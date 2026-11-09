"""Texto completo: parser JATS, troceado, BM25 y la herramienta `read_paper` (sin red)."""

import json
from pathlib import Path

import httpx
import pytest

from amr_agent.fulltext import Block, bm25_rank, chunk, parse_jats, tokenize
from amr_agent.tools import EFETCH, IDCONV, READ_PAPER, ReadPaperTools

XML = (Path(__file__).parent / "fixtures" / "pmc_article.xml").read_bytes()


# --- Parser ---------------------------------------------------------------------------------

def test_parse_metadata():
    a = parse_jats(XML)
    assert (a.pmcid, a.pmid, a.year) == ("PMC9999999", "11111111", 2024)
    assert a.title.startswith("A phase 1 trial")
    assert a.journal == "Journal of Invented Therapies"


def test_parse_keeps_section_path_and_floats():
    sections = [b.section for b in parse_jats(XML).blocks]
    assert sections == ["Body", "Results > Safety", "Results > Safety", "Table 1",
                        "Results > Efficacy", "Figure 2"]


def test_parse_skips_acknowledgements_and_does_not_duplicate_tables():
    blocks = parse_jats(XML).blocks
    assert not any("thank the patients" in b.text for b in blocks)
    efficacy = next(b for b in blocks if b.section == "Results > Efficacy")
    assert "Complete response 61%" not in efficacy.text  # la tabla va en su propio bloque
    assert "Complete response 61%" in next(b for b in blocks if b.section == "Table 1").text


# --- Troceado y BM25 ------------------------------------------------------------------------

def test_chunk_merges_same_section_only():
    blocks = [Block("A", "x" * 100), Block("A", "y" * 100), Block("B", "z" * 100)]
    assert [(c.section, len(c.text)) for c in chunk(blocks, size=1200)] == [("A", 201), ("B", 100)]


def test_chunk_never_splits_a_long_paragraph():
    assert chunk([Block("A", "x" * 5000)], size=1200) == [Block("A", "x" * 5000)]


def test_tokenize_keeps_technical_terms_whole():
    assert tokenize("CAR-T, IL-13Rα2 and 2.7-fold") == ["car-t", "il-13rα2", "and", "2.7-fold"]


def test_bm25_prefers_rare_terms():
    docs = ["lymphoma lymphoma lymphoma response", "tocilizumab was given", "lymphoma cohort"]
    assert bm25_rank("lymphoma tocilizumab", docs)[0] == 1  # "lymphoma" está en todos


def test_bm25_without_useful_terms_keeps_order():
    assert bm25_rank("what is the", ["a", "b"]) == [0, 1]


# --- read_paper -----------------------------------------------------------------------------

def tools_with(handler) -> ReadPaperTools:
    return ReadPaperTools(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def ncbi(xml: bytes = XML, pmcid: str | None = "PMC9999999"):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        url = str(request.url)
        if url.startswith(IDCONV):
            return httpx.Response(200, json={"records": [{"pmcid": pmcid} if pmcid else {}]})
        if url.startswith(EFETCH):
            return httpx.Response(200, content=xml)
        return httpx.Response(404)

    return handler, calls


async def test_read_paper_returns_best_passage_with_section_and_citation():
    handler, _ = ncbi()
    result = await tools_with(handler).call(
        "read_paper", {"paper_id": "PMC9999999", "question": "How many patients got tocilizumab?",
                       "k": 2})
    data = json.loads(result.content)
    assert not result.is_error
    assert data["passages"][0]["section"] == "Results > Safety"
    assert "26% of patients received tocilizumab" in data["passages"][0]["text"]
    assert len(data["passages"]) == 2
    assert result.pmids == frozenset({"11111111"})  # se puede citar como [PMID:11111111]
    assert result.papers[0].year == 2024


async def test_read_paper_accepts_a_pmid_and_caches_the_article():
    handler, calls = ncbi()
    tools = tools_with(handler)
    for q in ("response rate", "neurologic events"):
        result = await tools.call("read_paper", {"paper_id": "PMID: 11111111", "question": q})
        assert not result.is_error
    efetches = [c for c in calls if str(c.url).startswith(EFETCH)]
    assert len(efetches) == 1  # la segunda pregunta usa el artículo guardado
    assert efetches[0].url.params["id"] == "9999999"
    assert efetches[0].url.params["tool"] == "amr-research-agent"


@pytest.mark.parametrize("xml, pmcid", [
    (b"<pmc-articleset><article><front/></article></pmc-articleset>", "PMC1"),  # sin <body>
    (XML, None),  # el PMID no está en PMC
])
async def test_read_paper_without_full_text_is_an_error_the_model_can_read(xml, pmcid):
    handler, _ = ncbi(xml, pmcid)
    result = await tools_with(handler).call("read_paper", {"paper_id": "12345678",
                                                           "question": "dose"})
    assert result.is_error
    assert "no full text in PubMed Central" in result.content


@pytest.mark.parametrize("args", [
    {"paper_id": "doi:10.1/x", "question": "dose"},
    {"paper_id": "PMC1", "question": ""},
])
async def test_read_paper_validates_arguments_without_calling_ncbi(args):
    handler, calls = ncbi()
    result = await tools_with(handler).call("read_paper", args)
    assert result.is_error and calls == []


def test_read_paper_schema_has_its_limits_in_the_description():
    k = READ_PAPER.to_openai()["function"]["parameters"]["properties"]["k"]
    assert "maximum" not in k and "maximum 10" in k["description"]
