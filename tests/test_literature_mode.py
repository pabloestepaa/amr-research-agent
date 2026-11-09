"""Modo literatura: `search_papers` (PubMed + Europe PMC) y el agente que lo usa con read_paper.

Sin red: PubMed, Europe PMC y NCBI se simulan con `httpx.MockTransport`.
"""

import json
from pathlib import Path

import httpx

from amr_agent.agent import AMR_MODE, LITERATURE_MODE, SYSTEM_PROMPT, Agent
from amr_agent.tools import (
    EFETCH,
    ESEARCH,
    EUROPEPMC_SEARCH,
    IDCONV,
    SEARCH_ABSTRACT_CHARS,
    PaperSearchTools,
    ReadPaperTools,
    Toolset,
)
from tests.fakes import ScriptedLLM, text_turn, tool_turn

XML = (Path(__file__).parent / "fixtures" / "pmc_article.xml").read_bytes()


def epmc_record(pmid: str, in_pmc: bool = True, open_access: bool = True) -> dict:
    return {"pmid": pmid, "pmcid": f"PMC{pmid}" if in_pmc else None,
            "title": f"Article {pmid}", "pubYear": "2024",
            "journalInfo": {"journal": {"title": "Blood"}},
            "isOpenAccess": "Y" if open_access else "N", "inPMC": "Y" if in_pmc else "N",
            "abstractText": "CAR-T <i>cells</i> " + "x" * 2000}


def network(records: list[dict], pubmed: list[str] | None = None, esearch_down: bool = False):
    """PubMed ordena `pubmed` (por defecto, los PMIDs de `records`); Europe PMC describe
    `records`; NCBI devuelve el artículo de ejemplo (PMID 11111111)."""
    calls: list[httpx.Request] = []
    ids = [r["pmid"] for r in records if r.get("pmid")] if pubmed is None else pubmed

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        url = str(request.url)
        if url.startswith(ESEARCH):
            if esearch_down:  # NCBI a veces da el error con estado 200
                return httpx.Response(200, json={"ERROR": "Search Backend failed"})
            return httpx.Response(200, json={"esearchresult": {"count": str(len(ids)),
                                                                "idlist": ids}})
        if url.startswith(EUROPEPMC_SEARCH):
            return httpx.Response(200, json={"resultList": {"result": records}})
        if url.startswith(IDCONV):
            return httpx.Response(200, json={"records": [{"pmcid": "PMC9999999"}]})
        if url.startswith(EFETCH):
            return httpx.Response(200, content=XML)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


# --- search_papers --------------------------------------------------------------------------

async def test_search_papers_keeps_pubmed_order_and_returns_short_citable_results():
    records = [epmc_record("1"), epmc_record("2", in_pmc=False), {"title": "preprint, no PMID"}]
    http, calls = network(records, pubmed=["2", "1"])  # Europe PMC los da en otro orden
    result = await PaperSearchTools(http).call("search_papers", {"query": "CAR-T CRS", "k": 3})
    rows = json.loads(result.content)["results"]
    assert [(r["pmid"], r["full_text"]) for r in rows] == [("2", False), ("1", True)]
    assert rows[1]["abstract"].startswith("CAR-T cells x")  # sin etiquetas HTML
    assert len(rows[1]["abstract"]) == SEARCH_ABSTRACT_CHARS + 1  # recortado ("…")
    assert (rows[1]["journal"], rows[1]["year"]) == ("Blood", 2024)
    assert result.pmids == frozenset({"1", "2"})
    esearch, epmc = calls
    assert (esearch.url.params["term"], esearch.url.params["sort"]) == ("CAR-T CRS", "relevance")
    assert epmc.url.params["query"] == "(EXT_ID:2 OR EXT_ID:1) AND SRC:MED"


async def test_an_author_manuscript_in_pmc_counts_as_readable():
    # El informe de ZUMA-1 en NEJM no es de licencia abierta, pero está en PMC y efetch lo da.
    http, _ = network([epmc_record("29226797", in_pmc=True, open_access=False)])
    result = await PaperSearchTools(http).call("search_papers", {"query": "ZUMA-1"})
    assert json.loads(result.content)["results"][0]["full_text"] is True


async def test_search_papers_can_ask_only_for_readable_articles():
    http, calls = network([])
    result = await PaperSearchTools(http).call(
        "search_papers", {"query": "CAR-T", "full_text_only": True})
    assert calls[0].url.params["term"] == '(CAR-T) AND "pubmed pmc"[sb]'
    assert len(calls) == 1  # sin PMIDs no se pregunta a Europe PMC
    assert json.loads(result.content)["note"].startswith("no matching articles")


async def test_search_papers_falls_back_to_europepmc_when_pubmed_fails():
    http, calls = network([epmc_record("1")], esearch_down=True)
    result = await PaperSearchTools(http).call("search_papers", {"query": "CAR-T CRS"})
    assert json.loads(result.content)["results"][0]["pmid"] == "1"
    assert calls[-1].url.params["query"] == "TITLE_ABS:(CAR-T CRS) AND SRC:MED"


# --- El agente en modo literatura -----------------------------------------------------------

async def test_literature_mode_searches_reads_and_cites_the_full_text():
    http, _ = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "CAR-T tocilizumab"})),
        tool_turn(("read_paper", {"paper_id": "11111111",
                                  "question": "patients who received tocilizumab"})),
        text_turn("26% of patients received tocilizumab [PMID:11111111]."))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run(
        "¿Qué porcentaje de pacientes recibió tocilizumab?")

    system = llm.calls[0]["messages"][0]["content"]
    assert system.startswith("You are a research assistant for the biomedical literature.")
    assert "At most 2 searches" in " ".join(system.split())  # sin depender de los saltos de línea
    passages = json.loads(llm.calls[2]["messages"][-1]["content"])["passages"]
    assert "26% of patients received tocilizumab" in passages[0]["text"]
    assert [c.pmid for c in result.citations] == ["11111111"] and not result.invalid_citations
    assert result.citations[0].title.startswith("A phase 1 trial")  # del texto completo


async def test_literature_mode_caps_search_papers_not_read_paper():
    http, calls = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "a"*3})),
        tool_turn(("search_papers", {"query": "b"*3}),
                  ("search_papers", {"query": "c"*3}),
                  ("read_paper", {"paper_id": "PMC9999999", "question": "dose"})),
        text_turn("ok"))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("?")
    assert [(s.tool, s.is_error) for s in result.steps] == [
        ("search_papers", False), ("search_papers", False), ("search_papers", True),
        ("read_paper", False)]
    assert sum(str(c.url).startswith(EUROPEPMC_SEARCH) for c in calls) == 2
    blocked = llm.calls[2]["messages"][-2]["content"]
    assert blocked.startswith("ERROR: at most 2 search_papers calls per question")


async def test_final_turn_uses_the_mode_prompt():
    http, _ = network([epmc_record("1")])
    llm = ScriptedLLM(tool_turn(("search_papers", {"query": "CAR-T"})), text_turn("ok"))
    await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE,
                max_steps=2).run("?")
    final_system = llm.calls[-1]["messages"][0]["content"]
    assert llm.calls[-1]["tools"] == []
    assert final_system.startswith("You are a research assistant for the biomedical literature.")


def test_amr_mode_is_the_default_and_keeps_its_prompt():
    assert AMR_MODE.system_prompt is SYSTEM_PROMPT
    assert AMR_MODE.search_tool == "search_literature"
    assert Agent(ScriptedLLM(), Toolset([])).mode is AMR_MODE
