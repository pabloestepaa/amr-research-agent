"""Modo literatura: `search_papers` (PubMed + Europe PMC) y el agente que lo usa con read_paper.

Sin red: PubMed, Europe PMC y NCBI se simulan con `httpx.MockTransport`.
"""

import json
from pathlib import Path

import httpx

from amr_agent.agent import AMR_MODE, LITERATURE_MODE, SYSTEM_PROMPT, Agent
from amr_agent.rag import NO_EVIDENCE_PAPERS
from amr_agent.tools import (
    EFETCH,
    ESEARCH,
    EUROPEPMC_SEARCH,
    IDCONV,
    SEARCH_ABSTRACT_CHARS,
    PaperSearchTools,
    ReadPaperTools,
    Toolset,
    pubmed_term,
    reading_memory,
    shorter_queries,
    study_type,
)
from tests.fakes import ScriptedLLM, text_turn, tool_turn

XML = (Path(__file__).parent / "fixtures" / "pmc_article.xml").read_bytes()
# El artículo de ejemplo con 12 secciones más: da para leerlo dos veces sin repetir pasajes.
LONG_XML = XML.replace(b"</body>", b"".join(
    b"<sec><title>S%d</title><p>tocilizumab dose detail number %d in this section</p></sec>"
    % (i, i) for i in range(12)) + b"</body>")


def epmc_record(pmid: str, in_pmc: bool = True, open_access: bool = True) -> dict:
    return {"pmid": pmid, "pmcid": f"PMC{pmid}" if in_pmc else None,
            "title": f"Article {pmid}", "pubYear": "2024",
            "journalInfo": {"journal": {"title": "Blood"}},
            "isOpenAccess": "Y" if open_access else "N", "inPMC": "Y" if in_pmc else "N",
            "abstractText": "CAR-T <i>cells</i> " + "x" * 2000}


def network(records: list[dict], pubmed: list[str] | None = None, esearch_down: bool = False,
            xml: bytes = XML):
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
            return httpx.Response(200, content=xml)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


# --- search_papers --------------------------------------------------------------------------

async def test_search_papers_keeps_pubmed_order_and_returns_short_citable_results():
    records = [epmc_record("1"), epmc_record("2", in_pmc=False), {"title": "preprint, no PMID"}]
    http, calls = network(records, pubmed=["2", "1"])  # Europe PMC los da en otro orden
    result = await PaperSearchTools(http).call("search_papers", {"query": "CAR-T CRS", "top_k": 3})
    rows = json.loads(result.content)["results"]
    assert [(r["pmid"], r["full_text"]) for r in rows] == [("2", False), ("1", True)]
    assert rows[1]["abstract"].startswith("CAR-T cells x")  # sin etiquetas HTML
    assert len(rows[1]["abstract"]) == SEARCH_ABSTRACT_CHARS + 1  # recortado ("…")
    assert (rows[1]["journal"], rows[1]["year"]) == ("Blood", 2024)
    assert result.pmids == frozenset({"1", "2"})
    esearch, epmc = calls
    assert (esearch.url.params["term"], esearch.url.params["sort"]) == ('"CAR T" CRS', "relevance")
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
    assert calls[0].url.params["term"] == '("CAR T") AND "pubmed pmc"[sb]'
    assert len(calls) == 1  # sin PMIDs no se pregunta a Europe PMC
    assert json.loads(result.content)["note"].startswith("no matching articles")


async def test_search_papers_falls_back_to_europepmc_when_pubmed_fails():
    http, calls = network([epmc_record("1")], esearch_down=True)
    result = await PaperSearchTools(http).call("search_papers", {"query": "CAR-T CRS"})
    assert json.loads(result.content)["results"][0]["pmid"] == "1"
    assert calls[-1].url.params["query"] == "TITLE_ABS:(CAR-T CRS) AND SRC:MED"


def test_car_t_goes_to_pubmed_as_a_phrase_not_as_an_author():
    # PubMed traducía "CAR T" como "car t"[Author] y la búsqueda se quedaba sin el ensayo bueno.
    assert pubmed_term("GD2 CAR T H3K27M glioma") == 'GD2 "CAR T" H3K27M glioma'
    assert pubmed_term("CAR T-cell therapy") == '"CAR T" cell therapy'
    assert pubmed_term("car-t cells DIPG") == '"CAR T" cells DIPG'
    assert pubmed_term('"CAR T" glioma') == '"CAR T" glioma'  # ya entrecomillado
    assert pubmed_term("CART protein; CAR-NK cells") == "CART protein; CAR-NK cells"


async def test_search_papers_sends_the_car_t_phrase_to_pubmed():
    http, calls = network([])
    await PaperSearchTools(http).call(
        "search_papers", {"query": "B7-H3 CAR T DIPG", "full_text_only": True})
    assert calls[0].url.params["term"] == '(B7-H3 "CAR T" DIPG) AND "pubmed pmc"[sb]'


# --- Reintento con menos palabras (decisión 45) ---------------------------------------------

def test_shorter_queries_drop_generic_words_and_keep_the_study_first():
    # Consultas reales de la eval de literatura que dieron 0 artículos.
    assert shorter_queries("MAGE-A4 autologous T cells HLA-A*02 phase 1 solid cancers") == [
        "MAGE-A4 autologous solid cancers", "MAGE-A4 autologous"]  # "*" es el comodín
    assert shorter_queries("EPCORE NHL-1 epcoritamab large B-cell lymphoma 2-year follow-up "
                           "tumor lysis syndrome grade 3") == [
        "EPCORE NHL-1 epcoritamab large", "EPCORE NHL-1"]
    assert shorter_queries("GD2 CAR T cells H3K27M diffuse midline glioma manufacturing") == [
        'GD2 "CAR T" H3K27M diffuse', 'GD2 "CAR T"']  # la frase cuenta como una palabra
    assert shorter_queries("ADP-A2M4 phase 1") == ["ADP-A2M4"]
    assert shorter_queries("ZUMA-1 CRS") == []  # nada que quitar: no se repite la búsqueda


def test_shorter_queries_keep_the_number_of_a_type_or_class():
    # Antes, "type 2 diabetes" se quedaba en "type diabetes": el 2 contaba como un dato suelto.
    assert shorter_queries("type 2 diabetes islet transplantation insulin independence "
                           "5-year rate") == [
        "type 2 diabetes islet transplantation", "type 2 diabetes"]
    assert shorter_queries("class II HLA antibodies kidney graft loss phase 3") == [
        "class II HLA antibodies kidney", "class II HLA"]


async def test_search_papers_retries_an_empty_query_with_fewer_words():
    terms: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(ESEARCH):
            terms.append(request.url.params["term"])
            ids = [] if "phase" in terms[-1] else ["11111111"]
            return httpx.Response(200, json={"esearchresult": {"count": str(len(ids)),
                                                                "idlist": ids}})
        return httpx.Response(200, json={"resultList": {"result": [epmc_record("11111111")]}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await PaperSearchTools(http).call(
        "search_papers", {"query": "MAGE-A4 autologous T cells phase 1 solid cancers"})
    assert terms == ["MAGE-A4 autologous T cells phase 1 solid cancers",
                     "MAGE-A4 autologous solid cancers"]
    assert result.pmids == {"11111111"}
    # La nota va delante: el modelo sabe que la búsqueda que funcionó no es la suya.
    assert result.content.startswith('{"note":"no article matched every keyword; these are the '
                                     'results for a shorter query: MAGE-A4 autologous solid')


async def test_when_the_shorter_queries_fail_too_the_model_is_told_which_were_tried():
    http, calls = network([])
    result = await PaperSearchTools(http).call(
        "search_papers", {"query": "MAGE-A4 autologous T cells phase 1 solid cancers"})
    assert len(calls) == 3  # la suya y dos más cortas; sin PMIDs no se pregunta a Europe PMC
    note = json.loads(result.content)["note"]
    assert note.startswith("no matching articles")
    assert "(MAGE-A4 autologous solid cancers | MAGE-A4 autologous)" in note


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


async def test_a_second_read_brings_only_new_passages_and_a_third_is_refused():
    # Pregunta real (2026-11-12): leyó el mismo artículo tres veces y recibió casi lo mismo
    # (decisión 40). En la eval (ft-04), la 2.ª pregunta, más corta, habría encontrado la cifra:
    # ahora la 2.ª lectura trae los pasajes que no ha visto, y la 3.ª se rechaza (decisión 47).
    http, _ = network([epmc_record("11111111")], xml=LONG_XML)
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "tocilizumab dose",
                                  "k": 4})),
        tool_turn(("read_paper", {"paper_id": "PMID: 11111111", "question": "tocilizumab",
                                  "k": 4})),
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "dose", "k": 4})),
        text_turn("ok"))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("?")
    assert [(s.tool, s.is_error) for s in result.steps] == [
        ("read_paper", False), ("read_paper", False), ("read_paper", True)]
    first, second = (json.loads(llm.calls[i]["messages"][-1]["content"]) for i in (1, 2))
    texts = [{p["text"] for p in r["passages"]} for r in (first, second)]
    assert len(texts[0]) == len(texts[1]) == 4 and not texts[0] & texts[1]
    assert "note" not in first and second["note"].startswith("You read this article before")
    blocked = llm.calls[3]["messages"][-1]["content"]
    assert blocked.startswith("ERROR: you already read 11111111 2 times (steps 1 and 2)")


async def test_a_new_question_starts_with_nothing_shown():
    # La memoria es de la pregunta: la siguiente (u otra petición a la vez) lee desde el principio.
    http, _ = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    read = tool_turn(("read_paper", {"paper_id": "11111111", "question": "dose", "k": 8}))
    agent = Agent(ScriptedLLM(read, text_turn("ok"), read, text_turn("ok")), tools,
                  mode=LITERATURE_MODE)
    for _ in range(2):
        assert not (await agent.run("?")).steps[0].is_error


async def test_when_every_passage_was_shown_the_model_is_told_so():
    http, _ = network([])
    tools = ReadPaperTools(http)
    with reading_memory():
        await tools.call("read_paper", {"paper_id": "PMC9999999", "question": "dose", "k": 8})
        again = await tools.call("read_paper", {"paper_id": "PMC9999999", "question": "rate"})
    assert again.is_error and "every passage of PMC9999999 was already shown" in again.content


async def test_a_read_that_failed_can_be_asked_again():
    http, _ = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "dose", "pages": 2})),
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "dose"})),
        text_turn("ok"))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("?")
    assert [s.is_error for s in result.steps] == [True, False]  # argumento inventado, luego bien


async def test_the_literature_mode_abstains_with_its_own_message():
    # No tiene índice: "indexed literature" era el mensaje del modo AMR.
    result = await Agent(ScriptedLLM(text_turn("")), Toolset([]), mode=LITERATURE_MODE).run("?")
    assert result.abstained and "biomedical and life-science research" in result.answer
    assert NO_EVIDENCE_PAPERS in LITERATURE_MODE.system_prompt
    assert NO_EVIDENCE_PAPERS not in AMR_MODE.system_prompt


async def test_final_turn_uses_the_mode_prompt():
    http, _ = network([epmc_record("1")])
    llm = ScriptedLLM(tool_turn(("search_papers", {"query": "CAR-T"})), text_turn("ok"))
    await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE,
                max_steps=2).run("?")
    final_system = llm.calls[-1]["messages"][0]["content"]
    assert llm.calls[-1]["tools"] == []
    assert final_system.startswith("You are a research assistant for the biomedical literature.")


def test_only_the_papers_mode_asks_for_native_writing():
    # Decisión 41: la regla de idioma larga solo en Papers; el prompt AMR no cambia.
    assert "native" in LITERATURE_MODE.system_prompt
    assert "native" in LITERATURE_MODE.final_system_prompt
    assert "native" not in AMR_MODE.system_prompt


def test_amr_mode_is_the_default_and_keeps_its_prompt():
    assert AMR_MODE.system_prompt is SYSTEM_PROMPT
    assert AMR_MODE.search_tool == "search_literature"
    assert Agent(ScriptedLLM(), Toolset([])).mode is AMR_MODE


# --- Argumentos inventados (decisión 35) ----------------------------------------------------

async def test_invented_arguments_are_rejected_before_reaching_the_tool():
    http, calls = network([epmc_record("1")])
    result = await Toolset([PaperSearchTools(http)]).call(
        "search_papers", {"query": "CAR-T", "max_results": 10})
    assert result.is_error and calls == []  # ni PubMed ni Europe PMC
    assert result.content == ("ERROR: Unknown argument(s) for search_papers: max_results. "
                              "Valid arguments: query, top_k, full_text_only.")


async def test_null_optional_arguments_are_not_invented():
    http, _ = network([epmc_record("1")])
    result = await Toolset([PaperSearchTools(http)]).call(
        "search_papers", {"query": "CAR-T", "k": None, "full_text_only": None})
    assert not result.is_error


async def test_read_paper_returns_at_most_eight_passages():
    tools = ReadPaperTools(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=LONG_XML))))
    result = await tools.call("read_paper", {"paper_id": "PMC1", "question": "tocilizumab",
                                             "k": 10})
    assert len(json.loads(result.content)["passages"]) == 8


# --- Qué cuenta para el límite de búsquedas (decisión 37) -----------------------------------

async def test_rejected_and_empty_searches_do_not_use_up_the_limit():
    # En la demo: un argumento inventado y una consulta larga sin resultados dejaban al agente sin
    # búsquedas antes de ver un solo artículo.
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if str(request.url).startswith(ESEARCH):
            ids = [] if "long" in request.url.params["term"] else ["11111111"]
            return httpx.Response(200, json={"esearchresult": {"count": str(len(ids)),
                                                                "idlist": ids}})
        return httpx.Response(200, json={"resultList": {"result": [epmc_record("11111111")]}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "CAR-T", "max_results": 5})),  # rechazada
        tool_turn(("search_papers", {"query": "a long query with no results"})),  # vacía
        tool_turn(("search_papers", {"query": "CAR-T glioma"})),
        tool_turn(("search_papers", {"query": "CAR-T glioma phase 1"})),
        tool_turn(("search_papers", {"query": "CAR-T"})),  # tercera con resultados: bloqueada
        text_turn("ok"))
    result = await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE).run("?")
    assert [s.is_error for s in result.steps] == [True, False, False, False, True]
    assert result.steps[-1].preview.startswith("ERROR: at most 2 search_papers calls")
    empty = json.loads(llm.calls[2]["messages"][-1]["content"])
    assert "did not count towards your limit" in empty["note"]


async def test_two_searches_in_one_step_count_only_if_they_return_articles():
    http, _ = network([epmc_record("11111111")])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "CAR-T"}), ("search_papers", {"query": "x", "y": 1})),
        tool_turn(("search_papers", {"query": "CAR-T glioma"})),  # segunda que cuenta: pasa
        text_turn("ok"))
    result = await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE).run("?")
    assert [s.is_error for s in result.steps] == [False, True, False]


# --- Tipo de estudio y citas (decisión 43) --------------------------------------------------

def test_study_type_keeps_the_most_informative_label():
    # Combinaciones reales de Europe PMC (MeSH de PubMed y article-type de PMC mezclados).
    assert study_type(["Meta-Analysis", "Systematic Review", "review-article"]) == "Meta-analysis"
    assert study_type(["Clinical Trial, Phase I", "Clinical Trial, Phase II",
                       "Multicenter Study"]) == "Phase 1/2 trial"
    assert study_type(["Randomized Controlled Trial",
                       "Clinical Trial, Phase III"]) == "Phase 3 randomized trial"
    assert study_type(["Clinical Trial", "Multicenter Study"]) == "Clinical trial"
    assert study_type(["Scoping Review", "review-article"]) == "Scoping review"
    assert study_type(["Case Reports", "research-article"]) == "Case report"
    assert study_type(["Comparative Study", "research-article"]) == "Research article"
    assert study_type(["Journal Article", "Research Support, Non-U.S. Gov't"]) is None


async def test_search_papers_gives_the_study_type_and_only_meaningful_citation_counts():
    trial, recent = epmc_record("1"), epmc_record("2")
    trial.update(pubTypeList={"pubType": ["Clinical Trial, Phase II", "Journal Article"]},
                 citedByCount=1990)
    recent.update(pubTypeList={"pubType": ["Journal Article"]}, citedByCount=3)
    http, _ = network([trial, recent])
    result = await PaperSearchTools(http).call("search_papers", {"query": "CAR-T CRS"})
    rows = json.loads(result.content)["results"]
    assert (rows[0]["study_type"], rows[0]["cited_by"]) == ("Phase 2 trial", 1990)
    assert "study_type" not in rows[1] and "cited_by" not in rows[1]  # sin claves vacías


async def test_reading_an_article_keeps_its_study_type_in_the_reference():
    record = epmc_record("11111111")
    record.update(pubTypeList={"pubType": ["Clinical Trial, Phase I"]}, citedByCount=25)
    http, _ = network([record])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "CAR-T tocilizumab"})),
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "tocilizumab"})),
        text_turn("26% received tocilizumab [PMID:11111111]."))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("?")
    citation = result.citations[0]
    assert citation.title.startswith("A phase 1 trial")  # el título del texto completo
    assert (citation.study_type, citation.cited_by) == ("Phase 1 trial", 25)  # y no se pierde


def test_the_papers_prompt_prefers_stronger_designs_and_the_amr_prompt_is_unchanged():
    prompt = " ".join(LITERATURE_MODE.system_prompt.split())  # sin depender de los saltos
    assert "Prefer meta-analyses, systematic reviews and trials" in prompt
    assert "study_type" not in AMR_MODE.system_prompt
