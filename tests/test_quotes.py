"""Frases que respaldan cada cita en el modo Papers (decisión 42). Sin red ni LLM."""

from amr_agent.agent import (
    AMR_MODE,
    FINAL_SYSTEM_PROMPT,
    LITERATURE_MODE,
    SYSTEM_PROMPT,
    Agent,
    AgentAnswer,
    AgentUsage,
    Citation,
    Quote,
)
from amr_agent.evals.agent_eval import score_task, summarize
from amr_agent.evals.dataset import AgentTask
from amr_agent.quotes import RawQuote, locate, split_quotes
from amr_agent.tools import Excerpt, PaperSearchTools, ReadPaperTools, Toolset
from tests.fakes import ScriptedLLM, text_turn, tool_turn
from tests.test_literature_mode import epmc_record, network

PASSAGE = Excerpt("1", "Results > Safety",
                  "Cytokine release syndrome occurred in 43% of patients. For its management, "
                  "26% of patients received tocilizumab and 12% received corticosteroids.")


def test_the_quotes_block_is_split_from_the_answer():
    text = ("El 26 % recibió tocilizumab [PMID:1].\n\n"
            "**Quotes:**\n"
            '- [PMID:1] "26% of patients received tocilizumab"\n'
            '[PMID:2]: “Insulin independence occurred in 78.6% of recipients.”\n')
    body, quotes = split_quotes(text)
    assert body == "El 26 % recibió tocilizumab [PMID:1]."
    assert quotes == [RawQuote("1", "26% of patients received tocilizumab"),
                      RawQuote("2", "Insulin independence occurred in 78.6% of recipients.")]


def test_quote_lines_without_a_header_are_taken_from_the_end():
    # Con la respuesta en español, gpt-oss puede traducir la cabecera o saltársela.
    translated = 'Respuesta [PMID:1].\n\nCitas:\n"26% of patients received tocilizumab" [PMID:1]'
    missing = 'Respuesta [PMID:1].\n[PMID:1] "26% of patients received tocilizumab"'
    for text in (translated, missing):
        body, quotes = split_quotes(text)
        assert body == "Respuesta [PMID:1]."
        assert quotes == [RawQuote("1", "26% of patients received tocilizumab")]


def test_an_answer_without_quotes_is_left_as_it_is():
    text = 'Se usó "sin corticoides" [PMID:1].\nEl 26 % recibió tocilizumab [PMID:1].'
    assert split_quotes(text) == (text, [])


def test_a_quote_is_found_despite_gpt_oss_hyphens_spaces_and_cuts():
    # gpt-oss escribe U+2011 en vez de "-" y U+202F antes de "%", y puede cortar con "…".
    found = locate(RawQuote("1", "Cytokine release syndrome occurred in 43 % of patients… "
                                 "26 % of patients received tocilizumab"), [PASSAGE])
    assert found == PASSAGE
    assert locate(RawQuote("1", "IL‑6 blockade with tocilizumab was given"), [
        Excerpt("1", "Abstract", "IL-6 blockade with tocilizumab was given to 30 patients.")])


def test_paraphrases_short_quotes_and_other_articles_are_not_verified():
    assert locate(RawQuote("1", "a quarter of the patients were given tocilizumab"),
                  [PASSAGE]) is None  # parafraseada
    assert locate(RawQuote("1", "43% of patients"), [PASSAGE]) is None  # demasiado corta
    assert locate(RawQuote("2", "26% of patients received tocilizumab"),
                  [PASSAGE]) is None  # la frase existe, pero en otro artículo
    # Los trozos tienen que estar en orden.
    assert locate(RawQuote("1", "26% of patients received tocilizumab … Cytokine release "
                                "syndrome occurred"), [PASSAGE]) is None


async def test_the_agent_checks_each_quote_against_what_it_read():
    record = epmc_record("11111111")
    record["abstractText"] = "We report the safety and efficacy of an invented CAR-T product."
    http, _ = network([record])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "CAR-T tocilizumab"})),
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "tocilizumab"})),
        text_turn("El 26 % de los pacientes recibió tocilizumab [PMID:11111111].\n\nQuotes:\n"
                  '[PMID:11111111] "26% of patients received tocilizumab"\n'
                  '[PMID:11111111] "We report the safety and efficacy of an invented CAR-T"\n'
                  '[PMID:11111111] "Tocilizumab was given to every patient with CRS"\n'
                  '[PMID:22222222] "A quote from an article that no tool returned"'))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("¿Cuántos recibieron tocilizumab?")

    assert result.answer == "El 26 % de los pacientes recibió tocilizumab [PMID:11111111]."
    assert [c.pmid for c in result.citations] == ["11111111"]
    assert result.invalid_citations == ["22222222"]  # citado solo en las frases, y no visto
    assert [(q.verified, q.section) for q in result.citations[0].quotes] == [
        (True, "Results > Safety"), (True, "Abstract"), (False, None)]


async def test_a_refusal_needs_no_quotes():
    http, _ = network([])
    llm = ScriptedLLM(text_turn("I could not find evidence for this in the articles I found."))
    result = await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE).run("?")
    assert result.abstained and result.citations == []


def test_only_the_papers_mode_asks_for_quotes():
    assert LITERATURE_MODE.quotes and not AMR_MODE.quotes
    for prompt in (LITERATURE_MODE.system_prompt, LITERATURE_MODE.final_system_prompt):
        assert '"Quotes:"' in prompt
    assert "Quotes:" not in SYSTEM_PROMPT + FINAL_SYSTEM_PROMPT


async def test_the_amr_mode_does_not_parse_quotes():
    llm = ScriptedLLM(text_turn('Colistin [PMID:1].\nQuotes:\n[PMID:1] "mcr-1 confers colistin"'))
    result = await Agent(llm, Toolset([]), mode=AMR_MODE).run("?")
    assert result.answer.endswith('"mcr-1 confers colistin"')


def _answer(quotes: list[Quote]) -> AgentAnswer:
    return AgentAnswer(
        answer="x [PMID:1] [PMID:2]", invalid_citations=[], steps=[], stop_reason="answer",
        citations=[Citation(pmid="1", url="u", quotes=quotes), Citation(pmid="2", url="u")],
        usage=AgentUsage(model="m", llm_calls=1, prompt_tokens=1, completion_tokens=1),
        latency_ms={"total": 1.0})


def test_the_eval_counts_quoted_citations_and_verified_quotes():
    task = AgentTask(id="ft-x", category="full_text", question="?", expect_tools=[])
    quotes = [Quote(text="a", verified=True, section="Results"), Quote(text="b", verified=False)]
    row = score_task(task, _answer(quotes), quotes=True)
    assert (row["n_citations"], row["citations_quoted"], row["quotes"],
            row["quotes_verified"]) == (2, 1, 2, 1)
    summary = summarize([row])
    assert summary["quote_coverage"] == 0.5 and summary["quote_validity"] == 0.5
    # En el modo AMR no se piden frases: ni se cuentan ni salen en la tabla.
    amr = score_task(task, _answer([]))
    assert "quotes" not in amr and "quote_coverage" not in summarize([amr])


def test_footnotes_and_a_reference_list_give_way_to_the_quotes():
    # Prueba con gpt-oss (2026-11-12): con el bloque de frases, notas [^1] en el texto o una
    # lista "Citas" de PMIDs sueltos antes de "Quotes:".
    footnotes = ('Tacrolimus y sirolimus[^1][^2].\n\n**Citas:**\n'
                 '[PMID:1] "maintenance with sirolimus and tacrolimus"')
    body, quotes = split_quotes(footnotes)
    assert body == "Tacrolimus y sirolimus." and [q.pmid for q in quotes] == ["1"]
    listed = ('Puede causar ICANS.\n\n**Citas**\n- [PMID:1]\n- [PMID:2]\n\n**Quotes:**\n'
              '[PMID:1] "immune effector cell associated neurotoxicity syndrome"')
    assert split_quotes(listed)[0] == "Puede causar ICANS."


async def test_an_article_quoted_but_not_cited_in_the_text_is_still_cited():
    http, _ = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "tocilizumab"})),
        text_turn('El 26 % recibió tocilizumab[^1].\n\nQuotes:\n'
                  '[PMID:11111111] "26% of patients received tocilizumab"'))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("?")
    assert result.answer == "El 26 % recibió tocilizumab."
    assert [(c.pmid, [q.verified for q in c.quotes]) for c in result.citations] == [
        ("11111111", [True])]
