"""Tests de las evals del agente: puntuación de trayectorias, resumen, grabación y golden set."""

import json

import pytest

from amr_agent.agent import AgentAnswer, AgentUsage, ToolStep
from amr_agent.evals.agent_eval import failed_row, normalize_text, score_task, summarize
from amr_agent.evals.cassette import CassetteTools, cassette_key
from amr_agent.evals.dataset import LITERATURE_GOLDEN_PATH, AgentTask, load_agent_golden
from amr_agent.evals.report import agent_table
from amr_agent.rag import NO_EVIDENCE
from amr_agent.tools import (
    MCP_TOOLS,
    READ_PAPER,
    SEARCH_LITERATURE,
    SEARCH_PAPERS,
    ToolResult,
    ToolSpec,
)

CAT = "get_amr_gene|search_amr_genes"
HOST_TASK = AgentTask(
    id="host-01", category="hosts", question="Where is blaKPC-2 found?",
    expect_tools=[CAT, "get_protein_hosts"], expect_args={"get_protein_hosts": "WP_004199234"},
    answer_contains=["Klebsiella pneumoniae|K. pneumoniae"],
)


def step(tool: str, step: int = 1, error: bool = False, pmids=(), **args) -> ToolStep:
    return ToolStep(step=step, tool=tool, arguments=args, is_error=error, truncated=False,
                    result_chars=10, pmids=list(pmids), latency_ms=1.0, preview="")


def answer(text: str, steps: list[ToolStep], invalid=(), stop="answer") -> AgentAnswer:
    return AgentAnswer(answer=text, citations=[], invalid_citations=list(invalid), steps=steps,
                       stop_reason=stop,
                       usage=AgentUsage(model="m", llm_calls=len(steps) + 1, prompt_tokens=100,
                                        completion_tokens=10),
                       latency_ms={"llm": 1.0, "tools": 1.0, "total": 2.0})


GOOD_HOST_STEPS = [step("get_amr_gene", 1, name="blaKPC-2"),
                   step("get_protein_hosts", 2, accession="WP_004199234.1")]


# --- score_task ------------------------------------------------------------------------------

def test_successful_trajectory():
    row = score_task(HOST_TASK, answer("Mostly in K. pneumoniae.", GOOD_HOST_STEPS))
    assert row["success"] and row["tools_ok"] and row["args_ok"] and row["answer_ok"]
    assert row["tool_recall"] == 1.0 and row["extra_calls"] == 0
    assert row["trajectory"] == ['1:get_amr_gene({"name": "blaKPC-2"})',
                                 '2:get_protein_hosts({"accession": "WP_004199234.1"})']


def test_wrong_accession_fails_even_if_the_answer_looks_right():
    # El modelo acertó la especie, pero consultó otra proteína: la trayectoria es incorrecta.
    steps = [step("search_amr_genes", 1, query="KPC"),
             step("get_protein_hosts", 2, accession="WP_000027057.1")]
    row = score_task(HOST_TASK, answer("Klebsiella pneumoniae.", steps))
    assert row["tools_ok"] and row["answer_ok"]
    assert row["args_ok"] is False and not row["success"]


def test_missing_tool_and_missing_fact():
    row = score_task(HOST_TASK, answer("Found in Escherichia coli.", GOOD_HOST_STEPS[:1]))
    assert row["tool_recall"] == 0.5 and row["tools_ok"] is False
    assert row["answer_ok"] is False and not row["success"]


def test_counts_extra_and_failed_calls():
    steps = [*GOOD_HOST_STEPS, step("get_amr_gene", 3, error=True, name="kpc2"),
             step("search_literature", 3, query="KPC")]
    row = score_task(HOST_TASK, answer("K. pneumoniae", steps))
    assert row["success"]  # sobra una llamada, pero la tarea se resolvió
    assert row["extra_calls"] == 1 and row["error_calls"] == 1


def test_literature_task_needs_the_relevant_abstract():
    task = AgentTask(id="lit", category="literature", question="q",
                     expect_tools=["search_literature"], relevant_pmids=["42"])
    hit = score_task(task, answer("x [PMID:42]", [step("search_literature", pmids=["7", "42"])]))
    miss = score_task(task, answer("x [PMID:7]", [step("search_literature", pmids=["7"])]))
    assert hit["literature_hit"] and hit["success"]
    assert miss["literature_hit"] is False and not miss["success"]


def test_abstention_is_right_out_of_scope_and_wrong_in_scope():
    oos = AgentTask(id="oos", category="out_of_scope", question="capital?", should_abstain=True)
    assert score_task(oos, answer(NO_EVIDENCE, []))["success"]
    assert not score_task(oos, answer("Paris.", []))["success"]
    wrong = score_task(HOST_TASK, answer(NO_EVIDENCE, GOOD_HOST_STEPS))
    assert wrong["abstained"] and not wrong["success"]


def test_hallucinated_citation_fails_the_task():
    row = score_task(HOST_TASK, answer("K. pneumoniae [PMID:1]", GOOD_HOST_STEPS, invalid=["1"]))
    assert row["citations_valid"] is False and not row["success"]


# --- summarize y tabla ------------------------------------------------------------------------

def test_summarize_rates_and_categories():
    oos = AgentTask(id="oos", category="out_of_scope", question="?", should_abstain=True)
    rows = [
        score_task(HOST_TASK, answer("K. pneumoniae", GOOD_HOST_STEPS)),
        score_task(HOST_TASK, answer("E. coli", GOOD_HOST_STEPS[:1], stop="max_steps")),
        score_task(oos, answer(NO_EVIDENCE, [])),
    ]
    s = summarize(rows)
    assert s["task_success"] == round(2 / 3, 4)
    assert s["tool_selection"] == 0.5  # solo cuenta las tareas en ámbito
    assert s["abstention_accuracy"] == 1.0 and s["false_abstention_rate"] == 0.0
    assert s["literature_hit_rate"] is None  # ninguna tarea lo comprueba
    assert s["early_stop_rate"] == round(1 / 3, 4)
    assert s["by_category"] == {"hosts": {"n": 2, "success": 0.5},
                                "out_of_scope": {"n": 1, "success": 1.0}}

    report = {"model": "m", "n_tasks": 3, "max_steps": 5, "token_budget": 12000,
              "cassette_mode": "replay", "cassette_hits": 2, "cassette_misses": 1,
              "summary": s, "per_task": rows}
    table = agent_table(report)
    assert "| Task success (every check below passed) | 0.67 |" in table
    assert "- `host-01`: 1:get_amr_gene" in table  # la tarea fallida, con su trayectoria


def test_typography_does_not_hide_a_correct_fact():
    # host-02: la respuesta era correcta, pero gpt-oss separó "Escherichia coli" con un espacio
    # fino (U+202F) y la comparación no lo encontraba.
    text = "The top host is **Escherichia\u202fcoli** (82\u202f%), via bla\u2011KPC\u20112."
    task = HOST_TASK.model_copy(update={"answer_contains": ["Escherichia coli", "blaKPC-2|KPC-2"]})
    row = score_task(task, answer(text, GOOD_HOST_STEPS))
    assert row["answer_ok"] is True
    assert normalize_text("A\u00a0B\u2212C  D") == "a b-c d"
    assert normalize_text("5 of them (62.5\u202f%)") == "5 of them (62.5%)"


def test_a_crashed_task_counts_as_failed_but_not_in_the_cost_averages():
    # Un 400 del proveedor en una tarea no debe tumbar la ejecución ni perder las demás.
    ok = score_task(HOST_TASK, answer("K. pneumoniae", GOOD_HOST_STEPS))
    crashed = failed_row(HOST_TASK, ValueError("Tool choice is none, but model called a tool"),
                         latency_ms=50_000)
    s = summarize([ok, crashed])
    assert s["task_success"] == 0.5
    assert s["agent_errors"] == 1
    assert s["tool_selection"] == 1.0  # la que no terminó no se pudo comprobar
    assert s["mean_tokens"] == ok["tokens"] and s["latency_ms_p50"] == ok["latency_ms"]
    assert s["early_stop_rate"] == 0.5

    report = {"model": "m", "n_tasks": 2, "max_steps": 5, "token_budget": 12000,
              "cassette_mode": "replay", "cassette_hits": 0, "cassette_misses": 0,
              "summary": s, "per_task": [ok, crashed]}
    assert "- `host-01`: error: ValueError: Tool choice is none" in agent_table(report)


# --- grabación de respuestas -----------------------------------------------------------------

class CountingTools:
    def __init__(self) -> None:
        self.calls = 0

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec("get_amr_gene", "", {"type": "object", "properties": {}})]

    async def call(self, name, arguments) -> ToolResult:
        self.calls += 1
        if arguments.get("name") == "nope":
            return ToolResult("ERROR: not in catalog", is_error=True)
        return ToolResult(f'{{"call": {self.calls}}}', pmids=frozenset({"123"}))


async def test_cassette_records_then_replays(tmp_path):
    path = tmp_path / "cassette.jsonl"
    inner = CountingTools()
    first = CassetteTools(inner, path)
    live = await first.call("get_amr_gene", {"name": "mcr-1", "x": 1})
    first.save()

    second = CassetteTools(inner, path)
    replayed = await second.call("get_amr_gene", {"x": 1, "name": "mcr-1"})  # otro orden
    assert replayed.content == live.content and replayed.pmids == {"123"}
    assert inner.calls == 1 and (second.hits, second.misses) == (1, 0)


async def test_cassette_does_not_record_errors(tmp_path):
    cassette = CassetteTools(CountingTools(), tmp_path / "c.jsonl")
    await cassette.call("get_amr_gene", {"name": "nope"})
    cassette.save()
    assert (tmp_path / "c.jsonl").read_text() == ""


async def test_cassette_refresh_and_live_modes(tmp_path):
    path = tmp_path / "c.jsonl"
    inner = CountingTools()
    rec = CassetteTools(inner, path)
    await rec.call("get_amr_gene", {"name": "a"})
    rec.save()

    refresh = CassetteTools(inner, path, mode="refresh")
    assert (await refresh.call("get_amr_gene", {"name": "a"})).content == '{"call": 2}'
    refresh.save()
    assert json.loads(path.read_text())["content"] == '{"call": 2}'

    live = CassetteTools(inner, path, mode="live")
    await live.call("get_amr_gene", {"name": "a"})
    assert inner.calls == 3 and live.hits == 0


def test_cassette_key_ignores_argument_order():
    assert cassette_key("t", {"a": 1, "b": 2}) == cassette_key("t", {"b": 2, "a": 1})


# --- golden set ------------------------------------------------------------------------------

def test_agent_golden_set_is_valid():
    tasks = load_agent_golden()
    known = set(MCP_TOOLS) | {SEARCH_LITERATURE.name}
    for t in tasks:
        named = {n for group in t.expect_tools for n in group.split("|")}
        assert named <= known, f"{t.id}: unknown tools {named - known}"
    assert {t.category for t in tasks} == {
        "catalog", "hosts", "references", "literature", "multi_tool", "out_of_scope"}


def test_agent_golden_rejects_inconsistent_tasks(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q", "category": "catalog",
                               "expect_tools": ["get_amr_gene"],
                               "expect_args": {"get_protein_hosts": "WP_1"}}) + "\n")
    with pytest.raises(ValueError, match="expect_args"):
        load_agent_golden(bad)


# --- modo literatura -------------------------------------------------------------------------

FT_TASK = AgentTask(id="ft-01", category="full_text", question="Median duration of response?",
                    expect_tools=["search_papers", "read_paper"],
                    answer_contains=["25.6 weeks"], relevant_pmids=["36624315"])


def test_literature_task_counts_search_papers_and_read_paper():
    steps = [step("search_papers", 1, pmids=["36624315", "1"], query="MAGE-A4 T cells"),
             step("read_paper", 2, pmids=["36624315"], paper_id="36624315", question="DoR")]
    row = score_task(FT_TASK, answer("25.6 weeks [PMID:36624315].", steps))
    assert row["success"] and row["literature_hit"] and row["read_hit"]


def test_reading_another_paper_is_a_read_miss_but_not_by_itself_a_failure():
    # Encontró el artículo pero leyó otro y aun así dio el dato (p. ej., un seguimiento que lo
    # repite): la tarea cuenta, y read_hit dice que la lectura no fue la esperada.
    steps = [step("search_papers", 1, pmids=["36624315", "2"], query="MAGE-A4"),
             step("read_paper", 2, pmids=["2"], paper_id="2", question="DoR")]
    row = score_task(FT_TASK, answer("25.6 weeks [PMID:2].", steps))
    assert row["success"] and row["literature_hit"] and row["read_hit"] is False


def test_a_failed_read_does_not_count_as_reading():
    steps = [step("search_papers", 1, pmids=["36624315"], query="MAGE-A4"),
             step("read_paper", 2, error=True, pmids=["36624315"], paper_id="36624315",
                  question="DoR")]
    assert score_task(FT_TASK, answer("No data.", steps))["read_hit"] is False


def test_read_hit_rate_only_appears_in_literature_reports():
    amr = summarize([score_task(HOST_TASK, answer("K. pneumoniae", GOOD_HOST_STEPS))])
    assert "read_hit_rate" not in amr
    steps = [step("search_papers", 1, pmids=["36624315"], query="q"),
             step("read_paper", 2, pmids=["36624315"], paper_id="36624315", question="q")]
    lit = summarize([score_task(FT_TASK, answer("25.6 weeks", steps))])
    assert lit["read_hit_rate"] == 1.0
    report = {"model": "m", "n_tasks": 1, "max_steps": 6, "token_budget": 12_000,
              "cassette_mode": "replay", "cassette_hits": 0, "cassette_misses": 2,
              "summary": amr, "per_task": []}
    assert "read_paper read" not in agent_table(report)


def test_literature_golden_set_is_valid():
    tasks = load_agent_golden(LITERATURE_GOLDEN_PATH)
    for t in tasks:
        named = {n for group in t.expect_tools for n in group.split("|")}
        assert named <= {SEARCH_PAPERS.name, READ_PAPER.name}, t.id
        if not t.should_abstain:
            assert t.relevant_pmids and t.answer_contains, t.id
    assert {t.category for t in tasks} == {"full_text", "out_of_scope"}
