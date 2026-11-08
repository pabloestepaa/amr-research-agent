"""Tests del bucle del agente con un LLM con guion (ScriptedLLM) y herramientas reales.

Las herramientas son las de verdad: search_literature sobre un FakeRetriever y amr-mcp en memoria
con el catálogo de ejemplo. Lo único falso es el modelo, que sigue un guion.
"""

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from amr_mcp import server as amr_server
from amr_mcp.catalog import Catalog
from mcp import Client

from amr_agent.agent import FINAL_NUDGE, FINAL_SYSTEM_PROMPT, Agent, strip_tool_markup
from amr_agent.rag import NO_EVIDENCE
from amr_agent.tools import MCP_TOOLS, LiteratureTools, McpTools, ToolResult, Toolset, ToolSpec
from tests.fakes import FakeRetriever, ScriptedLLM, make_hit, text_turn, tool_turn

CATALOG = Path(__file__).parent / "fixtures" / "catalog_sample.tsv"


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    monkeypatch.setattr(amr_server.state, "catalog", Catalog.from_file(CATALOG))


class StubTools:
    """Sustituye herramientas que irían a la red (NCBI) por respuestas fijas."""

    def __init__(self, responses: dict[str, ToolResult]) -> None:
        self.responses = responses

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(n, "stub", {"type": "object", "properties": {}}) for n in self.responses]

    async def call(self, name, arguments) -> ToolResult:
        return self.responses[name]


@asynccontextmanager
async def toolset(hits=(), stubs: dict[str, ToolResult] | None = None):
    # Context manager y no fixture con `yield`: ver la explicación en test_tools.py.
    stubs = stubs or {}
    async with Client(amr_server.mcp) as client:
        mcp = await McpTools.load(client, allowed=tuple(n for n in MCP_TOOLS if n not in stubs))
        yield Toolset([LiteratureTools(FakeRetriever(list(hits))), mcp, StubTools(stubs)])


async def test_answers_directly_when_no_tool_is_needed():
    llm = ScriptedLLM(text_turn(NO_EVIDENCE))
    async with toolset() as tools:
        result = await Agent(llm, tools).run("What is the capital of France?")
    assert result.answer == NO_EVIDENCE
    assert result.steps == [] and result.stop_reason == "answer"
    assert llm.calls[0]["tool_choice"] == "auto"
    assert len(llm.calls[0]["tools"]) == 5  # search_literature + 4 de amr-mcp


async def test_chains_tools_and_feeds_results_back_to_the_model():
    llm = ScriptedLLM(
        tool_turn(("get_amr_gene", {"name": "mcr-1.1"})),
        tool_turn(("get_pubmed_references", {"pmids": ["26603172"]})),
        text_turn("mcr-1 confers colistin resistance [PMID:26603172]."),
    )
    pubmed = ToolResult('{"articles":[{"pmid":"26603172"}]}', pmids=frozenset({"26603172"}))
    async with toolset(stubs={"get_pubmed_references": pubmed}) as tools:
        result = await Agent(llm, tools).run("What does mcr-1 do?")

    assert [s.tool for s in result.steps] == ["get_amr_gene", "get_pubmed_references"]
    assert [s.step for s in result.steps] == [1, 2]
    assert result.steps[0].pmids == ["26603172"]  # el pubmed_reference del catálogo

    # En la 2.ª llamada el modelo ve su petición y el resultado real del catálogo.
    second = llm.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_amr_gene"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_0"
    assert json.loads(second[-1]["content"])["entries"][0]["class"] == "COLISTIN"

    assert [c.pmid for c in result.citations] == ["26603172"]
    assert result.invalid_citations == []
    assert result.usage.llm_calls == 3 and result.usage.prompt_tokens == 300


async def test_citations_carry_the_title_a_tool_returned():
    llm = ScriptedLLM(
        tool_turn(("get_amr_gene", {"name": "mcr-1.1"})),
        tool_turn(("search_literature", {"query": "colistin resistance"})),
        text_turn("Colistin [PMID:26603172]; poultry [PMID:111]."),
    )
    async with toolset(hits=[make_hit("111", "Colistin resistance in poultry")]) as tools:
        result = await Agent(llm, tools).run("What does mcr-1 do?")

    by_pmid = {c.pmid: c for c in result.citations}
    assert by_pmid["26603172"].title is None  # el catálogo solo da el PMID
    assert by_pmid["111"].title == "Colistin resistance in poultry"
    assert by_pmid["111"].year == 2024


async def test_parallel_tool_calls_all_get_a_reply():
    llm = ScriptedLLM(
        tool_turn(("search_literature", {"query": "colistin resistance"}),
                  ("get_amr_gene", {"name": "blaKPC-2"})),
        text_turn("Done [PMID:111]."),
    )
    async with toolset([make_hit("111")]) as tools:
        result = await Agent(llm, tools).run("colistin and KPC")
    assert {s.tool for s in result.steps} == {"search_literature", "get_amr_gene"}
    assert {s.step for s in result.steps} == {1}
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1"]


async def test_only_pmids_returned_by_tools_count_as_citations():
    llm = ScriptedLLM(
        tool_turn(("search_literature", {"query": "mcr-1"})),
        text_turn("Seen [PMID:111], invented 【PMID:99999999】."),
    )
    async with toolset([make_hit("111")]) as tools:
        result = await Agent(llm, tools).run("mcr-1?")
    assert [c.pmid for c in result.citations] == ["111"]
    assert result.invalid_citations == ["99999999"]
    assert "[PMID:99999999]" in result.answer  # citas de ancho completo normalizadas


async def test_tool_errors_reach_the_model_and_the_trajectory():
    llm = ScriptedLLM(
        tool_turn(("get_amr_gene", {"name": "nope"})),
        tool_turn(("get_amr_gene", "{not json")),
        tool_turn(("delete_database", {})),
        text_turn(NO_EVIDENCE),
    )
    async with toolset() as tools:
        result = await Agent(llm, tools).run("nope?")
    assert [s.is_error for s in result.steps] == [True, True, True]
    assert result.steps[1].arguments is None
    tool_msgs = [m["content"] for m in llm.calls[3]["messages"] if m["role"] == "tool"]
    assert "search_amr_genes" in tool_msgs[0]  # la pista de amr-mcp
    assert "JSON object" in tool_msgs[1]
    assert "unknown tool" in tool_msgs[2]


async def test_forces_a_final_answer_at_the_step_limit():
    # Un modelo que seguiría pidiendo herramientas: con max_steps=3, la 3.ª llamada es una
    # conversación nueva SIN herramientas, con la pregunta y los resultados como texto. Con
    # tool_choice="none" y el historial, gpt-oss seguía intentando llamarlas (Groq daba 400).
    kpc = tool_turn(("get_amr_gene", {"name": "blaKPC-2"}))
    llm = ScriptedLLM(kpc, kpc, text_turn("Final answer."))
    async with toolset() as tools:
        result = await Agent(llm, tools, max_steps=3).run("KPC?")
    assert [bool(c["tools"]) for c in llm.calls] == [True, True, False]
    final = llm.calls[-1]["messages"]
    assert [m["role"] for m in final] == ["system", "user"]  # ni llamadas ni mensajes `tool`
    assert final[0]["content"] == FINAL_SYSTEM_PROMPT
    user = final[1]["content"]
    assert user.startswith("Question: KPC?") and user.endswith(FINAL_NUDGE)
    assert '[1] get_amr_gene {"name": "blaKPC-2"}' in user and "[2] get_amr_gene" in user
    assert "blaKPC-2" in user.split("[1] get_amr_gene", 1)[1]  # el resultado, no solo la llamada
    assert result.stop_reason == "max_steps"
    assert result.answer == "Final answer."
    assert len(result.steps) == 2


async def test_stops_calling_tools_when_the_token_budget_is_spent():
    llm = ScriptedLLM(tool_turn(("get_amr_gene", {"name": "blaKPC-2"}), tokens=5000),
                      text_turn("Short answer."))
    async with toolset() as tools:
        result = await Agent(llm, tools, token_budget=4000).run("KPC?")
    assert [bool(c["tools"]) for c in llm.calls] == [True, False]
    assert result.stop_reason == "token_budget"


async def test_final_answer_ignores_tool_calls_and_strips_leaked_tool_markup():
    # Lo visto con gpt-oss en la vuelta final: una llamada escrita dentro del texto.
    leaked = ("NDM-1 is a carbapenemase "
              '【to=functions.search_amr_genes <|constrain|>json {"query":"blaNDM-1"}】 '
              "[PMID:111].")
    final = replace(tool_turn(("get_amr_gene", {"name": "x"})), text=leaked)
    kpc = tool_turn(("get_amr_gene", {"name": "blaKPC-2"}))
    llm = ScriptedLLM(kpc, final)
    async with toolset() as tools:
        result = await Agent(llm, tools, max_steps=2).run("NDM?")
    assert len(llm.calls) == 2  # no ejecuta lo que pida en la vuelta final
    assert result.answer == "NDM-1 is a carbapenemase [PMID:111]."


def test_strip_tool_markup_removes_tool_names_used_as_citations():
    # host-02: "...Salmonella enterica.【get_amr_gene】【get_protein_hosts】"
    text = "E. coli 82 % [PMID:1]【get_amr_gene】【get_protein_hosts】 and more【PMID:2】."
    assert strip_tool_markup(text) == "E. coli 82 % [PMID:1] and more【PMID:2】."


def test_strip_tool_markup_leaves_normal_text_alone():
    text = "Resistance to colistin [PMID:1]; see <b>table</b> (2024) ."
    assert strip_tool_markup(text) == text


async def test_limits_tool_calls_per_step():
    # Argumentos distintos: tres llamadas iguales serían "repetidas", que se prueba aparte.
    calls = [("get_amr_gene", {"name": n}) for n in ("blaKPC-2", "blaKPC-3", "mcr-1.1")]
    llm = ScriptedLLM(tool_turn(*calls), text_turn("ok"))
    async with toolset() as tools:
        result = await Agent(llm, tools, max_calls_per_step=2).run("KPC?")
    assert [s.is_error for s in result.steps] == [False, False, True]


async def test_empty_final_text_becomes_the_abstention_message():
    llm = ScriptedLLM(text_turn(""))
    async with toolset() as tools:
        result = await Agent(llm, tools).run("?")
    assert result.answer == NO_EVIDENCE


async def test_an_identical_call_is_not_run_again():
    lit = ("search_literature", {"query": "colistin poultry", "k": 2})
    llm = ScriptedLLM(tool_turn(lit), tool_turn(lit), text_turn("ok"))
    hits = [make_hit("1"), make_hit("2")]
    retriever = FakeRetriever(hits)
    tools = Toolset([LiteratureTools(retriever)])
    result = await Agent(llm, tools).run("?")
    assert len(retriever.calls) == 1  # la segunda no llega a buscar
    assert [s.is_error for s in result.steps] == [False, True]
    repeated = llm.calls[2]["messages"][-1]["content"]
    assert repeated.startswith("ERROR: you already called search_literature")
    assert "step 1" in repeated


async def test_rephrased_searches_only_show_new_abstracts():
    # La misma búsqueda reformulada: el buscador devuelve casi lo mismo cada vez.
    llm = ScriptedLLM(tool_turn(("search_literature", {"query": "AI-2 pathway", "k": 2})),
                      tool_turn(("search_literature", {"query": "Lsr AI-2 pathway", "k": 3})),
                      tool_turn(("search_literature", {"query": "AI-2 Lsr", "k": 3})),
                      text_turn("Answer [PMID:3]."))
    tools = Toolset([LiteratureTools(FakeRetriever([make_hit(p) for p in "123"]))])
    result = await Agent(llm, tools, max_steps=5, max_literature_searches=3).run("?")
    second = json.loads(llm.calls[2]["messages"][-1]["content"])
    assert [r["pmid"] for r in second["results"]] == ["3"]
    assert second["note"] == "Already shown above, so omitted here: PMIDs 1, 2."
    third = json.loads(llm.calls[3]["messages"][-1]["content"])
    assert third["results"] == [] and third["note"].startswith("No new abstracts")
    assert result.citations[0].pmid == "3" and not result.invalid_citations
    assert [s.pmids for s in result.steps][-1] == ["1", "2", "3"]  # siguen siendo citables


async def test_literature_searches_are_capped_per_question():
    # gpt-oss seguía reformulando la búsqueda hasta el límite de pasos (decisión 31).
    llm = ScriptedLLM(tool_turn(("search_literature", {"query": "AI-2 pathway"})),
                      tool_turn(("search_literature", {"query": "Lsr AI-2"}),
                                ("search_literature", {"query": "Lsr system"})),
                      text_turn("Answer [PMID:1]."))
    retriever = FakeRetriever([make_hit("1")])
    result = await Agent(llm, Toolset([LiteratureTools(retriever)]),
                         max_literature_searches=2).run("?")
    assert len(retriever.calls) == 2  # la tercera no llega al buscador
    assert [s.is_error for s in result.steps] == [False, False, True]
    blocked = llm.calls[2]["messages"][-1]["content"]
    assert blocked.startswith("ERROR: at most 2 search_literature calls per question")
    assert result.stop_reason == "answer" and result.citations[0].pmid == "1"


async def test_a_repeated_search_does_not_use_up_the_limit():
    lit = ("search_literature", {"query": "colistin poultry"})
    llm = ScriptedLLM(tool_turn(lit), tool_turn(lit),
                      tool_turn(("search_literature", {"query": "mcr-1 chickens"})),
                      text_turn("ok"))
    retriever = FakeRetriever([make_hit("1")])
    await Agent(llm, Toolset([LiteratureTools(retriever)]), max_literature_searches=2).run("?")
    assert len(retriever.calls) == 2


async def test_the_prompt_states_the_search_limit_and_the_scope():
    llm = ScriptedLLM(text_turn(NO_EVIDENCE))
    await Agent(llm, Toolset([LiteratureTools(FakeRetriever([]))]),
                max_literature_searches=3).run("Write a sorting function")
    system = llm.calls[0]["messages"][0]["content"]
    assert "At most\n  3 searches per question" in system
    assert "writing\n  code" in system and "{" not in system
