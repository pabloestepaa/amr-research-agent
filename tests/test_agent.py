"""Tests del bucle del agente con un LLM con guion (ScriptedLLM) y herramientas reales.

Las herramientas son las de verdad: search_literature sobre un FakeRetriever y amr-mcp en memoria
con el catálogo de ejemplo. Lo único falso es el modelo, que sigue un guion.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from amr_mcp import server as amr_server
from amr_mcp.catalog import Catalog
from mcp import Client

from amr_agent.agent import FINAL_NUDGE, Agent
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
    # Un modelo que seguiría pidiendo herramientas: con max_steps=3, la 3.ª llamada ya no las
    # permite (tool_choice="none") y lleva la instrucción de responder.
    kpc = tool_turn(("get_amr_gene", {"name": "blaKPC-2"}))
    llm = ScriptedLLM(kpc, kpc, text_turn("Final answer."))
    async with toolset() as tools:
        result = await Agent(llm, tools, max_steps=3).run("KPC?")
    assert [c["tool_choice"] for c in llm.calls] == ["auto", "auto", "none"]
    assert llm.calls[-1]["messages"][-1] == {"role": "user", "content": FINAL_NUDGE}
    assert result.stop_reason == "max_steps"
    assert result.answer == "Final answer."
    assert len(result.steps) == 2


async def test_stops_calling_tools_when_the_token_budget_is_spent():
    llm = ScriptedLLM(tool_turn(("get_amr_gene", {"name": "blaKPC-2"}), tokens=5000),
                      text_turn("Short answer."))
    async with toolset() as tools:
        result = await Agent(llm, tools, token_budget=4000).run("KPC?")
    assert [c["tool_choice"] for c in llm.calls] == ["auto", "none"]
    assert result.stop_reason == "token_budget"


async def test_limits_tool_calls_per_step():
    calls = [("get_amr_gene", {"name": "blaKPC-2"})] * 3
    llm = ScriptedLLM(tool_turn(*calls), text_turn("ok"))
    async with toolset() as tools:
        result = await Agent(llm, tools, max_calls_per_step=2).run("KPC?")
    assert [s.is_error for s in result.steps] == [False, False, True]


async def test_empty_final_text_becomes_the_abstention_message():
    llm = ScriptedLLM(text_turn(""))
    async with toolset() as tools:
        result = await Agent(llm, tools).run("?")
    assert result.answer == NO_EVIDENCE
