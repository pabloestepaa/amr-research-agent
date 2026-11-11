"""Trazas (telemetry.py): qué spans salen, colgando de quién y con qué atributos.

Los spans van a un exportador en memoria, sin red. El TracerProvider de OpenTelemetry es global y
solo se puede fijar una vez por proceso: lo fija el primer test que lo pide y los demás reutilizan
el mismo exportador, que se vacía antes de cada test.
"""

import json

import httpx2
import pytest
from amr_mcp import server as amr_server
from amr_mcp.catalog import Catalog
from mcp import Client
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from amr_agent.agent import Agent
from amr_agent.llm import OpenAICompatibleLLM
from amr_agent.rag import RagService
from amr_agent.telemetry import provider_name, setup_tracing, tracing_mode
from amr_agent.tools import MCP_TOOLS, LiteratureTools, McpTools, Toolset
from tests.fakes import FakeLLM, FakeRetriever, ScriptedLLM, make_hit, text_turn, tool_turn
from tests.test_agent import CATALOG

EXPORTER = InMemorySpanExporter()


@pytest.fixture
def spans(monkeypatch) -> InMemorySpanExporter:
    monkeypatch.setattr(amr_server.state, "catalog", Catalog.from_file(CATALOG))
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(provider)
    EXPORTER.clear()
    return EXPORTER


def by_name(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    out: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        out.setdefault(span.name, []).append(span)
    return out


def parent_of(span: ReadableSpan, spans: list[ReadableSpan]) -> ReadableSpan | None:
    return next((s for s in spans if span.parent and s.context.span_id == span.parent.span_id),
                None)


async def test_an_agent_question_is_one_trace_with_a_span_per_tool_call(spans):
    llm = ScriptedLLM(
        tool_turn(("search_literature", {"query": "NDM-1 Klebsiella"}),
                  ("get_amr_gene", {"name": "blaNDM-1"})),
        tool_turn(("search_literature", {"query": "NDM-1 Klebsiella"})),  # repetida: error
        text_turn("NDM-1 is a carbapenemase [PMID:1]."),
    )
    async with Client(amr_server.mcp) as client:
        mcp = await McpTools.load(client, allowed=MCP_TOOLS)
        tools = Toolset([LiteratureTools(FakeRetriever([make_hit("1")])), mcp])
        result = await Agent(llm, tools).run("¿Qué es NDM-1?")

    named = by_name(spans)
    [root] = named["invoke_agent amr-research-agent"]
    assert root.parent is None
    # Al conectar, el cliente MCP hace `initialize` y `tools/list` fuera de la pregunta: son
    # otras trazas. Todo lo de la pregunta comparte la del agente.
    finished = [s for s in spans.get_finished_spans()
                if s.context.trace_id == root.context.trace_id]
    attrs = root.attributes
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["amr_agent.mode"] == "amr"
    assert attrs["langfuse.observation.input"] == "¿Qué es NDM-1?"
    assert attrs["langfuse.observation.output"] == result.answer
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == result.answer
    assert (attrs["gen_ai.usage.input_tokens"], attrs["gen_ai.usage.output_tokens"]) == (300, 30)
    assert (attrs["amr_agent.stop_reason"], attrs["amr_agent.tool_calls"]) == ("answer", 3)

    searches = named["execute_tool search_literature"]
    assert [parent_of(s, finished) for s in searches] == [root, root]
    first, repeated = sorted(searches, key=lambda s: s.start_time)
    assert first.attributes["gen_ai.tool.call.id"] == "call_0"
    assert json.loads(first.attributes["gen_ai.tool.call.arguments"]) == {
        "query": "NDM-1 Klebsiella"}
    assert first.attributes["amr_agent.tool.pmids"] == 1
    assert first.status.status_code == StatusCode.UNSET
    assert repeated.status.status_code == StatusCode.ERROR
    assert repeated.attributes["error.type"] == "tool_error"
    # Ni resultados de herramientas ni prompts en la traza: solo pregunta, respuesta y argumentos.
    assert not any("gen_ai.tool.call.result" in s.attributes for s in finished)

    # El SDK de mcp crea su propio span y queda dentro de la herramienta que lo llamó. En
    # memoria es el del servidor (`tools/call ...`); por stdio, en la API, el del cliente
    # (`MCP send tools/call ...`), porque el subproceso no exporta trazas.
    [gene] = named["execute_tool get_amr_gene"]
    [mcp_span] = named["tools/call get_amr_gene"]
    assert parent_of(mcp_span, finished) == gene
    assert mcp_span.attributes["gen_ai.tool.name"] == "get_amr_gene"


async def test_each_llm_call_is_a_chat_span_with_the_provider_usage(spans):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    llm = OpenAICompatibleLLM("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b",
                              http_client=client)
    await llm.chat([{"role": "user", "content": "secret prompt"}], [])

    [span] = spans.get_finished_spans()
    assert span.name == "chat openai/gpt-oss-120b"
    assert span.kind == trace.SpanKind.CLIENT
    a = span.attributes
    assert (a["gen_ai.operation.name"], a["gen_ai.provider.name"]) == ("chat", "groq")
    assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == (12, 3)
    assert (a["gen_ai.response.id"], tuple(a["gen_ai.response.finish_reasons"])) == (
        "chatcmpl-1", ("stop",))
    assert "secret prompt" not in json.dumps(dict(a))  # los prompts no se guardan


async def test_a_failed_llm_call_is_marked_with_its_error_type(spans):
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(
        lambda r: httpx2.Response(400, json={"error": {"message": "bad"}})))
    llm = OpenAICompatibleLLM("https://api.groq.com/openai/v1", "k", "m", http_client=client)
    with pytest.raises(Exception):  # noqa: B017 - basta con que falle; importa el span
        await llm.chat([{"role": "user", "content": "x"}], [])
    [span] = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "BadRequestError"


async def test_ask_is_a_workflow_with_a_retrieval_span(spans):
    rag = RagService(FakeRetriever([make_hit("1"), make_hit("2")]),
                     FakeLLM("Answer [PMID:1] [PMID:9]."))
    await rag.ask("carbapenemases?", k=4, mode="vector")

    named = by_name(spans)
    [root] = named["invoke_workflow ask"]
    [retrieval] = named["retrieval"]
    assert retrieval.parent.span_id == root.context.span_id
    assert (retrieval.attributes["gen_ai.retrieval.top_k"],
            retrieval.attributes["amr_agent.search_mode"],
            retrieval.attributes["amr_agent.retrieval.hits"]) == (4, "vector", 2)
    assert root.attributes["langfuse.observation.input"] == "carbapenemases?"
    assert root.attributes["amr_agent.invalid_citations"] == 1  # PMID:9 no se recuperó


def test_tracing_is_off_unless_configured():
    assert setup_tracing({}) is None
    assert tracing_mode({}) == "off"
    assert tracing_mode({"OTEL_TRACES_EXPORTER": "console"}) == "console"
    assert tracing_mode({"OTEL_EXPORTER_OTLP_ENDPOINT": "https://x"}) == "otlp"


def test_provider_name_uses_the_convention_values():
    assert provider_name("https://api.groq.com/openai/v1") == "groq"
    assert provider_name("http://localhost:11434/v1") == "localhost"


def test_http_requests_alone_leave_no_spans(spans):
    # FastAPI trae sus propios spans HTTP; los desactivamos para que /health (Render lo llama
    # sin parar) y las visitas a la página no gasten unidades de Langfuse.
    from fastapi.testclient import TestClient

    from amr_agent.api.main import app

    assert TestClient(app).get("/health").status_code == 200
    assert spans.get_finished_spans() == ()


async def test_connecting_to_amr_mcp_is_one_trace(spans, monkeypatch):
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    from amr_agent.api import main

    monkeypatch.setattr(main, "amr_mcp_server", lambda: amr_server.mcp)  # en memoria
    settings = SimpleNamespace(mcp_timeout_s=5, agent_max_steps=5, agent_token_budget=12_000)
    async with AsyncExitStack() as stack:
        agent = await main.start_agent(stack, ScriptedLLM(), FakeRetriever([]), settings)
    assert agent is not None

    finished = spans.get_finished_spans()
    [root] = [s for s in finished if s.parent is None]
    assert root.name == "start amr-mcp"
    assert root.attributes["amr_agent.mcp.tools"] == len(MCP_TOOLS)
    assert len(finished) > 1 and {s.context.trace_id for s in finished} == {root.context.trace_id}


async def test_a_failed_amr_mcp_start_is_an_error_span(spans, monkeypatch):
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    from amr_agent.api import main

    def broken():
        raise FileNotFoundError("amr-mcp not installed")

    monkeypatch.setattr(main, "amr_mcp_server", broken)
    async with AsyncExitStack() as stack:
        agent = await main.start_agent(stack, ScriptedLLM(), FakeRetriever([]),
                                       SimpleNamespace(mcp_timeout_s=5))
    assert agent is None  # la API arranca sin /agent
    [span] = spans.get_finished_spans()
    assert (span.name, span.status.status_code) == ("start amr-mcp", StatusCode.ERROR)
    assert span.attributes["error.type"] == "FileNotFoundError"
