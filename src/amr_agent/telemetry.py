"""Trazas OpenTelemetry con las convenciones GenAI (decisión 39).

Una pregunta al agente queda así en Langfuse (o en cualquier backend que reciba OTLP):

    invoke_agent amr-research-agent        pregunta, respuesta, tokens totales, motivo de parada
    ├─ chat openai/gpt-oss-120b            tokens, finish_reason y latencia de cada vuelta
    ├─ execute_tool search_literature      argumentos, PMIDs devueltos, error
    ├─ execute_tool get_amr_gene
    │  └─ MCP send tools/call get_amr_gene    lo crea el SDK de mcp, no nosotros
    └─ chat openai/gpt-oss-120b

y una a /ask: `invoke_workflow ask` → `retrieval` + `chat`.

Tres familias de atributos:
- `gen_ai.*`: convenciones GenAI de OpenTelemetry (repo open-telemetry/semantic-conventions-genai,
  estado Development). Es lo que hace que cualquier backend entienda tokens, modelo y herramientas.
- `langfuse.*`: pistas para Langfuse (tipo de observación, entrada y salida); los demás backends las
  ignoran. Langfuse no documenta que lea `gen_ai.input.messages`, por eso van las dos.
- `amr_agent.*`: lo nuestro (modo, motivo de parada, PMIDs...).

Texto guardado: la pregunta, la respuesta final y los argumentos de las herramientas (las consultas
que escribe el modelo). Ni prompts, ni mensajes intermedios, ni resultados de herramientas: las
preguntas las escriben visitantes de una demo pública.

Sin configuración no se exporta nada: sin TracerProvider, la API de OpenTelemetry no hace nada y el
SDK ni se importa. Se activa con las variables estándar:
- OTEL_EXPORTER_OTLP_ENDPOINT (y OTEL_EXPORTER_OTLP_HEADERS): OTLP/HTTP, p. ej. a Langfuse.
- OTEL_TRACES_EXPORTER=console: las escribe en la consola, para probar en local.
"""

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, StatusCode

SERVICE_NAME = "amr-research-agent"
AGENT_NAME = "amr-research-agent"
# Valores de `gen_ai.provider.name` de la convención para los proveedores que podemos usar.
PROVIDERS = {"api.groq.com": "groq", "api.openai.com": "openai",
             "generativelanguage.googleapis.com": "gcp.gemini"}

tracer = trace.get_tracer("amr_agent")


def setup_tracing(environ: Mapping[str, str] = os.environ) -> Any:
    """Configura el TracerProvider global si hay dónde mandar las trazas. Devuelve el provider
    (hay que llamar a `shutdown()` al apagar para enviar los spans pendientes) o None."""
    if environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # En lotes y en otro hilo: exportar nunca retrasa la respuesta al visitante.
        processor = BatchSpanProcessor(OTLPSpanExporter())  # lee endpoint y cabeceras del entorno
    elif environ.get("OTEL_TRACES_EXPORTER") == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        processor = SimpleSpanProcessor(ConsoleSpanExporter())
    else:
        return None
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


def tracing_mode(environ: Mapping[str, str] = os.environ) -> str:
    """Para el log de arranque."""
    if environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        return "otlp"
    return "console" if environ.get("OTEL_TRACES_EXPORTER") == "console" else "off"


def provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or ""
    return PROVIDERS.get(host, host or "unknown")


@contextmanager
def _span(name: str, kind: SpanKind, attributes: dict[str, Any]) -> Iterator[Span]:
    """Un span que, si sale una excepción, la anota con `error.type` (lo pide la convención) y
    la deja seguir. OpenTelemetry ya guarda la excepción y marca el span como erróneo."""
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes) as span:
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def _text_messages(role: str, text: str, **extra: Any) -> str:
    """El formato de `gen_ai.input.messages` / `gen_ai.output.messages`: JSON con partes."""
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": text}], **extra}],
                      ensure_ascii=False)


def _input(text: str) -> dict[str, str]:
    return {"gen_ai.input.messages": _text_messages("user", text),
            "langfuse.observation.input": text}


def _set_output(span: Span, text: str) -> None:
    span.set_attribute("gen_ai.output.messages",
                       _text_messages("assistant", text, finish_reason="stop"))
    span.set_attribute("langfuse.observation.output", text)


# --- Llamadas al LLM ------------------------------------------------------------------------

@contextmanager
def chat_span(provider: str, model: str) -> Iterator[Span]:
    with _span(f"chat {model}", SpanKind.CLIENT, {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
        "gen_ai.request.temperature": 0.0,
    }) as span:
        yield span


def record_chat(span: Span, response: Any, retries: int) -> None:
    """Lo que devuelve el proveedor: modelo exacto, id, finish_reason y tokens."""
    span.set_attribute("gen_ai.response.model", response.model)
    if response.id:
        span.set_attribute("gen_ai.response.id", response.id)
    reasons = [c.finish_reason for c in response.choices if c.finish_reason]
    if reasons:
        span.set_attribute("gen_ai.response.finish_reasons", reasons)
    if response.usage:
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
    if retries:  # los 429 de Groq: cuánto de la latencia es espera
        span.set_attribute("amr_agent.llm.retries", retries)


# --- Agente y herramientas ------------------------------------------------------------------

@contextmanager
def agent_span(question: str, *, mode: str, provider: str, model: str) -> Iterator[Span]:
    with _span(f"invoke_agent {AGENT_NAME}", SpanKind.INTERNAL, {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": AGENT_NAME,
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
        "amr_agent.mode": mode,
        "langfuse.observation.type": "agent",
        **_input(question),
    }) as span:
        yield span


def record_agent(span: Span, *, answer: str, input_tokens: int, output_tokens: int,
                 llm_calls: int, tool_calls: int, stop_reason: str,
                 invalid_citations: int) -> None:
    _set_output(span, answer)
    span.set_attributes({
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "amr_agent.llm_calls": llm_calls,
        "amr_agent.tool_calls": tool_calls,
        "amr_agent.stop_reason": stop_reason,
        "amr_agent.invalid_citations": invalid_citations,
    })


@contextmanager
def tool_span(name: str, call_id: str, arguments: dict[str, Any] | None) -> Iterator[Span]:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.id": call_id,
        "gen_ai.tool.type": "function",
        "langfuse.observation.type": "tool",
    }
    if arguments is not None:
        attributes["gen_ai.tool.call.arguments"] = json.dumps(arguments, ensure_ascii=False)
    with _span(f"execute_tool {name}", SpanKind.INTERNAL, attributes) as span:
        yield span


def record_tool(span: Span, *, is_error: bool, error: str, pmids: int, truncated: bool,
                result_chars: int) -> None:
    span.set_attributes({"amr_agent.tool.pmids": pmids, "amr_agent.tool.truncated": truncated,
                         "amr_agent.tool.result_chars": result_chars})
    if is_error:
        # El agente sigue (el error vuelve al modelo como texto), pero en la traza debe verse.
        # El mensaje es nuestro ("ERROR: at most 2 search_papers calls..."), no del visitante.
        span.set_attribute("error.type", "tool_error")
        span.set_status(StatusCode.ERROR, error[:300])


# --- Arranque -------------------------------------------------------------------------------

@contextmanager
def startup_span(name: str) -> Iterator[Span]:
    """Agrupa lo que pasa al arrancar en una sola traza. Sin esto, cada paso de la conexión con
    amr-mcp (server/discover, initialize, tools/list...) era una traza suelta en Langfuse."""
    with _span(name, SpanKind.INTERNAL, {"langfuse.observation.type": "span"}) as span:
        yield span


def record_failure(span: Span, exc: BaseException) -> None:
    """Para errores que se capturan y no salen del span (el arranque sigue sin amr-mcp)."""
    span.set_attribute("error.type", type(exc).__qualname__)
    span.set_status(StatusCode.ERROR, str(exc)[:300])


# --- /ask -----------------------------------------------------------------------------------

@contextmanager
def workflow_span(question: str) -> Iterator[Span]:
    with _span("invoke_workflow ask", SpanKind.INTERNAL, {
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.agent.name": AGENT_NAME,  # la convención lo usa también para workflows
        "langfuse.observation.type": "chain",
        **_input(question),
    }) as span:
        yield span


def record_workflow(span: Span, answer: str, invalid_citations: int) -> None:
    _set_output(span, answer)
    span.set_attribute("amr_agent.invalid_citations", invalid_citations)


@contextmanager
def retrieval_span(top_k: int, mode: str) -> Iterator[Span]:
    with _span("retrieval", SpanKind.CLIENT, {
        "gen_ai.operation.name": "retrieval",
        "gen_ai.retrieval.top_k": top_k,
        "gen_ai.data_source.id": "amr-abstracts",  # la tabla documents de Postgres
        "amr_agent.search_mode": mode,
        "langfuse.observation.type": "retriever",
    }) as span:
        yield span
