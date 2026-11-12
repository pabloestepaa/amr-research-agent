"""API HTTP con FastAPI. Arrancar en local: `uv run uvicorn amr_agent.api.main:app --reload`."""

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import openai
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from mcp import Client
from pydantic import BaseModel, Field

from amr_agent import telemetry
from amr_agent.agent import LITERATURE_MODE, Agent, AgentAnswer
from amr_agent.api.diagnostics import memory_summary
from amr_agent.api.ratelimit import AskQuota
from amr_agent.config import get_settings
from amr_agent.db import apply_schema_async, make_pool
from amr_agent.embeddings import FastEmbedEmbedder
from amr_agent.llm import RETRY_WAIT, OpenAICompatibleLLM
from amr_agent.rag import RagAnswer, RagService, Source
from amr_agent.retrieval import Retriever, SearchBackend, SearchMode
from amr_agent.tools import (
    LiteratureTools,
    McpTools,
    PaperSearchTools,
    ReadPaperTools,
    Toolset,
    amr_mcp_server,
)

log = logging.getLogger("amr_agent")


def _setup_logging() -> None:
    """Que nuestros log.info salgan junto a los de uvicorn (por defecto solo saldrían WARNING+)."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta una vez al arrancar: crea los recursos caros y los comparte entre peticiones."""
    _setup_logging()
    settings = get_settings()
    # Tiempos por fase: en Render Free (0,15 CPU) el arranque tarda ~20 s y, si un despliegue
    # se atasca o falla, este log dice dónde.
    timings: dict[str, float] = {}
    t = time.perf_counter()
    await apply_schema_async(settings.database_url)
    async with AsyncExitStack() as stack:  # cierra todo lo abierto, en orden inverso, al apagar
        # Lo primero en abrirse es lo último en cerrarse: al apagar, el provider envía los spans
        # que queden en el lote después de que termine todo lo demás.
        if (tracer_provider := telemetry.setup_tracing()) is not None:
            stack.callback(tracer_provider.shutdown)
        pool = make_pool(settings.database_url)
        await pool.open()
        stack.push_async_callback(pool.close)
        timings["db"] = time.perf_counter() - t
        t = time.perf_counter()
        embedder = FastEmbedEmbedder(settings.embedding_model, threads=settings.embedding_threads)
        embedder.embed_query("warm-up")  # carga el modelo ya, no en la primera petición
        timings["model"] = time.perf_counter() - t
        retriever = Retriever(pool, embedder, settings.keyword_weight, settings.keyword_max_df)
        llm = OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s,
            max_retry_wait_s=settings.llm_max_retry_wait_s,
        )
        app.state.retriever = retriever
        app.state.rag = RagService(retriever, llm, top_k=settings.top_k)
        app.state.quota = AskQuota(settings.ask_per_ip_per_minute, settings.ask_per_day)
        app.state.agent_quota = AskQuota(settings.agent_per_ip_per_minute, settings.agent_per_day)
        app.state.literature_quota = AskQuota(settings.literature_per_ip_per_minute,
                                              settings.literature_per_day)
        app.state.agent = None
        if settings.agent_enabled:
            t = time.perf_counter()
            app.state.agent = await start_agent(stack, llm, retriever, settings)
            timings["amr-mcp"] = time.perf_counter() - t
        app.state.literature_agent = None
        if settings.literature_enabled:
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=30))
            app.state.literature_agent = literature_agent(http, settings)
        phases = ", ".join(f"{k} {v:.1f} s" for k, v in timings.items())
        log.info("Startup %.1f s (%s) · agent %s · literature %s · tracing %s · %s",
                 sum(timings.values()), phases, "on" if app.state.agent else "off",
                 "on" if app.state.literature_agent else "off", telemetry.tracing_mode(),
                 memory_summary())
        yield


async def start_agent(stack: AsyncExitStack, llm, retriever, settings) -> Agent | None:
    """Arranca amr-mcp (un subproceso para toda la vida de la API) y monta el agente.

    Si falla, la API arranca igualmente: /ask y /search no dependen de él y /agent responde 503.
    Mejor una demo a medias que ninguna. Todo el arranque es una sola traza (`start amr-mcp`).
    """
    with telemetry.startup_span("start amr-mcp") as span:
        try:
            client = await stack.enter_async_context(
                Client(amr_mcp_server(), read_timeout_seconds=settings.mcp_timeout_s))
            mcp_tools = await McpTools.load(client, timeout_s=settings.mcp_timeout_s)
        except Exception as exc:
            telemetry.record_failure(span, exc)
            log.exception("Could not start amr-mcp: /agent is disabled")
            return None
        span.set_attribute("amr_agent.mcp.tools", len(mcp_tools.specs()))
        # Cargamos el catálogo ya (en la imagen Docker viene incluido): la primera pregunta no
        # espera y, si no está, lo vemos en los logs al arrancar. Sin catálogo el agente sigue
        # siendo útil (literatura y especies hospedadoras), así que solo avisamos.
        try:
            warm = await client.call_tool("catalog_info", {})
            if warm.is_error:
                span.set_attribute("amr_agent.mcp.catalog", "unavailable")
                log.warning("amr-mcp catalog unavailable: %s", warm.content)
        except Exception:
            span.set_attribute("amr_agent.mcp.catalog", "failed")
            log.warning("amr-mcp catalog warm-up failed", exc_info=True)
    tools = Toolset([LiteratureTools(retriever), mcp_tools])
    return Agent(llm, tools, max_steps=settings.agent_max_steps,
                 token_budget=settings.agent_token_budget)


def literature_agent(http: httpx.AsyncClient, settings) -> Agent:
    """El agente del modo literatura: PubMed + texto completo de PMC, sin amr-mcp ni BD.

    Lleva su propio cliente del LLM porque espera más a Groq (ver config.py): una pregunta
    gasta el doble de lo que el plan gratuito deja por minuto.
    """
    llm = OpenAICompatibleLLM(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s,
        max_retry_wait_s=settings.literature_max_retry_wait_s,
    )
    tools = Toolset([PaperSearchTools(http, settings.ncbi_api_key, settings.ncbi_email),
                     ReadPaperTools(http, settings.ncbi_api_key, settings.ncbi_email)])
    return Agent(llm, tools, mode=LITERATURE_MODE,
                 token_budget=settings.literature_token_budget)


app = FastAPI(
    title="AMR Research Agent",
    description=("RAG with citations over PubMed literature on antimicrobial resistance, a "
                 "tool-calling agent that also queries NCBI databases through an MCP server, "
                 "and a literature mode that searches PubMed and reads full texts from PMC."),
    version="0.3.0",
    lifespan=lifespan,
    # FastAPI crea sus propios spans (POST /agent, fastapi.endpoint...) en cuanto hay un
    # TracerProvider. Los quitamos: cada visita a la página y cada /health de Render serían
    # una traza más contra las 50.000 unidades/mes de Langfuse, y Langfuse toma la entrada y la
    # salida de la traza del span raíz, que así es invoke_agent con la pregunta (decisión 39).
    telemetry={"tracing": False},
)


# Inyección de dependencias: los endpoints piden un servicio y FastAPI se lo da.
# En los tests se sustituyen por versiones falsas con `app.dependency_overrides`.
def get_rag(request: Request) -> RagService:
    return request.app.state.rag


def get_retriever(request: Request) -> SearchBackend:
    return request.app.state.retriever


def get_quota(request: Request) -> AskQuota:
    return request.app.state.quota


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=[
        "Which species carry blaKPC-2 most often, and what does recent literature say about "
        "treating KPC infections?"])
    mode: Literal["amr", "literature"] = Field(
        default="amr",
        description="amr: indexed AMR abstracts + NCBI gene catalog (amr-mcp). literature: any "
                    "biomedical question, searches PubMed and reads full texts from PubMed "
                    "Central (slower, ~15,000 tokens).")


# Las dependencias del agente leen el modo del cuerpo: FastAPI lo comparte con el endpoint.
def get_agent_quota(request: Request, body: AgentRequest) -> AskQuota:
    state = request.app.state
    return state.literature_quota if body.mode == "literature" else state.agent_quota


def get_agent(request: Request, body: AgentRequest) -> Agent:
    if body.mode == "literature":
        agent = getattr(request.app.state, "literature_agent", None)
    else:
        agent = request.app.state.agent
    if agent is None:
        raise HTTPException(503, "The agent is not available right now; /ask still works")
    return agent


def _check_quota(request: Request, quota: AskQuota) -> None:
    # Con `--proxy-headers` (ver Dockerfile) client.host es la IP real que manda el proxy de
    # Render en X-Forwarded-For, no la del propio proxy.
    ip = request.client.host if request.client else "unknown"
    if blocked := quota.acquire(ip):
        reason, wait = blocked
        raise HTTPException(429, reason, headers={"Retry-After": str(math.ceil(wait))})


def enforce_quota(request: Request, quota: Annotated[AskQuota, Depends(get_quota)]) -> None:
    _check_quota(request, quota)


def enforce_agent_quota(
    request: Request, quota: Annotated[AskQuota, Depends(get_agent_quota)]
) -> None:
    _check_quota(request, quota)


def llm_http_error(exc: openai.APIError) -> HTTPException:
    """El error del proveedor del LLM como respuesta HTTP con sentido."""
    if isinstance(exc, openai.RateLimitError):
        # Groq Free tiene dos límites: tokens por minuto y por día (TPD, ventana móvil de 24 h).
        # Decir "en un minuto" cuando es el diario manda al visitante a fallar otra vez.
        if "per day" in str(exc):
            return HTTPException(503, "LLM free-tier daily token limit reached, try again later")
        return HTTPException(503, "LLM free-tier rate limit reached, try again in a minute")
    log.exception("LLM provider error")
    return HTTPException(502, "LLM provider error")


@contextmanager
def llm_errors():
    try:
        yield
    except openai.APIError as exc:
        raise llm_http_error(exc) from exc


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=[
        "Which genes confer colistin resistance in E. coli?"])
    k: int = Field(default=5, ge=1, le=10, description="number of abstracts to retrieve")
    mode: SearchMode = "hybrid"


@app.get("/", include_in_schema=False)
async def demo() -> FileResponse:
    """Página de demo: HTML + JS sin dependencias que llama a /ask desde el navegador."""
    # no-cache: el navegador revalida (ETag) y nadie se queda con la página de antes de un deploy.
    return FileResponse(DEMO_PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health() -> dict[str, str]:
    # No toca la base de datos a propósito: Render llama a este endpoint a menudo y, si
    # consultara Neon, impediría que se apague y gastaría las horas de cómputo gratuitas.
    return {"status": "ok"}


@app.get("/search", response_model=list[Source])
async def search(
    retriever: Annotated[SearchBackend, Depends(get_retriever)],
    q: Annotated[str, Query(min_length=3, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    mode: SearchMode = "hybrid",
) -> list[Source]:
    """Solo recuperación, sin LLM: rápido y sin gastar cuota."""
    return [Source.from_hit(h) for h in await retriever.search(q, k, mode)]


@app.post("/ask", response_model=RagAnswer, dependencies=[Depends(enforce_quota)])
async def ask(body: AskRequest, rag: Annotated[RagService, Depends(get_rag)]) -> RagAnswer:
    with llm_errors():
        return await rag.ask(body.question, k=body.k, mode=body.mode)


# get_agent va primero en `dependencies`: si el agente no está disponible, la petición no
# llega a gastar cuota. FastAPI cachea la dependencia, así que no se ejecuta dos veces.
@app.post("/agent", response_model=AgentAnswer,
          dependencies=[Depends(get_agent), Depends(enforce_agent_quota)])
async def agent(body: AgentRequest, agent: Annotated[Agent, Depends(get_agent)]) -> AgentAnswer:
    """Agente con herramientas. Modo `amr`: literatura indexada + catálogo de genes y proteínas de
    NCBI (MCP). Modo `literature`: busca en PubMed y lee el texto completo de PMC.

    Devuelve la trayectoria completa (qué herramientas usó, con qué argumentos y cuánto tardó).
    """
    with llm_errors():
        return await agent.run(body.question)


@app.post("/agent/stream", response_class=EventSourceResponse,
          dependencies=[Depends(get_agent), Depends(enforce_agent_quota)])
async def agent_stream(
    body: AgentRequest, agent: Annotated[Agent, Depends(get_agent)]
) -> AsyncIterable[ServerSentEvent]:
    """Lo mismo que `/agent`, contado mientras ocurre (Server-Sent Events).

    Eventos: `round` (empieza una llamada al LLM; `final` si es la respuesta forzada sin
    herramientas), `wait` (espera la cuota del LLM, en segundos), `step` (una herramienta ha
    terminado: un paso de la trayectoria), `answer` (la respuesta, igual que la de `/agent`) y
    `error` (`status` y `detail`, los mismos que daría `/agent`). La cuota y los errores
    anteriores al primer evento (422, 429, 503) son respuestas HTTP normales.
    """
    events: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    def emit(event: str, data: Any) -> None:
        events.put_nowait(ServerSentEvent(event=event, data=data))

    async def run() -> None:
        # Dentro de la tarea: la variable de contexto solo vale para esta petición.
        RETRY_WAIT.set(lambda seconds: emit("wait", {"seconds": round(seconds, 1)}))
        try:
            emit("answer", await agent.run(body.question, on_event=emit))
        except openai.APIError as exc:
            error = llm_http_error(exc)
            emit("error", {"status": error.status_code, "detail": error.detail})
        except Exception:  # noqa: BLE001 - con el stream abierto ya no hay código HTTP que dar
            log.exception("Agent failed while streaming")
            emit("error", {"status": 500, "detail": "Internal error"})
        finally:
            events.put_nowait(None)

    task = asyncio.create_task(run())
    try:
        while (event := await events.get()) is not None:
            yield event
    finally:
        task.cancel()  # si el visitante cierra la página, el agente deja de gastar cuota
