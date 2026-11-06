"""API HTTP con FastAPI. Arrancar en local: `uv run uvicorn amr_agent.api.main:app --reload`."""

import logging
import math
import time
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from mcp import Client
from pydantic import BaseModel, Field

from amr_agent.agent import Agent, AgentAnswer
from amr_agent.api.diagnostics import memory_summary
from amr_agent.api.ratelimit import AskQuota
from amr_agent.config import get_settings
from amr_agent.db import apply_schema_async, make_pool
from amr_agent.embeddings import FastEmbedEmbedder
from amr_agent.llm import OpenAICompatibleLLM
from amr_agent.rag import RagAnswer, RagService, Source
from amr_agent.retrieval import Retriever, SearchBackend, SearchMode
from amr_agent.tools import LiteratureTools, McpTools, Toolset, amr_mcp_server

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
        app.state.agent = None
        if settings.agent_enabled:
            t = time.perf_counter()
            app.state.agent = await start_agent(stack, llm, retriever, settings)
            timings["amr-mcp"] = time.perf_counter() - t
        phases = ", ".join(f"{k} {v:.1f} s" for k, v in timings.items())
        log.info("Startup %.1f s (%s) · agent %s · %s", sum(timings.values()), phases,
                 "on" if app.state.agent else "off", memory_summary())
        yield


async def start_agent(stack: AsyncExitStack, llm, retriever, settings) -> Agent | None:
    """Arranca amr-mcp (un subproceso para toda la vida de la API) y monta el agente.

    Si falla, la API arranca igualmente: /ask y /search no dependen de él y /agent responde 503.
    Mejor una demo a medias que ninguna.
    """
    try:
        client = await stack.enter_async_context(
            Client(amr_mcp_server(), read_timeout_seconds=settings.mcp_timeout_s))
        mcp_tools = await McpTools.load(client, timeout_s=settings.mcp_timeout_s)
    except Exception:
        log.exception("Could not start amr-mcp: /agent is disabled")
        return None
    # Cargamos el catálogo ya (en la imagen Docker viene incluido): la primera pregunta no espera
    # y, si no está, lo vemos en los logs al arrancar. Sin catálogo el agente sigue siendo útil
    # (literatura y especies hospedadoras), así que solo avisamos.
    try:
        warm = await client.call_tool("catalog_info", {})
        if warm.is_error:
            log.warning("amr-mcp catalog unavailable: %s", warm.content)
    except Exception:
        log.warning("amr-mcp catalog warm-up failed", exc_info=True)
    tools = Toolset([LiteratureTools(retriever), mcp_tools])
    return Agent(llm, tools, max_steps=settings.agent_max_steps,
                 token_budget=settings.agent_token_budget)


app = FastAPI(
    title="AMR Research Agent",
    description=("RAG with citations over PubMed literature on antimicrobial resistance, and a "
                 "tool-calling agent that also queries NCBI databases through an MCP server."),
    version="0.2.0",
    lifespan=lifespan,
)


# Inyección de dependencias: los endpoints piden un servicio y FastAPI se lo da.
# En los tests se sustituyen por versiones falsas con `app.dependency_overrides`.
def get_rag(request: Request) -> RagService:
    return request.app.state.rag


def get_retriever(request: Request) -> SearchBackend:
    return request.app.state.retriever


def get_quota(request: Request) -> AskQuota:
    return request.app.state.quota


def get_agent_quota(request: Request) -> AskQuota:
    return request.app.state.agent_quota


def get_agent(request: Request) -> Agent:
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


@contextmanager
def llm_errors():
    """Traduce los errores del proveedor del LLM a respuestas HTTP con sentido."""
    try:
        yield
    except openai.RateLimitError as exc:
        # Groq Free tiene dos límites: tokens por minuto y por día (TPD, ventana móvil de 24 h).
        # Decir "en un minuto" cuando es el diario manda al visitante a fallar otra vez.
        if "per day" in str(exc):
            raise HTTPException(
                503, "LLM free-tier daily token limit reached, try again later") from exc
        raise HTTPException(503, "LLM free-tier rate limit reached, try again in a minute") from exc
    except openai.APIError as exc:
        log.exception("LLM provider error")
        raise HTTPException(502, "LLM provider error") from exc


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


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=[
        "Which species carry blaKPC-2 most often, and what does recent literature say about "
        "treating KPC infections?"])


# get_agent va primero en `dependencies`: si el agente no está disponible, la petición no
# llega a gastar cuota. FastAPI cachea la dependencia, así que no se ejecuta dos veces.
@app.post("/agent", response_model=AgentAnswer,
          dependencies=[Depends(get_agent), Depends(enforce_agent_quota)])
async def agent(body: AgentRequest, agent: Annotated[Agent, Depends(get_agent)]) -> AgentAnswer:
    """Agente con herramientas: literatura indexada + catálogo de genes y proteínas de NCBI (MCP).

    Devuelve la trayectoria completa (qué herramientas usó, con qué argumentos y cuánto tardó).
    """
    with llm_errors():
        return await agent.run(body.question)
