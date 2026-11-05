"""API HTTP con FastAPI. Arrancar en local: `uv run uvicorn amr_agent.api.main:app --reload`."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from amr_agent.api.ratelimit import AskQuota
from amr_agent.config import get_settings
from amr_agent.db import apply_schema_async, make_pool
from amr_agent.embeddings import FastEmbedEmbedder
from amr_agent.llm import OpenAICompatibleLLM
from amr_agent.rag import RagAnswer, RagService, Source
from amr_agent.retrieval import Retriever, SearchBackend, SearchMode

log = logging.getLogger("amr_agent")

DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta una vez al arrancar: crea los recursos caros y los comparte entre peticiones."""
    settings = get_settings()
    await apply_schema_async(settings.database_url)
    pool = make_pool(settings.database_url)
    await pool.open()
    embedder = FastEmbedEmbedder(settings.embedding_model, threads=settings.embedding_threads)
    embedder.embed_query("warm-up")  # carga el modelo ya, no en la primera petición
    retriever = Retriever(pool, embedder, settings.keyword_weight, settings.keyword_max_df)
    llm = OpenAICompatibleLLM(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s
    )
    app.state.retriever = retriever
    app.state.rag = RagService(retriever, llm, top_k=settings.top_k)
    app.state.quota = AskQuota(settings.ask_per_ip_per_minute, settings.ask_per_day)
    yield
    await pool.close()


app = FastAPI(
    title="AMR Research Agent",
    description="RAG with citations over PubMed literature on antimicrobial resistance.",
    version="0.1.0",
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


def enforce_quota(request: Request, quota: Annotated[AskQuota, Depends(get_quota)]) -> None:
    # Con `--proxy-headers` (ver Dockerfile) client.host es la IP real que manda el proxy de
    # Render en X-Forwarded-For, no la del propio proxy.
    ip = request.client.host if request.client else "unknown"
    if blocked := quota.acquire(ip):
        reason, wait = blocked
        raise HTTPException(429, reason, headers={"Retry-After": str(math.ceil(wait))})


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=[
        "Which genes confer colistin resistance in E. coli?"])
    k: int = Field(default=5, ge=1, le=10, description="number of abstracts to retrieve")
    mode: SearchMode = "hybrid"


@app.get("/", include_in_schema=False)
async def demo() -> FileResponse:
    """Página de demo: HTML + JS sin dependencias que llama a /ask desde el navegador."""
    return FileResponse(DEMO_PAGE)


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
    try:
        return await rag.ask(body.question, k=body.k, mode=body.mode)
    except openai.RateLimitError as exc:
        raise HTTPException(503, "LLM free-tier rate limit reached, try again in a minute") from exc
    except openai.APIError as exc:
        log.exception("LLM provider error")
        raise HTTPException(502, "LLM provider error") from exc
