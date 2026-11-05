"""Recuperación híbrida: palabras clave (full-text de Postgres) + semántica (pgvector).

Por qué híbrida: la búsqueda semántica entiende sinónimos y paráfrasis ("carbapenem
resistance" ~ "resistance to imipenem"), pero falla con identificadores exactos como
"blaNDM-1" o "mcr-1", que es justo donde la búsqueda por palabras clave brilla.
Las dos listas se combinan con Reciprocal Rank Fusion (RRF).
"""

import asyncio
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from amr_agent.embeddings import Embedder

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass(frozen=True)
class Hit:
    pmid: str
    title: str
    abstract: str
    journal: str | None
    year: int | None
    score: float


class SearchBackend(Protocol):
    async def search(self, query: str, k: int, mode: SearchMode = "hybrid") -> list[Hit]: ...


VECTOR_SQL = """
select pmid, title, abstract, journal, year, 1 - (embedding <=> %(qvec)s) as score
from documents
order by embedding <=> %(qvec)s
limit %(k)s
"""

# plainto_tsquery une los términos con AND ('a' & 'b'); una pregunta en lenguaje natural casi
# nunca contiene todas sus palabras en un abstract, así que lo convertimos en OR ('a' | 'b')
# y dejamos que ts_rank_cd premie a los documentos que cubren más términos.
KEYWORD_SQL = """
with q as (
    select to_tsquery('english',
        replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')) as query
)
select d.pmid, d.title, d.abstract, d.journal, d.year, ts_rank_cd(d.tsv, q.query) as score
from documents d, q
where d.tsv @@ q.query
order by score desc
limit %(k)s
"""


def rrf_fuse(rankings: list[list[Hit]], top_n: int, k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum(1 / (k + posición de d en cada lista)).

    Solo usa posiciones, no puntuaciones: así podemos mezclar una similitud coseno (0-1)
    con un ts_rank (escala arbitraria) sin tener que normalizarlas. k=60 es el valor del
    paper original (Cormack et al., 2009) y amortigua el peso de los primeros puestos.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.pmid] = scores.get(hit.pmid, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.pmid, hit)
    ordered = sorted(scores, key=lambda pmid: (-scores[pmid], pmid))
    return [replace(first_seen[pmid], score=scores[pmid]) for pmid in ordered[:top_n]]


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder) -> None:
        self._pool = pool
        self._embedder = embedder

    async def _vector(self, query: str, k: int) -> list[Hit]:
        # Calcular el embedding usa CPU y es síncrono: lo mandamos a un hilo para no
        # bloquear el event loop de FastAPI mientras tanto.
        qvec = await asyncio.to_thread(self._embedder.embed_query, query)
        return await self._fetch(VECTOR_SQL, {"qvec": qvec, "k": k})

    async def _keyword(self, query: str, k: int) -> list[Hit]:
        return await self._fetch(KEYWORD_SQL, {"q": query, "k": k})

    async def _fetch(self, sql: str, params: dict) -> list[Hit]:
        async with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return [Hit(**row) for row in await cur.fetchall()]

    async def search(self, query: str, k: int, mode: SearchMode = "hybrid") -> list[Hit]:
        if mode == "vector":
            return await self._vector(query, k)
        if mode == "keyword":
            return await self._keyword(query, k)
        candidates = max(4 * k, 20)  # cada lista aporta más candidatos de los que devolvemos
        vector_hits, keyword_hits = await asyncio.gather(
            self._vector(query, candidates), self._keyword(query, candidates)
        )
        return rrf_fuse([vector_hits, keyword_hits], top_n=k)
