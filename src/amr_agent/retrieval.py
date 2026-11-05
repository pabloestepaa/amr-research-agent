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

# Postgres no tiene IDF: `ts_rank_cd` puntúa igual un término que sale en todos los abstracts
# ("resistance", "coli") que uno raro ("blaNDM-5"). En un corpus temático eso hunde la búsqueda
# por palabras clave: gana el documento que repite más palabras comunes. Arreglo barato: quitar
# de la consulta los lexemas cuya frecuencia documental (`term_stats.df`, fracción de abstracts
# que lo contienen) supera `max_df`, y unir los que quedan con OR.
# Si `term_stats` está vacía, no se descarta nada (coalesce a 0).
KEYWORD_SQL = """
with terms as (
    select t.word
    from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) as t(word)
    left join term_stats s on s.word = t.word
    where coalesce(s.df, 0) <= %(max_df)s
),
q as (
    select string_agg(quote_literal(word), ' | ')::tsquery as query from terms
)
select d.pmid, d.title, d.abstract, d.journal, d.year, ts_rank_cd(d.tsv, q.query) as score
from documents d, q
where q.query is not null and d.tsv @@ q.query
order by score desc
limit %(k)s
"""


def rrf_fuse(
    rankings: list[list[Hit]], top_n: int, k: int = 60, weights: list[float] | None = None
) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = Σ w_i / (k + posición de d en la lista i).

    Solo usa posiciones, no puntuaciones: así podemos mezclar una similitud coseno (0-1)
    con un ts_rank (escala arbitraria) sin tener que normalizarlas. k=60 es el valor del
    paper original (Cormack et al., 2009) y amortigua el peso de los primeros puestos.
    Los pesos `w_i` (1.0 por defecto) permiten fiarse más de una lista que de otra.
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.pmid] = scores.get(hit.pmid, 0.0) + weight / (k + rank)
            first_seen.setdefault(hit.pmid, hit)
    ordered = sorted(scores, key=lambda pmid: (-scores[pmid], pmid))
    return [replace(first_seen[pmid], score=scores[pmid]) for pmid in ordered[:top_n]]


class Retriever:
    def __init__(
        self, pool: AsyncConnectionPool, embedder: Embedder, keyword_weight: float = 1.0,
        max_df: float = 0.1,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        # Peso de la lista de palabras clave en la fusión (la vectorial pesa 1.0).
        self.keyword_weight = keyword_weight
        self.max_df = max_df  # 1.0 = no descartar ningún término

    async def _vector(self, query: str, k: int) -> list[Hit]:
        # Calcular el embedding usa CPU y es síncrono: lo mandamos a un hilo para no
        # bloquear el event loop de FastAPI mientras tanto.
        qvec = await asyncio.to_thread(self._embedder.embed_query, query)
        return await self._fetch(VECTOR_SQL, {"qvec": qvec, "k": k})

    async def _keyword(self, query: str, k: int) -> list[Hit]:
        return await self._fetch(KEYWORD_SQL, {"q": query, "k": k, "max_df": self.max_df})

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
        return rrf_fuse([vector_hits, keyword_hits], top_n=k, weights=[1.0, self.keyword_weight])
