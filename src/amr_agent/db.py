"""Esquema y acceso a Postgres + pgvector."""

from collections.abc import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg_pool import AsyncConnectionPool

from amr_agent.ingest.pubmed import Article

EMBEDDING_DIM = 384  # dimensión de BAAI/bge-small-en-v1.5

# Una fila por abstract: los abstracts son cortos (~250 palabras), así que no hace falta
# trocearlos (chunking). La columna `tsv` la calcula Postgres sola y alimenta la búsqueda
# por palabras clave; `embedding` alimenta la búsqueda semántica.
SCHEMA = f"""
create extension if not exists vector;

create table if not exists documents (
    pmid       text primary key,
    title      text not null,
    abstract   text not null,
    journal    text,
    year       int,
    embedding  vector({EMBEDDING_DIM}) not null,
    tsv        tsvector generated always as (
                   setweight(to_tsvector('english', title), 'A') ||
                   setweight(to_tsvector('english', abstract), 'B')
               ) stored,
    created_at timestamptz not null default now()
);

create index if not exists documents_embedding_idx
    on documents using hnsw (embedding vector_cosine_ops);
create index if not exists documents_tsv_idx on documents using gin (tsv);

-- Frecuencia documental de cada lexema (fracción de abstracts que lo contienen). La usa la
-- búsqueda por palabras clave para ignorar términos que aparecen en casi todo el corpus.
create table if not exists term_stats (
    word text primary key,
    df   real not null
);
"""

# ts_stat recorre todos los tsvector: con 3.000 abstracts tarda menos de un segundo.
REFRESH_TERM_STATS = """
delete from term_stats;
insert into term_stats (word, df)
select word, ndoc::real / (select count(*) from documents)
from ts_stat('select tsv from documents');
"""

UPSERT = """
insert into documents (pmid, title, abstract, journal, year, embedding)
values (%s, %s, %s, %s, %s, %s)
on conflict (pmid) do update set
    title = excluded.title, abstract = excluded.abstract, journal = excluded.journal,
    year = excluded.year, embedding = excluded.embedding
"""


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
    register_vector(conn)


def refresh_term_stats(conn: psycopg.Connection) -> None:
    """Recalcula `term_stats`. Hay que llamarlo después de cada ingesta."""
    conn.execute(REFRESH_TERM_STATS)
    conn.commit()


async def apply_schema_async(database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        await conn.execute(SCHEMA)
        await conn.commit()


def upsert_articles(
    conn: psycopg.Connection, articles: Sequence[Article], embeddings: Sequence[np.ndarray]
) -> int:
    rows = [
        (a.pmid, a.title, a.abstract, a.journal, a.year, emb)
        for a, emb in zip(articles, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)
    conn.commit()
    return len(rows)


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


def make_pool(database_url: str, max_size: int = 5) -> AsyncConnectionPool:
    """Pool de conexiones asíncrono.

    - `check`: Neon apaga la base de datos cuando está inactiva y corta las conexiones;
      comprobamos cada conexión antes de prestarla para no devolver una muerta.
    - `prepare_threshold=None`: desactiva los prepared statements, que dan problemas
      detrás de poolers como PgBouncer (el que usa Neon en su URL "pooled").
    """
    return AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=max_size,
        open=False,
        configure=_configure,
        check=AsyncConnectionPool.check_connection,
        kwargs={"prepare_threshold": None},
    )
