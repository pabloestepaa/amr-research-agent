import os

import pytest

from amr_agent.db import apply_schema_async, make_pool, upsert_articles
from amr_agent.ingest.pubmed import Article
from amr_agent.retrieval import Retriever
from tests.fakes import HashingEmbedder

CORPUS = [
    Article("1001", "Plasmid-mediated colistin resistance gene mcr-1 in Escherichia coli",
            "We report the mcr-1 gene on an IncI2 plasmid conferring polymyxin resistance in "
            "E. coli isolates from pigs.", "Lancet Infect Dis", 2016),
    Article("1002", "Carbapenemase-producing Enterobacterales and blaNDM-1",
            "NDM-1 metallo-beta-lactamase hydrolyses carbapenems such as imipenem and "
            "meropenem, leaving few therapeutic options.", "J Antimicrob Chemother", 2019),
    Article("1003", "Fluoroquinolone resistance through gyrA mutations",
            "Point mutations in the quinolone resistance-determining region of gyrA and parC "
            "reduce susceptibility to ciprofloxacin.", "Antimicrob Agents Chemother", 2018),
]


@pytest.fixture
async def retriever():
    """Retriever real contra un Postgres de test con un corpus pequeño y conocido."""
    url = os.environ.get("DATABASE_URL_TEST")
    if not url:
        pytest.skip("DATABASE_URL_TEST not set")
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(url) as conn:
        conn.execute("drop table if exists documents")
        conn.commit()
    await apply_schema_async(url)
    embedder = HashingEmbedder()
    with psycopg.connect(url) as conn:
        register_vector(conn)
        upsert_articles(conn, CORPUS, embedder.embed_documents(
            [a.text_for_embedding() for a in CORPUS]))

    pool = make_pool(url, max_size=2)
    await pool.open()
    yield Retriever(pool, embedder)
    await pool.close()
