"""CLI de ingesta: `uv run amr-ingest --max 3000`.

Descarga abstracts de PubMed, calcula sus embeddings y los guarda en Postgres.
Es idempotente (upsert por PMID): se puede relanzar sin duplicar nada.
"""

import argparse
import time

import httpx
import psycopg

from amr_agent.config import get_settings
from amr_agent.db import apply_schema, upsert_articles
from amr_agent.embeddings import FastEmbedEmbedder
from amr_agent.ingest.pubmed import Article, PubMedClient

DEFAULT_QUERY = (
    '("drug resistance, bacterial"[MeSH Terms] OR "antimicrobial resistance"[Title/Abstract])'
    ' AND "Escherichia coli"[MeSH Terms] AND hasabstract AND english[lang]'
    ' AND 2015:2026[dp]'
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest PubMed abstracts into pgvector")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max", type=int, default=3000, help="max PMIDs to fetch")
    parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    start, total = time.perf_counter(), 0

    def flush(buffer: list[Article]) -> int:
        vectors = embedder.embed_documents([a.text_for_embedding() for a in buffer])
        return upsert_articles(conn, buffer, vectors)

    with psycopg.connect(settings.database_url) as conn, httpx.Client(timeout=60) as http:
        apply_schema(conn)
        client = PubMedClient(http, settings.ncbi_api_key, settings.ncbi_email)
        buffer: list[Article] = []
        for article in client.iter_articles(args.query, args.max):
            buffer.append(article)
            if len(buffer) >= args.batch:
                total += flush(buffer)
                buffer = []
                print(f"  {total} abstracts indexed...", flush=True)
        if buffer:
            total += flush(buffer)

    print(f"Done: {total} abstracts in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
