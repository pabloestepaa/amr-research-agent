"""CLI de ingesta: `uv run amr-ingest --max 3000`.

Descarga abstracts de PubMed, calcula sus embeddings y los guarda en Postgres.
Es idempotente (upsert por PMID): se puede relanzar sin duplicar nada.
"""

import argparse
import time

import httpx
import psycopg

from amr_agent.config import get_settings
from amr_agent.db import apply_schema, refresh_term_stats, upsert_articles
from amr_agent.embeddings import FastEmbedEmbedder
from amr_agent.ingest.pubmed import Article, PubMedClient, PubMedError, search_europepmc

PUBMED_QUERY = (
    '("drug resistance, bacterial"[MeSH Terms] OR "antimicrobial resistance"[Title/Abstract])'
    ' AND "Escherichia coli"[MeSH Terms] AND hasabstract AND english[lang]'
    ' AND 2015:2026[dp]'
)
# La misma intención expresada con la sintaxis de Europe PMC.
EUROPEPMC_QUERY = (
    'TITLE_ABS:"Escherichia coli" AND TITLE_ABS:(resistance AND (antibiotic OR antimicrobial'
    ' OR carbapenem OR colistin OR ESBL)) AND SRC:MED AND HAS_ABSTRACT:Y AND LANG:eng'
    ' AND PUB_YEAR:[2015 TO 2026]'
)


def find_pmids(client: PubMedClient, http: httpx.Client, source: str, n: int) -> list[str]:
    if source == "pubmed":
        try:
            return client.search(PUBMED_QUERY, n)
        except (PubMedError, httpx.HTTPError) as exc:
            print(f"PubMed search failed ({exc!s:.80}); falling back to Europe PMC")
    return search_europepmc(http, EUROPEPMC_QUERY, n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest PubMed abstracts into pgvector")
    parser.add_argument("--max", type=int, default=3000, help="max PMIDs to fetch")
    parser.add_argument("--search-with", choices=["pubmed", "europepmc"], default="pubmed")
    parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    start, total = time.perf_counter(), 0

    with psycopg.connect(settings.database_url) as conn, httpx.Client(timeout=60) as http:
        apply_schema(conn)
        client = PubMedClient(http, settings.ncbi_api_key, settings.ncbi_email)
        pmids = find_pmids(client, http, args.search_with, args.max)
        print(f"Found {len(pmids)} PMIDs; fetching abstracts from PubMed...")

        def flush(buffer: list[Article]) -> int:
            vectors = embedder.embed_documents([a.text_for_embedding() for a in buffer])
            return upsert_articles(conn, buffer, vectors)

        buffer: list[Article] = []
        for article in client.iter_articles(pmids):
            buffer.append(article)
            if len(buffer) >= args.batch:
                total += flush(buffer)
                buffer = []
                print(f"  {total} abstracts indexed...", flush=True)
        if buffer:
            total += flush(buffer)
        refresh_term_stats(conn)

    print(f"Done: {total} abstracts in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
