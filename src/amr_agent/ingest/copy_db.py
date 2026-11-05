"""Copia el corpus ya indexado de una base de datos a otra: `uv run amr-copy-db --target URL`.

Para qué: cargar Neon (producción) con los abstracts que ya están en el Postgres local, sin
volver a descargarlos de PubMed ni recalcular los embeddings (los vectores se copian tal cual).

Cómo: `COPY ... TO STDOUT` en el origen y `COPY ... FROM STDIN` a una tabla temporal en el
destino, y de ahí un upsert por PMID. Es idempotente: se puede relanzar sin duplicar nada.
Al final recalcula `term_stats`, que la búsqueda por palabras clave necesita.
"""

import argparse
import time

import psycopg

from amr_agent.config import get_settings
from amr_agent.db import apply_schema, refresh_term_stats

COLUMNS = "pmid, title, abstract, journal, year, embedding"  # `tsv` la recalcula Postgres

MERGE = f"""
insert into documents ({COLUMNS})
select {COLUMNS} from incoming
on conflict (pmid) do update set
    title = excluded.title, abstract = excluded.abstract, journal = excluded.journal,
    year = excluded.year, embedding = excluded.embedding
"""


def copy_documents(source_url: str, target_url: str) -> int:
    with psycopg.connect(source_url) as src, psycopg.connect(target_url) as dst:
        apply_schema(dst)
        dst.execute("create temp table incoming (like documents including defaults) "
                    "on commit drop")
        # Formato texto: no depende de que las dos BD tengan la misma versión de pgvector.
        with (src.cursor().copy(f"copy documents ({COLUMNS}) to stdout") as out,
              dst.cursor().copy(f"copy incoming ({COLUMNS}) from stdin") as into):
            for chunk in out:
                into.write(chunk)
        n = dst.execute(MERGE).rowcount
        dst.commit()
        refresh_term_stats(dst)
        return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy indexed abstracts to another database")
    parser.add_argument("--source", default=None, help="source URL (default: DATABASE_URL)")
    parser.add_argument("--target", required=True,
                        help="target URL; for Neon use the direct (non-pooled) connection")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    n = copy_documents(args.source or get_settings().database_url, args.target)
    print(f"Copied {n} abstracts in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
