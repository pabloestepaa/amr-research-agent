"""Evals de búsqueda: deterministas, sin LLM y gratis, así que se ejecutan en cada PR.

    uv run amr-eval-retrieval --load-corpus --check   # como en CI: snapshot + umbrales
    uv run amr-eval-retrieval --database-url postgresql://localhost/amr_agent  # corpus completo

Para cada pregunta del conjunto de referencia y cada modo de búsqueda (`hybrid`, `vector`,
`keyword`) se piden los 10 primeros resultados y se calculan recall@k, MRR y nDCG@k frente
a los PMIDs marcados como relevantes.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from amr_agent.evals.dataset import (
    CORPUS_PATH,
    EVALS_DIR,
    GOLDEN_PATH,
    RESULTS_DIR,
    GoldenItem,
    load_golden,
)
from amr_agent.evals.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from amr_agent.retrieval import SearchBackend, SearchMode

MODES: tuple[SearchMode, ...] = ("hybrid", "vector", "keyword")
KS = (1, 3, 5, 10)
THRESHOLDS_PATH = EVALS_DIR / "thresholds.json"


def score_ranking(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in KS}
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    scores["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return scores


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {key: round(mean([r[key] for r in rows]), 4) for key in keys}


async def evaluate(
    backend: SearchBackend, items: list[GoldenItem], modes: tuple[SearchMode, ...] = MODES
) -> dict:
    """Ejecuta la búsqueda para cada pregunta y modo y devuelve métricas globales y por tipo."""
    items = [i for i in items if i.in_scope]  # out_of_scope no tiene relevantes que medir
    report: dict = {"n_questions": len(items), "modes": {}, "per_question": []}
    for mode in modes:
        rows, by_type, latencies = [], defaultdict(list), []
        for item in items:
            t0 = time.perf_counter()
            hits = await backend.search(item.question, max(KS), mode)
            latencies.append((time.perf_counter() - t0) * 1000)
            ranked = [h.pmid for h in hits]
            scores = score_ranking(ranked, set(item.relevant_pmids))
            rows.append(scores)
            by_type[item.type].append(scores)
            report["per_question"].append(
                {"id": item.id, "mode": mode, "top5": ranked[:5], **scores}
            )
        report["modes"][mode] = {
            "overall": aggregate(rows),
            "by_type": {t: aggregate(r) | {"n": len(r)} for t, r in sorted(by_type.items())},
            "latency_ms_p50": round(sorted(latencies)[len(latencies) // 2], 1)
            if latencies else None,
        }
    return report


def check_thresholds(report: dict, thresholds: dict[str, dict[str, float]]) -> list[str]:
    """Devuelve los fallos (vacío si todo pasa). Formato: {"hybrid": {"recall@5": 0.8}}."""
    failures = []
    for mode, minimums in thresholds.items():
        overall = report["modes"].get(mode, {}).get("overall", {})
        for metric, minimum in minimums.items():
            value = overall.get(metric)
            if value is None or value < minimum:
                failures.append(f"{mode} {metric} = {value} < {minimum}")
    return failures


def load_snapshot(database_url: str, corpus_path: Path) -> int:
    """Carga el snapshot del corpus en una base de datos de evals, desde cero."""
    import psycopg

    from amr_agent.config import get_settings
    from amr_agent.db import apply_schema, refresh_term_stats, upsert_articles
    from amr_agent.embeddings import FastEmbedEmbedder
    from amr_agent.evals.dataset import load_corpus

    articles = load_corpus(corpus_path)
    embedder = FastEmbedEmbedder(get_settings().embedding_model)
    vectors = embedder.embed_documents([a.text_for_embedding() for a in articles])
    with psycopg.connect(database_url) as conn:
        conn.execute("drop table if exists documents")
        apply_schema(conn)
        n = upsert_articles(conn, articles, vectors)
        refresh_term_stats(conn)
        return n


def resolve_eval_db(arg: str | None, load_corpus: bool) -> str:
    from amr_agent.config import get_settings

    url = arg or os.environ.get("DATABASE_URL_EVAL")
    if not url:
        sys.exit("Set DATABASE_URL_EVAL or pass --database-url")
    # --load-corpus borra la tabla: nunca contra la base de datos de desarrollo/producción.
    if load_corpus and url == get_settings().database_url:
        sys.exit("Refusing to drop the main database: use a dedicated eval database")
    return url


async def run(database_url: str, keyword_weight: float, max_df: float) -> dict:
    from amr_agent.config import get_settings
    from amr_agent.db import make_pool
    from amr_agent.embeddings import FastEmbedEmbedder
    from amr_agent.retrieval import Retriever

    settings = get_settings()
    pool = make_pool(database_url, max_size=2)
    await pool.open()
    try:
        async with pool.connection() as conn:
            n_docs = (await (await conn.execute("select count(*) from documents")).fetchone())[0]
        retriever = Retriever(
            pool, FastEmbedEmbedder(settings.embedding_model), keyword_weight, max_df
        )
        report = await evaluate(retriever, load_golden(GOLDEN_PATH))
    finally:
        await pool.close()
    return {"corpus_size": n_docs, "embedding_model": settings.embedding_model,
            "keyword_weight": keyword_weight, "keyword_max_df": max_df} | report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrieval evals (recall@k, MRR, nDCG)")
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL_EVAL")
    parser.add_argument("--load-corpus", action="store_true",
                        help=f"drop and reload the eval DB from {CORPUS_PATH}")
    parser.add_argument("--check", action="store_true",
                        help=f"exit 1 if a metric falls below {THRESHOLDS_PATH}")
    # Por defecto, la misma configuración que la API (variables de entorno / .env).
    from amr_agent.config import get_settings

    settings = get_settings()
    parser.add_argument("--keyword-weight", type=float, default=settings.keyword_weight,
                        help="weight of the keyword ranking in hybrid RRF")
    parser.add_argument("--max-df", type=float, default=settings.keyword_max_df,
                        help="ignore keyword terms found in more than this share of abstracts")
    parser.add_argument("--name", default="retrieval", help="results file name (no extension)")
    args = parser.parse_args(argv)

    url = resolve_eval_db(args.database_url, args.load_corpus)
    if args.load_corpus:
        print(f"Loaded {load_snapshot(url, CORPUS_PATH)} abstracts from {CORPUS_PATH}")
    report = asyncio.run(run(url, args.keyword_weight, args.max_df))

    out = RESULTS_DIR / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    from amr_agent.evals.report import retrieval_table

    print(retrieval_table(report))
    print(f"\nSaved {out}")
    if args.check:
        failures = check_thresholds(report, json.loads(THRESHOLDS_PATH.read_text()))
        if failures:
            print("\nRetrieval quality regressed:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("All retrieval thresholds passed.")


if __name__ == "__main__":
    main()
