"""Construcción del conjunto de referencia y del snapshot del corpus (se ejecuta pocas veces).

    uv run amr-eval-build questions   # un LLM propone preguntas -> evals/golden.jsonl (revisar)
    uv run amr-eval-build snapshot    # relevantes + distractores -> evals/corpus.jsonl

Ambos leen el corpus completo de DATABASE_URL (los 2.999 abstracts de desarrollo).

Por qué semiautomático: escribir 40 preguntas a mano lleva horas; pedírselas a un LLM es
rápido pero puede producir preguntas malas o demasiado fáciles. El LLM propone y una persona
revisa cada pregunta (campo `reviewed`).

Por qué distractores "difíciles" en el snapshot: si el snapshot tuviera solo los abstracts
relevantes más documentos al azar, cualquier búsqueda acertaría y la eval no mediría nada.
Por eso se incluyen los documentos que los tres modos de búsqueda devuelven para cada
pregunta en el corpus completo: son justo los que pueden confundir al buscador.
"""

import argparse
import asyncio
import json
import random
import re

import psycopg
from psycopg.rows import dict_row

from amr_agent.config import get_settings
from amr_agent.evals.dataset import CORPUS_PATH, GOLDEN_PATH, GoldenItem, load_golden, write_jsonl

SEED = 2026
# Nombres de genes/enzimas de resistencia: blaNDM-5, blaCTX-M-15, mcr-1, KPC-2, OXA-48, qnrS...
GENE_RE = re.compile(r"\b(bla[A-Z]{2,}[\w-]*|mcr-\d+(\.\d+)?|(NDM|KPC|OXA|CTX-M|VIM|IMP)-\d+"
                     r"|qnr[A-Z]\d*|tet\([A-Z]\)|aac\(6'\)-Ib(-cr)?|fosA\d*|armA|rmtB)")

PROMPT_COMMON = """You write evaluation questions for a search engine over ~3,000 PubMed abstracts
about antimicrobial resistance in Escherichia coli. Given ONE abstract, write ONE question that:
- can be answered from this abstract alone, and is specific enough that this abstract would be
  among the best matches in the whole collection (not a generic question any abstract answers);
- is phrased as a real researcher would ask it, without "this study", "the authors" or "the
  abstract", in English, at most 30 words;
- does not copy sentences from the abstract.
{extra}
Reply with JSON only: {{"question": "..."}}"""

EXTRA = {
    "gene_specific": "The question MUST name the exact gene or enzyme (for example {genes}) "
                     "as written in the abstract.",
    "factual": "Ask about a concrete finding (a prevalence, a mechanism, a risk factor, a "
               "comparison). Using the key technical terms is fine.",
    "paraphrase": "Do NOT reuse the distinctive words of the abstract: use synonyms, describe "
                  "concepts instead of naming them, name drugs instead of drug classes or vice "
                  "versa. It should read like someone who never saw the abstract.",
}

# Escritas a mano: el corpus no las responde y lo correcto es abstenerse. Algunas están
# cerca del dominio (otros patógenos, otros fármacos) para que la tentación de contestar
# con conocimiento propio sea real.
OUT_OF_SCOPE = [
    "How does Plasmodium falciparum become resistant to artemisinin?",
    "Which antifungal drugs remain effective against Candida auris?",
    "How is HIV resistance to integrase inhibitors detected in the clinic?",
    "What is the mechanism of action of mRNA vaccines against COVID-19?",
    "Which mutations make Mycobacterium tuberculosis resistant to bedaquiline?",
    "Who won the 2022 FIFA World Cup?",
]


def _pick(conn: psycopg.Connection, n_per_type: dict[str, int]) -> list[tuple[str, dict]]:
    rows = conn.execute("select pmid, title, abstract from documents order by pmid").fetchall()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    with_genes = [r for r in rows if GENE_RE.search(r["title"])]
    used: set[str] = set()
    picked: list[tuple[str, dict]] = []
    for qtype, n in n_per_type.items():
        pool = with_genes if qtype == "gene_specific" else rows
        for row in pool:
            if len([p for p in picked if p[0] == qtype]) == n:
                break
            if row["pmid"] not in used:
                used.add(row["pmid"])
                picked.append((qtype, row))
    return picked


async def build_questions(per_type: int) -> None:
    from amr_agent.llm import OpenAICompatibleLLM

    s = get_settings()
    llm = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model, 60, 10)
    with psycopg.connect(s.database_url, row_factory=dict_row) as conn:
        picked = _pick(conn, dict.fromkeys(EXTRA, per_type))

    items: list[GoldenItem] = []
    for n, (qtype, row) in enumerate(picked, start=1):
        genes = ", ".join(dict.fromkeys(m.group(0) for m in GENE_RE.finditer(row["title"])))
        system = PROMPT_COMMON.format(extra=EXTRA[qtype].format(genes=genes or "blaNDM-1"))
        reply = await llm.complete(system, f"{row['title']}\n\n{row['abstract']}", json_mode=True)
        question = json.loads(reply.text)["question"].strip()
        items.append(GoldenItem(id=f"{qtype[:4]}-{n:02d}", question=question, type=qtype,
                                relevant_pmids=[row["pmid"]], source_pmid=row["pmid"]))
        print(f"[{n}/{len(picked)}] {qtype}: {question}", flush=True)
    items += [GoldenItem(id=f"oos-{i:02d}", question=q, type="out_of_scope")
              for i, q in enumerate(OUT_OF_SCOPE, start=1)]
    write_jsonl(GOLDEN_PATH, (i.model_dump() for i in items))
    print(f"Wrote {len(items)} questions to {GOLDEN_PATH}: review them before trusting results")


async def build_snapshot(size: int, per_mode: int) -> None:
    from amr_agent.db import make_pool
    from amr_agent.embeddings import FastEmbedEmbedder
    from amr_agent.retrieval import Retriever

    s = get_settings()
    golden = load_golden(GOLDEN_PATH)
    keep: dict[str, None] = {p: None for i in golden for p in i.relevant_pmids}
    n_relevant = len(keep)
    pool = make_pool(s.database_url, max_size=2)
    await pool.open()
    try:
        retriever = Retriever(pool, FastEmbedEmbedder(s.embedding_model))
        for item in golden:  # también las out_of_scope: sus distractores importan
            for mode in ("hybrid", "vector", "keyword"):
                for hit in await retriever.search(item.question, per_mode, mode):
                    keep.setdefault(hit.pmid, None)
    finally:
        await pool.close()
    n_hard = len(keep) - n_relevant

    with psycopg.connect(s.database_url, row_factory=dict_row) as conn:
        rows = conn.execute("select pmid, title, abstract, journal, year from documents"
                            " order by pmid").fetchall()
    others = [r["pmid"] for r in rows if r["pmid"] not in keep]
    random.Random(SEED).shuffle(others)
    keep.update(dict.fromkeys(others[: max(0, size - len(keep))]))
    by_pmid = {r["pmid"]: r for r in rows}
    write_jsonl(CORPUS_PATH, (by_pmid[p] for p in sorted(keep)))
    print(f"Wrote {len(keep)} abstracts to {CORPUS_PATH}: {n_relevant} relevant, "
          f"{n_hard} hard negatives, {len(keep) - n_relevant - n_hard} random")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the golden set and the corpus snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("questions", help="LLM-proposed questions (overwrites golden.jsonl)")
    q.add_argument("--per-type", type=int, default=12)
    q.add_argument("--force", action="store_true", help="overwrite a reviewed golden set")
    snap = sub.add_parser("snapshot", help="relevant + hard negatives + random abstracts")
    snap.add_argument("--size", type=int, default=300, help="minimum snapshot size")
    snap.add_argument("--per-mode", type=int, default=5, help="hard negatives per question/mode")
    args = parser.parse_args(argv)
    if args.command == "questions":
        if GOLDEN_PATH.exists() and not args.force:
            parser.error(f"{GOLDEN_PATH} exists (it may be reviewed); pass --force to overwrite")
        asyncio.run(build_questions(args.per_type))
    else:
        asyncio.run(build_snapshot(args.size, args.per_mode))


if __name__ == "__main__":
    main()
