"""Evals de respuestas con un LLM como juez. Gastan cuota de Groq: se lanzan a mano.

    uv run amr-eval-answers --load-corpus          # snapshot de evals, todas las preguntas
    uv run amr-eval-answers --limit 5              # prueba rápida

Qué se mide y cómo:
- Sin LLM (reglas): validez de las citas (ningún PMID inventado), cobertura de citas (cada
  respuesta en el ámbito cita algo), acierto al abstenerse en `out_of_scope` y tasa de
  abstenciones erróneas en preguntas que sí tienen respuesta.
- Con un LLM juez (`openai/gpt-oss-20b`, temperatura 0, salida JSON):
  · fidelidad: el juez separa la respuesta en afirmaciones y marca si cada una está
    respaldada por los abstracts citados; fidelidad = respaldadas / total.
  · relevancia: de 1 a 5, si la respuesta contesta a la pregunta (se escala a 0–1).
El juez es un modelo distinto y más pequeño que el generador: más barato, con su propio
límite de tokens en Groq y sin "autoevaluarse".
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass

from amr_agent.evals.dataset import CORPUS_PATH, GOLDEN_PATH, RESULTS_DIR, GoldenItem, load_golden
from amr_agent.evals.metrics import mean
from amr_agent.llm import LLMClient
from amr_agent.rag import NO_EVIDENCE, RagService
from amr_agent.retrieval import Hit, SearchBackend, SearchMode

JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a research assistant.
You receive a QUESTION, an ANSWER and the SOURCES the answer cites (PubMed abstracts).

1. Split the ANSWER into its individual factual claims (ignore citation markers).
2. For each claim decide if it is fully supported by the SOURCES. Use only the SOURCES,
   not your own knowledge: a claim that is true but absent from the SOURCES is unsupported.
3. Rate how well the ANSWER addresses the QUESTION from 1 (off-topic) to 5 (fully answers it).

Reply with JSON only, in this shape:
{"claims": [{"claim": "...", "supported": true}], "relevance": 4}"""


@dataclass
class Judgement:
    supported: int
    total: int
    relevance: int
    unsupported_claims: list[str]

    @property
    def faithfulness(self) -> float | None:
        return self.supported / self.total if self.total else None


def parse_judgement(text: str) -> Judgement:
    """Lee el JSON del juez. Tolera texto alrededor por si el modelo no respeta el formato."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}")
    data = json.loads(match.group(0))
    claims = data.get("claims") or []
    supported = [c for c in claims if c.get("supported") is True]
    relevance = int(data.get("relevance", 0))
    if not 1 <= relevance <= 5:
        raise ValueError(f"relevance out of range: {relevance}")
    return Judgement(
        supported=len(supported),
        total=len(claims),
        relevance=relevance,
        unsupported_claims=[c.get("claim", "") for c in claims if c.get("supported") is not True],
    )


class RecordingRetriever:
    """Envuelve al buscador y guarda los últimos resultados: el juez necesita el texto de los
    abstracts citados y `RagAnswer` solo devuelve sus metadatos."""

    def __init__(self, inner: SearchBackend) -> None:
        self._inner = inner
        self.last: dict[str, Hit] = {}

    async def search(self, query: str, k: int, mode: SearchMode = "hybrid") -> list[Hit]:
        hits = await self._inner.search(query, k, mode)
        self.last = {h.pmid: h for h in hits}
        return hits


def is_abstention(answer: str) -> bool:
    return NO_EVIDENCE.rstrip(".").lower() in answer.lower()


async def judge(llm: LLMClient, question: str, answer: str, cited: list[Hit]) -> Judgement:
    sources = "\n\n".join(f"[PMID:{h.pmid}] {h.title}\n{h.abstract}" for h in cited) or "(none)"
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{sources}"
    return parse_judgement((await llm.complete(JUDGE_SYSTEM, user, json_mode=True)).text)


async def evaluate_item(
    item: GoldenItem, rag: RagService, recorder: RecordingRetriever, judge_llm: LLMClient,
    k: int, mode: SearchMode,
) -> dict:
    result = await rag.ask(item.question, k=k, mode=mode)
    abstained = is_abstention(result.answer)
    retrieved = [s.pmid for s in result.sources]
    row = {
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "answer": result.answer,
        "abstained": abstained,
        "citations": [c.pmid for c in result.citations],
        "invalid_citations": result.invalid_citations,
        "context_hit": bool(set(item.relevant_pmids) & set(retrieved)) if item.in_scope else None,
        "latency_ms": sum(result.latency_ms.values()),
        "faithfulness": None,
        "relevance": None,
        "unsupported_claims": [],
    }
    # Solo se juzga lo que tiene algo que juzgar: una abstención no contiene afirmaciones.
    if item.in_scope and not abstained:
        cited = [recorder.last[p] for p in row["citations"] if p in recorder.last]
        j = await judge(judge_llm, item.question, result.answer, cited)
        row |= {"faithfulness": j.faithfulness, "relevance": (j.relevance - 1) / 4,
                "unsupported_claims": j.unsupported_claims}
    return row


def summarize(rows: list[dict]) -> dict:
    in_scope = [r for r in rows if r["type"] != "out_of_scope"]
    out_scope = [r for r in rows if r["type"] == "out_of_scope"]
    answered = [r for r in in_scope if not r["abstained"]]
    latencies = sorted(r["latency_ms"] for r in rows)

    def avg(key: str, subset: list[dict]) -> float | None:
        values = [r[key] for r in subset if r[key] is not None]
        return round(mean(values), 4) if values else None

    def share(subset: list[dict], pred) -> float | None:
        return round(sum(1 for r in subset if pred(r)) / len(subset), 4) if subset else None

    return {
        "faithfulness": avg("faithfulness", answered),
        "relevance": avg("relevance", answered),
        "citation_validity": share(rows, lambda r: not r["invalid_citations"]),
        "citation_coverage": share(answered, lambda r: bool(r["citations"])),
        "abstention_accuracy": share(out_scope, lambda r: r["abstained"]),
        "false_abstention_rate": share(in_scope, lambda r: r["abstained"]),
        "context_hit_rate": share(in_scope, lambda r: r["context_hit"]),
        "latency_ms_p50": round(latencies[len(latencies) // 2], 1) if latencies else None,
    }


def is_daily_limit(exc: Exception) -> bool:
    # Groq devuelve 429 tanto por tokens/minuto (el SDK espera y reintenta) como por
    # tokens/día (TPD, 200.000 en gpt-oss-120b): ese no se arregla esperando unos segundos.
    return "per day" in str(exc)


async def run(database_url: str, items: list[GoldenItem], k: int, mode: SearchMode,
              judge_model: str, done: list[dict] | None = None) -> dict:
    """Evalúa `items`. `done` son filas de una ejecución anterior que se reutilizan."""
    import openai

    from amr_agent.config import get_settings
    from amr_agent.db import make_pool
    from amr_agent.embeddings import FastEmbedEmbedder
    from amr_agent.llm import OpenAICompatibleLLM
    from amr_agent.retrieval import Retriever

    s = get_settings()
    if not s.llm_api_key:
        raise SystemExit("LLM_API_KEY is not set")
    # Groq Free: 8.000 tokens/minuto y 200.000/día por modelo; cada /ask gasta ~3.000, así
    # que una ejecución completa (~150.000) consume casi todo el día de gpt-oss-120b.
    # Muchos reintentos (el SDK espera lo que diga `retry-after`) convierten los 429 por minuto
    # en esperas, no en fallos.
    generator = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model, 60, 10)
    judge_llm = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, judge_model, 60, 10)
    pool = make_pool(database_url, max_size=2)
    await pool.open()
    try:
        recorder = RecordingRetriever(Retriever(
            pool, FastEmbedEmbedder(s.embedding_model), s.keyword_weight, s.keyword_max_df))
        rag = RagService(recorder, generator, top_k=k)
        rows = list(done or [])
        finished = {r["id"] for r in rows}
        complete = True
        for n, item in enumerate(items, start=1):  # en serie: el límite es de la cuenta
            if item.id in finished:
                continue
            try:
                row = await evaluate_item(item, rag, recorder, judge_llm, k, mode)
            except openai.RateLimitError as exc:
                if not is_daily_limit(exc):
                    raise
                # Guardamos lo hecho para continuar mañana con --resume en vez de perderlo.
                print(f"Daily token limit reached at {item.id}; saving partial results")
                complete = False
                break
            rows.append(row)
            print(f"[{n}/{len(items)}] {item.id} abstained={row['abstained']} "
                  f"faithfulness={row['faithfulness']} relevance={row['relevance']}", flush=True)
    finally:
        await pool.close()
    return {"generator_model": s.llm_model, "judge_model": judge_model, "mode": mode, "k": k,
            "keyword_max_df": s.keyword_max_df, "keyword_weight": s.keyword_weight,
            "complete": complete, "n_questions": len(rows), "summary": summarize(rows),
            "per_question": rows}


def main(argv: list[str] | None = None) -> None:
    from amr_agent.evals.report import answers_table
    from amr_agent.evals.retrieval_eval import load_snapshot, resolve_eval_db

    parser = argparse.ArgumentParser(description="Answer evals with an LLM judge (uses quota)")
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL_EVAL")
    parser.add_argument("--load-corpus", action="store_true",
                        help=f"drop and reload the eval DB from {CORPUS_PATH}")
    parser.add_argument("--judge-model", default="openai/gpt-oss-20b")
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--name", default="answers", help="results file name (no extension)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the questions already in the results file, run the rest")
    args = parser.parse_args(argv)
    out = RESULTS_DIR / f"{args.name}.json"

    url = resolve_eval_db(args.database_url, args.load_corpus)
    if args.load_corpus:
        print(f"Loaded {load_snapshot(url, CORPUS_PATH)} abstracts from {CORPUS_PATH}")
    items = load_golden(GOLDEN_PATH)[: args.limit]
    done = json.loads(out.read_text())["per_question"] if args.resume and out.exists() else []
    report = asyncio.run(run(url, items, args.k, args.mode, args.judge_model, done))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\n" + answers_table(report) + f"\n\nSaved {out}")
    if not report["complete"]:
        print(f"Incomplete: run again with --resume --name {args.name} when the quota resets")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
