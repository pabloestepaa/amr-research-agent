"""Evals de trayectoria del agente. Gastan cuota de Groq: se lanzan a mano.

    uv run amr-eval-agent --load-corpus        # snapshot de evals, todas las tareas
    uv run amr-eval-agent --only host-01,oos-01   # solo algunas

No basta con mirar la respuesta final: un agente puede acertar por casualidad o fallar porque
eligió mal una herramienta. Para cada tarea de `evals/golden_agent.jsonl` se comprueba, sin LLM
juez (todo son reglas deterministas):
- herramientas: ¿llamó a las que hacían falta? (p. ej., get_protein_hosts para especies)
- argumentos: ¿con los valores correctos? (la accesión WP_ que sacó del catálogo)
- respuesta: ¿contiene los hechos verificables? (la clase de fármaco, la especie principal)
- literatura: ¿search_literature encontró el abstract que responde?
- abstención: ¿se negó en las preguntas fuera de ámbito?
- coste: llamadas innecesarias, llamadas con error, pasos, tokens y paradas por límite.

Las respuestas de amr-mcp se graban en `evals/agent_cassette.jsonl` (ver cassette.py) para que
la eval no dependa de que NCBI esté disponible ni de la versión del catálogo.
"""

import argparse
import asyncio
import json
import re
import time
import unicodedata
from itertools import chain

from amr_agent.agent import AgentAnswer
from amr_agent.evals.answers_eval import is_abstention, is_daily_limit
from amr_agent.evals.dataset import (
    AGENT_GOLDEN_PATH,
    CASSETTE_PATH,
    CORPUS_PATH,
    RESULTS_DIR,
    AgentTask,
    load_agent_golden,
)
from amr_agent.evals.metrics import mean

HYPHENS_RE = re.compile(r"[‐-―−]")


def normalize_text(text: str) -> str:
    """Para comparar hechos sin que la tipografía cuente.

    gpt-oss separa "Escherichia coli" con un espacio fino (U+202F) y escribe "blaNDM-1" con un
    guion no separable (U+2011):
    la respuesta es correcta pero "escherichia coli" no aparecía en ella (falso negativo en
    host-02). NFKC convierte los espacios especiales en espacios normales; los guiones los
    unificamos a mano porque NFKC no los toca.
    """
    text = HYPHENS_RE.sub("-", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).lower()


def _alternatives(text: str) -> list[str]:
    return [normalize_text(a.strip()) for a in text.split("|") if a.strip()]


def score_task(task: AgentTask, result: AgentAnswer) -> dict:
    """Compara la trayectoria y la respuesta con lo esperado. Sin red ni LLM: se puede testear."""
    steps = result.steps
    called = {s.tool for s in steps}
    answer = normalize_text(result.answer)

    tools_hit = [bool(called & set(g.split("|"))) for g in task.expect_tools]

    def args_match(group: str, needle: str) -> bool:
        tools = set(group.split("|"))
        return any(s.tool in tools and needle.lower() in
                   json.dumps(s.arguments or {}, ensure_ascii=False).lower() for s in steps)

    expected_names = set(chain.from_iterable(g.split("|") for g in task.expect_tools))
    literature_pmids = set(chain.from_iterable(
        s.pmids for s in steps if s.tool == "search_literature"))

    args_ok = (all(args_match(g, n) for g, n in task.expect_args.items())
               if task.expect_args else None)
    answer_ok = (all(any(a in answer for a in _alternatives(item))
                     for item in task.answer_contains) if task.answer_contains else None)
    literature_hit = (bool(literature_pmids & set(task.relevant_pmids))
                      if task.relevant_pmids else None)
    abstained = is_abstention(result.answer)
    citations_valid = not result.invalid_citations

    if task.should_abstain:
        success = abstained and citations_valid
    else:
        # None significa "no se comprueba en esta tarea": solo un False la hace fallar.
        success = (all(tools_hit) and args_ok is not False and answer_ok is not False
                   and literature_hit is not False and not abstained and citations_valid)

    return {
        "id": task.id,
        "category": task.category,
        "question": task.question,
        "answer": result.answer,
        "success": success,
        "tools_ok": all(tools_hit) if task.expect_tools else None,
        "tool_recall": mean([float(h) for h in tools_hit]) if tools_hit else None,
        "args_ok": args_ok,
        "answer_ok": answer_ok,
        "literature_hit": literature_hit,
        "abstained": abstained,
        "should_abstain": task.should_abstain,
        "citations_valid": citations_valid,
        "invalid_citations": result.invalid_citations,
        "trajectory": [f"{s.step}:{s.tool}({json.dumps(s.arguments, ensure_ascii=False)})"
                       + (" ERROR" if s.is_error else "") for s in steps],
        "n_tool_calls": len(steps),
        "extra_calls": sum(1 for s in steps if s.tool not in expected_names),
        "error_calls": sum(1 for s in steps if s.is_error),
        "llm_calls": result.usage.llm_calls,
        "tokens": result.usage.prompt_tokens + result.usage.completion_tokens,
        "stop_reason": result.stop_reason,
        "latency_ms": result.latency_ms["total"],
    }


def failed_row(task: AgentTask, exc: Exception, latency_ms: float) -> dict:
    """Una tarea en la que el agente lanzó una excepción (p. ej. un 400 del proveedor).

    Cuenta como fallo: en la API sería un 502 para el usuario. Las comprobaciones quedan en None
    (no se pudieron hacer) y el coste en 0, porque no lo conocemos; `summarize` las excluye de
    las medias de coste y latencia.
    """
    return {
        "id": task.id, "category": task.category, "question": task.question,
        "answer": "", "success": False, "error": f"{type(exc).__name__}: {exc}"[:500],
        "tools_ok": None, "tool_recall": None, "args_ok": None, "answer_ok": None,
        "literature_hit": None, "abstained": None, "should_abstain": task.should_abstain,
        "citations_valid": None, "invalid_citations": [], "trajectory": [],
        "n_tool_calls": 0, "extra_calls": 0, "error_calls": 0, "llm_calls": 0, "tokens": 0,
        "stop_reason": "error", "latency_ms": round(latency_ms, 1),
    }


def summarize(rows: list[dict]) -> dict:
    def rate(key: str, subset: list[dict] | None = None) -> float | None:
        values = [r[key] for r in (rows if subset is None else subset) if r[key] is not None]
        return round(sum(1 for v in values if v) / len(values), 4) if values else None

    ran = [r for r in rows if not r.get("error")]  # las que terminaron: coste y latencia

    def avg(key: str) -> float | None:
        return round(mean([float(r[key]) for r in ran]), 2) if ran else None

    in_scope = [r for r in rows if not r["should_abstain"]]
    out_scope = [r for r in rows if r["should_abstain"]]
    total_calls = sum(r["n_tool_calls"] for r in ran)
    latencies = sorted(r["latency_ms"] for r in ran)
    categories = sorted({r["category"] for r in rows})
    return {
        "task_success": rate("success"),
        "tool_selection": rate("tools_ok", in_scope),
        "argument_accuracy": rate("args_ok"),
        "answer_accuracy": rate("answer_ok"),
        "literature_hit_rate": rate("literature_hit"),
        "abstention_accuracy": rate("abstained", out_scope),
        "false_abstention_rate": rate("abstained", in_scope),
        "citation_validity": rate("citations_valid"),
        "agent_errors": len(rows) - len(ran),
        "mean_tool_calls": avg("n_tool_calls"),
        "mean_extra_calls": avg("extra_calls"),
        "error_call_rate": (round(sum(r["error_calls"] for r in ran) / total_calls, 4)
                            if total_calls else None),
        "early_stop_rate": (round(sum(1 for r in rows if r["stop_reason"] != "answer")
                                  / len(rows), 4) if rows else None),
        "mean_tokens": avg("tokens"),
        "latency_ms_p50": round(latencies[len(latencies) // 2], 1) if latencies else None,
        "by_category": {c: {"n": sum(1 for r in rows if r["category"] == c),
                            "success": rate("success", [r for r in rows if r["category"] == c])}
                        for c in categories},
    }


async def run(database_url: str, tasks: list[AgentTask], cassette_mode: str,
              done: list[dict] | None = None) -> dict:
    """Ejecuta el agente real (Groq + amr-mcp por stdio) sobre `tasks`."""
    import openai
    from mcp import Client

    from amr_agent.agent import Agent
    from amr_agent.config import get_settings
    from amr_agent.db import make_pool
    from amr_agent.embeddings import FastEmbedEmbedder
    from amr_agent.evals.cassette import CassetteTools
    from amr_agent.llm import OpenAICompatibleLLM
    from amr_agent.retrieval import Retriever
    from amr_agent.tools import LiteratureTools, McpTools, Toolset, amr_mcp_server

    s = get_settings()
    if not s.llm_api_key:
        raise SystemExit("LLM_API_KEY is not set")
    # Como en las evals de respuestas: muchos reintentos y esperas largas permitidas, porque
    # aquí sí compensa esperar a que pase el límite por minuto de Groq.
    llm = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model, 60, max_retries=10)
    pool = make_pool(database_url, max_size=2)
    await pool.open()
    # Al reanudar, las tareas que acabaron en error se repiten (el error pudo ser pasajero).
    rows = [r for r in done or [] if not r.get("error")]
    finished = {r["id"] for r in rows}
    complete = True
    cassette = None
    try:
        retriever = Retriever(pool, FastEmbedEmbedder(s.embedding_model), s.keyword_weight,
                              s.keyword_max_df)
        async with Client(amr_mcp_server(), read_timeout_seconds=s.mcp_timeout_s) as client:
            cassette = CassetteTools(await McpTools.load(client), CASSETTE_PATH, cassette_mode)
            agent = Agent(llm, Toolset([LiteratureTools(retriever), cassette]),
                          max_steps=s.agent_max_steps, token_budget=s.agent_token_budget)
            for n, task in enumerate(tasks, start=1):  # en serie: el límite es de la cuenta
                if task.id in finished:
                    continue
                # Capturamos aquí dentro: fuera del `async with` la excepción saldría envuelta
                # en un ExceptionGroup de anyio y `except openai.RateLimitError` no la vería.
                t0 = time.perf_counter()
                try:
                    result = await agent.run(task.question)
                except openai.RateLimitError as exc:
                    if not is_daily_limit(exc):
                        raise
                    print(f"Daily token limit reached at {task.id}; saving partial results")
                    complete = False
                    break
                except Exception as exc:  # noqa: BLE001 - un fallo del agente es un resultado
                    row = failed_row(task, exc, (time.perf_counter() - t0) * 1000)
                    rows.append(row)
                    print(f"[{n}/{len(tasks)}] {task.id} ERROR {row['error']}", flush=True)
                    continue
                row = score_task(task, result)
                rows.append(row)
                print(f"[{n}/{len(tasks)}] {task.id} success={row['success']} "
                      f"tools={row['trajectory']} tokens={row['tokens']}", flush=True)
    finally:
        await pool.close()
        if cassette is not None:
            cassette.save()  # también si se corta: lo grabado ya no hay que volver a pedirlo
    return {
        "model": s.llm_model, "max_steps": s.agent_max_steps,
        "token_budget": s.agent_token_budget, "cassette_mode": cassette_mode,
        "cassette_hits": cassette.hits if cassette else 0,
        "cassette_misses": cassette.misses if cassette else 0,
        "complete": complete, "n_tasks": len(rows), "summary": summarize(rows),
        "per_task": rows,
    }


def main(argv: list[str] | None = None) -> None:
    from amr_agent.evals.report import agent_table
    from amr_agent.evals.retrieval_eval import load_snapshot, resolve_eval_db

    parser = argparse.ArgumentParser(description="Agent trajectory evals (uses LLM quota)")
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL_EVAL")
    parser.add_argument("--load-corpus", action="store_true",
                        help=f"drop and reload the eval DB from {CORPUS_PATH}")
    parser.add_argument("--limit", type=int, help="only the first N tasks")
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--name", default="agent", help="results file name (no extension)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the tasks already in the results file, run the rest")
    parser.add_argument("--cassette", choices=["replay", "refresh", "live"], default="replay",
                        help="tool responses: replay recorded ones (default), re-record, or live")
    args = parser.parse_args(argv)
    out = RESULTS_DIR / f"{args.name}.json"

    url = resolve_eval_db(args.database_url, args.load_corpus)
    if args.load_corpus:
        print(f"Loaded {load_snapshot(url, CORPUS_PATH)} abstracts from {CORPUS_PATH}")
    tasks = load_agent_golden(AGENT_GOLDEN_PATH)
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t.id in wanted]
    tasks = tasks[: args.limit]
    done = json.loads(out.read_text())["per_task"] if args.resume and out.exists() else []
    report = asyncio.run(run(url, tasks, args.cassette, done))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\n" + agent_table(report) + f"\n\nSaved {out}")
    if not report["complete"]:
        print(f"Incomplete: run again with --resume --name {args.name} when the quota resets")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
