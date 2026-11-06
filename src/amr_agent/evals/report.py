"""Tablas Markdown a partir de los resultados, para pegarlas en el README.

    uv run amr-eval-report   # regenera evals/RESULTS.md con lo que haya en evals/results/
"""

import json

from amr_agent.evals.dataset import EVALS_DIR, RESULTS_DIR

COLUMNS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def retrieval_table(report: dict) -> str:
    lines = [
        f"Corpus: {report.get('corpus_size', '?')} abstracts · "
        f"{report['n_questions']} questions · {report.get('embedding_model', '')} · "
        f"keyword max_df={report.get('keyword_max_df', 1.0)}, "
        f"weight={report.get('keyword_weight', 1.0)}",
        "",
        "| mode | " + " | ".join(COLUMNS) + " | p50 latency (ms) |",
        "|---" * (len(COLUMNS) + 2) + "|",
    ]
    for mode, data in report["modes"].items():
        cells = [_fmt(data["overall"][c]) for c in COLUMNS]
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {_fmt(data['latency_ms_p50'])} |")

    types = sorted({t for d in report["modes"].values() for t in d["by_type"]})
    lines += ["", "recall@5 by question type:", "",
              "| mode | " + " | ".join(f"{t} (n={_n(report, t)})" for t in types) + " |",
              "|---" * (len(types) + 1) + "|"]
    for mode, data in report["modes"].items():
        cells = [_fmt(data["by_type"].get(t, {}).get("recall@5")) for t in types]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _n(report: dict, qtype: str) -> int:
    return next(iter(report["modes"].values()))["by_type"].get(qtype, {}).get("n", 0)


ANSWER_METRICS = (
    ("faithfulness", "Faithfulness (claims supported by the cited abstracts)"),
    ("relevance", "Answer relevance (judge, 1–5 scaled to 0–1)"),
    ("citation_validity", "Answers without hallucinated PMIDs"),
    ("citation_coverage", "In-scope answers that cite at least one source"),
    ("abstention_accuracy", "Out-of-scope questions correctly refused"),
    ("false_abstention_rate", "In-scope questions wrongly refused (lower is better)"),
    ("context_hit_rate", "Questions whose relevant abstract was retrieved"),
)


def answers_table(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"Generator: `{report['generator_model']}` · judge: `{report['judge_model']}` · "
        f"{report['n_questions']} questions · retrieval `{report['mode']}`, k={report['k']}, "
        f"keyword max_df={report.get('keyword_max_df', 1.0)}"
        + ("" if report.get("complete", True) else " · **incomplete run**"),
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {label} | {_fmt(s.get(key))} |" for key, label in ANSWER_METRICS]
    lines.append(f"| p50 latency (ms) | {_fmt(s.get('latency_ms_p50'))} |")
    return "\n".join(lines)


AGENT_METRICS = (
    ("task_success", "Task success (every check below passed)"),
    ("tool_selection", "Called the tools the task needs"),
    ("argument_accuracy", "Called them with the right arguments (e.g. the WP_ accession)"),
    ("answer_accuracy", "Answer contains the verifiable facts (drug class, top species...)"),
    ("literature_hit_rate", "search_literature found the relevant abstract"),
    ("abstention_accuracy", "Out-of-scope questions correctly refused"),
    ("false_abstention_rate", "In-scope questions wrongly refused (lower is better)"),
    ("citation_validity", "Answers without hallucinated PMIDs"),
    ("agent_errors", "Tasks where the agent crashed, e.g. a provider 400 (lower is better)"),
    ("mean_tool_calls", "Tool calls per task"),
    ("mean_extra_calls", "Unneeded tool calls per task (lower is better)"),
    ("error_call_rate", "Tool calls that returned an error (lower is better)"),
    ("early_stop_rate", "Tasks stopped by the step or token limit (lower is better)"),
    ("mean_tokens", "Tokens per task"),
    ("latency_ms_p50", "p50 latency (ms)"),
)


def agent_table(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"Model: `{report['model']}` · {report['n_tasks']} tasks · max {report['max_steps']} "
        f"steps, {report['token_budget']:,} token budget · tool responses: "
        f"{report['cassette_mode']} ({report['cassette_hits']} replayed, "
        f"{report['cassette_misses']} live)"
        + ("" if report.get("complete", True) else " · **incomplete run**"),
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {label} | {_fmt(s.get(key))} |" for key, label in AGENT_METRICS]
    by_cat = s.get("by_category", {})
    lines += ["", "Task success by category:", "",
              "| " + " | ".join(f"{c} (n={v['n']})" for c, v in by_cat.items()) + " |",
              "|---" * len(by_cat) + "|",
              "| " + " | ".join(_fmt(v["success"]) for v in by_cat.values()) + " |"]
    failed = [r for r in report["per_task"] if not r["success"]]
    if failed:
        lines += ["", "Failed tasks:", ""]
        lines += [f"- `{r['id']}`: " + (f"error: {r['error']}" if r.get("error") else
                                         " → ".join(r["trajectory"]) or "no tool calls")
                  for r in failed]
    return "\n".join(lines)


def main() -> None:
    sections = ["# Eval results", "",
                "Generated by `uv run amr-eval-report` from `evals/results/*.json`.", ""]
    for path in sorted(RESULTS_DIR.glob("*.json")):
        report = json.loads(path.read_text())
        title = path.stem.replace("_", " ")
        if "modes" in report:
            sections += [f"## Retrieval — {title}", "", retrieval_table(report), ""]
        elif "per_task" in report:
            sections += [f"## Agent — {title}", "", agent_table(report), ""]
        elif "summary" in report:
            sections += [f"## Answers — {title}", "", answers_table(report), ""]
    out = EVALS_DIR / "RESULTS.md"
    out.write_text("\n".join(sections))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
