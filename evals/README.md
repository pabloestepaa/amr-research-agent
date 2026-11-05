# Evals

Two kinds of evaluation, with different cost and cadence:

| | Retrieval evals | Answer evals |
|---|---|---|
| What | recall@k, MRR, nDCG@10 for `hybrid`, `vector`, `keyword` | faithfulness, relevance, citation validity, abstention |
| How | deterministic, local embeddings, no LLM | the RAG pipeline + an LLM judge (`openai/gpt-oss-20b`) |
| Cost | free, ~1 min | Groq free-tier quota, ~20–30 min (8k tokens/min limit) |
| When | every push and PR ([ci.yml](../.github/workflows/ci.yml)), fails below [thresholds.json](thresholds.json) | manual or weekly ([answer-evals.yml](../.github/workflows/answer-evals.yml)) |

Latest numbers: [RESULTS.md](RESULTS.md).

## Files

- `golden.jsonl` — 42 questions: 12 `factual`, 12 `gene_specific` (exact gene names such as
  `blaNDM-5`, `mcr-1`), 12 `paraphrase` (no distinctive words from the abstract) and 6
  `out_of_scope` (the system must refuse). Each in-scope question has the PMID(s) that answer it.
  Generated semi-automatically (`amr-eval-build questions`: an LLM proposes one question per
  sampled abstract) and then reviewed by hand (`reviewed` field).
- `corpus.jsonl` — a 315-abstract snapshot used in CI instead of the full 3,000-abstract
  corpus: the relevant abstracts plus **hard negatives** (the top results of every search mode
  for every question on the full corpus), built with `amr-eval-build snapshot`.
  Random distractors would make the task trivially easy.
- `thresholds.json` — minimum metric values; CI fails if a change drops below them.
- `results/*.json` — full outputs, including per-question results for error analysis.

## Run locally

```bash
createdb amr_agent_eval && psql amr_agent_eval -c "create extension vector"
export DATABASE_URL_EVAL=postgresql://localhost/amr_agent_eval
uv run amr-eval-retrieval --load-corpus --check                  # snapshot, like CI
uv run amr-eval-retrieval --database-url postgresql://localhost/amr_agent --name retrieval_full
uv run amr-eval-answers --limit 5                                 # quick check, uses quota
uv run amr-eval-report                                            # regenerates RESULTS.md
```

## Known limitations

- **Incomplete relevance labels.** Each question was generated from one abstract and only that
  abstract is marked relevant, although others may also answer it. Absolute scores are therefore
  a lower bound; comparisons between modes and between commits remain fair.
- **The snapshot is a proxy.** Its scores track the full corpus closely (see RESULTS.md), but
  numbers on 315 documents are not numbers on 3,000.
- **LLM judges are noisy.** Temperature 0 and a JSON output format help, but a single run of 36
  judged answers has a margin of several points. Look at `unsupported_claims` before trusting
  a faithfulness drop.
- **Data.** The snapshot stores titles and abstracts from PubMed for non-commercial evaluation
  only; abstracts may be copyrighted by their publishers. Each record links back to its PMID.

## What the first run (2026-11-05) found

- **Hybrid retrieval is not the best mode here.** `vector` beats `hybrid` on recall@5 (0.86 vs
  0.83) and MRR (0.78 vs 0.63) on the full corpus. Keyword search is weak (MRR 0.34) and equal
  weighting in RRF drags hybrid down on paraphrased questions (recall@5 0.58 vs 0.83); hybrid only
  wins on exact gene names (1.00 vs 0.83). A weighted fusion or a reranker is the next experiment.
- **Every wrong refusal was a retrieval miss.** All 6 in-scope questions the model refused were
  ones where the relevant abstract was not retrieved, so the generator abstained honestly instead of
  guessing; the fix belongs in retrieval, not in the prompt.
- **One out-of-scope leak.** Asked how COVID-19 mRNA vaccines work, the model answered from its
  own knowledge and cited an unrelated (but retrieved) abstract, so citation validation did not
  catch it. Candidate fixes: judge out-of-scope answers for faithfulness too, or refuse when the
  retrieval scores are low.
- Faithfulness 0.96: the few unsupported claims are small additions (e.g. "myricetin is a
  plant-derived polyphenol") rather than invented findings. No hallucinated PMIDs.
- Latency in the answer evals (p50 ~22 s) is dominated by waiting for Groq's free-tier
  token-per-minute limit; it is not representative of a single `/ask`.
