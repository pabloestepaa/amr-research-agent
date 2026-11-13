# Evals

Three kinds of evaluation, with different cost and cadence:

| | Retrieval evals | Answer evals |
|---|---|---|
| What | recall@k, MRR, nDCG@10 for `hybrid`, `vector`, `keyword` | faithfulness, relevance, citation validity, abstention |
| How | deterministic, local embeddings, no LLM | the RAG pipeline + an LLM judge (`openai/gpt-oss-20b`) |
| Cost | free, ~1 min | ~150k tokens and ~25 min: Groq Free allows 8k tokens/min and 200k/day per model, so about one full run per day (`--resume` continues an interrupted run) |
| When | every push and PR ([ci.yml](../.github/workflows/ci.yml)), fails below [thresholds.json](thresholds.json) | manual or weekly ([answer-evals.yml](../.github/workflows/answer-evals.yml)) |

The third kind evaluates the agent (`/agent`), where the final answer is not enough: an agent
can get it right by luck, or wrong because it picked the wrong tool. Agent evals check the
trajectory with deterministic rules, no LLM judge: did it call the tools the task needs, with
the right arguments (e.g. the `WP_` accession it read from the catalog), does the answer contain
the verifiable facts, did `search_literature` find the relevant abstract, did it refuse
out-of-scope questions, and at what cost (extra or failed calls, steps, tokens). About 130k tokens
for the 22 tasks, so they run by hand only ([agent-evals.yml](../.github/workflows/agent-evals.yml)).

Latest numbers: [RESULTS.md](RESULTS.md).

## Files

- `golden.jsonl`: 42 questions: 12 `factual`, 12 `gene_specific` (exact gene names such as
  `blaNDM-5`, `mcr-1`), 12 `paraphrase` (no distinctive words from the abstract) and 6
  `out_of_scope` (the system must refuse). Each in-scope question has the PMID(s) that answer it.
  Generated semi-automatically (`amr-eval-build questions`: an LLM proposes one question per
  sampled abstract) and then reviewed by hand (`reviewed` field).
- `corpus.jsonl`: a 354-abstract snapshot used in CI instead of the full 3,000-abstract
  corpus: the relevant abstracts plus hard negatives (the top results of every search mode
  for every question on the full corpus), built with `amr-eval-build snapshot`.
  Random distractors would make the task trivially easy.
- `golden_agent.jsonl`: 22 agent tasks in six categories: `catalog` (what a gene is),
  `hosts` (catalog → `WP_` accession → species), `references` (catalog → PMIDs → PubMed),
  `literature` (questions from `golden.jsonl`), `multi_tool` and `out_of_scope`. Two tasks are in
  Spanish (`host-04`, `lit-05`): the agent must still query the English-only index in English. Expected facts were
  checked against the catalog and NCBI on 2026-11-05. Alternatives are written with `|`
  (`"Klebsiella pneumoniae|K. pneumoniae"`).
- `agent_cassette.jsonl`: recorded `amr-mcp` responses (tool + arguments → result). Agent evals
  replay them, so a slow or failing NCBI, or a new catalog version, does not change the score;
  calls not recorded yet go live and are added. `--cassette refresh` re-records everything.
- `golden_literature.jsonl`: 9 literature-mode tasks (decision 36): a figure or a name that is
  in the full text of a cell-therapy article but not in its abstract, so the agent has to find the
  study with `search_papers` and read it with `read_paper`. Two are in Spanish. Three `general`
  questions name no study (the first is a real user's question, which the agent refused to
  answer, see decision 40): they pass if the answer gives a percentage with valid citations and
  does not refuse, since there is no single right figure. Two more tasks are out of scope. Built from the synthetic set of the viability study and checked by Claude, not by
  a domain expert. Their tool responses go to `literature_cassette.jsonl`, which is not in git
  (it holds passages of articles).
- `thresholds.json`: minimum metric values; CI fails if a change drops below them.
- `results/*.json`: full outputs, including per-question results for error analysis.

## Run locally

```bash
createdb amr_agent_eval && psql amr_agent_eval -c "create extension vector"
export DATABASE_URL_EVAL=postgresql://localhost/amr_agent_eval
uv run amr-eval-retrieval --load-corpus --check                  # snapshot, like CI
uv run amr-eval-retrieval --database-url postgresql://localhost/amr_agent --name retrieval_full
uv run amr-eval-answers --limit 5                                 # quick check, uses quota
uv run amr-eval-agent --load-corpus --only host-01,oos-01         # agent evals, uses quota
uv run amr-eval-agent --mode literature                          # no DB, ~140k tokens
uv run amr-eval-report                                            # regenerates RESULTS.md
```

## Known limitations

- Relevance labels are incomplete. Each question was generated from one abstract and only that
  abstract is marked relevant, although others may also answer it. Absolute scores are therefore
  a lower bound; comparisons between modes and between commits remain fair.
- The snapshot is a proxy. Its scores track the full corpus closely (see RESULTS.md), but it
  has 354 documents, and the production corpus has 3,000.
- LLM judges are noisy. Temperature 0 and a JSON output format help, but a single run of 36
  judged answers has a margin of several points. Look at `unsupported_claims` before trusting
  a faithfulness drop.
- The snapshot stores titles and abstracts from PubMed for non-commercial evaluation
  only; abstracts may be copyrighted by their publishers. Each record links back to its PMID.

## What the evals found (2026-11-05)

- The first run showed hybrid search losing to vector search (MRR 0.66 vs 0.78 on the full
  corpus), because keyword search was weak (MRR 0.35): Postgres full-text ranking has no IDF, so
  words found in every abstract ("resistance", "coli") dominated. Dropping query terms present in
  more than 10% of abstracts fixed it (decision 14 in [docs/decisions.md](../docs/decisions.md)):

  | full corpus (2,999 abstracts) | keyword MRR | hybrid MRR | hybrid recall@5 | hybrid recall@10 |
  |---|---|---|---|---|
  | before | 0.35 | 0.66 | 0.81 | 0.86 |
  | after (`max_df=0.1`) | 0.61 | 0.80 | 0.86 | 0.97 |
  | vector only, for reference | — | 0.78 | 0.86 | 0.86 |

  Down-weighting keyword search in RRF never beat vector-only search. A cross-encoder reranker
  reached MRR 0.85 but needs ~4 s and >1 GB per query, too much for Render Free (decision 15).
- Every wrong refusal in the first answer run was a retrieval miss. All 6 in-scope questions
  the model refused were ones where the relevant abstract was not retrieved, so the generator
  abstained instead of guessing. The fix belongs in retrieval.
- One out-of-scope question leaked. Asked how COVID-19 mRNA vaccines work, the model answered from its
  own knowledge and cited an unrelated (but retrieved) abstract, so citation validation did not
  catch it. Candidate fixes: judge out-of-scope answers for faithfulness too, or refuse when the
  retrieval scores are low.
- Faithfulness 0.96: the few unsupported claims are small additions (e.g. "myricetin is a
  plant-derived polyphenol") rather than invented findings. No hallucinated PMIDs.
- The answer results in RESULTS.md were measured before the keyword fix (`max_df=1.0`); a
  rerun is pending because the first rerun hit Groq's daily token limit.
- Latency in the answer evals (p50 ~22 s) is dominated by waiting for Groq's free-tier
  token-per-minute limit; it is not representative of a single `/ask`.
