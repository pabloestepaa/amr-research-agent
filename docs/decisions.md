# Design decisions

Short, dated records of the technical choices behind this project: the context, the decision and
what was traded off. Newest last.

## 1. Free-tier-only stack (2026-11-04)
**Context.** The project must run publicly at zero cost.
**Decision.** Groq (LLM), Neon Postgres + pgvector (storage and vectors), Render Free (API hosting),
GitHub Actions (CI). Hugging Face Spaces was ruled out: since mid-2026, Docker and Gradio Spaces on
CPU require a paid plan for new accounts.
**Trade-offs.** Render Free sleeps after 15 min idle (30–60 s cold start) and has 512 MB of RAM;
Groq's free tier allows about 30 requests per minute.

## 2. One store for text, keywords and vectors: Postgres + pgvector (2026-11-04)
**Context.** Hybrid retrieval needs both full-text search and vector search.
**Decision.** A single `documents` table with a generated `tsvector` column (GIN index) and a
`vector(384)` column (HNSW index, cosine distance), instead of a separate vector database.
**Why.** One system to operate, transactional upserts, and SQL that any backend engineer can read.
3,000 abstracts take about 27 MB, well within Neon's 1 GB free tier.

## 3. No chunking: one row per abstract (2026-11-04)
**Context.** PubMed abstracts average about 1,900 characters.
**Decision.** Embed title + abstract as a single unit.
**Why.** Abstracts are short and self-contained; chunking would add complexity and break citations
(one PMID = one source). Revisit if full texts are ingested.

## 4. Local embeddings inside the API process (2026-11-04)
**Context.** Queries must be embedded at request time.
**Decision.** `BAAI/bge-small-en-v1.5` (384 dimensions) via fastembed (ONNX, CPU), loaded once at
startup with a single thread.
**Why.** No extra API, no rate limits, and deterministic results in CI. Measured peak RSS of the
whole API process: about 324 MB, which fits Render's 512 MB.
**Trade-offs.** English-only model; questions in other languages retrieve worse.

## 5. Hybrid retrieval fused with Reciprocal Rank Fusion (2026-11-05)
**Context.** Semantic search handles paraphrases ("resistance to imipenem" ≈ "carbapenem
resistance") but misses exact identifiers such as `blaNDM-1` or `mcr-1`, which is where keyword
search excels.
**Decision.** Run both searches in parallel and fuse the two rankings with RRF
(`score = Σ 1 / (60 + rank)`).
**Why.** RRF uses only ranks, so a cosine similarity and a `ts_rank_cd` score can be combined
without normalising them. Keyword search turns the question into an OR query, because a natural
language question rarely has all of its words in a single abstract.

## 6. Provider-agnostic LLM client (2026-11-05)
**Decision.** Use the OpenAI SDK with a configurable `base_url`. Groq, Gemini, Ollama and OpenRouter
all expose OpenAI-compatible endpoints, so switching providers means changing two environment
variables. The current model is `openai/gpt-oss-120b` on Groq; Groq had retired the originally
chosen `llama-3.3-70b-versatile`.

## 7. Citations are validated, not trusted (2026-11-05)
**Context.** The prompt asks for `[PMID:x]` citations, but models do not always comply:
`gpt-oss-120b` sometimes cites with full-width brackets such as `【PMID:x】`. This was caught on the
first real query, when the answer looked fine but the API reported zero citations.
**Decision.** Parse tolerant citation variants, normalise them, and split them into `citations`
(PMIDs that were actually retrieved) and `invalid_citations` (hallucinated PMIDs). If retrieval
returns nothing, the LLM is not called and the API returns a fixed "no evidence" answer.

## 8. Resilient ingestion with a fallback search (2026-11-05)
**Context.** On 2026-11-05 NCBI's `esearch` was down, returning HTTP 500 or an HTTP 200 body with an
`ERROR` field, while `efetch` still worked.
**Decision.** Retries with exponential backoff, detection of the error-in-200 case, and an automatic
fallback to Europe PMC (which indexes PubMed) to obtain PMIDs. Abstracts are still fetched from
PubMed. Ingestion is idempotent (upsert by PMID).

## 9. Health check does not touch the database (2026-11-05)
**Decision.** `/health` returns a static response.
**Why.** Render polls it frequently; querying Neon on every check would stop the database from
scaling to zero and burn the free compute hours.

## 10. Tests without network or API keys (2026-11-05)
**Decision.** Test doubles for the embedder (a hashing bag-of-words model), the LLM and the
retriever, plus FastAPI dependency overrides. Integration tests run against a real Postgres with
pgvector (`DATABASE_URL_TEST`, a service container in CI) on a 3-document corpus.

## 11. Own eval metrics instead of an eval framework (2026-11-05)
**Context.** Ragas and DeepEval provide retrieval and LLM-judge metrics out of the box.
**Decision.** Implement recall@k, MRR, nDCG and a single-call LLM judge in ~200 lines
(`src/amr_agent/evals/`).
**Why.** Every metric is a few readable lines with unit tests, there are no extra dependencies or
hidden prompts, and the judge's prompt and output are versioned with the code. The frameworks
remain a reasonable alternative if the metric set grows.

## 12. Retrieval evals on every PR, answer evals on demand (2026-11-05)
**Decision.** Retrieval evals are deterministic and free, so CI runs them on every push and fails
below fixed thresholds. Answer evals call the LLM twice per question and Groq's free tier allows
8,000 tokens per minute per model, so they run manually or weekly.
**How CI gets a corpus.** A versioned 315-abstract snapshot: relevant abstracts plus hard
negatives (what each search mode retrieves on the full corpus). Its scores are within about
0.03 of the full corpus, which validates it as a proxy.

## 13. The judge is a different, smaller model (2026-11-05)
**Decision.** Answers are generated by `gpt-oss-120b` and judged by `gpt-oss-20b` at temperature 0
with JSON output. The judge splits the answer into claims and checks each one against the
abstracts the answer cites; faithfulness is supported / total claims. Abstentions are not judged.
**Why.** A different model avoids grading its own output, and on Groq each model has its own
token budget, so judging does not slow generation down further.

## 14. Keyword search ignores corpus-wide terms (2026-11-05)
**Context.** The first retrieval evals showed hybrid search losing to plain vector search (MRR
0.66 vs 0.78 on the full corpus). The cause was keyword search (MRR 0.35): Postgres `ts_rank_cd`
has no inverse document frequency, so in a corpus where every abstract says "resistance" and
"Escherichia coli", the abstracts that repeat common words outrank the ones that match the rare,
informative term.
**Decision.** A `term_stats` table stores each lexeme's document frequency (refreshed after every
ingestion with `ts_stat`, under a second for 3,000 abstracts). Query terms present in more than 10%
of abstracts are dropped before the OR query is built.
**Result (full corpus, 36 questions).** Keyword MRR 0.35 → 0.61; hybrid MRR 0.66 → 0.80 and
recall@10 0.86 → 0.97, now ahead of vector search (0.78), and keeping its edge on exact gene names.
**Alternatives tried.** Down-weighting the keyword list in RRF only moved hybrid towards
vector-only search: it never beat it. A true BM25 extension (ParadeDB `pg_search`) is not
available on Neon Free.
**Caveat.** `max_df` (0.1) was chosen on the same 36 questions it is measured on. Results on two
halves of the set (MRR 0.83 and 0.76) suggest it is not a fluke, but the gain is an estimate.

## 15. No cross-encoder reranker for now (2026-11-05)
**Measured.** Re-ranking the top 20 hybrid candidates with `ms-marco-MiniLM-L-6-v2` (fastembed)
raised MRR to 0.85 and recall@5 to 0.92, but took ~4 s per query on an M-series laptop and pushed
the process's peak memory above 1 GB.
**Decision.** Not shipped: Render Free has 512 MB and a tenth of a CPU, so a single query would
take tens of seconds. Worth revisiting with paid hosting or a hosted rerank API.
