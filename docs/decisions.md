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
