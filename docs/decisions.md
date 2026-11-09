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
**Update (2026-11-06).** The agent evals found more variants: `【3†PMID:x】` (the model's browsing
citation format), `(PMID x)`, several PMIDs in one bracket (`【PMID:1, PMID:2】`) and bare
references such as `【2】`. The first three are normalised to `[PMID:x]`, so they are validated
too; an unrecognised citation is also an unchecked one. Bare references are removed.

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

## 16. A Docker image with the embedding model baked in (2026-11-05)
**Context.** Render Free has no persistent disk and sleeps after 15 minutes idle, so every cold
start is a fresh container.
**Decision.** A two-stage Dockerfile: uv installs the locked dependencies and downloads
`bge-small-en-v1.5` (~64 MB) at build time; the runtime stage runs as a non-root user with
`HF_HUB_OFFLINE=1`. CI builds the image, runs it with a 512 MB memory cap against an empty
Postgres, smoke-tests `/`, `/health` and `/search`, and reports its memory use.
**Why.** Cold starts do not depend on Hugging Face being reachable, and a broken image is caught
before Render tries to deploy it. Render deploys only after CI passes (`autoDeployTrigger:
checksPass` in `render.yaml`).

## 17. Rate limits on the public demo (2026-11-05)
**Context.** Each `/ask` uses about 3,000 Groq tokens and the free tier allows 200,000 tokens per
day per model, shared with the answer evals: one script in a loop could exhaust it in minutes.
**Decision.** An in-memory sliding-window limiter: 5 questions per minute per IP and 50 per day in
total (both configurable), answering `429` with `Retry-After`. `/search` is not limited because it
does not call the LLM.
**Trade-offs.** State lives in one process, which is correct for Render Free's single instance and
resets on restart. Several instances would need a shared store (Redis or Postgres). The client IP
comes from Render's `X-Forwarded-For`, which a client can prepend to, so the global daily cap is
the real protection for the quota.

## 18. Copy the index to production instead of re-ingesting (2026-11-05)
**Decision.** `amr-copy-db` streams `documents` (including the vectors) from the local database to
Neon with `COPY`, upserts by PMID through a temporary table, and refreshes `term_stats`. It took
11 s for 2,999 abstracts locally and is idempotent.
**Why.** Production gets exactly the corpus the evals were measured on, without calling PubMed
again (its search API was down that day) or re-computing embeddings. Text `COPY` format avoids
depending on matching pgvector versions.

## 19. A dependency-free demo page served by the API (2026-11-05)
**Decision.** One static HTML file with plain JavaScript at `/`, calling `/ask` on the same origin.
**Why.** No frontend build, no second service to host and no CORS. The model's answer is
HTML-escaped before citations and bold/italic are turned into markup, so model output can never
inject HTML into the page.

## 20. A tool-calling agent with a hand-written loop (2026-11-05)
**Context.** `/ask` can only answer from the indexed abstracts. Questions such as "in which species
is blaKPC-2 found most often?" need databases: NCBI's Reference Gene Catalog and its sequence
records, which the author's `amr-mcp` server already exposes.
**Decision.** A `/agent` endpoint where the model chooses tools through OpenAI-compatible tool
calling, in a loop of about 100 lines (`agent.py`): call the model, run the requested tools (in
parallel when it asks for several), append the results, repeat. The response includes the full
trajectory: every tool, its arguments, errors, truncation and latency.
**Why not LangGraph, PydanticAI or the OpenAI Agents SDK.** The loop is small enough to read in one
sitting, every message sent to the model is visible, and the same `ChatLLM` protocol is easy to
replace with a scripted fake in tests. A framework would pay off with persistent state, branching
graphs or human-in-the-loop steps, none of which this agent needs.

## 21. amr-mcp through a real MCP client (2026-11-05)
**Decision.** The API starts `amr-mcp` once, as a stdio subprocess, in the FastAPI lifespan
(`AsyncExitStack` closes it on shutdown), and `McpTools` turns the server's tool list into agent
tools by converting each MCP `inputSchema` into an OpenAI `parameters` schema. Nothing in that
class is specific to AMR: it works with any MCP server.
**Why.** The project shows both sides of the protocol: the server published on PyPI and a client
that consumes it. The server stays a separate process with its own dependencies and cache.
**Cost.** The subprocess uses about 95 MB once the gene catalog is loaded. The whole container
(API + subprocess) measured 367 MiB of Render's 512 MiB in CI, up from 274 MiB. If memory becomes
tight, the same `mcp.Client` can connect to the server in-process, which is what the tests do.
**Failure mode.** If `amr-mcp` cannot start, the API still starts: `/ask` and `/search` keep
working and `/agent` answers 503 without spending the visitor's quota. The NCBI catalog is baked
into the Docker image, like the embedding model, so the first question does not download it.

## 22. Few tools, compact results (2026-11-05)
**Context.** Groq's free tier allows 8,000 tokens per minute, and an agent re-sends the whole
history, including every tool schema and result, on each step. One `search_amr_genes` call
returned about 10,000 characters.
**Decision.** An allowlist of four `amr-mcp` tools (`search_amr_genes`, `get_amr_gene`,
`get_protein_hosts`, `get_pubmed_references`) plus `search_literature`. Excluded: `run_amrfinder`
(it reads file paths on the server), `fetch_sequence` (a FASTA is thousands of tokens that do not
help answer) and the UniProt tools. Results are serialised as compact JSON, fields the model never
needs are dropped, and ranked lists are halved until the result fits in 6,000 characters, so the
JSON stays valid and the least relevant items go first.
**Errors are data.** A failing tool (unknown gene, NCBI returning 500, invalid JSON arguments)
returns `ERROR: ...` to the model instead of raising, so it can correct itself; `amr-mcp`'s own
hint ("try search_amr_genes for a fuzzy match") reaches the model unchanged.

## 23. Literature as a tool, citations checked across all tools (2026-11-05)
**Decision.** `search_literature` returns abstracts (up to 1,500 characters each) with their
PMIDs, not an answer from the RAG pipeline: a nested LLM call would double the tokens and add a
second place to hallucinate. A `[PMID:x]` citation is valid only if some tool returned that PMID
(literature hits, a catalog entry's references, or PubMed metadata). Facts from databases are
attributed in the text ("according to the NCBI Reference Gene Catalog") rather than with a second
citation format.
**Language.** The prompt asks the agent to query the literature in English whatever the language
of the question and to answer in the question's language, a cheap workaround for the English-only
embedding model (decision 4). Two Spanish tasks in the agent evals check it.

## 24. Limits that end in an answer, not an error (2026-11-05)
**Decision.** At most 5 steps, a 12,000-token budget per question and 4 tool calls per step. When a
limit is reached, the agent makes one last call without tools and an instruction to answer with
what it has (first done with `tool_choice="none"`, which `gpt-oss` ignored: see decision 28);
`stop_reason` records why it stopped. `/agent` has its own, stricter
quota: 15 per day and, since 2026-11-06, 1 question per minute per IP (it was 2): one question
spends ~11,700 tokens and Groq Free allows 8,000 per minute, so a second one in the same minute
failed with the provider's 503 instead of our own 429 with a countdown.

## 25. gpt-oss-120b for the agent, not gpt-oss-20b (2026-11-05)
**Measured.** On the same question, `gpt-oss-120b` chained `get_amr_gene("mcr-1")` and then
`get_protein_hosts` with the accession taken from the first result. `gpt-oss-20b` wrapped its
arguments in an extra object (`{"arguments": {...}, "name": ...}`), repeated the same call three
times and ended with a 400 "Parsing failed" from Groq.
**Decision.** The larger model runs the agent; the smaller one stays as the answer-eval judge,
where it only has to return JSON.

## 26. Fail fast on LLM rate limits in the API (2026-11-05)
**Context.** With Groq's daily quota exhausted, `/ask` and `/agent` took about 35 s to return 503:
the OpenAI SDK honours a `Retry-After` of up to two minutes on every retry, so a visitor watched a
spinner only to get an error.
**Decision.** `OpenAICompatibleLLM` retries 429, 5xx and network errors itself (the SDK's own
retries are off) and only when the wait the server asks for is short: at most 5 s in the API,
120 s in the evals, which can afford to wait. The 503 now arrives in under half a second.
**Testing note.** OpenAI SDK 3.x is built on `httpx2`, not `httpx`, so `respx` does not intercept
it; the tests pass an `httpx2` client with a `MockTransport` instead.

## 27. Trajectory evals with deterministic rules and recorded tools (2026-11-05)
**Context.** For an agent, a correct final answer is not enough evidence: it may come from the
wrong tool, or from the right tool with the wrong arguments.
**Decision.** 22 reviewed tasks (`evals/golden_agent.jsonl`) state the tools each one needs, the
argument values that prove the chain worked (the `WP_` accession read from the catalog), the facts
the answer must contain, the abstract the literature search should find, or that it must refuse.
Scoring is rule-based, with no LLM judge, so a trajectory always gets the same score. The
`amr-mcp` responses are recorded once and replayed (`evals/agent_cassette.jsonl`), so an NCBI
outage or a catalog update cannot change the result; only successful responses are recorded.
**Cost.** About 130,000 tokens per full run, most of Groq's daily budget, so it runs by hand
(`amr-eval-agent`, `agent-evals.yml`) and supports `--only` and `--resume`. The agent loop itself
is tested on every push with a scripted LLM against the real tools (`amr-mcp` in-process).

## 28. The forced final answer is a fresh conversation without tools (2026-11-06)
**Context.** The first full run of the trajectory evals crashed on task 6. At the step limit the
agent re-sent the whole conversation with `tool_choice="none"` and a "now answer" message, but
`gpt-oss-120b`, still seeing the tool schemas and its own tool calls, kept trying to call a tool.
Sometimes Groq rejected the request (`400 Tool choice is none, but model called a tool`); other
times the call leaked into the answer as text (`【to=functions.search_amr_genes ...】`) and the
citations came out as `(PMID x)`, which the citation check does not recognise.
**Decision.** The last turn is a new two-message conversation with no tools at all: a system
prompt with the same rules and a user message holding the question and every tool result as plain
text. A model with no tools in sight has nothing to call. Any leaked tool-call markup is still
stripped from the answer as a safety net. The eval runner now records a crashed task as a failure
(with the error, counted in `agent_errors`) and moves on, instead of losing the whole run; a
`--resume` retries those tasks.
**Why it matters.** Unit tests with a scripted LLM could not have caught this: it is a behaviour
of one model behind one provider, and only running the real agent over many tasks exposed it.

## 29. Validate tool arguments on our side, compare facts without typography (2026-11-06)
**Context.** The first full agent eval scored 19/22, and two of the three failures were not the
agent's fault. In `lit-02` the model asked `search_literature` for `k=10`; the schema said
`maximum: 6`, so Groq rejected the whole request with a 400 before our code saw it, and `/agent`
would have answered 502. In `host-02` the answer was right, but `gpt-oss` writes "Escherichia coli"
with a narrow no-break space (U+202F), so a plain substring check did not find it.
**Decision.** Numeric bounds (`minimum`, `maximum`...) are moved from the JSON Schema sent to the
model into each argument's description (`ToolSpec.to_openai`). The model still sees the range,
but an out-of-range value now reaches us: `search_literature` clamps `k`, and `amr-mcp` returns a
tool error the model can read and correct, which is recoverable, unlike a 400. The trajectory
scorer compares facts after Unicode normalisation (NFKC, unified hyphens, collapsed spaces).
The third failure, `host-03`, was a real one: asked which species carry blaNDM-1, the model searched
the literature instead of the catalog and `get_protein_hosts`. The system prompt now says when to use
`get_protein_hosts`, not only what it does.
**Lesson.** Before tuning the agent on a failing task, read the failure. One of the three was a
bug in the evaluator.

## 30. No repeated tool calls, no re-read abstracts, no silently ignored arguments (2026-11-06)
**Context.** In the first full agent eval, 5 of 22 tasks hit the step limit. The agent rarely
repeated a call word for word: it rephrased the same literature search up to four times, and the
retriever returned mostly the same abstracts (`lit-01`: 5, then 3, 2 and 0 new ones). Each
rephrasing re-sent ~5,000 characters the model had already read and gave it no sign that it was
going in circles. Separately, the model invented arguments in 3 tasks (`name`, `gene_family`,
`gene_class` on `search_amr_genes`), which the MCP SDK drops silently: `gene_family: "blaNDM"`
returned 10,261 unrelated entries.
**Decision.**
- A call identical to an earlier one in the same question (same tool, same arguments) is not
  run again; the model gets an error pointing to the step where its result already is.
- `search_literature` results only list abstracts not yet shown in this question, with a note
  naming the omitted PMIDs; when nothing is new, the note tells the model to stop searching.
  Novelty is measured on the results, not on how similar the queries look: two different
  queries can return the same abstracts, and two similar ones can find new evidence.
- `amr-mcp` 0.1.4 rejects undeclared arguments with an error that lists the valid ones (an
  undeclared argument set to `null` is dropped: it means "no filter"). This is a runtime check,
  not `additionalProperties: false` in the schema, for the same reason as decision 29: Groq would
  turn a schema violation into a 400 for the whole request.

## 31. Route by question type in the prompt, cap literature searches at two (2026-11-08)
**Context.** The second full agent eval scored 17/22. The fixes of decision 29 worked, but five
other tasks failed on the model's side: twice it answered "what drug class does gene X confer
resistance to?" from the literature instead of the gene catalog, once it wrote the Python code it
was asked for instead of refusing, and twice it kept searching the literature until the step
limit. Decision 30 made rephrased searches show only new abstracts, but `gpt-oss` kept
rephrasing anyway. Replaying the literature queries of both runs against the eval database
showed where the relevant abstract first appeared: in the first search in 9 of 12 cases, in the
second in 2, in the third once (a question in Spanish), and the fourth search never found it.
In `lit-02` the agent found the right abstract in its first search, spent two more searches on
another compound, and in the final turn said it had no evidence.
**Decision.**
- The system prompt says which tool fits each kind of question: gene identity and drug class go
  to the catalog, species to `get_protein_hosts`, findings of studies to `search_literature`.
- The rules say the agent only answers AMR research questions and refuses anything else (code,
  general knowledge) without calling tools. The final turn shares these rules.
- At most two `search_literature` calls per question, enforced in the loop: a third one gets an
  error asking the model to answer with what it has. A repeated identical call does not count,
  since it is not run. The limit is also in the prompt, but the prompt alone did not stop the
  model in decision 30.
**Trade-off.** Two searches cover 11 of the 12 hits observed; the Spanish question that needed a
third search may now miss its abstract. The limit is a constructor argument
(`max_literature_searches`) if a deployment wants more.
**Result.** Re-running only the five failed tasks: 4 of 5 pass, none reaches the step limit, and
`lit-02` answers correctly after its third search is blocked. `multi-01` still fails: its query
adds "2023 2024", which pushes the relevant abstract out of the top 5. The full re-run
(2026-11-09) scored 19/22: right tools and arguments in every task, no task at the step limit
(5 before), 4,351 tokens per task (6,088 before) and every out-of-scope question refused. Two of
the three failures are a new pattern: in `lit-02` and `lit-05` the first search found the
relevant abstract, and the agent still said it had no evidence, because the abstract uses other
terms than the question (ESBL-producing for cephalosporin-resistant; myricetin, a flavonoid,
for polyphenol). The cap did not cause the refusals (`lit-02` also refused in the run without
it), but it no longer gives the model more searches to talk itself out of them either.
