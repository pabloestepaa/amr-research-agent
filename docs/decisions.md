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

## 7. Check every citation against what was retrieved (2026-11-05)
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

## 24. Hitting a limit still produces an answer (2026-11-05)
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

## 32. Abstain only when no result is about the question (2026-11-09)
**Context.** After decision 31, two of the three failures of the agent eval were refusals with
the answer in hand: in `lit-02` and `lit-05` the first search returned the relevant abstract and
the agent replied that it had no evidence. The abstracts use other terms than the questions (a
specific compound for a broader class, an acronym for a description). The rule said "if the
tools return nothing relevant, reply exactly: no evidence", and `gpt-oss` read "relevant" as
"uses the words of the question".
**Decision.** The rule now says that results often use other words (a broader or narrower term,
a synonym, an acronym), that a result answering the question in other words should be used and
the link between its terms and the question stated, that a partial answer is better than none if
it says what is missing, and that the fixed no-evidence reply is only for when no result is about
the question. The rule has no examples taken from the eval tasks, so that the eval still measures
the behaviour and not the memorised cases. The out-of-scope rule (refuse code and general
knowledge without calling tools) is unchanged.
**Trade-off.** A looser rule can turn honest refusals into stretched answers. The out-of-scope
tasks and the citation check watch for that: a stretched answer still has to cite PMIDs the tools
returned.
A first test of the new rule still refused `lit-05`. Its abstract never says "cephalosporin",
only "ESBL-producing", and linking the two takes background knowledge the prompt forbade ("do not
answer from memory"). So the rule also says that textbook knowledge may link terms, never supply
the findings.
**Result.** Partial runs only, since the full eval does not fit in the day's quota after the run
of decision 31. With the first version of the rule, on the five literature, two multi-tool and
three out-of-scope tasks: 8 of 10 pass, `lit-02` answers "myricetin" with the right citation, and
the three out-of-scope questions are still refused. With the final version, on `lit-02`, `lit-05`
and the out-of-scope tasks: 4 of 5. `lit-05` still refuses, although replaying its queries shows
the abstract in the first search: in both runs the model searched twice more and refused after
the cap. It stays as a known failure rather than adding an example close to the task.
`multi-01` still adds "2023 2024" to its query and misses the abstract.
The full run (2026-11-10) did not confirm the partial one: 19/22 again, with the same three
failures. `lit-02` and `lit-05` both found their abstract (literature hit) and refused after a
third search was blocked by the cap; `multi-01` added "2023 2024" again. Everything else held:
right tools and arguments in every task, all out-of-scope questions refused, no invalid
citations, no task at the step limit, ~4,560 tokens per task. The rule did not hurt anything, but
on this model it does not fix refusals reliably either: with Groq not deterministic at
temperature 0, the partial pass of `lit-02` was one good draw, not a fix.

## 33. Read the full text with BM25 over passages, no embeddings (2026-11-09)
**Context.** To make the project useful beyond AMR (a biotechnology student searching cell
therapy papers), the agent has to find specific facts, and those are rarely in the abstract. In
40 open-access cell therapy articles, 96% of the numbers in results paragraphs and tables did not
appear in the abstract (a rough count, but the samples were the kind of fact people look for:
"26% received tocilizumab", "the 5-year OS rate was 42.6%"). A full text is ~38 times an abstract
and splits into ~50 passages of 1,200 characters.
**Decision.** A `read_paper(paper_id, question, k)` tool. It accepts a PMID or a PMCID, fetches
the JATS XML of PubMed Central with `efetch db=pmc` (Europe PMC's `fullTextXML` returned 500
after 12-28 s when tested; NCBI answered in under a second), keeps paragraphs, figure captions
and tables with their section path, joins consecutive blocks of the same section up to 1,200
characters and ranks them with BM25 against the question, with IDF computed within the article.
It returns 5 passages by default and up to 10. Section names are not used to pick results:
Nature papers have no "Results" heading, each section is titled with its finding. The last 8
articles read are kept in memory.
**Measured** (scripts outside the repo; the datasets are not versioned):
- LitQA2 (FutureHouse, CC BY-SA 4.0): 199 questions whose answer needs the full text, each with
  the passage that contains it. 133 of 190 articles are open access in PMC; 131 questions were
  scored. The passage is in the top 5 for 0.82 of them and in the top 10 for 0.93.
- 24 cell therapy questions generated from open-access articles by `gpt-oss-20b`, asked to
  paraphrase (word overlap with the passage 0.30, LitQA2 0.40), reviewed by Claude, not by a
  domain expert: top 5 0.79, top 10 0.92.
- The same questions with `bge-small` embeddings: top 5 0.77 (LitQA2) and 0.62 (paraphrased);
  BM25 and embeddings fused with RRF: 0.86 and 0.88, both 0.92-0.95 in the top 10.
**Trade-off.** The fusion finds the passage a little more often in the top 5, but embedding ~50
passages takes ~9 s per article on a laptop, more on Render's shared CPU. With 10 passages BM25
matches it. Embeddings stay an option for articles already read, if they are stored. Only open
access articles in PMC can be read: 70% of LitQA2's.
**Not yet wired into the agent:** a new tool changes how the AMR agent picks tools, and its evals
would have to be re-run. That comes with the literature mode.

## 34. A literature mode: PubMed ranks, Europe PMC describes, the agent reads (2026-11-09)
**Context.** With `read_paper` (decision 33), the agent can answer questions outside AMR if it
can find the right article. The AMR agent searches a fixed index of 3,000 abstracts, which is
useless for other topics, and adding tools to it would change how it picks them and invalidate
its evals.
**Decision.**
- An `AgentMode` holds what differs between modes: system prompt, final-turn prompt, the search
  tool whose calls are capped at two and de-duplicated (decisions 30 and 31), and the step limit.
  The AMR mode is the default and its prompt is byte-for-byte the same as before.
- The literature mode has two tools, `search_papers` and `read_paper`, and six steps instead of
  five, so that it can search, read an article, read a second one and answer.
- `search_papers` takes the order from PubMed's Best Match (`esearch sort=relevance`) and the
  abstracts from Europe PMC, in one request for all PMIDs. Europe PMC's own ranking favours recent
  articles and also matches the full text of open-access papers: for "ZUMA-1 axicabtagene
  ciloleucel" none of its first 8 results was a report of the trial, while PubMed ranked the
  5-year follow-up first and the original NEJM report third. If `esearch` fails, it searches
  Europe PMC restricted to title and abstract.
- Each result says `full_text: true` when the article is in PubMed Central, whatever its
  licence. The NEJM report of ZUMA-1 is not open access but is in PMC as an NIH author
  manuscript, and `efetch` returns its full text; filtering on open access hid it.
- The prompt tells the agent to search for the study (trial, therapy, disease) rather than for
  the detail, because details are rarely in abstracts (decision 33).
**Result.** "¿Qué porcentaje de pacientes del ensayo ZUMA-1 con axicabtagene ciloleucel recibió
tocilizumab?": the first two versions (Europe PMC ranking, open-access filter) read a 2026
retrospective study that only mentions ZUMA-1 in its introduction and refused. The final version
found the NEJM report, read it and answered 43% with a valid citation. It took 16,400 tokens and
58 s, because it read two articles with 10 passages each; Groq's free tier allows 8,000 tokens per
minute. One question is not an evaluation: the literature mode needs its own eval set.
**Open.** Fewer passages per call (5 by default, but the model asked for 10), and rejecting
invented arguments in local tools as amr-mcp does (the model sent `top_k` instead of `k`).

## 35. Reject invented arguments everywhere, cap read_paper at 8 passages (2026-11-09)
**Context.** In the first real literature-mode question (decision 34) the model sent `top_k`
instead of `k` to `search_papers`, which ignored it silently, and asked `read_paper` for its
maximum of 10 passages twice. The question cost 16,400 tokens: each passage is ~1,200 characters
and the whole history is re-sent in every later call to the model.
**Decision.**
- `Toolset` rejects arguments that the tool's schema does not declare, before calling the tool,
  with the same message amr-mcp uses since decision 30 ("Unknown argument(s) for X: ... Valid
  arguments: ..."), so the model can fix the call. Null values do not count, as in amr-mcp. In
  the last full AMR eval no call used an undeclared argument, so the AMR agent is unaffected.
- `read_paper` returns 5 passages by default and at most 8. On LitQA2 the passage with the
  answer is in the top 5 for 0.82 of the questions, top 6 for 0.85, top 8 for 0.92 and top 10
  for 0.93: 8 keeps almost all of it with ~20% less text than 10.
**Not done: smaller passages.** Measured with chunks of 400, 600, 800 and 1,200 characters: the
results are almost identical, because the real unit is the paragraph (scientific paragraphs are
~1,200 characters and are never split). The characters returned are about k x 1,200 whatever the
chunk size. Cutting the cost further would mean returning only the best sentences of each
passage, which needs its own measurement.

## 36. An eval set for the literature mode, built from full-text facts (2026-11-10)
**Context.** Decision 34 tested the literature mode on one question. The target user (a
biotechnology student working on cell therapy) has not sent real questions yet, and the AMR
eval set cannot measure this mode: its answers come from the gene catalog or from abstracts.
**Decision.**
- `evals/golden_literature.jsonl`: 9 questions whose answer is a number or a name that is in the
  full text of a cell-therapy article but not in its abstract (CARVac starting dose, median
  duration of response, manufacturing platform...), plus 2 out-of-scope questions. They come
  from the synthetic set of the viability study (a Results sentence picked without an LLM, the
  question paraphrased by `gpt-oss-20b`, then checked against the sentence). Two are in Spanish,
  as the target user would ask them. Each question names the study, since a real user knows
  which trial they are asking about.
- The same trajectory checks as the AMR eval (`amr-eval-agent --mode literature`): the agent
  must call `search_papers` and `read_paper`, a search must find the article, and the answer must
  contain the fact. A new metric, `read_hit_rate`, says whether `read_paper` read that article. It
  does not decide success (another paper may report the same figure), but it tells a search
  failure from a reading or writing failure.
- No database: the tools call PubMed, Europe PMC and PMC. Their responses are recorded in
  `evals/literature_cassette.jsonl`, which stays out of git because it holds passages of articles
  and not all of them have a licence that allows redistribution (some are NIH author
  manuscripts). The eval set only stores questions, PMIDs and facts.
- One question was dropped: its answer was "8%", and a substring check cannot tell it from
  "18%". Answer checks avoid bare numbers ("3 patients", not "3").
**Limits.** 9 questions written from known sentences favour the retriever (the wording overlaps
with the passage: 0.33 word overlap in the synthetic set against 0.40 in LitQA2). The real test
is still the student's own questions.

## 37. The literature mode in the API and the demo, and which searches count (2026-11-10)
**Context.** The literature mode (decision 34) only ran from a script. To show it, and for the
target user to try it, it has to be reachable from the demo, but it costs ~15,000 tokens per
question, about twice what Groq's free tier allows per minute.
**Decision.**
- `POST /agent` takes `mode`: `amr` (default, unchanged) or `literature`. The dependencies read
  the mode from the body, so an unavailable agent answers 503 before any quota is spent, as
  before. The literature agent needs neither amr-mcp nor the database, so it works even if
  amr-mcp fails to start.
- It has its own quota (1 question per minute per visitor, 5 per day in total, ~80,000 of the
  200,000 daily tokens) and its own LLM client that waits up to 60 s for Groq's per-minute limit
  instead of 5 s: without waiting, most questions would end in a 503 half way through. Token
  budget 16,000 (the AMR agent keeps 12,000).
- The demo gets a third tab, "Papers", marked beta: one lane in Methods (everything runs in the
  app), its own lead text, round limit, waiting bar and stop messages.
- Searches that return nothing no longer count towards the search cap, nor do rejected calls (an
  invented argument, a network error). In the first demo test the model sent `top_k` (rejected)
  and then a long query with no PubMed results (PubMed requires every word), and the cap blocked
  every later search before it had seen a single article. The empty-result note now explains
  that every keyword must match. For the AMR agent this changes nothing in practice: its search
  always returns abstracts.
**Result.** Four live questions on 2026-11-10: one correct (decision 34's ZUMA-1 question, in
Spanish, the day before); three wrong. ELARA: the main report has no full text in PMC, partial
answer from a subgroup paper. KarMMa: found the NEJM report but read another article, then hit
the round limit and refused. ZUMA-1 in English: read the 2-year follow-up twice (no tocilizumab
figure) and refused. The pattern is choosing which article to read and stopping too late, not
search or reading. `gpt-oss` also sends `top_k` instead of `k` on the first call in every
question. These are the first things the literature eval (decision 36) has to measure before
any prompt change.

## 38. What the first literature eval found: PubMed reads "CAR T" as an author (2026-11-11)
**Context.** First full run of the literature eval (decision 36), 11 tasks, ~100,000 tokens:
5/11. Decision 37 guessed that the problem was choosing which article to read. The eval said
otherwise: in the four real failures the right article never appeared in any search.
**Findings.**
- PubMed translates `CAR T` into `car t[Author]`, a search for an author called "Car T". "GD2
  CAR T H3K27M glioma" returned 3 unrelated articles; with `"CAR T"` in quotes it returns 15, with
  the trial the task is about second. `CAR T-cell` does not go to the author field but expands
  "car" to automobiles and a protein. In a cell-therapy search tool this breaks most queries.
- Long queries return nothing, because PubMed requires every word to match: 9 of the 29 searches
  were empty. The tool's note says so and the model keeps adding words anyway. "TCR" is
  translated as a journal name and "HLA-A*02+" is indexed in a way the query never matches.
- The first call of every task (9/9) sent `top_k` instead of `k` and was rejected: one wasted
  step per question.
- Two failures were the evaluator's: the answers were right but wrote "62.5 %" with a space and
  "3 of the 12 treated patients".
**Decision.**
- `search_papers` rewrites every `CAR T` / `CAR-T` / `CAR T-cell(s)` as the phrase `"CAR T"`
  before calling PubMed (`pubmed_term`). Replaying the agent's own 20 recorded queries against
  live PubMed, the fix finds the missing article for two of the four failed tasks (ft-03 second,
  ft-05 first) and changes nothing for the others.
- The parameter is renamed from `k` to `top_k`, the name the model uses anyway. Accepting both
  would hide invented arguments again, which decision 35 removed on purpose.
- The evaluator ignores a space before `%`, and ft-06 accepts "3 of 12" and "3 of the 12". This
  adds answer forms after seeing the answers, as in decision 29; they are other ways of writing
  the same fact, not looser facts. The saved run was re-scored with the fixed evaluator: 7/11.
- Not done yet: retrying an empty search with fewer words. The two remaining failures (ft-04,
  ft-07) are queries with too many words, which is the model's choice; the next run says whether
  that needs a code change or a prompt change.
**Result.** Second run on 2026-11-12, with every tool response recorded again: 10/11, against
7/11 for the first run after re-scoring. The right article came up in 8 of 9 tasks (5 before):
ft-03 and ft-05 as the replay predicted, and ft-07 because the model chose a shorter query this
time. No tool call failed (24 % of them before, all `k` against `top_k`). Tokens per
task fell from ~9,000 to ~6,400, the median latency from 42 s to 33 s, and one task hit the step
limit instead of four. The failure left, ft-04, is the empty-search problem: four long queries
returned nothing, while "autologous T cell therapy MAGE-A4 solid cancers" puts the article
("Autologous T cell therapy for MAGE-A4(+) solid cancers in HLA-A*02(+) patients") first. The
automatic retry with fewer words is still not done.

## 39. Traces with OpenTelemetry's GenAI conventions, sent to Langfuse (2026-11-11)
**Context.** In production the only window into the agent was one startup log line and the
JSON each request returns. When a demo question went wrong there was no record of which tools
it called, how long each step took or how many tokens it spent, and the evals only cover the
questions in the golden sets.
**Decision.**
- Spans follow OpenTelemetry's GenAI semantic conventions (`invoke_agent`, `chat`,
  `execute_tool`, `invoke_workflow`, `retrieval`, with `gen_ai.usage.*`, `gen_ai.tool.*`...).
  The code depends on OpenTelemetry, not on a vendor SDK: any OTLP backend can receive them,
  and a few `langfuse.*` attributes only add hints (observation type, input, output).
- The spans are written by hand in [`telemetry.py`](../src/amr_agent/telemetry.py) instead of
  using `opentelemetry-instrumentation-openai-v2`. That package is in beta, does not clearly
  support `openai` 3.x (which uses `httpx2`), and would record full prompts. Hand-written spans
  are ~200 lines and record exactly what was chosen.
- What is stored: the question, the final answer and the tool arguments (the queries the model
  writes). Not the prompts, intermediate messages or tool results. Questions come from strangers
  on a public demo, so the page now says that questions are logged and asks visitors to leave
  out personal details. No account or IP address goes into a trace.
- Backend: Langfuse Cloud, free Hobby plan (50,000 units a month, data visible for 30 days, no
  card). One agent question is about 11 units (1 trace plus ~10 observations), so the agent's
  own daily quota cannot exhaust it.
- The `mcp` SDK already creates a client span for each `tools/call` and propagates the trace
  context, so MCP calls appear under the tool that made them at no cost. FastAPI 0.162 also
  creates HTTP spans whenever a tracer provider exists; they are off, so Render's health checks
  and page views spend no units. The connection to `amr-mcp` at startup is one trace
  (`start amr-mcp`) instead of five loose ones.
- Off unless configured: without `OTEL_EXPORTER_OTLP_ENDPOINT` the SDK is not even imported.
  `OTEL_TRACES_EXPORTER=console` prints spans locally. Spans are exported in batches from
  another thread, so exporting never delays an answer.
**Trade-offs.** About 20 MB more memory (the container measured 375–397 MB of 512). The GenAI
conventions are still marked Development and attribute names may change. Export failures only
show up in the Render logs, never in Langfuse (the first setup failed with a 401 for that
reason, see `docs/deploy.md`). Langfuse shows the cost as $0 because it has no price for
`gpt-oss-120b` on Groq.
**Result.** The first trace already answered something the evals never measured: in a
three-tool question (6.7 s in total), the four LLM calls took 3.6 s together and a single
`get_protein_hosts` call took 2.6 s, almost all of it waiting on NCBI inside `amr-mcp`. For
latency, the external databases matter as much as the model.

## 40. General questions in the Papers mode: answer from the abstracts, read each article once (2026-11-12)
**Context.** The first question from a real user ("¿Qué porcentaje de pacientes deja de
necesitar insulina después de un trasplante de islotes pancreáticos?") ended in a refusal after
about two minutes. The trace in Langfuse showed why. The first search returned an on-topic
article from the international islet transplant registry, and the agent read it three times
with slightly different questions or more passages, searched again, read it a fourth time and
hit the step limit. Its percentages are in figures, which `read_paper` cannot see, so with what
it had read the refusal was honest. Two other abstracts in the search results did give figures
("Insulin independence occurred in 78.6%" in a 20-year single-centre cohort, "26.7% vs. 0%" in a
phase 3 trial), but the agent never saw them: `search_papers` cut every abstract to its first
700 characters, and the results came after character 1,000. Every task in the eval named a
study, so this case had never been measured.
**Decision.**
- The prompt tells the agent that a question naming no study has no single right article. It
  answers from the abstracts with the figures of two or three studies, their setting and a
  citation each, and says results vary. It reads an article only if no abstract has a figure.
- `search_papers` keeps the abstract sentences that best match the query, ranked with the same
  BM25 as `read_paper`, in their original order and with "…" where something was left out,
  instead of the first 700 characters. The size is the same. The sentence splitter handles
  Europe PMC's structured abstracts ("P = 0.03.ResultsCrude...") and does not cut after "vs.".
- Each article can be read once per question. A second `read_paper` on the same article (by
  PMID, with or without a `PMID:` prefix) returns an error that asks for another article or an
  answer, like repeated calls already do (decision 30). Only successful reads count: a read
  rejected for an invented argument can be retried.
- The Papers mode refuses with its own message, "I could not find evidence for this in the
  articles I found". "Indexed literature" was the AMR mode's wording, and this mode has no
  index. The evaluator and the demo accept both messages.
- Three `general` tasks join the literature eval, the user's question among them. There is no
  single right figure, so they pass if the answer gives a percentage, cites valid PMIDs and does
  not refuse. The tasks were written after seeing the failure, as in decision 29.
The AMR prompt is unchanged, byte for byte.
**Result.** The user's question, run once against Groq after the prompt and read-once changes
only: it read the article, was stopped from reading it again, searched twice more and refused.
After the abstract change: it answered with the 78.6% from the single-centre cohort and the
registry's note that insulin independence beyond ten years is exceptional, both cited. It
still read the registry article first and used all six steps (17,000 tokens, 69 s), so the
prompt alone does not stop it from reading. The full literature run is pending (~100,000
tokens, not on the same day as the answer eval).

## 41. Papers answers in the user's language, written as a native would (2026-11-12)
**Context.** The second real question ("¿Qué medicamentos tiene que tomar un paciente después de
un trasplante de islotes para evitar el rechazo?") got a good answer with four citations, but its
Spanish read like a translation: "estéril de esteroides" for *steroid-free*, "mycophenolate
mofetil" left in English, "inducion" misspelled. The Langfuse trace shows that the model wrote it
that way; nothing in the app rewrites the text. The rule only said "Answer in the language of the
question".
**Decision.**
- In the Papers mode, an answer that is not in English has to read as if a native scientific
  writer of that language wrote it. Every word is in that language, with the standard names of
  drugs, procedures, diseases and outcomes. Disease and outcome acronyms are the local ones,
  spelled out the first time ("leucemia mieloide aguda (LMA)", not AML), while gene, protein,
  drug, trial and product names stay as they are (IL-6, CD19, ZUMA-1, CAR-T). The model rereads
  the answer before giving it. The Spanish examples come from the defects above.
- The AMR mode keeps its one-line rule and its prompt is unchanged byte for byte, so its eval
  still holds. Its users write in English, and changing it would mean another AMR run.
- `normalize_citations` keeps a PMID that gpt-oss writes as a browsing reference,
  `【19104407†L1-L4】`. Five digits or more is a PMID; browsing numbers are small (`【2】`). Before,
  these were removed as bare references, so the answer lost its citations and they skipped
  validation.
- The evaluator reads a decimal comma as a point, so "83,3 %" matches "83.3%".
**Result.** Measured without the agent loop, to spend little quota: the tool results of five real
runs went to the final answer call, once with each rule. Three were the questions the examples
came from and two were on other topics (chronic graft-versus-host disease, neurological effects
of CAR-T). With the new rule the drug names are translated ("micofenolato de mofetilo") and AML
became "leucemia mieloide aguda". Some English is left: CRS and cGVHD stay as they are (Spanish
texts use SLC and EICH), and one new calque appeared ("agentes depleción‑T‑celular"). The
tocilizumab question was refused under both rules, with the same evidence. A second pass with
gpt-oss-20b as a copy editor was tried and dropped: two of four outputs came back empty and the
rest were unchanged. The full measure comes with the next literature run, which has five Spanish
tasks.
