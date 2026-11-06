# AMR Research Agent

[![CI](https://github.com/pabloestepaa/amr-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/pabloestepaa/amr-research-agent/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)

A research assistant for **antimicrobial resistance (AMR)**. It answers questions with hybrid RAG
over ~3,000 recent PubMed abstracts, and with a **tool-calling agent** that also queries NCBI's AMR
gene catalog and sequence records through [`amr-mcp`](https://pypi.org/project/amr-mcp/), an MCP
server I wrote and published. Every citation is checked against what the tools actually returned,
and retrieval, answers and agent trajectories are all measured by evals.

**Live demo: <https://amr-research-agent.onrender.com>** · API docs at
[`/docs`](https://amr-research-agent.onrender.com/docs)
<sub>Free tier: after 15 minutes idle, the first request takes 30–60 s while the server wakes up.</sub>

- **Hybrid retrieval** (Postgres full-text + pgvector, fused with Reciprocal Rank Fusion):
  recall@10 of **0.97** on a 36-question golden set, against 0.86 for vector search alone.
- **Agent with a hand-written tool-calling loop** (no framework): it chains catalog lookups, NCBI
  protein records and the literature search, with step, token and per-step call limits.
- **Citations are validated, not trusted**: a PMID only counts if a tool returned it.
  **0 hallucinated PMIDs** across 42 RAG answers and 22 agent tasks.
- **Three layers of evals**: retrieval (on every push, with thresholds that fail CI), answers
  (LLM-as-judge) and agent trajectories (deterministic rules over recorded tool responses).
- **Runs entirely on free tiers**: Render (512 MB), Neon Postgres and Groq. CI builds the Docker
  image and runs it under the same 512 MB limit.

## What it does

The API has two ways to ask:

- **`POST /ask`**: classic RAG. Search the abstracts, then answer citing them as `[PMID:x]`.
- **`POST /agent`**: the model decides which tools to call and in what order, then answers.
  The response includes the full trajectory: each tool call, its arguments, latency and result.

An example from the agent evals (task `multi-01`), which needs two different sources:

> **What drug class does mcr-1 confer resistance to, and what does recent literature report about
> polymyxin resistance in poultry?**
>
> 1. `get_amr_gene({"name": "mcr-1"})`: NCBI Reference Gene Catalog
> 2. `search_literature({"query": "polymyxin resistance poultry 2023 2024 2025", "k": 5})`
> 3. `search_literature({"query": "colistin resistance poultry 2024", "k": 5})`
>
> According to the NCBI Reference Gene Catalog, the **mcr-1** gene family encodes a
> phosphoethanolamine-lipid A transferase that confers resistance to **colistin (polymyxin E)**,
> which is classified under the COLISTIN drug class.
>
> Recent literature shows that the prevalence of polymyxin (colistin) resistance in poultry varies
> by region. A national survey of Australian meat chickens found **no colistin-resistant E. coli**
> among 2,950 isolates [PMID:41759465]. In contrast, studies from Algeria and Pakistan reported
> higher rates: 3.9 % phenotypic colistin resistance in broiler breeders and laying hens, with 78 %
> of resistant isolates carrying **mcr-1** [PMID:41238024]; and **24.8 % colistin-resistant E. coli**
> in broiler chickens [PMID:40777829]. *(trimmed)*

The demo page at `/` lets you switch between the two modes and shows the agent's trajectory step
by step.

## Architecture

```mermaid
flowchart TB
    U(["Browser or API client"]) --> ASK & AG

    subgraph api["FastAPI on Render (512 MB)"]
        ASK["POST /ask<br/>RAG"]
        AG["POST /agent<br/>tool-calling loop"]
        R["Retriever<br/>keyword + vector, fused with RRF"]
        MCP["amr-mcp subprocess<br/>MCP over stdio"]
        ASK --> R
        AG -- search_literature --> R
        AG -- "catalog, species, references" --> MCP
    end

    ASK --> LLM["Groq · gpt-oss-120b"]
    AG --> LLM
    R --> DB[("Postgres + pgvector<br/>text · tsvector · vector(384)")]
    MCP --> NCBI["NCBI Reference Gene Catalog<br/>+ E-utilities"]
    ING["amr-ingest (offline)<br/>PubMed / Europe PMC"] --> DB
```

- **Ingestion** ([`ingest/`](src/amr_agent/ingest/)): NCBI E-utilities, falling back to Europe PMC
  when NCBI search is down. One row per abstract (no chunking: abstracts are short), with its text,
  a `tsvector` for keyword search and a 384-dimension embedding from `bge-small-en-v1.5`, computed
  locally with ONNX ([`embeddings.py`](src/amr_agent/embeddings.py)).
- **Retrieval** ([`retrieval.py`](src/amr_agent/retrieval.py)): full-text and vector search run in
  parallel and are fused with RRF. Keyword search ignores terms present in more than 10 % of the
  corpus, which `ts_rank_cd` alone would not (it has no IDF).
- **RAG** ([`rag.py`](src/amr_agent/rag.py)): each source is labelled with its PMID in the prompt,
  the answer cites them as `[PMID:x]`, and there is a fixed refusal message when the sources do
  not answer the question.
- **Agent** ([`agent.py`](src/amr_agent/agent.py), [`tools.py`](src/amr_agent/tools.py)): a loop of
  about fifty lines. Tools are the literature search plus four `amr-mcp` tools reached through a
  real MCP client. Results are trimmed to 6,000 characters, independent calls run in parallel, and
  when a limit is hit the model is asked for a final answer with the evidence it has.
- **LLM client** ([`llm.py`](src/amr_agent/llm.py)): any OpenAI-compatible API through `base_url`.
  It retries rate limits only when the wait is short, so the API fails fast instead of hanging.

## Evals

Three layers, from cheap and deterministic to expensive and manual. Methodology and golden sets
are in [`evals/README.md`](evals/README.md); full tables, regenerated with `amr-eval-report`,
are in [`evals/RESULTS.md`](evals/RESULTS.md).

| Layer | What it measures | How | When |
|---|---|---|---|
| Retrieval | recall@k, MRR, nDCG@10 per search mode | 36 reviewed questions, local embeddings, no LLM | every push; CI fails below [thresholds](evals/thresholds.json) |
| Answers | faithfulness, relevance, citation validity, abstention | RAG + an LLM judge from a different model (`gpt-oss-20b`) | manual or weekly |
| Agent | right tools, right arguments, facts in the answer, refusals, cost | 22 reviewed tasks, deterministic rules, recorded tool responses | manual |

**Retrieval** (full corpus, 2,999 abstracts, 36 questions):

| mode | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **hybrid** | 0.72 | 0.86 | **0.97** | **0.80** | **0.84** |
| vector | 0.72 | 0.86 | 0.86 | 0.78 | 0.80 |
| keyword | 0.50 | 0.75 | 0.89 | 0.61 | 0.68 |

Keyword search alone is weak on paraphrases (recall@5 0.50) and vector search on exact gene names
(0.83 against 0.92); hybrid holds up on both.

**Answers** (42 questions, 6 of them out of scope): faithfulness **0.96**, relevance 0.99, no
hallucinated PMIDs, every in-scope answer cites a source; 5 of 6 out-of-scope questions refused,
but also 6 of 36 in-scope questions wrongly refused. This run predates the keyword-search fix
below, which retrieves the relevant abstract more often; a re-run is pending.

**Agent** (22 tasks over catalog, species, references, literature, multi-tool and out-of-scope
questions): **19/22** tasks fully correct, right tools 0.94, right arguments 0.93, no hallucinated
PMIDs, all out-of-scope questions refused, ~6,000 tokens per task. The three failures led to the
fixes below; the run that measures them is pending.

### What the evals caught

The point of the evals is not the scores. These problems looked fine by eye and were only visible
in the numbers:

- **Hybrid search was worse than vector search alone** (MRR 0.66 against 0.78). Postgres
  `ts_rank_cd` has no IDF, so abstracts repeating "resistance" and "E. coli" outranked those
  matching the rare, informative term. Dropping query terms present in more than 10 % of the corpus
  fixed it: hybrid MRR 0.80, recall@10 0.86 → 0.97.
- **The API reported zero citations for a correct answer**: `gpt-oss` sometimes writes
  `【PMID:x】` instead of `[PMID:x]`. Citations are now parsed tolerantly and then validated.
- **The agent crashed at its step limit.** Asked for a final answer with `tool_choice="none"`,
  `gpt-oss` kept calling tools: Groq returned a 400, or the call leaked into the answer as text.
  The final turn is now a fresh conversation with no tools in it.
- **A provider-side 400 for an out-of-range argument** (`k=10` with a schema maximum of 6). Numeric
  bounds now go in the description and are enforced on our side, where an error is recoverable.
- **A bug in the evaluator, not the agent**: a correct answer failed because `gpt-oss` writes
  "Escherichia coli" with a narrow no-break space. Facts are now compared after Unicode
  normalisation.

## Design decisions

All 29 decisions, with context and trade-offs, are in [`docs/decisions.md`](docs/decisions.md).
The ones that shaped the project most:

| Decision | Why | Trade-off |
|---|---|---|
| **Postgres + pgvector** for text, keywords and vectors | one system to run, transactional upserts, plain SQL; 3,000 abstracts take ~27 MB | no built-in BM25, so keyword ranking needed the document-frequency fix above |
| **No chunking**: one row per abstract | abstracts are short and self-contained, and one PMID = one citable source | to revisit if full texts are ingested |
| **Citations are validated, not trusted** | a PMID counts only if retrieval or a tool returned it; the rest go to `invalid_citations` | tolerant parsing has to keep up with each model's citation habits |
| **Hand-written agent loop**, no framework | every message sent to the model is visible, and a scripted fake LLM tests the loop | no persistent state or branching graphs, which this agent does not need |
| **`amr-mcp` through a real MCP client** in a subprocess | the project shows both sides of MCP; the client works with any MCP server | ~95 MB of the 512 MB budget (the container measures 367 MiB) |
| **Own eval metrics** instead of Ragas or DeepEval | each metric is a few tested lines, and the judge prompt is versioned with the code | fewer metrics out of the box |
| **No cross-encoder reranker** | it raised MRR to 0.85, but took ~4 s per query and over 1 GB of RAM | worth adding on paid hosting or with a hosted rerank API |
| **Fail fast on LLM rate limits** | with the daily quota spent, a 503 in 0.5 s beats a 35 s spinner ending in the same error | the evals, which can wait, allow longer retries |

## Run it locally

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), PostgreSQL 17 with
[pgvector](https://github.com/pgvector/pgvector), and a free [Groq API key](https://console.groq.com/keys)
(any OpenAI-compatible endpoint works).

```bash
cp .env.example .env        # set DATABASE_URL and LLM_API_KEY
uv sync
uv run amr-ingest --max 3000                      # fetch, embed and index abstracts (idempotent)
uv run uvicorn amr_agent.api.main:app --port 8011 # demo at /, API docs at /docs
```

Tests use no network and no API keys (fake LLM and retriever, `amr-mcp` in-process), only a
throwaway database:

```bash
DATABASE_URL_TEST=postgresql://localhost/amr_agent_test uv run pytest -q
uv run ruff check .
```

Evals (the last two spend LLM quota):

```bash
export DATABASE_URL_EVAL=postgresql://localhost/amr_agent_eval
uv run amr-eval-retrieval --load-corpus --check   # as in CI
uv run amr-eval-answers                           # ~150k tokens, ~25 min on Groq Free
uv run amr-eval-agent --load-corpus               # ~130k tokens
uv run amr-eval-report                            # rewrites evals/RESULTS.md
```

Deployment to Render and Neon is described in [`docs/deploy.md`](docs/deploy.md).

## Stack and limitations

**Stack:** Python 3.13 · FastAPI · PostgreSQL + pgvector (Neon) · fastembed (`bge-small-en-v1.5`,
ONNX) · OpenAI SDK against Groq (`gpt-oss-120b`, judge `gpt-oss-20b`) · MCP (`mcp` client,
[`amr-mcp`](https://github.com/pabloestepaa/amr-mcp) server) · pytest · ruff · Docker ·
GitHub Actions · Render.

**Limitations:**

- **Small, recent corpus**: ~3,000 abstracts (2022–2026), abstracts only, no full texts.
- **English-only embeddings**: questions in other languages retrieve worse. The agent is told to
  query the literature in English, which mitigates it.
- **Small eval sets**: 36 retrieval questions, 42 answers and 22 agent tasks are enough to catch
  regressions and compare options, not to give tight confidence intervals. The keyword threshold
  (10 %) was tuned on the same questions it is measured on.
- **Not deterministic**: even at temperature 0, the same agent task can take a different path
  from one run to the next.
- **Free-tier limits**: the demo sleeps when idle, and Groq's daily token budget is shared by the
  demo and the evals, so the demo has per-IP and daily quotas.
- **A research tool, not clinical advice.** Answers summarise abstracts and public databases and
  can be wrong; check the cited sources.

## License

MIT, see [`LICENSE`](LICENSE).
