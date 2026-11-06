# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary: hiring managers, recruiters and AI/LLM engineers** evaluating Pablo Estepa for a remote
  AI Engineer / LLM Engineer role. They arrive from a CV, LinkedIn or the GitHub README, give the
  page one or two minutes, ask one question (often an example), look at the answer and the agent's
  trajectory, and then go to the source code. They need to grasp what it does and why it is
  technically serious without reading documentation.
- **Secondary: antimicrobial-resistance researchers** who may use it as a tool to query recent
  literature and the NCBI gene catalog. Their needs (accuracy, checkable sources) shape the honesty
  of the product, not its priorities.

## Product Purpose

A research assistant for antimicrobial resistance (AMR) that answers questions from ~3,000 recent
PubMed abstracts (hybrid RAG) and, in agent mode, also queries NCBI's AMR gene catalog and protein
records through `amr-mcp`, an MCP server Pablo wrote and published on PyPI. It exists as a portfolio
project that proves generalist, employable AI engineering (retrieval, tool calling, MCP, evals,
deployment on free tiers) through a domain that connects to his bachelor's thesis on AMR.

Success: a visitor runs a question, sees a cited answer and the step-by-step tool calls, understands
in seconds what makes it different, and follows through to the repository.

## Positioning

- **Citations are validated, not trusted**: a PMID only counts if retrieval or a tool returned it
  (0 hallucinated PMIDs across 42 RAG answers and 22 agent tasks).
- **Its own MCP server on both sides**: the agent reaches `amr-mcp` through a real MCP client.
- **Measured, not demoed**: three layers of evals (retrieval in CI with thresholds, LLM-judge
  answers, agent trajectories), and the README lists what the evals caught.
- **Hand-written agent loop** with visible trajectory (each call, arguments, latency, result size).

## Operating Context

- Served at `/` by the FastAPI app on Render Free (https://amr-research-agent.onrender.com); API docs
  at `/docs`. The page calls `POST /ask` (RAG, search mode hybrid/vector/keyword) and `POST /agent`.
- Cold start of 30–60 s after 15 idle minutes (Render Free).
- LLM on Groq Free (`gpt-oss-120b`): an agent question spends ~11,700 tokens against 8,000
  tokens/min, so the agent handles about one question per minute; `/agent` also has its own quota
  (2/min per IP, 15/day) and `/ask` (5/min per IP, 50/day). Rate-limit and cold-start states are
  normal, expected situations for visitors.
- Visitors arrive on desktop mostly, but links shared on LinkedIn open on phones.

## Capabilities and Constraints

- Two modes: Literature (RAG) and Agent (tools). The agent's tools: `search_literature`,
  `search_amr_genes`, `get_amr_gene`, `get_protein_hosts`, `get_pubmed_references`.
- Responses include answer text with `[PMID:x]` citations, retrieved abstracts (cited ones marked),
  timings and token counts; agent responses include the trajectory.
- **Stack (decided by Pablo, 2026-11-06)**: one static HTML file
  (`src/amr_agent/api/static/index.html`), no build step, no framework. Web fonts (e.g. Google
  Fonts) or a small CDN library are allowed when they earn their place.
- Must fit Render Free (512 MB container, currently ~367 MiB) and the CI `docker` job.
- Corpus: ~3,000 abstracts, 2022–2026, English-only embeddings (questions in other languages
  retrieve worse; the agent queries in English).

## Brand Commitments

- Name: **AMR Research Agent**. No logo or established visual identity.
- Voice: plain, precise, honest about limits. It is "a research tool, not clinical advice"; answers
  can be wrong and visitors are told to check the cited sources. No hype or inflated claims.
- Author: Pablo Estepa (GitHub `pabloestepaa`, LinkedIn `pabloestepa`). Related projects:
  `amr-mcp` (PyPI, GitHub) and the thesis `amr-metaheuristics-xai`.

## Evidence on Hand

- Live demo and screenshots: `docs/img/demo-agent.png`, `docs/img/demo-ask.png`,
  `docs/img/social-preview.png`.
- Eval results (real, from `evals/RESULTS.md` and the README): hybrid retrieval recall@10 0.97 and
  MRR 0.80 on 36 questions; answers faithfulness 0.96; agent 19/22 tasks fully correct; 0
  hallucinated PMIDs. A re-run of the answer and agent evals is pending, so these figures may change.
- 30 design decisions in `docs/decisions.md`; architecture diagram in the README.
- **Absent, never to be fabricated**: users, testimonials, usage numbers, press, institutional
  endorsement, clinical validation.

## Product Principles

1. **Show the mechanism.** The trajectory, citations and sources are the product's proof; make them
   visible, not hidden behind the answer.
2. **Understood in one glance, verifiable in one click.** A recruiter gets the point immediately; an
   engineer can open the code, the API docs or the evals from the page.
3. **Honest under limits.** Cold starts, rate limits, refusals and uncertain answers are designed
   states with clear explanations, never silent failures.
4. **Real numbers only.** Every claim on the page links back to something measured in the repo.

## Accessibility & Inclusion

No product-specific requirement established; WCAG 2.2 AA as the baseline. Questions may come in
Spanish (example prompt included), so copy and layout should not assume English-only input.
