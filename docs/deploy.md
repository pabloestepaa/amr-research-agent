# Deployment

Live demo: <https://amr-research-agent.onrender.com> (API docs at `/docs`).

Production runs on free tiers: **Neon** (Postgres + pgvector) and **Render** (the Docker image in
this repo). Render deploys `main` automatically once CI passes.

```
Browser ──► Render (FastAPI + demo page + embeddings) ──► Neon (pgvector)
                                                   ├────► Groq (LLM)
                                                   └────► Langfuse (traces, OTLP)
```

## 1. Database on Neon
1. Create a project at <https://console.neon.tech> in **AWS Europe Central 1 (Frankfurt)**, next to
   the Render region.
2. Copy two connection strings from *Connect*: the **direct** one (for loading data) and the
   **pooled** one (host contains `-pooler`, for the API).
3. Load the indexed corpus from your local database (schema is created if missing):
   ```bash
   uv run amr-copy-db --target 'postgresql://...direct...?sslmode=require'
   ```

## 2. API on Render
1. *New → Blueprint* and pick this repository; Render reads `render.yaml`.
2. Fill in the secrets it asks for: `DATABASE_URL` (Neon **pooled** URL) and `LLM_API_KEY` (Groq).
3. Wait for the first deploy, then open the service URL: `/` is the demo, `/docs` the API.

## 3. Traces on Langfuse (optional)
The API exports OpenTelemetry traces (GenAI semantic conventions) when an OTLP endpoint is set, and
does nothing otherwise. Any OTLP/HTTP backend works; production uses Langfuse Cloud (free Hobby plan).
1. Create a project at <https://cloud.langfuse.com> (EU region) and an API key pair in *Settings → API
   keys*. The secret key is shown once.
2. Build the header value in a terminal, without the keys ending up on screen or in the shell
   history (zsh; run everything in the same tab):
   ```bash
   read -s "pk?Public key: "; echo
   read -s "sk?Secret key: "; echo
   printf 'Authorization=Basic%%20%s,x-langfuse-ingestion-version=4' "$(printf '%s:%s' "$pk" "$sk" | base64)" | pbcopy
   pbpaste | head -c 26; echo; pbpaste | wc -c   # Authorization=Basic%20cGst, ~169 characters
   unset pk sk
   ```
   `%20` is the space after `Basic`: OpenTelemetry reads these headers URL-encoded. The ingestion
   header makes traces show up within seconds instead of up to 10 minutes.
3. In Render, set `OTEL_EXPORTER_OTLP_HEADERS` to that value (`OTEL_EXPORTER_OTLP_ENDPOINT` comes
   from `render.yaml`) and choose *Save, rebuild and deploy*: *Save only* does not restart.
4. The startup log should say `tracing otlp`. Ask a question and the trace appears in *Tracing*.

To see traces locally without an account, start the API with `OTEL_TRACES_EXPORTER=console`.

## Operating notes
- **Cold start.** Render Free sleeps after 15 minutes idle; the first request then takes 30–60 s.
  Neon also scales to zero, which adds about a second.
- **Quota.** `/ask` is limited to 5 questions per minute per IP and 50 per day in total
  (`ASK_PER_IP_PER_MINUTE`, `ASK_PER_DAY`), to protect Groq's free daily token budget.
  `/agent` costs 3–4 times as much per question and has its own limits: 1 per minute per IP (a question spends ~11,700
  tokens and Groq Free allows 8,000 per minute) and 15 per day (`AGENT_PER_IP_PER_MINUTE`, `AGENT_PER_DAY`).
- **Agent.** The API starts `amr-mcp` as a subprocess at startup (about 95 MB with the gene catalog
  loaded; the container measured 367 MiB of 512 MiB in CI). If it fails to start, `/ask` and
  `/search` still work and `/agent` returns 503: check the logs for "Could not start amr-mcp".
  `AGENT_ENABLED=false` skips it entirely to save memory. `NCBI_API_KEY` (optional) raises
  `amr-mcp`'s NCBI limit from 3 to 10 requests per second.
- **Startup log.** Each start logs one line with the time per phase and the memory in use, e.g.
  `Startup 18.2 s (db 0.7 s, model 5.3 s, amr-mcp 12.2 s) · agent on · container 368 MB
  (peak … MB) of 512 MB · RSS api 354 MB, children 95 MB`. Render Free has no memory metrics,
  so this is the way to see how close the service is to its limit. The `container` figure comes
  from the cgroup and is what the limit applies to; the per-process RSS shows the split between
  the API and `amr-mcp` but cannot be added up, because both count the shared libraries.
  Connecting to Neon times out after 10 s, so an unreachable database makes the deploy fail fast
  instead of hanging at "Waiting for application startup".
- **Traces.** Export failures never reach Langfuse; they show up in the Render logs as
  `Failed to export spans batch code: 401, reason: Unauthorized`. A 401 means the header is wrong:
  `Basic%20Og==` is the base64 of an empty `:`, i.e. the keys were empty when the header was built.
  Tracing adds about 20 MB (the container measured 375–397 MB with it). Only GenAI spans are exported
  (FastAPI's own HTTP spans are off), so `/health` checks and page views cost no Langfuse units.
- **Updating the corpus.** Ingest locally (`amr-ingest`), check the evals, then run `amr-copy-db`
  again: it upserts by PMID and refreshes the keyword statistics.
