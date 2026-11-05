# Deployment

Production runs on free tiers: **Neon** (Postgres + pgvector) and **Render** (the Docker image in
this repo). Render deploys `main` automatically once CI passes.

```
Browser ──► Render (FastAPI + demo page + embeddings) ──► Neon (pgvector)
                                                   └────► Groq (LLM)
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

## Operating notes
- **Cold start.** Render Free sleeps after 15 minutes idle; the first request then takes 30–60 s.
  Neon also scales to zero, which adds about a second.
- **Quota.** `/ask` is limited to 5 questions per minute per IP and 50 per day in total
  (`ASK_PER_IP_PER_MINUTE`, `ASK_PER_DAY`), to protect Groq's free daily token budget.
- **Updating the corpus.** Ingest locally (`amr-ingest`), check the evals, then run `amr-copy-db`
  again: it upserts by PMID and refreshes the keyword statistics.
