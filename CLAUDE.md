# asklineage

An open-source assistant that answers data-team questions about a dbt project: where a model's data comes from, what breaks downstream, who owns it, whether it has broken before and how it was fixed. It's a reference build for small teams, compared against Snowflake Cortex, dbt Wizard, Atlan and DataHub.

The author is a data engineering architect building this to learn enterprise RAG, publish a reusable public repo, and prepare for architect interviews. Every design choice should be explainable as a trade-off.

## Scope

- A reference build, not a service: no real users, no production operations (no on-call, SLAs, multi-tenancy, cloud deployment). Code and design choices are still production grade: typed, tested, bounded, honest failure modes.
- The deliverable is a repo anyone can clone and run on a laptop, plus an article explaining the trade-offs.
- An eval earns its place only if it decides something or guards a promise (lineage is exact; answers cite what they used). Each ADR names the number that decided it. If no decision depends on a measurement, don't build the harness; argue the trade-off in the ADR instead.

## How to work with the author

- **Write the code; don't run commands.** Give numbered steps with each command in its own `bash` block, what it does, and the expected output. The author runs everything in the VS Code terminal and pastes results back. Reading files to check things is fine.
- Explain the idea and the design briefly before writing code, then stop after each step for questions.
- Keep chat replies crisp and short to save tokens: bullets over prose, no recaps of what the code already shows, no long rationale. Put depth in ADRs, not in chat.
- Be direct and push back when a choice doesn't hold up.
- For external downloads, give the browser option first (URL, button, where to unzip), then the CLI.
- No LinkedIn angles mid-build. There will be one launch post at the end (demo video, GitHub link, one trade-off) plus a longer article. Record real trade-offs in `docs/decisions/` as they come up; that's the article's source material.
- Don't copy code from `../ATLAS-Enterprise-AI`. It's someone else's Apache-2.0 project; this repo is a clean rewrite.

## Stack and conventions

- Python 3.12, uv (`uv add`, `uv sync`, `uv run`), FastAPI, psycopg 3 with a connection pool, Postgres 17 + pgvector in `docker-compose.yml`. LangGraph orchestrates `/api/ask` (graph runtime only, no LangChain models; ADR 0008).
- Layout: `src/asklineage/` (package; `eval/` holds the eval harness, `web/` the static UI page, `corpus/` the synthetic document format and scaffold), `tests/`, `scripts/`, `config/`, `corpus/calitp/` (synthetic incidents, runbooks, Slack threads; committed), `docs/decisions/`, `tools/dbt/`.
- Ruff for lint and format (line length 100); CI runs `ruff check`, `ruff format --check` and `pytest` (`.github/workflows/ci.yml`).
- Keep parsing pure (no database) so it's unit-testable with `tests/fixtures/mini_manifest.json`; database writes live in `ingest/load.py`.
- `src/asklineage/sql/schema.sql` is idempotent (`IF NOT EXISTS`) and applied by `asklineage init-db`. Changing an existing column needs a manual migration; say so when it comes up.
- Add a dependency only in the step that first needs it.
- Default to no comments; one short line only when the reason isn't obvious.

## Common commands

```bash
docker compose up -d
uv run ruff format . && uv run ruff check . && uv run pytest
uv run asklineage init-db
uv run asklineage load-manifest data/manifests/calitp/manifest.json --owners config/owners.calitp.yml
uv run asklineage lineage dim_schedule_feeds --direction downstream --depth 2
uv run asklineage lineage dim_schedule_feeds --direction downstream --depth 2 --summarize -v
uv run python scripts/manifest_stats.py data/manifests/calitp/manifest.json
uv run asklineage llm-check
uv run asklineage embed-check
uv run asklineage embed
uv run asklineage search "Which table has the latitude and longitude of every stop?"
uv run uvicorn asklineage.main:app --reload
open http://localhost:8000/
curl -s localhost:8000/api/info
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{"node": "dim_schedule_feeds"}'
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{"question": "What depends on dim_schedule_feeds?", "start_thread": true}'
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{"question": "Has dim_stops broken before, and how was it fixed?"}'
uv run asklineage prune-threads --older-than-hours 24
uv run asklineage eval conformance data/manifests/calitp/manifest.json
uv run asklineage eval golden data/manifests/calitp/manifest.json
uv run asklineage eval search-set data/manifests/calitp/manifest.json
uv run asklineage eval search data/manifests/calitp/manifest.json
uv run asklineage eval run data/manifests/calitp/manifest.json --source-repo data/raw/data-infra
uv run asklineage corpus check data/manifests/calitp/manifest.json
uv run asklineage load-docs
uv run asklineage eval incidents
```

Stop `uvicorn --reload` before `uv add` or `uv sync`: the reloader watches `.venv`, restarts mid-install, and can leave port 8000 held by a dead worker, so requests hang instead of failing.

## Sample data

- **Cal-ITP `data-infra`** (github.com/cal-itp/data-infra, BigQuery, AGPL-3.0). The dbt project is in `warehouse/`, cloned into `data/raw/data-infra/` (git-ignored). Never commit their files or the manifest.
- GitLab's analytics repo was the first choice but went private in 2026 (403, docs site behind sign-in).
- Manifest produced with `uvx --python 3.12 --from "dbt-core==1.10.8" --with "dbt-bigquery==1.10.2" dbt parse --project-dir data/raw/data-infra/warehouse`, using `DBT_PROFILES_DIR=tools/dbt` (placeholder BigQuery profile `calitp_warehouse`). Parse warnings about missing `external_enghouse` sources come from a bug in their project and can be ignored.
- Manifest stats: 628 models, 220 sources, 17 seeds, 351 tests, 1,198 edges into models, 67% of models and 79% of columns described, **0 owners in dbt**.
- Owners come from `config/owners.calitp.yml`: synthetic, path-based, with some folders left unowned on purpose (27 nodes have no owner).
- `parse` gives no `catalog.json` (column types) or `run_results.json` (run history); those need a live warehouse.

## Architecture (target)

Sources (dbt manifests, model descriptions, incident docs) → ingestion (manifest parser, chunk and embed, incident loader) → Postgres + pgvector (lineage tables, vector chunks, incidents) → LangGraph flow (guard → classify → retrieve → answer + grounding check) → FastAPI. The LLM sits behind a narrow provider interface (`llm/base.py`). `LLM_PROVIDER=none` (the default) uses a template summary so a fresh clone needs no model; the first real adapter is OpenAI-compatible (local Ollama `qwen2.5:7b` for testers; any compatible gateway for enterprises). The author has no Bedrock account; a native Bedrock adapter is optional and later. A separate eval harness measures every layer.

Key design rules:
- **For lineage answers, code writes the facts; the LLM only adds highlights.** The table and the summary paragraph (counts, teams affected, `owner_source`, unowned, truncation) come from the SQL result. The LLM may add 1–3 relation highlights, each checked in code (edge or path exists in the stated direction, names, no numbers, no owner wording; code adds owner tags); failing highlights are dropped after one retry. Compute what is computable; generate only what needs language (ADR 0007).
- Lineage is model-level only; column-level lineage needs SQL parsing (sqlglot, Phase 5).
- Answers about owners must say where the owner came from (`owner_source`) and say when none is recorded.

## Plan and status

| Phase | Goal | Status |
|---|---|---|
| 0 | Starter kit: FastAPI, Postgres + pgvector, CI | Done |
| 1 | Lineage MVP | Done |
| 2 | pgvector semantic search + question classifier | Done |
| 3 | Incident retrieval: synthetic corpus, hybrid search, cited "has it broken before?" answers | Done |
| 4 | Change awareness: manifest diff and incremental refresh; a tiny second project as the reader quickstart | Planned |
| 5 | Measured experiments on existing evals: HNSW vs exact scan, a second LLM, highlights chosen in code | Planned |
| 6 | Release: one-command Docker stack, API key and tracing, cheap evals in CI, README, demo, article | Planned |

Phase 1 is built (steps 1 to 8); the history and numbers live in ADRs 0001–0011 and the README, so they aren't repeated here. What to know when coding:
- Modules: `ingest/` (manifest parser, loader, owner rules), `lineage/` (name resolution, bounded walk), `llm/` (provider interface, OpenAI-compatible adapter, `ScriptedProvider` for tests), `embed/` (embedding interface, OpenAI-compatible adapter, `HashingEmbedder` for tests; ADR 0012), `corpus/` (synthetic document format, checks, scaffold; ADR 0014), `incidents/` (incident chunks, name detection, index; ADR 0015), `answer/` (facts, grounding, `generate_checked`, parse checks), `api/` (flow, ask, threads, models, info), `eval/` (truth, conformance, golden set, score, run, highlights), `web/index.html` (static page, no `innerHTML`, no JS tests; checked by hand).
- openai 3.x runs on `httpx2`, so adapter tests mock with `httpx2.MockTransport`.
- `/api/ask` threads use `PostgresSaver`; every type stored in flow state must be in `STATE_TYPES` (`api/flow.py`). `LineageResponse.edges` is required and the eval checks it against the manifest.
- Eval baseline on Cal-ITP: conformance 8,650 walks, 0 mismatches; qwen2.5:7b final run 55 correct, 1 acceptable, 0 wrong of 56. Highlights are true but chosen worse than random; choosing them in code is a Phase 5 experiment.
- Open items: replace the clone URL placeholder in `README.md` and add `docs/images/web-ui.png`.

Phase 2 steps:
- 2a embedding provider interface (ADR 0012): `EmbeddingProvider.embed(texts, kind)`, separate `EMBED_*` settings, task prefixes as settings (nomic defaults), dimension check, `none` default, `HashingEmbedder` for CI, `asklineage embed-check` — built, awaiting review
- 2b first the hand-written search questions (`config/search_questions.calitp.yml`, 35 questions in 5 categories: described, column, jargon, undescribed, unanswerable; `eval/search_set.py` checks them against the manifest and shows word overlap; `asklineage eval search-set`) — drafted (d10 reworded); the author may add questions. Then chunks (`embed/chunks.py`: summary/columns/sql, 2,000-char pieces, header on each) + `node_chunks` (no FK to nodes; re-embed on text hash or model change) + `asklineage embed` (`embed/store.py`) — done; Cal-ITP: 2,799 chunks (875 summary, 952 columns, 972 sql) in 62 s, rerun embeds 0
- 2c search query and CLI; keyword (full-text) fallback when `EMBED_PROVIDER=none`, and the answer says which path ran — done (`search/query.py`: best chunk per node by cosine, `--kinds` filter, chunks only from the configured model; keyword = OR'd full-text over name, description, columns, SQL; no abstain threshold yet); spot checks: c06 hit (staging first), u01 miss, n01 top 0.667
- 2d `intent` added to the parse call (not a second LLM call), with owner lookup answered from SQL
  - 2d.1 (ADR 0013): `intent` replaces `is_lineage`; `owner` node + `OwnerResponse` from SQL; `out_of_scope` replaces `not_lineage`; golden set v2 (owner items) — done; qwen2.5:7b: 57 correct, 1 acceptable, 0 wrong of 58 after `named_node_wins` (a follow-up naming a node beats same_node); owner 4/4
  - 2d.2 search in `/api/ask` (`SearchResponse`, `low_confidence` flag under `SEARCH_LOW_CONFIDENCE_BELOW`, `SearchOptions` through `run_ask`; golden v3 with 2 search-intent turns) — done; qwen2.5:7b 59 correct, 1 acceptable, 0 wrong of 60
  - 2d.3 web UI for owner and search kinds (`renderOwner`, `renderSearch`, ADR 0011 addendum) — done, checked by hand
- 2e search eval: recall@k on the hand-written questions, both search paths — built (`eval/search_eval.py`, `asklineage eval search`: 6 configs, hit@k/MRR/recall@10 by category, cutoff table, writes `data/eval/search/`); vectors now tagged with model + document prefix (`index_name`), so a prefix change re-embeds. Cal-ITP (ADR 0012 table): semantic all kinds hit@5 19/30, MRR 0.525 vs keyword 9/30, 0.161; prefixes kept (better cutoff); cutoff 0.667 catches 3/5 unanswerable, blocks 1/30, so it is a flag, not an abstain. Then: snake_case split in column/sql chunks (kept, noise-level); family metric fam@1 18 vs hit@1 12; lineage-aware ranking (`SEARCH_PREFERRED_PATHS`, lift ≤3 marts within 2 hops, 0.98 for non-preferred, `via`) on by default; Cal-ITP all kinds: hit@1 16/30, hit@5 22/30, MRR 0.614 (from 12, 20, 0.519); cutoff unchanged

Phase 3 steps:
- 3a corpus (ADR 0014): `corpus/docs.py` (Markdown + YAML front-matter; front-matter is ground truth; `check_corpus` holds facts to the manifest and lineage, mention exact/informal enforced), `corpus/scaffold.py` (seeded skeletons, Zipf-skewed symptoms, decoys, refuses to overwrite), `asklineage corpus scaffold|check` — done; Cal-ITP seed 14: 40 incidents (11 informal, 6 decoys, no incremental_gap drawn), 12 runbooks, 25 threads (15 about incidents), prose hand-written; `dim_schedule_feeds` has 9 incidents
Phase 3 is kept minimal: only incidents are indexed. Runbooks and Slack threads stay in the corpus, unloaded. They are the obvious next step; ADR 0015 explains why they aren't built.
- 3b (ADR 0015) `incidents/chunks.py` (one chunk per section with an id/date/severity/title header; `detect_nodes` finds exact names in the text: front-matter is never read by the app), `incidents/store.py` (`index_incidents`: embed first, then replace the project's `documents` + `doc_chunks` in one transaction; vectors optional, `tsvector` generated), `asklineage load-docs` — done; Cal-ITP: 40 incidents, 200 chunks in 3 s; name detection links all 29 exact incidents fully and none of the 11 informal ones (63 links). Index, not record: `source` + `url` point back at the system of record
- 3c (ADR 0015) `config/incident_questions.calitp.yml` (17 questions: named 4, symptom 5, loose 3, decoy 3, none 2), `incidents/search.py` (`find_incidents`: names = detected names + 4 hops upstream; keyword = OR'd full-text; vector; hybrid = RRF k=60 over the top 30 of each; routed = names first, then hybrid), `eval/incident_eval.py`, `asklineage eval incidents` — done; Cal-ITP hit@1/MRR: names 3/0.233, keyword 14/0.956, vector 11/0.844, hybrid 13/0.922, routed 12/0.889; all but names saturate at hit@3 15/15; nothing abstains on the 2 unanswerable questions. Keyword's lead comes from one author writing both prose and questions. Decision: 3d uses hybrid (ADR 0015 addendum)
- 3d (ADR 0016) `history` intent in the parse call (no node required; question goes to hybrid search as asked); `answer/history.py` (code lists top 3 incidents from SQL; LLM writes ≤4 sentences or `answered: false`; each sentence must cite given ids, mention only cited ids, name only nodes its cited incidents name, use only numbers from its cited incidents or their dates; one retry; else list only); flow node `history`, `HistoryResponse`, `renderHistory`; golden v4 adds 3 history turns (intent only) — first run: right incidents, right abstention, but the model invented years ("7 May 2023" for a 2025 incident); number check added; rerun right years and passes first try; web card checked by hand — done

Phases 4–6 (slim; cut items are argued in ADRs, not built):
- 4 `asklineage diff old.json new.json` (added, removed, changed nodes and edges, and what is downstream of the changes); `load-manifest` + `embed` already re-embed only changed chunks, so refresh is a documented sequence, not a service. A tiny public dbt project (jaffle-shop style) committed as the 5-minute quickstart and the portability check. Cut: deploy webhooks, a Snowflake project
- 5 three experiments, each one table in an ADR, each on an eval that already exists: pgvector HNSW vs exact scan (recall@k and latency on the search and incident sets); a hosted model vs qwen2.5:7b on the golden set; highlights chosen in code vs by the LLM (existing highlight eval). Cut: Pinecone (argued, not run), column-level lineage (named as a limitation; needs compiled SQL, which `dbt parse` doesn't give)
- 6 `docker compose up` runs app + Postgres (+ optional Ollama) from a clean clone; an API-key dependency on `/api/*` to show the pattern; OpenTelemetry spans per flow node and LLM/embedding call (console or Jaeger in compose); CI runs conformance and the retrieval evals with `HashingEmbedder` (no LLM); README quickstart, demo video, launch post and article from the ADRs. Cut: real auth, cloud deployment, bigger golden set, hand labels, LLM-as-judge

## Evaluation approach

Each layer has one eval, and each eval guards a promise or decides a choice:
- Lineage: exhaustive conformance against an independent BFS; must be 100% (a miss is a bug).
- Parse/intent and name resolution: the golden set (regression; small additions only).
- Search and incident retrieval: hand-written questions, hit@k and MRR, compared across configs; the cutoff table says whether abstaining is possible.
- Answers: grounding is enforced in code at runtime (checked highlights, cited sentences), so it needs no judge model. The golden run counts correct, acceptable and wrong.
No LLM-as-judge and no hand-labelled answer set in this build; the ADRs say why.

## Decision records

- `0001` Postgres adjacency list + recursive CTEs for lineage, not a graph database
- `0002` Read dbt artifacts; never import dbt
- `0003` Path-based owner rules when dbt has no owners
- `0004` Name resolution returns candidates; a fuzzy match never answers
- `0005` Bounded traversal: depth cap + per-level dedup, not path tracking
- `0006` Narrow LLM provider interface; OpenAI-compatible first; `none` by default
- `0007` Code writes the lineage facts; the LLM adds checked relation highlights; generic retry loop
- `0008` LangGraph orchestrates `/api/ask` (for 6b checkpoints and Phase 3 loops, and to learn it); plain functions would do for 6a
- `0009` Conversation threads: opt-in, server-issued IDs, Postgres checkpoints, turn-scoped state reset, `interrupt()` clarifications, follow-ups, locks and pruning
- `0010` Eval v1: exhaustive lineage conformance against an independent stdlib BFS over `depends_on`; synthetic manifest in CI; question set per layer (7b–7d)
- `0011` Web UI: one static HTML page served by FastAPI, using only `/api/ask` and `/api/info`, with a hand-written SVG graph; not Streamlit/Gradio or React
- `0012` Separate embedding provider interface; task prefixes and dimension as settings; nomic-embed-text default
- `0013` Intent in the parse call (lineage, owner, search, other); owner answers from SQL
- `0014` Synthetic incident corpus: seeded scaffold, hand-written prose, ground truth in front-matter, checked against lineage
- `0015` Incidents: a search index copied from the system of record (not queried live); node links from names detected in the text; incidents only; hybrid search chosen (3c)
- `0016` History answers: code lists incidents, the LLM summarises with citations and node names checked in code; `answered: false` for abstention
