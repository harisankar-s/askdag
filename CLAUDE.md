# asklineage

An open-source assistant that answers data-team questions about a dbt project: where a model's data comes from, what breaks downstream, who owns it, whether it has broken before and how it was fixed. It's a reference build for small teams, compared against Snowflake Cortex, dbt Wizard, Atlan and DataHub.

The author is a data engineering architect building this to learn enterprise RAG, publish a reusable public repo, and prepare for architect interviews. Every design choice should be explainable as a trade-off.

## How to work with the author

- **Write the code; don't run commands.** Give numbered steps with each command in its own `bash` block, what it does, and the expected output. The author runs everything in the VS Code terminal and pastes results back. Reading files to check things is fine.
- Explain the idea and the design before writing code, then stop after each step for questions.
- Be direct and push back when a choice doesn't hold up.
- For external downloads, give the browser option first (URL, button, where to unzip), then the CLI.
- No LinkedIn angles mid-build. There will be one launch post at the end (demo video, GitHub link, one trade-off) plus a longer article. Record real trade-offs in `docs/decisions/` as they come up; that's the article's source material.
- Don't copy code from `../ATLAS-Enterprise-AI`. It's someone else's Apache-2.0 project; this repo is a clean rewrite.

## Stack and conventions

- Python 3.12, uv (`uv add`, `uv sync`, `uv run`), FastAPI, psycopg 3 with a connection pool, Postgres 17 + pgvector in `docker-compose.yml`. LangGraph orchestrates `/api/ask` (graph runtime only, no LangChain models; ADR 0008).
- Layout: `src/asklineage/` (package; `eval/` holds the eval harness), `tests/`, `scripts/`, `config/`, `docs/decisions/`, `tools/dbt/`.
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
uv run uvicorn asklineage.main:app --reload
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{"node": "dim_schedule_feeds"}'
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{"question": "What depends on dim_schedule_feeds?", "start_thread": true}'
uv run asklineage prune-threads --older-than-hours 24
uv run asklineage eval conformance data/manifests/calitp/manifest.json
uv run asklineage eval golden data/manifests/calitp/manifest.json
uv run asklineage eval run data/manifests/calitp/manifest.json --source-repo data/raw/data-infra
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
| 1 | Lineage MVP | In progress |
| 2 | pgvector semantic search + question classifier | Planned |
| 3 | Synthetic incidents, runbooks, Slack threads; hybrid retrieval | Planned |
| 4 | Refresh on dbt deploy; second project (Snowflake); manifest diffing | Planned |
| 5 | Experiments: pgvector vs. Pinecone, LLM providers, column-level lineage | Planned |
| 6 | Hardening: auth, OpenTelemetry, deployment, eval in CI | Planned |

Phase 1 steps:
1. Project setup — done
2. Get the dbt manifest (Cal-ITP) — done
3. Schema, manifest parser, loader, owner rules — done
4. Lineage queries — exact-or-candidates name resolution (pg_trgm suggestions), depth-bounded upstream/downstream CTE with `truncated` flag, `asklineage lineage` CLI, Postgres tests + CI service — done
5. LLM provider interface + lineage answer: fixed table from SQL + LLM summary + grounding check
   - 5a provider Protocol, `ScriptedProvider`, LLM settings, ADR 0006 — done
   - 5b OpenAI-compatible adapter + provider factory + `asklineage llm-check` (adds `openai`; openai 3.x runs on `httpx2`, so adapter tests mock with `httpx2.MockTransport`) — done
   - 5c `LineageFacts` (pure; counts cover every node, names capped by `LLM_MAX_NAMED_NODES`) + claims schema + prompt — done
   - 5d grounding check (pure) + template + reusable `generate_checked` loop (`answer/generate.py`); ADR 0007 — done
   - 5e `lineage --summarize [-v]` CLI — done. Redesigned to option B after three Cal-ITP runs; run 4 moved owners out of LLM text into code-added tags (ADR 0007 has the history and the A/B/C/D comparison); run 5: both directions grounded on the first attempt (highlights true but not yet well chosen; eval in Step 7)
6. LangGraph flow and `/api/ask` (ADR 0008)
   - 6a done (158 tests; first Cal-ITP runs in ADR 0008): stateless graph (`api/flow.py`): `parse` (LLM reads the question; entity must appear in it as a whole name) or `read_fields` (structured, works with `LLM_PROVIDER=none`) → `resolve` → `lineage` → `answer`, with `abstain` and `clarify` branches; `POST /api/ask` returns `kind` = lineage | clarification | abstain, attempts tagged by stage
   - 6b done (ADR 0009; tested on Cal-ITP: typo → resumable clarification → choose → follow-up): opt-in threads (`start_thread` → server-issued `thread_id`, `ask_threads` table, 404/409), `PostgresSaver` on a second autocommit pool (`init-db` runs `setup()`), `begin` clears turn-scoped keys, `clarify` uses `interrupt()` and `choose` resumes it (choice must be a candidate), follow-ups via `same_node` reuse `previous`, `durability="exit"`, explicit serializer allowlist `STATE_TYPES`, `prune-threads`; flow tests use `InMemorySaver` with the same serializer
7. **In progress:** Eval v1 (ADR 0010)
   - 7a ground truth + conformance: `eval/truth.py` (stdlib only, BFS over `depends_on`), `eval/conformance.py` (every node × both directions × depths 1, 2, 3, 5, 25: node set, fewest hops, edges, `truncated`; refuses a stale load), `asklineage eval conformance`; CI runs it on `tests/synthetic_manifest.py` (seeded, about 170 nodes) with a mutation test — done; Cal-ITP: 8,650 walks, 0 mismatches, 8.7 s
   - 7b.1 golden question set (`eval/golden.py`, `eval/templates.py`, `asklineage eval golden`): generated at run time, never committed; role-based node selection, expected outcomes from the manifest and ADR 0004's exact-name rule, phrasing seeded per item id, set sha; about 46 items / 56 turns — done; reviewed on Cal-ITP (a follow-up with no depth keeps the previous depth, confirmed)
   - 7b.2 `eval/score.py` (one outcome per turn: correct, acceptable, safe_wrong, silent_wrong, blocked, error; the answer's own walk is always checked against the manifest), `eval/run.py` (`run_ask` per turn, threads on `InMemorySaver`, warm-up parse, run metadata), `eval/report.py` (per-layer summary, counts with % only from 20 turns), `asklineage eval run` (exits 1 on lineage, facts or error bugs) writing `data/eval/runs/<time>-<model>/` — done; Cal-ITP without an LLM: 55 correct, 1 expected block, 0 silent wrong, lineage and facts 13/13
   - 7c `AttemptInfo.rules` (violation rule names); `eval/highlights.py`: first-attempt failures split into hallucination vs output rules, kept highlights re-checked against manifest edges and paths, usefulness proxy (minority-team hit, coverage, alphabetical bias, reach past one hop) vs code baseline (first named nodes via `build_facts`) and seeded random — done. qwen2.5:7b on Cal-ITP: 51 correct / 1 acceptable / 2 safe wrong / 1 silent wrong of 55; reading 24/25; highlights true (0 of 96 contradicted) but worse than random at choosing (minority-team hit 3/17 vs code 17/17, random 16/17); parse p50 691 ms, highlights p50 4.95 s. Then: `check_parse` rule `same_node_names_another_node` (project node names passed on follow-ups) and `recover_reading` (a typo the model corrected is resolved from the question's only name-like token) — done. Rerun: 53 correct / 1 acceptable / 0 safe wrong / 2 silent wrong; both fixes worked, and the two silent errors are now direction errors the old behaviour hid. Next: put the parse reply on `AttemptInfo` so a direction error can be traced to the model or to the flow's carry-over, then rerun
   - **Deferred (own ADR, before Phase 2):** choose highlights in code; the LLM picks worse than random (ADR 0010). Also still to do: more follow-up and typo cases in the golden set before quoting rates
   - 7d (optional) hand labels to check the usefulness proxy
8. README update: architecture, quickstart, sample data with the Cal-ITP commit hash, first results

## Evaluation approach

Evaluate each layer separately: classifier accuracy, name resolution, lineage set precision/recall (must be 100%; a miss is a bug), recall@k for semantic and incident search, answer correctness, correct abstention on unanswerable questions, and hallucination rate per 100 lineage answers. Lineage ground truth is generated from the manifest. LLM-as-judge uses a different model family from the generator and is checked against hand labels.

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
