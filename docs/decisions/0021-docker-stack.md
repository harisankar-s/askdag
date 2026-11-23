# 21. One-command Docker stack: app and a seed job in compose, the shop project as demo data, Ollama on the host

**Status:** accepted, 2026-11-23

## Context
A reader should get a working answer from a clean clone with one command and no model, key or `.env`. Until now the quickstart needed uv, Python 3.12 and five CLI commands. The Cal-ITP data can't be part of that path: the manifest is built from an AGPL project and never committed.

## Decision
- **`docker compose up` runs three services:** `postgres` (unchanged), `seed` (one-shot), `app` (uvicorn on port 8000). `app` waits for `seed` to exit 0; `seed` waits for Postgres to be healthy.
- **The demo data is `examples/shop`**, the committed quickstart project (9 models, owners recorded in dbt, 3 incidents). `seed` runs `init-db`, `load-manifest`, `embed --project shop`, `load-docs`. Each replaces what it loaded, so it runs again on every `up` with no "already seeded" flag. `embed` does nothing under `EMBED_PROVIDER=none`, so the default answers search with keywords and says so.
- **One image for both jobs**, built in two stages: uv installs the locked, non-dev dependencies and the package as a wheel into `/app/.venv`; the runtime stage copies only the venv and the shop example, and runs as a non-root user. `.dockerignore` is an allow-list, so `.env`, `data/` and `.venv` can't leak into a layer.
- **A seed job, not an entrypoint script.** Loading at app start would rerun on every restart and every replica, and would hide load failures inside a crash loop. A separate job fails visibly (`docker compose logs seed`) and the app never starts on a half-loaded database. This is the migration-job pattern of a real deployment, at laptop scale.
- **Settings come from `.env` if it exists** (`env_file`, `required: false`). Compose rewrites only `DATABASE_URL` and the two base URLs, because `localhost` inside a container is the container. `DOCKER_LLM_BASE_URL` and `DOCKER_EMBED_BASE_URL` default to the host's Ollama; for Claude, set `DOCKER_LLM_BASE_URL` to Anthropic's endpoint. Separate names, so the same `.env` still works for the CLI on the host.
- **Ollama stays on the host, not in compose.** Docker on macOS has no GPU access, so qwen2.5:7b in a container runs on the CPU only, several times slower than Ollama on the host using Metal. On Linux with a GPU a container would work, but it needs the NVIDIA toolkit and a multi-gigabyte model volume, and the host install is one command on every OS. `host.docker.internal` (with `host-gateway` for Linux) reaches it from the containers.

## Consequences
- The README quickstart becomes `git clone`, `docker compose up --build`, open `http://localhost:8000/`. The uv path stays for development and for Cal-ITP.
- The image holds no Cal-ITP data or code from their repo; loading Cal-ITP in Docker means mounting `data/` and running the CLI in the container, documented, not automated.
- CI runs the full `up` and one question on every push (ADR 0024 reverses the earlier "image build only" call here).

## Measurements
First run on an Apple Silicon laptop, Docker Desktop, Postgres volume already holding Cal-ITP, `.env` with qwen2.5:7b and nomic-embed-text on the host's Ollama:

| | |
|---|---|
| image size | 393 MB on disk, 79.5 MB compressed |
| `up --build -d` with the image cached | 5.8 s |
| seed run (init-db, load, embed, load-docs) | 3.4 s |
| app start to healthy | under 10 s |
| cold build | not timed (first run interrupted); TODO on a clean machine |

Shop questions asked through the containers (13): lineage, follow-up direction flip, the three owner sources, `none recorded` and search all right. History found the right incident first every time, but the LLM summarised only one of the two `mart_revenue_daily` incidents and declined "Why were customer regions blank?" although inc-003 ranked first. "What's the weather in Lisbon?" was read as search, not out of scope; the low-confidence flag (0.468 under 0.667) caught it. None of these is a Docker issue: host Ollama works through `host.docker.internal`.
