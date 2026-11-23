# 23. Tracing: OpenTelemetry spans per request, flow node and model call; no text in spans; off by default

**Status:** accepted, 2026-11-23

## Context
An `/api/ask` answer runs up to five flow nodes, one or two LLM calls, an embedding call and several SQL queries. The golden run already records per-call latency and tokens in `attempts`, but only for that one response, and only for the LLM. The question an operator asks is "where did this slow answer spend its time?", across requests, in a tool they already have.

## Decision
- **OpenTelemetry, not a vendor SDK or LangSmith.** OTLP goes to Jaeger, Tempo, Honeycomb, Datadog or a cloud collector without code changes. LangSmith would trace LangGraph with no code at all, but it is a hosted service tied to LangChain, and asklineage uses LangGraph only as a runtime (ADR 0008).
- **Library code uses only the OpenTelemetry API**, a no-op until `setup_tracing` installs a provider. The app installs it at import, before the middleware stack is fixed; the CLI and the evals never do, so they pay nothing.
- **`TRACING` = off (default) | console | otlp**, plus `TRACING_OTLP_ENDPOINT`. The standard `OTEL_*` variables would not be read from `.env`, so the endpoint is an ordinary setting.
- **Spans:**
  - `POST /api/ask` from the FastAPI instrumentation (health checks excluded, per-message ASGI spans dropped);
  - `ask`: the root of the flow, with input kind (question, node, choose), project, answer kind, LLM attempts, threaded;
  - `flow.<node>`: one per LangGraph node, by a wrapper at `add_node` (`functools.wraps` keeps the signature LangGraph reads to inject `runtime`). An `interrupt()` for a clarification is tagged, not marked as an error;
  - `chat <model>` and `embeddings <model>`: client spans in the two OpenAI-compatible adapters, named and tagged per the GenAI semantic conventions (operation, request and response model, server, input and output tokens, error type).
- **No question, prompt, reply or incident text in any span.** Traces leave the process, usually to a shared backend with looser access than the database. Counts, names, models and timings answer the latency question; a test asserts the question text never appears. Node names and project names are recorded: they are metadata in the same manifest everyone can read.
- **Jaeger in compose under a `tracing` profile** (`TRACING=otlp docker compose --profile tracing up`). Not in the default stack: a fresh clone shouldn't start a service it doesn't need.
- **No SQL spans yet.** The psycopg instrumentation would add one span per query; the flow node spans already bound the database time per step. Added when a node's time is unexplained.

## Measurement
One history question through the Docker stack, qwen2.5:7b and nomic-embed-text on the host's Ollama (first call after start):

| span | ms |
|---|---|
| `POST /api/ask` | 6,429 |
| `flow.parse` → `chat qwen2.5:7b` (1,225 in / 23 out tokens) | 3,921 |
| `flow.history` | 2,464 |
| ↳ `embeddings nomic-embed-text` | 15 |
| ↳ `chat qwen2.5:7b` (1,014 in / 15 out) | 2,439 |

A structured lineage request: 11 ms in all, `flow.lineage` 2.2 ms. 99.5% of the history answer is the two LLM calls; the embedding and the SQL are noise. This is what the trace is for: it says where to look (model size, prompt length, a hosted model) and where not to (the index, ADR 0018).
