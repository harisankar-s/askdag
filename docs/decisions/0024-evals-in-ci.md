# 24. CI runs the evals that guard a promise: lineage conformance on real manifests and the quickstart; retrieval evals stay local

**Status:** accepted, 2026-11-23

## Context
The Phase 6 plan said "CI runs conformance and the retrieval evals with `HashingEmbedder`". Looking at what each would guard:

- **Lineage conformance** already runs in `pytest` on the synthetic manifest (ADR 0010). That manifest is written by our own code, so it can't catch a dbt manifest shape our parser misreads.
- **Search and incident evals** need their question sets, and both sets are about Cal-ITP, whose manifest is never committed. Run on the shop project with `HashingEmbedder`, hit@k would measure hash collisions: a number nobody can act on. The eval code paths themselves are already covered by tests on Postgres (`test_search_eval.py`, `test_incident_search.py`).
- **The quickstart** (ADR 0021) is the promise a reader meets first, and nothing checked it.

## Decision
- **`test` job, one more step:** load each committed shop manifest (`manifest.v1.json`, `manifest.json`, both written by dbt 1.10.8) and run `asklineage eval conformance` on it. Every walk from every node, both directions, every depth, must match an independent BFS over `depends_on`; any mismatch fails the build. It runs through the CLI, so loading and the truth graph are tested as a user runs them.
- **`quickstart` job:** `scripts/smoke_quickstart.sh` runs `docker compose up --build` from the clean checkout with no `.env`, waits for `/health/ready`, checks `/api/info` lists `shop`, and asks for `int_order_payments` downstream, expecting exactly its four marts. On failure it prints the seed and app logs. The same script runs on a laptop.
  This reverses ADR 0021's "a full `up` in CI isn't worth the minutes": the image build alone would not catch a seed that fails or a wrong path in the compose file, and those are the failures a reader would hit.
- **Retrieval evals are not in CI.** Their job is to compare configurations (ADR 0012, 0015, 0018), which needs real embeddings and the Cal-ITP questions, so they stay local commands with results written into ADRs.
- **No LLM in CI.** No key in the repo's secrets, no cost per push, no flaky gate on a model's wording. The golden set stays a local run (ADR 0020).

## Measurement
Local timings (Apple Silicon); the GitHub runner numbers go here after the first push.

| step | time |
|---|---|
| conformance, shop v1 + current | 250 walks, 0 mismatches; 4.5 s with loads (0.1 s each in walks) |
| quickstart smoke, image cached | 10.6 s |
| quickstart job on a GitHub runner | TODO |
