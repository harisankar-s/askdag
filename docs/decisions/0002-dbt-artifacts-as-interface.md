# 2. Read dbt artifacts; don't depend on dbt

**Status:** accepted, 2026-11-21

## Context
asklineage needs a project's models, lineage, docs and tests. It could import dbt as a library, or read the JSON artifacts dbt writes.

## Decision
asklineage reads `manifest.json` (and later `catalog.json` and `run_results.json`) and never imports dbt. dbt runs separately, for example through `uvx`, to produce the artifacts. The parser accepts only manifest schema versions it knows (currently v12) and fails with a clear error otherwise.

## Consequences
- Works with any warehouse adapter and needs no warehouse credentials.
- No version conflicts between asklineage and a team's dbt install.
- Each new manifest schema version needs a small parser update and a test.
- Without a warehouse connection there's no `catalog.json`, so column data types are mostly empty.

## Alternatives considered
- **Import dbt-core:** gives parsing for free, but ties asklineage to one dbt version and its dependencies.
- **dbt Cloud Discovery API:** richer, but only for dbt Cloud customers.
