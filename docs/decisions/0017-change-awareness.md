# 17. Change awareness: a pure manifest diff, data versus docs-only changes, refresh as a documented sequence

**Status:** accepted, 2026-11-22

## Context
A data team's first question about a change is "what does this break, and whom do I tell?" asklineage answers lineage questions about one loaded manifest. Phase 4 adds a comparison of two manifests and checks that refreshing after a deploy keeps the stored lineage, chunks and incident links in step. It also adds a second, tiny project, so a reader can try asklineage in five minutes and the build is shown not to depend on Cal-ITP.

## Decision
- **`asklineage diff old.json new.json` compares two parsed manifests in memory, with no database.** It reuses the parser (ADR 0002), so it can run in CI on a pull request before anything is loaded. `--exit-code` exits 1 when the manifests differ, like `git diff`. `--json` gives unique ids, every edge and an `affects_data` flag on each change.
- **Every changed node says which fields changed, and each field is one of two kinds:**
  - *data*: SQL, materialization, the relation it writes, its parents;
  - *docs only*: description, documented columns, tags, owner, file path.
  Only data changes start the downstream walk. Columns count as docs: without `catalog.json` they are what the YAML says, not the warehouse schema.
- **Downstream is an unbounded multi-source BFS over the new manifest.** Each node reports its distance in hops, its owner with `owner_source`, and the nearest changed node it was reached from (`via`). The header counts nodes per owner, so the answer to "whom do I tell?" is one line. Changed and added nodes are not listed again. A docs-only node below a data change is.
- **Renames are a remove plus an add.** dbt has no identity across a rename. A removed and an added node with identical SQL, each unique in its set, are reported as a possible rename. Matching names fuzzily would break ADR 0004.
- **Refresh is three commands, documented, not a service:** `load-manifest` (replaces the project's rows in one transaction), `embed` (re-embeds changed chunks, deletes chunks of removed nodes), `load-docs` (re-detects incident links against the new nodes). Search inner-joins chunks to nodes, so a removed node never appears in results, even between `load-manifest` and `embed`.
- **The quickstart project is written for this repo** (`examples/shop/`), not copied from jaffle-shop. It has 9 models, 3 sources, 1 seed and 3 incidents. Owners come from dbt `meta` and a dbt group, with some nodes left unowned on purpose. Two committed manifests, one change apart, demo `diff` without any git setup.

## Options compared
| | **In-memory diff over the parser (chosen)** | Load both manifests, diff in SQL | `dbt ls --select state:modified+` |
|---|---|---|---|
| Needs a database | no | yes, two project copies | no, but needs dbt and the project |
| Runs before a merge | yes | after a load | yes |
| Says why a node changed | per field | per field | no; sub-selectors such as `state:modified.body` filter by kind but don't report it |
| Separates docs-only changes | yes | possible | partly, through the sub-selectors |
| Owners of downstream nodes | yes, with source | yes | no |
| Second lineage walk to keep correct | yes (BFS, about 20 lines) | no | no |

## Measurements
**Diff on Cal-ITP**, between the current manifest and the commit 30 model-touching commits earlier:
- 26 added, 26 removed, 57 changed, edges +48 −28.
- **16 of the 57 changes are docs only** (almost all documented columns). Without the split, each of them would look like a reason to warn teams downstream.
- Fields changed: SQL 41, columns 38, parents 6, materialization 3, description 2. Every data change includes a SQL change. No relation, tag, owner or file-path changes occurred, so those checks are covered only by unit tests.
- Downstream of the 41 data changes: 185 nodes across 5 teams (gtfs 97, ntd-reporting 47, transit-database 23, payments 15, tides 3).
- Possible renames found: 0. Pairs such as `bridge_paratransit_services` → `bridge_services_x_paratransit` changed their SQL as well, so the exact-SQL hint misses them.

**Refresh on Cal-ITP**, loading the old manifest and then the current one:
- `embed` re-embedded **235 of 2,799 chunks (8%)**, deleted 58 and kept 2,564, in **5.8 s against 62 s** for a full embed. That is why refresh needs no service: run the three commands after a deploy.
- `load-docs` took 2.2 s, and incident links stayed at 29 incidents and 63 links. No incident names a node that was added or removed.

**Quickstart (`shop`)**, v1 to the current manifest: 1 added, 2 changed (1 data, 1 docs only), 2 downstream, owned by `Growth analytics (dbt_group)` and `finance (dbt_meta)`. This is the first real data where owners come from dbt rather than the owners file: Cal-ITP records none (ADR 0003).

## Consequences
- **Two lineage walks.** `diff` walks the graph in Python, and `/api/ask` walks it in SQL (ADR 0001). The SQL walk is held to an independent BFS by the conformance eval (ADR 0010). The diff's walk is small and unit-tested on diamonds, renames and multi-source cases, but it has no exhaustive check. Running conformance against it would be cheap if the two ever disagree.
- **`via` names one changed node**: the nearest, ties broken by id. A node below several changes shows only one. The JSON could list them all if a reader needs it.
- **Docs-only is a claim about data, not about consumers.** Renaming a documented column without changing the SQL is docs only here, but a BI tool that reads the docs could still break. Exposures would cover that; this build doesn't load them.
- **Incident links go stale between deploys** until `load-docs` runs. A refresh that skips it misses new nodes named in old incidents and keeps links to removed ones. Those links match nothing, so the cost is a missed link, not a wrong one.
- **Portability fix:** `SEARCH_PREFERRED_PATHS` defaulted to Cal-ITP's `models/mart/*`. It now also matches dbt's usual `models/marts/*`, which `shop` uses. It is still one global setting, not per project.
- Cut, as planned: deploy webhooks and a Snowflake project. Webhooks would trigger the same three commands, and a second warehouse changes nothing that `dbt parse` doesn't hide.
