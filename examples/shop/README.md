# shop: the quickstart project

A small dbt project written for asklineage. It is the 5-minute quickstart, and it checks that nothing in asklineage is specific to Cal-ITP.

- 3 sources (`app.orders`, `app.customers`, `app.payments`), 1 seed, 9 models in `staging/`, `intermediate/` and `marts/`
- Owners recorded in dbt, as most teams would: `meta.owner` on staging (from `dbt_project.yml`) and on the finance marts, a dbt group on `dim_customers`. `int_order_payments`, the sources and the seed have no owner on purpose.
- `manifests/manifest.v1.json` and `manifests/manifest.json` are two parses of this project, one change apart, for `asklineage diff`. The project files match `manifest.json`.
- `corpus/` holds three hand-written incidents in the ADR 0014 format.

Rebuild a manifest from the repo root (dbt `parse` never connects to a database):

```bash
DBT_PROFILES_DIR=tools/dbt DBT_SEND_ANONYMOUS_USAGE_STATS=false uvx --python 3.12 --from "dbt-core==1.10.8" --with "dbt-duckdb==1.11.1" dbt parse --project-dir examples/shop
```

The manifest is written to `examples/shop/target/manifest.json`.
