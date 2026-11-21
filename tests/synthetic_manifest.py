"""A seeded synthetic dbt manifest, big enough to catch walk bugs the mini manifest can't.

It has diamonds whose paths differ in length, a hub with many children, a chain longer than
MAX_DEPTH with a shortcut across it, a snapshot, test nodes that must be ignored, and two
sources that share a table name. parent_map and depends_on are written from the same edges,
as dbt writes them.
"""

import random

PROJECT = "synthetic"
SCHEMA_URL = "https://schemas.getdbt.com/dbt/manifest/v12.json"
SOURCES = {
    "raw_app": ("events", "sessions", "users"),
    "raw_billing": ("events", "invoices", "payments"),
    "raw_crm": ("accounts", "contacts"),
}
SEEDS = ("country_codes", "currencies", "plan_tiers")
CHAIN_LENGTH = 30


def chain_id(position: int) -> str:
    return f"model.{PROJECT}.chain_{position:02d}"


def build_manifest(seed: int = 7, models: int = 120) -> dict:
    rng = random.Random(seed)
    nodes: dict[str, dict] = {}
    sources: dict[str, dict] = {}

    def add(resource_type: str, name: str, path: str, parents: list[str] | None = None) -> str:
        unique_id = f"{resource_type}.{PROJECT}.{name}"
        nodes[unique_id] = {
            "unique_id": unique_id,
            "resource_type": resource_type,
            "name": name,
            "package_name": PROJECT,
            "original_file_path": path,
            "config": {},
            "depends_on": {"macros": [], "nodes": list(parents or [])},
        }
        return unique_id

    staging = []
    for source_name, tables in SOURCES.items():
        for table in tables:
            source_id = f"source.{PROJECT}.{source_name}.{table}"
            sources[source_id] = {
                "unique_id": source_id,
                "resource_type": "source",
                "name": table,
                "source_name": source_name,
                "package_name": PROJECT,
                "original_file_path": f"models/staging/{source_name}.yml",
                "config": {},
            }
            name = f"stg_{source_name}__{table}"
            staging.append(add("model", name, f"models/staging/{name}.sql", [source_id]))
    seeds = [add("seed", name, f"seeds/{name}.csv") for name in SEEDS]
    accounts = f"source.{PROJECT}.raw_crm.accounts"
    snapshot = add("snapshot", "snap_accounts", "snapshots/snap_accounts.sql", [accounts])
    hub = add("model", "int_hub", "models/intermediate/int_hub.sql", [staging[0]])

    # Each model reads only from nodes made before it, so the graph stays acyclic.
    pool = [*staging, *seeds, snapshot, hub]
    for position in range(models):
        layer, prefix = ("intermediate", "int") if position < models // 2 else ("marts", "mart")
        parents = set(rng.sample(pool, rng.choice((1, 1, 2, 2, 3))))
        if rng.random() < 0.3:
            parents.add(hub)
        name = f"{prefix}_{position:03d}"
        pool.append(add("model", name, f"models/{layer}/{name}.sql", sorted(parents)))

    previous = staging[-1]
    for position in range(CHAIN_LENGTH):
        name = f"chain_{position:02d}"
        previous = add("model", name, f"models/chain/{name}.sql", [previous])
    # The last chain model is 2 hops from chain_00 through chain_01, not 29 along the chain.
    nodes[chain_id(CHAIN_LENGTH - 1)]["depends_on"]["nodes"].append(chain_id(1))

    for kind, parent in (("not_null", hub), ("unique", pool[-1])):
        name = f"{kind}_{nodes[parent]['name']}_id"
        unique_id = f"test.{PROJECT}.{name}.0a"
        nodes[unique_id] = {
            "unique_id": unique_id,
            "resource_type": "test",
            "name": name,
            "package_name": PROJECT,
            "attached_node": parent,
            "column_name": "id",
            "test_metadata": {"name": kind},
            "config": {"severity": "ERROR"},
            "depends_on": {"macros": [], "nodes": [parent]},
        }

    parent_map = {unique_id: list(node["depends_on"]["nodes"]) for unique_id, node in nodes.items()}
    parent_map |= {source_id: [] for source_id in sources}
    return {
        "metadata": {
            "dbt_schema_version": SCHEMA_URL,
            "dbt_version": "1.10.8",
            "project_name": PROJECT,
            "generated_at": "2026-11-21T00:00:00Z",
            "invocation_id": f"synthetic-{seed}",
        },
        "nodes": nodes,
        "sources": sources,
        "parent_map": parent_map,
    }
