"""Command-line entry point. Run `uv run asklineage --help`."""

import argparse
import json
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from asklineage.config import get_settings
from asklineage.ingest.load import apply_schema, load_manifest
from asklineage.ingest.manifest import UnsupportedManifestError, parse_manifest
from asklineage.ingest.owners import load_owner_rules


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asklineage")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create or update the database tables")
    load = commands.add_parser("load-manifest", help="Load a dbt manifest.json")
    load.add_argument("manifest", type=Path)
    load.add_argument("--owners", type=Path, help="YAML file of path-based owner rules")
    args = parser.parse_args(argv)

    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        apply_schema(conn)
        if args.command == "init-db":
            print("Schema is up to date.")
            return

        rules = load_owner_rules(args.owners) if args.owners else ()
        try:
            parsed = parse_manifest(json.loads(args.manifest.read_text()), rules)
        except UnsupportedManifestError as exc:
            parser.error(str(exc))
        load_id = load_manifest(conn, parsed)

    node_types = Counter(node.resource_type for node in parsed.nodes)
    owner_sources = Counter(node.owner_source or "none" for node in parsed.nodes)
    unattached = sum(1 for test in parsed.tests if test.attached_node_id is None)
    print(f"Loaded {parsed.project_name} (dbt {parsed.dbt_version}, load id {load_id})")
    print(f"  nodes    {len(parsed.nodes):>6}  {dict(node_types)}")
    print(f"  edges    {len(parsed.edges):>6}")
    print(f"  columns  {len(parsed.columns):>6}")
    print(f"  tests    {len(parsed.tests):>6}  ({unattached} not attached to a node)")
    print(f"  owners   {dict(owner_sources)}")


if __name__ == "__main__":
    main()
