"""Command-line entry point. Run `uv run asklineage --help`."""

import argparse
import json
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

from asklineage.config import get_settings
from asklineage.ingest.load import apply_schema, load_manifest
from asklineage.ingest.manifest import UnsupportedManifestError, parse_manifest
from asklineage.ingest.owners import load_owner_rules
from asklineage.lineage.graph import DEFAULT_DEPTH, MAX_DEPTH, Direction, Lineage, lineage
from asklineage.lineage.resolve import Ambiguous, NotFound, Resolved, resolve_node
from asklineage.llm.base import LLMError
from asklineage.llm.factory import get_provider


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asklineage")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create or update the database tables")
    load = commands.add_parser("load-manifest", help="Load a dbt manifest.json")
    load.add_argument("manifest", type=Path)
    load.add_argument("--owners", type=Path, help="YAML file of path-based owner rules")
    show = commands.add_parser("lineage", help="Show what a node depends on or feeds")
    show.add_argument("name", help="model name, source_name.table, or unique_id")
    show.add_argument(
        "--direction", type=Direction, choices=list(Direction), default=Direction.DOWNSTREAM
    )
    show.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help=f"1 to {MAX_DEPTH} hops")
    show.add_argument("--project", help="only match nodes in this dbt project")
    commands.add_parser("llm-check", help="Send one small request to the configured LLM")
    args = parser.parse_args(argv)
    if args.command == "lineage" and not 1 <= args.depth <= MAX_DEPTH:
        parser.error(f"--depth must be between 1 and {MAX_DEPTH}")
    if args.command == "llm-check":
        _check_llm()
        return

    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        apply_schema(conn)
        if args.command == "init-db":
            print("Schema is up to date.")
        elif args.command == "lineage":
            _show_lineage(conn, args)
        else:
            _load_manifest(conn, args, parser)


class _Ping(BaseModel):
    ok: bool
    reply: str


def _check_llm() -> None:
    settings = get_settings()
    provider = get_provider(settings)
    if provider is None:
        print("LLM_PROVIDER=none: summaries come from the template, and no model is called.")
        return
    try:
        completion = provider.complete_json(
            system="You are checking that a JSON API works.",
            prompt='Set ok to true and reply to the word "ready".',
            output=_Ping,
        )
    except LLMError as exc:
        print(f"LLM check failed: {exc}")
        raise SystemExit(1) from exc
    print(
        f"{completion.model} at {settings.llm_base_url}: ok={completion.data.ok}, "
        f"reply={completion.data.reply!r}, {completion.latency_ms} ms, "
        f"tokens in/out {completion.input_tokens}/{completion.output_tokens}"
    )


def _load_manifest(
    conn: psycopg.Connection, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
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


def _show_lineage(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    match resolve_node(conn, args.name, args.project):
        case Resolved(node=node):
            _print_lineage(lineage(conn, node.unique_id, args.direction, args.depth))
        case Ambiguous(candidates=candidates):
            print(f"{args.name!r} matches {len(candidates)} nodes; pass one of these unique_ids:")
            for candidate in candidates:
                print(f"  {candidate.unique_id}")
            raise SystemExit(1)
        case NotFound(suggestions=suggestions):
            print(f"No node named {args.name!r}.")
            if suggestions:
                print("Did you mean:")
                for candidate in suggestions:
                    print(f"  {candidate.score:.2f}  {candidate.unique_id}")
            raise SystemExit(1)


def _print_lineage(result: Lineage) -> None:
    print(
        f"{result.direction} of {result.root_id}, up to {result.max_depth} hops: "
        f"{len(result.nodes)} nodes, {len(result.edges)} edges"
    )
    if result.nodes:
        width = max(len(node.display_name) for node in result.nodes)
        print(f"  {'depth':>5}  {'type':<8}  {'name':<{width}}  owner")
        for node in result.nodes:
            owner = f"{node.owner} ({node.owner_source})" if node.owner else "none recorded"
            name = f"{node.display_name:<{width}}"
            print(f"  {node.depth:>5}  {node.resource_type:<8}  {name}  {owner}")
    if result.truncated:
        print(f"More nodes lie beyond {result.max_depth} hops; raise --depth to see them.")


if __name__ == "__main__":
    main()
