"""Command-line entry point. Run `uv run asklineage --help`."""

import argparse
import json
import textwrap
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from pydantic import BaseModel

from asklineage.answer.claims import Highlights
from asklineage.answer.generate import Attempt
from asklineage.answer.lineage_answer import HighlightStatus, LineageAnswer, answer_lineage
from asklineage.api.flow import checkpoint_serde
from asklineage.api.threads import prune_threads
from asklineage.config import get_settings
from asklineage.eval.conformance import (
    DEPTHS,
    StaleLoadError,
    check_loaded,
    loaded_owners,
    run_conformance,
)
from asklineage.eval.golden import PHRASINGS, GoldenSetError, build_golden, describe
from asklineage.eval.truth import truth_graph
from asklineage.ingest.load import apply_schema, load_manifest
from asklineage.ingest.manifest import UnsupportedManifestError, parse_manifest
from asklineage.ingest.owners import load_owner_rules
from asklineage.lineage.graph import DEFAULT_DEPTH, MAX_DEPTH, Direction, Lineage, lineage
from asklineage.lineage.resolve import (
    Ambiguous,
    Candidate,
    NotFound,
    Resolved,
    node_display_names,
    resolve_node,
)
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
    show.add_argument(
        "--summarize", action="store_true", help="add a summary, checked against the result"
    )
    show.add_argument(
        "-v", "--verbose", action="store_true", help="with --summarize, show each LLM attempt"
    )
    commands.add_parser("llm-check", help="Send one small request to the configured LLM")
    prune = commands.add_parser("prune-threads", help="Delete /api/ask threads unused for a while")
    prune.add_argument("--older-than-hours", type=float, default=24.0)
    evaluate = commands.add_parser("eval", help="Measure asklineage against a manifest")
    evals = evaluate.add_subparsers(dest="eval_command", required=True)
    conformance = evals.add_parser(
        "conformance", help="Check every lineage walk against a BFS over manifest.json"
    )
    conformance.add_argument("manifest", type=Path, help="the manifest that was loaded")
    conformance.add_argument(
        "--depths", type=int, nargs="+", default=list(DEPTHS), help="depths to walk at"
    )
    conformance.add_argument("--show", type=int, default=20, help="mismatches to print")
    golden = evals.add_parser("golden", help="Show the golden question set built from a manifest")
    golden.add_argument("manifest", type=Path, help="the manifest that was loaded")
    golden.add_argument("--phrasings", choices=PHRASINGS, default="one")
    golden.add_argument("--json", type=Path, dest="json_path", help="also write the set here")
    args = parser.parse_args(argv)
    if args.command == "lineage" and not 1 <= args.depth <= MAX_DEPTH:
        parser.error(f"--depth must be between 1 and {MAX_DEPTH}")
    checks_depths = args.command == "eval" and args.eval_command == "conformance"
    if checks_depths and not all(1 <= depth <= MAX_DEPTH for depth in args.depths):
        parser.error(f"--depths must be between 1 and {MAX_DEPTH}")
    if args.command == "llm-check":
        _check_llm()
        return

    url = get_settings().database_url
    with (
        psycopg.connect(url, row_factory=dict_row) as conn,
        # LangGraph's saver needs autocommit (its migrations create indexes concurrently).
        psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row) as raw,
    ):
        apply_schema(conn)
        saver = PostgresSaver(raw, serde=checkpoint_serde())
        if args.command == "init-db":
            saver.setup()
            print("Schema is up to date, including LangGraph's checkpoint tables.")
        elif args.command == "prune-threads":
            pruned = prune_threads(conn, saver, timedelta(hours=args.older_than_hours))
            print(f"Pruned {pruned} thread(s) unused for {args.older_than_hours:g} hours or more.")
        elif args.command == "lineage":
            _show_lineage(conn, args)
        elif args.command == "eval" and args.eval_command == "golden":
            _eval_golden(conn, args)
        elif args.command == "eval":
            _eval_conformance(conn, args)
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


def _eval_golden(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    graph = truth_graph(manifest)
    try:
        check_loaded(conn, graph)
        owners = loaded_owners(conn, graph.project_name)
        golden = build_golden(manifest, graph, owners, args.phrasings)
    except (StaleLoadError, GoldenSetError) as exc:
        print(exc)
        raise SystemExit(1) from exc
    print(describe(golden))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(golden.to_dict(), indent=2, sort_keys=True))
        print(f"Wrote {args.json_path}")


def _eval_conformance(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    graph = truth_graph(json.loads(args.manifest.read_text()))
    started = time.perf_counter()
    try:
        report = run_conformance(conn, graph, sorted(set(args.depths)))
    except StaleLoadError as exc:
        print(exc)
        raise SystemExit(1) from exc
    seconds = time.perf_counter() - started

    depths = ", ".join(str(depth) for depth in report.depths)
    print(f"Conformance of {report.project} against {args.manifest}")
    print(f"  manifest    {report.nodes:>6} nodes, {report.edges} edges")
    print(
        f"  walks       {report.walks:>6}  ({report.roots} roots x 2 directions x depths "
        f"{depths}) in {seconds:.1f} s"
    )
    if report.passed:
        print("  mismatches       0  every walk matches the manifest")
        return
    rules = Counter(mismatch.rule for mismatch in report.mismatches)
    print(f"  mismatches  {len(report.mismatches):>6}  {dict(rules)}")
    for mismatch in report.mismatches[: args.show]:
        print(f"    - {mismatch}")
    hidden = len(report.mismatches) - args.show
    if hidden > 0:
        print(f"    ... {hidden} more; raise --show to see them")
    raise SystemExit(1)


def _show_lineage(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    match resolve_node(conn, args.name, args.project):
        case Resolved(node=node):
            result = lineage(conn, node.unique_id, args.direction, args.depth)
            if args.summarize:
                print(_summarize(conn, result, node, args))
            _print_lineage(result)
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


def _summarize(
    conn: psycopg.Connection, result: Lineage, root: Candidate, args: argparse.Namespace
) -> str:
    settings = get_settings()
    answer = answer_lineage(
        result,
        root,
        get_provider(settings),
        max_named=settings.llm_max_named_nodes,
        known_names=node_display_names(conn, args.project),
    )
    return format_summary(answer, verbose=args.verbose)


def _wrap(text: str, first: str, rest: str) -> str:
    return textwrap.fill(
        text, width=100, initial_indent=first, subsequent_indent=rest, break_on_hyphens=False
    )


def format_summary(answer: LineageAnswer, verbose: bool = False) -> str:
    attempts = answer.attempts
    latency = sum(a.latency_ms or 0 for a in attempts)
    model = next((a.model for a in attempts if a.model), None)
    lines = ["Summary", _wrap(answer.summary, "  ", "  ")]
    match answer.status:
        case HighlightStatus.PASSED:
            lines.append(f"Highlights (grounded: {model}, {latency} ms)")
        case HighlightStatus.PASSED_AFTER_RETRY:
            lines.append(
                f"Highlights (grounded on retry: {model}, {len(attempts)} attempts, {latency} ms)"
            )
        case HighlightStatus.REJECTED if attempts and attempts[-1].error:
            lines.append("Highlights: none (the LLM could not be reached)")
        case HighlightStatus.REJECTED:
            hint = "" if verbose else "; run with --verbose to see why"
            lines.append(f"Highlights: none (rejected by the check {len(attempts)} times{hint})")
        case HighlightStatus.OFF:
            lines.append("Highlights: off (no LLM configured)")
        case HighlightStatus.SKIPPED:
            pass
    lines += [_wrap(text, "  - ", "    ") for text in answer.highlight_texts]
    if verbose:
        lines += _describe_attempts(attempts)
    return "\n".join(lines) + "\n"


def _describe_attempts(attempts: tuple[Attempt, ...]) -> list[str]:
    lines = []
    for number, attempt in enumerate(attempts, start=1):
        if attempt.error:
            lines.append(f"  attempt {number}: error: {attempt.error}")
            continue
        count = len(attempt.violations)
        outcome = "passed" if attempt.passed else f"{count} problem{'' if count == 1 else 's'}"
        if attempt.model is None:
            details = "reply was not valid JSON for the schema"
        else:
            details = f"{attempt.model}, {attempt.latency_ms} ms"
            if attempt.input_tokens is not None:
                details += f", {attempt.input_tokens} in / {attempt.output_tokens} out tokens"
        lines.append(f"  attempt {number}: {outcome} ({details})")
        for violation in attempt.violations:
            lines.append(f"    - {violation}")
            if violation.claim and isinstance(attempt.output, Highlights):
                said = attempt.output.claims[violation.claim - 1].text
                lines.append(f"      said: {said!r}")
    return lines


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
