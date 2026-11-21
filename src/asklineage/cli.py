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
from asklineage.api.flow import checkpoint_serde, search_options
from asklineage.api.threads import prune_threads
from asklineage.config import get_settings
from asklineage.embed.base import EmbeddingError, cosine, index_name
from asklineage.embed.chunks import CHUNK_KINDS
from asklineage.embed.factory import get_embedder
from asklineage.embed.store import sync_chunks
from asklineage.eval.conformance import (
    DEPTHS,
    StaleLoadError,
    check_loaded,
    loaded_owners,
    run_conformance,
)
from asklineage.eval.golden import PHRASINGS, GoldenSetError, NameIndex, build_golden, describe
from asklineage.eval.report import render, summarize
from asklineage.eval.run import EvalContext, run_golden, run_info, write_run
from asklineage.eval.search_eval import KIND_SETS, Config, families, run_search_eval
from asklineage.eval.search_eval import render as render_search
from asklineage.eval.search_set import (
    SearchSetError,
    check_against_manifest,
    description_overlap,
    load_search_set,
)
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
from asklineage.search.query import SearchResult, search


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
    commands.add_parser("embed-check", help="Embed two documents and a question, and compare them")
    embed = commands.add_parser("embed", help="Embed new or changed node text for search")
    embed.add_argument("--project", help="only this dbt project (default: every loaded project)")
    find = commands.add_parser("search", help="Find nodes by what a question describes")
    find.add_argument("question")
    find.add_argument("-k", type=int, default=10, help="how many nodes to show")
    find.add_argument(
        "--kinds", nargs="+", choices=CHUNK_KINDS, default=list(CHUNK_KINDS), help="text to match"
    )
    find.add_argument("--keyword", action="store_true", help="keywords only, even with embeddings")
    find.add_argument("--project", help="only match nodes in this dbt project")
    find.add_argument(
        "--no-lineage", action="store_true", help="rank by similarity only, without lifting marts"
    )
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
    run = evals.add_parser("run", help="Ask the golden set and score every layer")
    run.add_argument("manifest", type=Path, help="the manifest that was loaded")
    run.add_argument("--phrasings", choices=PHRASINGS, default="one")
    run.add_argument("--out", type=Path, default=Path("data/eval/runs"), help="runs folder")
    run.add_argument(
        "--source-repo",
        type=Path,
        help="checkout the manifest was built from, to record its commit",
    )
    search_set = evals.add_parser(
        "search-set", help="Check the hand-written search questions against a manifest"
    )
    search_set.add_argument("manifest", type=Path)
    search_set.add_argument(
        "--questions", type=Path, default=Path("config/search_questions.calitp.yml")
    )
    search_eval = evals.add_parser("search", help="Score search on the hand-written questions")
    search_eval.add_argument("manifest", type=Path, help="the manifest that was loaded")
    search_eval.add_argument(
        "--questions", type=Path, default=Path("config/search_questions.calitp.yml")
    )
    search_eval.add_argument("--out", type=Path, default=Path("data/eval/search"))
    args = parser.parse_args(argv)
    if args.command == "lineage" and not 1 <= args.depth <= MAX_DEPTH:
        parser.error(f"--depth must be between 1 and {MAX_DEPTH}")
    checks_depths = args.command == "eval" and args.eval_command == "conformance"
    if checks_depths and not all(1 <= depth <= MAX_DEPTH for depth in args.depths):
        parser.error(f"--depths must be between 1 and {MAX_DEPTH}")
    if args.command == "llm-check":
        _check_llm()
        return
    if args.command == "embed-check":
        _check_embedder()
        return
    if args.command == "embed":
        _embed(args)
        return
    if args.command == "eval" and args.eval_command == "search-set":
        _eval_search_set(args)
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
        elif args.command == "search":
            _print_search(
                search(
                    conn,
                    args.question,
                    get_embedder(get_settings()),
                    project=args.project,
                    kinds=args.kinds,
                    k=args.k,
                    keyword_only=args.keyword,
                    prefer=() if args.no_lineage else get_settings().search_preferred_paths,
                )
            )
        elif args.command == "eval" and args.eval_command == "golden":
            _eval_golden(conn, args)
        elif args.command == "eval" and args.eval_command == "run":
            _eval_run(conn, args)
        elif args.command == "eval" and args.eval_command == "search":
            _eval_search(conn, args)
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


EMBED_CHECK_QUERY = "Which model has the locations of bus stops?"
EMBED_CHECK_DOCUMENTS = (
    "dim_stops: one row per transit stop, with its latitude, longitude and the feed it came from.",
    "fct_fare_payments: one row per fare payment, with the amount and the card type.",
)


def _check_embedder() -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    if embedder is None:
        print("EMBED_PROVIDER=none: search uses keywords only, and no embedding model is called.")
        return
    try:
        documents = embedder.embed(EMBED_CHECK_DOCUMENTS, kind="document")
        query = embedder.embed([EMBED_CHECK_QUERY], kind="query")
    except EmbeddingError as exc:
        print(f"Embedding check failed: {exc}")
        raise SystemExit(1) from exc
    print(
        f"{documents.model} at {settings.embed_base_url}: {embedder.dimensions} dimensions, "
        f"{documents.latency_ms + query.latency_ms} ms for 2 documents and 1 question"
    )
    print(f"  question  {EMBED_CHECK_QUERY!r}")
    for text, vector in zip(EMBED_CHECK_DOCUMENTS, documents.vectors, strict=True):
        name = text.split(":")[0]
        print(f"  {cosine(query.vectors[0], vector):.3f}  {name}")
    print("The stops model should score higher than the fares model.")


def _embed(args: argparse.Namespace) -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    if embedder is None:
        print("EMBED_PROVIDER=none: nothing to embed; search uses keywords only.")
        return
    # Autocommit, so every finished batch is kept if the run stops part way.
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        apply_schema(conn)
        projects = (
            [args.project]
            if args.project
            else [
                row[0] for row in conn.execute("SELECT DISTINCT project_name FROM nodes ORDER BY 1")
            ]
        )
        if not projects:
            print("No nodes loaded; run load-manifest first.")
            raise SystemExit(1)
        for project in projects:

            def progress(done: int, total: int) -> None:
                print(f"  embedded {done}/{total}", end="\r", flush=True)

            try:
                report = sync_chunks(conn, project, embedder, on_batch=progress)
            except EmbeddingError as exc:
                print(f"Embedding failed: {exc}")
                raise SystemExit(1) from exc
            rate = report.embedded / report.seconds if report.seconds else 0
            print(
                f"{project} with {index_name(embedder)}: {sum(report.chunks.values())} chunks "
                f"{dict(report.chunks)}; embedded {report.embedded}, kept {report.kept}, "
                f"deleted {report.deleted} in {report.seconds:.1f} s ({rate:.0f} chunks/s)"
            )


def _print_search(result: SearchResult) -> None:
    how = f"semantic, {result.model}" if result.mode == "semantic" else "keyword"
    note = f"; {result.note}" if result.note else ""
    print(f"{len(result.hits)} nodes ({how}{note})")
    if not result.hits:
        return
    width = max(len(hit.display_name) for hit in result.hits)
    print(f"  {'score':>6}  {'match':<7}  {'name':<{width}}  owner")
    for hit in result.hits:
        owner = f"{hit.owner} ({hit.owner_source})" if hit.owner else "none recorded"
        via = f"  via {hit.via}" if hit.via else ""
        print(f"  {hit.score:6.3f}  {hit.kind:<7}  {hit.display_name:<{width}}  {owner}{via}")


def _eval_search(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    settings = get_settings()
    embedder = get_embedder(settings)
    graph = truth_graph(manifest)
    try:
        check_loaded(conn, graph)
        search_set = load_search_set(args.questions)
        expected = check_against_manifest(search_set, manifest)
    except (StaleLoadError, SearchSetError) as exc:
        print(exc)
        raise SystemExit(1) from exc
    modes = ["semantic", "keyword"] if embedder else ["keyword"]
    configs = [Config(mode, kinds) for mode in modes for kinds in KIND_SETS]
    prefer = tuple(settings.search_preferred_paths)
    if embedder and prefer:
        configs[3:3] = [Config("semantic", kinds, prefer) for kinds in KIND_SETS]
    tag = index_name(embedder) if embedder else "keyword-only"
    header = (
        f"Search eval: set {search_set.sha}, {len(search_set.questions)} questions, "
        f"vectors {tag!r}, query prefix {settings.embed_query_prefix!r}"
        if embedder
        else f"Search eval: set {search_set.sha}, keyword only (EMBED_PROVIDER=none)"
    )
    print(header)
    try:
        results = run_search_eval(
            conn,
            search_set,
            expected,
            embedder,
            configs,
            on_config=lambda config: print(f"  running {config.label}"),
            family=families(graph.parents, expected),
        )
    except (RuntimeError, EmbeddingError) as exc:
        print(f"Search eval failed: {exc}")
        raise SystemExit(1) from exc
    report = render_search(results, header)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out / f"{stamp}-{''.join(c if c.isalnum() else '-' for c in tag)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "search_set": search_set.sha,
        "questions_file": str(args.questions),
        "manifest": str(args.manifest),
        "vectors": tag,
        "query_prefix": settings.embed_query_prefix if embedder else None,
    }
    (run_dir / "results.json").write_text(
        json.dumps({"info": info, "results": results}, indent=2, sort_keys=True)
    )
    (run_dir / "report.txt").write_text(report)
    print(report)
    print(f"Wrote {run_dir}")


def _eval_search_set(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    try:
        search_set = load_search_set(args.questions)
        check_against_manifest(search_set, manifest)
    except SearchSetError as exc:
        print(exc)
        raise SystemExit(1) from exc
    categories = Counter(question.category for question in search_set.questions)
    print(
        f"Search set {search_set.sha} ({args.questions}): {len(search_set.questions)} questions, "
        f"every expected model is in the manifest"
    )
    print(f"  categories  {dict(categories)}")
    print("  overlap = share of the question's words in the expected model's name or description")
    for question in search_set.questions:
        overlap = description_overlap(question, manifest, search_set.project)
        shown = "    -" if overlap is None else f"{overlap:5.2f}"
        print(f"  {shown}  {question.id:<4} {question.category:<12} {question.question}")


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


def _eval_run(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    graph = truth_graph(manifest)
    settings = get_settings()
    provider = get_provider(settings)
    try:
        check_loaded(conn, graph)
        owners = loaded_owners(conn, graph.project_name)
        golden = build_golden(manifest, graph, owners, args.phrasings)
    except (StaleLoadError, GoldenSetError) as exc:
        print(exc)
        raise SystemExit(1) from exc
    info = run_info(
        args.manifest,
        raw,
        golden,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model if provider else None,
        max_named=settings.llm_max_named_nodes,
        source_repo=args.source_repo,
    )
    context = EvalContext(
        conn,
        provider,
        graph,
        NameIndex(manifest, graph),
        owners,
        settings.llm_max_named_nodes,
        search_options(settings),
    )
    total = sum(len(item.turns) for item in golden.items)
    print(
        f"Asking golden set {golden.sha}: {len(golden.items)} items, {total} turns, "
        f"{info.llm_model or 'no LLM'}"
    )
    done = iter(range(1, total + 1))

    def progress(score) -> None:
        print(f"  [{next(done):>3}/{total}] {score.item_id} turn {score.turn}: {score.outcome}")

    scores = run_golden(context, golden, on_turn=progress)
    summary = summarize(scores, llm=provider is not None)
    run_dir = write_run(
        args.out, info, golden, scores, summary, render(info, golden, summary, scores)
    )

    outcomes = Counter(score.outcome for score in scores)
    print(f"Outcomes: {dict(outcomes)}; silent wrong per 100: {summary['silent_wrong_per_100']}")
    print(f"Report: {run_dir / 'report.md'}")
    if summary["bugs"]:
        print(f"Bugs (errors, or tables or facts unlike the manifest): {summary['bugs']}")
        raise SystemExit(1)


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
