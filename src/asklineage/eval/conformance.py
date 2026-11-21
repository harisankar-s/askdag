"""Check every lineage walk the app returns against the ground truth from the manifest.

Exhaustive, not sampled: every node, both directions, several depths. The walk is
deterministic and cheap, so any mismatch is a bug to fix, not a score.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from asklineage.eval.truth import DIRECTIONS, TruthGraph, TruthWalk, walk
from asklineage.lineage.graph import DEFAULT_DEPTH, MAX_DEPTH, Direction, Lineage, lineage

DEPTHS = tuple(sorted({1, 2, 3, DEFAULT_DEPTH, MAX_DEPTH}))
MAX_IDS_SHOWN = 5

LATEST_LOAD = """
    SELECT invocation_id FROM manifest_loads
    WHERE project_name = %s
    ORDER BY id DESC
    LIMIT 1
"""

PROJECT_NODES = "SELECT unique_id FROM nodes WHERE project_name = %s"

PROJECT_OWNERS = "SELECT unique_id, owner FROM nodes WHERE project_name = %s"

PROJECT_EDGES = """
    SELECT e.parent_id, e.child_id
    FROM lineage_edges e
    JOIN nodes n ON n.unique_id = e.child_id
    WHERE n.project_name = %s
"""


class StaleLoadError(RuntimeError):
    pass


def loaded_owners(conn: psycopg.Connection, project: str) -> dict[str, str | None]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(PROJECT_OWNERS, (project,))
        return {row["unique_id"]: row["owner"] for row in cur.fetchall()}


@dataclass(frozen=True)
class Mismatch:
    rule: str
    detail: str
    root: str | None = None  # None for checks on the whole graph
    direction: str | None = None
    depth: int | None = None

    def __str__(self) -> str:
        where = f"{self.direction} of {self.root}, depth {self.depth}: " if self.root else ""
        return f"{where}{self.rule}: {self.detail}"


@dataclass(frozen=True)
class ConformanceReport:
    project: str
    nodes: int
    edges: int
    roots: int
    depths: tuple[int, ...]
    walks: int
    mismatches: tuple[Mismatch, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _ids(ids: Iterable[str]) -> str:
    ordered = sorted(ids)
    shown = ", ".join(ordered[:MAX_IDS_SHOWN])
    more = len(ordered) - MAX_IDS_SHOWN
    return f"{len(ordered)}: {shown}" + (f" (+{more} more)" if more > 0 else "")


def _edge_ids(edges: Iterable[tuple[str, str]]) -> list[str]:
    return [f"{parent} -> {child}" for parent, child in edges]


def check_loaded(conn: psycopg.Connection, graph: TruthGraph) -> None:
    """A database loaded from another manifest would show up as false bugs, so refuse it."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(LATEST_LOAD, (graph.project_name,))
        row = cur.fetchone()
    if row is None:
        raise StaleLoadError(
            f"{graph.project_name} is not loaded; run load-manifest on this manifest first."
        )
    if row["invocation_id"] != graph.invocation_id:
        raise StaleLoadError(
            f"The database holds another load of {graph.project_name} (invocation_id "
            f"{row['invocation_id']}; the manifest has {graph.invocation_id}). "
            "Run load-manifest on this manifest first."
        )


def compare_graph(
    graph: TruthGraph, nodes: set[str], edges: set[tuple[str, str]]
) -> list[Mismatch]:
    checks = (
        ("missing_node", "in the manifest, not in the database", graph.node_ids - nodes),
        ("extra_node", "in the database, not in the manifest", nodes - graph.node_ids),
        ("missing_edge", "in the manifest, not in the database", _edge_ids(graph.edges - edges)),
        ("extra_edge", "in the database, not in the manifest", _edge_ids(edges - graph.edges)),
    )
    return [Mismatch(rule, f"{what} ({_ids(ids)})") for rule, what, ids in checks if ids]


def compare_walk(result: Lineage, truth: TruthWalk) -> list[Mismatch]:
    if result.root_id != truth.root:
        raise ValueError(f"Comparing a walk from {result.root_id!r} with one from {truth.root!r}")
    got = {node.unique_id: node.depth for node in result.nodes}
    expected = truth.depths
    counts = Counter(node.unique_id for node in result.nodes)
    repeated = [uid for uid, count in counts.items() if count > 1]
    wrong_depth = [
        f"{uid} at {got[uid]} hops, expected {expected[uid]}"
        for uid in got.keys() & expected.keys()
        if got[uid] != expected[uid]
    ]
    edges = set(result.edges)
    checks = (
        ("missing_node", "expected, not returned", expected.keys() - got.keys()),
        ("extra_node", "returned, not expected", got.keys() - expected.keys()),
        ("duplicate_node", "returned more than once", repeated),
        ("wrong_depth", "not at the fewest hops", wrong_depth),
        ("missing_edge", "expected, not returned", _edge_ids(truth.edges - edges)),
        ("extra_edge", "returned, not expected", _edge_ids(edges - truth.edges)),
    )
    where = {"root": truth.root, "direction": truth.direction, "depth": truth.max_depth}
    found = [Mismatch(rule, f"{what} ({_ids(ids)})", **where) for rule, what, ids in checks if ids]
    if result.truncated != truth.truncated:
        detail = f"truncated is {result.truncated}, expected {truth.truncated}"
        found.append(Mismatch("truncated", detail, **where))
    return found


def _loaded_graph(conn: psycopg.Connection, project: str) -> tuple[set[str], set[tuple[str, str]]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(PROJECT_NODES, (project,))
        nodes = {row["unique_id"] for row in cur.fetchall()}
        cur.execute(PROJECT_EDGES, (project,))
        edges = {(row["parent_id"], row["child_id"]) for row in cur.fetchall()}
    return nodes, edges


def run_conformance(
    conn: psycopg.Connection, graph: TruthGraph, depths: Sequence[int] = DEPTHS
) -> ConformanceReport:
    check_loaded(conn, graph)
    nodes, edges = _loaded_graph(conn, graph.project_name)
    mismatches = compare_graph(graph, nodes, edges)
    # Walk the roots both sides know; a node on one side only is already reported above.
    roots = sorted(graph.node_ids & nodes)
    for root in roots:
        for direction in DIRECTIONS:
            for depth in depths:
                result = lineage(conn, root, Direction(direction), depth)
                mismatches.extend(compare_walk(result, walk(graph, root, direction, depth)))
    return ConformanceReport(
        project=graph.project_name,
        nodes=len(graph.node_ids),
        edges=len(graph.edges),
        roots=len(roots),
        depths=tuple(depths),
        walks=len(roots) * len(DIRECTIONS) * len(depths),
        mismatches=tuple(mismatches),
    )
