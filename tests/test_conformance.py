from dataclasses import replace

import psycopg
import pytest
from synthetic_manifest import build_manifest, chain_id

from asklineage.eval.conformance import (
    DEPTHS,
    StaleLoadError,
    compare_walk,
    run_conformance,
)
from asklineage.eval.truth import TruthGraph, TruthWalk, truth_graph
from asklineage.ingest.load import load_manifest
from asklineage.ingest.manifest import parse_manifest
from asklineage.lineage.graph import MAX_DEPTH, Direction, Lineage, LineageNode

ROOT, A, B, C = (f"model.p.{name}" for name in ("stg", "a", "b", "c"))
TRUTH = TruthWalk(
    root=ROOT,
    direction="downstream",
    max_depth=2,
    depths={A: 1, B: 2},
    edges=frozenset({(ROOT, A), (A, B)}),
    truncated=True,
)


def result_of(
    nodes: list[tuple[str, int]], edges: list[tuple[str, str]], truncated: bool
) -> Lineage:
    return Lineage(
        root_id=ROOT,
        direction=Direction.DOWNSTREAM,
        max_depth=2,
        nodes=tuple(
            LineageNode(uid, uid.rsplit(".", 1)[-1], None, "model", depth, None, None)
            for uid, depth in nodes
        ),
        edges=tuple(edges),
        truncated=truncated,
    )


def test_a_matching_walk_has_no_mismatches() -> None:
    result = result_of([(A, 1), (B, 2)], [(ROOT, A), (A, B)], truncated=True)
    assert compare_walk(result, TRUTH) == []


def test_each_difference_is_reported_under_its_own_rule() -> None:
    result = result_of([(A, 2), (C, 1)], [(ROOT, A), (ROOT, C)], truncated=False)
    found = compare_walk(result, TRUTH)
    assert {m.rule for m in found} == {
        "missing_node",
        "extra_node",
        "wrong_depth",
        "missing_edge",
        "extra_edge",
        "truncated",
    }
    assert all(m.root == ROOT and m.depth == 2 for m in found)


def test_a_node_returned_twice_is_reported() -> None:
    result = result_of([(A, 1), (B, 2), (B, 2)], [(ROOT, A), (A, B)], truncated=True)
    assert [m.rule for m in compare_walk(result, TRUTH)] == ["duplicate_node"]


@pytest.fixture
def synthetic(conn: psycopg.Connection) -> TruthGraph:
    manifest = build_manifest()
    load_manifest(conn, parse_manifest(manifest))
    return truth_graph(manifest)


def test_every_walk_matches_the_manifest(conn: psycopg.Connection, synthetic: TruthGraph) -> None:
    report = run_conformance(conn, synthetic)
    assert report.passed, "\n".join(str(m) for m in report.mismatches[:20])
    assert report.walks == len(synthetic.node_ids) * 2 * len(DEPTHS)


@pytest.mark.usefixtures("synthetic")
def test_a_wrong_edge_shows_up_in_the_graph_and_the_walks(conn: psycopg.Connection) -> None:
    # The database keeps chain_10 -> chain_11; the truth loses it, so the chain stops short.
    manifest = build_manifest()
    manifest["nodes"][chain_id(11)]["depends_on"]["nodes"].remove(chain_id(10))
    report = run_conformance(conn, truth_graph(manifest), depths=(MAX_DEPTH,))
    rules = {m.rule for m in report.mismatches}
    assert {"extra_edge", "extra_node", "truncated"} <= rules
    assert any(m.root is None and m.rule == "extra_edge" for m in report.mismatches)


def test_refuses_a_database_loaded_from_another_manifest(
    conn: psycopg.Connection, synthetic: TruthGraph
) -> None:
    with pytest.raises(StaleLoadError, match="another load"):
        run_conformance(conn, replace(synthetic, invocation_id="another-run"))


def test_refuses_a_project_that_is_not_loaded(conn: psycopg.Connection) -> None:
    manifest = build_manifest()
    manifest["metadata"]["project_name"] = "never_loaded"
    with pytest.raises(StaleLoadError, match="not loaded"):
        run_conformance(conn, truth_graph(manifest))
