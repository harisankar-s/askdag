import ast
import json
import sys
from pathlib import Path

import pytest
from synthetic_manifest import CHAIN_LENGTH, build_manifest, chain_id

from asklineage.eval import truth
from asklineage.eval.truth import truth_graph, walk
from asklineage.ingest.manifest import parse_manifest
from asklineage.lineage.graph import MAX_DEPTH

MINI_MANIFEST = Path(__file__).parent / "fixtures" / "mini_manifest.json"

SRC = "source.p.raw.events"
STG, A, B, C, D, E = (f"model.p.{name}" for name in ("stg", "a", "b", "c", "d", "e"))
LOOP = "model.p.loop"
TEST = "test.p.not_null_d_id"


# raw.events -> stg -> a -> c -> d -> e
#                   -> b ------> d
#                   -> loop (also lists itself as a parent)
# test.p.not_null_d_id reads d
def manifest_of(parents: dict[str, list[str]], sources: tuple[str, ...] = (SRC,)) -> dict:
    nodes = {
        unique_id: {
            "unique_id": unique_id,
            "resource_type": unique_id.split(".")[0],
            "depends_on": {"nodes": ids},
        }
        for unique_id, ids in parents.items()
    }
    return {
        "metadata": {"project_name": "p", "invocation_id": "i"},
        "nodes": nodes,
        "sources": {source: {"unique_id": source, "resource_type": "source"} for source in sources},
    }


GRAPH = truth_graph(
    manifest_of(
        {
            STG: [SRC],
            A: [STG],
            B: [STG],
            C: [A],
            D: [C, B],
            E: [D],
            LOOP: [LOOP, STG],
            TEST: [D],
        }
    )
)


def test_keeps_models_seeds_snapshots_and_sources_but_not_tests() -> None:
    assert GRAPH.node_ids == {SRC, STG, A, B, C, D, E, LOOP}


def test_drops_self_references_and_edges_to_tests() -> None:
    assert (LOOP, LOOP) not in GRAPH.edges
    assert (STG, LOOP) in GRAPH.edges
    assert all(TEST not in edge for edge in GRAPH.edges)
    assert len(GRAPH.edges) == 8


def test_each_node_is_at_its_fewest_hops() -> None:
    # d is 3 hops away through a and c, but 2 through b.
    assert walk(GRAPH, STG, "downstream", 5).depths == {A: 1, B: 1, LOOP: 1, C: 2, D: 2, E: 3}


def test_upstream_follows_parents() -> None:
    assert walk(GRAPH, D, "upstream", 5).depths == {C: 1, B: 1, A: 2, STG: 2, SRC: 3}


@pytest.mark.parametrize(
    ("max_depth", "truncated"),
    [(1, True), (2, True), (3, False), (4, False)],
)
def test_truncated_only_when_a_node_at_the_cap_has_an_unreached_neighbour(
    max_depth: int, truncated: bool
) -> None:
    assert walk(GRAPH, STG, "downstream", max_depth).truncated is truncated


def test_edges_are_every_edge_between_reached_nodes() -> None:
    # a -> c is kept although d was reached through b; d -> e is past the cap.
    assert walk(GRAPH, STG, "downstream", 2).edges == {
        (STG, A),
        (STG, B),
        (STG, LOOP),
        (A, C),
        (B, D),
        (C, D),
    }


def test_unknown_root_raises() -> None:
    with pytest.raises(LookupError):
        walk(GRAPH, "model.p.nope", "downstream", 1)


def test_truth_imports_only_the_standard_library() -> None:
    nodes = list(ast.walk(ast.parse(Path(truth.__file__).read_text())))
    imported = {alias.name for n in nodes if isinstance(n, ast.Import) for alias in n.names}
    imported |= {n.module for n in nodes if isinstance(n, ast.ImportFrom)}
    assert all(name.split(".")[0] in sys.stdlib_module_names for name in imported)


@pytest.mark.parametrize(
    "manifest",
    [json.loads(MINI_MANIFEST.read_text()), build_manifest()],
    ids=["mini", "synthetic"],
)
def test_depends_on_gives_the_same_graph_as_the_parser(manifest: dict) -> None:
    parsed = parse_manifest(manifest)
    graph = truth_graph(manifest)
    assert graph.node_ids == {node.unique_id for node in parsed.nodes}
    assert graph.edges == set(parsed.edges)


def test_synthetic_manifest_has_the_hard_cases() -> None:
    graph = truth_graph(build_manifest())
    chain = walk(graph, chain_id(0), "downstream", MAX_DEPTH)
    assert chain.truncated
    assert chain.depths[chain_id(CHAIN_LENGTH - 1)] == 2
    assert len(graph.children["model.synthetic.int_hub"]) > 10
    assert "snapshot.synthetic.snap_accounts" in graph.node_ids
    assert not any(node.startswith("test.") for node in graph.node_ids)


def test_synthetic_manifest_is_the_same_every_time() -> None:
    assert build_manifest() == build_manifest()
