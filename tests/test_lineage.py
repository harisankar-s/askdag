import json
from pathlib import Path

import pytest
from helpers import make_node

from asklineage.ingest.load import load_manifest
from asklineage.ingest.manifest import parse_manifest
from asklineage.lineage.graph import Direction, lineage

SRC = make_node("events", "source", source_name="raw")
SEED = make_node("country_codes", "seed")
STG = make_node("stg_events")
INT_A = make_node("int_a")
INT_B = make_node("int_b")
FCT = make_node("fct_d", owner="finance-team")
MART = make_node("mart_e")
REPORT = make_node("report_f")

# raw.events -> stg_events -> int_a -> fct_d -> mart_e -> report_f
#                          -> int_b ->
# country_codes ------------> int_b
# stg_events --------------------------------------------> report_f (shortcut)
NODES = [SRC, SEED, STG, INT_A, INT_B, FCT, MART, REPORT]
EDGES = [
    (SRC.unique_id, STG.unique_id),
    (STG.unique_id, INT_A.unique_id),
    (STG.unique_id, INT_B.unique_id),
    (SEED.unique_id, INT_B.unique_id),
    (INT_A.unique_id, FCT.unique_id),
    (INT_B.unique_id, FCT.unique_id),
    (FCT.unique_id, MART.unique_id),
    (MART.unique_id, REPORT.unique_id),
    (STG.unique_id, REPORT.unique_id),
]

MINI_MANIFEST = Path(__file__).parent / "fixtures" / "mini_manifest.json"


@pytest.fixture
def graph(load_graph) -> None:
    load_graph(NODES, EDGES)


def depths(result) -> dict[str, int]:
    return {node.name: node.depth for node in result.nodes}


@pytest.mark.usefixtures("graph")
def test_downstream_reports_each_node_once_at_its_shortest_depth(conn) -> None:
    result = lineage(conn, STG.unique_id, Direction.DOWNSTREAM)

    assert [node.name for node in result.nodes] == [
        "int_a",
        "int_b",
        "report_f",
        "fct_d",
        "mart_e",
    ]
    assert depths(result) == {"int_a": 1, "int_b": 1, "report_f": 1, "fct_d": 2, "mart_e": 3}
    assert result.truncated is False


@pytest.mark.usefixtures("graph")
def test_upstream_reaches_sources_and_seeds(conn) -> None:
    result = lineage(conn, FCT.unique_id, Direction.UPSTREAM)

    assert depths(result) == {
        "int_a": 1,
        "int_b": 1,
        "country_codes": 2,
        "stg_events": 2,
        "events": 3,
    }
    source = next(node for node in result.nodes if node.resource_type == "source")
    assert source.display_name == "raw.events"


@pytest.mark.usefixtures("graph")
def test_depth_limit_cuts_the_walk_and_flags_truncation(conn) -> None:
    result = lineage(conn, STG.unique_id, Direction.DOWNSTREAM, max_depth=2)

    assert depths(result) == {"int_a": 1, "int_b": 1, "report_f": 1, "fct_d": 2}
    assert result.truncated is True


@pytest.mark.usefixtures("graph")
def test_not_truncated_when_the_frontier_only_points_at_seen_nodes(conn) -> None:
    # mart_e sits at the cap, but its only child report_f was already reached at depth 1.
    result = lineage(conn, STG.unique_id, Direction.DOWNSTREAM, max_depth=3)

    assert "mart_e" in depths(result)
    assert result.truncated is False


@pytest.mark.usefixtures("graph")
def test_edges_are_those_between_returned_nodes_and_keep_their_direction(conn) -> None:
    result = lineage(conn, MART.unique_id, Direction.UPSTREAM, max_depth=1)

    assert depths(result) == {"fct_d": 1}
    assert result.edges == ((FCT.unique_id, MART.unique_id),)


@pytest.mark.usefixtures("graph")
def test_nodes_carry_owner_and_owner_source(conn) -> None:
    result = lineage(conn, INT_A.unique_id, Direction.DOWNSTREAM, max_depth=1)

    (fct,) = result.nodes
    assert (fct.owner, fct.owner_source) == ("finance-team", "owners_file")


@pytest.mark.usefixtures("graph")
def test_leaf_has_no_downstream(conn) -> None:
    result = lineage(conn, REPORT.unique_id, Direction.DOWNSTREAM)

    assert result.nodes == ()
    assert result.edges == ()
    assert result.truncated is False


@pytest.mark.usefixtures("graph")
def test_a_cycle_terminates_and_never_lists_the_root(conn) -> None:
    conn.execute(
        "INSERT INTO lineage_edges (parent_id, child_id) VALUES (%s, %s)",
        (REPORT.unique_id, STG.unique_id),
    )

    result = lineage(conn, STG.unique_id, Direction.DOWNSTREAM, max_depth=25)

    assert set(depths(result)) == {"int_a", "int_b", "report_f", "fct_d", "mart_e"}


@pytest.mark.usefixtures("graph")
def test_unknown_root_and_bad_depth_are_errors(conn) -> None:
    with pytest.raises(LookupError):
        lineage(conn, "model.lineage_test.nope", Direction.DOWNSTREAM)
    with pytest.raises(ValueError):
        lineage(conn, STG.unique_id, Direction.DOWNSTREAM, max_depth=0)
    with pytest.raises(ValueError):
        lineage(conn, STG.unique_id, Direction.DOWNSTREAM, max_depth=26)


def test_parsed_and_loaded_manifest_gives_the_expected_lineage(conn) -> None:
    load_manifest(conn, parse_manifest(json.loads(MINI_MANIFEST.read_text())))

    result = lineage(conn, "source.shop.raw.orders", Direction.DOWNSTREAM)

    assert depths(result) == {"stg_orders": 1, "dim_customers": 2, "fct_orders": 2}
