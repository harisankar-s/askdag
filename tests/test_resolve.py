import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.lineage.resolve import (
    Ambiguous,
    NotFound,
    Resolved,
    node_display_names,
    resolve_node,
)

RAW_ORDERS = make_node("orders", "source", source_name="raw")
ORDERS = make_node("orders")
STG_ORDERS = make_node("stg_orders")
FCT_ORDERS = make_node("fct_orders")


@pytest.fixture(autouse=True)
def graph(load_graph) -> None:
    load_graph([RAW_ORDERS, ORDERS, STG_ORDERS, FCT_ORDERS], [])


def resolve(conn, query: str):
    return resolve_node(conn, query, project=TEST_PROJECT)


def test_unique_id_resolves(conn) -> None:
    result = resolve(conn, FCT_ORDERS.unique_id)

    assert isinstance(result, Resolved)
    assert result.node.unique_id == FCT_ORDERS.unique_id


def test_name_match_ignores_case_and_surrounding_space(conn) -> None:
    result = resolve(conn, "  FCT_Orders ")

    assert isinstance(result, Resolved)
    assert result.node.unique_id == FCT_ORDERS.unique_id


def test_source_resolves_by_source_name_and_table(conn) -> None:
    result = resolve(conn, "raw.orders")

    assert isinstance(result, Resolved)
    assert result.node.unique_id == RAW_ORDERS.unique_id


def test_shared_name_is_ambiguous_not_guessed(conn) -> None:
    result = resolve(conn, "orders")

    assert isinstance(result, Ambiguous)
    assert {c.unique_id for c in result.candidates} == {ORDERS.unique_id, RAW_ORDERS.unique_id}


def test_typo_returns_suggestions_but_does_not_resolve(conn) -> None:
    result = resolve(conn, "fct_ordrs")

    assert isinstance(result, NotFound)
    assert result.suggestions[0].unique_id == FCT_ORDERS.unique_id
    assert all(0 < c.score < 1 for c in result.suggestions)


def test_nothing_similar_returns_no_suggestions(conn) -> None:
    assert resolve(conn, "zzzz") == NotFound("zzzz", ())
    assert resolve(conn, "   ") == NotFound("   ", ())


def test_project_filter_excludes_other_projects(conn) -> None:
    assert resolve_node(conn, "fct_orders", project="some_other_project") == NotFound(
        "fct_orders", ()
    )


def test_display_names_cover_every_node_with_sources_qualified(conn) -> None:
    names = node_display_names(conn, project=TEST_PROJECT)

    assert names == {"raw.orders", "orders", "stg_orders", "fct_orders"}
