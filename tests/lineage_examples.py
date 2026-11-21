"""Lineage results whose facts (names capped at 3) are exactly the prompt's worked examples."""

from asklineage.lineage.graph import Direction, Lineage, LineageNode
from asklineage.lineage.resolve import Candidate

EXAMPLE_ROOT = Candidate("model.p.stg_orders", "stg_orders", "model", None, 1.0)
EXAMPLE_MAX_NAMED = 3


def _node(name: str, depth: int, owner: str | None = "finance") -> LineageNode:
    return LineageNode(
        unique_id=f"model.p.{name}",
        name=name,
        source_name=None,
        resource_type="model",
        depth=depth,
        owner=owner,
        owner_source="owners_file" if owner else None,
    )


def _id(name: str) -> str:
    return f"model.p.{name}"


# stg_orders -> fct_orders      -> mart_revenue
#            -> int_order_items -> mart_unit_costs
#                               -> mart_weekly (no owner)
EXAMPLE_LINEAGE = Lineage(
    root_id=EXAMPLE_ROOT.unique_id,
    direction=Direction.DOWNSTREAM,
    max_depth=2,
    nodes=(
        _node("fct_orders", 1),
        _node("int_order_items", 1),
        _node("mart_revenue", 2),
        _node("mart_unit_costs", 2),
        _node("mart_weekly", 2, owner=None),
    ),
    edges=(
        (_id("fct_orders"), _id("mart_revenue")),
        (_id("int_order_items"), _id("mart_unit_costs")),
        (_id("int_order_items"), _id("mart_weekly")),
        (_id("stg_orders"), _id("fct_orders")),
        (_id("stg_orders"), _id("int_order_items")),
    ),
    truncated=True,
)


UPSTREAM_EXAMPLE_ROOT = Candidate("model.p.fct_orders", "fct_orders", "model", None, 1.0)


def _source(table: str, owner: str | None = "finance") -> LineageNode:
    return LineageNode(
        unique_id=f"source.p.raw.{table}",
        name=table,
        source_name="raw",
        resource_type="source",
        depth=2,
        owner=owner,
        owner_source="owners_file" if owner else None,
    )


# raw.orders   -> stg_orders   -> fct_orders
# raw.payments -> stg_payments -> (no owner on raw.payments)
UPSTREAM_EXAMPLE_LINEAGE = Lineage(
    root_id=UPSTREAM_EXAMPLE_ROOT.unique_id,
    direction=Direction.UPSTREAM,
    max_depth=2,
    nodes=(
        _node("stg_orders", 1),
        _node("stg_payments", 1),
        _source("orders"),
        _source("payments", owner=None),
    ),
    edges=(
        ("source.p.raw.orders", _id("stg_orders")),
        ("source.p.raw.payments", _id("stg_payments")),
        (_id("stg_orders"), _id("fct_orders")),
        (_id("stg_payments"), _id("fct_orders")),
    ),
    truncated=False,
)
