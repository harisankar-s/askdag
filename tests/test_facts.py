import pytest

from asklineage.answer.facts import OWNER_SOURCE_MEANING, OwnerGroup, build_facts
from asklineage.lineage.graph import Direction, Lineage, LineageNode
from asklineage.lineage.resolve import Candidate

ROOT = Candidate("model.p.stg_orders", "stg_orders", "model", None, 1.0)


def node(
    name: str,
    depth: int,
    owner: str | None = None,
    resource_type: str = "model",
    source_name: str | None = None,
) -> LineageNode:
    return LineageNode(
        unique_id=f"{resource_type}.p.{name}",
        name=name,
        source_name=source_name,
        resource_type=resource_type,
        depth=depth,
        owner=owner,
        owner_source="owners_file" if owner else None,
    )


def result(*nodes: LineageNode, truncated: bool = False) -> Lineage:
    return Lineage(ROOT.unique_id, Direction.DOWNSTREAM, 3, nodes, (), truncated)


NODES = (
    node("fct_orders", 1, owner="finance"),
    node("int_items", 1, owner="finance"),
    node("mart_revenue", 2, owner="analytics"),
    node("mart_returns", 2),
    node("snap_orders", 3, owner="finance", resource_type="snapshot"),
)


def test_counts_cover_every_node():
    facts = build_facts(result(*NODES), ROOT)

    assert facts.total == 5
    assert facts.by_depth == {1: 2, 2: 2, 3: 1}
    assert facts.by_type == {"model": 4, "snapshot": 1}


def test_owners_are_grouped_largest_first_with_their_source():
    facts = build_facts(result(*NODES), ROOT)

    meaning = OWNER_SOURCE_MEANING["owners_file"]
    assert facts.owners == (
        OwnerGroup("finance", "owners_file", meaning, 3),
        OwnerGroup("analytics", "owners_file", meaning, 1),
    )
    assert facts.unowned == 1


def test_root_and_direction_are_described():
    facts = build_facts(result(*NODES, truncated=True), ROOT)

    assert (facts.root, facts.root_type, facts.direction) == ("stg_orders", "model", "downstream")
    assert "read from the root" in facts.direction_meaning
    assert facts.truncated is True
    assert facts.max_depth == 3


def test_empty_result_has_zero_counts():
    facts = build_facts(result(), ROOT)

    assert (facts.total, facts.owners, facts.unowned) == (0, (), 0)


def test_wrong_root_is_rejected():
    other = Candidate("model.p.other", "other", "model", None, 1.0)

    with pytest.raises(ValueError, match="not the root"):
        build_facts(result(*NODES), other)
