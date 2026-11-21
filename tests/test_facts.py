import json

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


def test_counts_cover_every_node_even_when_names_are_capped():
    facts = build_facts(result(*NODES), ROOT, max_named=2)

    assert facts.total == 5
    assert facts.by_depth == {1: 2, 2: 2, 3: 1}
    assert facts.by_type == {"model": 4, "snapshot": 1}
    assert [n.name for n in facts.named] == ["fct_orders", "mart_revenue"]
    assert facts.unnamed == 3


def test_names_rotate_across_owners_so_every_team_is_named():
    facts = build_facts(result(*NODES), ROOT, max_named=4)

    assert [(n.name, n.owner) for n in facts.named] == [
        ("fct_orders", "finance"),
        ("mart_revenue", "analytics"),
        ("mart_returns", None),
        ("int_items", "finance"),
    ]


def test_a_larger_cap_names_every_node():
    facts = build_facts(result(*NODES), ROOT, max_named=50)

    assert len(facts.named) == 5
    assert facts.unnamed == 0


def test_owners_are_grouped_largest_first_with_their_source():
    facts = build_facts(result(*NODES), ROOT, max_named=20)

    meaning = OWNER_SOURCE_MEANING["owners_file"]
    assert facts.owners == (
        OwnerGroup("finance", "owners_file", meaning, 3),
        OwnerGroup("analytics", "owners_file", meaning, 1),
    )
    assert facts.unowned == 1


def test_root_and_direction_are_described():
    facts = build_facts(result(*NODES, truncated=True), ROOT, max_named=20)

    assert (facts.root, facts.root_type, facts.direction) == ("stg_orders", "model", "downstream")
    assert "read from the root" in facts.direction_meaning
    assert facts.truncated is True
    assert facts.max_depth == 3


def test_sources_use_their_display_name():
    source = node("events", 1, resource_type="source", source_name="raw")

    facts = build_facts(result(source), ROOT, max_named=20)

    assert facts.named[0].name == "raw.events"


def test_empty_result_has_zero_counts():
    facts = build_facts(result(), ROOT, max_named=20)

    assert (facts.total, facts.named, facts.unnamed, facts.owners, facts.unowned) == (
        0,
        (),
        0,
        (),
        0,
    )


def test_prompt_json_has_names_and_owners_but_no_counts():
    data = json.loads(build_facts(result(*NODES), ROOT, max_named=20).prompt_json())

    assert set(data) == {"root", "root_type", "direction", "direction_meaning", "named"}
    assert data["named"][0] == {
        "name": "fct_orders",
        "resource_type": "model",
        "depth": 1,
        "owner": "finance",
    }


def test_wrong_root_is_rejected():
    other = Candidate("model.p.other", "other", "model", None, 1.0)

    with pytest.raises(ValueError, match="not the root"):
        build_facts(result(*NODES), other, max_named=20)


def test_cap_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        build_facts(result(*NODES), ROOT, max_named=0)
