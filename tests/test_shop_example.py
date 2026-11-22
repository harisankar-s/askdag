"""The committed quickstart manifests still parse and diff the way examples/shop/README.md says."""

import json
from collections import Counter
from pathlib import Path

from asklineage.diff.compare import diff_manifests
from asklineage.ingest.manifest import parse_manifest

SHOP = Path(__file__).parent.parent / "examples" / "shop" / "manifests"


def parse(name: str):
    return parse_manifest(json.loads((SHOP / name).read_text()))


def test_owners_come_from_dbt_with_some_left_unowned() -> None:
    parsed = parse("manifest.json")
    owners = {node.name: (node.owner, node.owner_source) for node in parsed.nodes}

    assert Counter(node.resource_type for node in parsed.nodes) == {
        "model": 9,
        "source": 3,
        "seed": 1,
    }
    assert owners["stg_app__orders"] == ("data-platform", "dbt_meta")
    assert owners["fct_orders"] == ("finance", "dbt_meta")
    assert owners["dim_customers"] == ("Growth analytics", "dbt_group")
    assert owners["int_order_payments"] == (None, None)
    assert owners["orders"] == (None, None)


def test_v1_to_current_is_one_data_change_one_docs_change_one_new_model() -> None:
    diff = diff_manifests(parse("manifest.v1.json"), parse("manifest.json"))

    assert [ref.name for ref in diff.added] == ["mart_orders_by_region"]
    assert diff.removed == ()
    assert {(change.node.name, change.affects_data) for change in diff.changed} == {
        ("fct_orders", True),
        ("fct_refunds", False),
    }
    assert {item.node.name for item in diff.downstream} == {"dim_customers", "mart_revenue_daily"}
