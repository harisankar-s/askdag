import json
from pathlib import Path

import pytest

from asklineage.ingest.manifest import UnsupportedManifestError, parse_manifest
from asklineage.ingest.owners import OwnerRule

FIXTURE = Path(__file__).parent / "fixtures" / "mini_manifest.json"


@pytest.fixture
def manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def test_keeps_models_seeds_and_sources_but_not_tests(manifest: dict) -> None:
    parsed = parse_manifest(manifest)

    assert {node.unique_id for node in parsed.nodes} == {
        "model.shop.stg_orders",
        "model.shop.fct_orders",
        "model.shop.dim_customers",
        "seed.shop.country_codes",
        "source.shop.raw.orders",
    }


def test_edges_come_from_parent_map_without_tests(manifest: dict) -> None:
    parsed = parse_manifest(manifest)

    assert set(parsed.edges) == {
        ("source.shop.raw.orders", "model.shop.stg_orders"),
        ("model.shop.stg_orders", "model.shop.fct_orders"),
        ("model.shop.stg_orders", "model.shop.dim_customers"),
        ("seed.shop.country_codes", "model.shop.dim_customers"),
    }


def test_owner_precedence_is_meta_then_group_then_owner_rules(manifest: dict) -> None:
    rules = [OwnerRule(pattern="models/staging/*", owner="platform-team")]
    parsed = parse_manifest(manifest, rules)
    owners = {node.name: (node.owner, node.owner_source) for node in parsed.nodes}

    assert owners["fct_orders"] == ("finance-team", "dbt_meta")
    assert owners["dim_customers"] == ("Marketing analytics", "dbt_group")
    assert owners["stg_orders"] == ("platform-team", "owners_file")
    assert owners["orders"] == ("platform-team", "owners_file")
    assert owners["country_codes"] == (None, None)


def test_tests_attach_only_to_known_nodes(manifest: dict) -> None:
    tests = {test.name: test for test in parse_manifest(manifest).tests}

    not_null = tests["not_null_stg_orders_order_id"]
    assert not_null.attached_node_id == "model.shop.stg_orders"
    assert not_null.test_type == "not_null"
    assert not_null.column_name == "order_id"
    assert tests["broken_source_test"].attached_node_id is None


def test_columns_are_collected_for_models_and_sources(manifest: dict) -> None:
    columns = {(column.node_id, column.name) for column in parse_manifest(manifest).columns}

    assert columns == {
        ("model.shop.stg_orders", "order_id"),
        ("model.shop.stg_orders", "status"),
        ("source.shop.raw.orders", "id"),
    }


def test_rejects_unsupported_schema_version(manifest: dict) -> None:
    manifest["metadata"]["dbt_schema_version"] = "https://schemas.getdbt.com/dbt/manifest/v11.json"

    with pytest.raises(UnsupportedManifestError, match="v11"):
        parse_manifest(manifest)
