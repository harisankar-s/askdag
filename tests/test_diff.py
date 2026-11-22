import copy
import json
from pathlib import Path

import pytest

from asklineage.diff.compare import ProjectMismatchError, diff_manifests
from asklineage.ingest.manifest import ParsedManifest, parse_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_manifest.json"
STG = "model.shop.stg_orders"
FCT = "model.shop.fct_orders"
DIM = "model.shop.dim_customers"
SEED = "seed.shop.country_codes"
SOURCE = "source.shop.raw.orders"


@pytest.fixture
def manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def parse(manifest: dict) -> ParsedManifest:
    return parse_manifest(manifest)


def edited(manifest: dict) -> dict:
    return copy.deepcopy(manifest)


def downstream_of(diff) -> dict[str, tuple[int, str]]:
    return {item.node.unique_id: (item.hops, item.via) for item in diff.downstream}


def test_identical_manifests_have_no_changes(manifest: dict) -> None:
    diff = diff_manifests(parse(manifest), parse(manifest))

    assert diff.is_empty
    assert diff.added_edges == diff.removed_edges == diff.downstream == ()


def test_sql_change_reports_everything_downstream(manifest: dict) -> None:
    new = edited(manifest)
    new["nodes"][STG]["raw_code"] += " where status != 'test'"

    diff = diff_manifests(parse(manifest), parse(new))

    assert [(change.node.unique_id, change.fields) for change in diff.changed] == [
        (STG, ("raw_code",))
    ]
    assert diff.changed[0].affects_data
    assert downstream_of(diff) == {FCT: (1, STG), DIM: (1, STG)}


def test_metadata_only_change_has_no_downstream(manifest: dict) -> None:
    new = edited(manifest)
    new["nodes"][STG]["description"] = "Orders, cleaned and typed."
    new["nodes"][STG]["columns"]["status"]["description"] = "Order status."

    diff = diff_manifests(parse(manifest), parse(new))

    assert diff.changed[0].fields == ("description", "columns")
    assert not diff.changed[0].affects_data
    assert diff.downstream == ()


def test_owner_change_is_metadata(manifest: dict) -> None:
    new = edited(manifest)
    new["nodes"][FCT]["config"]["meta"]["owner"] = "revenue-team"

    diff = diff_manifests(parse(manifest), parse(new))

    assert diff.changed[0].fields == ("owner",)
    assert diff.downstream == ()


def test_added_model_adds_an_edge_without_changing_its_parent(manifest: dict) -> None:
    new = edited(manifest)
    added = "model.shop.fct_revenue"
    new["nodes"][added] = {
        **copy.deepcopy(manifest["nodes"][FCT]),
        "unique_id": added,
        "name": "fct_revenue",
        "raw_code": "select sum(amount) from {{ ref('fct_orders') }}",
    }
    new["parent_map"][added] = [FCT]

    diff = diff_manifests(parse(manifest), parse(new))

    assert [node.unique_id for node in diff.added] == [added]
    assert diff.added_edges == ((FCT, added),)
    assert diff.changed == ()
    assert diff.downstream == ()


def test_removed_model_removes_its_edges(manifest: dict) -> None:
    new = edited(manifest)
    del new["nodes"][DIM]
    del new["parent_map"][DIM]

    diff = diff_manifests(parse(manifest), parse(new))

    assert [node.unique_id for node in diff.removed] == [DIM]
    assert set(diff.removed_edges) == {(STG, DIM), (SEED, DIM)}
    assert diff.changed == ()


def test_dropped_parent_is_a_data_change(manifest: dict) -> None:
    new = edited(manifest)
    new["parent_map"][DIM] = [STG]

    diff = diff_manifests(parse(manifest), parse(new))

    assert diff.changed[0].node.unique_id == DIM
    assert diff.changed[0].fields == ("parents",)
    assert diff.removed_edges == ((SEED, DIM),)


def test_rename_is_a_remove_and_an_add_with_a_hint(manifest: dict) -> None:
    new = edited(manifest)
    renamed = "model.shop.stg_shop_orders"
    new["nodes"][renamed] = {
        **new["nodes"].pop(STG),
        "unique_id": renamed,
        "name": "stg_shop_orders",
    }
    new["parent_map"][renamed] = new["parent_map"].pop(STG)
    new["parent_map"][FCT] = [renamed]
    new["parent_map"][DIM] = [renamed, SEED]

    diff = diff_manifests(parse(manifest), parse(new))

    assert [node.unique_id for node in diff.removed] == [STG]
    assert [node.unique_id for node in diff.added] == [renamed]
    assert diff.possible_renames == ((STG, renamed),)
    assert {change.node.unique_id for change in diff.changed} == {FCT, DIM}
    assert diff.downstream == ()


def test_changed_nodes_downstream_of_another_change_are_listed_once(manifest: dict) -> None:
    new = edited(manifest)
    new["sources"][SOURCE]["relation_name"] = "raw_db.shop_v2.orders"
    new["nodes"][STG]["raw_code"] += " where true"

    diff = diff_manifests(parse(manifest), parse(new))

    assert {change.node.unique_id for change in diff.changed} == {SOURCE, STG}
    assert downstream_of(diff) == {FCT: (1, STG), DIM: (1, STG)}


def test_metadata_changed_node_still_shows_as_downstream(manifest: dict) -> None:
    new = edited(manifest)
    new["nodes"][STG]["raw_code"] += " where true"
    new["nodes"][FCT]["description"] = "One row per order, with status."

    diff = diff_manifests(parse(manifest), parse(new))

    assert FCT in downstream_of(diff)


def test_rejects_different_projects(manifest: dict) -> None:
    new = edited(manifest)
    new["metadata"]["project_name"] = "other"

    with pytest.raises(ProjectMismatchError, match="shop and other"):
        diff_manifests(parse(manifest), parse(new))
