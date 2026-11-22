import copy
import json
from pathlib import Path

import pytest

from asklineage.cli import main
from asklineage.diff.compare import diff_manifests
from asklineage.diff.report import diff_json, render_diff
from asklineage.ingest.manifest import parse_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_manifest.json"
STG = "model.shop.stg_orders"
FCT = "model.shop.fct_orders"
SOURCE = "source.shop.raw.orders"


@pytest.fixture
def manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def diff_of(old: dict, new: dict):
    return diff_manifests(parse_manifest(old), parse_manifest(new))


def test_no_changes_is_one_line(manifest: dict) -> None:
    assert render_diff(diff_of(manifest, manifest)) == "shop: no changes"


def test_sql_change_lists_downstream_with_owner_sources(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][STG]["raw_code"] += " where true"

    text = render_diff(diff_of(manifest, new))

    assert text.startswith("shop: 0 added, 0 removed, 1 changed (1 data, 0 docs only), edges +0 -0")
    owners = "owners: Marketing analytics 1, finance-team 1"
    assert f"Downstream of data changes: 2 node(s); {owners}" in text
    finance = next(line for line in text.splitlines() if "fct_orders" in line)
    assert "finance-team (dbt_meta)" in finance
    assert finance.endswith("via stg_orders")
    assert "1 hop   model" in text


def test_docs_only_change_has_no_downstream_section(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][FCT]["description"] = "Orders."

    text = render_diff(diff_of(manifest, new))

    assert "description  (docs only)" in text
    assert "Downstream" not in text


def test_data_change_with_nothing_downstream_says_zero(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][FCT]["config"]["materialized"] = "incremental"

    assert "Downstream of data changes: 0 node(s)" in render_diff(diff_of(manifest, new))


def test_sources_show_as_source_name_dot_table(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["sources"][SOURCE]["relation_name"] = "raw_db.shop_v2.orders"

    text = render_diff(diff_of(manifest, new))

    assert "raw.orders" in text
    assert "via raw.orders" in text


def test_sections_are_cut_at_the_limit(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][STG]["raw_code"] += " where true"

    text = render_diff(diff_of(manifest, new), limit=1)

    assert "  ... and 1 more" in text


def test_json_marks_data_changes(manifest: dict) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][STG]["raw_code"] += " where true"

    data = diff_json(diff_of(manifest, new))

    assert data["changed"][0]["affects_data"] is True
    assert {item["node"]["unique_id"] for item in data["downstream"]} == {
        FCT,
        "model.shop.dim_customers",
    }
    json.dumps(data)


def write(tmp_path: Path, name: str, manifest: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(manifest))
    return str(path)


def test_cli_exit_code_only_when_asked_and_different(
    manifest: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    new = copy.deepcopy(manifest)
    new["nodes"][STG]["raw_code"] += " where true"
    old_path, new_path = write(tmp_path, "old.json", manifest), write(tmp_path, "new.json", new)

    main(["diff", old_path, new_path])
    assert "1 changed" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exited:
        main(["diff", old_path, new_path, "--exit-code"])
    assert exited.value.code == 1

    main(["diff", old_path, old_path, "--exit-code"])
    assert capsys.readouterr().out.strip().endswith("shop: no changes")


def test_cli_rejects_different_projects(
    manifest: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = copy.deepcopy(manifest)
    other["metadata"]["project_name"] = "other"

    with pytest.raises(SystemExit) as exited:
        main(["diff", write(tmp_path, "a.json", manifest), write(tmp_path, "b.json", other)])

    assert exited.value.code == 2
    assert "Cannot diff different projects" in capsys.readouterr().err
