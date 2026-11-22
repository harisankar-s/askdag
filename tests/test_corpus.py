import json
from pathlib import Path

import pytest
from synthetic_manifest import PROJECT, build_manifest

from asklineage.corpus.docs import (
    SECTIONS,
    TODO,
    CorpusError,
    check_corpus,
    describe,
    names_in,
    render_document,
    split_front_matter,
)
from asklineage.corpus.scaffold import scaffold, write_scaffold

MARTS = ("models/marts/*",)
MINI = json.loads((Path(__file__).parent / "fixtures" / "mini_manifest.json").read_text())


def synthetic() -> dict:
    manifest = build_manifest()
    for name in ("int_000", "int_001", "int_hub"):
        manifest["nodes"][f"model.{PROJECT}.{name}"]["config"]["materialized"] = "incremental"
    return manifest


def prose(meta: dict) -> tuple[str, str]:
    """A title and body that satisfy the checks, as the hand-written prose must."""
    names = meta.get("nodes") or [*meta.get("symptom_nodes", []), meta.get("root_cause_node")]
    named = " and ".join(names) if meta["mention"] == "exact" else "the usual tables"
    if meta["type"] == "slack":
        return "Numbers look off", f"- 09:01 **ana**: is {named} late?\n- 09:05 **raj**: looking"
    sections = SECTIONS[meta["type"]]
    return "Something broke", "\n\n".join(f"## {s}\n\nAbout {named}." for s in sections)


def fill(directory: Path) -> None:
    for path in directory.glob("*/*.md"):
        meta, _ = split_front_matter(path.read_text())
        meta["title"], body = prose(meta)
        path.write_text(render_document(meta, body))


def write_corpus(directory: Path, project: str, docs: list[tuple[dict, str]]) -> Path:
    (directory / "corpus.yml").write_text(f"version: 1\nproject: {project}\nseed: 1\n")
    folders = {"incident": "incidents", "runbook": "runbooks", "slack": "slack"}
    for meta, body in docs:
        target = directory / folders[meta["type"]] / f"{meta['id']}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_document(meta, body))
    return directory


def incident(**changes) -> dict:
    meta = {
        "id": "inc-001",
        "type": "incident",
        "title": "Orders went missing",
        "date": "2025-02-01",
        "severity": "sev2",
        "mention": "exact",
        "cause": "source_outage",
        "symptom_nodes": ["fct_orders"],
        "root_cause_node": "raw.orders",
        "failed_tests": ["not_null_stg_orders_order_id"],
    }
    return {**meta, **changes}


def incident_body(text: str = "raw.orders stopped, so fct_orders was short.") -> str:
    return "\n\n".join(f"## {s}\n\n{text}" for s in SECTIONS["incident"])


def problems_for(tmp_path: Path, docs: list[tuple[dict, str]]) -> list[str]:
    return list(check_corpus(write_corpus(tmp_path, "shop", docs), MINI).problems)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scaffold_facts_pass_the_checks_once_prose_is_written(tmp_path, seed):
    skeletons = scaffold(synthetic(), PROJECT, seed, symptom_paths=MARTS)
    write_scaffold(tmp_path, PROJECT, seed, skeletons)

    drafts = check_corpus(tmp_path, synthetic())
    assert drafts.problems == ()
    assert len(drafts.drafts) == len(skeletons)

    fill(tmp_path)
    report = check_corpus(tmp_path, synthetic())
    assert report.problems == ()
    assert report.drafts == ()
    types = describe(report)["types"]
    assert (types["incident"], types["slack"]) == (40, 25)
    assert 1 <= types["runbook"] <= 12


def test_scaffold_is_deterministic_and_skewed():
    first = scaffold(synthetic(), PROJECT, 7, symptom_paths=MARTS)
    assert first == scaffold(synthetic(), PROJECT, 7, symptom_paths=MARTS)
    assert first != scaffold(synthetic(), PROJECT, 8, symptom_paths=MARTS)
    symptoms = [s.meta["symptom_nodes"][0] for s in first if s.folder == "incidents"]
    assert max(symptoms.count(name) for name in symptoms) >= 3
    assert sum("decoy_of" in s.meta for s in first) == 6
    assert all(name.startswith("mart_") for name in symptoms)


def test_write_scaffold_writes_nothing_if_any_file_exists(tmp_path):
    skeletons = scaffold(synthetic(), PROJECT, 1, symptom_paths=MARTS)
    existing = tmp_path / skeletons[5].relative_path
    existing.parent.mkdir(parents=True)
    existing.write_text("hand-written")

    with pytest.raises(FileExistsError):
        write_scaffold(tmp_path, PROJECT, 1, skeletons)
    assert existing.read_text() == "hand-written"
    assert not (tmp_path / "corpus.yml").exists()
    assert len(list(tmp_path.glob("*/*.md"))) == 1


def test_a_valid_incident_passes(tmp_path):
    assert problems_for(tmp_path, [(incident(), incident_body())]) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"root_cause_node": "dim_customers", "cause": "logic_bug"}, "is not upstream"),
        ({"cause": "logic_bug"}, "starts at a model"),
        ({"root_cause_node": "stg_orders"}, "starts at a source"),
        ({"root_cause_node": "stg_orders", "cause": "incremental_gap"}, "incremental model"),
        ({"symptom_nodes": ["fct_order"]}, "no node named 'fct_order'"),
        ({"failed_tests": ["nope"]}, "no test named"),
        ({"severity": "high"}, "severity must be"),
        ({"runbook": "rb-01"}, "is not a runbook"),
    ],
)
def test_incident_facts_must_agree_with_the_manifest(tmp_path, changes, expected):
    problems = problems_for(tmp_path, [(incident(**changes), incident_body())])
    assert any(expected in p for p in problems), problems


def test_a_failed_test_must_sit_between_root_cause_and_symptom(tmp_path):
    meta = incident(symptom_nodes=["dim_customers"], root_cause_node="country_codes")
    meta["cause"] = "logic_bug"
    body = incident_body("country_codes was wrong, so dim_customers was too.")
    problems = problems_for(tmp_path, [(meta, body)])
    assert any("not on a node between" in p for p in problems), problems


def test_mention_rules_are_enforced(tmp_path):
    informal = incident(mention="informal")
    problems = problems_for(tmp_path, [(informal, incident_body())])
    assert any("informal but 'fct_orders' is named" in p for p in problems)
    assert any("informal but 'raw.orders' is named" in p for p in problems)

    exact = incident()
    problems = problems_for(tmp_path, [(exact, incident_body("the orders table broke"))])
    assert any("exact but 'fct_orders' is not named" in p for p in problems)


def test_sections_and_slack_shape(tmp_path):
    body = "## Summary\n\nraw.orders and fct_orders."
    assert any("sections must be" in p for p in problems_for(tmp_path, [(incident(), body)]))

    thread = {
        "id": "sl-001",
        "type": "slack",
        "title": "fct_orders is short",
        "date": "2025-02-05",
        "mention": "exact",
        "channel": "#data-incidents",
        "nodes": ["fct_orders"],
        "incident": "inc-001",
    }
    problems = problems_for(
        tmp_path, [(incident(), incident_body()), (thread, "fct_orders looks short")]
    )
    assert any("two '- HH:MM" in p for p in problems)
    assert any("within 2 days" in p for p in problems)


def test_a_decoy_needs_a_shared_symptom_and_a_different_cause(tmp_path):
    copy = incident(id="inc-002", date="2025-03-01", decoy_of="inc-001")
    problems = problems_for(tmp_path, [(incident(), incident_body()), (copy, incident_body())])
    assert any("but not its root cause" in p for p in problems)


def test_drafts_are_reported_and_skip_prose_checks(tmp_path):
    report = check_corpus(write_corpus(tmp_path, "shop", [(incident(title=TODO), "")]), MINI)
    assert report.problems == ()
    assert report.drafts == ("inc-001",)


def test_names_in_matches_whole_names_only():
    assert names_in("see stg_orders today", "stg_orders")
    assert not names_in("see stg_orders_daily", "stg_orders")
    assert names_in("raw.orders was late", "raw.orders")
    assert not names_in("the orders were late", "raw.orders")
    assert names_in("download_outcomes was empty", "gtfs.download_outcomes")


def test_id_and_folder_must_match_the_file(tmp_path):
    write_corpus(tmp_path, "shop", [(incident(), incident_body())])
    (tmp_path / "incidents" / "inc-001.md").rename(tmp_path / "incidents" / "inc-009.md")
    problems = check_corpus(tmp_path, MINI).problems
    assert any("must match the file name" in p for p in problems)


def test_missing_corpus_settings_is_an_error(tmp_path):
    with pytest.raises(CorpusError):
        check_corpus(tmp_path, MINI)


def test_a_node_listed_twice_is_a_problem(tmp_path):
    runbook = {
        "id": "rb-01",
        "type": "runbook",
        "title": "Rebuild fct_orders",
        "date": "2025-02-01",
        "mention": "exact",
        "cause": "logic_bug",
        "nodes": ["fct_orders", "fct_orders"],
    }
    body = "\n\n".join(f"## {s}\n\nfct_orders." for s in SECTIONS["runbook"])
    assert any("listed twice" in p for p in problems_for(tmp_path, [(runbook, body)]))
