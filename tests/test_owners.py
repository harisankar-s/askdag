from pathlib import Path

from asklineage.ingest.owners import OwnerRule, load_owner_rules, match_owner


def test_first_matching_rule_wins() -> None:
    rules = [
        OwnerRule(pattern="models/*/payments*", owner="payments-team"),
        OwnerRule(pattern="models/*", owner="data-platform-team"),
    ]

    assert match_owner(rules, "models/mart/payments/fct_payments.sql") == "payments-team"
    assert match_owner(rules, "models/staging/gtfs/stg_gtfs__trips.sql") == "data-platform-team"
    assert match_owner(rules, "seeds/codes.csv") is None


def test_load_owner_rules_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "owners.yml"
    path.write_text('rules:\n  - path: "seeds/*"\n    owner: data-platform-team\n')

    assert load_owner_rules(path) == (OwnerRule(pattern="seeds/*", owner="data-platform-team"),)
