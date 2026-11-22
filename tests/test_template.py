from dataclasses import replace

from lineage_examples import EXAMPLE_LINEAGE, EXAMPLE_ROOT

from asklineage.answer.facts import OWNER_SOURCE_MEANING, OwnerGroup, build_facts
from asklineage.answer.template import template_summary
from asklineage.lineage.graph import Direction

EXAMPLE_FACTS = build_facts(EXAMPLE_LINEAGE, EXAMPLE_ROOT)

PATH_RULES = OWNER_SOURCE_MEANING["owners_file"]
DBT_META = OWNER_SOURCE_MEANING["dbt_meta"]


def test_downstream_summary_states_counts_teams_owner_source_and_truncation():
    assert template_summary(EXAMPLE_FACTS) == (
        "5 nodes depend on `stg_orders` within 2 hops (2 directly, 3 at 2 hops). "
        "Teams affected: finance (4 nodes). "
        f"Owners are {PATH_RULES}. "
        "1 node has no owner recorded. "
        "More nodes lie beyond 2 hops; raise the depth to see them."
    )


def test_upstream_summary_puts_the_root_first_and_says_who_owns_the_sources():
    facts = replace(EXAMPLE_FACTS, direction=Direction.UPSTREAM, truncated=False)

    summary = template_summary(facts)

    assert summary.startswith(
        "`stg_orders` depends on 5 nodes within 2 hops (2 directly, 3 at 2 hops). "
        "Owned by: finance (4 nodes)."
    )


def test_every_team_is_listed_largest_first():
    owners = (
        OwnerGroup("gtfs-team", "owners_file", PATH_RULES, 101),
        OwnerGroup("payments-team", "owners_file", PATH_RULES, 2),
        OwnerGroup("tides-team", "owners_file", PATH_RULES, 2),
    )
    facts = replace(EXAMPLE_FACTS, owners=owners, unowned=0, truncated=False)

    expected = (
        "Teams affected: gtfs-team (101 nodes), payments-team (2 nodes), tides-team (2 nodes)."
    )
    assert expected in template_summary(facts)


def test_a_team_with_two_owner_sources_is_counted_once():
    owners = (
        OwnerGroup("finance", "owners_file", PATH_RULES, 3),
        OwnerGroup("finance", "dbt_meta", DBT_META, 1),
    )
    facts = replace(EXAMPLE_FACTS, owners=owners, unowned=1)

    summary = template_summary(facts)

    assert "Teams affected: finance (4 nodes)." in summary
    assert f"Owners come from: {PATH_RULES} (3 nodes); {DBT_META} (1 node)." in summary


def test_long_team_lists_are_capped():
    owners = tuple(OwnerGroup(f"team-{i:02}", "owners_file", PATH_RULES, 1) for i in range(10))
    facts = replace(EXAMPLE_FACTS, owners=owners, unowned=0, truncated=False)

    summary = template_summary(facts)

    assert "team-07" in summary
    assert "team-08" not in summary
    assert "and 2 more teams." in summary


def test_single_node_uses_singular_wording():
    facts = replace(EXAMPLE_FACTS, total=1, by_depth={1: 1}, owners=(), unowned=1, truncated=False)

    assert template_summary(facts) == (
        "1 node depends on `stg_orders` within 2 hops (1 directly). 1 node has no owner recorded."
    )


def test_empty_results():
    empty = replace(EXAMPLE_FACTS, total=0, by_depth={}, owners=(), unowned=0, truncated=False)

    assert template_summary(empty) == "Nothing reads from `stg_orders`."
    assert template_summary(replace(empty, direction=Direction.UPSTREAM)) == (
        "`stg_orders` reads from no other node."
    )
