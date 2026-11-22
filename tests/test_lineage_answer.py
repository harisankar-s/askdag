from dataclasses import replace

from lineage_examples import (
    EXAMPLE_LINEAGE,
    EXAMPLE_ROOT,
    UPSTREAM_EXAMPLE_LINEAGE,
    UPSTREAM_EXAMPLE_ROOT,
)

from asklineage.answer.highlights import Highlight, majority_owner, pick
from asklineage.answer.lineage_answer import answer_lineage, tag_owners
from asklineage.answer.template import template_summary


def names(nodes):
    return [n.display_name for n in nodes]


def test_summary_comes_from_the_template():
    answer = answer_lineage(EXAMPLE_LINEAGE, EXAMPLE_ROOT)

    assert answer.summary == template_summary(answer.facts)
    assert answer.lineage is EXAMPLE_LINEAGE


def test_downstream_picks_another_team_then_the_furthest_then_a_direct_node():
    answer = answer_lineage(EXAMPLE_LINEAGE, EXAMPLE_ROOT)

    assert answer.highlights == (
        Highlight(
            "stg_orders",
            "mart_weekly",
            "indirect",
            "`mart_weekly` depends on `stg_orders` through other nodes.",
        ),
        Highlight(
            "stg_orders",
            "mart_revenue",
            "indirect",
            "`mart_revenue` depends on `stg_orders` through other nodes.",
        ),
        Highlight(
            "stg_orders", "fct_orders", "direct", "`fct_orders` reads directly from `stg_orders`."
        ),
    )
    assert answer.highlight_texts[0] == (
        "`mart_weekly` (no owner recorded) depends on `stg_orders` through other nodes."
    )


def test_upstream_sentences_put_the_root_first_and_name_sources():
    answer = answer_lineage(UPSTREAM_EXAMPLE_LINEAGE, UPSTREAM_EXAMPLE_ROOT)

    assert [(h.upstream, h.downstream, h.hops) for h in answer.highlights] == [
        ("raw.payments", "fct_orders", "indirect"),
        ("raw.orders", "fct_orders", "indirect"),
        ("stg_orders", "fct_orders", "direct"),
    ]
    assert answer.highlights[0].text == (
        "`fct_orders` depends on the source `raw.payments` through other nodes."
    )
    assert answer.highlights[2].text == "`fct_orders` reads directly from `stg_orders`."


def test_a_direct_highlight_is_an_edge_of_the_result():
    for lineage, root in (
        (EXAMPLE_LINEAGE, EXAMPLE_ROOT),
        (UPSTREAM_EXAMPLE_LINEAGE, UPSTREAM_EXAMPLE_ROOT),
    ):
        ids = {n.display_name: n.unique_id for n in lineage.nodes} | {
            root.display_name: root.unique_id
        }
        for h in answer_lineage(lineage, root).highlights:
            if h.hops == "direct":
                assert (ids[h.upstream], ids[h.downstream]) in lineage.edges


def test_each_other_team_is_named_before_the_majority_smallest_first():
    base = EXAMPLE_LINEAGE.nodes[0]
    nodes = (
        *(replace(base, unique_id=f"m.{i}", name=f"gtfs_{i}", owner="gtfs") for i in range(5)),
        replace(base, unique_id="m.pay1", name="pay_a", owner="payments", depth=2),
        replace(base, unique_id="m.pay2", name="pay_b", owner="payments"),
        replace(base, unique_id="m.tides", name="tides_a", owner="tides", depth=2),
        replace(base, unique_id="m.none", name="loose", owner=None, depth=2),
        replace(base, unique_id="m.deep", name="gtfs_deep", owner="gtfs", depth=2),
    )

    assert names(pick(nodes)) == ["loose", "tides_a", "pay_b"]


def test_one_team_and_only_direct_nodes_gives_one_direct_highlight():
    direct = tuple(n for n in EXAMPLE_LINEAGE.nodes if n.depth == 1)

    assert names(pick(direct)) == ["fct_orders"]


def test_when_most_nodes_are_unowned_an_owned_node_is_the_other_team():
    nodes = tuple(replace(n, owner=None) for n in EXAMPLE_LINEAGE.nodes[1:])
    nodes = (EXAMPLE_LINEAGE.nodes[0], *nodes)

    assert majority_owner(nodes) is None
    assert names(pick(nodes))[0] == "fct_orders"


def test_an_empty_result_has_no_highlights():
    empty = replace(EXAMPLE_LINEAGE, nodes=(), edges=(), truncated=False)

    answer = answer_lineage(empty, EXAMPLE_ROOT)

    assert answer.highlights == ()
    assert answer.summary == "Nothing reads from `stg_orders`."


def test_owner_tags_go_on_the_first_mention_only_and_skip_unknown_names():
    owners = {"a": "team-a", "b": None}

    assert tag_owners("`a` feeds `b`, and `a` feeds `root`.", owners) == (
        "`a` (team-a) feeds `b` (no owner recorded), and `a` feeds `root`."
    )
