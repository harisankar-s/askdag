import json
from pathlib import Path
from uuid import uuid4

import pytest
from synthetic_manifest import build_manifest

from asklineage.answer.parse import entity_in_question
from asklineage.api.models import AskRequest
from asklineage.eval.golden import (
    GoldenItem,
    GoldenSetError,
    NameIndex,
    build_golden,
    describe,
)
from asklineage.eval.templates import TEMPLATES, hops, levels
from asklineage.eval.truth import truth_graph
from asklineage.lineage.graph import MAX_DEPTH

MINI_MANIFEST = Path(__file__).parent / "fixtures" / "mini_manifest.json"
MANIFEST = build_manifest()
GRAPH = truth_graph(MANIFEST)
OWNERS = {
    uid: "team_a" if ".int_" in uid else "team_b" if ".mart_" in uid else None
    for uid in GRAPH.node_ids
}
GOLDEN = build_golden(MANIFEST, GRAPH, OWNERS)
ITEMS = {item.id: item for item in GOLDEN.items}
HUB = "model.synthetic.int_hub"
EVENTS = ("source.synthetic.raw_app.events", "source.synthetic.raw_billing.events")


def turns(items: tuple[GoldenItem, ...] = GOLDEN.items):
    return [turn for item in items for turn in item.turns]


def test_the_same_manifest_gives_the_same_set() -> None:
    again = build_golden(build_manifest(), truth_graph(build_manifest()), OWNERS)
    assert again == GOLDEN
    assert again.sha == GOLDEN.sha
    assert build_golden(MANIFEST, GRAPH, OWNERS, "all").sha != GOLDEN.sha


def test_roles_follow_the_rules() -> None:
    assert GOLDEN.missing_roles == ()
    roles = GOLDEN.roles
    assert roles["hub"] == "int_hub"
    assert roles["deep"] == "chain_28"
    assert roles["source"] == "raw_crm.accounts"
    assert roles["shared"] == "events"
    assert roles["prefix"] == "int_hub"
    assert roles["typo"] == "int_uhb"
    assert "multi_team" in roles


def test_expectations_follow_the_exact_name_rule() -> None:
    def expect(item_id: str, turn: int = 0):
        return ITEMS[item_id].turns[turn].expect

    assert expect("s-hub-upper-case").node_id == HUB
    assert expect("s-hub-unique-id").node_id == HUB
    assert expect("s-source-table-only").node_id == "source.synthetic.raw_crm.accounts"
    assert expect("s-shared").reason == "ambiguous"
    assert expect("s-shared").candidates == EVENTS
    assert expect("s-shared-qualified").node_id == EVENTS[0]
    assert expect("s-typo").reason == "not_found"
    assert expect("s-typo").suggest == HUB
    assert expect("s-unknown").reason == "not_found"
    assert expect("s-deep-up-max").depth == MAX_DEPTH
    assert expect("t-same-name-again", 1).depth == 2
    assert ITEMS["t-choose-structured"].turns[1].request == {"choose": EVENTS[0]}


def test_typo_and_unknown_names_match_no_node() -> None:
    index = NameIndex(MANIFEST, GRAPH)
    assert index.exact(GOLDEN.roles["typo"]) == ()
    assert sorted(GOLDEN.roles["typo"]) == sorted("int_hub")
    assert index.exact(ITEMS["s-unknown"].turns[0].request["node"]) == ()


def test_free_text_questions_name_their_node_as_a_whole_name() -> None:
    named = [turn for turn in turns() if turn.free_text and turn.about]
    assert named
    for turn in named:
        assert entity_in_question(turn.about, turn.request["question"]), turn.request


def test_every_request_is_a_valid_ask_request() -> None:
    for turn in turns():
        thread = {"thread_id": uuid4()} if "choose" in turn.request else {}
        AskRequest(**turn.request, **thread)


def test_item_ids_are_unique_with_every_phrasing() -> None:
    every = build_golden(MANIFEST, GRAPH, OWNERS, "all")
    ids = [item.id for item in every.items]
    assert len(ids) == len(set(ids))
    expanded = [item for item in every.items if item.id.startswith("f-hub-down~")]
    assert len(expanded) == len(TEMPLATES["down"])
    assert len(ITEMS) == len(GOLDEN.items)


def test_follow_ups_say_whether_to_reuse_the_previous_node() -> None:
    follow_ups = [t for item in GOLDEN.items if item.thread for t in item.turns[1:]]
    assert {turn.same_node for turn in follow_ups} >= {"yes", "no", "either"}


@pytest.mark.parametrize("depth", [1, 2])
def test_templates_render_every_placeholder_in_good_english(depth: int) -> None:
    for templates in TEMPLATES.values():
        for template in templates:
            text = template.render("stg_orders", depth)
            assert "{" not in text and "}" not in text
            assert "stg_orders" in text
            assert "1 hops" not in text and "one levels" not in text
    assert hops(1) == "1 hop"
    assert hops(3) == "3 hops"
    assert levels(1) == "one level"
    assert levels(2) == "two levels"
    assert levels(12) == "12 levels"


def test_set_is_json_and_describes_every_item() -> None:
    json.dumps(GOLDEN.to_dict())
    text = describe(GOLDEN)
    assert GOLDEN.sha in text
    assert all(item.id in text for item in GOLDEN.items)


def test_a_manifest_without_a_seed_is_refused() -> None:
    manifest = json.loads(MINI_MANIFEST.read_text())
    del manifest["nodes"]["seed.shop.country_codes"]
    with pytest.raises(GoldenSetError, match="seed"):
        build_golden(manifest, truth_graph(manifest), {})
