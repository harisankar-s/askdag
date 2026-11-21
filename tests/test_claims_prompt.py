import json

import pytest
from pydantic import ValidationError

from asklineage.answer.claims import Highlights
from asklineage.answer.prompt import EXAMPLE_FACTS, EXAMPLES, build_prompt, system_prompt
from asklineage.config import Settings
from asklineage.lineage.graph import Direction

CLAIM = {"upstream": "a", "downstream": "b", "hops": "direct", "text": "`b` reads from `a`."}


@pytest.mark.parametrize(
    "claims",
    [
        [],
        [CLAIM] * 4,
        [{**CLAIM, "confidence": 0.9}],
        [{key: value for key, value in CLAIM.items() if key != "hops"}],
        [{**CLAIM, "hops": "sideways"}],
        [{"kind": "total", "value": 3, "text": "3 nodes."}],
    ],
    ids=["none", "too-many", "extra-field", "missing-hops", "bad-hops", "old-count-claim"],
)
def test_malformed_highlights_are_rejected(claims):
    with pytest.raises(ValidationError):
        Highlights.model_validate({"claims": claims})


def test_schema_puts_text_after_the_checked_fields():
    claim = Highlights.model_json_schema()["$defs"]["RelationClaim"]

    assert list(claim["properties"])[-1] == "text"
    assert claim["additionalProperties"] is False


@pytest.mark.parametrize("direction", list(Direction))
def test_each_direction_gets_its_own_valid_example(direction):
    facts, highlights = EXAMPLES[direction]
    prompt = system_prompt(direction)

    assert facts.direction is direction
    assert f'"direction": "{direction}"' in prompt
    assert Highlights.model_validate_json(prompt.split("Example reply:\n", 1)[1]) == highlights


@pytest.mark.parametrize("direction", list(Direction))
def test_example_facts_add_up(direction):
    facts, _ = EXAMPLES[direction]

    assert sum(facts.by_depth.values()) == facts.total
    assert sum(facts.by_type.values()) == facts.total
    assert sum(g.nodes for g in facts.owners) + facts.unowned == facts.total
    assert len(facts.named) + facts.unnamed == facts.total


def test_prompt_carries_names_and_owners_but_no_counts():
    prompt = build_prompt(EXAMPLE_FACTS)
    data = json.loads(prompt.split("Facts:\n", 1)[1].rsplit("\n\n", 1)[0])

    assert data["root"] == "stg_orders"
    assert data["named"][1] == {
        "name": "mart_weekly",
        "resource_type": "model",
        "depth": 2,
        "owner": None,
    }
    assert "total" not in data


def test_named_node_cap_defaults_to_20_and_is_bounded(monkeypatch):
    monkeypatch.delenv("LLM_MAX_NAMED_NODES", raising=False)

    assert Settings(_env_file=None).llm_max_named_nodes == 20
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_max_named_nodes=0)
