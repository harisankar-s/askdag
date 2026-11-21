"""The instructions and the worked example (one per direction) the highlights model sees."""

from asklineage.answer.claims import Highlights, RelationClaim
from asklineage.answer.facts import (
    DIRECTION_MEANING,
    OWNER_SOURCE_MEANING,
    LineageFacts,
    NamedNode,
    OwnerGroup,
)
from asklineage.lineage.graph import Direction

RULES = """\
You pick the relationships in a dbt lineage result that matter most to someone about to change \
the root node, and state each in one sentence. The user already sees the counts, the owners \
and a table of every node, so never restate them.

You get the facts as JSON. Reply with 1 to 3 claims. Rules:
1. Use only the root and the nodes in "named". Never mention any other node.
2. Write every node name exactly as given, inside backticks, like `stg_orders`.
3. upstream is the node data comes from; downstream is the node that reads it. \
direction_meaning says which way data flows relative to the root.
4. hops is direct when the named node has depth 1, otherwise indirect.
5. Use each node's owner only to choose: if there are nodes owned by a different team from most \
of the others, or with owner null, prefer those; otherwise prefer the furthest nodes.
6. Never mention owners or teams in the text; they are added next to each name for you.
7. Never write a number."""

EXAMPLE_FACTS = LineageFacts(
    root="stg_orders",
    root_type="model",
    direction=Direction.DOWNSTREAM,
    direction_meaning=DIRECTION_MEANING[Direction.DOWNSTREAM],
    max_depth=2,
    truncated=True,
    total=5,
    by_depth={1: 2, 2: 3},
    by_type={"model": 5},
    named=(
        NamedNode("fct_orders", "model", 1, "finance"),
        NamedNode("mart_weekly", "model", 2, None),
        NamedNode("int_order_items", "model", 1, "finance"),
    ),
    unnamed=2,
    owners=(OwnerGroup("finance", "owners_file", OWNER_SOURCE_MEANING["owners_file"], 4),),
    unowned=1,
)

EXAMPLE_HIGHLIGHTS = Highlights(
    claims=[
        RelationClaim(
            upstream="stg_orders",
            downstream="mart_weekly",
            hops="indirect",
            text="`mart_weekly` depends on `stg_orders` through other models.",
        ),
        RelationClaim(
            upstream="stg_orders",
            downstream="fct_orders",
            hops="direct",
            text="`fct_orders` reads directly from `stg_orders`.",
        ),
    ]
)

UPSTREAM_EXAMPLE_FACTS = LineageFacts(
    root="fct_orders",
    root_type="model",
    direction=Direction.UPSTREAM,
    direction_meaning=DIRECTION_MEANING[Direction.UPSTREAM],
    max_depth=2,
    truncated=False,
    total=4,
    by_depth={1: 2, 2: 2},
    by_type={"model": 2, "source": 2},
    named=(
        NamedNode("stg_orders", "model", 1, "finance"),
        NamedNode("raw.payments", "source", 2, None),
        NamedNode("stg_payments", "model", 1, "finance"),
    ),
    unnamed=1,
    owners=(OwnerGroup("finance", "owners_file", OWNER_SOURCE_MEANING["owners_file"], 3),),
    unowned=1,
)

UPSTREAM_EXAMPLE_HIGHLIGHTS = Highlights(
    claims=[
        RelationClaim(
            upstream="raw.payments",
            downstream="fct_orders",
            hops="indirect",
            text="`fct_orders` depends on the source `raw.payments` through another model.",
        ),
        RelationClaim(
            upstream="stg_orders",
            downstream="fct_orders",
            hops="direct",
            text="`fct_orders` reads directly from `stg_orders`.",
        ),
    ]
)

EXAMPLES: dict[Direction, tuple[LineageFacts, Highlights]] = {
    Direction.DOWNSTREAM: (EXAMPLE_FACTS, EXAMPLE_HIGHLIGHTS),
    Direction.UPSTREAM: (UPSTREAM_EXAMPLE_FACTS, UPSTREAM_EXAMPLE_HIGHLIGHTS),
}


def system_prompt(direction: Direction) -> str:
    # One example in the question's direction: the model copies its phrasing, so the other
    # direction's example teaches backwards sentences.
    facts, highlights = EXAMPLES[direction]
    return (
        f"{RULES}\n\nExample facts:\n{facts.prompt_json()}\n\n"
        f"Example reply:\n{highlights.model_dump_json(indent=2)}"
    )


def build_prompt(facts: LineageFacts) -> str:
    return f"Facts:\n{facts.prompt_json()}\n\nWrite the highlight claims for these facts."
