"""Highlights: up to three relationships worth noticing, chosen and written in code (ADR 0019).

The rule is the one the LLM was once prompted with, followed exactly:
1. for each team other than the one owning most nodes (and for unowned nodes), its nearest
   node: the people someone changing the root is most likely to forget. Smallest teams first,
   unowned before a team of the same size, since then no one would be told;
2. with slots left, the furthest node, when it is more than one hop away;
3. then a direct node.
Candidates are every node in the walk, not only the capped list of names. Each sentence says
only what the walk proved: who reads from whom, directly or not.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from asklineage.answer.facts import LineageFacts
from asklineage.lineage.graph import Direction, LineageNode

MAX_HIGHLIGHTS = 3


@dataclass(frozen=True)
class Highlight:
    upstream: str
    downstream: str
    hops: Literal["direct", "indirect"]
    text: str  # names in backticks; owner tags are added later (lineage_answer.tag_owners)


def majority_owner(nodes: Sequence[LineageNode]) -> str | None:
    """The owner of most nodes, ties by name; None when unowned nodes outnumber every team."""
    counts = Counter(n.owner for n in nodes)
    return min(counts, key=lambda owner: (-counts[owner], owner is None, owner or ""))


def pick(nodes: Sequence[LineageNode]) -> tuple[LineageNode, ...]:
    ordered = sorted(nodes, key=lambda n: (n.depth, n.display_name))
    if not ordered:
        return ()
    majority = majority_owner(ordered)
    picked: list[LineageNode] = []

    def add(node: LineageNode | None) -> None:
        if node is not None and node not in picked:
            picked.append(node)

    others = Counter(n.owner for n in ordered if n.owner != majority)
    for owner in sorted(others, key=lambda o: (others[o], o is not None, o or "")):
        add(next(n for n in ordered if n.owner == owner))
    deepest = min(ordered, key=lambda n: (-n.depth, n.display_name))
    add(deepest if deepest.depth > 1 else None)
    add(next((n for n in ordered if n.depth == 1 and n not in picked), None))
    return tuple(picked[:MAX_HIGHLIGHTS])


def sentence(facts: LineageFacts, node: LineageNode) -> Highlight:
    root = f"`{facts.root}`"
    other = f"`{node.display_name}`"
    if facts.direction is Direction.UPSTREAM and node.resource_type == "source":
        other = f"the source {other}"
    direct = node.depth == 1
    if facts.direction is Direction.DOWNSTREAM:
        text = (
            f"{other} reads directly from {root}."
            if direct
            else (f"{other} depends on {root} through other nodes.")
        )
        upstream, downstream = facts.root, node.display_name
    else:
        text = (
            f"{root} reads directly from {other}."
            if direct
            else (f"{root} depends on {other} through other nodes.")
        )
        upstream, downstream = node.display_name, facts.root
    return Highlight(upstream, downstream, "direct" if direct else "indirect", text)


def highlights(facts: LineageFacts, nodes: Sequence[LineageNode]) -> tuple[Highlight, ...]:
    return tuple(sentence(facts, node) for node in pick(nodes))
