"""Score a lineage answer's highlights: are they true, and are they worth reading?

Truth is checked twice. The first attempt's broken rules say how often the model states
something false (hallucination) or only breaks an output rule (a number, owner wording, an
unmarked name). Each passed highlight is then checked against the manifest's edges, not the
result's, so a gap in grounding.check can't hide behind its own verdict.

Usefulness is a code-computed proxy, compared with two baselines over the same named nodes:
the first ones the LLM was shown (what moving the choice into code would give) and a seeded
random pick. If code matches the LLM, the LLM isn't adding judgment. Whether the proxy tracks
what a reader finds useful is checked against hand labels later (7d).
"""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from asklineage.answer.facts import NamedNode, build_facts
from asklineage.api.models import LineageResponse
from asklineage.eval.truth import TruthGraph, walk
from asklineage.lineage.graph import Lineage, LineageNode
from asklineage.lineage.resolve import Candidate

HALLUCINATION_RULES = frozenset(
    {
        "unknown_node",
        "wrong_direction",
        "no_such_edge",
        "no_such_path",
        "text_contradicts_direction",
    }
)
PASSED = ("passed", "passed_after_retry")


@dataclass(frozen=True)
class Picks:
    nodes: tuple[str, ...]
    minority_hit: bool | None  # a node outside the majority owner's team; None with one team
    coverage: float  # owners covered / the most this many nodes could cover
    alphabetical_bias: float | None  # share that are the alphabetically first direct nodes
    indirect_reach: bool | None  # reaches past one hop; None when nothing named is deeper


@dataclass(frozen=True)
class HighlightScore:
    attempted: bool  # the first attempt reached the model
    first_rules: tuple[str, ...]  # rules the first attempt broke
    hallucinated_first: bool  # one of them was a false relation, not an output rule
    claims: int  # highlights in the answer
    manifest_false: int  # of those, how many the manifest contradicts
    llm: Picks | None
    code: Picks | None
    random: Picks | None


def named_nodes(response: LineageResponse, max_named: int) -> tuple[NamedNode, ...]:
    """The named nodes the highlight prompt showed, rebuilt from the answer's table."""
    root = response.node
    # The display name stands in for the name; only display names reach the named list.
    nodes = tuple(
        LineageNode(
            unique_id=row.unique_id,
            name=row.display_name,
            source_name=None,
            resource_type=row.resource_type,
            depth=row.depth,
            owner=row.owner,
            owner_source=row.owner_source,
        )
        for row in response.table
    )
    result = Lineage(root.unique_id, response.direction, response.depth, nodes, (), False)
    candidate = Candidate(root.unique_id, root.name, root.resource_type, root.source_name, 1.0)
    return build_facts(result, candidate, max_named).named


def picks(
    chosen: Sequence[NamedNode],
    named: Sequence[NamedNode],
    majority: str | None,
    says_indirect: bool = False,
) -> Picks:
    teams = {n.owner for n in named}
    covered = {n.owner for n in chosen} & teams
    possible = min(len(chosen), len(teams))
    first_direct = sorted(n.name for n in named if n.depth == 1)[: len(chosen)]
    deeper = any(n.depth > 1 for n in named)
    return Picks(
        nodes=tuple(n.name for n in chosen),
        minority_hit=any(n.owner != majority for n in chosen) if len(teams) > 1 else None,
        coverage=round(len(covered) / possible, 3) if possible else 0.0,
        alphabetical_bias=(
            round(len({n.name for n in chosen} & set(first_direct)) / len(chosen), 3)
            if first_direct and chosen
            else None
        ),
        indirect_reach=(says_indirect or any(n.depth > 1 for n in chosen)) if deeper else None,
    )


def _holds(graph: TruthGraph, upstream: str, downstream: str, hops: str) -> bool:
    if hops == "direct":
        return (upstream, downstream) in graph.edges
    return downstream in walk(graph, upstream, "downstream", len(graph.node_ids)).depths


def score_highlights(
    response: LineageResponse, graph: TruthGraph, max_named: int, seed: str
) -> HighlightScore | None:
    attempts = [a for a in response.attempts if a.stage == "highlights"]
    if not attempts:
        return None  # no LLM, or nothing to highlight
    first = attempts[0]

    ids = {row.display_name: row.unique_id for row in response.table}
    ids[response.node.display_name] = response.node.unique_id
    false = 0
    for claim in response.highlights:
        upstream, downstream = ids.get(claim.upstream), ids.get(claim.downstream)
        if upstream is None or downstream is None:
            false += 1
        elif not _holds(graph, upstream, downstream, claim.hops):
            false += 1

    llm = code = randomly = None
    if str(response.highlight_status) in PASSED and response.highlights:
        named = named_nodes(response, max_named)
        rows = {row.display_name: row for row in response.table}
        majority = Counter(row.owner for row in response.table).most_common(1)[0][0]
        mentioned = dict.fromkeys(
            name
            for claim in response.highlights
            for name in (claim.upstream, claim.downstream)
            if name in rows
        )
        chosen = [
            NamedNode(name, rows[name].resource_type, rows[name].depth, rows[name].owner)
            for name in mentioned
        ]
        indirect = any(claim.hops == "indirect" for claim in response.highlights)
        llm = picks(chosen, named, majority, indirect)
        size = min(len(chosen), len(named))
        code = picks(named[:size], named, majority)
        randomly = picks(random.Random(seed).sample(list(named), size), named, majority)

    return HighlightScore(
        attempted=first.error is None,
        first_rules=tuple(first.rules),
        hallucinated_first=any(rule in HALLUCINATION_RULES for rule in first.rules),
        claims=len(response.highlights),
        manifest_false=false,
        llm=llm,
        code=code,
        random=randomly,
    )
