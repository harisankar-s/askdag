"""Check each highlight against the lineage result it describes. No LLM, no database."""

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator

from asklineage.answer.claims import Highlights, RelationClaim
from asklineage.answer.facts import LineageFacts
from asklineage.answer.generate import Violation
from asklineage.lineage.graph import Lineage

BACKTICKED = re.compile(r"`([^`]+)`")
NUMBER = re.compile(r"\b\d+\b")
NAME_LIKE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
# Owners are added to highlights by code, so any owner wording from the model is rejected.
# Not "own": "reads its own staging model" says nothing about owners.
OWNER_WORDS = re.compile(r"\b(?:owns|owned|owner|owners|ownership|team|teams)\b", re.IGNORECASE)

# Phrases that put the first-named node upstream ("`a` feeds `b`") or downstream ("`a` reads
# from `b`"). A sentence with neither, or both, is not checked for wording.
UPSTREAM_FIRST = ("feeds", "flows into", "is read by", "is used by", "is consumed by", "supplies")
DOWNSTREAM_FIRST = (
    "reads",
    "depends on",
    "is built from",
    "comes from",
    "uses",
    "pulls from",
    "is fed by",
    "consumes",
)

type Finding = tuple[str, str]


class _Graph:
    def __init__(self, result: Lineage, root_name: str) -> None:
        self.ids: dict[str, set[str]] = defaultdict(set)
        self.ids[root_name].add(result.root_id)
        for node in result.nodes:
            self.ids[node.display_name].add(node.unique_id)
        self.children: dict[str, set[str]] = defaultdict(set)
        for parent, child in result.edges:
            self.children[parent].add(child)

    @property
    def names(self) -> set[str]:
        return set(self.ids)

    def has_edge(self, upstream: str, downstream: str) -> bool:
        targets = self.ids[downstream]
        return any(self.children[u] & targets for u in self.ids[upstream])

    def reachable(self, upstream: str, downstream: str) -> bool:
        targets = self.ids[downstream]
        queue, seen = deque(self.ids[upstream]), set(self.ids[upstream])
        while queue:
            for child in self.children[queue.popleft()] - seen:
                if child in targets:
                    return True
                seen.add(child)
                queue.append(child)
        return False


def check(
    highlights: Highlights,
    facts: LineageFacts,
    result: Lineage,
    known_names: Iterable[str] = (),
) -> tuple[Violation, ...]:
    """known_names: every node name in the project, to catch real nodes named without backticks."""
    graph = _Graph(result, facts.root)
    node_names = graph.names | set(known_names)
    return tuple(
        Violation(index, rule, detail)
        for index, claim in enumerate(highlights.claims, start=1)
        for rule, detail in (
            *_check_relation(claim, graph),
            *_check_text(claim, facts, graph, node_names),
        )
    )


def _check_relation(claim: RelationClaim, graph: _Graph) -> Iterator[Finding]:
    up, down = claim.upstream, claim.downstream
    unknown = [name for name in (up, down) if name not in graph.ids]
    if unknown:
        yield "unknown_node", f"{', '.join(f'`{n}`' for n in unknown)} not in this lineage result"
        return
    holds = graph.has_edge if claim.hops == "direct" else graph.reachable
    if holds(up, down):
        return
    if holds(down, up):
        yield "wrong_direction", f"`{down}` feeds `{up}`, not the other way round"
    elif claim.hops == "direct":
        yield "no_such_edge", f"`{up}` does not feed `{down}` directly"
    else:
        yield "no_such_path", f"there is no path from `{up}` to `{down}` in this result"


def _check_text(
    claim: RelationClaim, facts: LineageFacts, graph: _Graph, node_names: set[str]
) -> Iterator[Finding]:
    claim_names = {claim.upstream, claim.downstream}
    teams = {group.owner for group in facts.owners}
    mentioned = BACKTICKED.findall(claim.text)
    for name in mentioned:
        if name in teams:
            yield "owner_mentioned", f"`{name}` is a team; owners are added to highlights for you"
            continue
        if name in claim_names or name == facts.root:
            continue
        if name in graph.ids:
            yield "name_not_in_claim", f"`{name}` is not in this claim's fields"
        else:
            yield "unknown_node", f"`{name}` is not in this lineage result"

    bare = BACKTICKED.sub(" ", claim.text)
    for token in NAME_LIKE.findall(bare):
        # Underscore or dot only: a model called `orders` must not flag the English word "orders".
        if ("_" in token or "." in token) and token in node_names and token not in teams:
            yield "unmarked_name", f"node name {token} must be written in backticks"

    for number in NUMBER.findall(bare):
        yield "unsupported_number", f"highlights state no numbers, but this one says {number}"

    found = {match.group(0).lower() for match in OWNER_WORDS.finditer(bare)}
    found |= {team for team in teams if re.search(rf"(?<![\w-]){re.escape(team)}(?![\w-])", bare)}
    if found:
        words = ", ".join(sorted(found))
        yield "owner_mentioned", f"don't mention owners ({words}); they are added for you"
    if mentioned:
        yield from _check_wording(claim, mentioned[0], bare.lower())


def _check_wording(claim: RelationClaim, first: str, text: str) -> Iterator[Finding]:
    up_first = any(re.search(rf"\b{phrase}\b", text) for phrase in UPSTREAM_FIRST)
    down_first = any(re.search(rf"\b{phrase}\b", text) for phrase in DOWNSTREAM_FIRST)
    if up_first == down_first or first not in (claim.upstream, claim.downstream):
        return
    expected = claim.upstream if up_first else claim.downstream
    if first != expected:
        role = "upstream" if up_first else "downstream"
        yield "text_contradicts_direction", f"`{first}` reads as {role}, against the fields"
