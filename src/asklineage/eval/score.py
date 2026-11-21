"""Score one golden turn against the response the flow gave. Pure: no database, no LLM.

Outcomes, worst first:
- silent_wrong: a lineage answer about the wrong node, direction or depth, or with a table,
  edges or facts that differ from the manifest. The user is told something false and isn't
  warned.
- safe_wrong: the wrong clarification or abstention. The user gets no answer, but no false one.
- acceptable: another outcome the golden set lists as safe for this turn.
- correct.
- blocked: a choose turn whose thread was not waiting for a choice; not scored.
- error: the flow raised; always a bug.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from asklineage.api.models import AskResponse, LineageResponse
from asklineage.eval.golden import Expect, GoldenItem, NameIndex, Turn
from asklineage.eval.highlights import HighlightScore, score_highlights
from asklineage.eval.truth import TruthGraph, walk

Outcome = Literal["correct", "acceptable", "safe_wrong", "silent_wrong", "blocked", "error"]
LLM_OFF = Expect("abstain", "llm_off")


@dataclass(frozen=True)
class TurnScore:
    item_id: str
    turn: int
    role: str
    tags: tuple[str, ...]
    request: dict
    free_text: bool
    same_node: str | None
    expected: Expect
    outcome: Outcome
    problems: tuple[str, ...] = ()
    kind: str | None = None
    reason: str | None = None
    node_id: str | None = None
    direction: str | None = None
    depth: int | None = None
    node_ok: bool | None = None
    direction_ok: bool | None = None
    depth_ok: bool | None = None
    lineage_ok: bool | None = None  # the table, edges and truncated flag match the manifest
    facts_ok: bool | None = None  # the summary's counts match the manifest
    candidates_ok: bool | None = None  # ambiguous: exactly the expected candidates
    suggestion_rank: int | None = None  # 1-based; 0 when the expected node wasn't suggested
    reused_previous: bool = False  # a follow-up naming a new node was answered about the old one
    highlight_status: str | None = None
    highlights: HighlightScore | None = None
    attempts: tuple[dict, ...] = ()
    seconds: float = 0.0
    response: dict | None = None

    @property
    def group(self) -> str:
        if "thread" in self.tags:
            return "thread"
        return "free_text" if self.free_text else "structured"


def expected_for(turn: Turn, llm: bool) -> Expect:
    """Without an LLM, every free-text question must abstain with llm_off."""
    return turn.expect if llm or not turn.free_text else LLM_OFF


def check_lineage(
    response: LineageResponse,
    graph: TruthGraph,
    index: NameIndex,
    owners: Mapping[str, str | None],
) -> tuple[bool, bool, list[str]]:
    """Check the answer's own walk (whatever node it chose) against the manifest."""
    root = response.node.unique_id
    try:
        truth = walk(graph, root, str(response.direction), response.depth)
    except LookupError:
        return False, False, [f"{root} is not in the manifest"]
    problems = []

    got = {row.unique_id: row.depth for row in response.table}
    table_ok = (
        got == truth.depths
        and len(response.table) == len(got)
        and response.facts.truncated == truth.truncated
    )
    if not table_ok:
        missing, extra = truth.depths.keys() - got.keys(), got.keys() - truth.depths.keys()
        problems.append(
            f"table differs from the manifest: {len(missing)} missing, {len(extra)} extra, "
            f"truncated {response.facts.truncated} (expected {truth.truncated})"
        )

    edges = {(edge.parent_id, edge.child_id) for edge in response.edges}
    edges_ok = edges == truth.edges and len(response.edges) == len(edges)
    if not edges_ok:
        missing, extra = truth.edges - edges, edges - truth.edges
        problems.append(
            f"edges differ from the manifest: {len(missing)} missing, {len(extra)} extra, "
            f"{len(response.edges) - len(edges)} duplicated"
        )
    lineage_ok = table_ok and edges_ok

    ids = truth.depths
    owner_counts: Counter[str] = Counter()
    for group in response.facts.owners:
        owner_counts[group.owner] += group.nodes
    expected_facts = {
        "total": len(ids),
        "by_depth": dict(Counter(ids.values())),
        "by_type": dict(Counter(index.names[uid].resource_type for uid in ids)),
        "owners": dict(Counter(owners[uid] for uid in ids if owners.get(uid))),
        "unowned": sum(1 for uid in ids if not owners.get(uid)),
    }
    got_facts = {
        "total": response.facts.total,
        "by_depth": response.facts.by_depth,
        "by_type": response.facts.by_type,
        "owners": dict(owner_counts),
        "unowned": response.facts.unowned,
    }
    wrong = [key for key, value in expected_facts.items() if got_facts[key] != value]
    if wrong:
        problems.append(f"facts differ from the manifest: {', '.join(wrong)}")
    return lineage_ok, not wrong, problems


def score_turn(
    item: GoldenItem,
    number: int,
    turn: Turn,
    response: AskResponse | None,
    *,
    llm: bool,
    graph: TruthGraph,
    index: NameIndex,
    owners: Mapping[str, str | None],
    previous_node: str | None = None,
    max_named: int = 20,
    seconds: float = 0.0,
    error: str | None = None,
    blocked: bool = False,
) -> TurnScore:
    expected = expected_for(turn, llm)
    base = {
        "item_id": item.id,
        "turn": number,
        "role": item.role,
        "tags": item.tags,
        "request": dict(turn.request),
        "free_text": turn.free_text,
        "same_node": turn.same_node,
        "expected": expected,
        "seconds": round(seconds, 3),
    }
    if blocked:
        return TurnScore(**base, outcome="blocked", problems=("the thread was not waiting",))
    if response is None:
        return TurnScore(**base, outcome="error", problems=(error or "no response",))

    kind = response.kind
    reason = getattr(response, "reason", None)
    problems: list[str] = []
    fields: dict = {}

    if kind != expected.kind:
        if kind in expected.also:
            outcome: Outcome = "acceptable"
        elif kind == "lineage":
            outcome = "silent_wrong"
            problems.append(f"answered with lineage; expected {expected.kind}")
        else:
            outcome = "safe_wrong"
            problems.append(f"expected {expected.kind}, got {kind} ({reason})")
    elif kind == "lineage":
        fields = {
            "node_ok": response.node.unique_id == expected.node_id,
            "direction_ok": str(response.direction) == expected.direction,
            "depth_ok": response.depth == expected.depth,
        }
        problems += [f"wrong {name.removesuffix('_ok')}" for name, ok in fields.items() if not ok]
        outcome = "correct" if all(fields.values()) else "silent_wrong"
    elif expected.reason and reason != expected.reason:
        outcome = "safe_wrong"
        problems.append(f"expected {kind} {expected.reason}, got {reason}")
    elif expected.reason == "ambiguous":
        got = tuple(sorted(c.unique_id for c in response.candidates))
        fields["candidates_ok"] = got == expected.candidates
        outcome = "correct" if fields["candidates_ok"] else "safe_wrong"
        if not fields["candidates_ok"]:
            problems.append(f"candidates {list(got)}, expected {list(expected.candidates)}")
    else:
        outcome = "correct"

    if kind == "clarification" and expected.suggest:
        suggested = [c.unique_id for c in response.candidates]
        rank = suggested.index(expected.suggest) + 1 if expected.suggest in suggested else 0
        fields["suggestion_rank"] = rank

    if kind == "lineage":
        node_id = response.node.unique_id
        lineage_ok, facts_ok, lineage_problems = check_lineage(response, graph, index, owners)
        problems += lineage_problems
        if not (lineage_ok and facts_ok):
            outcome = "silent_wrong"
        reused = (
            turn.same_node == "no"
            and node_id == previous_node
            and expected.node_id != previous_node
        )
        if reused:
            problems.append("reused the previous node instead of the one named")
        fields |= {
            "node_id": node_id,
            "direction": str(response.direction),
            "depth": response.depth,
            "lineage_ok": lineage_ok,
            "facts_ok": facts_ok,
            "reused_previous": reused,
            "highlight_status": str(response.highlight_status),
            "highlights": score_highlights(response, graph, max_named, f"{item.id}:{number}"),
        }

    return TurnScore(
        **base,
        outcome=outcome,
        problems=tuple(problems),
        kind=kind,
        reason=reason,
        attempts=tuple(a.model_dump(mode="json") for a in response.attempts),
        response=response.model_dump(mode="json"),
        **fields,
    )
