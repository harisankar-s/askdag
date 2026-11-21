"""The summary paragraph, written in code from the facts. The LLM never restates these facts."""

from collections import Counter

from asklineage.answer.facts import LineageFacts
from asklineage.lineage.graph import Direction

MAX_TEAMS_LISTED = 8


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def template_summary(facts: LineageFacts) -> str:
    root = f"`{facts.root}`"
    downstream = facts.direction is Direction.DOWNSTREAM
    if facts.total == 0:
        return f"Nothing reads from {root}." if downstream else f"{root} reads from no other node."

    nodes = _count(facts.total, "node")
    within = f"within {facts.max_depth} hops"
    if downstream:
        verb = "depends" if facts.total == 1 else "depend"
        opening = f"{nodes} {verb} on {root} {within}"
    else:
        opening = f"{root} depends on {nodes} {within}"
    hops = ", ".join(
        f"{n} directly" if depth == 1 else f"{n} at {depth} hops"
        for depth, n in facts.by_depth.items()
    )
    sentences = [f"{opening} ({hops})."]

    if facts.owners:
        teams = Counter()
        for group in facts.owners:
            teams[group.owner] += group.nodes
        ranked = sorted(teams.items(), key=lambda item: (-item[1], item[0]))
        listed = ", ".join(f"{team} ({_count(n, 'node')})" for team, n in ranked[:MAX_TEAMS_LISTED])
        hidden = len(ranked) - MAX_TEAMS_LISTED
        more = f", and {_count(hidden, 'more team')}" if hidden > 0 else ""
        label = "Teams affected" if downstream else "Owned by"
        sentences.append(f"{label}: {listed}{more}.")

        sources = Counter()
        for group in facts.owners:
            sources[group.owner_source_meaning] += group.nodes
        if len(sources) == 1:
            sentences.append(f"Owners are {next(iter(sources))}.")
        else:
            parts = "; ".join(f"{meaning} ({_count(n, 'node')})" for meaning, n in sources.items())
            sentences.append(f"Owners come from: {parts}.")

    if facts.unowned:
        verb = "has" if facts.unowned == 1 else "have"
        sentences.append(f"{_count(facts.unowned, 'node')} {verb} no owner recorded.")
    if facts.truncated:
        sentences.append(
            f"More nodes lie beyond {facts.max_depth} hops; raise the depth to see them."
        )
    return " ".join(sentences)
