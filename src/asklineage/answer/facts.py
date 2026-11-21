"""What a lineage answer states: counts over every node, plus a capped list of names."""

import json
from collections import Counter
from dataclasses import asdict, dataclass

from asklineage.lineage.graph import Direction, Lineage, LineageNode
from asklineage.lineage.resolve import Candidate

DIRECTION_MEANING = {
    Direction.DOWNSTREAM: "these nodes read from the root, directly or through other nodes",
    Direction.UPSTREAM: "the root reads from these nodes, directly or through other nodes",
}

OWNER_SOURCE_MEANING = {
    "dbt_meta": "set in the model's dbt meta.owner",
    "dbt_group": "from the owner of the model's dbt group",
    "owners_file": "assigned by path rules in the owners file, not recorded in dbt",
}


@dataclass(frozen=True)
class NamedNode:
    name: str
    resource_type: str
    depth: int
    owner: str | None


@dataclass(frozen=True)
class OwnerGroup:
    owner: str
    owner_source: str
    owner_source_meaning: str
    nodes: int


@dataclass(frozen=True)
class LineageFacts:
    root: str
    root_type: str
    direction: Direction
    direction_meaning: str
    max_depth: int
    truncated: bool
    total: int
    by_depth: dict[int, int]
    by_type: dict[str, int]
    named: tuple[NamedNode, ...]
    unnamed: int
    owners: tuple[OwnerGroup, ...]
    unowned: int

    def prompt_json(self) -> str:
        """Only what the highlights may use: no counts, so the model has none to restate."""
        keys = ("root", "root_type", "direction", "direction_meaning", "named")
        data = asdict(self)
        return json.dumps({key: data[key] for key in keys}, indent=2)


def build_facts(result: Lineage, root: Candidate, max_named: int) -> LineageFacts:
    if root.unique_id != result.root_id:
        raise ValueError(f"Root {root.unique_id!r} is not the root of this result")
    if max_named < 1:
        raise ValueError(f"max_named must be at least 1, got {max_named}")

    nodes = result.nodes
    owned = Counter((n.owner, n.owner_source or "unknown") for n in nodes if n.owner)
    groups = sorted(owned.items(), key=lambda item: (-item[1], item[0]))
    types = Counter(n.resource_type for n in nodes)
    named = _pick_named(nodes, max_named)

    return LineageFacts(
        root=root.display_name,
        root_type=root.resource_type,
        direction=result.direction,
        direction_meaning=DIRECTION_MEANING[result.direction],
        max_depth=result.max_depth,
        truncated=result.truncated,
        total=len(nodes),
        by_depth=dict(sorted(Counter(n.depth for n in nodes).items())),
        by_type=dict(sorted(types.items(), key=lambda item: (-item[1], item[0]))),
        named=tuple(NamedNode(n.display_name, n.resource_type, n.depth, n.owner) for n in named),
        unnamed=len(nodes) - len(named),
        owners=tuple(
            OwnerGroup(owner, source, OWNER_SOURCE_MEANING.get(source, source), count)
            for (owner, source), count in groups
        ),
        unowned=sum(1 for n in nodes if not n.owner),
    )


def _pick_named(nodes: tuple[LineageNode, ...], limit: int) -> list[LineageNode]:
    # Round-robin across owners, nearest first within each, so every team (and the unowned
    # nodes) gets named even when one team owns almost everything.
    by_owner: dict[str | None, list[LineageNode]] = {}
    for node in nodes:  # already ordered by depth, then name
        by_owner.setdefault(node.owner, []).append(node)
    queues = sorted(by_owner.items(), key=lambda item: (-len(item[1]), item[0] is None, item[0]))
    picked: list[LineageNode] = []
    for position in range(max((len(q) for _, q in queues), default=0)):
        for _, queue in queues:
            if position < len(queue):
                picked.append(queue[position])
                if len(picked) == limit:
                    return picked
    return picked
