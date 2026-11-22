"""What a lineage answer states: counts over every node, by depth, type and owner."""

from collections import Counter
from dataclasses import dataclass

from asklineage.lineage.graph import Direction, Lineage
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
    owners: tuple[OwnerGroup, ...]
    unowned: int


def build_facts(result: Lineage, root: Candidate) -> LineageFacts:
    if root.unique_id != result.root_id:
        raise ValueError(f"Root {root.unique_id!r} is not the root of this result")

    nodes = result.nodes
    owned = Counter((n.owner, n.owner_source or "unknown") for n in nodes if n.owner)
    groups = sorted(owned.items(), key=lambda item: (-item[1], item[0]))
    types = Counter(n.resource_type for n in nodes)

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
        owners=tuple(
            OwnerGroup(owner, source, OWNER_SOURCE_MEANING.get(source, source), count)
            for (owner, source), count in groups
        ),
        unowned=sum(1 for n in nodes if not n.owner),
    )
