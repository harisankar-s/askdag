"""Lineage ground truth, computed from manifest.json with the standard library only.

Independent of the app on purpose: it reads depends_on (the parser reads parent_map) and walks
in Python (the app walks in SQL), so a bug in either shows up as a difference. Which nodes
count, and that self-references are dropped, is the spec; it is written here a second time.
"""

from collections import defaultdict
from dataclasses import dataclass

NODE_TYPES = frozenset({"model", "seed", "snapshot"})
DIRECTIONS = ("downstream", "upstream")


@dataclass(frozen=True)
class TruthGraph:
    project_name: str
    invocation_id: str | None
    node_ids: frozenset[str]
    edges: frozenset[tuple[str, str]]  # (parent, child)
    children: dict[str, frozenset[str]]
    parents: dict[str, frozenset[str]]


@dataclass(frozen=True)
class TruthWalk:
    root: str
    direction: str
    max_depth: int
    depths: dict[str, int]  # every reached node except the root, at its fewest hops
    edges: frozenset[tuple[str, str]]  # every edge between two of the root and reached nodes
    truncated: bool  # a node at max_depth has a neighbour that was not reached


def truth_graph(manifest: dict) -> TruthGraph:
    nodes = {
        unique_id: node
        for unique_id, node in manifest["nodes"].items()
        if node["resource_type"] in NODE_TYPES
    }
    node_ids = frozenset(nodes) | frozenset(manifest.get("sources") or {})
    edges = frozenset(
        (parent, child)
        for child, node in nodes.items()
        for parent in (node.get("depends_on") or {}).get("nodes") or ()
        if parent in node_ids and parent != child
    )
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        children[parent].add(child)
        parents[child].add(parent)
    metadata = manifest["metadata"]
    return TruthGraph(
        project_name=metadata["project_name"],
        invocation_id=metadata.get("invocation_id"),
        node_ids=node_ids,
        edges=edges,
        children={node: frozenset(ids) for node, ids in children.items()},
        parents={node: frozenset(ids) for node, ids in parents.items()},
    )


def walk(graph: TruthGraph, root: str, direction: str, max_depth: int) -> TruthWalk:
    if root not in graph.node_ids:
        raise LookupError(f"No node with unique_id {root!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, got {max_depth}")
    step = graph.children if direction == "downstream" else graph.parents

    depths: dict[str, int] = {}
    seen = {root}
    frontier = [root]
    for hops in range(1, max_depth + 1):
        if not frontier:
            break
        reached = []
        for node in frontier:
            for neighbour in step.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    depths[neighbour] = hops
                    reached.append(neighbour)
        frontier = reached

    truncated = any(n not in seen for node in frontier for n in step.get(node, ()))
    edges = frozenset(
        (parent, child)
        for parent in seen
        for child in graph.children.get(parent, ())
        if child in seen
    )
    return TruthWalk(root, direction, max_depth, depths, edges, truncated)
