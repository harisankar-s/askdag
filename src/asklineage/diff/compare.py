"""Compare two parsed manifests of one project: what changed, and what sits downstream of it."""

from collections import defaultdict, deque
from dataclasses import dataclass

from asklineage.ingest.manifest import Column, Node, ParsedManifest

# A change to one of these alters the rows a node produces or where they land.
DATA_FIELDS = ("raw_code", "materialization", "relation_name", "parents")
# These change only what the docs say. Columns are the documented ones (no catalog.json).
METADATA_FIELDS = ("description", "columns", "tags", "owner", "file_path")


class ProjectMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class NodeRef:
    unique_id: str
    name: str
    resource_type: str
    owner: str | None
    owner_source: str | None


@dataclass(frozen=True)
class NodeChange:
    node: NodeRef
    fields: tuple[str, ...]

    @property
    def affects_data(self) -> bool:
        return any(field in DATA_FIELDS for field in self.fields)


@dataclass(frozen=True)
class Downstream:
    node: NodeRef
    hops: int
    via: str


@dataclass(frozen=True)
class ManifestDiff:
    project_name: str
    added: tuple[NodeRef, ...]
    removed: tuple[NodeRef, ...]
    changed: tuple[NodeChange, ...]
    added_edges: tuple[tuple[str, str], ...]
    removed_edges: tuple[tuple[str, str], ...]
    downstream: tuple[Downstream, ...]
    possible_renames: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def diff_manifests(old: ParsedManifest, new: ParsedManifest) -> ManifestDiff:
    if old.project_name != new.project_name:
        raise ProjectMismatchError(
            f"Cannot diff different projects: {old.project_name} and {new.project_name}"
        )
    old_nodes = {node.unique_id: node for node in old.nodes}
    new_nodes = {node.unique_id: node for node in new.nodes}
    old_snapshots = _snapshots(old)
    new_snapshots = _snapshots(new)

    added_ids = sorted(new_nodes.keys() - old_nodes.keys())
    removed_ids = sorted(old_nodes.keys() - new_nodes.keys())
    changed = []
    for unique_id in sorted(old_nodes.keys() & new_nodes.keys()):
        before, after = old_snapshots[unique_id], new_snapshots[unique_id]
        fields = tuple(
            field for field in DATA_FIELDS + METADATA_FIELDS if before[field] != after[field]
        )
        if fields:
            changed.append(NodeChange(_ref(new_nodes[unique_id]), fields))

    old_edges, new_edges = set(old.edges), set(new.edges)
    sources = [change.node.unique_id for change in changed if change.affects_data]
    return ManifestDiff(
        project_name=new.project_name,
        added=tuple(_ref(new_nodes[unique_id]) for unique_id in added_ids),
        removed=tuple(_ref(old_nodes[unique_id]) for unique_id in removed_ids),
        changed=tuple(changed),
        added_edges=tuple(sorted(new_edges - old_edges)),
        removed_edges=tuple(sorted(old_edges - new_edges)),
        downstream=_downstream(new, sources, skip=set(sources) | set(added_ids)),
        possible_renames=_possible_renames(
            [old_nodes[unique_id] for unique_id in removed_ids],
            [new_nodes[unique_id] for unique_id in added_ids],
        ),
    )


def _ref(node: Node) -> NodeRef:
    name = f"{node.source_name}.{node.name}" if node.source_name else node.name
    return NodeRef(node.unique_id, name, node.resource_type, node.owner, node.owner_source)


def _snapshots(parsed: ParsedManifest) -> dict[str, dict[str, object]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in parsed.edges:
        parents[child].add(parent)
    columns: dict[str, list[Column]] = defaultdict(list)
    for column in parsed.columns:
        columns[column.node_id].append(column)

    return {
        node.unique_id: {
            "raw_code": node.raw_code,
            "materialization": node.materialization,
            "relation_name": node.relation_name,
            "parents": frozenset(parents[node.unique_id]),
            "description": node.description,
            "columns": frozenset(
                (column.name, column.description, column.data_type)
                for column in columns[node.unique_id]
            ),
            "tags": frozenset(node.tags),
            "owner": (node.owner, node.owner_source),
            "file_path": node.file_path,
        }
        for node in parsed.nodes
    }


def _downstream(
    parsed: ParsedManifest, sources: list[str], skip: set[str]
) -> tuple[Downstream, ...]:
    """Multi-source BFS over the new graph, unbounded: a pre-merge check should see everything."""
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in parsed.edges:
        children[parent].append(child)
    nodes = {node.unique_id: node for node in parsed.nodes}

    reached: dict[str, tuple[int, str]] = {}
    queue = deque((source, 0, source) for source in sorted(sources))
    seen = set(sources)
    while queue:
        current, hops, via = queue.popleft()
        for child in sorted(children[current]):
            if child in seen:
                continue
            seen.add(child)
            reached[child] = (hops + 1, via)
            queue.append((child, hops + 1, via))

    return tuple(
        Downstream(_ref(nodes[unique_id]), hops, via)
        for unique_id, (hops, via) in sorted(
            reached.items(), key=lambda item: (item[1][0], item[0])
        )
        if unique_id not in skip
    )


def _possible_renames(removed: list[Node], added: list[Node]) -> tuple[tuple[str, str], ...]:
    """dbt has no rename identity; identical SQL on a removed and an added node is a hint."""
    gone = _unique_by_code(removed)
    new = _unique_by_code(added)
    return tuple(sorted((gone[code], new[code]) for code in gone.keys() & new.keys()))


def _unique_by_code(nodes: list[Node]) -> dict[str, str]:
    by_code: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        if node.raw_code:
            by_code[node.raw_code].append(node.unique_id)
    return {code: ids[0] for code, ids in by_code.items() if len(ids) == 1}
