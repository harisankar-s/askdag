"""Turn a dbt manifest.json into records for the lineage tables."""

from collections.abc import Iterable
from dataclasses import dataclass

from asklineage.ingest.owners import OwnerRule, match_owner

SUPPORTED_SCHEMA_VERSIONS = frozenset({"v12"})
NODE_RESOURCE_TYPES = frozenset({"model", "seed", "snapshot"})


class UnsupportedManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    unique_id: str
    resource_type: str
    name: str
    source_name: str | None
    package_name: str
    database_name: str | None
    schema_name: str | None
    relation_name: str | None
    materialization: str | None
    description: str
    file_path: str | None
    tags: tuple[str, ...]
    owner: str | None
    owner_source: str | None
    raw_code: str | None


@dataclass(frozen=True)
class Column:
    node_id: str
    name: str
    description: str
    data_type: str | None


@dataclass(frozen=True)
class DataTest:
    unique_id: str
    name: str
    test_type: str
    column_name: str | None
    attached_node_id: str | None
    severity: str | None


@dataclass(frozen=True)
class ParsedManifest:
    project_name: str
    dbt_version: str
    schema_version: str
    generated_at: str | None
    invocation_id: str | None
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    columns: tuple[Column, ...]
    tests: tuple[DataTest, ...]


def schema_version_of(manifest: dict) -> str:
    url = manifest.get("metadata", {}).get("dbt_schema_version", "")
    return url.rsplit("/", 1)[-1].removesuffix(".json")


def parse_manifest(manifest: dict, owner_rules: Iterable[OwnerRule] = ()) -> ParsedManifest:
    version = schema_version_of(manifest)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise UnsupportedManifestError(
            f"Manifest schema {version or 'unknown'} is not supported (supported: {supported})"
        )
    metadata = manifest["metadata"]
    if not metadata.get("project_name"):
        raise UnsupportedManifestError("Manifest metadata has no project_name")

    rules = tuple(owner_rules)
    group_owners = _group_owners(manifest.get("groups") or {})
    raw_nodes = [
        node for node in manifest["nodes"].values() if node["resource_type"] in NODE_RESOURCE_TYPES
    ]
    raw_nodes.extend((manifest.get("sources") or {}).values())

    nodes = tuple(_to_node(raw, group_owners, rules) for raw in raw_nodes)
    node_ids = {node.unique_id for node in nodes}

    return ParsedManifest(
        project_name=metadata["project_name"],
        dbt_version=metadata.get("dbt_version", ""),
        schema_version=version,
        generated_at=metadata.get("generated_at"),
        invocation_id=metadata.get("invocation_id"),
        nodes=nodes,
        edges=_edges(manifest.get("parent_map") or {}, node_ids),
        columns=tuple(column for raw in raw_nodes for column in _columns(raw)),
        tests=_tests(manifest["nodes"].values(), node_ids),
    )


def _to_node(raw: dict, group_owners: dict[str, str], rules: tuple[OwnerRule, ...]) -> Node:
    owner, owner_source = _resolve_owner(raw, group_owners, rules)
    return Node(
        unique_id=raw["unique_id"],
        resource_type=raw["resource_type"],
        name=raw["name"],
        source_name=raw.get("source_name"),
        package_name=raw["package_name"],
        database_name=raw.get("database"),
        schema_name=raw.get("schema"),
        relation_name=raw.get("relation_name"),
        materialization=(raw.get("config") or {}).get("materialized"),
        description=(raw.get("description") or "").strip(),
        file_path=raw.get("original_file_path"),
        tags=tuple(raw.get("tags") or ()),
        owner=owner,
        owner_source=owner_source,
        raw_code=raw.get("raw_code") or None,
    )


def _resolve_owner(
    raw: dict, group_owners: dict[str, str], rules: tuple[OwnerRule, ...]
) -> tuple[str | None, str | None]:
    config = raw.get("config") or {}
    meta = {**(raw.get("meta") or {}), **(config.get("meta") or {})}
    if meta.get("owner"):
        return str(meta["owner"]), "dbt_meta"

    group = raw.get("group") or config.get("group")
    if group:
        return group_owners.get(group, group), "dbt_group"

    owner = match_owner(rules, raw.get("original_file_path") or "")
    return (owner, "owners_file") if owner else (None, None)


def _group_owners(groups: dict) -> dict[str, str]:
    owners = {}
    for group in groups.values():
        owner = group.get("owner") or {}
        owners[group["name"]] = owner.get("name") or owner.get("email") or group["name"]
    return owners


def _edges(parent_map: dict[str, list[str]], node_ids: set[str]) -> tuple[tuple[str, str], ...]:
    edges = {
        (parent, child)
        for child, parents in parent_map.items()
        if child in node_ids
        for parent in parents
        if parent in node_ids and parent != child
    }
    return tuple(sorted(edges))


def _columns(raw: dict) -> list[Column]:
    return [
        Column(
            node_id=raw["unique_id"],
            name=column["name"],
            description=(column.get("description") or "").strip(),
            data_type=column.get("data_type"),
        )
        for column in (raw.get("columns") or {}).values()
    ]


def _tests(raw_nodes: Iterable[dict], node_ids: set[str]) -> tuple[DataTest, ...]:
    tests = []
    for raw in raw_nodes:
        if raw["resource_type"] != "test":
            continue
        attached = raw.get("attached_node")
        tests.append(
            DataTest(
                unique_id=raw["unique_id"],
                name=raw["name"],
                test_type=(raw.get("test_metadata") or {}).get("name") or "singular",
                column_name=raw.get("column_name"),
                attached_node_id=attached if attached in node_ids else None,
                severity=(raw.get("config") or {}).get("severity"),
            )
        )
    return tuple(tests)
