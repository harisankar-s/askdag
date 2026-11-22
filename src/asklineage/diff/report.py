"""Render a manifest diff as text for people or JSON for tools."""

from collections import Counter
from dataclasses import asdict

from asklineage.diff.compare import ManifestDiff, NodeRef

FIELD_LABELS = {"raw_code": "sql", "relation_name": "relation"}
UNOWNED = "no owner recorded"


def diff_json(diff: ManifestDiff) -> dict:
    data = asdict(diff)
    for change, raw in zip(diff.changed, data["changed"], strict=True):
        raw["affects_data"] = change.affects_data
    return data


def render_diff(diff: ManifestDiff, limit: int = 50) -> str:
    if diff.is_empty:
        return f"{diff.project_name}: no changes"
    names = {ref.unique_id: ref.name for ref in _refs(diff)}
    data_changes = sum(1 for change in diff.changed if change.affects_data)
    lines = [
        f"{diff.project_name}: {len(diff.added)} added, {len(diff.removed)} removed, "
        f"{len(diff.changed)} changed ({data_changes} data, "
        f"{len(diff.changed) - data_changes} docs only), "
        f"edges +{len(diff.added_edges)} -{len(diff.removed_edges)}"
    ]

    width = _width(diff.added, limit)
    lines += _section("Added", [_row(ref, width) for ref in diff.added], limit)
    width = _width(diff.removed, limit)
    lines += _section("Removed", [_row(ref, width) for ref in diff.removed], limit)
    width = _width([change.node for change in diff.changed], limit)
    lines += _section(
        "Changed",
        [
            _row(change.node, width, ", ".join(FIELD_LABELS.get(f, f) for f in change.fields))
            + ("" if change.affects_data else "  (docs only)")
            for change in diff.changed
        ],
        limit,
    )
    lines += _section(
        "Possible renames (same SQL)",
        [f"{names[old]} -> {names[new]}" for old, new in diff.possible_renames],
        limit,
    )

    if any(change.affects_data for change in diff.changed):
        owners = Counter(ref.node.owner or UNOWNED for ref in diff.downstream)
        teams = ", ".join(f"{owner} {count}" for owner, count in owners.most_common())
        title = f"Downstream of data changes: {len(diff.downstream)} node(s)"
        shown = diff.downstream[:limit]
        width = _width([item.node for item in shown], limit)
        owner_width = max((len(_owner(item.node)) for item in shown), default=0) + 2
        lines += _section(
            title + (f"; owners: {teams}" if teams else ""),
            [
                f"{item.hops} hop{'s' if item.hops > 1 else ' '}  "
                + _row(item.node, width, f"{_owner(item.node):<{owner_width}}via {names[item.via]}")
                for item in diff.downstream
            ],
            limit,
            always=True,
        )
    return "\n".join(lines)


def _refs(diff: ManifestDiff) -> list[NodeRef]:
    return [
        *diff.added,
        *diff.removed,
        *(change.node for change in diff.changed),
        *(item.node for item in diff.downstream),
    ]


def _width(refs: list[NodeRef] | tuple[NodeRef, ...], limit: int) -> int:
    return max((len(ref.name) for ref in refs[:limit]), default=0) + 2


def _row(ref: NodeRef, width: int, detail: str = "") -> str:
    return f"{ref.resource_type:<9}{ref.name:<{width}}{detail}".rstrip()


def _owner(ref: NodeRef) -> str:
    return f"{ref.owner} ({ref.owner_source})" if ref.owner else UNOWNED


def _section(title: str, rows: list[str], limit: int, always: bool = False) -> list[str]:
    if not rows and not always:
        return []
    shown = rows[:limit]
    more = [f"  ... and {len(rows) - limit} more"] if len(rows) > limit else []
    return ["", title, *(f"  {row}" for row in shown), *more]
