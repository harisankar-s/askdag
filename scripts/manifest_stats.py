"""Print what a dbt manifest.json contains and how well it is documented."""

import argparse
import json
from collections import Counter
from pathlib import Path


def print_ratio(label: str, part: int, whole: int) -> None:
    share = f"{part / whole:.0%}" if whole else "n/a"
    print(f"  {label:<28} {part:>6} / {whole:<6} ({share})")


def owner_of(node: dict) -> str | None:
    meta = {**node.get("meta", {}), **node.get("config", {}).get("meta", {})}
    return meta.get("owner") or node.get("group")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    manifest = json.loads(parser.parse_args().manifest.read_text())

    metadata = manifest["metadata"]
    print(f"dbt version:     {metadata.get('dbt_version')}")
    print(f"schema version:  {metadata.get('dbt_schema_version')}")
    print(f"project:         {metadata.get('project_name')}")

    nodes = manifest["nodes"].values()
    by_type = Counter(node["resource_type"] for node in nodes)
    by_type["source"] = len(manifest["sources"])
    print("\nResources:")
    for resource_type, count in by_type.most_common():
        print(f"  {resource_type:<12} {count:>6}")

    models = [node for node in nodes if node["resource_type"] == "model"]
    described = sum(1 for model in models if model.get("description", "").strip())
    owned = sum(1 for model in models if owner_of(model))
    columns = [column for model in models for column in model.get("columns", {}).values()]
    columns_described = sum(1 for column in columns if column.get("description", "").strip())

    print("\nModel documentation:")
    print_ratio("models with a description", described, len(models))
    print_ratio("models with an owner", owned, len(models))
    print_ratio("documented columns", columns_described, len(columns))

    parent_map = manifest["parent_map"]
    edges = sum(len(parent_map.get(model["unique_id"], [])) for model in models)
    print(f"\nLineage edges into models: {edges}")


if __name__ == "__main__":
    main()
