from asklineage.ingest.manifest import Node

TEST_PROJECT = "lineage_test"


def make_node(
    name: str,
    resource_type: str = "model",
    source_name: str | None = None,
    owner: str | None = None,
) -> Node:
    if resource_type == "source":
        unique_id = f"source.{TEST_PROJECT}.{source_name}.{name}"
    else:
        unique_id = f"{resource_type}.{TEST_PROJECT}.{name}"
    return Node(
        unique_id=unique_id,
        resource_type=resource_type,
        name=name,
        source_name=source_name,
        package_name=TEST_PROJECT,
        database_name=None,
        schema_name=None,
        relation_name=None,
        materialization=None,
        description="",
        file_path=None,
        tags=(),
        owner=owner,
        owner_source="owners_file" if owner else None,
        raw_code=None,
    )
