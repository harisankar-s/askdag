"""Write a parsed manifest into Postgres."""

from dataclasses import asdict
from importlib.resources import files

import psycopg

from asklineage.ingest.manifest import ParsedManifest

SCHEMA_SQL = files("asklineage").joinpath("sql/schema.sql")

INSERT_NODE = """
    INSERT INTO nodes (
        unique_id, project_name, resource_type, name, source_name, package_name,
        database_name, schema_name, relation_name, materialization, description,
        file_path, tags, owner, owner_source, raw_code
    ) VALUES (
        %(unique_id)s, %(project_name)s, %(resource_type)s, %(name)s, %(source_name)s,
        %(package_name)s, %(database_name)s, %(schema_name)s, %(relation_name)s,
        %(materialization)s, %(description)s, %(file_path)s, %(tags)s, %(owner)s,
        %(owner_source)s, %(raw_code)s
    )
"""

INSERT_EDGE = "INSERT INTO lineage_edges (parent_id, child_id) VALUES (%s, %s)"

INSERT_COLUMN = """
    INSERT INTO node_columns (node_id, name, description, data_type)
    VALUES (%(node_id)s, %(name)s, %(description)s, %(data_type)s)
"""

INSERT_TEST = """
    INSERT INTO data_tests (
        unique_id, project_name, name, test_type, column_name, attached_node_id, severity
    ) VALUES (
        %(unique_id)s, %(project_name)s, %(name)s, %(test_type)s, %(column_name)s,
        %(attached_node_id)s, %(severity)s
    )
"""

INSERT_LOAD = """
    INSERT INTO manifest_loads (
        project_name, dbt_version, schema_version, generated_at, invocation_id,
        node_count, edge_count
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id
"""


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL.read_text())


def load_manifest(conn: psycopg.Connection, parsed: ParsedManifest) -> int:
    """Replace everything stored for this project with the parsed manifest, atomically."""
    project = parsed.project_name
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM data_tests WHERE project_name = %s", (project,))
        cur.execute("DELETE FROM nodes WHERE project_name = %s", (project,))

        cur.executemany(
            INSERT_NODE,
            [
                {**asdict(node), "project_name": project, "tags": list(node.tags)}
                for node in parsed.nodes
            ],
        )
        cur.executemany(INSERT_EDGE, parsed.edges)
        cur.executemany(INSERT_COLUMN, [asdict(column) for column in parsed.columns])
        cur.executemany(
            INSERT_TEST, [{**asdict(test), "project_name": project} for test in parsed.tests]
        )

        cur.execute(
            INSERT_LOAD,
            (
                project,
                parsed.dbt_version,
                parsed.schema_version,
                parsed.generated_at,
                parsed.invocation_id,
                len(parsed.nodes),
                len(parsed.edges),
            ),
        )
        return cur.fetchone()["id"]
