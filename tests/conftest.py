import os
from collections.abc import Callable, Iterator

import psycopg
import pytest
from helpers import TEST_PROJECT
from psycopg.rows import dict_row

from asklineage.config import get_settings
from asklineage.ingest.load import apply_schema, load_manifest
from asklineage.ingest.manifest import Node, ParsedManifest


@pytest.fixture(scope="session")
def database_url() -> str:
    url = get_settings().database_url
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            apply_schema(conn)
    except psycopg.OperationalError as exc:
        # A missing database must fail CI, not quietly skip every database test.
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Postgres is not reachable ({exc}); start it with `docker compose up -d`")
    return url


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.transaction(force_rollback=True):
            yield conn


@pytest.fixture
def load_graph(conn: psycopg.Connection) -> Callable[..., None]:
    def load(nodes: list[Node], edges: list[tuple[str, str]]) -> None:
        load_manifest(
            conn,
            ParsedManifest(
                project_name=TEST_PROJECT,
                dbt_version="1.10.8",
                schema_version="v12",
                generated_at=None,
                invocation_id=None,
                nodes=tuple(nodes),
                edges=tuple(edges),
                columns=(),
                tests=(),
            ),
        )

    return load
