"""/api/ask conversation threads: ask_threads says which exist; LangGraph's tables hold state."""

from datetime import timedelta
from uuid import UUID

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from asklineage.api.flow import checkpoint_serde


def make_saver(pool: ConnectionPool) -> PostgresSaver:
    return PostgresSaver(pool, serde=checkpoint_serde())


def create_thread(conn: psycopg.Connection) -> UUID:
    row = conn.execute("INSERT INTO ask_threads DEFAULT VALUES RETURNING thread_id").fetchone()
    return row["thread_id"]


def touch_thread(conn: psycopg.Connection, thread_id: UUID) -> bool:
    """False when the thread doesn't exist (never created, or pruned)."""
    row = conn.execute(
        "UPDATE ask_threads SET last_used_at = now() WHERE thread_id = %s RETURNING thread_id",
        (thread_id,),
    ).fetchone()
    return row is not None


def try_lock_thread(conn: psycopg.Connection, thread_id: UUID) -> bool:
    # Held until the request's transaction ends, so two turns on one thread can't interleave.
    row = conn.execute(
        "SELECT pg_try_advisory_xact_lock(hashtextextended(%s::text, 0)) AS locked",
        (thread_id,),
    ).fetchone()
    return row["locked"]


def prune_threads(
    conn: psycopg.Connection, saver: BaseCheckpointSaver, older_than: timedelta
) -> int:
    """Delete threads unused for longer than older_than, with their checkpoints."""
    rows = conn.execute(
        "DELETE FROM ask_threads WHERE last_used_at < now() - %s RETURNING thread_id",
        (older_than,),
    ).fetchall()
    for row in rows:
        saver.delete_thread(str(row["thread_id"]))
    return len(rows)
