from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from asklineage.config import get_settings

_pool: ConnectionPool | None = None
_checkpoint_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_checkpoint_pool() -> ConnectionPool:
    """For LangGraph's PostgresSaver, which needs autocommit and no prepared statements."""
    global _checkpoint_pool
    if _checkpoint_pool is None:
        _checkpoint_pool = ConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
    return _checkpoint_pool


def close_pool() -> None:
    global _pool, _checkpoint_pool
    for pool in (_pool, _checkpoint_pool):
        if pool is not None:
            pool.close()
    _pool = _checkpoint_pool = None
