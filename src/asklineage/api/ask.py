"""POST /api/ask: a question in; a lineage answer, a clarification or an abstention out."""

from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends

from asklineage.api.flow import run_ask
from asklineage.api.models import AskRequest, AskResponse
from asklineage.config import get_settings
from asklineage.db import get_pool
from asklineage.llm.base import LLMProvider
from asklineage.llm.factory import get_provider

router = APIRouter(prefix="/api")


def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


@lru_cache
def get_llm() -> LLMProvider | None:
    return get_provider(get_settings())


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    conn: Annotated[psycopg.Connection, Depends(get_conn)],
    provider: Annotated[LLMProvider | None, Depends(get_llm)],
) -> AskResponse:
    return run_ask(conn, provider, request, max_named=get_settings().llm_max_named_nodes)
