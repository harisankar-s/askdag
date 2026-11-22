"""POST /api/ask: a question in; a lineage, owner or search answer, a clarification or an
abstention out."""

from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph

from asklineage.api.flow import (
    SearchOptions,
    build_graph,
    run_ask,
    search_options,
    waiting_for_choice,
)
from asklineage.api.models import AskRequest, AskResponse
from asklineage.api.threads import create_thread, make_saver, touch_thread, try_lock_thread
from asklineage.config import get_settings
from asklineage.db import get_checkpoint_pool, get_pool
from asklineage.llm.base import LLMProvider
from asklineage.llm.factory import get_provider

router = APIRouter(prefix="/api")


def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


@lru_cache
def get_llm() -> LLMProvider | None:
    return get_provider(get_settings())


@lru_cache
def get_search() -> SearchOptions:
    return search_options(get_settings())


@lru_cache
def get_thread_graph() -> CompiledStateGraph:
    return build_graph(checkpointer=make_saver(get_checkpoint_pool()))


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    conn: Annotated[psycopg.Connection, Depends(get_conn)],
    provider: Annotated[LLMProvider | None, Depends(get_llm)],
    thread_graph: Annotated[CompiledStateGraph, Depends(get_thread_graph)],
    search: Annotated[SearchOptions, Depends(get_search)],
) -> AskResponse:
    if not request.start_thread and request.thread_id is None:
        return run_ask(conn, provider, request, search_options=search)

    if request.start_thread:
        thread_id = create_thread(conn)
    else:
        thread_id = request.thread_id
        if not touch_thread(conn, thread_id):
            raise HTTPException(404, "No such thread; it may have been pruned. Start a new one.")
        if not try_lock_thread(conn, thread_id):
            raise HTTPException(409, "This thread is answering another question; try again.")
        if request.choose and not waiting_for_choice(thread_graph, thread_id):
            raise HTTPException(409, "This thread is not waiting for a choice.")
    return run_ask(
        conn,
        provider,
        request,
        graph=thread_graph,
        thread_id=thread_id,
        search_options=search,
    )
