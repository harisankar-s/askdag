"""GET /api/info: what the page needs to explain itself. Never returns the key or the base URL."""

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from asklineage.api.ask import get_conn
from asklineage.config import Settings, get_settings

router = APIRouter(prefix="/api")


class InfoResponse(BaseModel):
    projects: list[str]
    llm_provider: str
    llm_model: str | None  # None when no LLM is configured


@router.get("/info", response_model=InfoResponse)
def info(
    conn: Annotated[psycopg.Connection, Depends(get_conn)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> InfoResponse:
    rows = conn.execute("SELECT DISTINCT project_name FROM nodes ORDER BY project_name").fetchall()
    return InfoResponse(
        projects=[row["project_name"] for row in rows],
        llm_provider=settings.llm_provider,
        llm_model=None if settings.llm_provider == "none" else settings.llm_model,
    )
