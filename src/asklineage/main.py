from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from asklineage.api.ask import router as ask_router
from asklineage.api.info import router as info_router
from asklineage.db import close_pool, get_pool

INDEX_HTML = files("asklineage").joinpath("web/index.html")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    close_pool()


app = FastAPI(title="asklineage", lifespan=lifespan)
app.include_router(ask_router)
app.include_router(info_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    try:
        with get_pool().connection(timeout=2) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ready"}
