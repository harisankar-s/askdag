from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from asklineage.api.ask import router as ask_router
from asklineage.api.auth import require_api_key
from asklineage.api.info import router as info_router
from asklineage.config import get_settings
from asklineage.db import close_pool, get_pool
from asklineage.tracing import setup_tracing

INDEX_HTML = files("asklineage").joinpath("web/index.html")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    close_pool()
    if tracing is not None:
        tracing.shutdown()


app = FastAPI(title="asklineage", lifespan=lifespan)
# Health checks and the page stay open: probes carry no key, and the page holds no data.
app.include_router(ask_router, dependencies=[Depends(require_api_key)])
app.include_router(info_router, dependencies=[Depends(require_api_key)])
# Before the first request: the middleware stack is fixed once the app starts serving.
tracing = setup_tracing(get_settings(), app)


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
