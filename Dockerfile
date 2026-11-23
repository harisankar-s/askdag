FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
# The venv's scripts point at /app/.venv, so it must keep that path.
COPY --from=build /app/.venv /app/.venv
COPY examples/shop/manifests examples/shop/manifests
COPY examples/shop/corpus examples/shop/corpus
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=2)"]
CMD ["uvicorn", "asklineage.main:app", "--host", "0.0.0.0", "--port", "8000"]
