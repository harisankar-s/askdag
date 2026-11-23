"""OpenTelemetry for the API (ADR 0023): one span per request, per flow node and per model call.

Library code only uses the OpenTelemetry API, which is a no-op until setup_tracing installs a
provider. Spans carry counts, names, models and timings, never question, prompt or reply text.
"""

import functools
from collections.abc import Callable

from fastapi import FastAPI
from langgraph.errors import GraphInterrupt
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.trace import Status, StatusCode

from asklineage.config import Settings

SERVICE_NAME = "asklineage"
tracer = trace.get_tracer("asklineage.flow")


def setup_tracing(settings: Settings, app: FastAPI) -> TracerProvider | None:
    """Installs the global provider and the request spans; None (and nothing done) when off."""
    if settings.tracing == "off":
        return None
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    exporter: SpanExporter
    if settings.tracing == "console":
        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=settings.tracing_otlp_endpoint)
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # receive/send spans per ASGI message add nothing for a JSON API.
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls="health/.*",
        exclude_spans=["receive", "send"],
    )
    return provider


def traced_node[F: Callable](name: str, node: F) -> F:
    """Wraps a LangGraph node in a span. functools.wraps keeps the signature LangGraph reads to
    decide whether to pass `runtime`."""

    @functools.wraps(node)
    def run(*args, **kwargs):
        with tracer.start_as_current_span(
            f"flow.{name}", record_exception=False, set_status_on_exception=False
        ) as span:
            try:
                return node(*args, **kwargs)
            except GraphInterrupt:
                # interrupt() pauses a thread for a clarification; it is not a failure.
                span.set_attribute("asklineage.interrupted", True)
                raise
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise

    return run  # type: ignore[return-value]
