"""A client span around one model call, named and tagged per OpenTelemetry's GenAI conventions.

Only the OpenTelemetry API is used, so this is a no-op until a provider is installed (ADR 0023).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import openai
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

tracer = trace.get_tracer("asklineage.models")


@contextmanager
def model_span(operation: str, model: str, client: openai.OpenAI) -> Iterator[Span]:
    url = client.base_url
    with tracer.start_as_current_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("server.address", url.host)
        if url.port:
            span.set_attribute("server.port", url.port)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
