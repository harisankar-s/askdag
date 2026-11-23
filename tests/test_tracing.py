from collections.abc import Iterator

import httpx2
import pytest
from ask_examples import EDGES, NODES, PARSE, QUESTION
from fastapi import FastAPI
from helpers import TEST_PROJECT
from langgraph.errors import GraphInterrupt
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from test_embed import make_embedder
from test_openai_compatible import Reply, completion_body, make_provider

from asklineage.api.flow import run_ask
from asklineage.api.models import AskRequest
from asklineage.config import Settings
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider
from asklineage.tracing import setup_tracing, traced_node

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider() -> None:
    # The global provider can be set once per process; every tracer here is a proxy to it.
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(tracer_provider)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    EXPORTER.clear()
    yield EXPORTER
    EXPORTER.clear()


def by_name(finished: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {span.name: span for span in finished}


def attribute_text(finished: tuple[ReadableSpan, ...]) -> str:
    return " ".join(str(v) for span in finished for v in (span.attributes or {}).values())


def test_every_flow_node_is_a_child_of_the_ask_span(conn, load_graph, spans) -> None:
    load_graph(NODES, EDGES)

    run_ask(conn, None, AskRequest(project=TEST_PROJECT, node="stg_orders"))

    named = by_name(spans.get_finished_spans())
    flow = [name for name in named if name.startswith("flow.")]
    assert flow == ["flow.begin", "flow.read_fields", "flow.resolve", "flow.lineage", "flow.answer"]
    ask = named["ask"]
    assert all(named[name].parent.span_id == ask.context.span_id for name in flow)
    assert ask.attributes["asklineage.kind"] == "lineage"
    assert ask.attributes["asklineage.input"] == "node"
    assert ask.attributes["asklineage.project"] == TEST_PROJECT


def test_question_text_is_never_recorded(conn, load_graph, spans) -> None:
    load_graph(NODES, EDGES)

    run_ask(conn, ScriptedProvider([PARSE]), AskRequest(project=TEST_PROJECT, question=QUESTION))

    finished = spans.get_finished_spans()
    assert "flow.parse" in by_name(finished)
    assert by_name(finished)["ask"].attributes["asklineage.input"] == "question"
    assert QUESTION not in attribute_text(finished)


def test_an_llm_call_is_a_client_span_with_usage(spans) -> None:
    llm, _ = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summary": "ok", "count": 4}'))
    )

    llm.complete_json(system="secret system", prompt="secret prompt", output=Reply)

    (span,) = spans.get_finished_spans()
    assert span.name == "chat test-model"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["server.address"] == "llm.test"
    assert span.attributes["gen_ai.usage.input_tokens"] == 120
    assert span.attributes["gen_ai.usage.output_tokens"] == 30
    assert span.attributes["asklineage.output_schema"] == "Reply"
    assert "secret" not in attribute_text((span,))


def test_a_failed_llm_call_marks_its_span(spans) -> None:
    llm, _ = make_provider(lambda _: httpx2.Response(500, json={"error": "boom"}))

    with pytest.raises(LLMError):
        llm.complete_json(system="s", prompt="p", output=Reply)

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "LLMError"


def test_an_embedding_call_records_kind_count_and_tokens(spans) -> None:
    embedder, _ = make_embedder(batch_size=2)

    embedder.embed(["a", "bb", "ccc"], kind="document")

    (span,) = spans.get_finished_spans()
    assert span.name == "embeddings test-embed"
    assert span.attributes["asklineage.embed_kind"] == "document"
    assert span.attributes["asklineage.texts"] == 3
    assert span.attributes["gen_ai.usage.input_tokens"] == 30


def test_an_interrupt_is_not_an_error(spans) -> None:
    def clarify(state):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        traced_node("clarify", clarify)({})

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["asklineage.interrupted"] is True


def test_off_installs_nothing() -> None:
    assert setup_tracing(Settings(tracing="off"), FastAPI()) is None
