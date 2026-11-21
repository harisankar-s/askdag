"""The /api/ask flow as a LangGraph graph (ADR 0008).

START ─┬─ read_fields ─────┬─ resolve ─┬─ lineage ─ answer
       └─ parse ─┬────────┘           └─ clarify
                 └─ abstain

Nodes are thin wrappers over the pure functions; the connection and provider come through the
runtime context, not the state, so the state stays serializable for a checkpointer.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from asklineage.answer.lineage_answer import answer_lineage
from asklineage.answer.parse import clean_entity, parse_question
from asklineage.api.models import (
    AbstainReason,
    AbstainResponse,
    AskRequest,
    AskResponse,
    AttemptInfo,
    CandidateInfo,
    ClarificationResponse,
    FactsInfo,
    HighlightInfo,
    LineageResponse,
    NodeInfo,
    NodeRow,
)
from asklineage.lineage.graph import DEFAULT_DEPTH, Direction, Lineage, lineage
from asklineage.lineage.resolve import (
    Ambiguous,
    Candidate,
    NotFound,
    Resolved,
    node_display_names,
    resolve_node,
)
from asklineage.llm.base import LLMProvider

# The longest run is 4 steps; a run that needs more is a bug, not a slow question.
RECURSION_LIMIT = 10

ABSTAIN_MESSAGES: dict[str, str] = {
    "not_lineage": (
        "asklineage answers lineage questions about this dbt project: where a model's data "
        "comes from, and what reads from it downstream. Ask about a model or source by name."
    ),
    "not_understood": (
        "I couldn't tell which node and direction the question is about. Name the model or "
        "source exactly as it is in dbt, for example: what depends on stg_orders?"
    ),
    "llm_off": (
        "Free-text questions need an LLM, and LLM_PROVIDER is none. "
        "Pass node, and optionally direction and depth, instead."
    ),
    "llm_unavailable": (
        "The LLM could not be reached, so the question could not be read. "
        "Pass node, and optionally direction and depth, instead."
    ),
}


@dataclass(frozen=True)
class AskContext:
    conn: psycopg.Connection
    provider: LLMProvider | None
    max_named: int


@dataclass(frozen=True)
class LineageQuery:
    entity: str
    direction: Direction
    depth: int


class AskState(TypedDict, total=False):
    request: AskRequest
    query: LineageQuery
    abstain_reason: AbstainReason
    root: Candidate
    unresolved: Ambiguous | NotFound
    result: Lineage
    response: AskResponse
    # Each node returns only its own attempts; the reducer appends them in order.
    attempts: Annotated[list[AttemptInfo], operator.add]


def route_input(state: AskState) -> Literal["read_fields", "parse"]:
    return "read_fields" if state["request"].node else "parse"


def read_fields(state: AskState) -> AskState:
    request = state["request"]
    direction = request.direction or Direction.DOWNSTREAM
    query = LineageQuery(request.node, direction, request.depth or DEFAULT_DEPTH)
    return {"query": query}


def parse(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    provider = runtime.context.provider
    if provider is None:
        return {"abstain_reason": "llm_off"}
    generated = parse_question(provider, state["request"].question)
    attempts = [AttemptInfo.from_attempt("parse", a) for a in generated.attempts]
    parsed = generated.output
    if parsed is None:
        reason = "llm_unavailable" if generated.attempts[-1].error else "not_understood"
        return {"abstain_reason": reason, "attempts": attempts}
    if not parsed.is_lineage:
        return {"abstain_reason": "not_lineage", "attempts": attempts}
    query = LineageQuery(
        clean_entity(parsed.entity), parsed.direction, parsed.depth or DEFAULT_DEPTH
    )
    return {"query": query, "attempts": attempts}


def after_parse(state: AskState) -> Literal["abstain", "resolve"]:
    return "abstain" if "abstain_reason" in state else "resolve"


def abstain(state: AskState) -> AskState:
    reason = state["abstain_reason"]
    return {"response": AbstainResponse(reason=reason, message=ABSTAIN_MESSAGES[reason])}


def resolve(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    resolution = resolve_node(runtime.context.conn, state["query"].entity, state["request"].project)
    if isinstance(resolution, Resolved):
        return {"root": resolution.node}
    return {"unresolved": resolution}


def after_resolve(state: AskState) -> Literal["lineage", "clarify"]:
    return "lineage" if "root" in state else "clarify"


def clarify(state: AskState) -> AskState:
    match state["unresolved"]:
        case Ambiguous(query=query, candidates=candidates):
            reason = "ambiguous"
            message = (
                f"{query!r} matches {len(candidates)} nodes. "
                "Ask again with one of these unique_ids as node."
            )
        case NotFound(query=query, suggestions=candidates):
            reason = "not_found"
            hint = "Did you mean one of these?" if candidates else "Check the name in dbt."
            message = f"No node named {query!r}. {hint}"
    response = ClarificationResponse(
        reason=reason,
        query=query,
        message=message,
        candidates=[CandidateInfo.from_candidate(c) for c in candidates],
    )
    return {"response": response}


def walk(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    query = state["query"]
    result = lineage(runtime.context.conn, state["root"].unique_id, query.direction, query.depth)
    return {"result": result}


def answer(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    context, root, query = runtime.context, state["root"], state["query"]
    # Only the highlight check uses every node name; skip the query when there's no LLM.
    known = node_display_names(context.conn, state["request"].project) if context.provider else ()
    lineage_answer = answer_lineage(
        state["result"], root, context.provider, max_named=context.max_named, known_names=known
    )
    highlights = [
        HighlightInfo(text=text, upstream=c.upstream, downstream=c.downstream, hops=c.hops)
        for c, text in zip(lineage_answer.highlights, lineage_answer.highlight_texts, strict=True)
    ]
    response = LineageResponse(
        node=NodeInfo.from_candidate(root),
        direction=query.direction,
        depth=query.depth,
        summary=lineage_answer.summary,
        facts=FactsInfo.from_facts(lineage_answer.facts),
        highlights=highlights,
        highlight_status=lineage_answer.status,
        table=[NodeRow.from_node(node) for node in lineage_answer.lineage.nodes],
    )
    attempts = [AttemptInfo.from_attempt("highlights", a) for a in lineage_answer.attempts]
    return {"response": response, "attempts": attempts}


def build_graph():
    graph = StateGraph(AskState, context_schema=AskContext)
    graph.add_node("read_fields", read_fields)
    graph.add_node("parse", parse)
    graph.add_node("abstain", abstain)
    graph.add_node("resolve", resolve)
    graph.add_node("clarify", clarify)
    graph.add_node("lineage", walk)
    graph.add_node("answer", answer)

    graph.add_conditional_edges(START, route_input, ["read_fields", "parse"])
    graph.add_edge("read_fields", "resolve")
    graph.add_conditional_edges("parse", after_parse, ["abstain", "resolve"])
    graph.add_conditional_edges("resolve", after_resolve, ["lineage", "clarify"])
    graph.add_edge("lineage", "answer")
    for last in ("abstain", "clarify", "answer"):
        graph.add_edge(last, END)
    return graph.compile()


ASK_GRAPH = build_graph()


def run_ask(
    conn: psycopg.Connection,
    provider: LLMProvider | None,
    request: AskRequest,
    *,
    max_named: int,
) -> AskResponse:
    final = ASK_GRAPH.invoke(
        {"request": request},
        context=AskContext(conn, provider, max_named),
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return final["response"].model_copy(update={"attempts": final.get("attempts", [])})
