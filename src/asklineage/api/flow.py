"""The /api/ask flow as a LangGraph graph (ADR 0008, 0009).

START ─ begin ─┬─ read_fields ─────┬─ resolve ─┬─ lineage ─ answer
               └─ parse ─┬────────┘           └─ clarify ─┬─ lineage
                         ├─ lineage (follow-up)            └─ END
                         └─ abstain

Nodes are thin wrappers over the pure functions; the connection and provider come through the
runtime context, not the state, so the state can be checkpointed. One builder compiles twice:
without a checkpointer (stateless, the default) and with one (conversation threads).
"""

from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict
from uuid import UUID

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from asklineage.answer.lineage_answer import HighlightStatus, answer_lineage
from asklineage.answer.parse import (
    PreviousQuestion,
    clean_entity,
    parse_question,
    recover_reading,
)
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
    OwnerInfo,
)
from asklineage.lineage.graph import DEFAULT_DEPTH, Direction, Lineage, LineageNode, lineage
from asklineage.lineage.resolve import (
    Ambiguous,
    Candidate,
    NotFound,
    Resolved,
    node_display_names,
    node_names,
    resolve_node,
)
from asklineage.llm.base import LLMProvider

# The longest run is 5 steps; a run that needs more is a bug, not a slow question.
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
    resumable: bool = False  # true only on a checkpointed thread, where interrupt() can work


@dataclass(frozen=True)
class LineageQuery:
    entity: str
    direction: Direction
    depth: int


@dataclass(frozen=True)
class Remembered:
    root: Candidate
    direction: Direction
    depth: int


def add_attempts(
    current: list[AttemptInfo] | None, update: list[AttemptInfo] | None
) -> list[AttemptInfo]:
    """Append each node's attempts; None (from begin) starts a new turn's list."""
    if update is None:
        return []
    return [*(current or []), *update]


class AskState(TypedDict, total=False):
    # Turn-scoped: begin clears these, because a thread's checkpoint carries every key into
    # the next question, and a stale abstain_reason or root would route the new one wrongly.
    request: AskRequest
    query: LineageQuery | None
    abstain_reason: AbstainReason | None
    root: Candidate | None
    unresolved: Ambiguous | NotFound | None
    result: Lineage | None
    response: AskResponse | None
    attempts: Annotated[list[AttemptInfo], add_attempts]
    # Thread-scoped: what the last answer was about, for follow-up questions.
    previous: Remembered | None


# Every type stored in the state, so checkpoints deserialise them without LangGraph's
# "unregistered type" path, which a future version will block.
STATE_TYPES: tuple[type, ...] = (
    AskRequest,
    LineageQuery,
    Remembered,
    Candidate,
    Ambiguous,
    NotFound,
    Lineage,
    LineageNode,
    Direction,
    HighlightStatus,
    AttemptInfo,
    LineageResponse,
    ClarificationResponse,
    AbstainResponse,
    NodeInfo,
    CandidateInfo,
    NodeRow,
    FactsInfo,
    OwnerInfo,
    HighlightInfo,
)


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def begin(state: AskState) -> AskState:
    return {
        "query": None,
        "abstain_reason": None,
        "root": None,
        "unresolved": None,
        "result": None,
        "response": None,
        "attempts": None,
    }


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
    previous = state.get("previous")
    asked_before = (
        PreviousQuestion(previous.root.display_name, previous.direction, previous.depth)
        if previous
        else None
    )
    question = state["request"].question
    # Only a follow-up can claim the previous node, so only then are node names needed.
    known = node_names(runtime.context.conn, state["request"].project) if previous else frozenset()
    generated = parse_question(provider, question, asked_before, known)
    attempts = [AttemptInfo.from_attempt("parse", a) for a in generated.attempts]
    parsed = generated.output or recover_reading(generated, question)
    if parsed is None:
        reason = "llm_unavailable" if generated.attempts[-1].error else "not_understood"
        return {"abstain_reason": reason, "attempts": attempts}
    if not parsed.is_lineage:
        return {"abstain_reason": "not_lineage", "attempts": attempts}

    # The check guarantees a direction, or a previous question to keep one from.
    direction = parsed.direction or previous.direction
    depth = parsed.depth or (previous.depth if previous else DEFAULT_DEPTH)
    if parsed.same_node:
        query = LineageQuery(previous.root.display_name, direction, depth)
        return {"query": query, "root": previous.root, "attempts": attempts}
    query = LineageQuery(clean_entity(parsed.entity), direction, depth)
    return {"query": query, "attempts": attempts}


def after_parse(state: AskState) -> Literal["abstain", "lineage", "resolve"]:
    if state.get("abstain_reason"):
        return "abstain"
    return "lineage" if state.get("root") else "resolve"


def abstain(state: AskState) -> AskState:
    reason = state["abstain_reason"]
    return {"response": AbstainResponse(reason=reason, message=ABSTAIN_MESSAGES[reason])}


def resolve(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    resolution = resolve_node(runtime.context.conn, state["query"].entity, state["request"].project)
    if isinstance(resolution, Resolved):
        return {"root": resolution.node}
    return {"unresolved": resolution}


def after_resolve(state: AskState) -> Literal["lineage", "clarify"]:
    return "lineage" if state.get("root") else "clarify"


def candidates_of(unresolved: Ambiguous | NotFound) -> list[Candidate]:
    if isinstance(unresolved, Ambiguous):
        return list(unresolved.candidates)
    return list(unresolved.suggestions)


def clarification(unresolved: Ambiguous | NotFound, resumable: bool) -> ClarificationResponse:
    candidates = candidates_of(unresolved)
    query = unresolved.query
    if isinstance(unresolved, Ambiguous):
        reason, message = "ambiguous", f"{query!r} matches {len(candidates)} nodes."
    else:
        reason, message = "not_found", f"No node named {query!r}."
        if not candidates:
            message += " Check the name in dbt."
    if candidates:
        how = "send choose with" if resumable else "ask again with node set to"
        message += f" To use one of these, {how} its unique_id."
    return ClarificationResponse(
        reason=reason,
        query=query,
        message=message,
        candidates=[CandidateInfo.from_candidate(c) for c in candidates],
        resumable=resumable and bool(candidates),
    )


def clarify(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    # On resume this node runs again from the top, so nothing before interrupt() may have
    # side effects.
    unresolved = state["unresolved"]
    response = clarification(unresolved, runtime.context.resumable)
    if not response.resumable:
        return {"response": response}
    choice = interrupt(response.model_dump(mode="json"))

    picked = next((c for c in candidates_of(unresolved) if c.unique_id == choice), None)
    if picked is None:
        rejected = response.model_copy(
            update={
                "reason": "invalid_choice",
                "message": f"{choice!r} was not one of the candidates. Ask the question again.",
                "resumable": False,
            }
        )
        return {"response": rejected}
    return {"root": picked}


def after_clarify(state: AskState) -> Literal["lineage", "__end__"]:
    return "lineage" if state.get("root") else END


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
    return {
        "response": response,
        "attempts": attempts,
        "previous": Remembered(root, query.direction, query.depth),
        # The response already holds the table; don't checkpoint the whole walk as well.
        "result": None,
    }


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AskState, context_schema=AskContext)
    graph.add_node("begin", begin)
    graph.add_node("read_fields", read_fields)
    graph.add_node("parse", parse)
    graph.add_node("abstain", abstain)
    graph.add_node("resolve", resolve)
    graph.add_node("clarify", clarify)
    graph.add_node("lineage", walk)
    graph.add_node("answer", answer)

    graph.add_edge(START, "begin")
    graph.add_conditional_edges("begin", route_input, ["read_fields", "parse"])
    graph.add_edge("read_fields", "resolve")
    graph.add_conditional_edges("parse", after_parse, ["abstain", "lineage", "resolve"])
    graph.add_conditional_edges("resolve", after_resolve, ["lineage", "clarify"])
    graph.add_conditional_edges("clarify", after_clarify, ["lineage", END])
    graph.add_edge("lineage", "answer")
    for last in ("abstain", "answer"):
        graph.add_edge(last, END)
    return graph.compile(checkpointer=checkpointer)


ASK_GRAPH = build_graph()


def thread_config(thread_id: UUID) -> dict:
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": RECURSION_LIMIT}


def waiting_for_choice(graph: CompiledStateGraph, thread_id: UUID) -> bool:
    return bool(graph.get_state(thread_config(thread_id)).interrupts)


def run_ask(
    conn: psycopg.Connection,
    provider: LLMProvider | None,
    request: AskRequest,
    *,
    max_named: int,
    graph: CompiledStateGraph = ASK_GRAPH,
    thread_id: UUID | None = None,
) -> AskResponse:
    """Stateless unless thread_id is given, in which case graph must have a checkpointer."""
    context = AskContext(conn, provider, max_named, resumable=thread_id is not None)
    if thread_id is None:
        final = graph.invoke(
            {"request": request}, context=context, config={"recursion_limit": RECURSION_LIMIT}
        )
        response = final["response"]
    else:
        config = thread_config(thread_id)
        graph_input = Command(resume=request.choose) if request.choose else {"request": request}
        # "exit": one checkpoint per turn (or at the interrupt), not one per step.
        final = graph.invoke(graph_input, context=context, config=config, durability="exit")
        interrupts = graph.get_state(config).interrupts
        if interrupts:
            response = ClarificationResponse.model_validate(interrupts[0].value)
        else:
            response = final["response"]
    return response.model_copy(
        update={"attempts": final.get("attempts") or [], "thread_id": thread_id}
    )
