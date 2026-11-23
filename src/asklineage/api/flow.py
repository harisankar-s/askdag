"""The /api/ask flow as a LangGraph graph (ADR 0008, 0009).

START ─ begin ─┬─ read_fields ─────┬─ resolve ─┬─ lineage ─ answer
               └─ parse ─┬────────┘           ├─ owner
                         ├─ lineage | owner    └─ clarify ─┬─ lineage | owner
                         │   (follow-up)                    └─ END
                         ├─ search
                         ├─ history
                         └─ abstain

The parse call's intent picks lineage or owner after a node is resolved; owner answers come
from SQL alone. history searches incidents with the question as asked (ADR 0015, 0016).

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
from psycopg.rows import dict_row

from asklineage.answer.facts import OWNER_SOURCE_MEANING
from asklineage.answer.history import Retrieved, answer_history
from asklineage.answer.lineage_answer import answer_lineage
from asklineage.answer.parse import (
    Intent,
    PreviousQuestion,
    clean_entity,
    keep_first_reading,
    keep_previous_direction,
    named_node_wins,
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
    EdgeInfo,
    FactsInfo,
    HighlightInfo,
    HistoryResponse,
    HistorySentenceInfo,
    IncidentHit,
    LineageResponse,
    NodeInfo,
    NodeRow,
    OwnerInfo,
    OwnerResponse,
    SearchHit,
    SearchResponse,
)
from asklineage.config import Settings
from asklineage.embed.base import EmbeddingProvider
from asklineage.embed.factory import get_embedder
from asklineage.incidents.search import find_incidents, incident_sections, known_spellings
from asklineage.lineage.graph import DEFAULT_DEPTH, Direction, Lineage, LineageNode, lineage
from asklineage.lineage.resolve import (
    Ambiguous,
    Candidate,
    NotFound,
    Resolved,
    node_names,
    resolve_node,
)
from asklineage.llm.base import LLMProvider
from asklineage.search.query import search
from asklineage.search.scan import Scan
from asklineage.tracing import traced_node, tracer

HISTORY_INCIDENTS = 3
HISTORY_SECTIONS = ("Summary", "Root cause", "Fix")

# The longest run is 5 steps; a run that needs more is a bug, not a slow question.
RECURSION_LIMIT = 10

ABSTAIN_MESSAGES: dict[str, str] = {
    "out_of_scope": (
        "asklineage answers questions about this dbt project's models: where a model's data "
        "comes from, what reads from it downstream, who owns it, and whether it has broken "
        "before. Ask about a model or source by name."
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
class SearchOptions:
    embedder: EmbeddingProvider | None = None  # None: keyword search
    prefer: tuple[str, ...] = ()  # path patterns for lineage-aware ranking; () turns it off
    low_confidence_below: float = 0.667
    k: int = 10
    scan: Scan = "topn"  # how vector queries find the nearest chunks (search/scan.py)
    overfetch: int = 4
    ef_search: int | None = None


@dataclass(frozen=True)
class AskContext:
    conn: psycopg.Connection
    provider: LLMProvider | None
    resumable: bool = False  # true only on a checkpointed thread, where interrupt() can work
    search: SearchOptions = SearchOptions()


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
    intent: Intent | None
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
    AttemptInfo,
    LineageResponse,
    ClarificationResponse,
    AbstainResponse,
    OwnerResponse,
    SearchResponse,
    SearchHit,
    HistoryResponse,
    IncidentHit,
    HistorySentenceInfo,
    NodeInfo,
    CandidateInfo,
    NodeRow,
    EdgeInfo,
    FactsInfo,
    OwnerInfo,
    HighlightInfo,
)


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def begin(state: AskState) -> AskState:
    return {
        "intent": None,
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
    return {"intent": "lineage", "query": query}


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
    parsed = (
        generated.output
        or recover_reading(generated, question)
        or named_node_wins(generated, question)
    )
    if parsed is not None:
        parsed = keep_previous_direction(
            keep_first_reading(generated, parsed), question, asked_before
        )
    if parsed is None:
        reason = "llm_unavailable" if generated.attempts[-1].error else "not_understood"
        return {"abstain_reason": reason, "attempts": attempts}
    if parsed.intent in ("search", "history"):
        return {"intent": parsed.intent, "attempts": attempts}
    if parsed.intent == "other":
        return {"abstain_reason": "out_of_scope", "attempts": attempts}

    # For lineage the check guarantees a direction, or a previous question to keep one from.
    # An owner question keeps the previous direction and depth, so "and upstream?" still works.
    fallback = previous.direction if previous else Direction.DOWNSTREAM
    direction = parsed.direction or fallback
    depth = parsed.depth or (previous.depth if previous else DEFAULT_DEPTH)
    update: AskState = {"intent": parsed.intent, "attempts": attempts}
    if parsed.same_node:
        query = LineageQuery(previous.root.display_name, direction, depth)
        return update | {"query": query, "root": previous.root}
    return update | {"query": LineageQuery(clean_entity(parsed.entity), direction, depth)}


def answer_route(state: AskState) -> Literal["lineage", "owner"]:
    return "owner" if state.get("intent") == "owner" else "lineage"


def after_parse(
    state: AskState,
) -> Literal["abstain", "search", "history", "lineage", "owner", "resolve"]:
    if state.get("abstain_reason"):
        return "abstain"
    if state.get("intent") in ("search", "history"):
        return state["intent"]
    return answer_route(state) if state.get("root") else "resolve"


def abstain(state: AskState) -> AskState:
    reason = state["abstain_reason"]
    return {"response": AbstainResponse(reason=reason, message=ABSTAIN_MESSAGES[reason])}


def resolve(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    resolution = resolve_node(runtime.context.conn, state["query"].entity, state["request"].project)
    if isinstance(resolution, Resolved):
        return {"root": resolution.node}
    return {"unresolved": resolution}


def after_resolve(state: AskState) -> Literal["lineage", "owner", "clarify"]:
    return answer_route(state) if state.get("root") else "clarify"


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


def after_clarify(state: AskState) -> Literal["lineage", "owner", "__end__"]:
    return answer_route(state) if state.get("root") else END


def walk(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    query = state["query"]
    result = lineage(runtime.context.conn, state["root"].unique_id, query.direction, query.depth)
    return {"result": result}


def answer(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    root, query = state["root"], state["query"]
    lineage_answer = answer_lineage(state["result"], root)
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
        table=[NodeRow.from_node(node) for node in lineage_answer.lineage.nodes],
        edges=[EdgeInfo(parent_id=p, child_id=c) for p, c in lineage_answer.lineage.edges],
    )
    return {
        "response": response,
        "previous": Remembered(root, query.direction, query.depth),
        # The response already holds the table and edges; don't checkpoint the whole walk as well.
        "result": None,
    }


def owner(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    root, query = state["root"], state["query"]
    with runtime.context.conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            "SELECT owner, owner_source FROM nodes WHERE unique_id = %s", (root.unique_id,)
        ).fetchone()
    found = row["owner"] if row else None
    source = row["owner_source"] if found else None
    meaning = OWNER_SOURCE_MEANING.get(source, source) if source else None
    if found:
        message = f"{root.display_name} is owned by {found} ({meaning})."
    else:
        message = f"No owner is recorded for {root.display_name} in dbt or the owners file."
    response = OwnerResponse(
        node=NodeInfo.from_candidate(root),
        owner=found,
        owner_source=source,
        owner_source_meaning=meaning,
        message=message,
    )
    return {"response": response, "previous": Remembered(root, query.direction, query.depth)}


def find(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    options = runtime.context.search
    request = state["request"]
    result = search(
        runtime.context.conn,
        request.question,
        options.embedder,
        project=request.project,
        k=options.k,
        prefer=options.prefer,
        scan=options.scan,
        overfetch=options.overfetch,
        ef_search=options.ef_search,
    )
    hits = [
        SearchHit(
            unique_id=hit.unique_id,
            display_name=hit.display_name,
            resource_type=hit.resource_type,
            owner=hit.owner,
            owner_source=hit.owner_source,
            score=round(hit.score, 4),
            match=hit.kind,
            via=hit.via,
        )
        for hit in result.hits
    ]
    weak = result.mode == "semantic" and (not hits or hits[0].score < options.low_confidence_below)
    if not hits:
        message = "No node's name, description, columns or SQL matched the question."
    elif weak:
        message = (
            f"The best match scores {hits[0].score:.3f}, under {options.low_confidence_below}: "
            "the data asked about may not be in this project. Check the matches before using one."
        )
    elif result.mode == "keyword":
        message = "Matched by keywords only, which often misses; check the matches before use."
    else:
        message = f"{len(hits)} nodes ranked by how well their text matches the question."
    response = SearchResponse(
        mode=result.mode,
        model=result.model,
        hits=hits,
        low_confidence=weak or result.mode == "keyword",
        message=message,
        note=result.note,
    )
    return {"response": response}


def recall(state: AskState, runtime: Runtime[AskContext]) -> AskState:
    context, request = runtime.context, state["request"]
    embedder = context.search.embedder
    mode = "hybrid" if embedder else "keyword"
    hits = find_incidents(
        context.conn,
        request.question,
        mode,
        embedder,
        project=request.project,
        k=HISTORY_INCIDENTS,
        scan=context.search.scan,
        overfetch=context.search.overfetch,
        ef_search=context.search.ef_search,
    )
    found = incident_sections(context.conn, hits, HISTORY_SECTIONS)
    with context.conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT project_name, doc_id, doc_date, severity, url FROM documents "
            "WHERE (project_name, doc_id) IN (SELECT * FROM unnest(%s::text[], %s::text[]))",
            ([h.project for h in hits], [h.doc_id for h in hits]),
        ).fetchall()
    meta = {(r["project_name"], r["doc_id"]): r for r in rows}
    retrieved = [
        Retrieved(
            hit.doc_id,
            hit.title,
            tuple(
                (section, content.split("\n", 1)[-1].removeprefix(f"{section}: "))
                for section, content in found[(hit.project, hit.doc_id)]
            ),
            meta[(hit.project, hit.doc_id)]["doc_date"].isoformat(),
        )
        for hit in hits
    ]
    known = known_spellings(context.conn, request.project) if context.provider else {}
    result = answer_history(request.question, retrieved, context.provider, known)
    incidents = [
        IncidentHit(
            doc_id=hit.doc_id,
            title=hit.title,
            date=meta[(hit.project, hit.doc_id)]["doc_date"].isoformat(),
            severity=meta[(hit.project, hit.doc_id)]["severity"],
            url=meta[(hit.project, hit.doc_id)]["url"],
            score=round(hit.score, 4),
            match=hit.section or "",
        )
        for hit in hits
    ]
    summary = result.summary
    if not incidents:
        message = "No recorded incident matches the question."
    elif summary is None:
        why = "LLM_PROVIDER is none" if context.provider is None else "no reply passed the checks"
        message = f"The closest {len(incidents)} incidents, unsummarised ({why}); check they match."
    elif not summary.answered:
        message = "No recorded incident answers the question; these are only the closest matches."
    else:
        message = f"Summarised from {len(incidents)} incidents; each sentence cites its sources."
    response = HistoryResponse(
        mode=mode,
        incidents=incidents,
        answered=summary.answered if summary else None,
        summary=[
            HistorySentenceInfo(text=s.text, cites=list(s.cites))
            for s in (summary.sentences if summary and summary.answered else [])
        ],
        message=message,
    )
    attempts = [AttemptInfo.from_attempt("history", a) for a in result.attempts]
    return {"response": response, "attempts": attempts}


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AskState, context_schema=AskContext)
    graph.add_node("begin", traced_node("begin", begin))
    graph.add_node("read_fields", traced_node("read_fields", read_fields))
    graph.add_node("parse", traced_node("parse", parse))
    graph.add_node("abstain", traced_node("abstain", abstain))
    graph.add_node("resolve", traced_node("resolve", resolve))
    graph.add_node("clarify", traced_node("clarify", clarify))
    graph.add_node("lineage", traced_node("lineage", walk))
    graph.add_node("answer", traced_node("answer", answer))
    graph.add_node("owner", traced_node("owner", owner))
    graph.add_node("search", traced_node("search", find))
    graph.add_node("history", traced_node("history", recall))

    graph.add_edge(START, "begin")
    graph.add_conditional_edges("begin", route_input, ["read_fields", "parse"])
    graph.add_edge("read_fields", "resolve")
    graph.add_conditional_edges(
        "parse", after_parse, ["abstain", "search", "history", "lineage", "owner", "resolve"]
    )
    graph.add_conditional_edges("resolve", after_resolve, ["lineage", "owner", "clarify"])
    graph.add_conditional_edges("clarify", after_clarify, ["lineage", "owner", END])
    graph.add_edge("lineage", "answer")
    for last in ("abstain", "answer", "owner", "search", "history"):
        graph.add_edge(last, END)
    return graph.compile(checkpointer=checkpointer)


ASK_GRAPH = build_graph()


def search_options(settings: Settings) -> SearchOptions:
    return SearchOptions(
        embedder=get_embedder(settings),
        prefer=tuple(settings.search_preferred_paths),
        low_confidence_below=settings.search_low_confidence_below,
        scan=settings.search_vector_scan,
        overfetch=settings.search_overfetch,
        ef_search=settings.search_hnsw_ef_search,
    )


def thread_config(thread_id: UUID) -> dict:
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": RECURSION_LIMIT}


def waiting_for_choice(graph: CompiledStateGraph, thread_id: UUID) -> bool:
    return bool(graph.get_state(thread_config(thread_id)).interrupts)


def run_ask(
    conn: psycopg.Connection,
    provider: LLMProvider | None,
    request: AskRequest,
    *,
    graph: CompiledStateGraph = ASK_GRAPH,
    thread_id: UUID | None = None,
    search_options: SearchOptions = SearchOptions(),
) -> AskResponse:
    """Stateless unless thread_id is given, in which case graph must have a checkpointer."""
    with tracer.start_as_current_span("ask") as span:
        span.set_attribute("asklineage.threaded", thread_id is not None)
        span.set_attribute("asklineage.input", _input_kind(request))
        if request.project:
            span.set_attribute("asklineage.project", request.project)
        response = _run(conn, provider, request, graph, thread_id, search_options)
        span.set_attribute("asklineage.kind", response.kind)
        span.set_attribute("asklineage.llm_attempts", len(response.attempts))
        return response


def _input_kind(request: AskRequest) -> str:
    if request.choose:
        return "choose"
    return "question" if request.question else "node"


def _run(
    conn: psycopg.Connection,
    provider: LLMProvider | None,
    request: AskRequest,
    graph: CompiledStateGraph,
    thread_id: UUID | None,
    search_options: SearchOptions,
) -> AskResponse:
    context = AskContext(conn, provider, resumable=thread_id is not None, search=search_options)
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
