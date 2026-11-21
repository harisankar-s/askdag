"""The /api/ask contract: one request shape, and five kinds of answer told apart by `kind`."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from asklineage.answer.facts import LineageFacts
from asklineage.answer.generate import Attempt
from asklineage.answer.lineage_answer import HighlightStatus
from asklineage.answer.parse import MAX_QUESTION_CHARS
from asklineage.lineage.graph import MAX_DEPTH, Direction, LineageNode
from asklineage.lineage.resolve import Candidate


class AskRequest(BaseModel):
    """A free-text question (needs an LLM), or a node with optional direction and depth.

    Stateless by default. start_thread opens a conversation; thread_id continues one; choose
    (with thread_id, on its own) answers the thread's pending clarification with a unique_id.
    """

    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(default=None, min_length=1, max_length=MAX_QUESTION_CHARS)
    node: str | None = Field(default=None, min_length=1, max_length=300)
    direction: Direction | None = None
    depth: int | None = Field(default=None, ge=1, le=MAX_DEPTH)
    project: str | None = None
    start_thread: bool = False
    thread_id: UUID | None = None
    choose: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def question_or_node(self) -> Self:
        if self.start_thread and self.thread_id is not None:
            raise ValueError("pass start_thread or thread_id, not both")
        if self.choose is not None:
            if self.thread_id is None:
                raise ValueError("choose answers a thread's clarification; pass its thread_id")
            asked = (self.question, self.node, self.direction, self.depth, self.project)
            if any(field is not None for field in asked):
                raise ValueError("choose is sent with thread_id only")
            return self
        if self.node is None:
            if self.question is None:
                raise ValueError("pass a question, or a node with optional direction and depth")
            if self.direction is not None or self.depth is not None:
                raise ValueError("direction and depth are only used with node")
        return self


class NodeInfo(BaseModel):
    unique_id: str
    name: str
    source_name: str | None
    display_name: str
    resource_type: str

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> Self:
        return cls(
            unique_id=candidate.unique_id,
            name=candidate.name,
            source_name=candidate.source_name,
            display_name=candidate.display_name,
            resource_type=candidate.resource_type,
        )


class CandidateInfo(NodeInfo):
    score: float  # 1.0 for exact matches; pg_trgm similarity for suggestions

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> Self:
        return cls(**NodeInfo.from_candidate(candidate).model_dump(), score=candidate.score)


class NodeRow(BaseModel):
    unique_id: str
    display_name: str
    resource_type: str
    depth: int
    owner: str | None
    owner_source: str | None

    @classmethod
    def from_node(cls, node: LineageNode) -> Self:
        return cls(
            unique_id=node.unique_id,
            display_name=node.display_name,
            resource_type=node.resource_type,
            depth=node.depth,
            owner=node.owner,
            owner_source=node.owner_source,
        )


class EdgeInfo(BaseModel):
    parent_id: str
    child_id: str


class OwnerInfo(BaseModel):
    owner: str
    owner_source: str
    owner_source_meaning: str
    nodes: int


class FactsInfo(BaseModel):
    total: int
    by_depth: dict[int, int]
    by_type: dict[str, int]
    owners: list[OwnerInfo]
    unowned: int
    truncated: bool

    @classmethod
    def from_facts(cls, facts: LineageFacts) -> Self:
        return cls(
            total=facts.total,
            by_depth=facts.by_depth,
            by_type=facts.by_type,
            owners=[
                OwnerInfo(
                    owner=group.owner,
                    owner_source=group.owner_source,
                    owner_source_meaning=group.owner_source_meaning,
                    nodes=group.nodes,
                )
                for group in facts.owners
            ],
            unowned=facts.unowned,
            truncated=facts.truncated,
        )


class HighlightInfo(BaseModel):
    text: str  # with owner tags added by code
    upstream: str
    downstream: str
    hops: Literal["direct", "indirect"]


class AttemptInfo(BaseModel):
    stage: Literal["parse", "highlights"]
    passed: bool
    error: str | None = None
    violations: list[str] = []
    rules: list[str] = []  # the rule each violation broke, in the same order
    reply: str | None = None  # what the model returned, so a reading can be traced
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @classmethod
    def from_attempt(cls, stage: Literal["parse", "highlights"], attempt: Attempt) -> Self:
        return cls(
            stage=stage,
            passed=attempt.passed,
            error=attempt.error,
            violations=[str(violation) for violation in attempt.violations],
            rules=[violation.rule for violation in attempt.violations],
            reply=attempt.reply,
            model=attempt.model,
            latency_ms=attempt.latency_ms,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
        )


class LineageResponse(BaseModel):
    kind: Literal["lineage"] = "lineage"
    node: NodeInfo
    direction: Direction
    depth: int
    summary: str
    facts: FactsInfo
    highlights: list[HighlightInfo]
    highlight_status: HighlightStatus
    table: list[NodeRow]
    edges: list[EdgeInfo]  # parent -> child in both directions; may reference the root
    attempts: list[AttemptInfo] = []
    thread_id: UUID | None = None


class ClarificationResponse(BaseModel):
    kind: Literal["clarification"] = "clarification"
    reason: Literal["ambiguous", "not_found", "invalid_choice"]
    query: str
    message: str
    candidates: list[CandidateInfo]  # exact matches when ambiguous; suggestions when not found
    # True when the thread is waiting for `choose`; otherwise ask again.
    resumable: bool = False
    attempts: list[AttemptInfo] = []
    thread_id: UUID | None = None


class OwnerResponse(BaseModel):
    """Written by code from the nodes table; no LLM text."""

    kind: Literal["owner"] = "owner"
    node: NodeInfo
    owner: str | None  # None: no owner recorded in dbt or the owners file
    owner_source: str | None
    owner_source_meaning: str | None
    message: str
    attempts: list[AttemptInfo] = []
    thread_id: UUID | None = None


class SearchHit(BaseModel):
    unique_id: str
    display_name: str
    resource_type: str
    owner: str | None
    owner_source: str | None
    score: float  # cosine similarity for semantic; ts_rank for keyword; not comparable
    match: str  # the chunk kind that matched best: summary, columns or sql
    via: str | None = None  # the staging model or source that lifted this mart


class SearchResponse(BaseModel):
    """Nodes whose text matches what the question describes, ranked; written by code."""

    kind: Literal["search"] = "search"
    mode: Literal["semantic", "keyword"]
    model: str | None  # the vectors' tag in semantic mode
    hits: list[SearchHit]
    # True when the best match is weak (semantic: top score under the cutoff) or keyword
    # search ran; the data asked about may not be in the project.
    low_confidence: bool
    message: str
    note: str | None = None
    attempts: list[AttemptInfo] = []
    thread_id: UUID | None = None


AbstainReason = Literal["out_of_scope", "not_understood", "llm_off", "llm_unavailable"]


class AbstainResponse(BaseModel):
    kind: Literal["abstain"] = "abstain"
    reason: AbstainReason
    message: str
    attempts: list[AttemptInfo] = []
    thread_id: UUID | None = None


AskResponse = Annotated[
    LineageResponse | OwnerResponse | SearchResponse | ClarificationResponse | AbstainResponse,
    Field(discriminator="kind"),
]
