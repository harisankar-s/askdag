"""Ask the golden set through run_ask, the entry point /api/ask uses, and score every turn.

Threads run on an in-memory checkpointer, as the flow tests do: the eval measures the flow,
not Postgres checkpoint storage or the API's thread locks, which have their own tests.
"""

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from asklineage.answer.parse import parse_question
from asklineage.answer.parse import system_prompt as parse_prompt
from asklineage.answer.prompt import system_prompt as highlights_prompt
from asklineage.api.flow import (
    ASK_GRAPH,
    build_graph,
    checkpoint_serde,
    run_ask,
    waiting_for_choice,
)
from asklineage.api.models import AskRequest
from asklineage.eval.golden import GoldenItem, GoldenSet, NameIndex, Turn
from asklineage.eval.score import TurnScore, score_turn
from asklineage.eval.truth import TruthGraph
from asklineage.lineage.graph import Direction
from asklineage.llm.base import LLMProvider

# Not scored: one call so the first scored question doesn't pay for loading the model.
WARM_UP_QUESTION = "What depends on stg_orders?"


@dataclass(frozen=True)
class EvalContext:
    conn: psycopg.Connection
    provider: LLMProvider | None
    graph: TruthGraph
    index: NameIndex
    owners: Mapping[str, str | None]
    max_named: int


@dataclass(frozen=True)
class RunInfo:
    started_at: str
    manifest_path: str
    manifest_sha256: str
    project: str
    invocation_id: str | None
    golden_sha: str
    golden_version: int
    phrasings: str
    llm_provider: str
    llm_model: str | None
    max_named: int
    asklineage_commit: str | None
    source_commit: str | None
    prompt_shas: dict[str, str]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def git_commit(path: Path) -> str | None:
    """HEAD of the checkout at path, with +dirty when it has uncommitted changes."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    dirty = git("status", "--porcelain").stdout.strip()
    return head.stdout.strip() + ("+dirty" if dirty else "")


def prompt_shas() -> dict[str, str]:
    return {
        "parse": _sha(parse_prompt()),
        "highlights_downstream": _sha(highlights_prompt(Direction.DOWNSTREAM)),
        "highlights_upstream": _sha(highlights_prompt(Direction.UPSTREAM)),
    }


def run_info(
    manifest_path: Path,
    manifest_bytes: bytes,
    golden: GoldenSet,
    *,
    llm_provider: str,
    llm_model: str | None,
    max_named: int,
    source_repo: Path | None = None,
) -> RunInfo:
    return RunInfo(
        started_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        project=golden.project,
        invocation_id=golden.invocation_id,
        golden_sha=golden.sha,
        golden_version=golden.version,
        phrasings=golden.phrasings,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_named=max_named,
        asklineage_commit=git_commit(Path(__file__).parent),
        source_commit=git_commit(source_repo) if source_repo else None,
        prompt_shas=prompt_shas(),
    )


def _ask_turn(
    context: EvalContext,
    item: GoldenItem,
    number: int,
    turn: Turn,
    *,
    flow: CompiledStateGraph,
    thread_id: UUID | None,
    previous_node: str | None,
) -> TurnScore:
    score = partial(
        score_turn,
        item,
        number,
        turn,
        llm=context.provider is not None,
        graph=context.graph,
        index=context.index,
        owners=context.owners,
        previous_node=previous_node,
    )
    if "choose" in turn.request and not waiting_for_choice(flow, thread_id):
        return score(None, blocked=True)
    request = AskRequest(**turn.request, thread_id=thread_id)
    started = time.perf_counter()
    try:
        response = run_ask(
            context.conn,
            context.provider,
            request,
            max_named=context.max_named,
            graph=flow,
            thread_id=thread_id,
        )
    except Exception as exc:
        # Keep going: one failing turn is reported as an error, not the end of the run.
        context.conn.rollback()
        seconds = time.perf_counter() - started
        return score(None, seconds=seconds, error=f"{type(exc).__name__}: {exc}")
    return score(response, seconds=time.perf_counter() - started)


def run_golden(
    context: EvalContext,
    golden: GoldenSet,
    on_turn: Callable[[TurnScore], None] | None = None,
) -> list[TurnScore]:
    if context.provider is not None:
        parse_question(context.provider, WARM_UP_QUESTION)
    threads = build_graph(InMemorySaver(serde=checkpoint_serde()))
    scores = []
    for item in golden.items:
        thread_id = uuid4() if item.thread else None
        flow = threads if item.thread else ASK_GRAPH
        previous_node = None
        for number, turn in enumerate(item.turns, start=1):
            score = _ask_turn(
                context,
                item,
                number,
                turn,
                flow=flow,
                thread_id=thread_id,
                previous_node=previous_node,
            )
            scores.append(score)
            if on_turn:
                on_turn(score)
            if score.kind == "lineage":
                previous_node = score.node_id
    return scores


def write_run(
    out: Path,
    info: RunInfo,
    golden: GoldenSet,
    scores: list[TurnScore],
    summary: dict,
    report: str,
) -> Path:
    stamp = info.started_at.replace(":", "").replace("-", "")
    model = (info.llm_model or "none").replace("/", "-").replace(":", "-")
    run_dir = out / f"{stamp}-{model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "golden.json").write_text(json.dumps(golden.to_dict(), indent=2, sort_keys=True))
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(asdict(score), sort_keys=True) + "\n" for score in scores)
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"run": asdict(info), **summary}, indent=2, sort_keys=True)
    )
    (run_dir / "report.md").write_text(report)
    return run_dir
