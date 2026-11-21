"""A lineage answer: the table and summary from code, plus LLM highlights that passed the check."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from asklineage.answer.claims import Highlights, RelationClaim
from asklineage.answer.facts import LineageFacts, build_facts
from asklineage.answer.generate import Attempt, generate_checked
from asklineage.answer.grounding import check
from asklineage.answer.prompt import build_prompt, system_prompt
from asklineage.answer.template import template_summary
from asklineage.lineage.graph import Lineage
from asklineage.lineage.resolve import Candidate
from asklineage.llm.base import LLMProvider


class HighlightStatus(StrEnum):
    PASSED = "passed"
    PASSED_AFTER_RETRY = "passed_after_retry"
    REJECTED = "rejected"  # every attempt failed the check, or the LLM errored
    OFF = "off"  # no LLM configured
    SKIPPED = "skipped"  # nothing to highlight


@dataclass(frozen=True)
class LineageAnswer:
    lineage: Lineage
    facts: LineageFacts
    summary: str
    highlights: tuple[RelationClaim, ...]
    status: HighlightStatus
    attempts: tuple[Attempt, ...]

    @property
    def highlight_texts(self) -> tuple[str, ...]:
        owners = {node.display_name: node.owner for node in self.lineage.nodes}
        return tuple(tag_owners(claim.text, owners) for claim in self.highlights)


def tag_owners(text: str, owners: dict[str, str | None]) -> str:
    """Add each node's owner after its first mention; the root has none in the result."""
    tagged: set[str] = set()

    def tag(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in owners or name in tagged:
            return match.group(0)
        tagged.add(name)
        return f"`{name}` ({owners[name] or 'no owner recorded'})"

    return re.sub(r"`([^`]+)`", tag, text)


def answer_lineage(
    result: Lineage,
    root: Candidate,
    provider: LLMProvider | None,
    *,
    max_named: int,
    known_names: Iterable[str] = (),
) -> LineageAnswer:
    facts = build_facts(result, root, max_named)
    summary = template_summary(facts)

    def answer(
        status: HighlightStatus,
        highlights: tuple[RelationClaim, ...] = (),
        attempts: tuple[Attempt, ...] = (),
    ) -> LineageAnswer:
        return LineageAnswer(result, facts, summary, highlights, status, attempts)

    if provider is None:
        return answer(HighlightStatus.OFF)
    if facts.total == 0:
        return answer(HighlightStatus.SKIPPED)

    names = frozenset(known_names)
    generated = generate_checked(
        provider,
        system=system_prompt(facts.direction),
        prompt=build_prompt(facts),
        output=Highlights,
        check=lambda highlights: check(highlights, facts, result, names),
    )
    if generated.output is None:
        return answer(HighlightStatus.REJECTED, attempts=generated.attempts)
    passed_first = len(generated.attempts) == 1
    status = HighlightStatus.PASSED if passed_first else HighlightStatus.PASSED_AFTER_RETRY
    return answer(status, tuple(generated.output.claims), generated.attempts)
