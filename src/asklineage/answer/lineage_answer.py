"""A lineage answer, all from code: the table, the summary and the highlights (ADR 0019)."""

import re
from dataclasses import dataclass

from asklineage.answer.facts import LineageFacts, build_facts
from asklineage.answer.highlights import Highlight, highlights
from asklineage.answer.template import template_summary
from asklineage.lineage.graph import Lineage
from asklineage.lineage.resolve import Candidate


@dataclass(frozen=True)
class LineageAnswer:
    lineage: Lineage
    facts: LineageFacts
    summary: str
    highlights: tuple[Highlight, ...]

    @property
    def highlight_texts(self) -> tuple[str, ...]:
        owners = {node.display_name: node.owner for node in self.lineage.nodes}
        return tuple(tag_owners(h.text, owners) for h in self.highlights)


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


def answer_lineage(result: Lineage, root: Candidate) -> LineageAnswer:
    facts = build_facts(result, root)
    return LineageAnswer(result, facts, template_summary(facts), highlights(facts, result.nodes))
