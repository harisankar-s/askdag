"""Hand-written phrasings for golden questions, tagged by how close they are to the prompt.

`seen` marks a phrasing that mirrors one of the parse prompt's few-shot examples. The report
splits accuracy by the tag: a gap between seen and novel means the examples are being
matched, not the rules followed.
"""

from dataclasses import dataclass

NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class Template:
    text: str
    seen: bool = False

    def render(self, name: str, depth: int | None = None) -> str:
        return self.text.format(name=name, hops=hops(depth), levels=levels(depth))


def hops(depth: int | None) -> str:
    if depth is None:
        return ""
    return f"{depth} hop{'s' if depth > 1 else ''}"


def levels(depth: int | None) -> str:
    if depth is None:
        return ""
    count = NUMBER_WORDS[depth - 1] if depth <= len(NUMBER_WORDS) else str(depth)
    return f"{count} level{'s' if depth > 1 else ''}"


TEMPLATES: dict[str, tuple[Template, ...]] = {
    "down": (
        Template("What depends on {name}?", seen=True),
        Template("If I change {name}, what breaks downstream?"),
        Template("Which models read from `{name}`?"),
        Template("what uses {name}"),
    ),
    "down_depth": (
        Template("What depends on {name}, {levels} down?", seen=True),
        Template("Show me everything downstream of {name}, up to {hops}."),
        Template("Which nodes are within {hops} downstream of `{name}`?"),
    ),
    "up": (
        Template("Where does {name} get its data from?", seen=True),
        Template("What does {name} depend on?"),
        Template("Trace {name} back to its sources."),
        Template("What feeds into `{name}`?"),
    ),
    "up_depth": (
        Template("Where does {name} get its data from, {levels} up?", seen=True),
        Template("List the upstream dependencies of {name}, {hops} deep."),
        Template("What are the parents of {name} up to {levels}?"),
    ),
    "source_down": (
        Template("Which teams are affected if the {name} source changes?", seen=True),
        Template("What is built on top of the {name} source?"),
    ),
}
