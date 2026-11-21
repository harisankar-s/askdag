"""Summarise scored turns per layer, and write the markdown report.

Shares are kept as (count, total). A percentage is printed only from MIN_FOR_PERCENT turns
up: 4/5 and 5/5 are too few to be a rate.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from asklineage.eval.golden import GoldenSet, describe_expect, describe_request
from asklineage.eval.run import RunInfo
from asklineage.eval.score import TurnScore

MIN_FOR_PERCENT = 20
MAX_FAILURES_SHOWN = 15
GROUPS = ("structured", "free_text", "thread")
OUTCOMES = ("correct", "acceptable", "safe_wrong", "silent_wrong", "blocked", "error")
RIGHT = ("correct", "acceptable")

type Share = tuple[int, int]


def _share(scores: Iterable[TurnScore], passed: Callable[[TurnScore], bool]) -> Share:
    chosen = list(scores)
    return sum(1 for s in chosen if passed(s)), len(chosen)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _stage_cost(scores: Sequence[TurnScore], stage: str) -> dict | None:
    calls = [[a for a in s.attempts if a["stage"] == stage] for s in scores]
    calls = [attempts for attempts in calls if attempts]
    if not calls:
        return None
    attempts = [a for call in calls for a in call]
    latency = [a["latency_ms"] for a in attempts if a["latency_ms"] is not None]
    tokens_in = [a["input_tokens"] for a in attempts if a["input_tokens"] is not None]
    tokens_out = [a["output_tokens"] for a in attempts if a["output_tokens"] is not None]
    return {
        "calls": len(calls),
        "attempts": len(attempts),
        "first_attempt_passed": sum(1 for call in calls if call[0]["passed"]),
        "retried": sum(1 for call in calls if len(call) > 1),
        "errors": sum(1 for a in attempts if a["error"]),
        "latency_ms_p50": _percentile(latency, 0.5),
        "latency_ms_p95": _percentile(latency, 0.95),
        "input_tokens_mean": round(sum(tokens_in) / len(tokens_in)) if tokens_in else None,
        "output_tokens_mean": round(sum(tokens_out) / len(tokens_out)) if tokens_out else None,
    }


def summarize(scores: Sequence[TurnScore], llm: bool) -> dict:
    scored = [s for s in scores if s.outcome != "blocked"]
    lineage = [s for s in scored if s.kind == "lineage"]
    structured = [s for s in scored if not s.free_text and "choose" not in s.request]
    suggest = [s for s in scored if s.expected.suggest and s.kind == "clarification"]
    expects_lineage = [s for s in scored if s.free_text and s.expected.kind == "lineage"]
    single = [s for s in scored if s.free_text and s.group == "free_text"]

    summary: dict = {
        "llm": llm,
        "turns": len(scores),
        "outcomes": {
            group: dict(Counter(s.outcome for s in scores if s.group == group)) for group in GROUPS
        },
        "silent_wrong_per_100": round(
            100 * sum(1 for s in scored if s.outcome == "silent_wrong") / len(scored), 1
        )
        if scored
        else None,
        "bugs": [
            f"{s.item_id} turn {s.turn}"
            for s in scores
            if s.outcome == "error" or s.lineage_ok is False or s.facts_ok is False
        ],
        "lineage": {
            "table": _share(lineage, lambda s: bool(s.lineage_ok)),
            "facts": _share(lineage, lambda s: bool(s.facts_ok)),
        },
        "resolution": {
            "exact": _share(
                (s for s in structured if s.expected.kind == "lineage"),
                lambda s: s.outcome == "correct",
            ),
            "ambiguous": _share(
                (s for s in structured if s.expected.reason == "ambiguous"),
                lambda s: s.outcome == "correct",
            ),
            "not_found": _share(
                (s for s in structured if s.expected.reason == "not_found"),
                lambda s: s.outcome == "correct",
            ),
            "suggested": _share(suggest, lambda s: bool(s.suggestion_rank)),
            "suggestion_ranks": sorted(s.suggestion_rank for s in suggest),
        },
        "highlights": dict(Counter(s.highlight_status for s in lineage)),
        "seconds_p50": _percentile([s.seconds for s in scored], 0.5),
        "seconds_p95": _percentile([s.seconds for s in scored], 0.95),
        "cost": {stage: _stage_cost(scored, stage) for stage in ("parse", "highlights")},
    }
    if not llm:
        summary["llm_off"] = _share(
            (s for s in scored if s.expected.reason == "llm_off"),
            lambda s: s.outcome == "correct",
        )
        return summary

    answered = [s for s in expects_lineage if s.kind == "lineage"]
    summary["reading"] = {
        "answered": _share(expects_lineage, lambda s: s.kind == "lineage"),
        "node": _share(answered, lambda s: bool(s.node_ok)),
        "direction": _share(answered, lambda s: bool(s.direction_ok)),
        "depth": _share(answered, lambda s: bool(s.depth_ok)),
        "all_three": _share(expects_lineage, lambda s: s.outcome == "correct"),
        "seen": _share((s for s in single if "seen" in s.tags), lambda s: s.outcome in RIGHT),
        "novel": _share((s for s in single if "novel" in s.tags), lambda s: s.outcome in RIGHT),
    }
    summary["abstention"] = {
        "off_topic_abstained": _share(
            (s for s in scored if s.expected.reason == "not_lineage"),
            lambda s: s.kind == "abstain",
        ),
        "false_abstains": _share(expects_lineage, lambda s: s.kind == "abstain"),
    }
    summary["follow_ups"] = {}
    for value in ("yes", "no", "either"):
        turns = [s for s in scored if s.same_node == value]
        summary["follow_ups"][value] = {
            "right": _share(turns, lambda s: s.outcome in RIGHT),
            "reused_previous": sum(1 for s in turns if s.reused_previous),
        }
    return summary


def fmt(share: Share) -> str:
    count, total = share
    if not total:
        return "–"
    text = f"{count}/{total}"
    return f"{text} ({count / total:.0%})" if total >= MIN_FOR_PERCENT else text


def _table(header: tuple[str, ...], rows: Iterable[tuple]) -> list[str]:
    lines = [f"| {' | '.join(header)} |", f"|{'---|' * len(header)}"]
    lines += [f"| {' | '.join(str(cell) for cell in row)} |" for row in rows]
    return lines


def _got(score: TurnScore) -> str:
    if score.outcome == "error":
        return "error"
    if score.kind == "lineage":
        node = score.response["node"]["display_name"]
        return f"lineage {node} {score.direction} {score.depth}"
    return f"{score.kind} {score.reason}"


def render(info: RunInfo, golden: GoldenSet, summary: dict, scores: Sequence[TurnScore]) -> str:
    model = info.llm_model or "no LLM"
    run_rows = [
        ("golden set", f"`{info.golden_sha}` (version {info.golden_version}, {info.phrasings})"),
        ("project", f"{info.project}, invocation `{info.invocation_id}`"),
        ("manifest sha256", f"`{info.manifest_sha256[:16]}`"),
        ("source commit", f"`{info.source_commit}`" if info.source_commit else "not recorded"),
        ("asklineage commit", f"`{info.asklineage_commit}`"),
        ("LLM", f"{info.llm_provider}, {model}, max named nodes {info.max_named}"),
        ("prompt shas", ", ".join(f"{k} `{v}`" for k, v in info.prompt_shas.items())),
    ]
    lines = [f"# asklineage eval: {model}, {info.started_at}", ""]
    lines += _table(("Run", ""), run_rows)

    outcomes = summary["outcomes"]
    lines += ["", "## Outcomes", ""]
    rows = []
    for outcome in OUTCOMES:
        counts = [outcomes[group].get(outcome, 0) for group in GROUPS]
        rows.append((outcome, *counts, sum(counts)))
    lines += _table(("outcome", *GROUPS, "all"), rows)
    lines += [
        "",
        f"Silent wrong answers per 100 turns: {summary['silent_wrong_per_100']}. "
        "Bugs (errors, or a table or facts that differ from the manifest): "
        f"{', '.join(summary['bugs']) or 'none'}.",
    ]

    lineage, resolution = summary["lineage"], summary["resolution"]
    lines += ["", "## Lineage and facts (must be 100%)", ""]
    lines += _table(
        ("check", "right"),
        [("table and truncated flag", fmt(lineage["table"])), ("facts", fmt(lineage["facts"]))],
    )
    lines += ["", "## Name resolution (structured requests)", ""]
    lines += _table(
        ("case", "right"),
        [
            ("exact name, id or case", fmt(resolution["exact"])),
            ("ambiguous: exactly the candidates", fmt(resolution["ambiguous"])),
            ("not found", fmt(resolution["not_found"])),
            ("expected node suggested (top 5)", fmt(resolution["suggested"])),
        ],
    )
    lines.append(f"\nSuggestion ranks (0 = not suggested): {resolution['suggestion_ranks']}")

    if not summary["llm"]:
        lines += [
            "",
            "## Without an LLM",
            "",
            f"Free-text turns that abstained with `llm_off`: {fmt(summary['llm_off'])}. "
            "Question reading, abstention, follow-ups and highlights need an LLM run.",
        ]
    else:
        reading, abstention = summary["reading"], summary["abstention"]
        lines += ["", "## Question reading (free text expecting a lineage answer)", ""]
        lines += _table(
            ("check", "right"),
            [
                ("answered with lineage", fmt(reading["answered"])),
                ("right node, of answered", fmt(reading["node"])),
                ("right direction, of answered", fmt(reading["direction"])),
                ("right depth, of answered", fmt(reading["depth"])),
                ("all three", fmt(reading["all_three"])),
                ("single questions, seen phrasing", fmt(reading["seen"])),
                ("single questions, novel phrasing", fmt(reading["novel"])),
            ],
        )
        lines += ["", "## Abstention", ""]
        lines += _table(
            ("check", "result"),
            [
                ("off-topic questions that abstained", fmt(abstention["off_topic_abstained"])),
                ("lineage questions that abstained", fmt(abstention["false_abstains"])),
            ],
        )
        lines += ["", "## Follow-ups", ""]
        lines += _table(
            ("should reuse the previous node", "right", "reused it wrongly"),
            [
                (value, fmt(entry["right"]), entry["reused_previous"])
                for value, entry in summary["follow_ups"].items()
            ],
        )

    lines += ["", "## Highlights", ""]
    statuses = summary["highlights"]
    lines.append(", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())) or "none")

    lines += ["", "## Cost", ""]
    cost_rows = [
        (
            stage,
            c["calls"],
            c["attempts"],
            c["first_attempt_passed"],
            c["retried"],
            c["errors"],
            c["latency_ms_p50"],
            c["latency_ms_p95"],
            c["input_tokens_mean"],
            c["output_tokens_mean"],
        )
        for stage, c in summary["cost"].items()
        if c
    ]
    if cost_rows:
        header = ("stage", "calls", "attempts", "passed first", "retried", "errors")
        header += ("p50 ms", "p95 ms", "mean in", "mean out")
        lines += _table(header, cost_rows)
    else:
        lines.append("No LLM calls.")
    lines.append(
        f"\nWall time per turn: p50 {summary['seconds_p50']} s, p95 {summary['seconds_p95']} s."
    )

    failures = [s for s in scores if s.outcome not in (*RIGHT, "blocked")]
    lines += ["", f"## Not right ({len(failures)})", ""]
    for score in failures[:MAX_FAILURES_SHOWN]:
        expected = describe_expect(score.expected, golden.labels)
        lines.append(
            f"- `{score.item_id}` turn {score.turn} ({score.outcome}): "
            f"{describe_request(score.request)} → expected {expected}; got {_got(score)}. "
            f"{'; '.join(score.problems)}"
        )
    if len(failures) > MAX_FAILURES_SHOWN:
        lines.append(f"- … {len(failures) - MAX_FAILURES_SHOWN} more in results.jsonl")
    blocked = [f"{s.item_id} turn {s.turn}" for s in scores if s.outcome == "blocked"]
    if blocked:
        lines.append(f"\nBlocked (not scored): {', '.join(blocked)}")
    return "\n".join(lines) + "\n"
