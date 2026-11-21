"""Ask for typed output, check it in code, retry once with the problems, or give up.

Not lineage-specific: any answer type supplies its own prompt, schema and check.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from asklineage.llm.base import LLMError, LLMOutputError, LLMProvider

MAX_ATTEMPTS = 2
MAX_DETAIL_CHARS = 500
MAX_REPLY_CHARS = 4000


@dataclass(frozen=True)
class Violation:
    claim: int | None  # 1-based position in the reply; None for rules about the whole reply
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"claim {self.claim}: {self.detail}" if self.claim else self.detail


@dataclass(frozen=True)
class Attempt:
    violations: tuple[Violation, ...] = ()
    error: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reply: str | None = None  # the raw or re-serialised reply, sent back on retry
    output: BaseModel | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.violations


@dataclass(frozen=True)
class Generated[T: BaseModel]:
    output: T | None  # None when no attempt passed the check
    attempts: tuple[Attempt, ...]


def generate_checked[T: BaseModel](
    provider: LLMProvider,
    *,
    system: str,
    prompt: str,
    output: type[T],
    check: Callable[[T], Sequence[Violation]],
    max_attempts: int = MAX_ATTEMPTS,
) -> Generated[T]:
    attempts: list[Attempt] = []
    current = prompt
    for _ in range(max_attempts):
        try:
            completion = provider.complete_json(system=system, prompt=current, output=output)
        except LLMOutputError as exc:
            violation = Violation(None, "invalid_output", exc.detail[:MAX_DETAIL_CHARS])
            attempts.append(Attempt(violations=(violation,), reply=exc.raw[:MAX_REPLY_CHARS]))
        except LLMError as exc:
            # The client has already retried transport errors; another attempt won't help.
            attempts.append(Attempt(error=str(exc)))
            break
        else:
            attempt = Attempt(
                violations=tuple(check(completion.data)),
                model=completion.model,
                latency_ms=completion.latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                reply=completion.data.model_dump_json(exclude_none=True),
                output=completion.data,
            )
            attempts.append(attempt)
            if attempt.passed:
                return Generated(completion.data, tuple(attempts))
        current = prompt + retry_feedback(attempts[-1])
    return Generated(None, tuple(attempts))


def retry_feedback(attempt: Attempt) -> str:
    # Showing the reply lets the model edit it; without it, a rewrite drops claims that were right.
    problems = "\n".join(f"- {violation}" for violation in attempt.violations)
    return (
        f"\n\nYour previous reply was:\n{attempt.reply}\n\n"
        f"It broke these rules:\n{problems}\n\n"
        "Reply again with the whole corrected reply. Fix every problem listed, "
        "and keep the claims that had no problems."
    )
