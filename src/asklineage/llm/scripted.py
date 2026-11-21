"""A provider that replays canned replies, so answer and grounding code can be tested offline."""

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel

from asklineage.llm.base import Completion, LLMError, parse_output

type Reply = str | dict | LLMError


@dataclass(frozen=True)
class ScriptedCall:
    system: str
    prompt: str
    output: type[BaseModel]


class ScriptedProvider:
    def __init__(self, replies: Iterable[Reply], model: str = "scripted") -> None:
        self.model = model
        self.calls: list[ScriptedCall] = []
        self._replies: deque[Reply] = deque(replies)

    def complete_json[T: BaseModel](
        self, *, system: str, prompt: str, output: type[T]
    ) -> Completion[T]:
        self.calls.append(ScriptedCall(system, prompt, output))
        if not self._replies:
            raise LLMError(f"ScriptedProvider has no replies left (call {len(self.calls)})")
        reply = self._replies.popleft()
        if isinstance(reply, LLMError):
            raise reply
        # Raw strings use the real providers' parser, so malformed replies can be tested.
        raw = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(parse_output(raw, output), self.model, None, None, 0)
