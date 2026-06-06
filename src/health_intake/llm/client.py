"""LLM client protocol, OpenAI implementation, and a scripted fake for tests."""

import logging
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from health_intake.engine.extraction import FieldExtraction
from health_intake.models.conversation import ChatMessage

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction: ...

    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str: ...


def _to_openai(system: str, messages: Sequence[ChatMessage]) -> list[dict]:
    payload: list[dict] = [{"role": "system", "content": system}]
    payload.extend({"role": m.role, "content": m.content} for m in messages if m.role != "system")
    return payload


class OpenAIClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction:
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=_to_openai(system, messages),
            response_format=FieldExtraction,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        return parsed or FieldExtraction()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str:
        convo = _to_openai(system, messages)
        convo.append({"role": "system", "content": f"Situation: {situation}"})
        completion = self._client.chat.completions.create(
            model=self._model, messages=convo, temperature=0.3
        )
        return completion.choices[0].message.content or ""


class FakeLLMClient:
    """Deterministic scripted client for tests. Pops scripted values per call."""

    def __init__(self, extractions: list[FieldExtraction], replies: list[str]) -> None:
        self._extractions = list(extractions)
        self._replies = list(replies)

    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction:
        return self._extractions.pop(0) if self._extractions else FieldExtraction()

    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str:
        return self._replies.pop(0) if self._replies else "(no reply)"
