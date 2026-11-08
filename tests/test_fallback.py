"""Фолбек між Gemini-ключами: порядок, перемикання, проксі обох методів.

Без мережі й без ключа: ланцюжок — на стабах, білдер — на env-словниках.
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm_client import (
    DailyQuotaExceeded,
    FallbackClassifier,
    GeminiClient,
    LLMClassificationError,
    build_classifier,
)


class _Stub:
    def __init__(self, name: str, *, exc: Exception | None = None) -> None:
        self.name, self.exc, self.classify_calls, self.link_calls = name, exc, 0, 0

    async def classify(self, request_id: str, raw_text: str, max_retries: int = 1):
        self.classify_calls += 1
        if self.exc is not None:
            raise self.exc
        return self.name

    async def link_requests(self, items: list[dict]):
        self.link_calls += 1
        if self.exc is not None:
            raise self.exc
        return [self.name]


def test_classify_first_working_key_short_circuits() -> None:
    a, b = _Stub("a"), _Stub("b")
    out = asyncio.run(FallbackClassifier([("a", a), ("b", b)]).classify("r1", "txt"))
    assert out == "a" and (a.classify_calls, b.classify_calls) == (1, 0)


def test_classify_falls_through_on_quota() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=DailyQuotaExceeded("per-day"))
    b = _Stub("gemini:GEMINI_API_KEY_2")
    out = asyncio.run(FallbackClassifier([("gemini:GEMINI_API_KEY", a),
                                          ("gemini:GEMINI_API_KEY_2", b)]).classify("r", "t"))
    assert out == "gemini:GEMINI_API_KEY_2" and (a.classify_calls, b.classify_calls) == (1, 1)


def test_link_requests_is_also_proxied_with_fallback() -> None:
    a = _Stub("a", exc=LLMClassificationError("boom"))
    b = _Stub("b")
    out = asyncio.run(FallbackClassifier([("a", a), ("b", b)]).link_requests([{"x": 1}]))
    assert out == ["b"] and (a.link_calls, b.link_calls) == (1, 1)


def test_all_keys_failing_raises_naming_each() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=LLMClassificationError("x"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=DailyQuotaExceeded("y"))
    with pytest.raises(LLMClassificationError) as e:
        asyncio.run(FallbackClassifier([("gemini:GEMINI_API_KEY", a),
                                        ("gemini:GEMINI_API_KEY_2", b)]).classify("r", "t"))
    assert "GEMINI_API_KEY" in str(e.value) and "GEMINI_API_KEY_2" in str(e.value)


def test_build_classifier_orders_dedups_single_and_none() -> None:
    members = build_classifier(env={"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_2": "k2",
                                    "GEMINI_API_KEY_3": "k3"}).members
    assert [n for n, _ in members] == ["gemini:GEMINI_API_KEY", "gemini:GEMINI_API_KEY_2",
                                       "gemini:GEMINI_API_KEY_3"]
    assert isinstance(build_classifier(env={"GEMINI_API_KEY": "only"}), GeminiClient)
    assert build_classifier(env={}) is None
    assert isinstance(build_classifier(env={"GEMINI_API_KEY": "s", "GEMINI_API_KEY_2": "s"}),
                      GeminiClient)
