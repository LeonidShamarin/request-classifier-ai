"""
Тести стійкості LLM-клієнта — без мережі й без API-ключа.

Реальний genai.Client не створюється: конструктор обходиться через
object.__new__ + ручна підстановка полів. Нас цікавить лише поведінка
_call_model / classify навколо помилок, а не сам SDK.
"""

import asyncio
import json

import pytest
from google.genai import errors as genai_errors

from src.llm_client import (
    MAX_SERVER_RETRY_DELAY,
    TRANSPORT_MAX_ATTEMPTS,
    DailyQuotaExceeded,
    GeminiClient,
    LLMClassificationError,
    _is_retryable,
    _parse_json,
    _RateLimiter,
    _server_retry_delay,
)

VALID_PAYLOAD = {
    "category": "автоматизація",
    "target_department": "маркетинг",
    "priority": "high",
    "short_summary": "Автоматизувати щотижневий звіт.",
    "requested_actions": ["Налаштувати вивантаження"],
    "needs_clarification": False,
    "clarification_reason": None,
    "language": "uk",
    "confidence": 0.9,
}


async def _no_sleep(_delay):
    """Заміна asyncio.sleep — щоб тести backoff-у не чекали реальні секунди.
    Не викликає asyncio.sleep(0), бо він уже підмінений на цю ж функцію."""
    return None


def _client_without_network() -> GeminiClient:
    client = object.__new__(GeminiClient)
    client._client = None
    client._model_name = "test-model"
    client._config = None
    client._limiter = _RateLimiter(0)  # rpm=0 — обмеження вимкнене, тести не чекають
    client.daily_quota_hit = False
    return client


def _fake_sdk(generate_fn):
    """Мінімальна підробка genai.Client: лише ланцюжок .aio.models.generate_content."""
    models = type("M", (), {"generate_content": staticmethod(generate_fn)})()
    aio = type("A", (), {"models": models})()
    return type("C", (), {"aio": aio})()


def _client_error(code: int) -> genai_errors.ClientError:
    exc = genai_errors.ClientError.__new__(genai_errors.ClientError)
    Exception.__init__(exc, f"{code} error")
    exc.code = code
    return exc


def _server_error() -> genai_errors.ServerError:
    exc = genai_errors.ServerError.__new__(genai_errors.ServerError)
    Exception.__init__(exc, "503 unavailable")
    exc.code = 503
    return exc


# --- _is_retryable -----------------------------------------------------------


def test_rate_limit_is_retryable():
    assert _is_retryable(_client_error(429)) is True


def test_bad_request_is_not_retryable():
    # 400 не «розсмокчеться» — повторювати безглуздо, треба падати одразу.
    assert _is_retryable(_client_error(400)) is False


def test_server_error_is_retryable():
    assert _is_retryable(_server_error()) is True


def test_validation_style_error_is_not_retryable():
    assert _is_retryable(ValueError("bad json")) is False


def _daily_quota_error() -> genai_errors.ClientError:
    exc = genai_errors.ClientError.__new__(genai_errors.ClientError)
    Exception.__init__(
        exc,
        "429 RESOURCE_EXHAUSTED. {'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'retryDelay': '25s'}",
    )
    exc.code = 429
    return exc


def test_daily_quota_is_not_retryable():
    """Пастка: сервер радить retryDelay=25s навіть коли квота денна.
    Слухати цю пораду — значить крутити безнадійні retry годинами."""
    assert _is_retryable(_daily_quota_error()) is False


def test_per_minute_quota_stays_retryable():
    exc = _client_error(429)
    Exception.__init__(
        exc,
        "429 RESOURCE_EXHAUSTED. {'quotaId': "
        "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'retryDelay': '46s'}",
    )
    assert _is_retryable(exc) is True


@pytest.mark.asyncio
async def test_daily_quota_trips_circuit_breaker(monkeypatch):
    """Після денної квоти решта запитів має падати миттєво, без мережі."""
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client = _client_without_network()
    calls = {"n": 0}

    async def daily_quota(**kwargs):
        calls["n"] += 1
        raise _daily_quota_error()

    client._client = _fake_sdk(daily_quota)

    with pytest.raises(DailyQuotaExceeded):
        await client._call_model("prompt")
    assert client.daily_quota_hit is True
    assert calls["n"] == 1  # без повторів

    # наступний виклик уже навіть не торкається мережі
    with pytest.raises(DailyQuotaExceeded):
        await client._call_model("prompt")
    assert calls["n"] == 1


# --- retryDelay від сервера --------------------------------------------------


def test_server_retry_delay_is_extracted():
    """Gemini кладе RetryInfo у тіло 429 — воно точніше за наш сліпий backoff."""
    exc = _client_error(429)
    Exception.__init__(exc, "429 RESOURCE_EXHAUSTED. {'retryDelay': '46.5s'}")
    assert _server_retry_delay(exc) == 46.5


def test_server_retry_delay_is_capped():
    exc = _client_error(429)
    Exception.__init__(exc, "429 RESOURCE_EXHAUSTED. {'retryDelay': '3600s'}")
    assert _server_retry_delay(exc) == MAX_SERVER_RETRY_DELAY


def test_server_retry_delay_absent_returns_none():
    assert _server_retry_delay(ValueError("no retry info here")) is None


# --- backoff у _call_model ---------------------------------------------------


@pytest.mark.asyncio
async def test_call_model_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client = _client_without_network()
    calls = {"n": 0}

    async def fake_generate(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _client_error(429)

        class Resp:
            text = json.dumps(VALID_PAYLOAD)

        return Resp()

    client._client = _fake_sdk(fake_generate)

    raw = await client._call_model("prompt")
    assert json.loads(raw)["category"] == "автоматизація"
    assert calls["n"] == 3  # дві 429, потім успіх


@pytest.mark.asyncio
async def test_call_model_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client = _client_without_network()
    calls = {"n": 0}

    async def always_429(**kwargs):
        calls["n"] += 1
        raise _client_error(429)

    client._client = _fake_sdk(always_429)

    with pytest.raises(LLMClassificationError):
        await client._call_model("prompt")
    assert calls["n"] == TRANSPORT_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_call_model_does_not_retry_non_retryable(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client = _client_without_network()
    calls = {"n": 0}

    async def always_400(**kwargs):
        calls["n"] += 1
        raise _client_error(400)

    client._client = _fake_sdk(always_400)

    with pytest.raises(genai_errors.ClientError):
        await client._call_model("prompt")
    assert calls["n"] == 1  # жодного повтору


# --- self-repair у classify --------------------------------------------------


@pytest.mark.asyncio
async def test_classify_self_repairs_invalid_json():
    client = _client_without_network()
    responses = ["це взагалі не json", json.dumps(VALID_PAYLOAD)]
    seen_prompts = []

    async def fake_call(prompt):
        seen_prompts.append(prompt)
        return responses.pop(0)

    client._call_model = fake_call

    result = await client.classify("REQ-1", "текст", max_retries=1)
    assert result.retries_used == 1
    assert result.classification.priority == "high"
    # друга спроба має містити текст помилки — це і є "самопочинка"
    assert "невалідною" in seen_prompts[1]


@pytest.mark.asyncio
async def test_classify_raises_after_exhausting_repairs():
    client = _client_without_network()

    async def always_garbage(prompt):
        return "{}"  # валідний JSON, але не проходить Pydantic

    client._call_model = always_garbage

    with pytest.raises(LLMClassificationError):
        await client.classify("REQ-1", "текст", max_retries=1)


# --- прохід зв'язування ------------------------------------------------------


ITEMS = [
    {"id": "REQ-001", "category": "звіт/аналітика", "short_summary": "Звіт по Google Ads"},
    {"id": "REQ-013", "category": "звіт/аналітика", "short_summary": "Той самий звіт"},
]


@pytest.mark.asyncio
async def test_link_requests_returns_valid_links():
    client = _client_without_network()

    async def fake_call(prompt, linking=False):
        assert linking is True
        assert "REQ-013" in prompt  # у промпт іде саме перелік запитів
        return json.dumps(
            {
                "links": [
                    {
                        "request_id": "REQ-013",
                        "related_request_id": "REQ-001",
                        "reason": "той самий звіт по Google Ads",
                    }
                ]
            }
        )

    client._call_model = fake_call
    links = await client.link_requests(ITEMS)
    assert len(links) == 1
    assert links[0].request_id == "REQ-013"


@pytest.mark.asyncio
async def test_link_requests_drops_hallucinated_and_self_links():
    client = _client_without_network()

    async def fake_call(prompt, linking=False):
        return json.dumps(
            {
                "links": [
                    # id, якого немає в інбоксі
                    {"request_id": "REQ-999", "related_request_id": "REQ-001", "reason": "x"},
                    # запит сам на себе
                    {"request_id": "REQ-001", "related_request_id": "REQ-001", "reason": "x"},
                ]
            }
        )

    client._call_model = fake_call
    assert await client.link_requests(ITEMS) == []


@pytest.mark.asyncio
async def test_link_requests_is_non_fatal_on_failure():
    """Зв'язки — збагачення, а не основний результат: помилка не валить прогін."""
    client = _client_without_network()

    async def boom(prompt, linking=False):
        raise LLMClassificationError("linking exploded")

    client._call_model = boom
    assert await client.link_requests(ITEMS) == []


@pytest.mark.asyncio
async def test_link_requests_skips_call_for_single_item():
    client = _client_without_network()

    async def must_not_be_called(prompt, linking=False):
        raise AssertionError("не має викликатись на одному запиті")

    client._call_model = must_not_be_called
    assert await client.link_requests(ITEMS[:1]) == []


# --- _parse_json -------------------------------------------------------------


def test_parse_json_strips_markdown_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_rejects_non_object():
    with pytest.raises(ValueError):
        _parse_json("[1, 2, 3]")
