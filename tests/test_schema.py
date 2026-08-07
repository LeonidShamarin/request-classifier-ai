import pytest
from pydantic import ValidationError

from src.schema import LLMClassification, ProcessedRequest


def test_valid_classification_parses():
    c = LLMClassification(
        category="автоматизація",
        target_department="маркетинг",
        priority="high",
        short_summary="Автоматизувати щотижневий звіт по Google Ads.",
        requested_actions=["Налаштувати вивантаження CSV", "Автоматично заповнювати таблицю"],
        needs_clarification=False,
        language="uk",
        confidence=0.9,
    )
    assert c.category == "автоматизація"
    assert c.priority == "high"


def test_invalid_category_rejected():
    with pytest.raises(ValidationError):
        LLMClassification(
            category="не існуюча категорія",
            priority="high",
            short_summary="x",
            needs_clarification=False,
            language="uk",
            confidence=0.5,
        )


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        LLMClassification(
            category="інше",
            priority="low",
            short_summary="x",
            needs_clarification=False,
            language="uk",
            confidence=1.5,
        )


def test_empty_actions_are_stripped():
    c = LLMClassification(
        category="інше",
        priority="low",
        short_summary="x",
        requested_actions=["  ", "", "Реальна дія"],
        needs_clarification=False,
        language="uk",
        confidence=0.5,
    )
    assert c.requested_actions == ["Реальна дія"]


def test_processed_request_fallback_has_safe_defaults():
    rec = ProcessedRequest(
        id="REQ-999",
        channel="Slack",
        timestamp="2026-01-01 00:00",
        raw_text="щось незрозуміле",
        classification=None,
        processing_status="fallback_error",
        processing_error="parse error: bad json",
    )
    d = rec.to_output_dict()
    assert d["needs_clarification"] is True
    assert d["category"] == "інше"
    assert d["processing_status"] == "fallback_error"
    assert "parse error" in d["clarification_reason"]
