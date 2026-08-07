from src.report import build_report
from src.schema import LLMClassification, ProcessedRequest


def _ok_record(rid, category="автоматизація", needs_clarification=False, dept="маркетинг"):
    return ProcessedRequest(
        id=rid,
        channel="Slack",
        timestamp="2026-01-01 09:00",
        raw_text="text",
        classification=LLMClassification(
            category=category,
            target_department=dept,
            priority="medium",
            short_summary="summary",
            requested_actions=[],
            needs_clarification=needs_clarification,
            clarification_reason="незрозумілий обсяг" if needs_clarification else None,
            language="uk",
            confidence=0.7,
        ),
    )


def test_report_counts_categories_and_clarification():
    records = [
        _ok_record("REQ-1", category="автоматизація"),
        _ok_record("REQ-2", category="автоматизація", needs_clarification=True),
        _ok_record("REQ-3", category="звіт/аналітика", dept=None),
    ]
    report = build_report(records)
    assert "автоматизація: 2" in report
    assert "звіт/аналітика: 1" in report
    assert "Потребують уточнення (1)" in report
    assert "REQ-2" in report
    assert "не визначено: 1" in report
