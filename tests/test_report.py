from src.report import build_report
from src.schema import LLMClassification, ProcessedRequest


def _ok_record(
    rid,
    category="автоматизація",
    needs_clarification=False,
    dept="маркетинг",
    related=None,
    related_reason=None,
):
    return ProcessedRequest(
        id=rid,
        channel="Slack",
        timestamp="2026-01-01 09:00",
        raw_text="text",
        related_request_id=related,
        related_reason=related_reason,
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


def _fallback_record(rid):
    return ProcessedRequest(
        id=rid,
        channel="Telegram",
        timestamp="2026-01-01 09:00",
        raw_text="текст, який не вдалось класифікувати",
        classification=None,
        processing_status="fallback_error",
        processing_error="LLM returned garbage",
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


def test_fallback_records_excluded_from_aggregates():
    """Fallback має технічні заглушки (category=інше, priority=medium).
    Якщо рахувати їх разом з рештою — статистика мовчки бреше."""
    records = [
        _ok_record("REQ-1", category="автоматизація"),
        _fallback_record("REQ-BAD"),
    ]
    report = build_report(records)

    assert "Усього запитів у вхідному файлі: **2**" in report
    assert "Успішно класифіковано: **1**" in report
    assert "Не вдалось класифікувати: **1**" in report

    # заглушки fallback-у не потрапили в агрегати
    assert "- інше: 1" not in report
    assert "- medium: 1" in report and "- medium: 2" not in report

    # але сам запис не загубився: він у "потребують уточнення" і в помилках
    assert "Потребують уточнення (1)" in report
    assert "Помилки обробки LLM (1)" in report
    assert "REQ-BAD" in report


def test_report_lists_related_duplicates():
    records = [
        _ok_record("REQ-001"),
        _ok_record("REQ-013", related="REQ-001", related_reason="той самий звіт"),
    ]
    report = build_report(records)
    assert "Ймовірні дублікати (1)" in report
    assert "**REQ-013** → **REQ-001**" in report
    assert "той самий звіт" in report


def test_report_omits_duplicates_section_when_none():
    report = build_report([_ok_record("REQ-001")])
    assert "Ймовірні дублікати" not in report
