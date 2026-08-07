"""
Агрегація результатів у короткий Markdown-звіт для людини, яка розбиратиме чергу.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.schema import ProcessedRequest


def build_report(records: list[ProcessedRequest]) -> str:
    dicts = [r.to_output_dict() for r in records]

    total = len(dicts)
    by_category = Counter(d["category"] for d in dicts)
    by_priority = Counter(d["priority"] for d in dicts)
    by_department = Counter(d["target_department"] or "не визначено" for d in dicts)
    needs_clarification = [d for d in dicts if d["needs_clarification"]]
    fallback_errors = [d for d in dicts if d["processing_status"] == "fallback_error"]

    lines: list[str] = []
    lines.append("# Звіт по вхідних запитах\n")
    lines.append(f"Усього оброблено запитів: **{total}**\n")

    lines.append("## По категоріях\n")
    for cat, count in by_category.most_common():
        lines.append(f"- {cat}: {count}")
    lines.append("")

    lines.append("## По пріоритету\n")
    for pr in ("high", "medium", "low"):
        lines.append(f"- {pr}: {by_priority.get(pr, 0)}")
    lines.append("")

    lines.append("## По відділах\n")
    for dept, count in by_department.most_common():
        lines.append(f"- {dept}: {count}")
    lines.append("")

    lines.append(f"## Потребують уточнення ({len(needs_clarification)})\n")
    if needs_clarification:
        for d in needs_clarification:
            reason = d.get("clarification_reason") or "—"
            summary = d.get("short_summary") or d["raw_text"][:80]
            lines.append(f"- **{d['id']}** ({d['channel']}): {summary} — _{reason}_")
    else:
        lines.append("Немає.")
    lines.append("")

    if fallback_errors:
        lines.append(f"## Помилки обробки LLM ({len(fallback_errors)})\n")
        lines.append(
            "Ці запити не вдалось класифікувати навіть після retry — "
            "потребують ручного розбору.\n"
        )
        for d in fallback_errors:
            lines.append(f"- **{d['id']}**: {d['processing_error']}")
        lines.append("")

    return "\n".join(lines)


def write_report(records: list[ProcessedRequest], output_path: Path) -> None:
    output_path.write_text(build_report(records), encoding="utf-8")
