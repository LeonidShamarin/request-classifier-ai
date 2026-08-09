"""
Агрегація результатів у короткий Markdown-звіт для людини, яка розбиратиме чергу.

Важливо: у агрегати (категорії / пріоритети / відділи) потрапляють ЛИШЕ успішно
класифіковані записи. Fallback-записи (LLM провалився) мають технічні заглушки
замість реальних значень — якщо їх рахувати разом з рештою, вони мовчки роздують
"інше" і "medium" і зіпсують статистику. Тому вони йдуть окремою секцією.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.schema import ProcessedRequest


def build_report(records: list[ProcessedRequest]) -> str:
    dicts = [r.to_output_dict() for r in records]

    total = len(dicts)
    classified = [d for d in dicts if d["processing_status"] == "ok"]
    fallback_errors = [d for d in dicts if d["processing_status"] != "ok"]

    by_category = Counter(d["category"] for d in classified)
    by_priority = Counter(d["priority"] for d in classified)
    by_department = Counter(d["target_department"] or "не визначено" for d in classified)
    needs_clarification = [d for d in dicts if d["needs_clarification"]]

    lines: list[str] = []
    lines.append("# Звіт по вхідних запитах\n")
    lines.append(f"Усього запитів у вхідному файлі: **{total}**")
    lines.append(f"Успішно класифіковано: **{len(classified)}**")
    lines.append(f"Не вдалось класифікувати: **{len(fallback_errors)}**\n")

    if fallback_errors:
        lines.append(
            "> Агрегати нижче рахуються лише по успішно класифікованих "
            f"({len(classified)}). Решта — в секції «Помилки обробки LLM».\n"
        )

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

    related = [d for d in dicts if d.get("related_request_id")]
    if related:
        lines.append(f"## Ймовірні дублікати ({len(related)})\n")
        lines.append(
            "Запити, що просять фактично те саме, що вже замовлено іншим запитом — "
            "варто обʼєднати в одну задачу, а не вести двічі.\n"
        )
        for d in related:
            lines.append(
                f"- **{d['id']}** → **{d['related_request_id']}**: "
                f"_{d.get('related_reason') or '—'}_"
            )
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
