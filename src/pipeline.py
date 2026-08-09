"""
Оркестрація: читає CSV, батчево (з обмеженою конкурентністю) шле запити в LLM,
ловить помилки класифікації і перетворює їх на fallback-записи, повертає
список ProcessedRequest готовий до серіалізації в output.json.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path

from src.llm_client import GeminiClient, LLMClassificationError
from src.schema import ProcessedRequest

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"id", "channel", "timestamp", "raw_text"}


def read_requests(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Валідуємо заголовок ДО читання рядків — на великому файлі немає сенсу
        # вичитувати сотні мегабайт, щоб потім впасти через відсутню колонку.
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        return list(reader)


async def _process_one(
    client: GeminiClient,
    row: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> ProcessedRequest:
    async with semaphore:
        try:
            result = await client.classify(
                request_id=row["id"],
                raw_text=row["raw_text"],
                max_retries=max_retries,
            )
            return ProcessedRequest(
                id=row["id"],
                channel=row["channel"],
                timestamp=row["timestamp"],
                raw_text=row["raw_text"],
                classification=result.classification,
                processing_status="ok",
                retries_used=result.retries_used,
            )
        except LLMClassificationError as exc:
            logger.error("Fallback for %s: %s", row["id"], exc)
            return ProcessedRequest(
                id=row["id"],
                channel=row["channel"],
                timestamp=row["timestamp"],
                raw_text=row["raw_text"],
                classification=None,
                processing_status="fallback_error",
                processing_error=str(exc),
                retries_used=max_retries + 1,
            )
        except Exception as exc:  # noqa: BLE001 — навмисно широко: мережа/rate-limit/усе інше
            logger.error("Unexpected error for %s: %s", row["id"], exc)
            return ProcessedRequest(
                id=row["id"],
                channel=row["channel"],
                timestamp=row["timestamp"],
                raw_text=row["raw_text"],
                classification=None,
                processing_status="fallback_error",
                processing_error=f"Unexpected error: {exc}",
                retries_used=0,
            )


async def process_all(
    csv_path: Path,
    client: GeminiClient,
    concurrency: int = 5,
    max_retries: int = 1,
) -> list[ProcessedRequest]:
    rows = read_requests(csv_path)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [_process_one(client, row, semaphore, max_retries) for row in rows]
    results = await asyncio.gather(*tasks)

    # gather зберігає порядок вхідних тасків, але явно сортуємо по id
    # для детермінізму виводу незалежно від того, як асинхронно виконались задачі.
    order = {r["id"]: i for i, r in enumerate(rows)}
    records = sorted(results, key=lambda r: order[r.id])

    await _apply_related_links(client, records)
    return records


async def _apply_related_links(
    client: GeminiClient, records: list[ProcessedRequest]
) -> None:
    """
    Другий прохід по вже класифікованих записах — шукає дублікати запитів.
    Мутує records на місці. Не критичний: якщо не вийшло, зв'язки просто
    лишаються порожніми, а основний результат не страждає.
    """
    items = [
        {
            "id": r.id,
            "category": r.classification.category,
            "short_summary": r.classification.short_summary,
        }
        for r in records
        if r.classification is not None
    ]
    links = await client.link_requests(items)
    if not links:
        return

    by_id = {r.id: r for r in records}
    for link in links:
        record = by_id[link.request_id]
        record.related_request_id = link.related_request_id
        record.related_reason = link.reason
    logger.info("Знайдено зв'язків між запитами: %d", len(links))
