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


def read_requests(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"id", "channel", "timestamp", "raw_text"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return rows


async def _process_one(
    client: GeminiClient,
    row: dict,
    context_ids: list[str],
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> ProcessedRequest:
    async with semaphore:
        try:
            result = await client.classify(
                request_id=row["id"],
                raw_text=row["raw_text"],
                context_ids=context_ids,
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
    context_ids = [r["id"] for r in rows]
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _process_one(client, row, context_ids, semaphore, max_retries) for row in rows
    ]
    results = await asyncio.gather(*tasks)

    # gather зберігає порядок вхідних тасків, але явно сортуємо по id
    # для детермінізму виводу незалежно від того, як асинхронно виконались задачі.
    order = {r["id"]: i for i, r in enumerate(rows)}
    results = sorted(results, key=lambda r: order[r.id])
    return list(results)
