"""
CLI: python main.py --input data/input_requests.csv --output-dir output

Env: GEMINI_API_KEY (обов'язково), GOOGLE_SHEETS_ID + GOOGLE_SERVICE_ACCOUNT_JSON (опційно).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.llm_client import GeminiClient
from src.pipeline import process_all
from src.report import write_report
from src.sheets import write_to_sheet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Класифікація вхідних запитів через LLM")
    parser.add_argument("--input", type=Path, default=Path("data/input_requests.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--concurrency", type=int, default=5, help="Паралельні запити до LLM")
    parser.add_argument("--max-retries", type=int, default=1, help="Retry на невалідний вивід")
    parser.add_argument(
        "--sheets", action="store_true", help="Також записати результат у Google Sheet"
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY не знайдено. Додайте його в .env (див. .env.example)")
        sys.exit(1)

    if not args.input.exists():
        logger.error("Вхідний файл не знайдено: %s", args.input)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(api_key=api_key)

    logger.info("Обробка %s (concurrency=%d)...", args.input, args.concurrency)
    records = await process_all(
        csv_path=args.input,
        client=client,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    output_json_path = args.output_dir / "output.json"
    output_json_path.write_text(
        json.dumps([r.to_output_dict() for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Записано %s", output_json_path)

    report_path = args.output_dir / "report.md"
    write_report(records, report_path)
    logger.info("Записано %s", report_path)

    ok = sum(1 for r in records if r.processing_status == "ok")
    failed = len(records) - ok
    logger.info("Готово: %d ok, %d fallback (помилка LLM)", ok, failed)

    if args.sheets:
        write_to_sheet(records)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
