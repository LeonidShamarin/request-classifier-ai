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

from src.llm_client import DEFAULT_RPM, MODEL_NAME, GeminiClient
from src.pipeline import process_all
from src.report import write_report
from src.sheets import write_to_sheet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _use_system_trust_store() -> None:
    """
    httpx (усередині google-genai) за замовчуванням довіряє лише бандлу `certifi`.
    У мережах з TLS-інспекцією (корпоративний проксі, антивірус) сертифікат
    підмінюється власним коренем — він є у сховищі ОС, але не в certifi, і всі
    виклики падають з CERTIFICATE_VERIFY_FAILED.

    truststore перемикає перевірку на системне сховище (Windows/macOS/Linux).
    Це не послаблює перевірку — просто бере довіру звідти, де вона насправді
    налаштована. Якщо пакета немає — тихо працюємо на certifi.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logger.debug("truststore недоступний, лишаємось на certifi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Класифікація вхідних запитів через LLM")
    parser.add_argument("--input", type=Path, default=Path("data/input_requests.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--concurrency", type=int, default=3, help="Паралельні запити до LLM")
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help=f"Ліміт запитів на хвилину (дефолт {DEFAULT_RPM} — квота безкоштовного "
        "тіру Gemini). 0 вимикає обмеження.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help=f"Модель Gemini (дефолт {MODEL_NAME})")
    parser.add_argument("--max-retries", type=int, default=1, help="Retry на невалідний вивід")
    parser.add_argument(
        "--sheets", action="store_true", help="Також записати результат у Google Sheet"
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    _use_system_trust_store()
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY не знайдено. Додайте його в .env (див. .env.example)")
        sys.exit(1)

    if not args.input.exists():
        logger.error("Вхідний файл не знайдено: %s", args.input)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(api_key=api_key, model_name=args.model, rpm=args.rpm)

    logger.info(
        "Обробка %s (model=%s, concurrency=%d, rpm=%d)...",
        args.input,
        args.model,
        args.concurrency,
        args.rpm,
    )
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

    if client.daily_quota_hit:
        logger.warning(
            "УВАГА: денну квоту безкоштовного тіру вичерпано для моделі %s — "
            "частина запитів не класифікована. Результат неповний. "
            "Спробуйте іншу модель (--model) або повторіть після скидання квоти.",
            args.model,
        )

    if args.sheets:
        write_to_sheet(records)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
