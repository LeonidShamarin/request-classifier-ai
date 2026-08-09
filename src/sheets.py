"""
Опційний запис результатів у Google Sheet.

Вимикається сам, якщо не налаштовано (немає GOOGLE_SHEETS_ID або
GOOGLE_SERVICE_ACCOUNT_JSON) — це не має ламати основний пайплайн,
тому помилки тут тільки логуються, а не кидаються далі.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.schema import ProcessedRequest

logger = logging.getLogger(__name__)

HEADER = [
    "id",
    "channel",
    "timestamp",
    "category",
    "target_department",
    "priority",
    "short_summary",
    "requested_actions",
    "needs_clarification",
    "clarification_reason",
    "language",
    "related_request_id",
    "related_reason",
    "confidence",
    "processing_status",
]


def is_configured() -> bool:
    return bool(os.environ.get("GOOGLE_SHEETS_ID")) and bool(
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    )


def write_to_sheet(records: list[ProcessedRequest]) -> bool:
    """Повертає True, якщо запис вдався; False — якщо пропущено/провалилось.
    Ніколи не кидає виключення назовні (звіт і output.json не мають залежати
    від доступності Google API)."""
    if not is_configured():
        logger.info("Google Sheets not configured, skipping (see .env.example).")
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_path = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
        sheet_id = os.environ["GOOGLE_SHEETS_ID"]

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        worksheet.clear()

        rows = [HEADER]
        for r in records:
            d = r.to_output_dict()
            rows.append(
                [
                    d["id"],
                    d["channel"],
                    d["timestamp"],
                    d["category"],
                    d["target_department"] or "",
                    d["priority"],
                    d["short_summary"] or "",
                    "; ".join(d["requested_actions"] or []),
                    str(d["needs_clarification"]),
                    d["clarification_reason"] or "",
                    d["language"] or "",
                    d["related_request_id"] or "",
                    d.get("related_reason") or "",
                    str(d["confidence"]),
                    d["processing_status"],
                ]
            )
        worksheet.update(rows)
        logger.info("Wrote %d rows to Google Sheet %s", len(rows) - 1, sheet_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Google Sheets write failed (non-fatal): %s", exc)
        return False
