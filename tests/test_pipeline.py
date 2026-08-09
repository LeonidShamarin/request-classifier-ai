from pathlib import Path

import pytest

from src.llm_client import ClassificationResult, LLMClassificationError
from src.pipeline import process_all, read_requests
from src.schema import LLMClassification, RelatedLink

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_requests.csv"


class FakeClient:
    """Дублює інтерфейс GeminiClient, без реальних викликів API."""

    def __init__(self, behavior: dict[str, str], links: list[RelatedLink] | None = None):
        # behavior: request_id -> "ok" | "fail"
        self.behavior = behavior
        self.links = links or []
        self.linked_items = None

    async def classify(self, request_id, raw_text, max_retries=1):
        mode = self.behavior.get(request_id, "ok")
        if mode == "fail":
            raise LLMClassificationError(f"forced failure for {request_id}")
        return ClassificationResult(
            classification=LLMClassification(
                category="автоматизація",
                target_department="маркетинг",
                priority="medium",
                short_summary=f"Summary for {request_id}",
                requested_actions=["do the thing"],
                needs_clarification=False,
                language="uk",
                confidence=0.8,
            ),
            retries_used=0,
        )

    async def link_requests(self, items):
        self.linked_items = items
        return self.links


def test_read_requests_parses_csv():
    rows = read_requests(FIXTURE_CSV)
    assert len(rows) == 3
    assert rows[0]["id"] == "REQ-A"


@pytest.mark.asyncio
async def test_process_all_success_and_fallback_mixed():
    client = FakeClient(behavior={"REQ-B": "fail"})
    records = await process_all(FIXTURE_CSV, client, concurrency=2, max_retries=1)

    assert len(records) == 3
    by_id = {r.id: r for r in records}

    assert by_id["REQ-A"].processing_status == "ok"
    assert by_id["REQ-A"].classification.category == "автоматизація"

    assert by_id["REQ-B"].processing_status == "fallback_error"
    assert by_id["REQ-B"].classification is None
    out = by_id["REQ-B"].to_output_dict()
    assert out["needs_clarification"] is True

    assert by_id["REQ-C"].processing_status == "ok"


@pytest.mark.asyncio
async def test_process_all_preserves_input_order():
    client = FakeClient(behavior={})
    records = await process_all(FIXTURE_CSV, client, concurrency=5, max_retries=0)
    assert [r.id for r in records] == ["REQ-A", "REQ-B", "REQ-C"]


@pytest.mark.asyncio
async def test_related_links_are_applied_to_records():
    client = FakeClient(
        behavior={},
        links=[
            RelatedLink(
                request_id="REQ-C", related_request_id="REQ-A", reason="той самий звіт"
            )
        ],
    )
    records = await process_all(FIXTURE_CSV, client, concurrency=2, max_retries=0)
    by_id = {r.id: r for r in records}

    assert by_id["REQ-C"].related_request_id == "REQ-A"
    assert by_id["REQ-C"].to_output_dict()["related_reason"] == "той самий звіт"
    # решта записів лишається незв'язаною
    assert by_id["REQ-A"].related_request_id is None


@pytest.mark.asyncio
async def test_linker_receives_only_classified_records():
    """Fallback-записи не мають про що зв'язуватись — у них немає summary."""
    client = FakeClient(behavior={"REQ-B": "fail"})
    await process_all(FIXTURE_CSV, client, concurrency=2, max_retries=0)

    assert [it["id"] for it in client.linked_items] == ["REQ-A", "REQ-C"]
