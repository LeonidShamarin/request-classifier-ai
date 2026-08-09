"""
Структура даних для класифікації запитів.

Обов'язкові поля — прямо з ТЗ. Розширення нижче навмисно мінімальні:
кожне з них закриває конкретний edge case, який реально зустрічається
у вхідному CSV (див. README, розділ "Схема даних").
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "автоматизація",
    "інтеграція",
    "звіт/аналітика",
    "баг/підтримка",
    "питання/консультація",
    "поза скоупом",
    "інше",  # розширення: подяки, привітання, нерелевантний шум — не запит взагалі
]

Priority = Literal["low", "medium", "high"]

Language = Literal["uk", "en", "mixed"]


class LLMClassification(BaseModel):
    """
    Те, що ми очікуємо отримати від LLM для одного запиту.
    Це "контракт" для structured output — саме ця схема передається
    моделі (response_schema) і саме нею валідується сира відповідь.
    """

    category: Category
    target_department: Optional[str] = Field(
        default=None,
        description="Відділ-замовник, напр. 'маркетинг', 'продажі', 'HR'. "
        "null, якщо з тексту незрозуміло.",
    )
    priority: Priority
    short_summary: str = Field(min_length=1, max_length=400)
    requested_actions: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_reason: Optional[str] = Field(
        default=None,
        description="Чому запит розмитий. Обов'язково, якщо needs_clarification=true.",
    )
    language: Language
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("short_summary")
    @classmethod
    def _strip_summary(cls, v: str) -> str:
        return v.strip()

    @field_validator("requested_actions")
    @classmethod
    def _clean_actions(cls, v: list[str]) -> list[str]:
        return [a.strip() for a in v if a and a.strip()]


class RelatedLink(BaseModel):
    """Одна знайдена пара «цей запит — про те саме, що й той»."""

    request_id: str
    related_request_id: str
    reason: str = Field(min_length=1, max_length=300)


class LinkingResponse(BaseModel):
    """Відповідь окремого проходу зв'язування (див. GeminiClient.link_requests)."""

    links: list[RelatedLink] = Field(default_factory=list)


class ProcessedRequest(BaseModel):
    """
    Повний запис для output.json: вхідні метадані + результат класифікації
    + метадані обробки (чи були retry/помилки парсингу).
    """

    id: str
    channel: str
    timestamp: str
    raw_text: str

    classification: Optional[LLMClassification] = None

    # Заповнюється окремим batch-проходом ПІСЛЯ класифікації, а не самою
    # класифікацією: зв'язок між запитами видно лише коли бачиш весь інбокс
    # одночасно, а не кожен рядок окремо.
    related_request_id: Optional[str] = None
    related_reason: Optional[str] = None

    # Метадані пайплайну (не від LLM) — потрібні, щоб чесно показати,
    # де саме класифікація "здалась" і чому.
    processing_status: Literal["ok", "fallback_error"] = "ok"
    processing_error: Optional[str] = None
    retries_used: int = 0

    def to_output_dict(self) -> dict:
        """Пласка структура для output.json — зручніше читати/грепати, ніж вкладену."""
        base = {
            "id": self.id,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "raw_text": self.raw_text,
            "processing_status": self.processing_status,
            "processing_error": self.processing_error,
            "retries_used": self.retries_used,
        }
        if self.classification is not None:
            base.update(self.classification.model_dump())
            base["related_request_id"] = self.related_request_id
            base["related_reason"] = self.related_reason
        else:
            # fallback-запис без класифікації — усі поля класифікації null,
            # але needs_clarification примусово true, щоб запис не загубився
            # серед "оброблених" у звіті.
            base.update(
                {
                    "category": "інше",
                    "target_department": None,
                    "priority": "medium",
                    "short_summary": None,
                    "requested_actions": [],
                    "needs_clarification": True,
                    "clarification_reason": self.processing_error
                    or "LLM classification failed after retries",
                    "language": None,
                    "related_request_id": self.related_request_id,
                    "related_reason": self.related_reason,
                    "confidence": 0.0,
                }
            )
        return base
