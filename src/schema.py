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
    related_request_id: Optional[str] = Field(
        default=None,
        description="ID іншого запиту з цього ж інбоксу, якщо поточний явно "
        "посилається на нього (напр. 'той самий звіт, що X просив'). "
        "null, якщо зв'язку немає.",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("short_summary")
    @classmethod
    def _strip_summary(cls, v: str) -> str:
        return v.strip()

    @field_validator("requested_actions")
    @classmethod
    def _clean_actions(cls, v: list[str]) -> list[str]:
        return [a.strip() for a in v if a and a.strip()]


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
                    "related_request_id": None,
                    "confidence": 0.0,
                }
            )
        return base
