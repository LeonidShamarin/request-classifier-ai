"""
Обгортка над Gemini API: будує промпт, просить structured JSON output,
валідує Pydantic-схемою, і при провалі робить одну спробу "самопочинки"
(показує моделі її ж помилку і просить виправити).

Якщо і друга спроба провалилась — кидає LLMClassificationError,
яку pipeline.py ловить і перетворює на fallback-запис (жоден рядок
вхідного CSV не губиться, він просто позначається needs_clarification=true).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from src.schema import LLMClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = """\
Ти — асистент AI-юніту компанії, який класифікує вхідні запити від внутрішніх \
команд (маркетинг, продажі, аналітика, PM, HR, бухгалтерія тощо). Запити прилітають \
у вільній формі, українською або англійською, часто без контексту.

Визнач для кожного запиту:

- category — рівно одне з:
  * "автоматизація" — просять автоматизувати ручний процес
  * "інтеграція" — з'єднати два сервіси/системи між собою
  * "звіт/аналітика" — потрібен звіт, дашборд, агрегація даних, виявлення аномалій
  * "баг/підтримка" — щось зламалось, не працює, потрібна допомога з існуючим рішенням
  * "питання/консультація" — просто питання/думка, без прохання щось зробити
  * "поза скоупом" — запит не стосується AI/автоматизації взагалі (напр. закупівля обладнання)
  * "інше" — не є запитом по суті: подяка, привітання, шум, посилання без пояснення

- target_department — відділ-замовник, якщо очевидно з тексту (напр. "маркетинг", \
"продажі", "HR", "аналітика", "бухгалтерія", "контент", "SMM"). null, якщо незрозуміло.

- priority — high / medium / low. Зважай на явні маркери терміновості ("ГОРИТЬ", \
"терміново", дедлайн сьогодні) та на бізнес-вплив (зламане = high, "не горить" = low).

- short_summary — суть запиту одним реченням, українською.

- requested_actions — список конкретних дій, які просять зробити. Порожній список, \
якщо дій не просять (напр. category="питання/консультація" або "інше").

- needs_clarification — true, якщо запит надто розмитий, щоб братися в роботу як є \
(немає деталей про обсяг, формат, систему, дедлайн — і без цього не зрозуміло, що робити). \
Якщо true — обов'язково заповни clarification_reason.

- language — "uk", "en" або "mixed", якщо в одному тексті помітно змішані мови.

- related_request_id — якщо запит явно посилається на інший запит з цього ж інбоксу \
(напр. "той самий звіт, що X просив"), і тобі показали список інших запитів нижче — \
вкажи його id. Інакше null.

- confidence — твоя впевненість у класифікації, число від 0.0 до 1.0.

Відповідай ТІЛЬКИ валідним JSON-об'єктом за наданою схемою. Без пояснень, без markdown.
"""


class LLMClassificationError(Exception):
    """Обидві спроби (оригінал + repair) провалились."""


@dataclass
class ClassificationResult:
    classification: LLMClassification
    retries_used: int


def _build_user_prompt(raw_text: str, request_id: str, context_ids: list[str]) -> str:
    context_note = ""
    if context_ids:
        context_note = (
            f"\n\nІнші id запитів у цьому інбоксі (для related_request_id): "
            f"{', '.join(context_ids)}"
        )
    return f"id запиту: {request_id}\nТекст запиту:\n{raw_text}{context_note}"


def _get_response_schema() -> dict:
    """JSON-схема для Gemini response_schema (controlled generation)."""
    schema = LLMClassification.model_json_schema()
    # Gemini не любить деякі поля з JSON Schema (напр. $defs для Literal-ів
    # рендеряться як enum, це ок) — модель приймає стандартний draft-схему,
    # тож передаємо як є.
    return schema


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME, temperature: float = 0.0):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self._temperature = temperature

    async def classify(
        self, request_id: str, raw_text: str, context_ids: list[str], max_retries: int = 1
    ) -> ClassificationResult:
        prompt = _build_user_prompt(raw_text, request_id, context_ids)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                raw = await self._call_model(prompt)
                data = _parse_json(raw)
                parsed = LLMClassification.model_validate(data)
                return ClassificationResult(classification=parsed, retries_used=attempt)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("Classification attempt %d failed for %s: %s", attempt, request_id, exc)
                # self-repair: показуємо моделі її помилку і просимо виправити
                prompt = (
                    f"{_build_user_prompt(raw_text, request_id, context_ids)}\n\n"
                    f"Попередня відповідь була невалідною: {exc}\n"
                    f"Виправ і поверни ТІЛЬКИ коректний JSON за схемою."
                )

        raise LLMClassificationError(
            f"Failed to classify {request_id} after {max_retries + 1} attempts: {last_error}"
        )

    async def _call_model(self, prompt: str) -> str:
        response = await self._model.generate_content_async(
            prompt,
            generation_config=self._genai.GenerationConfig(
                temperature=self._temperature,
                response_mime_type="application/json",
                response_schema=_get_response_schema(),
            ),
        )
        return response.text


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # захист від випадків, коли модель все ж обгортає у ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data
