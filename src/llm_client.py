"""
Обгортка над Gemini API (SDK `google-genai`).

Рівні стійкості навмисно розділені, бо це різні класи проблем:

1. **Rate limiter** (`_RateLimiter`) — рознесення викликів у часі під квоту
   безкоштовного тіру, щоб узагалі не влітати в 429.
2. **Backoff** (`_call_model`) — транспорт: хвилинний 429, 5xx, обрив мережі.
   Модель тут ні до чого, просто чекаємо і повторюємо.
3. **Circuit breaker** (`daily_quota_hit`) — денна квота. Її перечекати не
   можна, тому решта запитів падає миттєво, без марних викликів.
4. **Self-repair** (`classify`) — зміст: відповідь не спарсилась як JSON або не
   пройшла Pydantic. Показуємо моделі її ж помилку і просимо виправити.

Якщо все вичерпано — кидаємо LLMClassificationError, яку pipeline.py ловить
і перетворює на fallback-запис (жоден рядок вхідного CSV не губиться, він просто
позначається needs_clarification=true).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from src.schema import LinkingResponse, LLMClassification, RelatedLink

logger = logging.getLogger(__name__)

# Класифікація за фіксованою схемою — легка задача, тут flash-lite достатньо,
# а денна квота безкоштовного тіру в неї помітно щедріша за flash.
# (gemini-2.5-flash з попередньої версії віддає 404 "no longer available
# to new users" — на нових ключах її вже не піднімеш.)
MODEL_NAME = "gemini-3.5-flash-lite"

# Ліміт безкоштовного тіру — 5 запитів/хв на модель. Клієнт сам розтягує
# виклики під цю квоту, щоб не влітати в 429 і не палити час на backoff.
DEFAULT_RPM = 5

TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BASE_DELAY = 2.0
# Стеля на паузу, яку просить сервер: без неї один невдалий запит може
# підвісити весь прогін на кілька хвилин.
MAX_SERVER_RETRY_DELAY = 65.0

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


def _server_retry_delay(exc: Exception) -> float | None:
    """
    Gemini у тілі 429 повертає RetryInfo.retryDelay (напр. 46s) — це набагато
    точніше за наш сліпий exponential backoff. Дістаємо, якщо є.
    """
    match = _RETRY_DELAY_RE.search(str(exc))
    if not match:
        return None
    return min(float(match.group(1)), MAX_SERVER_RETRY_DELAY)


class _RateLimiter:
    """
    Проста черга на N викликів за хвилину. Не обмежує паралелізм сам по собі —
    лише рознесення стартів у часі, решту робить семафор у pipeline.
    """

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._interval

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

- confidence — твоя впевненість у класифікації, число від 0.0 до 1.0.

Відповідай ТІЛЬКИ валідним JSON-об'єктом за наданою схемою. Без пояснень, без markdown.
"""

LINKER_SYSTEM_PROMPT = """\
Тобі дають список запитів з одного інбоксу: id, категорія і короткий опис суті.

Знайди пари, де один запит просить фактично ТЕ САМЕ, що вже просить інший — \
наприклад, дублює прохання ("той самий звіт, що Оля просила в понеділок"), або \
просить доступ/розсилку до результату, який уже замовлений іншим запитом.

Правила:
- Зв'язок має бути очевидним із тексту, а не здогадкою. Схожа тематика — НЕ зв'язок.
  Два різні звіти по різних даних — НЕ зв'язок.
- У парі request_id — це пізніший/похідний запит, related_request_id — первинний.
- reason — одне речення, чому ти вважаєш це одним і тим самим.
- Якщо жодної очевидної пари немає — поверни порожній список links.

Відповідай ТІЛЬКИ валідним JSON за наданою схемою.
"""


class LLMClassificationError(Exception):
    """Вичерпані всі спроби — і транспортні, і змістовні."""


@dataclass
class ClassificationResult:
    classification: LLMClassification
    retries_used: int


def _build_user_prompt(raw_text: str, request_id: str) -> str:
    return f"id запиту: {request_id}\nТекст запиту:\n{raw_text}"


class DailyQuotaExceeded(Exception):
    """Вичерпано денну квоту — до кінця доби прогін не відновиться."""


def _is_daily_quota(exc: Exception) -> bool:
    """
    Не всі 429 однакові. Хвилинну квоту перечекати можна, денну — ні.

    Пастка: Gemini у ВСІХ 429 повертає RetryInfo (напр. 'retryDelay: 25s'),
    навіть коли квота денна. Якщо слухати цю пораду наосліп, прогін годинами
    крутить безнадійні retry. Тому дивимось саме на quotaId.
    """
    return "PerDay" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    """Транспортна помилка, яку має сенс повторити після паузи."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        # Єдина 4xx, яку варто повторювати — хвилинний rate limit.
        # 400/401/403 не «розсмокчуться» самі, денна квота — тим паче.
        if getattr(exc, "code", None) != 429:
            return False
        return not _is_daily_quota(exc)
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = MODEL_NAME,
        temperature: float = 0.0,
        rpm: int = DEFAULT_RPM,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._limiter = _RateLimiter(rpm)
        # Запобіжник: денна квота вичерпується на весь проєкт одразу. Немає сенсу
        # ганяти решту запитів по мережі — вони гарантовано впадуть так само.
        self.daily_quota_hit = False
        self._config = self._make_config(SYSTEM_PROMPT, LLMClassification, temperature)
        self._linker_config = self._make_config(
            LINKER_SYSTEM_PROMPT, LinkingResponse, temperature
        )

    @staticmethod
    def _make_config(
        system_prompt: str, schema: type, temperature: float
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            # Pydantic-модель передається напряму: SDK сам мапить Optional[str]
            # у nullable-поле схеми Gemini. Ручна конвертація JSON Schema тут не
            # потрібна і шкідлива (anyOf/default API не приймає).
            response_schema=schema,
            # Класифікація за готовою схемою не потребує глибокого reasoning.
            # Саме thinking_level, а не thinking_budget: моделі Gemini 3.x
            # відповідають на budget=0 помилкою 400 INVALID_ARGUMENT.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

    async def classify(
        self, request_id: str, raw_text: str, max_retries: int = 1
    ) -> ClassificationResult:
        prompt = _build_user_prompt(raw_text, request_id)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                raw = await self._call_model(prompt)
                data = _parse_json(raw)
                parsed = LLMClassification.model_validate(data)
                return ClassificationResult(classification=parsed, retries_used=attempt)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Classification attempt %d failed for %s: %s", attempt, request_id, exc
                )
                # self-repair: показуємо моделі її помилку і просимо виправити
                prompt = (
                    f"{_build_user_prompt(raw_text, request_id)}\n\n"
                    f"Попередня відповідь була невалідною: {exc}\n"
                    f"Виправ і поверни ТІЛЬКИ коректний JSON за схемою."
                )

        raise LLMClassificationError(
            f"Failed to classify {request_id} after {max_retries + 1} attempts: {last_error}"
        )

    async def link_requests(self, items: list[dict]) -> list[RelatedLink]:
        """
        Другий прохід: шукає дублікати/похідні запити по всьому інбоксу одразу.

        Чому окремим проходом, а не полем у classify(): зв'язок між запитами
        видно лише коли бачиш увесь інбокс. Пхати весь інбокс у кожен з N
        промптів — це O(N²) токенів; один виклик по готових саммарі — O(N).

        Помилка тут не критична: зв'язки — це збагачення, а не основний результат,
        тому при провалі просто повертаємо порожній список.
        """
        if len(items) < 2:
            return []

        listing = "\n".join(
            f"- {it['id']} [{it['category']}]: {it['short_summary']}" for it in items
        )
        try:
            raw = await self._call_model(f"Запити в інбоксі:\n{listing}", linking=True)
            parsed = LinkingResponse.model_validate(_parse_json(raw))
        except Exception as exc:  # noqa: BLE001 — необов'язковий етап, не валимо прогін
            logger.warning("Linking pass failed (non-fatal): %s", str(exc)[:200])
            return []

        known = {it["id"] for it in items}
        valid = [
            link
            for link in parsed.links
            # модель цілком може вигадати неіснуючий id або зв'язати запит сам із
            # собою — обидва випадки відсікаємо, бо перевірити дешево
            if link.request_id in known
            and link.related_request_id in known
            and link.request_id != link.related_request_id
        ]
        dropped = len(parsed.links) - len(valid)
        if dropped:
            logger.warning("Linking: dropped %d link(s) with unknown/self ids", dropped)
        return valid

    async def _call_model(self, prompt: str, linking: bool = False) -> str:
        """Один логічний виклик моделі, з exponential backoff на 429/5xx/мережу."""
        last_error: Exception | None = None

        for attempt in range(TRANSPORT_MAX_ATTEMPTS):
            if self.daily_quota_hit:
                raise DailyQuotaExceeded(
                    f"Денна квота вичерпана для моделі {self._model_name}"
                )
            try:
                await self._limiter.acquire()
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=self._linker_config if linking else self._config,
                )
                text = response.text
                if not text:
                    # Порожня відповідь — напр. спрацював safety-фільтр. Це вже не
                    # транспорт, а зміст: віддаємо нагору, хай самопочиниться.
                    raise ValueError("Model returned empty response")
                return text
            except Exception as exc:  # noqa: BLE001 — розбираємо тип нижче
                if _is_daily_quota(exc):
                    self.daily_quota_hit = True
                    raise DailyQuotaExceeded(
                        f"Денна квота вичерпана для моделі {self._model_name}. "
                        "Спробуйте іншу модель (--model) або зачекайте до скидання квоти."
                    ) from exc
                if not _is_retryable(exc):
                    raise
                last_error = exc
                if attempt == TRANSPORT_MAX_ATTEMPTS - 1:
                    break
                # Якщо сервер сам сказав, скільки чекати — слухаємо його.
                # Наш exponential backoff тут лише запасний варіант.
                delay = _server_retry_delay(exc)
                if delay is None:
                    delay = TRANSPORT_BASE_DELAY * (2**attempt)
                delay += random.uniform(0, 1)
                logger.warning(
                    "Transport error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    TRANSPORT_MAX_ATTEMPTS,
                    delay,
                    str(exc)[:200],
                )
                await asyncio.sleep(delay)

        raise LLMClassificationError(
            f"Transport failed after {TRANSPORT_MAX_ATTEMPTS} attempts: {last_error}"
        )


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


# --- фолбек між кількома Gemini-ключами --------------------------------------
#
# Класифікація йде через `google-genai` з `response_schema`, тож крос-провайдерний
# фолбек непрактичний. Натомість ротуємо Gemini-ключі: основний GEMINI_API_KEY,
# далі ключі з ключі2.txt (GEMINI_API_KEY_2/_3) в останню чергу — це покриває
# мертвий ключ і вичерпану за добу квоту. Кожен ключ має власний клієнт (свій
# rate limiter і прапорець денної квоти). Проксіюємо обидва публічні методи.

_GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")


class FallbackClassifier:
    """Пробує Gemini-ключі по черзі; на `LLMClassificationError`/`DailyQuotaExceeded`
    бере наступний. Кидає помилку лише коли відмовили всі ключі, тож pipeline, як і
    раніше, перетворює її на запис `processing_status='fallback_error'`."""

    def __init__(self, members: list[tuple[str, GeminiClient]]):
        if not members:
            raise ValueError("порожній ланцюжок Gemini-ключів")
        self.members = members

    async def classify(self, request_id: str, raw_text: str, max_retries: int = 1):
        return await self._run("classify", request_id, raw_text, max_retries=max_retries)

    async def link_requests(self, items: list[dict]):
        return await self._run("link_requests", items)

    async def _run(self, method: str, *args, **kwargs):
        errors: list[str] = []
        for name, client in self.members:
            try:
                return await getattr(client, method)(*args, **kwargs)
            except (LLMClassificationError, DailyQuotaExceeded, genai_errors.APIError) as exc:
                errors.append(f"{name}: {str(exc)[:150]}")
        raise LLMClassificationError("усі Gemini-ключі відмовили -> " + " | ".join(errors))


def _gemini_keys(env: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for var in _GEMINI_KEY_VARS:
        key = (env.get(var) or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append((var, key))
    return out


def build_classifier(model_name: str = MODEL_NAME, rpm: int = DEFAULT_RPM,
                     temperature: float = 0.0, *, env: dict | None = None):
    """Збирає клієнт із ланцюжком Gemini-ключів, або `None`, якщо жодного ключа
    немає. Один ключ — звичайний `GeminiClient`, кілька — обгортка з фолбеком."""
    env = os.environ if env is None else env
    keys = _gemini_keys(env)
    if not keys:
        return None
    members = [(f"gemini:{var}", GeminiClient(api_key=key, model_name=model_name,
                                              temperature=temperature, rpm=rpm))
               for var, key in keys]
    if len(members) == 1:
        return members[0][1]
    return FallbackClassifier(members)
