"""LLM abstraction for code generation and the semantic verification signal.

Provider is selected through the environment:

    VKSK_LLM_PROVIDER = groq (default) | anthropic | mock
    VKSK_LLM_MODEL    = model override
    GROQ_API_KEY      = key for groq

Without a key for the selected provider it falls back to MOCK.

MOCK is not a general LLM simulation. It is a lookup table for the exact
queries in the two test sets, matched verbatim, so the pipeline can be
exercised offline. It also reproduces a few realistic generation failures on
purpose, to demonstrate the L0 and L2 paths. Values may be a dict keyed by
table name, because retrieval narrows the visible tables and the correct code
depends on which one was selected.
"""
import os
import re
import time

import config  # noqa: F401 - loads .env

# Groq is the default provider: the project runs on free API tiers only, and
# it is the one with a usable free tier. The anthropic branch is kept for
# compatibility but is not used in reproducible runs.

PROVIDER_ENV = "VKSK_LLM_PROVIDER"
MODEL_ENV = "VKSK_LLM_MODEL"

DEFAULT_PROVIDER = "groq"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Chosen by the hardest task it is given, which is pandas code generation
# rather than classification, and used for both the baseline and the method so
# that comparisons isolate the method rather than the model.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

MAX_TOKENS = 400
REQUEST_TIMEOUT = 60

# 429 is an expected state on the free tier, not an error: retry with backoff,
# honouring Retry-After when present.
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


class LLMError(RuntimeError):
    """Provider returned an error that retrying will not fix."""


# ---- cost and latency accounting ----

_STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "elapsed_s": 0.0, "retries": 0, "provider": None, "model": None}


def reset_stats():
    _STATS.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "elapsed_s": 0.0, "retries": 0})


def get_stats() -> dict:
    s = dict(_STATS)
    s["avg_latency_s"] = (s["elapsed_s"] / s["calls"]) if s["calls"] else 0.0
    s["total_tokens"] = s["prompt_tokens"] + s["completion_tokens"]
    return s


def active_provider() -> str:
    """Provider actually in effect: 'groq', 'anthropic' or 'mock'.

    MOCK is not a configured provider but the consequence of a missing key.
    """
    provider = os.environ.get(PROVIDER_ENV, DEFAULT_PROVIDER).strip().lower()
    if provider == "mock":
        return "mock"
    if provider == "anthropic":
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
    if provider == "groq":
        return "groq" if os.environ.get("GROQ_API_KEY") else "mock"
    raise LLMError(f"Unknown {PROVIDER_ENV}={provider!r}; expected groq|anthropic|mock")


def active_model() -> str:
    provider = active_provider()
    if provider == "mock":
        return "mock"
    default = GROQ_DEFAULT_MODEL if provider == "groq" else ANTHROPIC_DEFAULT_MODEL
    return os.environ.get(MODEL_ENV, default)


def call_llm(prompt: str, system: str = None, max_tokens: int = None,
             reasoning_effort: str = None) -> str:
    """Call the active provider.

    `max_tokens` overrides the default for one call. Batch tasks need far more
    than code generation, because gpt-oss models spend part of the budget on
    internal reasoning tokens; too small a limit yields an empty response
    rather than a truncated one.

    `reasoning_effort` ("low"/"medium"/"high") caps that internal reasoning.
    "low" is right for structured tasks such as label classification and keeps
    calls within the per-minute token limit. Leave it off for code generation.
    """
    provider = active_provider()
    if provider == "groq":
        return _call_groq(prompt, system, max_tokens, reasoning_effort)
    if provider == "anthropic":
        return _call_anthropic(prompt, system, os.environ.get("ANTHROPIC_API_KEY"), max_tokens)
    return _mock_llm(prompt)


def _record(elapsed: float, usage: dict, provider: str, model: str, retries: int = 0):
    _STATS["calls"] += 1
    _STATS["elapsed_s"] += elapsed
    _STATS["retries"] += retries
    _STATS["provider"] = provider
    _STATS["model"] = model
    if usage:
        _STATS["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        _STATS["completion_tokens"] += int(usage.get("completion_tokens") or 0)


def _call_groq(prompt, system, max_tokens=None, reasoning_effort=None):
    """Groq via its OpenAI-compatible endpoint.

    Plain HTTP rather than an SDK: no extra dependency, and the free-tier 429
    handling has to be custom anyway.
    """
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")

    model = os.environ.get(MODEL_ENV, GROQ_DEFAULT_MODEL)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {"model": model, "max_tokens": max_tokens or MAX_TOKENS,
               "messages": messages, "temperature": 0}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    started = time.time()
    retries = 0
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            retries += 1
            if attempt == MAX_RETRIES - 1:
                raise LLMError(f"Groq {resp.status_code} after {MAX_RETRIES} attempts: {resp.text[:200]}")
            wait = float(resp.headers.get("retry-after") or BACKOFF_BASE ** attempt)
            time.sleep(min(wait, 30.0))
            continue
        if resp.status_code != 200:
            raise LLMError(f"Groq {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        _record(time.time() - started, data.get("usage") or {}, "groq", model, retries)
        return _strip_code_fences(data["choices"][0]["message"]["content"])

    raise LLMError("Groq: retries exhausted")


def _call_anthropic(prompt, system, api_key, max_tokens=None):
    """Legacy branch, not used in reproducible runs."""
    import anthropic  # pip install anthropic

    model = os.environ.get(MODEL_ENV, ANTHROPIC_DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens or MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    started = time.time()
    resp = client.messages.create(**kwargs)
    usage = {"prompt_tokens": getattr(resp.usage, "input_tokens", 0),
             "completion_tokens": getattr(resp.usage, "output_tokens", 0)}
    _record(time.time() - started, usage, "anthropic", model)
    text = "".join(block.text for block in resp.content if block.type == "text")
    return _strip_code_fences(text)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


# ---- MOCK mode ----

_CODE_MAP = {
    "Яка виручка за Q3 2024?":
        "result = income.loc[income['Показник'] == 'Виручка', 'Q3_2024'].values[0]",
    "Який чистий прибуток за Q1 2023?":
        "result = income.loc[income['Показник'] == 'Чистий прибуток', 'Q1_2023'].values[0]",
    "Яка сукупна виручка за 2023 рік?":
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = income.loc[income['Показник'] == 'Виручка', cols].values.sum()",
    "Яка сумарна собівартість реалізації за 2024 рік?":
        "cols = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']\n"
        "result = income.loc[income['Показник'] == 'Собівартість реалізації', cols].values.sum()",
    "Який середній чистий прибуток за 2023 рік?":
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = income.loc[income['Показник'] == 'Чистий прибуток', cols].values.mean()",
    "Яка середня виручка за всі 8 кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = income.loc[income['Показник'] == 'Виручка', cols].values.mean()",
    "Який максимальний чистий прибуток серед усіх кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = income.loc[income['Показник'] == 'Чистий прибуток', cols].values.max()",
    "Яка мінімальна виручка серед усіх кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = income.loc[income['Показник'] == 'Виручка', cols].values.min()",
    "Наскільки виручка Q4 2024 більша за Q4 2023 в абсолютних цифрах?":
        "a = income.loc[income['Показник'] == 'Виручка', 'Q4_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Виручка', 'Q4_2023'].values[0]\n"
        "result = a - b",
    "Яка різниця між чистим прибутком Q1 2024 і Q1 2023?":
        "a = income.loc[income['Показник'] == 'Чистий прибуток', 'Q1_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Чистий прибуток', 'Q1_2023'].values[0]\n"
        "result = a - b",
    "На скільки відсотків зросла виручка з Q1 2023 по Q1 2024?":
        "a = income.loc[income['Показник'] == 'Виручка', 'Q1_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Виручка', 'Q1_2023'].values[0]\n"
        "result = (a - b) / b * 100",
    "Який приріст чистого прибутку з Q3 2023 по Q3 2024 у відсотках?":
        "a = income.loc[income['Показник'] == 'Чистий прибуток', 'Q3_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Чистий прибуток', 'Q3_2023'].values[0]\n"
        "result = (a - b) / b * 100",
    "Скільки кварталів мали чистий прибуток більше 150000?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "row = income.loc[income['Показник'] == 'Чистий прибуток', cols].values[0]\n"
        "result = (row > 150000).sum()",
    "Яка сумарна виручка за квартали, де чистий прибуток перевищив 200000?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "profit = income.loc[income['Показник'] == 'Чистий прибуток', cols].values[0]\n"
        "revenue = income.loc[income['Показник'] == 'Виручка', cols].values[0]\n"
        "result = revenue[profit > 200000].sum()",
    # Cross-table query: retrieval may route it to any of the three tables,
    # so the value is a dict keyed by table name.
    "Які грошові кошти на кінець Q2 2024?": {
        "income": "result = income.loc[income['Показник'] == 'Грошові кошти на кінець кварталу', 'Q2_2024'].values[0]",
        "balance": "result = balance.loc[balance['Показник'] == 'Грошові кошти', 'Q2_2024'].values[0]",
        "cashflow": "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Залишок коштів на кінець періоду', 'Q2_2024'].values[0]",
    },
    # ці дві мають "правильну" версію тут (використовується на retry) і
    # окрему навмисно хибну першу спробу в _FIRST_ATTEMPT_OVERRIDE нижче
    "Яка маржа чистого прибутку (%) за Q4 2024?":
        "a = income.loc[income['Показник'] == 'Чистий прибуток', 'Q4_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Виручка', 'Q4_2024'].values[0]\n"
        "result = a / b * 100",
    "Яка сума фінансових витрат за 2023 рік?":
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = income.loc[income['Показник'] == 'Фінансові витрати', cols].sum(axis=1).values[0]",
    # Adversarial: no such data exists, so the pipeline should abstain.
    "Яка рентабельність по сегменту B2B за МСФЗ 15 у Q2 2023?": {
        "income": "result = income.loc[income['Показник'] == 'Сегмент B2B МСФЗ 15', 'Q2_2023'].values[0]",
        "balance": "result = balance.loc[balance['Показник'] == 'Сегмент B2B МСФЗ 15', 'Q2_2023'].values[0]",
        "cashflow": "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Сегмент B2B МСФЗ 15', 'Q2_2023'].values[0]",
    },

    # ---- three-table query set ----

    # income
    "Який максимальний валовий прибуток серед усіх кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = income.loc[income['Показник'] == 'Валовий прибуток', cols].values.max()",

    # balance
    "Який власний капітал на Q2 2024?":
        "result = balance.loc[balance['Показник'] == 'Власний капітал', 'Q2_2024'].values[0]",
    "Скільки всього активів на Q4 2023?":
        "result = balance.loc[balance['Показник'] == 'Всього активів', 'Q4_2023'].values[0]",
    "На скільки зросли всього активів з Q1 2023 по Q4 2024?":
        "a = balance.loc[balance['Показник'] == 'Всього активів', 'Q4_2024'].values[0]\n"
        "b = balance.loc[balance['Показник'] == 'Всього активів', 'Q1_2023'].values[0]\n"
        "result = a - b",
    "Яка частка власного капіталу у всього активів (%) на Q3 2024?":
        "a = balance.loc[balance['Показник'] == 'Власний капітал', 'Q3_2024'].values[0]\n"
        "b = balance.loc[balance['Показник'] == 'Всього активів', 'Q3_2024'].values[0]\n"
        "result = a / b * 100",
    "Яка максимальна дебіторська заборгованість серед усіх кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = balance.loc[balance['Показник'] == 'Дебіторська заборгованість', cols].values.max()",
    "Яка середня величина всього зобов'язань за 2024 рік?":
        "cols = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']\n"
        "result = balance.loc[balance['Показник'] == 'Всього зобов\\'язань', cols].values.mean()",

    # cashflow (stitched from two pages)
    "Скільки виплачено дивідендів акціонерам за Q2 2023?":
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Виплачені дивіденди акціонерам', 'Q2_2023'].values[0]",
    "Який чистий рух коштів від операційної діяльності за Q4 2024?":
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Чистий рух коштів від операційної діяльності', 'Q4_2024'].values[0]",
    # correct version, used on retry; the faulty first attempt is below
    "Яка сума витрат на маркетинг за 2024 рік?":
        "cols = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']\n"
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Витрати на маркетинг', cols].values.sum()",
    "Скільки всього сплачено відсотків за кредитами за 2023 рік?":
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Сплачені відсотки за кредитами', cols].values.sum()",
    "Які найбільші надходження від реалізації продукції серед усіх кварталів?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Надходження від реалізації продукції', cols].values.max()",
    "Скільки кварталів мали чистий рух грошових коштів за період більше 80000?":
        "cols = ['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']\n"
        "row = cashflow.loc[cashflow['Стаття руху коштів'] == 'Чистий рух грошових коштів за період', cols].values[0]\n"
        "result = (row > 80000).sum()",
    "Який середній чистий рух коштів від інвестиційної діяльності за 2023 рік?":
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Чистий рух коштів від інвестиційної діяльності', cols].values.mean()",
    "На скільки відсотків зросли надходження від реалізації продукції з Q1 2023 по Q4 2024?":
        "a = cashflow.loc[cashflow['Стаття руху коштів'] == 'Надходження від реалізації продукції', 'Q4_2024'].values[0]\n"
        "b = cashflow.loc[cashflow['Стаття руху коштів'] == 'Надходження від реалізації продукції', 'Q1_2023'].values[0]\n"
        "result = (a - b) / b * 100",
}

# Deliberately faulty first attempts, demonstrating the L2 and L0 paths.
_FIRST_ATTEMPT_OVERRIDE = {
    "Яка маржа чистого прибутку (%) за Q4 2024?":
        # inverted formula: ~608% instead of ~16%, fails range and semantics -> L2
        "a = income.loc[income['Показник'] == 'Виручка', 'Q4_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Чистий прибуток', 'Q4_2024'].values[0]\n"
        "result = a / b * 100",
    "Яка сума фінансових витрат за 2023 рік?":
        # wrong row label -> IndexError -> L0 fuzzy repair without the LLM
        "cols = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']\n"
        "result = income.loc[income['Показник'] == 'Фінвитрати', cols].sum(axis=1).values[0]",
    # Same failure class on the stitched table. Note the `.sum(axis=1).values[0]`
    # form: `.values.sum()` returns 0.0 on an empty slice, so no exception is
    # raised and the pipeline silently accepts a wrong zero.
    "Яка сума витрат на маркетинг за 2024 рік?":
        "cols = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']\n"
        "result = cashflow.loc[cashflow['Стаття руху коштів'] == 'Витрати на маркетінг', cols].sum(axis=1).values[0]",
}

_CORRECT_RETRY = {
    "Яка маржа чистого прибутку (%) за Q4 2024?":
        "a = income.loc[income['Показник'] == 'Чистий прибуток', 'Q4_2024'].values[0]\n"
        "b = income.loc[income['Показник'] == 'Виручка', 'Q4_2024'].values[0]\n"
        "result = a / b * 100",
}


def _find_query(prompt: str):
    # longest match first: some queries are prefixes of others
    for q in sorted(_CODE_MAP, key=len, reverse=True):
        if q in prompt:
            return q
    return None


def _pick_by_scope(variants: dict, prompt: str) -> str:
    """Pick the variant matching whichever table retrieval placed in scope.

    Presence is detected from the schema block in the prompt.
    """
    positions = []
    for name, code in variants.items():
        idx = prompt.find(f"- {name}:")
        if idx >= 0:
            positions.append((idx, code))
    if positions:
        return min(positions)[1]
    return next(iter(variants.values()))


def _resolve(entry, prompt: str) -> str:
    return _pick_by_scope(entry, prompt) if isinstance(entry, dict) else entry


def _mock_llm(prompt: str) -> str:
    # semantic check branch
    if "Чи ця відповідь виглядає правдоподібно" in prompt:
        return _mock_semantic_check(prompt)

    q = _find_query(prompt)
    if q is None:
        return ("result = None  # MOCK: запит поза наборами test_queries_full.json / "
                "test_queries_phase1.json — встанови ANTHROPIC_API_KEY")

    is_retry = "попередня спроба" in prompt.lower()
    if is_retry and q in _CORRECT_RETRY:
        return _resolve(_CORRECT_RETRY[q], prompt)
    if not is_retry and q in _FIRST_ATTEMPT_OVERRIDE:
        return _resolve(_FIRST_ATTEMPT_OVERRIDE[q], prompt)
    return _resolve(_CODE_MAP[q], prompt)


def _mock_semantic_check(prompt: str) -> str:
    from core.query_semantics import is_relative_query

    m = re.search(r"Отримана числова відповідь:\s*([-\d.]+)", prompt)
    if not m:
        return "так"
    value = float(m.group(1))
    # Same predicate as the range signal, imported locally to avoid a cycle.
    if is_relative_query(prompt) and abs(value) > 200:
        return "ні"  # a margin in the hundreds of percent is implausible
    return "так"
