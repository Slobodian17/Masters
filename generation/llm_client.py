"""LLM abstraction for code generation and the semantic verification signal.

Provider is selected through the environment:

    VKSK_LLM_PROVIDER = groq (default) | gemini | mock
    VKSK_LLM_MODEL    = model override
    GROQ_API_KEY      = key for groq
    GEMINI_API_KEY    = key for gemini

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
from core import lang

# Groq is the default provider: the project runs on free API tiers only, and
# it is the one with a usable free tier for code generation.

PROVIDER_ENV = "VKSK_LLM_PROVIDER"
MODEL_ENV = "VKSK_LLM_MODEL"

DEFAULT_PROVIDER = "groq"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Chosen by the hardest task it is given, which is pandas code generation
# rather than classification, and used for both the baseline and the method so
# that comparisons isolate the method rather than the model.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

# Second free provider, added so a result can be repeated on a different model
# without spending the Groq bucket. Google AI Studio has its own quota, so the
# two never compete for budget.
#
# Catalogue checked live 2026-09-27: `gemma-3-27b-it` no longer exists -- the
# Gemma line moved to 4 (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`). The
# general-purpose Flash-Lite pointer is `gemini-flash-lite-latest`.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_DEFAULT_MODEL = "gemini-flash-lite-latest"

# Gemini 2.5+ spends output budget on internal thinking, the same way gpt-oss
# does, and the same way it produced empty replies at max_tokens=600 (defect
# 15). Unset means the provider default and is recorded as such; 0 disables
# thinking where the model allows it.
GEMINI_THINKING_ENV = "VKSK_GEMINI_THINKING"

# Gemma on the free tier is slow -- measured 14-40 s per call -- so the 60 s
# shared timeout sits too close to normal latency and turns a slow reply into
# a lost one.
GEMINI_TIMEOUT = float(os.environ.get("VKSK_GEMINI_TIMEOUT", "90"))
GEMINI_TIMEOUT_RETRIES = int(os.environ.get("VKSK_GEMINI_TIMEOUT_RETRIES", "2"))

MAX_TOKENS = 400

# Code generation needs far more room than a short answer, and running out of
# it does not truncate visibly -- it yields code cut mid-literal. Measured on
# FinQA: at 400 tokens, 19 of 21 failing iterations were a cut-off line, the
# commonest being an unterminated string in a long row label, or the code
# stopping on the bare word `result` before its assignment. Row labels in real
# filings run to sixty characters and the schema lists every one of them.
#
# Raising this costs throughput rather than requests: Groq reserves max_tokens
# against the per-minute token limit, so a larger value paces the run slower.
CODEGEN_MAX_TOKENS = int(os.environ.get("VKSK_CODEGEN_MAX_TOKENS", "1200"))
REQUEST_TIMEOUT = 60

# 429 is an expected state on the free tier, not an error: retry with backoff,
# honouring Retry-After when present.
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


class LLMError(RuntimeError):
    """Provider returned an error that retrying will not fix."""


# ---- cost and latency accounting ----
#
# Calls are also counted per `tag`, so a run can be broken down by purpose:
# code generation, the semantic verification signal, and the no-RAG baseline
# each hit the API separately and are budgeted separately.

_STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "elapsed_s": 0.0, "retries": 0, "provider": None, "model": None,
          "by_tag": {}, "last_finish_reason": None}


def _blank_tag() -> dict:
    return {"calls": 0, "retries": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "elapsed_s": 0.0}


def reset_stats():
    _STATS.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "elapsed_s": 0.0, "retries": 0, "by_tag": {}})


def get_stats() -> dict:
    s = dict(_STATS)
    s["avg_latency_s"] = (s["elapsed_s"] / s["calls"]) if s["calls"] else 0.0
    s["total_tokens"] = s["prompt_tokens"] + s["completion_tokens"]
    return s


def was_truncated() -> bool:
    """True when the provider stopped the last reply at the token limit.

    Read from finish_reason rather than guessed from the downstream failure.
    A cut reply is not reported as an error: the text simply ends mid-line,
    and whatever breaks next -- an unterminated string, a missing `result` --
    names the symptom and hides the cause. Measured on FinQA, that
    misattribution sent 112 of 116 failures down the repair path, where the
    model regenerated under the same limit and was cut again.
    """
    return _STATS.get("last_finish_reason") == "length"


def active_provider() -> str:
    """Provider actually in effect: 'groq', 'gemini' or 'mock'.

    MOCK is not a configured provider but the consequence of a missing key.
    """
    provider = os.environ.get(PROVIDER_ENV, DEFAULT_PROVIDER).strip().lower()
    if provider == "mock":
        return "mock"
    if provider == "groq":
        return "groq" if os.environ.get("GROQ_API_KEY") else "mock"
    if provider == "gemini":
        return "gemini" if os.environ.get("GEMINI_API_KEY") else "mock"
    raise LLMError(
        f"Unknown {PROVIDER_ENV}={provider!r}; expected groq|gemini|mock")


def active_model() -> str:
    provider = active_provider()
    if provider == "mock":
        return "mock"
    default = {"groq": GROQ_DEFAULT_MODEL,
               "gemini": GEMINI_DEFAULT_MODEL}[provider]
    return os.environ.get(MODEL_ENV, default)


def call_llm(prompt: str, system: str = None, max_tokens: int = None,
             reasoning_effort: str = None, tag: str = "other",
             temperature: float = None) -> str:
    """Call the active provider.

    `max_tokens` overrides the default for one call. Batch tasks need far more
    than code generation, because gpt-oss models spend part of the budget on
    internal reasoning tokens; too small a limit yields an empty response
    rather than a truncated one.

    `reasoning_effort` ("low"/"medium"/"high") caps that internal reasoning.
    "low" is right for structured tasks such as label classification and keeps
    calls within the per-minute token limit. Leave it off for code generation.

    `tag` labels the call's purpose so a run can be broken down by where the
    API budget went.

    `temperature` defaults to 0, which is what every reproducible run uses.
    It is exposed for one experiment only: comparing diversity obtained by
    sampling one model repeatedly against diversity obtained by taking a
    different route through it (5.28).
    """
    provider = active_provider()
    if provider == "groq":
        return _call_groq(prompt, system, max_tokens, reasoning_effort, tag,
                          temperature)
    if provider == "gemini":
        return _call_gemini(prompt, system, max_tokens, tag, temperature)
    return _mock_llm(prompt)


# ---- daily-token ledger ----
#
# The per-day token limit is invisible: it is absent from the response
# headers, which report the per-MINUTE limit under a name that looks daily,
# and it appears only inside the body of a 429. Section 5.14.5 therefore had
# to leave open whether it charges tokens a call actually used or the
# max_tokens it reserved -- a difference of roughly 4x in how long a run
# takes, and the reason three successive explanations of a slow run were
# wrong.
#
# Settling it needs both sums known at the instant the limit is reached. The
# ledger keeps them: one line per successful call with actual usage and the
# reservation beside it, and on the first daily 429 a snapshot pairing the
# provider's own "Used N" against both of our totals. Off unless a run asks
# for it, so MOCK and the frozen regressions are untouched.
LEDGER_ENV = "VKSK_TOKEN_LEDGER"

# A daily-limit 429 is not one situation but two, and the difference is in the
# body. TPD is a bucket of 200,000 tokens refilled continuously at
# 200,000/day = 2.3148 tokens/s -- measured, and confirmed by the retry-after
# matching deficit/rate to three decimals (5.14.5). So the wait Groq asks for
# is short whenever the deficit is small: the first one observed was 53.568 s.
#
# Failing fast was right while the deficit was believed to be hours. Applied to
# a 53-second wait it throws away a run over a pause, which is what would end a
# long paced pass over the remaining questions. Wait when the body says the
# wait is short; fail only when it is genuinely long.
TPD_WAIT_MAX = float(os.environ.get("VKSK_TPD_WAIT_MAX", "240"))
TPD_WAIT_TRIES = int(os.environ.get("VKSK_TPD_WAIT_TRIES", "6"))

_RETRY_IN = re.compile(
    r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", re.IGNORECASE)


def parse_retry_after_body(text: str):
    """Seconds Groq asks us to wait, read from the 429 body, or None.

    The headers carry Retry-After for per-minute limits but not for the daily
    one; only the message text has it, in the form "Please try again in
    53.568s" or "... in 1m2.5s".
    """
    m = _RETRY_IN.search(str(text))
    if not m:
        return None
    mins = float(m.group(1) or 0)
    return mins * 60.0 + float(m.group(2))


def _ledger_path():
    return os.environ.get(LEDGER_ENV) or ""


def _ledger_write(entry: dict):
    path = _ledger_path()
    if not path:
        return
    try:
        import json
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Accounting must never take a run down with it.
        pass


def _ledger_snapshot_429(body: str, model: str):
    """Record what our own totals were when the daily limit said no."""
    path = _ledger_path()
    if not path:
        return
    try:
        import json
        actual = reserved = calls = 0
        since = time.time() - 24 * 3600
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("model") != model or float(e.get("ts", 0)) < since:
                        continue
                    actual += int(e.get("total") or 0)
                    reserved += int(e.get("reserved") or 0)
                    calls += 1
        out = path.replace(".jsonl", "") + ".tpd429.json"
        if os.path.exists(out):
            return          # the FIRST 429 is the informative one
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "model": model,
                       "calls_24h": calls,
                       "actual_tokens_24h": actual,
                       "reserved_tokens_24h": reserved,
                       "body": body[:600]}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _record(elapsed: float, usage: dict, provider: str, model: str,
            retries: int = 0, tag: str = "other", reserved: int = 0):
    _STATS["calls"] += 1
    _STATS["elapsed_s"] += elapsed
    _STATS["retries"] += retries
    _STATS["provider"] = provider
    _STATS["model"] = model
    p = int((usage or {}).get("prompt_tokens") or 0)
    c = int((usage or {}).get("completion_tokens") or 0)
    _STATS["prompt_tokens"] += p
    _STATS["completion_tokens"] += c

    t = _STATS["by_tag"].setdefault(tag, _blank_tag())
    t["calls"] += 1
    t["retries"] += retries
    t["elapsed_s"] += elapsed
    t["prompt_tokens"] += p
    t["completion_tokens"] += c

    _ledger_write({"ts": time.time(), "model": model, "tag": tag,
                   "prompt": p, "completion": c, "total": p + c,
                   "reserved": int(reserved or 0), "retries": retries})


def _record_failure(tag: str, retries: int, elapsed: float):
    """A call that raised still consumed API requests; count it separately.

    Without this, a failed semantic check would look free, and the request
    budget in a report would come out lower than what the provider saw.
    """
    t = _STATS["by_tag"].setdefault(tag, _blank_tag())
    t.setdefault("failed_calls", 0)
    t["failed_calls"] += 1
    t["retries"] += retries
    t["elapsed_s"] += elapsed
    _STATS["retries"] += retries
    _STATS.setdefault("failed_calls", 0)
    _STATS["failed_calls"] += 1


def _call_groq(prompt, system, max_tokens=None, reasoning_effort=None,
               tag="other", temperature=None):
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
               "messages": messages,
               "temperature": 0 if temperature is None else float(temperature)}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    started = time.time()
    retries = 0
    tpd_waits = 0
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 429 and "tokens per day" in resp.text.lower():
            # A daily budget does not come back in thirty seconds. Retrying it
            # five times with backoff spends minutes to learn nothing, and it
            # was doing exactly that for two hours before the 429 body was
            # read rather than the headers.
            #
            # But the body also says how long the wait actually is, and the
            # bucket refills continuously, so a small deficit means seconds.
            # Wait those out; fail only on a wait that really is long.
            _ledger_snapshot_429(resp.text, model)
            wait = parse_retry_after_body(resp.text)
            if (wait is not None and wait <= TPD_WAIT_MAX
                    and tpd_waits < TPD_WAIT_TRIES):
                tpd_waits += 1
                retries += 1
                time.sleep(wait + 1.0)
                continue
            _record_failure(tag, retries, time.time() - started)
            raise LLMError(f"Groq TPD exhausted: {resp.text[:600]}")
        if resp.status_code == 429 or resp.status_code >= 500:
            retries += 1
            if attempt == MAX_RETRIES - 1:
                _record_failure(tag, retries, time.time() - started)
                raise LLMError(f"Groq {resp.status_code} after {MAX_RETRIES} attempts: {resp.text[:600]}")
            wait = float(resp.headers.get("retry-after") or BACKOFF_BASE ** attempt)
            time.sleep(min(wait, 30.0))
            continue
        if resp.status_code != 200:
            _record_failure(tag, retries + 1, time.time() - started)
            raise LLMError(f"Groq {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        _record(time.time() - started, data.get("usage") or {}, "groq", model,
                retries, tag, reserved=payload["max_tokens"])
        _STATS["last_finish_reason"] = data["choices"][0].get("finish_reason")
        return _strip_code_fences(data["choices"][0]["message"]["content"])

    _record_failure(tag, retries, 0.0)
    raise LLMError("Groq: retries exhausted")


def _call_gemini(prompt, system, max_tokens=None, tag="other",
                 temperature=None):
    """Google AI Studio, plain HTTP, mirroring the Groq branch.

    Written the same way and for the same reasons: explicit parameters rather
    than defaults nobody can see afterwards, actual token accounting from the
    provider's own usage block, and a fail-fast on the daily quota so a run
    does not spend an hour in backoff learning nothing (5.14.5).

    Its quota is separate from Groq's, which is the point: a repetition on
    another model costs nothing from the bucket the main results ran on.
    """
    import requests

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    model = os.environ.get(MODEL_ENV, GEMINI_DEFAULT_MODEL)
    gen = {"temperature": 0.0 if temperature is None else float(temperature),
           "maxOutputTokens": int(max_tokens or MAX_TOKENS)}
    thinking = os.environ.get(GEMINI_THINKING_ENV, "").strip()
    if thinking != "":
        # Recorded either way: absent means the provider default, which is
        # what the run report has to state rather than imply.
        gen["thinkingConfig"] = {"thinkingBudget": int(thinking)}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": gen}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
    started = time.time()
    retries = 0
    timeouts = 0
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url, headers={"x-goog-api-key": api_key,
                              "Content-Type": "application/json"},
                json=payload, timeout=GEMINI_TIMEOUT)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout) as exc:
            # Measured on the free tier: a reply occasionally never arrives.
            # Two more tries, then give up -- and the caller must not write a
            # cache row for it, because no answer was ever produced.
            timeouts += 1
            retries += 1
            if timeouts > GEMINI_TIMEOUT_RETRIES:
                _record_failure(tag, retries, time.time() - started)
                raise LLMError(
                    f"Gemini timeout after {timeouts} attempts: {type(exc).__name__}")
            time.sleep(2.0)
            continue

        if resp.status_code == 429:
            text = resp.text
            low = text.lower()
            # A per-day quota does not clear in thirty seconds. Per-minute does.
            daily = ("perday" in low.replace("_", "").replace(" ", "")
                     or "per day" in low)
            wait = _gemini_retry_delay(text)
            if daily and (wait is None or wait > TPD_WAIT_MAX):
                _record_failure(tag, retries, time.time() - started)
                _ledger_snapshot_429(text, model)
                raise LLMError(f"Gemini daily quota exhausted: {text[:600]}")
            retries += 1
            if attempt == MAX_RETRIES - 1:
                _record_failure(tag, retries, time.time() - started)
                raise LLMError(f"Gemini 429 after {MAX_RETRIES} attempts: {text[:600]}")
            time.sleep(min(wait if wait is not None else BACKOFF_BASE ** attempt, 60.0))
            continue

        if resp.status_code >= 500:
            retries += 1
            if attempt == MAX_RETRIES - 1:
                _record_failure(tag, retries, time.time() - started)
                raise LLMError(f"Gemini {resp.status_code} after {MAX_RETRIES}: {resp.text[:300]}")
            time.sleep(min(BACKOFF_BASE ** attempt, 30.0))
            continue

        if resp.status_code != 200:
            _record_failure(tag, retries + 1, time.time() - started)
            raise LLMError(f"Gemini {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        um = data.get("usageMetadata") or {}
        # thoughtsTokenCount is billed as output and is NOT inside
        # candidatesTokenCount, so leaving it out would understate the cost of
        # exactly the models that spend the most on it.
        usage = {"prompt_tokens": int(um.get("promptTokenCount") or 0),
                 "completion_tokens": int((um.get("candidatesTokenCount") or 0)
                                          + (um.get("thoughtsTokenCount") or 0))}
        _record(time.time() - started, usage, "gemini", model, retries, tag,
                reserved=gen["maxOutputTokens"])
        cands = data.get("candidates") or [{}]
        _STATS["last_finish_reason"] = (
            "length" if cands[0].get("finishReason") == "MAX_TOKENS"
            else cands[0].get("finishReason"))
        parts = ((cands[0].get("content") or {}).get("parts") or [])
        text = "".join(p.get("text", "") for p in parts)
        return _strip_code_fences(text)

    _record_failure(tag, retries, 0.0)
    raise LLMError("Gemini: retries exhausted")


_GEM_DELAY = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


def _gemini_retry_delay(text: str):
    """Seconds from the RetryInfo block of a 429 body, or None."""
    m = _GEM_DELAY.search(str(text))
    return float(m.group(1)) if m else None


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


def _find_query(prompt: str, code_map: dict):
    # longest match first: some queries are prefixes of others
    for q in sorted(code_map, key=len, reverse=True):
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


def _mock_maps():
    """MOCK lookup tables for the active language.

    Ukrainian lives in this module because the frozen phase-0 baseline is keyed
    on it verbatim; English lives in its own module.
    """
    if lang.active_lang() == "uk":
        return _CODE_MAP, _FIRST_ATTEMPT_OVERRIDE, _CORRECT_RETRY
    from generation import mock_en
    return mock_en.CODE_MAP, mock_en.FIRST_ATTEMPT_OVERRIDE, mock_en.CORRECT_RETRY


def _mock_llm(prompt: str) -> str:
    prof = lang.get()
    if prof.SEMANTIC_MARKER in prompt:
        return _mock_semantic_check(prompt)

    code_map, first_attempt, correct_retry = _mock_maps()
    q = _find_query(prompt, code_map)
    if q is None:
        return ("result = None  # MOCK: query outside the test sets "
                f"({prof.TEST_SET_FULL} / {prof.TEST_SET_PHASE1})")

    is_retry = prof.RETRY_MARKER in prompt.lower()
    if is_retry and q in correct_retry:
        return _resolve(correct_retry[q], prompt)
    if not is_retry and q in first_attempt:
        return _resolve(first_attempt[q], prompt)
    return _resolve(code_map[q], prompt)


def _mock_semantic_check(prompt: str) -> str:
    from core.query_semantics import is_relative_query

    prof = lang.get()
    m = re.search(prof.SEMANTIC_VALUE_RE, prompt)
    if not m:
        return prof.YES_WORDS[0]
    value = float(m.group(1))
    # Same predicate as the range signal, imported locally to avoid a cycle.
    if is_relative_query(prompt) and abs(value) > 200:
        return prof.NO_WORDS[0]  # a margin in the hundreds of percent is implausible
    return prof.YES_WORDS[0]
