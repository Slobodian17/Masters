"""LLM paraphrase layer plus an automatic label-disagreement detector.

The combinatorial core produces only the variety built into it, so a model
trained on it alone risks learning the generator's templates rather than the
language. Paraphrases add natural phrasing.

A paraphrase inherits its seed's label, but the model can shift the meaning,
so each one is checked by a second, different model. Agreement accepts it;
disagreement excludes it and sets it aside for manual review. Using a
different model matters: one model checking its own output would agree with
itself.
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import time

from generation import llm_client
from generation.dataset_core import TAXONOMY, build as build_core

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_PATH = os.path.join(DATA_DIR, "query_type_dataset.json")
PENDING_PATH = os.path.join(DATA_DIR, "query_type_disagreements.json")

GENERATOR_MODEL = "openai/gpt-oss-120b"
ARBITER_MODEL = "openai/gpt-oss-20b"

BATCH = 8  # скільки seed-запитів віддаємо генератору за один виклик

# Batch paraphrasing needs far more than the default budget, both for the
# output and for internal reasoning tokens.
PARAPHRASE_MAX_TOKENS = 4000

SYSTEM = (
    "Ти переписуєш запити українською мовою до фінансових таблиць. "
    "Зберігаєш ТОЧНО той самий тип операції та ті самі показники й періоди, "
    "змінюєш лише формулювання. Відповідаєш лише валідним JSON."
)


def _say(msg: str):
    """Progress line with an explicit flush.

    Without it a redirected long run looks hung: stdout stays buffered.
    """
    print(msg, flush=True)
    sys.stdout.flush()


def _prompt(items, per_seed: int) -> str:
    nl = chr(10)
    listing = nl.join(f'{i}. {it["query"]}' for i, it in enumerate(items, start=1))
    fmt = '{"1": ["...", "..."], "2": ["...", "..."]}'
    return (
        f"Ось {len(items)} запитів до фінансових таблиць:{nl}{nl}{listing}{nl}{nl}"
        f"Для КОЖНОГО запиту напиши {per_seed} різних перефразувань українською.{nl}"
        f"Вимоги:{nl}"
        f"- зберігай тип операції (пошук значення, сума, середнє, максимум, "
        f"порівняння, приріст у відсотках, фільтр за умовою, відношення);{nl}"
        f"- зберігай ті самі показники, періоди й числові пороги;{nl}"
        f"- пиши природно, як пише жива людина: можна змінювати порядок слів, "
        f"вживати розмовні звороти, скорочувати;{nl}"
        f"- НЕ додавай пояснень і не змінюй суть.{nl}{nl}"
        f"Формат відповіді — рівно такий JSON: {fmt}"
    )


def _parse_json(raw: str) -> dict:
    """Extract JSON, tolerating markdown fences or surrounding prose."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def generate(items, per_seed: int, verbose: bool = True) -> list:
    """items: a subset of the core. -> paraphrases with inherited labels."""
    os.environ["VKSK_LLM_MODEL"] = GENERATOR_MODEL
    out = []
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]

    for bi, batch in enumerate(batches, start=1):
        try:
            raw = llm_client.call_llm(_prompt(batch, per_seed), system=SYSTEM,
                                      max_tokens=PARAPHRASE_MAX_TOKENS)
        except Exception as e:
            if verbose:
                _say(f"  батч {bi}/{len(batches)}: ПОМИЛКА {type(e).__name__}: {str(e)[:120]}")
            continue

        data = _parse_json(raw)
        added = 0
        for idx_str, variants in data.items():
            try:
                src = batch[int(idx_str) - 1]
            except (ValueError, IndexError):
                continue
            if not isinstance(variants, list):
                continue
            for v in variants:
                v = str(v).strip()
                if len(v) < 8:
                    continue
                out.append({
                    "query": v,
                    "type": src["type"],        # inherited label (a hypothesis)
                    "seed_id": src["seed_id"],  # same seed group
                    "metric": src["metric"],
                    "source": "llm_paraphrase",
                    "seed_query": src["query"],
                })
                added += 1
        if verbose:
            _say(f"  батч {bi}/{len(batches)}: +{added} перефразувань")
    return out


# ---------------------------- АРБІТР ----------------------------

ARBITER_BATCH = 20
SHUFFLE_SEED = 1234

# The binding limit is tokens per minute, not requests, and the provider
# reserves `max_tokens` against it rather than the actual response length.
ARBITER_MAX_TOKENS = 700

# Low reasoning effort is what makes arbitration viable on the free tier:
# without it the whole token budget goes to internal reasoning.
ARBITER_REASONING = "low"

# Spacing between calls, so retries are not burned against the token limit.
ARBITER_PACE_S = 5.0

_ARBITER_SYSTEM = ("Ти — класифікатор запитів до фінансових таблиць. "
                   "Відповідаєш лише валідним JSON, без пояснень.")


def _arbiter_prompt(queries) -> str:
    from generation.query_classifier_zeroshot import TAXONOMY as ZS_TAXONOMY
    from generation.query_classifier_zeroshot import _DESCRIPTIONS

    nl = chr(10)
    labels = nl.join(f"- {n}: {_DESCRIPTIONS[n]}" for n in ZS_TAXONOMY)
    listing = nl.join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
    fmt = '{"1": "МІТКА", "2": "МІТКА"}'
    return (f"Типи операцій:{nl}{labels}{nl}{nl}"
            f"Класифікуй КОЖЕН запит:{nl}{listing}{nl}{nl}"
            f"Формат відповіді — рівно такий JSON: {fmt}")


def arbitrate(rows, verbose: bool = True) -> tuple:
    """Check labels with a second model, in batches.

    Returns (accepted, disagreements, failed).

    Batching matters: one call per example would consume half the daily request
    budget, which is shared with the code generator and every eval run.

    Shuffling matters more. Paraphrases arrive grouped by class, so unshuffled
    batches would each hold a single class, which nudges the model into giving
    them all the same label. Measured directly: grouped order yields 0.8%
    disagreement, shuffled order 2.7%.

    Labels are never corrected automatically; a disagreement only means two
    sources differ, and which is right is for a human to decide.
    """
    from generation.query_classifier_zeroshot import TAXONOMY as ZS_TAXONOMY
    from generation.query_classifier_zeroshot import _strip_reasoning

    os.environ["VKSK_LLM_MODEL"] = ARBITER_MODEL
    accepted, disagreements, failed = [], [], []

    rows = list(rows)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    batches = [rows[i:i + ARBITER_BATCH] for i in range(0, len(rows), ARBITER_BATCH)]

    for bi, batch in enumerate(batches, start=1):
        if bi > 1:
            time.sleep(ARBITER_PACE_S)

        verdicts, batch_error = {}, None
        try:
            raw = llm_client.call_llm(_arbiter_prompt([r["query"] for r in batch]),
                                      system=_ARBITER_SYSTEM,
                                      max_tokens=ARBITER_MAX_TOKENS,
                                      reasoning_effort=ARBITER_REASONING)
            parsed = _parse_json(_strip_reasoning(raw))
            for k, v in parsed.items():
                label = str(v).strip().upper()
                if label in ZS_TAXONOMY:
                    verdicts[k] = label
        except Exception as e:
            batch_error = f"{type(e).__name__}: {str(e)[:100]}"
            if verbose:
                _say(f"  арбітр, батч {bi}/{len(batches)}: ЗБІЙ {batch_error}")

        for j, r in enumerate(batch, start=1):
            verdict = verdicts.get(str(j))
            if verdict is None:
                # A failed call is not a disagreement. Counting it as one
                # would let a single dropped batch add 20 phantom
                # disagreements and corrupt the rate being measured.
                failed.append({**r, "arbiter": "FAILED", "error": batch_error})
                continue
            row = {**r, "arbiter": verdict}
            (accepted if verdict == r["type"] else disagreements).append(row)

        if verbose:
            _say(f"  арбітр: батч {bi}/{len(batches)}, "
                 f"розбіжностей {len(disagreements)}, збоїв {len(failed)}")
    return accepted, disagreements, failed


# ---------------------------- ЗБІРКА ----------------------------

def load_existing_paraphrases() -> list:
    """All previously generated paraphrases: accepted plus set aside."""
    rows = []
    for path in (OUT_PATH, PENDING_PATH):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rows += [r for r in json.load(f) if r.get("source") == "llm_paraphrase"]

    seen, out = set(), []
    for r in rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        out.append({k: v for k, v in r.items() if k not in ("arbiter", "error")})
    return out


def _save(core: list, accepted: list, rejected: list):
    dataset = core + accepted
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=1)

    _say("")
    _say(f"Датасет: {len(dataset)} прикладів "
         f"({len(core)} комбінаторних + {len(accepted)} LLM), "
         f"seed-груп: {len({d['seed_id'] for d in dataset})}")
    _say(str(dict(collections.Counter(d["type"] for d in dataset))))
    _say(f"Збережено: {OUT_PATH}")
    _say(f"Виключено на ручну перевірку: {PENDING_PATH}")


def rearbitrate(core: list):
    """Rebuild the dataset without regenerating paraphrases.

    Generation is non-deterministic, so re-running it would produce a different
    set and break comparability with numbers already measured.
    """
    rows = load_existing_paraphrases()
    _say(f"Повторний арбітраж наявних перефразувань: {len(rows)}")
    _say(f"Арбітр: {ARBITER_MODEL}, батч {ARBITER_BATCH}, "
         f"reasoning_effort={ARBITER_REASONING}")
    _say("")

    llm_client.reset_stats()
    accepted, disagreements, failed = arbitrate(rows)
    judged = len(accepted) + len(disagreements)
    rate = len(disagreements) / judged if judged else 0.0

    _say("")
    _say(f"  оцінено: {judged} з {len(rows)}")
    _say(f"  прийнято: {len(accepted)}")
    _say(f"  розбіжностей (виключено з тренування): {len(disagreements)} ({rate:.1%})")
    _say(f"  збоїв виклику (теж виключено): {len(failed)}")

    _save(core, accepted, disagreements + failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", type=int, default=2, help="перефразувань на seed")
    ap.add_argument("--seeds", type=int, default=240, help="скільки seed-ів брати з ядра")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rearbitrate", action="store_true",
                    help="не генерувати наново: перепровести арбітраж наявних "
                         "перефразувань і перезібрати датасет")
    args = ap.parse_args()

    if llm_client.active_provider() == "mock":
        _say("LLM-шар потребує GROQ_API_KEY (див. .env). Ядро працює й без нього:")
        _say("    python -m generation.dataset_core")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    core = build_core(seed=args.seed)

    if args.rearbitrate:
        return rearbitrate(core)

    rng = random.Random(args.seed)
    per_class = max(1, args.seeds // len(TAXONOMY))
    chosen = []
    for t in TAXONOMY:
        rows = [c for c in core if c["type"] == t]
        chosen.extend(rng.sample(rows, min(per_class, len(rows))))

    _say(f"Ядро: {len(core)} запитів. Обрано для перефразування: {len(chosen)} seed-ів "
         f"({per_class} на клас), по {args.per_seed} варіантів.")
    _say(f"Генератор: {GENERATOR_MODEL}   Арбітр: {ARBITER_MODEL}")
    _say("")

    llm_client.reset_stats()
    t0 = time.time()
    _say("1) Генерація перефразувань:")
    paraphrases = generate(chosen, args.per_seed)
    gen_stats = llm_client.get_stats()
    _say(f"   отримано {len(paraphrases)} перефразувань за {time.time() - t0:.0f} с "
         f"({gen_stats['calls']} викликів, {gen_stats['retries']} ретраїв)")
    _say("")

    llm_client.reset_stats()
    _say("2) Арбітраж міток:")
    accepted, disagreements, failed = arbitrate(paraphrases)
    arb_stats = llm_client.get_stats()

    judged = len(accepted) + len(disagreements)
    rate = len(disagreements) / judged if judged else 0.0
    _say(f"   прийнято: {len(accepted)}")
    _say(f"   розбіжностей (виключено): {len(disagreements)} ({rate:.1%} від оцінених)")
    _say(f"   збоїв виклику (виключено з частки): {len(failed)}")
    _say(f"   викликів арбітра: {arb_stats['calls']}, ретраїв: {arb_stats['retries']}")

    _save(core, accepted, disagreements + failed)


if __name__ == "__main__":
    main()
