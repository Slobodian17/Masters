"""How much daily token budget is actually left, by the measured mechanism.

The naive estimate -- 200,000 minus the last 24 hours of spend -- is wrong, and
it told me the bucket was empty while roughly a third of it was available. TPD
is not a 24-hour sum: it is a bucket of 200,000 tokens that refills
continuously at 200,000/day = 2.314815 tokens/s, confirmed to three decimals
against the provider's own retry-after (5.14.5).

So the level has to be simulated forward over the ledger: start full, subtract
each call, add refill for the elapsed time, clamp at [0, 200000].

Run:  python -m eval.tpd_budget
      python -m eval.tpd_budget --need 144800
"""
import argparse
import json
import os
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "finqa_cache", "token_ledger.jsonl")

LIMIT = 200000
REFILL = LIMIT / 86400.0          # tokens per second


# The ledger holds every provider's calls, but the bucket being simulated is
# Groq's. Without this filter a Gemini run would inflate the Groq estimate and
# a Groq run would be paused for budget it never spent.
GROQ_MODELS = ("openai/gpt-oss-120b",)


def entries(path=LEDGER, model=None):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if model is not None and e.get("model") != model:
                    continue
                if model is None and e.get("model") not in GROQ_MODELS:
                    continue
                out.append(e)
    out.sort(key=lambda e: e["ts"])
    return out


def level(now=None, path=LEDGER, model=None):
    """Simulate the bucket. -> (available, spent_24h, first_ts, last_ts).

    `model` defaults to the Groq model the main results ran on; other
    providers have their own quotas and must not be mixed into this one.
    """
    now = now or time.time()
    e = entries(path, model)
    if not e:
        return LIMIT, 0, None, None
    # Start the simulation 24 h before the first logged call: by then anything
    # earlier has fully refilled, so the bucket was certainly full.
    t = e[0]["ts"]
    avail = float(LIMIT)
    for x in e:
        avail = min(LIMIT, avail + (x["ts"] - t) * REFILL)
        avail = max(0.0, avail - float(x.get("total") or 0))
        t = x["ts"]
    avail = min(LIMIT, avail + (now - t) * REFILL)
    spent24 = sum(x["total"] for x in e if x["ts"] >= now - 86400)
    return avail, spent24, e[0]["ts"], e[-1]["ts"]


def when(need, now=None, path=LEDGER, model=None):
    """Seconds until `need` tokens are available."""
    avail, *_ = level(now, path, model)
    if avail >= need:
        return 0.0
    return (need - avail) / REFILL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=float, default=0)
    args = ap.parse_args()

    now = time.time()
    avail, spent24, first, last = level(now)
    f = lambda t: datetime.fromtimestamp(t).strftime("%d.%m %H:%M")

    print("=" * 66)
    print("ДОБОВИЙ БЮДЖЕТ ЗА ЗМІРЯНОЮ МОДЕЛЛЮ КОВША")
    print("=" * 66)
    print(f"  ліміт ковша              : {LIMIT}")
    print(f"  долив                    : {REFILL:.6f} токен/с "
          f"({REFILL * 3600:.0f}/год)")
    print(f"  журнал                   : {f(first)} .. {f(last)}")
    print(f"  витрачено за 24 год      : {spent24:.0f}")
    print(f"  ДОСТУПНО ЗАРАЗ           : {avail:.0f}  ({avail / LIMIT:.0%})")
    print()
    print("  Наївна оцінка «200000 мінус сума за добу» дала б "
          f"{max(0, LIMIT - spent24):.0f} —")
    print("  вона не враховує долив і тому занижує залишок.")
    if args.need:
        w = when(args.need, now)
        print()
        print(f"  потрібно {args.need:.0f} токенів")
        if w <= 0:
            print("  -> доступно вже зараз")
        else:
            print(f"  -> через {w / 3600:.1f} год, тобто близько {f(now + w)}")


if __name__ == "__main__":
    main()
