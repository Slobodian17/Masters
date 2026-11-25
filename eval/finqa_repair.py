"""Judge-guided repair, and the accuracy of the whole system under three policies.

The decomposition of 5.36 says where repair should aim: 72.7% of the wrong
answers read the wrong cells, and only 19.3% read the right cells and combine
them wrongly. So repair here does NOT regenerate the program. The cells are
already located -- by the judge, which chose them from candidates -- and the
model is asked for one thing it has not yet been asked for: the arithmetic over
those operands, named A, B, C, with no table lookup left to get wrong.

That is the `cell()` primitive of the proposed architecture in its cheapest
form: locating and combining are separated, and each is repaired on its own.

Neither the judge nor the repair is shown the direct route's answer, the code's
answer or the gold. The judge sees the question and a list of candidate cells;
the repair sees the question and the chosen cells. What differs between the
policies is only WHEN repair is triggered.

  P1  trigger = the two routes disagree (the registered rule of 5.26).
      Repair takes over; the direct route is used only if repair returns no
      number. R1 is a trigger here, never content.
  P2  trigger = the routes disagree OR the judge chose different cells than
      the program read. Catches the class route agreement cannot: both routes
      agreeing on the same wrong number (53 of 300 in 5.26).
  P3  P2's trigger, but R1 is not used at all -- a failed repair falls back to
      the program's own answer. This is the only policy whose accuracy can be
      compared with R1 without having consumed R1.

No policy abstains: every question gets an answer and is scored.

`--project` scores the policies with zero API calls, assuming repair succeeds
exactly when the judge's cells cover the gold operands. That is an upper
bound, reported as one.

Run:  python -m eval.finqa_repair --tag confirm300 --project
      python -m eval.finqa_repair --tag confirm300 --policy P2
"""
import argparse
import os
import re
import time
from math import comb

from eval import cache_io
from eval.finqa_confirm import build, decide, is_correct
from eval.finqa_judge import covers, same_set
from generation import llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

MAX_TOKENS = 900
SLEEP = 4.2

SYSTEM = ("You write one arithmetic expression. You answer with one line and "
          "nothing else.")

TEMPLATE = """Question: {q}

These are the numbers from the table that the question needs:
{ops}

Write a single Python expression over these names that answers the question.
You may use numeric constants such as 100 or 1000 for unit or percent
conversion. Do not look anything else up.

Reply with exactly one line, no explanation:
RESULT: <expression>"""

NAMES = "ABCDEFGHIJKL"
POLICIES = ("P1", "P2", "P3")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def render_ops(cells):
    return "\n".join(
        f'{NAMES[i]} = {c["value"]:g}   (row "{c["label"][:44]}", '
        f'column "{c["col"][:24]}")'
        for i, c in enumerate(cells[:len(NAMES)]))


def parse_expr(raw):
    m = re.search(r"RESULT\s*:?\s*(.+)", str(raw or ""))
    expr = (m.group(1) if m else str(raw or "")).strip()
    expr = expr.strip("`").split("\n")[0].strip()
    if not expr or re.search(r"[^A-L0-9_+\-*/(). ]", expr):
        return None
    return expr


def evaluate(expr, cells):
    ns = {NAMES[i]: c["value"] for i, c in enumerate(cells[:len(NAMES)])}
    try:
        v = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 - filtered above
        return float(v)
    except Exception:
        return None


# ---- what each policy does with one question ----

def triggers(r, j, policy):
    """Does this policy send the question to repair?"""
    route_agree = decide(r["r1"], r["r2"])[0] == "ACCEPT"
    judge_agree = j is not None and same_set(j["chosen"], j["measured"])
    if policy == "P1":
        return not route_agree
    return (not route_agree) or (not judge_agree)


def resolve(r, j, repaired, policy):
    """-> (value, source). No abstention."""
    if r["r2"] is None or j is None:
        # The program produced no number, so there is nothing to judge.
        if policy == "P3":
            return r["r2"], "code"
        return r["r1"], "direct"
    if not triggers(r, j, policy):
        return r["r2"], "code"
    rp = repaired.get(r["id"])
    v = rp.get("value") if rp else None
    if v is not None:
        return v, "repair"
    return (r["r2"], "code") if policy == "P3" else (r["r1"], "direct")


def score(rows, judge, repaired, policy):
    out = {"policy": policy, "ok": 0, "per_q": {},
           "src": {}, "ok_by_src": {}}
    for r in rows:
        pick, src = resolve(r, judge.get(r["id"]), repaired, policy)
        ok = pick is not None and bool(is_correct(pick, r["gold"]))
        out["ok"] += ok
        out["per_q"][r["id"]] = ok
        out["src"][src] = out["src"].get(src, 0) + 1
        out["ok_by_src"][src] = out["ok_by_src"].get(src, 0) + ok
    return out


def project(judge):
    """Zero-call upper bound: repair succeeds iff the judge's cells are right."""
    return {qid: {"value": (j["gold"] if covers(j["chosen"], j["gold_ops"])
                            is True else None)}
            for qid, j in judge.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--project", action="store_true",
                    help="нуль викликів: стеля, не вимір")
    ap.add_argument("--policy", default="P2",
                    help="для якої політики ремонтувати (P2 покриває P1)")
    args = ap.parse_args()

    jrows = [r for r in cache_io.load_rows(
        os.path.join(CACHE, f"judge_{args.tag}.jsonl"))[0] if not r.get("error")]
    judge = {r["id"]: r for r in jrows}

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)
    n = len(rows)
    r1 = sum(1 for r in rows if r["r1_correct"])
    r2 = sum(1 for r in rows if r["correct"])

    need = {p: [r for r in rows if r["r2"] is not None
                and judge.get(r["id"]) is not None
                and triggers(r, judge.get(r["id"]), p)] for p in POLICIES}
    print(f"суддя: {len(judge)} питань")
    for p in POLICIES:
        print(f"  {p}: ремонтувати {len(need[p])} питань")

    rep_path = os.path.join(CACHE, f"repair_{args.tag}.jsonl")
    if args.project:
        repaired = project(judge)
        print("РЕЖИМ СТЕЛІ: ремонт не запускався; припущення — "
              "«правильні клітинки -> правильна відповідь»")
    else:
        done = {r["id"] for r in cache_io.load_rows(rep_path)[0]}
        todo = [r for r in need[args.policy] if r["id"] not in done]
        print(f"ремонтувати зараз ({args.policy}): {len(todo)}, "
              f"уже є {len(done)}")
        with open(rep_path, "a", encoding="utf-8") as fh:
            for k, r in enumerate(todo, 1):
                j = judge[r["id"]]
                cells = j["chosen_cells"] or j["measured_cells"]
                if not cells:
                    continue
                prompt = TEMPLATE.format(q=j["question"], ops=render_ops(cells))
                llm_client.reset_stats()
                raw, err, err_class = "", None, None
                try:
                    raw = llm_client.call_llm(prompt, system=SYSTEM,
                                              tag="repair",
                                              max_tokens=MAX_TOKENS)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    err_class = type(exc).__name__
                expr = parse_expr(raw)
                val = evaluate(expr, cells) if expr else None
                row = {"id": r["id"], "expr": expr, "value": val,
                       "n_cells": len(cells), "gold": j["gold"],
                       "raw": str(raw)[:300],
                       "tokens": (llm_client.get_stats() or {}).get("total_tokens"),
                       "error": err, "error_class": err_class}
                try:
                    cache_io.write_row(fh, row)
                except cache_io.FakeRow as bad:
                    print(f"  [{k}] {r['id']}: транспорт, не записано — {bad}")
                    break
                good = val is not None and bool(is_correct(val, j["gold"]))
                print(f"  [{k}/{len(todo)}] {r['id']} {expr} = "
                      f"{val if val is None else round(val, 6)} "
                      f"(еталон {j['gold']}) {'OK' if good else 'ні'}")
                time.sleep(SLEEP)
        repaired = {r["id"]: r for r in cache_io.load_rows(rep_path)[0]
                    if not r.get("error")}

    print()
    print("=" * 78)
    print(f"ТОЧНІСТЬ СИСТЕМИ ПО ВСІХ {n} ПИТАННЯХ, БЕЗ ВІДМОВ — {args.tag}"
          + ("  [СТЕЛЯ]" if args.project else "  [ВИМІР]"))
    print("=" * 78)
    print()
    print(f"  {'варіант':40s} {'правильних':>11s} {'точність':>9s}")
    print(f"  {'кодовий маршрут R2 сам':40s} {r2:11d} {r2 / n:9.1%}")
    print(f"  {'прямий маршрут R1 сам':40s} {r1:11d} {r1 / n:9.1%}")
    res = {p: score(rows, judge, repaired, p) for p in POLICIES}
    labels = {"P1": "P1 розбіжність маршрутів -> ремонт",
              "P2": "P2 + суддя й на згодних",
              "P3": "P3 те саме, але БЕЗ R1 узагалі"}
    for p in POLICIES:
        s = res[p]
        print(f"  {labels[p]:40s} {s['ok']:11d} {s['ok'] / n:9.1%}")

    for p in POLICIES:
        s = res[p]
        b = sum(1 for r in rows if s["per_q"][r["id"]] and not r["r1_correct"])
        c = sum(1 for r in rows if not s["per_q"][r["id"]] and r["r1_correct"])
        print()
        print(f"--- {labels[p]} ПРОТИ R1 ({r1}/{n} = {r1 / n:.1%}) ---")
        print(f"  виправлено проти R1 : {b}")
        print(f"  зламано проти R1    : {c}")
        print(f"  різниця {(s['ok'] - r1) / n:+.1f} п.п."
              .replace(f"{(s['ok'] - r1) / n:+.1f}",
                       f"{100 * (s['ok'] - r1) / n:+.1f}")
              + f", Макнемар p = {mcnemar(b, c):.4f}")
        print("  звідки бралась відповідь:")
        for k in ("code", "repair", "direct"):
            tot = s["src"].get(k, 0)
            if tot:
                print(f"    {k:8s} {tot:3d} питань, правильних "
                      f"{s['ok_by_src'][k]:3d} = {s['ok_by_src'][k] / tot:5.1%}")

    if not args.project:
        tok = sum(r.get("tokens") or 0 for r in repaired.values())
        print()
        print(f"--- ЦІНА РЕМОНТУ: {len(repaired)} викликів, "
              f"{tok:,} токенів ---")


if __name__ == "__main__":
    main()
