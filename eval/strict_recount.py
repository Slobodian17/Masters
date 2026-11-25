"""Every headline number under the registered tolerance and under a strict one.

`is_correct` uses `tol * max(|expected|, 1)`. For |gold| >= 1 that is a 1%
relative tolerance, which is what FinQA's EA@1% means. For |gold| < 1 the
max() takes over and it becomes an ABSOLUTE window of 0.01 -- so on a gold of
0.00338 anything in [-0.0066, 0.0134] passes, including the value with the
opposite sign.

The registered verdict stands under the registered rule: 5.26 fixed it before
the run and rewriting it afterwards is exactly what pre-registration forbids.
This module reports both columns side by side so the reader can see how much
of any result rests on the looser window.

Zero API calls.

Run:  python -m eval.strict_recount
"""
import os

from eval import cache_io
from eval.finqa_confirm import build, decide

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")
TOL = 0.01


def registered(pred, exp, tol=TOL):
    if exp is None or pred is None:
        return False
    try:
        p, e = float(pred), float(exp)
    except (TypeError, ValueError):
        return False
    return abs(p - e) <= tol * max(abs(e), 1.0)


def strict(pred, exp, tol=TOL):
    """Relative only. A gold of 0 admits nothing but 0."""
    if exp is None or pred is None:
        return False
    try:
        p, e = float(pred), float(exp)
    except (TypeError, ValueError):
        return False
    if e == 0:
        return p == 0
    return abs(p - e) <= tol * abs(e)


def rows(tag):
    return cache_io.load_rows(os.path.join(CACHE, f"{tag}.jsonl"))[0]


def r2_of(x):
    v = x.get("candidate")
    return x.get("answer") if v is None else v


def line(label, a, b, n):
    d = (b - a) * 100
    flag = "" if abs(d) < 1e-9 else f"   {d:+.1f} п.п."
    print(f"  {label:38s} {a:6.1%}  {b:6.1%}{flag}")


def block(title, pairs, n):
    print()
    print(f"--- {title} ---")
    print(f"  {'метрика':38s} {'зареєстр.':>7s} {'строго':>7s}")
    for label, a, b in pairs:
        line(label, a, b, n)


def confirm300():
    code, dirr = rows("confirm300"), rows("baseline_table_confirm300")
    items = build(code, dirr)
    n = len(items)
    out = []
    for name, ok in (("зареєстроване", registered), ("строге", strict)):
        acc = [r for r in items if r["verdict"] == "ACCEPT"]
        corr = {r["id"]: ok(r["r2"], r["gold"]) for r in items}
        r1c = {r["id"]: ok(r["r1"], r["gold"]) for r in items}
        with_v = [r for r in items if r["r2"] is not None]
        wrong = [r for r in with_v if not corr[r["id"]]]
        rej = [r for r in with_v if r["verdict"] == "ABSTAIN"]
        out.append({
            "acc_full": sum(corr.values()) / n,
            "r1_full": sum(r1c.values()) / n,
            "accepted": sum(1 for r in acc if corr[r["id"]]) / len(acc),
            "coverage": len(acc) / n,
            "recall": sum(1 for r in wrong if r["verdict"] == "ABSTAIN") / len(wrong),
            "precision": sum(1 for r in rej if not corr[r["id"]]) / len(rej),
            "only_r1": sum(1 for r in items if r1c[r["id"]] and not corr[r["id"]]),
            "only_r2": sum(1 for r in items if corr[r["id"]] and not r1c[r["id"]]),
        })
    a, b = out
    block("ПІДТВЕРДЖУВАЛЬНИЙ ПРОГІН 5.26 (gpt-oss-120b, n=300)", [
        ("точність прийнятих", a["accepted"], b["accepted"]),
        ("покриття", a["coverage"], b["coverage"]),
        ("R2 при повному покритті", a["acc_full"], b["acc_full"]),
        ("R1 при повному покритті", a["r1_full"], b["r1_full"]),
        ("rejection recall", a["recall"], b["recall"]),
        ("rejection precision", a["precision"], b["precision"]),
    ], n)
    print(f"  {'лише R1 правий / лише R2':38s} "
          f"{a['only_r1']:3d}/{a['only_r2']:<3d} {b['only_r1']:6d}/{b['only_r2']}")
    print()
    print("  Критерії 5.26.4 під строгим правилом:")
    for lab, v, thr in (("recall >= 40%", b["recall"], 0.40),
                        ("precision >= 60%", b["precision"], 0.60),
                        ("приріст >= 10 п.п.",
                         b["accepted"] - b["acc_full"], 0.10)):
        print(f"    {lab:22s} {v:6.1%}  {'виконано' if v >= thr else 'НЕ виконано'}")


def dev100(label, code_tag, r1_tag, arb_tag):
    from eval.finqa_arbiter import decide as sysrule
    with open(os.path.join(HERE, "finqa_dev100.ids"), encoding="utf-8") as f:
        ids = {l.strip() for l in f if l.strip()}
    code = {x["id"]: x for x in rows(code_tag)}
    r1 = {x["id"]: x for x in rows(r1_tag)}
    arb = {x["id"]: x.get("arbiter") for x in rows(arb_tag)}
    ids = {q for q in ids if q in code and q in r1}
    n = len(ids)
    res = []
    for ok in (registered, strict):
        a = b = c = 0
        for q in ids:
            gold = code[q].get("gold")
            v2, v1 = r2_of(code[q]), r1[q].get("answer")
            route, val = sysrule(v1, v2)
            sysv = {"AGREE": val, "R1": v1, "R2": v2}.get(route, arb.get(q))
            a += ok(v1, gold)
            b += ok(v2, gold)
            c += ok(sysv, gold)
        res.append((a / n, b / n, c / n))
    (a1, b1, c1), (a2, b2, c2) = res
    block(f"{label} (n={n})", [
        ("R1 пряма", a1, a2), ("R2 код", b1, b2),
        ("система + розбір", c1, c2)], n)


def tatdqa():
    from eval import tatdqa_loader as T
    import random
    import zipfile
    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    random.Random(42).shuffle(sel)
    sample = sel[:40]
    by = {q["uid"]: q for q in sample}
    m = "gemini-flash-lite-latest"
    out = []
    for arm in ("gold", "pdf"):
        d = {x["id"]: x for x in rows(f"tatdqa_{arm}_{m}")}
        reg = sum(1 for q in by if d.get(q, {}).get("correct")) / 40
        st = 0
        for q, qq in by.items():
            x = d.get(q)
            if not x:
                continue
            g = T.numeric_answer(qq)
            v = x.get("candidate")
            if v is None or g is None:
                continue
            mult = T.SCALES.get((qq.get("scale") or "").strip(), 1.0)
            try:
                p = float(v)
            except (TypeError, ValueError):
                continue
            if any(strict(cand, g) for cand in (p, p / mult if mult else p, p * mult)):
                st += 1
        out.append((reg, st / 40))
    block("TAT-DQA наскрізний (flash-lite, n=40)", [
        ("gold — верхня межа", out[0][0], out[0][1]),
        ("pdf — реальний режим", out[1][0], out[1][1])], 40)
    print(f"  {'різниця gold − pdf':38s} "
          f"{(out[0][0] - out[1][0]) * 100:5.1f}  "
          f"{(out[0][1] - out[1][1]) * 100:5.1f} п.п.")


def main():
    print("=" * 74)
    print("ПЕРЕРАХУНОК: зареєстрований допуск проти строго відносного")
    print("=" * 74)
    print("  зареєстр. : |p-e| <= 0.01 * max(|e|, 1)   — як у 5.26, вердикт за ним")
    print("  строго    : |p-e| <= 0.01 * |e|           — без абсолютного вікна")
    confirm300()
    dev100("ЕТАП 1 (gpt-oss-120b, dev)", "confirm300_values",
           "baseline_table_confirm300", "arbiter_confirm300_values")
    dev100("ЕТАП 2 (flash-lite, dev)", "gemlite100",
           "baseline_table_confirm300_values_gemlite100", "arbiter_gemlite100")
    tatdqa()


if __name__ == "__main__":
    main()
