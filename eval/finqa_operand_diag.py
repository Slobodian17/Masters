"""Where the code route goes wrong: in the NUMBERS or in the FORMULA?

FinQA ships a gold program with the exact operands, e.g. `divide(420.5, 750.0)`.
That makes the split checkable rather than guessed: take the numbers the gold
program used, take the operation the generated code used, and see which of the
two, swapped in, would have produced the right answer.

The attribution is reproducible and needs no model:

  FORMULA   the code's answer is reproduced by applying SOME operation to the
            GOLD operands -- it had the right numbers and combined them wrong
  NUMBERS   the code's answer is reproduced by applying the GOLD operation to
            OTHER cells of the table -- right method, wrong cells
            (split further: wrong row / wrong column / scale / sign)
  DATASET   the gold itself is not a number, or contradicts its own question
  NO_VALUE  the code returned nothing

Zero API calls.

Run:  python -m eval.finqa_operand_diag
"""
import argparse
import ast
import collections
import itertools
import os
import re

from eval import cache_io
from eval.finqa_confirm import build
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")
TOL = 0.01

_NUM = re.compile(r"-?\d+\.?\d*")
_REF = re.compile(r"#\d+")
_OPNAME = re.compile(r"(add|subtract|multiply|divide|exp|greater)\(")

OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: (a / b) if b else None,
}


def close(a, b, tol=TOL):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(abs(b), 1.0)


def gold_operands(prog):
    # `#0`, `#1` are references to earlier steps, not operands. Left in, the
    # regex reads them as the numbers 0 and 1 and invents operands the gold
    # program never used (defect 21).
    out = []
    for m in _NUM.findall(_REF.sub("", str(prog or ""))):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def gold_op_names(prog):
    return _OPNAME.findall(str(prog or ""))


def code_ops(code):
    """Which arithmetic the generated program actually performs."""
    found = set()
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            found.add({ast.Add: "add", ast.Sub: "subtract",
                       ast.Mult: "multiply", ast.Div: "divide"}.get(
                type(node.op), "other"))
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fn in ("sum", "mean", "max", "min"):
                found.add(fn)
    return found


def table_cells(rec):
    """Every numeric cell with its row and column index."""
    out = []
    for i, row in enumerate(rec.get("table") or []):
        for j, c in enumerate(row):
            s = str(c).replace(",", "").replace("$", "").replace("%", "").strip()
            neg = s.startswith("(") and s.endswith(")")
            s = s.strip("()")
            try:
                v = float(s)
            except ValueError:
                continue
            out.append((i, j, -v if neg else v))
    return out


def reproduced_by(target, nums, ops):
    """Is `target` any op applied to an ordered pair from `nums`?"""
    for a, b in itertools.permutations(nums, 2):
        for name in ops:
            f = OPS.get(name)
            if not f:
                continue
            try:
                q = f(a, b)
            except Exception:
                continue
            if q is not None and close(target, q):
                return name, a, b
    return None


def classify(r, rec, code):
    cand, gold = r["r2"], r["gold"]
    if cand is None:
        return "NO_VALUE", "код не повернув числа"
    if gold is None:
        return "DATASET", "еталон не числовий (так/ні)"

    gops = gold_operands(r.get("gold_program"))
    gnames = gold_op_names(r.get("gold_program")) or ["divide", "subtract"]
    cops = code_ops(code) or set(OPS)

    # scale and sign first: they are unambiguous
    if close(-cand, gold):
        return "NUM_SIGN", f"знак: {cand} проти {gold}"
    for f in (100.0, 1e3, 1e6, 1e9):
        if close(cand / f, gold) or close(cand * f, gold):
            return "NUM_SCALE", f"масштаб ×{f:g}"

    # right numbers, wrong formula: gold operands recombined
    if len(gops) >= 2:
        hit = reproduced_by(cand, gops[:4], set(OPS))
        if hit:
            return "FORMULA", f"gold-операнди через {hit[0]}({hit[1]:g}, {hit[2]:g})"

    # right method, wrong cells
    cells = table_cells(rec)
    vals = [c[2] for c in cells]
    if len(vals) <= 60:
        hit = reproduced_by(cand, vals, set(gnames))
        if hit:
            _, a, b = hit
            ra = next((c for c in cells if c[2] == a), None)
            rb = next((c for c in cells if c[2] == b), None)
            if ra and rb and {x for x in gops} and ra[0] == rb[0]:
                return "NUM_COL", f"ті самі рядки, інші стовпці: {a:g}, {b:g}"
            return "NUM_ROW", f"gold-операція над іншими клітинками: {a:g}, {b:g}"
    return "OTHER", f"наш={cand} еталон={gold}"


def ceilings(rows, by_id, recs):
    """Two counterfactuals, both computed without a model.

    `numbers`  the code keeps its own operation but is handed the gold operands
    `formula`  the code keeps its own numbers but the gold operation is applied
    """
    fixed_by_numbers = fixed_by_formula = eligible = 0
    for r in rows:
        if r["correct"] or r["gold"] is None:
            continue
        x = by_id[r["id"]]
        code = next((s.get("code") for s in reversed(x.get("steps") or [])
                     if s.get("code")), "")
        rec = recs.get(r["id"])
        if not rec:
            continue
        gops = gold_operands(r.get("gold_program"))
        gnames = gold_op_names(r.get("gold_program"))
        cops = code_ops(code)
        eligible += 1

        if len(gops) >= 2 and cops:
            hit = reproduced_by(r["gold"], gops[:4], cops)
            if hit:
                fixed_by_numbers += 1

        cells = [c[2] for c in table_cells(rec)]
        if gnames and len(cells) <= 60:
            # the numbers the code addressed are unknown; approximate with the
            # cells whose combination under the CODE's operation gives its answer
            src = reproduced_by(r["r2"], cells, cops) if (r["r2"] is not None and cops) else None
            if src:
                _, a, b = src
                hit = reproduced_by(r["gold"], [a, b], set(gnames))
                if hit:
                    fixed_by_formula += 1
    return fixed_by_numbers, fixed_by_formula, eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)
    by_id = {x["id"]: x for x in code_rows}
    recs = {x["id"]: x for x in fq.load_split("test")}
    for r in rows:
        r["gold_program"] = by_id[r["id"]].get("gold_program")

    wrong = [r for r in rows if not r["correct"]]
    agreed = [r for r in wrong if r["verdict"] == "ACCEPT"]

    print("=" * 78)
    print(f"ЧИСЛА ЧИ ФОРМУЛА — {args.tag}, openai/gpt-oss-120b")
    print("=" * 78)
    print(f"питань {len(rows)}, хибних у коді {len(wrong)}, "
          f"з них спільних із прямою {len(agreed)}")
    print()

    tagged = {}
    for r in wrong:
        x = by_id[r["id"]]
        code = next((s.get("code") for s in reversed(x.get("steps") or [])
                     if s.get("code")), "")
        tagged[r["id"]] = classify(r, recs.get(r["id"], {}), code)

    names = {"FORMULA": "а) правильні числа, хибна формула",
             "NUM_ROW": "б) хибне число: не той рядок",
             "NUM_COL": "б) хибне число: не той стовпець/рік",
             "NUM_SCALE": "б) хибне число: масштаб",
             "NUM_SIGN": "б) хибне число: знак",
             "DATASET": "в) дефект набору",
             "NO_VALUE": "г) код не повернув числа",
             "OTHER": "не розкладено"}
    order = ["FORMULA", "NUM_ROW", "NUM_COL", "NUM_SCALE", "NUM_SIGN",
             "DATASET", "NO_VALUE", "OTHER"]

    def table(subset, title):
        cnt = collections.Counter(tagged[r["id"]][0] for r in subset)
        tot = max(len(subset), 1)
        print(f"--- {title} (n={len(subset)}) ---")
        for k in order:
            v = cnt.get(k, 0)
            if not v:
                continue
            print(f"  {names[k]:38s} {v:3d}  {v / tot:5.1%}  "
                  f"{'#' * int(round(26 * v / tot))}")
        num = sum(cnt.get(k, 0) for k in
                  ("NUM_ROW", "NUM_COL", "NUM_SCALE", "NUM_SIGN"))
        print(f"  {'РАЗОМ числа (б)':38s} {num:3d}  {num / tot:5.1%}")
        print(f"  {'РАЗОМ формула (а)':38s} {cnt.get('FORMULA', 0):3d}  "
              f"{cnt.get('FORMULA', 0) / tot:5.1%}")
        print()

    table(wrong, "УСІ ХИБНІ ВІДПОВІДІ КОДУ")
    table(agreed, "СПІЛЬНІ ХИБНІ (код і пряма погодились)")

    fn, ff, el = ceilings(rows, by_id, recs)
    print("--- СТЕЛЯ ПІДСТАНОВКИ (без LLM, на хибних із числовим еталоном) ---")
    print(f"  придатних для перевірки          : {el}")
    print(f"  виправилось, якби ЧИСЛА були gold: {fn}  ({fn / max(el, 1):.0%})")
    print(f"  виправилось, якби ФОРМУЛА була gold: {ff}  ({ff / max(el, 1):.0%})")
    base = sum(1 for r in rows if r["correct"])
    print(f"  точність коду зараз              : {base}/{len(rows)} = {base / len(rows):.1%}")
    print(f"  якби числа були gold             : {base + fn}/{len(rows)} = "
          f"{(base + fn) / len(rows):.1%}")
    print(f"  якби формула була gold           : {base + ff}/{len(rows)} = "
          f"{(base + ff) / len(rows):.1%}")
    print()

    print(f"--- ПРИКЛАДИ ГРУПИ (б): хибне число ---")
    shown = 0
    for r in wrong:
        k, why = tagged[r["id"]]
        if not k.startswith("NUM_") or shown >= args.examples:
            continue
        shown += 1
        x = by_id[r["id"]]
        print()
        print(f"  [{names[k]}]  {r['id']}")
        print(f"    {str(x.get('question'))[:88]}")
        print(f"    еталон={r['gold']}  наш={r['r2']}  "
              f"gold_program={r.get('gold_program')}")
        print(f"    -> {why}")

    out = os.path.join(HERE, f"finqa_operand_diag_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,bucket,why,gold,ours,gold_program,agreed\n")
        for r in wrong:
            k, why = tagged[r["id"]]
            gp = str(r.get("gold_program")).replace(",", ";")
            f.write(f"{r['id']},{k},\"{why}\",{r['gold']},{r['r2']},"
                    f"{gp},{int(r['verdict'] == 'ACCEPT')}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
