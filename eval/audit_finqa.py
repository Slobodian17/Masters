"""Audit of FinQA before anything is measured on it.

The point is to find out what this dataset can and cannot support, so that no
later number rests on an assumption nobody checked. Five questions:

  1. How many tables per question under gold context. The multi-table error
     category only exists if the answer needs more than one.
  2. How much of the gold evidence lives in the text rather than the table.
     A table-only pipeline cannot answer those at all, so this is a hard
     ceiling on execution accuracy, not a source of fixable errors.
  3. Whether percentage answers are scaled consistently. FinQA is known to
     mix 14.1 and 0.141; a comparator that does not know this under-reports.
  4. How much of each table survives numeric parsing. An unparsed cell is not
     a blank — it is a value the pipeline will never see.
  5. What the gold programs look like, since their operation sequence is the
     surrogate for program accuracy.

Nothing here calls an LLM.

Run:  python -m eval.audit_finqa
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd

from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "test"


def _pct(n, d):
    return f"{n}/{d} = {n / d:.1%}" if d else f"{n}/0"


def audit(records: list) -> dict:
    out = {}

    # --- 1. tables per question -------------------------------------------
    n_tables = collections.Counter()
    table_shapes = []
    for r in records:
        t = r.get("table") or []
        n_tables[1 if t else 0] += 1
        if t:
            table_shapes.append((len(t) - 1, len(t[0]) - 1 if t[0] else 0))
    out["tables_per_question"] = dict(n_tables)
    out["table_shapes"] = table_shapes

    # --- 2. where the gold evidence lives ---------------------------------
    src = collections.Counter(fq.evidence_source(r) for r in records)
    out["evidence_source"] = dict(src)

    # How many gold rows come from text, for the questions that use any.
    text_rows = [len(r["qa"].get("ann_text_rows") or []) for r in records]
    table_rows = [len(r["qa"].get("ann_table_rows") or []) for r in records]
    out["mean_text_rows"] = float(np.mean(text_rows))
    out["mean_table_rows"] = float(np.mean(table_rows))

    # --- 3. percentage answers and their scale ----------------------------
    pct_q, scaled, unscaled, ambiguous = 0, 0, 0, 0
    scale_examples = []
    for r in records:
        qa = r["qa"]
        ans_text = str(qa.get("answer", "")).strip()
        exe = fq.gold_answer(r)
        if "%" not in ans_text or exe is None:
            continue
        pct_q += 1
        txt_val, _ = fq.parse_number(ans_text)
        if txt_val is None:
            ambiguous += 1
            continue
        # The answer text is rounded for display and exe_ans is not, so the
        # two are compared with the same 1% tolerance the task itself uses.
        # An exact test here reports the rounding, not the scale.
        def close(a, b):
            return abs(a - b) <= 0.01 * max(abs(b), 1e-9)

        if close(exe, txt_val):
            unscaled += 1          # exe_ans is already 14.1
        elif close(exe * 100, txt_val):
            scaled += 1            # exe_ans is 0.141, answer text says 14.1%
            if len(scale_examples) < 4:
                scale_examples.append((ans_text, exe))
        else:
            ambiguous += 1
    out["percent"] = {"questions": pct_q, "exe_is_fraction": scaled,
                      "exe_is_percent": unscaled, "neither": ambiguous,
                      "examples": scale_examples}

    # --- 4. numeric parsing coverage --------------------------------------
    cells = parsed = 0
    empty_frames = 0
    unparsed_samples = []
    per_table_rate = []
    for r in records:
        df, st = fq.to_frame(r.get("table") or [])
        if df.empty:
            empty_frames += 1
            continue
        cells += st["cells"]
        parsed += st["parsed"]
        if st["cells"]:
            per_table_rate.append(st["parsed"] / st["cells"])
        for u in st["unparsed_examples"]:
            if len(unparsed_samples) < 15:
                unparsed_samples.append(u)
    out["parsing"] = {"cells": cells, "parsed": parsed,
                      "empty_frames": empty_frames,
                      "median_table_rate": float(np.median(per_table_rate)) if per_table_rate else 0.0,
                      "tables_below_50pct": int(sum(1 for x in per_table_rate if x < 0.5)),
                      "unparsed_samples": unparsed_samples}

    # --- 5. gold answers and programs -------------------------------------
    numeric = sum(1 for r in records if fq.gold_answer(r) is not None)
    ops = collections.Counter()
    op_counts = collections.Counter()
    for r in records:
        seq = fq.program_ops(fq.gold_program(r))
        ops.update(seq)
        op_counts[len(seq)] += 1
    out["gold"] = {"numeric_answers": numeric, "ops": dict(ops.most_common()),
                   "program_lengths": dict(sorted(op_counts.items()))}

    return out


def main():
    print("=" * 78)
    print(f"АУДИТ FinQA ({SPLIT}) — ЩО ЦЕЙ ДАТАСЕТ ВЗАГАЛІ ДОЗВОЛЯЄ МІРЯТИ")
    print("=" * 78)
    path = fq.fetch(SPLIT)
    records = fq.load_split(SPLIT)
    print(f"Файл: {path}")
    print(f"Прикладів: {len(records)}")
    print()

    a = audit(records)
    n = len(records)

    print("--- 1. Скільки таблиць на питання (gold context) ---")
    for k, v in sorted(a["tables_per_question"].items()):
        print(f"  таблиць={k}: {_pct(v, n)}")
    shapes = a["table_shapes"]
    rows = [s[0] for s in shapes]
    cols = [s[1] for s in shapes]
    print(f"  рядків у таблиці: медіана {np.median(rows):.0f}, "
          f"min {min(rows)}, max {max(rows)}")
    print(f"  стовпців: медіана {np.median(cols):.0f}, min {min(cols)}, max {max(cols)}")
    print()
    print("  ВИСНОВОК: під gold context це ОДНОТАБЛИЧНИЙ бенчмарк. Категорія")
    print("  multi-table тут не виникає в принципі — її треба міряти в")
    print("  corpus-retrieval режимі, де пул таблиць спільний.")

    print()
    print("--- 2. Де лежить еталонне свідчення (стеля для табличного конвеєра) ---")
    for k in ("table", "text", "both", "none"):
        v = a["evidence_source"].get(k, 0)
        print(f"  {k:6s}: {_pct(v, n)}")
    only_table = a["evidence_source"].get("table", 0)
    needs_text = a["evidence_source"].get("text", 0) + a["evidence_source"].get("both", 0)
    print(f"  середньо рядків з таблиці: {a['mean_table_rows']:.2f}, "
          f"з тексту: {a['mean_text_rows']:.2f}")
    print()
    print(f"  СТЕЛЯ: наш конвеєр читає лише таблиці, тож без роботи з текстом")
    print(f"  доступні щонайбільше {_pct(only_table, n)} питань.")
    print(f"  Решта {_pct(needs_text, n)} потребує чисел з pre_text/post_text.")

    print()
    print("--- 3. Масштаб у відсоткових відповідях ---")
    p = a["percent"]
    print(f"  питань з '%' у тексті відповіді: {_pct(p['questions'], n)}")
    print(f"    exe_ans — частка (0.141 при '14.1%'): {p['exe_is_fraction']}")
    print(f"    exe_ans — відсоток (14.1 при '14.1%'): {p['exe_is_percent']}")
    print(f"    ні те, ні те: {p['neither']}")
    for txt, exe in p["examples"]:
        print(f"      приклад: answer='{txt}'  exe_ans={exe}")
    print()
    if p["exe_is_fraction"] and p["exe_is_percent"]:
        print("  УВАГА: обидві форми присутні одночасно. Компаратор, який цього")
        print("  не знає, занижує Execution Accuracy на відсоткових питаннях у")
        print("  100 разів помилки. Еталоном беремо exe_ans, а не текст.")
    else:
        print("  Масштаб узгоджений; окремої обробки не потрібно.")

    print()
    print("--- 4. Що виживає при розборі чисел ---")
    pr = a["parsing"]
    print(f"  клітинок усього: {pr['cells']}, розібрано: {_pct(pr['parsed'], pr['cells'])}")
    print(f"  медіана по таблиці: {pr['median_table_rate']:.1%}")
    print(f"  таблиць з розбором <50%: {pr['tables_below_50pct']}")
    print(f"  порожніх фреймів: {pr['empty_frames']}")
    print(f"  приклади нерозібраного: {pr['unparsed_samples'][:10]}")

    print()
    print("--- 5. Еталонні відповіді й програми ---")
    g = a["gold"]
    print(f"  числових exe_ans: {_pct(g['numeric_answers'], n)}")
    print(f"  довжина програми (операцій): {g['program_lengths']}")
    print(f"  операції: {g['ops']}")
    print()
    print("  Ці послідовності — основа сурогата Program Accuracy")
    print("  (operation-type match). Це НЕ Program Accuracy FinQA.")

    out = os.path.join(HERE, f"finqa_audit_{SPLIT}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(a, f, ensure_ascii=False, indent=1, default=str)
    print()
    print("Збережено:", out)
    return a


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
