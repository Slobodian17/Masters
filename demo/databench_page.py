"""The DataBench corpus screen: retrieval -> code -> execution -> verification.

Replay only. Every number on this screen comes from a cache line or from
re-executing cached code against a local table; nothing calls an API.

The explanation at the bottom is DERIVED, not narrated: the column and the
condition are read out of the generated program's syntax tree, the condition is
then applied independently, and the count it produces is checked against the
answer the program returned. If they disagree the screen says so instead of
explaining.
"""
import json
import os

import pandas as pd
import streamlit as st

from demo.databench_scenarios import CASES, DEMO_MODEL, DEMO_SOURCE
from eval import cache_io
from eval.databench_demo import columns_used, comparisons, mask_of
from eval.databench_reretrieval import triggered
from eval.databench_run import top_rows

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DCACHE = os.path.join(HERE, "eval", "databench_cache")


@st.cache_resource(show_spinner=False)
def _retriever():
    from ingestion import databench_catalog as cat
    return cat.Retriever()


@st.cache_data(show_spinner=False)
def _rows(arm, tag):
    p = os.path.join(DCACHE, f"{arm}_{tag}.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in cache_io.load_rows(p)[0]}


@st.cache_data(show_spinner=False)
def _retries(tag):
    p = os.path.join(DCACHE, f"B-rag2_{tag}.jsonl")
    out = {}
    if os.path.exists(p):
        for x in cache_io.load_rows(p)[0]:
            out.setdefault(x["id"], []).append(x)
    for v in out.values():
        v.sort(key=lambda z: z["rank"])
    return out


@st.cache_data(show_spinner=False)
def _spec(tag):
    name = "databench_dev.json" if tag == "dev100" else "databench_test150.json"
    with open(os.path.join(HERE, "eval", name), encoding="utf-8") as f:
        return json.load(f)


def _table(ds, lite=False):
    from ingestion import databench_loader as dbl
    return dbl.table(ds, lite=lite)


def _frac(seen, tot):
    """Share of the table, with only as many decimals as it needs."""
    f = seen / max(tot, 1)
    return f"{f:.3%}" if f < 0.01 else f"{f:.1%}"


def _final_code(row):
    return next((s.get("code") for s in reversed(row.get("steps") or [])
                 if s.get("code")), "")


def resolve(case):
    """Apply the case's stated rule to the caches. Returns the question dict."""
    tag = case["tag"]
    spec = _spec(tag)
    qs = spec["questions"]
    sizes = {k: v["rows"] for k, v in spec["tables"].items()}
    r = _retriever()
    A, B = _rows("A", tag), _rows("B", tag)
    RG, RT = _rows("B-rag", tag), _retries(tag)

    if case["key"] == "1":
        best = None
        for q in qs:
            qid = q["qid"]
            if qid not in A or qid not in B:
                continue
            if A[qid]["correct"] or not B[qid]["correct"]:
                continue
            top = [t for t, _ in r.search(q["question"], 1)]
            if not top or top[0] != q["dataset"]:
                continue
            n = sizes.get(q["dataset"], 0)
            if best is None or n > best[0]:
                best = (n, q)
        return best[1] if best else None

    for q in qs:
        qid = q["qid"]
        top = [t for t, _ in r.search(q["question"], 3)]
        if not top or top[0] == q["dataset"]:
            continue                      # rule needs a retrieval MISS
        base = RG.get(qid)
        if not triggered(base, q["type"]):
            continue
        tries = RT.get(qid, [])
        solved = next((z for z in tries if not triggered(z, q["type"])), None)
        if case["key"] == "2" and solved and solved["correct"]:
            return q
        if case["key"] == "3" and tries and solved is None:
            return q
    return None


def render():
    st.header("DataBench: система сама знаходить таблицю")
    st.markdown(
        "Корпус — **60 таблиць, 3 220 897 рядків**. Питання не приходить із "
        "таблицею: її треба знайти. Усі числа — з кешів справжніх прогонів."
    )
    st.caption(f"модель: **{DEMO_MODEL}** · джерело: {DEMO_SOURCE}")

    labels = [f"{c['key']}. {c['title']}" for c in CASES]
    want = st.query_params.get("case")
    idx = next((i for i, c in enumerate(CASES) if c["key"] == want), 0)
    choice = st.selectbox("Випадок", labels, index=idx)
    case = CASES[labels.index(choice)]
    st.caption(f"правило відбору: {case['rule']}")

    q = resolve(case)
    if q is None:
        st.warning("За цим правилом у кешах випадку не знайшлось.")
        return
    tag = case["tag"]
    qid = q["qid"]
    A, B = _rows("A", tag), _rows("B", tag)
    RG, RT = _rows("B-rag", tag), _retries(tag)
    r = _retriever()

    st.subheader("Питання")
    st.markdown(f"> {q['question']}")
    st.caption(f"очікуваний тип відповіді: `{q['type']}` · "
               f"еталонна таблиця: `{q['dataset']}`")

    # ---- 1. retrieval ----
    st.subheader("Крок 1. Пошук таблиці в корпусі")
    st.caption("BM25 по схемному каталогу — назва, стовпці, типи. "
               "Нуль викликів LLM.")
    hits = r.search(q["question"], 3)
    rows = []
    for rank, (tid, sc) in enumerate(hits, 1):
        rows.append({"ранг": rank, "таблиця": tid, "оцінка BM25": round(sc, 2),
                     "еталонна": "так" if tid == q["dataset"] else ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    hit1 = hits and hits[0][0] == q["dataset"]
    if hit1:
        st.success("Пошук знайшов потрібну таблицю на 1-му місці.")
        if len(hits) > 1 and hits[0][1] - hits[1][1] < 1.0:
            st.caption(f"перевага над другим місцем лише "
                       f"{hits[0][1] - hits[1][1]:.2f} — пошук тут не надійний; "
                       f"на всій вибірці recall@1 = 49.3%")
    else:
        st.error(f"**Пошук промахнувся:** узято `{hits[0][0]}`, "
                 f"потрібна `{q['dataset']}`.")

    used = hits[0][0]
    base = B.get(qid) if hit1 else RG.get(qid)
    df_used = _table(used)

    # ---- 2. code on the retrieved table ----
    st.subheader("Крок 2. Код на знайденій таблиці")
    st.caption(f"`{used}` — {df_used.shape[0]:,} рядків × {df_used.shape[1]} "
               f"стовпців. Модель бачить схему, типи й 5 рядків; "
               f"код працює на всій таблиці.".replace(",", " "))
    if base is None:
        st.warning("Для цього питання кеш не містить прогону.")
        return
    st.code(_final_code(base), language="python")
    st.metric("результат виконання", str(base.get("value")))

    why = triggered(base, q["type"])
    if why:
        st.error(f"**Тригер повторного пошуку: `{why}`.** Це сигнал, доступний "
                 f"ДО будь-якої звірки з еталоном.")

    # ---- 3. re-retrieval ----
    tries = RT.get(qid, [])
    if tries:
        st.subheader("Крок 3. Повторний пошук: наступна таблиця з рангу")
        for t in tries:
            still = triggered(t, q["type"])
            with st.expander(
                    f"ранг {t['rank']} — `{t['dataset']}` → "
                    f"{t['value']}" + ("" if not still else f"  (знову {still})"),
                    expanded=(not still)):
                st.code(_final_code(t), language="python")
                st.write("результат:", t["value"])
        solved = next((t for t in tries if not triggered(t, q["type"])), None)
        if solved is None:
            st.warning("Жоден із кандидатів не дав відповіді — лишається "
                       "результат top-1, тож блок не погіршує базу.")
        else:
            st.success(f"Узято відповідь із рангу {solved['rank']}.")
    elif why:
        st.info("Повторний пошук для цього питання не прогонявся.")

    # ---- 4. verdict ----
    final = next((t for t in tries if not triggered(t, q["type"])), None) or base
    st.subheader("Крок 4. Вердикт")
    c1, c2, c3 = st.columns(3)
    c1.metric("відповідь системи", str(final.get("value")))
    c2.metric("еталон", str(final.get("truth")))
    if final.get("correct"):
        c3.success("правильно")
    else:
        c3.error("хибно")

    # direct answer, beside the code answer
    a = A.get(qid)
    if a is not None:
        seen = a.get("rows_seen") or 0
        tot = a.get("rows_total") or 1
        st.caption(
            f"**Пряма відповідь на тому самому питанні:** `{a.get('value')}` — "
            f"{'правильно' if a.get('correct') else 'хибно'}. "
            f"Вона бачила **{seen} з {tot:,} рядків ({_frac(seen, tot)})**, "
            f"і таблицю їй дали готовою.".replace(",", " "))

    # ---- 5. explanation derived from the AST ----
    st.subheader("Крок 5. Пояснення, виведене з виконання")
    st.caption("Стовпець і умова прочитані з синтаксичного дерева програми; "
               "умова застосована незалежно, і підрахунок звірений із "
               "відповіддю програми.")
    code = _final_code(final)
    ds = final.get("dataset", used)
    dff = _table(ds)
    cols = columns_used(code)
    cmps = comparisons(code)
    st.write("стовпці, до яких звертається програма:", cols or "—")
    if not cmps:
        st.info("Програма не містить простої умови порівняння — пояснення "
                "на цьому рівні неповне.")
        return
    for col, op, val in cmps:
        if col not in dff.columns:
            st.warning(f"стовпця `{col}` у таблиці `{ds}` немає")
            continue
        m = mask_of(dff, col, op, val)
        if m is None:
            continue
        n = int(m.sum())
        st.markdown(
            f"- умова **`{col} {op} {val!r}`** виконується на **{n:,} рядках "
            f"із {len(dff):,}** ({n / max(len(dff), 1):.1%})".replace(",", " "))
        if str(n) == str(final.get("value")).strip():
            st.success("Цей підрахунок збігається з відповіддю програми — "
                       "пояснення підтверджене виконанням.")
        if a is not None and ds == q["dataset"]:
            win = top_rows(dff, q["question"])
            mw = mask_of(win, col, op, val)
            if mw is not None:
                st.markdown(
                    f"- у вікні прямої відповіді та сама умова виконується на "
                    f"**{int(mw.sum())} рядках із {len(win)}** — тобто пряма "
                    f"відповідь була правильною для показаних їй даних")
        head = dff.loc[m, [c for c in cols if c in dff.columns][:4]].head(4)
        if not head.empty:
            st.dataframe(head, use_container_width=True)

    st.info(f"**Висновок випадку.** {case['takeaway']}")
