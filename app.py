"""Демо методу: перевірка відповіді через згоду двох незалежних прочитань.

REPLAY ONLY. Кожен екран будується з кешів справжніх прогонів — жодного
виклику до API, жодного очікування. Для захисту це важливіше за «живий»
режим: показане число можна звірити з артефактом, а не з тим, що модель
відповіла цієї хвилини.

Підписи інтерфейсу українською; документи, питання й код — англійською, як
у самих датасетах.

Запуск:  streamlit run app.py
"""
import json
import os
import zipfile

import pandas as pd
import streamlit as st

from demo.scenarios import DEMO_MODEL, DEMO_SOURCE, SCENARIOS, TATDQA_NOTE, TATDQA_PAGE

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "eval", "finqa_cache")
FIGS = os.path.join(HERE, "eval", "figures")

st.set_page_config(page_title="ВКСК — демо методу", layout="wide")


# ---------- дані ----------

@st.cache_data(show_spinner=False)
def load_cache(tag):
    path = os.path.join(CACHE, f"{tag}.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


@st.cache_data(show_spinner=False)
def load_finqa_tables(qid):
    from ingestion import finqa_loader as fq
    for rec in fq.load_split("test"):
        if rec.get("id") == qid:
            return fq.to_tables(rec), fq.question(rec)
    return {}, ""


def fmt(v, nd=6):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{nd}g}"


def agree(r1, r2, tol=0.01):
    if r1 is None or r2 is None:
        return None
    return abs(r1 - r2) <= tol * max(abs(r2), 1.0)


def provenance(extra=""):
    st.caption(f"модель: **{DEMO_MODEL}** · джерело: {DEMO_SOURCE}{extra}")


# ---------- сторінка 1 ----------

def page_pdf():
    st.header("Від PDF до таблиці")
    st.markdown(
        "Найслабша ланка ланцюга, і зміряна окремо: витягування з PDF коштує "
        "методу **30 пунктів точності** (5.33). Тут парсер працює **живцем** — "
        "сторінка читається зараз, не з кеша."
    )
    st.caption(TATDQA_NOTE)

    pdf = os.path.join(HERE, "data", "tatdqa", "dev", f"{TATDQA_PAGE}.pdf")
    if not os.path.exists(pdf):
        st.error(f"Немає файлу {pdf}. Розпакуйте сторінку з tatdqa_docs_dev.zip.")
        return

    os.environ.setdefault("VKSK_PDF_TEXT_TABLES", "1")
    os.environ.setdefault("VKSK_PDF_TABLE_FILTER", "1")
    from ingestion.ingest_pdf import extract_tables

    with st.spinner("Парсинг сторінки…"):
        frags = extract_tables(pdf)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(f"Витягнуто парсером: {len(frags)} фрагментів")
        st.caption(
            "Текстова стратегія плюс сегментація: рядок із двома й більше "
            "числами починає таблицю, один прозовий рядок усередині дозволений."
        )
        for i, df in enumerate(frags):
            with st.expander(f"фрагмент {i} — {df.shape[0]}×{df.shape[1]}",
                             expanded=(i == 0)):
                st.dataframe(df, use_container_width=True, height=180)
    with c2:
        st.subheader("Розмітка TAT-DQA")
        st.caption("Еталонна таблиця тієї самої сторінки — для порівняння.")
        try:
            import random
            from eval import tatdqa_loader as T
            from eval.tatdqa_run import gold_table
            zf = zipfile.ZipFile(T.DOCS_ZIP)
            sel = T.select(zf=zf)
            random.Random(42).shuffle(sel)
            q = next(x for x in sel[:40] if x["doc_uid"] == TATDQA_PAGE)
            gt = gold_table(q, zf)
            if gt:
                st.dataframe(list(gt.values())[0], use_container_width=True,
                             height=260)
            st.info(f"Питання сторінки: *{q['question']}*")
        except Exception as exc:
            st.warning(f"Розмітку не вдалось показати: {type(exc).__name__}")

    st.divider()
    st.markdown(
        "**Що тут видно.** Парсер повертає кілька фрагментів; потрібну таблицю "
        "має знайти система, і на 40 сторінках це вдається у **40%** випадків "
        "проти 92% у розмітці. Саме ця різниця й дає ті 30 пунктів."
    )


# ---------- сторінка 2 ----------

def page_how():
    st.header("Як система відповідає")
    code = load_cache("confirm300")
    direct = load_cache("baseline_table_confirm300")

    labels = [f"{s['key']}. {s['title']}" for s in SCENARIOS]
    # ?scenario=<key> preselects one. Needed for reproducible screenshots and
    # handy for linking straight to a case during the defence.
    want = st.query_params.get("scenario")
    idx = next((i for i, x in enumerate(SCENARIOS) if x["key"] == want), 0)
    choice = st.selectbox("Сценарій", labels, index=idx)
    s = SCENARIOS[labels.index(choice)]
    qid = s["id"]
    c, d = code[qid], direct.get(qid, {})
    provenance(f" · питання **{qid}**")
    st.caption(f"правило відбору сценарію: {s['rule']}")

    tables, question = load_finqa_tables(qid)
    st.subheader("Питання")
    st.markdown(f"> {c.get('question') or question}")

    st.subheader("Таблиця")
    for name, df in tables.items():
        st.dataframe(df, use_container_width=True)

    # Live gate demo: the gate is deterministic, so a question about data the
    # table does not carry can be judged here and now rather than replayed.
    if s.get("live_gate"):
        from core import scope_gate
        st.subheader("Питання, яке ставимо натомість")
        st.markdown(f"> *{s['live_gate']}*")
        st.caption("Гейт детермінований — рахується живцем, не з кеша.")
        g = scope_gate.check(s["live_gate"], tables)
        passed = g["status"] == scope_gate.IN_SCOPE
        if passed:
            st.success(f"Гейт пропустив (покриття {g['coverage']:.0%})")
        else:
            st.error(f"**Гейт відмовив.** Покриття схеми — {g['coverage']:.0%}: "
                     f"жоден токен не знайшов відповідника.")
        st.write("токени питання без відповідника в схемі:", g["unmatched"])
        st.info(f"**Висновок сценарію.** {s['takeaway']}")
        return

    r1 = d.get("answer")
    r2 = c.get("candidate") if c.get("candidate") is not None else c.get("answer")
    gate = c.get("abstain_reason") == "OUT_OF_SCOPE"

    a, b = st.columns(2)
    with a:
        st.subheader("R1 — пряма відповідь")
        st.caption("Модель бачить таблицю і називає число.")
        st.metric("відповідь", fmt(r1))
    with b:
        st.subheader("R2 — згенерований код")
        if gate:
            st.warning("Гейт визнав питання поза обсягом таблиці — "
                       "**код не запускався**.")
        else:
            steps = c.get("steps") or []
            for i, stp in enumerate(steps):
                ok = stp.get("error") is None and stp.get("exception_type") is None
                head = (f"ітерація {i + 1}"
                        + ("" if ok else f" — {stp.get('exception_type') or 'відхилено'}"))
                with st.expander(head, expanded=(i == len(steps) - 1)):
                    if stp.get("code"):
                        st.code(stp["code"], language="python")
                    st.write("результат виконання:", fmt(stp.get("result")))
                    if stp.get("error"):
                        st.error(str(stp["error"])[:200])
            st.metric("підсумкове значення", fmt(r2))

    st.divider()
    st.subheader("Вердикт згоди")
    same = agree(r1, r2)
    if gate or r2 is None:
        st.warning("**Відмова.** Код не дав числа — порівнювати нема з чим.")
        decision = None
    elif same:
        st.success(f"**Збіг** у межах 1%: {fmt(r1)} ≈ {fmt(r2)} → відповідь прийнято.")
        decision = r2
    else:
        st.error(f"**Розбіжність**: {fmt(r1)} ≠ {fmt(r2)} → "
                 "система не впевнена, потрібна перевірка людиною.")
        decision = None

    gold = c.get("gold")
    st.subheader("Звірка з еталоном датасету")
    e1, e2, e3 = st.columns(3)
    e1.metric("еталон FinQA", fmt(gold))
    e2.metric("рішення системи", fmt(decision) if decision is not None else "відмова")
    if gold is None:
        e3.warning("еталон не числовий")
    elif decision is None:
        e3.info("система утрималась")
    elif abs(float(decision) - float(gold)) <= 0.01 * max(abs(float(gold)), 1.0):
        e3.success("рішення правильне")
    else:
        e3.error("рішення ХИБНЕ")

    st.info(f"**Висновок сценарію.** {s['takeaway']}")

    with st.expander("Деталі каскаду φ(v) — старий механізм, замінений згодою"):
        st.write("сигнали:", c.get("signals"))
        st.write("довіра φ(v):", c.get("confidence"),
                 " · статус каскаду:", c.get("status"),
                 " · причина:", c.get("abstain_reason"))
        st.caption("Зміряно на 300 питаннях: каскад ловить 24.2% хибних "
                   "відповідей при precision 68.6%, згода маршрутів — "
                   "46.5% при 93.9%.")


# ---------- сторінка 3 ----------

FIG_CAPTIONS = [
    ("01_cascade_vs_agreement.png", "Каскад φ(v) проти згоди маршрутів",
     "передреєстроване (5.26), gpt-oss-120b, n=300",
     "Згода відхиляє хибні відповіді вдвічі частіше й майже без хибних "
     "спрацювань."),
    ("03_accuracy_coverage.png", "Точність маршрутів і прийнятих",
     "передреєстроване (5.26), gpt-oss-120b, n=300",
     "Точність прийнятих 76.8% при покритті 76% — покриття подається "
     "поруч завжди."),
    ("04_route_matrix.png", "Матриця R1 × R2",
     "пост-хок, gpt-oss-120b, n=300",
     "Код правий там, де пряма відповідь хибна, лише на 4 питаннях із 300."),
    ("11_tatdqa.png", "TAT-DQA: витягування і наскрізна точність",
     "розвідувальне (5.33), flash-lite, n=40",
     "Витягування з PDF коштує 30 пунктів: 45% проти 15%."),
]


def page_results():
    st.header("Результати")
    st.markdown(
        "Графіки будуються **лише з артефактів** — жодне число не вписане "
        "в код. Відтворення: `python -m eval.plots`."
    )
    for i in range(0, len(FIG_CAPTIONS), 2):
        cols = st.columns(2)
        for col, (fn, title, kind, note) in zip(cols, FIG_CAPTIONS[i:i + 2]):
            with col:
                st.subheader(title)
                st.caption(kind)
                p = os.path.join(FIGS, fn)
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                else:
                    st.warning(f"немає {fn} — запустіть `python -m eval.plots`")
                st.markdown(note)


# ---------- каркас ----------

st.sidebar.title("ВКСК")
st.sidebar.caption("Пошук таблиці · генерація коду · верифікація")
PAGES = ["DataBench: пошук у корпусі", "Від PDF до таблиці",
         "FinQA: згода маршрутів", "Результати"]
_want_page = st.query_params.get("page")
page = st.sidebar.radio(
    "Розділ", PAGES,
    index=PAGES.index(_want_page) if _want_page in PAGES else 0)
st.sidebar.divider()
st.sidebar.caption(
    "**Режим відтворення.** Усі числа — з кешів справжніх прогонів, "
    "нуль викликів API.\n\n"
    "Моделі: `openai/gpt-oss-120b` — FinQA, "
    "`gemini-flash-lite-latest` — DataBench і TAT-DQA."
)

if page == "DataBench: пошук у корпусі":
    from demo.databench_page import render as page_databench
    page_databench()
elif page == "Від PDF до таблиці":
    page_pdf()
elif page == "FinQA: згода маршрутів":
    page_how()
else:
    page_results()
