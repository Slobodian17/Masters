"""Streamlit UI.

Shows the retrieval decision, the answer with its confidence, the full
iteration trace and the provenance record, so the mechanics can be inspected
rather than only described.
"""
import streamlit as st

from core.pipeline import run
from core import provenance
from load_data import load_tables
from retrieval.retrieve import explain, select_tables

st.set_page_config(page_title="ВКСК — демо", layout="wide")
st.title("ВКСК: верифікаційно-керована ступінчаста корекція коду")
st.caption("Демо-прототип на синтетичних фінансових даних (PDF генерується автоматично)")

DOCS = {
    "Три таблиці, одна розірвана між сторінками": "sample_report_multipage.pdf",
    "Дві таблиці на одній сторінці": "sample_report.pdf",
}

with st.sidebar:
    st.header("Документ")
    doc_label = st.selectbox("PDF-джерело", list(DOCS.keys()))
    pdf_path = DOCS[doc_label]

    if st.session_state.get("_doc") != pdf_path:
        with st.spinner("Парсинг PDF (ingest_pdf + М1 stitching)..."):
            st.session_state.tables = load_tables(pdf_path)
            st.session_state._doc = pdf_path

    tables = st.session_state.tables

    st.header("Таблиці")
    for name, df in tables.items():
        st.subheader(f"{name} ({df.shape[0]}×{df.shape[1]})")
        st.dataframe(df, use_container_width=True, height=160)

    st.header("Параметри")
    use_retrieval = st.checkbox("Використовувати retrieval", value=True,
                                help="Вимкнення дає генератору всі таблиці документа")
    threshold = st.slider("Поріг confidence (θ)", 0.0, 1.0, 0.7, 0.05)
    max_iter = st.slider("Ліміт ітерацій (N)", 1, 8, 5)

query = st.text_input("Запит", value="Скільки виплачено дивідендів акціонерам за Q2 2023?")

if st.button("Виконати", type="primary") and query:
    names = None
    if use_retrieval:
        with st.spinner("Retrieval: структурний + часовий + векторний -> RRF..."):
            detail = explain(query, tables)
            names = detail["selected"]

        st.subheader("Рішення retrieval")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Обрана таблиця", ", ".join(names) if names else "—")
            st.caption(f"Векторний бекенд: {detail['vector_backend']}")
            st.write("**RRF-скор:**")
            st.dataframe(
                [{"таблиця": n, "rrf": round(s, 4)} for n, s in detail["fused"]],
                use_container_width=True, hide_index=True,
            )
        with c2:
            st.write("**Внесок кожного індексу:**")
            rows = []
            for source, per_table in detail["per_index"].items():
                for tname, d in per_table.items():
                    rows.append({"індекс": source, "таблиця": tname,
                                 "ранг": d["rank"], "скор": d["score"]})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.spinner("Виконується цикл корекції..."):
        result = run(query, tables, retrieved_names=names, max_iter=max_iter,
                     threshold=threshold, verbose=False, source_file=pdf_path)

    st.subheader("Результат")
    col1, col2 = st.columns([1, 1])
    with col1:
        if result["status"] == "ACCEPT":
            st.success(f"Відповідь: **{result['answer']}**")
            st.metric("Confidence (φ)", f"{result['confidence']:.2f}")
        else:
            st.warning("ABSTAIN — система не досягла достатньої довіри й чесно відмовилась вгадувати.")
        st.caption(f"Ітерацій витрачено: {len(result['trace'])}")

    with col2:
        st.write("**Трейс виконання**")
        for step in result["trace"]:
            level = step.get("level", "?")
            if "exception_type" in step:
                st.write(f"`iter={step['iter']}` **{step['exception_type']}** -> `{level}`")
                st.code(step["error"], language="text")
            else:
                st.write(f"`iter={step['iter']}` результат=`{step['result']}`, "
                         f"φ=`{step['confidence']:.2f}` -> `{level}`")
                st.json(step["signals"])

    st.subheader("Provenance-запис (аудиторський слід)")
    st.code(provenance.to_json(result["provenance"]), language="json")

st.divider()
st.caption(
    "Без GROQ_API_KEY працює MOCK-режим: генератор коду відповідає лише на запити "
    "з тестових наборів. Для довільних запитів додай ключ у .env і перезапусти."
)
