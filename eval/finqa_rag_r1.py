"""Probe: does the direct answer lose accuracy when the context is RETRIEVED
(top-k tables of the company's reports) instead of the gold table?

Exploratory, dev only (finqa_dev100.ids). The gold-table direct answer for the
same questions and model is already cached; only the RAG arm is new.

Both arms get the same system prompt, instructions and answer marker; the only
difference is the context. Each retrieved table is labelled with the report it
came from (ticker, year) and its caption -- the metadata a real document store
carries, given identically to any route that will use this retrieval.

Costs one call per question.

Run:  VKSK_LLM_PROVIDER=gemini VKSK_ANSWER_MARKER=marker \
      python -m eval.finqa_rag_r1 --k 3
"""
import argparse
import json
import os
import time

from eval import cache_io
from eval import finqa_baseline as fb
from eval.finqa_rag_conflict import company_corpus, retrieve
from eval.finqa_run import FINQA_INSTRUCTIONS, auto_pace, is_correct
from generation import llm_client
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")


def build_prompt(question, rec, hits):
    co, _ = fq.filing_of(rec)
    parts = []
    for i, (_h, raw, cap, yr) in enumerate(hits, start=1):
        df, _ = fq.to_frame(raw)
        parts.append(f"Table {i} -- source: {co} annual report {yr}; "
                     f'caption: "{cap}"\n{fb.serialise(df)}')
    ctx = "\n\n".join(parts)
    return (f"Retrieved tables:\n{ctx}\n\n{FINQA_INSTRUCTIONS}\n\n"
            f"Question: {question}\nAnswer with one number only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(HERE, "finqa_dev100.ids"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pace", type=float, default=None)
    args = ap.parse_args()
    if llm_client.active_provider() == "mock":
        print("MOCK: no FinQA answers; set a real provider.")
        return
    pace = args.pace if args.pace is not None else auto_pace() / 2

    allrec = []
    for sp in ("train", "dev", "test"):
        allrec += fq.load_split(sp)
    byid = {}
    for r in allrec:
        byid.setdefault(r["id"], r)
    corpus = company_corpus(allrec)
    with open(args.ids, encoding="utf-8") as f:
        ids = [l.strip() for l in f if l.strip()]

    model = llm_client.active_model().split("/")[-1]
    out_path = os.path.join(CACHE_DIR, f"rag_r1_k{args.k}_{model}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    n_new = 0
    n_fake = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for qid in ids:
            if qid in done:
                continue
            if args.limit and n_new >= args.limit:
                break
            rec = byid[qid]
            gold = fq.gold_answer(rec)
            hits = retrieve(rec, corpus, args.k)
            own = fq.content_hash(rec["table"])
            t0 = time.perf_counter()
            try:
                raw = llm_client.call_llm(
                    build_prompt(fq.question(rec), rec, hits),
                    system=fb.system_prompt(), tag="rag_r1",
                    max_tokens=fb.MAX_TOKENS)
                err = None
                err_class = None
            except Exception as exc:
                raw, err = "", f"{type(exc).__name__}: {exc}"
                err_class = type(exc).__name__
            value = fb.parse_answer(raw)
            row = {"id": qid, "gold": gold, "answer": value,
                   "correct": bool(is_correct(value, gold)) if value is not None
                   and gold is not None else False,
                   "gold_in_context": own in [h[0] for h in hits],
                   "gold_rank": next((i + 1 for i, h in enumerate(hits)
                                      if h[0] == own), None),
                   "raw": raw[-600:], "elapsed_s": round(time.perf_counter() - t0, 2),
                   "error": err, "error_class": err_class}

            # A 429 or a timeout means the model was never reached, so there is
            # no result to record. Writing one anyway marks the question done
            # and a resume skips it forever. Same defect, same single place to
            # fix it as in finqa_run and finqa_baseline.
            try:
                cache_io.write_row(f, row, on_unknown=lambda m: print(
                    f"  УВАГА: нерозпізнане походження помилки — {m}"))
            except cache_io.FakeRow as exc:
                n_fake += 1
                print(f"    {qid:32s} ВИКЛИК НЕ ВІДБУВСЯ, рядок НЕ записано: {exc}")
                if "tpd" in str(exc).lower() or "tokens per day" in str(exc).lower():
                    print("  добовий ліміт токенів вичерпано — зупиняюсь; "
                          "решта лишається непоміченою для відновлення")
                    break
                time.sleep(pace)
                continue
            n_new += 1
            print(f"{n_new:3d} {qid:32s} ok={row['correct']!s:5s} "
                  f"gold_in_ctx={row['gold_in_context']}")
            time.sleep(pace)

    if n_fake:
        print(f"\nвикликів, що не відбулись: {n_fake} — ці питання лишились "
              f"непоміченими, повторний запуск їх добере")


if __name__ == "__main__":
    main()
