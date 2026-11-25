"""Direct-answer baseline on FinQA: the model without the method.

Needed to say what the method is worth. Every number so far compares the
method against itself under different settings; none says whether generating
code and verifying it beats simply asking.

Two baselines, because "baseline" is ambiguous and the ambiguity matters:

  schema  only the column names and row labels, no values. This is what
          `eval.eval` calls configuration A on the synthetic set, where it
          scores 0% -- the model cannot invent the numbers. Kept for
          continuity, not for comparison.
  table   the whole table serialised, and a request for the number. This is
          the honest comparison: the same information the method receives,
          without code generation, execution or verification.

Only `table` is comparable with the method. Reporting `schema` as the
baseline would flatter the method by handing the baseline less information.

Costs one call per question per baseline.

Run:  python -m eval.finqa_baseline --tag ab_yes_v2
"""
import argparse
import json
import os
import re
import time

import pandas as pd

from eval import cache_io
from eval.finqa_replay import load_cache
from eval.finqa_run import FINQA_INSTRUCTIONS, auto_pace, is_correct, scale_error
from generation import llm_client
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

# The baseline must do the arithmetic in context, with no sandbox to run it
# in, so it needs MORE room than code generation, not less. The first run set
# this to 600 and 7 of 25 replies came back empty -- the gpt-oss models spend
# the budget on reasoning tokens and return nothing rather than truncating.
# That understated the baseline by 28 points and flattered the method it is
# meant to measure against.
# Must equal the method's code-generation budget within one model, or the
# comparison measures the budget rather than the method. Both read the same
# environment override.
MAX_TOKENS = int(os.environ.get("VKSK_CODEGEN_MAX_TOKENS", "2400"))

# Some models ignore "answer with a single number" and reason in prose first.
# Measured on Gemma 4 26B: 6 of 10 replies were cut at 2400 tokens before
# reaching an answer, and in the rest the FIRST number in the reply was a year
# from a column header -- so the route scored 0/10 for a parsing reason, not a
# model one. That is defect 15 again: the harness understating a route.
#
# The marker makes the answer findable wherever the prose ends. Off by default,
# because switching it on changes what every earlier run was asked, and those
# runs have to stay reproducible as measured.
ANSWER_MARKER = "ANSWER:"
MARKER_ENV = "VKSK_ANSWER_MARKER"


def marker_mode() -> str:
    """'' (off), 'marker', or 'strict'.

    'strict' forbids the prose outright. Measured on Gemma 4 26B: asking for
    the marker at the END of the reply does not help, because the model spends
    whatever budget it is given on reasoning and never reaches the marker --
    2188 tokens at a 2400 limit, 4186 at a 4000 limit, 9 of 10 replies cut.
    The budget was never the constraint; the instruction was.
    """
    v = os.environ.get(MARKER_ENV, "0").strip().lower()
    if v in ("", "0", "false", "no"):
        return ""
    return "strict" if v == "strict" else "marker"


def use_marker() -> bool:
    return marker_mode() != ""


SYSTEM_PLAIN = (
    "You are a financial analyst. Answer with a single number and nothing "
    "else: no units, no explanation, no currency symbol."
)

SYSTEM_MARKED = (
    "You are a financial analyst. Work out the answer, then end your reply "
    "with a final line of exactly this form:\n"
    "ANSWER: <number>\n"
    "The number carries no units, no percent sign and no currency symbol."
)


SYSTEM_STRICT = (
    "You are a financial analyst. Reply with exactly one line: "
    "ANSWER: <number>. Do not explain, do not show steps."
)


def system_prompt() -> str:
    return {"": SYSTEM_PLAIN, "marker": SYSTEM_MARKED,
            "strict": SYSTEM_STRICT}[marker_mode()]


SYSTEM = SYSTEM_PLAIN


def serialise(df: pd.DataFrame) -> str:
    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join("" if pd.isna(v) else str(v) for v in row.tolist()))
    return "\n".join(lines)


def schema_only(df: pd.DataFrame) -> str:
    rows = [str(v) for v in df.iloc[:, 0].tolist()] if df.shape[1] else []
    return f"columns: {list(df.columns)}\nrow labels: {rows}"


def build_prompt(question: str, df: pd.DataFrame, mode: str,
                 caption: str = None) -> str:
    body = serialise(df) if mode == "table" else schema_only(df)
    head = "Table:" if mode == "table" else "Table structure (values withheld):"
    cap = ""
    if caption:
        # The same wording the code route gets, for the same reason: the
        # caption names the unit the table itself often omits. Given to BOTH
        # routes or the comparison would credit the method with information
        # rather than with method -- route independence rests on different
        # ways of reasoning, not on different inputs.
        cap = (f'\nTable caption (describes the table; do NOT take any '
               f'number from it): "{caption}"\n')
    return (f"{head}\n{body}\n{cap}\n{FINQA_INSTRUCTIONS}\n\n"
            f"Question: {question}\nAnswer with one number only.")


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


_MARKED = re.compile(r"ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


def parse_answer(text: str):
    """The marked answer if the reply carries one, else the first number.

    The fallback is unchanged, so a reply without the marker parses exactly as
    it did before and no earlier result moves.
    """
    s = str(text).replace("$", "")
    marks = _MARKED.findall(str(text))
    if marks:
        try:
            return float(marks[-1].replace(",", ""))
        except ValueError:
            pass
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ab_yes_v2",
                    help="cache whose questions to reuse")
    ap.add_argument("--mode", default="table", choices=["table", "schema"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--pace", type=float, default=None)
    # Re-asking a named subset into a file of its own. Needed to regenerate
    # the rows produced before the prompt template was hashed: they have to be
    # compared against the originals side by side, not overwrite them.
    ap.add_argument("--only-ids", default="",
                    help="file with one question id per line; ask only those")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output cache, to keep a rerun apart")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW questions; 0 means all. "
                         "A trial start before committing the daily budget.")
    args = ap.parse_args()
    pace = args.pace if args.pace is not None else auto_pace() / 2

    provider = llm_client.active_provider()
    if provider == "mock":
        print("MOCK не містить питань FinQA — потрібен GROQ_API_KEY.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault(llm_client.LEDGER_ENV,
                          os.path.join(CACHE_DIR, "token_ledger.jsonl"))

    rows = load_cache(args.tag)
    records = {r["id"]: r for r in fq.load_split(args.split)}
    # A run with the caption is a different experiment and gets its own file,
    # exactly as finqa_run does. Overwriting the caption-free cache would
    # destroy the paired comparison the run exists to make.
    suffix = args.out_suffix + ("_cap" if fq.use_caption() else "")
    out_path = os.path.join(
        CACHE_DIR, f"baseline_{args.mode}_{args.tag}{suffix}.jsonl")

    if args.only_ids:
        with open(args.only_ids, encoding="utf-8") as f:
            wanted = {l.strip() for l in f if l.strip()}
        missing = wanted - {r["id"] for r in rows}
        if missing:
            print(f"УВАГА: {len(missing)} id немає в кеші {args.tag}, пропускаю")
        rows = [r for r in rows if r["id"] in wanted]
        print(f"обмежено списком: {len(rows)} питань з {len(wanted)} запитаних")

    print("=" * 78)
    print(f"БАЗОВА ЛІНІЯ «{args.mode}» на питаннях кешу {args.tag}")
    print("=" * 78)
    print(f"модель: {llm_client.active_model()}   питань: {len(rows)}")
    print("Пряме запитання, без генерації коду, без sandbox, без верифікації.")
    print()

    # Append-only and resumable, for the same reason finqa_run is: a session
    # that ends mid-run must not cost the budget already spent.
    done, results = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done.add(row["id"])
                    results.append(row)
    if done:
        print(f"вже пораховано: {len(done)}, продовжую")

    n_fake = n_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            if r["id"] in done:
                continue
            rec = records.get(r["id"])
            df = fq.to_tables(rec).get("table") if rec else None
            if df is None or df.empty:
                continue
            llm_client.reset_stats()
            t0 = time.perf_counter()
            try:
                raw = llm_client.call_llm(
                    build_prompt(r["question"], df, args.mode,
                                 caption=fq.caption(rec) if fq.use_caption() else None),
                    system=system_prompt(), tag="baseline",
                    max_tokens=MAX_TOKENS)
                err = err_class = None
            except Exception as exc:
                raw = ""
                err, err_class = f"{type(exc).__name__}: {exc}", type(exc).__name__
            truncated = llm_client.was_truncated()
            value = parse_answer(raw)
            ok = is_correct(value, r["gold"]) if value is not None else False
            st = llm_client.get_stats()
            row = {"id": r["id"], "question": r["question"], "gold": r["gold"],
                   "raw": raw[:200], "answer": value, "correct": bool(ok),
                   "scale_error": scale_error(value, r["gold"]),
                   "elapsed_s": round(time.perf_counter() - t0, 2),
                   "empty_reply": not str(raw).strip(),
                   "truncated": truncated,
                   # Actual usage against the reservation, so the first 429
                   # can say which of the two the daily budget charges.
                   "tokens": st.get("total_tokens", 0),
                   "prompt_tokens": st.get("prompt_tokens", 0),
                   "completion_tokens": st.get("completion_tokens", 0),
                   "reserved_tokens": MAX_TOKENS * st.get("calls", 0),
                   "error": err, "error_class": err_class}

            # The call may never have reached the model, and then there is no
            # result to record. Writing one marks the question done and a
            # resume skips it forever -- which is what the daily-token
            # fail-fast caused: 105 questions cached as empty answers that
            # were never actually asked. cache_io is the single place that
            # rule lives now; finqa_run had the same defect independently.
            try:
                cache_io.write_row(f, row, on_unknown=lambda m: print(
                    f"  УВАГА: нерозпізнане походження помилки — {m}"))
            except cache_io.FakeRow as exc:
                n_fake += 1
                print(f"[{i}/{len(rows)}] ВИКЛИК НЕ ВІДБУВСЯ, рядок НЕ записано: {exc}")
                if "tpd" in str(exc).lower() or "tokens per day" in str(exc).lower():
                    print("  добовий ліміт токенів вичерпано — зупиняюсь; "
                          "решта лишається непоміченою для відновлення")
                    break
                continue
            results.append(row)
            n_new += 1
            # No correctness in the line: this run is blind, and a per-question
            # OK/WRONG stream is the primary quantity arriving one token at a
            # time. What is shown is that the call happened and what it cost.
            print(f"[{i}/{len(rows)}] записано  tok={row['tokens']:4d} "
                  f"{row['elapsed_s']}s"
                  + ("  ПОРОЖНЯ ВІДПОВІДЬ" if row["empty_reply"] else "")
                  + ("  ОБІРВАНО" if row["truncated"] else ""))
            if args.limit and n_new >= args.limit:
                print(f"  пробний ліміт {args.limit} досягнуто — зупиняюсь; "
                      f"решта лишається непоміченою для відновлення")
                break
            if i < len(rows):
                time.sleep(pace)

    # NOTHING here may be an accuracy. This script runs while a
    # pre-registered experiment is still blind, and the baseline's own
    # Execution Accuracy is a primary quantity of 5.26 in its own right --
    # printing it leaks the experiment even without a comparison, which is
    # how it leaked once already. What is printed below is health of the
    # apparatus: whether the calls happened and what they cost.
    df_out = pd.DataFrame(results)
    n = len(df_out)
    print()
    print(f"--- Базова лінія «{args.mode}»: стан прогону ---")
    print(f"  рядків у кеші: {n}   викликів не відбулось: {n_fake}")
    if n:
        empty = int(df_out.get("empty_reply", pd.Series(dtype=bool)).sum())
        trunc = int(df_out.get("truncated", pd.Series(dtype=bool)).sum())
        noans = int(df_out.answer.isna().sum())
        print(f"  без числа у відповіді: {noans}"
              f"   з них порожня відповідь: {empty}, обірвано: {trunc}")
        if empty:
            print("  УВАГА: порожні відповіді означають замалий max_tokens, а не")
            print("  нездатність моделі — це дефект обв'язки, не результат.")
        tok = df_out.get("tokens", pd.Series(dtype=float)).fillna(0)
        res = df_out.get("reserved_tokens", pd.Series(dtype=float)).fillna(0)
        print(f"  токенів фактично: {int(tok.sum())}   "
              f"зарезервовано: {int(res.sum())}   "
              f"(відношення {int(res.sum()) / max(int(tok.sum()), 1):.1f}x)")

    a = cache_io.audit(out_path)
    print(f"  аудит кешу: справжніх {a['real']}, фальшивих {a['transport']}, "
          f"нерозпізнаних {a['unknown']}, дублікатів {a['duplicates']}, "
          f"пошкоджених {a['broken']}")
    print()
    print("Жодної точності тут НЕ друкується: прогін сліпий.")
    print("Аналіз запускається окремо, після завершення обох маршрутів.")
    print("Збережено:", out_path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
