"""Generation pass over FinQA, cached so the cascade can be replayed for free.

The repair loop and the cascade are entangled: a verifier that rejects makes
the pipeline regenerate, so two cascade configurations do not see the same
candidates unless the run is done once and replayed. This module does the
expensive half exactly once and writes everything a replay needs — the code,
the value it produced and the signal vector, at every iteration.

Budget is the reason the cache is append-only and resumable. A full pass over
the table-only subset costs about two days of the free Groq tier; losing it to
a crash at question 200 is not an acceptable failure mode, so every finished
question is flushed to disk before the next one starts and a restart skips
what is already there.

Evaluation is restricted to the questions whose gold evidence lives in the
table (57.5% of the test split, see eval.audit_finqa). That is a property of
the method, which verifies table arithmetic, not a limitation discovered
afterwards.

Run:  python -m eval.finqa_run --n 10          # pilot
      python -m eval.finqa_run --n 300         # first cycle
      python -m eval.finqa_run --n 300 --resume
"""
import argparse
import collections
import json
import os
import random
import time
import traceback

from core import decision
from core.pipeline import run
from eval import cache_io
from generation import llm_client
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

THRESHOLD = 0.7
MAX_ITER = 3          # lower than the synthetic runs: every iteration is budget

# Two limits bind on the Groq free tier, and the tighter one is not the
# obvious one. 30 requests a minute is easy at two calls per question; 8000
# TOKENS a minute is not, because the provider reserves max_tokens rather
# than charging actual usage. Raising code generation to 1200 makes the token
# limit bite roughly four times sooner than the request limit, so the pace is
# computed from tokens.
TPM_LIMIT = 8000

# WRONG, and corrected 2026-09-26 -- kept here because the wording is the
# mistake. "Groq charges the RESERVED max_tokens" was inferred from 291
# retries at low real usage and labelled MEASURED. The retries were real; the
# explanation was not. The token ledger later showed 222,400 reserved against
# a 200,000 daily limit with ZERO daily 429s, which refutes reservation
# charging outright (5.14.5). The retries came from pacing with no headroom
# for the burst a repair iteration makes -- with the headroom below, the same
# route over the same questions collected 0 retries at the same max_tokens.
#
# Two corrections follow, and they are the ones that mattered. A question
# costs more than one code-generation call
# -- measured 2.0 of them plus one semantic check -- and the pace needs
# headroom, because a repair iteration arrives in a burst rather than spread
# evenly. Without headroom the first burst triggers a 429 and the backoff
# cascade makes the run six times slower than pacing correctly would.
CODEGEN_CALLS_PER_QUESTION = 2.0
SEMANTIC_CALLS_PER_QUESTION = 1.0
TPM_HEADROOM = 0.8


def auto_pace() -> float:
    """Seconds between questions that keep the run inside the token limit."""
    from generation.llm_client import CODEGEN_MAX_TOKENS, MAX_TOKENS
    reserved = (CODEGEN_CALLS_PER_QUESTION * CODEGEN_MAX_TOKENS
                + SEMANTIC_CALLS_PER_QUESTION * MAX_TOKENS)
    usable = TPM_LIMIT * TPM_HEADROOM
    questions_per_minute = usable / max(reserved, 1)
    return round(60.0 / max(questions_per_minute, 0.1), 1)


PACE_SECONDS = None         # None means auto_pace()

# FinQA's answer conventions, stated to the generator. These are properties of
# the corpus, not of the method: its gold programs multiply by const_1000000
# when the caption says "( in millions )", and store a percentage as the raw
# fraction. A system evaluated on FinQA has to follow both, and a run that does
# not is measuring prompt hygiene rather than verification quality.
#
# The unit caption is usually inside the table itself -- measured on the pilot,
# it sits in the first header cell in two cases out of three -- so honouring it
# needs no access to the surrounding text.
FINQA_INSTRUCTIONS = (
    "Conventions of this dataset:\n"
    "- The table caption may state a unit, e.g. '( in millions )' or "
    "'( in thousands )'. When it does, scale the final answer to absolute "
    "units accordingly.\n"
    "- Express a percentage or a ratio as a plain fraction, not multiplied "
    "by 100: write 0.141, not 14.1.\n"
    "- `result` must be a single number, never a DataFrame or a Series."
)


# The scope gate was built for the synthetic Ukrainian corpus, where an
# out-of-scope question was a real category. On FinQA every question is about
# its own table by construction, and the gate refused 23 of 300 -- the direct
# route answered 10 of those correctly (5.34.2). Off for FinQA runs; the
# default stays on so the frozen regressions do not move.
NO_GATE_ENV = "VKSK_NO_SCOPE_GATE"


def no_scope_gate() -> bool:
    return os.environ.get(NO_GATE_ENV, "0").strip() not in ("", "0", "false", "no")


def load_cache_ids(tag: str) -> list:
    path = cache_path(tag)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cache_path(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}.jsonl")


# ---- sampling ----

def table_only(records: list) -> list:
    """Questions whose gold evidence is entirely in the table."""
    return [r for r in records if fq.evidence_source(r) == "table"]


def stratified_sample(records: list, n: int, seed: int = 42) -> list:
    """Sample stratified by gold program length.

    Program length is the natural difficulty axis in FinQA: a one-operation
    lookup and a three-operation chain fail for different reasons, and an
    unstratified draw at n=300 would under-represent the long tail that the
    invariant signal is most likely to touch.
    """
    by_len = collections.defaultdict(list)
    for r in records:
        by_len[min(len(fq.program_ops(fq.gold_program(r))), 3)].append(r)

    rng = random.Random(seed)
    for v in by_len.values():
        rng.shuffle(v)

    total = sum(len(v) for v in by_len.values())
    out = []
    for k in sorted(by_len):
        share = round(n * len(by_len[k]) / total)
        out.extend(by_len[k][:share])

    # rounding can leave the sample short or long
    pool = [r for k in sorted(by_len) for r in by_len[k][len(out):]] if len(out) < n else []
    while len(out) < n and pool:
        out.append(pool.pop())
    out = out[:n]
    rng.shuffle(out)
    return out


# ---- correctness ----

def is_correct(predicted, expected, tol: float = 0.01):
    """FinQA execution accuracy at 1% relative tolerance."""
    if expected is None:
        return None
    try:
        p, e = float(predicted), float(expected)
    except (TypeError, ValueError):
        return False
    return abs(p - e) <= tol * max(abs(e), 1.0)


def scale_error(predicted, expected, tol: float = 0.01) -> bool:
    """True when the value is right but the unit is off by a power of ten.

    FinQA stores most percentage answers as a fraction (0.141 for "14.1%"),
    and a model that answers in percent is scored wrong for a reason that has
    nothing to do with arithmetic. FinAgent-RAG's five error categories have
    no place for this, so it is counted separately rather than folded into
    `arithmetic`.
    """
    if expected is None:
        return False
    try:
        p, e = float(predicted), float(expected)
    except (TypeError, ValueError):
        return False
    if abs(p - e) <= tol * max(abs(e), 1.0):
        return False
    # Powers of ten only. 100 is the percent/fraction convention; 1e3 and 1e6
    # are the "in thousands" / "in millions" caption, which lives in the text
    # the table-only pipeline does not read -- the gold programs say so
    # outright, multiplying by const_1000000.
    for factor in (1e2, 1e3, 1e6, 1e9):
        for q in (p * factor, p / factor):
            if abs(q - e) <= tol * max(abs(e), 1.0):
                return True
    return False


# ---- one question ----

def run_one(record: dict, instructions: str = None, tables: dict = None,
            gold_name: str = "") -> dict:
    """Full pipeline pass over one FinQA question, with everything a replay needs."""
    qid = record.get("id") or record.get("filename", "?")
    question = fq.question(record)
    gold = fq.gold_answer(record)
    tables = tables if tables is not None else fq.to_tables(record)

    llm_client.reset_stats()
    from core import pipeline as _pl
    _pl.salvage_reset()
    t0 = time.perf_counter()
    error = error_class = None
    try:
        result = run(question, tables, threshold=THRESHOLD, max_iter=MAX_ITER,
                     verbose=False, source_file=f"finqa:{qid}",
                     use_scope_gate=not no_scope_gate(),
                     instructions=instructions,
                     caption=fq.caption(record) if fq.use_caption() else None)
    except Exception as exc:
        # The class, not only the text. eval.cache_io decides from it whether
        # the failure happened in the transport -- in which case there is no
        # result and the row must not be written -- or inside the method, in
        # which case it is a result and counts as wrong.
        error, error_class = f"{type(exc).__name__}: {exc}", type(exc).__name__
        result = {"status": "ERROR", "trace": []}
    elapsed = time.perf_counter() - t0
    stats = llm_client.get_stats()

    o = decision.outcome(result, THRESHOLD)
    answer = o["answer"]
    candidate = o["candidate"]

    by_tag = stats.get("by_tag", {})
    steps = []
    for st in result.get("trace", []):
        steps.append({
            "iter": st.get("iter"),
            "level": st.get("level"),
            "result": _jsonable(st.get("result")),
            "signals": st.get("signals"),
            "confidence": st.get("confidence"),
            "exception_type": st.get("exception_type"),
            # The code and the error text are what a failure diagnosis needs,
            # and they were missing from the first caches: 112 of 116
            # exceptions were NameError or SyntaxError with nothing recorded
            # about which name or which line.
            "code": st.get("code"),
            "error": st.get("error"),
        })

    return {
        "id": qid,
        "question": question,
        "gold": gold,
        "gold_program": fq.gold_program(record),
        "gold_ops": fq.program_ops(fq.gold_program(record)),
        "n_ops": len(fq.program_ops(fq.gold_program(record))),
        "table_shape": [len(record.get("table") or []) - 1,
                        len((record.get("table") or [[]])[0]) - 1],
        "status": result.get("status"),
        "abstain_reason": o["abstain_reason"],
        "answer": _jsonable(answer),
        "candidate": _jsonable(candidate),
        "has_candidate": o["has_candidate"],
        "confidence": o["confidence"],
        "signals": o["signals"],
        "iterations": o["iterations"],
        "answer_correct": is_correct(answer, gold) if answer is not None else None,
        "candidate_correct": is_correct(candidate, gold) if candidate is not None else None,
        "candidate_scale_error": scale_error(candidate, gold),
        "steps": steps,
        "calls_codegen": by_tag.get("codegen", {}).get("calls", 0),
        "calls_semantic": by_tag.get("semantic", {}).get("calls", 0),
        # Budget retries are full calls. They are tagged "codegen" and so are
        # already inside calls_codegen; this counts them separately so the
        # cost of the truncation path is visible instead of hiding inside
        # generation, and so that "solved without an LLM repair" is not
        # quietly credited with work that did reach the model.
        "calls_truncation_retry": sum(1 for st in result.get("trace", [])
                                      if st.get("level") == "LB"),
        "used_caption": fq.use_caption(),
        "n_tables": len(tables),
        "gold_table": gold_name,
        # Which tables the generated code actually names. More than one is the
        # multi-table signature: the category FinQA cannot produce under gold
        # context and FinAgent-RAG report as 28.9% of their residual errors.
        "tables_referenced": sorted(
            {n for st in result.get("trace", []) if st.get("code")
             for n in tables if n in (st.get("code") or "")}),
        # Numbers the program introduced from outside the tables. Non-zero is
        # not proof of error, but with the caption switched on it is the thing
        # to watch: metadata must not become data.
        "foreign_literals": sorted({v for st in result.get("trace", [])
                                    for v in (st.get("foreign_literals") or [])}),
        "calls_total": (by_tag.get("codegen", {}).get("calls", 0)
                        + by_tag.get("semantic", {}).get("calls", 0)),
        "retries": stats.get("retries", 0),
        # Actual usage, split, and the reservation beside it. The daily budget
        # is invisible until a 429 states it, so the only way to learn whether
        # it charges what a call used or what it reserved is to know both
        # sums at the moment the limit is hit (5.14.5 left this open).
        "tokens": stats.get("total_tokens", 0),
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "completion_tokens": stats.get("completion_tokens", 0),
        "reserved_tokens": (
            by_tag.get("codegen", {}).get("calls", 0) * llm_client.CODEGEN_MAX_TOKENS
            + by_tag.get("semantic", {}).get("calls", 0) * llm_client.MAX_TOKENS),
        "elapsed_s": round(elapsed, 2),
        # How often the reply had to be salvaged from prose. Recorded because a
        # model that needs it is being helped in a way the others were not, and
        # that asymmetry has to be visible in the report rather than implied.
        "salvage": _pl.salvage_stats(),
        "error": error,
        "error_class": error_class,
    }


def _jsonable(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


# ---- driver ----

def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None, help="cache name; default sample_<n>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pace", type=float, default=None,
                    help="seconds between questions; default is computed from "
                         "the token-per-minute limit")
    ap.add_argument("--no-conventions", action="store_true",
                    help="omit the dataset conventions from the prompt (ablation)")
    ap.add_argument("--only-ids", default="",
                    help="файл зі списком id (по одному на рядок); прогнати "
                         "лише їх із відтвореної вибірки")
    ap.add_argument("--exclude", default="",
                    help="comma-separated cache tags whose question ids to "
                         "exclude; use to keep a confirmatory run clear of the "
                         "questions a hypothesis was found on")
    ap.add_argument("--corpus", action="store_true",
                    help="put every table of the filing in the prompt, not just "
                         "the question's own one")
    ap.add_argument("--shuffle-filings", action="store_true",
                    help="null control: draw the pool from a DIFFERENT filing, "
                         "so any cross-table finding is noise by construction")
    args = ap.parse_args()
    if args.pace is None:
        args.pace = auto_pace()

    tag = args.tag or f"{args.split}_n{args.n}_seed{args.seed}"
    if args.corpus:
        tag = f"{tag}_corpus"
    if args.shuffle_filings:
        tag = f"{tag}_shuffled"
    if fq.use_caption():
        # A run with the caption is a different experiment and gets its own
        # file. Overwriting the earlier cache would destroy the comparison
        # this option exists to make.
        tag = f"{tag}_cap"
    path = cache_path(tag)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault(llm_client.LEDGER_ENV,
                          os.path.join(CACHE_DIR, "token_ledger.jsonl"))

    provider = llm_client.active_provider()
    print("=" * 78)
    print(f"FinQA: ПРОГІН ГЕНЕРАЦІЇ (конфігурація A) -> {os.path.basename(path)}")
    print("=" * 78)
    print(f"провайдер: {provider}   модель: {llm_client.active_model()}")
    if provider == "mock":
        print("MOCK не містить запитів FinQA — прогін безглуздий. Потрібен GROQ_API_KEY.")
        return

    records = fq.load_split(args.split)
    subset = table_only(records)

    filing_pool = None
    if args.corpus:
        pool_records = []
        for sp in ("train", "dev", "test"):
            try:
                pool_records += fq.load_split(sp)
            except Exception:
                pass
        filing_pool = fq.build_filing_pool(pool_records)
        print(f"корпус: пул з {len(filing_pool)} філінгів, БЕЗ обмеження розміру")
        if args.shuffle_filings:
            print("НУЛЬОВИЙ КОНТРОЛЬ: пул береться з ЧУЖОГО філінга")
    if args.exclude:
        # Questions a hypothesis was discovered on cannot also test it. The
        # exploratory set is removed BEFORE sampling, so the confirmatory
        # sample is drawn from untouched questions only.
        seen = set()
        for tag in (t.strip() for t in args.exclude.split(",") if t.strip()):
            try:
                seen |= {r["id"] for r in load_cache_ids(tag)}
            except FileNotFoundError:
                print(f"  попередження: кеш {tag} не знайдено, пропускаю")
        before = len(subset)
        subset = [r for r in subset if r["id"] not in seen]
        print(f"виключено розвідувальних питань: {before - len(subset)} "
              f"(залишилось {len(subset)})")

    sample = stratified_sample(subset, args.n, args.seed)

    if args.only_ids:
        # The sample is still reproduced in full from the seed, then narrowed.
        # Narrowing before sampling would draw a different sample and the
        # comparison would no longer be paired.
        with open(args.only_ids, encoding="utf-8") as f:
            want = {l.strip() for l in f if l.strip()}
        before = len(sample)
        sample = [r for r in sample if r["id"] in want]
        print(f"обмежено списком: {len(sample)} з {before} "
              f"(у списку {len(want)})")

    done = load_done(path) if args.resume else set()
    if done and not args.resume:
        print(f"УВАГА: {os.path.basename(path)} вже існує. Запусти з --resume або зміни --tag.")
        return
    todo = [r for r in sample if (r.get("id") or r.get("filename")) not in done]

    print(f"підмножина «еталон у таблиці»: {len(subset)} з {len(records)}")
    print(f"вибірка: {len(sample)} (стратифікація за довжиною gold-програми)")
    print(f"вже пораховано: {len(done)}   лишилось: {len(todo)}")
    print(f"темп: {args.pace} с між питаннями   оцінка: ~{len(todo) * (args.pace + 2) / 60:.0f} хв")
    print()

    instructions = None if args.no_conventions else FINQA_INSTRUCTIONS
    print(f"конвенції датасету в промпті: {'НІ (ablation)' if args.no_conventions else 'так'}")
    print(f"підпис таблиці ({fq.CAPTION_ENV}): "
          f"{'УВІМКНЕНО' if fq.use_caption() else 'вимкнено (дефолт)'}")
    print()

    t_start = time.perf_counter()
    n_ok = n_err = n_fake = 0
    tpd_stop = False
    with open(path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, start=1):
            try:
                tables, gold_name = None, ""
                if filing_pool is not None:
                    src = rec
                    if args.shuffle_filings:
                        # A pool from an unrelated filing. Anything a
                        # cross-table check finds here is noise, and that
                        # number is the floor the real run must beat.
                        keys = sorted(filing_pool)
                        other = keys[(keys.index(fq.filing_of(rec)) + 7) % len(keys)] \
                            if fq.filing_of(rec) in filing_pool else keys[0]
                        src = {"filename": f"{other[0]}/{other[1]}/x.pdf",
                               "table": rec.get("table"), "qa": rec.get("qa")}
                    tables = fq.corpus_tables(src, filing_pool, seed=args.seed)
                    gold_name = fq.gold_table_name(rec, tables)
                row = run_one(rec, instructions=instructions, tables=tables,
                              gold_name=gold_name)
            except Exception:
                n_err += 1
                print(f"[{i}/{len(todo)}] НЕОЧІКУВАНИЙ ЗБІЙ:")
                traceback.print_exc()
                continue

            # flush immediately: a crash must not cost the budget already
            # spent. cache_io refuses a row whose call never reached the
            # model, so a question the transport dropped stays unmarked and
            # a resume asks it again instead of recording a false failure.
            try:
                cache_io.write_row(f, row, on_unknown=lambda r: print(
                    f"  УВАГА: нерозпізнане походження помилки — {r}"))
            except cache_io.FakeRow as exc:
                n_fake += 1
                print(f"[{i}/{len(todo)}] ВИКЛИК НЕ ВІДБУВСЯ, рядок НЕ записано: {exc}")
                if "tpd" in str(exc).lower() or "tokens per day" in str(exc).lower():
                    print("  добовий ліміт токенів вичерпано — зупиняюсь; "
                          "решта питань лишається непоміченою для відновлення")
                    tpd_stop = True
                    break
                continue

            n_ok += 1
            if row["error"]:
                n_err += 1
            mark = {"ACCEPT": "+", "ABSTAIN": ".", "ERROR": "!"}.get(row["status"], "?")
            ok = row["answer_correct"]
            print(f"[{i}/{len(todo)}] {mark} {row['status']:7s} "
                  f"ans={row['answer']} gold={row['gold']} "
                  f"{'OK' if ok else ('WRONG' if ok is False else '-')} "
                  f"iter={row['iterations']} calls={row['calls_codegen'] + row['calls_semantic']} "
                  f"{row['elapsed_s']}s")

            if i < len(todo):
                time.sleep(args.pace)

    print()
    print(f"записано: {n_ok}   збоїв методу: {n_err}   "
          f"викликів не відбулось: {n_fake}   "
          f"час: {(time.perf_counter() - t_start) / 60:.1f} хв")
    if n_fake:
        print(f"  {n_fake} питань лишились НЕПОМІЧЕНИМИ — повторний запуск "
              f"з --resume перепитає саме їх")
    if tpd_stop:
        print("  зупинено добовим лімітом токенів, не кінцем вибірки")
    a = cache_io.audit(path)
    print(f"кеш: {a['rows']} рядків, справжніх {a['real']}, "
          f"фальшивих {a['transport']}, нерозпізнаних {a['unknown']}, "
          f"дублікатів {a['duplicates']}, пошкоджених {a['broken']}")
    print("Збережено:", path)
    print()
    print("Каскад тепер програється офлайн: python -m eval.finqa_replay --tag " + tag)


if __name__ == "__main__":
    main()
