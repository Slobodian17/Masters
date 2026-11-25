"""A failed call must not create a cache row.

This is the test that was missing when the same defect appeared twice. It has
two halves, because the defect has two halves: the rule itself, and the
guarantee that every script goes through the rule instead of writing its own
line to the file.

Run:  python -m tests.test_cache_io
"""
import io
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import cache_io

PASS, FAIL = [], []


def ok(label, cond, detail=""):
    if cond:
        PASS.append(label)
    else:
        FAIL.append(f"{label}" + (f" — {detail}" if detail else ""))


# ---- the rule ----

def test_transport_refused():
    """The exact rows that poisoned both caches."""
    cases = [
        {"id": "a", "error": "Groq 429 after 5 attempts: ...",
         "error_class": "LLMError", "status": "ERROR"},
        {"id": "b", "error": "Groq TPD exhausted: tokens per day (TPD): Limit 200000",
         "error_class": "LLMError", "status": "ERROR"},
        # written before error_class existed: classified from the text
        {"id": "c", "error": "LLMError: Groq 429 after 5 attempts", "status": "ERROR"},
        {"id": "d", "error": "ConnectionError: Connection aborted", "status": "ERROR"},
    ]
    for row in cases:
        kind, _ = cache_io.classify(row)
        ok(f"transport {row['id']}", kind == "transport", kind)

    path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for row in cases:
            try:
                cache_io.write_row(f, row)
                FAIL.append(f"write_row записав фальшивий рядок {row['id']}")
            except cache_io.FakeRow:
                pass
    ok("файл лишився порожнім після 4 збоїв",
       (not os.path.exists(path)) or os.path.getsize(path) == 0,
       str(os.path.getsize(path)) if os.path.exists(path) else "no file")

    # and a resume must not see those ids
    rows, _ = cache_io.load_rows(path)
    ok("жоден id збою не потрапив у кеш", {r.get("id") for r in rows} == set())


def test_method_failure_is_kept():
    """A crash INSIDE the method is a result and must be recorded.

    The over-strict rule -- refuse anything whose status is not success --
    would drop these, and that biases in the method's favour, which is worse
    than the defect it fixes.
    """
    rows = [
        {"id": "e", "status": "ERROR", "error": "KeyError: 'revenue'"},
        {"id": "f", "status": "ERROR", "error": "ValueError: could not convert"},
        {"id": "g", "status": "ABSTAIN", "abstain_reason": "LOW_CONFIDENCE"},
        {"id": "h", "status": "ABSTAIN", "abstain_reason": "OUT_OF_SCOPE"},
        {"id": "i", "status": "ACCEPT", "answer": 1.0},
    ]
    path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
    seen_unknown = []
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            try:
                cache_io.write_row(f, row, on_unknown=seen_unknown.append)
            except cache_io.FakeRow as exc:
                FAIL.append(f"write_row відкинув справжній рядок {row['id']}: {exc}")
    back, broken = cache_io.load_rows(path)
    ok("усі 5 справжніх рядків записані", len(back) == 5, str(len(back)))
    ok("0 пошкоджених", broken == 0)
    ok("збій методу позначений як нерозпізнаний, але записаний",
       len(seen_unknown) == 2, str(len(seen_unknown)))


def test_archive_moves_not_deletes():
    path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
    rows = [{"id": "ok1", "status": "ACCEPT"},
            {"id": "bad1", "status": "ERROR", "error": "LLMError: Groq 429 after 5 attempts"},
            {"id": "ok2", "status": "ABSTAIN", "abstain_reason": "LOW_CONFIDENCE"},
            {"id": "bad2", "status": "ERROR", "error": "LLMError: Groq TPD exhausted"}]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    res = cache_io.archive_fakes(path, note="тест")
    ok("перенесено 2", res["moved"] == 2, str(res["moved"]))
    ok("лишилось 2", res["kept"] == 2, str(res["kept"]))
    kept, _ = cache_io.load_rows(path)
    ok("у кеші лише справжні", {r["id"] for r in kept} == {"ok1", "ok2"})
    arch, _ = cache_io.load_rows(res["archive"])
    ok("в архіві є обидва фальшиві",
       {r.get("id") for r in arch} >= {"bad1", "bad2"})
    ok("архів — окремий файл поруч", res["archive"].endswith(".rejected.jsonl"))


def test_audit_counts():
    path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "status": "ACCEPT"}) + "\n")
        f.write(json.dumps({"id": "a", "status": "ACCEPT"}) + "\n")
        f.write("{ broken\n")
        f.write(json.dumps({"id": "b", "error": "LLMError: 429"}) + "\n")
    a = cache_io.audit(path)
    ok("audit бачить дублікат", a["duplicates"] == 1, str(a["duplicates"]))
    ok("audit бачить пошкоджений рядок", a["broken"] == 1, str(a["broken"]))
    ok("audit бачить фальшивий рядок", a["transport"] == 1, str(a["transport"]))


# ---- the guarantee ----

def test_no_script_writes_rows_directly():
    """No script may append its own JSON line to a cache.

    The rule is only worth as much as this check: both defects were written by
    a script that had its own `f.write(json.dumps(row))`, and a third script
    would have had one too.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaldir = os.path.join(here, "eval")
    pat = re.compile(r"\.write\(\s*json\.dumps\(\s*row")
    bad = []
    for name in sorted(os.listdir(evaldir)):
        if not name.endswith(".py") or name == "cache_io.py":
            continue
        src = io.open(os.path.join(evaldir, name), encoding="utf-8").read()
        if pat.search(src):
            bad.append(name)
    ok("жоден скрипт не пише рядок кешу напряму", not bad, ", ".join(bad))

    for name in ("finqa_run.py", "finqa_baseline.py"):
        src = io.open(os.path.join(evaldir, name), encoding="utf-8").read()
        ok(f"{name} пише через cache_io", "cache_io.write_row" in src)


def main():
    test_transport_refused()
    test_method_failure_is_kept()
    test_archive_moves_not_deletes()
    test_audit_counts()
    test_no_script_writes_rows_directly()

    print(f"passed: {len(PASS)}")
    if FAIL:
        print(f"FAILED: {len(FAIL)}")
        for f in FAIL:
            print("  -", f)
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
