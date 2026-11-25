"""The one place a cache row is written, and the one place it can be refused.

Two scripts have now had the same defect independently. A call that never
reached the model still produced a cache row, and because every cache here is
append-only and resumable, a resume reads that row as "done" and never asks
the question again. In `finqa_baseline` it cost 105 questions; in `finqa_run`
it cost 33, found only by auditing the file afterwards. Patching each script
where it was noticed would leave the next script to repeat it, so the rule
lives here and both write through this module.

The rule is NOT "refuse anything that is not a success". That version would
introduce the opposite bias, and of the worse kind: a question where the
pipeline itself crashes is a genuine result -- the method failed on it -- and
dropping it would quietly raise the method's score. What must never be written
is a row whose failure happened in the TRANSPORT, before any reply existed:
there is no result to record, only a note that the network said no.

So the test is on the cause, not on the status:

  transport failure   no reply was ever obtained (429, daily budget, timeout,
                      connection reset) -> not a result, refuse to write, the
                      question stays unmarked and a resume asks it again.
  method failure      a reply came back and something downstream broke -> a
                      result, written with status ERROR and counted as wrong.

Anything unrecognised is written and flagged loudly rather than dropped:
silently discarding a row is the failure mode this module exists to prevent,
so an unknown cause must not inherit it.
"""
import json
import os

# Failures that mean "no reply was ever obtained". LLMError is raised by
# generation.llm_client only after the transport gave up -- exhausted 429
# retries, the daily token budget, a non-200 the provider will not fix.
TRANSPORT_CLASSES = (
    "LLMError",
    "Timeout", "ReadTimeout", "ConnectTimeout",
    "ConnectionError", "ConnectionResetError",
    "RequestException", "HTTPError", "SSLError", "ChunkedEncodingError",
)

# Signatures for rows written before `error_class` was recorded. The existing
# caches carry the class name inside the text, so an old file can be audited
# by the same rule that governs new writes.
TRANSPORT_MARKS = (
    "llmerror",
    "tokens per day", "tpd exhausted",
    "rate limit", "429",
    "timed out", "timeout",
    "connection aborted", "connection reset",
    "max retries exceeded",
)


class FakeRow(ValueError):
    """A row that records a call which never reached the model."""


def classify(row):
    """('real'|'transport'|'unknown', reason).

    `real` covers both a successful pass and a genuine method failure.
    """
    if not isinstance(row, dict):
        return "unknown", "рядок не є словником"
    if not row.get("id"):
        return "unknown", "немає id"

    err = row.get("error")
    if not err:
        return "real", ""

    cls = str(row.get("error_class") or "")
    if cls in TRANSPORT_CLASSES:
        return "transport", f"{cls}: {str(err)[:80]}"

    text = str(err).lower()
    if not cls:
        # Old rows put the class name at the front of the text.
        head = text.split(":", 1)[0].strip()
        if head in {c.lower() for c in TRANSPORT_CLASSES}:
            return "transport", str(err)[:90]
    if any(m in text for m in TRANSPORT_MARKS):
        return "transport", str(err)[:90]

    return "unknown", f"помилка нерозпізнаного походження: {str(err)[:80]}"


def is_transport_failure(row) -> bool:
    return classify(row)[0] == "transport"


def write_row(handle, row, on_unknown=None):
    """Append one row, refusing a transport failure.

    Raises FakeRow instead of returning a flag: a caller that ignores the
    return value is exactly how 105 rows were written, and an exception cannot
    be ignored by accident.
    """
    kind, reason = classify(row)
    if kind == "transport":
        raise FakeRow(reason)
    if kind == "unknown" and on_unknown is not None:
        on_unknown(reason)
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return True


def load_rows(path):
    rows, broken = [], 0
    if not os.path.exists(path):
        return rows, broken
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                broken += 1
    return rows, broken


def audit(path):
    """Counts a run must be able to state about its own cache."""
    rows, broken = load_rows(path)
    kinds = {"real": 0, "transport": 0, "unknown": 0}
    offenders = []
    for r in rows:
        kind, reason = classify(r)
        kinds[kind] += 1
        if kind != "real":
            offenders.append((r.get("id"), kind, reason))
    ids = [r.get("id") for r in rows]
    return {"path": path, "rows": len(rows), "broken": broken,
            "duplicates": len(ids) - len(set(ids)),
            "real": kinds["real"], "transport": kinds["transport"],
            "unknown": kinds["unknown"], "offenders": offenders}


def archive_fakes(path, note=""):
    """Move transport failures out of the cache into a sibling archive.

    They are not deleted. What happened during a run is part of the record of
    the run, and the count of calls the free tier refused is itself a measured
    quantity; it just must not sit in the file the analysis reads.
    """
    rows, broken = load_rows(path)
    keep, moved = [], []
    for r in rows:
        (moved if is_transport_failure(r) else keep).append(r)
    if not moved:
        return {"moved": 0, "kept": len(keep), "archive": None, "broken": broken}

    arch = path.replace(".jsonl", "") + ".rejected.jsonl"
    with open(arch, "a", encoding="utf-8") as f:
        if note:
            f.write(json.dumps({"_note": note}, ensure_ascii=False) + "\n")
        for r in moved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"moved": len(moved), "kept": len(keep), "archive": arch,
            "broken": broken, "ids": [r.get("id") for r in moved]}
