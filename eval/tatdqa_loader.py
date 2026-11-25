"""TAT-DQA: real PDF pages of financial reports, with questions and answers.

Used for the end-to-end test the FinQA data cannot support: FinQA ships tables
already extracted, so nothing there measures the step from a page to a table.
TAT-DQA ships the PDF.

What the dev release actually contains, checked rather than assumed:
  - 274 single-page PDFs with a real text layer (10 of 10 sampled)
  - one JSON per page whose `blocks` are LAYOUT blocks: a table row arrives as
    one text string, "Cost of revenue 43 % 50 % 55 %", and a question's
    `block_mapping` points at a character span inside it
  - no ready-made table matrix; the grid has to be recovered from the rows

So "their annotation" is a set of row strings, not a table. That is still
enough to compare against our own extraction, and it is stated here rather
than implied, because it is weaker than a gold grid.

ANSWER SCALE. An answer is a number plus a scale word: `answer=105,
scale='percent'` means 105%, `answer=5, scale='million'` means five million.
A prediction is correct within 1% of the answer either as given or after the
scale is applied, so a model answering 0.105 or 105000000 is not punished for
a convention.
"""
import json
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "tatdqa")
QA_JSON = os.path.join(DATA, "tatdqa_dataset_dev.json")
DOCS_ZIP = os.path.join(DATA, "tatdqa_docs_dev.zip")

TOL = 0.01
SCALES = {"": 1.0, None: 1.0, "percent": 0.01, "thousand": 1e3,
          "million": 1e6, "billion": 1e9}

_NUMTOK = re.compile(r"\(?-?[\d,]+\.?\d*\)?%?")
_ROWISH = re.compile(r"[\d,]+\.?\d*")


def load_qa():
    with open(QA_JSON, encoding="utf-8") as f:
        return json.load(f)


def page_blocks(uid, zf=None):
    zf = zf or zipfile.ZipFile(DOCS_ZIP)
    j = json.loads(zf.read(f"dev/{uid}.json").decode("utf-8"))
    return [b for p in j.get("pages", []) for b in p.get("blocks", [])]


def is_table_row(text):
    """A layout block that is a table row rather than prose.

    Reproducible test: at least two numeric tokens and few enough words that it
    cannot be a sentence. Prose about numbers usually has one figure and many
    words; a row has several figures and short labels.
    """
    t = str(text or "").strip()
    if not t:
        return False
    nums = _ROWISH.findall(t)
    words = t.split()
    return len(nums) >= 2 and len(words) <= 24


def question_uses_table(q, blocks_by_uid):
    """Every cell the answer was derived from lies in a table row."""
    mapping = q.get("block_mapping") or []
    if not mapping:
        return False
    hit = False
    for m in mapping:
        for uid in m:
            b = blocks_by_uid.get(uid)
            if b is None:
                return False
            if not is_table_row(b.get("text")):
                return False
            hit = True
    return hit


def numeric_answer(q):
    """The answer as one number, or None when it is text or a list."""
    a = q.get("answer")
    if isinstance(a, bool):
        return None
    if isinstance(a, (int, float)):
        return float(a)
    if isinstance(a, str):
        s = a.replace(",", "").replace("%", "").replace("$", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def is_correct(pred, q, tol=TOL):
    """Value against the answer, before and after its scale, within 1%."""
    gold = numeric_answer(q)
    if gold is None or pred is None:
        return None
    try:
        p = float(pred)
    except (TypeError, ValueError):
        return False
    mult = SCALES.get((q.get("scale") or "").strip(), 1.0)
    for cand in (p, p / mult if mult else p, p * mult):
        try:
            if abs(cand - gold) <= tol * max(abs(gold), 1.0):
                return True
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return False


def select(docs=None, zf=None):
    """Questions whose answer is ONE number computed from the table."""
    docs = docs or load_qa()
    zf = zf or zipfile.ZipFile(DOCS_ZIP)
    out = []
    for doc in docs:
        uid = doc["doc"]["uid"]
        try:
            blocks = page_blocks(uid, zf)
        except KeyError:
            continue
        by_uid = {b.get("uuid"): b for b in blocks}
        for q in doc.get("questions", []):
            if q.get("answer_type") != "arithmetic":
                continue
            if numeric_answer(q) is None:
                continue
            if not question_uses_table(q, by_uid):
                continue
            # NOT "uid": a question has its own `uid`, and spreading **q
            # after it silently overwrote the page id -- which made the count
            # of pages equal the count of questions.
            out.append({**q, "doc_uid": uid,
                        "source": doc["doc"].get("source"),
                        "page": doc["doc"].get("page")})
    return out


if __name__ == "__main__":
    import collections
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    docs = load_qa()
    zf = zipfile.ZipFile(DOCS_ZIP)
    total = sum(len(d.get("questions", [])) for d in docs)
    sel = select(docs, zf)
    print("=" * 74)
    print("TAT-DQA dev: відбір питань для наскрізного тесту")
    print("=" * 74)
    print(f"  сторінок (PDF)                       : {len(docs)}")
    print(f"  питань усього                        : {total}")
    ar = [q for d in docs for q in d.get("questions", [])
          if q.get("answer_type") == "arithmetic"]
    print(f"  з них arithmetic                     : {len(ar)}")
    print(f"  arithmetic з ОДНИМ числом у відповіді: "
          f"{sum(1 for q in ar if numeric_answer(q) is not None)}")
    print(f"  + усі операнди в рядках ТАБЛИЦІ      : {len(sel)}")
    print()
    print(f"  шкали у відібраних: "
          f"{dict(collections.Counter((q.get('scale') or '-') for q in sel))}")
    print(f"  сторінок, яких вони торкаються       : "
          f"{len({q['doc_uid'] for q in sel})}")
    print()
    print("--- ПРАВИЛО ОЦІНКИ, приклад на 3 питаннях ---")
    print("значення правильне, якщо воно в межах 1% від відповіді ДО або ПІСЛЯ")
    print("застосування шкали (percent=0.01, thousand=1e3, million=1e6)")
    print()
    for q in sel[:3]:
        g = numeric_answer(q)
        sc = (q.get("scale") or "").strip() or "-"
        mult = SCALES.get(sc if sc != "-" else "", 1.0)
        print(f"  {q['question'][:82]}")
        print(f"     answer={g}  scale={sc}  derivation={str(q.get('derivation'))[:46]}")
        ok = [x for x in (g, g * mult, g / mult if mult else g)]
        print(f"     зараховується: {g} (як є), {g * mult:g} (×шкала), "
              f"{(g / mult) if mult else g:g} (÷шкала), кожне ±1%")
