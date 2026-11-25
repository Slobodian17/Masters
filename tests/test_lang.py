"""Unit tests for the language profiles.

The yes/no cases are the point of this file. `_v_semantic` used to read the
reply with a substring test, which is correct for a one-word answer and wrong
for every hedged one: "no" is a substring of "not", "none", "nothing",
"cannot" and "know", so "I do not know" would have been recorded as a
confident "no" and the semantic signal would have failed without raising
anything. The same shape of bug has already appeared four times in this
project, always as a mechanism that looked like it worked while measuring
something else.

Run:  python -m tests.test_lang
"""
import os
import sys

from core import lang
from core import text_utils

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append(f"{name}: got {got!r}, want {want!r}")


def with_lang(code: str):
    os.environ[lang.LANG_ENV] = code
    text_utils.reset_cache()
    return lang.get()


# --- English: words that merely contain "no" must not read as a refusal ---

EN_NOT_NO = [
    "not", "none", "nothing", "cannot", "know", "normal", "notice",
    "I do not know.", "There is nothing to compare against.",
    "I cannot tell from the schema alone.",
    "None of the columns are numeric.",
    "Not enough information.",
]

EN_YES = [
    "yes", "Yes", "YES", "yes.", "Yes, it is plausible.",
    "yes — the order of magnitude matches",
]

EN_NO = [
    "no", "No", "NO", "no.", "No, that is far too large.",
    "no - the value exceeds the table maximum",
]

# A reply carrying both words resolves to whichever comes first.
EN_MIXED = [
    ("No, this does not look plausible, yes I am sure.", False),
    ("Yes, this is plausible; no reason to doubt it.", True),
]

# --- Ukrainian: the same trap exists, with "ні" inside longer words ---

UK_NOT_NO = [
    "ніколи", "ніяк", "ніщо", "нічого", "жодної інформації",
    "Ніяк не можу визначити.",
]

UK_YES = ["так", "Так", "так.", "Так, виглядає правдоподібно."]
UK_NO = ["ні", "Ні", "ні.", "Ні, завелика величина."]


def test_yes_no():
    prof = with_lang("en")
    for s in EN_NOT_NO:
        check(f"en not-a-no {s!r}", lang.parse_yes_no(s, prof), None)
    for s in EN_YES:
        check(f"en yes {s!r}", lang.parse_yes_no(s, prof), True)
    for s in EN_NO:
        check(f"en no {s!r}", lang.parse_yes_no(s, prof), False)
    for s, want in EN_MIXED:
        check(f"en mixed {s!r}", lang.parse_yes_no(s, prof), want)

    prof = with_lang("uk")
    for s in UK_NOT_NO:
        check(f"uk not-a-no {s!r}", lang.parse_yes_no(s, prof), None)
    for s in UK_YES:
        check(f"uk yes {s!r}", lang.parse_yes_no(s, prof), True)
    for s in UK_NO:
        check(f"uk no {s!r}", lang.parse_yes_no(s, prof), False)


def test_empty_pattern_family():
    """An empty pattern list must match nothing, not everything.

    English has no percent/interest homonymy, so INTEREST_PATTERNS is empty.
    Joining an empty tuple gives the empty regex, which matches at position 0
    of every string -- every English query would have been classified as a
    loan-interest query.
    """
    from core import query_semantics

    with_lang("en")
    check("en interest family empty", len(lang.get().INTEREST_PATTERNS), 0)
    check("en interest never matches",
          query_semantics.is_interest_query("What was the revenue in Q3 2024?"), False)
    check("en interest never matches on empty string",
          query_semantics.is_interest_query(""), False)
    check("en percent is relative",
          query_semantics.is_relative_query("net profit margin in percent"), True)
    check("en plain sum is not relative",
          query_semantics.is_relative_query("total interest paid on loans in 2023"), False)

    with_lang("uk")
    check("uk interest wins over relative",
          query_semantics.is_relative_query("Скільки сплачено відсотків за кредитами?"), False)
    check("uk percent query is relative",
          query_semantics.is_relative_query("На скільки відсотків зросла виручка?"), True)


def test_stemmer_agreement():
    """Singular and plural of the same label must reach the same stem.

    Retrieval matches a query token against a label token; if "expenses" in the
    label and "expense" in the query stem differently, the hit is lost.
    """
    with_lang("en")
    pairs = [("expenses", "expense"), ("assets", "asset"), ("costs", "cost"),
             ("liabilities", "liability"), ("activities", "activity"),
             ("proceeds", "proceed"), ("borrowings", "borrowing"),
             ("payments", "payment"), ("receivables", "receivable"),
             ("dividends", "dividend"), ("shares", "share")]
    for a, b in pairs:
        check(f"en stem {a}=={b}", text_utils.stem(a) == text_utils.stem(b), True)

    with_lang("uk")
    check("uk stem is one pass", lang.get().STEM_ITERATE, False)


def test_default_language():
    os.environ.pop(lang.LANG_ENV, None)
    check("default lang", lang.active_lang(), "en")


def main():
    test_yes_no()
    test_empty_pattern_family()
    test_stemmer_agreement()
    test_default_language()

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
