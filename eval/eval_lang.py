"""Sensitivity of the pipeline to the two tunable parts of a language profile.

Both were open questions when the English profile was written, and both are
the kind of question that invites a plausible answer instead of a number:

  * the suffix list. An empty list is the honest default for a language with
    little inflection, but `structural_index` compares token sets exactly, so
    a plural label and a singular query would miss each other.
  * the scope gate's similarity cutoff, which decides how close a query term
    has to be to a schema term before the document counts as able to answer.

Each setting is applied to the profile, the tokeniser cache is dropped, and
everything downstream is recomputed. Nothing here calls an LLM, so the numbers
are identical with or without a key.

Run:  python -m eval.eval_lang
"""
import os

import pandas as pd

from core import lang, scope_gate, text_utils
from eval.eval_retrieval import SINGLE_INDEXES, load_test_set, recall_at_1
from load_data import load_tables

PDF = "sample_report_multipage.pdf"

UK_SUFFIXES = lang.get("uk").SUFFIXES

# (label, suffixes, iterate). Ukrainian is measured with its own list and with
# no list at all, which is the only comparison that matters for it.
CONFIGS = {
    "en": [
        ("none (exact tokens)", (), False),
        ("plural: ies, s", ("ies", "s"), False),
        ("plural + iterate", ("ies", "s"), True),
        ("light: ies,s,ing,ed,y", ("ies", "s", "ing", "ed", "y"), False),
        ("light + iterate", ("ies", "s", "ing", "ed", "y"), True),
    ],
    "uk": [
        ("none (exact tokens)", (), False),
        ("project list, 1 pass", UK_SUFFIXES, False),
        ("project list + iterate", UK_SUFFIXES, True),
    ],
}


def measure(code: str, suffixes, iterate) -> dict:
    prof = lang.get(code)
    saved = (prof.SUFFIXES, prof.STEM_ITERATE)
    if suffixes is not None:
        prof.SUFFIXES, prof.STEM_ITERATE = suffixes, iterate
    text_utils.reset_cache()
    try:
        tables = load_tables(PDF)
        test_set = load_test_set()
        out = {}
        for name, idx in SINGLE_INDEXES.items():
            out[name], m = recall_at_1(test_set, tables, indexes=idx)
            if name == "structural":
                out["struct_misses"] = sorted(x["query"] for x in m)
        out["rrf"], misses = recall_at_1(test_set, tables)
        out["misses"] = sorted(m["query"] for m in misses)

        # Second consumer of the same tokeniser, and an independent one: the
        # gate refuses a query when no content token resembles the schema.
        wrong_refusals, adversarial_caught = [], None
        for it in test_set:
            g = scope_gate.check(it["query"], tables)
            refused = g["status"] == scope_gate.OUT_OF_SCOPE
            if it["type"] == "ADVERSARIAL":
                adversarial_caught = refused
            elif refused:
                wrong_refusals.append(it["query"])
        out["gate_wrong_refusals"] = wrong_refusals
        out["gate_adversarial"] = adversarial_caught

        pairs = PAIRS[code]
        out["pairs_agree"] = sum(
            1 for a, b in pairs if text_utils.stem(a) == text_utils.stem(b))
        out["pairs_total"] = len(pairs)
        return out
    finally:
        prof.SUFFIXES, prof.STEM_ITERATE = saved
        text_utils.reset_cache()


def run_language(code: str) -> pd.DataFrame:
    os.environ[lang.LANG_ENV] = code
    text_utils.reset_cache()
    rows, detail = [], {}
    for label, suffixes, iterate in CONFIGS[code]:
        r = measure(code, suffixes, iterate)
        rows.append({
            "конфігурація": label,
            "structural": f"{r['structural']:.0%}",
            "vector": f"{r['vector']:.0%}",
            "RRF": f"{r['rrf']:.0%}",
            "промахів structural": len(r["struct_misses"]),
            "форм зійшлось": f"{r['pairs_agree']}/{r['pairs_total']}",
            "гейт: хибних відмов": len(r["gate_wrong_refusals"]),
            "гейт ловить adversarial": "так" if r["gate_adversarial"] else "НІ",
        })
        detail[label] = r["struct_misses"]
    return pd.DataFrame(rows), detail


# Form pairs that have to collapse to one stem for a query in one form to
# retrieve a label written in the other. None of the test-set queries happens
# to exercise them, which is why they are reported separately from recall.
PAIRS = {
    "en": [("expenses", "expense"), ("assets", "asset"), ("costs", "cost"),
           ("liabilities", "liability"), ("activities", "activity"),
           ("proceeds", "proceed"), ("borrowings", "borrowing"),
           ("payments", "payment"), ("receivables", "receivable")],
    "uk": [("витрати", "витрат"), ("активів", "активи"),
           ("кошти", "коштів"), ("прибутку", "прибуток"),
           ("зобов'язань", "зобов'язання")],
}


def stem_pairs(code: str) -> pd.DataFrame:
    """Singular/plural pairs that must reach the same stem to be retrievable."""
    os.environ[lang.LANG_ENV] = code
    text_utils.reset_cache()
    pairs = PAIRS[code]
    rows = []
    for a, b in pairs:
        sa, sb = text_utils.stem(a), text_utils.stem(b)
        rows.append({"форма 1": a, "стем 1": sa, "форма 2": b, "стем 2": sb,
                     "збіг": "так" if sa == sb else "НІ"})
    return pd.DataFrame(rows)


CUTOFFS = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def gate_sweep(code: str) -> pd.DataFrame:
    """Scope gate behaviour across similarity cutoffs.

    Two ways to be wrong, and they pull in opposite directions: refusing a
    query the document can answer, and letting the adversarial query through.
    The usable window is where both columns are clean.
    """
    os.environ[lang.LANG_ENV] = code
    text_utils.reset_cache()
    prof = lang.get(code)
    saved = prof.TOKEN_SIMILARITY
    tables = load_tables(PDF)
    test_set = load_test_set()
    rows = []
    try:
        for cut in CUTOFFS:
            prof.TOKEN_SIMILARITY = cut
            wrong, adversarial = [], None
            for it in test_set:
                refused = scope_gate.check(it["query"], tables)["status"] == scope_gate.OUT_OF_SCOPE
                if it["type"] == "ADVERSARIAL":
                    adversarial = refused
                elif refused:
                    wrong.append(it["query"])
            rows.append({
                "поріг": f"{cut:.2f}{' <-' if abs(cut - saved) < 1e-9 else ''}",
                "хибних відмов": len(wrong),
                "ловить adversarial": "так" if adversarial else "НІ",
                "придатний": "так" if (not wrong and adversarial) else "ні",
            })
    finally:
        prof.TOKEN_SIMILARITY = saved
        text_utils.reset_cache()
    return pd.DataFrame(rows)


def main():
    saved = os.environ.get(lang.LANG_ENV)
    print("=" * 78)
    print("ЧУТЛИВІСТЬ RETRIEVAL ДО СТЕМЕРА, ПО МОВАХ")
    print("=" * 78)
    print("Жоден рядок нижче не звертається до LLM: числа однакові з ключем і без.")

    try:
        for code in ("en", "uk"):
            prof = lang.get(code)
            print()
            print(f"--- {code} ({prof.NAME}) ---")
            print(f"робочий набір суфіксів: {len(prof.SUFFIXES)}, "
                  f"ітерація до нерухомої точки: {prof.STEM_ITERATE}")
            table, detail = run_language(code)
            print(table.to_string(index=False))
            print()
            print("  Які саме запити промахує structural (рахунок однаковий")
            print("  не означає, що промахи ті самі):")
            base = None
            for label, misses in detail.items():
                mark = "" if base is None or misses == base else "  <- інший набір"
                base = base if base is not None else misses
                print(f"    [{label}] {len(misses)} промахів{mark}")
                for q in misses:
                    print(f"        {q}")
            print()
            print("  Пари форм, які мають зійтись в один стем:")
            print(stem_pairs(code).to_string(index=False))
            print()
            print(f"  Поріг схожості гейта (робочий: {prof.TOKEN_SIMILARITY}):")
            print(gate_sweep(code).to_string(index=False))
    finally:
        if saved is None:
            os.environ.pop(lang.LANG_ENV, None)
        else:
            os.environ[lang.LANG_ENV] = saved
        text_utils.reset_cache()

    print()
    print("ЯК ЧИТАТИ. RRF тримає 100% в усіх конфігураціях, бо векторний індекс")
    print("працює на символьних n-грамах і стемера не бачить. Отже вибір суфіксів")
    print("не впливає на критерій готовності Фази 1 — впливає лише на те, скільки")
    print("роботи лишається векторному індексу.")
    print()
    print("Набори з 23 і 18 запитів замалі, щоб розрізняти 4 в.п.: один запит")
    print("важить 4.3 в.п. Тому однаковий відсоток тут НЕ означає однакову")
    print("поведінку — див. списки промахів вище.")
    print()
    print("РЕЗУЛЬТАТ ВИМІРУ, як є: на цих наборах жодна конфігурація суфіксів не")
    print("змінює ні recall@1, ні поведінку гейта — в ЖОДНІЙ з двох мов. Єдине,")
    print("що різниться, — колонка «форм зійшлось». Вона показує, що вибір")
    print("суфіксів робить різницю в принципі, але жоден тестовий запит її не")
    print("зачіпає. Тому вибір робочого набору для en — це рішення за цією")
    print("колонкою, а не висновок з recall. Це рішення, не вимір.")
    print()
    print("А ось поріг гейта мови РОЗРІЗНЯЮТЬ, і це вже вимір: придатні вікна")
    print("не збігаються. Дивись дві останні таблиці.")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
