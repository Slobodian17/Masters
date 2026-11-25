"""Combinatorial core of the query-type dataset, English.

Deliberately smaller than its Ukrainian counterpart, and smaller for a reason
rather than out of haste. The Ukrainian generator carries a grammar: every
metric has nominative, genitive and accusative forms plus a gender, and an
agreement table supplies the right adjective ending for each template slot.
That machinery exists because Ukrainian would otherwise produce ungrammatical
queries in about half the combinations, and a classifier trained on broken
language would be learning our carelessness.

English inflects almost none of this, so the same coverage needs one metric
form and no agreement table. Fewer templates per class is then a choice about
how much of the API budget to spend on paraphrasing, not a compromise on
correctness: the classifier learns the FORM of an operation, and three
templates per class already give it three distinct forms.

Same interface as `dataset_core`: build() -> [{query, type, seed_id, ...}].
"""
import hashlib
import random

from generation.dataset_core import TAXONOMY

# One form each, no gender. Includes metrics absent from the fixtures so the
# classifier does not bind itself to one document's labels.
METRICS = [
    "revenue", "net profit", "gross profit", "operating profit",
    "cost of sales", "operating expenses", "finance costs", "income tax",
    "total assets", "total liabilities", "total equity", "trade receivables",
    "inventory", "cash and cash equivalents", "capital expenditure",
    "net cash from operating activities", "dividends paid", "interest paid",
]

QUARTERS = [f"Q{q} {y}" for y in (2023, 2024) for q in (1, 2, 3, 4)]
YEARS = ["2023", "2024"]
THRESHOLDS = ["50000", "100000", "150000", "200000", "500000", "1000000"]

TEMPLATES = {
    "SCALAR_LOOKUP": [
        "What was {m} in {period}?",
        "Give the value of {m} for {period}.",
        "{m} in {period} — how much?",
    ],
    "AGGREGATE_SUM": [
        "What was total {m} for {year}?",
        "Sum {m} across all quarters of {year}.",
        "How much {m} in total over {year}?",
    ],
    "AGGREGATE_MEAN": [
        "What was the average {m} for {year}?",
        "Mean {m} across the quarters of {year}?",
        "On average, how much {m} per quarter in {year}?",
    ],
    "FIND_MAX_MIN": [
        "What was the maximum {m} across all quarters?",
        "In which quarter was {m} lowest?",
        "Find the largest {m} over the whole period.",
    ],
    "COMPARE_TWO": [
        "What is the difference between {m} in {period2} and {period}?",
        "How much larger was {m} in {period2} than in {period}?",
        "Compare {m} in {period} and {period2} in absolute terms.",
    ],
    "GROWTH_RATE": [
        "By what percent did {m} grow from {period} to {period2}?",
        "What was the growth of {m} from {period} to {period2} in percent?",
        "Percentage change in {m} between {period} and {period2}?",
    ],
    "CONDITIONAL_FILTER": [
        "How many quarters had {m} above {threshold}?",
        "List the quarters where {m} exceeded {threshold}.",
        "In how many quarters did {m} stay below {threshold}?",
    ],
    "RATIO": [
        "What was the {m} margin (%) in {period}?",
        "What was the share of {m} in total assets (%) in {period}?",
        "Ratio of {m} to revenue in {period}?",
    ],
}


def _fill(template: str, metric: str, rng) -> str:
    i, j = sorted(rng.sample(range(len(QUARTERS)), 2))
    return (template
            .replace("{m}", metric)
            .replace("{period2}", QUARTERS[j])
            .replace("{period}", QUARTERS[i])
            .replace("{year}", rng.choice(YEARS))
            .replace("{threshold}", rng.choice(THRESHOLDS)))


def seed_id(qtype: str, template_idx: int, metric: str) -> str:
    """Stable identifier of a seed group, same scheme as the Ukrainian core."""
    raw = f"en|{qtype}|{template_idx}|{metric}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def build(seed: int = 42, per_combo: int = 1) -> list:
    rng = random.Random(seed)
    out, seen = [], set()

    combos = [(t, i, m)
              for t in TAXONOMY
              for i in range(len(TEMPLATES[t]))
              for m in METRICS]
    combos.sort(key=lambda c: (c[0], c[1], c[2]))

    for qtype, tmpl_idx, metric in combos:
        for _ in range(per_combo):
            q = _fill(TEMPLATES[qtype][tmpl_idx], metric, rng)
            if q in seen:
                continue
            seen.add(q)
            out.append({
                "query": q,
                "type": qtype,
                "seed_id": seed_id(qtype, tmpl_idx, metric),
                "template_idx": tmpl_idx,
                "metric": metric,
                "source": "combinatorial",
            })
    return out


def _assert_no_placeholders(data: list):
    bad = [d for d in data if "{" in d["query"] or "}" in d["query"]]
    if bad:
        raise AssertionError(f"unfilled placeholders in {len(bad)} queries, "
                             f"e.g.: {bad[0]['query']}")


if __name__ == "__main__":
    import collections

    data = build()
    _assert_no_placeholders(data)
    print(f"generated: {len(data)} queries, seed groups: "
          f"{len({d['seed_id'] for d in data})}")
    print(dict(collections.Counter(d["type"] for d in data)))

    rng = random.Random(7)
    print("\nrandom sample:")
    for t in TAXONOMY:
        rows = [d for d in data if d["type"] == t]
        for d in rng.sample(rows, 2):
            print(f"  [{t:19s}] {d['query']}")
