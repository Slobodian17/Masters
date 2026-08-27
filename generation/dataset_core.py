"""Combinatorial core of the query-type dataset.

Generates (query, type) pairs from a grammar of metric x period x template.
Deterministic, free and offline, so this layer always reproduces.

Templates agree with the grammatical gender and number of each metric.
Without that, roughly half the dataset would be ungrammatical, and the gap
between synthetic and real language would come from our own carelessness
rather than from the task.

Semantic plausibility of individual combinations is deliberately not checked:
the classifier learns the FORM of an operation, not whether a metric suits it.

Each row carries a `seed_id` so train/validation splits can be made by whole
groups, keeping paraphrases of one seed on the same side of the split.
"""
import hashlib
import random

TAXONOMY = [
    "SCALAR_LOOKUP", "AGGREGATE_SUM", "AGGREGATE_MEAN", "FIND_MAX_MIN",
    "COMPARE_TWO", "GROWTH_RATE", "CONDITIONAL_FILTER", "RATIO",
]

# (nominative, genitive, accusative, gender) where gender is f/m/n/pl.
# Includes metrics absent from the fixtures, so the classifier does not bind
# itself to one document's labels.
METRICS = [
    ("виручка", "виручки", "виручку", "f"),
    ("чистий прибуток", "чистого прибутку", "чистий прибуток", "m"),
    ("валовий прибуток", "валового прибутку", "валовий прибуток", "m"),
    ("собівартість реалізації", "собівартості реалізації", "собівартість реалізації", "f"),
    ("операційні витрати", "операційних витрат", "операційні витрати", "pl"),
    ("фінансові витрати", "фінансових витрат", "фінансові витрати", "pl"),
    ("власний капітал", "власного капіталу", "власний капітал", "m"),
    ("активи", "активів", "активи", "pl"),
    ("дебіторська заборгованість", "дебіторської заборгованості", "дебіторську заборгованість", "f"),
    ("грошові кошти", "грошових коштів", "грошові кошти", "pl"),
    ("податок на прибуток", "податку на прибуток", "податок на прибуток", "m"),
    ("витрати на маркетинг", "витрат на маркетинг", "витрати на маркетинг", "pl"),
    ("виплачені дивіденди", "виплачених дивідендів", "виплачені дивіденди", "pl"),
    ("амортизація", "амортизації", "амортизацію", "f"),
    ("запаси", "запасів", "запаси", "pl"),
    ("короткострокові зобов'язання", "короткострокових зобов'язань",
     "короткострокові зобов'язання", "pl"),
    ("операційний грошовий потік", "операційного грошового потоку",
     "операційний грошовий потік", "m"),
    ("нерозподілений прибуток", "нерозподіленого прибутку", "нерозподілений прибуток", "m"),
]

# Forms agreeing with the metric's gender.
AGREE = {
    "W":         {"f": "Яка", "m": "Який", "n": "Яке", "pl": "Які"},
    "SUM_ADJ":   {"f": "сукупна", "m": "сукупний", "n": "сукупне", "pl": "сукупні"},
    "TOTAL_ADJ": {"f": "сумарна", "m": "сумарний", "n": "сумарне", "pl": "сумарні"},
    "AVG_ADJ":   {"f": "середня", "m": "середній", "n": "середнє", "pl": "середні"},
    "MAX_ADJ":   {"f": "максимальна", "m": "максимальний", "n": "максимальне", "pl": "максимальні"},
    "IS":        {"f": "становить", "m": "становить", "n": "становить", "pl": "становлять"},
    "WAS":       {"f": "становила", "m": "становив", "n": "становило", "pl": "становили"},
    "GREW":      {"f": "зросла", "m": "зріс", "n": "зросло", "pl": "зросли"},
    "CHANGED":   {"f": "змінилася", "m": "змінився", "n": "змінилося", "pl": "змінилися"},
    "EXCEEDED":  {"f": "перевищила", "m": "перевищив", "n": "перевищило", "pl": "перевищили"},
    "BIGGER":    {"f": "більша", "m": "більший", "n": "більше", "pl": "більші"},
    "DIFFERS":   {"f": "відрізняється", "m": "відрізняється", "n": "відрізняється",
                  "pl": "відрізняються"},
    "WAS_TOP":   {"f": "була найбільшою", "m": "був найбільшим", "n": "було найбільшим",
                  "pl": "були найбільшими"},
    "WAS_BELOW": {"f": "була нижчою за", "m": "був нижчим за", "n": "було нижчим за",
                  "pl": "були нижчими за"},
}

QUARTERS = [f"Q{q} {y}" for y in (2023, 2024) for q in (1, 2, 3, 4)]
YEARS = ["2023", "2024"]
THRESHOLDS = ["50000", "100000", "150000", "200000", "500000", "1000000"]

# {n}/{g}/{a} are the metric's cases; other placeholders come from AGREE.
TEMPLATES = {
    "SCALAR_LOOKUP": [
        "{W} {n} за {period}?",
        "Яке значення {g} у {period}?",
        "Скільки {IS} {n} за {period}?",
        "Покажи {a} за {period}",
        "{n} за {period} — скільки?",
        "Назви {a} у {period}",
    ],
    "AGGREGATE_SUM": [
        "{W} {SUM_ADJ} {n} за {year} рік?",
        "{W} {TOTAL_ADJ} {n} за {year} рік?",
        "Скільки всього {g} за {year} рік?",
        "Порахуй загальну суму {g} за {year} рік",
        "Яка сума {g} за {year} рік?",
        "Підсумуй {a} за всі квартали {year} року",
    ],
    "AGGREGATE_MEAN": [
        "{W} {AVG_ADJ} {n} за {year} рік?",
        "Яке середнє значення {g} за {year} рік?",
        "Порахуй середнє значення {g} за {year} рік",
        "Скільки в середньому {WAS} {n} у {year} році?",
        "{W} {n} у середньому по кварталах {year} року?",
        "Обчисли середньоквартальне значення {g} у {year} році",
    ],
    "FIND_MAX_MIN": [
        "{W} {MAX_ADJ} {n} серед усіх кварталів?",
        "Яке мінімальне значення {g} серед усіх кварталів?",
        "У якому кварталі {n} {WAS_TOP}?",
        "Знайди найменше значення {g} за всі періоди",
        "Яке найбільше значення {g} у таблиці?",
        "Покажи пікове значення {g}",
    ],
    "COMPARE_TWO": [
        "Яка різниця у значенні {g} між {period} і {period2}?",
        "На скільки {n} у {period} {BIGGER}, ніж у {period2}?",
        "Наскільки змінилося значення {g} з {period} до {period2} в абсолютних цифрах?",
        "Порівняй {a} у {period} і {period2}",
        "Скільки складає розрив у {g} між {period} та {period2}?",
        "На скільки одиниць {DIFFERS} {n} у {period} від {period2}?",
    ],
    "GROWTH_RATE": [
        "На скільки відсотків {GREW} {n} з {period} по {period2}?",
        "Який приріст {g} з {period} по {period2} у відсотках?",
        "Яке відсоткове зростання {g} між {period} і {period2}?",
        "На скільки відсотків {CHANGED} {n} з {period} до {period2}?",
        "Який темп зростання {g} з {period} по {period2}?",
        "Динаміка {g} у відсотках з {period} по {period2}",
    ],
    "CONDITIONAL_FILTER": [
        "Скільки кварталів мали {a} більше {threshold}?",
        "У скількох кварталах {n} {EXCEEDED} {threshold}?",
        "Порахуй суму {g} за квартали, де значення перевищило {threshold}",
        "Порахуй квартали, у яких {n} {WAS_BELOW} {threshold}",
        "Скільки періодів мали {a} менше {threshold}?",
        "Знайди кількість кварталів зі значенням {g} понад {threshold}",
    ],
    "RATIO": [
        "Яка маржа {g} (%) за {period}?",
        "Яка частка {g} у виручці за {period}?",
        "Яке відношення {g} до виручки у {period} у відсотках?",
        "Скільки відсотків {IS} {n} від виручки за {period}?",
        "Яка рентабельність за показником {g} у {period}?",
        "Питома вага {g} у структурі за {period}, %",
    ],
}


def _fill(template: str, metric, rng) -> str:
    n, g, a, gender = metric
    # two distinct quarters in chronological order
    i, j = sorted(rng.sample(range(len(QUARTERS)), 2))
    period, period2 = QUARTERS[i], QUARTERS[j]

    text = template
    for key, forms in AGREE.items():
        text = text.replace("{" + key + "}", forms[gender])
    return (text
            .replace("{n}", n).replace("{g}", g).replace("{a}", a)
            .replace("{period2}", period2)
            .replace("{period}", period)
            .replace("{year}", rng.choice(YEARS))
            .replace("{threshold}", rng.choice(THRESHOLDS)))


def seed_id(qtype: str, template_idx: int, metric_name: str) -> str:
    """Stable identifier of a seed group."""
    raw = f"{qtype}|{template_idx}|{metric_name}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def build(seed: int = 42, per_combo: int = 1) -> list:
    """Build the dataset core.

    `per_combo` sets how many variants per template x metric pair.
    """
    rng = random.Random(seed)
    out, seen = [], set()

    combos = [(t, i, m)
              for t in TAXONOMY
              for i in range(len(TEMPLATES[t]))
              for m in METRICS]
    combos.sort(key=lambda c: (c[0], c[1], c[2][0]))  # deterministic order

    for qtype, tmpl_idx, metric in combos:
        for _ in range(per_combo):
            q = _fill(TEMPLATES[qtype][tmpl_idx], metric, rng)
            if q in seen:
                continue
            seen.add(q)
            out.append({
                "query": q,
                "type": qtype,
                "seed_id": seed_id(qtype, tmpl_idx, metric[0]),
                "template_idx": tmpl_idx,
                "metric": metric[0],
                "source": "combinatorial",
            })
    return out


def _assert_no_placeholders(data: list):
    """No placeholder may survive into a finished query."""
    bad = [d for d in data if "{" in d["query"] or "}" in d["query"]]
    if bad:
        raise AssertionError(f"незамінені плейсхолдери у {len(bad)} запитах, напр.: {bad[0]['query']}")


if __name__ == "__main__":
    import collections

    data = build()
    _assert_no_placeholders(data)
    print(f"Згенеровано: {len(data)} запитів, seed-груп: {len({d['seed_id'] for d in data})}")
    print(dict(collections.Counter(d["type"] for d in data)))

    rng = random.Random(7)
    print("\nВипадкова вибірка (перевірка граматики):")
    for t in TAXONOMY:
        rows = [d for d in data if d["type"] == t]
        for d in rng.sample(rows, 2):
            print(f"  [{t:19s}] {d['query']}")
