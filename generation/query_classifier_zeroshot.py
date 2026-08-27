"""Zero-shot query-type classification through an LLM, used as a baseline.

Not a contribution in itself: it establishes what a prompted model achieves,
so the trained classifier can be judged against accuracy, latency and cost
rather than accuracy alone.
"""
import re

from generation.llm_client import active_model, active_provider, call_llm

# The eight operation types.
TAXONOMY = [
    "SCALAR_LOOKUP",
    "AGGREGATE_SUM",
    "AGGREGATE_MEAN",
    "FIND_MAX_MIN",
    "COMPARE_TWO",
    "GROWTH_RATE",
    "CONDITIONAL_FILTER",
    "RATIO",
]

# OUT_OF_SCOPE is not a class of the classifier; the scope gate decides it.
# Kept here only to reproduce the original nine-class measurement.
OUT_OF_SCOPE = "OUT_OF_SCOPE"
TAXONOMY_WITH_OOS = TAXONOMY + [OUT_OF_SCOPE]

_DESCRIPTIONS = {
    "SCALAR_LOOKUP": "зчитати одне конкретне значення з таблиці за показником і періодом",
    "AGGREGATE_SUM": "підсумувати значення показника за кілька періодів",
    "AGGREGATE_MEAN": "обчислити середнє значення показника за кілька періодів",
    "FIND_MAX_MIN": "знайти максимум або мінімум показника серед періодів",
    "COMPARE_TWO": "різниця між двома значеннями в абсолютних одиницях",
    "GROWTH_RATE": "зміна показника між періодами У ВІДСОТКАХ",
    "CONDITIONAL_FILTER": "відібрати періоди за умовою і порахувати кількість або суму",
    "RATIO": "відношення двох показників у відсотках (маржа, частка, рентабельність)",
    OUT_OF_SCOPE: "жодна таблиця документа не містить даних для відповіді",
}


def build_prompt(query: str, allow_oos: bool = False) -> str:
    labels = TAXONOMY_WITH_OOS if allow_oos else TAXONOMY
    lines = "\n".join(f"- {n}: {_DESCRIPTIONS[n]}" for n in labels)
    return (
        "Класифікуй запит користувача до фінансової таблиці за типом операції.\n\n"
        f"Доступні типи:\n{lines}\n\n"
        f"Запит: {query}\n\n"
        "Відповідай РІВНО однією міткою з переліку вище, великими літерами, "
        "без пояснень і без розділових знаків."
    )


SYSTEM = ("Ти — класифікатор запитів до табличних даних. Виводиш рівно одну "
          "мітку типу і нічого більше.")


_REASONING_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>",
                           re.IGNORECASE | re.DOTALL)


def _strip_reasoning(raw: str) -> str:
    """Strip reasoning blocks before parsing.

    Some models emit <think>...</think> weighing options; taking the first
    label mentioned would pick one from the reasoning rather than the answer.
    An unclosed block, from hitting the token limit, is also cut.
    """
    text = _REASONING_RE.sub(" ", raw or "")
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text


def _parse(raw: str, allow_oos: bool) -> str:
    """Extract the label: the first valid one mentioned after stripping
    any reasoning block."""
    labels = TAXONOMY_WITH_OOS if allow_oos else TAXONOMY
    text = _strip_reasoning(raw).upper()
    hits = [(text.find(lbl), lbl) for lbl in labels if lbl in text]
    if hits:
        return min(hits)[1]
    return "UNPARSED"


def classify(query: str, allow_oos: bool = False) -> str:
    if active_provider() == "mock":
        raise RuntimeError(
            "Zero-shot класифікація потребує реального LLM. Встанови GROQ_API_KEY "
            "(безкоштовний тариф console.groq.com). MOCK не може імітувати "
            "zero-shot узагальнення — саме воно й вимірюється."
        )
    return _parse(call_llm(build_prompt(query, allow_oos), system=SYSTEM), allow_oos)


def describe() -> dict:
    return {"provider": active_provider(), "model": active_model(),
            "classes": len(TAXONOMY)}
