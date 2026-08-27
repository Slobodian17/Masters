"""Text normalisation for queries and table labels.

Shared by the structural index and the scope gate. Retrieval accuracy depends
on this tokeniser, so changes to the suffix or stopword lists require
re-running `python -m eval.eval_retrieval`.
"""
import re


# Endings stripped so inflected forms of a word collapse to one stem.
# Order matters: longer suffixes first.
_SUFFIXES = (
    "ями", "ами", "ові", "еві", "ією", "ями", "ах", "ях", "ам", "ям", "ою", "ею",
    "ів", "ей", "ий", "ій", "их", "ім", "ом", "ем", "ів", "ти", "ся",
    "а", "я", "у", "ю", "е", "и", "і", "о",
)

_STOPWORDS = {
    "яка", "який", "яке", "які", "що", "за", "на", "у", "в", "з", "із", "до", "по",
    "від", "для", "та", "і", "й", "а", "але", "чи", "скільки", "наскільки", "як",
    "усіх", "всіх", "усі", "всі", "рік", "року", "роки", "було", "є", "більше",
    "менше", "між", "цей", "ця", "це",
}

_MIN_TOKEN = 3


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if len(word) - len(suf) >= 4 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> set:
    """Текст -> множина нормалізованих токенів. Числа зберігаються як є
    (напр. '150000' із запиту-фільтра), роки та квартали теж — їх ловить
    temporal_index, але й тут вони не шкодять."""
    raw = re.findall(r"[0-9]+|[^\W\d_]+", str(text).lower(), flags=re.UNICODE)
    out = set()
    for w in raw:
        if w in _STOPWORDS or len(w) < _MIN_TOKEN:
            continue
        out.add(_stem(w) if not w.isdigit() else w)
    return out
