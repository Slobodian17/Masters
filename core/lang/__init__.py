"""Language profiles.

Everything that ties the pipeline to one natural language lives in a profile
module: stemmer suffixes, stopwords, the regexes that separate relative from
absolute queries, the scope gate's operation vocabulary, the prompts sent to
the LLM, and the words its yes/no answer is read with.

English is the default because the main experiment runs on FinQA. Ukrainian is
kept complete, not as a legacy branch: comparing the two is the point. The
phenomena that make Ukrainian harder — inflection the stemmer must undo, and
the percent/interest homonymy that English does not have — are visible only
when both profiles run the same pipeline.

    VKSK_LANG = en (default) | uk

The profile is read at call time rather than import time, so a test can switch
languages inside one process.
"""
import importlib
import os
import re
from functools import lru_cache

LANG_ENV = "VKSK_LANG"
DEFAULT_LANG = "en"
SUPPORTED = ("en", "uk")


def active_lang() -> str:
    code = os.environ.get(LANG_ENV, DEFAULT_LANG).strip().lower()
    if code not in SUPPORTED:
        raise ValueError(f"Unknown {LANG_ENV}={code!r}; expected one of {SUPPORTED}")
    return code


@lru_cache(maxsize=None)
def _load(code: str):
    return importlib.import_module(f"core.lang.{code}")


def get(code: str = None):
    """The active profile module."""
    return _load(code or active_lang())


# ---- regex helpers shared by every profile ----

NEVER = re.compile(r"(?!x)x")  # matches nothing


@lru_cache(maxsize=None)
def compile_any(patterns: tuple):
    """Alternation over `patterns`, or a never-matching regex when empty.

    An empty alternation compiles to the empty pattern, which matches at
    position 0 of every string. A profile that legitimately has no patterns
    would then answer True to everything.
    """
    if not patterns:
        return NEVER
    return re.compile("|".join(patterns), re.IGNORECASE | re.UNICODE)


@lru_cache(maxsize=None)
def _word_re(words: tuple):
    body = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE | re.UNICODE)


def parse_yes_no(text, prof=None):
    """Read a yes/no answer out of an LLM reply. -> True | False | None.

    Matching is on whole words, not substrings. The substring form is a silent
    failure waiting to happen: "no" sits inside "not", "none", "nothing",
    "cannot" and "know", so a refusal to answer would be read as a firm "no".
    Ukrainian has the same trap with "ні" inside "ніколи" and "ніяк".

    When both a yes-word and a no-word appear, the earlier one wins: a reply
    opening "no, that is not plausible" is a no, even though a yes-word may
    follow later.
    """
    prof = prof or get()
    s = str(text)
    y = _word_re(tuple(prof.YES_WORDS)).search(s)
    n = _word_re(tuple(prof.NO_WORDS)).search(s)
    if y and n:
        return y.start() < n.start()
    if y:
        return True
    if n:
        return False
    return None
