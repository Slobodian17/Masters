"""English profile — the default.

Written as a mirror of `uk`, position for position, so the two documents hold
the same numbers under different labels. Any difference the pipeline shows
between the profiles is then a property of the language, not of the data.

Two places differ in kind rather than in wording. Both were checked against
`eval.eval_lang`, and the check did not confirm what was expected of either:

  * INTEREST_PATTERNS is empty here, because English has two separate words
    where Ukrainian has one ambiguous one. This is a structural difference,
    not a tuning choice.
  * TOKEN_SIMILARITY differs, and this one is measured: the gate needs at
    least 0.75 in English to refuse the adversarial query, while Ukrainian
    catches it from 0.60 and instead starts refusing valid queries at 0.95.
    The usable windows are [0.75, 0.95] and [0.60, 0.90] — overlapping, but
    not the same, so one shared constant would sit at the edge of one of them.

The stemmer was expected to be a third such difference and is not. On the
current test sets no suffix configuration changes recall@1 or gate behaviour
in either language; see SUFFIXES below.
"""

CODE = "en"
NAME = "English"

# ---- tokenisation (core.text_utils) ----
#
# MEASURED, and the result is negative: `eval.eval_lang` reports recall@1 = 96%
# for structural, 100% for RRF, and zero wrong gate refusals under EVERY variant
# below, the empty one included. The 23-query set does not contain a query whose
# wording differs in number from its label, so it cannot tell the variants apart.
#
# The one column that does move is form-pair agreement: 0/9 with no suffixes,
# 9/9 with the list below. Those pairs do not affect any current measurement,
# so keeping this list is a decision taken on the pairs, not a conclusion drawn
# from recall. Stated plainly so it is not mistaken for the latter.
#
# Order is not the obvious one. "es" before "s" would turn "expenses" into
# "expens" while "expense" stays whole, giving one word two stems; stripping
# the bare "s" first keeps the pair together. "ies"/"y" are listed so that
# "liabilities" and "liability" both reduce to "liabilit".
SUFFIX_VARIANTS = {
    "none": (),
    "plural": ("ies", "s"),
    "light": ("ies", "s", "ing", "ed", "y"),
}

SUFFIXES = SUFFIX_VARIANTS["light"]

STEM_MIN_KEEP = 4

# Applied repeatedly until nothing more strips. This is what carries the last
# two form pairs: "proceeds" -> "proceed" -> "proce" agrees with "proceed" ->
# "proce" only at a fixed point. It changes no recall or gate number.
STEM_ITERATE = True

STOPWORDS = frozenset({
    "what", "which", "who", "how", "when", "where", "why",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "and", "or", "but", "not", "for", "from", "with", "without",
    "of", "in", "on", "at", "to", "by", "as", "than", "then", "that", "this",
    "these", "those", "it", "its", "there", "their", "all", "any", "each",
    "between", "over", "under", "into", "per", "much", "many", "more", "less",
    "year", "years",
})

MIN_TOKEN = 3

# ---- relative vs absolute queries (core.query_semantics) ----

RELATIVE_PATTERNS = (
    r"%",
    r"\bpercent",
    r"\bpct\b",
    r"\bmargin",
    r"\bprofitabilit",
    r"\bshare\s+of\b",
    r"\bproportion\b",
    r"\bratio\b",
    r"\bgrowth\b",
    r"\bgrew\b",
    r"\bgrowth\s+rate\b",
    r"\byear[-\s]over[-\s]year\b",
    r"\byoy\b",
)

# Deliberately empty. Ukrainian needs this family because "відсотки" is both a
# rate and a sum of money, so "сплачені відсотки за кредитами" has to be pulled
# out of the relative branch by hand. English says "percent" for the rate and
# "interest" for the money, and those two never collide.
#
# That is not the same as English being unambiguous. It has its own collision
# in the same place, and the classifier found it: "By how much did total assets
# GROW from Q1 2023 to Q4 2024?" asks for an absolute difference, while "By
# what percent did revenue GROW ..." asks for a rate, and only the opening
# phrase separates them. Hence "grow" is absent from the patterns below while
# "growth" and "grew" are present -- the bare verb carries no unit.
#
# So the disambiguation layer does not disappear between the languages, it
# moves: Ukrainian disambiguates a noun, English a verb frame.
INTEREST_PATTERNS = ()

# ---- scope gate (core.scope_gate) ----
#
# Measured in `eval.eval_lang`. Below 0.75 the gate lets the adversarial query
# through; no cutoff up to 0.95 produces a wrong refusal. 0.75 is the boundary
# itself, so the working value sits one step inside the window rather than on
# its edge. Ukrainian's usable window starts lower and ends earlier, which is
# why the two profiles do not share this number.
TOKEN_SIMILARITY = 0.80

OPERATION_WORDS = frozenset({
    "sum", "total", "overall", "aggregate", "combined",
    "averag", "mean", "median",
    "max", "maximum", "min", "minimum", "highest", "lowest", "largest",
    "smallest", "biggest", "peak",
    "differ", "difference", "compar", "exceed", "greater", "grow", "growth",
    "percent", "percentag", "margin", "ratio", "proportion", "share", "rate",
    "increas", "decreas", "chang", "trend", "dynamic",
    "count", "number", "quarter", "year", "period", "month",
    "metric", "valu", "amount", "volume", "figur", "item", "line",
})

# ---- table naming (load_data) ----

TABLE_NAME_RULES = (
    ("income", ("revenue", "net profit")),
    ("balance", ("total assets", "total equity")),
    ("cashflow", ("net cash from", "proceeds from sales")),
)

# ---- code generation prompts (core.pipeline) ----

CODEGEN_SYSTEM = (
    "You are an analyst who writes short Python/Pandas code. The tables are "
    "available under the names given in the schema below (the first table is "
    "also available as `df`). Always assign the final answer to a variable "
    "named `result`. Output ONLY code, with no explanation and no markdown "
    "fences."
)

SCHEMA_LINE = "- {name}: columns={cols}; first-column values (line items)={rows}"
PROMPT_HEAD = "Table schema:\n{schema}\n\nQuery: {query}\n"
PROMPT_FEEDBACK = "\nThe previous attempt had a problem: {feedback}\nTake this into account and write corrected code.\n"
PROMPT_TAIL = "Write the Pandas code."

RETRY_MARKER = "the previous attempt"

FEEDBACK_LOW_CONFIDENCE = (
    "The result {result!r} has low confidence (signals: {signals}). "
    "Check the formula and the units."
)

# ---- semantic verification signal (core.verifier) ----

SEMANTIC_PROMPT = (
    "User query: {query}\n"
    "Numeric answer obtained: {value}\n"
    "Does this answer look plausible as the result of this particular query "
    "(ignoring exact precision — only order of magnitude and common sense)? "
    "Reply with exactly one word: yes or no."
)

SEMANTIC_MARKER = "Does this answer look plausible"
SEMANTIC_VALUE_RE = r"Numeric answer obtained:\s*([-\d.]+)"

# Whole-word matching, not substrings: "no" is inside "not", "none",
# "nothing", "cannot" and "know", all of which are ordinary in a hedged reply.
YES_WORDS = ("yes",)
NO_WORDS = ("no",)

# ---- synthetic document (ingestion.synthetic_data, ingestion.make_sample_pdf) ----

INCOME_HEADER = "Line item"
BALANCE_HEADER = "Line item"
CASHFLOW_HEADER = "Cash flow item"

INCOME_LABELS = (
    "Revenue", "Cost of sales", "Gross profit",
    "Operating expenses", "Operating profit",
    "Finance costs", "Profit before tax",
    "Income tax", "Net profit",
    "Cash at end of quarter",
)

BALANCE_LABELS = (
    "Cash and cash equivalents", "Trade receivables", "Total assets",
    "Total liabilities", "Total equity",
)

CASHFLOW_LABELS = (
    "Proceeds from sales of goods",
    "Proceeds from rent",
    "Other operating receipts",
    "Payments to suppliers of materials",
    "Payments to suppliers of services",
    "Payments to employees",
    "Social security contributions",
    "Payments for office rent",
    "Marketing expenses",
    "Income tax paid",
    "Interest paid on loans",
    "Net cash from operating activities",
    "Purchase of property and equipment",
    "Purchase of intangible assets",
    "Proceeds from sale of equipment",
    "Loans granted to related parties",
    "Repayment of loans granted",
    "Dividends received",
    "Net cash from investing activities",
    "Proceeds from long-term borrowings",
    "Repayment of long-term borrowings",
    "Proceeds from short-term borrowings",
    "Repayment of short-term borrowings",
    "Dividends paid to shareholders",
    "Purchase of treasury shares",
    "Net cash from financing activities",
    "Net change in cash for the period",
    "Cash at end of period",
)

# English documents get their own file names. Without this an English run
# would regenerate — or worse, silently read — the frozen Ukrainian PDFs.
PDF_SUFFIX = "_en"

PDF_MULTIPAGE_TITLE = "Annual report (demonstration data) — page 1"
PDF_TITLE = "Statement of financial results (demonstration data)"
PDF_INCOME_HEADING = "Statement of financial results"
PDF_BALANCE_HEADING = "Balance sheet (extract)"
PDF_CASHFLOW_HEADING = "Statement of cash flows"
PDF_CASHFLOW_CONT_HEADING = "Statement of cash flows (continued)"

# ---- evaluation ----

TEST_SET_FULL = "test_queries_full_en.json"
TEST_SET_PHASE1 = "test_queries_phase1_en.json"

# The table with its VALUES, in the same pipe-delimited shape the direct-answer
# route gets. Measured (5.30.1): the schema-only prompt gives the code
# generator labels and no numbers, so it cannot see that a reserve is stored
# negative or that a cumulative-return column is indexed to 100 -- 28% of its
# wrong answers had the right magnitude and only the sign or scale wrong.
SCHEMA_VALUES_HEAD = "- {name}: columns={cols}"
SCHEMA_VALUES_ROW = "  {cells}"
SCHEMA_VALUES_TRUNCATED = "  ... ({n} more rows not shown)"

NO_RESULT = "The code did not define a variable `result`"
