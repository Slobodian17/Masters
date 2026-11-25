"""Ukrainian profile.

Every constant here was measured or tuned against the Ukrainian test sets
during phases 0-2 and is reproduced verbatim; the frozen MOCK baseline depends
on it byte for byte.
"""

CODE = "uk"
NAME = "українська"

# ---- tokenisation (core.text_utils) ----
#
# Longer suffixes first. Kept as measured in phases 0-2; the frozen baseline is
# reproducible only with this exact list.
#
# Worth recording that `eval.eval_lang` does NOT show this list earning its
# keep on retrieval: structural recall@1 is 78% both with it and with no
# suffixes at all. What changes is which five queries miss, not how many. The
# stemmer is therefore justified here by reproducibility, not by a measured
# gain -- and the intuition that an inflected language cannot retrieve without
# one is, on this test set, unsupported.
SUFFIXES = (
    "ями", "ами", "ові", "еві", "ією", "ями", "ах", "ях", "ам", "ям", "ою", "ею",
    "ів", "ей", "ий", "ій", "их", "ім", "ом", "ем", "ів", "ти", "ся",
    "а", "я", "у", "ю", "е", "и", "і", "о",
)

# Minimum length of what remains after stripping a suffix.
STEM_MIN_KEEP = 4

# One pass, because that is what the frozen baseline was measured with.
# Iterating to a fixed point was expected to eat real roots; measured, it
# changes nothing at all -- same recall, same misses, same form pairs. The
# reason to keep one pass is byte-identical regression, not the expectation.
STEM_ITERATE = False

STOPWORDS = frozenset({
    "яка", "який", "яке", "які", "що", "за", "на", "у", "в", "з", "із", "до", "по",
    "від", "для", "та", "і", "й", "а", "але", "чи", "скільки", "наскільки", "як",
    "усіх", "всіх", "усі", "всі", "рік", "року", "роки", "було", "є", "більше",
    "менше", "між", "цей", "ця", "це",
})

MIN_TOKEN = 3

# ---- relative vs absolute queries (core.query_semantics) ----
#
# "відсотки" is genuinely ambiguous: a rate in one context, a sum of money in
# another. Prefix matching on "відсот" cannot tell them apart, hence the
# explicit pattern families. English has two separate words and needs none of
# this, which is the clearest single instance of language specificity in the
# whole pipeline.
RELATIVE_PATTERNS = (
    r"%",
    r"\bу\s+відсотк",
    r"\bв\s+відсотк",
    r"на\s+скільки\s+відсотк",
    r"відсотк\w*\s+(?:зрост|приріст|зменш|падінн)",
    r"маржа",
    r"маржинальн",
    r"рентабельн",
    r"частк",
    r"приріст",
    r"зростанн",
    r"темп\w*\s+зрост",
)

# Contexts where "відсотки" means loan interest. Checked first, takes priority.
INTEREST_PATTERNS = (
    r"відсотк\w*\s+за\s+(?:кредит|позик|депозит|облігац)",
    r"(?:сплачен|нарахован|отриман)\w*\s+відсотк",
    r"відсотков\w*\s+(?:витрат|дохід|ставк)",
)

# ---- scope gate (core.scope_gate) ----
#
# Measured in `eval.eval_lang`: usable from 0.60 up to 0.90, with two wrong
# refusals appearing at 0.95. English needs a higher floor, so the two
# profiles keep separate values.
TOKEN_SIMILARITY = 0.75

OPERATION_WORDS = frozenset({
    "сум", "сумарн", "сукупн", "загальн", "всьог", "разом",
    "середн", "середнь", "максимальн", "мінімальн", "найбільш", "найменш",
    "різниц", "порівн", "більш", "менш", "перевищ",
    "відсотк", "відсот", "процент", "маржа", "частк", "рентабельн",
    "приріст", "зрост", "зросл", "динамік", "темп",
    "кільк", "квартал", "рік", "рок", "період", "місяц",
    "показник", "значенн", "величин", "обсяг", "стаття",
})

# ---- table naming (load_data) ----

TABLE_NAME_RULES = (
    ("income", ("виручка", "чистий прибуток")),
    ("balance", ("всього активів", "власний капітал")),
    ("cashflow", ("рух коштів", "надходження від реалізації")),
)

# ---- code generation prompts (core.pipeline) ----

CODEGEN_SYSTEM = (
    "Ти — аналітик, що пише короткий Python/Pandas код. Таблиці доступні "
    "під іменами, вказаними у схемі нижче (перша таблиця також доступна як "
    "`df`). Завжди записуй фінальну відповідь у змінну `result`. "
    "Виводь ЛИШЕ код, без пояснень і без markdown-огорож."
)

SCHEMA_LINE = "- {name}: стовпці={cols}; значення першого стовпця (показники)={rows}"
PROMPT_HEAD = "Схема таблиць:\n{schema}\n\nЗапит: {query}\n"
PROMPT_FEEDBACK = "\nПопередня спроба мала проблему: {feedback}\nВрахуй це і напиши виправлений код.\n"
PROMPT_TAIL = "Напиши Pandas-код."

# Substring that tells MOCK a prompt is a repair attempt, matched lowercased.
RETRY_MARKER = "попередня спроба"

FEEDBACK_LOW_CONFIDENCE = (
    "Результат {result!r} має низьку довіру (сигнали: {signals}). "
    "Перевір формулу й одиниці виміру."
)

# ---- semantic verification signal (core.verifier) ----

SEMANTIC_PROMPT = (
    "Запит користувача: {query}\n"
    "Отримана числова відповідь: {value}\n"
    "Чи ця відповідь виглядає правдоподібно як результат саме такого запиту "
    "(без урахування точності — лише порядок величини й здоровий глузд)? "
    "Відповідай рівно одним словом: так або ні."
)

# Marker MOCK uses to recognise the semantic question.
SEMANTIC_MARKER = "Чи ця відповідь виглядає правдоподібно"
SEMANTIC_VALUE_RE = r"Отримана числова відповідь:\s*([-\d.]+)"

YES_WORDS = ("так",)
NO_WORDS = ("ні",)

# ---- synthetic document (ingestion.synthetic_data, ingestion.make_sample_pdf) ----

INCOME_HEADER = "Показник"
BALANCE_HEADER = "Показник"
CASHFLOW_HEADER = "Стаття руху коштів"

INCOME_LABELS = (
    "Виручка", "Собівартість реалізації", "Валовий прибуток",
    "Операційні витрати", "Прибуток від операційної діяльності",
    "Фінансові витрати", "Прибуток до оподаткування",
    "Податок на прибуток", "Чистий прибуток",
    "Грошові кошти на кінець кварталу",
)

BALANCE_LABELS = (
    "Грошові кошти", "Дебіторська заборгованість", "Всього активів",
    "Всього зобов'язань", "Власний капітал",
)

CASHFLOW_LABELS = (
    "Надходження від реалізації продукції",
    "Надходження від оренди",
    "Інші операційні надходження",
    "Оплата постачальникам сировини",
    "Оплата постачальникам послуг",
    "Виплати персоналу",
    "Відрахування на соціальні заходи",
    "Оплата оренди приміщень",
    "Витрати на маркетинг",
    "Сплата податку на прибуток",
    "Сплачені відсотки за кредитами",
    "Чистий рух коштів від операційної діяльності",
    "Придбання основних засобів",
    "Придбання нематеріальних активів",
    "Надходження від продажу обладнання",
    "Надання позик пов'язаним особам",
    "Повернення раніше наданих позик",
    "Отримані дивіденди",
    "Чистий рух коштів від інвестиційної діяльності",
    "Надходження від довгострокових кредитів",
    "Погашення довгострокових кредитів",
    "Надходження від короткострокових кредитів",
    "Погашення короткострокових кредитів",
    "Виплачені дивіденди акціонерам",
    "Витрати на викуп власних акцій",
    "Чистий рух коштів від фінансової діяльності",
    "Чистий рух грошових коштів за період",
    "Залишок коштів на кінець періоду",
)

# Ukrainian keeps the bare file names: the frozen phase-0 PDFs carry them.
PDF_SUFFIX = ""

PDF_MULTIPAGE_TITLE = "Річний звіт (демонстраційні дані) — сторінка 1"
PDF_TITLE = "Звіт про фінансові результати (демонстраційні дані)"
PDF_INCOME_HEADING = "Звіт про фінансові результати"
PDF_BALANCE_HEADING = "Баланс (витяг)"
PDF_CASHFLOW_HEADING = "Звіт про рух грошових коштів"
PDF_CASHFLOW_CONT_HEADING = "Звіт про рух грошових коштів (продовження)"

# ---- evaluation ----

TEST_SET_FULL = "test_queries_full.json"
TEST_SET_PHASE1 = "test_queries_phase1.json"

# Таблиця РАЗОМ ІЗ ЗНАЧЕННЯМИ, у тому самому вигляді з вертикальною рискою,
# який отримує маршрут прямої відповіді. Зміряно (5.30.1): промпт лише зі
# схемою дає генераторові мітки без чисел, тож він не бачить, що резерв
# збережено від'ємним — 28% його хибних відповідей мали правильну величину
# і хибний лише знак або масштаб.
SCHEMA_VALUES_HEAD = "- {name}: стовпці={cols}"
SCHEMA_VALUES_ROW = "  {cells}"
SCHEMA_VALUES_TRUNCATED = "  ... (ще {n} рядків не показано)"

NO_RESULT = "Код не визначив змінну `result`"
