"""DataBench demo cases, chosen by stated rules from the real caches.

A demo that shows only what flatters the method is an advertisement. The rules
are here so the selection is checkable, and one case is deliberately a failure
of the retrieval stage.
"""

DEMO_MODEL = "gemini-flash-lite-latest"
DEMO_SOURCE = ("кеші A/B/B-rag/B-rag2 на dev100 і test150, "
               "офіційне порівняння відповідей організатора задачі")

# Each entry names the cache tag and the question id, so every screen can be
# traced back to a line in a file.
CASES = [
    {
        "key": "1",
        "tag": "dev100",
        "qid": None,                      # resolved by rule at load time
        "rule": "найбільша таблиця серед dev-питань, де пряма відповідь хибна, "
                "код правильний, І пошук сам знаходить таблицю на 1-му місці",
        "title": "713 тисяч рядків: пряма відповідь не бачить даних",
        "takeaway": "Пряма відповідь дала 0 — і це було правильно для тих "
                    "30 рядків, які їй показали. Умова виконується на 0 із 30 "
                    "рядків вікна і на 144 099 із 713 107 у таблиці. "
                    "Хибним було вікно, не міркування.",
    },
    {
        "key": "2",
        "tag": "test150",
        "qid": None,
        "rule": "перший випадок на тесті, де пошук промахнувся @1, "
                "спрацював тригер повторного пошуку, і наступна таблиця "
                "дала правильну відповідь",
        "title": "Повторний пошук за сигналом виконання",
        "takeaway": "Код на знайденій таблиці повернув порожнечу — це сигнал, "
                    "доступний ДО будь-якої звірки з еталоном. Система взяла "
                    "наступну таблицю з рангу й відповіла правильно. "
                    "На тесті так виправлено 10 питань, зламано 0.",
    },
    {
        "key": "3",
        "tag": "test150",
        "qid": None,
        "rule": "перший випадок на тесті, де пошук промахнувся @1 і жоден "
                "із трьох кандидатів не дав відповіді",
        "title": "ОБМЕЖЕННЯ: питання не ідентифікує свою таблицю",
        "takeaway": "Третина питань DataBench не містить нічого, що вказує на "
                    "таблицю («Provide the bottom 6 scores in the dataset»). "
                    "Тут повторний пошук безсилий, і це межа бенчмарку, "
                    "а не пошуку: recall@3 = 66% при стелі розрізнювальності 57–67%.",
    },
]
