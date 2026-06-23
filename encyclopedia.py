import sqlite3
import os
from datetime import date, timedelta

DB_PATH = os.getenv("ENCYCLOPEDIA_DB_PATH", "/app/japanese_encyclopedia.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS japanese_encyclopedia (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                hiragana            TEXT    NOT NULL UNIQUE,
                katakana            TEXT,
                kanji               TEXT,
                romaji              TEXT,
                chinese_translation TEXT,
                etymology           TEXT,
                jlpt_level          TEXT,
                video_batch_id      TEXT,
                review_count        INTEGER NOT NULL DEFAULT 0,
                next_review_date    TEXT    NOT NULL DEFAULT (date('now')),
                ease_factor         REAL    NOT NULL DEFAULT 2.5,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def upsert_word(hiragana: str, **kwargs) -> None:
    allowed = {
        "katakana", "kanji", "romaji", "chinese_translation",
        "etymology", "jlpt_level", "video_batch_id",
    }
    data = {k: v for k, v in kwargs.items() if k in allowed}

    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM japanese_encyclopedia WHERE hiragana = ?", (hiragana,)
        ).fetchone()

        if row:
            if data:
                set_clause = ", ".join(f"{k} = ?" for k in data)
                values = list(data.values()) + [hiragana]
                conn.execute(
                    f"UPDATE japanese_encyclopedia SET {set_clause}, updated_at = datetime('now') "
                    f"WHERE hiragana = ?",
                    values,
                )
        else:
            cols = ["hiragana"] + list(data.keys())
            placeholders = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO japanese_encyclopedia ({', '.join(cols)}) VALUES ({placeholders})",
                [hiragana] + list(data.values()),
            )
        conn.commit()


def update_srs(hiragana: str, quality: int) -> bool:
    """SM-2 spaced repetition algorithm. quality: 0 (blackout) to 5 (perfect)."""
    if not 0 <= quality <= 5:
        raise ValueError("quality must be between 0 and 5")

    with _connect() as conn:
        row = conn.execute(
            "SELECT ease_factor, review_count FROM japanese_encyclopedia WHERE hiragana = ?",
            (hiragana,),
        ).fetchone()

        if not row:
            return False

        ef = row["ease_factor"]
        n = row["review_count"]

        if quality >= 3:
            if n == 0:
                interval = 1
            elif n == 1:
                interval = 6
            else:
                interval = max(1, round(6 * (ef ** (n - 1))))
            ef = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            n += 1
        else:
            interval = 1
            n = 0

        next_date = (date.today() + timedelta(days=interval)).isoformat()

        conn.execute(
            """UPDATE japanese_encyclopedia
               SET ease_factor = ?, review_count = ?, next_review_date = ?,
                   updated_at = datetime('now')
               WHERE hiragana = ?""",
            (ef, n, next_date, hiragana),
        )
        conn.commit()
    return True


def get_due_words(limit: int = 20) -> list[dict]:
    today = date.today().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT hiragana, katakana, kanji, romaji, chinese_translation,
                      jlpt_level, ease_factor, review_count, next_review_date
               FROM japanese_encyclopedia
               WHERE next_review_date <= ?
               ORDER BY next_review_date ASC
               LIMIT ?""",
            (today, limit),
        ).fetchall()
    return [dict(row) for row in rows]
