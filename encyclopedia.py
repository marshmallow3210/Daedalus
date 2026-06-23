import sqlite3
import os
from datetime import date

DB_PATH = os.getenv("ENCYCLOPEDIA_DB_PATH", "/app/japanese_encyclopedia.db")

_WORD_NEW_COLUMNS = [
    ("part_of_speech",           "TEXT"),
    ("tags",                     "TEXT"),
    ("example_sentence",         "TEXT"),
    ("example_sentence_reading", "TEXT"),
    ("video_count",              "INTEGER NOT NULL DEFAULT 0"),
    ("emoji",                    "TEXT"),
    ("youtube_video_id",         "TEXT"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────

def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS japanese_encyclopedia (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                hiragana                 TEXT    NOT NULL UNIQUE,
                katakana                 TEXT,
                kanji                    TEXT,
                romaji                   TEXT,
                chinese_translation      TEXT,
                etymology                TEXT,
                jlpt_level               TEXT,
                part_of_speech           TEXT,
                tags                     TEXT,
                example_sentence         TEXT,
                example_sentence_reading TEXT,
                emoji                    TEXT,
                video_count              INTEGER NOT NULL DEFAULT 0,
                video_batch_id           TEXT,
                youtube_video_id         TEXT,
                created_at               TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at               TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_batches (
                batch_id         TEXT PRIMARY KEY,
                status           TEXT NOT NULL DEFAULT 'pending',
                word_count       INTEGER NOT NULL DEFAULT 0,
                youtube_video_id TEXT,
                local_path       TEXT,
                error_message    TEXT,
                retry_after      TEXT,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at     TEXT
            )
        """)
        conn.commit()
    migrate_db()


def migrate_db() -> None:
    """Add new columns to existing tables without wiping data."""
    with _connect() as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(japanese_encyclopedia)")}
        for col, defn in _WORD_NEW_COLUMNS:
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE japanese_encyclopedia ADD COLUMN {col} {defn}"
                )
        # Ensure video_batches exists (idempotent)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_batches (
                batch_id         TEXT PRIMARY KEY,
                status           TEXT NOT NULL DEFAULT 'pending',
                word_count       INTEGER NOT NULL DEFAULT 0,
                youtube_video_id TEXT,
                local_path       TEXT,
                error_message    TEXT,
                retry_after      TEXT,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at     TEXT
            )
        """)
        conn.commit()


# ──────────────────────────────────────────────────────────────
# Word CRUD
# ──────────────────────────────────────────────────────────────

def upsert_word(hiragana: str, **kwargs) -> None:
    allowed = {
        "katakana", "kanji", "romaji", "chinese_translation", "etymology",
        "jlpt_level", "part_of_speech", "tags", "example_sentence",
        "example_sentence_reading", "emoji", "video_batch_id", "youtube_video_id",
    }
    data = {k: v for k, v in kwargs.items() if k in allowed}
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM japanese_encyclopedia WHERE hiragana = ?", (hiragana,)
        ).fetchone()
        if row:
            if data:
                clause = ", ".join(f"{k} = ?" for k in data)
                conn.execute(
                    f"UPDATE japanese_encyclopedia "
                    f"SET {clause}, updated_at = datetime('now') WHERE hiragana = ?",
                    [*data.values(), hiragana],
                )
        else:
            cols = ["hiragana", *data.keys()]
            ph   = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO japanese_encyclopedia ({', '.join(cols)}) VALUES ({ph})",
                [hiragana, *data.values()],
            )
        conn.commit()


def search_words(query: str, limit: int = 20) -> list[dict]:
    p = f"%{query}%"
    with _connect() as conn:
        rows = conn.execute(
            """SELECT hiragana, katakana, kanji, romaji, chinese_translation,
                      jlpt_level, emoji, video_count, youtube_video_id
               FROM japanese_encyclopedia
               WHERE hiragana LIKE ? OR kanji LIKE ?
                  OR chinese_translation LIKE ? OR tags LIKE ?
               ORDER BY jlpt_level ASC LIMIT ?""",
            (p, p, p, p, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Video candidate queries
# ──────────────────────────────────────────────────────────────

def get_words_for_video(limit: int = 50, jlpt_level: str = None) -> list[dict]:
    """Return words that have not appeared in any video (video_count = 0)."""
    with _connect() as conn:
        base = """
            SELECT hiragana, katakana, kanji, romaji, chinese_translation,
                   etymology, jlpt_level, part_of_speech, tags,
                   example_sentence, example_sentence_reading, emoji
            FROM japanese_encyclopedia
            WHERE video_count = 0
        """
        if jlpt_level:
            rows = conn.execute(
                base + " AND jlpt_level = ? ORDER BY id ASC LIMIT ?",
                (jlpt_level, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                base + " ORDER BY jlpt_level ASC, id ASC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def mark_words_in_video(hiragana_list: list, batch_id: str, youtube_video_id: str = "") -> int:
    """Increment video_count and record batch/YouTube info. Returns updated count."""
    with _connect() as conn:
        total = 0
        for h in hiragana_list:
            conn.execute(
                """UPDATE japanese_encyclopedia
                   SET video_count      = video_count + 1,
                       video_batch_id   = ?,
                       youtube_video_id = COALESCE(NULLIF(?, ''), youtube_video_id),
                       updated_at       = datetime('now')
                   WHERE hiragana = ?""",
                (batch_id, youtube_video_id, h),
            )
            total += conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
    return total


# ──────────────────────────────────────────────────────────────
# video_batches management
# ──────────────────────────────────────────────────────────────

def create_batch(batch_id: str, word_count: int, local_path: str) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO video_batches (batch_id, status, word_count, local_path)
               VALUES (?, 'generating', ?, ?)""",
            (batch_id, word_count, local_path),
        )
        conn.commit()


def update_batch_status(batch_id: str, status: str, error: str = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE video_batches SET status = ?, error_message = ? WHERE batch_id = ?",
            (status, error, batch_id),
        )
        conn.commit()


def update_batch_youtube_id(batch_id: str, youtube_video_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            """UPDATE video_batches
               SET youtube_video_id = ?, status = 'completed',
                   completed_at = datetime('now')
               WHERE batch_id = ?""",
            (youtube_video_id, batch_id),
        )
        conn.commit()


def get_batch(batch_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM video_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
    return dict(row) if row else None


def clear_batch_local_path(batch_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE video_batches SET local_path = NULL WHERE batch_id = ?", (batch_id,)
        )
        conn.commit()


def get_today_upload_count() -> int:
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM video_batches
               WHERE status = 'completed'
               AND date(completed_at) = date('now')""",
        ).fetchone()
    return row[0] if row else 0
