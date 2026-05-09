# code/registry/db.py
"""
SQLite реестр документов.
Файл БД: /app/data/registry.db
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import date

logger = logging.getLogger(__name__)

DB_PATH = Path("/app/data/registry.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                category        TEXT NOT NULL DEFAULT 'Прочее',
                department      TEXT NOT NULL DEFAULT 'ВолгоградНИПИнефть',
                filename        TEXT,
                html_path       TEXT,
                file_url        TEXT,
                milvus_doc_id   TEXT,
                version         INTEGER NOT NULL DEFAULT 1,
                valid_until     DATE,
                revision        TEXT,
                revised_at      DATE,
                index_status    TEXT NOT NULL DEFAULT 'queued',
                status          TEXT NOT NULL DEFAULT 'active',
                added_at        DATE NOT NULL DEFAULT (date('now')),
                author          TEXT,
                error_msg       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_name
                ON documents(name);
            CREATE INDEX IF NOT EXISTS idx_documents_status
                ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_index_status
                ON documents(index_status);
            CREATE INDEX IF NOT EXISTS idx_documents_valid_until
                ON documents(valid_until);
        """)
        # Миграции для старых БД
        for col in ["file_url TEXT", "revision TEXT", "revised_at DATE"]:
            try:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {col}")
            except Exception:
                pass
    logger.info(f"БД инициализирована: {DB_PATH}")


def get_all(
    status: Optional[str] = None,
    index_status: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Dict]:
    sql = "SELECT * FROM documents WHERE 1=1"
    params = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if index_status:
        sql += " AND index_status = ?"
        params.append(index_status)
    if category:
        sql += " AND category = ?"
        params.append(category)
    if department:
        sql += " AND department = ?"
        params.append(department)
    if search:
        sql += " AND LOWER(name) LIKE LOWER(?)"
        params.append(f"%{search}%")
    sql += " ORDER BY added_at DESC, id DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_by_id(doc_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def get_by_name(name: str) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE name = ? AND status = 'active' LIMIT 1",
            (name,)
        ).fetchone()
    return dict(row) if row else None


def create(
    name: str,
    category: str,
    department: str,
    filename: Optional[str],
    author: Optional[str],
    valid_until: Optional[str] = None,
    file_url: Optional[str] = None,
    revision: Optional[str] = None,
    revised_at: Optional[str] = None,
    version: int = 1,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO documents
               (name, category, department, filename, author, valid_until,
                version, index_status, status, file_url, revision, revised_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'active', ?, ?, ?)""",
            (name, category, department, filename, author, valid_until,
             version, file_url, revision, revised_at),
        )
        return cur.lastrowid


def update_fields(doc_id: int, **kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    params = list(kwargs.values()) + [doc_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE documents SET {sets} WHERE id = ?", params)


def archive_document(doc_id: int):
    update_fields(doc_id, status="archived")
    logger.info(f"Документ #{doc_id} архивирован")


def get_expired() -> List[Dict]:
    today = date.today().isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM documents
               WHERE status = 'active'
                 AND valid_until IS NOT NULL
                 AND valid_until <= ?""",
            (today,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete(doc_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    logger.info(f"Документ #{doc_id} удалён из реестра")