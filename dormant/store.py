"""SQLite schema and access. Local file only. Nothing here talks to a network."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import PATHS, VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    roots         TEXT NOT NULL,
    files_seen    INTEGER NOT NULL DEFAULT 0,
    files_stored  INTEGER NOT NULL DEFAULT 0,
    with_history  INTEGER NOT NULL DEFAULT 0,
    bytes_scanned INTEGER NOT NULL DEFAULT 0,
    seconds       REAL,
    mode          TEXT NOT NULL DEFAULT 'real'
);

CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    volume        TEXT NOT NULL,
    root          TEXT NOT NULL,
    name          TEXT NOT NULL,
    ext           TEXT NOT NULL,
    dir_depth     INTEGER NOT NULL,
    size_bytes    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    modified_at   TEXT NOT NULL,
    added_at      TEXT,
    last_used_at  TEXT,
    first_used_at TEXT,
    open_count    INTEGER NOT NULL DEFAULT 0,
    used_dates    TEXT NOT NULL DEFAULT '[]',
    has_history   INTEGER NOT NULL DEFAULT 0,
    context       TEXT NOT NULL DEFAULT 'personal',
    archived_at   TEXT,
    scanned_run   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_files_last_used ON files(last_used_at);
CREATE INDEX IF NOT EXISTS idx_files_size      ON files(size_bytes);
CREATE INDEX IF NOT EXISTS idx_files_archived  ON files(archived_at);

-- One row per recorded open event. This is the event log the labels come from.
CREATE TABLE IF NOT EXISTS opens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    opened_at  TEXT NOT NULL,
    ordinal    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_opens_file ON opens(file_id);
CREATE INDEX IF NOT EXISTS idx_opens_time ON opens(opened_at);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ts          TEXT NOT NULL,
    p_reopen    REAL NOT NULL,
    rank        INTEGER,
    model       TEXT NOT NULL,
    horizon     INTEGER NOT NULL,
    holdout     INTEGER NOT NULL DEFAULT 0,
    label       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);

CREATE TABLE IF NOT EXISTS snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at     TEXT NOT NULL,
    volume       TEXT NOT NULL,
    total_bytes  INTEGER NOT NULL,
    free_bytes   INTEGER NOT NULL,
    used_bytes   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest      TEXT NOT NULL,
    file_id       INTEGER,
    origin        TEXT NOT NULL,
    staged        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    undone_at     TEXT,
    undo_complete INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bench (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    model     TEXT NOT NULL,
    n_train   INTEGER NOT NULL,
    n_test    INTEGER NOT NULL,
    auc       REAL,
    pr_auc    REAL,
    precision REAL,
    recall    REAL,
    reclaimed REAL,
    fit_s     REAL,
    predict_s REAL,
    peak_mb   REAL,
    notes     TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """One SQLite file, several connections: the server, the CLI and a nightly launch
    agent can all legitimately be open at once. WAL plus a busy timeout means a
    short-lived writer waits instead of throwing at the reader."""
    target = db_path or PATHS.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "context" not in columns:
        conn.execute("ALTER TABLE files ADD COLUMN context TEXT NOT NULL DEFAULT 'personal'")
    set_meta(conn, "schema_version", "1")
    set_meta(conn, "app_version", VERSION)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def upsert_file(conn: sqlite3.Connection, rec: dict) -> int | None:
    cur = conn.execute("SELECT id FROM files WHERE path = ?", (rec["path"],))
    row = cur.fetchone()
    used = json.dumps(rec.get("used_dates", []))
    context = rec.get("context", "personal")
    if row:
        conn.execute(
            """UPDATE files SET volume=?, root=?, name=?, ext=?, dir_depth=?, size_bytes=?,
               created_at=?, modified_at=?, added_at=?, last_used_at=?, first_used_at=?,
               open_count=?, used_dates=?, has_history=?, context=?, scanned_run=? WHERE id=?""",
            (rec["volume"], rec["root"], rec["name"], rec["ext"], rec["dir_depth"], rec["size_bytes"],
             rec["created_at"], rec["modified_at"], rec.get("added_at"), rec.get("last_used_at"),
             rec.get("first_used_at"), rec.get("open_count", 0), used, rec.get("has_history", 0),
             context, rec.get("scanned_run"), row["id"]),
        )
        return int(row["id"])
    conn.execute(
        """INSERT INTO files(path, volume, root, name, ext, dir_depth, size_bytes, created_at,
           modified_at, added_at, last_used_at, first_used_at, open_count, used_dates, has_history, context, scanned_run)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rec["path"], rec["volume"], rec["root"], rec["name"], rec["ext"], rec["dir_depth"],
         rec["size_bytes"], rec["created_at"], rec["modified_at"], rec.get("added_at"),
         rec.get("last_used_at"), rec.get("first_used_at"), rec.get("open_count", 0), used,
         rec.get("has_history", 0), context, rec.get("scanned_run")),
    )
    cur = conn.execute("SELECT id FROM files WHERE path = ?", (rec["path"],))
    return int(cur.fetchone()["id"])


def replace_opens(conn: sqlite3.Connection, file_id: int, dates: Iterable[str]) -> int:
    conn.execute("DELETE FROM opens WHERE file_id = ?", (file_id,))
    ordered = sorted(set(dates))
    conn.executemany(
        "INSERT INTO opens(file_id, opened_at, ordinal) VALUES(?,?,?)",
        [(file_id, d, i + 1) for i, d in enumerate(ordered)],
    )
    return len(ordered)


def clear_predictions(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM predictions")


def record_snapshot(conn: sqlite3.Connection, volume: str, total: int, free: int) -> None:
    conn.execute(
        "INSERT INTO snapshots(taken_at, volume, total_bytes, free_bytes, used_bytes) VALUES(?,?,?,?,?)",
        (utcnow(), volume, total, free, total - free),
    )


def file_count(conn: sqlite3.Connection, archived: bool | None = None) -> int:
    if archived is None:
        return int(conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"])
    if archived:
        return int(conn.execute("SELECT COUNT(*) c FROM files WHERE archived_at IS NOT NULL").fetchone()["c"])
    return int(conn.execute("SELECT COUNT(*) c FROM files WHERE archived_at IS NULL").fetchone()["c"])
