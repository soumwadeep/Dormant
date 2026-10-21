"""Feature and label construction.

The naive version of this file asks "will this file be opened again?" at the moment
it is created. That is a different question (does this file matter at all?) and it
makes the dataset almost all-positive, because of course a fresh download gets read.

Dormant asks the question it can actually answer and act on: a file has already been
sitting shut for at least MIN_DORMANT_AGE_DAYS, and the observation is placed at a
sampled date inside that streak. The label then comes from what the index already
proves happened afterwards:

    1  some later open falls within W days of the observation date
    0  the next open falls more than W days after it, or there is no next open and
       the observation is at least W days in the past

Both cases are knowable from data macOS already recorded, which is what makes this a
real holdout rather than a vibe. Features only ever use opens up to the observation
date, never after it.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from .config import (
    MAX_TRAIN_ROWS,
    MIN_DORMANT_AGE_DAYS,
    PROJECT_MARKERS,
    PROJECT_SEGMENTS,
    REOPEN_WINDOW_DAYS,
)
from .store import parse_ts

#: Offsets, in days of dormancy, sampled inside each streak. Deliberately sparse:
#: a thousand near-duplicate rows per file would flatter every model equally.
STREAK_OFFSETS = (14, 21, 30, 45, 60, 90, 150, 240, 365)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".heic", ".heif", ".tiff", ".tif", ".webp", ".bmp", ".svg"}
DOC_EXTS = {".pdf", ".doc", ".docx", ".rtf", ".txt", ".md", ".markdown", ".pages", ".key",
            ".numbers", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".pptm", ".epub", ".tex", ".bib"}
MEDIA_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".dmg", ".iso", ".7z", ".rar"}
CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".swift", ".java",
             ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".php", ".sh", ".sql", ".json",
             ".yaml", ".yml", ".toml", ".ipynb"}

FEATURES = [
    "age_days",
    "days_since_last_open",
    "days_since_modified",
    "open_count",
    "gap_mean_days",
    "gap_max_days",
    "log_size",
    "dir_depth",
    "is_image",
    "is_document",
    "is_media",
    "is_archive",
    "is_code",
    "day_of_week",
    "is_weekend",
    "hour",
    "month",
    "in_downloads",
    "on_desktop",
    "in_pictures",
]


def _days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


@lru_cache(maxsize=200000)
def is_project_path(path: str) -> bool:
    """Fallback for ad-hoc paths. Scans label context while walking instead."""
    segments = set(path.split("/"))
    if segments & set(PROJECT_SEGMENTS):
        return True
    p = Path(path)
    for parent in [p.parent, *p.parents]:
        if parent.name in (".git", "node_modules") or parent.name in PROJECT_SEGMENTS:
            return True
        for marker in PROJECT_MARKERS:
            try:
                if marker in (".git", "node_modules"):
                    if os.path.isdir(parent / marker):
                        return True
                elif os.path.exists(parent / marker):
                    return True
            except OSError:
                continue
        if str(parent) in ("/", ""):
            break
    return False


def _ext_flags(ext: str) -> dict:
    return {
        "is_image": int(ext in IMAGE_EXTS),
        "is_document": int(ext in DOC_EXTS),
        "is_media": int(ext in MEDIA_EXTS),
        "is_archive": int(ext in ARCHIVE_EXTS),
        "is_code": int(ext in CODE_EXTS),
    }


@dataclass
class Row:
    file_id: int
    as_of: str
    label: int | None
    features: dict
    path: str
    size_bytes: int


def _features(file: dict, as_of: datetime, seen: list[datetime], last_open: datetime | None) -> dict:
    created = parse_ts(file["created_at"])
    modified = parse_ts(file["modified_at"])
    age = max(_days_between(created, as_of), 0.0)
    gaps = [_days_between(seen[i - 1], seen[i]) for i in range(1, len(seen))]
    low = f"/{file['path'].lower()}"
    return {
        "age_days": age,
        "days_since_last_open": age if last_open is None else max(_days_between(last_open, as_of), 0.0),
        "days_since_modified": max(_days_between(modified, as_of), 0.0) if modified else age,
        "open_count": float(len(seen)),
        "gap_mean_days": sum(gaps) / len(gaps) if gaps else 0.0,
        "gap_max_days": max(gaps) if gaps else 0.0,
        "log_size": math.log1p(float(file["size_bytes"])),
        "dir_depth": float(file["dir_depth"]),
        "day_of_week": as_of.weekday(),
        "is_weekend": int(as_of.weekday() >= 5),
        "hour": as_of.hour,
        "month": as_of.month,
        "in_downloads": int("/downloads/" in low),
        "on_desktop": int("/desktop/" in low),
        "in_pictures": int("/pictures/" in low),
        **_ext_flags(file["ext"]),
    }


def observations_for(created: datetime, opens: list[datetime], now: datetime,
                     horizon: int, min_dormant: int) -> list[tuple[datetime, int, int | None]]:
    """Sampled (as_of, opens_seen, index_of_next_open) triples that are labelable today."""
    out: list[tuple[datetime, int, int | None]] = []
    anchors: list[tuple[datetime, int, int | None]] = [(created, 0, 0 if opens else None)]
    for i, d in enumerate(opens):
        anchors.append((d, i + 1, i + 1 if i + 1 < len(opens) else None))
    latest = now - timedelta(days=horizon)
    for anchor, seen_count, next_idx in anchors:
        lo = anchor + timedelta(days=min_dormant)
        hi = latest
        next_open = opens[next_idx] if next_idx is not None else None
        if next_open is not None:
            hi = min(hi, next_open - timedelta(days=1))
        if hi < lo:
            continue
        points: list[datetime] = [lo, hi]
        for off in STREAK_OFFSETS:
            t = anchor + timedelta(days=off)
            if lo <= t <= hi:
                points.append(t)
        if next_open is not None:
            for back in (4, 11, 20, 27):
                t = next_open - timedelta(days=back)
                if lo <= t <= hi:
                    points.append(t)
        uniq: dict[str, datetime] = {}
        for t in points:
            uniq[t.strftime("%Y-%m-%dT%H")] = t
        for t in sorted(uniq.values()):
            out.append((t, seen_count, next_idx))
    return out


def build_rows(conn, horizon: int = REOPEN_WINDOW_DAYS, *, include_project: bool = False,
               archived=None, max_rows: int = MAX_TRAIN_ROWS,
               min_dormant: int = MIN_DORMANT_AGE_DAYS) -> list[Row]:
    """Every labelable observation the index currently supports."""
    now = datetime.now(timezone.utc)
    sql = ("SELECT id, path, ext, size_bytes, created_at, modified_at, dir_depth, context "
           "FROM files WHERE 1=1")
    args: list = []
    if archived is False:
        sql += " AND archived_at IS NULL"
    elif archived is True:
        sql += " AND archived_at IS NOT NULL"
    rows: list[Row] = []
    for file in conn.execute(sql, args).fetchall():
        f = dict(file)
        if not include_project and f.get("context") == "project":
            continue
        created = parse_ts(f["created_at"])
        if created is None:
            continue
        raw = conn.execute("SELECT opened_at FROM opens WHERE file_id = ? ORDER BY opened_at",
                           (f["id"],)).fetchall()
        opens = sorted({parse_ts(r["opened_at"]) for r in raw if parse_ts(r["opened_at"])})
        for as_of, seen_count, next_idx in observations_for(created, opens, now, horizon, min_dormant):
            seen = opens[:seen_count]
            last_open = seen[-1] if seen else None
            next_open = opens[next_idx] if next_idx is not None else None
            if next_open is not None and _days_between(as_of, next_open) <= horizon:
                label = 1
            else:
                label = 0
            rows.append(Row(file_id=f["id"], as_of=as_of.isoformat(timespec="seconds"), label=label,
                            features=_features(f, as_of, seen, last_open),
                            path=f["path"], size_bytes=f["size_bytes"]))
    if len(rows) > max_rows:
        # Deterministic stride keeps the whole time span instead of the newest cluster.
        step = math.ceil(len(rows) / max_rows)
        rows = rows[::step]
    return rows


def scoring_rows(conn, horizon: int = REOPEN_WINDOW_DAYS, *, include_project: bool = False,
                 min_dormant: int = MIN_DORMANT_AGE_DAYS) -> list[Row]:
    """One row per live file, observed at now(). Only files already stale enough to ask about."""
    now = datetime.now(timezone.utc)
    sql = ("SELECT id, path, ext, size_bytes, created_at, modified_at, dir_depth, context "
           "FROM files WHERE archived_at IS NULL")
    out: list[Row] = []
    watching = 0
    for file in conn.execute(sql).fetchall():
        f = dict(file)
        if not include_project and f.get("context") == "project":
            continue
        created = parse_ts(f["created_at"])
        if created is None:
            continue
        raw = conn.execute("SELECT opened_at FROM opens WHERE file_id = ? ORDER BY opened_at",
                           (f["id"],)).fetchall()
        opens = sorted({parse_ts(r["opened_at"]) for r in raw if parse_ts(r["opened_at"])})
        last_open = opens[-1] if opens else None
        feats = _features(f, now, opens, last_open)
        if feats["days_since_last_open"] < min_dormant:
            watching += 1
            continue
        out.append(Row(file_id=f["id"], as_of=now.isoformat(timespec="seconds"), label=None,
                       features=feats, path=f["path"], size_bytes=f["size_bytes"]))
    scoring_rows.watching = watching  # type: ignore[attr-defined]
    return out


def rule_score(row: Row, dormant_after_days: int) -> float:
    """The no-model baseline a person would write by hand: dormant if untouched for N days."""
    recency = row.features["days_since_last_open"]
    return 1.0 if recency >= dormant_after_days else 0.0
