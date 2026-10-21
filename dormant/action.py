"""Archiving that can be undone, and is not allowed to delete.

Dormant moves files into a staging folder outside the scanned roots, keeps a manifest
of every move, and can reverse any manifest in full. It never overwrites, never
follows a symlink, never touches a path that is not already in the index, and refuses
anything that would land outside the staging tree. The worst case it can produce is a
folder you can drag back.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import PATHS
from .store import utcnow

#: System locations Dormant will never move anything out of. Deliberately not a bare
#: "/private": that would also ban /private/var/folders, where your own documents,
#: Mail attachments and every app's temp working copy actually live.
PROTECTED_PREFIXES = ("/System", "/usr", "/bin", "/sbin", "/Library", "/Applications",
                      "/private/etc", "/private/var/db", "/private/var/root", "/private/var/log")


class UnsafeAction(Exception):
    pass


def _stage_path(origin: str, staged_root: Path, name: str) -> Path:
    """Collision-safe destination: never overwrite, ever."""
    target = staged_root / name
    if not target.exists():
        return target
    if "." in name:
        stem, dot, ext = name.partition(".")
    else:
        stem, dot, ext = name, "", ""
    for i in range(1, 1000):
        candidate = staged_root / f"{stem}.dup{i}{dot}{ext}"
        if not candidate.exists():
            return candidate
    raise UnsafeAction(f"could not name a free destination for {name}")


def validate(conn, file_id: int) -> dict:
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    if row is None:
        raise UnsafeAction(f"file {file_id} is not in the index")
    rec = dict(row)
    path = rec["path"]
    if any(path.startswith(p) for p in PROTECTED_PREFIXES):
        raise UnsafeAction(f"refusing to touch a protected path: {path}")
    if os.path.islink(path):
        raise UnsafeAction(f"refusing to move a symlink: {path}")
    if rec.get("archived_at"):
        raise UnsafeAction(f"already archived: {path}")
    if not os.path.exists(path):
        raise UnsafeAction(f"indexed but missing on disk: {path}")
    return rec


def archive(conn, file_ids: list[int], *, max_bytes: int | None = None, reason: str = "model") -> dict:
    staged_root = PATHS.archive_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    staged_root.mkdir(parents=True, exist_ok=True)
    PATHS.manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_id = uuid.uuid4().hex[:12]
    moves: list[dict] = []
    skipped: list[dict] = []
    budget = max_bytes or 0
    total = 0
    for fid in file_ids:
        try:
            rec = validate(conn, fid)
        except UnsafeAction as exc:
            skipped.append({"file_id": fid, "why": str(exc)})
            continue
        size = rec["size_bytes"]
        if max_bytes and total + size > budget:
            skipped.append({"file_id": fid, "why": "over byte budget", "path": rec["path"]})
            continue
        origin = rec["path"]
        dest = _stage_path(origin, staged_root, rec["name"])
        try:
            shutil.move(origin, str(dest))
        except OSError as exc:
            skipped.append({"file_id": fid, "why": f"move failed: {exc}"})
            continue
        if not dest.exists() or os.path.exists(origin):
            raise UnsafeAction(f"post-move invariant broken for {origin}")
        conn.execute("UPDATE files SET archived_at = ? WHERE id = ?", (utcnow(), fid))
        conn.execute(
            """INSERT INTO actions(manifest, file_id, origin, staged, size_bytes, created_at)
               VALUES(?,?,?,?,?,?)""",
            (manifest_id, fid, origin, str(dest), size, utcnow()),
        )
        moves.append({"file_id": fid, "origin": origin, "staged": str(dest), "size_bytes": size})
        total += size

    payload = {
        "manifest": manifest_id,
        "created_at": utcnow(),
        "reason": reason,
        "count": len(moves),
        "bytes": total,
        "staged_root": str(staged_root),
        "moves": moves,
        "skipped": skipped,
        "deletes": 0,
    }
    path = PATHS.manifest_dir / f"{manifest_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.commit()
    return payload


def latest_manifest(conn) -> str | None:
    row = conn.execute("SELECT manifest FROM actions WHERE undone_at IS NULL ORDER BY id DESC LIMIT 1").fetchone()
    return row["manifest"] if row else None


def undo(conn, manifest_id: str | None = None) -> dict:
    manifest_id = manifest_id or latest_manifest(conn)
    if not manifest_id:
        return {"restored": 0, "reason": "nothing to undo"}
    file = PATHS.manifest_dir / f"{manifest_id}.json"
    rows = conn.execute("SELECT * FROM actions WHERE manifest = ? ORDER BY id", (manifest_id,)).fetchall()
    if not rows:
        return {"manifest": manifest_id, "restored": 0, "reason": "no rows"}

    restored, failed = [], []
    for r in rows:
        origin, staged, size = r["origin"], r["staged"], r["size_bytes"]
        if r["undone_at"]:
            continue
        if not Path(staged).exists():
            failed.append({"origin": origin, "why": "staged copy missing"})
            continue
        if Path(origin).exists():
            failed.append({"origin": origin, "why": "origin already occupied"})
            continue
        try:
            Path(origin).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(staged, origin)
        except OSError as exc:
            failed.append({"origin": origin, "why": str(exc)})
            continue
        conn.execute("UPDATE files SET archived_at = NULL WHERE id = ?", (r["file_id"],))
        conn.execute("UPDATE actions SET undone_at = ?, undo_complete = 1 WHERE id = ?", (utcnow(), r["id"]))
        restored.append({"origin": origin, "size_bytes": size})

    complete = not failed
    if file.exists() and complete:
        data = json.loads(file.read_text(encoding="utf-8"))
        data["undone_at"] = utcnow()
        data["restored"] = len(restored)
        file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    conn.commit()
    return {"manifest": manifest_id, "restored": len(restored),
            "bytes": sum(x["size_bytes"] for x in restored), "failed": failed, "complete": complete}


def archive_state(conn) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) c, COALESCE(SUM(size_bytes),0) b FROM files WHERE archived_at IS NOT NULL").fetchone()
    pending = conn.execute("SELECT COUNT(DISTINCT manifest) c FROM actions WHERE undone_at IS NULL").fetchone()
    return {"archived_files": row["c"], "archived_bytes": row["b"],
            "reversible_manifests": pending["c"] if pending else 0}
