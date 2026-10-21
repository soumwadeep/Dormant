"""Three levels of letting go, in order of how much regret is possible.

  archive()  moves files into a dated staging folder outside the scanned roots. Fully
             reversible from the index, and the default.
  trash()    moves files into the macOS Trash, which is what Finder would have done,
             so the user keeps the system's own "Put Back". Reversible from Dormant
             too, via the manifest.
  purge()    deletes for real. Requires the caller to pass the exact confirmation
             phrase, writes a manifest of what is about to go *before* it goes, and
             cannot be undone by anything in this program.

All three share the same refusals: a path that is not in the index, a protected system
prefix, a symlink, or anything already gone. Nothing ever overwrites a destination;
collisions produce file.dup1.ext. The point of the ladder is that the destructive rung
exists but is deliberately harder to reach than the recoverable ones.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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

#: The exact string a caller must produce for purge(). Chosen to be typed, not clicked.
DELETE_PHRASE = "DELETE"

TRASH_DIR = Path.home() / ".Trash"

#: Preview kinds, by extension. Anything not listed is offered as a download, not a view.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".svg"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".toml", ".log", ".rtf", ".html"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

#: Bytes returned for a text preview. More than this and the terminal is the wrong tool.
TEXT_PREVIEW_BYTES = 24_000


def preview_kind(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in TEXT_EXTS:
        return "text"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return "other"


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
    if rec.get("disposition") not in (None, "", "present"):
        raise UnsafeAction(f"already {rec['disposition']}: {path}")
    if rec.get("archived_at") and not rec.get("disposition"):
        raise UnsafeAction(f"already archived: {path}")
    if not os.path.exists(path):
        raise UnsafeAction(f"indexed but missing on disk: {path}")
    return rec


def _relocate(conn, file_ids: list[int], *, dest_root: Path, kind: str,
              max_bytes: int | None = None, reason: str = "model",
              manifest_id: str | None = None) -> dict:
    """Move files out of their original location, recording every move.

    Shared by archive and trash. Both are reversible from the manifest, and neither
    can overwrite a destination, because _stage_path names around collisions.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    PATHS.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_id = manifest_id or uuid.uuid4().hex[:12]

    moves: list[dict] = []
    skipped: list[dict] = []
    total = 0
    budget = max_bytes or 0
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
        dest = _stage_path(origin, dest_root, rec["name"])
        try:
            shutil.move(origin, str(dest))
        except OSError as exc:
            skipped.append({"file_id": fid, "why": f"move failed: {exc}"})
            continue
        if not dest.exists() or os.path.exists(origin):
            raise UnsafeAction(f"post-move invariant broken for {origin}")
        conn.execute("UPDATE files SET archived_at = ?, disposition = ? WHERE id = ?", (utcnow(), kind, fid))
        conn.execute(
            """INSERT INTO actions(manifest, file_id, origin, staged, size_bytes, created_at, kind)
               VALUES(?,?,?,?,?,?,?)""",
            (manifest_id, fid, origin, str(dest), size, utcnow(), kind),
        )
        moves.append({"file_id": fid, "origin": origin, "staged": str(dest), "size_bytes": size})
        total += size

    payload = {
        "manifest": manifest_id,
        "created_at": utcnow(),
        "kind": kind,
        "reason": reason,
        "count": len(moves),
        "bytes": total,
        "staged_root": str(dest_root),
        "moves": moves,
        "skipped": skipped,
        "deletes": 0,
        "reversible": True,
    }
    (PATHS.manifest_dir / f"{manifest_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.commit()
    return payload


def archive(conn, file_ids: list[int], *, max_bytes: int | None = None, reason: str = "model") -> dict:
    """Stage into Dormant's own dated folder. The default, and the safest."""
    root = PATHS.archive_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _relocate(conn, file_ids, dest_root=root, kind="staged", max_bytes=max_bytes, reason=reason)


def trash(conn, file_ids: list[int], *, max_bytes: int | None = None, reason: str = "model") -> dict:
    """Send to the macOS Trash, which is what Finder would have done.

    Going through ~/.Trash rather than unlinking means the system still offers Put
    Back, so the file has two independent recovery routes: the manifest, or the Dock.
    """
    return _relocate(conn, file_ids, dest_root=TRASH_DIR, kind="trashed", max_bytes=max_bytes, reason=reason)


def purge(conn, file_ids: list[int], *, confirm: str = "", max_bytes: int | None = None,
          reason: str = "model") -> dict:
    """Delete for real. The only function in this program that destroys data.

    Three things make it harder to reach than it should be, on purpose: the caller
    must supply the exact confirmation phrase, the manifest of what is about to be
    lost is written and committed *before* the first unlink, and the result says
    plainly that it cannot be undone.
    """
    if confirm != DELETE_PHRASE:
        raise UnsafeAction(
            f"permanent delete needs the confirmation phrase {DELETE_PHRASE!r}, got {confirm!r}")

    PATHS.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_id = uuid.uuid4().hex[:12]
    victims: list[dict] = []
    skipped: list[dict] = []
    total = 0
    for fid in file_ids:
        try:
            rec = validate(conn, fid)
        except UnsafeAction as exc:
            skipped.append({"file_id": fid, "why": str(exc)})
            continue
        if max_bytes and total + rec["size_bytes"] > (max_bytes or 0):
            skipped.append({"file_id": fid, "why": "over byte budget", "path": rec["path"]})
            continue
        victims.append({"file_id": fid, "origin": rec["path"], "size_bytes": rec["size_bytes"],
                        "name": rec["name"], "last_used_at": rec.get("last_used_at"),
                        "created_at": rec.get("created_at")})
        total += rec["size_bytes"]

    # Written first, so a crash mid-loop still leaves a record of what was lost.
    payload = {
        "manifest": manifest_id,
        "created_at": utcnow(),
        "kind": "deleted",
        "reason": reason,
        "count": len(victims),
        "bytes": total,
        "moves": victims,
        "skipped": skipped,
        "deletes": len(victims),
        "reversible": False,
        "warning": "these files are gone; the Trash does not hold them",
    }
    (PATHS.manifest_dir / f"{manifest_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.commit()

    removed, failed = [], []
    for v in victims:
        try:
            os.remove(v["origin"])
        except OSError as exc:
            failed.append({"origin": v["origin"], "why": str(exc)})
            continue
        if os.path.exists(v["origin"]):
            failed.append({"origin": v["origin"], "why": "still present after unlink"})
            continue
        conn.execute("UPDATE files SET archived_at = ?, disposition = 'deleted' WHERE id = ?",
                     (utcnow(), v["file_id"]))
        conn.execute(
            """INSERT INTO actions(manifest, file_id, origin, staged, size_bytes, created_at, kind)
               VALUES(?,?,?,?,?,?, 'deleted')""",
            (manifest_id, v["file_id"], v["origin"], "", v["size_bytes"], utcnow()),
        )
        removed.append(v)
    payload["removed"] = len(removed)
    payload["failed"] = failed
    payload["bytes_freed"] = sum(v["size_bytes"] for v in removed)
    (PATHS.manifest_dir / f"{manifest_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.commit()
    return payload


def preview_info(conn, file_id: int) -> dict:
    """Metadata for the preview drawer: what it is, how big, and whether it still exists."""
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    if row is None:
        raise UnsafeAction(f"file {file_id} is not in the index")
    rec = dict(row)
    path = rec["path"]
    exists = os.path.isfile(path) and not os.path.islink(path)
    size = os.path.getsize(path) if exists else rec["size_bytes"]
    return {
        "id": rec["id"], "name": rec["name"], "path": path, "ext": rec["ext"],
        "kind": preview_kind(rec["ext"]), "size_bytes": size, "exists": exists,
        "disposition": rec.get("disposition") or ("staged" if rec.get("archived_at") else "present"),
        "last_used_at": rec.get("last_used_at"), "created_at": rec.get("created_at"),
        "open_count": rec.get("open_count"),
    }


def quicklook(paths: list[str]) -> dict:
    """Hand files to macOS Quick Look. Native preview, no rendering code of our own."""
    live = [p for p in paths if os.path.exists(p)]
    if not live:
        return {"opened": 0, "reason": "no live paths"}
    try:
        subprocess.Popen(["qlmanage", "-p", *live], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except OSError as exc:
        return {"opened": 0, "reason": str(exc)[:120]}
    return {"opened": len(live), "skipped": len(paths) - len(live)}


def reveal(paths: list[str]) -> dict:
    """Show in Finder."""
    live = [p for p in paths if os.path.exists(p)]
    if not live:
        return {"revealed": 0}
    subprocess.Popen(["open", "-R", live[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"revealed": 1}


def latest_manifest(conn) -> str | None:
    row = conn.execute(
        "SELECT manifest FROM actions WHERE undone_at IS NULL AND kind != 'deleted' ORDER BY id DESC LIMIT 1").fetchone()
    return row["manifest"] if row else None


def undo(conn, manifest_id: str | None = None) -> dict:
    manifest_id = manifest_id or latest_manifest(conn)
    if not manifest_id:
        return {"restored": 0, "reason": "nothing to undo"}
    file = PATHS.manifest_dir / f"{manifest_id}.json"
    rows = conn.execute("SELECT * FROM actions WHERE manifest = ? ORDER BY id", (manifest_id,)).fetchall()
    if not rows:
        return {"manifest": manifest_id, "restored": 0, "reason": "no rows"}
    kinds = {r["kind"] for r in rows}
    if "deleted" in kinds:
        return {"manifest": manifest_id, "restored": 0,
                "reason": "a permanent delete cannot be undone; the manifest lists what was lost"}

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
        "SELECT COUNT(*) c, COALESCE(SUM(size_bytes),0) b FROM files WHERE disposition = 'staged'").fetchone()
    trashed = conn.execute(
        "SELECT COUNT(*) c, COALESCE(SUM(size_bytes),0) b FROM files WHERE disposition = 'trashed'").fetchone()
    gone = conn.execute(
        "SELECT COUNT(*) c, COALESCE(SUM(size_bytes),0) b FROM files WHERE disposition = 'deleted'").fetchone()
    pending = conn.execute(
        "SELECT COUNT(DISTINCT manifest) c FROM actions WHERE undone_at IS NULL AND kind != 'deleted'").fetchone()
    return {"archived_files": row["c"], "archived_bytes": row["b"],
            "trashed_files": trashed["c"], "trashed_bytes": trashed["b"],
            "deleted_files": gone["c"], "deleted_bytes": gone["b"],
            "reversible_manifests": pending["c"] if pending else 0}
