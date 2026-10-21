"""Ingestion: read what macOS already knows about your files, and nothing else.

Two passes, deliberately:

  inventory()  os.walk + os.stat. Cheap, complete, covers files Spotlight never
               indexed. This is the universe Dormant can act on.
  history()    mdls per openable file, run through a thread pool because each call
               is really a round trip to the Spotlight daemon, not disk work.
               This is where kMDItemUsedDates comes from: macOS already recorded
               every time a document app opened the file. Dormant does not have to
               watch for weeks to get labels, it just has to read them.

No file contents are ever opened, hashed, copied or uploaded. Paths, sizes and
timestamps are the entire feature space, which is what makes the offline claim
true rather than cosmetic.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    MAX_SCAN_FILES,
    MAX_TRAVERSAL_FILES,
    MDLS_BATCH,
    MIN_FILE_BYTES,
    OPENABLE_EXTS,
    PROJECT_MARKERS,
    PROJECT_SEGMENTS,
    SCAN_EXCLUDE_DIRS,
    SCAN_EXCLUDE_SUFFIXES,
)
from .store import parse_ts, utcnow

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")

#: Project records are context, not inventory. Capped so repos cannot crowd out documents.
MAX_PROJECT_RECORDS = 12_000


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def normalise_date(raw: str) -> str | None:
    """Spotlight prints '2026-10-19 06:32:23 +0000' or a quoted variant."""
    m = DATE_RE.search(raw)
    if not m:
        return None
    stamp = m.group(1).replace(" ", "T")
    return f"{stamp}+00:00"


def expand(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def is_excluded_dir(name: str) -> bool:
    return name.startswith(".") and name not in {".config"} or name in SCAN_EXCLUDE_DIRS


def volume_of(path: Path) -> str:
    """The mount root, so a forecast is per disk and not nonsense across disks."""
    parts = str(path).split("/")
    if len(parts) > 2 and parts[1] == "Volumes":
        return f"/Volumes/{parts[2]}"
    return "/"


def _dir_is_project(dirpath: str) -> bool:
    """One directory, checked once: does it contain build-system furniture?

    Downloaded repository zips have no .git, so markers alone are not enough. The
    segment list catches vendored dependencies and source roots wherever they appear,
    which is what actually keeps Kotlin and PHP out of a list of documents.
    """
    base = os.path.basename(dirpath)
    if base in PROJECT_SEGMENTS or base in (".git", "node_modules"):
        return True
    segments = set(dirpath.split("/"))
    if segments & set(PROJECT_SEGMENTS):
        return True
    for marker in PROJECT_MARKERS:
        try:
            if marker in (".git", "node_modules"):
                if os.path.isdir(os.path.join(dirpath, marker)):
                    return True
            elif os.path.exists(os.path.join(dirpath, marker)):
                return True
        except OSError:
            continue
    return False


def inventory(roots: list[Path], limit: int = MAX_SCAN_FILES,
              max_seen: int = MAX_TRAVERSAL_FILES) -> tuple[list[dict], int]:
    """Walk the filesystem and return stat-only records. No Spotlight involved.

    Project context is propagated top-down while walking, so a file under a repo is
    labelled 'project' without a single extra stat call per file.
    """
    out: list[dict] = []
    seen = 0
    project_kept = 0
    project_dirs: dict[str, bool] = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
            parent = os.path.dirname(dirpath)
            inside = project_dirs.get(parent, False) or _dir_is_project(dirpath)
            project_dirs[dirpath] = inside
            dirnames[:] = [d for d in dirnames if not is_excluded_dir(d) and not d.endswith(SCAN_EXCLUDE_SUFFIXES)]
            for fname in filenames:
                seen += 1
                if seen > max_seen:
                    return out, seen
                if fname.startswith("."):
                    continue
                suffix = Path(fname).suffix.lower()
                if suffix and suffix not in OPENABLE_EXTS:
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full, follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if st.st_size < MIN_FILE_BYTES:
                    continue
                created = getattr(st, "st_birthtime", st.st_ctime)
                try:
                    rel = Path(full).relative_to(root)
                except ValueError:
                    rel = Path(full).name
                rec = {
                    "path": full,
                    "volume": volume_of(Path(full)),
                    "root": str(root),
                    "name": fname,
                    "ext": suffix or "",
                    "dir_depth": len(rel.parts) - 1 if isinstance(rel, Path) else 0,
                    "size_bytes": st.st_size,
                    "created_at": iso(datetime.fromtimestamp(created, tz=timezone.utc)),
                    "modified_at": iso(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)),
                    "context": "project" if inside else "personal",
                }
                if rec["context"] == "project":
                    # Project trees are counted, not catalogued. A repository of
                    # 200,000 files should not crowd out the 600 documents that matter.
                    project_kept += 1
                    if project_kept > MAX_PROJECT_RECORDS:
                        continue
                out.append(rec)
                if len(out) >= limit:
                    return out, seen
    return out, seen


def _mdls_one(path: str, attrs: tuple[str, ...]) -> dict:
    """One mdls call. Returns {attr: raw value}; UsedDates keeps its list form."""
    args = ["mdls"]
    for a in attrs:
        args += ["-name", a]
    args.append(path)
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=8)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    text = res.stdout
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if "=" in stripped and (line[:1] not in (" ", "\t", "(") or stripped.startswith("kMDItem")):
            key = stripped.split("=", 1)[0].strip()
            if key.startswith("kMDItem"):
                current = key
                found.setdefault(current, [])
                rest = stripped.split("=", 1)[1].strip()
                if rest and rest != "(" and rest != "(null)":
                    found[current].append(rest)
                continue
        if current is not None:
            found.setdefault(current, []).append(stripped)
    return {k: v for k, v in found.items()}


def history(paths: list[str], workers: int = 8) -> dict[str, dict]:
    """Fetch open history for many files through the Spotlight daemon, in parallel."""
    attrs = ("kMDItemUsedDates", "kMDItemLastUsedDate", "kMDItemContentCreationDate", "kMDItemDateAdded")
    result: dict[str, dict] = {}
    if not paths:
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p, raw in zip(paths, pool.map(lambda x: _mdls_one(x, attrs), paths)):
            used = [normalise_date(x) for x in raw.get("kMDItemUsedDates", [])]
            used = sorted({u for u in used if u})
            last = raw.get("kMDItemLastUsedDate", [])
            added = raw.get("kMDItemDateAdded", [])
            result[p] = {
                "used_dates": used,
                "last_used_at": used[-1] if used else (normalise_date(last[0]) if last else None),
                "first_used_at": used[0] if used else None,
                "open_count": len(used),
                "has_history": int(bool(used)),
                "added_at": normalise_date(added[0]) if added else None,
            }
    return result


def spotlight_candidates(roots: list[Path]) -> set[str]:
    """Ask the index directly which files have ever been opened. Instant, and a
    useful prefilter when the inventory is large."""
    found: set[str] = set()
    floor = "$time.iso(2000-01-01T00:00:00Z)"
    for root in roots:
        if not root.exists():
            continue
        query = f'kMDItemLastUsedDate >= {floor}'
        try:
            res = subprocess.run(["mdfind", "-onlyin", str(root), query],
                                 capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            continue
        for line in res.stdout.splitlines():
            if line.strip():
                found.add(line.strip())
    return found


@dataclass
class ScanResult:
    files_seen: int = 0
    files_stored: int = 0
    with_history: int = 0
    bytes_scanned: int = 0
    inventory_s: float = 0.0
    history_s: float = 0.0
    roots: tuple[str, ...] = ()
    mode: str = "real"
    volumes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "files_stored": self.files_stored,
            "with_history": self.with_history,
            "bytes_scanned": self.bytes_scanned,
            "inventory_s": round(self.inventory_s, 2),
            "history_s": round(self.history_s, 2),
            "roots": list(self.roots),
            "mode": self.mode,
        }


def scan(conn, roots: list[Path], *, use_index: bool = True, workers: int = 8,
         limit: int = MAX_SCAN_FILES, mode: str = "real") -> ScanResult:
    """Index the given roots. Two-stage, and the second stage is the only slow one.

    Spotlight knows about files that os.walk never reaches because our excludes prune
    them, so the mdls candidate list is the union of the personal inventory and the
    index's own answer. Project files are skipped entirely: nobody reopens a .php in a
    vendor folder, and paying 180 seconds to ask 8,000 times about it is not a trade
    anyone should make.
    """
    from .store import record_snapshot, replace_opens, upsert_file

    started = time.time()
    run_cur = conn.execute(
        "INSERT INTO scan_runs(started_at, roots, mode) VALUES(?,?,?)",
        (utcnow(), ",".join(str(r) for r in roots), mode),
    )
    run_id = int(run_cur.lastrowid)

    t0 = time.time()
    records, seen = inventory(roots, limit=limit, max_seen=MAX_TRAVERSAL_FILES)
    inventory_s = time.time() - t0

    t1 = time.time()
    known = {r["path"] for r in records}
    personal = [r["path"] for r in records if r["context"] == "personal"]
    extras: list[dict] = []
    if use_index:
        for p in sorted(spotlight_candidates(roots)):
            if p in known or not os.path.isfile(p):
                continue
            if "/node_modules/" in p or "/.git/" in p:
                continue
            suffix = Path(p).suffix.lower()
            if suffix and suffix not in OPENABLE_EXTS:
                continue
            try:
                st = os.stat(p, follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size < MIN_FILE_BYTES:
                continue
            rel = Path(p)
            extras.append({
                "path": p,
                "volume": volume_of(rel),
                "root": str(rel),
                "name": rel.name,
                "ext": suffix or "",
                "dir_depth": max(len(rel.parts) - 4, 0),
                "size_bytes": st.st_size,
                "created_at": iso(datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_ctime), tz=timezone.utc)),
                "modified_at": iso(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)),
                "context": "personal",
            })
    candidates = personal + [e["path"] for e in extras]
    hist = history(candidates, workers=workers)
    history_s = time.time() - t1

    stored = 0
    with_hist = 0
    total_bytes = 0
    for rec in records + extras:
        h = hist.get(rec["path"])
        if h:
            rec.update(h)
            if h["used_dates"]:
                with_hist += 1
        file_id = upsert_file(conn, {**rec, "scanned_run": run_id})
        if file_id and rec.get("used_dates"):
            replace_opens(conn, file_id, rec["used_dates"])
        stored += 1
        total_bytes += rec["size_bytes"]

    vols: dict[str, dict] = {}
    for root in roots:
        vol = volume_of(root)
        try:
            usage = shutil.disk_usage(str(root))
        except OSError:
            continue
        vols[vol] = {"total_bytes": usage.total, "free_bytes": usage.free}
        record_snapshot(conn, vol, usage.total, usage.free)

    conn.execute(
        """UPDATE scan_runs SET finished_at=?, files_seen=?, files_stored=?, with_history=?,
           bytes_scanned=?, seconds=? WHERE id=?""",
        (utcnow(), seen, stored, with_hist, total_bytes, time.time() - started, run_id),
    )
    conn.commit()

    return ScanResult(
        files_seen=seen, files_stored=stored, with_history=with_hist,
        bytes_scanned=total_bytes, inventory_s=inventory_s, history_s=history_s,
        roots=tuple(str(r) for r in roots), mode=mode, volumes=vols,
    )
