"""Synthetic corpus, so the screenshots and the public demo leak nothing real.

Dormant's index contains file paths, sizes and the dates you opened things. That is
personal. The demo generator produces the same shape of data with fake names, which
means the repository, the DEV post and any deployed preview can show real numbers
without showing anybody's Documents folder.

The behaviour is modelled per file kind, because that is what makes the learned
scores interesting: a tax PDF and a screenshot have completely different lifespans,
and a rule that says "90 days idle means dead" is wrong about one of them. Revisits
are deliberately placed both inside and outside the 30-day window, so the stale
population is neither all-positive nor all-negative.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .store import replace_opens, set_meta, upsert_file

KINDS = {
    "tax": {"ext": ".pdf", "dirs": ["Documents/Tax/2024", "Documents/Tax/2025", "Downloads"],
            "n": 240, "size": (90_000, 4_000_000), "pattern": "paperwork"},
    "invoices": {"ext": ".pdf", "dirs": ["Documents/Invoices", "Downloads"],
                 "n": 200, "size": (60_000, 2_500_000), "pattern": "paperwork"},
    "screenshots": {"ext": ".png", "dirs": ["Desktop", "Desktop/Captures"],
                    "n": 380, "size": (200_000, 9_000_000), "pattern": "one_shot"},
    "receipts": {"ext": ".jpg", "dirs": ["Documents/Expenses"], "n": 160,
                 "size": (300_000, 6_000_000), "pattern": "one_shot"},
    "photos": {"ext": ".heic", "dirs": ["Pictures/2024", "Pictures/2025", "Pictures/Singapore"],
               "n": 420, "size": (1_800_000, 90_000_000), "pattern": "media"},
    "videos": {"ext": ".mov", "dirs": ["Movies/Clips", "Movies/Screen"], "n": 70,
               "size": (40_000_000, 900_000_000), "pattern": "media"},
    "working": {"ext": ".docx", "dirs": ["Documents/Actizant", "Documents/Proposals", "Desktop"],
                "n": 150, "size": (40_000, 12_000_000), "pattern": "recurring"},
    "readings": {"ext": ".md", "dirs": ["Documents/Reading", "Documents/Notes"], "n": 130,
                 "size": (4_000, 120_000), "pattern": "recurring"},
    "archives": {"ext": ".zip", "dirs": ["Downloads", "Documents/Backups"], "n": 100,
                 "size": (5_000_000, 1_200_000_000), "pattern": "archive"},
    "songs": {"ext": ".mp3", "dirs": ["Music/Demo", "Downloads"], "n": 70,
              "size": (3_000_000, 30_000_000), "pattern": "one_shot"},
}

PROJECT_KINDS = {
    "src": {"ext": ".ts", "dirs": ["Repos/aurora-web/src", "Repos/dormant/web"], "n": 700, "size": (1_000, 60_000)},
    "asset": {"ext": ".png", "dirs": ["Repos/aurora-web/public"], "n": 160, "size": (5_000, 900_000)},
    "lock": {"ext": ".json", "dirs": ["Repos/aurora-web"], "n": 40, "size": (10_000, 2_000_000)},
}

WORDS = ["itr", "form16", "invoice", "quote", "agreement", "statement", "screenshot", "capture",
         "receipt", "deck", "notes", "draft", "final", "v2", "scan", "passport", "lease", "summary"]


def _push(out: list[datetime], base: datetime, lo: int, hi: int, today: datetime,
          rng: random.Random) -> None:
    d = base + timedelta(days=rng.randint(lo, hi), hours=rng.randint(0, 23))
    if d <= today:
        out.append(d)


def _dates_for(pattern: str, created: datetime, rng: random.Random, today: datetime) -> list[datetime]:
    """Draw an open history per behaviour model."""
    age = int((today - created).total_seconds() // 86400)
    if age <= 0:
        age = 1
    out: list[datetime] = []

    def pick(p: float) -> bool:
        return rng.random() < p

    if pattern == "paperwork":
        # Filed soon after download, then nothing until a deadline drags it back.
        n = rng.randint(1, 3)
        last = created
        for _ in range(n):
            _push(out, last, 0, min(10, age), today, rng)
            if out:
                last = max(out)
        if pick(0.40):
            _push(out, last, 16, 29, today, rng)   # revisit inside the window: a positive
        if pick(0.34):
            _push(out, last, 48, 240, today, rng)  # revisit outside it: a negative, then maybe a positive
            if pick(0.45):
                _push(out, max(out), 16, 29, today, rng)
    elif pattern == "one_shot":
        if rng.random() < 0.86:
            _push(out, created, 0, min(4, age), today, rng)
        if out and pick(0.07):
            _push(out, max(out), 20, 29, today, rng)
        elif out and pick(0.09):
            _push(out, max(out), 70, 320, today, rng)
    elif pattern == "media":
        if rng.random() < 0.7:
            _push(out, created, 0, min(2, age), today, rng)
        if out and pick(0.26):
            _push(out, max(out), 3, 13, today, rng)
        if out and pick(0.10):
            _push(out, max(out), 18, 29, today, rng)
        if out and pick(0.11):
            _push(out, max(out), 90, 420, today, rng)
    elif pattern == "recurring":
        gap = rng.choice([3, 4, 5, 6, 7, 9])
        d = created + timedelta(days=rng.randint(0, max(gap // 2, 1)))
        while d < today:
            out.append(d)
            d += timedelta(days=gap + rng.randint(-1, 2))
    elif pattern == "archive":
        if rng.random() < 0.05:
            _push(out, created, 0, min(3, age), today, rng)
        if out and pick(0.02):
            _push(out, max(out), 18, 28, today, rng)
    return sorted({x for x in out if x <= today})


def seed(conn, *, days: int = 900, rng_seed: int = 7, include_project: bool = True) -> dict:
    rng = random.Random(rng_seed)
    today = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
    home = "/Users/demo"
    total_bytes = 0
    n_files = 0
    n_events = 0

    for kind, spec in KINDS.items():
        for i in range(spec["n"]):
            created = today - timedelta(
                days=rng.randint(0, days), hours=rng.randint(0, 23))
            d = rng.choice(spec["dirs"])
            word = rng.choice(WORDS)
            name = f"{word}_{created.strftime('%Y%m%d')}_{i:03d}{spec['ext']}"
            path = f"{home}/{d}/{name}"
            size = rng.randint(*spec["size"])
            rel_depth = len(d.split("/"))
            opens = _dates_for(spec["pattern"], created, rng, today)
            iso = [o.isoformat(timespec="seconds") for o in opens]
            rec = {
                "path": path, "volume": "/demo-boot", "root": home, "name": name, "ext": spec["ext"],
                "dir_depth": rel_depth, "size_bytes": size,
                "created_at": created.isoformat(timespec="seconds"),
                "modified_at": (created + timedelta(hours=rng.randint(0, 30))).isoformat(timespec="seconds"),
                "added_at": created.isoformat(timespec="seconds"),
                "last_used_at": iso[-1] if iso else None,
                "first_used_at": iso[0] if iso else None,
                "open_count": len(iso), "used_dates": iso, "has_history": int(bool(iso)),
                "context": "personal",
            }
            fid = upsert_file(conn, rec)
            if fid and iso:
                n_events += replace_opens(conn, fid, iso)
            total_bytes += size
            n_files += 1

    if include_project:
        for kind, spec in PROJECT_KINDS.items():
            for i in range(spec["n"]):
                created = today - timedelta(days=rng.randint(0, days))
                d = rng.choice(spec["dirs"])
                name = f"{rng.choice(WORDS)}_{i:04d}{spec['ext']}"
                path = f"{home}/{d}/{name}"
                size = rng.randint(*spec["size"])
                upsert_file(conn, {
                    "path": path, "volume": "/demo-boot", "root": home, "name": name, "ext": spec["ext"],
                    "dir_depth": len(d.split("/")), "size_bytes": size,
                    "created_at": created.isoformat(timespec="seconds"),
                    "modified_at": created.isoformat(timespec="seconds"),
                    "added_at": created.isoformat(timespec="seconds"),
                    "last_used_at": None, "first_used_at": None, "open_count": 0,
                    "used_dates": [], "has_history": 0, "context": "project",
                })
                total_bytes += size
                n_files += 1

    conn.execute("DELETE FROM snapshots")
    # A volume that looks like a real 256 GB laptop mid-week: 118 GB free a quarter
    # ago, filling at a few hundred MB a day, now around the low fifties.
    total = 245_100_000_000
    drift = rng.uniform(0.30e9, 0.55e9)
    free = 118_000_000_000
    for back in range(120, 0, -1):
        when = today - timedelta(days=back)
        free -= drift * rng.uniform(0.25, 1.75)
        free = max(free, 46_000_000_000)
        conn.execute(
            "INSERT INTO snapshots(taken_at, volume, total_bytes, free_bytes, used_bytes) VALUES(?,?,?,?,?)",
            (when.isoformat(timespec="seconds"), "/demo-boot", total, int(free), int(total - free)))
    conn.commit()
    set_meta(conn, "data_mode", "demo")
    set_meta(conn, "demo_seed", rng_seed)
    conn.commit()
    return {"files": n_files, "open_events": n_events, "bytes": total_bytes}
