"""Safety invariants for the only part of Dormant that touches real files.

Dormant must be able to move things and undo them. It must never be able to delete a
thing, overwrite a thing, or touch a path it has not indexed. Those are asserted here
by making the destructive syscalls raise, rather than by trusting the code not to
call them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from conftest import add_file
from dormant import action
from dormant.action import UnsafeAction

NOW = datetime.now(timezone.utc)
iso = lambda d: (NOW - timedelta(days=d)).isoformat(timespec="seconds")  # noqa: E731


def make_real(tmp_path, name: str, size: int = 4096) -> str:
    p = tmp_path / "corpus" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return str(p)


def test_archive_moves_and_undo_restores_exactly(fresh_db, tmp_path):
    path_a = make_real(tmp_path, "alpha.pdf")
    path_b = make_real(tmp_path, "beta.pdf")
    ida = add_file(fresh_db, path_a, created=iso(400), size=4096)
    idb = add_file(fresh_db, path_b, created=iso(400), size=4096)

    res = action.archive(fresh_db, [ida, idb], reason="test")
    assert res["count"] == 2 and res["deletes"] == 0
    assert not os.path.exists(path_a) and not os.path.exists(path_b)
    assert all(os.path.exists(m["staged"]) for m in res["moves"])
    assert fresh_db.execute("SELECT archived_at FROM files WHERE id=?", (ida,)).fetchone()[0]

    undo = action.undo(fresh_db, res["manifest"])
    assert undo["restored"] == 2 and undo["complete"]
    assert os.path.exists(path_a) and os.path.exists(path_b)
    assert fresh_db.execute("SELECT archived_at FROM files WHERE id=?", (ida,)).fetchone()[0] is None


def test_manifest_is_a_real_artifact_on_disk(fresh_db, tmp_path):
    p = make_real(tmp_path, "gamma.docx")
    fid = add_file(fresh_db, p, created=iso(300))
    res = action.archive(fresh_db, [fid])
    from dormant.config import PATHS

    manifest = json.loads((PATHS.manifest_dir / f"{res['manifest']}.json").read_text())
    assert manifest["moves"][0]["origin"] == p
    assert manifest["deletes"] == 0


def test_never_overwrites_an_existing_file(fresh_db, tmp_path):
    """Two files with the same name must both survive, staged separately."""
    a = make_real(tmp_path, "dup.pdf", 1000)
    b_dir = tmp_path / "other"
    b_dir.mkdir(exist_ok=True)
    b = b_dir / "dup.pdf"
    b.write_bytes(b"y" * 2000)
    ida = add_file(fresh_db, a, created=iso(300), size=1000)
    idb = add_file(fresh_db, str(b), created=iso(300), size=2000)

    res = action.archive(fresh_db, [ida, idb])
    staged = [m["staged"] for m in res["moves"]]
    assert len(set(staged)) == 2, "collision must produce a second destination, not a clobber"
    assert all(os.path.exists(s) for s in staged)
    assert sum(os.path.getsize(s) for s in staged) == 3000


def test_refuses_protected_paths(fresh_db):
    fid = add_file(fresh_db, "/System/Library/LaunchDaemons/evil.pdf", created=iso(300))
    res = action.archive(fresh_db, [fid])
    assert res["count"] == 0 and len(res["skipped"]) == 1
    assert "protected" in res["skipped"][0]["why"]


def test_refuses_unknown_and_missing(fresh_db, tmp_path):
    assert action.archive(fresh_db, [999999])["skipped"], "an id that is not indexed cannot move"
    ghost = str(tmp_path / "gone.png")
    fid = add_file(fresh_db, ghost, created=iso(300))
    res = action.archive(fresh_db, [fid])
    assert res["count"] == 0 and "missing" in res["skipped"][0]["why"]


def test_refuses_symlinks(fresh_db, tmp_path):
    real = make_real(tmp_path, "target.pdf")
    link = str(tmp_path / "link.pdf")
    os.symlink(real, link)
    fid = add_file(fresh_db, link, created=iso(300))
    res = action.archive(fresh_db, [fid])
    assert res["count"] == 0 and "symlink" in res["skipped"][0]["why"]
    assert os.path.exists(real), "the link target must not be touched"


def test_no_delete_syscall_is_ever_reached(fresh_db, tmp_path, monkeypatch):
    """Patch the destructive primitives: if Dormant calls one, the test fails loudly."""
    def tripwire(*a, **k):
        raise AssertionError("Dormant attempted a filesystem delete")

    for fn in ("remove", "unlink", "rmdir"):
        monkeypatch.setattr(os, fn, tripwire)
    monkeypatch.setattr(os, "replace", tripwire)

    p = make_real(tmp_path, "delta.txt")
    fid = add_file(fresh_db, p, created=iso(300))
    res = action.archive(fresh_db, [fid])
    assert res["count"] == 1
    action.undo(fresh_db, res["manifest"])
    assert os.path.exists(p)


def test_undo_is_not_double_applied(fresh_db, tmp_path):
    p = make_real(tmp_path, "epsilon.pdf")
    fid = add_file(fresh_db, p, created=iso(300))
    res = action.archive(fresh_db, [fid])
    first = action.undo(fresh_db, res["manifest"])
    second = action.undo(fresh_db, res["manifest"])
    assert first["restored"] == 1
    assert second["restored"] == 0, "a manifest must not be reversible twice"


def test_byte_budget_is_respected(fresh_db, tmp_path):
    ids = [add_file(fresh_db, make_real(tmp_path, f"big{i}.zip", 900), created=iso(300), size=900)
           for i in range(5)]
    res = action.archive(fresh_db, ids, max_bytes=2000)
    assert res["count"] == 2 and res["bytes"] <= 2000
