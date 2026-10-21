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
    """archive() and trash() must never unlink. Only purge() is allowed to, and it is
    a different function so this tripwire still means something."""
    def tripwire(*a, **k):
        raise AssertionError("archive/trash attempted a filesystem delete")

    for fn in ("remove", "unlink", "rmdir"):
        monkeypatch.setattr(os, fn, tripwire)
    monkeypatch.setattr(os, "replace", tripwire)

    p = make_real(tmp_path, "delta.txt")
    fid = add_file(fresh_db, p, created=iso(300))
    res = action.archive(fresh_db, [fid])
    assert res["count"] == 1
    action.undo(fresh_db, res["manifest"])
    assert os.path.exists(p)


# --------------------------------------------------------------------- trash

def test_trash_moves_to_the_system_trash_and_undoes(fresh_db, tmp_path, monkeypatch):
    """Never point at the developer's real ~/.Trash in a test."""
    fake_trash = tmp_path / "Trash"
    fake_trash.mkdir()
    monkeypatch.setattr(action, "TRASH_DIR", fake_trash)

    p = make_real(tmp_path, "throwaway.pdf")
    fid = add_file(fresh_db, p, created=iso(300))
    res = action.trash(fresh_db, [fid])
    assert res["count"] == 1 and res["kind"] == "trashed" and res["reversible"]
    assert not os.path.exists(p)
    assert any(f.name.startswith("throwaway") for f in fake_trash.iterdir())

    undo = action.undo(fresh_db, res["manifest"])
    assert undo["restored"] == 1 and os.path.exists(p)


def test_latest_manifest_skips_permanent_deletes(fresh_db, tmp_path, monkeypatch):
    fake_trash = tmp_path / "Trash2"
    fake_trash.mkdir()
    monkeypatch.setattr(action, "TRASH_DIR", fake_trash)
    p = make_real(tmp_path, "gone.pdf")
    fid = add_file(fresh_db, p, created=iso(300))
    action.purge(fresh_db, [fid], confirm=action.DELETE_PHRASE)
    assert action.latest_manifest(fresh_db) is None, "undo must never offer to reverse a purge"


# --------------------------------------------------------------------- purge

def test_purge_refuses_without_the_phrase(fresh_db, tmp_path):
    p = make_real(tmp_path, "precious.pdf")
    fid = add_file(fresh_db, p, created=iso(300))
    import pytest

    with pytest.raises(UnsafeAction):
        action.purge(fresh_db, [fid], confirm="yes")
    with pytest.raises(UnsafeAction):
        action.purge(fresh_db, [fid])
    assert os.path.exists(p), "a refused purge must leave the file exactly where it was"


def test_purge_deletes_and_records_what_was_lost(fresh_db, tmp_path):
    paths = [make_real(tmp_path, f"loss{i}.pdf", 2048) for i in range(3)]
    ids = [add_file(fresh_db, p, created=iso(300), size=2048) for p in paths]
    res = action.purge(fresh_db, ids, confirm=action.DELETE_PHRASE)
    assert res["removed"] == 3 and res["reversible"] is False and res["deletes"] == 3
    assert all(not os.path.exists(p) for p in paths)
    assert res["bytes_freed"] == 3 * 2048

    from dormant.config import PATHS

    manifest = json.loads((PATHS.manifest_dir / f"{res['manifest']}.json").read_text())
    assert {m["origin"] for m in manifest["moves"]} == set(paths), "the record must name what is gone"
    row = fresh_db.execute("SELECT disposition FROM files WHERE id=?", (ids[0],)).fetchone()
    assert row[0] == "deleted"


def test_purge_survives_a_vanishing_file(fresh_db, tmp_path):
    """A file that disappeared since the scan is skipped, not an exception mid-loop."""
    here = make_real(tmp_path, "still_here.pdf")
    away = make_real(tmp_path, "about_to_vanish.pdf")
    id_here = add_file(fresh_db, here, created=iso(300))
    id_away = add_file(fresh_db, away, created=iso(300))
    os.remove(away)
    res = action.purge(fresh_db, [id_here, id_away], confirm=action.DELETE_PHRASE)
    assert res["removed"] == 1 and len(res["skipped"]) == 1
    assert not os.path.exists(here)


# --------------------------------------------------------------------- preview

def test_preview_kind_mapping(fresh_db, tmp_path):
    cases = {"a.png": "image", "b.pdf": "pdf", "c.txt": "text", "d.mp4": "video",
             "e.mp3": "audio", "f.zip": "other"}
    for name, kind in cases.items():
        p = make_real(tmp_path, name, 3000)
        fid = add_file(fresh_db, p, created=iso(300), ext=os.path.splitext(name)[1])
        assert action.preview_info(fresh_db, fid)["kind"] == kind


def test_preview_reports_missing_files(fresh_db, tmp_path):
    p = make_real(tmp_path, "evaporated.pdf")
    fid = add_file(fresh_db, p, created=iso(300))
    os.remove(p)
    info = action.preview_info(fresh_db, fid)
    assert info["exists"] is False


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
