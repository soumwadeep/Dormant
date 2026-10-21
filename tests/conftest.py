"""Test fixtures.

Dormant's paths are resolved at import time from the environment, so the isolation
has to happen before anything in the package is imported. That is what this file is
for: it points the index and the staging folder at throwaway directories, which means
no test can ever move a file in the developer's home folder.
"""

from __future__ import annotations

import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="dormant-tests-")
os.environ["DORMANT_DATA"] = os.path.join(_TMP, "data")
os.environ["DORMANT_ARCHIVE"] = os.path.join(_TMP, "archive")
os.environ["DORMANT_CHECKPOINTS"] = os.path.join(_TMP, "checkpoints")

import pytest  # noqa: E402

from dormant.store import connect, init_db, replace_opens, upsert_file  # noqa: E402


@pytest.fixture(scope="session")
def db():
    conn = connect()
    init_db(conn)
    yield conn
    conn.close()
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def fresh_db():
    conn = connect()
    init_db(conn)
    for table in ("predictions", "opens", "actions", "files", "snapshots", "bench", "scan_runs"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    yield conn
    conn.commit()  # release any write lock the test left open before the next fixture runs


@pytest.fixture(autouse=True)
def _release_api_connection():
    """The server opens its own connection; leave no writer behind between tests."""
    yield
    import sys

    mod = sys.modules.get("api.app")
    conn = getattr(mod, "_conn", None) if mod else None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        mod._conn = None


def add_file(conn, path: str, *, created: str, modified: str | None = None, size: int = 10_000,
             opens: list[str] | None = None, ext: str | None = None, context: str = "personal") -> int:
    from pathlib import Path as P

    from dormant.scanner import volume_of

    p = P(path)
    fid = upsert_file(conn, {
        "path": path, "volume": volume_of(p),
        "root": str(p.parent), "name": p.name, "ext": ext if ext is not None else p.suffix.lower(),
        "dir_depth": 2, "size_bytes": size, "created_at": created,
        "modified_at": modified or created, "added_at": created,
        "last_used_at": opens[-1] if opens else None, "first_used_at": opens[0] if opens else None,
        "open_count": len(opens or []), "used_dates": opens or [], "has_history": int(bool(opens)),
        "context": context,
    })
    if opens and fid:
        replace_opens(conn, fid, opens)
    conn.commit()
    return int(fid)
