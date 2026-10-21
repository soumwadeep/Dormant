"""FastAPI service for the Dormant dashboard.

Bound to 127.0.0.1 by default. There is no authentication because there is nothing
to authenticate against: the only data it can serve is the index of the machine it is
running on, and the only client it listens for is your own browser.

The database connection is opened lazily. Importing this module must not touch the
disk, because tests, the CLI and `uvicorn --reload` all import it for different
reasons.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dormant import engine
from dormant.config import API_HOST, API_PORT, DEFAULT_ROOTS, VERSION
from dormant.store import connect, get_meta, init_db

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Dormant", version=VERSION, docs_url=None, redoc_url=None)

_conn = None
_state_lock = threading.Lock()
_job: dict[str, Any] = {"running": False, "kind": None, "started": None, "log": [], "error": None}


def db():
    """One connection for the process, created on first use."""
    global _conn
    if _conn is None:
        _conn = connect()
        init_db(_conn)
    return _conn


def _job_log(msg: str) -> None:
    _job["log"].append({"ts": time.strftime("%H:%M:%S"), "msg": msg[:200]})
    _job["log"] = _job["log"][-60:]


class ArchiveIn(BaseModel):
    top: int = Field(50, ge=1, le=5000)
    max_gb: float = Field(10.0, gt=0, le=2000)
    confirm: bool = False


class UndoIn(BaseModel):
    manifest: str | None = None
    confirm: bool = False


class AgentIn(BaseModel):
    text: str = Field(min_length=1, max_length=400)


class ScanIn(BaseModel):
    roots: list[str] | None = None


def _background(kind: str, fn) -> None:
    """Long jobs run on a worker thread so the dashboard stays responsive."""
    if _job["running"]:
        raise HTTPException(409, f"a {kind} job is already running")
    _job.update(running=True, kind=kind, started=time.time(), log=[], error=None, result=None)

    def worker():
        try:
            _job_log(f"{kind} started")
            _job["result"] = fn()
            _job_log(f"{kind} finished")
        except Exception as exc:
            _job["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            _job_log(f"error: {_job['error']}")
        finally:
            _job["running"] = False
            _job["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()


@app.get("/")
def index():
    target = WEB_DIR / "index.html"
    if not target.exists():
        return JSONResponse({"error": "web/index.html missing"}, status_code=500)
    return FileResponse(target)


def _has_index(conn) -> bool:
    try:
        return conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] > 0
    except Exception:
        return False


@app.get("/api/health")
def health():
    import shutil

    conn = db()
    usage = shutil.disk_usage("/")
    counts = conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"] if _has_index(conn) else 0
    return {
        "ok": True,
        "app": "Dormant",
        "version": VERSION,
        "mode": get_meta(conn, "data_mode", "real"),
        "files_indexed": counts,
        "bind": f"{API_HOST}:{API_PORT}",
        "network_egress": "none",
        "disk_free_gb": round(usage.free / 1e9, 1),
    }


@app.get("/api/state")
def state():
    with _state_lock:
        return engine.state(db())


@app.get("/api/field")
def field(limit: int = 4000):
    return {"points": engine.field_points(db(), limit=min(limit, 20000))}


@app.get("/api/top")
def top(limit: int = 100):
    return {"files": engine.top_flagged(db(), limit=min(limit, 2000))}


@app.get("/api/bench")
def bench():
    conn = db()
    return {"rows": conn.execute("SELECT * FROM bench ORDER BY id DESC LIMIT 12").fetchall(),
            "scarcity": engine._json_meta(conn, "scarcity"),
            "split": engine._json_meta(conn, "eval_split")}


@app.get("/api/forecast")
def forecast(volume: str | None = None):
    from dormant.forecast import forecast as f

    conn = db()
    if volume is None:
        row = conn.execute("SELECT volume FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        volume = row["volume"] if row else "/"
    return f(conn, volume=volume)


@app.post("/api/scan")
def scan(payload: ScanIn):
    from pathlib import Path as P

    from dormant import scanner

    conn = db()
    roots = [P(r).expanduser().resolve() for r in (payload.roots or DEFAULT_ROOTS)]
    roots = [r for r in roots if r.exists()]
    _background("scan", lambda: scanner.scan(conn, roots, workers=12).as_dict())
    return {"queued": True, "roots": [str(r) for r in roots]}


@app.post("/api/score")
def score():
    _background("score", lambda: engine.score_live(db()))
    return {"queued": True}


@app.post("/api/bench")
def run_bench():
    from dormant.model import bench as run

    conn = db()
    _background("bench", lambda: run(conn))
    return {"queued": True}


@app.post("/api/archive")
def archive(payload: ArchiveIn):
    """Two-step by design: without confirm the call only previews, and nothing moves.

    The worst outcome this endpoint can produce is a dated folder you can undo,
    because action.archive has no delete path and refuses collisions.
    """
    from dormant import action

    conn = db()
    rows = engine.top_flagged(conn, limit=payload.top)
    if not payload.confirm:
        return {"confirm_required": True, "would_stage": len(rows),
                "bytes": sum(r["size_bytes"] for r in rows), "sample": rows[:12]}
    if get_meta(conn, "data_mode", "real") == "demo":
        raise HTTPException(409, "the demo corpus is synthetic: those files do not exist on disk")
    return action.archive(conn, [r["id"] for r in rows], max_bytes=int(payload.max_gb * 1e9),
                          reason="dashboard")


@app.post("/api/undo")
def undo(payload: UndoIn):
    from dormant import action

    conn = db()
    manifest = payload.manifest or action.latest_manifest(conn)
    if not payload.confirm:
        return {"confirm_required": True, "manifest": manifest}
    if get_meta(conn, "data_mode", "real") == "demo":
        raise HTTPException(409, "nothing staged in demo mode")
    return action.undo(conn, manifest)


@app.post("/api/agent")
def agent(payload: AgentIn):
    from dormant.agent import handle

    return handle(db(), payload.text)


@app.get("/api/job")
def job():
    return {**{k: v for k, v in _job.items() if k != "result"},
            "result": _job.get("result"),
            "elapsed": round(time.time() - (_job["started"] or time.time()), 1)}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
