"""Orchestration: scan, label, bench, score, forecast, brief, one call.

The live scoring pass fits the open-weight model on every labelled observation the
index currently supports and predicts once over the live files. That is the whole
"training" story: there is no training, there is context. Adding rows to the prompt
is what a fine-tuning run is to a closed model.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import action as act
from . import scanner
from .brief import build_brief
from .config import API_PORT, DEFAULT_ROOTS, PATHS, REOPEN_WINDOW_DAYS, VERSION
from .dataset import labelled, time_split
from .features import build_rows, scoring_rows
from .forecast import forecast as forecast_volume
from .model import TabPFNProvider, bench as run_bench, model_choice
from .store import clear_predictions, connect, get_meta, init_db, parse_ts, set_meta, utcnow

_MODELS: dict[str, TabPFNProvider] = {}


def _json_meta(conn, key: str):
    raw = get_meta(conn, key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _provider() -> TabPFNProvider:
    key = "default"
    if key not in _MODELS:
        _MODELS[key] = TabPFNProvider(n_estimators=2)
    return _MODELS[key]


def best_threshold(conn) -> tuple[float, float]:
    """Return (threshold, precision) from the most recent benchmark that hit the target."""
    row = conn.execute(
        "SELECT model, notes FROM bench ORDER BY id DESC LIMIT 1").fetchone()
    ts_row = conn.execute(
        "SELECT precision, reclaimed FROM bench WHERE precision IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    import json
    last_full = conn.execute("SELECT notes FROM bench WHERE model = 'tabpfn' ORDER BY id DESC LIMIT 1").fetchone()
    if last_full:
        try:
            data = json.loads(last_full["notes"])
            if data.get("threshold"):
                return float(data["threshold"]), float(ts_row["precision"] or 0) if ts_row else 0.0
        except (json.JSONDecodeError, TypeError):
            pass
    return 0.85, 0.0


def score_live(conn, *, horizon: int = REOPEN_WINDOW_DAYS) -> dict:
    rows = scoring_rows(conn, horizon=horizon)
    if not rows:
        return {"scored": 0, "reason": "no live files"}
    train_rows = labelled(build_rows(conn, horizon=horizon))
    if len(train_rows) < 20:
        return {"scored": 0, "reason": f"only {len(train_rows)} labelled rows; scan more"}
    split = time_split(train_rows, frac=1.0)
    prov = _provider()
    t0 = time.time()
    try:
        X = [[r.features[k] for k in split.feature_names] for r in split.train]
        y = [int(r.label or 0) for r in split.train]
        clf = prov.load()
        if clf is None:
            return {"scored": 0, "reason": f"model unavailable: {prov._error}"}
        clf.fit(X, y)
        Xt = [[r.features[k] for k in split.feature_names] for r in rows]
        probs = clf.predict_proba(Xt)[:, 1]
    except Exception as exc:
        return {"scored": 0, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    elapsed = time.time() - t0

    threshold, precision = best_threshold(conn)
    ts = utcnow()
    clear_predictions(conn)
    order = sorted(zip(rows, probs), key=lambda pair: -pair[1])
    rows_in = []
    for rank, (r, p) in enumerate(order, start=1):
        rows_in.append((r.file_id, ts, float(p), rank, "tabpfn", horizon, 0, None))
    conn.executemany(
        "INSERT INTO predictions(file_id, ts, p_reopen, rank, model, horizon, holdout, label) VALUES(?,?,?,?,?,?,?,?)",
        rows_in,
    )
    conn.commit()
    flagged = [x for x, p in order if (1 - p) >= threshold]
    bytes_flagged = sum(x.size_bytes for x in flagged)
    return {
        "scored": len(rows_in),
        "seconds": round(elapsed, 2),
        "threshold": threshold,
        "precision_at_threshold": precision,
        "flagged": len(flagged),
        "flagged_bytes": bytes_flagged,
        "model_available": prov.available,
    }


def top_flagged(conn, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        """SELECT f.id, f.path, f.name, f.ext, f.size_bytes, f.volume, f.created_at, f.modified_at,
                  f.last_used_at, f.open_count, f.context, p.p_reopen, p.rank
           FROM predictions p JOIN files f ON f.id = p.file_id
           WHERE f.archived_at IS NULL
           ORDER BY p.p_reopen ASC LIMIT ?""", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        out.append({
            "id": d["id"],
            "name": d["name"],
            "path": d["path"],
            "ext": d["ext"],
            "size_bytes": d["size_bytes"],
            "p_reopen": round(d["p_reopen"], 4),
            "dormant_confidence": round(1 - d["p_reopen"], 4),
            "last_used_at": d["last_used_at"],
            "created_at": d["created_at"],
            "open_count": d["open_count"],
            "context": d["context"],
            "rank": d["rank"],
        })
    return out


def field_points(conn, limit: int = 3000) -> list[dict]:
    """Coordinates for the 3D view: age on the radius, dormancy on the vertical axis."""
    rows = conn.execute(
        """SELECT f.size_bytes, f.ext, f.created_at, f.last_used_at, p.p_reopen, f.context
           FROM predictions p JOIN files f ON f.id = p.file_id
           WHERE f.archived_at IS NULL ORDER BY p.rank LIMIT ?""", (limit,)).fetchall()
    now = datetime.now(timezone.utc)
    pts = []
    for r in rows:
        created = parse_ts(r["created_at"]) or now
        age_days = max((now - created).total_seconds() / 86400.0, 0.0)
        pts.append({
            "age": round(age_days, 1),
            "p": round(float(r["p_reopen"]), 4),
            "size": r["size_bytes"],
            "ext": r["ext"],
        })
    return pts


def largest_cluster(conn, threshold: float) -> str | None:
    rows = conn.execute(
        """SELECT f.path, f.size_bytes, p.p_reopen FROM predictions p
           JOIN files f ON f.id = p.file_id WHERE f.archived_at IS NULL""").fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        if (1 - r["p_reopen"]) < threshold:
            continue
        parent = str(Path(r["path"]).parent)
        counts[parent] = counts.get(parent, 0) + r["size_bytes"]
    if not counts:
        return None
    top = max(counts.items(), key=lambda kv: kv[1])
    return f"{Path(top[0]).name or top[0]} ({top[1] / 1e9:.1f} GB)"


def state(conn) -> dict:
    mode = get_meta(conn, "data_mode", "real")
    volume_row = conn.execute("SELECT volume FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
    volume = volume_row["volume"] if volume_row else "/"
    fc = forecast_volume(conn, volume=volume)
    counts = conn.execute(
        """SELECT COUNT(*) total, SUM(has_history) hist, SUM(size_bytes) bytes,
                  SUM(CASE WHEN context='project' THEN 1 ELSE 0 END) project
           FROM files WHERE archived_at IS NULL""").fetchone()
    last_scan = conn.execute("SELECT * FROM scan_runs ORDER BY id DESC LIMIT 1").fetchone()
    bench_rows = conn.execute("SELECT * FROM bench ORDER BY id DESC LIMIT 12").fetchall()
    threshold, precision = best_threshold(conn)
    flagged_row = conn.execute(
        """SELECT COUNT(*) c, COALESCE(SUM(f.size_bytes),0) b FROM predictions p
           JOIN files f ON f.id = p.file_id
           WHERE f.archived_at IS NULL AND (1 - p.p_reopen) >= ?""", (threshold,)).fetchone()
    days_bought = None
    if fc.get("growth_gb_per_day") and fc["growth_gb_per_day"] > 0:
        days_bought = (flagged_row["b"] / 1e9) / fc["growth_gb_per_day"]
    conf_row = conn.execute(
        """SELECT AVG(1 - p.p_reopen) m FROM predictions p JOIN files f ON f.id = p.file_id
           WHERE f.archived_at IS NULL AND (1 - p.p_reopen) >= ?""", (threshold,)).fetchone()
    mean_conf = float(conf_row["m"] or 0.0)
    brief = build_brief(
        count=flagged_row["c"], bytes_total=flagged_row["b"], mean_confidence=mean_conf,
        holdout_precision=precision or None, forecast=fc, reclaimed_days=days_bought,
        top_dir=largest_cluster(conn, threshold),
        model_name=f"TabPFN-{model_choice()} (open weights, local)",
    )
    return {
        "app": "Dormant",
        "version": VERSION,
        "mode": mode,
        "generated_at": utcnow(),
        "counts": {
            "files": counts["total"] or 0,
            "with_history": counts["hist"] or 0,
            "project_files": counts["project"] or 0,
            "bytes": counts["bytes"] or 0,
        },
        "threshold": threshold,
        "precision": precision,
        "flagged": {"files": flagged_row["c"], "bytes": flagged_row["b"],
                    "days_bought": round(days_bought, 1) if days_bought else None},
        "brief": brief,
        "forecast": fc,
        "bench": [dict(b) for b in bench_rows],
        "scarcity": _json_meta(conn, "scarcity"),
        "eval_split": _json_meta(conn, "eval_split"),
        "top": top_flagged(conn, limit=200),
        "archive": act.archive_state(conn),
        "last_scan": dict(last_scan) if last_scan else None,
        "api_port": API_PORT,
    }


def full_run(roots: list[str] | None = None, *, horizon: int = REOPEN_WINDOW_DAYS,
             mode: str = "real", conn=None) -> dict:
    own = conn is None
    conn = conn or connect()
    init_db(conn)
    set_meta(conn, "data_mode", mode)
    started = time.time()
    scanned = None
    if mode == "real":
        target = [Path(r).expanduser().resolve() for r in (roots or DEFAULT_ROOTS)]
        target = [t for t in target if t.exists()]
        scanned = scanner.scan(conn, target, use_index=True, workers=12)
    bench = run_bench(conn, horizon=horizon)
    if bench.get("scarcity") is not None:
        set_meta(conn, "scarcity", json.dumps(bench["scarcity"]))
    set_meta(conn, "eval_split", json.dumps(bench.get("split") or {}))
    scoring = score_live(conn, horizon=horizon)
    out = state(conn)
    out["timings"] = {"scan": getattr(scanned, "as_dict", lambda: None)(),
                      "bench_seconds": round(bench.get("wall", 0) or 0, 2),
                      "total_seconds": round(time.time() - started, 1)}
    out["scoring"] = scoring
    out["bench_detail"] = bench
    return out
