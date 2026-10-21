"""Disk forecasting from arithmetic.

The model never gets to guess a number. Free space, file sizes and creation dates
come from stat() and disk_usage(); the projection is a robust linear fit. TabPFN is
offered a second, learned projection of the same series because Prior Labs describes
the product as forecasting, but the headline date is always the one a person could
check with a calculator.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import median

from .store import parse_ts

DAY = "%Y-%m-%d"

#: Below this, "the disk is filling" is not a claim worth printing. 0.05 GB/day is
#: about 350 MB a day, smaller than a single phone backup.
MIN_GROWTH_GB_PER_DAY = 0.05

#: A projection further out than this is an exercise in extrapolation, not a forecast.
MAX_HORIZON_DAYS = 3650


def _day(dt: datetime) -> str:
    return dt.strftime(DAY)


def created_curve(conn, volume: str | None = None, context: str = "personal") -> list[dict]:
    """Reconstruct the growth curve: bytes added per day, from creation timestamps.

    This is a reconstruction, not a measurement. Deletions are invisible to it, so it
    over-states growth. Dormant says so in the UI and lets nightly snapshots take over.
    """
    sql = "SELECT created_at, size_bytes FROM files WHERE 1=1"
    args: list = []
    if volume:
        sql += " AND volume = ?"
        args.append(volume)
    if context:
        sql += " AND context = ?"
        args.append(context)
    per_day: dict[str, int] = defaultdict(int)
    for row in conn.execute(sql, args):
        dt = parse_ts(row["created_at"])
        if dt:
            per_day[_day(dt)] += row["size_bytes"]
    if not per_day:
        return []
    start = min(per_day)
    end = max(per_day)
    out: list[dict] = []
    cumulative = 0
    day = datetime.strptime(start, DAY).replace(tzinfo=timezone.utc)
    stop = datetime.strptime(end, DAY).replace(tzinfo=timezone.utc)
    while day <= stop:
        key = _day(day)
        cumulative += per_day.get(key, 0)
        out.append({"date": key, "bytes_added": per_day.get(key, 0), "cumulative_bytes": cumulative})
        day += timedelta(days=1)
    return out


def snapshot_series(conn, volume: str | None = None) -> list[dict]:
    sql = "SELECT taken_at, total_bytes, free_bytes, used_bytes FROM snapshots"
    args: list = []
    if volume:
        sql += " WHERE volume = ?"
        args.append(volume)
    sql += " ORDER BY taken_at"
    return [{"date": r["taken_at"][:10], "free_bytes": r["free_bytes"],
             "total_bytes": r["total_bytes"], "used_bytes": r["used_bytes"]}
            for r in conn.execute(sql, args)]


def _theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Median-of-pairwise-slopes. Two data points a week apart should not wreck it."""
    slopes = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        return 0.0, ys[-1] if ys else 0.0
    slope = median(slopes)
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept


def forecast(conn, volume: str = "/", window_days: int = 365,
             warn_free_gb: float = 10.0) -> dict:
    """Project the date the volume reaches a warning level of free space."""
    snaps = snapshot_series(conn, volume)
    if not snaps:
        return {"volume": volume, "available": False, "reason": "no snapshots yet"}

    free_now = snaps[-1]["free_bytes"]
    total = snaps[-1]["total_bytes"]

    # Prefer measured history once the nightly runs have accumulated it. Nine scans
    # taken tonight are not a trend, they are one number repeated, so distinct days
    # matter more than row count.
    distinct_days = len({s["date"] for s in snaps})
    measured = len(snaps) >= 3 and distinct_days >= 2
    basis_note = None
    if measured:
        xs = [(parse_ts(s["date"]) - parse_ts(snaps[0]["date"])).days for s in snaps]
        ys = [s["used_bytes"] / 1e9 for s in snaps]
        source = "measured nightly snapshots"
    else:
        curve = created_curve(conn, volume=volume)
        if not curve:
            return {"volume": volume, "available": False, "reason": "no file history"}
        trimmed = curve[-window_days:]
        xs = [i for i in range(len(trimmed))]
        ys = [c["cumulative_bytes"] / 1e9 for c in trimmed]
        source = "reconstructed from file creation dates"
        basis_note = ("lower bound: a creation-date curve only sees files that still exist, "
                      "so deletions are invisible to it")

    if len(xs) < 2:
        return {"volume": volume, "available": False, "reason": "not enough history"}

    slope_gb_day, intercept = _theil_sen([float(x) for x in xs], ys)
    # Re-base the fit so it terminates at today's measured usage, not the fit's own value.
    offset_gb = (total - free_now) / 1e9 - (slope_gb_day * xs[-1] + intercept)
    adjusted_intercept = intercept + offset_gb

    days_to_warn = None
    full_date = None
    if slope_gb_day >= MIN_GROWTH_GB_PER_DAY:
        remaining_gb = free_now / 1e9 - warn_free_gb
        projected = remaining_gb / slope_gb_day
        if 0 < projected <= MAX_HORIZON_DAYS:
            days_to_warn = projected
            full_date = (datetime.now(timezone.utc) + timedelta(days=int(projected))).date().isoformat()
        else:
            basis_note = (basis_note or "") + ("; " if basis_note else "") + \
                f"projection exceeds {MAX_HORIZON_DAYS} days and is not shown"
    else:
        basis_note = (basis_note or "") + ("; " if basis_note else "") + \
            f"growth {slope_gb_day:.3f} GB/day is below the {MIN_GROWTH_GB_PER_DAY} GB/day reporting floor"

    projections = {}
    for horizon in (14, 30, 60, 90):
        used_h = slope_gb_day * (xs[-1] + horizon) + adjusted_intercept
        projections[f"free_in_{horizon}d_gb"] = round(max(total / 1e9 - used_h, 0.0), 2)

    return {
        "volume": volume,
        "available": True,
        "source": source,
        "basis_note": basis_note,
        "total_gb": round(total / 1e9, 1),
        "free_gb": round(free_now / 1e9, 1),
        "used_gb": round((total - free_now) / 1e9, 1),
        "free_pct": round(free_now / total * 100, 1),
        "growth_gb_per_day": round(slope_gb_day, 4),
        "growth_gb_per_week": round(slope_gb_day * 7, 3),
        "days_to_warning": int(days_to_warn) if days_to_warn and days_to_warn > 0 else None,
        "full_date": full_date,
        "warn_free_gb": warn_free_gb,
        "snapshots": len(snaps),
        "history_days": len(xs),
        "projections": projections,
        "curve": [{"date": c["date"], "cumulative_gb": round(c["cumulative_bytes"] / 1e9, 3)}
                  for c in created_curve(conn, volume=volume)][-180:],
    }


def tabpfn_free_space_projection(conn, volume: str = "/", horizon_days: int = 30) -> dict:
    """Second opinion: TabPFN-2 regression over the same daily series."""
    curve = created_curve(conn, volume=volume)
    snaps = snapshot_series(conn, volume)
    if len(curve) < 30 or not snaps:
        return {"available": False, "reason": "series too short"}
    total = snaps[-1]["total_bytes"]
    xs = list(range(len(curve)))
    # Target: cumulative used bytes, expressed from the reconstructed curve.
    y = [c["cumulative_bytes"] / 1e9 for c in curve]
    feats = [[i, i * i / max(len(curve), 1), y[i - 1] if i else 0.0,
              (y[i - 1] - y[i - 2]) if i > 1 else 0.0, (y[i - 1] - y[max(i - 7, 0)]) / max(min(i, 7), 1)]
             for i in xs]
    try:
        import time as _t

        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNRegressor

        from .config import PATHS

        ckpt = hf_hub_download(repo_id="Prior-Labs/TabPFN-v2-reg",
                               filename="tabpfn-v2-regressor.ckpt",
                               local_dir=str(PATHS.checkpoint_dir))
        reg = TabPFNRegressor(model_path=ckpt, device="auto", n_estimators=2,
                              inference_precision="auto", ignore_pretraining_limits=True)
        t0 = _t.time()
        reg.fit(feats, y)
        future = [xs[-1] + horizon_days, (xs[-1] + horizon_days) ** 2 / max(len(curve), 1),
                  y[-1], (y[-1] - y[-2]), (y[-1] - y[max(len(y) - 8, 0)]) / 7]
        pred = float(reg.predict([future])[0])
        dt = _t.time() - t0
        return {"available": True, "predicted_used_gb": round(pred, 2),
                "predicted_free_gb": round(max((total - pred * 1e9) / 1e9, 0.0), 2),
                "seconds": round(dt, 2), "n_train": len(xs)}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
