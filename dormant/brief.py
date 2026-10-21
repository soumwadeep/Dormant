"""The two-line brief.

Every number in these sentences is computed. The model is never asked to restate
data, because a language model that rounds 8.6 GB to "about 9 GB" is a bug, not a
feature. Anything Dormant can calculate with an integer it calculates with an
integer; prose is a template over those integers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import REOPEN_WINDOW_DAYS


def _gb(n: int | float) -> str:
    return f"{n / 1e9:.1f} GB"


def build_brief(count: int, bytes_total: float, mean_confidence: float, holdout_precision: float | None,
               forecast: dict, reclaimed_days: float | None, top_dir: str | None,
               horizon: int = REOPEN_WINDOW_DAYS, model_name: str = "TabPFN-2") -> dict:
    """Line 1 states the model's own average confidence on the listed files. The
    measured holdout precision is reported separately, because the two numbers mean
    different things and only one of them is a claim about accuracy."""
    line1 = (
        f"Dormant found {count:,} files worth {_gb(bytes_total)} it is {mean_confidence:.0%} sure "
        f"you will not reopen inside {horizon} days."
    )
    if holdout_precision:
        line1 += f" On the held-out slice it was right {holdout_precision:.0%} of the time it said stop."
    if forecast.get("available") and forecast.get("full_date"):
        tail = (f"At {_gb(int(forecast['growth_gb_per_week'] * 1e9))} a week of growth, "
                f"{forecast['volume']} hits {forecast['warn_free_gb']:.0f} GB free on {forecast['full_date']}.")
        if forecast.get("basis_note"):
            tail += f" Treat that as a floor: {forecast['basis_note'].split(':')[0]}."
        if reclaimed_days:
            tail += f" Archiving this list buys you about {reclaimed_days:.0f} more days."
        line2 = tail
    elif forecast.get("available"):
        line2 = (f"{forecast['volume']} has {_gb(int(forecast['free_gb'] * 1e9))} free and no measurable "
                 f"fill rate yet. Run Dormant nightly and the date arrives on its own.")
    else:
        line2 = f"Run Dormant again tomorrow and it will have a growth rate to project from."

    detail = []
    if top_dir:
        detail.append(f"largest cluster: {top_dir}")
    detail.append(f"model: {model_name}, {horizon}-day horizon")
    if forecast.get("basis_note"):
        detail.append("disk projection is a floor: " + forecast["basis_note"].split(":")[0])
    return {
        "line1": line1,
        "line2": line2,
        "detail": detail,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numbers_computed_not_generated": True,
    }
