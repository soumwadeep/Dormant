"""Time-split datasets and rolling-origin folds.

Splitting observation rows randomly would leak: two rows from the same file share a
history, so a random split puts yesterday inside tomorrow. Dormant splits on the
observation date only, and validates with rolling origin, which is the version of
cross-validation that respects the arrow of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import REOPEN_WINDOW_DAYS
from .features import FEATURES, Row


@dataclass
class Split:
    train: list[Row]
    test: list[Row]
    cut: str
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def positive_rate(self) -> float:
        if not self.test:
            return 0.0
        return sum(r.label or 0 for r in self.test) / len(self.test)

    def arrays(self):
        X = [[r.features[k] for k in self.feature_names] for r in self.train]
        y = [int(r.label or 0) for r in self.train]
        Xt = [[r.features[k] for k in self.feature_names] for r in self.test]
        yt = [int(r.label or 0) for r in self.test]
        return X, y, Xt, yt


def labelled(rows: list[Row]) -> list[Row]:
    """Rows whose outcome window has fully closed. Censored rows are dropped, never guessed."""
    return [r for r in rows if r.label is not None]


def time_split(rows: list[Row], frac: float = 0.7) -> Split:
    rows = sorted(labelled(rows), key=lambda r: r.as_of)
    if len(rows) < 12:
        return Split(train=rows, test=[], cut="", feature_names=list(FEATURES))
    cut_idx = min(max(1, int(len(rows) * frac)), len(rows))
    if cut_idx >= len(rows):
        # frac=1.0 means "use everything as context", which is how live scoring fits.
        return Split(train=rows, test=[], cut=rows[-1].as_of, feature_names=list(FEATURES))
    cut_row = rows[cut_idx]
    return Split(train=rows[:cut_idx], test=rows[cut_idx:], cut=cut_row.as_of,
                 feature_names=list(FEATURES))


def rolling_origin_folds(rows: list[Row], folds: int = 3) -> list[Split]:
    """Expanding-window validation: train on all history up to a cut, test just after it."""
    rows = sorted(labelled(rows), key=lambda r: r.as_of)
    out: list[Split] = []
    if len(rows) < folds * 12:
        return out
    step = len(rows) // (folds + 1)
    for i in range(1, folds + 1):
        boundary = step * i
        if boundary >= len(rows) - 4:
            break
        train = rows[:boundary]
        test = rows[boundary:boundary + step]
        if len(train) < 20 or len(test) < 5:
            continue
        out.append(Split(train=train, test=test, cut=test[0].as_of, feature_names=list(FEATURES)))
    return out


def newest_label_date(rows: list[Row]) -> str | None:
    done = labelled(rows)
    return max((r.as_of for r in done), default=None) if done else None


def horizon_label(horizon: int = REOPEN_WINDOW_DAYS) -> str:
    return f"reopened within {horizon} days"
