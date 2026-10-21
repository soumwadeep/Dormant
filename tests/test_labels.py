"""Labels, censoring and leakage.

These are the tests that decide whether the benchmark numbers on the dashboard mean
anything, so they are the ones worth reading.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import add_file
from dormant.dataset import labelled, time_split
from dormant.features import FEATURES, build_rows, scoring_rows

NOW = datetime.now(timezone.utc)


def iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")


def day(rounding: int = 1) -> timedelta:
    """Tolerate the seconds between module import and the code's own now()."""
    return timedelta(days=rounding)


def rows_for(conn, suffix: str) -> list:
    return [r for r in build_rows(conn) if r.path.endswith(suffix)]


def test_reopen_inside_window_is_positive(fresh_db):
    """A file opened 40 days ago that came back 20 days later must be a positive."""
    add_file(fresh_db, "/Users/demo/Documents/Tax/returns_2024.pdf", created=iso(400),
             opens=[iso(40), iso(20)])
    rows = rows_for(fresh_db, "returns_2024.pdf")
    assert rows, "the observation points inside a 20-day streak must be sampled"
    assert any(r.label == 1 for r in rows), "an as-of within 90 days of the next open is a positive"


def test_long_gap_produces_both_classes(fresh_db):
    """A 400-day gap is mostly negative, and turns positive in its last 90 days.

    That shape is the whole product: dormancy is not a cliff, it is a decaying
    probability, and a fixed "90 days idle" rule cannot see the difference.
    """
    add_file(fresh_db, "/Users/demo/Downloads/old_claim.pdf", created=iso(800),
             opens=[iso(700), iso(300)])
    rows = rows_for(fresh_db, "old_claim.pdf")
    assert rows
    assert any(r.label == 0 for r in rows), "most of a 400-day gap is genuinely dormant"
    assert any(r.label == 1 for r in rows), "the last 90 days before a return are not"
    open_dates = [datetime.fromisoformat(iso(d)) for d in (700, 300)]
    for r in rows:
        as_of = datetime.fromisoformat(r.as_of)
        later = [o for o in open_dates if o > as_of]
        expected = 1 if later and (min(later) - as_of).days <= 90 else 0
        assert r.label == expected, f"{r.as_of}: label must be the distance to the next open"


def test_never_opened_file_yields_only_negatives(fresh_db):
    """A file that has never been opened and is still unopened cannot be a positive."""
    add_file(fresh_db, "/Users/demo/Downloads/forgotten.zip", created=iso(500), opens=None)
    rows = rows_for(fresh_db, "forgotten.zip")
    assert rows, "never-opened files must still be scored, that is the point of the product"
    assert all(r.label == 0 for r in rows)


def test_no_row_is_younger_than_the_dormancy_floor(fresh_db):
    add_file(fresh_db, "/Users/demo/Desktop/fresh.png", created=iso(5), opens=[iso(4)])
    for r in build_rows(fresh_db):
        assert r.features["days_since_last_open"] >= 21.0 - 1e-6


def test_features_never_look_ahead(fresh_db):
    """At an as-of date, open_count and recency must describe only opens before it."""
    opens_days = (200, 150, 100)
    add_file(fresh_db, "/Users/demo/Documents/Work/plan.docx", created=iso(300),
             opens=[iso(d) for d in opens_days])
    for r in rows_for(fresh_db, "plan.docx"):
        as_of = datetime.fromisoformat(r.as_of)
        seen = sum(1 for d in opens_days if NOW - timedelta(days=d) <= as_of)
        assert r.features["open_count"] == float(seen)
        if seen:
            expected_gap = (as_of - (NOW - timedelta(days=opens_days[seen - 1]))).days
        else:
            # Before the first open, "since last open" is by definition the age.
            expected_gap = (as_of - (NOW - timedelta(days=300))).days
        assert abs(r.features["days_since_last_open"] - expected_gap) <= 1.0


def test_censored_rows_are_dropped_not_guessed(fresh_db):
    """An outcome window that has not closed yet must not appear anywhere."""
    add_file(fresh_db, "/Users/demo/Documents/Invoices/new_recent.pdf", created=iso(1000),
             opens=[iso(30)])
    rows = build_rows(fresh_db, horizon=90)
    for r in rows:
        as_of = datetime.fromisoformat(r.as_of)
        assert NOW - as_of >= timedelta(days=90) - day(), "row observed less than a horizon ago is censored"
        assert r.label is not None


def test_time_split_respects_the_arrow_of_time(fresh_db):
    for i in range(8):
        add_file(fresh_db, f"/Users/demo/Documents/s{i}.pdf", created=iso(700 - i * 30),
                 opens=[iso(600 - i * 30), iso(300 - i * 20)])
    split = time_split(build_rows(fresh_db), frac=0.6)
    assert split.train and split.test
    assert max(r.as_of for r in split.train) <= min(r.as_of for r in split.test)


def test_frac_one_leaves_no_test_slice(fresh_db):
    for i in range(8):
        add_file(fresh_db, f"/Users/demo/Documents/full{i}.pdf", created=iso(700 - i * 30),
                 opens=[iso(600 - i * 30), iso(300 - i * 20)])
    split = time_split(labelled(build_rows(fresh_db)), frac=1.0)
    assert split.test == [] and split.train, "frac=1.0 means use everything as context"


def test_scoring_rows_exclude_project_trees(fresh_db):
    add_file(fresh_db, "/Volumes/x/Repos/app/node_modules/left-pad.js", created=iso(400),
             context="project")
    add_file(fresh_db, "/Users/demo/Documents/keep.pdf", created=iso(400), context="personal")
    paths = [r.path for r in scoring_rows(fresh_db)]
    assert any(p.endswith("keep.pdf") for p in paths)
    assert not any("node_modules" in p for p in paths)


def test_feature_list_matches_rows(fresh_db):
    add_file(fresh_db, "/Users/demo/Documents/match.pdf", created=iso(400), opens=[iso(200)])
    for r in build_rows(fresh_db):
        assert set(r.features) == set(FEATURES)
