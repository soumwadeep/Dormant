"""Evaluation integrity, forecast arithmetic, API contract and the privacy guardrail.

These exist because a project that reports an AUC it derived from a leak, or a disk
date it derived from a unit bug, is worse than a project that reports nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import add_file
from dormant.dataset import labelled, time_split
from dormant.features import build_rows
from dormant.forecast import _theil_sen, created_curve, forecast
from dormant.model import metrics

NOW = datetime.now(timezone.utc)
iso = lambda d: (NOW - timedelta(days=d)).isoformat(timespec="seconds")  # noqa: E731


# ------------------------------------------------------------------- operating point

def test_perfect_separation_scores_one():
    """Inputs are dormancy scores against dormancy labels, which is how bench calls it."""
    m = metrics([1, 1, 1, 0, 0], [0.95, 0.93, 0.91, 0.2, 0.1])
    assert m["auc"] == 1.0
    assert m["best"] and m["best"]["precision"] == 1.0


def test_thresholds_below_half_are_not_eligible():
    """Flagging everything also hits high precision when 91% of rows are dormant.

    That is not a decision, so the search is not allowed to land there.
    """
    labels = [0] * 91 + [1] * 9
    scores = [0.99] * 91 + [0.98] * 9  # a constant predictor: dormancy 0.01 vs 0.02
    m = metrics(labels, scores)
    assert m["best"] is None
    if m["fallback"]:
        assert m["fallback"]["threshold"] >= 0.5


def test_single_class_holdout_has_no_auc():
    m = metrics([0, 0, 0], [0.9, 0.8, 0.7])
    assert m["auc"] is None and m["pr_auc"] is None


# ------------------------------------------------------------------- censoring

def test_rows_observed_recently_are_never_labelled(fresh_db):
    """An outcome window still open is dropped, it is not silently counted as negative."""
    add_file(fresh_db, "/Users/demo/Documents/x/fresh_tax.pdf", created=iso(600), opens=[iso(5)])
    rows = build_rows(fresh_db, horizon=90)
    assert not any(datetime.fromisoformat(r.as_of) > NOW - timedelta(days=89) for r in rows), \
        "as-of must be at least a horizon in the past"
    assert all(r.label is not None for r in labelled(rows))


def test_split_has_no_temporal_overlap(fresh_db):
    add_file(fresh_db, "/Users/demo/Documents/a.pdf", created=iso(700), opens=[iso(600), iso(400)])
    add_file(fresh_db, "/Users/demo/Documents/b.pdf", created=iso(650), opens=[iso(300), iso(120)])
    add_file(fresh_db, "/Users/demo/Documents/c.zip", created=iso(500), opens=None)
    split = time_split(build_rows(fresh_db), frac=0.5)
    if split.train and split.test:
        assert max(r.as_of for r in split.train) <= min(r.as_of for r in split.test)


# ------------------------------------------------------------------- forecast maths

def test_theil_sen_recovers_a_known_slope():
    xs = [float(i) for i in range(20)]
    ys = [3.0 + 2.5 * x for x in xs]
    slope, intercept = _theil_sen(xs, ys)
    assert abs(slope - 2.5) < 1e-6 and abs(intercept - 3.0) < 1e-6


def test_outliers_do_not_move_the_projection():
    xs = [float(i) for i in range(20)]
    clean = [3.0 + 2.5 * x for x in xs]
    noisy = list(clean)
    noisy[7] += 900.0
    s_clean, _ = _theil_sen(xs, clean)
    s_noisy, _ = _theil_sen(xs, noisy)
    assert abs(s_noisy - s_clean) < 0.25


def test_projections_are_gigabytes_not_bytes(fresh_db):
    """Regression test: free space in the projections once came out as the whole volume."""
    total = 250_000_000_000
    for back in range(30, 0, -1):
        used = 100_000_000_000 + (30 - back) * 2_000_000_000
        fresh_db.execute(
            "INSERT INTO snapshots(taken_at, volume, total_bytes, free_bytes, used_bytes) VALUES(?,?,?,?,?)",
            ((NOW - timedelta(days=back)).isoformat(timespec="seconds"), "/proj-test",
             total, total - used, used))
    fresh_db.commit()
    f = forecast(fresh_db, volume="/proj-test")
    assert f["available"], f
    for k, v in f["projections"].items():
        assert 0 <= v <= total / 1e9, f"{k} left the building: {v}"
    assert f["projections"]["free_in_30d_gb"] < f["projections"]["free_in_14d_gb"], \
        "a filling disk must project monotonically"
    assert f["full_date"], "a disk filling 2 GB a day has to reach the warning line"


def test_created_curve_is_cumulative_and_non_negative(fresh_db):
    add_file(fresh_db, "/Users/demo/Documents/one.pdf", created=iso(120), size=5_000)
    add_file(fresh_db, "/Users/demo/Documents/two.pdf", created=iso(60), size=7_000)
    curve = created_curve(fresh_db, volume=None)
    assert curve and all(c["cumulative_bytes"] >= 0 for c in curve)
    assert curve[-1]["cumulative_bytes"] >= curve[0]["cumulative_bytes"]


# ------------------------------------------------------------------- demo privacy

def test_demo_corpus_contains_no_real_paths(fresh_db):
    from dormant import demo

    demo.seed(conn=fresh_db, rng_seed=3)
    paths = [r[0] for r in fresh_db.execute("SELECT path FROM files")]
    assert paths
    assert all(p.startswith("/Users/demo/") for p in paths), "a seeded demo must never name a real home folder"


# ------------------------------------------------------------------- api contract

def test_api_serves_state_and_refuses_unconfirmed_archive(fresh_db):
    from fastapi.testclient import TestClient

    import api.app as api_app

    api_app._conn = fresh_db
    client = TestClient(api_app.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] and body["network_egress"] == "none"

    preview = client.post("/api/archive", json={"top": 5, "max_gb": 1, "confirm": False})
    assert preview.status_code == 200 and preview.json()["confirm_required"]

    st = client.get("/api/state")
    assert st.status_code == 200
    payload = st.json()
    for key in ("brief", "forecast", "counts", "top", "threshold"):
        assert key in payload


def test_api_agent_never_moves_files_without_confirmation(fresh_db):
    from fastapi.testclient import TestClient

    import api.app as api_app

    api_app._conn = fresh_db
    client = TestClient(api_app.app)
    res = client.post("/api/agent", json={"text": "free up 12 gigabytes"}).json()
    assert res["intent"]
    for blocked in res["blocked"]:
        assert blocked["tool"] in ("archive", "undo")
        assert "confirm" in blocked["why"]


def test_api_health_reports_index_size(fresh_db):
    from fastapi.testclient import TestClient

    import api.app as api_app

    add_file(fresh_db, "/Users/demo/Documents/zz.pdf", created=iso(300))
    api_app._conn = fresh_db
    client = TestClient(api_app.app)
    assert client.get("/api/state").json()["counts"]["files"] >= 1


# ------------------------------------------------------------------- preview guardrails

def test_preview_requires_the_token(fresh_db, tmp_path):
    """File contents are the most sensitive thing this server can hand out."""
    from fastapi.testclient import TestClient

    import api.app as api_app

    p = tmp_path / "secret.pdf"
    p.write_bytes(b"%PDF-1.4 " + b"z" * 500)
    fid = add_file(fresh_db, str(p), created=iso(300), size=p.stat().st_size)
    api_app._conn = fresh_db
    client = TestClient(api_app.app)

    assert client.get(f"/api/file/{fid}/content").status_code == 403
    assert client.get(f"/api/file/{fid}/meta").status_code == 403
    ok = client.get(f"/api/file/{fid}/content?token={api_app.PREVIEW_TOKEN}")
    assert ok.status_code == 200 and ok.content.startswith(b"%PDF")


def test_preview_supports_range_requests(fresh_db, tmp_path):
    from fastapi.testclient import TestClient

    import api.app as api_app

    p = tmp_path / "clip.mp4"
    p.write_bytes(bytes(range(256)) * 40)
    fid = add_file(fresh_db, str(p), created=iso(300), size=p.stat().st_size)
    api_app._conn = fresh_db
    client = TestClient(api_app.app)
    r = client.get(f"/api/file/{fid}/content?token={api_app.PREVIEW_TOKEN}",
                   headers={"range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 10-19/")
    assert len(r.content) == 10


def test_index_html_carries_the_token_for_its_own_page(fresh_db):
    from fastapi.testclient import TestClient

    import api.app as api_app

    api_app._conn = fresh_db
    client = TestClient(api_app.app)
    html = client.get("/").text
    assert api_app.PREVIEW_TOKEN in html
    assert "__PREVIEW_TOKEN__" not in html


def test_api_delete_needs_the_phrase(fresh_db):
    from fastapi.testclient import TestClient

    import api.app as api_app

    add_file(fresh_db, "/Users/demo/Documents/keepme.pdf", created=iso(300))
    api_app._conn = fresh_db
    client = TestClient(api_app.app)
    res = client.post("/api/delete", json={"ids": [1], "phrase": "yes"}).json()
    assert res["confirm_required"] and res["phrase_expected"] == "DELETE"
    assert client.get("/api/state").json()["counts"]["files"] >= 1, "nothing was deleted"
