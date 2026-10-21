"""Model providers: an open-weight tabular foundation model, and the baselines it has to beat.

TabPFN is not a language model pretending to do arithmetic. It is a transformer that
was trained once, offline, on millions of synthetic tabular datasets, and then does
in-context inference: your labelled rows go in as context, out comes a posterior over
classes. No training run, no gradient step on your data, no per-token bill, nothing
uploaded. That is the entire reason a solo machine can carry this product.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

from .config import PATHS, PRECISION_TARGET, REOPEN_WINDOW_DAYS, RULE_DORMANT_AFTER_DAYS
from .dataset import Split
from .features import rule_score

#: TabPFN refuses CPU inference above 1000 rows unless you say you mean it.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

V2_CLASSIFIER_URL = ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt")

#: Opt-in table. DORMANT_TABPFN_MODEL=3.5-fast switches the product to the version
#: Prior Labs recommends. It is not the default, and that is a product decision:
#: the fast checkpoint is 334 MB against 28 MB, it sits behind a licence-acceptance
#: wall that needs an account, and this whole project exists to run on a laptop with
#: no network and no sign-in. The scoreboard is more interesting with both on it.
CHECKPOINTS = {
    "v2": ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt", "tabpfn-v2-classifier.ckpt"),
    "3.5-fast": ("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-fast-20260909.safetensors",
                 "tabpfn-v3.5-fast-20260909.safetensors"),
    "3.5": ("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors",
            "tabpfn-v3.5-20260909.safetensors"),
}


def model_choice() -> str:
    return os.environ.get("DORMANT_TABPFN_MODEL", "v2").strip().lower()


def ensure_checkpoint(choice: str | None = None) -> str | None:
    """Fetch the selected weights once, locally. Returns None if unavailable.

    The v2 checkpoint is ungated. The 3.x checkpoints require you to accept the
    TABPFN-3.5 Non-Commercial License and authenticate to Hugging Face first, which
    is a human step; the error message says so instead of silently falling back.
    """
    key = (choice or model_choice())
    if key not in CHECKPOINTS:
        key = "v2"
    repo, filename, local_name = CHECKPOINTS[key]
    target = PATHS.checkpoint_dir / local_name
    if target.exists():
        return str(target)
    try:
        from huggingface_hub import hf_hub_download
        PATHS.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(PATHS.checkpoint_dir))
        if got != str(target):
            os.replace(got, target)
        return str(target)
    except Exception as exc:
        if key != "v2" and "gated" in str(exc).lower() or "401" in str(exc):
            raise RuntimeError(
                f"{key} weights need licence acceptance: agree at "
                f"https://huggingface.co/{repo} then run `hf auth login`. "
                f"Set DORMANT_TABPFN_MODEL=v2 to use the ungated 28 MB checkpoint."
            ) from exc
        return None


@dataclass
class BenchResult:
    model: str
    n_train: int
    n_test: int
    auc: float | None = None
    pr_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    threshold: float | None = None
    reclaimed_gb: float | None = None
    wrong: int | None = None
    fit_s: float | None = None
    predict_s: float | None = None
    peak_mb: float | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def metrics(y_true: list[int], scores: list[float], target: float = PRECISION_TARGET) -> dict:
    """Pick the operating point.

    First preference: the most generous threshold that still hits the target
    precision. If the model cannot reach the target at any threshold, fall back to
    the best precision-weighted point and report the precision that was actually
    measured, instead of quietly printing the aspiration.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict = {"n": len(y_true), "positives": sum(y_true)}
    if len(set(y_true)) < 2:
        out.update(auc=None, pr_auc=None, best=None, fallback=None)
        return out
    out["auc"] = float(roc_auc_score(y_true, scores))
    out["pr_auc"] = float(average_precision_score(y_true, scores))
    best = None
    fallback = None
    # Only thresholds that express a real belief count. With a 91% dormant base rate,
    # a threshold of 0.03 "meets 95% precision" by flagging everything, which is not
    # a decision, it is a description of the majority class.
    for thr in [round(0.5 + x * 0.02, 3) for x in range(25)]:
        pred = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, t in zip(pred, y_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred, y_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred, y_true) if p == 0 and t == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        cand = {"threshold": thr, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
        if prec >= target and rec > 0 and (best is None or rec > best["recall"]):
            best = cand
        score = prec * (rec ** 0.4)
        if tp and (fallback is None or score > fallback.get("score", 0)):
            fallback = {**cand, "score": score}
    out["best"] = best
    out["fallback"] = fallback
    return out


class TabPFNProvider:
    name = "tabpfn"

    def __init__(self, checkpoint: str | None = None, n_estimators: int = 4):
        self.checkpoint = checkpoint
        self.n_estimators = n_estimators
        self.available = False
        self._clf = None
        self._error = ""

    def load(self):
        if self._clf is not None:
            return self._clf
        ckpt = self.checkpoint or ensure_checkpoint()
        if not ckpt:
            self._error = "no checkpoint available offline"
            return None
        try:
            from tabpfn import TabPFNClassifier
            self._clf = TabPFNClassifier(
                model_path=ckpt,
                device="auto",
                n_estimators=self.n_estimators,
                inference_precision="auto",
                ignore_pretraining_limits=True,
                random_state=0,
            )
            self.available = True
            return self._clf
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = f"{type(exc).__name__}: {exc}"[:200]
            self._clf = None
            return None

    def fit_predict(self, split: Split) -> tuple[list[float], float, float]:
        X, y, Xt, _ = split.arrays()
        clf = self.load()
        if clf is None:
            raise RuntimeError(self._error or "tabpfn unavailable")
        t0 = time.time()
        clf.fit(X, y)
        t1 = time.time()
        probs = clf.predict_proba(Xt)[:, 1]
        t2 = time.time()
        return [float(p) for p in probs], t1 - t0, t2 - t1


class SklearnProvider:
    """Deterministic baselines. If the foundation model cannot beat these, it is decoration."""

    def __init__(self, kind: str = "hist_gb"):
        self.kind = kind
        self.name = kind

    def fit_predict(self, split: Split) -> tuple[list[float], float, float]:
        X, y, Xt, _ = split.arrays()
        t0 = time.time()
        if self.kind == "hist_gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        elif self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
        elif self.kind == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0)
        else:
            raise ValueError(self.kind)
        model.fit(X, y)
        t1 = time.time()
        probs = model.predict_proba(Xt)[:, 1]
        t2 = time.time()
        return [float(p) for p in probs], t1 - t0, t2 - t1


def rule_provider(split: Split) -> tuple[list[float], float, float]:
    """The heuristic a person writes by hand: ninety days untouched means it is dead.

    Returned in the model's own units (probability of reopening) so the dormancy
    flip downstream is applied uniformly to every provider.
    """
    scores = [0.0 if rule_score(r, RULE_DORMANT_AFTER_DAYS) == 1.0 else 1.0 for r in split.test]
    return scores, 0.0, 0.0


def peak_rss_mb() -> float:
    import subprocess
    import uuid
    probe = uuid.uuid4().hex[:8]
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) / 1024
    except Exception:
        return 0.0


def cap_holdout(split: Split, limit: int = 1800) -> Split:
    """TabPFN is slow on CPU by design, so the bench runs on the most recent slice.

    Taking the newest rows keeps the time-split property: the model still only ever
    sees history that precedes everything it is scored on.
    """
    if split.n_test <= limit:
        return split
    return Split(train=split.train, test=split.test[-limit:], cut=split.cut,
                 feature_names=list(split.feature_names))


def _stride(items: list, n: int) -> list:
    if n >= len(items) or n <= 0:
        return items
    step = math.ceil(len(items) / n)
    return items[::step][:n]


def scarcity_curve(conn, sizes: tuple[int, ...] = (120, 240, 480, 960, 1920),
                   horizon: int = REOPEN_WINDOW_DAYS, eval_limit: int = 1400) -> list[dict]:
    """The question a personal corpus actually asks: how well does each model do when
    there is almost nothing to learn from? Stride sampling keeps the samples spread
    across the whole timeline rather than bunched at one end."""
    from .dataset import time_split
    from .features import build_rows

    rows = build_rows(conn, horizon=horizon)
    full = time_split(rows, frac=0.7)
    if full.n_test < 20:
        return []
    test = cap_holdout(full, eval_limit).test
    labels = [int(r.label or 0) for r in test]
    out: list[dict] = []
    for n in sizes:
        if n > len(full.train):
            continue
        sub = Split(train=_stride(full.train, n), test=test, cut=full.cut,
                    feature_names=list(full.feature_names))
        row: dict = {"n_train": n, "n_test": len(test), "positives": sum(labels),
                     "positive_rate": round(sum(labels) / max(len(labels), 1), 3)}
        for name, factory in (("logistic", lambda: SklearnProvider("logistic")),
                              ("random_forest", lambda: SklearnProvider("random_forest")),
                              ("hist_gb", lambda: SklearnProvider("hist_gb")),
                              ("tabpfn", lambda: TabPFNProvider(n_estimators=2))):
            try:
                scores, _, _ = factory().fit_predict(sub)
                row[f"{name}_auc"] = round(metrics(labels, scores).get("auc") or 0.0, 4)
            except Exception:
                row[f"{name}_auc"] = None
        wins = [k for k in row if k.endswith("_auc") and row[k] is not None]
        row["best"] = max(wins, key=lambda k: row[k]).replace("_auc", "") if wins else None
        out.append(row)
    return out


def bench(conn, *, providers: list[str] | None = None, horizon: int = REOPEN_WINDOW_DAYS) -> dict:
    """Run every model over the same time-split holdout and record the scoreboard."""
    from .features import build_rows
    from .dataset import rolling_origin_folds, time_split

    rows = build_rows(conn, horizon=horizon)
    split = cap_holdout(time_split(rows, frac=0.7))
    wanted = providers or ["rule_90d", "logistic", "random_forest", "hist_gb", "tabpfn"]
    results: list[BenchResult] = []
    if split.n_test < 5:
        return {"split": {"train": split.n_train, "test": split.n_test, "cut": split.cut},
                "results": [], "error": "holdout too small; run more scans"}

    for name in wanted:
        try:
            if name == "rule_90d":
                scores, fit_s, pred_s = rule_provider(split)
            elif name == "tabpfn":
                prov = TabPFNProvider(n_estimators=2)
                scores, fit_s, pred_s = prov.fit_predict(split)
            else:
                prov = SklearnProvider(name)
                scores, fit_s, pred_s = prov.fit_predict(split)
        except Exception as exc:
            results.append(BenchResult(model=name, n_train=split.n_train, n_test=split.n_test,
                                       notes=f"failed: {exc}"[:160]))
            continue
        reopen_labels = [int(r.label or 0) for r in split.test]
        # The product decision is "is this dormant", which is the complement of the
        # model's positive class. Scoring the wrong orientation turns a good model
        # into an AUC of 0.33 and an unusable threshold, so it is done explicitly here.
        dormancy = [1 - s for s in scores]
        dormant_labels = [1 - t for t in reopen_labels]
        m = metrics(dormant_labels, dormancy)
        point = m.get("best") or m.get("fallback") or {}
        source = "target precision met" if m.get("best") else ("best measured point" if m.get("fallback") else "none")
        reclaimed = 0.0
        wrong = None
        if point:
            thr = point["threshold"]
            for r, s in zip(split.test, dormancy):
                if s >= thr and (r.label or 0) == 0:
                    reclaimed += r.size_bytes
                if s >= thr and (r.label or 0) == 1:
                    wrong = (wrong or 0) + 1
        results.append(BenchResult(
            model=name, n_train=split.n_train, n_test=split.n_test, auc=m.get("auc"),
            pr_auc=m.get("pr_auc"), precision=point.get("precision"), recall=point.get("recall"),
            threshold=point.get("threshold"), reclaimed_gb=reclaimed / 1e9 if point else None,
            wrong=wrong, fit_s=round(fit_s, 3), predict_s=round(pred_s, 3),
            peak_mb=round(peak_rss_mb(), 1),
            notes=f"dormancy {source}: tp={point.get('tp')} fp={point.get('fp')} fn={point.get('fn')}",
        ))
        conn.execute(
            """INSERT INTO bench(ts, model, n_train, n_test, auc, pr_auc, precision, recall,
               reclaimed, fit_s, predict_s, peak_mb, notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), name, split.n_train, split.n_test,
             m.get("auc"), m.get("pr_auc"), point.get("precision"), point.get("recall"),
             reclaimed / 1e9 if point else None, fit_s, pred_s, peak_rss_mb(),
             json.dumps({"threshold": point.get("threshold"), "tp": point.get("tp"),
                         "fp": point.get("fp"), "fn": point.get("fn"), "source": source})),
        )
    conn.commit()
    folds = rolling_origin_folds(rows, folds=3)
    fold_auc = {}
    for f in folds:
        try:
            scores, _, _ = TabPFNProvider(n_estimators=2).fit_predict(cap_holdout(f, 1200))
            fold_auc[f.cut[:10]] = metrics([int(r.label or 0) for r in f.test], scores).get("auc")
        except Exception:
            fold_auc[f.cut[:10]] = None
    return {
        "split": {"train": split.n_train, "test": split.n_test, "cut": split.cut,
                  "positive_rate": round(split.positive_rate, 4)},
        "results": [r.as_dict() for r in results],
        "folds": [{"cut": f.cut, "n_train": f.n_train, "n_test": f.n_test, "auc": v}
                  for f, v in zip(folds, [fold_auc.get(f.cut[:10]) for f in folds])],
        "scarcity": scarcity_curve(conn, horizon=horizon),
        "rows_total": len(rows),
        "rows_labelled": len([r for r in rows if r.label is not None]),
    }
