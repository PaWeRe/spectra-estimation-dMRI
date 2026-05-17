"""Alternative-classifier replication for §6 step 1 of the 2026-05-14 session.

Question: does any non-linear classifier extract information from the 6
intermediate NUTS bins that the LR cannot? Threshold from the session notes:
"if any classifier beats the 2-feature NUTS-LR by Delta AUC >= 0.05 with
non-overlapping bootstrap CIs", Path A's "8-bin spectrum is effectively
2-dim" claim is in trouble.

Protocol mirrors recompute.loocv_auc:
- LOOCV, StandardScaler fit inside fold
- features.csv as the single source of truth (Session 8 NUTS posteriors)
- fixed hyperparameters per classifier (no per-fold tuning)
- bootstrap 95% CI on the resulting AUC (1000 iter, fixed seed)

Tasks:
- PZ tumor-vs-normal (n=81, 27/54)
- TZ tumor-vs-normal (n=68, 13/55)
- GGG>=3 on tumor-only (n=40 with known ggg)

Feature sets:
- ADC alone (raw rank, no classifier - reference baseline)
- 8-bin NUTS posterior means
- 2-bin NUTS {D=0.25, D=3.00}

Classifiers:
- LR (C=1.0, lbfgs)            -- same as recompute
- RF (n_estimators=300)
- GBM (n_estimators=300, lr=0.05, max_depth=3)
- SVM-RBF (C=1.0, gamma='scale', probability=True)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


FEATURES_CSV = Path("results/biomarkers/features.csv")
OUTPUT_CSV = Path("results/biomarkers/classifier_comparison.csv")

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
NUTS_ALL_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
NUTS_TWO_COLS = ["nuts_D_0.25", "nuts_D_3.00"]


def make_classifier(name: str):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=1,
        )
    if name == "GBM":
        return GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
        )
    if name == "SVM-RBF":
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    raise ValueError(name)


def loocv_predict(X: np.ndarray, y: np.ndarray, clf_name: str) -> np.ndarray:
    """Return LOOCV predicted positive-class probabilities."""
    n = len(y)
    y_pred = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = make_classifier(clf_name)
        clf.fit(Xtr, y[tr])
        y_pred[te] = clf.predict_proba(Xte)[0, 1]
    return y_pred


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def raw_rank_auc(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return score oriented so AUC >= 0.5 (matches recompute.raw_rank_auc)."""
    auc = roc_auc_score(y, score)
    return score if auc >= 0.5 else -score


def build_tasks(df: pd.DataFrame) -> dict:
    tasks = {}
    tasks["PZ"] = df[df["zone"] == "pz"].copy()
    tasks["PZ"]["_y"] = tasks["PZ"]["is_tumor"].astype(int)

    tasks["TZ"] = df[df["zone"] == "tz"].copy()
    tasks["TZ"]["_y"] = tasks["TZ"]["is_tumor"].astype(int)

    ggg = df[(df["is_tumor"]) & df["ggg"].notna() & (df["ggg"] != 0)].copy()
    ggg["_y"] = (ggg["ggg"] >= 3).astype(int)
    tasks["GGG"] = ggg
    return tasks


def evaluate(task_name: str, task_df: pd.DataFrame, feat_name: str,
             feat_cols: list, clf_name: str) -> dict:
    y = task_df["_y"].values
    if feat_cols is None:
        scores = raw_rank_auc(task_df["adc"].values, y)
    else:
        X = task_df[feat_cols].values
        scores = loocv_predict(X, y, clf_name)
    auc = roc_auc_score(y, scores)
    lo, hi = bootstrap_auc_ci(y, scores)
    return {
        "task": task_name,
        "n": len(y),
        "n_pos": int(y.sum()),
        "feature_set": feat_name,
        "classifier": clf_name,
        "auc": auc,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def main():
    df = pd.read_csv(FEATURES_CSV)
    tasks = build_tasks(df)

    rows = []
    for task_name, task_df in tasks.items():
        y = task_df["_y"].values
        if len(np.unique(y)) < 2:
            print(f"[skip] {task_name}: single class")
            continue
        print(f"\n=== {task_name}  n={len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())}) ===")

        # ADC baseline: raw rank (no classifier training)
        rows.append(evaluate(task_name, task_df, "ADC (raw)", None, "raw"))

        # The two NUTS feature sets x four classifiers
        for feat_name, feat_cols in [
            ("NUTS 8-bin", NUTS_ALL_COLS),
            ("NUTS 2-bin {0.25, 3.00}", NUTS_TWO_COLS),
        ]:
            for clf_name in ["LR", "RF", "GBM", "SVM-RBF"]:
                row = evaluate(task_name, task_df, feat_name, feat_cols, clf_name)
                rows.append(row)
                print(f"  {feat_name:25s}  {clf_name:8s}  "
                      f"AUC={row['auc']:.3f}  [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")

        # Print ADC last for visual contrast
        adc_row = rows[-(1 + 2 * 4)]
        print(f"  {'ADC (raw)':25s}  {'-':8s}  AUC={adc_row['auc']:.3f}  "
              f"[{adc_row['ci_lo']:.3f}, {adc_row['ci_hi']:.3f}]")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}  ({len(out)} rows)")

    # Quick falsification check: max non-LR AUC vs NUTS 2-bin LR per task
    print("\n--- §6 step-1 falsification check ---")
    for task_name in tasks:
        sub = out[out["task"] == task_name]
        if sub.empty:
            continue
        baseline = sub[(sub["feature_set"] == "NUTS 2-bin {0.25, 3.00}")
                       & (sub["classifier"] == "LR")]
        if baseline.empty:
            continue
        b_auc = baseline["auc"].iloc[0]
        b_lo, b_hi = baseline["ci_lo"].iloc[0], baseline["ci_hi"].iloc[0]

        challengers = sub[(sub["classifier"].isin(["RF", "GBM", "SVM-RBF"]))]
        if challengers.empty:
            continue
        best = challengers.loc[challengers["auc"].idxmax()]
        delta = best["auc"] - b_auc
        non_overlap = best["ci_lo"] > b_hi
        verdict = "FALSIFIES" if (delta >= 0.05 and non_overlap) else "OK"
        print(f"  {task_name}: 2-bin LR={b_auc:.3f} [{b_lo:.3f},{b_hi:.3f}]   "
              f"best non-linear={best['classifier']}/{best['feature_set']} "
              f"={best['auc']:.3f} [{best['ci_lo']:.3f},{best['ci_hi']:.3f}]   "
              f"Δ={delta:+.3f}  {verdict}")


if __name__ == "__main__":
    main()
