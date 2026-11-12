"""
coverage_by_resplit.py: conformal coverage averaged over repeated re-splits of the
held-out patients.

Split conformal prediction guarantees marginal coverage over the joint draw of
calibration and test patients, so coverage averaged over many splits should reach
1 - alpha for any score function. Both representations share one split, so the
same 404 patients calibrate both and the same 505 are tested. The script pools the
909 held-out patients and re-splits them 500 times into 404 calibration and 505
test patients, stratified by class like the original split. Coverage is recomputed
on each split. The stored mean predicted probabilities of the fitted models
already cover all 909 patients. RUNS holds the newest full run
folder of each representation.

Output: results/coverage_by_resplit.csv, one row per representation and model.

Run from the repository root:  python -m tools.coverage_by_resplit
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tables_figures.figures import newest_run

ALPHA = 0.05
N_SPLITS = 500
SEED = 2026
DIR = Path(__file__).resolve().parent.parent

RUNS = [(newest_run("indvA").name, "indv", "vA_indv", "A"),
        (newest_run("sumB").name, "sum", "vB_sum", "B")]
MODELS = ["bayesian", "bootstrap", "mcd", "svm", "xgboost"]
LABELS = {"bayesian": "Bayesian Logistic Regression",
          "bootstrap": "Bootstrap Logistic Regression",
          "mcd": "MC Dropout Neural Network",
          "svm": "SVM Ensemble", "xgboost": "XGBoost Ensemble"}


def q_level(n, alpha):
    """
    Returns the finite-sample conformal quantile level ceil((n + 1)(1 - alpha)) / n.

    Input: n, the number of calibration patients; alpha, the miscoverage level.
    Output: the level as a float.
    """
    return float(np.ceil((n + 1) * (1 - alpha)) / n)


def coverage_once(probs, y, cal_idx, test_idx, alpha=ALPHA):
    """
    Computes conformal coverage for one split: the threshold from the calibration
    rows and the coverage on the test rows.

    The threshold is the k-th smallest calibration score, k = ceil((n + 1)(1 - alpha)),
    and coverage is measured on the sets as constructed.

    Input: probs, (n, K) mean predicted probabilities of the pooled patients; y,
    zero-indexed labels; cal_idx, test_idx, the row positions of each part; alpha.
    Output: the coverage and the threshold q.
    """
    s = np.sort(1.0 - probs[cal_idx, y[cal_idx]])
    k = int(np.ceil((len(s) + 1) * (1.0 - alpha)))
    q = float(s[min(max(k, 1), len(s)) - 1])
    inset = (1.0 - probs[test_idx, y[test_idx]]) <= q
    return float(inset.mean()), q


def stratified_splits(y, n_cal, rng, n_splits):
    """
    Yields random calibration and test partitions stratified by class.

    Each class contributes the same share to the calibration part as it holds in the
    pooled set.

    Input: y, the pooled labels; n_cal, the calibration size; rng, a numpy
    Generator; n_splits, the number of partitions.
    Output: a generator of (cal_idx, test_idx) pairs.
    """
    classes = np.unique(y)
    per_class = {c: np.flatnonzero(y == c) for c in classes}
    frac = n_cal / len(y)
    for _ in range(n_splits):
        cal = []
        for c in classes:
            idx = per_class[c].copy()
            rng.shuffle(idx)
            k = int(round(len(idx) * frac))
            cal.append(idx[:k])
        cal = np.concatenate(cal)
        mask = np.ones(len(y), bool)
        mask[cal] = False
        yield cal, np.flatnonzero(mask)


def main():
    """
    Computes the observed and re-split coverage for every model and representation,
    writes the table and prints a summary.
    """
    rows = []
    for run_id, tag, ver, rep in RUNS:
        run = DIR / "results" / run_id
        y_cal = pd.read_csv(DIR / f"data/cal/y_{tag}_calibration.csv").iloc[:, -1].values.astype(int) - 1
        y_test = pd.read_csv(DIR / f"data/test/y_{tag}_test.csv").iloc[:, -1].values.astype(int) - 1
        y_pool = np.concatenate([y_cal, y_test])
        n_cal = len(y_cal)

        for m in MODELS:
            with open(run / f"models/results_{m}_{ver}.pkl", "rb") as fh:
                d = pickle.load(fh)
            probs = np.vstack([np.asarray(d["mean_probs_cal"]),
                               np.asarray(d["mean_probs_test"])])

            observed, q_obs = coverage_once(
                probs, y_pool, np.arange(n_cal), np.arange(n_cal, len(y_pool)))

            rng = np.random.default_rng(SEED)
            covs, qs = [], []
            for cal_idx, test_idx in stratified_splits(y_pool, n_cal, rng, N_SPLITS):
                c, q = coverage_once(probs, y_pool, cal_idx, test_idx)
                covs.append(c); qs.append(q)
            covs, qs = np.asarray(covs), np.asarray(qs)

            rows.append({
                "rep": rep, "model": LABELS[m],
                "observed_coverage": round(observed, 4),
                "mean_over_splits": round(float(covs.mean()), 4),
                "sd_over_splits": round(float(covs.std(ddof=1)), 4),
                "range_over_splits": f"[{np.quantile(covs, .025):.4f}, {np.quantile(covs, .975):.4f}]",
                "pct_splits_at_or_above_target": round(float((covs >= 0.95).mean()) * 100, 1),
                "percentile_of_observed": round(float((covs <= observed).mean()) * 100, 1),
                "q_observed": round(q_obs, 4),
                "q_mean_over_splits": round(float(qs.mean()), 4),
            })

    df = pd.DataFrame(rows)
    out = DIR / "results" / "coverage_by_resplit.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(f"{N_SPLITS} stratified re-splits of the 909 held-out patients, seed {SEED}, "
          f"alpha {ALPHA}, no refitting.\n")
    print(df.to_string(index=False))
    print(f"\nwritten: {out}")
    print("\nMean over splits, by representation:")
    print(df.groupby("rep")["mean_over_splits"].agg(["min", "mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
