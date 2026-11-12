"""
table1_bootstrap.py: patient-level bootstrap spread for every Table 1 metric.

The script resamples the 505 test patients with the shared resampling matrix
(config.N_BOOT draws, config.BOOT_SEED), so each Table 1 metric carries a standard
deviation and a 95 percent percentile interval. The point values equal the saved
test performance table, since each metric uses the same performance functions.
RUNS holds the newest full run folder and the version of each representation.

The work runs in two steps, so the step reading the large results pickles uses
numpy alone:
    python -m tables_figures.table1_bootstrap extract   numpy only, caches the test predictions
    python -m tables_figures.table1_bootstrap boot      needs scikit-learn, writes the tables
    python -m tables_figures.table1_bootstrap           both steps
"""

import sys
import pickle

import numpy as np
import pandas as pd

from pipeline import config as C
from tables_figures.figures import newest_run

RUNS = {"indv": (newest_run("indvA").name, "vA_indv"),
        "sum":  (newest_run("sumB").name,  "vB_sum")}
MODELS = ["bootstrap", "bayesian", "svm", "mcd", "xgboost"]
LABELS = {"bootstrap": "Boot LR", "bayesian": "Bayes LR", "svm": "SVM",
          "mcd": "NN-MCD", "xgboost": "XGB"}
METRICS = [("balanced_accuracy", "Balanced accuracy"),
           ("accuracy",          "Overall accuracy"),
           ("auc_roc_macro",     "Macro AUC-ROC"),
           ("f1_macro",          "Macro F1 score"),
           ("ece",               "ECE (10 bins)"),
           ("brier_score",       "Normalized Brier score"),
           ("nll",               "NLL score")]
CLASSES = ["Depression", "Panic disorder", "Social phobia", "Specific phobia"]


def _eval_dir(rep: str):
    """
    Returns the eval_0 folder of one representation's run.

    Input: rep, a key of RUNS ('indv' or 'sum').
    Output: the folder path.
    """
    run, _ = RUNS[rep]
    return C.DIR_RESULTS / run / f"eval_0_{rep}_modelPerformance"


def extract() -> None:
    """
    Caches the test labels and each model's mean_probs_test and y_pred_test in one
    compressed file per representation.

    Output: none; writes table1_test_predictions_{rep}.npz to the eval_0 folder.
    """
    for rep, (run, tag) in RUNS.items():
        y = pd.read_csv(C.DIR_CODE / "data" / "test" / f"y_{rep}_test.csv",
                        index_col=0)["TreatmentType"].to_numpy()
        arrays = {"y_test_0idx": (y - 1).astype(int)}
        for m in MODELS:
            path = C.DIR_RESULTS / run / "models" / f"results_{m}_{tag}.pkl"
            with open(path, "rb") as f:
                res = pickle.load(f)
            arrays[f"{m}__mean_probs_test"] = res["mean_probs_test"]
            arrays[f"{m}__y_pred_test"]     = res["y_pred_test"]
        out = _eval_dir(rep) / f"table1_test_predictions_{rep}.npz"
        np.savez_compressed(out, **arrays)
        print(f"  saved {out}")


def boot() -> None:
    """
    Bootstraps every Table 1 metric, including the four per-class recalls, over the
    shared test resamples.

    Output: none; writes performance_test_bootstrap_{rep}.csv to the eval_0 folder,
    in long format with point, sd, lo, hi and n_boot per model and metric.
    """
    from evaluation import stats as S
    from uq.performance import compute_classification_metrics

    for rep in RUNS:
        z = np.load(_eval_dir(rep) / f"table1_test_predictions_{rep}.npz")
        y = z["y_test_0idx"]
        n = len(y)
        idx = S.boot_indices(n)  # the shared resampling matrix
        rows = []
        for m in MODELS:
            probs, pred = z[f"{m}__mean_probs_test"], z[f"{m}__y_pred_test"]

            def all_metrics(i):
                r = compute_classification_metrics(y[i], pred[i], probs[i])
                out = {k: (np.nan if r[k] is None else r[k]) for k, _ in METRICS}
                for c in range(len(CLASSES)):
                    out[f"recall_T{c+1}"] = r["per_class_recall"][c]
                return out

            point = all_metrics(np.arange(n))
            draws = pd.DataFrame([all_metrics(row) for row in idx])
            keys = [k for k, _ in METRICS] + [f"recall_T{c+1}" for c in range(len(CLASSES))]
            names = dict(METRICS) | {f"recall_T{c+1}": f"Recall, {CLASSES[c]}"
                                      for c in range(len(CLASSES))}
            for k in keys:
                d = draws[k].dropna().to_numpy()
                lo, hi = np.quantile(d, [(1 - C.CI_LEVEL) / 2, 1 - (1 - C.CI_LEVEL) / 2])
                rows.append({"Model": LABELS[m], "metric_key": k, "Metric": names[k],
                             "point": point[k], "sd": float(np.std(d, ddof=1)),
                             "lo": float(lo), "hi": float(hi), "n_boot": len(d)})
            print(f"  {rep} {LABELS[m]} done")
        out = _eval_dir(rep) / f"performance_test_bootstrap_{rep}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  saved {out}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("extract", "all"):
        extract()
    if step in ("boot", "all"):
        boot()
