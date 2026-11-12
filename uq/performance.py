"""
performance.py: classification and calibration metrics for each model on the
training, calibration and test splits.

Classification metrics: balanced accuracy (primary), accuracy, AUC-ROC (macro and
weighted, one-vs-rest), F1 (macro and weighted), precision and recall (macro and
weighted), per-class recall and the confusion matrix.

Calibration metrics: expected calibration error (ECE, overall and one-vs-rest per
class), Brier score (overall and per class) and negative log-likelihood (NLL,
overall and per class). Every metric is computed on the mean predicted
probabilities. The per-class ECE treats each class k as a binary problem, with
probs[:, k] as the confidence and (y_true == k) as the outcome.

The module also provides summary tables and plots of these metrics.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES  = {0: "Depression", 1: "Panic Disorder",
                2: "Social Phobia", 3: "Specific Phobia"}
CLASS_LABELS = ["T1: Depression", "T2: Panic Disorder",
                "T3: Social Phobia", "T4: Specific Phobia"]
MODEL_LABELS = {
    "bayesian":  "Bayesian PyMC",
    "bootstrap": "Bootstrap LR",
    "mcd":       "MC Dropout",
    "svm":       "SVM (RBF)",
    "xgboost":   "XGBoost",
}
N_CLASSES = 4
EPS       = 1e-12


# Calibration helpers
# -------------------

def _ece_one_vs_rest(y_true_binary: np.ndarray, probs_k: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the expected calibration error for one class in one-vs-rest form.

    Patients are binned by their predicted probability for class k. The ECE is the
    size-weighted mean gap between the mean predicted probability and the observed
    share of class k in each bin.

    Input: y_true_binary, (n,) with 1 where the true class is k; probs_k, (n,) the
    predicted probability of class k; n_bins, the number of equal-width bins.
    Output: the ECE as a float.
    """
    confs = np.array(probs_k)
    corr  = np.array(y_true_binary, dtype=float)
    ece   = 0.0
    n     = len(confs)
    edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi  = edges[i], edges[i + 1]
        in_bin  = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        n_bin   = in_bin.sum()
        if n_bin == 0:
            continue
        ece += (n_bin / n) * abs(float(corr[in_bin].mean()) - float(confs[in_bin].mean()))

    return float(ece)


def compute_nll(y_true: np.ndarray, y_pred_probs: np.ndarray):
    """
    Computes the negative log-likelihood of the true class, overall and per class.

    Probabilities are clipped at EPS before the log.

    Input: y_true, (n,) zero-indexed labels; y_pred_probs, (n, K) predicted
    probabilities.
    Output: the overall NLL and an array of K per-class values.
    """
    y_true    = np.ravel(y_true).astype(int)
    probs     = np.clip(np.array(y_pred_probs), EPS, 1.0)
    K         = probs.shape[1]
    true_p    = probs[np.arange(len(y_true)), y_true]
    nll       = float(-np.mean(np.log(true_p)))
    nll_pc    = np.zeros(K)
    for c in range(K):
        mask = y_true == c
        if mask.sum() > 0:
            nll_pc[c] = float(-np.mean(np.log(probs[mask, c])))
    return nll, nll_pc


def compute_ece(y_true: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10):
    """
    Computes the overall expected calibration error with equal-width confidence bins.

    The confidence is the largest predicted probability and the prediction is its
    class. The ECE is the size-weighted mean gap between accuracy and mean confidence
    across bins.

    Input: y_true, (n,) zero-indexed labels; y_pred_probs, (n, K); n_bins, the
    number of bins.
    Output: the ECE as a float, and a dict of per-bin accuracies, confidences and sizes.
    """
    y_true = np.ravel(y_true)
    probs  = np.array(y_pred_probs)
    y_pred = probs.argmax(axis=1)
    confs  = probs.max(axis=1)
    ece    = 0.0
    n      = len(y_true)
    bounds = np.linspace(0, 1, n_bins + 1)
    stats  = {"accuracies": [], "confidences": [], "sizes": []}

    for i in range(n_bins):
        lo, hi  = bounds[i], bounds[i + 1]
        mask    = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        n_bin   = mask.sum()
        if n_bin == 0:
            stats["accuracies"].append(0.0)
            stats["confidences"].append(0.0)
            stats["sizes"].append(0)
            continue
        bin_acc  = float(np.mean(y_pred[mask] == y_true[mask]))
        bin_conf = float(np.mean(confs[mask]))
        ece     += (n_bin / n) * abs(bin_acc - bin_conf)
        stats["accuracies"].append(bin_acc)
        stats["confidences"].append(bin_conf)
        stats["sizes"].append(int(n_bin))

    return float(ece), stats


def compute_brier(y_true: np.ndarray, y_pred_probs: np.ndarray):
    """
    Computes the multiclass Brier score, overall and per class.

    The score is the mean squared difference between the predicted probabilities and
    the one-hot true labels.

    Input: y_true, (n,) zero-indexed labels; y_pred_probs, (n, K).
    Output: the overall score and an array of K per-class values.
    """
    y_true = np.ravel(y_true).astype(int)
    probs  = np.array(y_pred_probs)
    K      = probs.shape[1]
    ohe    = np.zeros((len(y_true), K))
    ohe[np.arange(len(y_true)), y_true] = 1
    sq     = (probs - ohe) ** 2
    return float(np.mean(sq)), np.mean(sq, axis=0)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Computes every classification and calibration metric for one model on one split.

    Input: y_true, (n,) zero-indexed labels; y_pred, (n,) zero-indexed predictions,
    the argmax of the mean probabilities; y_pred_probs, (n, K) mean predicted
    probabilities; n_bins, the number of ECE bins.
    Output: dict of metrics, including per_class_recall, confusion_matrix,
    ece_bin_stats and the per-class arrays brier_per_class, nll_per_class and
    ece_per_class.
    """
    y_true = np.ravel(y_true).astype(int)
    y_pred = np.ravel(y_pred).astype(int)
    probs  = np.array(y_pred_probs)
    m      = {}

    # Classification
    m["balanced_accuracy"]  = float(balanced_accuracy_score(y_true, y_pred))
    m["accuracy"]           = float(accuracy_score(y_true, y_pred))
    m["f1_macro"]           = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    m["f1_weighted"]        = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["precision_macro"]    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    m["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    m["recall_macro"]       = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
    m["recall_weighted"]    = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class recall and confusion matrix
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    totals = cm.sum(axis=1)
    m["per_class_recall"] = np.where(totals > 0, np.diag(cm) / totals, 0.0)
    m["confusion_matrix"] = cm

    # AUC-ROC
    try:
        yb = label_binarize(y_true, classes=list(range(N_CLASSES)))
        m["auc_roc_macro"]    = float(roc_auc_score(yb, probs, average="macro",    multi_class="ovr"))
        m["auc_roc_weighted"] = float(roc_auc_score(yb, probs, average="weighted", multi_class="ovr"))
    except Exception:
        m["auc_roc_macro"]    = None
        m["auc_roc_weighted"] = None

    # Calibration: overall ECE, Brier, NLL
    ece, bin_stats = compute_ece(y_true, probs, n_bins)
    m["ece"]           = ece
    m["ece_bin_stats"] = bin_stats

    brier, brier_pc = compute_brier(y_true, probs)
    m["brier_score"]     = brier
    m["brier_per_class"] = brier_pc

    nll, nll_pc = compute_nll(y_true, probs)
    m["nll"]           = nll
    m["nll_per_class"] = nll_pc

    # One-vs-rest ECE per class
    m["ece_per_class"] = np.array([
        _ece_one_vs_rest((y_true == c).astype(int), probs[:, c], n_bins)
        for c in range(N_CLASSES)
    ])

    return m


def _to_0idx(y) -> np.ndarray:
    """
    Converts one-indexed labels to a zero-indexed integer array.

    Input: y, a Series or array of labels 1 to K.
    Output: numpy array of labels 0 to K - 1.
    """
    arr = y.values.ravel() if hasattr(y, "values") else np.ravel(y)
    return (arr - 1).astype(int)


# Main entry point
# ----------------

def run_performance(
    all_models: dict,
    y_train,
    y_cal,
    y_test,
    config: dict,
    n_bins: int = 10,
) -> dict:
    """
    Computes every metric for every model on all three splits and saves the result.

    Input: all_models, model key to results dict with mean_probs_{split} and
    y_pred_{split}; y_train, y_cal, y_test, one-indexed labels 1 to 4; config, a dict
    with results_dir and optionally version; n_bins, the number of ECE bins.
    Output: nested dict {model: {'train': m, 'cal': m, 'test': m}}, each m from
    compute_classification_metrics. The dict is also saved as a pickle named with
    the version, so each representation keeps its own file.
    """
    print("=" * 70)
    print("PERFORMANCE EVALUATION")
    print("=" * 70)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    y_idxs = {
        "train": _to_0idx(y_train),
        "cal":   _to_0idx(y_cal),
        "test":  _to_0idx(y_test),
    }

    performance_results = {}

    for model_name, res in all_models.items():
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"\n  {label}")
        performance_results[model_name] = {}

        for split_name, y_idx in y_idxs.items():
            mean_probs = res[f"mean_probs_{split_name}"]
            y_pred     = res[f"y_pred_{split_name}"]
            m          = compute_classification_metrics(y_idx, y_pred, mean_probs, n_bins)
            performance_results[model_name][split_name] = m

            pcr = m["per_class_recall"]
            print(f"    {split_name:5s}: bal_acc={m['balanced_accuracy']:.4f}  "
                  f"ECE={m['ece']:.4f}  Brier={m['brier_score']:.4f}  "
                  f"NLL={m['nll']:.4f}")
            print(f"           T1={pcr[0]:.3f}  T2={pcr[1]:.3f}  "
                  f"T3={pcr[2]:.3f}  T4={pcr[3]:.3f}")

    # The file name carries the version, so two representations writing to one folder
    # keep separate files.
    _version = config.get("version", "")
    save_path = results_dir / (f"performance_results_{_version}.pkl" if _version
                               else "performance_results_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(performance_results, f)
    print(f"\n  Saved: {save_path}")
    print("=" * 70)

    return performance_results


# Summary tables
# --------------

def make_performance_table(performance_results: dict, split: str = "test") -> pd.DataFrame:
    """
    Builds the table of headline and per-class metrics for one split.

    Input: performance_results, from run_performance; split, 'train', 'cal' or 'test'.
    Output: DataFrame with one row per model and one column per metric.
    """
    rows = []
    for mn, splits in performance_results.items():
        m   = splits[split]
        pcr = m["per_class_recall"]
        pcn = m["nll_per_class"]
        pcb = m["brier_per_class"]
        auc = round(m["auc_roc_macro"], 4) if m["auc_roc_macro"] is not None else None
        rows.append({
            "Model":           MODEL_LABELS.get(mn, mn),
            "Bal. Acc.":       round(m["balanced_accuracy"], 4),
            "Accuracy":        round(m["accuracy"], 4),
            "AUC-ROC (macro)": auc,
            "F1 (macro)":      round(m["f1_macro"], 4),
            "F1 (weighted)":   round(m["f1_weighted"], 4),
            "Prec. (macro)":   round(m["precision_macro"], 4),
            "Recall (macro)":  round(m["recall_macro"], 4),
            "ECE":             round(m["ece"], 4),
            "Brier":           round(m["brier_score"], 4),
            "NLL":             round(m["nll"], 4),
            "T1 Recall":       round(float(pcr[0]), 4),
            "T2 Recall":       round(float(pcr[1]), 4),
            "T3 Recall":       round(float(pcr[2]), 4),
            "T4 Recall":       round(float(pcr[3]), 4),
            "T1 NLL":          round(float(pcn[0]), 4),
            "T2 NLL":          round(float(pcn[1]), 4),
            "T3 NLL":          round(float(pcn[2]), 4),
            "T4 NLL":          round(float(pcn[3]), 4),
            "T1 Brier":        round(float(pcb[0]), 4),
            "T2 Brier":        round(float(pcb[1]), 4),
            "T3 Brier":        round(float(pcb[2]), 4),
            "T4 Brier":        round(float(pcb[3]), 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def make_per_class_calibration_table(
    performance_results: dict,
    split: str = "test",
) -> pd.DataFrame:
    """
    Builds the table of one-vs-rest ECE per class for one split.

    A per-class view separates models whose overall ECE is similar.

    Input: performance_results, from run_performance; split, 'train', 'cal' or 'test'.
    Output: DataFrame with one row per model and columns T1 to T4.
    """
    rows = []
    for mn, splits in performance_results.items():
        m   = splits[split]
        row = {"Model": MODEL_LABELS.get(mn, mn)}
        for k, label in enumerate(["T1", "T2", "T3", "T4"]):
            row[label] = round(float(m["ece_per_class"][k]), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# Visualisations
# --------------

def plot_balanced_accuracy(performance_results: dict, figsize=(12, 5)) -> None:
    """
    Plots balanced accuracy per model and split as grouped bars, with the four-class
    chance level of 0.25 as a dashed line.

    Input: performance_results, from run_performance; figsize.
    Output: none; the figure is shown.
    """
    models = list(performance_results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = {"train": "#4C72B0", "cal": "#DD8452", "test": "#55A868"}
    x, w   = np.arange(len(models)), 0.25

    fig, ax = plt.subplots(figsize=figsize)
    for i, split in enumerate(["train", "cal", "test"]):
        vals = [performance_results[m][split]["balanced_accuracy"] for m in models]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=split.capitalize(),
                      color=colors[split], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy — All Models and Splits", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.25, color="grey", linestyle=":", linewidth=0.8, label="Chance (0.25)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_calibration_comparison(
    performance_results: dict,
    split: str = "test",
    figsize=(14, 4),
) -> None:
    """
    Plots ECE, Brier score and NLL per model on one split as three bar panels.

    Input: performance_results; split; figsize.
    Output: none; the figure is shown.
    """
    models = list(performance_results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    cfgs   = [
        ("ece",         "ECE",         "#4C72B0"),
        ("brier_score", "Brier Score", "#DD8452"),
        ("nll",         "NLL",         "#55A868"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, title, color) in zip(axes, cfgs):
        vals = [performance_results[m][split][key] for m in models]
        bars = ax.bar(labels, vals, color=color, alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{title}\n(lower = better)", fontsize=10)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Calibration Metrics — {split.capitalize()} Set",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_per_class_recall(
    performance_results: dict,
    split: str = "test",
    figsize=(13, 4),
) -> None:
    """
    Plots per-class recall as a heatmap with models as rows and classes as columns.

    Input: performance_results; split; figsize.
    Output: none; the figure is shown.
    """
    models = list(performance_results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    data   = np.array([performance_results[m][split]["per_class_recall"]
                        for m in models])
    df = pd.DataFrame(data, index=labels, columns=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Recall (class-wise)"})
    ax.set_title(f"Per-Class Recall — {split.capitalize()} Set", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    performance_results: dict,
    split: str = "test",
    figsize=(15, 3),
) -> None:
    """
    Plots one confusion matrix per model, side by side.

    Input: performance_results; split; figsize.
    Output: none; the figure is shown.
    """
    models = list(performance_results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    tick = ["T1", "T2", "T3", "T4"]
    for ax, mn in zip(axes, models):
        cm = performance_results[mn][split]["confusion_matrix"]
        ba = performance_results[mn][split]["balanced_accuracy"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=tick, yticklabels=tick,
                    ax=ax, cbar=False, linewidths=0.4)
        ax.set_title(f"{MODEL_LABELS.get(mn, mn)}\nBA={ba:.3f}", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Confusion Matrices — {split.capitalize()} Set",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_per_class_calibration_heatmap(
    performance_results: dict,
    split: str = "test",
    figsize=(9, 4),
) -> None:
    """
    Plots one-vs-rest ECE per class as a heatmap with models as rows.

    Input: performance_results; split; figsize.
    Output: none; the figure is shown.
    """
    df = make_per_class_calibration_table(performance_results, split)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "One-vs-rest ECE (lower = better)"})
    ax.set_title(
        f"Per-Class Calibration (One-vs-Rest ECE) — {split.capitalize()} Set",
        fontsize=11,
    )
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_per_class_nll(
    performance_results: dict,
    split: str = "test",
    figsize=(13, 4),
) -> None:
    """
    Plots per-class NLL as a heatmap with models as rows and classes as columns.

    Input: performance_results; split; figsize.
    Output: none; the figure is shown.
    """
    models = list(performance_results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    data   = np.array([performance_results[m][split]["nll_per_class"]
                        for m in models])
    df = pd.DataFrame(data, index=labels, columns=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "NLL (lower = better)"})
    ax.set_title(f"Per-Class NLL — {split.capitalize()} Set", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
