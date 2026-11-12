"""
uncertainty.py: entropy decomposition, conformal prediction thresholds, the
matched-volume entropy threshold, patient-level flags and their concordance.

Entropy decomposition, from the M probability vectors per patient:
    H  = H(E[p])       total uncertainty, the Shannon entropy of the mean vector
    MI = H - E[H(p)]   epistemic uncertainty (mutual information)
    AU = E[H(p)]       aleatoric uncertainty

Conformal prediction:
    Inductive conformal prediction (ICP) uses one threshold q from the calibration
    nonconformity scores s = 1 - p(true class). Mondrian thresholds, one per
    class, are computed alongside for the prediction-set columns.

Matched-volume entropy threshold:
    r = the share of calibration patients the ICP flag marks uncertain.
    t = the (1 - r) quantile of calibration entropy H.
    Both come from calibration alone and are then applied to test.

Concordance of the ICP flag and the entropy flag, per test patient:
    Concordant Certain    ICP singleton,          H at or below t
    Concordant Uncertain  ICP empty or 2+ labels, H above t
    Discordant Type A     ICP singleton,          H above t
    Discordant Type B     ICP empty or 2+ labels, H at or below t
"""

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "Depression",
    1: "Panic Disorder",
    2: "Social Phobia",
    3: "Specific Phobia",
}

MODEL_LABELS = {
    "bayesian":  "Bayesian PyMC",
    "bootstrap": "Bootstrap LR",
    "mcd":       "MC Dropout",
    "svm":       "SVM (RBF)",
    "xgboost":   "XGBoost",
}

CONCORDANCE_CATEGORIES = [
    "Concordant Certain",
    "Concordant Uncertain",
    "Discordant Type A",
    "Discordant Type B",
]

EPS       = 1e-10
N_CLASSES = 4


# Entropy decomposition
# --------------------

def _h(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes the Shannon entropy (nats) along one axis, clipping at EPS to avoid log(0).

    Input: probs, an array of probabilities; axis, the class axis.
    Output: the entropy array.
    """
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def compute_uncertainty_metrics(all_probs: np.ndarray) -> pd.DataFrame:
    """
    Decomposes predictive uncertainty into total entropy H, mutual information MI and
    aleatoric uncertainty AU for one model on one split.

    The decomposition depends only on the (M, n, K) array, so it applies to all five
    models alike.

    Input: all_probs, (M, n, K), the M probability vectors per patient.
    Output: DataFrame with one row per patient: entropy, mutual_information,
    aleatoric_uncertainty, aleatoric_residual (H - MI) and aleatoric_match, a
    consistency check of the two aleatoric values.
    """
    mean_probs   = all_probs.mean(axis=0)
    entropy      = _h(mean_probs, axis=-1)
    entropy_each = _h(all_probs,  axis=-1)
    aleatoric    = entropy_each.mean(axis=0)
    mi           = np.clip(entropy - aleatoric, 0, None)
    aleatoric_residual = entropy - mi

    return pd.DataFrame({
        "entropy":               entropy,
        "mutual_information":    mi,
        "aleatoric_uncertainty": aleatoric,
        "aleatoric_residual":    aleatoric_residual,
        "aleatoric_match":       np.abs(aleatoric_residual - aleatoric) < 1e-4,
    })


# CP threshold computation
# ------------------------

def _q_level(n: int, alpha: float) -> float:
    """
    Returns the finite-sample conformal quantile level ceil((n + 1)(1 - alpha)) / n.

    Input: n, the number of calibration scores; alpha, the miscoverage level.
    Output: the level as a float.
    """
    return float(np.ceil((n + 1) * (1 - alpha)) / n)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Returns the split-conformal threshold: the k-th smallest calibration score, with
    k = ceil((n + 1)(1 - alpha)).

    The order statistic comes from sorting and indexing, which states the rule
    exactly. NumPy's quantile methods place the index at q(n - 1), so passing k / n
    returns a neighbouring rank.

    Input: scores, the calibration nonconformity scores; alpha, the miscoverage level.
    Output: the threshold q, with k clamped to the available ranks at extreme alpha.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[min(max(k, 1), n) - 1])


def compute_icp_threshold(
    mean_probs_cal: np.ndarray,
    y_cal_idx: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Computes the inductive conformal prediction threshold from the calibration set.

    The nonconformity score of patient i is s_i = 1 - p(y_i | x_i), one minus the
    mean predicted probability of the true class. The threshold q_icp gives marginal
    coverage of at least 1 - alpha on exchangeable test data.

    Input: mean_probs_cal, (n_cal, K) mean predicted probabilities; y_cal_idx,
    (n_cal,) zero-indexed labels; alpha, the miscoverage level.
    Output: dict with q_icp, q_level, n_cal and alpha.
    """
    n_cal      = len(y_cal_idx)
    true_probs = mean_probs_cal[np.arange(n_cal), y_cal_idx]
    nc_scores  = 1.0 - true_probs
    ql         = _q_level(n_cal, alpha)

    return {
        "q_icp":   conformal_quantile(nc_scores, alpha),
        "q_level": round(ql, 6),
        "n_cal":   n_cal,
        "alpha":   alpha,
    }


def compute_mondrian_thresholds(
    mean_probs_cal: np.ndarray,
    y_cal_idx: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Computes one Mondrian conformal threshold per class, each from the calibration
    patients of that class.

    A class with no calibration patients falls back to the scores of all calibration
    patients for that class column, with a printed warning.

    Input: mean_probs_cal, (n_cal, K); y_cal_idx, (n_cal,) zero-indexed labels;
    alpha, the miscoverage level.
    Output: dict keyed by class index 0 to 3, each with q_mondrian, q_level,
    n_cal_class and alpha.
    """
    out = {}
    for k in range(N_CLASSES):
        mask_k = y_cal_idx == k
        n_k    = int(mask_k.sum())

        if n_k == 0:
            nc_k = 1.0 - mean_probs_cal[:, k]
            n_k  = len(nc_k)
            print(f"  Warning: class {k} ({CLASS_NAMES[k]}) has no calibration "
                  f"samples. Global nonconformity scores used as fallback.")
        else:
            nc_k = 1.0 - mean_probs_cal[mask_k, k]

        ql_k = _q_level(n_k, alpha)
        out[k] = {
            # the same order statistic, within class k
            "q_mondrian":  conformal_quantile(nc_k, alpha),
            "q_level":     round(ql_k, 6),
            "n_cal_class": n_k,
            "alpha":       alpha,
        }
    return out


# Matched-volume entropy threshold
# --------------------------

def compute_rate_matched_threshold(uq_cal: pd.DataFrame, cp_flags_cal: pd.DataFrame) -> dict:
    """
    Derives the entropy threshold t from the calibration flag rate of conformal
    prediction.

    r is the share of calibration patients with icp_flag == 1, and t is the (1 - r)
    quantile of calibration entropy. The entropy flag therefore marks the same share
    of calibration patients as the conformal flag. r is measured from the
    calibration prediction sets, and both r and t use calibration data alone.

    Input: uq_cal, from compute_uncertainty_metrics on calibration; cp_flags_cal,
    from evaluate_cp_flags on calibration.
    Output: dict with r, t and q_level (1 - r).
    """
    r       = float((cp_flags_cal["icp_flag"] == 1).mean())
    q_level = 1.0 - r
    t       = float(uq_cal["entropy"].quantile(q_level))

    return {"r": round(r, 6), "t": round(t, 6), "q_level": round(q_level, 6)}


# Flag evaluation
# ---------------

def evaluate_cp_flags(
    mean_probs_test: np.ndarray,
    thr_icp: dict,
    thr_mondrian: dict,
    y_test_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Builds the ICP and Mondrian prediction sets and flags for each patient.

    A set contains class k when 1 - p(k) <= q. The flag is 0 for a singleton set
    (certain) and 1 for an empty set or a set of two or more classes (uncertain). An
    empty set means no class reaches a mean predicted probability of 1 - q, so the
    patient is flagged uncertain. At alpha 0.05 no set can be empty, since the largest
    of four probabilities is at least 0.25 and q is above 0.75.

    Input: mean_probs_test, (n, K) mean predicted probabilities; thr_icp, from
    compute_icp_threshold; thr_mondrian, from compute_mondrian_thresholds;
    y_test_idx, (n,) zero-indexed labels.
    Output: DataFrame with one row per patient: true and predicted class, both
    prediction sets (names and indices), set sizes, icp_flag and mondrian_flag.
    """
    q_icp = thr_icp["q_icp"]
    rows  = []

    for i in range(len(y_test_idx)):
        p = mean_probs_test[i]

        ps_icp = [k for k in range(N_CLASSES) if (1.0 - p[k]) <= q_icp]

        ps_mondrian = [k for k in range(N_CLASSES)
                       if (1.0 - p[k]) <= thr_mondrian[k]["q_mondrian"]]

        rows.append({
            "patient_pos":     i,
            "true_class":      int(y_test_idx[i]),
            "true_class_name": CLASS_NAMES[int(y_test_idx[i])],
            "y_pred":          int(p.argmax()),
            "y_pred_name":     CLASS_NAMES[int(p.argmax())],
            "ps_icp":          [CLASS_NAMES[k] for k in ps_icp],
            "ps_mondrian":     [CLASS_NAMES[k] for k in ps_mondrian],
            "ps_icp_idx":      ps_icp,
            "ps_mondrian_idx": ps_mondrian,
            "n_set_icp":       len(ps_icp),
            "n_set_mondrian":  len(ps_mondrian),
            "icp_flag":        0 if len(ps_icp) == 1 else 1,
            "mondrian_flag":   0 if len(ps_mondrian) == 1 else 1,
        })

    return pd.DataFrame(rows)


def evaluate_ppu_flags_rate_matched(uq_test: pd.DataFrame, t: float) -> pd.DataFrame:
    """
    Applies the entropy threshold t to each patient.

    h_flag is 0 when H <= t (certain) and 1 when H > t (uncertain). MI and AU are
    carried for description only.

    Input: uq_test, from compute_uncertainty_metrics; t, the entropy threshold.
    Output: DataFrame with patient_pos, entropy, mutual_information,
    aleatoric_uncertainty and h_flag.
    """
    df = uq_test.reset_index(drop=True)
    return pd.DataFrame({
        "patient_pos":           df.index,
        "entropy":               df["entropy"].values,
        "mutual_information":    df["mutual_information"].values,
        "aleatoric_uncertainty": df["aleatoric_uncertainty"].values,
        "h_flag":                (df["entropy"] > t).astype(int).values,
    })


# Concordance analysis
# --------------------

def compute_concordance_table(
    cp_flags_df: pd.DataFrame,
    ppu_flags_df: pd.DataFrame,
    cp_col: str,
    ppu_col: str,
) -> dict:
    """
    Cross-tabulates one conformal flag column against one entropy flag column.

    Input: cp_flags_df, ppu_flags_df, the two flag tables; cp_col, ppu_col, the flag
    columns to compare.
    Output: dict with counts and proportions of the four concordance categories,
    n_test, the two column names, and detail_df with one row per patient
    (patient_pos, cp_flag, ppu_flag, category).
    """
    cp  = cp_flags_df[cp_col].values
    ppu = ppu_flags_df[ppu_col].values
    n   = len(cp)

    cats = []
    for c, p in zip(cp, ppu):
        if   c == 0 and p == 0: cats.append("Concordant Certain")
        elif c == 1 and p == 1: cats.append("Concordant Uncertain")
        elif c == 0 and p == 1: cats.append("Discordant Type A")
        else:                    cats.append("Discordant Type B")

    counts = {cat: cats.count(cat) for cat in CONCORDANCE_CATEGORIES}
    props  = {cat: round(counts[cat] / n, 4) for cat in CONCORDANCE_CATEGORIES}

    return {
        "counts":      counts,
        "proportions": props,
        "n_test":      n,
        "cp_col":      cp_col,
        "ppu_col":     ppu_col,
        "detail_df":   pd.DataFrame({
            "patient_pos": cp_flags_df["patient_pos"].values,
            "cp_flag":     cp,
            "ppu_flag":    ppu,
            "category":    cats,
        }),
    }


def run_concordance_analysis_icp(
    cp_flags_df: pd.DataFrame,
    ppu_flags_df: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Builds the ICP flag by entropy flag concordance table for one model.

    Input: cp_flags_df, from evaluate_cp_flags; ppu_flags_df, from
    evaluate_ppu_flags_rate_matched; model_name.
    Output: dict with icp_x_h, from compute_concordance_table, and model_name.
    """
    return {
        "icp_x_h":    compute_concordance_table(cp_flags_df, ppu_flags_df,
                                                 "icp_flag", "h_flag"),
        "model_name": model_name,
    }


def make_concordance_summary_df_icp_h(all_concordance: dict) -> pd.DataFrame:
    """
    Collects the concordance results of all models into one table.

    Input: all_concordance, model key to the output of run_concordance_analysis_icp.
    Output: DataFrame with one row per model and the count and percentage of each
    concordance category.
    """
    rows = []
    for model_name, conc in all_concordance.items():
        tbl = conc["icp_x_h"]
        c   = tbl["counts"]
        p   = tbl["proportions"]
        n   = tbl["n_test"]
        rows.append({
            "Model":                    MODEL_LABELS.get(model_name, model_name),
            "CP Method":                "Standard ICP",
            "PPU Metric":               "H (Total Entropy)",
            "N Concordant Certain":     c["Concordant Certain"],
            "% Concordant Certain":     round(p["Concordant Certain"] * 100, 1),
            "N Concordant Uncertain":   c["Concordant Uncertain"],
            "% Concordant Uncertain":   round(p["Concordant Uncertain"] * 100, 1),
            "N Discordant Type A":      c["Discordant Type A"],
            "% Discordant Type A":      round(p["Discordant Type A"] * 100, 1),
            "N Discordant Type B":      c["Discordant Type B"],
            "% Discordant Type B":      round(p["Discordant Type B"] * 100, 1),
            "N Total":                  n,
        })
    return pd.DataFrame(rows)


def print_concordance_table(concordance_result: dict, model_name: str) -> None:
    """
    Prints the concordance counts, percentages and totals for one model.

    Input: concordance_result, from compute_concordance_table; model_name.
    Output: none.
    """
    c   = concordance_result["counts"]
    p   = concordance_result["proportions"]
    n   = concordance_result["n_test"]
    cp  = concordance_result["cp_col"].replace("_flag", "").upper()
    ppu = concordance_result["ppu_col"].replace("_flag", "").upper()

    sep = "-" * 64
    print(f"\n{sep}")
    print(f"  {MODEL_LABELS.get(model_name, model_name)}  |  {cp} x {ppu}  |  n={n}")
    print(sep)
    for cat in CONCORDANCE_CATEGORIES:
        print(f"  {cat:<30s}: {c[cat]:4d}  ({p[cat]*100:5.1f}%)")
    total_c = c["Concordant Certain"] + c["Concordant Uncertain"]
    print(sep)
    print(f"  Total concordant : {total_c:4d}  ({total_c/n*100:.1f}%)")
    print(f"  Total discordant : {n-total_c:4d}  ({(n-total_c)/n*100:.1f}%)")
    print(sep)


# Visualisations
# --------------

def plot_uncertainty_decomposition(uncertainty_results: dict, figsize=(14, 5)) -> None:
    """
    Plots mean H, MI and AU per model and split as three bar panels.

    Input: uncertainty_results, from run_uncertainty_rate_matched; figsize.
    Output: none; the figure is shown.
    """
    models  = list(uncertainty_results.keys())
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    splits  = ["train", "cal", "test"]
    metrics = [
        ("entropy",               "Total H",      "#4C72B0"),
        ("mutual_information",    "Epistemic MI", "#DD8452"),
        ("aleatoric_uncertainty", "Aleatoric AU", "#55A868"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    w = 0.22
    x = np.arange(len(models))

    for ax, (key, title, color) in zip(axes, metrics):
        alphas  = [0.5, 0.72, 1.0]
        hatches = ["/", "", ""]
        for s_i, (split, alph, hatch) in enumerate(zip(splits, alphas, hatches)):
            vals = [uncertainty_results[m][split][key].mean() for m in models]
            bars = ax.bar(x + (s_i - 1) * w, vals, w, color=color, alpha=alph,
                          hatch=hatch, label=split.capitalize(), edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.set_ylabel("Mean (nats)", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Uncertainty Decomposition — All Models", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_distributions_model_rate_matched(
    uncertainty_results: dict,
    model_name: str,
    figsize=(10, 5),
) -> None:
    """
    Plots kernel density estimates of entropy H on the training, calibration and test
    splits for one model, with the threshold t as a dashed line.

    SciPy is imported inside this function, the only one in the module that needs it.

    Input: uncertainty_results, from run_uncertainty_rate_matched; model_name; figsize.
    Output: none; the figure is shown.
    """
    res = uncertainty_results[model_name]
    lbl = MODEL_LABELS.get(model_name, model_name)
    t   = res["rate_matched"]["t"]
    r   = res["rate_matched"]["r"]

    tr = res["train"]["entropy"].values
    ca = res["cal"]["entropy"].values
    te = res["test"]["entropy"].values

    xg = np.linspace(min(tr.min(), ca.min(), te.min()),
                     max(tr.max(), ca.max(), te.max()) * 1.05, 500)
    from scipy.stats import gaussian_kde
    kde_tr = gaussian_kde(tr, bw_method="scott")(xg)
    kde_ca = gaussian_kde(ca, bw_method="scott")(xg)
    kde_te = gaussian_kde(te, bw_method="scott")(xg)
    yt = max(kde_tr.max(), kde_ca.max(), kde_te.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(xg, kde_tr, alpha=0.12, color="steelblue")
    ax.plot(xg, kde_tr, lw=0.9, ls=":", color="steelblue", alpha=0.7,
            label=f"Train (n={len(tr)})")
    ax.fill_between(xg, kde_ca, alpha=0.35, color="grey")
    ax.plot(xg, kde_ca, lw=1.5, color="grey", label=f"Cal (n={len(ca)})")
    ax.fill_between(xg, kde_te, alpha=0.35, color="steelblue")
    ax.plot(xg, kde_te, lw=1.5, color="#3a7aba", label=f"Test (n={len(te)})")

    ax.axvline(t, color="red", ls="--", lw=1.5,
               label=f"Rate-matched t = {t:.4f}  (r = {r:.3f})")
    ax.text(t + (xg[-1] - xg[0]) * 0.02, yt * 0.88,
            f"t = {t:.4f}", color="red", fontsize=9, fontweight="bold")

    ry = -yt * 0.04
    ax.plot(ca, np.full_like(ca, ry),             "|", color="grey",     alpha=0.4, ms=4)
    ax.plot(te, np.full_like(te, ry - yt * 0.03), "|", color="steelblue", alpha=0.4, ms=4)

    ax.set_title(f"{lbl}  |  Entropy H distribution  |  rate-matched threshold",
                 fontsize=11)
    ax.set_xlabel("Entropy H (nats)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(ry - yt * 0.04, yt * 1.15)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()


# Percentile sweep and Type A emergence
# --------------------------------------

def compute_percentile_sweep(
    uq_cal: pd.DataFrame,
    uq_test: pd.DataFrame,
    cp_flags_test: pd.DataFrame,
    percentiles,
) -> pd.DataFrame:
    """
    Sweeps the entropy threshold across calibration entropy percentiles and records
    the concordance with the fixed ICP flag at each threshold.

    At percentile p the threshold is the p-th percentile of calibration entropy, and a
    test patient is flagged when its entropy exceeds it.

    Input: uq_cal, uq_test, from compute_uncertainty_metrics; cp_flags_test, from
    evaluate_cp_flags; percentiles, an iterable of integers from 1 to 99.
    Output: DataFrame with one row per percentile: t, the four category percentages
    (pct_CC, pct_CU, pct_DA, pct_DB) and their counts.
    """
    cal_h    = uq_cal["entropy"].values
    test_h   = uq_test["entropy"].values
    icp_flag = cp_flags_test["icp_flag"].values
    n        = len(test_h)

    rows = []
    for p in percentiles:
        t_p    = float(np.percentile(cal_h, p))
        h_flag = (test_h > t_p).astype(int)

        cc = int(((icp_flag == 0) & (h_flag == 0)).sum())
        cu = int(((icp_flag == 1) & (h_flag == 1)).sum())
        da = int(((icp_flag == 0) & (h_flag == 1)).sum())
        db = int(((icp_flag == 1) & (h_flag == 0)).sum())

        rows.append({
            "percentile": int(p),
            "t":          round(t_p, 6),
            "pct_CC":     round(cc / n * 100, 2),
            "pct_CU":     round(cu / n * 100, 2),
            "pct_DA":     round(da / n * 100, 2),
            "pct_DB":     round(db / n * 100, 2),
            "n_CC": cc, "n_CU": cu, "n_DA": da, "n_DB": db,
        })

    return pd.DataFrame(rows)


def plot_concordance_sweep(
    sweep_df: pd.DataFrame,
    model_label: str,
    rate_matched_percentile: float,
    figsize=(10, 5),
) -> None:
    """
    Plots the four concordance category percentages across the threshold sweep, with
    the matched-volume threshold as a dashed line.

    Input: sweep_df, from compute_percentile_sweep; model_label, the title label;
    rate_matched_percentile, the matched-volume threshold as a percentile,
    rate_matched['q_level'] * 100; figsize.
    Output: none; the figure is shown.
    """
    color_map = {
        "pct_CC": ("#2ecc71", "Concordant Certain"),
        "pct_CU": ("#3498db", "Concordant Uncertain"),
        "pct_DA": ("#e67e22", "Discordant Type A"),
        "pct_DB": ("#e74c3c", "Discordant Type B"),
    }

    fig, ax = plt.subplots(figsize=figsize)
    x = sweep_df["percentile"].values

    for col, (color, label) in color_map.items():
        ax.plot(x, sweep_df[col].values, color=color, lw=1.8, label=label)

    ax.axvline(rate_matched_percentile, color="black", ls="--", lw=1.3,
               label=f"Rate-matched ({rate_matched_percentile:.1f}th percentile)")

    ax.set_xlabel("Calibration H percentile used as PPU threshold", fontsize=10)
    ax.set_ylabel("% Test patients", fontsize=10)
    ax.set_title(f"{model_label}  |  Concordance proportions across PPU threshold sweep",
                 fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()


def find_type_a_emergence(sweep_df: pd.DataFrame, thresholds=(0.01, 0.05)) -> tuple:
    """
    Finds the first calibration percentile at which Discordant Type A exceeds each
    given proportion.

    Input: sweep_df, from compute_percentile_sweep; thresholds, proportions such as
    (0.01, 0.05).
    Output: tuple with one percentile per threshold, or None where the sweep never
    exceeds it.
    """
    results = []
    for thr in thresholds:
        match = sweep_df[sweep_df["pct_DA"] > thr * 100]
        results.append(int(match.iloc[0]["percentile"]) if len(match) > 0 else None)
    return tuple(results)


# Worked example and treatment pair analysis
# ------------------------------------------

def get_worked_type_b_example(
    model_name: str,
    cp_flags: pd.DataFrame,
    ppu_flags: pd.DataFrame,
    concordance: dict,
    mean_probs_test: np.ndarray,
    rate_matched: dict,
) -> dict:
    """
    Returns the uncertainty profile of the first Discordant Type B patient for a
    worked example.

    A Type B patient has an empty or multi-label ICP set (uncertain) and entropy at or
    below t (certain).

    Input: model_name; cp_flags, ppu_flags, the test flag tables; concordance, from
    run_concordance_analysis_icp (its icp_x_h entry); mean_probs_test, (n, K);
    rate_matched, from compute_rate_matched_threshold.
    Output: dict with patient_pos, true and predicted class, the ICP set and its
    size, entropy H, threshold t, rate r and the mean probability vector, or a dict
    with 'error' when no Type B patient exists.
    """
    detail = concordance["detail_df"]
    db     = detail[detail["category"] == "Discordant Type B"]

    if len(db) == 0:
        return {"error": f"No Discordant Type B patients found for {model_name}."}

    pos     = int(db.iloc[0]["patient_pos"])
    cp_row  = cp_flags[cp_flags["patient_pos"] == pos].iloc[0]
    ppu_row = ppu_flags[ppu_flags["patient_pos"] == pos].iloc[0]

    return {
        "model":       MODEL_LABELS.get(model_name, model_name),
        "patient_pos": pos,
        "true_class":  cp_row["true_class_name"],
        "y_pred":      cp_row["y_pred_name"],
        "ps_icp":      cp_row["ps_icp"],
        "n_set_icp":   int(cp_row["n_set_icp"]),
        "icp_flag":    int(cp_row["icp_flag"]),
        "entropy_H":   round(float(ppu_row["entropy"]), 4),
        "threshold_t": round(float(rate_matched["t"]), 4),
        "rate_r":      round(float(rate_matched["r"]), 4),
        "mean_probs":  {CLASS_NAMES[k]: round(float(mean_probs_test[pos, k]), 4)
                        for k in range(N_CLASSES)},
        "category":    "Discordant Type B (CP uncertain, PPU certain)",
    }


def compute_type_b_pair_distribution(cp_flags: pd.DataFrame, concordance: dict) -> pd.DataFrame:
    """
    Counts the treatment class pairs in the ICP sets of Discordant Type B patients.

    A set of two classes gives one pair, and a larger set gives every pairwise
    combination, each sorted by class index.

    Input: cp_flags, the test flag table; concordance, with a detail_df.
    Output: DataFrame with Pair, N and %, sorted by N descending.
    """
    detail = concordance["detail_df"]
    db_pos = set(detail[detail["category"] == "Discordant Type B"]["patient_pos"].values)
    db_cp  = cp_flags[cp_flags["patient_pos"].isin(db_pos)]

    pair_counts = {}
    for _, row in db_cp.iterrows():
        idx_set = row["ps_icp_idx"]
        if len(idx_set) < 2:
            continue
        for a, b in combinations(sorted(idx_set), 2):
            key = (CLASS_NAMES[a], CLASS_NAMES[b])
            pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["Pair", "N", "%"])

    total = sum(pair_counts.values())
    rows  = [{"Pair": f"{a} + {b}", "N": n, "%": round(n / total * 100, 1)}
             for (a, b), n in sorted(pair_counts.items(), key=lambda x: -x[1])]
    return pd.DataFrame(rows)


# Internal helpers

def _to_0idx(y) -> np.ndarray:
    """
    Converts one-indexed labels to a zero-indexed integer array.

    Input: y, a Series or array of labels 1 to K.
    Output: numpy array of labels 0 to K - 1.
    """
    arr = y.values.ravel() if hasattr(y, "values") else np.ravel(y)
    return (arr - 1).astype(int)


# Main orchestration
# ------------------

def run_uncertainty_rate_matched(
    all_models: dict,
    y_train,
    y_cal,
    y_test,
    split_indices: dict,
    config: dict,
    alpha: float = 0.05,
) -> dict:
    """
    Runs the uncertainty analysis for every model and saves the result.

    Per model the steps are: entropy decomposition on all three splits; ICP and
    Mondrian thresholds from calibration; ICP flags on calibration, used only to
    measure r; the entropy threshold t; ICP and entropy flags on test; and the ICP by
    entropy concordance table.

    Input: all_models, model key to results dict with all_probs_{split},
    mean_probs_{split} and y_pred_{split}; y_train, y_cal, y_test, one-indexed
    labels; split_indices, carried for the caller and unused here; config, a dict
    with results_dir and version; alpha, the miscoverage level.
    Output: dict keyed by model, each with train, cal and test (per-patient H, MI and
    AU), thr_icp, thr_mondrian, rate_matched (r, t, q_level), cp_flags and ppu_flags
    for test, and concordance. The dict is also saved as a pickle.
    """
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION")
    print("=" * 70)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    y_tr = _to_0idx(y_train)
    y_ca = _to_0idx(y_cal)
    y_te = _to_0idx(y_test)

    out = {}

    for model_name, res in all_models.items():
        lbl = MODEL_LABELS.get(model_name, model_name)
        print(f"\n  {lbl}")
        print(f"  {'─' * 55}")

        # Step 1: entropy decomposition across all splits
        print("  Computing entropy decomposition...")
        uq = {
            "train": compute_uncertainty_metrics(res["all_probs_train"]),
            "cal":   compute_uncertainty_metrics(res["all_probs_cal"]),
            "test":  compute_uncertainty_metrics(res["all_probs_test"]),
        }
        for split, df in uq.items():
            print(f"    {split:5s}: H={df['entropy'].mean():.4f}  "
                  f"MI={df['mutual_information'].mean():.4f}  "
                  f"AU={df['aleatoric_uncertainty'].mean():.4f}")

        # Steps 2 and 3: CP thresholds from calibration
        print(f"  ICP threshold (alpha={alpha})...")
        thr_icp = compute_icp_threshold(res["mean_probs_cal"], y_ca, alpha)
        print(f"    q_icp={thr_icp['q_icp']:.4f}  "
              f"(q_level={thr_icp['q_level']:.4f},  n_cal={thr_icp['n_cal']})")

        thr_mondrian = compute_mondrian_thresholds(res["mean_probs_cal"], y_ca, alpha)

        # Step 4: calibration CP flags, used only to compute r
        cp_flags_cal = evaluate_cp_flags(res["mean_probs_cal"], thr_icp, thr_mondrian, y_ca)
        n_cal        = len(cp_flags_cal)
        n_cal_multi  = int((cp_flags_cal["icp_flag"] == 1).sum())

        # Step 5: rate-matched PPU threshold
        print("  Rate-matched PPU threshold...")
        rate_matched = compute_rate_matched_threshold(uq["cal"], cp_flags_cal)
        print(f"    r={rate_matched['r']:.4f}  "
              f"(calibration multi-label rate: {n_cal_multi}/{n_cal} patients)")
        print(f"    t={rate_matched['t']:.4f}  "
              f"({rate_matched['q_level']:.4f} quantile of calibration H)")

        # Step 6: CP flags on test set
        print("  CP flags on test set...")
        cp_flags = evaluate_cp_flags(res["mean_probs_test"], thr_icp, thr_mondrian, y_te)
        n  = len(cp_flags)
        ni = int((cp_flags["icp_flag"] == 0).sum())
        print(f"    ICP: singleton={ni} ({ni/n:.1%})   "
              f"multi-label={n-ni} ({(n-ni)/n:.1%})")

        # Step 7: PPU flags on test set (H thresholded at t)
        print("  PPU flags on test set (H thresholded at t)...")
        ppu_flags = evaluate_ppu_flags_rate_matched(uq["test"], rate_matched["t"])
        nh = int((ppu_flags["h_flag"] == 1).sum())
        print(f"    H above t: {nh} ({nh/n:.1%})")

        # Step 8: concordance, ICP by entropy
        print("  Concordance (ICP x H)...")
        concordance = run_concordance_analysis_icp(cp_flags, ppu_flags, model_name)
        tbl = concordance["icp_x_h"]
        for cat in CONCORDANCE_CATEGORIES:
            cnt = tbl["counts"][cat]
            pct = tbl["proportions"][cat]
            print(f"    {cat:<30s}: {cnt:4d}  ({pct*100:.1f}%)")

        out[model_name] = {
            "train":        uq["train"],
            "cal":          uq["cal"],
            "test":         uq["test"],
            "thr_icp":      thr_icp,
            "thr_mondrian": thr_mondrian,
            "rate_matched": rate_matched,
            "cp_flags":     cp_flags,
            "ppu_flags":    ppu_flags,
            "concordance":  concordance,
        }

    # Save full results
    version   = config.get("version", "v01")
    save_path = results_dir / f"uncertainty_results_{version}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n  Saved: {save_path}")
    print("=" * 70)

    return out


# =====================================================================
# M CONVERGENCE AND ALPHA SENSITIVITY
# Convergence uses subsets of the stored draws, and the alpha sensitivity reuses
# the stored entropy values. Neither retrains a model.
# =====================================================================

def _balanced_accuracy(y_true_idx, y_pred_idx) -> float:
    """
    Computes balanced accuracy, the mean per-class recall, with numpy alone.

    Input: y_true_idx, y_pred_idx, zero-indexed labels and predictions.
    Output: the balanced accuracy as a float.
    """
    y_true_idx = np.asarray(y_true_idx)
    y_pred_idx = np.asarray(y_pred_idx)
    recalls = []
    for k in np.unique(y_true_idx):
        mask = y_true_idx == k
        if mask.sum() > 0:
            recalls.append(float((y_pred_idx[mask] == k).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def run_m_convergence(res: dict, y_cal, y_test, m_grid, alpha: float = 0.05) -> pd.DataFrame:
    """
    Checks whether the number of draws M is large enough for one model.

    For each M in m_grid the first M stored draws are used, and the mean
    probabilities, balanced accuracy, entropy and ICP by entropy concordance are
    recomputed.

    Input: res, the results dict with all_probs_cal and all_probs_test; y_cal,
    y_test, one-indexed labels; m_grid, the draw counts (values above the stored M
    are skipped); alpha, the miscoverage level.
    Output: DataFrame with one row per M: balanced accuracy, mean test entropy, the
    four concordance percentages and the concordant percentage (CC + CU).
    """
    y_ca = _to_0idx(y_cal)
    y_te = _to_0idx(y_test)
    ap_cal  = res["all_probs_cal"]
    ap_test = res["all_probs_test"]
    m_max   = ap_test.shape[0]

    rows = []
    for m in m_grid:
        if m > m_max:
            continue
        mp_cal_m  = ap_cal[:m].mean(axis=0)
        mp_test_m = ap_test[:m].mean(axis=0)
        uq_cal_m  = compute_uncertainty_metrics(ap_cal[:m])
        uq_test_m = compute_uncertainty_metrics(ap_test[:m])

        y_pred_m = mp_test_m.argmax(axis=1)
        bal_acc  = _balanced_accuracy(y_te, y_pred_m)

        thr_icp = compute_icp_threshold(mp_cal_m, y_ca, alpha)
        thr_mon = compute_mondrian_thresholds(mp_cal_m, y_ca, alpha)
        cp_cal  = evaluate_cp_flags(mp_cal_m, thr_icp, thr_mon, y_ca)
        rm      = compute_rate_matched_threshold(uq_cal_m, cp_cal)
        cp_test = evaluate_cp_flags(mp_test_m, thr_icp, thr_mon, y_te)
        ppu_te  = evaluate_ppu_flags_rate_matched(uq_test_m, rm["t"])
        conc    = compute_concordance_table(cp_test, ppu_te, "icp_flag", "h_flag")
        p       = conc["proportions"]

        rows.append({
            "M":                      m,
            "Bal. Acc. (test)":       round(bal_acc, 4),
            "Mean H (test)":          round(float(uq_test_m["entropy"].mean()), 4),
            "% Concordant Certain":   round(p["Concordant Certain"] * 100, 1),
            "% Concordant Uncertain": round(p["Concordant Uncertain"] * 100, 1),
            "% Discordant Type A":    round(p["Discordant Type A"] * 100, 1),
            "% Discordant Type B":    round(p["Discordant Type B"] * 100, 1),
            "% Concordant (CC+CU)":   round((p["Concordant Certain"] + p["Concordant Uncertain"]) * 100, 1),
        })
    return pd.DataFrame(rows)


def find_m_stabilisation(conv_df: pd.DataFrame, tol_ba: float = 0.005,
                         tol_h: float = 0.005, tol_conc: float = 0.5) -> dict:
    """
    Finds the smallest M at which balanced accuracy, mean test entropy and the
    concordant percentage all lie within tolerance of their values at the largest M.

    Input: conv_df, from run_m_convergence; tol_ba, the balanced accuracy tolerance;
    tol_h, the entropy tolerance in nats; tol_conc, the concordance tolerance in
    percentage points.
    Output: dict with m_stable, the reference values at the largest M, and the
    three tolerances.
    """
    df     = conv_df.sort_values("M").reset_index(drop=True)
    ref_ba = float(df["Bal. Acc. (test)"].iloc[-1])
    ref_h  = float(df["Mean H (test)"].iloc[-1])
    ref_c  = float(df["% Concordant (CC+CU)"].iloc[-1])
    m_stable = int(df["M"].iloc[-1])
    for _, r in df.iterrows():
        if (abs(r["Bal. Acc. (test)"] - ref_ba) <= tol_ba
                and abs(r["Mean H (test)"] - ref_h) <= tol_h
                and abs(r["% Concordant (CC+CU)"] - ref_c) <= tol_conc):
            m_stable = int(r["M"])
            break
    return {"m_stable": m_stable, "ref_bal_acc": round(ref_ba, 4),
            "ref_mean_H": round(ref_h, 4), "ref_concordant_pct": round(ref_c, 1),
            "tol_ba": tol_ba, "tol_h": tol_h, "tol_conc": tol_conc}


def plot_m_convergence(conv_df: pd.DataFrame, model_label: str, m_stable: int = None, figsize=(15, 4)) -> None:
    """
    Plots balanced accuracy, mean test entropy and the concordant percentage against M.

    Input: conv_df, from run_m_convergence; model_label; m_stable, marked as a dashed
    line when given; figsize.
    Output: none; the figure is shown.
    """
    df = conv_df.sort_values("M")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    ax0.plot(df["M"], df["Bal. Acc. (test)"], marker="o", color="#7030a0")
    ax0.set_xlabel("M (samples)"); ax0.set_ylabel("Balanced accuracy (test)")
    ax0.set_title(f"{model_label}: balanced accuracy vs M")
    ax1.plot(df["M"], df["Mean H (test)"], marker="o", color="#4472c4")
    ax1.set_xlabel("M (samples)"); ax1.set_ylabel("Mean test entropy H")
    ax1.set_title(f"{model_label}: entropy vs M")
    ax2.plot(df["M"], df["% Concordant (CC+CU)"], marker="o", color="#2e8b57")
    ax2.set_xlabel("M (samples)"); ax2.set_ylabel("% Concordant (CC+CU)")
    ax2.set_title(f"{model_label}: concordant rate vs M")
    if m_stable is not None:
        for ax in (ax0, ax1, ax2):
            ax.axvline(m_stable, color="#c00000", ls="--", lw=1, label=f"M stable = {m_stable}")
        ax2.legend(fontsize=8)
    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.show()


def run_alpha_sensitivity(res: dict, uq_model: dict, y_cal, y_test, alphas) -> pd.DataFrame:
    """
    Recomputes the ICP by entropy concordance for one model at each miscoverage level
    alpha.

    Entropy does not depend on alpha, so the stored calibration and test entropy
    tables are reused. For each alpha the ICP threshold, the rate r and the threshold
    t are recomputed with all M draws.

    Input: res, the results dict with mean_probs_cal and mean_probs_test; uq_model,
    a dict with the 'cal' and 'test' entropy tables; y_cal, y_test, one-indexed
    labels; alphas, the miscoverage levels.
    Output: DataFrame with one row per alpha: the four concordance percentages, the
    concordant percentage, r and t.
    """
    y_ca = _to_0idx(y_cal)
    y_te = _to_0idx(y_test)
    mp_cal  = res["mean_probs_cal"]
    mp_test = res["mean_probs_test"]
    uq_cal  = uq_model["cal"]
    uq_test = uq_model["test"]

    rows = []
    for a in alphas:
        thr_icp = compute_icp_threshold(mp_cal, y_ca, a)
        thr_mon = compute_mondrian_thresholds(mp_cal, y_ca, a)
        cp_cal  = evaluate_cp_flags(mp_cal, thr_icp, thr_mon, y_ca)
        rm      = compute_rate_matched_threshold(uq_cal, cp_cal)
        cp_test = evaluate_cp_flags(mp_test, thr_icp, thr_mon, y_te)
        ppu_te  = evaluate_ppu_flags_rate_matched(uq_test, rm["t"])
        conc    = compute_concordance_table(cp_test, ppu_te, "icp_flag", "h_flag")
        p       = conc["proportions"]
        rows.append({
            "alpha":                  a,
            "% Concordant Certain":   round(p["Concordant Certain"] * 100, 1),
            "% Concordant Uncertain": round(p["Concordant Uncertain"] * 100, 1),
            "% Discordant Type A":    round(p["Discordant Type A"] * 100, 1),
            "% Discordant Type B":    round(p["Discordant Type B"] * 100, 1),
            "% Concordant (CC+CU)":   round((p["Concordant Certain"] + p["Concordant Uncertain"]) * 100, 1),
            "r (multi-label rate)":   round(rm["r"], 4),
            "t (H threshold)":        round(rm["t"], 4),
        })
    return pd.DataFrame(rows)


def plot_alpha_sensitivity(alpha_df: pd.DataFrame, model_label: str, primary_alpha: float = 0.05, figsize=(11, 4)) -> None:
    """
    Plots the concordance categories, and the rate r and threshold t, against alpha.

    Input: alpha_df, from run_alpha_sensitivity; model_label; primary_alpha, marked
    as a dotted line; figsize.
    Output: none; the figure is shown.
    """
    df = alpha_df.sort_values("alpha")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for col, colr in [("% Concordant Certain", "#2ecc71"), ("% Concordant Uncertain", "#3498db"),
                      ("% Discordant Type A", "#e67e22"), ("% Discordant Type B", "#e74c3c")]:
        ax1.plot(df["alpha"], df[col], marker="o", label=col.replace("% ", ""), color=colr)
    ax1.set_xlabel("alpha"); ax1.set_ylabel("% of test patients")
    ax1.set_title(f"{model_label}: concordance categories vs alpha"); ax1.legend(fontsize=7)
    ax2.plot(df["alpha"], df["r (multi-label rate)"], marker="o", label="r", color="#8e44ad")
    ax2.plot(df["alpha"], df["t (H threshold)"], marker="s", label="t", color="#16a085")
    ax2.set_xlabel("alpha"); ax2.set_ylabel("value")
    ax2.set_title(f"{model_label}: rate-matched r and t vs alpha"); ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.axvline(primary_alpha, color="#999999", ls=":", lw=1)
        ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.show()
