"""
figures.py: the paper figures, drawn from a finished run of each representation.

Thresholds, flag rates and the alpha sweep come from the saved tables of the run.
The two per-patient
coordinates of the flag scatter are the only values computed here, and they are
checked against the saved tables before any figure is drawn.

    Figure3_flag_scatter         entropy against the second-largest mean predicted
                                 probability for the 505 test patients, one panel
                                 per model and representation, with both thresholds
                                 drawn so the four quadrants are the four flag groups
    Figure4_coverage_resampling  conformal coverage over 200 calibration resamples
                                 and 500 calibration and test re-splits, as a mean
                                 with the 2.5 to 97.5 percentile range, and the
                                 reported split as a dot
    Figure5_coverage_by_class    conformal coverage within each treatment class,
                                 with Wilson intervals
    Figure_alpha_sweep           coverage, set size, flag rates, agreement and error
                                 recall as alpha runs from 0 to 1, as the median
                                 across the five models with the range between them

Three checks run before any figure is drawn, each comparing a quantity computed
here with the same quantity in the saved tables:
    check 1   the conformal threshold recomputed from the stored calibration
              probabilities against q_conformal
    check 2   the entropy threshold recomputed with the entropy function below
              against t_entropy
    check 3   the two test flag rates rebuilt from those thresholds against the
              saved test flag rates
A failed check stops the script with no figure written. Check 1 also recognises a
run folder whose conformal threshold came from NumPy's interpolated quantile. The
figures from such a run carry a PROVISIONAL suffix and a matching caption note.

Run from the repository root:  python -m tables_figures.figures
"""

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import config as C

DIR_CODE = Path(__file__).resolve().parent.parent
DIR_OUT = DIR_CODE / "results" / "paper"

# EPS and the entropy expression match the pipeline's entropy function. Check 2
# recomputes t from the entropy below and compares it with the saved t.
EPS = 1e-10

# TOL: one unit of the last decimal in the saved tables.
TOL = 1.5e-4

# Okabe-Ito hues keep their separation under deuteranopia and protanopia. Marker
# shape carries the same distinction a second time.
COL = {"CC": "#0072B2", "CU": "#E69F00", "DA": "#009E73", "DB": "#D55E00"}
MARK = {"CC": "o", "CU": "o", "DA": "^", "DB": "^"}
GROUP_LABEL = {
    "CC": "Both certain",
    "CU": "Both uncertain",
    "DA": "Entropy only",
    "DB": "Conformal only",
}
COL_CP, COL_H = "#0072B2", "#D55E00"
INK, MUTED = "#1c1c1a", "#6f6f68"


def entropy(p: np.ndarray) -> np.ndarray:
    """
    Computes the Shannon entropy (nats) of each row, with the same EPS clipping as the
    pipeline's entropy function.

    Input: p, (n, K) probabilities.
    Output: (n,) entropies.
    """
    p = np.clip(p, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def newest_run(representation: str) -> Path:
    """
    Returns the most recent full (non-smoke) run folder for one representation.

    Input: representation, a key of config.REPRESENTATIONS.
    Output: the run folder path. Raises FileNotFoundError when none exists.
    """
    runs = sorted(d for d in (DIR_CODE / "results").glob(f"run_*_{representation}")
                  if d.is_dir() and not d.name.startswith("run_smoke"))
    if not runs:
        raise FileNotFoundError(f"No run folder for {representation} under results.")
    return runs[-1]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Returns the split-conformal threshold: the k-th smallest score, with
    k = ceil((n + 1)(1 - alpha)).

    Input: scores, the calibration nonconformity scores; alpha, the miscoverage level.
    Output: the threshold q.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[min(max(k, 1), n) - 1])


def superseded_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Returns the threshold from NumPy's default interpolated quantile at the conformal
    level ceil((n + 1)(1 - alpha)) / n.

    The interpolated value lies between the k-th and the (k + 1)-th smallest score.
    Check 1 compares a saved threshold with both rules, so a run folder made with
    the interpolated quantile is recognised and labelled.

    Input: scores; alpha.
    Output: the interpolated threshold.
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    level = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
    return float(np.quantile(s, level))


def load(representation: str) -> dict:
    """
    Reads one representation's newest run and rebuilds the per-patient coordinates
    of the flag scatter.

    Input: representation, a key of config.REPRESENTATIONS.
    Output: dict with run (the folder), models (per-model p2, entropy, both cut
    points and flag group), sweep (the full alpha sweep), n_test and checks (one row
    per model with the three verification results).
    """
    rep = C.REPRESENTATIONS[representation]
    tag, ver = rep["tag"], rep["version"]
    run = newest_run(representation)

    y_cal = pd.read_csv(DIR_CODE / f"data/cal/y_{tag}_calibration.csv").iloc[:, -1].values.astype(int) - 1
    y_test = pd.read_csv(DIR_CODE / f"data/test/y_{tag}_test.csv").iloc[:, -1].values.astype(int) - 1

    thr = pd.read_csv(run / f"eval_2_{tag}_thresholdsAndFlags/thresholds_and_flag_rates_{tag}.csv")
    sweep = pd.read_csv(run / f"eval_5_{tag}_sensitivity/alpha_sweep_full_{tag}.csv")

    models, checks = {}, []
    for key in C.MODEL_KEYS:
        label = C.MODEL_LABELS[key]
        with open(run / f"models/results_{key}_{ver}.pkl", "rb") as fh:
            d = pickle.load(fh)
        mp_cal = np.asarray(d["mean_probs_cal"])
        mp_test = np.asarray(d["mean_probs_test"])

        row = thr.loc[thr["Model"] == label]
        if row.empty:
            raise KeyError(f"{label} missing from the eval_2 table in {run.name}.")
        row = row.iloc[0]
        q_saved = float(row["q_conformal"])
        t_saved = float(row["t_entropy"])
        r_saved = float(row["r_cal_conformal"])

        h_cal, h_test = entropy(mp_cal), entropy(mp_test)
        p2 = np.sort(mp_test, axis=1)[:, -2]

        # The saved tables carry four decimals, so two values agree when they lie within
        # one unit of the last digit (TOL). One patient moves a rate on 505 patients by
        # 0.0020, so this tolerance still separates a single patient.
        scores_cal = 1.0 - mp_cal[np.arange(len(y_cal)), y_cal]
        # The flags are rebuilt from the full-precision threshold. With the rounded value,
        # a patient whose second-largest probability lies inside the last rounded digit of
        # q would cross the cut.
        q_rule_full = conformal_quantile(scores_cal, C.CP_ALPHA)
        q_old_full = superseded_quantile(scores_cal, C.CP_ALPHA)
        q_rule, q_old = round(q_rule_full, 4), round(q_old_full, 4)
        t_full = float(pd.Series(h_cal).quantile(1.0 - r_saved))
        t_here = round(t_full, 4)

        # The threshold behind the run is the candidate the saved value rounds to, and the
        # flags are rebuilt from it.
        q_use = q_rule_full if abs(q_saved - q_rule) <= TOL else q_old_full
        cp = (p2 >= 1.0 - q_use).astype(int)
        h = (h_test > t_full).astype(int)

        # The two candidate rules can round to the same value, and then this model cannot
        # separate them. The verdict comes from the models where they differ.
        d_rule, d_old = abs(q_saved - q_rule), abs(q_saved - q_old)
        if abs(q_rule - q_old) <= TOL:
            verdict = "indistinguishable"
        elif d_rule <= TOL and d_rule < d_old:
            verdict = "order statistic"
        elif d_old <= TOL:
            verdict = "superseded call"
        else:
            verdict = "neither"

        cp_rate, h_rate = float(cp.mean()), float(h.mean())
        checks.append({
            "representation": representation, "Model": label,
            "q_saved": q_saved, "q_order_statistic": q_rule, "q_superseded_call": q_old,
            "q_verdict": verdict,
            "t_saved": t_saved, "t_recomputed": t_here,
            "t_match": abs(t_here - t_saved) <= TOL,
            "cp_rate_saved": round(float(row["test_flag_rate_conformal"]), 4),
            "cp_rate_rebuilt": round(cp_rate, 4),
            "cp_match": abs(cp_rate - float(row["test_flag_rate_conformal"])) <= TOL,
            "h_rate_saved": round(float(row["test_flag_rate_entropy"]), 4),
            "h_rate_rebuilt": round(h_rate, 4),
            "h_match": abs(h_rate - float(row["test_flag_rate_entropy"])) <= TOL,
        })

        group = np.where((cp == 0) & (h == 0), "CC",
                 np.where((cp == 1) & (h == 1), "CU",
                  np.where((cp == 0) & (h == 1), "DA", "DB")))
        models[key] = {"label": label, "p2": p2, "h": h_test,
                       "cut_p2": 1.0 - q_use, "cut_h": t_full, "group": group}

    return {"run": run, "models": models, "sweep": sweep, "n_test": len(y_test),
            "checks": pd.DataFrame(checks)}


def figure_3(data: dict, provisional: bool) -> Path:
    """
    Draws the flag scatter: entropy against the second-largest mean predicted
    probability, with both thresholds.

    The vertical line lies at 1 - q, since a prediction set holds two or more classes
    exactly when the second-largest probability reaches it. The horizontal line lies
    at t. The four quadrants are therefore the four flag groups.

    Input: data, representation key to the output of load; provisional, adds the
    PROVISIONAL suffix when True.
    Output: the path of the PNG written (a PDF is written beside it).
    """
    reps = ["indvA", "sumB"]
    # Every panel shares the same axes. The second-largest of four probabilities is at
    # most 0.5, and the entropy of four classes is at most ln(4) = 1.386, so both
    # limits come from the task. Shared axes show where t lies across models.
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.6), sharex=True, sharey=True)

    for r, rep in enumerate(reps):
        for c, key in enumerate(C.MODEL_KEYS):
            ax = axes[r, c]
            m = data[rep]["models"][key]
            for g in ["CC", "CU", "DA", "DB"]:
                sel = m["group"] == g
                ax.scatter(m["p2"][sel], m["h"][sel], s=9, marker=MARK[g],
                           c=COL[g], alpha=0.65, linewidths=0.3, edgecolors="white",
                           label=GROUP_LABEL[g] if (r == 0 and c == 0) else None)
            ax.axvline(m["cut_p2"], color=INK, lw=1.0, ls="--")
            ax.axhline(m["cut_h"], color=INK, lw=1.0, ls="--")
            ax.set_title(m["label"].replace(" Neural Network", "").replace(" Ensemble", "")
                         .replace(" Logistic Regression", " LR"), fontsize=9, color=INK)
            ax.set_xlim(0.0, 0.5)
            ax.set_ylim(0.0, 1.4)
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#d8d8d2")
            if c == 0:
                ax.set_ylabel(f"{C.REPRESENTATIONS[rep]['label']}\n\npredictive entropy",
                              fontsize=8, color=INK)
            if r == 1:
                ax.set_xlabel("second-largest probability", fontsize=8, color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005), markerscale=1.8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    name = "Figure3_flag_scatter" + ("_PROVISIONAL" if provisional else "")
    out = DIR_OUT / f"{name}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(DIR_OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    return out


def _band(ax, df, col, colour, label, scale=1.0):
    """
    Draws the median across the five models as a line, with the range between them as
    a band.

    Input: ax, the axes; df, the alpha sweep; col, the column to draw; colour; label;
    scale, a multiplier for the values.
    Output: none.
    """
    g = df.groupby("alpha")[col]
    med, lo, hi = g.median() * scale, g.min() * scale, g.max() * scale
    ax.fill_between(med.index, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.plot(med.index, med, color=colour, lw=1.6, label=label)


def figure_4(data: dict, provisional: bool) -> Path:
    """
    Draws the alpha sweep figure: the comparison recomputed at every alpha from 0 to 1
    in steps of 0.01.

    Each panel shows the median across the five models with the range between them,
    and the per-model values stay in the saved sweep table. The dashed vertical line
    marks the pre-specified alpha of 0.05.

    Input: data, representation key to the output of load; provisional, adds the
    PROVISIONAL suffix when True.
    Output: the path of the PNG written (a PDF is written beside it).
    """
    rows = [
        ("coverage_test", None, "empirical coverage", 1.0),
        ("mean_set_size", None, "mean set size", 1.0),
        ("test_flag_rate_conformal", "test_flag_rate_entropy", "test flag rate", 1.0),
        ("n_flag_conformal", "n_flag_entropy", "patients in review queue", 1.0),
        ("pct_concordant", "kappa", "agreement and kappa", None),
        ("n_concordant", None, "concordant patients", 1.0),
        ("n_discordant", None, "discordant patients", 1.0),
        ("recall_conformal", "recall_entropy", "error recall", 1.0),
    ]
    reps = ["indvA", "sumB"]
    # The panels cover alpha 0 to 1. An empty set is flagged uncertain, and empty sets
    # appear once q falls below 0.75, from about alpha 0.17, so the flag rates,
    # agreement and recall keep changing up to alpha 1.
    x_max = 1.0
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.8, 2.45 * len(rows)), sharex=True)

    for c, rep in enumerate(reps):
        # The sweep is cut to the drawn range first, so each panel's vertical scale fits
        # the part of the sweep it shows.
        df = data[rep]["sweep"]
        df = df.loc[df["alpha"] <= x_max].copy()
        # The three count rows are patients out of n_test. The sweep stores rates to four
        # decimals, so each count is the rate times n_test rounded to the nearest patient,
        # and the check below stops the figure when a product lies further than 0.05 from
        # a whole patient. Concordant patients receive the same flag from both methods;
        # discordant patients are flagged by one method only.
        n = data[rep]["n_test"]
        for col, src, sc in [("n_flag_conformal", "test_flag_rate_conformal", 1.0),
                             ("n_flag_entropy", "test_flag_rate_entropy", 1.0),
                             ("n_concordant", "pct_concordant", 0.01)]:
            raw = df[src] * sc * n
            if (raw - raw.round()).abs().max() > 0.05:
                raise SystemExit(f"{src} times {n} is not a whole patient count in {rep}.")
            df[col] = raw.round()
        df["n_discordant"] = n - df["n_concordant"]
        for r, (a_col, b_col, ylab, scale) in enumerate(rows):
            ax = axes[r, c]
            if ylab == "agreement and kappa":
                _band(ax, df, "pct_concordant", COL_CP, "observed agreement", 0.01)
                _band(ax, df, "kappa", COL_H, "kappa", 1.0)
            elif b_col is None:
                _band(ax, df, a_col, COL_CP, ylab, scale)  # the y axis label names a single-series panel
                if a_col == "coverage_test":
                    ax.plot(sorted(df["alpha"].unique()),
                            1 - np.array(sorted(df["alpha"].unique())),
                            color=MUTED, lw=1.0, ls=":", label="target 1 - alpha")
            else:
                _band(ax, df, a_col, COL_CP, "conformal", scale)
                _band(ax, df, b_col, COL_H, "entropy", scale)
            ax.axvline(C.CP_ALPHA, color=INK, lw=0.9, ls="--")
            ax.set_xlim(0.0, x_max)
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#d8d8d2")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8, color=INK)
            if r == 0:
                ax.set_title(C.REPRESENTATIONS[rep]["label"], fontsize=10, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("alpha", fontsize=8, color=INK)
            # A single-series panel is named by its axis label, so a legend appears only where
            # two series share a panel.
            handles, labs = ax.get_legend_handles_labels()
            if c == 1 and len(handles) > 1:
                # The coverage panel falls to zero at alpha 1, so its legend stays top right. The
                # other panels rise towards 1 there, so their legend sits at mid height on the
                # right, clear of the lines.
                ax.legend(fontsize=7, frameon=False,
                          loc="upper right" if ylab == "empirical coverage" else "center right")

    fig.tight_layout()
    name = "Figure_alpha_sweep" + ("_PROVISIONAL" if provisional else "")
    out = DIR_OUT / f"{name}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(DIR_OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    return out


# Model order and short names for the paper tables.
TABLE_ORDER = ["bootstrap", "bayesian", "svm", "mcd", "xgboost"]
SHORT = {"bootstrap": "Boot LR", "bayesian": "Bayes LR", "svm": "SVM",
         "mcd": "NN-MCD", "xgboost": "XGB"}
DATASET = {"indvA": "Dataset A", "sumB": "Dataset B"}


def _range(txt: str) -> tuple:
    """
    Parses a saved "[lo, hi]" range string into two floats.

    Input: txt, the string.
    Output: (lo, hi).
    """
    lo, hi = txt.strip("[]").split(",")
    return float(lo), float(hi)


def _style(ax):
    """
    Applies the shared axis style: light spines and a light horizontal grid.

    Input: ax, the axes.
    Output: none.
    """
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color("#d8d8d2")
    ax.grid(axis="y", color="#ececea", lw=0.6)
    ax.set_axisbelow(True)


def figure_coverage_resampling() -> Path:
    """
    Draws conformal coverage over two resampling schemes, with the reported split as
    a dot.

    The 200 calibration resamples hold the 505 test patients fixed, and the 500
    re-splits redraw calibration and test together, the draw the coverage guarantee
    averages over. The re-split values come from results/coverage_by_resplit.csv.

    Output: the path of the PNG written (a PDF is written beside it).
    """
    resplit = pd.read_csv(DIR_CODE / "results" / "coverage_by_resplit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    w = 0.26
    for ax, rep in zip(axes, ["indvA", "sumB"]):
        tag = C.REPRESENTATIONS[rep]["tag"]
        run = newest_run(rep)
        cal = pd.read_csv(run / f"eval_5_{tag}_sensitivity/calibration_resampling_{tag}.csv")
        obs = pd.read_csv(run / f"eval_1_{tag}_conformalValidity/coverage_marginal_test_{tag}.csv")
        rs = resplit[resplit["rep"] == rep[-1]]
        for i, key in enumerate(TABLE_ORDER):
            label = C.MODEL_LABELS[key]
            c = cal.loc[cal["Model"] == label].iloc[0]
            r = rs.loc[rs["model"] == label].iloc[0]
            o = float(obs.loc[obs["Model"] == label, "empirical_coverage"].iloc[0])
            for x, mean, rng, col, lab in [
                (i - w / 2 - 0.02, c["coverage_mean"], c["coverage_range"], COL_CP,
                 "calibration resampled 200 times, test fixed"),
                (i + w / 2 + 0.02, r["mean_over_splits"], r["range_over_splits"], COL_H,
                 "calibration and test re-split 500 times")]:
                lo, hi = _range(rng)
                ax.add_patch(plt.Rectangle((x - w / 2, lo), w, hi - lo, color=col,
                                           alpha=0.22, lw=0,
                                           label=lab if i == 0 else None))
                ax.plot([x - w / 2, x + w / 2], [mean, mean], color=col, lw=2.0)
            ax.scatter([i], [o], s=34, color=INK, zorder=5, edgecolors="white",
                       linewidths=0.8, label="reported split" if i == 0 else None)
        ax.axhline(0.95, color=INK, lw=0.9, ls="--", label="target 0.95")
        ax.set_xticks(range(len(TABLE_ORDER)))
        ax.set_xticklabels([SHORT[k] for k in TABLE_ORDER], fontsize=8, color=INK)
        ax.set_xlim(-0.6, len(TABLE_ORDER) - 0.4)
        ax.set_title(DATASET[rep], fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("empirical coverage", fontsize=9, color=INK)
    axes[0].set_ylim(0.86, 0.99)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = DIR_OUT / "Figure4_coverage_resampling.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(DIR_OUT / "Figure4_coverage_resampling.pdf", facecolor="white")
    plt.close(fig)
    return out


def figure_coverage_by_class() -> Path:
    """
    Draws conformal coverage within each treatment class, with Wilson intervals, from
    the saved coverage-by-class tables.

    There is one panel per class with ten points each, five models under dataset A and
    dataset B, offset and separated by marker shape as well as colour.

    Output: the path of the PNG written (a PDF is written beside it).
    """
    frames = []
    for rep in ["indvA", "sumB"]:
        tag = C.REPRESENTATIONS[rep]["tag"]
        df = pd.read_csv(newest_run(rep) /
                         f"eval_1_{tag}_conformalValidity/coverage_by_class_test_{tag}.csv")
        frames.append(df.assign(rep=rep))
    df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, C.N_CLASSES, figsize=(12.4, 3.6), sharey=True)
    style = {"indvA": (COL_CP, "o", -0.14), "sumB": (COL_H, "s", 0.14)}
    for k, ax in enumerate(axes):
        sub = df[df["class_idx"] == k]
        n = int(sub["n_test"].iloc[0])
        for rep, (col, mk, dx) in style.items():
            for i, key in enumerate(TABLE_ORDER):
                row = sub[(sub["rep"] == rep) & (sub["Model"] == C.MODEL_LABELS[key])].iloc[0]
                cov = float(row["coverage"])
                ax.errorbar(i + dx, cov,
                            yerr=[[cov - float(row["ci_lo"])], [float(row["ci_hi"]) - cov]],
                            fmt=mk, color=col, ms=5.5, elinewidth=1.2, capsize=2.5,
                            mec="white", mew=0.6,
                            label=DATASET[rep] if (k == 0 and i == 0) else None)
        ax.axhline(0.95, color=INK, lw=0.9, ls="--",
                   label="target 0.95" if k == 0 else None)
        ax.set_title(f"{C.CLASS_NAMES[k]} (n = {n})", fontsize=9.5, color=INK)
        ax.set_xticks(range(len(TABLE_ORDER)))
        ax.set_xticklabels([SHORT[m] for m in TABLE_ORDER], fontsize=7.5, color=INK,
                           rotation=30, ha="right")
        ax.set_xlim(-0.6, len(TABLE_ORDER) - 0.4)
        _style(ax)
    axes[0].set_ylabel("coverage within class", fontsize=9, color=INK)
    axes[0].set_ylim(0.58, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = DIR_OUT / "Figure5_coverage_by_class.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(DIR_OUT / "Figure5_coverage_by_class.pdf", facecolor="white")
    plt.close(fig)
    return out


def main():
    """
    Runs the three checks, then writes the four figures, the verification table and
    the figure captions to results/paper.
    """
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    data = {rep: load(rep) for rep in ["indvA", "sumB"]}
    checks = pd.concat([data[r]["checks"] for r in data], ignore_index=True)

    pd.set_option("display.width", 220)
    print("VERIFICATION")
    for col, what in [("t_match", "entropy threshold"),
                      ("cp_match", "conformal test flag rate"),
                      ("h_match", "entropy test flag rate")]:
        ok = int(checks[col].sum())
        print(f"  {what:28s} {ok} of {len(checks)} reproduce the saved table")
        if ok < len(checks):
            print(checks.loc[~checks[col]].to_string(index=False))
            raise SystemExit("A rebuilt quantity disagrees with the pipeline output. "
                             "No figure written.")

    counts = checks["q_verdict"].value_counts().to_dict()
    print(f"  {'conformal threshold':28s} " +
          ", ".join(f"{v} {k}" for k, v in counts.items()))
    if counts.get("neither"):
        print(checks.loc[checks["q_verdict"] == "neither"].to_string(index=False))
        raise SystemExit("A saved threshold matches neither the order statistic nor the "
                         "call it replaced. No figure written.")
    provisional = bool(counts.get("superseded call"))
    if provisional:
        print("\n  This run's conformal thresholds come from NumPy's interpolated quantile,")
        print("  and every saved threshold matches that quantile. The")
        print("  coordinates below are faithful to the run that produced them, and the")
        print("  figures carry a PROVISIONAL suffix. Run tools/regenerate_evaluations.py, then this")
        print("  script again, for the versions that go in the paper.\n")
        print(checks.loc[checks["q_verdict"] == "superseded call",
                         ["representation", "Model", "q_saved",
                          "q_order_statistic", "q_superseded_call"]].to_string(index=False))

    for rep in data:
        print(f"\n  {rep}: {data[rep]['run'].name}")
    print()
    print("written:", figure_3(data, provisional))
    print("written:", figure_4(data, provisional))
    print("written:", figure_coverage_resampling())
    print("written:", figure_coverage_by_class())
    checks.to_csv(DIR_OUT / "figure_verification.csv", index=False)
    print("written:", DIR_OUT / "figure_verification.csv")

    note = ("\n\nPROVISIONAL. This run's conformal thresholds come from NumPy's interpolated "
            "quantile, so the two threshold lines sit where that quantile "
            "put them." if provisional else "")
    caption = (
        "# Figure captions\n\n"
        "## Figure 3\n\n"
        "Figure 3. Predictive entropy against the second-largest mean predicted probability "
        "for the 505 test patients, for each of the five architectures under both feature "
        "representations. The vertical line marks 1 minus the conformal threshold q, since a "
        "prediction set holds two or more classes exactly when the second-largest probability "
        "reaches it, and the horizontal line marks the rate-matched entropy threshold t. The "
        "four quadrants are therefore the four flag groups, and the point counts in each one "
        "are the four cells of Table 3. Axes are shared across panels: the second-largest of "
        "four probabilities cannot exceed 0.5, and the entropy of four classes cannot exceed "
        "ln(4) = 1.386." + note + "\n\n"
        "## Figure 4\n\n"
        "Figure 4. Conformal coverage over resampling for the five models under dataset A and "
        "dataset B. Blue bands span the 2.5 to 97.5 percentile range of coverage over 200 "
        "resamples of the 404 calibration patients with the 505 test patients held fixed, and "
        "orange bands the same range over 500 re-splits of the 909 held-out patients into 404 "
        "calibration and 505 test patients; the line in each band is the mean. The dot is the "
        "coverage of the reported split and the dashed line marks 0.95.\n\n"
        "## Figure 5\n\n"
        "Figure 5. Conformal coverage within each treatment class on the 505 test patients, one "
        "panel per class, with 95 percent Wilson score intervals. Circles are dataset A and "
        "squares dataset B; the dashed line marks 0.95.\n\n"
        "## Alpha sweep figure (number set in the master draft)\n\n"
        "Empirical coverage, mean prediction-set size, test flag rate, agreement and error "
        "recall as alpha runs from 0 to 1, under both feature representations, with the "
        "number of the 505 test patients in each flag's review queue, the number both flags "
        "place in the same group (concordant) and the number flagged by one flag only "
        "(discordant). Each line is "
        "the median across the five architectures and each band is the range between them, "
        "with the per-model values in the saved sweep table. The dashed vertical line marks the "
        "pre-specified alpha of 0.05. An empty prediction set raises the conformal flag, so "
        "past about alpha 0.17, where empty sets first appear, both flag rates rise again "
        "toward 1." + note + "\n")
    (DIR_OUT / "Figure_captions.md").write_text(caption)
    print("written:", DIR_OUT / "Figure_captions.md")


if __name__ == "__main__":
    main()
