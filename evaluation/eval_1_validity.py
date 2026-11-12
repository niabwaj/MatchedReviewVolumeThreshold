"""
eval_1_validity.py: held-out validity and efficiency of the conformal prediction sets.

The entropy threshold comes from the conformal flag rate, so the conformal sets
need checking on held-out data first. The module reports, for every model:
    empirical marginal coverage on test against the 1 - alpha target
    coverage within each treatment class, with the class size
    the set-size distribution, the number of empty sets, and the share of
    flagged sets holding three or four of the four classes
"""

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S

GROUP = "eval_1"


def prediction_sets(mean_probs: np.ndarray, q: float) -> np.ndarray:
    """
    Builds the LAC (least ambiguous set-valued classifier) prediction sets as a
    boolean mask.

    A class is in a patient's set when its nonconformity score 1 - p(k) is at or
    below q.

    Input: mean_probs, (n, K) mean predicted probabilities; q, the conformal threshold.
    Output: mask, (n, K), True where class k is in the set.
    """
    return (1.0 - np.asarray(mean_probs)) <= q


def coverage_table(mean_probs: np.ndarray, y_idx: np.ndarray, q: float,
                   alpha: float) -> dict:
    """
    Computes empirical marginal coverage: the share of patients whose true class is in
    their prediction set.

    Coverage is measured on the sets as constructed. An empty set never holds the
    true class, so it counts as not covered.

    Input: mean_probs, (n, K); y_idx, (n,) zero-indexed labels; q, the conformal
    threshold; alpha, the miscoverage level.
    Output: dict with n, n_covered, coverage, the Wilson interval ci_lo and ci_hi,
    target (1 - alpha) and meets_target (True when ci_hi reaches the target).
    """
    mask = prediction_sets(mean_probs, q)
    n    = len(y_idx)
    covered = int(mask[np.arange(n), y_idx].sum())
    lo, hi  = S.wilson_ci(covered, n)
    return {"n": n, "n_covered": covered, "coverage": covered / n,
            "ci_lo": lo, "ci_hi": hi, "target": 1 - alpha,
            "meets_target": bool(hi >= 1 - alpha)}


def run(run_ctx: dict, all_models: dict, uq: dict, data: dict,
        alpha: float = None) -> dict:
    """
    Builds and saves the three validity tables for every model.

    Input: run_ctx, the run context; all_models, model key to results dict; uq,
    the per-model uncertainty state with thr_icp; data, the loaded splits; alpha,
    the miscoverage level (default config.CP_ALPHA).
    Output: dict with coverage, coverage_by_class and set_efficiency DataFrames.
    """
    alpha = C.CP_ALPHA if alpha is None else alpha
    R.banner("EVALUATION 1 — CONFORMAL VALIDITY AND SET EFFICIENCY")

    y_idx = S.to_0idx(data["y"]["test"])
    cov_rows, cls_rows, eff_rows = [], [], []

    for mn, res in uq.items():
        q    = res["thr_icp"]["q_icp"]
        mp   = all_models[mn]["mean_probs_test"]
        mask = prediction_sets(mp, q)
        size = mask.sum(axis=1)

        # --- marginal coverage -------------------------------------------------
        cov = coverage_table(mp, y_idx, q, alpha)
        cov_rows.append({
            "Model": C.MODEL_LABELS[mn], "alpha": alpha,
            "target_coverage": round(cov["target"], 4),
            "q_conformal": round(q, 4), "n_test": cov["n"],
            "n_covered": cov["n_covered"],
            "empirical_coverage": round(cov["coverage"], 4),
            "ci_lo": round(cov["ci_lo"], 4), "ci_hi": round(cov["ci_hi"], 4),
            "meets_target": cov["meets_target"],
            "gap_vs_target": round(cov["coverage"] - cov["target"], 4),
        })

        # The marginal guarantee holds over all patients, so coverage is reported for each
        # class with n shown.
        for k in range(C.N_CLASSES):
            sel = y_idx == k
            nk  = int(sel.sum())
            ck  = int(mask[sel, k].sum()) if nk else 0
            lo, hi = S.wilson_ci(ck, nk) if nk else (np.nan, np.nan)
            cls_rows.append({
                "Model": C.MODEL_LABELS[mn], "class_idx": k,
                "class": C.CLASS_NAMES[k], "n_test": nk, "n_covered": ck,
                "coverage": round(ck / nk, 4) if nk else np.nan,
                "ci_lo": round(lo, 4) if nk else np.nan,
                "ci_hi": round(hi, 4) if nk else np.nan,
                "target": round(1 - alpha, 4),
                "meets_target": bool(hi >= 1 - alpha) if nk else False,
            })

        # --- set efficiency and informativeness --------------------------------
        n        = len(size)
        n_empty  = int((size == 0).sum())
        # An empty set stays empty and is flagged uncertain, as is a set of two or more
        # classes. The *_empty_before_fallback columns count the empty sets.
        eff_size = size
        flagged  = (size == 0) | (size >= 2)
        n_flag   = int(flagged.sum())
        big      = int((eff_size[flagged] >= 3).sum()) if n_flag else 0  # empty sets are excluded
        eff_rows.append({
            "Model": C.MODEL_LABELS[mn],
            "mean_set_size": round(float(eff_size.mean()), 4),
            "median_set_size": float(np.median(eff_size)),
            "pct_size_0": round(float((eff_size == 0).mean()) * 100, 2),
            "pct_size_1": round(float((eff_size == 1).mean()) * 100, 2),
            "pct_size_2": round(float((eff_size == 2).mean()) * 100, 2),
            "pct_size_3": round(float((eff_size == 3).mean()) * 100, 2),
            "pct_size_4": round(float((eff_size == 4).mean()) * 100, 2),
            "n_empty_before_fallback": n_empty,
            "pct_empty_before_fallback": round(n_empty / n * 100, 2),
            "n_flagged": n_flag,
            "pct_flagged": round(n_flag / n * 100, 2),
            # pct_flagged_3_or_4: a flagged set holding three or four of the four classes
            # excludes almost nothing, so this share measures how informative the flagged
            # sets are.
            "pct_flagged_3_or_4": round(big / n_flag * 100, 2) if n_flag else np.nan,
        })

    df_cov = pd.DataFrame(cov_rows)
    df_cls = pd.DataFrame(cls_rows)
    df_eff = pd.DataFrame(eff_rows)

    R.save_table(run_ctx, GROUP, "coverage_marginal_test", df_cov,
                 "Empirical marginal coverage on the held-out test split against the "
                 "1 - alpha target, with a Wilson interval.",
                 item="19")
    R.save_table(run_ctx, GROUP, "coverage_by_class_test", df_cls,
                 "Coverage within each treatment class, with the class size shown. A marginal "
                 "guarantee does not transfer to any single class.", item="20")
    R.save_table(run_ctx, GROUP, "set_efficiency_and_categories", df_eff,
                 "Set-size distribution, the empty-set count, and "
                 "the share of flagged patients whose set holds three or four of the four "
                 "classes.", item="21")

    R.show(df_cov, "Empirical coverage on test, all models", full=True)
    R.show(df_eff, "Set efficiency and informativeness", full=True)
    R.show(df_cls[df_cls["Model"] == C.MODEL_LABELS[C.REFERENCE_MODEL]],
           f"Coverage by class — {C.MODEL_LABELS[C.REFERENCE_MODEL]}", full=True)

    under = df_cov[~df_cov["meets_target"]]
    if len(under):
        print(f"\n  Below target coverage: {', '.join(under['Model'])}. "
              f"The conformal anchor is weaker than the nominal level for these models, "
              f"which qualifies every downstream claim about them.")
    else:
        print(f"\n  All models reach the {1 - alpha:.2f} target within the interval. "
              f"The conformal branch is verified, so the entropy threshold derived from "
              f"it rests on a checked anchor.")

    return {"coverage": df_cov, "coverage_by_class": df_cls, "set_efficiency": df_eff}
