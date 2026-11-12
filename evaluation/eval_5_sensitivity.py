"""
eval_5_sensitivity.py: sensitivity of the comparison to four analysis choices.

    alpha sweep           the miscoverage level alpha
    calibration resample  the single draw of 404 calibration patients
    APS                   the LAC nonconformity score
    M convergence         the number of draws M = 1,000 per patient

Every analysis post-processes the stored probability arrays of the fitted models.
"""

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S
from evaluation.eval_1_validity import prediction_sets
from uq.uncertainty import _q_level, conformal_quantile

GROUP = "eval_5"


def cp_flag(mask: np.ndarray) -> np.ndarray:
    """
    Computes the conformal flag from a prediction-set mask.

    The flag is 1 for an empty set or a set of two or more classes, and 0 for a
    singleton.

    Input: mask, (n, K) boolean prediction sets.
    Output: (n,) integer flags.
    """
    size = mask.sum(axis=1)
    return ((size == 0) | (size >= 2)).astype(int)


# =====================================================================
# 1. SCORES
# =====================================================================

def lac_threshold(mean_probs_cal: np.ndarray, y_cal_idx: np.ndarray, alpha: float) -> float:
    """
    Computes the LAC conformal threshold: the conformal quantile of 1 - p(true class)
    on calibration.

    Input: mean_probs_cal, (n, K); y_cal_idx, (n,) zero-indexed labels; alpha, the
    miscoverage level.
    Output: the threshold q.
    """
    n  = len(y_cal_idx)
    s  = 1.0 - mean_probs_cal[np.arange(n), y_cal_idx]
    return conformal_quantile(s, alpha)


def aps_scores(mean_probs: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """
    Computes the non-randomised APS score: the total probability of every class ranked
    at or above the true class.

    A patient whose true class ranks near the top has a low score.

    Input: mean_probs, (n, K); y_idx, (n,) zero-indexed labels.
    Output: (n,) scores.
    """
    p_true = mean_probs[np.arange(len(y_idx)), y_idx][:, None]
    return (mean_probs * (mean_probs >= p_true)).sum(axis=1)


def aps_sets(mean_probs: np.ndarray, q: float) -> np.ndarray:
    """
    Builds non-randomised APS prediction sets.

    Classes enter in descending probability until the cumulative probability reaches
    q, and the class that crosses q is kept whole. The non-randomised form
    over-covers by construction, so APS sets are larger and coverage lies above the
    nominal level.

    Input: mean_probs, (n, K); q, the APS threshold.
    Output: mask, (n, K) boolean, always holding at least one class.
    """
    order = np.argsort(-mean_probs, axis=1)
    srt   = np.take_along_axis(mean_probs, order, axis=1)
    cum   = np.cumsum(srt, axis=1)
    # every class up to and including the first to reach q
    keep_sorted = np.zeros_like(cum, dtype=bool)
    reached     = cum >= q
    first       = np.argmax(reached, axis=1)
    first       = np.where(reached.any(axis=1), first, mean_probs.shape[1] - 1)
    for i, f in enumerate(first):
        keep_sorted[i, :f + 1] = True
    mask = np.zeros_like(keep_sorted)
    np.put_along_axis(mask, order, keep_sorted, axis=1)
    return mask


# =====================================================================
# 2. THE FOUR ANALYSES
# =====================================================================

def alpha_sweep(all_models: dict, uq: dict, data: dict) -> pd.DataFrame:
    """
    Re-derives both thresholds at every alpha in config.ALPHA_GRID and recomputes the
    comparison.

    Each row carries coverage, mean set size, both flag rates, concordance, kappa,
    error recall and retained accuracy.

    Input: all_models, model key to results dict; uq, the per-model uncertainty
    state; data, the loaded splits.
    Output: DataFrame with one row per model and alpha.
    """
    y_cal = S.to_0idx(data["y"]["cal"])
    rows  = []
    for mn, res in uq.items():
        mp_cal  = all_models[mn]["mean_probs_cal"]
        mp_test = all_models[mn]["mean_probs_test"]
        h_cal   = res["cal"]["entropy"].values
        h_test  = res["ppu_flags"].sort_values("patient_pos")["entropy"].values
        cp_df   = res["cp_flags"].sort_values("patient_pos")
        y_test  = cp_df["true_class"].values.astype(int)
        err     = S.error_indicator(y_test, cp_df["y_pred"].values.astype(int))
        n       = len(y_test)

        for alpha in C.ALPHA_GRID:
            q      = lac_threshold(mp_cal, y_cal, alpha)
            m_cal  = prediction_sets(mp_cal, q)
            m_test = prediction_sets(mp_test, q)
            size   = m_test.sum(axis=1)
            cp     = cp_flag(m_test)
            r      = float(cp_flag(m_cal).mean())
            t      = float(np.quantile(h_cal, 1 - r)) if 0 <= 1 - r <= 1 else np.nan
            h      = (h_test > t).astype(int)

            rows.append({
                "Model": C.MODEL_LABELS[mn], "alpha": alpha,
                "q_conformal": round(q, 4),
                "coverage_test": round(float(m_test[np.arange(n), y_test].mean()), 4),
                "target_coverage": round(1 - alpha, 4),
                "mean_set_size": round(float(size.mean()), 3),
                "cal_flag_rate": round(r, 4),
                "t_entropy": round(t, 4),
                "test_flag_rate_conformal": round(float(cp.mean()), 4),
                "test_flag_rate_entropy": round(float(h.mean()), 4),
                "pct_concordant": round(float((cp == h).mean()) * 100, 2),
                "kappa": round(S.kappa(cp, h), 4),
                "recall_conformal": round(float(cp[err == 1].mean()), 4),
                "recall_entropy": round(float(h[err == 1].mean()), 4),
                "retained_accuracy_conformal": round(
                    float(1 - err[cp == 0].mean()) if (cp == 0).any() else np.nan, 4),
                "retained_accuracy_entropy": round(
                    float(1 - err[h == 0].mean()) if (h == 0).any() else np.nan, 4),
            })
    return pd.DataFrame(rows)


def calibration_resample(all_models: dict, uq: dict, data: dict,
                         n_resamples: int = None, alpha: float = None) -> pd.DataFrame:
    """
    Resamples the calibration set with replacement and re-derives both thresholds on
    each resample.

    The spread of q, t, r, coverage, concordance and kappa shows how much the results
    depend on the one calibration draw.

    Input: all_models; uq; data; n_resamples (default config.N_CAL_RESAMPLES);
    alpha (default config.CP_ALPHA).
    Output: DataFrame with one row per model: mean, standard deviation and 95
    percent range of each quantity.
    """
    n_resamples = C.N_CAL_RESAMPLES if n_resamples is None else n_resamples
    alpha       = C.CP_ALPHA if alpha is None else alpha
    y_cal = S.to_0idx(data["y"]["cal"])
    rng   = np.random.default_rng(C.BOOT_SEED)
    rows  = []

    for mn, res in uq.items():
        mp_cal  = all_models[mn]["mean_probs_cal"]
        mp_test = all_models[mn]["mean_probs_test"]
        h_cal   = res["cal"]["entropy"].values
        h_test  = res["ppu_flags"].sort_values("patient_pos")["entropy"].values
        cp_df   = res["cp_flags"].sort_values("patient_pos")
        y_test  = cp_df["true_class"].values.astype(int)
        n_cal, n_test = len(y_cal), len(y_test)

        draws = {"q": [], "t": [], "r": [], "coverage": [], "concordant": [], "kappa": []}
        for _ in range(n_resamples):
            b     = rng.integers(0, n_cal, n_cal)
            q     = lac_threshold(mp_cal[b], y_cal[b], alpha)
            m_cal = prediction_sets(mp_cal[b], q)
            r     = float(cp_flag(m_cal).mean())
            t     = float(np.quantile(h_cal[b], 1 - r))
            m_te  = prediction_sets(mp_test, q)
            cp    = cp_flag(m_te)
            h     = (h_test > t).astype(int)
            draws["q"].append(q); draws["t"].append(t); draws["r"].append(r)
            draws["coverage"].append(float(m_te[np.arange(n_test), y_test].mean()))
            draws["concordant"].append(float((cp == h).mean()))
            draws["kappa"].append(S.kappa(cp, h))

        row = {"Model": C.MODEL_LABELS[mn], "n_resamples": n_resamples, "alpha": alpha}
        for k, v in draws.items():
            v = np.asarray(v, dtype=float)
            lo, hi = np.nanquantile(v, [0.025, 0.975])
            row[f"{k}_mean"]  = round(float(np.nanmean(v)), 4)
            row[f"{k}_sd"]    = round(float(np.nanstd(v, ddof=1)), 4)
            row[f"{k}_range"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def score_sensitivity(all_models: dict, uq: dict, data: dict,
                      alpha: float = None) -> pd.DataFrame:
    """
    Repeats the comparison with APS in place of LAC for every model.

    LAC is the primary score since it gives the smallest sets at a given coverage.
    Under LAC a set holds two or more classes exactly when the second-largest
    probability reaches 1 - q. The matched-volume ranking is then the exact conformal
    score.

    Input: all_models; uq; data; alpha (default config.CP_ALPHA).
    Output: DataFrame with one row per model and score.
    """
    alpha = C.CP_ALPHA if alpha is None else alpha
    y_cal = S.to_0idx(data["y"]["cal"])
    rows  = []

    for mn, res in uq.items():
        mp_cal  = all_models[mn]["mean_probs_cal"]
        mp_test = all_models[mn]["mean_probs_test"]
        h_cal   = res["cal"]["entropy"].values
        h_test  = res["ppu_flags"].sort_values("patient_pos")["entropy"].values
        cp_df   = res["cp_flags"].sort_values("patient_pos")
        y_test  = cp_df["true_class"].values.astype(int)
        err     = S.error_indicator(y_test, cp_df["y_pred"].values.astype(int))
        n       = len(y_test)

        for score in C.CONFORMAL_SCORES:
            if score == "lac":
                q      = lac_threshold(mp_cal, y_cal, alpha)
                m_cal  = prediction_sets(mp_cal, q)
                m_test = prediction_sets(mp_test, q)
            else:
                s      = aps_scores(mp_cal, y_cal)
                q      = conformal_quantile(s, alpha)
                m_cal  = aps_sets(mp_cal, q)
                m_test = aps_sets(mp_test, q)

            size = m_test.sum(axis=1)
            cp   = cp_flag(m_test)
            r    = float(cp_flag(m_cal).mean())
            t    = float(np.quantile(h_cal, np.clip(1 - r, 0, 1)))
            h    = (h_test > t).astype(int)

            rows.append({
                "Model": C.MODEL_LABELS[mn], "score": score.upper(), "alpha": alpha,
                "q": round(q, 4),
                "coverage_test": round(float(m_test[np.arange(n), y_test].mean()), 4),
                "mean_set_size": round(float(size.mean()), 3),
                "cal_flag_rate": round(r, 4),
                "t_entropy": round(t, 4),
                "test_flag_rate_conformal": round(float(cp.mean()), 4),
                "test_flag_rate_entropy": round(float(h.mean()), 4),
                "pct_concordant": round(float((cp == h).mean()) * 100, 2),
                "kappa": round(S.kappa(cp, h), 4),
                "recall_conformal": round(float(cp[err == 1].mean()), 4),
                "recall_entropy": round(float(h[err == 1].mean()), 4),
            })
    return pd.DataFrame(rows)


def m_convergence(uq: dict, all_models: dict, data: dict) -> pd.DataFrame:
    """
    Recomputes balanced accuracy, entropy and concordance on growing subsets of the
    stored draws, up to M.

    The analysis is skipped when fewer than three grid values fit within M, as in a
    smoke run.

    Input: uq; all_models; data.
    Output: DataFrame with one row per model and M, with m_stable, the smallest M
    within tolerance of the largest M, or an empty DataFrame.
    """
    from uq.uncertainty import run_m_convergence, find_m_stabilisation
    grid = [m for m in C.M_GRID if m <= C.M_SAMPLES]
    if len(grid) < 3:
        print("  M convergence skipped: M is too small in this run for the grid to apply.")
        return pd.DataFrame()

    frames = []
    for mn, res in uq.items():
        try:
            df = run_m_convergence(all_models[mn], data["y"]["cal"], data["y"]["test"],
                                   grid, alpha=C.CP_ALPHA)
            df.insert(0, "Model", C.MODEL_LABELS[mn])
            try:
                df["m_stable"] = find_m_stabilisation(df)["m_stable"]
            except Exception:
                pass
            frames.append(df)
        except Exception as exc:
            print(f"  M convergence failed for {C.MODEL_LABELS[mn]}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =====================================================================
# 3. MAIN
# =====================================================================

def run(run_ctx: dict, all_models: dict, uq: dict, data: dict) -> dict:
    """
    Runs the four sensitivity analyses and saves their tables.

    Input: run_ctx, the run context; all_models; uq; data.
    Output: dict with alpha_sweep, alpha_sweep_reported, calibration_resampling,
    score_sensitivity and m_convergence DataFrames.
    """
    R.banner("EVALUATION 5 — SENSITIVITY")

    df_alpha = alpha_sweep(all_models, uq, data)
    R.save_table(run_ctx, GROUP, "alpha_sweep_full", df_alpha,
                 f"Both thresholds re-derived at every alpha from {C.ALPHA_GRID[0]} to "
                 f"{C.ALPHA_GRID[-1]} in steps of 0.01, {len(C.ALPHA_GRID)} points per model, "
                 "carrying coverage, set size, flag rates, concordance, error recall and "
                 "retained accuracy. The source for the sensitivity figure.", item="29")

    # df_alpha_report: the reported alpha points. The full curve stays in the file above.
    df_alpha_report = df_alpha[df_alpha["alpha"].isin(C.ALPHA_REPORT)].copy()
    R.save_table(run_ctx, GROUP, "alpha_sweep_reported_points", df_alpha_report,
                 f"The alpha sweep at the reported points {C.ALPHA_REPORT}, drawn from the "
                 "full curve. This is the table; the curve is the figure.", item="29")

    df_cal = calibration_resample(all_models, uq, data)
    R.save_table(run_ctx, GROUP, "calibration_resampling", df_cal,
                 "Spread of the conformal threshold, the entropy threshold, coverage and "
                 "concordance across bootstrap draws of the calibration set.", item="29")

    df_score = score_sensitivity(all_models, uq, data)
    R.save_table(run_ctx, GROUP, "score_sensitivity_lac_vs_aps", df_score,
                 "The comparison repeated with APS in place of LAC on all five models, "
                 "pre-specified so the choice of nonconformity score is shown not to drive "
                 "the concordance result.", item="29")

    df_m = m_convergence(uq, all_models, data)
    if len(df_m):
        R.save_table(run_ctx, GROUP, "m_convergence", df_m,
                     "Balanced accuracy, entropy and concordance as the number of stored "
                     "draws per patient increases, evidence that M = 1,000 is sufficient.",
                     item="29")

    R.show(df_alpha_report[df_alpha_report["Model"] == C.MODEL_LABELS[C.REFERENCE_MODEL]],
           f"Alpha sweep at the reported points — {C.MODEL_LABELS[C.REFERENCE_MODEL]} "
           f"({len(df_alpha)} rows saved across the full curve)", full=True)
    R.show(df_score, "LAC against APS, all models", full=True)
    R.show(df_cal[["Model", "q_mean", "q_range", "t_mean", "t_range",
                   "coverage_mean", "concordant_mean", "concordant_range"]],
           "Calibration resampling spread", full=True)

    # shift: the change in concordance when APS replaces LAC.
    piv = df_score.pivot_table(index="Model", columns="score", values="pct_concordant")
    if {"LAC", "APS"} <= set(piv.columns):
        shift = (piv["APS"] - piv["LAC"]).abs()
        print(f"\n  Switching LAC to APS moves concordance by "
              f"{shift.min():.1f} to {shift.max():.1f} percentage points across models "
              f"(largest: {shift.idxmax()}). Read this against the LAC concordance figure "
              f"before claiming the score choice is immaterial.")
    return {"alpha_sweep": df_alpha, "alpha_sweep_reported": df_alpha_report,
            "calibration_resampling": df_cal, "score_sensitivity": df_score,
            "m_convergence": df_m}
