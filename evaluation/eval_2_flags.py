"""
eval_2_flags.py: the conformal and entropy thresholds and the flags they produce.

For every model the module derives the conformal threshold q from calibration and
measures the calibration flag rate r of conformal prediction. It sets the entropy
threshold t at the (1 - r) quantile of calibration entropy and applies both
thresholds to test. It then compares calibration and test flag rates for both
methods as an empirical check of exchangeability. The per-model state it returns
holds the thresholds, the per-patient flags and the concordance tables.
"""

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S
from uq.uncertainty import (run_uncertainty_rate_matched, compute_uncertainty_metrics,
                            evaluate_cp_flags, evaluate_ppu_flags_rate_matched)

GROUP = "eval_2"


def _cal_flags(res: dict, uq_model: dict, y_cal_idx: np.ndarray) -> tuple:
    """
    Rebuilds the conformal and entropy flags on the calibration split.

    The calibration flag rates sit beside the test flag rates in the exchangeability
    table.

    Input: res, the model's results dict; uq_model, its uncertainty state;
    y_cal_idx, zero-indexed calibration labels.
    Output: cp_cal and ppu_cal, the calibration flag tables.
    """
    cp_cal  = evaluate_cp_flags(res["mean_probs_cal"], uq_model["thr_icp"],
                                uq_model["thr_mondrian"], y_cal_idx)
    ppu_cal = evaluate_ppu_flags_rate_matched(uq_model["cal"], uq_model["rate_matched"]["t"])
    return cp_cal, ppu_cal


def _two_sample_diff(a: np.ndarray, b: np.ndarray, seed: int = None) -> dict:
    """
    Bootstraps the difference in flag rate between two independent splits.

    Calibration and test hold different patients, so each split is resampled
    independently.

    Input: a, b, binary flag arrays; seed, the bootstrap seed (default
    config.BOOT_SEED).
    Output: dict with the two rates a and b, diff (a - b), the interval lo and hi at
    config.CI_LEVEL, and excludes_zero.
    """
    rng = np.random.default_rng(C.BOOT_SEED if seed is None else seed)
    a, b = np.asarray(a), np.asarray(b)
    d = np.array([rng.choice(a, len(a), replace=True).mean()
                  - rng.choice(b, len(b), replace=True).mean()
                  for _ in range(C.N_BOOT)])
    lo, hi = np.quantile(d, [(1 - C.CI_LEVEL) / 2, 1 - (1 - C.CI_LEVEL) / 2])
    return {"a": float(a.mean()), "b": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run(run_ctx: dict, all_models: dict, data: dict, config: dict,
        alpha: float = None) -> dict:
    """
    Builds the uncertainty state for every model and saves the threshold tables.

    Input: run_ctx, the run context; all_models, model key to results dict; data,
    the loaded splits; config, the shared config dict from config.base_config;
    alpha, the miscoverage level (default config.CP_ALPHA).
    Output: uq, the per-model state {train, cal, test, thr_icp, thr_mondrian,
    rate_matched, cp_flags, ppu_flags, concordance}. The saved outputs are
    thresholds_and_flag_rates, exchangeability_cal_vs_test, uncertainty_distribution
    and the uq_state pickle.
    """
    alpha = C.CP_ALPHA if alpha is None else alpha
    R.banner(f"EVALUATION 2 — THRESHOLDS AND FLAGS (alpha = {alpha})")

    uq = run_uncertainty_rate_matched(
        all_models=all_models, y_train=data["y"]["train"], y_cal=data["y"]["cal"],
        y_test=data["y"]["test"], split_indices=data["split_indices"],
        config=config, alpha=alpha)

    y_cal_idx = S.to_0idx(data["y"]["cal"])

    thr_rows, exch_rows, dist_rows = [], [], []
    for mn, res in uq.items():
        cp_cal, ppu_cal = _cal_flags(all_models[mn], res, y_cal_idx)
        cp_test, ppu_test = res["cp_flags"], res["ppu_flags"]

        thr_rows.append({
            "Model":            C.MODEL_LABELS[mn],
            "alpha":            alpha,
            "q_level":          res["thr_icp"]["q_level"],
            "q_conformal":      round(res["thr_icp"]["q_icp"], 4),
            "n_cal":            res["thr_icp"]["n_cal"],
            "r_cal_conformal":  round(float(cp_cal["icp_flag"].mean()), 4),
            "t_entropy":        round(res["rate_matched"]["t"], 4),
            "entropy_quantile": res["rate_matched"]["q_level"],
            "cal_flag_rate_conformal": round(float(cp_cal["icp_flag"].mean()), 4),
            "cal_flag_rate_entropy":   round(float(ppu_cal["h_flag"].mean()), 4),
            "test_flag_rate_conformal": round(float(cp_test["icp_flag"].mean()), 4),
            "test_flag_rate_entropy":   round(float(ppu_test["h_flag"].mean()), 4),
        })

        # The shift in flag rate from calibration to test is an empirical check of
        # exchangeability. A large shift would weaken the coverage guarantee.
        d_cp = _two_sample_diff(cp_cal["icp_flag"].values, cp_test["icp_flag"].values)
        d_h  = _two_sample_diff(ppu_cal["h_flag"].values,  ppu_test["h_flag"].values)
        exch_rows.append({
            "Model": C.MODEL_LABELS[mn],
            "conformal_cal": round(d_cp["a"], 4), "conformal_test": round(d_cp["b"], 4),
            "conformal_diff_CI": S.fmt_ci(d_cp, dp=3),
            "conformal_shift_detected": d_cp["excludes_zero"],
            "entropy_cal": round(d_h["a"], 4), "entropy_test": round(d_h["b"], 4),
            "entropy_diff_CI": S.fmt_ci(d_h, dp=3),
            "entropy_shift_detected": d_h["excludes_zero"],
        })

        for split in ["train", "cal", "test"]:
            df = res[split]
            dist_rows.append({
                "Model": C.MODEL_LABELS[mn], "split": split, "n": len(df),
                "H_mean": round(float(df["entropy"].mean()), 4),
                "H_sd":   round(float(df["entropy"].std()), 4),
                "H_median": round(float(df["entropy"].median()), 4),
                "MI_mean": round(float(df["mutual_information"].mean()), 4),
                "AU_mean": round(float(df["aleatoric_uncertainty"].mean()), 4),
            })

    df_thr  = pd.DataFrame(thr_rows)
    df_exch = pd.DataFrame(exch_rows)
    df_dist = pd.DataFrame(dist_rows)

    R.save_table(run_ctx, GROUP, "thresholds_and_flag_rates", df_thr,
                 "Conformal threshold q, the calibration multi-label rate r that sets the "
                 "review budget, the entropy threshold t derived at the (1-r) quantile, and "
                 "the flag rate each produces on calibration and on test.", item="22")
    R.save_table(run_ctx, GROUP, "exchangeability_cal_vs_test", df_exch,
                 "Flag rate on calibration against flag rate on test for both methods, with "
                 "a bootstrap interval on the difference. Evidence for the exchangeability "
                 "assumption behind the conformal guarantee.", item="15")
    R.save_table(run_ctx, GROUP, "uncertainty_distribution", df_dist,
                 "Entropy, mutual information and aleatoric uncertainty by split and model. "
                 "Context for the threshold table; no claim rests on MI or AU.", item="22")
    R.save_object(run_ctx, GROUP, "uq_state", uq,
                  "Full per-model uncertainty state: thresholds, per-patient flags and the "
                  "concordance tables.", item="22")

    R.show(df_thr, "Thresholds and flag rates, all models", full=True)
    R.show_reference(df_exch, "Calibration against test flag rate")

    ref = C.REFERENCE_MODEL
    if ref in uq:
        rm = uq[ref]["rate_matched"]
        print(f"\n  {C.MODEL_LABELS[ref]} check: r = {rm['r']:.4f} of calibration patients "
              f"carry a multi-label set, so t is the {rm['q_level']:.4f} quantile of "
              f"calibration entropy, t = {rm['t']:.4f}.")
        print(f"  Test flag rates: conformal {uq[ref]['cp_flags']['icp_flag'].mean():.4f}, "
              f"entropy {uq[ref]['ppu_flags']['h_flag'].mean():.4f}. "
              f"These are matched by construction on calibration, not on test.")
    return uq
