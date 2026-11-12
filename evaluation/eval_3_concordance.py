"""
eval_3_concordance.py: agreement between the conformal flag and the entropy flag.

With both flags set at the same calibration flag rate, the module measures how far
they select the same test patients, using:
    joint table       the cross of the two binary flags, four categories
    kappa             agreement beyond chance at these flag rates
    PABAK, ppos       prevalence-robust agreement, since kappa falls when the
                      flagged share is far from one half
    by class          agreement within each treatment class
    continuous scale  Spearman correlation between entropy and the second-largest
                      mean probability, the exact score behind the conformal set,
                      before either is cut into a flag
    flag-rate gap     the test flag rates of the two methods with a paired
                      interval, since the rates are matched on calibration only
"""

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S

GROUP = "eval_3"


def _aligned(res: dict) -> tuple:
    """
    Returns the two flag vectors of one model, aligned on patient position.

    Input: res, the model's uncertainty state with cp_flags and ppu_flags.
    Output: cp, h, true_class and y_pred as integer arrays in patient order.
    """
    cp = res["cp_flags"].sort_values("patient_pos")
    h  = res["ppu_flags"].sort_values("patient_pos")
    return (cp["icp_flag"].values.astype(int), h["h_flag"].values.astype(int),
            cp["true_class"].values.astype(int), cp["y_pred"].values.astype(int))


def run(run_ctx: dict, all_models: dict, uq: dict, data: dict) -> dict:
    """
    Builds and saves the four concordance tables for every model.

    Input: run_ctx, the run context; all_models, model key to results dict, for the
    continuous scores; uq, the per-model uncertainty state; data, the loaded splits.
    Output: dict with summary, by_class, continuous and flag_rate_gap DataFrames.
    """
    R.banner("EVALUATION 3 — CONCORDANCE BETWEEN THE TWO FLAGS")

    sum_rows, cls_rows, cont_rows, gap_rows = [], [], [], []

    for mn, res in uq.items():
        cp, h, true_cls, _ = _aligned(res)
        n   = len(cp)
        idx = S.boot_indices(n)  # one resample matrix, shared by every interval
        parts = S.agreement_parts(cp, h)

        # Every agreement statistic carries a patient-level bootstrap interval drawn from
        # the same resamples.
        k_ci  = S.boot_ci(lambda i: S.kappa(cp[i], h[i]), n, idx=idx)
        oa_ci = S.boot_ci(lambda i: float((cp[i] == h[i]).mean()), n, idx=idx)

        sum_rows.append({
            "Model": C.MODEL_LABELS[mn], "n_test": n,
            "CC_both_certain":   parts["n00"],
            "CU_both_uncertain": parts["n11"],
            "DA_entropy_only":   parts["n01"],
            "DB_conformal_only": parts["n10"],
            "pct_CC": round(parts["n00"] / n * 100, 2),
            "pct_CU": round(parts["n11"] / n * 100, 2),
            "pct_DA": round(parts["n01"] / n * 100, 2),
            "pct_DB": round(parts["n10"] / n * 100, 2),
            "pct_concordant": round(parts["observed_agreement"] * 100, 2),
            "agreement_CI": S.fmt_ci(oa_ci, dp=2, pct=True),
            "kappa": round(parts["kappa"], 4),
            "kappa_CI": S.fmt_ci(k_ci, dp=3),
            "pabak": round(parts["pabak"], 4),
            "ppos": round(parts["ppos"], 4),
            "pneg": round(parts["pneg"], 4),
            "prevalence_index": round(parts["prevalence_index"], 4),
        })

        # --- by treatment class --------------------------------------------------
        for k in range(C.N_CLASSES):
            sel = true_cls == k
            nk  = int(sel.sum())
            if nk == 0:
                continue
            p_k = S.agreement_parts(cp[sel], h[sel])
            cls_rows.append({
                "Model": C.MODEL_LABELS[mn], "class": C.CLASS_NAMES[k], "n": nk,
                "pct_concordant": round(p_k["observed_agreement"] * 100, 2),
                "kappa": round(p_k["kappa"], 4), "pabak": round(p_k["pabak"], 4),
                "flag_rate_conformal": round(float(cp[sel].mean()), 4),
                "flag_rate_entropy":   round(float(h[sel].mean()), 4),
            })

        # --- continuous scale ----------------------------------------------------
        # p2: the second-largest mean probability, the exact score behind the LAC set.
        # rho: their rank agreement before either score is cut into a flag.
        p2      = S.conformal_score_continuous(all_models[mn]["mean_probs_test"])
        entropy = res["ppu_flags"].sort_values("patient_pos")["entropy"].values
        sp      = S.spearman(entropy, p2)
        rho_ci  = S.boot_ci(lambda i: S.spearman(entropy[i], p2[i])["rho"], n, idx=idx)
        cont_rows.append({
            "Model": C.MODEL_LABELS[mn], "n": n,
            "spearman_rho": round(sp["rho"], 4), "p_value": sp["p"],
            "rho_CI": S.fmt_ci(rho_ci, dp=3),
            "interpretation": ("strong monotone association on the continuous scale"
                               if abs(sp["rho"]) >= 0.7 else
                               "moderate association on the continuous scale"
                               if abs(sp["rho"]) >= 0.4 else
                               "weak association on the continuous scale"),
        })

        # --- flag-rate gap on test ----------------------------------------------
        gap = S.paired_boot_diff(lambda i: float(cp[i].mean()),
                                 lambda i: float(h[i].mean()), n, idx=idx)
        gap_rows.append({
            "Model": C.MODEL_LABELS[mn],
            "flag_rate_conformal": round(gap["a"], 4),
            "flag_rate_entropy":   round(gap["b"], 4),
            "difference": round(gap["diff"], 4),
            "difference_CI": S.fmt_ci(gap, dp=4),
            "rates_differ_on_test": gap["excludes_zero"],
        })

    df_sum  = pd.DataFrame(sum_rows)
    df_cls  = pd.DataFrame(cls_rows)
    df_cont = pd.DataFrame(cont_rows)
    df_gap  = pd.DataFrame(gap_rows)

    R.save_table(run_ctx, GROUP, "concordance_summary", df_sum,
                 "The cross of the two binary flags per model: four categories, observed "
                 "agreement, kappa, PABAK, positive and negative agreement, each with a "
                 "patient-level bootstrap interval.", item="23")
    R.save_table(run_ctx, GROUP, "concordance_by_class", df_cls,
                 "Agreement within each treatment class, to show whether the headline "
                 "agreement is carried by one class.", item="30")
    R.save_table(run_ctx, GROUP, "continuous_association", df_cont,
                 "Spearman correlation between predictive entropy and the second-largest "
                 "mean probability, the exact continuous conformal score. Tests agreement "
                 "before either quantity is cut into a flag.", item="24")
    R.save_table(run_ctx, GROUP, "flag_rate_gap_test", df_gap,
                 "Conformal against entropy flag rate on test with a paired interval. The "
                 "rates are matched on calibration by construction, not on test.", item="29")

    R.show(df_sum[["Model", "pct_concordant", "agreement_CI", "kappa", "kappa_CI",
                   "pabak", "pct_DA", "pct_DB"]],
           "Concordance summary, all models", full=True)
    R.show(df_cont, "Entropy against the continuous conformal score", full=True)

    ref = C.MODEL_LABELS[C.REFERENCE_MODEL]
    row = df_sum[df_sum["Model"] == ref]
    if len(row):
        r = row.iloc[0]
        print(f"\n  {ref}: {r['pct_concordant']:.1f}% of patients receive the same flag "
              f"from both methods, kappa {r['kappa']:.3f}. "
              f"{r['pct_DA']:.1f}% are flagged by entropy alone and {r['pct_DB']:.1f}% "
              f"by the conformal set alone.")
    return {"summary": df_sum, "by_class": df_cls,
            "continuous": df_cont, "flag_rate_gap": df_gap}
