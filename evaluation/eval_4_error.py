"""
eval_4_error.py: the share of the model's errors in each review queue.

Error recall rises with the share of patients flagged: a queue holding 70 percent
of patients at random contains about 70 percent of the errors. Every recall value
therefore sits beside the random queue of the same size, and the difference
carries a paired interval.

At alpha 0.05 both flag rates are far above a feasible review volume, so the
comparison is repeated at the fixed review rates in config.REVIEW_RATES, 5 to 40
percent. There both methods flag the same number of patients by construction. A third table
tests union, intersection and two rank-combination rules against the better
single flag at equal review volume.
"""

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S

GROUP = "eval_4"


# =====================================================================
# 1. THE FOUR QUANTITIES, EACH AS A FUNCTION OF A PATIENT INDEX
# =====================================================================

def _recall(flag, err):
    """
    Returns a function of a patient index for error recall: the share of all errors
    among the flagged patients.

    Input: flag, the binary flag array; err, the binary error array.
    Output: a function of an index array, returning NaN when the resample holds no error.
    """
    return lambda i: (float(flag[i][err[i] == 1].mean())
                      if (err[i] == 1).any() else np.nan)


def _precision(flag, err):
    """
    Returns a function of a patient index for precision: the error rate among the
    flagged patients.

    Input: flag; err.
    Output: a function of an index array, returning NaN when nothing is flagged.
    """
    return lambda i: (float(err[i][flag[i] == 1].mean())
                      if (flag[i] == 1).any() else np.nan)


def _retained_accuracy(flag, err):
    """
    Returns a function of a patient index for retained accuracy: the accuracy among
    the patients left unflagged.

    Input: flag; err.
    Output: a function of an index array, returning NaN when every patient is flagged.
    """
    return lambda i: (float(1 - err[i][flag[i] == 0].mean())
                      if (flag[i] == 0).any() else np.nan)


def _enrichment(flag, err):
    """
    Returns a function of a patient index for enrichment: precision divided by the
    base error rate, where 1.0 equals random review.

    Input: flag; err.
    Output: a function of an index array, returning NaN when the base rate is zero or
    nothing is flagged.
    """
    def f(i):
        base = float(err[i].mean())
        if base == 0 or not (flag[i] == 1).any():
            return np.nan
        return float(err[i][flag[i] == 1].mean()) / base
    return f


def _row(label, model_label, flag, err, idx, extra=None) -> dict:
    """
    Builds one table row for a flag against the model's errors, with bootstrap
    intervals.

    Recall always sits beside the random queue of the same size, since a longer queue
    contains more of the misclassified patients.

    Input: label, the flag rule name; model_label; flag, err, binary arrays; idx,
    the shared bootstrap index matrix; extra, optional columns to add.
    Output: dict with the flag count and rate, recall, precision, enrichment and
    retained accuracy with intervals, the random-queue recall, the difference from
    random with its interval, and beats_random.
    """
    n    = len(err)
    rate = float(flag.mean())
    rec  = S.boot_ci(_recall(flag, err), n, idx=idx)
    pre  = S.boot_ci(_precision(flag, err), n, idx=idx)
    ret  = S.boot_ci(_retained_accuracy(flag, err), n, idx=idx)
    enr  = S.boot_ci(_enrichment(flag, err), n, idx=idx)
    # A random queue of the same size contains, in expectation, a share of errors
    # equal to the share of patients it holds.
    vs_random = S.paired_boot_diff(_recall(flag, err),
                                   lambda i: float(flag[i].mean()), n, idx=idx)
    out = {
        "Model": model_label, "rule": label,
        "n_flagged": int(flag.sum()), "flag_rate": round(rate, 4),
        "error_recall": round(rec["point"], 4), "recall_CI": S.fmt_ci(rec),
        "precision": round(pre["point"], 4), "precision_CI": S.fmt_ci(pre),
        "base_error_rate": round(float(err.mean()), 4),
        "enrichment": round(enr["point"], 4), "enrichment_CI": S.fmt_ci(enr),
        "retained_accuracy": round(ret["point"], 4), "retained_CI": S.fmt_ci(ret),
        "random_queue_recall": round(rate, 4),
        "recall_minus_random": round(vs_random["diff"], 4),
        "recall_minus_random_CI": S.fmt_ci(vs_random),
        "beats_random": vs_random["excludes_zero"],
    }
    if extra:
        out.update(extra)
    return out


# =====================================================================
# 2. MAIN
# =====================================================================

def run(run_ctx: dict, all_models: dict, uq: dict, data: dict) -> dict:
    """
    Builds and saves the matched-rate, fixed-budget and combination-rule tables.

    Input: run_ctx, the run context; all_models, model key to results dict, for the
    continuous scores; uq, the per-model uncertainty state; data, the loaded splits.
    Output: dict with matched, budgets and combinations DataFrames.
    """
    R.banner("EVALUATION 4 — ERROR DETECTION AGAINST A RANDOM QUEUE")

    matched_rows, budget_rows, combo_rows = [], [], []

    for mn, res in uq.items():
        label = C.MODEL_LABELS[mn]
        cp_df = res["cp_flags"].sort_values("patient_pos")
        h_df  = res["ppu_flags"].sort_values("patient_pos")

        cp  = cp_df["icp_flag"].values.astype(int)
        h   = h_df["h_flag"].values.astype(int)
        err = S.error_indicator(cp_df["true_class"].values, cp_df["y_pred"].values)
        n   = len(err)
        idx = S.boot_indices(n)

        # The two continuous scores behind the flags, used at the fixed review rates where
        # both methods flag the same number of patients.
        p2      = S.conformal_score_continuous(all_models[mn]["mean_probs_test"])
        entropy = h_df["entropy"].values

        # --- at the matched review volume -----------------------------------------
        matched_rows.append(_row("conformal set", label, cp, err, idx))
        matched_rows.append(_row("entropy threshold", label, h, err, idx))

        # The paired difference goes onto both rows of the pair, so a table filtered to
        # one method still shows the comparison.
        diff = S.paired_boot_diff(_recall(cp, err), _recall(h, err), n, idx=idx)
        for row in (matched_rows[-2], matched_rows[-1]):
            row["conformal_minus_entropy_recall"] = round(diff["diff"], 4)
            row["difference_CI"] = S.fmt_ci(diff)
            row["methods_differ"] = diff["excludes_zero"]

        # --- at fixed review budgets --------------------------------------------
        # Both methods flag exactly n_flag patients, so recall compares directly.
        for rate in C.REVIEW_RATES:
            n_flag = int(round(rate * n))
            f_cp   = S.top_n_flag(p2, n_flag)
            f_h    = S.top_n_flag(entropy, n_flag)
            extra  = {"review_rate": rate, "n_budget": n_flag}
            budget_rows.append(_row("conformal score, top n", label, f_cp, err, idx, extra))
            budget_rows.append(_row("entropy, top n", label, f_h, err, idx, extra))

            d = S.paired_boot_diff(_recall(f_cp, err), _recall(f_h, err), n, idx=idx)
            overlap = round(float(((f_cp == 1) & (f_h == 1)).sum() / max(n_flag, 1)), 4)
            for row in (budget_rows[-2], budget_rows[-1]):
                row["conformal_minus_entropy_recall"] = round(d["diff"], 4)
                row["difference_CI"] = S.fmt_ci(d)
                row["methods_differ"] = d["excludes_zero"]
                # overlap_of_flagged: the share of the budget both methods give to the same
                # patients.
                row["overlap_of_flagged"] = overlap

        # --- combination rules ---------------------------------------------------
        # Union and intersection are also tested at matched volume: patients are ranked
        # on the higher or the lower of their two percentile ranks and cut at the conformal
        # flag count. The comparison with the better single flag is then at equal review
        # volume.
        n_ref  = int(cp.sum())
        r_cp   = pd.Series(p2).rank(pct=True).values
        r_h    = pd.Series(entropy).rank(pct=True).values
        rules  = {
            "union, own volume":        ((cp == 1) | (h == 1)).astype(int),
            "intersection, own volume": ((cp == 1) & (h == 1)).astype(int),
            "max rank, matched volume": S.top_n_flag(np.maximum(r_cp, r_h), n_ref),
            "min rank, matched volume": S.top_n_flag(np.minimum(r_cp, r_h), n_ref),
        }
        best_single = cp if _recall(cp, err)(np.arange(n)) >= _recall(h, err)(np.arange(n)) else h
        best_name   = "conformal set" if best_single is cp else "entropy threshold"

        for rname, f in rules.items():
            d = S.paired_boot_diff(_recall(f, err), _recall(best_single, err), n, idx=idx)
            # A rule beats the single flag only at equal review volume, since the union
            # contains more patients and therefore more errors.
            vol_diff = int(f.sum() - best_single.sum())
            combo_rows.append(_row(rname, label, f, err, idx, {
                "compared_with": best_name,
                "recall_minus_best_single": round(d["diff"], 4),
                "difference_CI": S.fmt_ci(d),
                "matched_volume": vol_diff == 0,
                "beats_best_single_at_matched_volume": bool(d["lo"] > 0 and vol_diff == 0),
                "volume_vs_best_single": vol_diff,
            }))

    df_matched = pd.DataFrame(matched_rows)
    df_budget  = pd.DataFrame(budget_rows)
    df_combo   = pd.DataFrame(combo_rows)

    R.save_table(run_ctx, GROUP, "error_detection_matched_rate", df_matched,
                 "Error recall, precision, enrichment and retained accuracy for each flag at "
                 "the rate-matched operating point, each against the random queue of the same "
                 "size, with patient-level intervals.", item="25, 26")
    R.save_table(run_ctx, GROUP, "error_detection_fixed_review_rates", df_budget,
                 "The same quantities at fixed review budgets of 5, 10, 20 and 30 percent, "
                 "where both methods flag exactly the same number of patients.", item="26")
    R.save_table(run_ctx, GROUP, "combination_rules", df_combo,
                 "Union, intersection and two matched-volume rank rules against the better "
                 "single flag, with a paired interval on the difference in recall.", item="33")

    R.show(df_matched[["Model", "rule", "flag_rate", "error_recall",
                       "random_queue_recall", "recall_minus_random_CI", "beats_random"]],
           "Error detection at the matched rate, against random review", full=True)
    R.show(df_budget[df_budget["Model"] == C.MODEL_LABELS[C.REFERENCE_MODEL]][
               ["rule", "review_rate", "n_budget", "error_recall", "recall_minus_random",
                "enrichment", "overlap_of_flagged"]],
           f"Fixed review budgets — {C.MODEL_LABELS[C.REFERENCE_MODEL]}", full=True)

    wins = df_combo[df_combo["beats_best_single_at_matched_volume"]]
    if len(wins) == 0:
        print("\n  No combination rule beats the better single flag at matched volume in any "
              "model, so the two flags are redundant at equal review volume.")
    else:
        print(f"\n  {len(wins)} combination rule(s) beat the better single flag at equal "
              f"volume with an interval excluding zero: "
              f"{', '.join(wins['Model'] + ' / ' + wins['rule'])}. "
              f"This contradicts the redundancy of the two flags, so check it twice.")
    return {"matched": df_matched, "budgets": df_budget, "combinations": df_combo}
