"""
tables.py: Tables 1 to 5 of the paper, assembled from a finished run of each
representation.

Every value is copied from a saved evaluation table, apart from four derived
columns. Each derived column is arithmetic on printed columns, is named in its
caption, and is checked against a second route to the same quantity before it is
written.

    Table 1   classification performance and probability quality
    Table 2   panel A, thresholds and flag rates; panel B, conformal coverage and
              set efficiency
    Table 3   the four flag groups, agreement, kappa and rank correlation
    Table 4   agreement within each treatment class
    Table 5   error detection at the matched flag rate

Each table is written as CSV per representation, and all of them together as one
markdown file. Run folders whose conformal threshold came from NumPy's
interpolated quantile are recognised, and their outputs carry a PROVISIONAL suffix.

Run from the repository root:  python -m tables_figures.tables
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as C
from tables_figures.figures import TOL, conformal_quantile, newest_run, superseded_quantile

DIR_CODE = Path(__file__).resolve().parent.parent
DIR_OUT = DIR_CODE / "results" / "paper"

# PERF_LABEL: the model labels in the performance table, mapped to the labels in
# config.MODEL_LABELS used by every other table.
PERF_LABEL = {
    "Bayesian PyMC": "Bayesian Logistic Regression",
    "Bootstrap LR": "Bootstrap Logistic Regression",
    "MC Dropout": "MC Dropout Neural Network",
    "SVM (RBF)": "SVM Ensemble",
    "XGBoost": "XGBoost Ensemble",
}
# CLASS_RENAME: maps the alternative label "Social Anxiety" onto "Social Phobia",
# so every table uses one class name.
CLASS_RENAME = {"Social Anxiety": "Social Phobia"}

CI = re.compile(r"^\s*(-?[\d.]+)\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]\s*$")


def _cell(v) -> str:
    """
    Formats one table cell as text, trimming trailing zeros from a float.

    Input: v, the cell value.
    Output: the text, empty for NaN.
    """
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def to_pipe(df: pd.DataFrame) -> str:
    """
    Formats a DataFrame as a markdown pipe table with the standard library alone.

    Input: df, the table.
    Output: the pipe table as a string.
    """
    cols = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in r] for r in df.itertuples(index=False, name=None)]
    width = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(width[i]) for i, c in enumerate(cols)) + " |"
    rule = "|" + "|".join("-" * (width[i] + 2) for i in range(len(cols))) + "|"
    body = ["| " + " | ".join(r[i].ljust(width[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join([head, rule] + body)


def parse_ci(s: str) -> tuple:
    """
    Parses a 'point [lo, hi]' interval string.

    Input: s, the string.
    Output: (point, lo, hi) as floats. Raises ValueError for any other form.
    """
    m = CI.match(str(s))
    if not m:
        raise ValueError(f"Interval string not in the expected form: {s!r}")
    return tuple(float(g) for g in m.groups())


def flip_ci(s: str) -> str:
    """
    Converts a proportion and its interval to one minus the proportion, with the
    bounds swapped.

    Input: s, a 'point [lo, hi]' string.
    Output: the converted string, to three decimals.
    """
    p, lo, hi = parse_ci(s)
    return f"{1 - p:.3f} [{1 - hi:.3f}, {1 - lo:.3f}]"


def read(run: Path, group: str, name: str, tag: str) -> pd.DataFrame:
    """
    Reads one saved evaluation table and harmonises the class label.

    Input: run, the run folder; group, an evaluation group key; name, the table
    name; tag, the representation tag.
    Output: the table as a DataFrame.
    """
    folder = f"{group}_{tag}_{C.EVAL_GROUPS[group]}"
    df = pd.read_csv(run / folder / f"{name}_{tag}.csv")
    if "class" in df.columns:
        df["class"] = df["class"].replace(CLASS_RENAME)
    return df


def order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the rows into the model order of config.MODEL_KEYS, then by class and rule
    where present.

    Input: df, a table with a Model column.
    Output: the sorted copy.
    """
    labels = [C.MODEL_LABELS[k] for k in C.MODEL_KEYS]
    df = df.copy()
    df["Model"] = pd.Categorical(df["Model"], categories=labels, ordered=True)
    return df.sort_values([c for c in ["Model", "class", "rule"] if c in df.columns]).reset_index(drop=True)


def staleness(run: Path, representation: str, tag: str) -> bool:
    """
    Reports whether the run's conformal thresholds came from NumPy's interpolated
    quantile.

    The test is the same as check 1 of figures.py, so tables and figures built from
    one run folder carry the same verdict.

    Input: run, the run folder; representation; tag.
    Output: True when any model's saved threshold matches the interpolated quantile.
    """
    rep = C.REPRESENTATIONS[representation]
    y_cal = pd.read_csv(DIR_CODE / f"data/cal/y_{tag}_calibration.csv").iloc[:, -1].values.astype(int) - 1
    thr = read(run, "eval_2", "thresholds_and_flag_rates", tag)
    import pickle
    verdicts = []
    for key in C.MODEL_KEYS:
        with open(run / f"models/results_{key}_{rep['version']}.pkl", "rb") as fh:
            mp_cal = np.asarray(pickle.load(fh)["mean_probs_cal"])
        s = 1.0 - mp_cal[np.arange(len(y_cal)), y_cal]
        q_saved = float(thr.loc[thr["Model"] == C.MODEL_LABELS[key], "q_conformal"].iloc[0])
        q_rule = round(conformal_quantile(s, C.CP_ALPHA), 4)
        q_old = round(superseded_quantile(s, C.CP_ALPHA), 4)
        if abs(q_rule - q_old) > TOL:
            verdicts.append("superseded" if abs(q_saved - q_old) <= TOL else "rule")
    return "superseded" in verdicts


def build(representation: str) -> dict:
    """
    Assembles the five tables for one representation and runs the derived-column checks.

    Chance agreement must reproduce the saved kappa, and the counts of correct
    predictions among flagged and unflagged patients must add up to the total.

    Input: representation, a key of config.REPRESENTATIONS.
    Output: dict with run, stale and the tables Table1, Table2A, Table2B, Table3,
    Table4 and Table5.
    """
    tag = C.REPRESENTATIONS[representation]["tag"]
    run = newest_run(representation)

    perf = read(run, "eval_0", "performance_test", tag)
    unknown = set(perf["Model"]) - set(PERF_LABEL)
    if unknown:
        raise KeyError(f"Unmapped model label in the performance table: {unknown}")
    perf["Model"] = perf["Model"].map(PERF_LABEL)

    thr = read(run, "eval_2", "thresholds_and_flag_rates", tag)
    gap = read(run, "eval_3", "flag_rate_gap_test", tag)
    cov = read(run, "eval_1", "coverage_marginal_test", tag)
    eff = read(run, "eval_1", "set_efficiency_and_categories", tag)
    con = read(run, "eval_3", "concordance_summary", tag)
    rho = read(run, "eval_3", "continuous_association", tag)
    cls = read(run, "eval_3", "concordance_by_class", tag)
    err = read(run, "eval_4", "error_detection_matched_rate", tag)

    t1 = order(perf[["Model", "Bal. Acc.", "Accuracy", "AUC-ROC (macro)", "F1 (macro)",
                     "T1 Recall", "T2 Recall", "T3 Recall", "T4 Recall",
                     "ECE", "Brier", "NLL"]])

    t2a = order(thr[["Model", "q_conformal", "r_cal_conformal", "t_entropy",
                     "cal_flag_rate_conformal", "cal_flag_rate_entropy",
                     "test_flag_rate_conformal", "test_flag_rate_entropy"]]
                .merge(gap[["Model", "difference", "difference_CI"]], on="Model")
                .rename(columns={"difference": "flag_rate_gap",
                                 "difference_CI": "flag_rate_gap_CI"}))

    t2b = order(cov[["Model", "empirical_coverage", "gap_vs_target"]]
                .merge(eff[["Model", "mean_set_size", "pct_size_1", "pct_size_2",
                            "pct_size_3", "pct_size_4", "pct_flagged_3_or_4",
                            "n_empty_before_fallback"]], on="Model"))

    n = int(con["n_test"].iloc[0])
    t3 = con[["Model", "n_test", "CC_both_certain", "CU_both_uncertain",
              "DA_entropy_only", "DB_conformal_only", "pct_concordant",
              "agreement_CI", "kappa", "kappa_CI"]].copy()
    p_cp = (t3["CU_both_uncertain"] + t3["DB_conformal_only"]) / n
    p_h = (t3["CU_both_uncertain"] + t3["DA_entropy_only"]) / n
    t3["chance_agreement"] = (p_cp * p_h + (1 - p_cp) * (1 - p_h)).round(4)
    # check: kappa rebuilt from observed and chance agreement against the saved value
    po = t3["pct_concordant"] / 100.0
    rebuilt = ((po - t3["chance_agreement"]) / (1 - t3["chance_agreement"])).round(4)
    bad = (rebuilt - t3["kappa"]).abs() > 1e-3
    if bad.any():
        raise ValueError(f"Chance agreement does not reproduce kappa for "
                         f"{list(t3.loc[bad, 'Model'])} in {representation}.")
    t3 = order(t3.merge(rho[["Model", "spearman_rho", "rho_CI"]], on="Model"))
    t3 = t3[["Model", "n_test", "CC_both_certain", "CU_both_uncertain",
             "DA_entropy_only", "DB_conformal_only", "pct_concordant", "agreement_CI",
             "chance_agreement", "kappa", "kappa_CI", "spearman_rho", "rho_CI"]]

    t4 = order(cls[["Model", "class", "n", "pct_concordant", "kappa",
                    "flag_rate_conformal", "flag_rate_entropy"]])

    t5 = err[["Model", "rule", "n_flagged", "flag_rate", "precision", "precision_CI",
              "retained_accuracy", "retained_CI", "error_recall", "recall_CI",
              "random_queue_recall", "recall_minus_random",
              "recall_minus_random_CI", "base_error_rate"]].copy()
    t5["acc_flagged"] = (1 - t5["precision"]).round(4)
    t5["acc_flagged_CI"] = t5["precision_CI"].apply(flip_ci)
    flagged_correct = t5["n_flagged"] * (1 - t5["precision"])
    unflagged_correct = (n - t5["n_flagged"]) * t5["retained_accuracy"]
    total_correct = n * (1 - t5["base_error_rate"])
    off = (flagged_correct + unflagged_correct - total_correct).abs()
    if (off > 1.0).any():
        raise ValueError(f"Correct-prediction counts do not reconcile in {representation}: "
                         f"largest gap {off.max():.2f} patients.")
    t5["pct_correct_flagged"] = (100 * flagged_correct / total_correct).round(2)
    t5 = order(t5)[["Model", "rule", "n_flagged", "flag_rate",
                    "acc_flagged", "acc_flagged_CI",
                    "retained_accuracy", "retained_CI",
                    "error_recall", "recall_CI", "random_queue_recall",
                    "recall_minus_random", "recall_minus_random_CI",
                    "pct_correct_flagged"]]

    return {"run": run, "stale": staleness(run, representation, tag),
            "Table1": t1, "Table2A": t2a, "Table2B": t2b,
            "Table3": t3, "Table4": t4, "Table5": t5}


CAPTIONS = {
    "Table1": ("Table 1. Classification performance and probability quality of the five "
               "architectures on the 505 test patients. Balanced accuracy is the mean of the "
               "four per-class recalls. T1 to T4 are depression, panic disorder, social phobia "
               "and specific phobia. ECE is the expected calibration error over ten equal-width "
               "bins of the top-class probability, the Brier score is normalized over four "
               "classes, and NLL is the mean negative log likelihood of the recorded class. "
               "Source: eval_0."),
    "Table2A": ("Table 2, panel A. The two thresholds and the flag rate each produces. q is the "
                "385th smallest of the 404 calibration nonconformity scores, r is the share of "
                "calibration patients whose conformal set holds two or more classes, and t is the "
                "(1 - r) quantile of calibration entropy. The gap is the conformal test flag rate "
                "minus the entropy test flag rate. Source: eval_2 and eval_3."),
    "Table2B": ("Table 2, panel B. Conformal coverage and set efficiency on the 505 test patients. "
                "Coverage is measured on the set the procedure returned. An empty set, or a set "
                "of two or more classes, raises the conformal flag. The size columns sum to 100, "
                "and the one-class column is 100 minus the conformal test flag rate in panel A. "
                "Source: eval_1."),
    "Table3": ("Table 3. Agreement between the conformal flag and the entropy flag on the 505 test "
               "patients. CC both certain, CU both uncertain, DA entropy flag only, DB conformal "
               "flag only. Observed agreement is (CC + CU) / 505. Chance agreement is computed "
               "from the two test flag rates the cells imply and reproduces the saved kappa through "
               "kappa = (observed - chance) / (1 - chance). Rho is the Spearman rank correlation "
               "between entropy and the second-largest mean predicted probability over all 505 "
               "patients. Intervals are 95 percent from 2,000 patient-level bootstrap resamples. "
               "Source: eval_3."),
    "Table4": ("Table 4. Agreement between the two flags within each recorded treatment class, with "
               "the class size. Source: eval_3."),
    "Table5": ("Table 5. Error detection at the matched flag rate. Accuracy among flagged patients "
               "is one minus the error rate among them, with the interval bounds swapped. Retained "
               "accuracy is the accuracy among unflagged patients, which is the accuracy of the "
               "cases a service would automate. A random queue of the same size catches a share of "
               "errors equal to its flag rate in expectation, so recall minus random is the column "
               "to read. The last column is the share of correctly classified patients the flag "
               "marks anyway, computed as flagged correct over all correct. Source: eval_4."),
}


def main():
    """
    Builds the tables for both representations and writes the CSVs and one markdown
    file to results/paper.
    """
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    built = {rep: build(rep) for rep in ["indvA", "sumB"]}
    stale = any(built[r]["stale"] for r in built)
    suffix = "_PROVISIONAL" if stale else ""

    captions = dict(CAPTIONS)
    if stale:
        # On a run from the interpolated quantile, the Table 2 caption describes the
        # threshold that run used.
        captions["Table2A"] = captions["Table2A"].replace(
            "q is the 385th smallest of the 404 calibration nonconformity scores",
            "q is a high quantile of the 404 calibration nonconformity scores, taken here "
            "under NumPy's interpolated quantile in place of "
            "the 385th smallest")

    lines = ["# Paper 1. Tables 1 to 5", ""]
    lines.append("Built by tables_figures/tables.py from the saved evaluation tables. "
                 "Every value is copied from a CSV except the four columns the captions name.")
    lines.append("")
    for rep in built:
        lines.append(f"Run for {C.REPRESENTATIONS[rep]['label']}: {built[rep]['run'].name}")
    if stale:
        lines += ["", "PROVISIONAL. These runs have conformal thresholds from NumPy's interpolated "
                      "quantile. Run tools/regenerate_evaluations.py, then this script again, for the "
                      "versions that go in the paper."]
    lines.append("")

    for name in ["Table1", "Table2A", "Table2B", "Table3", "Table4", "Table5"]:
        lines += [f"## {name}", "", captions[name], ""]
        for rep in built:
            df = built[rep][name]
            df.to_csv(DIR_OUT / f"{name}_{C.REPRESENTATIONS[rep]['tag']}{suffix}.csv", index=False)
            lines += [f"### {C.REPRESENTATIONS[rep]['label']}", "",
                      to_pipe(df), ""]

    out = DIR_OUT / f"Tables_1_to_5{suffix}.md"
    out.write_text("\n".join(lines))
    print("checks: chance agreement reproduces kappa, and the correct-prediction counts reconcile, "
          "for every model in both representations.")
    print("stale run:", stale)
    print("written:", out)
    for rep in built:
        print(f"  {rep}: {built[rep]['run'].name}")


if __name__ == "__main__":
    main()
