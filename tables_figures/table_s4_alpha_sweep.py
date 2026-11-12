"""
table_s4_alpha_sweep.py: Supplementary Table S4, the alpha sweep at the reported
alpha values.

The table covers ten runs, five models under dataset A and five under dataset B,
at every alpha in config.ALPHA_REPORT. For each run it gives the empirical
coverage, both flag rates, the observed agreement and Cohen's kappa between the
two flags, and the error recall of each flag. The entropy flag rate is the random-
review baseline for entropy error recall.

Every cell is copied from the saved alpha sweep table of the newest run of each
representation, and a check traces each cell back to that table. Kappa reads
"undefined" where one flag marks every patient or none.

Output: Table_S4_alpha_sweep.csv in results/paper, one row per alpha and metric
and one column per run.

Run from the repository root:  python -m tables_figures.table_s4_alpha_sweep
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as C
from tables_figures.figures import newest_run

DIR_CODE = Path(__file__).resolve().parent.parent
DIR_OUT = DIR_CODE / "results" / "paper"

# column order and short labels of the table
MODELS = [
    ("Bootstrap Logistic Regression", "Boot LR"),
    ("Bayesian Logistic Regression", "Bayes LR"),
    ("SVM Ensemble", "SVM"),
    ("MC Dropout Neural Network", "MCD"),
    ("XGBoost Ensemble", "XGB"),
]
# row label, sweep column, decimals
METRICS = [
    ("Coverage", "coverage_test", 4),
    ("CP flag rate", "test_flag_rate_conformal", 4),
    ("Entropy flag rate", "test_flag_rate_entropy", 4),
    ("Observed agreement (%)", "pct_concordant", 2),  # stored in percent
    ("Kappa", "kappa", 3),
    ("CP error recall", "recall_conformal", 3),
    ("Entropy error recall", "recall_entropy", 3),
]
REPS = [("indvA", "A"), ("sumB", "B")]


def fmt(v: float, dec: int) -> str:
    """
    Formats one value to a fixed number of decimals.

    Input: v, the value; dec, the decimals.
    Output: the text, "undefined" for a missing value.
    """
    return "undefined" if pd.isna(v) else f"{v:.{dec}f}"


def main():
    """Builds Supplementary Table S4, checks every cell, and writes it to results/paper."""
    sweeps = {}
    for rep, _ in REPS:
        tag = C.REPRESENTATIONS[rep]["tag"]
        run = newest_run(rep)
        df = pd.read_csv(run / f"eval_5_{tag}_sensitivity/alpha_sweep_full_{tag}.csv")
        df["alpha"] = df["alpha"].round(2)
        sweeps[rep] = df.set_index(["alpha", "Model"])
        print("read:", run.name)

    rows = []
    for a in C.ALPHA_REPORT:
        a = round(a, 2)
        for label, col, dec in METRICS:
            row = {"alpha": f"{a:.2f}", "metric": label}
            for rep, ds in REPS:
                for model, short in MODELS:
                    row[f"{ds} {short}"] = fmt(sweeps[rep].loc[(a, model), col], dec)
            rows.append(row)
    out = pd.DataFrame(rows)

    # check: every cell traces back to the sweep table it came from
    for rep, ds in REPS:
        for model, short in MODELS:
            for a in C.ALPHA_REPORT:
                v = sweeps[rep].loc[(round(a, 2), model), "coverage_test"]
                cell = out.loc[(out.alpha == f"{a:.2f}") & (out.metric == "Coverage"), f"{ds} {short}"].item()
                assert np.isclose(float(cell), v, atol=5e-5), (rep, model, a)

    path = DIR_OUT / "Table_S4_alpha_sweep.csv"
    out.to_csv(path, index=False)
    print("written:", path, out.shape)


if __name__ == "__main__":
    main()
