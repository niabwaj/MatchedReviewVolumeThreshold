"""
table_s2_training_specs.py: Supplementary Table S2, the training specification of
each model.

For each of the five models the table lists the hyperparameter grid and its tuning
method, the fixed training settings and the seed of every random step. It also
lists the software versions behind the reported runs. The grids, seeds and run
sizes come from config, and the software versions from the manifest of the newest
run of each representation. The fixed settings, coded in the model training
modules, appear as text with the module each comes from.

Output: Table_S2_training_specs.csv in results/paper, one row per panel, model
and setting.

Run from the repository root:  python -m tables_figures.table_s2_training_specs
"""

import json
from pathlib import Path

import pandas as pd

from pipeline import config as C
from tables_figures.figures import newest_run

DIR_CODE = Path(__file__).resolve().parent.parent
DIR_OUT = DIR_CODE / "results" / "paper"


def grid(values) -> str:
    """
    Formats a hyperparameter grid as '{a, b, c}', trimming trailing zeros and adding
    thousands separators from 1,000.

    Input: values, the grid.
    Output: the formatted string.
    """
    def one(v):
        """Formats one grid value."""
        t = f"{v:.10f}".rstrip("0").rstrip(".")
        return f"{int(float(t)):,}" if float(t) >= 1000 else t
    return "{" + ", ".join(one(v) for v in values) + "}"


def main():
    """
    Builds Supplementary Table S2 and writes it to results/paper.

    Stops when config.SMOKE is True, since the table describes the full run.
    """
    if C.SMOKE:
        raise SystemExit("config.SMOKE is True; the table must describe the full run.")
    M, S = C.M_SAMPLES, C.DATA_SEED
    n_xgb = len(C.XGB_MAX_DEPTH_GRID) * len(C.XGB_LR_GRID)
    n_mcd = len(C.MCD_HIDDEN_UNITS_GRID) * len(C.MCD_DROPOUT_GRID) * len(C.MCD_L1_GRID)

    A = "A. Hyper-parameter grids and training settings"
    rows = [
        (A, "All models", "Data split",
         "Stratified partition into 1,616 training, 404 calibration and 505 test patients, identical for both representations"),
        (A, "All models", "Draws per patient (M)", f"{M:,} probability vectors per patient"),
        (A, "All models", "Tuning criterion",
         "Balanced accuracy, stratified 5-fold cross-validation within the training data, for the four models with a grid"),

        (A, "Bayesian logistic regression", "Likelihood", "Categorical (softmax) over four classes, linear logits (ml_Bayesian.py)"),
        (A, "Bayesian logistic regression", "Priors", "Intercepts Normal(0, 2.0); coefficients Normal(0, 1.5) (ml_Bayesian.py)"),
        (A, "Bayesian logistic regression", "Sampler",
         f"NUTS, {C.BAYES_CHAINS} chains of {C.BAYES_DRAWS:,} draws after {C.BAYES_TUNE:,} tuning iterations, target acceptance {C.BAYES_TARGET_ACC}"),
        (A, "Bayesian logistic regression", "Draws used",
         f"{M:,} of {C.BAYES_CHAINS * C.BAYES_DRAWS:,} posterior draws, selected at random without replacement"),

        (A, "Bootstrap logistic regression", "Grid",
         f"C in {grid(C.BOOTSTRAP_C_GRID)} ({len(C.BOOTSTRAP_C_GRID)} values), tuned separately for every member by {C.BOOTSTRAP_INNER_FOLDS}-fold cross-validation on its resample"),
        (A, "Bootstrap logistic regression", "Members",
         f"{M:,} multinomial logistic regressions, L2 penalty, lbfgs solver, up to 1,000 iterations, each fitted on a stratified bootstrap resample of the training set (ml_LogReg.py)"),

        (A, "SVM ensemble", "Grid",
         f"C in {grid(C.SVM_C_GRID)} ({len(C.SVM_C_GRID)} values), tuned separately for every member by {C.SVM_INNER_FOLDS}-fold cross-validation on its resample"),
        (A, "SVM ensemble", "Members",
         f"{M:,} RBF-kernel support vector machines, gamma '{C.SVM_GAMMA}', Platt scaling for class probabilities, each fitted on a stratified bootstrap resample of the training set (ml_SVM.py)"),

        (A, "XGBoost ensemble", "Grid",
         f"max_depth in {grid(C.XGB_MAX_DEPTH_GRID)} and learning_rate in {grid(C.XGB_LR_GRID)} ({n_xgb} combinations), tuned once by {C.XGB_INNER_FOLDS}-fold cross-validation on the full training set with {C.XGB_N_TREES} trees and row and column subsampling at 1.0"),
        (A, "XGBoost ensemble", "Members",
         f"{M:,} models on the full training set with the selected max_depth and learning_rate, {C.XGB_N_TREES} trees, row subsampling {C.XGB_SUBSAMPLE}, column subsampling {C.XGB_COLSAMPLE}, multiclass log-loss, each with its own seed (ml_XGBoost.py)"),

        (A, "Neural network with MC dropout", "Architecture",
         "Questionnaire features pass through dropout and a dense ReLU layer of h units with L1 penalty; demographic features join after this layer; dropout, then a dense softmax output of four units with L1 penalty; Glorot uniform initialisation (ml_MCD.py)"),
        (A, "Neural network with MC dropout", "Grid",
         f"h in {grid(C.MCD_HIDDEN_UNITS_GRID)}, dropout rate in {grid(C.MCD_DROPOUT_GRID)}, L1 strength in {grid(C.MCD_L1_GRID)} ({n_mcd} combinations), tuned by {C.MCD_INNER_FOLDS}-fold cross-validation"),
        (A, "Neural network with MC dropout", "Training",
         f"Adam, learning rate 0.001, sparse categorical cross-entropy, batch size {C.MCD_BATCH_SIZE}, up to {C.MCD_EPOCHS} epochs, early stopping on validation loss with patience {C.MCD_PATIENCE} and best weights restored; final model on the full training set with {C.MCD_VAL_SPLIT:.0%} held for early stopping (ml_MCD.py)"),
        (A, "Neural network with MC dropout", "Draws",
         f"{M:,} forward passes with both dropout layers active"),
    ]

    B = "B. Seeds"
    rows += [
        (B, "All models", "Train, calibration and test partition", f"{S}"),
        (B, "Bayesian logistic regression", "NUTS sampler; selection of the draws used", f"{S}; {S}"),
        (B, "Bootstrap logistic regression", "Member i (i = 0 to 999): resample, cross-validation folds and model",
         f"{S} + i"),
        (B, "SVM ensemble", "Member i (i = 0 to 999): resample, cross-validation folds, Platt scaling and model",
         f"{S} + i"),
        (B, "XGBoost ensemble", "Grid search folds and model; member i (i = 0 to 999)", f"{S}; {S} + i"),
        (B, "Neural network with MC dropout",
         "Cross-validation folds; fold model k (k = 0 to 4); final model initialisation and dropout masks; forward passes",
         f"{S}; {S} + k; {S}; {S}"),
        (B, "Evaluation", "Patient-level bootstrap intervals ({:,} resamples) and calibration resampling ({} draws)".format(C.N_BOOT, C.N_CAL_RESAMPLES),
         f"{C.BOOT_SEED}"),
    ]

    # software versions, read from the manifest of both reported runs
    envs = []
    for rep in ["indvA", "sumB"]:
        run = newest_run(rep)
        m = json.loads((run / "_manifest.json").read_text())
        envs.append((run.name, m["environment"]))
    assert envs[0][1]["packages"] == envs[1][1]["packages"], "the two runs used different package versions"
    env = envs[0][1]
    pk = env["packages"]
    Cp = "C. Software"
    used = [
        ("Python", env["python"], "All steps"),
        ("NumPy", pk["numpy"], "All steps"),
        ("pandas", pk["pandas"], "All steps"),
        ("SciPy", pk["scipy"], "Statistics"),
        ("scikit-learn", pk["sklearn"], "Bootstrap logistic regression, SVM ensemble, cross-validation, metrics"),
        ("PyMC", pk["pymc"], "Bayesian logistic regression"),
        ("ArviZ", pk["arviz"], "Posterior trace storage"),
        ("XGBoost", pk["xgboost"], "XGBoost ensemble"),
        ("TensorFlow", pk["tensorflow"], "Neural network with MC dropout"),
        ("matplotlib", pk["matplotlib"], "Figures"),
    ]
    rows += [(Cp, name, use, ver) for name, ver, use in used]
    rows.append((Cp, "Platform", "Operating system", env["platform"]))

    out = pd.DataFrame(rows, columns=["panel", "group", "setting", "value"])
    path = DIR_OUT / "Table_S2_training_specs.csv"
    out.to_csv(path, index=False)
    print("read:", ", ".join(r for r, _ in envs))
    print("written:", path, out.shape)


if __name__ == "__main__":
    main()
