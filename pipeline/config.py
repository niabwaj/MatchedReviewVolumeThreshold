"""
config.py: every constant of the pipeline in one place.

Paths, seeds, run sizes, hyperparameter grids, the alpha grid, the review rates
and the labels all live here, so a change to any of them reaches the whole run.
Every path is relative to the repository root, so the repository runs unchanged
on another machine. The run manifest records a snapshot of these constants for
every execution.
"""

from datetime import datetime
from pathlib import Path

# =====================================================================
# 1. PATHS
# =====================================================================

# Paths are relative to the repository root.
DIR_CODE    = Path(__file__).resolve().parent.parent
DIR_DATA    = DIR_CODE / "data"
DIR_SPLITS  = {"train": DIR_DATA / "train", "cal": DIR_DATA / "cal", "test": DIR_DATA / "test"}
DIR_RAW     = DIR_DATA / "raw_data"
DIR_MODELS  = DIR_CODE / "model_objects"
DIR_RESULTS = DIR_CODE / "results"

# =====================================================================
# 2. REPRESENTATIONS
# =====================================================================

# REPRESENTATIONS: the two feature representations. Every result is reported once
# per representation, from the same code.
REPRESENTATIONS = {
    "indvA": {
        "label":      "Individual items (A)",
        "tag":        "indv",
        "version":    "vA_indv",
        "x_files":    {"train": "x_indv_train.csv",
                       "cal":   "x_indv_calibration.csv",
                       "test":  "x_indv_test.csv"},
        "y_files":    {"train": "y_indv_train.csv",
                       "cal":   "y_indv_calibration.csv",
                       "test":  "y_indv_test.csv"},
        "split_pkl":  "split_indices_indv_seed42.pkl",
        "n_features": 86,
        "prescaled":  True,  # items divided by their instrument maximum during data preparation
    },
    "sumB": {
        "label":      "Summed scores (B)",
        "tag":        "sum",
        "version":    "vB_sum",
        "x_files":    {"train": "x_sum_train.csv",
                       "cal":   "x_sum_calibration.csv",
                       "test":  "x_sum_test.csv"},
        "y_files":    {"train": "y_sum_train.csv",
                       "cal":   "y_sum_calibration.csv",
                       "test":  "y_sum_test.csv"},
        "split_pkl":  "split_indices_sum_seed42.pkl",
        "n_features": 30,
        "prescaled":  True,  # sums divided by their maximum possible score during data preparation
    },
}

TARGET_COL  = "TreatmentType"
CLASS_NAMES = {0: "Depression", 1: "Panic Disorder",
               2: "Social Phobia", 3: "Specific Phobia"}
N_CLASSES   = 4

# =====================================================================
# 3. SCALING
# =====================================================================

# SCALE_ON_LOAD False keeps the instrument scale. Both representations arrive on 0 to
# 1, since every questionnaire item or summed score is divided by its maximum possible
# score during data preparation: PHQ sums by 27 (9 items times 3), and individual
# items by 3, 4 or 8 depending on the instrument. The divisors are fixed properties of
# the questionnaires and independent of the patients, so the exchangeability
# assumption behind the conformal guarantee holds. The instrument scale also keeps
# the information of how close each patient came to the top of each item.
# SCALE_ON_LOAD True switches to a min-max scaler fit on the training split. The
# preflight gate adapts to either rule.
SCALER        = "minmax"
SCALE_ON_LOAD = False

# =====================================================================
# 4. REPRODUCIBILITY AND RUN SIZE
# =====================================================================

DATA_SEED = 42
M_SAMPLES = 1000          # posterior draws / ensemble members per patient

# SMOKE = True runs reduced draws, grids and epochs, which tests the pipeline in
# minutes. Smoke run folders carry the prefix smoke_ and are for testing only.
SMOKE = False

SMOKE_OVERRIDES = {
    "M_SAMPLES":        25,
    "BAYES_DRAWS":      100,
    "BAYES_TUNE":       100,
    "BAYES_CHAINS":     2,
    "MCD_EPOCHS":       20,
    "MCD_DROPOUT_GRID": [0.2],
    "MCD_HIDDEN_UNITS_GRID": [16],
    "MCD_L1_GRID":      [0.01],
    "MCD_PATIENCE":     5,
    "BOOTSTRAP_C_GRID": [0.1, 1.0],
    "SVM_C_GRID":       [1.0],
    "XGB_MAX_DEPTH_GRID": [3],
    "XGB_LR_GRID":      [0.1],
    "N_BOOT":           200,
    "N_CAL_RESAMPLES":  20,
}

# =====================================================================
# 5. MODELS
# =====================================================================

MODEL_KEYS   = ["bayesian", "bootstrap", "mcd", "svm", "xgboost"]
MODEL_LABELS = {
    "bayesian":  "Bayesian Logistic Regression",
    "bootstrap": "Bootstrap Logistic Regression",
    "mcd":       "MC Dropout Neural Network",
    "svm":       "SVM Ensemble",
    "xgboost":   "XGBoost Ensemble",
}

# REFERENCE_MODEL: the model printed in full at every checkpoint, so the format of
# every table is checked by eye on one model.
REFERENCE_MODEL = "bayesian"

BOOTSTRAP_C_GRID      = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_INNER_FOLDS = 5

BAYES_DRAWS      = 1000
BAYES_TUNE       = 1000
BAYES_CHAINS     = 4
BAYES_TARGET_ACC = 0.95
BAYES_PROGRESSBAR = False  # console output only; see model_config below

# The three grids the MC dropout network tunes over by cross-validation, with the
# training settings it uses. The key names match the keys the training function
# reads. The 24 dropout-network combinations are already the largest tuning budget of
# the five models, against 9 for XGBoost and 5 for the SVM.
MCD_HIDDEN_UNITS_GRID = [16, 32, 64]
MCD_DROPOUT_GRID      = [0.1, 0.2, 0.3, 0.4]
MCD_L1_GRID           = [0.001, 0.01]
MCD_EPOCHS            = 500
MCD_BATCH_SIZE        = 32
MCD_VAL_SPLIT         = 0.2
MCD_PATIENCE          = 50
MCD_INNER_FOLDS       = 5

SVM_C_GRID      = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_GAMMA       = "scale"
SVM_INNER_FOLDS = 5

XGB_N_TREES        = 300
XGB_MAX_DEPTH_GRID = [3, 4, 6]
XGB_LR_GRID        = [0.05, 0.1, 0.2]
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8
XGB_INNER_FOLDS    = 5

# =====================================================================
# 6. UNCERTAINTY AND EVALUATION SETTINGS
# =====================================================================

CP_ALPHA = 0.05  # pre-specified operating point

# ALPHA_GRID: the sensitivity analysis sweeps alpha from 0.00 to 1.00 in steps of
# 0.01. Each point re-derives the conformal threshold from the stored calibration
# probabilities, rebuilds the prediction sets, re-matches the entropy threshold to
# the new flag rate, and recomputes the comparison, so 101 points take seconds.
#
# Both endpoints bound the curve. At alpha = 0 every set holds all four classes, so
# the conformal flag marks every patient. At alpha = 1 almost every set is empty, so
# the conformal flag marks almost every patient. Agreement statistics are undefined where
# a flag is constant, and the tables report those cells as undefined.
ALPHA_GRID = [round(i / 100, 2) for i in range(0, 101)]

# ALPHA_REPORT: the alpha values tabulated in Supplementary Table S4, across the
# whole range in steps of 0.10 past 0.20. The full sweep stays in its own file.
ALPHA_REPORT = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.80, 0.90, 1.00]

# REVIEW_RATES: fixed review budgets, reported alongside the alpha sweep, since alpha
# 0.05 flags more patients than a service can review.
REVIEW_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]

# LAC is the primary nonconformity score. APS is a pre-specified sensitivity analysis
# on all five models and both representations.
CONFORMAL_SCORES = ["lac", "aps"]
PRIMARY_SCORE    = "lac"

# Patient-level paired bootstrap, so every interval reflects sampling of patients.
N_BOOT     = 2000
BOOT_SEED  = 2026
CI_LEVEL   = 0.95

# N_CAL_RESAMPLES: resamples of the 404 calibration patients, to measure how much
# the thresholds depend on one calibration draw.
N_CAL_RESAMPLES = 200

M_GRID = [25, 50, 100, 200, 300, 500, 700, 1000]  # M convergence check

CONCORDANCE_CATEGORIES = ["Concordant Certain", "Concordant Uncertain",
                          "Discordant Type A", "Discordant Type B"]

# =====================================================================
# 7. OUTPUT GROUPS
# =====================================================================

# EVAL_GROUPS: one output folder per evaluation group, named by its content.
EVAL_GROUPS = {
    "eval_0": "modelPerformance",
    "eval_1": "conformalValidity",
    "eval_2": "thresholdsAndFlags",
    "eval_3": "concordance",
    "eval_4": "errorDetection",
    "eval_5": "sensitivity",
}


# =====================================================================
# 8. HELPERS
# =====================================================================

def make_run_id(smoke: bool = None) -> str:
    """
    Returns the timestamp that names a run folder, in the form YYYYMMDD-HHMM, with the
    prefix smoke_ for a smoke run.

    Input: smoke, overrides SMOKE when given.
    Output: the run id string.
    """
    smoke = SMOKE if smoke is None else smoke
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"smoke_{stamp}" if smoke else stamp


# _FULL_VALUES: the full-run values, captured the first time apply_smoke_overrides
# runs, before any change.
_FULL_VALUES: dict = {}


def apply_smoke_overrides(verbose: bool = True) -> dict:
    """
    Sets the run-size constants for the current SMOKE setting.

    The full values are captured on the first call and restored whenever SMOKE is
    False. A kernel that ran in smoke mode therefore returns to the full sizes for a
    full run.

    Input: verbose, prints the settings in effect when True.
    Output: dict of the governed constants and their current values, recorded in the
    run manifest.
    """
    g = globals()
    if not _FULL_VALUES:
        _FULL_VALUES.update({k: g[k] for k in SMOKE_OVERRIDES if k in g})

    target = SMOKE_OVERRIDES if SMOKE else _FULL_VALUES
    for k, v in target.items():
        g[k] = v

    if verbose:
        if SMOKE:
            print(f"  SMOKE MODE: {len(SMOKE_OVERRIDES)} settings reduced, "
                  f"M_SAMPLES = {g['M_SAMPLES']}. Not reportable.")
        else:
            print(f"  FULL RUN: M_SAMPLES = {g['M_SAMPLES']}, "
                  f"N_BOOT = {g['N_BOOT']}, N_CAL_RESAMPLES = {g['N_CAL_RESAMPLES']}.")
    return {k: g[k] for k in SMOKE_OVERRIDES}


def model_config(model_key: str, representation: str, run_dir) -> dict:
    """
    Builds the config dict a model training function reads.

    Input: model_key, a key of MODEL_KEYS; representation, a key of
    REPRESENTATIONS; run_dir, the run folder.
    Output: dict with the shared keys (data_seed, version, model_dir, results_dir,
    m_samples, n_posterior_samples) plus the model's hyperparameters.
    """
    rep  = REPRESENTATIONS[representation]
    run_dir = Path(run_dir)
    base = {
        "data_seed":   DATA_SEED,
        "version":     rep["version"],
        "model_dir":   str(run_dir / "models"),
        "results_dir": str(run_dir / "models"),
        "m_samples":   M_SAMPLES,
        "n_posterior_samples": M_SAMPLES,
    }
    extra = {
        "bootstrap": {"c_grid": BOOTSTRAP_C_GRID, "inner_folds": BOOTSTRAP_INNER_FOLDS},
        "bayesian":  {"draws": BAYES_DRAWS, "tune": BAYES_TUNE, "chains": BAYES_CHAINS,
                      "target_accept": BAYES_TARGET_ACC,
                      # console output only: PyMC redraws the sampler progress bar thousands of times,
                      # and a notebook stores every redraw.
                      "progressbar": BAYES_PROGRESSBAR},
        "mcd":       {"hidden_units_grid": MCD_HIDDEN_UNITS_GRID,
                      "dropout_grid": MCD_DROPOUT_GRID, "l1_grid": MCD_L1_GRID,
                      "n_fwd_passes": M_SAMPLES, "epochs": MCD_EPOCHS,
                      "batch_size": MCD_BATCH_SIZE, "val_split": MCD_VAL_SPLIT,
                      "patience": MCD_PATIENCE, "inner_folds": MCD_INNER_FOLDS},
        "svm":       {"c_grid": SVM_C_GRID, "gamma": SVM_GAMMA,
                      "inner_folds": SVM_INNER_FOLDS},
        "xgboost":   {"n_trees": XGB_N_TREES, "max_depth_grid": XGB_MAX_DEPTH_GRID,
                      "lr_grid": XGB_LR_GRID, "subsample": XGB_SUBSAMPLE,
                      "colsample_bytree": XGB_COLSAMPLE, "inner_folds": XGB_INNER_FOLDS},
    }[model_key]
    return {**base, **extra}


def base_config(representation: str, run_dir) -> dict:
    """
    Builds the shared config dict the uncertainty and evaluation functions read.

    Input: representation, a key of REPRESENTATIONS; run_dir, the run folder.
    Output: dict with data_seed, version, model_dir, results_dir and m_samples.
    """
    rep = REPRESENTATIONS[representation]
    run_dir = Path(run_dir)
    return {
        "data_seed":   DATA_SEED,
        "version":     rep["version"],
        "model_dir":   str(run_dir / "models"),
        "results_dir": str(run_dir),
        "m_samples":   M_SAMPLES,
    }
