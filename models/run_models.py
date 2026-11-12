"""
run_models.py: trains the five models for one representation.

Each model returns a results dict of the same shape, so every evaluation treats
the five models identically:
    all_probs_{train,cal,test}   (M, n, K)   the M draws per patient
    mean_probs_{train,cal,test}  (n, K)      the mean predictive distribution
    y_pred_{train,cal,test}      (n,)        argmax of the mean, zero-indexed
"""

import time
import traceback

import numpy as np
import pandas as pd

from pipeline import checks
from pipeline import config as C
from pipeline import data_io
from pipeline import report as R

# TRAINERS: the module and training function of each model. Each module is
# imported inside train_one, so a missing optional package stops one model only.
TRAINERS = {
    "bootstrap": ("models.ml_LogReg",   "_run_bootstrap"),
    "bayesian":  ("models.ml_Bayesian", "_run_bayesian"),
    "mcd":       ("models.ml_MCD",      "_run_mcd"),
    "svm":       ("models.ml_SVM",      "_run_svm"),
    "xgboost":   ("models.ml_XGBoost",  "_run_xgboost"),
}

# PAPER_OUTPUTS: the reported results each model feeds, printed after it trains.
PAPER_OUTPUTS = {
    "bayesian":  "reference architecture: Table 3 row, all thresholds, all worked examples",
    "bootstrap": "Table 3 row, concordance row, error-detection row",
    "mcd":       "Table 3 row, concordance row, the worst-calibrated comparison",
    "svm":       "Table 3 row, concordance row, error-detection row",
    "xgboost":   "Table 3 row, concordance row, the scale-invariant comparison",
}


def _results_path(run: dict, model_key: str):
    """
    Returns the path of a model's results pickle in the run folder.

    Input: run, the run context; model_key, a key of TRAINERS.
    Output: the pickle path.
    """
    return run["run_dir"] / "models" / f"results_{model_key}_{run['version']}.pkl"


def train_one(run: dict, data: dict, model_key: str, reuse: bool = True) -> dict:
    """
    Trains one model, or reloads its stored fit, and returns its results dict.

    Input: run, the run context; data, the loaded splits; model_key, a key of
    config.MODEL_KEYS; reuse, loads the stored pickle instead of retraining when the
    run folder already holds this model's fit.
    Output: the results dict, with train_seconds added for a new fit. The training
    function saves the dict to the run folder itself.
    """
    path = _results_path(run, model_key)
    if reuse and path.exists():
        res = data_io.load_pickle(path)
        print(f"  {C.MODEL_LABELS[model_key]}: reused existing fit ({path.name})")
        return res

    module_name, fn_name = TRAINERS[model_key]
    module  = __import__(module_name, fromlist=[fn_name])
    trainer = getattr(module, fn_name)
    cfg     = C.model_config(model_key, run["representation"], run["run_dir"])

    t0  = time.time()
    res = trainer(data["X"]["train"], data["y"]["train"],
                  data["X"]["cal"],   data["y"]["cal"],
                  data["X"]["test"],  data["y"]["test"],
                  cfg)
    res["train_seconds"] = round(time.time() - t0, 1)
    return res


def train_all(run: dict, data: dict, models_to_run: list = None,
              reuse: bool = True, stop_on_error: bool = False) -> dict:
    """
    Trains every selected model in turn and records each in the manifest.

    The package check runs first, so the manifest records a model blocked by the
    environment as blocked, apart from an excluded model. A model with
    a stored fit in the run folder counts as runnable, since reuse loads the pickle
    without importing the framework.

    Input: run, the run context; data, the loaded splits; models_to_run, a list of
    model keys (default config.MODEL_KEYS); reuse, as in train_one; stop_on_error,
    re-raises the first training error when True.
    Output: all_models, a dict of model key to results dict.
    """
    models_to_run = list(models_to_run or C.MODEL_KEYS)
    unknown = [m for m in models_to_run if m not in TRAINERS]
    if unknown:
        raise KeyError(f"Unknown model key(s): {unknown}. Expected {list(TRAINERS)}.")

    # A model blocked by a missing package is recorded as blocked in the manifest. A
    # model with a stored results pickle stays runnable in any kernel, since reuse loads
    # the pickle without importing its framework.
    env      = checks.check_environment(verbose=False)
    already  = [k for k in models_to_run if reuse and _results_path(run, k).exists()]
    blocked  = {k: env["blocked"][k] for k in models_to_run
                if k in env["blocked"] and k not in already}
    runnable = [k for k in models_to_run if k not in blocked]

    R.banner(f"TRAINING {len(runnable)} ARCHITECTURE(S) — {run['rep']['label']}")
    if C.SMOKE:
        print("  SMOKE MODE: reduced draws, grids and epochs. Not reportable.\n")

    if blocked:
        print(f"  Kernel: {env['executable']}")
        for key, missing in blocked.items():
            hints = "; ".join(checks.INSTALL_HINT.get(m, f"install {m}") for m in missing)
            print(f"  SKIPPED — {C.MODEL_LABELS[key]}: no {', '.join(missing)} in this kernel")
            print(f"     {hints}")
            run["manifest"]["models"][key] = {"label": C.MODEL_LABELS[key],
                                              "blocked_by_environment": missing}
        print("  If these packages exist in another conda environment, switch the "
              "notebook kernel rather than installing into this one.\n")
        data_io.write_manifest(run)

    if not runnable:
        print("  Nothing can be trained in this kernel. Fix the environment and re-run.")
        return {}

    all_models, failures = {}, {}
    for key in runnable:
        try:
            res = train_one(run, data, key, reuse=reuse)
            all_models[key] = res
            run["manifest"]["models"][key] = {
                "label":    C.MODEL_LABELS[key],
                "seconds":  res.get("train_seconds"),
                "shapes":   {s: list(np.shape(res[f"all_probs_{s}"]))
                             for s in ["train", "cal", "test"]},
                "bal_acc":  {k: v for k, v in res.get("metadata", {}).items()
                             if k.startswith("bal_acc")},
                "path":     str(_results_path(run, key)),
            }
        except Exception as exc:
            failures[key] = f"{type(exc).__name__}: {exc}"
            run["manifest"]["models"][key] = {"label": C.MODEL_LABELS[key],
                                              "failed": failures[key]}
            print(f"\n  FAILED — {C.MODEL_LABELS[key]}: {failures[key]}")
            traceback.print_exc()
            if stop_on_error:
                data_io.write_manifest(run)
                raise
        data_io.write_manifest(run)

    _print_confirmation(run, all_models, failures)
    return all_models


def _print_confirmation(run: dict, all_models: dict, failures: dict) -> None:
    """
    Prints one summary block per trained model and lists any failures.

    Input: run, the run context; all_models, from train_all; failures, model key
    to error message.
    Output: none.
    """
    R.banner("TRAINING COMPLETE")
    for key, res in all_models.items():
        meta = res.get("metadata", {})
        print(f"\n  {C.MODEL_LABELS[key]}")
        print(f"    fitted in {res.get('train_seconds', float('nan'))} s   "
              f"M = {np.shape(res['all_probs_test'])[0]}")
        print(f"    bal. acc   train {meta.get('bal_acc_tr', float('nan')):.4f}   "
              f"cal {meta.get('bal_acc_cal', float('nan')):.4f}   "
              f"test {meta.get('bal_acc_test', float('nan')):.4f}")
        print(f"    arrays     "
              f"cal {np.shape(res['all_probs_cal'])}   "
              f"test {np.shape(res['all_probs_test'])}")
        print(f"    feeds      {PAPER_OUTPUTS[key]}")
    if failures:
        print("\n  Not trained:")
        for k, v in failures.items():
            print(f"    {C.MODEL_LABELS[k]}: {v}")
    print(f"\n  all_models carries {len(all_models)} architecture(s): "
          f"{', '.join(all_models)}")
    if len(all_models) < len(C.MODEL_KEYS):
        absent = [C.MODEL_LABELS[k] for k in C.MODEL_KEYS if k not in all_models]
        print(f"  Not in this run: {', '.join(absent)}. Every evaluation below reports "
              f"only the architectures present, so the tables stay internally consistent.")


def load_all(run_dir, representation: str) -> dict:
    """
    Rebuilds all_models from the stored pickles of a finished run folder.

    Input: run_dir, the run folder; representation, a key of config.REPRESENTATIONS.
    Output: dict of model key to results dict, for every model with a stored pickle.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    version = C.REPRESENTATIONS[representation]["version"]
    out = {}
    for key in C.MODEL_KEYS:
        p = run_dir / "models" / f"results_{key}_{version}.pkl"
        if p.exists():
            out[key] = data_io.load_pickle(p)
    print(f"  Loaded {len(out)} model(s) from {run_dir.name}: {', '.join(out)}")
    return out
