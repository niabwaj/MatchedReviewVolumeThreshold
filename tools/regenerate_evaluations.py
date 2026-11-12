"""
regenerate_evaluations.py: rebuilds every evaluation table of a finished run from
its stored model fits.

For each representation the script reopens the newest full run folder and loads
the five stored results pickles of the fitted models. It then runs the six
evaluations in order, overwriting the folder's tables. The model pickles, the data
and every other run folder stay untouched. Most of the run time goes to the alpha
sweep and the calibration resampling.

Run from the repository root:
    python -m tools.regenerate_evaluations
    %run -m tools.regenerate_evaluations        inside a notebook
"""

from pipeline import quiet
quiet.silence_known_noise()

from pipeline import config as C
from pipeline import data_io
from models import run_models
from pipeline import report as R
from evaluation import eval_0_performance as eval_0
from evaluation import eval_1_validity as eval_1
from evaluation import eval_2_flags as eval_2
from evaluation import eval_3_concordance as eval_3
from evaluation import eval_4_error as eval_4
from evaluation import eval_5_sensitivity as eval_5

NOTE = ("evaluation tables regenerated from the stored fits. "
        "No model refitted.")


def rerun(representation: str, run_id: str = None) -> dict:
    """
    Reopens one run folder and rebuilds all six evaluation groups from its stored fits.

    Input: representation, a key of config.REPRESENTATIONS; run_id, the folder to
    reopen, or None for the newest full run.
    Output: dict with run, perf, uq, validity, concordance, error, sensitivity and
    inventory. The rebuilt tables are written to the run folder.
    """
    R.banner(f"REGENERATING EVALUATIONS — {C.REPRESENTATIONS[representation]['label']}")

    run = data_io.resume_run(representation, run_id, note=NOTE)
    cfg = C.base_config(representation, run["run_dir"])
    data = data_io.load_splits(run)
    data = data_io.scale_splits(run, data)

    # reuse=True with every model key, so each model loads from the pickle in this
    # folder.
    all_models = run_models.train_all(run, data, C.MODEL_KEYS, reuse=True)
    missing = [k for k in C.MODEL_KEYS if k not in all_models]
    if missing:
        raise RuntimeError(f"No stored results for {missing} in {run['run_dir'].name}. "
                           f"Point run_id at the folder holding all five fits.")

    perf = eval_0.run(run, all_models, data, cfg)
    uq = eval_2.run(run, all_models, data, cfg, alpha=C.CP_ALPHA)
    validity = eval_1.run(run, all_models, uq, data, alpha=C.CP_ALPHA)
    concordance = eval_3.run(run, all_models, uq, data)
    error = eval_4.run(run, all_models, uq, data)
    sensitivity = eval_5.run(run, all_models, uq, data)
    inventory = R.finalise(run)

    return {"run": run, "perf": perf, "uq": uq, "validity": validity,
            "concordance": concordance, "error": error,
            "sensitivity": sensitivity, "inventory": inventory}


if __name__ == "__main__":
    out = {rep: rerun(rep) for rep in ["indvA", "sumB"]}
    print("\nDone. Both representations regenerated from stored fits.")
    for rep, o in out.items():
        print(f"  {rep}: {o['run']['run_dir']}")
