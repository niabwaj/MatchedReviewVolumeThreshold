"""
eval_0_performance.py: classification and calibration performance of each model.

The module computes balanced accuracy, AUC-ROC, F1, precision, recall, ECE, Brier
score and NLL for every model on all three splits. It saves the test table, the
calibration table and the per-class ECE table to the eval_0 group folder. Each
model is evaluated as fitted, and ECE, Brier score and NLL measure the quality of
its probabilities.
"""

import pandas as pd

from pipeline import config as C
from pipeline import report as R
from evaluation import stats as S
from uq.performance import (run_performance, make_performance_table,
                            make_per_class_calibration_table)

GROUP = "eval_0"


def run(run_ctx: dict, all_models: dict, data: dict, config: dict) -> dict:
    """
    Computes the performance metrics for every model and split and saves three tables.

    Input: run_ctx, the run context; all_models, model key to results dict; data,
    the loaded splits; config, the shared config dict from config.base_config.
    Output: perf, the nested metrics dict {model: {split: metrics}}. The saved
    tables are performance_test, performance_calibration and ece_per_class_test.
    """
    R.banner("EVALUATION 0 — MODEL PERFORMANCE")

    perf = run_performance(all_models=all_models,
                           y_train=data["y"]["train"], y_cal=data["y"]["cal"],
                           y_test=data["y"]["test"], config=config)

    # The table builders return Model as the index. Resetting it makes Model a column
    # in every saved CSV.
    df_test = make_performance_table(perf, split="test").reset_index()
    df_cal  = make_performance_table(perf, split="cal").reset_index()
    df_ece  = make_per_class_calibration_table(perf, split="test").reset_index()

    R.save_table(run_ctx, GROUP, "performance_test", df_test,
                 "Discrimination and calibration on the held-out test split, one row per "
                 "architecture. The paper's model table.", item="18")
    R.save_table(run_ctx, GROUP, "performance_calibration", df_cal,
                 "The same metrics on the calibration split, reported so the calibration "
                 "set is shown to be comparable to test before exchangeability is claimed.",
                 item="18")
    R.save_table(run_ctx, GROUP, "ece_per_class_test", df_ece,
                 "One-vs-rest expected calibration error per treatment class per model. "
                 "Supports the statement that no probability calibration was applied.",
                 item="18")

    R.show(df_test, "Test performance, all models", full=True)
    R.show(df_ece, "One-vs-rest ECE per class, test split", full=True)

    # worst_calibrated: the model with the highest test ECE.
    worst = df_test.sort_values("ECE", ascending=False).iloc[0]  # Model is a column here
    print(f"\n  Worst calibrated on test: {worst['Model']} (ECE {worst['ECE']:.4f}). "
          f"Expect the highest entropy threshold and the lowest concordance here.")
    return perf
