# Matched-volume uncertainty flags: conformal prediction and entropy

This repository holds the code for the manuscript "Deriving an Entropy Threshold from Conformal Prediction at a Matched Review Volume for Patient-Level Uncertainty Flags in Mental Health Treatment Classification: Development and Evaluation Study".

## What the pipeline does

Five machine learning models predict one of four treatment classes for patients referred to an internet-delivered cognitive behavioural therapy service. Each model returns M = 1,000 predicted probability vectors per patient. Two uncertainty quantification methods then read the same mean predictive distribution:

- **Conformal prediction** (inductive, LAC score) returns a set of plausible treatment classes for each patient, with marginal coverage of at least 1 - alpha. A patient whose set holds two or more classes, or none, is flagged uncertain.
- **Entropy** measures the spread of the mean predicted probabilities across the four classes. Entropy has no agreed flagging threshold of its own.

The pipeline derives the entropy threshold from conformal prediction at a matched review volume. The conformal flag rate r is measured on the calibration split. The entropy threshold t is the (1 - r) quantile of calibration entropy, so both methods flag the same share of calibration patients. Both thresholds then apply to the held-out test split.

On the test split the pipeline measures:

1. the validity of the conformal sets: coverage overall and within each treatment class, and set size
2. the agreement between the two flags: the four flag groups, kappa, PABAK, agreement by class, and the rank correlation of the two continuous scores
3. the share of the model's errors in each review queue, against a random queue of the same size, at the matched volume and at fixed review budgets
4. the sensitivity of these results to alpha, the calibration draw, the nonconformity score (APS in place of LAC) and the number of draws M

Every analysis runs twice, once per feature representation of the same 2,525 patients. Dataset A holds the 67 individual questionnaire items (86 features in total), and dataset B the 11 summed questionnaire scores (30 features in total).

## Repository layout

```
data_clean.ipynb          cleans the source referral records
data_split.ipynb          builds the training, calibration and test splits
run_all_indvA.ipynb       full pipeline, dataset A
run_all_sumB.ipynb        full pipeline, dataset B
paper_results.ipynb       paper figures and Tables 1 to 5

pipeline/                 configuration, data input and output, preflight checks, reporting
    config.py             every constant: paths, seeds, grids, alpha values, labels
    data_io.py            run folders, split loading, scaling rule, manifest
    checks.py             preflight gate and environment check
    report.py             saves every table with a caption and records it in the manifest
    quiet.py              silences three named, harmless console warnings
models/                   the five models and the training loop
    ml_LogReg.py          bootstrap logistic regression ensemble
    ml_Bayesian.py        Bayesian multinomial logistic regression (PyMC, NUTS)
    ml_SVM.py             bootstrap ensemble of RBF support vector machines
    ml_MCD.py             Monte Carlo dropout neural network
    ml_XGBoost.py         stochastic XGBoost ensemble
    run_models.py         trains or reloads the selected models
uq/                       metrics and uncertainty methods
    performance.py        classification and calibration metrics
    uncertainty.py        entropy decomposition, conformal thresholds, flags, concordance
evaluation/               the six evaluation groups and shared statistics
    eval_0_performance.py   model performance
    eval_1_validity.py      conformal coverage and set efficiency
    eval_2_flags.py         thresholds and flags
    eval_3_concordance.py   agreement between the two flags
    eval_4_error.py         error detection against a random queue
    eval_5_sensitivity.py   alpha sweep, calibration resampling, APS, M convergence
    stats.py                patient-level bootstrap and agreement statistics
tables_figures/           paper figures and tables, built from saved results
    figures.py            Figures 3 to 5 and the alpha sweep figure
    tables.py             Tables 1 to 5
    table1_bootstrap.py   bootstrap spread for Table 1
    table_s2_training_specs.py   Supplementary Table S2
    table_s4_alpha_sweep.py      Supplementary Table S4
tools/
    regenerate_evaluations.py    rebuilds every evaluation table from stored fits
    coverage_by_resplit.py       coverage over 500 re-splits of the held-out patients
data/README.md            the expected data files
environment.yml           the conda environment
```

## Environment

```
conda env create -f environment.yml
conda activate uq-flags
python -m ipykernel install --user --name uq-flags --display-name "UQ flags"
```

Every notebook opens on the "UQ flags" kernel. Every notebook and command below runs from the repository root, so the subfolders import as packages.

## How to run

1. **Data.** `data_clean.ipynb` reads the source records from the data custodian's secure database and writes `sx_clean_df.csv` and `sx_clean_Total_df.csv`, which go into `data/raw_data/`. `data_split.ipynb` then writes the training, calibration and test splits under `data/` and the split indices `split_indices_indv_seed42.pkl` and `split_indices_sum_seed42.pkl` at the repository root. `data/README.md` lists every file.
2. **Smoke run.** In `run_all_indvA.ipynb`, set `C.SMOKE = True` in section 0 and run the whole notebook. A smoke run uses reduced draws, grids and epochs, takes minutes, and writes to a run folder with the prefix `smoke_`.
3. **Full run.** Set `C.SMOKE = False` and run `run_all_indvA.ipynb` again, then `run_all_sumB.ipynb`. Each writes one run folder under `results/`. The preflight gate in section 2 halts on any data fault and states the resolution.
4. **Coverage over re-splits.** Run `python -m tools.coverage_by_resplit`, which reads the newest full run of each representation. Figure 4 reads its output.
5. **Paper figures and Tables 1 to 5.** Run `paper_results.ipynb`. It rebuilds the evaluation tables of the newest full run of each representation from the stored fits, then writes the figures and tables to `results/paper/`.
6. **Table 1 bootstrap and supplementary tables.** Each script reads the newest full run of each representation:

```
python -m tables_figures.table1_bootstrap
python -m tables_figures.table_s2_training_specs
python -m tables_figures.table_s4_alpha_sweep
```

## Data flow within one run

```
data/{train,cal,test}/x_*.csv, y_*.csv
        |
        v
data_io.load_splits        loads the splits as stored, casts boolean columns to int
        |
        v
checks.run_preflight       halts on any fault and names the models at risk
        |
        v
data_io.scale_splits       records the scaling rule: features keep their instrument scale
        |
        v
run_models.train_all       five models, M = 1,000 draws per patient
        |                  -> all_probs (M, n, 4), mean_probs (n, 4), y_pred (n,)
        v
eval_0  performance        balanced accuracy, AUC-ROC, ECE, Brier score, NLL
        |
        v
eval_2  thresholds         q from the calibration nonconformity scores
                           r = calibration flag rate of conformal prediction
                           t = (1 - r) quantile of calibration entropy
                           both applied to test
        |
        +--> eval_1  coverage, coverage by class, set size and informativeness
        +--> eval_3  concordance: flag groups, kappa, PABAK, by class, Spearman
        +--> eval_4  error recall against a random queue, at matched and fixed rates
        +--> eval_5  alpha sweep, calibration resampling, LAC against APS, M convergence
        |
        v
report.finalise            closes the manifest and prints the output inventory
        |
        v
results/run_<YYYYMMDD-HHMM>_<representation>/
```

## Output layout

```
results/
  run_<YYYYMMDD-HHMM>_indvA/
    _manifest.json                    configuration, data hashes, package versions, timings
    models/                           one results pickle per model, plus the fitted models
    eval_0_indv_modelPerformance/
    eval_1_indv_conformalValidity/
    eval_2_indv_thresholdsAndFlags/
    eval_3_indv_concordance/
    eval_4_indv_errorDetection/
    eval_5_indv_sensitivity/
  run_<YYYYMMDD-HHMM>_sumB/
    ...
  coverage_by_resplit.csv             coverage over re-splits, from tools/coverage_by_resplit.py
  paper/                              figures, Tables 1 to 5, Supplementary Tables S2 and S4
```

Every saved table has a JSON sidecar with a one-sentence caption, its shape and a timestamp, and the run manifest lists every saved file.

## Method decisions in the code

- **Nonconformity score.** LAC is the primary score. Under LAC a set holds two or more classes exactly when the second-largest mean probability reaches 1 - q, so ranking patients by that probability gives the exact conformal score at any review budget. APS, in its non-randomised form, runs on all five models and both representations as a pre-specified sensitivity analysis. Non-randomised APS over-covers by construction, so its sets are larger.
- **Conformal threshold.** q is the k-th smallest calibration score, with k = ceil((n + 1)(1 - alpha)), taken by sorting and indexing.
- **Empty sets.** An empty prediction set stays empty and raises the conformal flag, as a set of two or more classes does. Coverage is measured on the sets as constructed. At alpha 0.05 no set can be empty, since the largest of four probabilities is at least 0.25 and q is above 0.75.
- **Matched volume.** The entropy threshold comes from the conformal flag rate measured on calibration, and both thresholds then apply to test.
- **Feature scaling.** Every questionnaire item or summed score is divided by its maximum possible score, a fixed property of the instrument. The transformation therefore uses no patient data, and the exchangeability assumption behind the conformal guarantee holds.
- **Probabilities.** Each model is evaluated as fitted, and ECE, Brier score and NLL measure the quality of its probabilities.
- **Random-review baseline.** Every error-recall value sits beside the random queue of the same size, since recall rises with the flag rate.
- **Review budgets.** Fixed review rates of 5, 10, 20, 30 and 40 percent run alongside the alpha sweep, since alpha 0.05 flags more patients than a service can review.
- **Intervals.** Patient-level bootstrap with 2,000 resamples, paired where two quantities are compared on the same patients.
- **Patient records.** One record per patient, the most recent referral.

## Pinned packages and console warnings

Two scikit-learn parameters in the code are deprecated. `environment.yml` pins scikit-learn below 1.10, so both behave exactly as they did for the reported models, and `pipeline/quiet.py` silences the two notices by name.

| Where | Parameter | Deprecated | Removed | Replacement |
| --- | --- | --- | --- | --- |
| `models/ml_LogReg.py` | `LogisticRegression(penalty='l2')` | 1.8 | 1.10 | `l1_ratio=0` |
| `models/ml_SVM.py` | `SVC(probability=True)` | 1.9 | 1.11 | `CalibratedClassifierCV(SVC(), ensemble=False)` |

The logistic regression replacement renames the same penalty, so `l1_ratio=0` with the same `C` and solver optimises an identical objective and gives the same coefficients. The SVM replacement changes the method. `SVC(probability=True)` fits Platt scaling inside the estimator, and `CalibratedClassifierCV` calibrates from cross-validated predictions. Every SVM probability would shift, and with it the entropy values, the conformal sets and the concordance results.

TensorFlow messages beginning `operand #N does not dominate this use` come from an optional MLIR graph optimisation. The pass fails, TensorFlow reports `Ignoring multi-device function optimization failure` and falls back to the unoptimised path, and the fitted network and its predictions are unaffected.

## Reproducibility

- Seed 42 drives the data split and the internal randomness of every model. The bootstrap uses its own fixed seed, 2026, so every interval repeats exactly.
- Every run writes `_manifest.json` with the full configuration, SHA-256 prefixes of each input file, class counts per split, the scaling record, the preflight result, per-model timings and array shapes, and the version of every package that affects a result.
- The PyMC NUTS sampler and GPU-backed dropout are seeded. Both can still differ at bit level across machines.

## Data availability

The referral data are clinical records from Internetpsykiatrien, Centre for Digital Psychiatry, Region of Southern Denmark, held under Danish data protection law, and the repository holds code only. The access route is in the Data Availability statement of the manuscript. The pipeline runs on any dataset with training, calibration and test splits and a one-indexed multiclass target: `config.REPRESENTATIONS` names the files.

## License

MIT, see `LICENSE`.
