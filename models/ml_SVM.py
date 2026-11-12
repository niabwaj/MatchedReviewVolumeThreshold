"""
ml_SVM.py: bootstrap ensemble of RBF-kernel support vector machines.

The ensemble holds M SVC models with probability=True. Each member is fitted on
its own stratified bootstrap resample of the training set, with C tuned by inner
cross-validation on that resample. The kernel width gamma is fixed at 'scale'.
Platt scaling converts each member's margin scores into class probabilities over
the four treatment classes. The M members give M probability vectors per patient,
and their disagreement is the model's uncertainty.

Platt scaling runs an internal cross-validation inside every fit, so training time
grows quickly with M. A smaller m_samples in config shortens a test run.

Saved files:
    model_dir/svm_ensemble_seed{seed}_{version}.pkl   the M fitted models
    model_dir/svm_metadata_seed{seed}_{version}.pkl   the metadata dict
    results_dir/results_svm_{version}.pkl             the results dict
"""

# IMPORTS ================================================================
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score


# HELPERS ================================================================

def _predict_ensemble(models, x_data):
    """
    Predicts Platt-scaled class probabilities for every patient with every member.

    The class order is checked, so the probability columns line up with every other
    model's predictions.

    Input: models, a list of fitted SVC(probability=True) models trained on
    zero-indexed labels [0, 1, 2, 3]; x_data, a DataFrame or array of features.
    Output: dict with all_probs (M, n, 4), mean_probs (n, 4) and y_pred (n,).
    Raises ValueError when the class order is not [0, 1, 2, 3].
    """
    X = x_data.astype(float).to_numpy() if hasattr(x_data, 'to_numpy') else np.asarray(x_data, dtype=float)
    n_models   = len(models)
    n_patients = len(X)

    expected = np.array([0, 1, 2, 3])
    if not np.array_equal(models[0].classes_, expected):
        raise ValueError(
            f"Unexpected SVM classes: {models[0].classes_}. "
            "Expected [0,1,2,3]. Ensure 0-indexed labels were used during training."
        )
    n_classes = len(models[0].classes_)

    all_probs = np.zeros((n_models, n_patients, n_classes))
    for i, model in enumerate(models):
        all_probs[i] = model.predict_proba(X)   # Platt-scaled probabilities

    mean_probs = all_probs.mean(axis=0)
    y_pred     = mean_probs.argmax(axis=1)

    return {
        'all_probs':  all_probs,
        'mean_probs': mean_probs,
        'y_pred':     y_pred,
    }


def _tune_c_svm(x_tr, y_tr, c_grid, gamma, inner_folds, data_seed):
    """
    Selects C for an RBF SVM by grid search with stratified k-fold cross-validation on
    balanced accuracy.

    Input: x_tr, y_tr, the data to tune on; c_grid, the candidate C values; gamma,
    the fixed kernel width; inner_folds, the number of folds; data_seed, the seed.
    Output: the best C and its mean cross-validated balanced accuracy.
    """
    base = SVC(kernel='rbf', gamma=gamma, probability=True,
               random_state=data_seed, cache_size=500)
    gs = GridSearchCV(
        base,
        {'C': c_grid},
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=data_seed),
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    gs.fit(x_tr, y_tr)
    return gs.best_params_['C'], gs.best_score_


# MAIN ENSEMBLE FN ================================================================

def _run_main_ensemble(x_train, y_train_idx, m_samples, c_grid, gamma, inner_folds, data_seed):
    """
    Fits the M bootstrap SVM members.

    Each member receives a stratified bootstrap resample of the training set and its
    own C, tuned on that resample. gamma stays fixed for every member, so C alone
    varies the regularisation and the search stays tractable.

    Input: x_train, the training features (n_train, n_features); y_train_idx, the
    zero-indexed labels; m_samples, the number of members M; c_grid, the candidate C
    values; gamma, the fixed kernel width; inner_folds, the number of inner folds;
    data_seed, the base seed (member i uses data_seed + i).
    Output: the list of fitted models and the list of selected C values.
    """
    print(f"\n  Main SVM bootstrap ensemble: {m_samples} models, per-bootstrap C tuning...")
    models   = []
    c_values = []

    for i in range(m_samples):
        if (i + 1) % 25 == 0:
            print(f"    SVM model {i+1}/{m_samples}...")

        # Stratified bootstrap resample
        x_boot, y_boot = resample(
            x_train, y_train_idx,
            n_samples=len(x_train),
            random_state=data_seed + i,
            stratify=y_train_idx
        )

        # Tune C on this bootstrap sample
        boot_c, _ = _tune_c_svm(x_boot, y_boot, c_grid, gamma, inner_folds, data_seed + i)
        c_values.append(boot_c)

        # Train SVC on this resample with its optimal C
        svm = SVC(
            kernel='rbf', gamma=gamma, C=boot_c,
            probability=True, cache_size=500,
            random_state=data_seed + i
        )
        svm.fit(x_boot, y_boot)
        models.append(svm)

    return models, c_values


# MAIN CALL ================================================================

def _run_svm(x_train, y_train, x_cal, y_cal, x_test, y_test, config):
    """
    Trains the SVM ensemble and predicts the training, calibration and test splits.

    Input: x_train, y_train, x_cal, y_cal, x_test, y_test, the three splits with
    one-indexed labels 1 to 4; config, a dict with data_seed, version, model_dir,
    results_dir, m_samples (falling back to n_estimators), and optionally c_grid,
    gamma and inner_folds.
    Output: the standard results dict with all_probs, mean_probs and y_pred for each
    split, feature_names, model_path and metadata. The dict, the models and the
    metadata are also saved as pickles.
    """
    print("\n" + "="*70)
    print("BOOTSTRAP SVM (RBF KERNEL) ENSEMBLE")
    print("="*70)

    data_seed    = config['data_seed']
    version      = config['version']
    model_dir    = Path(config['model_dir'])
    results_dir  = Path(config['results_dir'])
    m_samples    = config.get('m_samples', config.get('n_estimators', 200))
    c_grid       = config.get('c_grid', [0.01, 0.1, 1.0, 10.0, 100.0])
    gamma        = config.get('gamma', 'scale')
    inner_folds  = config.get('inner_folds', 5)

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Convert features to numpy --
    X_train = x_train.astype(float).to_numpy()
    X_cal   = x_cal.astype(float).to_numpy()
    X_test  = x_test.astype(float).to_numpy()

    # -- Convert labels to 0-indexed --
    y_tr = np.ravel((y_train.values - 1).astype(int) if hasattr(y_train, 'values') else (y_train - 1).astype(int))
    y_ca = np.ravel((y_cal.values - 1).astype(int)   if hasattr(y_cal, 'values')   else (y_cal - 1).astype(int))
    y_te = np.ravel((y_test.values - 1).astype(int)  if hasattr(y_test, 'values')  else (y_test - 1).astype(int))

    K = len(np.unique(y_tr))

    print(f"  Training patients:    {len(X_train)}")
    print(f"  Calibration patients: {len(X_cal)}")
    print(f"  Test patients:        {len(X_test)}")
    print(f"  Features:             {X_train.shape[1]}")
    print(f"  Classes:              {K}")
    print(f"  Ensemble size:        {m_samples}")
    print(f"  Kernel:               RBF, gamma={gamma}")
    print(f"  C grid:               {c_grid}")
    print(f"  Inner folds:          {inner_folds}")

    # -- Main bootstrap ensemble --
    print(f"\n{'='*70}")
    print("STEP 1: Main bootstrap SVM ensemble (per-bootstrap C tuning)")
    main_models, c_values = _run_main_ensemble(
        X_train, y_tr, m_samples, c_grid, gamma, inner_folds, data_seed
    )

    main_path = model_dir / f"svm_ensemble_seed{data_seed}_{version}.pkl"
    with open(main_path, 'wb') as f:
        pickle.dump(main_models, f)
    print(f"  Main ensemble saved: {main_path}")

    # -- Predict all three splits --
    print(f"\n{'='*70}")
    print("STEP 2: Generating predictions for all splits")
    print("  Predicting: training set")
    pred_train = _predict_ensemble(main_models, x_train)
    print("  Predicting: calibration set")
    pred_cal   = _predict_ensemble(main_models, x_cal)
    print("  Predicting: test set")
    pred_test  = _predict_ensemble(main_models, x_test)

    # -- Balanced accuracy per split --
    bal_acc_tr   = balanced_accuracy_score(y_tr, pred_train['y_pred'])
    bal_acc_cal  = balanced_accuracy_score(y_ca, pred_cal['y_pred'])
    bal_acc_test = balanced_accuracy_score(y_te, pred_test['y_pred'])

    # -- Assemble metadata --
    metadata = {
        'data_seed':          data_seed,
        'version':            version,
        'm_samples':          m_samples,
        'n_classes':          K,
        'n_train':            len(X_train),
        'n_cal':              len(X_cal),
        'n_test':             len(X_test),
        'base_model':         f'SVC(rbf, gamma={gamma}, probability=True)',
        'feature_names':      x_train.columns.tolist(),
        'kernel':             'rbf',
        'gamma':              gamma,
        'c_grid':             c_grid,
        'inner_folds':        inner_folds,
        'bootstrap_c_values': c_values,
        'bootstrap_c_mean':   float(np.mean(c_values)),
        'bootstrap_c_std':    float(np.std(c_values)),
        'bal_acc_tr':         float(bal_acc_tr),
        'bal_acc_cal':        float(bal_acc_cal),
        'bal_acc_test':       float(bal_acc_test),
    }

    meta_path = model_dir / f"svm_metadata_seed{data_seed}_{version}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    # -- Assemble standard results dict --
    results_svm = {
        'model_name':       'svm',
        'all_probs_train':  pred_train['all_probs'],
        'all_probs_cal':    pred_cal['all_probs'],
        'all_probs_test':   pred_test['all_probs'],
        'mean_probs_train': pred_train['mean_probs'],
        'mean_probs_cal':   pred_cal['mean_probs'],
        'mean_probs_test':  pred_test['mean_probs'],
        'y_pred_train':     pred_train['y_pred'],
        'y_pred_cal':       pred_cal['y_pred'],
        'y_pred_test':      pred_test['y_pred'],
        'feature_names':    x_train.columns.tolist(),
        'model_path':       str(main_path),
        'metadata':         metadata,
    }

    save_path = results_dir / f'results_svm_{version}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_svm, f)

    print(f"\n  Results saved: {save_path}")
    print("\n" + "="*70)
    print("SVM BOOTSTRAP TRAINING COMPLETE")
    print(f"  Training bal. accuracy:     {bal_acc_tr:.4f}")
    print(f"  Calibration bal. accuracy:  {bal_acc_cal:.4f}")
    print(f"  Test bal. accuracy:         {bal_acc_test:.4f}")
    print(f"  C range: [{min(c_values):.4f}, {max(c_values):.4f}]  "
          f"mean={np.mean(c_values):.4f}  std={np.std(c_values):.4f}")
    print(f"  all_probs_train shape:      {results_svm['all_probs_train'].shape}")
    print(f"  all_probs_cal shape:        {results_svm['all_probs_cal'].shape}")
    print(f"  all_probs_test shape:       {results_svm['all_probs_test'].shape}")
    print("="*70)

    return results_svm
