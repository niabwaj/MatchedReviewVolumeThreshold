"""
ml_LogReg.py: bootstrap logistic regression ensemble.

The ensemble holds M multinomial logistic regressions (L2 penalty, lbfgs solver).
Each member is fitted on its own stratified bootstrap resample of the training
set, with the regularisation strength C tuned by inner cross-validation on that
resample. The M members give M predicted probability vectors per patient, and
their spread is the model's uncertainty.

Saved files:
    model_dir/bootstrap_ensemble_seed{seed}_{version}.pkl   the M fitted models
    model_dir/bootstrap_metadata_seed{seed}_{version}.pkl   the metadata dict
    results_dir/results_bootstrap_{version}.pkl             the results dict
"""

# IMPORTS ================================================================
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score

# HELPERS ================================================================
def _predict_ensemble(models, x_data):
    """
    Predicts class probabilities for every patient with every ensemble member.

    Input: models, a list of fitted classifiers trained on zero-indexed labels
    [0, 1, 2, 3]; x_data, a DataFrame or array of features.
    Output: dict with all_probs (M, n, 4), mean_probs (n, 4) and y_pred (n,),
    the argmax of mean_probs. Raises ValueError when the class order is not [0, 1, 2, 3].
    """

    # Features as a float array, and the ensemble dimensions
    X = x_data.astype(float).to_numpy() if hasattr(x_data, 'to_numpy') else np.array(x_data) # feature data
    n_models = len(models) # number of bootstrap sample models ----> M
    n_patients = len(X)  # number of patients in dataset ----> n
    
    expected_classes = np.array([0, 1, 2, 3])  # class order of the first member, checked against [0, 1, 2, 3]
    if not np.array_equal(models[0].classes_, expected_classes):
        raise ValueError(
            f"Unexpected model classes: {models[0].classes_}. "
            "Expected [0,1,2,3]. Ensure labels were converted to 0-indexed before training."
        )
    n_classes = len(models[0].classes_) # number of target classes ----> k = 4
    
    all_probs = np.zeros((n_models, n_patients, n_classes)) # initialize all_probs array
    for i, model in enumerate(models):
        all_probs[i] = model.predict_proba(X)  # class probabilities from member i
    mean_probs = all_probs.mean(axis=0) # average over M samples for each patient, per class
    y_pred     = mean_probs.argmax(axis=1) # store the class index of the maximum probability class from mean_probs
    return {
        'all_probs':  all_probs,
        'mean_probs': mean_probs,
        'y_pred':     y_pred,
    }
        
def _tune_c(x_tr, y_tr, c_grid, inner_folds, data_seed):
    """
    Selects C by grid search with stratified k-fold cross-validation on balanced accuracy.

    Input: x_tr, y_tr, the data to tune on; c_grid, the candidate C values;
    inner_folds, the number of folds; data_seed, the random seed.
    Output: the best C and its mean cross-validated balanced accuracy.
    """
    base_model = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        max_iter=1000,
        random_state=data_seed
    )
    gs = GridSearchCV(
        base_model,
        {'C': c_grid},
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=data_seed),
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    gs.fit(x_tr, y_tr)
    return gs.best_params_['C'], gs.best_score_

# MAIN ENSEMBLE FN ================================================================
def _run_main_ensemble(x_train, y_train_idx, m_samples, c_grid, inner_folds, data_seed):
    """
    Fits the M bootstrap logistic regression members.

    Each member receives a stratified bootstrap resample of the training set and its
    own C, tuned on that resample. Per-member tuning varies the regularisation across
    members, which widens the spread of the M predictions.

    Input: x_train, the training features (n_train, n_features); y_train_idx, the
    zero-indexed labels; m_samples, the number of members M; c_grid, the candidate C
    values; inner_folds, the number of inner folds; data_seed, the base seed (member
    i uses data_seed + i).
    Output: the list of fitted models and the list of selected C values.
    """
    print(f"\n  Main bootstrap ensemble: {m_samples} models with per-bootstrap C tuning...")
    models   = []
    c_values = []

    for i in range(m_samples):
        if (i + 1) % 50 == 0:
            print(f"    Bootstrap model {i+1}/{m_samples}...")

        # Stratified bootstrap resample
        x_boot, y_boot = resample(
            x_train, y_train_idx,
            n_samples=len(x_train),
            random_state=data_seed + i,
            stratify=y_train_idx
        )

        # Tune C on this bootstrap sample
        boot_c, _ = _tune_c(x_boot, y_boot, c_grid, inner_folds, data_seed + i)
        c_values.append(boot_c)

        # Train on this resample with its optimal C
        model = LogisticRegression(
            solver='lbfgs', penalty='l2', C=boot_c,
            max_iter=1000, random_state=data_seed + i
        )
        model.fit(x_boot, y_boot)
        models.append(model)

    return models, c_values

# MAIN CALL ================================================================

def _run_bootstrap(x_train, y_train, x_cal, y_cal, x_test, y_test, config):
    """
    Trains the bootstrap ensemble and predicts the training, calibration and test splits.

    Input: x_train, y_train, x_cal, y_cal, x_test, y_test, the three splits with
    one-indexed labels 1 to 4; config, a dict with data_seed, version, model_dir,
    results_dir and m_samples, and optionally c_grid and inner_folds.
    Output: the standard results dict with all_probs, mean_probs and y_pred for each
    split, feature_names, model_path and metadata. The dict, the models and the
    metadata are also saved as pickles.
    """
    print("\n" + "="*70)
    print("BOOTSTRAP LOGISTIC REGRESSION ENSEMBLE")
    print("="*70)

    # LOAD AND PRINT INPUT CONFIGURATIONS --------------------------------
    data_seed    = config['data_seed']
    version      = config['version']
    model_dir    = Path(config['model_dir'])
    results_dir  = Path(config['results_dir'])
    m_samples = config['m_samples']
    c_grid       = config.get('c_grid', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    inner_folds  = config.get('inner_folds', 5)

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # -- Convert features to numpy --------------------------------------
    X_train = x_train.astype(float).to_numpy()
    X_cal   = x_cal.astype(float).to_numpy()
    X_test  = x_test.astype(float).to_numpy()

    # -- Convert labels to 0-indexed --------------------------------------
    y_tr  = np.ravel((y_train.values - 1).astype(int) if hasattr(y_train, 'values') else (y_train - 1).astype(int))
    y_ca  = np.ravel((y_cal.values - 1).astype(int)   if hasattr(y_cal, 'values')   else (y_cal - 1).astype(int))
    y_te  = np.ravel((y_test.values - 1).astype(int)  if hasattr(y_test, 'values')  else (y_test - 1).astype(int))

    K = len(np.unique(y_tr))
    
    print(f"  Training patients:    {len(X_train)}")
    print(f"  Calibration patients: {len(X_cal)}")
    print(f"  Test patients:        {len(X_test)}")
    print(f"  Features:             {X_train.shape[1]}")
    print(f"  Classes:              {K}")
    print(f"  Ensemble size:        {m_samples}")
    print(f"  C grid:               {c_grid}")
    print(f"  Inner folds:          {inner_folds}")

    # -- Main bootstrap ensemble --------------------------------------
    print(f"\n{'='*70}")
    print("Main bootstrap ensemble (per-bootstrap C tuning)")
    main_models, c_values = _run_main_ensemble(
        X_train, y_tr, m_samples, c_grid, inner_folds, data_seed
    )
    # Save main ensemble
    main_path = model_dir / f"bootstrap_ensemble_seed{data_seed}_{version}.pkl"
    with open(main_path, 'wb') as f:
        pickle.dump(main_models, f)
    print(f"  -> Main ensemble saved: {main_path}")

    # -- Predict all three splits --------------------------------------
    print(f"\n{'='*70}")
    print("Generating predictions for all splits")
    print("  Predicting: training set")
    pred_train = _predict_ensemble(main_models, X_train)
    bal_acc_tr   = balanced_accuracy_score(y_tr, pred_train['y_pred'])
    print(f"Balanced accuracy on training set = {bal_acc_tr:.4f}")

    print("  Predicting: calibration set")
    pred_cal   = _predict_ensemble(main_models, X_cal)
    bal_acc_cal  = balanced_accuracy_score(y_ca, pred_cal['y_pred'])
    print(f"Balanced accuracy on calibration set = {bal_acc_cal:.4f}")

    print("  Predicting: test set")
    pred_test  = _predict_ensemble(main_models, X_test)
    bal_acc_test = balanced_accuracy_score(y_te, pred_test['y_pred'])
    print(f"Balanced accuracy on test set = {bal_acc_test:.4f}")

    # -- Assemble metadata --------------------------------------
    metadata = {
        'data_seed':          data_seed,
        'm_samples':          m_samples,
        'n_classes':          K,
        'n_train':            len(X_train),
        'n_cal':              len(X_cal),
        'n_test':             len(X_test),
        'base_model':         'LogisticRegression(l2, lbfgs)',
        'feature_names':      x_train.columns.tolist(),
        'c_grid':             c_grid,
        'inner_folds':        inner_folds,
        'bootstrap_c_values': c_values,
        'bootstrap_c_mean':   float(np.mean(c_values)),
        'bootstrap_c_std':    float(np.std(c_values)),
        'bal_acc_tr':         float(bal_acc_tr),
        'bal_acc_cal':        float(bal_acc_cal),
        'bal_acc_test':        float(bal_acc_test),
    }

    meta_path = model_dir / f"bootstrap_metadata_seed{data_seed}_{version}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    # -- Assemble standard results dict --------------------------------------
    results_bootstrap = {
        'model_name':        'bootstrap',

        'all_probs_train':   pred_train['all_probs'],   # (N, n_train, 4)
        'all_probs_cal':     pred_cal['all_probs'],      # (N, n_cal,   4)
        'all_probs_test':    pred_test['all_probs'],     # (N, n_test,  4)

        'mean_probs_train':  pred_train['mean_probs'],   # (n_train, 4)
        'mean_probs_cal':    pred_cal['mean_probs'],     # (n_cal,   4)
        'mean_probs_test':   pred_test['mean_probs'],    # (n_test,  4)

        'y_pred_train':      pred_train['y_pred'],       # (n_train,) 0-indexed
        'y_pred_cal':        pred_cal['y_pred'],         # (n_cal,)
        'y_pred_test':       pred_test['y_pred'],        # (n_test,)

        'feature_names':     x_train.columns.tolist(),
        'model_path':        str(main_path),

        'metadata':          metadata,
    }

    # Save results
    save_path = results_dir / f'results_bootstrap_{version}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_bootstrap, f)

    print(f"\n  Results saved: {save_path}")
    print("\n" + "="*70)
    print("BOOTSTRAP TRAINING COMPLETE")
    print(f"  Training bal. accuracy:     {bal_acc_tr:.4f}")
    print(f"  Calibration bal. accuracy:  {bal_acc_cal:.4f}")
    print(f"  Test bal. accuracy:         {bal_acc_test:.4f}")
    print(f"  all_probs_train shape:      {results_bootstrap['all_probs_train'].shape}")
    print(f"  all_probs_cal shape:        {results_bootstrap['all_probs_cal'].shape}")
    print(f"  all_probs_test shape:       {results_bootstrap['all_probs_test'].shape}")
    print("="*70)

    return results_bootstrap




