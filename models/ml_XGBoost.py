"""
ml_XGBoost.py: stochastic ensemble of gradient-boosted trees (XGBClassifier).

max_depth and learning_rate are tuned once on the full training set with
subsampling switched off, so the tuning scores are reproducible. The M members
then train on the full training set, each with its own random seed. The seed drives
row subsampling (subsample) and column subsampling (colsample_bytree) inside every
tree, so the members differ through subsampling while the data stay fixed. The M
members give M probability vectors per patient, and their disagreement is the
model's uncertainty.

Saved files:
    model_dir/xgb_ensemble_seed{seed}_{version}.pkl   the M fitted models
    model_dir/xgb_metadata_seed{seed}_{version}.pkl   the metadata dict
    results_dir/results_xgboost_{version}.pkl         the results dict
"""

# IMPORTS ================================================================
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


# HELPERS ================================================================

def _predict_ensemble(models, x_data):
    """
    Predicts class probabilities for every patient with every ensemble member.

    Members trained on zero-indexed labels [0, 1, 2, 3] return the probability columns
    in class order.

    Input: models, a list of fitted XGBClassifier models; x_data, a DataFrame or
    array of features.
    Output: dict with all_probs (M, n, 4), mean_probs (n, 4) and y_pred (n,).
    """
    X = x_data.astype(float).to_numpy() if hasattr(x_data, 'to_numpy') else np.asarray(x_data, dtype=float)
    n_models   = len(models)
    n_patients = len(X)
    n_classes  = models[0].n_classes_

    all_probs = np.zeros((n_models, n_patients, n_classes))
    for i, model in enumerate(models):
        all_probs[i] = model.predict_proba(X)

    mean_probs = all_probs.mean(axis=0)
    y_pred     = mean_probs.argmax(axis=1)

    return {
        'all_probs':  all_probs,
        'mean_probs': mean_probs,
        'y_pred':     y_pred,
    }


def _tune_xgb(x_tr, y_tr, n_trees, max_depth_grid, lr_grid, inner_folds, data_seed):
    """
    Selects max_depth and learning_rate by grid search with stratified k-fold
    cross-validation on balanced accuracy.

    Subsampling is off during tuning (subsample=1.0, colsample_bytree=1.0), so the
    scores depend on max_depth and learning_rate alone and repeat exactly.

    Input: x_tr, y_tr, the data to tune on; n_trees, the boosting rounds;
    max_depth_grid, lr_grid, the candidate values; inner_folds, the number of folds;
    data_seed, the seed.
    Output: the best max_depth, the best learning rate and the best mean
    cross-validated balanced accuracy.
    """
    param_grid = {'max_depth': max_depth_grid, 'learning_rate': lr_grid}

    base = XGBClassifier(
        n_estimators=n_trees,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=data_seed
    )
    gs = GridSearchCV(
        base,
        param_grid,
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=data_seed),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )
    gs.fit(x_tr, y_tr)
    best = gs.best_params_
    return best['max_depth'], best['learning_rate'], gs.best_score_


# MAIN ENSEMBLE FN ================================================================

def _run_main_ensemble(x_train, y_train_idx, m_samples, n_trees, max_depth, lr,
                       subsample, colsample_bytree, data_seed):
    """
    Fits the M XGBoost members on the full training set.

    Each member uses its own random_state (data_seed + i), which changes the row and
    column masks drawn inside every tree.

    Input: x_train, y_train_idx, the training features and zero-indexed labels;
    m_samples, the number of members M; n_trees, the boosting rounds per member;
    max_depth, lr, the tuned values; subsample, the row fraction per tree;
    colsample_bytree, the column fraction per tree; data_seed, the base seed.
    Output: the list of fitted models.
    """
    print(f"\n  Main XGBoost ensemble: {m_samples} models with subsampling stochasticity")
    print(f"    (subsample={subsample}, colsample_bytree={colsample_bytree}, varied random_state)")
    models = []

    for i in range(m_samples):
        if (i + 1) % 50 == 0:
            print(f"    XGBoost model {i+1}/{m_samples}...")

        model = XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='mlogloss',
            verbosity=0,
            random_state=data_seed + i  # varied per member, so each member draws different masks
        )
        model.fit(x_train, y_train_idx)
        models.append(model)

    return models


# MAIN CALL ================================================================

def _run_xgboost(x_train, y_train, x_cal, y_cal, x_test, y_test, config):
    """
    Trains the XGBoost ensemble and predicts the training, calibration and test splits.

    Input: x_train, y_train, x_cal, y_cal, x_test, y_test, the three splits with
    one-indexed labels 1 to 4; config, a dict with data_seed, version, model_dir,
    results_dir, m_samples (falling back to n_estimators), and optionally n_trees,
    max_depth_grid, lr_grid, subsample, colsample_bytree and inner_folds.
    Output: the standard results dict with all_probs, mean_probs and y_pred for each
    split, feature_names, model_path and metadata. The dict, the models and the
    metadata are also saved as pickles.
    """
    print("\n" + "="*70)
    print("XGBOOST STOCHASTIC ENSEMBLE")
    print("="*70)

    data_seed        = config['data_seed']
    version          = config['version']
    model_dir        = Path(config['model_dir'])
    results_dir      = Path(config['results_dir'])
    m_samples        = config.get('m_samples', config.get('n_estimators', 200))
    n_trees          = config.get('n_trees', 300)
    max_depth_grid   = config.get('max_depth_grid', [3, 4, 6])
    lr_grid          = config.get('lr_grid', [0.05, 0.1, 0.2])
    subsample        = config.get('subsample', 0.8)
    colsample_bytree = config.get('colsample_bytree', 0.8)
    inner_folds      = config.get('inner_folds', 5)

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
    print(f"  Trees per model:      {n_trees}")
    print(f"  subsample:            {subsample}  colsample_bytree: {colsample_bytree}")
    print(f"  max_depth grid:       {max_depth_grid}")
    print(f"  lr grid:              {lr_grid}")
    print(f"  Inner folds:          {inner_folds}")

    # -- Tune max_depth and learning_rate (subsampling off) --
    print(f"\n{'='*70}")
    print("STEP 1: Tuning hyperparameters (max_depth, learning_rate)")
    best_depth, best_lr, cv_acc = _tune_xgb(
        X_train, y_tr, n_trees, max_depth_grid, lr_grid, inner_folds, data_seed
    )
    print(f"  Best max_depth={best_depth}, learning_rate={best_lr}  CV balanced accuracy={cv_acc:.4f}")

    # -- Main stochastic ensemble --
    print(f"\n{'='*70}")
    print("STEP 2: Main XGBoost stochastic ensemble")
    main_models = _run_main_ensemble(
        X_train, y_tr, m_samples, n_trees,
        best_depth, best_lr, subsample, colsample_bytree, data_seed
    )

    main_path = model_dir / f"xgb_ensemble_seed{data_seed}_{version}.pkl"
    with open(main_path, 'wb') as f:
        pickle.dump(main_models, f)
    print(f"  Main ensemble saved: {main_path}")

    # -- Predict all three splits --
    print(f"\n{'='*70}")
    print("STEP 3: Generating predictions for all splits")
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
        'data_seed':        data_seed,
        'version':          version,
        'm_samples':        m_samples,
        'n_classes':        K,
        'n_train':          len(X_train),
        'n_cal':            len(X_cal),
        'n_test':           len(X_test),
        'base_model':       'XGBClassifier',
        'feature_names':    x_train.columns.tolist(),
        'n_trees':          n_trees,
        'max_depth':        best_depth,
        'learning_rate':    best_lr,
        'subsample':        subsample,
        'colsample_bytree': colsample_bytree,
        'inner_folds':      inner_folds,
        'cv_acc':           float(cv_acc),
        'bal_acc_tr':       float(bal_acc_tr),
        'bal_acc_cal':      float(bal_acc_cal),
        'bal_acc_test':     float(bal_acc_test),
    }

    meta_path = model_dir / f"xgb_metadata_seed{data_seed}_{version}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    # -- Assemble standard results dict --
    results_xgboost = {
        'model_name':       'xgboost',
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

    save_path = results_dir / f'results_xgboost_{version}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_xgboost, f)

    print(f"\n  Results saved: {save_path}")
    print("\n" + "="*70)
    print("XGBOOST ENSEMBLE TRAINING COMPLETE")
    print(f"  max_depth={best_depth}, lr={best_lr}, subsample={subsample}, colsample={colsample_bytree}")
    print(f"  Training bal. accuracy:     {bal_acc_tr:.4f}")
    print(f"  Calibration bal. accuracy:  {bal_acc_cal:.4f}")
    print(f"  Test bal. accuracy:         {bal_acc_test:.4f}")
    print(f"  all_probs_train shape:      {results_xgboost['all_probs_train'].shape}")
    print(f"  all_probs_cal shape:        {results_xgboost['all_probs_cal'].shape}")
    print(f"  all_probs_test shape:       {results_xgboost['all_probs_test'].shape}")
    print("="*70)

    return results_xgboost
