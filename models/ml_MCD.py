"""
ml_MCD.py: Monte Carlo dropout neural network with a two-branch input.

The network follows the two-branch input structure of the PrISM model (Kelly et
al., 2025), built with the Keras functional API. The questionnaire features pass
through dropout and a Dense(ReLU, L1) layer, and every hidden unit draws on every
questionnaire feature. The demographic features skip the hidden layer and join its
output by concatenation. A second dropout layer precedes the softmax output layer
over the four treatment classes. Stratified k-fold cross-validation selects the
hidden units, the dropout rate and the L1 strength.

At inference, M forward passes run with training=True, so each pass applies a new
dropout mask and returns a different probability vector. Dropout before every
weight layer is the condition under which Monte Carlo dropout approximates a
Bayesian posterior (Gal and Ghahramani, 2016). The spread of the M vectors is the
model's uncertainty.

Saved files:
    model_dir/mcd_model_seed{seed}_{version}.keras      the trained network
    model_dir/mcd_metadata_seed{seed}_{version}.pkl     the metadata dict
    results_dir/results_mcd_{version}.pkl               the results dict
"""

# IMPORTS ================================================================
import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import tensorflow as tf
    _gpus = tf.config.list_physical_devices('GPU')
    if _gpus:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


# _MHQ_PREFIXES: a column whose name starts with one of these prefixes joins the
# questionnaire branch, and every other column is demographic. The list covers the
# summed-score names (phq, gad, sias, pdss, fq_*) and the item names (Phq*, Gad*,
# Sias*, Pdss*, Fq*), so both representations group correctly.
_MHQ_PREFIXES = (
    'phq', 'gad', 'sias', 'pdss',
    'fq_',
    'Phq', 'Gad', 'Sias', 'Pdss', 'Fq',
    'mansa', 'Mansa',
)


# HELPERS ================================================================

def _identify_feature_groups(feature_names: list):
    """
    Splits the feature names into questionnaire and demographic groups by name prefix.

    Input: feature_names, the list of column names.
    Output: mhq_idx, demo_idx, the column positions of each group, and mhq_names,
    demo_names, the matching names.
    """
    mhq_idx    = [i for i, n in enumerate(feature_names)
                  if any(n.startswith(p) for p in _MHQ_PREFIXES)]
    demo_idx   = [i for i in range(len(feature_names)) if i not in set(mhq_idx)]
    mhq_names  = [feature_names[i] for i in mhq_idx]
    demo_names = [feature_names[i] for i in demo_idx]
    return mhq_idx, demo_idx, mhq_names, demo_names


def _build_mcd_model(n_mhq, n_demo, n_classes, hidden_units, dropout_rate, l1_lambda, seed):
    """
    Builds and compiles the two-branch Monte Carlo dropout network.

        questionnaire input (n_mhq,) -> Dropout(p) -> Dense(h, ReLU, L1)
        demographic input (n_demo,) ---------------------------------> Concatenate (h + n_demo)
                                                                      -> Dropout(p)
                                                                      -> Dense(n_classes, softmax, L1)

    L1 regularisation on both Dense layers, kernel and bias, favours sparse weights.
    The optimiser is Adam with learning rate 0.001 and sparse categorical
    cross-entropy loss.

    Input: n_mhq, n_demo, the number of questionnaire and demographic features;
    n_classes, the number of classes; hidden_units, the width h of the hidden layer;
    dropout_rate, the dropout probability p; l1_lambda, the L1 strength; seed, the
    seed for weight initialisation and dropout.
    Output: a compiled tf.keras.Model with two inputs.
    """
    tf.random.set_seed(seed)
    l1_reg = tf.keras.regularizers.l1(l1_lambda)
    init   = tf.keras.initializers.GlorotUniform(seed=seed)

    mhq_input  = tf.keras.Input(shape=(n_mhq,),  name='mhq_input')
    demo_input = tf.keras.Input(shape=(n_demo,), name='demo_input')

    # Dropout before the hidden weight layer. With the dropout before the output layer
    # below, every weight layer is preceded by dropout, and both dropout layers run with
    # training=True at inference.
    mhq_dropped = tf.keras.layers.Dropout(rate=dropout_rate, seed=seed, name='mc_dropout_in')(mhq_input)

    mhq_rep = tf.keras.layers.Dense(
        hidden_units, activation='relu',
        kernel_regularizer=l1_reg, bias_regularizer=l1_reg,
        kernel_initializer=init, name='mhq_block'
    )(mhq_dropped)

    combined = tf.keras.layers.Concatenate(name='combined')([mhq_rep, demo_input])

    # Dropout before the output weight layer.
    dropped = tf.keras.layers.Dropout(rate=dropout_rate, seed=seed, name='mc_dropout_out')(combined)

    out = tf.keras.layers.Dense(
        n_classes, activation='softmax',
        kernel_regularizer=l1_reg, bias_regularizer=l1_reg,
        kernel_initializer=init, name='output'
    )(dropped)

    model = tf.keras.Model(
        inputs=[mhq_input, demo_input], outputs=out,
        name=f'mcd_prism_h{hidden_units}_dp{int(dropout_rate * 10)}'
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _train_model(model, Xm, Xd, y, epochs, batch_size, patience,
                 val_data=None, val_split=0.2, verbose=0):
    """
    Fits the network with early stopping on validation loss, restoring the best weights.

    Input: model, the compiled network; Xm, Xd, the questionnaire and demographic
    arrays; y, the zero-indexed labels; epochs, the maximum number of epochs;
    batch_size; patience, the early-stopping patience; val_data, a held-out
    ([Xm, Xd], y) pair, or None to hold out val_split of the training data instead;
    verbose, the Keras verbosity.
    Output: the fitted model and its Keras History object.
    """
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
    )
    if val_data is not None:
        history = model.fit(
            [Xm, Xd], y, epochs=epochs, batch_size=batch_size,
            validation_data=val_data, callbacks=[es], verbose=verbose
        )
    else:
        history = model.fit(
            [Xm, Xd], y, epochs=epochs, batch_size=batch_size,
            validation_split=val_split, callbacks=[es], verbose=verbose
        )
    return model, history


def _cross_validate(Xm_tr, Xd_tr, y_tr, n_classes, config):
    """
    Selects hidden_units, dropout_rate and l1_lambda by stratified k-fold
    cross-validation on balanced accuracy.

    Every combination in the product of the three grids is trained on each fold and
    scored on the held-out fold.

    Input: Xm_tr, Xd_tr, the questionnaire and demographic training arrays; y_tr, the
    zero-indexed labels; n_classes, the number of classes; config, a dict with
    data_seed and optionally hidden_units_grid, dropout_grid, l1_grid, inner_folds,
    epochs, batch_size and patience.
    Output: best_params, a dict with hidden_units, dropout_rate and l1_lambda, and
    cv_results, one dict per combination sorted by mean balanced accuracy.
    """
    data_seed         = config['data_seed']
    hidden_units_grid = config.get('hidden_units_grid', [16, 32, 64])
    dropout_grid      = config.get('dropout_grid', [0.1, 0.2, 0.3, 0.4])
    l1_grid           = config.get('l1_grid', [0.001, 0.01])
    inner_folds       = config.get('inner_folds', 5)
    epochs            = config.get('epochs', 500)
    batch_size        = config.get('batch_size', 32)
    patience          = config.get('patience', 50)

    combos = list(itertools.product(hidden_units_grid, dropout_grid, l1_grid))
    n_mhq, n_demo = Xm_tr.shape[1], Xd_tr.shape[1]
    n_total = len(combos) * inner_folds

    print(f"\n  Grid CV: {len(combos)} combinations x {inner_folds} folds = {n_total} model fits")
    print(f"  hidden_units: {hidden_units_grid}  dropout: {dropout_grid}  l1: {l1_grid}")

    skf     = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=data_seed)
    results = []
    fit_idx = 0

    for h, dp, l1 in combos:
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(Xm_tr, y_tr)):
            fit_idx += 1
            if fit_idx % 20 == 0:
                print(f"    Fit {fit_idx}/{n_total}: h={h}, dp={dp}, l1={l1} ...")

            Xm_f_tr, Xm_f_va = Xm_tr[tr_idx], Xm_tr[va_idx]
            Xd_f_tr, Xd_f_va = Xd_tr[tr_idx], Xd_tr[va_idx]
            y_f_tr,  y_f_va  = y_tr[tr_idx],  y_tr[va_idx]

            fold_model = _build_mcd_model(n_mhq, n_demo, n_classes, h, dp, l1, seed=data_seed + fold_i)
            val_data = ([Xm_f_va, Xd_f_va], y_f_va)
            fold_model, _ = _train_model(
                fold_model, Xm_f_tr, Xd_f_tr, y_f_tr,
                epochs, batch_size, patience, val_data=val_data
            )
            val_probs = fold_model.predict([Xm_f_va, Xd_f_va], verbose=0)
            fold_scores.append(balanced_accuracy_score(y_f_va, val_probs.argmax(axis=1)))

        mean_acc = float(np.mean(fold_scores))
        std_acc  = float(np.std(fold_scores))
        results.append({
            'hidden_units': h, 'dropout_rate': dp, 'l1_lambda': l1,
            'mean_bal_acc': mean_acc, 'std_bal_acc': std_acc, 'fold_scores': fold_scores,
        })
        print(f"    h={h}, dp={dp}, l1={l1}: bal_acc={mean_acc:.4f} (+/-{std_acc:.4f})")

    results.sort(key=lambda r: r['mean_bal_acc'], reverse=True)
    best = results[0]
    print(f"\n  Best: hidden={best['hidden_units']}, dropout={best['dropout_rate']}, "
          f"l1={best['l1_lambda']}  (CV bal_acc={best['mean_bal_acc']:.4f})")

    return {k: best[k] for k in ('hidden_units', 'dropout_rate', 'l1_lambda')}, results


def _predict_mcd(model, Xm, Xd, m_samples, data_seed):
    """
    Predicts class probabilities with M Monte Carlo forward passes.

    Each pass runs with training=True, so the dropout layers apply a new random mask
    and return a different probability vector per patient.

    Input: model, the fitted two-input network; Xm, Xd, the questionnaire and
    demographic arrays (float32); m_samples, the number of passes M; data_seed, the
    seed for the dropout mask sequence.
    Output: dict with all_probs (M, n, 4), mean_probs (n, 4) and y_pred (n,).
    """
    n_patients = len(Xm)
    n_classes  = model.output_shape[-1]

    tf.random.set_seed(data_seed)
    all_probs = np.zeros((m_samples, n_patients, n_classes), dtype=np.float32)

    for i in range(m_samples):
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{m_samples} forward passes...")
        all_probs[i] = model([Xm, Xd], training=True).numpy()

    mean_probs = all_probs.mean(axis=0)
    y_pred     = mean_probs.argmax(axis=1)

    return {'all_probs': all_probs, 'mean_probs': mean_probs, 'y_pred': y_pred}


# MAIN CALL ================================================================

def _run_mcd(x_train, y_train, x_cal, y_cal, x_test, y_test, config):
    """
    Trains the Monte Carlo dropout network and predicts the training, calibration and
    test splits.

    The steps are: group the features by name, select the hyperparameters by
    cross-validation, train the final network on the full training set, then run M
    forward passes on each split.

    Input: x_train, y_train, x_cal, y_cal, x_test, y_test, the three splits with
    one-indexed labels 1 to 4; config, a dict with data_seed, version, model_dir,
    results_dir, m_samples (falling back to n_estimators, then n_fwd_passes, then
    200), and optionally hidden_units_grid, dropout_grid, l1_grid, inner_folds,
    epochs, batch_size, val_split and patience.
    Output: the standard results dict with all_probs, mean_probs and y_pred for each
    split, feature_names, model_path and metadata. The dict, the network and the
    metadata are also saved.
    """
    print("\n" + "=" * 70)
    print("MONTE CARLO DROPOUT - PRISM-INSPIRED 2-BRANCH ARCHITECTURE")
    print("=" * 70)

    data_seed   = config['data_seed']
    version     = config['version']
    model_dir   = Path(config['model_dir'])
    results_dir = Path(config['results_dir'])
    m_samples   = config.get('m_samples', config.get('n_estimators', config.get('n_fwd_passes', 200)))

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(data_seed)
    tf.random.set_seed(data_seed)

    # zero-indexed labels (input labels are 1 to 4)
    def _to_idx(y):
        arr = y.values.ravel() if hasattr(y, 'values') else np.ravel(y)
        return (arr - 1).astype(np.int32)

    y_tr = _to_idx(y_train)
    y_ca = _to_idx(y_cal)
    y_te = _to_idx(y_test)
    K    = len(np.unique(y_tr))

    # float32 feature arrays, the TensorFlow default dtype
    X_train = x_train.astype(float).to_numpy().astype(np.float32)
    X_cal   = x_cal.astype(float).to_numpy().astype(np.float32)
    X_test  = x_test.astype(float).to_numpy().astype(np.float32)

    # questionnaire branch and demographic skip connection
    feature_names                            = x_train.columns.tolist()
    mhq_idx, demo_idx, mhq_names, demo_names = _identify_feature_groups(feature_names)
    n_mhq, n_demo                            = len(mhq_idx), len(demo_idx)

    print(f"  MHQ features  ({n_mhq}): {mhq_names}")
    print(f"  Demo features ({n_demo}): {demo_names}")
    print(f"  Training patients:    {len(X_train)}")
    print(f"  Calibration patients: {len(X_cal)}")
    print(f"  Test patients:        {len(X_test)}")
    print(f"  Forward passes (M):   {m_samples}")
    print(f"  batch_size:           {config.get('batch_size', 32)}  patience: {config.get('patience', 50)}")

    # questionnaire and demographic arrays for all three splits
    mhq_idx_arr  = np.array(mhq_idx)
    demo_idx_arr = np.array(demo_idx)

    Xm_tr, Xd_tr = X_train[:, mhq_idx_arr], X_train[:, demo_idx_arr]
    Xm_ca, Xd_ca = X_cal[:,   mhq_idx_arr], X_cal[:,   demo_idx_arr]
    Xm_te, Xd_te = X_test[:,  mhq_idx_arr], X_test[:,  demo_idx_arr]

    # -- Grid CV for best hyperparameters --
    print(f"\n{'=' * 70}")
    print("STEP 1: Grid cross-validation (hidden_units x dropout x l1)")
    best_params, cv_results = _cross_validate(Xm_tr, Xd_tr, y_tr, K, config)
    best_h  = best_params['hidden_units']
    best_dp = best_params['dropout_rate']
    best_l1 = best_params['l1_lambda']

    # -- Train the final MCD model --
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Training final MCD model (h={best_h}, dp={best_dp}, l1={best_l1})")
    epochs     = config.get('epochs', 500)
    batch_size = config.get('batch_size', 32)
    val_split  = config.get('val_split', 0.2)
    patience   = config.get('patience', 50)

    mcd_model = _build_mcd_model(n_mhq, n_demo, K, best_h, best_dp, best_l1, data_seed)
    mcd_model, mcd_hist = _train_model(
        mcd_model, Xm_tr, Xd_tr, y_tr, epochs, batch_size, patience, val_split=val_split
    )
    epochs_mcd = len(mcd_hist.history['loss'])
    print(f"  Epochs trained: {epochs_mcd}")

    mcd_path = model_dir / f"mcd_model_seed{data_seed}_{version}.keras"
    mcd_model.save(str(mcd_path))
    print(f"  MCD model saved: {mcd_path}")

    # -- MC forward passes for all three splits --
    print(f"\n{'=' * 70}")
    print(f"STEP 3: MC predictions ({m_samples} passes, dropout={best_dp} active)")
    print("  Predicting: training set")
    pred_train = _predict_mcd(mcd_model, Xm_tr, Xd_tr, m_samples, data_seed)
    print("  Predicting: calibration set")
    pred_cal   = _predict_mcd(mcd_model, Xm_ca, Xd_ca, m_samples, data_seed)
    print("  Predicting: test set")
    pred_test  = _predict_mcd(mcd_model, Xm_te, Xd_te, m_samples, data_seed)

    # -- Balanced accuracy per split --
    bal_acc_tr   = float(balanced_accuracy_score(y_tr, pred_train['y_pred']))
    bal_acc_cal  = float(balanced_accuracy_score(y_ca, pred_cal['y_pred']))
    bal_acc_test = float(balanced_accuracy_score(y_te, pred_test['y_pred']))

    # -- Assemble metadata --
    metadata = {
        'data_seed':         data_seed,
        'version':           version,
        'm_samples':         m_samples,
        'n_classes':         K,
        'n_train':           len(X_train),
        'n_cal':             len(X_cal),
        'n_test':            len(X_test),
        'feature_names':     feature_names,
        'mhq_features':      mhq_names,
        'demo_features':     demo_names,
        'n_mhq':             n_mhq,
        'n_demo':            n_demo,
        'best_hidden_units': best_h,
        'best_dropout':      best_dp,
        'best_l1':           best_l1,
        'cv_results':        cv_results,
        'epochs_mcd':        epochs_mcd,
        'batch_size':        batch_size,
        'patience':          patience,
        'architecture':      f'MHQ({n_mhq})->Dropout({best_dp})->Dense({best_h},relu,L1)'
                             f'->concat(demo={n_demo})->Dropout({best_dp})->Dense({K},softmax,L1)',
        'bal_acc_tr':        bal_acc_tr,
        'bal_acc_cal':       bal_acc_cal,
        'bal_acc_test':      bal_acc_test,
    }

    meta_path = model_dir / f"mcd_metadata_seed{data_seed}_{version}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    # -- Assemble standard results dict --
    results_mcd = {
        'model_name':       'mcd',
        'all_probs_train':  pred_train['all_probs'],   # (M, n_train, 4)
        'all_probs_cal':    pred_cal['all_probs'],      # (M, n_cal,   4)
        'all_probs_test':   pred_test['all_probs'],     # (M, n_test,  4)
        'mean_probs_train': pred_train['mean_probs'],   # (n_train, 4)
        'mean_probs_cal':   pred_cal['mean_probs'],     # (n_cal,   4)
        'mean_probs_test':  pred_test['mean_probs'],    # (n_test,  4)
        'y_pred_train':     pred_train['y_pred'],       # (n_train,) 0-indexed
        'y_pred_cal':       pred_cal['y_pred'],         # (n_cal,)
        'y_pred_test':      pred_test['y_pred'],        # (n_test,)
        'feature_names':    feature_names,
        'model_path':       str(mcd_path),
        'metadata':         metadata,
    }

    save_path = results_dir / f'results_mcd_{version}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_mcd, f)

    print(f"\n  Results saved: {save_path}")
    print("\n" + "=" * 70)
    print("MCD TRAINING COMPLETE")
    print(f"  Architecture:               {metadata['architecture']}")
    print(f"  Training bal. accuracy:     {bal_acc_tr:.4f}")
    print(f"  Calibration bal. accuracy:  {bal_acc_cal:.4f}")
    print(f"  Test bal. accuracy:         {bal_acc_test:.4f}")
    print(f"  all_probs_train shape:      {results_mcd['all_probs_train'].shape}")
    print(f"  all_probs_cal shape:        {results_mcd['all_probs_cal'].shape}")
    print(f"  all_probs_test shape:       {results_mcd['all_probs_test'].shape}")
    print("=" * 70)

    return results_mcd
