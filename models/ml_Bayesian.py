"""
ml_Bayesian.py: Bayesian multinomial logistic regression in PyMC.

The model places weakly informative Gaussian priors on the intercepts,
Normal(0, 2.0), and on the coefficients, Normal(0, 1.5), with a categorical
softmax likelihood over the four treatment classes. NUTS (the No-U-Turn Sampler)
samples the posterior. Each of M posterior draws passes through the softmax to
give one predicted probability vector per patient, and the mean of the M vectors
is the mean predictive distribution. Where the data constrain the weights, the
draws agree and the mean is peaked; where the data are thin, the draws disagree
and the mean is flatter.

Saved files:
    model_dir/bayesian_trace_seed{seed}_{version}.nc        the posterior trace (NetCDF)
    model_dir/bayesian_metadata_seed{seed}_{version}.pkl    the metadata dict
    results_dir/results_bayesian_{version}.pkl              the results dict
"""

# IMPORTS ================================================================
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score

# PyMC and ArviZ are imported inside _train_pymc. A module-level pymc import raises
# a pytensor circular-import error when numpy, pandas and matplotlib are already
# loaded in the same kernel.


# HELPERS ================================================================

def _train_pymc(x_train, y_train, x_cal, y_cal, config):
    """
    Builds the PyMC model, samples the posterior with NUTS and saves the trace.

    Input: x_train, the training features; y_train, one-indexed labels 1 to 4,
    converted to zero-indexed internally; x_cal, y_cal, the calibration split (its
    size goes into the metadata); config, a dict with draws, tune, chains,
    target_accept, data_seed, version, model_dir and optionally progressbar.
    Output: dict with trace, model, metadata, trace_path and meta_path.
    """
    import pymc as pm  # imported here to avoid the circular import noted above
    import arviz as az  # provides the NetCDF backend for trace.to_netcdf

    draws      = config['draws']
    tune       = config['tune']
    chains     = config['chains']
    target_acc = config['target_accept']
    data_seed  = config['data_seed']
    version    = config['version']
    model_dir  = Path(config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert features to numpy, labels to 0-indexed
    X_train = x_train.astype(float).to_numpy()
    y_tr    = np.ravel((y_train.values - 1).astype(int)
                       if hasattr(y_train, 'values') else (y_train - 1).astype(int))

    K          = int(y_tr.max()) + 1
    n_features = X_train.shape[1]

    print(f"\n{'='*70}")
    print("BAYESIAN PyMC MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Training patients:     {len(X_train)}")
    print(f"  Features:              {n_features}")
    print(f"  Classes:               {K}")
    print(f"  Priors:                intercept ~ Normal(0, 2.0), coeff ~ Normal(0, 1.5)")
    print(f"  Chains: {chains}  |  Draws: {draws}  |  Tune: {tune}  |  target_accept: {target_acc}")

    # Build and sample the PyMC model
    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0.0, sigma=2.0, shape=K)
        coeff     = pm.Normal('coeff',     mu=0.0, sigma=1.5, shape=(n_features, K))

        logits = pm.math.dot(X_train, coeff) + intercept   # (n_train, K)
        p      = pm.math.softmax(logits, axis=1)            # (n_train, K)
        pm.Categorical('y', p=p, observed=y_tr)

        print(f"\n  Total parameters: {K + n_features * K}")
        print(f"  Running MCMC: {chains} chains x {draws} draws (+ {tune} tuning iterations)")

        # progressbar controls console output only and leaves the fit unchanged. A notebook
        # stores every redraw of the PyMC progress bar, so config switches it off.
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_acc,
            random_seed=data_seed,
            progressbar=config.get('progressbar', True)
        )

    total_samples = chains * draws
    print(f"\n  Sampling complete. {total_samples} posterior samples generated.")

    # Save the trace
    trace_path = model_dir / f"bayesian_trace_seed{data_seed}_{version}.nc"
    trace.to_netcdf(trace_path)
    print(f"  Trace saved: {trace_path}")

    metadata = {
        'data_seed':       data_seed,
        'version':         version,
        'model_type':      'Bayesian logistic regression (PyMC, NUTS)',
        'draws':           draws,
        'tune':            tune,
        'chains':          chains,
        'target_accept':   target_acc,
        'total_samples':   total_samples,
        'n_classes':       K,
        'n_features':      n_features,
        'n_train':         len(X_train),
        'n_cal':           len(x_cal),
        'prior_intercept': 'Normal(0, 2.0)',
        'prior_coeff':     'Normal(0, 1.5)',
        'feature_names':   x_train.columns.tolist(),
    }

    meta_path = model_dir / f"bayesian_metadata_seed{data_seed}_{version}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved: {meta_path}")

    return {
        'trace':      trace,
        'model':      model,
        'metadata':   metadata,
        'trace_path': str(trace_path),
        'meta_path':  str(meta_path),
    }


def _predict_posterior(trace, x_data, m_samples, data_seed):
    """
    Computes posterior predictive class probabilities for one split.

    The posterior draws come straight from the trace, and each draw passes through
    the softmax, so the output has the same shapes as every other model's predictions.

    Input: trace, ArviZ InferenceData with a posterior group; x_data, the features;
    m_samples, the number of posterior draws to use; data_seed, the seed for the
    draw subset.
    Output: dict with all_probs (M, n, 4), mean_probs (n, 4) and y_pred (n,).
    """
    X = x_data.astype(float).to_numpy() if hasattr(x_data, 'to_numpy') else np.asarray(x_data, dtype=float)
    n_patients = len(X)

    intercept_samples = trace.posterior['intercept'].values   # (chains, draws, K)
    coeff_samples     = trace.posterior['coeff'].values        # (chains, draws, n_feat, K)

    n_chains, n_draws, K = intercept_samples.shape
    total = n_chains * n_draws

    intercept_flat = intercept_samples.reshape(total, K)
    coeff_flat     = coeff_samples.reshape(total, *coeff_samples.shape[2:])

    # Reproducible subset of M posterior samples
    if m_samples < total:
        rng = np.random.default_rng(data_seed)
        idx = rng.choice(total, size=m_samples, replace=False)
        intercept_flat = intercept_flat[idx]
        coeff_flat     = coeff_flat[idx]
        n_used = m_samples
    else:
        n_used = total

    print(f"    Predicting: {n_patients} patients x {n_used} posterior samples...")

    all_probs = np.zeros((n_used, n_patients, K))
    for i in range(n_used):
        if (i + 1) % 250 == 0:
            print(f"      {i + 1}/{n_used} samples processed...")
        logits       = X @ coeff_flat[i] + intercept_flat[i]
        exp_l        = np.exp(logits - logits.max(axis=1, keepdims=True))
        all_probs[i] = exp_l / exp_l.sum(axis=1, keepdims=True)

    mean_probs = all_probs.mean(axis=0)
    y_pred     = mean_probs.argmax(axis=1)

    return {
        'all_probs':  all_probs,
        'mean_probs': mean_probs,
        'y_pred':     y_pred,
    }


# MAIN CALL ================================================================

def _run_bayesian(x_train, y_train, x_cal, y_cal, x_test, y_test, config):
    """
    Trains the Bayesian model and predicts the training, calibration and test splits.

    Input: x_train, y_train, x_cal, y_cal, x_test, y_test, the three splits with
    one-indexed labels 1 to 4; config, a dict with data_seed, version, model_dir,
    results_dir, draws, tune, chains, target_accept and m_samples (falling back to
    n_posterior_samples, then 1000).
    Output: the standard results dict with all_probs, mean_probs and y_pred for each
    split, feature_names, model_path and metadata. The dict is also saved as a pickle.
    """
    print("\n" + "="*70)
    print("BAYESIAN PyMC LOGISTIC REGRESSION")
    print("="*70)

    data_seed   = config['data_seed']
    version     = config['version']
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    m_samples   = config.get('m_samples', config.get('n_posterior_samples', 1000))

    # -- 0-indexed labels for scoring --
    y_tr = np.ravel((y_train.values - 1).astype(int) if hasattr(y_train, 'values') else (y_train - 1).astype(int))
    y_ca = np.ravel((y_cal.values - 1).astype(int)   if hasattr(y_cal, 'values')   else (y_cal - 1).astype(int))
    y_te = np.ravel((y_test.values - 1).astype(int)  if hasattr(y_test, 'values')  else (y_test - 1).astype(int))

    # -- Train --
    print(f"\n{'='*70}")
    print("STEP 1: Posterior sampling (NUTS)")
    train_out = _train_pymc(x_train, y_train, x_cal, y_cal, config)
    trace     = train_out['trace']

    # -- Predict all three splits --
    print(f"\n{'='*70}")
    print(f"STEP 2: Posterior predictions for all splits ({m_samples} samples)")
    print("  Predicting: training set")
    pred_train = _predict_posterior(trace, x_train, m_samples, data_seed)
    print("  Predicting: calibration set")
    pred_cal   = _predict_posterior(trace, x_cal,   m_samples, data_seed)
    print("  Predicting: test set")
    pred_test  = _predict_posterior(trace, x_test,  m_samples, data_seed)

    # -- Balanced accuracy per split (ensemble argmax) --
    bal_acc_tr   = balanced_accuracy_score(y_tr, pred_train['y_pred'])
    bal_acc_cal  = balanced_accuracy_score(y_ca, pred_cal['y_pred'])
    bal_acc_test = balanced_accuracy_score(y_te, pred_test['y_pred'])

    # -- Complete metadata and re-save --
    metadata = train_out['metadata']
    metadata.update({
        'm_samples':    m_samples,
        'n_test':       len(x_test),
        'bal_acc_tr':   float(bal_acc_tr),
        'bal_acc_cal':  float(bal_acc_cal),
        'bal_acc_test': float(bal_acc_test),
    })
    with open(train_out['meta_path'], 'wb') as f:
        pickle.dump(metadata, f)

    # -- Assemble standard results dict --
    results_bayesian = {
        'model_name':       'bayesian',
        'all_probs_train':  pred_train['all_probs'],   # (M, n_train, 4)
        'all_probs_cal':    pred_cal['all_probs'],      # (M, n_cal,   4)
        'all_probs_test':   pred_test['all_probs'],     # (M, n_test,  4)
        'mean_probs_train': pred_train['mean_probs'],   # (n_train, 4)
        'mean_probs_cal':   pred_cal['mean_probs'],     # (n_cal,   4)
        'mean_probs_test':  pred_test['mean_probs'],    # (n_test,  4)
        'y_pred_train':     pred_train['y_pred'],       # (n_train,) 0-indexed
        'y_pred_cal':       pred_cal['y_pred'],         # (n_cal,)
        'y_pred_test':      pred_test['y_pred'],        # (n_test,)
        'feature_names':    x_train.columns.tolist(),
        'model_path':       train_out['trace_path'],
        'metadata':         metadata,
    }

    # -- Save results --
    save_path = results_dir / f'results_bayesian_{version}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_bayesian, f)
    print(f"\n  Results saved: {save_path}")

    print("\n" + "="*70)
    print("BAYESIAN TRAINING COMPLETE")
    print(f"  Training bal. accuracy:     {bal_acc_tr:.4f}")
    print(f"  Calibration bal. accuracy:  {bal_acc_cal:.4f}")
    print(f"  Test bal. accuracy:         {bal_acc_test:.4f}")
    print(f"  Posterior samples used:     {m_samples}")
    print(f"  all_probs_train shape:      {results_bayesian['all_probs_train'].shape}")
    print(f"  all_probs_cal shape:        {results_bayesian['all_probs_cal'].shape}")
    print(f"  all_probs_test shape:       {results_bayesian['all_probs_test'].shape}")
    print("="*70)

    return results_bayesian
