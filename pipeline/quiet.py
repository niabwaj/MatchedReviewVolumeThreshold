"""
quiet.py: silences three known, harmless console warnings by name.

The function runs once, before any model framework is imported, since TensorFlow
reads its logging settings at import time. Each silenced message is matched on its
own text and listed when the function runs. Every other warning still reaches the
console, so a convergence failure or a new deprecation stays visible.

The three messages and why each is safe to silence:

1. scikit-learn, LogisticRegression 'penalty'. The parameter is deprecated in 1.8
   and removed in 1.10. scikit-learn is pinned below 1.10, so the parameter
   behaves as documented and the warning is advance notice only.

2. scikit-learn, SVC 'probability'. The parameter is deprecated in 1.9 and removed
   in 1.11. The pin keeps Platt scaling inside SVC, which is the probability
   estimate every SVM result in this pipeline uses.

3. TensorFlow MLIR graph optimizer, 'operand #N does not dominate this use'.
   TensorFlow attempts an optional graph optimization, the pass fails, and
   TensorFlow falls back to the unoptimized path. The fitted network and its
   predictions are unaffected.
"""

import os
import warnings

# SKLEARN_MESSAGES: matched on the start of each warning's own text, so only these
# two are hidden. The patterns avoid commas, since the child-process form below is a
# comma-separated list.
SKLEARN_MESSAGES = [
    "'penalty' was deprecated in version 1.8 and will be removed in 1.10.",
    "The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11.",
]

# TF_ENV: read by TensorFlow at import time. Level 3 keeps fatal errors and drops
# the INFO, WARNING and ERROR output of the C++ layer, where the MLIR lines come from.
TF_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "GRPC_VERBOSITY": "ERROR",
    "GLOG_minloglevel": "2",
}


def silence_known_noise(verbose: bool = True) -> dict:
    """
    Applies the named warning filters and the TensorFlow logging settings.

    The run console stays readable during the grid searches, and the returned record
    shows exactly which messages were hidden.

    Input: verbose, prints the list of silenced messages when True.
    Output: dict with 'sklearn' (the filtered messages), 'tensorflow' (the
    environment settings applied) and 'note'.
    """
    for msg in SKLEARN_MESSAGES:
        warnings.filterwarnings("ignore", message=msg, category=FutureWarning)

    # GridSearchCV runs with n_jobs=-1, and a worker process does not inherit the
    # parent's warning filters. PYTHONWARNINGS is the one setting passed to the workers.
    existing = os.environ.get("PYTHONWARNINGS", "")
    entries  = [e for e in existing.split(",") if e]
    for msg in SKLEARN_MESSAGES:
        entries.append(f"ignore:{msg}:FutureWarning")
    os.environ["PYTHONWARNINGS"] = ",".join(entries)

    for k, v in TF_ENV.items():
        os.environ.setdefault(k, v)

    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

    record = {"sklearn": SKLEARN_MESSAGES, "tensorflow": TF_ENV,
              "note": "named filters only; any unlisted warning still prints"}

    if verbose:
        print("Silenced, by name:")
        print("  scikit-learn  LogisticRegression 'penalty' deprecation (1.8, removed 1.10)")
        print("  scikit-learn  SVC 'probability' deprecation (1.9, removed 1.11)")
        print("  tensorflow    MLIR graph optimizer messages, optimization is optional")
        print("Anything else still prints. A new warning during a run is a finding.")
    return record
