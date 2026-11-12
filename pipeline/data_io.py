"""
data_io.py: disk input and output for a run.

The module opens a timestamped run folder, loads the training, calibration and
test splits, applies the feature scaling rule, and writes the run manifest. Each
execution gets its own folder, and the manifest records the configuration, the
input file hashes and the package versions behind every result in it.
"""

import hashlib
import json
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as C

SPLITS = ["train", "cal", "test"]


# =====================================================================
# 1. RUN REGISTRY
# =====================================================================

def new_run(representation: str, note: str = "") -> dict:
    """
    Creates the timestamped folder for one execution and returns its run context.

    Input: representation, a key of config.REPRESENTATIONS; note, a free-text line
    stored in the manifest.
    Output: run, a dict with run_id, run_dir, representation, rep (the
    representation's config block), version, eval_dirs and manifest.
    """
    if representation not in C.REPRESENTATIONS:
        raise KeyError(f"Unknown representation {representation!r}. "
                       f"Expected one of {list(C.REPRESENTATIONS)}.")

    rep     = C.REPRESENTATIONS[representation]
    run_id  = C.make_run_id()
    run_dir = C.DIR_RESULTS / f"run_{run_id}_{representation}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    # eval_dirs: one folder per evaluation group.
    eval_dirs = {}
    for key, name in C.EVAL_GROUPS.items():
        d = run_dir / f"{key}_{rep['tag']}_{name}"
        d.mkdir(exist_ok=True)
        eval_dirs[key] = d

    manifest = {
        "run_id":         run_id,
        "representation": representation,
        "version":        rep["version"],
        "note":           note,
        "started":        datetime.now().isoformat(timespec="seconds"),
        "smoke":          C.SMOKE,
        "smoke_overrides": C.apply_smoke_overrides(),
        "config":         _config_snapshot(),
        "environment":    environment_report(),
        "data":           {},
        "scaling":        {},
        "preflight":      {},
        "models":         {},
        "outputs":        [],
    }

    run = {"run_id": run_id, "run_dir": run_dir, "representation": representation,
           "rep": rep, "version": rep["version"], "eval_dirs": eval_dirs,
           "manifest": manifest}
    write_manifest(run)

    print(f"  Run registered: {run_dir.name}")
    print(f"  Representation: {rep['label']}  ({rep['n_features']} features)")
    print(f"  Mode:           {'SMOKE, not reportable' if C.SMOKE else 'full run'}")
    return run


def resume_run(representation: str, run_id: str = None, note: str = "") -> dict:
    """
    Reopens an existing run folder and returns its run context.

    A run that stops part way, for example when a kernel restarts, continues in the
    same folder, so all five models share one folder and one manifest.

    Input: representation, a key of config.REPRESENTATIONS; run_id, a folder to
    reopen, or None for the newest folder matching config.SMOKE; note, a line stored
    in the manifest.
    Output: run, the same dict new_run returns, with the stored manifest reloaded.
    """
    rep     = C.REPRESENTATIONS[representation]
    run_dir = find_run(representation, run_id)
    mpath   = run_dir / "_manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"No manifest in {run_dir}. This folder was not written "
                                f"by new_run, so it cannot be resumed.")
    manifest = json.loads(mpath.read_text())
    manifest.setdefault("resumed", []).append(
        {"at": datetime.now().isoformat(timespec="seconds"), "note": note,
         "environment": environment_report()})

    eval_dirs = {}
    for key, name in C.EVAL_GROUPS.items():
        d = run_dir / f"{key}_{rep['tag']}_{name}"
        d.mkdir(exist_ok=True)
        eval_dirs[key] = d

    run = {"run_id": manifest.get("run_id", run_dir.name), "run_dir": run_dir,
           "representation": representation, "rep": rep, "version": rep["version"],
           "eval_dirs": eval_dirs, "manifest": manifest}
    write_manifest(run)

    trained = [k for k, v in manifest.get("models", {}).items() if "shapes" in v]
    print(f"  Resumed: {run_dir.name}")
    print(f"  Already fitted: {', '.join(trained) if trained else 'nothing yet'}")
    print(f"  Outputs recorded so far: {len(manifest.get('outputs', []))}")
    return run


def write_manifest(run: dict) -> Path:
    """
    Writes run['manifest'] to _manifest.json in the run folder.

    Input: run, the run context.
    Output: the manifest path.
    """
    path = run["run_dir"] / "_manifest.json"
    path.write_text(json.dumps(run["manifest"], indent=2, default=str))
    return path


def _config_snapshot() -> dict:
    """
    Captures every upper-case constant in config, so the manifest holds the full
    configuration of the run.

    Output: dict of constant name to value.
    """
    return {k: v for k, v in vars(C).items()
            if k.isupper() and isinstance(v, (int, float, str, bool, list, dict, type(None)))
            and not isinstance(v, Path)}


def environment_report() -> dict:
    """
    Records the Python version, the platform and the versions of the packages that
    affect results.

    A difference between two sets of numbers can then be checked against a
    difference in library versions.

    Output: dict with 'python', 'platform' and 'packages'.
    """
    versions = {}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "pymc", "arviz",
                "xgboost", "tensorflow", "torch", "matplotlib"]:
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}


def _sha256(path: Path, n_bytes: int = 1 << 20) -> str:
    """
    Returns the first 16 hex characters of a file's SHA-256 hash, read in chunks.

    Input: path, the file; n_bytes, the chunk size.
    Output: the hash prefix as a string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# =====================================================================
# 2. DATA LOADING
# =====================================================================

def load_splits(run: dict, verbose: bool = True) -> dict:
    """
    Loads the three splits exactly as stored on disk.

    The preflight check then sees the data in its stored state and reports any
    scaling fault. Boolean columns are cast to int.

    Input: run, the run context; verbose, prints the split shapes when True.
    Output: data, a dict with X and y per split, feature_names, split_indices and
    scaled (False). File hashes and class counts go into the manifest.
    """
    rep = run["rep"]
    X, y, hashes = {}, {}, {}
    cast_report = {}

    for split in SPLITS:
        xp = C.DIR_SPLITS[split] / rep["x_files"][split]
        yp = C.DIR_SPLITS[split] / rep["y_files"][split]
        for p in (xp, yp):
            if not p.exists():
                raise FileNotFoundError(f"Expected split file not found: {p}")
        X[split] = pd.read_csv(xp, index_col=0)
        # One-hot columns can return from a CSV as bool or as text. A single cast to int
        # gives every later check one dtype to expect.
        bool_cols = [c for c in X[split].columns
                     if pd.api.types.is_bool_dtype(X[split][c])]
        if bool_cols:
            X[split][bool_cols] = X[split][bool_cols].astype(int)
            cast_report[split] = bool_cols
        y[split] = pd.read_csv(yp, index_col=0)[C.TARGET_COL]
        hashes[xp.name] = _sha256(xp)
        hashes[yp.name] = _sha256(yp)

    # split_pkl: the patient indices behind the three splits, so a result row traces
    # back to its source record.
    with open(C.DIR_CODE / rep["split_pkl"], "rb") as f:
        split_indices = pickle.load(f)

    data = {"X": X, "y": y, "feature_names": X["train"].columns.tolist(),
            "split_indices": split_indices, "scaled": False}

    run["manifest"]["data"] = {
        "files": hashes,
        "n": {s: int(len(X[s])) for s in SPLITS},
        "n_features": int(X["train"].shape[1]),
        "class_counts": {s: {str(k): int(v) for k, v in y[s].value_counts().sort_index().items()}
                         for s in SPLITS},
        "bool_columns_cast_to_int": {k: len(v) for k, v in cast_report.items()},
    }
    write_manifest(run)

    if verbose:
        print(f"  Loaded (unscaled, as stored):")
        for s in SPLITS:
            print(f"    {s:5s}  X {str(X[s].shape):>12s}   y {len(y[s]):>5d}")
        print(f"    features: {X['train'].shape[1]}   target: {C.TARGET_COL}   "
              f"classes: {sorted(y['train'].unique())}")
        if cast_report:
            n = len(next(iter(cast_report.values())))
            print(f"    cast {n} boolean columns to int in each split")
    return data


def preview(data: dict, n: int = 5) -> pd.DataFrame:
    """
    Summarises each column of the training split for a visual check before the
    preflight gate.

    Input: data, from load_splits; n, unused.
    Output: DataFrame with one row per feature: dtype, min, max, mean, number of
    unique values and number of missing values.
    """
    Xtr = data["X"]["train"]
    return pd.DataFrame({
        "dtype":   Xtr.dtypes.astype(str),
        "min":     Xtr.min(),
        "max":     Xtr.max(),
        "mean":    Xtr.mean().round(4),
        "n_unique": Xtr.nunique(),
        "n_missing": Xtr.isna().sum(),
    })


# =====================================================================
# 3. SCALING
# =====================================================================

def scale_splits(run: dict, data: dict, verbose: bool = True) -> dict:
    """
    Applies the feature scaling rule and records it in the manifest.

    With config.SCALE_ON_LOAD False, the default, features keep their instrument
    scale. Every questionnaire item already arrives divided by its maximum possible
    score, a fixed property of the instrument and independent of every split. With
    SCALE_ON_LOAD True, a min-max scaler is fit on the training split alone and
    applied to all three splits.

    Input: run, the run context; data, from load_splits; verbose, prints the
    feature ranges when True.
    Output: data, with X replaced and scaler set only when a scaler runs.
    """
    if not C.SCALE_ON_LOAD:
        ranges = {s: (round(float(data["X"][s].min().min()), 4),
                      round(float(data["X"][s].max().max()), 4)) for s in SPLITS}
        run["manifest"]["scaling"] = {
            "method": "none", "reason": (
                "Features arrive on the instrument scale: every questionnaire item is "
                "divided by its maximum possible score during data preparation. Those "
                "divisors are fixed constants, so no split informs any transformation."),
            "ranges": ranges}
        write_manifest(run)
        if verbose:
            print("  No scaler applied. Features arrive divided by their instrument maxima.")
            for s in SPLITS:
                print(f"    {s:5s}  range {ranges[s][0]:.3f} to {ranges[s][1]:.3f}")
        return data

    if data.get("scaled"):
        print("  Already scaled. No action taken.")
        return data

    # Imported here, since only the SCALE_ON_LOAD True path uses scikit-learn.
    from sklearn.preprocessing import MinMaxScaler

    Xtr = data["X"]["train"]
    scaler = MinMaxScaler().fit(Xtr)

    before = {s: (float(data["X"][s].min().min()), float(data["X"][s].max().max()))
              for s in SPLITS}
    for s in SPLITS:
        data["X"][s] = pd.DataFrame(scaler.transform(data["X"][s]),
                                    index=data["X"][s].index,
                                    columns=data["X"][s].columns)
    after = {s: (float(data["X"][s].min().min()), float(data["X"][s].max().max()))
             for s in SPLITS}

    data["scaled"] = True
    data["scaler"] = scaler

    run["manifest"]["scaling"] = {
        "method": "MinMaxScaler", "fit_on": "train", "applied_to": SPLITS,
        "range_before": before, "range_after": after,
        "note": ("Summed files arrive pre-scaled on the build set; re-fitting on "
                 "train is equivalent to scaling the unscaled source from train."),
    }
    write_manifest(run)

    if verbose:
        print("  Min-max scaler fit on TRAIN only, applied to train, cal and test.")
        for s in SPLITS:
            print(f"    {s:5s}  range {before[s][0]:.3f} to {before[s][1]:.3f}"
                  f"   ->   {after[s][0]:.3f} to {after[s][1]:.3f}")
    return data


# =====================================================================
# 4. ARTEFACT IO
# =====================================================================

def save_pickle(obj, path) -> Path:
    """
    Writes a Python object to a pickle, creating the parent folder when needed.

    Input: obj, the object; path, the target file.
    Output: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path):
    """
    Reads a pickle.

    Input: path, the file.
    Output: the stored object, or None when the file is absent.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def find_run(representation: str, run_id: str = None) -> Path:
    """
    Locates a run folder for one representation.

    With config.SMOKE True the newest smoke folder is returned, and with it False the
    newest full folder, so smoke and full runs stay apart.

    Input: representation, a key of config.REPRESENTATIONS; run_id, part of a
    folder name to match, or None for the newest folder.
    Output: the run folder path.
    """
    runs = sorted(C.DIR_RESULTS.glob(f"run_*_{representation}"))
    if run_id:
        hits = [r for r in runs if run_id in r.name]
        if not hits:
            raise FileNotFoundError(f"No run folder matching {run_id} for {representation}.")
        return hits[-1]

    want_smoke = C.SMOKE
    match = [r for r in runs if ("smoke" in r.name) == want_smoke]
    if not match:
        kind  = "smoke" if want_smoke else "full"
        other = "full" if want_smoke else "smoke"
        have  = [r.name for r in runs][-3:]
        raise FileNotFoundError(
            f"No {kind} run for {representation}. Set C.SMOKE={not want_smoke} to reach a "
            f"{other} run, or pass run_id explicitly. Folders present: {have or 'none'}")
    return match[-1]
