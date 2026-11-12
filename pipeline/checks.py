"""
checks.py: the preflight gate between loading data and training models.

The gate inspects the loaded splits, reports every fault, names the models each
fault puts at risk, and halts the run with a resolution for each fault. The run
order is load, preflight, act on the reported resolution, then preflight again, so
a clean second pass is the record of data fit for training. A separate environment
check reports which models can train with the installed packages.
"""

import sys

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import data_io

SPLITS = ["train", "cal", "test"]

# RISK_MAP: the models each fault type threatens, so the report names the
# affected models as well as the column.
RISK_MAP = {
    "unscaled_features":   ["bootstrap", "svm", "mcd"],
    "out_of_range":        ["bootstrap", "svm", "mcd"],
    "missing_values":      ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "non_numeric":         ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "constant_column":     ["bayesian", "bootstrap", "svm"],
    "absent_in_split":     [],
    "column_mismatch":     ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "duplicate_patient":   ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "split_overlap":       ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "label_domain":        ["bayesian", "bootstrap", "mcd", "svm", "xgboost"],
    "small_calibration_class": [],
}

RISK_REASON = {
    "bootstrap": "L2 penalty and the C grid assume comparable feature magnitudes",
    "svm":       "RBF kernel distances and the C grid assume comparable magnitudes",
    "mcd":       "gradient scale and dropout regularisation assume bounded inputs",
    "bayesian":  "sampler geometry degrades and divergences appear",
    "xgboost":   "tree splits are scale invariant, but missing or non-numeric values still fail",
}


class PreflightError(RuntimeError):
    """Raised when the preflight gate finds a halting fault and halt is True."""


# SCALED_TOL: after a training-only fit, calibration and test values may lie a
# little outside 0 to 1, since a held-out patient can hold a more extreme value than
# anyone in training. An excursion within this tolerance is a note; a larger one
# halts the run.
SCALED_TOL = 0.25


def _finding(code, where, detail, resolution, severity="halt"):
    """
    Builds one row of the preflight report.

    Input: code, the fault type (a key of RISK_MAP); where, the split or splits
    affected; detail, what was found; resolution, the action to take; severity,
    "halt" to stop the run or "note" to record context only.
    Output: dict with the six report columns.
    """
    return {"code": code, "where": where, "detail": detail,
            "models_at_risk": ", ".join(RISK_MAP.get(code, [])) or "none",
            "resolution": resolution, "severity": severity}


# =====================================================================
# INDIVIDUAL CHECKS
# =====================================================================

def check_shapes_and_overlap(data: dict) -> list:
    """
    Checks split sizes, duplicate patients within a split, and patients shared
    between splits.

    A patient in two splits breaks exchangeability and the conformal coverage guarantee.

    Input: data, from data_io.load_splits.
    Output: list of findings.
    """
    out = []
    X, y = data["X"], data["y"]
    for s in SPLITS:
        if len(X[s]) != len(y[s]):
            out.append(_finding("column_mismatch", s,
                                f"X has {len(X[s])} rows, y has {len(y[s])}",
                                "Re-export this split from data_split.ipynb."))
        if X[s].index.duplicated().any():
            n = int(X[s].index.duplicated().sum())
            out.append(_finding("duplicate_patient", s, f"{n} duplicated index values",
                                "One record per patient is assumed throughout. Deduplicate at source."))
    for a, b in [("train", "cal"), ("train", "test"), ("cal", "test")]:
        shared = set(X[a].index) & set(X[b].index)
        if shared:
            out.append(_finding("split_overlap", f"{a} and {b}",
                                f"{len(shared)} patients appear in both",
                                "Exchangeability and the conformal guarantee both fail. Re-split."))
    return out


def check_columns(data: dict) -> list:
    """
    Checks that all three splits carry the same features in the same order.

    Input: data, from data_io.load_splits.
    Output: list of findings.
    """
    out = []
    ref = list(data["X"]["train"].columns)
    for s in ["cal", "test"]:
        cols = list(data["X"][s].columns)
        if cols != ref:
            missing = set(ref) - set(cols)
            extra   = set(cols) - set(ref)
            detail  = (f"order differs" if not missing and not extra
                       else f"missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
            out.append(_finding("column_mismatch", s, detail,
                                "Reindex this split onto the training column order."))
    return out


def check_dtypes_and_missing(data: dict) -> list:
    """
    Checks for non-numeric columns and missing values in features and labels.

    Input: data, from data_io.load_splits.
    Output: list of findings.
    """
    out = []
    for s in SPLITS:
        Xs = data["X"][s]
        # bool is not a numpy number subtype, but a 0/1 one-hot column is numeric for every
        # model, so both dtypes pass.
        def _ok(d):
            return np.issubdtype(d, np.number) or pd.api.types.is_bool_dtype(d)
        bad = Xs.columns[~Xs.dtypes.apply(_ok)].tolist()
        if bad:
            out.append(_finding("non_numeric", s, f"non-numeric columns: {bad[:8]}",
                                "Encode or drop these columns in data_split.ipynb."))
        miss = Xs.isna().sum()
        if miss.any():
            top = miss[miss > 0].sort_values(ascending=False)
            out.append(_finding("missing_values", s,
                                f"{int(miss.sum())} missing across {len(top)} columns, "
                                f"worst {top.index[0]} ({int(top.iloc[0])})",
                                "Impute or drop at source. No model here handles NaN."))
        if data["y"][s].isna().any():
            out.append(_finding("missing_values", f"{s} labels",
                                f"{int(data['y'][s].isna().sum())} missing labels",
                                "Drop these patients at source."))
    return out


def check_ranges(data: dict) -> list:
    """
    Checks feature ranges against the scaling rule in config.SCALE_ON_LOAD.

    With SCALE_ON_LOAD False, the default, features keep their instrument scale. Every questionnaire item is
    divided by its maximum possible score during data preparation, so both
    representations must arrive inside 0 to 1, and any value outside halts the run.

    With SCALE_ON_LOAD True, a value above 1 before scaling means the features are
    still raw. After a training-only fit, training values must lie inside 0 to 1.
    Calibration and test values may lie slightly outside, since a held-out patient
    can be more extreme than anyone in training. An excursion within SCALED_TOL is a
    note, and a larger one halts the run.

    Input: data, from data_io.load_splits or data_io.scale_splits.
    Output: list of findings.
    """
    out    = []
    scaled = data.get("scaled", False)

    for s in SPLITS:
        Xs = data["X"][s]
        hi = float(Xs.max().max())
        lo = float(Xs.min().min())
        over  = [c for c in Xs.columns if Xs[c].max() > 1.0 + 1e-6]
        under = [c for c in Xs.columns if Xs[c].min() < -1e-6]
        if not (over or under):
            continue

        if not C.SCALE_ON_LOAD:
            # Features keep their instrument scale, so the data must arrive inside 0 to 1.
            out.append(_finding(
                "out_of_range", s,
                f"{len(over)} columns above 1 (max {hi:.3f}) and {len(under)} below 0 "
                f"(min {lo:.3f}), with no scaler in the pipeline",
                "Every item should be divided by its maximum possible score in "
                "data_split.ipynb (individual items) or data_clean.ipynb (summed scores). "
                "Check that the scaling cell ran and that the divisor matches the item "
                "count. Set config.SCALE_ON_LOAD True only to reinstate a train-fit "
                "min-max instead.",
                severity="halt"))
            continue

        if not scaled:
            out.append(_finding(
                "unscaled_features", s,
                f"{len(over)} columns above 1 (max {hi:.3f}), {len(under)} below 0",
                "Run the scaling cell (data_io.scale_splits) before training.",
                severity="halt"))
            continue

        far = hi > 1 + SCALED_TOL or lo < -SCALED_TOL
        if s == "train":
            out.append(_finding(
                "out_of_range", s,
                f"training values outside 0 to 1 after scaling (min {lo:.3f}, max {hi:.3f})",
                "The scaler was not fit on this split. Refit on train.",
                severity="halt"))
        elif far:
            out.append(_finding(
                "out_of_range", s,
                f"values well outside the training range (min {lo:.3f}, max {hi:.3f}, "
                f"tolerance {SCALED_TOL})",
                "An excursion this large means the scaler saw the wrong split, or this "
                "split holds values the training data never saw. Check both before training.",
                severity="halt"))
        else:
            out.append(_finding(
                "out_of_range", s,
                f"{len(over)} columns just above 1 and {len(under)} just below 0 "
                f"(min {lo:.3f}, max {hi:.3f}), inside the {SCALED_TOL} tolerance",
                "Expected after a train-only fit. No action needed; the values are "
                "outside the training range, which is what a held-out split can do.",
                severity="note"))
    return out


def check_constants(data: dict) -> list:
    """
    Flags columns constant in the training split, and columns constant in calibration
    or test while varying in training.

    Input: data, from data_io.load_splits.
    Output: list of findings.
    """
    out = []
    Xtr = data["X"]["train"]
    const = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) <= 1]
    if const:
        out.append(_finding("constant_column", "train",
                            f"{len(const)} constant columns: {const[:8]}",
                            "Drop these features. They carry no signal and break the scaler."))
    for s in ["cal", "test"]:
        Xs = data["X"][s]
        gone = [c for c in Xs.columns if Xs[c].nunique(dropna=False) <= 1
                and Xtr[c].nunique(dropna=False) > 1]
        if gone:
            # A rare one-hot category can be absent from a held-out split by chance. Two
            # income levels carry 8 and 12 positives across 2,525 patients, so a 16 percent
            # calibration share often holds none. The columns stay in, and a penalised model
            # shrinks them towards zero.
            out.append(_finding("absent_in_split", s,
                                f"{len(gone)} columns constant here but varying in train: {gone[:8]}",
                                "Expected for a rare category. Kept deliberately; note it in "
                                "Limitations. Investigate only if a column with many positives "
                                "in training appears here.",
                                severity="note"))
    return out


def check_labels(data: dict) -> list:
    """
    Checks the label domain and the calibration count of the smallest class.

    Fewer than 30 calibration patients in a class makes the class-conditional coverage
    estimate too noisy to report.

    Input: data, from data_io.load_splits.
    Output: list of findings.
    """
    out = []
    expected = set(range(1, C.N_CLASSES + 1))
    for s in SPLITS:
        seen = set(pd.unique(data["y"][s].dropna()).astype(int).tolist())
        if not seen <= expected:
            out.append(_finding("label_domain", s, f"labels seen: {sorted(seen)}",
                                f"Labels must be 1 to {C.N_CLASSES}, one-indexed."))
    cal_counts = data["y"]["cal"].value_counts().sort_index()
    if cal_counts.min() < 30:
        out.append(_finding("small_calibration_class", "cal",
                            f"smallest class has {int(cal_counts.min())} calibration patients",
                            "Below 30 the class-conditional coverage estimate is too noisy to "
                            "report. Say so in Methods rather than fitting a class threshold."))
    return out


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def run_preflight(run: dict, data: dict, halt: bool = True, verbose: bool = True) -> dict:
    """
    Runs every check, prints the findings and the per-model risk, and halts on a fault.

    Input: run, the run context; data, from data_io.load_splits; halt, raises
    PreflightError on a halting finding when True; verbose, prints the report.
    Output: dict with findings, halts, notes, risk (one row per model), safe_models,
    passed and scaled. The result is also written into the run manifest.
    """
    findings = []
    for fn in (check_shapes_and_overlap, check_columns, check_dtypes_and_missing,
               check_ranges, check_constants, check_labels):
        findings.extend(fn(data))

    cols  = ["code", "where", "detail", "models_at_risk", "resolution", "severity"]
    df    = pd.DataFrame(findings, columns=cols)
    halts = [f for f in findings if f["severity"] == "halt"]
    notes = [f for f in findings if f["severity"] == "note"]

    # risk: one row per model with the faults that threaten it.
    rows = []
    for m in C.MODEL_KEYS:
        hits = [f["code"] for f in halts if m in RISK_MAP.get(f["code"], [])]
        rows.append({"Model": C.MODEL_LABELS[m], "key": m,
                     "n_faults": len(hits),
                     "faults": ", ".join(sorted(set(hits))) or "none",
                     "why_sensitive": RISK_REASON[m] if hits else "",
                     "verdict": "AT RISK" if hits else "safe to train"})
    risk = pd.DataFrame(rows)
    safe = risk.loc[risk["verdict"] == "safe to train", "key"].tolist()

    result = {"findings": df, "halts": pd.DataFrame(halts, columns=cols),
              "notes": pd.DataFrame(notes, columns=cols), "risk": risk,
              "safe_models": safe, "passed": len(halts) == 0,
              "scaled": data.get("scaled", False)}

    run["manifest"]["preflight"] = {
        "passed": result["passed"], "scaled_at_check": result["scaled"],
        "halting_findings": halts, "notes": notes, "safe_models": safe,
    }
    data_io.write_manifest(run)

    if verbose:
        _print_preflight(result, data)

    if halts and halt:
        raise PreflightError(
            f"{len(halts)} preflight finding(s). Nothing has been trained. "
            f"Act on the resolution column above, then run this cell again."
        )
    return result


def _print_preflight(result: dict, data: dict) -> None:
    """
    Prints the split summary, the findings and the per-model verdict.

    Input: result, from run_preflight; data, the loaded splits.
    Output: none.
    """
    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    for s in SPLITS:
        Xs, ys = data["X"][s], data["y"][s]
        counts = ys.value_counts().sort_index()
        dist   = "  ".join(f"T{int(k)}={int(v)}" for k, v in counts.items())
        print(f"  {s:5s}  n={len(Xs):5d}  features={Xs.shape[1]:3d}  "
              f"range {Xs.min().min():.3f} to {Xs.max().max():.3f}   {dist}")
    print(f"  scaled at this check: {result['scaled']}")

    print("-" * 70)
    notes = result["notes"]
    if len(notes):
        print(f"  {len(notes)} note(s), no action needed:")
        for _, r in notes.iterrows():
            print(f"   [{r['code']}]  {r['where']}: {r['detail']}")
        print()

    if result["passed"]:
        print("  No halting findings. All five architectures are safe to train.")
    else:
        print(f"  {len(result['halts'])} finding(s) that stop the run:\n")
        for _, r in result["halts"].iterrows():
            print(f"   [{r['code']}]  {r['where']}")
            print(f"     {r['detail']}")
            print(f"     at risk: {r['models_at_risk']}")
            print(f"     do this: {r['resolution']}\n")
        print(result["risk"][["Model", "n_faults", "verdict"]].to_string(index=False))
    print("=" * 70)

# =====================================================================
# ENVIRONMENT CHECK
# =====================================================================

# MODEL_REQUIREMENTS: the third-party packages each model imports.
MODEL_REQUIREMENTS = {
    "bayesian":  ["pymc", "arviz", "sklearn"],
    "bootstrap": ["sklearn"],
    "mcd":       ["tensorflow", "sklearn"],
    "svm":       ["sklearn"],
    "xgboost":   ["xgboost", "sklearn"],
}

# CORE_REQUIREMENTS: packages the evaluation modules need for any model.
CORE_REQUIREMENTS = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]

INSTALL_HINT = {
    "pymc":       "conda install -c conda-forge pymc",
    "arviz":      "conda install -c conda-forge arviz",
    "tensorflow": "pip install tensorflow",
    "xgboost":    "conda install -c conda-forge xgboost",
    "sklearn":    "conda install -c conda-forge scikit-learn",
    "scipy":      "conda install -c conda-forge scipy",
}


def _import_version(mod: str):
    """
    Returns the installed version of a package.

    Input: mod, the import name.
    Output: the version string, "installed" when the package has no version
    attribute, or None when the import fails.
    """
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def check_environment(verbose: bool = True) -> dict:
    """
    Reports which packages are installed and which models can therefore train.

    The check runs before any data is loaded, so a missing package shows up before
    training starts.

    Input: verbose, prints the report when True.
    Output: dict with versions, missing_core, trainable (model keys with every
    package present), blocked (model key to missing packages), python and executable.
    """
    mods = sorted(set(CORE_REQUIREMENTS + [m for v in MODEL_REQUIREMENTS.values() for m in v]))
    versions = {m: _import_version(m) for m in mods}

    missing_core = [m for m in CORE_REQUIREMENTS if versions[m] is None]
    trainable, blocked = [], {}
    for key, reqs in MODEL_REQUIREMENTS.items():
        gone = [m for m in reqs if versions[m] is None]
        (blocked.setdefault(key, gone) if gone else trainable.append(key))

    result = {"versions": versions, "missing_core": missing_core,
              "trainable": [k for k in C.MODEL_KEYS if k in trainable],
              "blocked": blocked,
              "python": sys.version.split()[0], "executable": sys.executable}

    if verbose:
        print("=" * 70)
        print("ENVIRONMENT CHECK")
        print("=" * 70)
        print(f"  python {result['python']}")
        print(f"  kernel {sys.executable}")
        print()
        for m in mods:
            v = versions[m]
            print(f"    {m:12s} {v if v else 'NOT INSTALLED'}")
        print()
        if missing_core:
            print(f"  CORE PACKAGES MISSING: {missing_core}. Nothing will run.")
        for key, gone in blocked.items():
            hints = "; ".join(INSTALL_HINT.get(m, f"install {m}") for m in gone)
            print(f"  {C.MODEL_LABELS[key]} cannot train: missing {gone}")
            print(f"     {hints}")
        print(f"\n  Trainable now: {result['trainable'] or 'none'}")
        if blocked:
            print("  If these packages exist in another conda environment, switching the "
                  "notebook kernel is faster and safer than installing into this one.")
        print("=" * 70)
    return result
