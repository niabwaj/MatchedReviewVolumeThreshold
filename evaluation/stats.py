"""
stats.py: patient-level bootstrap, agreement statistics and shared score utilities.

Every interval resamples patients, so it measures sampling uncertainty in the test
patients. One resampling matrix can be shared between two statistics, which makes
their difference a paired comparison.
"""

import numpy as np
import pandas as pd

from pipeline import config as C


# =====================================================================
# 1. BOOTSTRAP
# =====================================================================

def boot_indices(n: int, n_boot: int = None, seed: int = None) -> np.ndarray:
    """
    Draws the bootstrap resampling matrix.

    Input: n, the number of patients; n_boot, the number of resamples (default
    config.N_BOOT); seed (default config.BOOT_SEED).
    Output: idx, (n_boot, n), each row a with-replacement draw of patient positions.
    """
    n_boot = C.N_BOOT if n_boot is None else n_boot
    rng    = np.random.default_rng(C.BOOT_SEED if seed is None else seed)
    return rng.integers(0, n, size=(n_boot, n))


def boot_ci(stat_fn, n: int, idx: np.ndarray = None, n_boot: int = None,
            seed: int = None, level: float = None) -> dict:
    """
    Bootstraps a statistic defined as a function of a patient index array.

    A statistic with no finite resample value, such as recall with no errors, returns
    its point value with an undefined interval.

    Input: stat_fn, a function of an index array returning a float; n, the number
    of patients; idx, a shared resampling matrix, or None to draw one; n_boot; seed;
    level, the interval level (default config.CI_LEVEL).
    Output: dict with point, lo, hi, se and n_boot (the finite resamples used).
    """
    level = C.CI_LEVEL if level is None else level
    idx   = boot_indices(n, n_boot, seed) if idx is None else idx
    point = float(stat_fn(np.arange(n)))
    draws = np.array([stat_fn(row) for row in idx], dtype=float)
    draws = draws[np.isfinite(draws)]
    # A degenerate statistic (no errors, no flagged patients, a constant flag) leaves
    # no finite resample, so the point returns with an undefined interval.
    if draws.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_boot": 0}
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "se": float(np.std(draws, ddof=1)), "n_boot": len(draws)}


def paired_boot_diff(stat_a, stat_b, n: int, idx: np.ndarray = None,
                     n_boot: int = None, seed: int = None, level: float = None) -> dict:
    """
    Bootstraps the difference between two statistics on the same patients.

    Both statistics are evaluated on the same resample, so the interval reflects the
    paired design.

    Input: stat_a, stat_b, functions of an index array; n; idx; n_boot; seed; level.
    Output: dict with a, b, diff (a - b), lo, hi, excludes_zero and n_boot.
    """
    level = C.CI_LEVEL if level is None else level
    idx   = boot_indices(n, n_boot, seed) if idx is None else idx
    base  = np.arange(n)
    a0, b0 = float(stat_a(base)), float(stat_b(base))
    d = np.array([float(stat_a(r)) - float(stat_b(r)) for r in idx], dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"a": a0, "b": b0, "diff": a0 - b0, "lo": float("nan"),
                "hi": float("nan"), "excludes_zero": False, "n_boot": 0}
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"a": a0, "b": b0, "diff": a0 - b0, "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(d)}


def fmt_ci(d: dict, dp: int = 3, pct: bool = False) -> str:
    """
    Formats a bootstrap result as 'point [lo, hi]' for a table cell.

    Input: d, a dict from boot_ci or paired_boot_diff; dp, decimal places; pct,
    multiplies by 100 when True.
    Output: the formatted string, with '[undefined]' when the interval is missing.
    """
    m = 100.0 if pct else 1.0
    key = "diff" if "diff" in d else "point"
    if not np.isfinite(d.get("lo", np.nan)):
        return f"{d[key]*m:.{dp}f} [undefined]"
    return f"{d[key]*m:.{dp}f} [{d['lo']*m:.{dp}f}, {d['hi']*m:.{dp}f}]"


def wilson_ci(k: int, n: int, level: float = None) -> tuple:
    """
    Computes the Wilson score interval for a proportion.

    Empirical coverage lies near 0.95, where the normal approximation interval is
    unreliable at these sample sizes.

    Input: k, the successes; n, the trials; level (default config.CI_LEVEL).
    Output: (lo, hi), or (NaN, NaN) when n is 0.
    """
    level = C.CI_LEVEL if level is None else level
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z  = norm.ppf(1 - (1 - level) / 2)
    p  = k / n
    d  = 1 + z**2 / n
    c  = p + z**2 / (2 * n)
    h  = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float((c - h) / d), float((c + h) / d))


# =====================================================================
# 2. AGREEMENT
# =====================================================================

def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes Cohen's kappa for two binary flags on the same patients.

    Kappa corrects for the agreement two flags at these rates reach by chance.

    Input: a, b, binary arrays.
    Output: kappa, or NaN when chance agreement is 1.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if np.isclose(pe, 1.0) else float((po - pe) / (1 - pe))


def pabak(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the prevalence-adjusted bias-adjusted kappa, 2 x observed agreement - 1.

    Kappa falls when the flagged share is far from one half, so PABAK sits beside it.

    Input: a, b, binary arrays.
    Output: PABAK as a float.
    """
    a, b = np.asarray(a), np.asarray(b)
    return float(2 * (a == b).mean() - 1)


def agreement_parts(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Computes the 2 x 2 counts and agreement statistics for two binary flags.

    Input: a, b, binary arrays.
    Output: dict with n11, n10, n01, n00, n, observed_agreement, ppos (positive
    agreement), pneg (negative agreement), prevalence_index, kappa and pabak.
    """
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    n   = n11 + n10 + n01 + n00
    ppos = 2 * n11 / (2 * n11 + n10 + n01) if (2 * n11 + n10 + n01) else float("nan")
    pneg = 2 * n00 / (2 * n00 + n10 + n01) if (2 * n00 + n10 + n01) else float("nan")
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n": n,
            "observed_agreement": (n11 + n00) / n if n else float("nan"),
            "ppos": float(ppos), "pneg": float(pneg),
            "prevalence_index": abs(n11 - n00) / n if n else float("nan"),
            "kappa": kappa(a, b), "pabak": pabak(a, b)}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Computes the Spearman rank correlation between two scores.

    Input: x, y, arrays of equal length.
    Output: dict with rho and p.
    """
    from scipy.stats import spearmanr
    r, p = spearmanr(np.asarray(x), np.asarray(y))
    return {"rho": float(r), "p": float(p)}


# =====================================================================
# 3. SCORES AND FLAGS
# =====================================================================

def conformal_score_continuous(mean_probs: np.ndarray) -> np.ndarray:
    """
    Returns each patient's second-largest mean predicted probability, p(2).

    p(2) is the exact continuous score behind the LAC prediction set. Under LAC a set
    holds two or more classes exactly when 1 - p(2) <= q, that is when p(2) >= 1 - q.
    Ranking patients by p(2) and cutting at a review budget therefore gives the
    conformal flag at that budget.

    Input: mean_probs, (n, K).
    Output: (n,) array of p(2).
    """
    p = np.sort(np.asarray(mean_probs), axis=1)
    return p[:, -2]


def top_n_flag(score: np.ndarray, n_flag: int) -> np.ndarray:
    """
    Flags the n_flag highest-scoring patients, breaking ties by position.

    Both methods then flag exactly the same number of patients at a fixed review rate.

    Input: score, (n,); n_flag, the number to flag.
    Output: (n,) integer flags.
    """
    score = np.asarray(score, dtype=float)
    flag  = np.zeros(len(score), dtype=int)
    if n_flag > 0:
        flag[np.argsort(-score, kind="stable")[:n_flag]] = 1
    return flag


def error_indicator(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> np.ndarray:
    """
    Marks the patients whose top-1 prediction is wrong.

    Input: y_true_idx, y_pred_idx, zero-indexed labels and predictions.
    Output: (n,) array, 1 for an error and 0 otherwise.
    """
    return (np.asarray(y_true_idx) != np.asarray(y_pred_idx)).astype(int)


def to_0idx(y) -> np.ndarray:
    """
    Converts one-indexed labels to a zero-indexed integer array.

    Input: y, a Series or array of labels 1 to K.
    Output: numpy array of labels 0 to K - 1.
    """
    arr = y.values.ravel() if hasattr(y, "values") else np.ravel(y)
    return (np.asarray(arr) - 1).astype(int)
