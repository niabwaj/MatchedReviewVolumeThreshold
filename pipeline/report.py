"""
report.py: saves and prints every result table of a run.

Every table passes through save_table, which writes the CSV into its evaluation
group folder, adds a JSON sidecar with a one-sentence caption, and records the file
in the run manifest. The manifest therefore lists the source of every reported
number.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline import config as C
from pipeline import data_io


# =====================================================================
# 1. SAVING
# =====================================================================

def save_table(run: dict, group: str, name: str, df: pd.DataFrame,
               caption: str, item: str = "", index: bool = False) -> Path:
    """
    Writes one result table as CSV with a JSON sidecar and records it in the manifest.

    Input: run, the run context; group, an evaluation group key from
    config.EVAL_GROUPS; name, the file name stem; df, the table; caption, one
    sentence on what the table answers; item, an optional reference label; index,
    whether to write the DataFrame index.
    Output: the path of the saved CSV.
    """
    d    = run["eval_dirs"][group]
    stem = f"{name}_{run['rep']['tag']}"
    path = d / f"{stem}.csv"
    df.to_csv(path, index=index)

    (d / f"{stem}.json").write_text(json.dumps({
        "caption": caption, "requirements_item": item,
        "rows": int(len(df)), "columns": list(df.columns),
        "run_id": run["run_id"], "representation": run["representation"],
        "written": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    _record(run, {"group": group, "file": str(path.relative_to(run["run_dir"])),
                  "caption": caption, "item": item, "rows": int(len(df))})
    return path


def _record(run: dict, entry: dict) -> None:
    """
    Adds one saved output to the run manifest, replacing any earlier entry for the
    same file.

    A repeated evaluation overwrites its CSV, so the manifest keeps one entry per file.

    Input: run, the run context; entry, the manifest record for the file.
    Output: none; the manifest is updated on disk.
    """
    outs = run["manifest"]["outputs"]
    for i, e in enumerate(outs):
        if e.get("file") == entry["file"]:
            outs[i] = entry
            break
    else:
        outs.append(entry)
    data_io.write_manifest(run)


def save_object(run: dict, group: str, name: str, obj, caption: str,
                item: str = "") -> Path:
    """
    Writes a non-tabular result, such as a threshold dict or a nested per-model state,
    as a pickle and records it in the run manifest.

    A pickle keeps structure a flat table would lose.

    Input: run, group, name, caption and item, as in save_table; obj, the object.
    Output: the path of the saved pickle.
    """
    d    = run["eval_dirs"][group]
    path = d / f"{name}_{run['rep']['tag']}.pkl"
    data_io.save_pickle(obj, path)
    _record(run, {"group": group, "file": str(path.relative_to(run["run_dir"])),
                  "caption": caption, "item": item, "rows": None})
    return path


# =====================================================================
# 2. PRINTING
# =====================================================================

def show(df: pd.DataFrame, title: str, full: bool = False, n: int = 8) -> None:
    """
    Prints a result table in the run console.

    Input: df, the table; title, the heading; full, prints every row when True;
    n, the number of rows otherwise.
    Output: none.
    """
    print(f"\n--- {title} ---")
    if full or len(df) <= n:
        try:
            from IPython.display import display
            display(df)
        except Exception:
            print(df.to_string(index=False))
    else:
        print(df.head(n).to_string(index=False))
        print(f"  ... {len(df) - n} more rows saved to file")


def show_reference(df: pd.DataFrame, title: str, model_col: str = "Model") -> None:
    """
    Prints the reference model's rows in full and one summary line for each other model.

    The full output for one model is enough to check the format of a table by eye.

    Input: df, a table with one or more rows per model; title, the heading;
    model_col, the column naming the model.
    Output: none.
    """
    label = C.MODEL_LABELS[C.REFERENCE_MODEL]
    if model_col in df.columns and (df[model_col] == label).any():
        print(f"\n--- {title} — {label} shown in full, all models saved ---")
        try:
            from IPython.display import display
            display(df[df[model_col] == label])
        except Exception:
            print(df[df[model_col] == label].to_string(index=False))
        print(f"  {len(df)} rows total across {df[model_col].nunique()} models.")
    else:
        show(df, title)


def banner(text: str) -> None:
    """Prints a section banner in the run console."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# =====================================================================
# 3. CLOSING THE RUN
# =====================================================================

def finalise(run: dict) -> pd.DataFrame:
    """
    Closes the run: stamps the manifest with an end time and prints the output inventory.

    Input: run, the run context.
    Output: inv, a DataFrame with one row per saved file and its group, caption
    and item.
    """
    run["manifest"]["finished"] = datetime.now().isoformat(timespec="seconds")
    data_io.write_manifest(run)

    inv = pd.DataFrame(run["manifest"]["outputs"])
    banner(f"RUN COMPLETE — {run['run_dir'].name}")
    if inv.empty:
        print("  No outputs recorded.")
        return inv

    for group in C.EVAL_GROUPS:
        sub = inv[inv["group"] == group]
        if sub.empty:
            continue
        print(f"\n  {group}_{run['rep']['tag']}_{C.EVAL_GROUPS[group]}")
        for _, r in sub.iterrows():
            item = f"  [item {r['item']}]" if r["item"] else ""
            print(f"    {Path(r['file']).name}{item}")
            print(f"      {r['caption']}")

    print(f"\n  {len(inv)} files written under {run['run_dir']}")
    print(f"  Manifest: {run['run_dir'] / '_manifest.json'}")
    if C.SMOKE:
        print("\n  SMOKE RUN. These numbers are not reportable.")
    return inv
