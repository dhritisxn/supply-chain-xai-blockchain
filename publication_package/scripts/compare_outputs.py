"""
compare_outputs.py

Numerically compares every CSV in two publication_package/ directories
(e.g. the original run vs a clean-environment re-run). Prints, per file,
whether it is identical, numerically equal within rtol, or different.

    python compare_outputs.py <dir_a> <dir_b> [--rtol 1e-9]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args()
    worst = 0.0
    bad = []
    for fa in sorted(args.a.glob("*.csv")):
        if fa.name == "reproducibility_manifest.csv":
            continue
        fb = args.b / fa.name
        if not fb.exists():
            print(f"MISSING  {fa.name}")
            bad.append(fa.name)
            continue
        if fa.read_bytes() == fb.read_bytes():
            print(f"IDENTICAL {fa.name}")
            continue
        da, db = pd.read_csv(fa), pd.read_csv(fb)
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            print(f"SHAPE-DIFF {fa.name} {da.shape} vs {db.shape}")
            bad.append(fa.name)
            continue
        num = da.select_dtypes("number").columns
        rel = 0.0
        if len(num):
            x, y = da[num].to_numpy(float), db[num].to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0, np.abs(x - y) / np.maximum(np.abs(x), 1e-12))
            rel = float(np.nanmax(d))
        non = [c for c in da.columns if c not in num and c != "chain"]
        text_same = da[non].astype(str).equals(db[non].astype(str))
        worst = max(worst, rel)
        status = "NUM-EQUAL" if rel <= args.rtol and text_same else "DIFFERENT"
        if status == "DIFFERENT":
            bad.append(fa.name)
        print(f"{status:9s} {fa.name}  max_rel_diff={rel:.3e}  text_cols_equal={text_same}")
    print(f"\nworst relative difference: {worst:.3e};  files differing beyond rtol={args.rtol}: {bad or 'none'}")


if __name__ == "__main__":
    main()
