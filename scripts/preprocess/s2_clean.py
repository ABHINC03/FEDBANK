"""
s2_clean.py — Cleaning stage (S2).
FEDBANK_preprocessing_spec.md Section S2.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S2-start")
    t0 = time.time()
    missing_rate = state["s1"]["missing_rate"]
    nunique_map   = state["s1"]["nunique"]

    # ── 1. Replace ±inf with null ──────────────────────────────────────────
    float_cols = [c for c in df.columns if df[c].dtype in (cudf.dtype("float32"), cudf.dtype("float64"))]
    print(f"[S2] Replacing ±inf in {len(float_cols)} float columns …")
    for col in float_cols:
        s = df[col]
        df[col] = s.replace([float("inf"), float("-inf")], None)
    free_gpu()

    # ── 2. Extreme values abs > 1e30 → null; report abs > 1e9 ─────────────
    extreme_1e9_report = []
    for col in float_cols:
        s = df[col]
        # abs > 1e30
        mask_extreme = (s.abs() > 1e30)
        if int(mask_extreme.sum()) > 0:
            df[col] = s.where(~mask_extreme, other=None)
            df[col] = df[col].nans_to_nulls()
        # report abs > 1e9
        mask_1e9 = (s.abs() > 1e9)
        cnt = int(mask_1e9.sum())
        if cnt > 0:
            extreme_1e9_report.append((col, cnt))

    if extreme_1e9_report:
        print(f"[S2] Columns with abs>1e9 values (report only): {extreme_1e9_report[:20]}")

    # ── 3. nans_to_nulls after ops ─────────────────────────────────────────
    for col in float_cols:
        df[col] = df[col].nans_to_nulls()

    # ── 4. Drop columns (train stats only) ────────────────────────────────
    drop_missing = []
    drop_constant = []
    drop_nearconst = []

    train_mask = df["split"] == 0
    n_train = int(train_mask.sum())

    for col in list(df.columns):
        if col in ("TransactionID", "TransactionDT", "isFraud", "split", "window_id"):
            continue
        if col in cfg.NEVER_DROP:
            continue

        mr = missing_rate.get(col, 0.0)
        nu = nunique_map.get(col, 2)

        # Drop by missing rate
        if mr > cfg.DROP_MISSING:
            drop_missing.append(col)
            continue

        # Drop constant / near-constant (evaluate on train)
        if nu <= 1:
            drop_constant.append(col)
            continue

        # Near-constant: top value share >= 0.995
        try:
            train_col = df[train_mask][col]
            top_count = int(train_col.value_counts().iloc[0])
            if top_count / n_train >= cfg.NEAR_CONST_SHARE:
                drop_nearconst.append(col)
        except Exception:
            pass

    all_drops = set(drop_missing + drop_constant + drop_nearconst)
    print(f"[S2] Dropping {len(drop_missing)} high-missing, "
          f"{len(drop_constant)} constant, {len(drop_nearconst)} near-constant columns")
    df = df.drop(columns=list(all_drops))
    free_gpu()

    state["s2"] = {
        "drop_missing": drop_missing,
        "drop_constant": drop_constant,
        "drop_nearconst": drop_nearconst,
        "extreme_1e9_report": extreme_1e9_report[:50],
    }
    vram(tag=" S2-end")
    print(f"[S2] Done in {time.time()-t0:.1f}s  | columns remaining: {df.shape[1]}")
    return df, state
