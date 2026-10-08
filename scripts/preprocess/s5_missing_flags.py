"""
s5_missing_flags.py — Missing-value indicator flags (S5).
FEDBANK_preprocessing_spec.md Section S5.
Must run BEFORE imputation.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S5-start")
    t0 = time.time()

    missing_rate = state["s1"]["missing_rate"]
    train_mask   = df["split"] == 0
    n_train      = int(train_mask.sum())

    meta_cols = {"TransactionID", "TransactionDT", "isFraud", "split", "window_id",
                 "card_key_id", "transaction_month", "transaction_year"}

    # Columns to flag: train missing_rate in (FLAG_MIN, DROP_MISSING]
    flag_cols = []
    for col in df.columns:
        if col in meta_cols:
            continue
        mr = missing_rate.get(col, 0.0)
        if cfg.FLAG_MIN < mr <= cfg.DROP_MISSING:
            flag_cols.append(col)

    print(f"[S5] Creating {len(flag_cols)} missing flags …")

    # Create flags
    added_flags = []
    flag_data = {}   # col → int8 cupy array (full df)
    for col in flag_cols:
        flag = df[col].isna().astype("int8")
        flag_data[col] = flag.to_cupy()

    # ── De-duplicate identical flag columns ──────────────────────────────
    # Use (sum, sum_of_squares) as a fingerprint; then exact compare within collision groups
    fingerprints: dict[str, tuple] = {}
    for col, arr in flag_data.items():
        s  = int(arr.sum())
        s2 = int((arr.astype("int32") ** 2).sum())
        fingerprints[col] = (s, s2)

    # Group by fingerprint
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for col, fp in fingerprints.items():
        groups[fp].append(col)

    kept_flags: dict[str, str] = {}   # original col → representative name
    seen_reps: list[str] = []         # list of representative column names

    for fp, cols_in_group in groups.items():
        if len(cols_in_group) == 1:
            rep = cols_in_group[0]
            kept_flags[rep] = f"{rep}_missing"
            seen_reps.append(f"{rep}_missing")
        else:
            # Exact compare within group; keep one
            buckets: list[list[str]] = []
            for col in cols_in_group:
                placed = False
                for bucket in buckets:
                    if int((flag_data[col] != flag_data[bucket[0]]).sum()) == 0:
                        bucket.append(col)
                        placed = True
                        break
                if not placed:
                    buckets.append([col])

            for bucket in buckets:
                # representative: prefer paper-known names, else first alphabetically
                paper_preferred = [c for c in bucket if any(
                    c.startswith(p) for p in ("D12", "D", "C", "TransactionAmt")
                )]
                rep_col = paper_preferred[0] if paper_preferred else sorted(bucket)[0]

                # name by block pattern
                if all(c.startswith("V") for c in bucket):
                    # Extract block number from first V col digits
                    import re
                    nums = [re.search(r"\d+", c) for c in bucket]
                    nums = [m.group() for m in nums if m]
                    rep_name = f"Vgrp{'_'.join(sorted(set(nums[:3])))}_missing"
                else:
                    rep_name = f"{rep_col}_missing"

                for col in bucket:
                    kept_flags[col] = rep_name
                if rep_name not in seen_reps:
                    seen_reps.append(rep_name)

    # Write representative flags into df
    written: set[str] = set()
    for orig_col, flag_name in kept_flags.items():
        if flag_name in written:
            continue
        arr = flag_data[orig_col]
        # Drop if constant on train
        train_arr = arr[train_mask.to_cupy()]
        if int(train_arr.max()) == int(train_arr.min()):
            continue
        df[flag_name] = cudf.Series(arr.astype(cp.int8), index=df.index)
        written.add(flag_name)
        added_flags.append(flag_name)

    print(f"[S5] {len(added_flags)} unique missing flags added (de-duplicated from {len(flag_cols)})")

    state["s5"] = {
        "flag_cols_before_dedup": len(flag_cols),
        "added_flags": added_flags,
    }

    del flag_data
    free_gpu()
    vram(tag=" S5-end")
    print(f"[S5] Done in {time.time()-t0:.1f}s  | columns: {df.shape[1]}")
    return df, state
