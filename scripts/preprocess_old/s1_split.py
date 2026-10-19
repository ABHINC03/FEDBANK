"""
s1_split.py — Load, sort, split (S1).
FEDBANK_preprocessing_spec.md Section S1.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import numpy as np
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram
from .io_utils import save_state


def run(cfg, state: dict) -> tuple[cudf.DataFrame, dict]:
    """
    Load raw_merged.parquet, sort by TransactionDT, assign splits, compute
    per-column stats on train only.

    Returns (df, state) where df has a `split` column and meta columns are kept.
    """
    check_vram(cfg.VRAM_MIN_MB, tag=" S1-start")
    t0 = time.time()

    parquet_path = cfg.INTERIM_DIR / "raw_merged.parquet"
    print(f"[S1] Loading {parquet_path} …")
    df = cudf.read_parquet(parquet_path)
    print(f"[S1] Loaded: {df.shape}")

    # ── Sample mode ────────────────────────────────────────────────────────
    if cfg.SAMPLE_N > 0:
        # Take first SAMPLE_N rows after sort (preserve time structure)
        df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
        df = df.iloc[:cfg.SAMPLE_N].reset_index(drop=True)
        print(f"[S1] Sample mode: {len(df)} rows")
        ratio = cfg.SAMPLE_N / 590_540
        train_n = int(cfg.TRAIN_N * ratio)
        val_n   = int(cfg.VAL_N   * ratio)
        test_n  = int(cfg.TEST_N  * ratio)
    else:
        train_n = cfg.TRAIN_N
        val_n   = cfg.VAL_N
        test_n  = cfg.TEST_N

    # ── Sort ───────────────────────────────────────────────────────────────
    print("[S1] Sorting by TransactionDT, TransactionID …")
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

    n = len(df)
    print(f"[S1] Total rows: {n}")

    # ── Assign splits ──────────────────────────────────────────────────────
    split = cp.zeros(n, dtype=cp.int8)
    split[train_n : train_n + val_n] = 1
    split[train_n + val_n : train_n + val_n + test_n] = 2
    split[train_n + val_n + test_n :] = 3
    df["split"] = cudf.Series(split, index=df.index)

    # ── window_id for monitor split ────────────────────────────────────────
    monitor_mask = split == 3
    row_in_monitor = cp.where(monitor_mask, cp.arange(n, dtype=cp.int64) - (train_n + val_n + test_n), -1)
    window_id = cp.where(monitor_mask, row_in_monitor // 3000, -1).astype(cp.int32)
    window_id = cp.where(window_id >= 50, -1, window_id)  # windows >= 50 → -1
    df["window_id"] = cudf.Series(window_id.astype(cp.int32), index=df.index)

    # ── Assertions ─────────────────────────────────────────────────────────
    sizes = {
        0: int((df["split"] == 0).sum()),
        1: int((df["split"] == 1).sum()),
        2: int((df["split"] == 2).sum()),
        3: int((df["split"] == 3).sum()),
    }
    assert sum(sizes.values()) == n, f"[S1] Split sizes don't sum: {sizes}"

    # Time-order monotone within and across splits
    for s in range(3):
        max_dt = int(df[df["split"] == s]["TransactionDT"].max())
        min_dt_next = int(df[df["split"] == s + 1]["TransactionDT"].min())
        if max_dt > min_dt_next:
            raise AssertionError(
                f"[S1] Time order violated: split {s} max DT {max_dt} > split {s+1} min DT {min_dt_next}"
            )

    # Fraud rate check (warn, don't fail)
    expected_fraud = {0: 0.0333, 1: 0.0362, 2: 0.0370, 3: 0.0364}
    for s in range(4):
        mask = df["split"] == s
        fr = float(df[mask]["isFraud"].mean())
        exp = expected_fraud[s]
        if abs(fr - exp) > 0.005:
            print(f"[S1] WARNING: split {s} fraud rate {fr:.4f} deviates from expected {exp:.4f} (>±0.5 pp)")
        else:
            print(f"[S1] split {s} fraud rate {fr:.4f} ✓ (expected {exp:.4f})")

    print(f"[S1] Split sizes: {sizes}")

    # ── Per-column train stats ─────────────────────────────────────────────
    print("[S1] Computing per-column train stats …")
    train_mask = df["split"] == 0
    train_df = df[train_mask]
    n_train = int(train_mask.sum())

    missing_rate = {}
    nunique = {}

    for col in df.columns:
        if col in ("TransactionID", "TransactionDT", "isFraud", "split", "window_id"):
            continue
        try:
            mr = float(train_df[col].isna().sum()) / n_train
            missing_rate[col] = mr
            nu = int(train_df[col].nunique())
            nunique[col] = nu
        except Exception as e:
            print(f"[S1] Warning: could not compute stats for {col}: {e}")

    # ── Meta table ────────────────────────────────────────────────────────
    meta_cols = ["TransactionID", "TransactionDT", "isFraud", "split", "window_id"]
    meta = df[meta_cols]
    meta_path = cfg.INTERIM_DIR / "meta.parquet"
    meta.to_parquet(meta_path)
    print(f"[S1] Meta saved → {meta_path}")

    state["s1"] = {
        "n_total": n,
        "split_sizes": sizes,
        "train_n": train_n,
        "val_n": val_n,
        "test_n": test_n,
        "missing_rate": missing_rate,
        "nunique": nunique,
    }

    free_gpu()
    vram(tag=" S1-end")
    print(f"[S1] Done in {time.time()-t0:.1f}s")
    return df, state
