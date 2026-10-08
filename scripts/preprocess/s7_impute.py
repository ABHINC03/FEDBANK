"""
s7_impute.py — Imputation (S7).
FEDBANK_preprocessing_spec.md Section S7.
KNN imputation with cuML brute-force NearestNeighbors; fallback = median.
Fit on TRAIN rows only.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import numpy as np
import cudf
import cupy as cp
import cuml
from cuml.neighbors import NearestNeighbors

from .gpu_env import vram, free_gpu, check_vram


_META_COLS = {"TransactionID", "TransactionDT", "isFraud", "split", "window_id",
              "card_key_id", "transaction_month", "transaction_year"}
_SKIP_DTYPES = {"int8"}


def _numeric_continuous(df: cudf.DataFrame) -> list[str]:
    """Return continuous numeric columns (not flags/ordinals/meta)."""
    out = []
    for col in df.columns:
        if col in _META_COLS:
            continue
        if col.endswith("_missing"):
            continue
        dtype = str(df[col].dtype)
        if dtype in ("int8",):
            continue
        if dtype in ("float32", "float64", "int16", "int32", "int64"):
            out.append(col)
    return out


def knn_impute_block(Az: cp.ndarray, T: cp.ndarray,
                     ref_pos: cp.ndarray, k: int = 5,
                     batch: int = 20_000, medians=None) -> cp.ndarray:
    """
    Az: (n, d) float32 anchor matrix (no NaN).
    T:  (n, c) float32 target matrix with NaN for missing.
    ref_pos: cupy int array of reference row positions (train only).
    Returns T imputed in-place.
    """
    with cuml.using_output_type("cupy"):
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        nn.fit(Az[ref_pos])
        Tref = T[ref_pos]   # (R, c)

        for s in range(0, Az.shape[0], batch):
            _, ind = nn.kneighbors(Az[s:s + batch])   # (b, k)
            # Gather neighbor target values and nanmean
            gathered = Tref[ind]                        # (b, k, c)
            est = cp.nanmean(gathered, axis=1)      # (b, c)
            blk = T[s:s + batch]
            m = cp.isnan(blk)
            blk[m] = est[m]
            # Remaining all-NaN → median
            m2 = cp.isnan(blk)
            if m2.any() and medians is not None:
                blk[m2] = cp.broadcast_to(medians, blk.shape)[m2]
            T[s:s + batch] = blk

    return T


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S7-start")
    t0 = time.time()

    missing_rate = state["s1"]["missing_rate"]
    train_mask   = df["split"] == 0
    train_idx    = cp.where(train_mask.to_cupy())[0]
    n_total      = len(df)

    numeric_cols = _numeric_continuous(df)

    # ── Classify columns by imputation method ─────────────────────────────
    knn_cols    = [c for c in numeric_cols if 0 < missing_rate.get(c, 0) < cfg.KNN_MAX]
    median_cols = [c for c in numeric_cols if cfg.KNN_MAX <= missing_rate.get(c, 0) <= cfg.DROP_MISSING]
    no_imp_cols = [c for c in numeric_cols if missing_rate.get(c, 0) == 0]

    print(f"[S7] Imputation plan: {len(no_imp_cols)} no-op, "
          f"{len(knn_cols)} KNN, {len(median_cols)} median")

    # ── Compute train medians for all cols ────────────────────────────────
    print("[S7] Computing train medians …")
    medians: dict[str, float] = {}
    for col in knn_cols + median_cols:
        try:
            m = float(df[col][train_mask].dropna().quantile(0.5))
            medians[col] = m if not np.isnan(m) else 0.0
        except Exception:
            medians[col] = 0.0

    # ── Apply median imputation (high-missing columns) ────────────────────
    print(f"[S7] Median-imputing {len(median_cols)} columns …")
    for col in median_cols:
        m = medians[col]
        df[col] = df[col].fillna(m)
        df[col] = df[col].nans_to_nulls().fillna(m)
    free_gpu()

    # ── Build anchor matrix for KNN ────────────────────────────────────────
    # Anchor cols: train missing rate < 1%, non-zero variance, up to 24
    ANCHOR_CANDIDATES = [
        "TransactionAmt_log", "transaction_hour",
        "card1_freq", "addr1_freq", "card_prior_txn_count", "card_velocity",
    ] + [f"C{i}" for i in range(1, 15)]

    anchor_cols = []
    for c in ANCHOR_CANDIDATES:
        if c not in df.columns:
            continue
        mr = missing_rate.get(c, 0)
        if mr < 0.01:
            train_std = float(df[c][train_mask].std())
            if train_std > 1e-6:
                anchor_cols.append(c)
        if len(anchor_cols) >= cfg.KNN_ANCHOR_MAX:
            break

    print(f"[S7] Anchor columns ({len(anchor_cols)}): {anchor_cols}")

    if not knn_cols or not anchor_cols:
        print("[S7] No KNN imputation needed or no anchor columns.")
        state["s7"] = {
            "knn_cols": knn_cols, "median_cols": median_cols,
            "anchor_cols": anchor_cols, "medians": medians,
        }
        return df, state

    # Z-score anchors using TRAIN stats
    anchor_means: dict[str, float] = {}
    anchor_stds:  dict[str, float] = {}
    for c in anchor_cols:
        mu  = float(df[c][train_mask].mean())
        sig = float(df[c][train_mask].std())
        sig = sig if sig > 1e-8 else 1.0
        anchor_means[c] = mu
        anchor_stds[c]  = sig

    # Build anchor matrix for all rows
    print("[S7] Building anchor matrix …")
    Az_cols = []
    for c in anchor_cols:
        col_arr = df[c].to_cupy(dtype="float32", na_value=float("nan"))
        col_z   = (col_arr - anchor_means[c]) / anchor_stds[c]
        col_z   = cp.nan_to_num(col_z, nan=0.0)   # fill anchor NaN with 0 (mean)
        Az_cols.append(col_z)
    Az = cp.stack(Az_cols, axis=1)   # (n, d)
    Az = cp.ascontiguousarray(Az.astype("float32"))
    del Az_cols
    free_gpu()

    # Reference set: random sample of ≤ KNN_REF train rows
    cp.random.seed(cfg.SEED)
    n_ref = min(cfg.KNN_REF, len(train_idx))
    perm  = cp.random.permutation(len(train_idx))[:n_ref]
    ref_pos = train_idx[perm]   # shape (n_ref,)

    medians_arr = cp.array([medians.get(c, 0.0) for c in knn_cols], dtype="float32")

    # ── KNN impute in column blocks ────────────────────────────────────────
    BLOCK = 64
    imputed_count = 0
    print(f"[S7] KNN-imputing {len(knn_cols)} columns in blocks of {BLOCK} …")
    for b_start in range(0, len(knn_cols), BLOCK):
        block = knn_cols[b_start : b_start + BLOCK]
        T = cp.stack(
            [df[c].to_cupy(dtype="float32", na_value=float("nan")) for c in block],
            axis=1
        )   # (n, c)
        T = cp.ascontiguousarray(T)
        block_medians = cp.array([medians.get(c, 0.0) for c in block], dtype="float32")

        T = knn_impute_block(Az, T, ref_pos, k=cfg.KNN_K,
                             batch=cfg.KNN_BATCH, medians=block_medians)

        for j, col in enumerate(block):
            col_arr = T[:, j]
            # Remaining NaN → median
            col_arr = cp.where(cp.isnan(col_arr), medians.get(col, 0.0), col_arr)
            df[col] = cudf.Series(col_arr.astype("float32"), index=df.index).nans_to_nulls()
            df[col] = df[col].fillna(medians.get(col, 0.0))
            imputed_count += 1

        del T
        free_gpu()

    del Az, ref_pos
    free_gpu()

    # ── Assertion: no nulls in numeric cols ───────────────────────────────
    null_counts = {}
    for col in numeric_cols:
        nc = int(df[col].isna().sum())
        if nc > 0:
            null_counts[col] = nc
    if null_counts:
        print(f"[S7] WARNING: {len(null_counts)} columns still have nulls after imputation: "
              f"{list(null_counts.keys())[:10]}")
        # Force-fill with median
        for col in null_counts:
            df[col] = df[col].fillna(medians.get(col, 0.0))

    state["s7"] = {
        "knn_cols": knn_cols,
        "median_cols": median_cols,
        "anchor_cols": anchor_cols,
        "medians": medians,
        "anchor_means": anchor_means,
        "anchor_stds": anchor_stds,
        "n_ref": int(n_ref),
    }

    vram(tag=" S7-end")
    print(f"[S7] Done in {time.time()-t0:.1f}s | KNN-imputed: {imputed_count} cols")
    return df, state
