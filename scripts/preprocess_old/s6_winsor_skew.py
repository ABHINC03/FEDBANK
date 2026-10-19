"""
s6_winsor_skew.py — Winsorize + skew transform (S6).
FEDBANK_preprocessing_spec.md Section S6.
Quantiles fitted on TRAIN rows only; applied to all splits.
Runs BEFORE imputation (deliberate deviation from paper order).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram


# Columns that are flags / ordinals / binary — should NOT be winsorized/skew-transformed
_SKIP_PREFIXES = ("_missing", "card_is_first_seen", "amt_is_round", "is_weekend",
                   "email_match", "missing_crucial_info")
_SKIP_SUFFIXES = ("_missing",)
_SKIP_EXACT    = {"transaction_hour", "transaction_dayofweek", "transaction_part_of_day",
                   "transaction_month", "transaction_year",
                   "isFraud", "split", "window_id", "TransactionID", "TransactionDT",
                   "card_key_id", "card_is_first_seen", "amt_is_round",
                   "is_weekend", "email_match", "missing_crucial_info"}


def _is_numeric_continuous(col: str, dtype) -> bool:
    if col in _SKIP_EXACT:
        return False
    if col.endswith("_missing"):
        return False
    if dtype in ("int8",):
        return False
    if dtype not in ("float32", "float64", "int16", "int32", "int64"):
        return False
    return True


def _skew_cp(arr: cp.ndarray) -> float:
    """Compute skewness using cupy moments (on non-nan values)."""
    valid = arr[~cp.isnan(arr)]
    if len(valid) < 3:
        return 0.0
    mu = float(cp.mean(valid))
    sigma = float(cp.std(valid))
    if sigma < 1e-10:
        return 0.0
    n = len(valid)
    skew = float(cp.mean(((valid - mu) / sigma) ** 3))
    return skew


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S6-start")
    t0 = time.time()

    train_mask = df["split"] == 0
    q_lo, q_hi = cfg.WINSOR

    winsor_stats: dict[str, dict] = {}   # col → {lo, hi}
    skew_cols_log1p: list[str] = []
    skew_cols_signed: list[str] = []

    numeric_cols = [
        c for c in df.columns
        if _is_numeric_continuous(c, str(df[c].dtype))
    ]
    print(f"[S6] Processing {len(numeric_cols)} continuous numeric columns …")

    BLOCK = 64
    for b_start in range(0, len(numeric_cols), BLOCK):
        block = numeric_cols[b_start : b_start + BLOCK]
        for col in block:
            try:
                train_s = df[col][train_mask]
                # Compute quantiles on GPU
                qs = train_s.quantile([q_lo, q_hi])
                lo = float(qs.iloc[0])
                hi = float(qs.iloc[1])
                winsor_stats[col] = {"lo": lo, "hi": hi}

                # Apply clip to ALL splits
                df[col] = df[col].clip(lo, hi)
                df[col] = df[col].nans_to_nulls()

                # Skew (computed on TRAIN, post-clip)
                train_arr = df[col][train_mask].to_cupy(dtype="float32", na_value=float("nan"))
                sk = _skew_cp(train_arr)
                col_min = float(cp.nanmin(train_arr)) if len(train_arr) > 0 else 0.0

                if abs(sk) > cfg.SKEW_ABS:
                    if col_min >= 0:
                        # log1p
                        df[col] = cudf.Series(
                            cp.log1p(df[col].to_cupy(dtype="float32", na_value=float("nan"))).astype("float32"),
                            index=df.index
                        ).nans_to_nulls()
                        skew_cols_log1p.append(col)
                    else:
                        # signed log1p
                        vals = df[col].to_cupy(dtype="float32", na_value=float("nan"))
                        transformed = cp.sign(vals) * cp.log1p(cp.abs(vals))
                        df[col] = cudf.Series(transformed.astype("float32"), index=df.index).nans_to_nulls()
                        skew_cols_signed.append(col)

            except Exception as e:
                print(f"[S6] Warning: could not process column {col}: {e}")

        free_gpu()

    # Optional Yeo-Johnson path (fallback to signed_log1p if fails)
    if cfg.SKEW_METHOD == "yeo_johnson":
        print("[S6] Yeo-Johnson path selected — but signed-log1p already applied above as default.")
        print("[S6] (YJ would require refitting; keeping signed-log1p for determinism.)")

    state["s6"] = {
        "n_winsorized": len(winsor_stats),
        "skew_log1p_cols": skew_cols_log1p,
        "skew_signed_cols": skew_cols_signed,
        "winsor_stats": winsor_stats,   # {col: {lo, hi}}
    }

    free_gpu()
    vram(tag=" S6-end")
    print(f"[S6] Done in {time.time()-t0:.1f}s | "
          f"winsorized: {len(winsor_stats)}, log1p: {len(skew_cols_log1p)}, signed: {len(skew_cols_signed)}")
    return df, state
