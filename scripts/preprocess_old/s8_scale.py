"""
s8_scale.py — Manual RobustScaler (S8).
FEDBANK_preprocessing_spec.md Section S8.
Fitted on TRAIN rows; applied to all splits.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import json
import time
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram


_META_COLS = {"TransactionID", "TransactionDT", "isFraud", "split", "window_id",
              "card_key_id", "transaction_month", "transaction_year"}

_SKIP_DTYPES = {"int8"}


def _is_continuous(col: str, dtype: str) -> bool:
    if col in _META_COLS:
        return False
    if col.endswith("_missing"):
        return False
    if dtype == "int8":
        return False
    # ordinal codes (int16) may stay unscaled unless range > 100
    if dtype in ("float32", "float64"):
        return True
    return False


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S8-start")
    t0 = time.time()

    train_mask = df["split"] == 0
    scaler_stats: dict[str, dict] = {}   # col → {center, scale}

    continuous_cols = [
        c for c in df.columns if _is_continuous(c, str(df[c].dtype))
    ]
    print(f"[S8] Scaling {len(continuous_cols)} continuous columns …")

    CLIP = 10.0
    BLOCK = 64
    for b_start in range(0, len(continuous_cols), BLOCK):
        block = continuous_cols[b_start : b_start + BLOCK]
        for col in block:
            try:
                train_col = df[col][train_mask].dropna()
                if len(train_col) == 0:
                    continue
                qs = train_col.quantile([0.25, 0.5, 0.75])
                center = float(qs.iloc[1])   # median
                iqr    = float(qs.iloc[2]) - float(qs.iloc[0])
                if iqr > 1e-8:
                    scale = iqr
                else:
                    std = float(train_col.std())
                    scale = std if std > 1e-8 else 1.0

                df[col] = ((df[col] - center) / scale).clip(-CLIP, CLIP)
                df[col] = df[col].nans_to_nulls()
                scaler_stats[col] = {"center": center, "scale": scale, "clip": CLIP}

            except Exception as e:
                print(f"[S8] Warning: could not scale {col}: {e}")

        free_gpu()

    # ── Assertion: on train, scaled continuous medians ≈ 0 ────────────────
    fails = []
    for col in continuous_cols[:50]:  # spot-check first 50
        try:
            m = float(df[col][train_mask].dropna().quantile(0.5))
            if abs(m) > 1e-2:
                fails.append((col, m))
        except Exception:
            pass
    if fails:
        print(f"[S8] WARNING: {len(fails)} columns with |median| > 1e-2 after scaling: {fails[:5]}")

    # Persist
    scaler_path = cfg.ARTIFACTS_DIR / "scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)

    state["s8"] = {
        "n_scaled": len(scaler_stats),
        "scaler_json": str(scaler_path),
    }

    free_gpu()
    vram(tag=" S8-end")
    print(f"[S8] Done in {time.time()-t0:.1f}s | scaled: {len(scaler_stats)} cols")
    return df, state
