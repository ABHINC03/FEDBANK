"""
io_utils.py — CSV→Parquet conversion, load helpers, state JSON persistence.
FEDBANK_preprocessing_spec.md Section S0.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# S0: CSV → Parquet (CPU, once)
# ─────────────────────────────────────────────────────────────────────────────

# Column dtype maps
_TXN_FLOAT32_SKIP = {"TransactionID", "TransactionDT", "isFraud"}

_KNOWN_CATS = {
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "DeviceType", "DeviceInfo",
}
_KNOWN_CATS.update({f"id_{i:02d}" for i in range(12, 39)})


def _build_dtype_map(cols: list[str]) -> dict[str, Any]:
    dtype_map: dict[str, Any] = {}
    for c in cols:
        if c == "TransactionID":
            dtype_map[c] = "int32"
        elif c == "TransactionDT":
            dtype_map[c] = "int64"
        elif c == "isFraud":
            dtype_map[c] = "int8"
        elif c in _KNOWN_CATS:
            dtype_map[c] = str
        else:
            dtype_map[c] = "float32"
    return dtype_map


def csv_to_parquet(cfg, force: bool = False) -> Path:
    """
    Read train_transaction.csv + train_identity.csv, merge, write raw_merged.parquet.
    Skipped if parquet already exists (unless --force).
    """
    out = cfg.INTERIM_DIR / "raw_merged.parquet"
    if out.exists() and not force:
        print(f"[S0] raw_merged.parquet exists, skipping (pass --force to overwrite).")
        return out

    t0 = time.time()
    txn_path = cfg.RAW_DIR / "train_transaction.csv"
    idn_path = cfg.RAW_DIR / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"[S0] train_transaction.csv not found at {txn_path}\n"
            "Place the Kaggle IEEE-CIS CSVs in data/ before running."
        )
    if not idn_path.exists():
        raise FileNotFoundError(
            f"[S0] train_identity.csv not found at {idn_path}."
        )

    print("[S0] Reading train_transaction.csv …")
    txn_cols = pd.read_csv(txn_path, nrows=0).columns.tolist()
    txn_dtype = _build_dtype_map(txn_cols)
    txn = pd.read_csv(txn_path, dtype=txn_dtype, low_memory=False)
    print(f"[S0]   txn shape: {txn.shape}")

    print("[S0] Reading train_identity.csv …")
    idn_cols = pd.read_csv(idn_path, nrows=0).columns.tolist()
    # Normalize hyphens → underscores
    idn_cols_normalized = [c.replace("-", "_") for c in idn_cols]
    idn_dtype = _build_dtype_map(idn_cols_normalized)
    idn = pd.read_csv(idn_path, low_memory=False)
    idn.columns = idn_cols_normalized
    idn = idn.astype({c: idn_dtype.get(c, "float32") for c in idn.columns if c in idn_dtype})
    print(f"[S0]   idn shape: {idn.shape}")

    print("[S0] Left-joining identity onto transaction on TransactionID …")
    merged = txn.merge(idn, on="TransactionID", how="left")
    print(f"[S0]   merged shape: {merged.shape}")

    assert len(merged) == len(txn), \
        f"[S0] Row count mismatch after merge: {len(merged)} vs expected {len(txn)}"
    assert merged["TransactionID"].nunique() == len(merged), \
        "[S0] TransactionID not unique after merge!"

    print(f"[S0] Writing {out} …")
    merged.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    print(f"[S0] Done in {time.time()-t0:.1f}s")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# State JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_serializable(obj):
    """Recursively make object JSON-serializable."""
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _json_serializable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_serializable(v) for v in obj]
    return obj


def save_state(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_serializable(state), f, indent=2)
    print(f"[state] Saved → {path}")


def load_state(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
