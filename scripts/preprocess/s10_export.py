"""
s10_export.py — Export processed outputs + feature manifest (S10).
FEDBANK_preprocessing_spec.md Section 9.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import json
import time
import numpy as np
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu
from .io_utils import save_state


_SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "monitor"}
_META_COLS   = ["TransactionID", "TransactionDT", "isFraud", "split", "window_id",
                "card_key_id", "transaction_month", "transaction_year"]


def run(cfg, df: cudf.DataFrame, state: dict, tracks: dict) -> dict:
    t0 = time.time()
    track_a = tracks["track_a"]
    track_b = tracks["track_b"]

    meta_cols_present = [c for c in _META_COLS if c in df.columns]

    importance = state["s9"].get("importance", {})
    psi_vals   = state["s9"].get("psi_vals", {})
    aucs       = state["s9"].get("aucs", {})
    missing_rate = state["s1"].get("missing_rate", {})

    for split_val, split_name in _SPLIT_NAMES.items():
        mask = df["split"] == split_val
        sub  = df[mask]

        # Meta
        meta_sub = sub[meta_cols_present]
        meta_path = cfg.PROCESSED_DIR / f"meta_{split_name}.parquet"
        try:
            meta_sub.to_parquet(meta_path)
        except Exception as e:
            print(f"[S10] cudf write failed for meta_{split_name}: {e}; trying pandas …")
            meta_sub.to_pandas().to_parquet(meta_path, engine="pyarrow")

        # Track A (sel)
        feat_a = sub[track_a].astype("float32")
        feat_a_path = cfg.PROCESSED_DIR / f"features_sel_{split_name}.parquet"
        try:
            feat_a.to_parquet(feat_a_path)
        except Exception as e:
            print(f"[S10] cudf write failed for features_sel_{split_name}: {e}; using pandas …")
            feat_a.to_pandas().to_parquet(feat_a_path, engine="pyarrow")

        # Track B (pca)
        track_b_present = [c for c in track_b if c in df.columns]
        feat_b = sub[track_b_present].astype("float32")
        feat_b_path = cfg.PROCESSED_DIR / f"features_pca_{split_name}.parquet"
        try:
            feat_b.to_parquet(feat_b_path)
        except Exception as e:
            print(f"[S10] cudf write failed for features_pca_{split_name}: {e}; using pandas …")
            feat_b.to_pandas().to_parquet(feat_b_path, engine="pyarrow")

        print(f"[S10] {split_name}: meta={meta_sub.shape}, sel={feat_a.shape}, pca={feat_b.shape}")
        del meta_sub, feat_a, feat_b, sub
        free_gpu()

    # ── Feature manifest ──────────────────────────────────────────────────
    all_feats = sorted(set(track_a + track_b))
    manifest = []
    for col in all_feats:
        source = "engineered" if col in state.get("s3", {}).get("engineered_cols", []) else \
                 ("V_pca" if col.startswith("Vpca") else "raw")
        entry = {
            "name": col,
            "source": source,
            "track_a": col in track_a,
            "track_b": col in track_b,
            "importance_gain": float(importance.get(col, 0.0)),
            "psi_train_val": float(psi_vals.get(col, 0.0)),
            "missing_rate_train": float(missing_rate.get(col, 0.0)),
            "univariate_auc": float(aucs.get(col, 0.5)),
        }
        manifest.append(entry)

    # Sort by importance
    manifest.sort(key=lambda x: x["importance_gain"], reverse=True)
    for i, entry in enumerate(manifest):
        entry["importance_rank"] = i + 1

    manifest_path = cfg.ARTIFACTS_DIR / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── preprocess_state.json ─────────────────────────────────────────────
    state_path = cfg.ARTIFACTS_DIR / "preprocess_state.json"
    save_state(state, state_path)

    print(f"[S10] Done in {time.time()-t0:.1f}s")
    vram(tag=" S10-end")
    return {"manifest": manifest, "manifest_path": str(manifest_path)}
