"""
s4_encode.py — Categorical encoding (S4).
FEDBANK_preprocessing_spec.md Section S4.
All vocab/freq tables fitted on TRAIN rows only.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import json
import time
import numpy as np
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram
from .io_utils import save_state


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers (from spec Section S4)
# ─────────────────────────────────────────────────────────────────────────────

def fit_vocab(s_train: cudf.Series):
    """Returns (vocab_list, counts_cupy)."""
    s = s_train.astype("str").fillna("unknown")
    vc = s.value_counts()
    vocab = vc.index.to_pandas().tolist()
    counts = vc.values                   # cupy array
    return vocab, counts


def encode_codes(s: cudf.Series, vocab: list) -> cp.ndarray:
    """Ordinal codes; unseen → -1. Handles cuDF uint8 encoding (255 = unseen)."""
    s = s.astype("str").fillna("unknown")
    dtype = cudf.CategoricalDtype(categories=cudf.Index(vocab))
    raw = s.astype(dtype).cat.codes.values  # may be uint8 in cuDF 26+
    # Convert to int16; map max uint8 (255) → -1 for unseen
    codes = raw.astype(cp.int16)
    codes = cp.where(raw == 255, cp.int16(-1), codes)
    return codes



def apply_freq(s: cudf.Series, vocab: list, counts_cp) -> cudf.Series:
    """Frequency encoding; unseen → 0."""
    codes = encode_codes(s, vocab)
    out = cp.where(codes >= 0, counts_cp[cp.clip(codes, 0, None)], 0)
    return cudf.Series(out.astype("float32"), index=s.index)


def apply_ordinal(s: cudf.Series, vocab: list) -> cudf.Series:
    """Ordinal codes as int16; unseen/missing → -1."""
    return cudf.Series(encode_codes(s, vocab).astype("int16"), index=s.index)


# ─────────────────────────────────────────────────────────────────────────────

_HIGH_CARD = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo",
]

_LOW_CARD = [
    "ProductCD", "card4", "card6",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "DeviceType",
] + [f"id_{i:02d}" for i in range(12, 39)]

_M_MAP = {"T": 1, "F": 0, "unknown": -1}


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S4-start")
    t0 = time.time()

    train_mask = df["split"] == 0
    encoders = {}       # will be serialized to JSON (top-50k per column)
    counts_store = {}   # full counts arrays (numpy), saved to .npz

    # ── High-cardinality: frequency encoding ──────────────────────────────
    print("[S4] Frequency encoding high-cardinality columns …")
    for col in _HIGH_CARD:
        if col not in df.columns:
            continue
        s_train = df[col][train_mask]
        vocab, counts = fit_vocab(s_train)
        # Keep top 50k
        top = 50_000
        vocab_trunc  = vocab[:top]
        counts_trunc = counts[:top]

        freq_col = f"{col}_freq"
        df[freq_col] = apply_freq(df[col], vocab_trunc, counts_trunc)

        encoders[freq_col] = {"type": "freq", "vocab": vocab_trunc[:200]}   # store sample
        counts_store[freq_col] = cp.asnumpy(counts_trunc)

    # card1_freq_log
    if "card1_freq" in df.columns:
        df["card1_freq_log"] = cudf.Series(
            cp.log1p(df["card1_freq"].to_cupy(na_value=0.0)).astype("float32"),
            index=df.index
        )

    # card_addr_freq  (card1_addr1 key — from _card1_addr1_key computed in S3)
    if "_card1_addr1_key" in df.columns:
        s_train_ca = df["_card1_addr1_key"][train_mask]
        vocab_ca, counts_ca = fit_vocab(s_train_ca)
        vocab_ca_t = vocab_ca[:50_000]
        counts_ca_t = counts_ca[:50_000]
        df["card_addr_freq"] = apply_freq(df["_card1_addr1_key"], vocab_ca_t, counts_ca_t)
        df = df.drop(columns=["_card1_addr1_key"])
        encoders["card_addr_freq"] = {"type": "freq", "vocab": vocab_ca_t[:200]}
        counts_store["card_addr_freq"] = cp.asnumpy(counts_ca_t)
    free_gpu()

    # ── Low-cardinality: ordinal encoding ─────────────────────────────────
    print("[S4] Ordinal encoding low-cardinality columns …")
    for col in _LOW_CARD:
        if col not in df.columns:
            continue

        s_train = df[col][train_mask]

        # M columns: map T→1, F→0
        if col in ("M1","M2","M3","M4","M5","M6","M7","M8","M9"):
            s_str_all   = df[col].astype("str").fillna("unknown")
            s_str_train = s_str_all[train_mask]
            vocab, _ = fit_vocab(s_str_train)
            # Use T/F mapping if only T/F/unknown
            mapped = s_str_all.map(_M_MAP)
            if mapped.isna().sum() == 0 or True:
                # replace unmapped (None) with -1
                mapped = mapped.fillna(-1).astype("int16")
                df[col] = mapped
            else:
                df[col] = apply_ordinal(df[col], vocab)
            encoders[col] = {"type": "ordinal_M", "vocab": ["T","F","unknown"]}
        else:
            vocab, _ = fit_vocab(s_train)
            df[col] = apply_ordinal(df[col], vocab)
            encoders[col] = {"type": "ordinal", "vocab": vocab[:200]}
        free_gpu()

    # ── Drop raw string columns ────────────────────────────────────────────
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        print(f"[S4] Dropping remaining string columns: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # ── Interactions (S3 spec: after freq tables exist) ────────────────────
    if "card1_freq" in df.columns and "amt_to_avg_card_ratio" in df.columns:
        df["card1_freq_x_ratio"] = (df["card1_freq"] * df["amt_to_avg_card_ratio"]).astype("float32")
    if "addr1_freq" in df.columns and "TransactionAmt_log" in df.columns:
        df["addr1_freq_x_amt_log"] = (df["addr1_freq"] * df["TransactionAmt_log"]).astype("float32")

    # Assert no object columns remain
    remaining_obj = [c for c in df.columns if df[c].dtype == object]
    assert not remaining_obj, f"[S4] Object columns remain: {remaining_obj}"

    # ── Persist encoders ──────────────────────────────────────────────────
    enc_json_path = cfg.ARTIFACTS_DIR / "encoders.json"
    with open(enc_json_path, "w") as f:
        json.dump(encoders, f, indent=2)

    counts_npz_path = cfg.ARTIFACTS_DIR / "encoders_counts.npz"
    np.savez_compressed(counts_npz_path, **counts_store)

    state["s4"] = {
        "high_card_cols": [c for c in _HIGH_CARD if c in df.columns],
        "low_card_cols":  [c for c in _LOW_CARD  if c in df.columns],
        "encoders_json": str(enc_json_path),
    }

    free_gpu()
    vram(tag=" S4-end")
    print(f"[S4] Done in {time.time()-t0:.1f}s  | columns: {df.shape[1]}")
    return df, state
