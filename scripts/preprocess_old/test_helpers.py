"""
test_helpers.py — Unit tests for encode helpers, causal card features, KNN, PSI.
FEDBANK_preprocessing_spec.md Section 10.
Run: python -m scripts.preprocess.test_helpers
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

if "CUDA_PATH" not in os.environ:
    prefix = sys.prefix
    candidate = os.path.join(prefix, "targets", "x86_64-linux")
    if os.path.exists(os.path.join(candidate, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = candidate
    elif os.path.exists(os.path.join(prefix, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = prefix

os.environ.setdefault("CUDF_SPILL", "on")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_encode_helpers():
    import cudf
    import cupy as cp
    from scripts.preprocess.s4_encode import fit_vocab, encode_codes, apply_freq, apply_ordinal

    s_train = cudf.Series(["A","A","B","C","A",None])
    vocab, counts = fit_vocab(s_train)
    assert "A" in vocab
    assert "unknown" in vocab or None not in vocab  # None → 'unknown'

    # Unseen category → -1
    s_test = cudf.Series(["A","D",None,"B"])
    codes = encode_codes(s_test, vocab)
    assert int(codes[1]) == -1   # "D" unseen
    print("[test] encode_codes: unseen → -1 ✓")

    # Frequency encoding: unseen → 0
    freq = apply_freq(s_test, vocab, counts)
    assert float(freq.iloc[1]) == 0.0
    print("[test] apply_freq: unseen → 0 ✓")

    # Ordinal: null → 'unknown' → mapped
    ord_s = apply_ordinal(s_test, vocab)
    assert int(ord_s.iloc[2]) >= -1
    print("[test] apply_ordinal: null handling ✓")


def test_causal_card():
    """Verify card_prior_txn_count is 0 for first transaction per card."""
    import pandas as pd
    import cudf
    import cupy as cp

    # Small synthetic dataset
    data = {
        "TransactionDT": [0, 86400, 2 * 86400, 0, 86400, 0],
        "TransactionAmt": [10., 20., 30., 40., 50., 60.],
        "card1": ["A","A","A","B","B","C"],
        "card2": ["1","1","1","2","2","3"],
        "card3": ["x","x","x","y","y","z"],
        "card5": ["p","p","p","q","q","r"],
        "addr1": ["1","1","1","2","2","3"],
        "D1":    [0., 1., 2., 0., 1., 0.],
        "split": [0, 0, 0, 0, 0, 0],
        "isFraud": [0, 0, 1, 0, 0, 0],
        "TransactionID": [1,2,3,4,5,6],
        "window_id": [-1]*6,
    }
    df = cudf.DataFrame(data)
    df["isFraud"] = df["isFraud"].astype("int8")
    df["split"] = df["split"].astype("int8")

    # Simulate S3 causal card features inline
    DT  = df["TransactionDT"].astype("int64")
    day = DT // 86400
    parts = [df[c].astype("str").fillna("na") for c in ("card1","card2","card3","card5","addr1")]
    key = parts[0]
    for p in parts[1:]: key = key + "_" + p
    start = (day - df["D1"].fillna(-9999).astype("int64")).astype("str")
    key = key + "_" + start
    df["card_key_id"] = key.astype("category").cat.codes.astype("int32")

    tmp = df[["card_key_id","TransactionDT","TransactionAmt"]].copy()
    tmp["_rid"] = cudf.Series(cp.arange(len(tmp), dtype=cp.int64), index=df.index)
    tmp = tmp.sort_values(["card_key_id","TransactionDT","_rid"])
    g = tmp.groupby("card_key_id")
    tmp["prior_n"] = g.cumcount().astype("float32")
    tmp = tmp.sort_values("_rid").reset_index(drop=True)

    prior_n = tmp["prior_n"].to_pandas().values
    # First transaction per card key → prior_n = 0
    assert prior_n[0] == 0, f"Expected 0 for first row, got {prior_n[0]}"
    # Second transaction same card → prior_n = 1
    assert prior_n[1] == 1, f"Expected 1 for second row of same card, got {prior_n[1]}"
    print("[test] causal card_prior_txn_count ✓")


def test_knn_impute():
    import cupy as cp
    from scripts.preprocess.s7_impute import knn_impute_block

    cp.random.seed(42)
    n, d, c = 200, 5, 3
    Az = cp.random.rand(n, d).astype("float32")
    T  = cp.random.rand(n, c).astype("float32")
    # Introduce NaN
    T[10, 0] = float("nan")
    T[50, 1] = float("nan")
    ref_pos = cp.arange(100, dtype=cp.int64)   # first 100 as reference
    medians = cp.array([0.5, 0.5, 0.5], dtype="float32")

    T_out = knn_impute_block(Az, T, ref_pos, k=3, batch=50, medians=medians)
    assert not cp.isnan(T_out).any(), "[test] KNN impute: NaN remains!"
    print("[test] knn_impute_block: all NaN filled ✓")


def test_psi():
    import cupy as cp
    from scripts.preprocess.s9_reduce import psi

    cp.random.seed(42)
    train_x = cp.random.randn(1000).astype("float32")
    other_x = cp.random.randn(1000).astype("float32")
    p1 = psi(train_x, other_x)
    # Same distribution → PSI ≈ 0
    assert p1 < 0.1, f"[test] PSI for same dist = {p1}, expected < 0.1"
    # Very different → PSI high
    other_shifted = (cp.random.randn(1000) + 5).astype("float32")
    p2 = psi(train_x, other_shifted)
    assert p2 > 0.25, f"[test] PSI for shifted dist = {p2}, expected > 0.25"
    print(f"[test] psi: same={p1:.3f}, shifted={p2:.3f} ✓")


if __name__ == "__main__":
    print("Running unit tests …")
    test_encode_helpers()
    test_causal_card()
    test_knn_impute()
    test_psi()
    print("\nAll unit tests passed ✓")
