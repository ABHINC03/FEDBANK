"""
s3_engineer.py — Feature engineering (S3).
FEDBANK_preprocessing_spec.md Section S3.
All features are causal (no future leakage).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import time
import cudf
import cupy as cp

from .gpu_env import vram, free_gpu, check_vram


_REF_EPOCH = 1512086400   # 2017-12-01 00:00:00 UTC


def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S3-start")
    t0 = time.time()

    DT = df["TransactionDT"].astype("int64")
    day = DT // 86400

    # ── Temporal features ──────────────────────────────────────────────────
    df["transaction_hour"]      = ((DT // 3600) % 24).astype("int8")
    df["transaction_dayofweek"] = ((day + 4) % 7).astype("int8")   # 2017-12-01=Friday; Mon=0
    df["is_weekend"]            = (df["transaction_dayofweek"] >= 5).astype("int8")
    df["transaction_part_of_day"] = (df["transaction_hour"] // 6).astype("int8")

    ts = cudf.to_datetime(DT + _REF_EPOCH, unit="s")
    df["transaction_month"] = ts.dt.month.astype("int8")
    df["transaction_year"]  = ts.dt.year.astype("int16")

    # ── Amount features ────────────────────────────────────────────────────
    amt = df["TransactionAmt"].astype("float32")
    df["TransactionAmt_log"] = cp.log1p(amt.to_cupy()).view("float32")
    df["TransactionAmt_log"] = cudf.Series(
        cp.log1p(amt.to_cupy(na_value=float("nan"))), index=df.index
    ).nans_to_nulls()
    df["amt_cents"]    = ((amt * 1000) % 1000).astype("float32")
    df["amt_is_round"] = (amt % 1 == 0).astype("int8")

    # ── Card key ───────────────────────────────────────────────────────────
    parts = []
    for c in ("card1", "card2", "card3", "card5", "addr1"):
        s = df[c].astype("str").fillna("na") if c in df.columns else cudf.Series(["na"] * len(df), index=df.index)
        parts.append(s)

    key = parts[0]
    for p in parts[1:]:
        key = key + "_" + p

    if cfg.USE_D1_UID and "D1" in df.columns:
        start = (day - df["D1"].fillna(-9999).astype("int64")).astype("str")
        key = key + "_" + start

    df["card_key_id"] = key.astype("category").cat.codes.astype("int32")

    # ── Causal card-level features ─────────────────────────────────────────
    print("[S3] Computing causal card features …")
    tmp = df[["card_key_id", "TransactionDT", "TransactionAmt"]].copy()
    tmp["_rid"] = cudf.Series(cp.arange(len(tmp), dtype=cp.int64), index=df.index)
    tmp = tmp.sort_values(["card_key_id", "TransactionDT", "_rid"])

    g = tmp.groupby("card_key_id")

    # prior_n = cumcount
    try:
        tmp["prior_n"] = g.cumcount().astype("float32")
    except Exception as e:
        print(f"[S3] cumcount failed ({e}), using fallback …")
        tmp = tmp.reset_index(drop=True)
        tmp["prior_n"] = (tmp.groupby("card_key_id").cumcount()).astype("float32")

    # prior_sum = cumsum - current
    try:
        tmp["prior_sum"] = (g["TransactionAmt"].cumsum() - tmp["TransactionAmt"]).astype("float32")
    except Exception as e:
        print(f"[S3] cumsum fallback: {e}")
        tmp["prior_sum"] = (tmp.groupby("card_key_id")["TransactionAmt"].cumsum() - tmp["TransactionAmt"]).astype("float32")

    # first_dt
    try:
        tmp["first_dt"] = g["TransactionDT"].transform("min")
    except Exception as e:
        print(f"[S3] transform(min) fallback: {e}")
        first_dt_agg = g.agg({"TransactionDT": "min"}).rename(columns={"TransactionDT": "first_dt"})
        tmp = tmp.merge(first_dt_agg, on="card_key_id", how="left")

    # prev_dt via shift
    try:
        tmp["prev_dt"] = g["TransactionDT"].shift(1)
        # mask where card_key_id changes
        shifted_key = tmp["card_key_id"].shift(1)
        different_card = tmp["card_key_id"] != shifted_key
        tmp["prev_dt"] = tmp["prev_dt"].where(~different_card, other=None)
    except Exception as e:
        print(f"[S3] shift fallback: {e}")
        tmp["prev_dt"] = tmp["TransactionDT"].shift(1)
        tmp["prev_dt"] = tmp["prev_dt"].where(
            tmp["card_key_id"] == tmp["card_key_id"].shift(1), other=None
        )

    # Restore original order
    tmp = tmp.sort_values("_rid").reset_index(drop=True)

    # Derived causal features
    DT_orig = tmp["TransactionDT"].astype("float32")
    first_dt_f = tmp["first_dt"].astype("float32")
    prev_dt_f  = tmp["prev_dt"].astype("float32")
    prior_n    = tmp["prior_n"]
    prior_sum  = tmp["prior_sum"]

    df["card_prior_txn_count"] = prior_n.values
    df["card_is_first_seen"]   = (prior_n == 0).astype("int8").values

    # card_time_since_last: log1p(DT - prev_dt); -1 for first seen
    dt_diff = (DT_orig - prev_dt_f)
    tsince = cudf.Series(
        cp.where(
            cp.isnan(prev_dt_f.to_cupy(na_value=float("nan"))),
            cp.float32(-1.0),
            cp.log1p(dt_diff.to_cupy(na_value=0.0))
        ).astype("float32"),
        index=df.index
    )
    df["card_time_since_last"] = tsince

    # card_velocity = prior_n / ((DT - first_dt)/86400 + 1)
    elapsed_days = (DT_orig - first_dt_f) / 86400 + 1
    velocity = prior_n / elapsed_days
    df["card_velocity"] = velocity.values.astype("float32")

    # amt_to_avg_card_ratio
    amt_vals = tmp["TransactionAmt"].astype("float32")
    avg_prior = prior_sum / prior_n.where(prior_n > 0, other=1.0)
    ratio = amt_vals / avg_prior.where(prior_n > 0, other=1.0)
    ratio = ratio.clip(0, 100)
    ratio = ratio.where(prior_n > 0, other=1.0)
    df["amt_to_avg_card_ratio"] = ratio.values.astype("float32")

    del tmp
    free_gpu()

    # ── Other domain features ──────────────────────────────────────────────
    crucial_cols = ["card1","card2","card3","card4","card5","card6",
                    "addr1","addr2","P_emaildomain"]
    existing_crucial = [c for c in crucial_cols if c in df.columns]
    missing_counts = cudf.Series(cp.zeros(len(df), dtype=cp.int32), index=df.index)
    for c in existing_crucial:
        missing_counts = missing_counts + df[c].isna().astype("int32")
    df["missing_crucial_info"] = missing_counts.astype("int8")

    # email_match
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        p = df["P_emaildomain"]
        r = df["R_emaildomain"]
        match = ((p == r) & p.notna() & r.notna()).astype("int8")
        df["email_match"] = match
    else:
        df["email_match"] = cudf.Series(cp.zeros(len(df), dtype=cp.int8), index=df.index)

    # card_addr_freq and interactions are computed in S4 after freq tables are fitted
    # Store card1_addr1 key for later
    if "card1" in df.columns and "addr1" in df.columns:
        df["_card1_addr1_key"] = df["card1"].astype("str").fillna("na") + "_" + df["addr1"].astype("str").fillna("na")

    free_gpu()
    state["s3"] = {"engineered_cols": [
        "transaction_hour","transaction_dayofweek","is_weekend","transaction_part_of_day",
        "transaction_month","transaction_year",
        "TransactionAmt_log","amt_cents","amt_is_round",
        "card_key_id","card_prior_txn_count","card_is_first_seen",
        "card_time_since_last","card_velocity","amt_to_avg_card_ratio",
        "missing_crucial_info","email_match",
    ]}
    vram(tag=" S3-end")
    print(f"[S3] Done in {time.time()-t0:.1f}s  | columns: {df.shape[1]}")
    return df, state
