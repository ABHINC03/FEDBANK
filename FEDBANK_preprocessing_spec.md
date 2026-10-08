# FEDBANK — IEEE-CIS Preprocessing + Dimensionality Reduction (GPU / RAPIDS / 6 GB)

> **Audience:** an AI coding agent (Antigravity) working inside the repo `ABHINC03/FEDBANK`.
> **Scope:** data preprocessing and dimensionality reduction ONLY. **Do not create any model, GNN, graph, SMOTE, or training code.**
> **Reference method:** Menezes & Filho, *"Investigating the Robustness of GNNs to Data Drift: A Case Study on Financial Transaction Data"*, IEEE Access 2025 (Sec. III-B, III-F).
> **Hardware/stack:** WSL2 Ubuntu, conda env `fedbank` (Python 3.11), NVIDIA RTX 4050 with **6 GB VRAM**, RAPIDS (cuDF, cuML, cuPy, RMM). No scikit-learn in the hot path.

---

## 0. How you (the agent) must work

1. **Inspect first.** List `scripts/`, `notebooks/`, `requirements.txt`, `package_installer.py`, `sample.py`. Read existing preprocessing code and any error logs. **Do not delete the user's existing files**; put all new code in `scripts/preprocess/` and keep old code untouched (you may add a short note in `README.md`).
2. **Never reinstall the environment blindly.** Run the checks in Section 1 first and report. Only fix the environment if a check fails, and then only as described in Section 1.2.
3. **Smoke test before full run.** Every stage must first run with `--sample 20000` (stratified-by-time head sample is fine) and pass all assertions, then run on the full data.
4. **Stages are resumable.** Each stage writes its output to `data/interim/` as parquet plus a JSON state file. A crash in stage N must never require re-running stages < N.
5. **If something fails, stop and diagnose.** Do not "work around" by silently dropping rows, columns, or the leakage rules in Section 3. Report the exact traceback and the fix you applied.
6. **Everything that is learned from data (quantiles, medians, vocabularies, scalers, PCA, selected features) is fitted on the TRAIN split only** and applied unchanged to validation / test / monitoring. This is non-negotiable (see Section 3).
7. Fixed seed everywhere: `SEED = 42`.
8. At the end, produce `reports/preprocess_report.md` (Section 11) and tick every item in the Definition of Done (Section 12).

---

## 1. Environment (verify, then only fix what is broken)

### 1.1 Mandatory checks (write them as `scripts/preprocess/check_env.py`)

```python
import os
os.environ.setdefault("CUDF_SPILL", "on")          # MUST be set before importing cudf
import sys, subprocess
print("python", sys.version)
import cupy as cp, cudf, cuml, rmm
print("cudf", cudf.__version__, "| cuml", cuml.__version__, "| cupy", cp.__version__, "| rmm", rmm.__version__)
free, total = cp.cuda.runtime.memGetInfo()
print(f"GPU free/total: {free/2**30:.2f} / {total/2**30:.2f} GiB")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                     capture_output=True, text=True).stdout)
# tiny functional tests
s = cudf.Series([1.0, None, 3.0]); assert int(s.isna().sum()) == 1
from cuml.neighbors import NearestNeighbors
X = cp.random.rand(100, 4).astype("float32")
nn = NearestNeighbors(n_neighbors=3, algorithm="brute").fit(X); nn.kneighbors(X[:5])
from cuml.decomposition import PCA
PCA(n_components=2).fit(X)
print("ENV OK")
```

Expected: Python 3.11, GPU name contains `4050`, total ≈ 6 GiB (usable free is typically 5.0–5.6 GiB because Windows reserves some). `ENV OK` printed.

### 1.2 If imports fail or versions conflict

The current `SETUP.md` installs `cuml`, `cudf`, `cugraph` with **three separate** `conda install` commands. Separate solves frequently produce mismatched RAPIDS/CUDA builds — a very likely source of the errors seen so far. If the check fails with import/ABI/`libcudf`/`libcuda` errors, recreate the environment with **one** solve using the selector at <https://docs.rapids.ai/install> (choose: Conda, WSL2, Python 3.11, CUDA 12, latest stable RAPIDS), e.g. pattern:

```bash
conda create -n fedbank_rapids -c rapidsai -c conda-forge -c nvidia \
    rapids=<LATEST_STABLE> python=3.11 'cuda-version>=12.0,<=12.9' \
    pyarrow pandas numpy scipy pyyaml tqdm
conda activate fedbank_rapids
conda install -c conda-forge py-xgboost-gpu     # optional, used in Sec. 8.3
```

(Do not hard-code a RAPIDS version from this document; take the selector's current output. Do not `pip install` torch/cudf wheels into this env.)

### 1.3 WSL2-specific rules (6 GB card)

* Keep **all data inside the WSL filesystem** (`~/FEDBANK/data/...`), never under `/mnt/c/...` (very slow I/O, causes timeouts).
* **Do NOT use `rmm.reinitialize(managed_memory=True)`** — CUDA unified-memory oversubscription is not supported on WSL2 and fails/hangs.
* Use `CUDF_SPILL=on` (host spilling) as a safety net, but design the pipeline so it should not be needed (Section 2).
* If WSL itself runs out of host RAM, the user can raise it in `C:\Users\<user>\.wslconfig` (`[wsl2] memory=...`), then `wsl --shutdown`. Mention this in the report only if host-RAM errors occur.

---

## 2. GPU memory strategy (the #1 source of crashes)

Raw merged data: **590,540 rows × ~434 columns**. In float64 that is ≈ 2 GB *per copy*; pandas-style operations create several copies → OOM on 6 GB. Rules:

| Rule | Detail |
|---|---|
| **float32 / int32 everywhere** | Downcast right after load. Labels `int8`. |
| **CSV → Parquet once, on CPU** | Do NOT parse the 650 MB CSV on the GPU (parser temp buffers spike memory). Convert with pandas (`dtype` float32, chunked) to `data/interim/raw_merged.parquet`, then use `cudf.read_parquet(columns=[...])`. |
| **Column-block processing** | Process V-features / D-features / ID-features in blocks of ≤ 64–100 columns for heavy ops (KNN, correlation, PCA). |
| **Row-batch processing** | KNN queries in batches of ≤ 20,000 rows. |
| **Free aggressively** | After each block: `del` temporaries, `gc.collect()`, `cp.get_default_memory_pool().free_all_blocks()` (or RMM equivalent). |
| **No `df.copy()` of the whole frame** | Add columns in place; drop columns in place. |
| **One index** | `reset_index(drop=True)` right after sorting and never again re-index — all column assignments assume the same `RangeIndex`. |
| **Log memory** | Print `memGetInfo()` at the start/end of every stage; abort with a clear message if free < 600 MB at a stage start. |
| **Target** | Peak VRAM < 4.5 GiB. |

Shared allocator setup (`scripts/preprocess/gpu_env.py`):

```python
import os
os.environ.setdefault("CUDF_SPILL", "on")
import gc, cupy as cp, rmm
from rmm.allocators.cupy import rmm_cupy_allocator

def init_gpu():
    rmm.reinitialize(pool_allocator=False, managed_memory=False)   # managed_memory=False is REQUIRED on WSL2
    cp.cuda.set_allocator(rmm_cupy_allocator)                      # cuPy + cuDF share one allocator

def free_gpu():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

def vram(tag=""):
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[VRAM]{tag} used {(total-free)/2**30:.2f} / {total/2**30:.2f} GiB")
    return free
```

---

## 3. Leakage & ordering rules (apply to every step)

1. Sort by `TransactionDT` ascending (stable), assign the split by **row position** (Section 4), only then fit anything.
2. **Fit on train only:** column drop lists, missing-rate thresholds, extreme-value caps, winsorization quantiles, skew list, medians, KNN reference set, category vocabularies, frequency tables, scaler stats, correlation pruning, feature importance, PCA, PSI baselines.
3. Engineered "history" features (card velocity, ratio to card average) must be **causal**: they may only use the *same row and earlier rows*. Never use a group mean/count computed over the whole dataset.
4. Validation may be used only for XGBoost early stopping (Sec. 8.3) and for reporting PSI. Test and monitoring are never used for any decision.
5. `isFraud`, `TransactionID`, `TransactionDT` are **never** model features. They go in a separate meta table.

---

## 4. Data, splits and expected numbers

**Input:** Kaggle IEEE-CIS: `train_transaction.csv` (590,540 × 394) and `train_identity.csv` (144,233 × 41). `test_*.csv` are unlabeled → **ignore**.
Put them in `data/raw/` (git-ignored). Expect the user to place them there; if missing, stop and say so.

**Merge:** left join identity onto transaction on `TransactionID` → 590,540 rows. Assert row count; assert `TransactionID` unique.

**Chronological split (derived from paper Table 1 node counts; the paper itself gives normalized-time boundaries, which map to these counts):**

| Split | Rows (by position after sorting by `TransactionDT`) | Paper reference |
|---|---|---|
| train | first **273,000** | 303,495 nodes *after* graph-SMOTE; original ≈ 273,000, fraud ≈ 3.33 % |
| val | next **58,500** | 58,500, fraud ≈ 3.62 % |
| test | next **58,500** | 58,500, fraud ≈ 3.70 % |
| monitor | remaining **200,540** | 200,540, fraud ≈ 3.64 % |

Assertions: sizes sum to 590,540; `max(DT of split i) <= min(DT of split i+1)`; fraud rates within ±0.5 pp of the table (warn, don't fail, if outside — report it).
The monitoring split will later be cut into 50 windows × 3,000 rows (first 150,000 rows). **Store monitor rows in strict time order and add a `window_id` meta column (`row_in_monitor // 3000`, windows ≥ 50 → `-1`).**

Do **not** do SMOTE or any resampling here.

---

## 5. Pipeline stages (execute in this order)

```
S0  csv_to_parquet        CPU     -> data/interim/raw_merged.parquet
S1  load_sort_split       GPU     -> meta + split labels, drop-list, dtypes
S2  clean                 GPU     -> extreme values, drop cols, inf->null
S3  engineer              GPU     -> temporal, card (causal), amount, counts
S4  encode                GPU     -> categoricals (train-fitted vocab)
S5  missing_flags         GPU     -> indicator flags (before imputation)
S6  winsor_skew           GPU     -> winsorize (train q), signed-log1p skewed
S7  impute                GPU     -> median / approximate KNN (train-fitted)
S8  scale                 GPU     -> RobustScaler (train-fitted, manual impl.)
S9  reduce                GPU     -> prune -> correlation -> importance -> (PCA track)
S10 export + report       CPU/GPU -> data/processed/*.parquet, manifests, report
```

Order deviates from the paper in one deliberate way: **skew transform and winsorization happen before KNN imputation** so Euclidean neighbor distances are not dominated by heavy tails. State this in the report.

### S0 — CSV → Parquet (CPU, once)

* Read `train_transaction.csv` and `train_identity.csv` with pandas using `dtype` maps: all numeric → `float32` (except `TransactionID` `int32`, `TransactionDT` `int64`, `isFraud` `int8`); known categoricals → `str`. Use `usecols=None`, `low_memory=False`.
* Merge (left on `TransactionID`), write `data/interim/raw_merged.parquet` (pyarrow, snappy).
* Identity column names in train_identity use `id_01 … id_38`, `DeviceType`, `DeviceInfo` (some copies of the dataset use `id-01` with hyphens in the *test* identity file; normalize `-` → `_`).
* Skip if the parquet exists and `--force` not given.

### S1 — Load, sort, split

* `cudf.read_parquet`. Sort by `TransactionDT` (then `TransactionID` for tie-break), `reset_index(drop=True)`.
* Create `split` as int8 column from row position (0=train,1=val,2=test,3=monitor).
* Save meta table: `TransactionID, TransactionDT, isFraud, split, window_id`.
* Compute on **train rows only** `missing_rate[col]` and `nunique[col]`; persist in state JSON.

### S2 — Clean

1. Replace `±inf` with null in all float columns.
2. **Extreme values:** any numeric value with `abs(x) > 1e30` → null (paper). Also report (don't act on) columns with `abs(x) > 1e9`.
3. After any arithmetic that can create NaN, call `col = col.nans_to_nulls()` — in cuDF **NaN and null are different**; `fillna` only fills nulls. Keep one convention: *missing = null*.
4. **Drop columns** (decision from train rows only):
   * missing rate > 0.90 (paper: V-features) — apply to all families (V, D, id);
   * constant / near-constant: `nunique <= 1`, or top-value share ≥ 0.995.
   Persist the dropped list with reason.
5. Never drop `TransactionAmt`, `card1`, `card2`, `card3`, `card4`, `card5`, `card6`, `addr1`, `addr2`, `ProductCD`, `C1–C14`, `D1`, `D15`, `P_emaildomain`, `R_emaildomain`, `M1–M9` unless constant.

### S3 — Feature engineering (all causal)

Temporal (from `TransactionDT`, seconds since reference; reference date from paper `2017-12-01 00:00:00 UTC` = epoch `1512086400`):

```python
DT = df["TransactionDT"].astype("int64")
day  = DT // 86400
df["transaction_hour"]      = ((DT // 3600) % 24).astype("int8")
df["transaction_dayofweek"] = ((day + 4) % 7).astype("int8")      # 2017-12-01 was a Friday; Monday=0
df["is_weekend"]            = (df["transaction_dayofweek"] >= 5).astype("int8")
df["transaction_part_of_day"] = (df["transaction_hour"] // 6).astype("int8")   # 0 night,1 morning,2 afternoon,3 evening
ts = cudf.to_datetime(DT + 1512086400, unit="s")
df["transaction_month"] = ts.dt.month.astype("int8")              # computed for the report; see exclusion rule below
df["transaction_year"]  = ts.dt.year.astype("int16")
```

> **Exclusion rule (important):** `transaction_month` and `transaction_year` are computed (the paper lists them) but are **excluded from the final model feature sets** because under a strict temporal split they are monotone proxies for time and cannot generalize to later windows; they would also encode the drift being studied. Keep them only in the meta table. Make this a config flag `EXCLUDE_TIME_PROXIES = True`.

Amount features: `TransactionAmt_log = log1p(TransactionAmt)`, `amt_cents = ((TransactionAmt*1000) % 1000)` (decimal-part), `amt_is_round = (TransactionAmt % 1 == 0)`.

Causal card/client features. Build a card key as a string, then an integer id:

```python
parts = [df[c].astype("str").fillna("na") for c in ("card1", "card2", "card3", "card5", "addr1")]
key = parts[0]
for p in parts[1:]:
    key = key + "_" + p
if USE_D1_UID:      # Kaggle-winner client id: account start day = day_of_txn - D1
    start = (day - df["D1"].fillna(-9999).astype("int64")).astype("str")
    key = key + "_" + start
df["card_key_id"] = key.astype("category").cat.codes.astype("int32")    # identifier only; never a model feature
```

Then, on a sorted copy of **only 4 columns** (to save memory), compute and write back in original row order:

```python
tmp = df[["card_key_id", "TransactionDT", "TransactionAmt"]].copy()
tmp["_rid"] = cp.arange(len(tmp), dtype="int64")
tmp = tmp.sort_values(["card_key_id", "TransactionDT", "_rid"])
g = tmp.groupby("card_key_id")
tmp["prior_n"]     = g.cumcount().astype("float32")
tmp["prior_sum"]   = (g["TransactionAmt"].cumsum() - tmp["TransactionAmt"]).astype("float32")
tmp["first_dt"]    = g["TransactionDT"].transform("min")
tmp["prev_dt"]     = g["TransactionDT"].shift(1)
tmp = tmp.sort_values("_rid")          # restore original order; assert (tmp._rid.values == arange).all()
```

Derived (all use only earlier rows of the same key):

| Feature | Definition |
|---|---|
| `card_prior_txn_count` | `prior_n` |
| `card_is_first_seen` | `prior_n == 0` (int8) |
| `card_time_since_last` | `log1p(DT - prev_dt)`; first-seen → `-1` (documented sentinel) |
| `card_velocity` | `prior_n / ((DT - first_dt)/86400 + 1)` |
| `amt_to_avg_card_ratio` | `TransactionAmt / (prior_sum / prior_n)` when `prior_n > 0`, else `1.0`; then `clip(0, 100)` |

If `groupby.transform("min")` or `groupby.shift` fails in the installed cuDF, fallback: compute `first_dt` via `g.agg({"TransactionDT": "min"})` and merge back **by `card_key_id`** while carrying `_rid` and re-sorting on `_rid`; compute `prev_dt` as `tmp["TransactionDT"].shift(1)` masked where `card_key_id != card_key_id.shift(1)`.

Other domain features:
* `missing_crucial_info` = row-wise count of nulls over `["card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain"]` (compute as sum of `isna()` columns; do not use `.apply`).
* `card_addr_freq` — frequency of the `card1_addr1` pair; fit on train (see S4). **Do not** feed the raw hash/id as a number (meaningless magnitude), contrary to the paper's `card_addr_hash` name — use its frequency encoding.
* `email_match` = `P_emaildomain == R_emaildomain` (both non-null) as int8.
* Interaction (paper "interactions between card/address and ratio features"): `card1_freq * amt_to_avg_card_ratio` and `addr1_freq * TransactionAmt_log` (after S4 frequencies exist; compute at end of S4).

Counters: `C1–C14` kept as numeric. `D1–D15`: keep numeric; additionally `D1_over_DT_days = D1 / (day+1)` is optional (off by default).

Don't use `df.apply`, Python loops over rows, or `.str` ops on large string columns more than necessary (cuDF string ops are fine on the ~15 categorical columns, not on 400 numeric ones).

### S4 — Categorical encoding (fit vocab on TRAIN rows)

Categorical columns: `ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1–M9, DeviceType, DeviceInfo, id_12 … id_38` (those that are strings in the parquet) plus integer-coded categoricals `card1, card2, card3, card5, addr1, addr2`.

* Convert to string, `fillna("unknown")` — **in this order** (`astype("str")` first, then `fillna`; `fillna("unknown")` on a float column raises).
* **High-cardinality** (`card1, card2, card3, card5, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo`, `card_key` pair): **frequency encoding** — count in train, unseen → 0; name `<col>_freq`. Also add `<col>_freq_log = log1p(freq)` only for `card1`.
* **Low-cardinality** (`ProductCD, card4, card6, M1–M9, DeviceType, id_12 … id_38` categoricals): **ordinal codes** from train vocabulary; unseen or missing → `-1`. For `M1–M9`, map `T→1, F→0, unknown→-1`.
* Drop the raw string columns after encoding (never leave `object` columns in the numeric matrix). `DeviceInfo` may also be reduced to a *brand* token (first token, lower-cased) before frequency encoding — optional.

Order-safe helpers (use these; do **not** use `merge` for encoding, because cuDF merge can reorder rows):

```python
import cupy as cp, cudf

def fit_vocab(s_train: cudf.Series):
    s = s_train.astype("str").fillna("unknown")
    vc = s.value_counts()                              # index = category, values = count
    return vc.index.to_pandas().tolist(), vc.values    # list[str], cupy counts

def encode_codes(s: cudf.Series, vocab: list) -> cp.ndarray:
    s = s.astype("str").fillna("unknown")
    dtype = cudf.CategoricalDtype(categories=cudf.Index(vocab))
    return s.astype(dtype).cat.codes.values            # cupy int; unseen -> -1

def apply_freq(s, vocab, counts_cp):
    codes = encode_codes(s, vocab)
    out = cp.where(codes >= 0, counts_cp[cp.clip(codes, 0, None)], 0)
    return cudf.Series(out.astype("float32"), index=s.index)

def apply_ordinal(s, vocab):
    return cudf.Series(encode_codes(s, vocab).astype("int16"), index=s.index)
```

Unit test both helpers on a toy Series including an unseen category and a null.
Persist vocabularies (≤ top 50,000 per column for frequency tables; the rest → 0) in `artifacts/encoders.json` + counts in `artifacts/encoders_counts.npz`.

### S5 — Missing-value indicators (BEFORE imputation)

* For each column with train missing rate in **(5 %, 90 %]** create `<col>_missing` (int8).
* **De-duplicate identical flag columns** (V-features in the same block share the same null pattern): hash each flag column (e.g. `flag.astype('int8').sum()` plus a few moments, then exact compare within collisions) and keep one representative per identical group, named `<block>_missing` (e.g. `Vgrp3_missing`). Also keep the paper's examples such as `D12_missing` if it exists.
* Drop flags that are constant on train.
* Expect only **tens** of unique flags, not hundreds. Later stages (S9) will keep ~8 as in the paper if they carry signal.

### S6 — Winsorize + skew transform (quantiles from TRAIN)

* Numeric columns only (exclude flags, ordinals, codes, binary).
* Winsorize to train **1st/99th percentile**: `lo, hi = q01, q99` from train non-null values; apply `clip(lo, hi)` to all splits.
* Skew list: train `abs(skew) > 2` **and** `min >= 0` → apply `log1p`; if the column has negatives and `abs(skew) > 2` → apply `sign(x)*log1p(abs(x))`. Persist the list.
* Paper uses Yeo-Johnson (`PowerTransformer`) for skewed columns. Default here is signed-log1p (deterministic, zero API risk). Optional config `SKEW_METHOD = "yeo_johnson"` may try `cuml.preprocessing.PowerTransformer`; if the import or fit fails, fall back to signed-log1p automatically and log it.
* Compute skew on GPU with `cp` moments on train only, in column blocks.

### S7 — Imputation (fit on TRAIN)

Per column family, using train missing rate `m`:

| Missing rate `m` | Method |
|---|---|
| `m == 0` | nothing |
| `0 < m < 0.50` | **approximate KNN imputation (k = 5)**, fallback median |
| `0.50 ≤ m ≤ 0.90` | train median |
| categorical/codes | already filled in S4 |

cuML has **no `KNNImputer`**. Implement it with `cuml.neighbors.NearestNeighbors` (brute force, euclidean):

* **Anchor columns** (distance space): up to 24 numeric columns with train missing rate < 1 % and non-zero variance — typically `TransactionAmt_log`, `C1–C14`, `transaction_hour`, `card1_freq`, `addr1_freq`, `card_prior_txn_count`, `card_velocity`. Z-score them with train mean/std (after S6).
* **Reference set:** random sample of **≤ 50,000 train rows** (seed 42). Neighbors are always searched in this set → no leakage from val/test/monitor.
* **Targets:** the columns with `0 < m < 0.50`; process in blocks of ≤ 64 columns and query rows in batches of 20,000.
* A cell is imputed with `nanmean` of the neighbors' *observed* values for that column; if all 5 neighbors are missing → train median.

```python
import cuml, cupy as cp
from cuml.neighbors import NearestNeighbors

def knn_impute_block(Az, T, ref_pos, k=5, batch=20000, medians=None):
    """Az: (n,d) float32 cupy anchor matrix (no NaN). T: (n,c) float32 cupy targets with NaN for missing.
       ref_pos: cupy int array of reference row positions (train only). Returns T imputed in place."""
    with cuml.using_output_type("cupy"):
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(Az[ref_pos])
        Tref = T[ref_pos]                                  # (R,c)
        for s in range(0, Az.shape[0], batch):
            _, ind = nn.kneighbors(Az[s:s+batch])          # (b,k)
            est = cp.nanmean(Tref[ind], axis=1)            # (b,c); all-NaN -> NaN (cupy may warn; ignore)
            blk = T[s:s+batch]
            m = cp.isnan(blk)
            blk[m] = est[m]
            m = cp.isnan(blk)                              # remaining -> median
            if m.any() and medians is not None:
                blk[m] = cp.broadcast_to(medians, blk.shape)[m]
    return T

# Getting cupy from cudf WITH nulls: df.values raises. Use:
# T = df[cols].to_cupy(dtype="float32", na_value=float("nan"))
# ...then write back column-by-column: df[c] = cudf.Series(T[:, j], index=df.index).nans_to_nulls()
```

Memory check: neighbor gather `(20000, 5, 64)` float32 ≈ 25 MB — fine.
Persist: anchor list, ref row ids, medians. After imputation assert `df[numeric].isna().sum().sum() == 0`.

### S8 — Scaling (manual RobustScaler, train-fitted)

Do not depend on `sklearn`/`cuml` scaler APIs. For every continuous numeric column (exclude int8 flags, ordinal codes, binary):

```python
q = train_col.quantile([0.25, 0.5, 0.75])      # on GPU; convert with .to_pandas()
center, iqr = q[0.5], q[0.75] - q[0.25]
scale = iqr if iqr > 1e-8 else (train_col.std() or 1.0)
x = ((x - center) / scale).clip(-10, 10)       # clip protects the GNN from rare blow-ups
```

Persist `center`, `scale`, clip bound in `artifacts/scaler.json`. Binary flags and ordinal codes stay unscaled (codes get a second pass only if the agent finds range > 100).
Assertion: on train, scaled continuous columns have median ≈ 0 (|median| < 1e-3).

---

## 6. Feature inventory target after S8

Roughly: ~15 engineered (temporal 4–5, card/amount 8–10), ~12–20 kept raw `C`/`D` features, up to ~80–120 V-features before reduction, ~15 frequency/ordinal encodings, ~10–30 unique missing flags. Typical total before S9: **150–250 columns**. S9 reduces this.

---

## 7. Dimensionality reduction — objective

Reduce redundant / noisy columns (V-features are heavily collinear in blocks; many flags are identical or uninformative) so the later GNN trains on a compact, stable feature vector. The paper ended with a hand/analysis-selected **38 features**; our target:

* **Track A — `sel`**: pruned + importance-selected original features. Default size **K = 48** (config; allowed range 38–64). 
* **Track B — `sel_pca`**: Track A's non-V features **plus** PCA components of the V-feature blocks. Default ≤ ~48 total.

Both tracks are exported; the modeling stage will choose.

---

## 8. Dimensionality reduction — steps (all fitted on TRAIN)

### 8.1 Hard prune
* Drop columns with train `std < 1e-6` or ≥ 99.5 % one value.
* Drop duplicate columns (exact equality on a 20k-row train sample, then verify on all train).

### 8.2 Correlation pruning (|r| ≥ 0.95)
* Compute the Pearson matrix on train in float32 using `cp.corrcoef` per **block pair** if > 250 columns (full 250×250 on 273k×250 ≈ 270 MB is fine; above that block it).
* Greedy: sort columns by univariate train AUC (Mann–Whitney via cupy ranks: `auc = (rank_sum_pos - n_pos(n_pos+1)/2) / (n_pos*n_neg)`, use `max(auc, 1-auc)`); for each pair with `|r| ≥ 0.95` drop the lower-AUC column.
* Protect list (never drop in 8.2): `TransactionAmt_log, card_velocity, amt_to_avg_card_ratio, transaction_hour, card1_freq`.

### 8.3 Importance-based selection (GPU XGBoost; fallback = univariate AUC)
* `xgboost.XGBClassifier(tree_method="hist", device="cuda", n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, scale_pos_weight=neg/pos, eval_metric="aucpr", early_stopping_rounds=30, random_state=42)`.
* Fit on **train**, early-stop on **validation**. Use `DMatrix`/`QuantileDMatrix` from float32 cupy to keep VRAM < 2 GB (reduce `max_bin` to 128 if needed).
* Use **gain** importance. Rank; keep the top `K` features subject to protect list; also record the cumulative-gain curve.
* If `xgboost` is unavailable or fails on VRAM: rank by univariate AUC from 8.2 and keep top `K`. Log which path ran.
* This model is *only* a selector. Do not save it as a deliverable and do not evaluate on test/monitor.

### 8.4 Drift-aware screen (report + optional drop)
* PSI per candidate feature between train and **validation**, 10 train-quantile bins, `eps=1e-6`:

```python
def psi(train_x, other_x, bins=10, eps=1e-6):
    qs = cp.unique(cp.nanquantile(train_x, cp.linspace(0, 1, bins + 1)))
    qs[0], qs[-1] = -cp.inf, cp.inf
    a = cp.histogram(train_x, qs)[0] / len(train_x) + eps
    b = cp.histogram(other_x, qs)[0] / len(other_x) + eps
    return float(cp.sum((a - b) * cp.log(a / b)))
```
* Report PSI > 0.25 as "high drift". Config `DRIFT_FILTER = False` by default: **the paper studies drift on unmitigated features, so default is report-only.** If `True`, drop selected features with PSI(train→val) > 0.25 (never using test/monitor).

### 8.5 PCA track for V-blocks
* Group the surviving V-features by **identical train null-count** (same-block columns share it); within a block, standardize on train (mean/std).
* `cuml.decomposition.PCA(n_components=min(10, n_cols), svd_solver="full")` per block, fit on train (float32, column-major cupy). Keep the smallest `n` with cumulative explained variance ≥ **0.95**, hard cap 5 per block; name `Vpca_<block>_<i>`.
* Total V-PCA components should be ≤ ~20; if more, raise the variance threshold down to 0.90.
* Transform val/test/monitor with the *train-fitted* components. Persist components, means, stds (`artifacts/pca_V.npz`).
* Track B = Track A non-V features + V-PCA components (then cap at 64; drop lowest-importance non-protected if over).

### 8.6 Final checks per track
* No nulls, no inf; dtype float32 (flags int8 may be cast to float32 at export).
* Max pairwise |corr| on train < 0.95 for Track A.
* Feature names unique; saved order = column order of parquet files.

---

## 9. Outputs (exact filenames)

```
data/processed/
  meta_{train,val,test,monitor}.parquet        # TransactionID, TransactionDT, isFraud, split, window_id, card_key_id, transaction_month, transaction_year
  features_sel_{train,val,test,monitor}.parquet
  features_pca_{train,val,test,monitor}.parquet
artifacts/
  preprocess_state.json     # drop lists, thresholds, quantiles, skew list, medians, anchors, config used, versions
  encoders.json, encoders_counts.npz
  scaler.json
  pca_V.npz
  feature_manifest.json     # per feature: name, source, transform chain, track membership, importance rank, PSI, missing_rate
reports/preprocess_report.md
```

Row order in `features_*` and `meta_*` is identical (same time-sorted order); `len(features)==len(meta)` per split. Parquet written with `cudf.to_parquet` (or via pandas if a write error occurs, split by column blocks).

Add to `.gitignore`: `data/raw/`, `data/interim/`, `data/processed/`, `artifacts/*.npz`. (Small JSONs may be committed.)

---

## 10. Code layout and CLI

```
scripts/preprocess/
  __init__.py
  config.py            # dataclass Config (SEED, K, thresholds, flags in this doc)
  gpu_env.py           # init_gpu, free_gpu, vram
  check_env.py
  io_utils.py          # csv_to_parquet, load, save_state
  s1_split.py … s9_reduce.py   # one file per stage; each exposes run(cfg, state)
  run.py               # python -m scripts.preprocess.run --stage all|S3 --sample 20000 --force
  validate.py          # all assertions in Sec. 11
  test_helpers.py      # unit tests for encode/freq/causal-card/knn/psi on toy data
```

Config defaults: `SEED=42, TRAIN_N=273000, VAL_N=58500, TEST_N=58500, DROP_MISSING=0.90, FLAG_MIN=0.05, KNN_MAX=0.50, KNN_K=5, KNN_REF=50000, KNN_BATCH=20000, WINSOR=(0.01,0.99), SKEW_ABS=2.0, CORR_THR=0.95, K_SEL=48, PCA_VAR=0.95, USE_D1_UID=True, EXCLUDE_TIME_PROXIES=True, DRIFT_FILTER=False`.

`--sample N` for smoke runs: take the first `N` rows after sorting (so the time structure is preserved) and scale `TRAIN_N/VAL_N/TEST_N` proportionally (0.4623 / 0.0990 / 0.0990). The sample must run S1→S10 end-to-end in < 3 minutes.

---

## 11. Validation (`validate.py`) and the report

Hard assertions (fail = stop):

1. Merged rows = 590,540 (sample runs: = N); `TransactionID` unique.
2. Split sizes as in Sec. 4; time order monotone within and across splits; `window_id` correct.
3. No `object`/string columns in any `features_*` file; all float32.
4. No null/inf/NaN in any `features_*` file.
5. Same feature list, same order, in all four splits of a track.
6. **Leakage test:** re-run S6–S8 fit on train only after *deleting* val/test/monitor rows from memory; the saved quantiles/medians/scaler stats must be bit-identical (or within 1e-6) to the full run.
7. **Causality test:** for 1,000 random rows, recompute `card_prior_txn_count` and `amt_to_avg_card_ratio` with a slow pandas groupby on rows with strictly earlier (time, row) and match.
8. Train: scaled continuous medians ≈ 0; Track A max |corr| < 0.95.
9. Peak VRAM logged and < 5 GiB.

Report (`reports/preprocess_report.md`) must include: library versions; per-stage runtime and peak VRAM; split table with fraud rates vs. paper; dropped columns by reason (counts + names for non-V); imputation counts by method; number of unique missing flags kept; Track A and Track B feature lists with importance rank and PSI; top-10 PSI features; list of **deviations from the paper** (below); any warnings.

### Known deviations from the paper (state them in the report)

| Topic | Paper | This implementation | Why |
|---|---|---|---|
| Split | normalized-DT boundaries | row-count boundaries matching paper Table 1 node counts | exact boundaries are not reproducible without their scaler |
| KNN imputation | sklearn-style KNN | cuML brute-force KNN on 24 anchor columns + 50k train reference | no GPU `KNNImputer`; memory |
| Order | impute → transform | winsorize + transform → impute | distance quality |
| Skew transform | Yeo-Johnson | signed-log1p (YJ optional) | determinism / API risk |
| `card_addr_hash` | raw hash feature | frequency-encoded | raw hash magnitude is meaningless |
| Velocity / ratio features | undefined | causal definitions in S3 | avoid future leakage |
| Time proxies | month/year listed | computed, excluded from features | cannot generalize under temporal split |
| 38 features | selected "by analysis" | data-driven K=48 (range 38–64) + PCA track | reproducible, reduces noise |
| SMOTE / graph | in paper | **out of scope** here | modeling stage |

---

## 12. Definition of Done (tick all)

- [ ] `check_env.py` prints `ENV OK`; versions recorded.
- [ ] S0–S10 run with `--sample 20000` without errors, then on full data.
- [ ] Peak VRAM < 5 GiB on full data; no use of managed memory.
- [ ] All Section 11 assertions pass.
- [ ] Outputs in Section 9 exist, aligned row-for-row with meta.
- [ ] `feature_manifest.json` and `reports/preprocess_report.md` written.
- [ ] Existing user files untouched; new code only under `scripts/preprocess/`; `.gitignore` updated.
- [ ] No model/graph/SMOTE code was added.

---

## 13. Pitfall cheat-sheet (read before coding)

| Symptom | Cause | Fix |
|---|---|---|
| `cudaErrorMemoryAllocation` / `std::bad_alloc` | float64, whole-frame copies, huge KNN batch | float32, in-place ops, column blocks, `free_gpu()`, smaller batch |
| Hang or crash after `managed_memory=True` | UVM oversubscription unsupported on WSL2 | never enable it |
| Rows misaligned after encoding | `merge` reordered rows, or index mismatch | use `encode_codes`/cupy gather; single `RangeIndex`; assert `_rid` order |
| `ValueError` on `df.values` | nulls present | `to_cupy(dtype="float32", na_value=nan)` |
| `fillna` "didn't work" | NaN ≠ null in cuDF | `.nans_to_nulls()` first |
| `fillna("unknown")` raises on float col | wrong order | `astype("str")` then `fillna` |
| `groupby.transform/shift` unsupported | older cuDF | fallbacks in S3 |
| `NearestNeighbors` dtype error | float64 or non-contiguous | `astype("float32")`, `cp.ascontiguousarray` |
| Output types switch (cudf vs cupy) | cuML default output type | wrap in `cuml.using_output_type("cupy")` |
| Slow disk / timeouts | data on `/mnt/c` | keep in WSL home |
| `object` columns reaching scaler/PCA | forgot to drop raw strings | assert dtypes at end of S4 |
| Different results between runs | no seed | `SEED=42` for cupy (`cp.random.seed`) and sampling |
| Selected features differ run to run | XGBoost non-determinism on GPU | fix seed, record chosen list in `feature_manifest.json`, treat that file as source of truth |

---

## 14. One-paragraph prompt to paste above this file in Antigravity

> Read `FEDBANK_preprocessing_spec.md` completely before writing code. Implement exactly the pipeline in it under `scripts/preprocess/`, preprocessing and dimensionality reduction only (no models, graphs, or SMOTE). Run the environment check first, then each stage on `--sample 20000`, then on the full data. Stop and report the full traceback if any assertion in Section 11 fails; never weaken a leakage rule to get past an error. When done, show `reports/preprocess_report.md` and the Definition-of-Done checklist.
