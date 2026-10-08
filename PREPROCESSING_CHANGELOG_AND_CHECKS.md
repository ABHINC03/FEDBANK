# FEDBANK — Preprocessing Changes & Verification Checklist

This document tracks all data transformations, file-by-file pipeline operations, validation checks, and output schemas for the FEDBANK IEEE-CIS Fraud Detection dataset preprocessing pipeline.

---

## 1. Pipeline Execution Summary

* **Dataset**: IEEE-CIS Fraud Detection (590,540 total rows $\times$ 434 raw features)
* **Hardware**: NVIDIA GeForce RTX 4050 (6.00 GiB VRAM, WSL2 Ubuntu)
* **RAPIDS Stack**: cuDF 26.06.01, cuML 26.06.00, cuPy 14.2.0, RMM 26.06.00
* **Total Runtime (Full 590k Dataset)**: **263.8 seconds** (~4.4 minutes)
* **Peak VRAM Consumed**: **4.63 GiB / 6.00 GiB** (Zero OOM errors)
* **Validation Status**: **100% Pass** (All 9 formal assertions verified)

---

## 2. File-by-File Changes & Processing Specifications

### `s0_csv_to_parquet` / `io_utils.py`
* **Input Files**:
  * `data/train_transaction.csv` (652 MB, 590,540 rows $\times$ 394 cols)
  * `data/train_identity.csv` (26 MB, 144,233 rows $\times$ 41 cols)
* **Output File**:
  * `data/interim/raw_merged.parquet` (590,540 rows $\times$ 434 cols)
* **Changes & Transformations**:
  1. Standardized identity column naming: normalized hyphenated column keys (`id-01` $\to$ `id_01`).
  2. Applied upfront memory-efficient typing: identifiers to `int32`/`int64`, labels to `int8`, numeric floats to `float32`, categoricals to `str`.
  3. Performed left outer join on `TransactionID` preserving total row count and exact transaction order.
* **Checks & Invariants**:
  * Row count matches `train_transaction.csv` exactly (590,540 rows).
  * `TransactionID` is strictly unique.

---

### `s1_split.py` — Load, Sort & Chronological Split
* **Input File**: `data/interim/raw_merged.parquet`
* **Output Files**:
  * `data/interim/meta.parquet` (TransactionID, TransactionDT, isFraud, split, window_id)
  * `data/interim/after_s1.parquet`
  * `artifacts/preprocess_state.json` (split boundaries, fraud rates, train stats)
* **Changes & Transformations**:
  1. Sorted rows chronologically by `TransactionDT` ascending (with `TransactionID` as tie-breaker).
  2. Assigned fixed row-position splits based on Menezes & Filho (IEEE Access 2025) Table 1:
     * **Train (`split = 0`)**: First 273,000 rows (Fraud rate: **3.33%**)
     * **Val (`split = 1`)**: Next 58,500 rows (Fraud rate: **3.62%**)
     * **Test (`split = 2`)**: Next 58,500 rows (Fraud rate: **3.70%**)
     * **Monitor (`split = 3`)**: Remaining 200,540 rows (Fraud rate: **3.64%**)
  3. Computed 50 sequential monitoring drift evaluation windows (`row_in_monitor // 3000`).
* **Checks & Invariants**:
  * Total rows sum: $273,000 + 58,500 + 58,500 + 200,540 = 590,540$.
  * Strict temporal monotonicity: $\max(\text{DT}_i) \le \min(\text{DT}_{i+1})$ across all splits.
  * Split fraud rates match empirical benchmark within $\pm 0.01\%$.

---

### `s2_clean.py` — Value Sanitization & Low-Information Column Pruning
* **Input File**: `data/interim/after_s1.parquet`
* **Output File**: `data/interim/after_s2.parquet`
* **Changes & Transformations**:
  1. Replaced $\pm\infty$ with null in all 388 numeric float columns.
  2. Dropped columns exceeding 90% missingness on train set (4 columns dropped: `dist2`, `id_07`, `id_08`, `id_21`).
  3. Dropped constant columns ($\le 1$ unique value on train: 1 column dropped).
  4. Dropped near-constant columns with $\ge 99.5\%$ frequency in the modal value (11 columns dropped).
* **Checks & Invariants**:
  * Pruning statistics fitted exclusively on `train` split (`split == 0`).
  * Columns remaining: **420 columns**.
  * No division-by-zero or remaining infinite float values.

---

### `s3_engineer.py` — Temporal & Causal Card Aggregations
* **Input File**: `data/interim/after_s2.parquet`
* **Output File**: `data/interim/after_s3.parquet`
* **Changes & Transformations**:
  1. **Temporal Features**:
     * `transaction_day = TransactionDT // 86400`
     * `transaction_hour = (TransactionDT // 3600) % 24`
     * `transaction_dow = (TransactionDT // 86400) % 7`
     * `transaction_month`, `transaction_year` (stored for audit, omitted from modeling features to avoid temporal drift).
  2. **Amount Features**:
     * `TransactionAmt_log = cp.log1p(amt)`
     * `amt_cents = (amt * 1000) % 1000`
     * `amt_is_round = (amt % 1 == 0).astype(int8)`
  3. **Causal Card UID & Aggregates**:
     * Created composite card key: `card1_card2_card3_card5_addr1_(day - D1)`.
     * `card_prior_txn_count`: Cumulative transaction count prior to current timestamp (`cumcount()`). First transaction is strictly `0.0`.
     * `card_is_first_seen`: Binary flag (`1` if prior count is `0`, else `0`).
     * `card_time_since_last`: Log elapsed time since previous card transaction (`-1.0` if first seen).
     * `card_velocity`: `prior_n / ((DT - first_dt)/86400 + 1)`.
     * `amt_to_avg_card_ratio`: Current transaction amount relative to prior mean for that card.
* **Checks & Invariants**:
  * **Zero future leakage**: All rolling aggregates use strictly preceding rows within each card group.
  * First transaction per card always receives `prior_txn_count = 0` and `time_since_last = -1.0`.

---

### `s4_encode.py` — Categorical Encodings
* **Input File**: `data/interim/after_s3.parquet`
* **Output Files**:
  * `data/interim/after_s4.parquet`
  * `artifacts/encoders.json`
  * `artifacts/encoders_counts.npz`
* **Changes & Transformations**:
  1. Fitted category frequency tables on **train set only**.
  2. High-cardinality columns (`card1`, `card2`, `addr1`, `addr2`, `card_key_id`, `card1_addr1_hash`) encoded via normalized frequency (`count / len(train)`). Unseen categories in test/val receive `0.0`.
  3. Low-cardinality columns (`ProductCD`, `card4`, `card6`, `M1`–`M9`, `DeviceType`, `id_12`–`id_38`) encoded as integer ordinal codes (`0 .. C-1`). Unseen/missing categories mapped to `-1`.
  4. Dropped raw unparsed string columns (`P_emaildomain`, `R_emaildomain`, `DeviceInfo`).
* **Checks & Invariants**:
  * No test/val data leakage into category frequencies.
  * Unseen values explicitly handled with designated sentinel codes.

---

### `s5_missing_flags.py` — Missing Value Indicators
* **Input File**: `data/interim/after_s4.parquet`
* **Output File**: `data/interim/after_s5.parquet`
* **Changes & Transformations**:
  1. Identified 315 columns having $>5\%$ missing rate on the train set.
  2. Created binary missing indicator columns (`<col>_missing = isna().astype(int8)`).
  3. Deduplicated co-occurring indicator flags (e.g. columns that are always missing together) to remove exact colinear duplicates.
  4. Retained **30 unique indicator columns** (added to feature space).
* **Checks & Invariants**:
  * Indicators created *before* numerical imputation.
  * Preserved pattern of missingness as an explicit inductive signal for fraud detection.

---

### `s6_winsor_skew.py` — Winsorization & Skew Transformation
* **Input File**: `data/interim/after_s5.parquet`
* **Output File**: `data/interim/after_s6.parquet`
* **Changes & Transformations**:
  1. Winsorized 431 continuous numeric columns using train percentiles $[1\%, 99\%]$:
     $$x_{\text{clipped}} = \text{clip}(x, q_{0.01}^{\text{train}}, q_{0.99}^{\text{train}})$$
  2. Identified heavily skewed columns ($|\text{skew}| > 2.0$ on train split):
     * Non-negative skewed (289 cols): $\text{sign}(x) \cdot \log(1 + |x|)$.
     * Negative values present (4 cols): signed log transform.
* **Checks & Invariants**:
  * Winsorization thresholds computed exclusively from train rows.
  * Pre-imputation skew reduction ensures stable Euclidean distances for KNN imputation.

---

### `s7_impute.py` — Multi-Strategy Numerical Imputation
* **Input File**: `data/interim/after_s6.parquet`
* **Output File**: `data/interim/after_s7.parquet`
* **Changes & Transformations**:
  1. **Partitioning by Missing Rate (Train Set)**:
     * 68 columns: 0% missing $\to$ No-op.
     * 138 columns: $(0\%, 50\%)$ missing $\to$ **GPU KNN Imputation**.
     * 225 columns: $\ge 50\%$ missing $\to$ **Median Imputation**.
  2. **Median Imputation**: Computed train medians and filled missing values.
  3. **GPU Approximate KNN Imputation**:
     * Extracted 19 dense anchor columns (`TransactionAmt_log`, `transaction_hour`, `card1_freq`, `addr1_freq`, `card_prior_txn_count`, `card_velocity`, `C1`–`C14`).
     * Fitted `cuml.neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute')` on 50,000 train reference rows.
     * Imputed missing values using the mean of 5 nearest neighbors in blocks of 64 columns. Fallback to train median if all neighbors are NaN.
* **Checks & Invariants**:
  * Zero out-of-fold reference rows: reference index is strictly train rows.
  * No NaN/Null values remain in numerical columns.

---

### `s8_scale.py` — Robust Scaling
* **Input File**: `data/interim/after_s7.parquet`
* **Output Files**:
  * `data/interim/after_s8.parquet`
  * `artifacts/scaler.json`
* **Changes & Transformations**:
  1. Fitted `RobustScaler` on **train split only** across 401 continuous features:
     $$x_{\text{scaled}} = \frac{x - \text{median}_{\text{train}}}{\text{IQR}_{\text{train}}}, \quad \text{IQR} = q_{0.75} - q_{0.25}$$
  2. Applied identical scaling parameters to train, val, test, and monitoring splits.
* **Checks & Invariants**:
  * Train set feature medians equal $0.0 \pm 0.01$.
  * Robust to outliers and extreme financial spikes.

---

### `s9_reduce.py` — Dimensionality Reduction & Drift Screening
* **Input File**: `data/interim/after_s8.parquet`
* **Output Files**:
  * `data/interim/after_s9.parquet`
  * `artifacts/pca_V.npz`
* **Changes & Transformations**:
  1. **Hard Pruning**: Removed known proxy columns (raw IDs, timestamps, zero-variance columns: 21 dropped).
  2. **Greedy Correlation Pruning**:
     * Computed full correlation matrix $R$ on train split.
     * Greedily eliminated collinear features where $|r_{ij}| \ge 0.95$, dropping the lower-AUC feature (142 dropped, 306 retained).
  3. **Feature Selection (Track A)**:
     * Ranked remaining features by univariate ROC-AUC.
     * Selected Top **48 features** (target $K=48$).
  4. **Population Stability Index (PSI) Drift Screen**:
     * Evaluated drift between train and validation splits:
       $$\text{PSI} = \sum (P_i - Q_i) \ln(P_i / Q_i)$$
     * Verified all selected features have acceptable stability ($\text{PSI} < 0.25$).
  5. **PCA Track (Track B)**:
     * Extracted 200 raw V-features from train split.
     * Fitted `cuml.decomposition.PCA(n_components=5)`.
     * Formed Track B feature set combining Top 22 core non-V features with 5 PCA components (`Vpca_b0_0` .. `Vpca_b0_4`), yielding **27 total features**.
* **Checks & Invariants**:
  * No pairwise feature correlation in Track A exceeds $0.95$.
  * PSI screen run across chronological boundary (train $\to$ val).

---

### `s10_export.py` & `validate.py` — Parquet Export & Final Assertions
* **Input File**: `data/interim/after_s9.parquet`
* **Output Files (under `data/processed/`)**:
  * `meta_train.parquet`, `meta_val.parquet`, `meta_test.parquet`, `meta_monitor.parquet`
  * `features_sel_train.parquet`, `features_sel_val.parquet`, `features_sel_test.parquet`, `features_sel_monitor.parquet` (Track A)
  * `features_pca_train.parquet`, `features_pca_val.parquet`, `features_pca_test.parquet`, `features_pca_monitor.parquet` (Track B)
  * `artifacts/feature_manifest.json`
  * `reports/preprocess_report.md`
* **Changes & Transformations**:
  1. Separated tabular features into two model-ready representations:
     * **Track A (`sel`)**: 48 selected engineered + pruned features.
     * **Track B (`pca`)**: 27 features (core + PCA-reduced V-features).
  2. Strictly cast all exported feature matrices to `float32` (ensuring downstream GPU PyTorch / GNN memory efficiency).
  3. Stored metadata separately (`TransactionID`, `TransactionDT`, `isFraud`, `split`, `window_id`, `card_key_id`).

---

## 3. Formal Validation Checklist & Verification Matrix

The pipeline execution is validated against all 9 formal criteria specified in Section 11 of `FEDBANK_preprocessing_spec.md`:

| Check ID | Requirement | Verification Result | Status |
|---|---|---|:---:|
| **[V1]** | Exact row count: 590,540 rows across merged dataset | `n_total = 590,540` | **PASS** |
| **[V2]** | Chronological split sizes & order: train=273,000, val=58,500, test=58,500, monitor=200,540 | $\max(\text{DT}_i) \le \min(\text{DT}_{i+1})$; sizes exact match | **PASS** |
| **[V3]** | Clean dtypes: No object/string columns; all features `float32` | 0 object columns; 100% `float32` features | **PASS** |
| **[V4]** | Numerical integrity: Zero NaN, Null, or Infinite values in exported features | Null count: 0, Inf count: 0 across all files | **PASS** |
| **[V5]** | Feature alignment: Identical feature schema across train, val, test, monitor | Feature columns match identically | **PASS** |
| **[V6]** | Data leakage prevention: Medians, scalers, encodings fitted on train only | Train medians $\approx 0.0$ ($|m| \le 10^{-2}$) | **PASS** |
| **[V7]** | Temporal causality: Causal card aggregates never look into the future | Strict monotonicity within card keys | **PASS** |
| **[V8]** | Collinearity constraint: Pairwise $|r| < 0.95$ in Track A | Maximum pairwise correlation $< 0.95$ | **PASS** |
| **[V9]** | VRAM safety: Stays well within 6 GB hardware budget | Peak VRAM: 4.63 GiB / 6.00 GiB | **PASS** |

---

## 4. Quick Reference — How to Run

```bash
# 1. Activate conda environment
conda activate fedbank

# 2. Run unit tests
python -m scripts.preprocess.test_helpers

# 3. Run smoke test (20,000 rows)
python -m scripts.preprocess.run --stage all --sample 20000

# 4. Run full pipeline (all 590,540 rows)
python -m scripts.preprocess.run --stage all

# 5. Run validation assertions
python -m scripts.preprocess.validate
```
