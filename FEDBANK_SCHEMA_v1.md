# FEDBANK Feature Schema v1 (frozen)

Single contract that every bank (client), the server, and the centralized baseline must follow.
Machine-readable copy: `fedbank_schema_v1.json`. Any change = new version (`v2`), never an in-place edit.

## 1. What is fixed (identical everywhere)
- **Features:** the 48 Track A features below, in exactly this column order.
- **Dtype:** float32 for every feature. Label: `isFraud` (int8).
- **Missing handling:** no NaN may reach the model. After imputation, zero NaNs (assert at export).
- **Categorical vocabularies:** fixed lists (below / in JSON). Unseen or missing -> `-1`.
- **Post-scaling clip:** [-10, 10].
- **Split rule:** train/val/test by `TransactionDT` cutoffs, the same cutoffs in all banks. Take them from the centralized run: `min/max TransactionDT` per split in `data/processed/meta_*.parquet`.
- **Partition key:** `card_key_id`. All rows of one card live in one bank, so the causal card features match the centralized run.
- **Model input dim:** 48, first op in the model is `LayerNorm(48)`.

## 2. What is fitted locally (per bank, on that bank's train rows only)
| Step | Rule |
|---|---|
| Winsorize (1%/99%) | local quantiles |
| Skew transform | signed-log1p where local |skew| > 2 (same rule, local decision) |
| Imputation | local median + local KNN (reference rows only from this bank); missing flag `addr1_missing` kept |
| RobustScaler | local median / IQR, then clip [-10, 10] |
| `card1_freq` | local relative frequency: count in bank train / bank train size, then log1p |
| Class weights / focal alpha | local fraud rate |

Never shared: raw rows, features, graph edges, labels, local statistics. Shared: model weights/updates only.

## 3. Design-time assumptions (state these in the report)
1. The 48-feature list was chosen on the **pooled** centralized train data (univariate AUC, correlation pruning, PSI screen). This is a one-time design decision, not a runtime exchange.
2. Categorical vocabularies are public, small, and non-private (device type, id_* codes).
3. PCA (Track B) is excluded from the federated path: it is a fitted projection tied to pooled data. Federated PCA is future work.
4. Ablation: compare this local-fit setting against a shared-statistics setting (global scaler/medians) to measure the cost of not sharing.

## 4. Columns (fixed order)
| # | Feature | Group | Notes |
|---|---|---|---|
| 0 | `addr1_missing` | binary_flag | binary flag: addr1 is null |
| 1 | `email_match` | binary_flag | binary |
| 2 | `DeviceType` | categorical_ordinal | vocab 4 values, unseen=-1 |
| 3 | `id_12` | categorical_ordinal | vocab 3 values, unseen=-1 |
| 4 | `id_16` | categorical_ordinal | vocab 4 values, unseen=-1 |
| 5 | `id_18` | categorical_ordinal | vocab 18 values, unseen=-1 |
| 6 | `id_24` | categorical_ordinal | vocab 10 values, unseen=-1 |
| 7 | `id_34` | categorical_ordinal | vocab 6 values, unseen=-1 |
| 8 | `id_35` | categorical_ordinal | vocab 4 values, unseen=-1 |
| 9 | `id_36` | categorical_ordinal | vocab 4 values, unseen=-1 |
| 10 | `id_38` | categorical_ordinal | vocab 4 values, unseen=-1 |
| 11 | `TransactionAmt_log` | engineered_local | log1p(TransactionAmt) |
| 12 | `amt_to_avg_card_ratio` | engineered_local | amt / mean(prior amt), clip[0,100], 1.0 if first seen |
| 13 | `card_velocity` | engineered_local | prior_n / ((DT-first_dt)/86400+1) |
| 14 | `transaction_hour` | engineered_local | int 0-23 (cyclical optional) |
| 15 | `card1_freq` | frequency_local |  |
| 16 | `C1` | count_C |  |
| 17 | `C12` | count_C |  |
| 18 | `C2` | count_C |  |
| 19 | `C4` | count_C |  |
| 20 | `C8` | count_C |  |
| 21 | `card3` | raw_numeric |  |
| 22 | `V168` | vesta_V |  |
| 23 | `V203` | vesta_V |  |
| 24 | `V217` | vesta_V |  |
| 25 | `V218` | vesta_V |  |
| 26 | `V219` | vesta_V |  |
| 27 | `V229` | vesta_V |  |
| 28 | `V230` | vesta_V |  |
| 29 | `V232` | vesta_V |  |
| 30 | `V257` | vesta_V |  |
| 31 | `V258` | vesta_V |  |
| 32 | `V263` | vesta_V |  |
| 33 | `V264` | vesta_V |  |
| 34 | `V274` | vesta_V |  |
| 35 | `V283` | vesta_V |  |
| 36 | `V294` | vesta_V |  |
| 37 | `V303` | vesta_V |  |
| 38 | `V31` | vesta_V |  |
| 39 | `V34` | vesta_V |  |
| 40 | `V40` | vesta_V |  |
| 41 | `V45` | vesta_V |  |
| 42 | `V50` | vesta_V |  |
| 43 | `V52` | vesta_V |  |
| 44 | `V60` | vesta_V |  |
| 45 | `V72` | vesta_V |  |
| 46 | `V81` | vesta_V |  |
| 47 | `V93` | vesta_V |  |

Engineered formulas are causal (use a card's own past only) and need no fitting.

## 5. Validation every bank must pass before round 1
- Column names and order equal `fedbank_schema_v1.json`.
- Shape `(N, 48)`, dtype float32, no NaN/inf, all values within [-10, 10] (binary/ordinal excluded from scaling).
- Ordinal columns contain only vocab codes and `-1`.
- Bank has >= some minimum fraud rows in train (set a threshold, e.g. 100) so local loss is defined.
- `schema_version` is stored in each exported parquet's metadata and in every Flower message; the server rejects a mismatch.
