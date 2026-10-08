# FEDBANK Preprocessing Report

Generated: 2026-10-08 23:21  |  Sample: full

## Library Versions

| Library | Version |
|---|---|
| cuDF | 26.06.01 |
| cuML | 26.06.00 |
| cuPy | 14.2.0 |
| RMM  | 26.06.00 |

## Split Table

| Split | Rows | Expected | Fraud Rate | Paper Ref |
|---|---|---|---|---|
| train | 273000 | 273000 | (see pipeline output) | 3.33% |
| val | 58500 | 58500 | (see pipeline output) | 3.62% |
| test | 58500 | 58500 | (see pipeline output) | 3.70% |
| monitor | 200540 | 200540 | (see pipeline output) | 3.64% |

## Stage Runtimes

| Stage | Time (s) |
|---|---|
| S1 | 4.0 |
| S2 | 102.9 |
| S3 | 1.0 |
| S4 | 5.4 |
| S5 | 1.3 |
| S6 | 15.9 |
| S7 | 14.5 |
| S8 | 10.9 |
| S9 | 20.6 |
| S10 | 3.0 |

## Dropped Columns

| Reason | Count |
|---|---|
| Missing rate > 0.9 | 4 |
| Constant (nunique ≤ 1) | 1 |
| Near-constant (top share ≥ 0.995) | 11 |
| Hard prune (S9) | 21 |
| Correlation prune (S9) | 142 |

## Imputation Counts

| Method | Columns |
|---|---|
| No-op (0% missing) | (remainder) |
| KNN (k=5) | 138 |
| Median | 225 |

KNN anchor columns (19):
`TransactionAmt_log, transaction_hour, card1_freq, addr1_freq, card_prior_txn_count, card_velocity, C1, C2, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14`

## Missing Flags

Original flaggable columns: 315
Unique flags kept: 30

## Track A — 48 Features (sel)

| Rank | Feature | Importance Gain | PSI(train→val) |
|---|---|---|---|
| 1 | TransactionAmt_log | 0.5016 | 0.0165 |
| 2 | card_velocity | 0.6434 | 0.0134 |
| 3 | amt_to_avg_card_ratio | 0.5265 | 0.0263 |
| 4 | transaction_hour | 0.5050 | 0.0019 |
| 5 | card1_freq | 0.5242 | 0.0049 |
| 6 | C4 | 0.7030 | 0.0639 |
| 7 | C8 | 0.6885 | 0.0954 |
| 8 | email_match | 0.6841 | 0.0000 |
| 9 | C12 | 0.6818 | 0.0250 |
| 10 | V52 | 0.6777 | 0.0023 |
| 11 | id_35 | 0.6741 | 0.1021 |
| 12 | V50 | 0.6725 | 0.0000 |
| 13 | V93 | 0.6716 | 0.0000 |
| 14 | card3 | 0.6714 | 0.0000 |
| 15 | V218 | 0.6702 | 0.0000 |
| 16 | V303 | 0.6693 | 0.0923 |
| 17 | V40 | 0.6690 | 0.0145 |
| 18 | V264 | 0.6689 | 0.0000 |
| 19 | V219 | 0.6679 | 0.0000 |
| 20 | V81 | 0.6650 | 0.0160 |
| 21 | id_24 | 0.6645 | 0.1022 |
| 22 | C2 | 0.6643 | 0.0114 |
| 23 | V258 | 0.6640 | 0.0000 |
| 24 | id_34 | 0.6639 | 0.1360 |
| 25 | id_16 | 0.6635 | 0.1078 |
| 26 | DeviceType | 0.6632 | 0.1021 |
| 27 | V203 | 0.6615 | 0.0000 |
| 28 | V168 | 0.6612 | 0.0000 |
| 29 | V72 | 0.6609 | 0.0000 |
| 30 | V229 | 0.6595 | 0.0000 |
| 31 | V283 | 0.6590 | 0.0049 |
| 32 | V217 | 0.6589 | 0.0000 |
| 33 | id_36 | 0.6587 | 0.1021 |
| 34 | V34 | 0.6584 | 0.0000 |
| 35 | id_12 | 0.6583 | 0.1021 |
| 36 | addr1_missing | 0.6580 | 0.0000 |
| 37 | V263 | 0.6578 | 0.0000 |
| 38 | V60 | 0.6563 | 0.0060 |
| 39 | id_38 | 0.6554 | 0.1021 |
| 40 | id_18 | 0.6551 | 0.1022 |
| 41 | V230 | 0.6549 | 0.0000 |
| 42 | V232 | 0.6545 | 0.0000 |
| 43 | V257 | 0.6542 | 0.0000 |
| 44 | V274 | 0.6541 | 0.0000 |
| 45 | C1 | 0.6538 | 0.0059 |
| 46 | V45 | 0.6529 | 0.0000 |
| 47 | V31 | 0.6525 | 0.0000 |
| 48 | V294 | 0.6521 | 0.0013 |

## Track B — 27 Features (pca)

| # | Feature |
|---|---|
| 1 | TransactionAmt_log |
| 2 | card_velocity |
| 3 | amt_to_avg_card_ratio |
| 4 | transaction_hour |
| 5 | card1_freq |
| 6 | C4 |
| 7 | C8 |
| 8 | email_match |
| 9 | C12 |
| 10 | id_35 |
| 11 | card3 |
| 12 | id_24 |
| 13 | C2 |
| 14 | id_34 |
| 15 | id_16 |
| 16 | DeviceType |
| 17 | id_36 |
| 18 | id_12 |
| 19 | addr1_missing |
| 20 | id_38 |
| 21 | id_18 |
| 22 | C1 |
| 23 | Vpca_b0_0 |
| 24 | Vpca_b0_1 |
| 25 | Vpca_b0_2 |
| 26 | Vpca_b0_3 |
| 27 | Vpca_b0_4 |

## Top-10 PSI Features (train→val)

| Feature | PSI |
|---|---|
| id_34 | 0.1360 |
| id_16 | 0.1078 |
| id_24 | 0.1022 |
| id_18 | 0.1022 |
| id_35 | 0.1021 |
| DeviceType | 0.1021 |
| id_36 | 0.1021 |
| id_12 | 0.1021 |
| id_38 | 0.1021 |
| C8 | 0.0954 |

## Deviations from the Paper

| Topic | Paper | This implementation | Why |
|---|---|---|---|
| Split | normalized-DT boundaries | row-count boundaries matching paper Table 1 | exact boundaries not reproducible without their scaler |
| KNN imputation | sklearn KNNImputer | cuML brute-force KNN on 24 anchor cols + 50000 train reference | no GPU KNNImputer; memory |
| Order | impute → transform | winsorize + transform → impute | distance quality under heavy tails |
| Skew transform | Yeo-Johnson | signed-log1p (YJ optional) | determinism / API risk |
| card_addr_hash | raw hash feature | frequency-encoded | raw hash magnitude is meaningless |
| Velocity / ratio | undefined | causal definitions in S3 | avoid future leakage |
| Time proxies | month/year listed | computed, excluded from features | cannot generalize under temporal split |
| 38 features | selected 'by analysis' | data-driven K=48 + PCA track | reproducible, reduces noise |
| SMOTE / graph | in paper | out of scope | modeling stage |

---
*Report generated automatically by `scripts/preprocess/run.py`.*