# FEDBANK — Graph Construction & GNN Baseline Report

This report documents the execution, validation checks, edge calibration, and model baseline results for Phase 1 of FEDBANK.

---

## 1. System & Environment Audit

- **Operating System:** Ubuntu under WSL2 (Linux 6.6)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB total VRAM)
- **PyTorch:** `2.6.0+cu126` with CUDA 12.6 support verified (`torch.cuda.is_available() == True`)
- **PyTorch Geometric (PyG):** `2.6.1` installed and verified
- **Python:** Python 3.11 (`/home/abhin/miniconda3/envs/fedbank/bin/python`)
- **Schema Reference:** `fedbank_schema_v1.json` / `FEDBANK_SCHEMA_v1.md` (frozen, 48 features)

---

## 2. Processed Data Audit & Schema Check

Before graph construction, all parquet files in `data/processed/` were audited against `fedbank_schema_v1.json`:
1. **Column Match:** All 48 features specified in the schema are present in `features_sel_{split}.parquet`.
2. **Missing / Extra Columns:** Zero missing features, zero extra features.
3. **Column Order:** Columns in raw exported parquet were reordered in-memory to strictly match `fedbank_schema_v1.json` column sequence.
4. **V140 & V261 Absence:** Confirmed that `V140` and `V261` are not among the 48 schema features (as anticipated in spec Section 3 G2d). Per spec, Stage G2d anomaly builder selected the top-2 schema features with highest univariate ROC-AUC on train split only: `C1` and `C2`.
5. **Metadata Columns:** `['TransactionID', 'TransactionDT', 'isFraud', 'split', 'window_id', 'card_key_id', 'transaction_month', 'transaction_year']`.

---

## 3. Stage G1 — Node Features Validation

### Track A: Selected Features (48 dims)

| Split | Rows ($N$) | Features ($D$) | Fraud Count | Fraud Rate | Paper Target | Finite (no NaN/inf) | No Leakage | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 273,000 | 48 | 9,091 | 3.33% | 3.33% | PASS | PASS | **PASS** |
| **Validation** | 58,500 | 48 | 2,115 | 3.62% | 3.62% | PASS | PASS | **PASS** |
| **Test** | 58,500 | 48 | 2,164 | 3.70% | 3.70% | PASS | PASS | **PASS** |
| **Monitoring** | 200,540 | 48 | 7,293 | 3.64% | 3.64% | PASS | PASS | **PASS** |

---

## 4. Stage G2 & G3 — Edge Calibration and Graph Assembly

All 4 edge builders were implemented and assembled into homogeneous graphs:
- **Similarity edges:** Cosine similarity > 0.8, max 5 edges/node (GPU chunked).
- **Temporal edges:** Sliding window within 7,200s (2 hrs), max 5 edges/node (vectorized).
- **Cluster edges:** `MiniBatchKMeans` (k=3, batch=1024) fitted on train only, max 5 edges/node, 10,000 edges/cluster cap.
- **Anomaly edges:** Cosine similarity > 0.75 on top-2 univariate AUC anomaly features (`C1`, `C2` with train z-scores), max 3 edges/node.

### Edge Counts vs Paper Targets

| Split | Nodes ($N$) | Undirected Edges ($E$) | Paper Target ($E$) | Target Match Ratio | Avg Degree | File Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 273,000 | **3,159,849** | ~3,170,000 | **99.68%** | 23.15 | 198.0 MB |
| **Validation** | 58,500 | **699,965** | ~641,000 | **109.20%** | 23.93 | 43.5 MB |
| **Test** | 58,500 | **700,845** | ~644,000 | **108.83%** | 23.96 | 43.5 MB |
| **Monitoring** | 200,540 | **3,049,955** | ~1,810,000 | Subgraph slices | 30.42 | 178.9 MB |

### Train Edge Composition by Type (Pre-deduplication)
- **Similarity:** 950,359 undirected edges (30.08%)
- **Temporal:** 1,364,979 undirected edges (43.20%)
- **Cluster:** 30,000 undirected edges (0.95%)
- **Anomaly:** 818,013 undirected edges (25.89%)

---

## 5. Stage G4 — Formal Graph Validation & Homophily Analysis

### Pass/Fail Table

| Split | Nodes ($N$) | Undirected Edges | Symmetric | No Loops / Dup | Isolated Frac | Degree (p50/p90/p99) | Homophily Lift | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 273,000 | 3,159,849 | PASS | PASS | 0.00% | 20 / 23 / 26 | 4.82x | **PASS** |
| **Val** | 58,500 | 699,965 | PASS | PASS | 0.00% | 21 / 24 / 28 | 4.39x | **PASS** |
| **Test** | 58,500 | 700,845 | PASS | PASS | 0.00% | 21 / 24 / 28 | 3.26x | **PASS** |
| **Monitor (50w)**| 200,540 | 3,049,955 | PASS | PASS | 0.00% | 30 / 34 / 39 | 3.01x | **PASS** |

### Homophily Analysis (Train Split)
- **Base Random Fraud-Fraud Rate ($p^2$):** 0.1109%
- **Actual Graph Overall FF Rate:** 0.5344% (**4.82x lift** over random baseline)
  - `similarity`: 1.1509% FF rate (**10.38x lift**) — strongly connects fraudulent transactions
  - `anomaly`: 0.4119% FF rate (**3.71x lift**) — significant fraud concentration
  - `temporal`: 0.1869% FF rate (**1.69x lift**) — positive homophily
  - `cluster`: 0.1134% FF rate (**1.02x lift**) — neutral homophily
- **Verdict:** All edge types show homophily at or above random wiring. No edge type adds destructive noise.
