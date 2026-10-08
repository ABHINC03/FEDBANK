# FedBank — IEEE-CIS Fraud Detection with GNN + Federated Learning

A research implementation based on Menezes & Filho, *"Investigating the Robustness of GNNs to Data Drift: A Case Study on Financial Transaction Data"*, IEEE Access 2025.

## Preprocessing Pipeline

All preprocessing and dimensionality reduction code is under `scripts/preprocess/`.
See [`FEDBANK_preprocessing_spec.md`](FEDBANK_preprocessing_spec.md) for the full specification.

### Quick Start

```bash
conda activate fedbank   # or fedbank_rapids

# Step 1: Verify environment
python -m scripts.preprocess.check_env

# Step 2: Smoke test (20k rows, ~3 min)
python -m scripts.preprocess.run --stage all --sample 20000

# Step 3: Full run
python -m scripts.preprocess.run --stage all

# Step 4: View report
cat reports/preprocess_report.md
```

### Pipeline Stages

| Stage | Description | Output |
|---|---|---|
| S0 | CSV → Parquet (CPU) | `data/interim/raw_merged.parquet` |
| S1 | Load, sort, split | meta + state |
| S2 | Clean (inf, extremes, drop cols) | — |
| S3 | Feature engineering (temporal, card, amount) | — |
| S4 | Categorical encoding (train-fitted vocab) | `artifacts/encoders.json` |
| S5 | Missing-value flags (before imputation) | — |
| S6 | Winsorize + skew transform | — |
| S7 | Imputation (KNN + median, train-fitted) | — |
| S8 | RobustScaler (train-fitted) | `artifacts/scaler.json` |
| S9 | Dim reduction: prune → corr → importance → PCA | `artifacts/pca_V.npz` |
| S10 | Export parquet files + feature manifest | `data/processed/`, `artifacts/feature_manifest.json` |

### Outputs

```
data/processed/
  meta_{train,val,test,monitor}.parquet
  features_sel_{train,val,test,monitor}.parquet   # Track A (K=48 features)
  features_pca_{train,val,test,monitor}.parquet   # Track B (non-V + V-PCA)
artifacts/
  preprocess_state.json
  encoders.json, encoders_counts.npz
  scaler.json
  pca_V.npz
  feature_manifest.json
reports/preprocess_report.md
```

### Data

Place Kaggle IEEE-CIS CSVs in `data/`:
- `data/train_transaction.csv`
- `data/train_identity.csv`

(Test CSVs are ignored — they are unlabeled.)

## Environment Setup

See [`SETUP.md`](SETUP.md) for full WSL2 + Conda + RAPIDS setup instructions.

> **Note (preprocessing pipeline):** The three separate `conda install` commands in SETUP.md (cuml / cudf / cugraph) may produce mismatched RAPIDS builds. If `check_env.py` fails, recreate the environment with a single `conda create` using the RAPIDS selector at <https://docs.rapids.ai/install> (Conda, WSL2, Python 3.11, CUDA 12, latest stable). See `FEDBANK_preprocessing_spec.md` Section 1.2 for the exact pattern.
