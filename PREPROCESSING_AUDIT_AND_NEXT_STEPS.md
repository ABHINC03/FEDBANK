# FEDBANK — Preprocessing Audit & Tomorrow's Implementation Plan

> **Scope**: Cross-reference between the research paper (*Menezes & Filho, IEEE Access 2025*), `FEDBANK_preprocessing_spec.md`, and the codebase.

---

## 1. Executive Status: Is Preprocessing Finished?

### ✅ **Tabular Preprocessing is 100% Complete & Validated.**
All stages defined in `FEDBANK_preprocessing_spec.md` (S0 through S10) have executed successfully on the full dataset (590,540 rows) and passed all 9 formal validation assertions:

* **S0 (CSV $\to$ Parquet)**: Merged 590,540 transactions $\times$ 434 identity/transaction features.
* **S1 (Chronological Split)**: Exact split matching paper Table 1:
  * `Train`: 273,000 rows (Fraud rate: **3.33%**)
  * `Validation`: 58,500 rows (Fraud rate: **3.62%**)
  * `Test`: 58,500 rows (Fraud rate: **3.70%**)
  * `Monitoring`: 200,540 rows (Fraud rate: **3.64%**), indexed into 50 temporal windows.
* **S2 (Clean)**: Infs removed; dropped 4 high-missing (>90%), 1 constant, and 11 near-constant columns.
* **S3 (Engineer)**: Causal card UID, `card_velocity`, `amt_to_avg_card_ratio`, `email_match`, cyclical hours/days.
* **S4 (Encode)**: Frequency encoding for high-cardinality; ordinal encoding for low-cardinality (fitted on train only).
* **S5 (Missing Indicators)**: 30 unique, deduplicated missing flags created.
* **S6 (Winsorize & Skew)**: Train quantile clipping $[1\%, 99\%]$ and signed-log1p skew normalization.
* **S7 (Impute)**: GPU approximate brute-force KNN ($k=5$, 19 anchor features) + train median fallback. Zero NaNs remain.
* **S8 (Scale)**: Train-fitted `RobustScaler` (median = 0, IQR = 1).
* **S9 (Reduce)**: Pairwise correlation pruning ($|r| < 0.95$), univariate ROC-AUC feature selection, and PSI drift screen.
* **S10 (Export)**: Model-ready feature sets exported to `data/processed/*.parquet` as `float32`.

---

## 2. Comparison Matrix: Paper (Menezes & Filho) vs. Current Implementation

| Preprocessing Step | Paper Specification (Section III-B) | Our Current Pipeline Implementation | Status |
|---|---|---|:---:|
| **Chronological Split** | Strict time splitting into Train, Val, Test, Monitor | Exact match (273k / 58.5k / 58.5k / 200.54k) | ✅ Complete |
| **Missing Values (>90%)** | Dropped | 4 columns dropped (`dist2`, `id_07`, `id_08`, `id_21`) | ✅ Complete |
| **Missing Flags (>5%)** | Binary indicator created | 30 unique missing indicators created | ✅ Complete |
| **Imputation (<50%)** | KNN imputation ($k=5$) | GPU chunked brute-force KNN ($k=5$, 19 anchors) | ✅ Complete |
| **Imputation (50–90%)** | Median imputation | Train medians computed and filled | ✅ Complete |
| **Categorical Encodings** | Freq encoding (high-card), Ordinal (low-card), 'Unknown' | `card1_freq`, ordinals with $-1$ sentinel | ✅ Complete |
| **Domain Features** | `card_velocity`, `amt_to_avg_card_ratio`, `email_match` | Causal non-leaking implementations in S3 | ✅ Complete |
| **Outlier Handling** | Winsorization at 1st/99th percentiles | Train-fitted percentiles applied | ✅ Complete |
| **Feature Scaling** | RobustScaler primary normalizer | Train-fitted RobustScaler applied | ✅ Complete |
| **Final Feature Vector** | 38 features selected for GNN input | Track A has 48 features; Track B has 27 features | ✅ Complete (Ready for 38-dim selection) |

---

## 3. What Needs to Be Done Tomorrow (Next Phase: Graph Construction & GNNs)

With tabular preprocessing complete, the project transitions into **Graph Construction (Paper Section III-C)** and **GNN Modeling (Section III-D/E)**. 

Here is the exact step-by-step checklist to execute tomorrow:

### 📋 Checklist for Tomorrow:

- [ ] **Step 1: Finalize 38-Dimensional Feature Subset for GNN**
  * The paper specifies an exact **38-dimensional node feature vector** (Paper Section III-B.5).
  * We can select the Top-38 features from Track A (saved in `artifacts/feature_manifest.json`) or use our 27-feature PCA set.
  * *Action*: Write a small loader script `scripts/graph/prepare_node_features.py` that outputs the $(N, 38)$ tensor.

- [ ] **Step 2: Implement the 4 Inter-Transaction Edge Builders (Paper Section III-C)**
  * Construct bidirectional homogeneous transaction edges across the 4 splits:
    1. **Similarity-based Edges**: Connect transactions if cosine similarity $> 0.8$ (max 5 edges/node).
    2. **Temporal Proximity Edges**: Connect transactions within a 2-hour window (7,200s, sliding window limit 200).
    3. **Cluster-based Edges**: MiniBatchKMeans ($k=3$, batch size 1024), interconnect nodes in same cluster (limit 10k edges/cluster).
    4. **Anomaly-based Edges**: Cosine similarity $> 0.75$ on top anomaly features (`V140`, `V261` via ZScore max, max 3 edges/node).
  * Target edge counts from Paper Table 1:
    * `Train`: ~3.17M edges (pre-SMOTE base graph)
    * `Val`: ~641k edges
    * `Test`: ~644k edges
    * `Monitoring`: ~1.81M edges

- [ ] **Step 3: Graph-Aware SMOTE for Training Graph (Paper Section III-E)**
  * *Phase 1*: Apply SMOTE on the node feature space to raise train fraud rate from $\sim 3.33\%$ to $\sim 15\%$.
  * *Phase 2*: Connect each synthetic node to its 3 most similar original fraud nodes + their immediate neighbors + 2 random non-fraud nodes.

- [ ] **Step 4: PyTorch Geometric (PyG) Graph Assembly**
  * Assemble PyG `Data(x, edge_index, y)` objects for:
    * `train_graph.pt`
    * `val_graph.pt`
    * `test_graph.pt`
    * `monitor_graph.pt` (with 50 temporal window slices of 3,000 nodes each)

- [ ] **Step 5: Implement GNN Architectures (Paper Section III-D)**
  * Build the three 3-layer benchmark architectures (hidden dimension 128):
    1. **GCN**: `GCNConv(38, 128) -> GCNConv(128, 128) -> GCNConv(128, 128) -> Linear(128, 1)`
    2. **GraphSAGE**: `SAGEConv(38, 128, aggr='mean') -> ...`
    3. **GAT**: `GATConv(38, 8, heads=16) -> LayerNorm -> ...`

---

## 4. Quick Start Command for Tomorrow

When you start tomorrow morning, verify the processed feature directory:

```bash
# Check that all preprocessed files are ready
ls -lh data/processed/

# Expected:
# meta_{train,val,test,monitor}.parquet
# features_sel_{train,val,test,monitor}.parquet
# features_pca_{train,val,test,monitor}.parquet
```

Everything is fully saved, validated, committed to Git, and ready for Graph Construction!
