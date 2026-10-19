# FEDBANK Phase 1 Gate Verdict & Diagnostic Report

**Evaluation Date:** 2026-10-17  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.6, PyTorch 2.6.0, PyG 2.6.1)  
**Status:** **Phase 1 Gate 1 FAILED — Do not advance to Phase 2.**

---

## 1. Executive Summary & Gate Criteria Evaluation (§6.3)

| # | Criterion | Required Condition | Actual Result | Status |
|:---|:---|:---|:---|:---:|
| 1 | **GAT beats MLP** | GAT Test AUPRC > MLP Test AUPRC | GAT: **0.2459 ± 0.0067** vs MLP: **0.3770 ± 0.0029** | **FAIL** |
| 2 | **GAT vs GBDT Benchmark** | Close to / competitive with LightGBM | GAT: **0.2459** vs Tuned LGBM: **0.4250** (Gap: -0.1791) | **FAIL** |
| 3 | **Multi-Seed Stability** | Test AUPRC std across 3 seeds $\le 0.01$ | GAT Std: **0.0067** (Seeds: 0.2399, 0.2426, 0.2553) | **PASS** |
| 4 | **Drift Gracefulness** | No severe collapse in later monitor windows | Slope: **-0.002965**/window (Mean AUPRC: 0.2681) | **PASS** |
| 5 | **Memory & Compute Budget** | Peak GPU memory < 5.5 GB | Peak VRAM: **1.63 GB** (Budget: 6.0 GB) | **PASS** |

**Verdict:** The Phase 1 gate fails on Criterion 1 and Criterion 2. As instructed, Phase 2 is frozen pending architectural and graph topology remediation.

---

## 2. Diagnostic Step 1: Multi-Seed Re-evaluation (Seeds 42, 43, 44)

In the initial pipeline run, only GAT was tested across 3 seeds; LightGBM, MLP, GCN, and GraphSAGE were single-seed runs (seed 42). Below are the empirical results from re-running all 5 models under identical conditions across seeds `42`, `43`, and `44`:

| Model | Val AUPRC (Mean ± Std) | Test AUPRC (Mean ± Std) | Test AUPRC by Seed `[42, 43, 44]` |
| :--- | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **0.5227 ± 0.0052** | **0.4250 ± 0.0009** | [0.4240, 0.4262, 0.4249] |
| **Tabular MLP** | **0.4294 ± 0.0037** | **0.3770 ± 0.0029** | [0.3737, 0.3765, 0.3807] |
| **GraphSAGE (3-Layer)** | **0.4037 ± 0.0020** | **0.3556 ± 0.0034** | [0.3589, 0.3569, 0.3509] |
| **GCN (3-Layer)** | **0.3156 ± 0.0036** | **0.2472 ± 0.0056** | [0.2547, 0.2412, 0.2456] |
| **GAT (3-Layer, 4-Head)** | **0.2929 ± 0.0083** | **0.2459 ± 0.0067** | [0.2399, 0.2426, 0.2553] |

*Raw data saved to `results/multiseed_benchmarks.csv`.*

---

## 3. Diagnostic Step 2: LightGBM Deflation Diagnosis & Re-Tuning

### Root Cause of Initial Weak Performance
In the initial baseline run, LightGBM yielded **0.2528 Test AUPRC**, trailing even the GNN. Investigation revealed:
1. `scale_pos_weight = 29.03` was set alongside default `binary_logloss`.
2. The extreme penalty on false negatives distorted probability calibration, causing validation logloss to explode on early false alarms.
3. Early stopping triggered at **Tree 1**, halting training before any non-linear splits were learned.

### Re-tuning Sweep (Early Stopping on Validation AUPRC)
We conducted a sweep varying `scale_pos_weight` ($1.0$, $\sqrt{29.03} \approx 5.39$, $29.03$), depth/leaves, and learning rate:

| Config | PosWeight | Depth / Leaves | LR | Best Trees | Val AUPRC | Test AUPRC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `unweighted_depth6` | 1.00 | 6 / 31 | 0.05 | 500 | 0.5108 | 0.4195 | 0.8528 |
| `unweighted_depth8` | 1.00 | 8 / 63 | 0.05 | 441 | 0.5198 | 0.4185 | 0.8536 |
| **`unweighted_lr03` (Optimal)** | **1.00** | **8 / 63** | **0.03** | **489** | **0.5210** | **0.4240** | **0.8542** |
| `sqrt_pos_weight` | 5.39 | 6 / 31 | 0.05 | 7 | 0.4015 | 0.3411 | 0.8229 |
| `full_pos_weight` | 29.03 | 6 / 31 | 0.05 | 1 | 0.3088 | 0.2528 | 0.7856 |

**Selected Configuration:**
```python
lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
# Early stopping: 30 rounds monitored strictly on validation average_precision (AUPRC)
```

---

## 4. Diagnostic Step 3: GAT / GCN Ablation Suite

To isolate why GAT and GCN suffered relative to MLP and GraphSAGE, we executed an ablation varying layer depth, self-feature skip connections, and edge topologies:

| Experiment | Val AUPRC | Test AUPRC | Test ROC-AUC | Observation |
| :--- | :---: | :---: | :---: | :--- |
| **GAT (1-Layer, No Skip)** | 0.3211 | 0.2815 | 0.7606 | Shallowest network preserves most ego-signal |
| **GAT (2-Layer, No Skip)** | 0.3156 | 0.2736 | 0.7528 | Feature dilution begins |
| **GAT (3-Layer, No Skip)** | 0.2799 | 0.2434 | 0.7353 | **Severe oversmoothing** across ~23 neighbors |
| **GCN (1-Layer, No Skip)** | 0.3172 | 0.2710 | 0.7598 | - |
| **GCN (2-Layer, No Skip)** | 0.3267 | 0.2650 | 0.7494 | - |
| **GCN (3-Layer, No Skip)** | 0.3156 | 0.2531 | 0.7386 | Same collapse pattern as GAT |
| **GAT (1-Layer, WITH Skip)** | 0.3783 | 0.3318 | 0.7977 | **+0.0503 AUPRC lift** over No Skip |
| **GAT (2-Layer, WITH Skip)** | 0.3856 | 0.3360 | 0.7926 | **+0.0624 AUPRC lift** over No Skip |
| **GAT (3-Layer, WITH Skip)** | 0.3865 | **0.3384** | 0.7999 | **+0.0950 AUPRC lift** over No Skip |
| **GCN (1-Layer, WITH Skip)** | 0.3895 | **0.3562** | 0.8096 | Matches GraphSAGE performance |
| **GCN (2-Layer, WITH Skip)** | 0.3900 | 0.3428 | 0.7989 | Residual path stabilizes deeper aggregation |
| **GCN (3-Layer, WITH Skip)** | 0.3916 | 0.3379 | 0.8041 | - |
| **GAT 2L+Skip [No Edges]** | 0.3898 | 0.3424 | 0.8020 | Equivalent to Tabular MLP |
| **GAT 2L+Skip [Similarity-Only]**| 0.3955 | 0.3429 | 0.8064 | Negligible gain (+0.0005) over No Edges |
| **GAT 2L+Skip [Temporal-Only]** | 0.3958 | 0.3449 | 0.7994 | Minor gain (+0.0025) over No Edges |

*Raw data saved to `results/gnn_ablations.csv`.*

### Core Architectural Finding
1. **Self-Feature Skip Connection is Essential:** In graphs without self-loops ($0.0\%$ self-loops by specification), standard `GATConv` computes $h_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W x_j$. The ego-node's own features $x_i$ are excluded from the attention pool! Adding an explicit residual skip ($h_i + W_{\text{skip}} x_i$) boosts GAT Test AUPRC from **0.2434 to 0.3384 (+39.0% relative improvement)**.
2. **Why GraphSAGE Performed Better (0.3556):** PyG's `SAGEConv` implements `root_weight=True` by default, executing $W_1 x_i + W_2 \cdot \text{aggr}(\{x_j\})$. Hence, GraphSAGE inherently preserved ego-features, whereas standard GAT destroyed them.
3. **Synthetic Edges Provide Insufficient Signal:** Evaluated under GAT 2L+Skip, Similarity edges (0.3429) and Temporal edges (0.3449) barely move the needle beyond the No-Edge baseline (0.3424). Dense unsupervised cosine similarity links transactions with similar amounts/cards regardless of fraud syndicates, diluting fraud-specific patterns.

---

## 5. Diagnostic Step 4: Training Curves (GAT vs GraphSAGE)

The training dynamics were plotted and exported to `results/gat_vs_sage_curves.png`:
- **GAT (No Skip):** Validation loss drops early but validation AUPRC plateaus quickly around epoch 3 at $\approx 0.28\text{–}0.30$. Neighborhood feature averaging prevents the attention heads from isolating fraudulent clusters.
- **GraphSAGE:** Validation AUPRC rises steadily to $> 0.40$, remaining consistently $0.10\text{–}0.12$ above GAT across all epochs due to ego-feature preservation.

---

## 6. Diagnostic Step 5: Proposal for Entity-Based Graph Construction

### Motivation
Synthetic similarity and temporal edges fail to beat Tabular MLP because fraud syndicates operate through **shared physical and digital identities**, not abstract feature cosine similarity. 

### Proposed Entity Edge Types (IEEE-CIS)
We propose supplementing the graph with three deterministic entity-relation edge types:

1. **`same_card` Edges:**
   - **Definition:** Link transactions sharing identical card fingerprints (`card1` + `card2` + `card3` + `card4` + `card5` + `card6`).
   - **Rationale:** Card replay and rapid succession attacks represent the primary vector of payment fraud. IEEE-CIS literature demonstrates $> 25\times$ fraud homophily on card identity cliques.
2. **`same_address` Edges:**
   - **Definition:** Link transactions sharing identical billing/shipping regions (`addr1` combined with `addr2` or `dist1`).
   - **Rationale:** Compromised card testing rings often ship to fraudulent consolidation hubs or reshipping addresses.
3. **`same_email` Edges:**
   - **Definition:** Link transactions sharing identical email domains or synthetic account hashes (`P_emaildomain` / `R_emaildomain`).
   - **Rationale:** Catch-all disposable domains (`mailinator`, `protonmail`, fraudulent Gmail clusters) cluster fraud instances together.

### Architectural Recommendation
To decisively pass Phase 1 Gate 1 before starting Phase 2:
1. **Adopt Ego-Residual GNN:** Integrate `GATWithSkip` or PyG `ResGatedGraphConv` / `GraphSAGE` with skip connections.
2. **Rebuild Graph with Entity Edges:** Add `same_card` and `same_address` with a degree cap (e.g., $k \le 10$) to prevent supernode hub explosion.
