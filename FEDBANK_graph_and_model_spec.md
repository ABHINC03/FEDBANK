# FEDBANK — Phase 1 Spec: Graph Construction → Single-Device GNN (Notebook)

**Audience:** coding agent (Antigravity). Read this whole file before writing code.
**Goal of Phase 1:** prove the GNN works on ONE device (centralized) and produce a trusted baseline. Federation comes later (Phase 2), so everything here must be *federation-ready*.
**Inputs already done:** preprocessing S0–S10 (`scripts/preprocess/`), outputs in `data/processed/`, frozen schema in `FEDBANK_SCHEMA_v1.md` + `fedbank_schema_v1.json`.
**Do not modify** anything under `scripts/preprocess/`, `artifacts/`, or `data/processed/`. Read-only.

---

## 0. Ground rules

1. **Stop and report, don't improvise.** If a check in this file fails, or an assumption is false (e.g. a column is missing), stop that stage, write what you found to `reports/graph_report.md`, and continue only with what is unambiguous. Never silently change the spec.
2. **Paths are relative to the repo root.** No `/home/abhin/...`. Resolve root via `Path(__file__).resolve().parents[N]` in scripts and `Path.cwd()` search for `.git` in the notebook.
3. **Every stage ends with assertions** (listed per stage). A stage is "done" only when they pass.
4. **Config in YAML** (`configs/graph.yaml`, `configs/gnn.yaml`). No magic numbers in code. Seed = 42 everywhere; model runs use seeds 42, 43, 44.
5. **Hardware:** one 6 GB GPU under WSL2, RAPIDS conda env. Design everything to fit in 6 GB. Use chunking and CPU fallback.
6. **Federation-ready API** (important): graph building takes a dataframe, not file paths:
   `build_graph(features: DataFrame, meta: DataFrame, cfg) -> GraphBundle`
   so Phase 2 can call the exact same function once per bank. Training is `train_local(model, graph, cfg) -> (state_dict, metrics)`.

---

## 1. Folder layout to create

```
configs/graph.yaml            configs/gnn.yaml
scripts/graph/
  prepare_node_features.py    # Stage G1
  edges_similarity.py         # Stage G2a
  edges_temporal.py           # Stage G2b
  edges_cluster.py            # Stage G2c
  edges_anomaly.py            # Stage G2d
  build_graph.py              # Stage G3 (calls the above, merges, saves)
  validate_graph.py           # Stage G4
data/graphs/                  # gitignored: train_graph.pt val_graph.pt test_graph.pt monitor_graph.pt
reports/graph_report.md       # auto-generated
notebooks/04_gnn_single_device.ipynb   # Stage M (model building, live metrics)
results/                      # metrics CSV/JSON per run, plots (small files only)
```

Add `data/graphs/` and `results/checkpoints/` to `.gitignore`.

---

## 2. Stage G1 — Node features

**Input:** `data/processed/features_sel_{train,val,test,monitor}.parquet` (Track A, 48 features) and `meta_{split}.parquet`.

Steps:
1. Load each split. Assert `list(features.columns) == schema["columns"] names in order` from `fedbank_schema_v1.json`. If the processed parquet column order differs, **reorder to the schema order**, don't fail; if a column is missing or extra, fail.
2. Assert `len(features) == len(meta)` and rows are aligned (same `TransactionID` order). Never join by position without checking IDs.
3. Cast to float32. Assert no NaN/inf.
4. Output tensors `x` of shape `(N, 48)`, labels `y` (int8) from `meta.isFraud`, plus `transaction_dt` and `window_id` (monitor only) kept as non-feature attributes.

**Assertions:** shape `(N,48)`; split sizes 273000 / 58500 / 58500 / 200540; fraud rates close to paper (3.33% / 3.62% / 3.70% / 3.64%); `isFraud`, `TransactionDT`, `TransactionID`, `card_key_id`, `split`, `window_id` are **never** in `x`.

Keep Track B (PCA, 27 features) available behind a config flag `features: sel | pca` for an ablation. Default `sel`.

---

## 3. Stage G2 — Edge builders

General rules for ALL edge types:
- **Each split gets its own separate graph.** Edges never cross splits (train nodes never connect to val/test nodes). This avoids temporal leakage.
- Edges are **undirected**: store both directions. No self-loops (GCN/GAT add their own). Deduplicate.
- Keep an `edge_type` array (0=similarity, 1=temporal, 2=cluster, 3=anomaly) so edge types can be ablated. A pair connected by several rules keeps all its types in a separate record (or keep the min type id and log counts). The model uses the merged homogeneous graph.
- Every builder must have a **degree cap** (config). Memory and hubs are the main risk.
- Builders must be pure functions of `(x, dt, cfg)`. No file I/O inside them.
- Log edge counts per type and per split into `reports/graph_report.md`.

### G2a — Similarity edges
Connect transactions with cosine similarity > `sim_threshold` (0.8), at most `sim_max_per_node` (5) neighbors per node.
- Compute top-k cosine neighbors on GPU in **chunks** (start with 1024 query rows; reduce automatically if free VRAM < 1.5 GB). L2-normalize `x` first. Use `torch.topk` on the chunk × N matrix, keep neighbors above the threshold.
- Exclude self matches. Fall back to CPU chunks if CUDA OOM, never crash.
- Note: with 48 mostly-continuous features, 0.8 may connect a lot of rows; the cap handles that.

### G2b — Temporal-proximity edges
Connect transactions within `temporal_window_s` (7200 s) of each other.
- Sort by `TransactionDT`. For each node connect to later nodes inside the window, with a **per-node cap** `temporal_max_per_node` (start at 5, config-driven). The paper's note of "sliding window limit 200" would create ~50M edges on the train split, far above the paper's reported ~3.17M total, so treat the cap as a calibration knob (see calibration below).
- Vectorized with sorted arrays and offsets, no Python loops over nodes.

### G2c — Cluster-based edges
`MiniBatchKMeans(n_clusters=3, batch_size=1024, random_state=seed)` **fitted on the train split only**; assign val/test/monitor with `predict`. Within each cluster connect each node to `cluster_max_per_node` random same-cluster nodes (seeded), with a total cap of 10,000 edges per cluster per split. Save the fitted KMeans (joblib) to `artifacts/graph/kmeans.joblib`.

### G2d — Anomaly-based edges
Paper: cosine similarity > 0.75 on the top anomaly features (`V140`, `V261`, z-score max), max 3 edges per node.
- **Check first:** `V140` and `V261` are probably NOT among the 48 schema features. Look at the processed parquet columns. If they are absent, do NOT go back to raw data. Instead pick the `anomaly_k` (default 2) schema features with highest univariate AUC computed **on the train split only**, and document this deviation in the report.
- Anomaly score per node = max |z-score| over the chosen features, z-scored with **train** mean/std.
- Connect nodes whose anomaly-feature vectors have cosine similarity > 0.75, cap 3 per node.

### Calibration (do this before the full build)
Run the builders on a 20k-row sample of train and print edges-per-node per type. Then run full train and compare totals with the paper's targets: train ≈ 3.17M, val ≈ 641k, test ≈ 644k, monitor ≈ 1.81M. Targets are **guides, not assertions**: stay within a factor of ~2, adjust caps in YAML to get close, and record final counts and the gap in the report. Average degree should land roughly 10–25.

---

## 4. Stage G3 — Assemble graphs

For each split build a PyG `Data`:
`Data(x, edge_index, y, edge_type, transaction_dt)`; for monitor also `window_id`.
- Save to `data/graphs/{split}_graph.pt` with `torch.save`.
- **Monitor graph:** build **one subgraph per `window_id`** (edges only within a window), stored as a list. There are 50 windows of ~4,000 nodes (200,540 / 50). Do not hardcode 3,000; use `window_id` from `meta_monitor`.
- No SMOTE in v1. Leave a config flag `smote: false` and a stub, but do not implement graph-aware SMOTE until baselines exist (it complicates federated DP later and is an ablation, not a default).
- Store `graph_meta.json` next to the graphs: schema_version, config hash, git commit, edge counts, seeds, creation time.

---

## 5. Stage G4 — Graph validation (`validate_graph.py`)

Assert and print a pass/fail table:
1. `edge_index` values in `[0, N)`; symmetric; no self-loops; no duplicates.
2. No isolated-node rate above `max_isolated_frac` (default 10%); report it (isolated nodes learn from features only).
3. Max degree ≤ configured hard cap; report degree percentiles (50/90/99/max).
4. **Homophily check:** fraction of edges joining two fraud nodes vs the base rate (fraud-fraud edge rate should exceed what random wiring gives: fraud_rate² share). Report per edge type. If an edge type shows homophily at or below random, flag it (it adds noise).
5. No cross-split edges (nodes are split-local by construction; assert via ID sets).
6. `x` has no NaN/inf and matches the schema; labels binary.
7. Estimated memory: print `x`, `edge_index` size in MB and confirm train graph fits in GPU memory for a neighbor-sampled run.

---

## 6. Stage M — Model building notebook (`notebooks/04_gnn_single_device.ipynb`)

The user wants to **watch the model build and train live**, so this is a notebook, not a script. Use clearly separated cells with markdown headings. Clear outputs of heavy cells before committing (use `nbstripout`).

### Cell plan

| # | Section | Contents |
|---|---|---|
| 0 | Setup | imports, seed everything (`torch`, `numpy`, `random`, cuDNN deterministic flags), device check, print GPU free memory, load YAML config |
| 1 | Load data | load the 4 graph files, print node/edge counts, fraud rates, degree stats, schema version |
| 2 | Graph sanity visuals | degree histogram, edge-type counts, a small sampled subgraph plot (e.g. 150 nodes around a fraud node, fraud nodes coloured) |
| 3 | Metrics module | functions for AUPRC, ROC-AUC, best-F1 and its threshold, recall at 1% FPR, precision at top-k; **threshold chosen on validation only** |
| 4 | Live-plot helper | a callback that redraws a figure each epoch with `IPython.display.clear_output(wait=True)` (see §6.2) |
| 5 | Baselines (no graph) | LightGBM or XGBoost on the 48 features, then an MLP with identical hidden size. Same metrics, same splits |
| 6 | Model definitions | `GCN`, `GraphSAGE`, `GAT` (see §6.1). Federation-ready: one `state_dict`, LayerNorm only |
| 7 | `train_local(...)` | single function: trains with `NeighborLoader`, early stopping on val AUPRC, returns state_dict + history. Keep it pure and self-contained so it can be lifted into `fedbank/model.py` for Flower unchanged |
| 8 | Train GAT (main) | run with live plots, seed 42 |
| 9 | Compare architectures | GCN vs SAGE vs GAT, same budget |
| 10 | Seeds | repeat best model for seeds 42/43/44, report mean ± std |
| 11 | Test evaluation | evaluate once on test with the val-chosen threshold. Never tune on test |
| 12 | Drift analysis | per-window AUPRC on the 50 monitor windows, plot trend |
| 13 | Attention inspection | for a few flagged transactions, show top-attended neighbors (this later feeds the analyst explanation view) |
| 14 | Ablations | edge types removed one at a time; `features: pca` vs `sel`; with/without focal loss |
| 15 | Export | save metrics to `results/phase1_metrics.csv` + `.json`, best weights to `results/checkpoints/` |

### 6.1 Model specs
- 3 GNN layers, hidden 128, dropout 0.3 (config), ReLU/ELU, final `Linear(128, 1)` logits.
- **Input normalization:** `LayerNorm(48)` as the first operation (federation-friendly, replaces dependence on shared scaler statistics).
- **Normalization inside the network: LayerNorm only. No BatchNorm** (batch statistics break under federated averaging).
- `GCN`: `GCNConv` × 3. `GraphSAGE`: `SAGEConv(aggr="mean")` × 3.
- `GAT` (the main model): `GATConv` × 3 with **4 heads** per layer (hidden 32 per head, concatenated = 128; last conv may average). Expose `heads` in config; also run the paper's 16-head variant (8 per head) as an ablation.
- Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` computed from **train labels**, plus a switchable focal loss (`gamma=2`).
- Optimizer AdamW, lr 1e-3, weight decay 1e-4, up to 100 epochs, early stopping patience 10 on **val AUPRC**.
- Mini-batch training: `NeighborLoader(num_neighbors=[15, 10, 5], batch_size=2048)` for training; for val/test inference use the same loader with the **same neighbor counts or larger**, in eval mode, and never accumulate gradients. Tune `batch_size` down if GPU memory exceeds ~5 GB.
- Mixed precision (`torch.autocast`, fp16) is optional; if used, keep the loss in fp32 and verify metrics match fp32 within noise.

### 6.2 Live metrics requirements
Per epoch, log and display:
- train loss, val loss, val AUPRC, val ROC-AUC, val best-F1, val recall@1%FPR
- learning rate, epoch time, peak GPU memory (`torch.cuda.max_memory_allocated`)
- a live figure with (a) loss curves, (b) val AUPRC curve with the best epoch marked, (c) the current precision–recall curve, and (d) a small GPU-memory bar.
Use `tqdm.auto` for per-batch progress. Optionally write the same scalars to TensorBoard (`runs/`) so a browser view also works. Persist the full history as a DataFrame at the end of every run.

### 6.3 Pass criteria (the Phase 1 gate)
Phase 1 is complete only if ALL hold. Write the verdict into `reports/phase1_verdict.md`:
1. GAT val/test **AUPRC beats the MLP** (otherwise the graph adds no value: fix graph/edges before moving on).
2. GAT is within a small margin of, or better than, LightGBM/XGBoost AUPRC. Report the gap honestly even if GNN loses.
3. Std of test AUPRC across 3 seeds is small (report; flag if std > 0.01).
4. Monitor-window AUPRC degrades gracefully (no collapse in later windows). Report the slope.
5. Peak GPU memory < 5.5 GB and full training under a reasonable time (report epoch time).
If any fails, record the diagnosis and stop. Do not start federation code.

---

## 7. Precautions (read carefully)

**Data leakage**
- No edges across splits. Fit KMeans, z-score stats, and any threshold on **train only**.
- Never put `isFraud`, `TransactionDT`, `TransactionID`, `card_key_id`, `split`, `window_id`, or any label-derived value into `x`.
- Do not use neighbor **labels** as input features. The graph passes features only.
- Choose the decision threshold on validation; report test once.
- Do not shuffle the chronological order or re-split. Splits are fixed by preprocessing.

**Imbalance and metrics**
- Fraud is ~3.5%. Never report accuracy as a headline. Use AUPRC (primary), ROC-AUC, F1 at val-chosen threshold, recall@1%FPR.
- Report the trivial baseline AUPRC (= fraud rate) next to every result for context.
- Do not apply SMOTE in v1.

**Memory and hardware (6 GB, WSL2)**
- Chunk all N×N similarity work. Delete large tensors and call `torch.cuda.empty_cache()` between stages.
- RAPIDS and PyTorch can conflict inside one conda env (CUDA/cuDNN versions). Install PyG and torch with the **same CUDA version as the env**, check with `torch.cuda.is_available()` and a tiny `GATConv` forward pass **before** building anything. If installs conflict, create a second env for the GNN notebook (RAPIDS is only needed for preprocessing). Write the working install commands to `SETUP.md`.
- Keep `CUDF_SPILL=on` if cuDF is used. Prefer NumPy/PyTorch in the graph stage; cuDF is optional.
- If OOM occurs, reduce batch size and neighbor counts first; never silently shrink the graph.

**Reproducibility**
- Seed `torch`, `numpy`, `random`, and the `NeighborLoader` generator. Enable deterministic algorithms where available; note that scatter ops in GNNs may stay slightly nondeterministic. Report it.
- Save config + git commit hash + schema version with every result file.
- Never overwrite earlier results; write to timestamped run folders.

**Graph quality**
- Watch degree explosion. Hubs from similarity/cluster edges can dominate message passing. Enforce caps.
- Isolated nodes and weak-homophily edge types add noise. Use the G4 report to drop or tighten them, and record the decision.

**Federation readiness (do not break these)**
- Same input dim (48), same column order, same architecture hyperparameters everywhere.
- LayerNorm only; no BatchNorm; no global statistics baked into the model.
- `train_local` and `build_graph` take plain arguments (no globals, no hard-coded paths).
- Standard metrics format (`results/*.csv`) so Phase 2 results line up with Phase 1.

**Repo hygiene**
- Do not commit data, graphs, checkpoints, or large notebook outputs. Strip outputs before committing.
- Add a short `scripts/graph/README.md` with run order and expected runtimes.
- Remove the empty `sample.py` or document it.

---

## 8. Run order and deliverables

```bash
python -m scripts.graph.prepare_node_features
python -m scripts.graph.build_graph --stage all --sample 20000   # smoke test + calibration
python -m scripts.graph.build_graph --stage all                  # full build
python -m scripts.graph.validate_graph
jupyter lab notebooks/04_gnn_single_device.ipynb
```

Deliverables: the scripts above, `data/graphs/*.pt` (local only), `reports/graph_report.md`, the executed notebook, `results/phase1_metrics.csv/json`, and `reports/phase1_verdict.md` with the gate result.

**Out of scope for this file:** partitioning into banks, Flower, FLTrust, differential privacy, attacks, dashboards. Those are Phase 2.
