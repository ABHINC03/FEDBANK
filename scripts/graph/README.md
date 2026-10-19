# FEDBANK — Graph Pipeline Scripts

This directory contains the Stage G1–G4 graph construction scripts for FEDBANK.

## Pipeline Scripts & Run Order

```bash
# Stage G1: Prepare Node Features & Schema Validation
python -m scripts.graph.prepare_node_features

# Stage G3: Graph Construction (Calibration / Smoke test on 20k rows)
python -m scripts.graph.build_graph --stage all --sample 20000

# Stage G3: Full Graph Construction (Builds all 4 splits)
python -m scripts.graph.build_graph --stage all

# Stage G4: Graph Integrity & Homophily Validation
python -m scripts.graph.validate_graph
```

## Module Descriptions

- `prepare_node_features.py` (Stage G1): Extracts, validates, and aligns tabular node features against `fedbank_schema_v1.json`. Checks zero NaN/inf and row alignments with metadata.
- `edges_similarity.py` (Stage G2a): Cosine similarity edges (threshold > 0.8, max 5 edges/node) using GPU chunking with CPU fallback.
- `edges_temporal.py` (Stage G2b): Vectorized sliding-window temporal proximity edges within 7,200s (2 hrs) with per-node degree cap.
- `edges_cluster.py` (Stage G2c): `MiniBatchKMeans` (k=3) fitted on train only, with 10,000 edges/cluster cap per split.
- `edges_anomaly.py` (Stage G2d): Top univariate ROC-AUC anomaly feature selection on train split (`C1`, `C2`), z-scoring, and cosine similarity (>0.75, max 3 edges/node).
- `build_graph.py` (Stage G3): Assembles homogeneous PyG `Data` graphs for train, val, test, and 50 monitor window subgraphs.
- `validate_graph.py` (Stage G4): Comprehensive integrity check, symmetry validation, degree percentiles, and fraud homophily lift verification.

## Expected Runtimes (WSL2, RTX 4050 6GB)

| Stage | Command | Typical Runtime | Output |
|---|---|---|---|
| G1 | `prepare_node_features` | ~4s | In-memory `NodeFeaturesBundle` |
| G3 Sample | `build_graph --sample 20000` | ~15s | `data/graphs/*_sample_20000.pt` |
| G3 Full | `build_graph --stage all` | ~2.5 min | `data/graphs/*_graph.pt` |
| G4 | `validate_graph` | ~10s | Pass/fail summary & homophily report |
