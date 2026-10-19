"""
Stage G3 — Graph Assembly Pipeline.

Assembles PyG Data objects for train, val, test, and monitor splits.
Coordinates edge builders (similarity, temporal, cluster, anomaly).
Ensures models (KMeans, anomaly stats) are fitted on train only.
For monitor, constructs 50 separate window subgraphs (edges within window only).
Saves graphs to data/graphs/*.pt with graph_meta.json.

Run as:
    python -m scripts.graph.build_graph --stage all --sample 20000  # smoke test / calibration
    python -m scripts.graph.build_graph --stage all                 # full build
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from scripts.graph.edges_anomaly import build_anomaly_edges, select_top_anomaly_features
from scripts.graph.edges_cluster import build_cluster_edges, fit_cluster_model
from scripts.graph.edges_similarity import build_similarity_edges
from scripts.graph.edges_temporal import build_temporal_edges
from scripts.graph.prepare_node_features import (
    NodeFeaturesBundle,
    find_repo_root,
    load_schema,
    load_split_node_features,
    prepare_node_features,
)


def load_config(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration YAML."""
    if cfg_path is None:
        cfg_path = find_repo_root() / "configs" / "graph.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_and_deduplicate_edges(
    edge_indices: List[torch.Tensor],
    edge_types: List[torch.Tensor],
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Merge edges from multiple builders, remove self-loops, and deduplicate.
    If an edge appears in multiple types, keeps the minimum edge_type id.
    
    Returns:
        merged_edge_index: (2, E_unique)
        merged_edge_type: (E_unique,)
        edge_counts_by_type: dict with original counts per type
    """
    type_names = ["similarity", "temporal", "cluster", "anomaly"]
    raw_counts = {name: 0 for name in type_names}

    valid_indices = []
    valid_types = []

    for idx_tensor, type_tensor in zip(edge_indices, edge_types):
        if idx_tensor.numel() == 0:
            continue
        # Remove self-loops
        non_self = idx_tensor[0] != idx_tensor[1]
        filt_idx = idx_tensor[:, non_self]
        filt_typ = type_tensor[non_self]

        if filt_idx.numel() > 0:
            t_id = int(filt_typ[0].item())
            raw_counts[type_names[t_id]] += filt_idx.shape[1] // 2  # undirected pairs
            valid_indices.append(filt_idx)
            valid_types.append(filt_typ)

    if not valid_indices:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long), raw_counts

    all_indices = torch.cat(valid_indices, dim=1)
    all_types = torch.cat(valid_types, dim=0)

    # Dedup using canonical undirected representation
    src = all_indices[0]
    tgt = all_indices[1]
    canon_src = torch.minimum(src, tgt)
    canon_tgt = torch.maximum(src, tgt)

    # Pack pair into int64
    pair_keys = canon_src * num_nodes + canon_tgt

    # Group by unique keys and take min edge_type
    unique_keys, inverse_indices = torch.unique(pair_keys, return_inverse=True)
    num_unique = unique_keys.shape[0]

    min_types = torch.full((num_unique,), 999, dtype=torch.long)
    min_types.scatter_reduce_(0, inverse_indices, all_types, reduce="amin")

    dedup_src = unique_keys // num_nodes
    dedup_tgt = unique_keys % num_nodes

    # Store both directions
    final_src = torch.cat([dedup_src, dedup_tgt])
    final_tgt = torch.cat([dedup_tgt, dedup_src])
    final_types = torch.cat([min_types, min_types])

    merged_edge_index = torch.stack([final_src, final_tgt], dim=0)
    return merged_edge_index, final_types, raw_counts


def build_graph(
    features_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: Dict[str, Any],
    split: str = "train",
    kmeans_model: Optional[Any] = None,
    anomaly_stats: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
) -> Tuple[Data | List[Data], Dict[str, Any], Any, Dict[str, Any]]:
    """
    Federation-ready graph builder: takes in-memory DataFrames, produces PyG Data.
    """
    if sample_size is not None and sample_size < len(features_df):
        features_df = features_df.iloc[:sample_size].copy()
        meta_df = meta_df.iloc[:sample_size].copy()

    schema_cols = load_schema()
    bundle = prepare_node_features(
        features_df=features_df,
        meta_df=meta_df,
        split=split,
        schema_cols=schema_cols,
        track=cfg.get("features", {}).get("track", "sel"),
        to_torch_tensor=False,
    )

    x_np = bundle.x
    y_np = bundle.y
    dt_np = bundle.transaction_dt
    num_nodes = len(x_np)

    edge_cfg = cfg.get("edges", {})
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("seed", 42)

    # 1. Fit models on train if not provided
    if split == "train":
        if kmeans_model is None:
            c_cfg = edge_cfg.get("cluster", {})
            kmeans_model = fit_cluster_model(
                x_train=x_np,
                n_clusters=c_cfg.get("n_clusters", 3),
                batch_size=c_cfg.get("batch_size", 1024),
                random_state=seed,
            )

        if anomaly_stats is None:
            a_cfg = edge_cfg.get("anomaly", {})
            top_idx, top_names, t_means, t_stds = select_top_anomaly_features(
                x_train=x_np,
                y_train=y_np,
                k=a_cfg.get("k_features", 2),
                feature_names=bundle.feature_names,
            )
            anomaly_stats = {
                "top_indices": top_idx,
                "top_names": top_names,
                "train_means": t_means,
                "train_stds": t_stds,
            }

    # Special handling for monitor split: build 50 subgraphs (one per window_id)
    if split == "monitor" and bundle.window_id is not None:
        window_ids = np.unique(bundle.window_id)
        subgraphs = []
        total_edge_counts = {"similarity": 0, "temporal": 0, "cluster": 0, "anomaly": 0}

        for wid in window_ids:
            mask = (bundle.window_id == wid)
            w_indices = np.where(mask)[0]
            w_x = x_np[w_indices]
            w_y = y_np[w_indices]
            w_dt = dt_np[w_indices]
            w_N = len(w_x)

            # Build edges within this window
            indices_list = []
            types_list = []

            # Similarity
            s_cfg = edge_cfg.get("similarity", {})
            s_idx, s_typ = build_similarity_edges(
                x=w_x,
                threshold=s_cfg.get("threshold", 0.8),
                max_per_node=s_cfg.get("max_per_node", 5),
                chunk_size=s_cfg.get("chunk_size", 1024),
                device=device,
            )
            indices_list.append(s_idx)
            types_list.append(s_typ)

            # Temporal
            t_cfg = edge_cfg.get("temporal", {})
            t_idx, t_typ = build_temporal_edges(
                transaction_dt=w_dt,
                window_s=t_cfg.get("window_s", 7200),
                max_per_node=t_cfg.get("max_per_node", 5),
            )
            indices_list.append(t_idx)
            types_list.append(t_typ)

            # Cluster
            c_cfg = edge_cfg.get("cluster", {})
            c_idx, c_typ = build_cluster_edges(
                x=w_x,
                kmeans_model=kmeans_model,
                max_per_node=c_cfg.get("max_per_node", 5),
                max_edges_per_cluster=c_cfg.get("max_edges_per_cluster", 10000),
                random_state=seed + int(wid),
            )
            indices_list.append(c_idx)
            types_list.append(c_typ)

            # Anomaly
            a_cfg = edge_cfg.get("anomaly", {})
            anom_idx, anom_typ = build_anomaly_edges(
                x=w_x,
                anomaly_feature_indices=anomaly_stats["top_indices"],
                train_means=anomaly_stats["train_means"],
                train_stds=anomaly_stats["train_stds"],
                threshold=a_cfg.get("similarity_threshold", 0.75),
                max_per_node=a_cfg.get("max_per_node", 3),
                device=device,
            )
            indices_list.append(anom_idx)
            types_list.append(anom_typ)

            merged_idx, merged_typ, counts = merge_and_deduplicate_edges(
                indices_list, types_list, w_N
            )
            for k_typ, cnt in counts.items():
                total_edge_counts[k_typ] += cnt

            subgraph = Data(
                x=torch.from_numpy(w_x).to(torch.float32),
                edge_index=merged_idx,
                edge_type=merged_typ,
                y=torch.from_numpy(w_y).to(torch.int8),
                transaction_dt=torch.from_numpy(w_dt).to(torch.int64),
                window_id=torch.full((w_N,), int(wid), dtype=torch.int32),
            )
            subgraphs.append(subgraph)

        meta_info = {
            "split": split,
            "num_windows": len(window_ids),
            "num_nodes_total": num_nodes,
            "edge_counts_by_type": total_edge_counts,
            "total_directed_edges": sum(g.edge_index.shape[1] for g in subgraphs),
        }
        return subgraphs, meta_info, kmeans_model, anomaly_stats

    # Standard split (train, val, test)
    indices_list = []
    types_list = []

    # Similarity edges
    s_cfg = edge_cfg.get("similarity", {})
    s_idx, s_typ = build_similarity_edges(
        x=x_np,
        threshold=s_cfg.get("threshold", 0.8),
        max_per_node=s_cfg.get("max_per_node", 5),
        chunk_size=s_cfg.get("chunk_size", 1024),
        device=device,
    )
    indices_list.append(s_idx)
    types_list.append(s_typ)

    # Temporal edges
    t_cfg = edge_cfg.get("temporal", {})
    t_idx, t_typ = build_temporal_edges(
        transaction_dt=dt_np,
        window_s=t_cfg.get("window_s", 7200),
        max_per_node=t_cfg.get("max_per_node", 5),
    )
    indices_list.append(t_idx)
    types_list.append(t_typ)

    # Cluster edges
    c_cfg = edge_cfg.get("cluster", {})
    c_idx, c_typ = build_cluster_edges(
        x=x_np,
        kmeans_model=kmeans_model,
        max_per_node=c_cfg.get("max_per_node", 5),
        max_edges_per_cluster=c_cfg.get("max_edges_per_cluster", 10000),
        random_state=seed,
    )
    indices_list.append(c_idx)
    types_list.append(c_typ)

    # Anomaly edges
    a_cfg = edge_cfg.get("anomaly", {})
    anom_idx, anom_typ = build_anomaly_edges(
        x=x_np,
        anomaly_feature_indices=anomaly_stats["top_indices"],
        train_means=anomaly_stats["train_means"],
        train_stds=anomaly_stats["train_stds"],
        threshold=a_cfg.get("similarity_threshold", 0.75),
        max_per_node=a_cfg.get("max_per_node", 3),
        device=device,
    )
    indices_list.append(anom_idx)
    types_list.append(anom_typ)

    merged_idx, merged_typ, raw_counts = merge_and_deduplicate_edges(
        indices_list, types_list, num_nodes
    )

    pyg_data = Data(
        x=torch.from_numpy(x_np).to(torch.float32),
        edge_index=merged_idx,
        edge_type=merged_typ,
        y=torch.from_numpy(y_np).to(torch.int8),
        transaction_dt=torch.from_numpy(dt_np).to(torch.int64),
    )
    if bundle.window_id is not None:
        pyg_data.window_id = torch.from_numpy(bundle.window_id).to(torch.int32)

    meta_info = {
        "split": split,
        "num_nodes": num_nodes,
        "num_edges_directed": merged_idx.shape[1],
        "num_edges_undirected": merged_idx.shape[1] // 2,
        "edge_counts_by_type": raw_counts,
        "avg_degree": float(merged_idx.shape[1]) / num_nodes if num_nodes > 0 else 0.0,
    }

    return pyg_data, meta_info, kmeans_model, anomaly_stats


def run_build_pipeline(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Execute complete graph building pipeline and save to disk."""
    repo_root = find_repo_root()
    cfg = load_config(repo_root / "configs" / "graph.yaml")
    graph_dir = repo_root / "data" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    mode_str = f"SAMPLE ({sample_size:,} rows)" if sample_size else "FULL DATASET"
    print(f"STAGE G3: Building Homogeneous Graphs ({mode_str})")
    print("=" * 78)

    all_meta = {}
    kmeans_model = None
    anomaly_stats = None

    for split in ["train", "val", "test", "monitor"]:
        print(f"\nProcessing split: {split.upper()}...")
        feat_path = repo_root / "data" / "processed" / f"features_sel_{split}.parquet"
        meta_path = repo_root / "data" / "processed" / f"meta_{split}.parquet"

        features_df = pd.read_parquet(feat_path)
        meta_df = pd.read_parquet(meta_path)

        graph_obj, info, kmeans_model, anomaly_stats = build_graph(
            features_df=features_df,
            meta_df=meta_df,
            cfg=cfg,
            split=split,
            kmeans_model=kmeans_model,
            anomaly_stats=anomaly_stats,
            sample_size=sample_size,
        )

        save_name = f"{split}_graph.pt"
        if sample_size is not None:
            save_name = f"{split}_graph_sample_{sample_size}.pt"
        
        save_path = graph_dir / save_name
        torch.save(graph_obj, save_path)
        print(f"Saved {save_name} ({save_path.stat().st_size / (1024**2):.1f} MB)")
        all_meta[split] = info

        # Print summary
        if split != "monitor":
            print(f"  Nodes: {info['num_nodes']:,} | Undirected Edges: {info['num_edges_undirected']:,} | Avg Degree: {info['avg_degree']:.2f}")
            print(f"  Edge breakdown: {info['edge_counts_by_type']}")
        else:
            print(f"  Windows: {info['num_windows']} | Total Nodes: {info['num_nodes_total']:,} | Directed Edges: {info['total_directed_edges']:,}")

    # Save fitted models and statistics
    if kmeans_model is not None and sample_size is None:
        joblib.dump(kmeans_model, graph_dir / "kmeans.joblib")
    if anomaly_stats is not None and sample_size is None:
        # Convert numpy to lists for json serialization
        serializable_anomaly = {
            "top_indices": [int(i) for i in anomaly_stats["top_indices"]],
            "top_names": [str(n) for n in anomaly_stats["top_names"]],
            "train_means": [float(m) for m in anomaly_stats["train_means"]],
            "train_stds": [float(s) for s in anomaly_stats["train_stds"]],
        }
        with open(graph_dir / "anomaly_stats.json", "w") as f:
            json.dump(serializable_anomaly, f, indent=2)

    # Save graph_meta.json
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"

    meta_record = {
        "schema_version": "fedbank_schema_v1",
        "created_at": datetime.datetime.now().isoformat(),
        "git_commit": git_hash,
        "is_sample": sample_size is not None,
        "sample_size": sample_size,
        "config": cfg,
        "splits": all_meta,
    }
    meta_json_path = graph_dir / ("graph_meta_sample.json" if sample_size else "graph_meta.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2)

    print("\n" + "=" * 78)
    print("STAGE G3 GRAPH BUILD COMPLETE")
    print("=" * 78)
    return all_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEDBANK Stage G3 Graph Builder")
    parser.add_argument("--stage", type=str, default="all", help="Stage to run")
    parser.add_argument("--sample", type=int, default=None, help="Row sample for calibration/smoke test")
    args = parser.parse_args()

    run_build_pipeline(sample_size=args.sample)
