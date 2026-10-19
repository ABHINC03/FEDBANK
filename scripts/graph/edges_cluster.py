"""
Stage G2c — Cluster-Based Edge Builder.

Fits MiniBatchKMeans (k=3) on the train split only.
Predicts cluster assignments for any split.
Within each cluster, connects each node to cluster_max_per_node random same-cluster nodes,
with a hard cap of 10,000 edges per cluster per split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch


def fit_cluster_model(
    x_train: np.ndarray | torch.Tensor,
    n_clusters: int = 3,
    batch_size: int = 1024,
    random_state: int = 42,
    save_path: Optional[str | Path] = None,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans model on the train split only and optionally save."""
    if isinstance(x_train, torch.Tensor):
        x_arr = x_train.cpu().numpy()
    else:
        x_arr = np.asarray(x_train)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )
    kmeans.fit(x_arr)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans, p)

    return kmeans


def build_cluster_edges(
    x: np.ndarray | torch.Tensor,
    kmeans_model: MiniBatchKMeans,
    max_per_node: int = 5,
    max_edges_per_cluster: int = 10000,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct undirected cluster-based edges within predicted clusters.

    Args:
        x: Feature matrix of shape (N, D).
        kmeans_model: Fitted MiniBatchKMeans instance.
        max_per_node: Max random same-cluster connections per node.
        max_edges_per_cluster: Maximum undirected edges allowed per cluster.
        random_state: Seed for reproducible random neighbor selection.

    Returns:
        edge_index: LongTensor of shape (2, E).
        edge_type: LongTensor of shape (E,) filled with 2.
    """
    if isinstance(x, torch.Tensor):
        x_arr = x.cpu().numpy()
    else:
        x_arr = np.asarray(x)

    N = len(x_arr)
    if N <= 1 or max_per_node <= 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    cluster_labels = kmeans_model.predict(x_arr)
    rng = np.random.default_rng(random_state)

    src_list = []
    tgt_list = []

    for c in range(kmeans_model.n_clusters):
        c_nodes = np.where(cluster_labels == c)[0]
        n_c = len(c_nodes)
        if n_c <= 1:
            continue

        c_src = []
        c_tgt = []

        # Connect each node in cluster to up to max_per_node random same-cluster peers
        k_sample = min(max_per_node, n_c - 1)
        for i in range(n_c):
            node_idx = c_nodes[i]
            # Pick random peers excluding self
            peer_pool = np.delete(c_nodes, i)
            chosen = rng.choice(peer_pool, size=k_sample, replace=False)
            c_src.extend([node_idx] * k_sample)
            c_tgt.extend(chosen)

        if not c_src:
            continue

        # Canonicalize and deduplicate within cluster
        u_c = np.minimum(np.asarray(c_src, dtype=np.int64), np.asarray(c_tgt, dtype=np.int64))
        v_c = np.maximum(np.asarray(c_src, dtype=np.int64), np.asarray(c_tgt, dtype=np.int64))

        cluster_keys = u_c * N + v_c
        unique_keys = np.unique(cluster_keys)

        # Enforce per-cluster edge cap
        if len(unique_keys) > max_edges_per_cluster:
            unique_keys = rng.choice(unique_keys, size=max_edges_per_cluster, replace=False)

        src_list.append(unique_keys // N)
        tgt_list.append(unique_keys % N)

    if not src_list:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    dedup_src = np.concatenate(src_list)
    dedup_tgt = np.concatenate(tgt_list)

    # Store both directions
    final_src = np.concatenate([dedup_src, dedup_tgt])
    final_tgt = np.concatenate([dedup_tgt, dedup_src])

    edge_index = torch.from_numpy(np.stack([final_src, final_tgt], axis=0)).to(torch.long)
    edge_type = torch.full((edge_index.shape[1],), 2, dtype=torch.long)

    return edge_index, edge_type
