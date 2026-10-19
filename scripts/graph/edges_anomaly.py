"""
Stage G2d — Anomaly-Based Edge Builder.

Selects top anomaly_k (default 2) features based on highest univariate ROC-AUC
computed on the train split only (since V140 and V261 are absent from schema).
Computes anomaly feature z-scores using train mean and std.
Connects transactions with cosine similarity > threshold (default 0.75) on the
anomaly feature vectors, capped at max_per_node (default 3) edges per node.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def select_top_anomaly_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 2,
    feature_names: List[str] | None = None,
) -> Tuple[List[int], List[str], np.ndarray, np.ndarray]:
    """
    Select top k features with highest univariate ROC-AUC on train split only.

    Returns:
        feature_indices: Indices of chosen features.
        selected_names: Names of chosen features.
        train_means: Mean of selected features on train split.
        train_stds: Std of selected features on train split.
    """
    N, D = x_train.shape
    auc_scores = np.zeros(D, dtype=np.float32)

    for col_idx in range(D):
        col_vals = x_train[:, col_idx]
        try:
            # Check AUC in both directions (positive and inverse correlation)
            score = roc_auc_score(y_train, col_vals)
            auc_scores[col_idx] = max(score, 1.0 - score)
        except Exception:
            auc_scores[col_idx] = 0.5

    top_indices = np.argsort(auc_scores)[-k:][::-1].tolist()
    if feature_names is not None:
        selected_names = [feature_names[i] for i in top_indices]
    else:
        selected_names = [f"feat_{i}" for i in top_indices]

    train_means = np.mean(x_train[:, top_indices], axis=0)
    train_stds = np.std(x_train[:, top_indices], axis=0)
    train_stds[train_stds < 1e-6] = 1.0  # Avoid division by zero

    return top_indices, selected_names, train_means, train_stds


def build_anomaly_edges(
    x: np.ndarray | torch.Tensor,
    anomaly_feature_indices: List[int],
    train_means: np.ndarray,
    train_stds: np.ndarray,
    threshold: float = 0.75,
    max_per_node: int = 3,
    chunk_size: int = 2048,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct undirected edges based on cosine similarity of anomaly feature z-scores.

    Args:
        x: Node feature matrix of shape (N, D).
        anomaly_feature_indices: Selected feature column indices.
        train_means: Train mean vector for selected features.
        train_stds: Train std vector for selected features.
        threshold: Cosine similarity threshold (default 0.75).
        max_per_node: Max connections per node (default 3).
        chunk_size: Query chunk size for matrix multiplication.
        device: 'cuda' or 'cpu'.

    Returns:
        edge_index: LongTensor of shape (2, E).
        edge_type: LongTensor of shape (E,) filled with 3.
    """
    if isinstance(x, torch.Tensor):
        x_sub = x[:, anomaly_feature_indices].cpu().numpy()
    else:
        x_sub = x[:, anomaly_feature_indices]

    N = len(x_sub)
    if N <= 1 or max_per_node <= 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    # Compute z-scores using train statistics
    z_scores = (x_sub - train_means) / train_stds
    z_tensor = torch.from_numpy(z_scores).to(torch.float32)

    # Normalize L2
    norm = torch.norm(z_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    z_norm = z_tensor / norm

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    k = min(max_per_node, N - 1)

    def _run_chunked(target_device: torch.device):
        src_list = []
        tgt_list = []
        z_norm_dev = z_norm.to(target_device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_q = z_norm_dev[start_idx:end_idx]

            sim = torch.matmul(chunk_q, z_norm_dev.t())

            # Mask self
            diag_idx = torch.arange(start_idx, end_idx, device=target_device)
            sim[torch.arange(end_idx - start_idx, device=target_device), diag_idx] = -2.0

            top_vals, top_inds = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)

            valid_mask = top_vals >= threshold
            if valid_mask.any():
                row_offsets = torch.arange(start_idx, end_idx, device=target_device).unsqueeze(1).expand_as(top_inds)
                src_list.append(row_offsets[valid_mask].cpu())
                tgt_list.append(top_inds[valid_mask].cpu())

            if target_device.type == "cuda" and start_idx % (chunk_size * 10) == 0:
                torch.cuda.empty_cache()

        if target_device.type == "cuda":
            del z_norm_dev
            torch.cuda.empty_cache()

        return src_list, tgt_list

    if use_cuda:
        try:
            sources, targets = _run_chunked(torch.device("cuda"))
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            sources, targets = _run_chunked(torch.device("cpu"))
    else:
        sources, targets = _run_chunked(torch.device("cpu"))

    if not sources:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    all_src = torch.cat(sources)
    all_tgt = torch.cat(targets)

    # Make undirected and deduplicate
    canon_src = torch.minimum(all_src, all_tgt)
    canon_tgt = torch.maximum(all_src, all_tgt)

    pair_keys = canon_src * N + canon_tgt
    unique_keys = torch.unique(pair_keys)

    dedup_src = unique_keys // N
    dedup_tgt = unique_keys % N

    final_src = torch.cat([dedup_src, dedup_tgt])
    final_tgt = torch.cat([dedup_tgt, dedup_src])

    edge_index = torch.stack([final_src, final_tgt], dim=0)
    edge_type = torch.full((edge_index.shape[1],), 3, dtype=torch.long)

    return edge_index, edge_type
