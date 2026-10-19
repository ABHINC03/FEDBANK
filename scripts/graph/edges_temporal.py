"""
Stage G2b — Temporal-Proximity Edge Builder.

Connects transactions within temporal_window_s (default 7200s / 2 hours) of each other.
Vectorized across sorted TransactionDT arrays with per-node cap.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch


def build_temporal_edges(
    transaction_dt: np.ndarray | torch.Tensor,
    window_s: int = 7200,
    max_per_node: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct undirected temporal edges within a sliding time window.

    Args:
        transaction_dt: 1D array of transaction timestamps.
        window_s: Window size in seconds (default 7200).
        max_per_node: Maximum future transactions to connect per node.

    Returns:
        edge_index: LongTensor of shape (2, E).
        edge_type: LongTensor of shape (E,) filled with 1.
    """
    if isinstance(transaction_dt, torch.Tensor):
        dt_arr = transaction_dt.cpu().numpy()
    else:
        dt_arr = np.asarray(transaction_dt)

    N = len(dt_arr)
    if N <= 1 or max_per_node <= 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    # Sort nodes chronologically
    perm = np.argsort(dt_arr, kind="mergesort")
    sorted_dt = dt_arr[perm]

    src_list = []
    tgt_list = []

    # Vectorized check for each offset up to max_per_node
    for offset in range(1, min(max_per_node + 1, N)):
        src_idx = np.arange(0, N - offset)
        dst_idx = src_idx + offset
        
        # Check time window condition
        dt_diff = sorted_dt[dst_idx] - sorted_dt[src_idx]
        valid_mask = (dt_diff >= 0) & (dt_diff <= window_s)

        if np.any(valid_mask):
            valid_src = perm[src_idx[valid_mask]]
            valid_dst = perm[dst_idx[valid_mask]]
            src_list.append(valid_src)
            tgt_list.append(valid_dst)

    if not src_list:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    all_src = np.concatenate(src_list)
    all_tgt = np.concatenate(tgt_list)

    # Canonical order for deduplication
    canon_src = np.minimum(all_src, all_tgt)
    canon_tgt = np.maximum(all_src, all_tgt)

    # Pack into int64 for unique
    pair_keys = canon_src.astype(np.int64) * N + canon_tgt.astype(np.int64)
    unique_keys = np.unique(pair_keys)

    dedup_src = unique_keys // N
    dedup_tgt = unique_keys % N

    # Store both directions
    final_src = np.concatenate([dedup_src, dedup_tgt])
    final_tgt = np.concatenate([dedup_tgt, dedup_src])

    edge_index = torch.from_numpy(np.stack([final_src, final_tgt], axis=0)).to(torch.long)
    edge_type = torch.full((edge_index.shape[1],), 1, dtype=torch.long)

    return edge_index, edge_type
