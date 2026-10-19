"""
Stage G2a — Cosine Similarity Edge Builder.

Connects transactions with cosine similarity > threshold (default 0.8),
at most max_per_node (default 5) neighbors per node.
Chunked top-k on GPU with automatic CPU fallback on OOM.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch


def build_similarity_edges(
    x: np.ndarray | torch.Tensor,
    threshold: float = 0.8,
    max_per_node: int = 5,
    chunk_size: int = 1024,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct undirected similarity edges based on feature cosine similarity.

    Args:
        x: Node feature matrix of shape (N, D).
        threshold: Minimum cosine similarity to form an edge.
        max_per_node: Maximum neighbors to connect per query node.
        chunk_size: Query chunk size for matrix multiplication.
        device: 'cuda' or 'cpu'.

    Returns:
        edge_index: LongTensor of shape (2, E).
        edge_type: LongTensor of shape (E,) filled with 0.
    """
    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x).to(torch.float32)
    else:
        x_tensor = x.to(torch.float32).cpu()

    N, D = x_tensor.shape
    if N <= 1 or max_per_node <= 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    # Normalize L2
    norm = torch.norm(x_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    x_norm = x_tensor / norm

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    k = min(max_per_node, N - 1)

    sources = []
    targets = []

    def _run_chunked(target_device: torch.device, csize: int):
        src_list = []
        tgt_list = []
        x_norm_dev = x_norm.to(target_device)

        for start_idx in range(0, N, csize):
            end_idx = min(start_idx + csize, N)
            chunk_q = x_norm_dev[start_idx:end_idx]  # (B, D)

            # Cosine similarity matrix (B, N)
            sim = torch.matmul(chunk_q, x_norm_dev.t())

            # Mask self-connections
            diag_idx = torch.arange(start_idx, end_idx, device=target_device)
            sim[torch.arange(end_idx - start_idx, device=target_device), diag_idx] = -2.0

            # Top-k candidates
            top_vals, top_inds = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)

            # Filter by threshold
            valid_mask = top_vals >= threshold
            if valid_mask.any():
                row_offsets = torch.arange(start_idx, end_idx, device=target_device).unsqueeze(1).expand_as(top_inds)
                valid_src = row_offsets[valid_mask].cpu()
                valid_tgt = top_inds[valid_mask].cpu()
                src_list.append(valid_src)
                tgt_list.append(valid_tgt)

            # Free GPU memory after each chunk
            if target_device.type == "cuda":
                del sim, top_vals, top_inds, valid_mask
                if start_idx % (csize * 10) == 0:
                    torch.cuda.empty_cache()

        if target_device.type == "cuda":
            del x_norm_dev
            torch.cuda.empty_cache()

        return src_list, tgt_list

    if use_cuda:
        try:
            # Check available VRAM
            free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
            active_chunk_size = chunk_size if free_vram >= 1.5 else max(256, chunk_size // 2)
            sources, targets = _run_chunked(torch.device("cuda"), active_chunk_size)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"[Warning] CUDA similarity OOM/error ({e}), falling back to CPU chunking...")
            torch.cuda.empty_cache()
            sources, targets = _run_chunked(torch.device("cpu"), chunk_size)
    else:
        sources, targets = _run_chunked(torch.device("cpu"), chunk_size)

    if not sources:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long)

    all_src = torch.cat(sources)
    all_tgt = torch.cat(targets)

    # Make undirected: (u, v) and (v, u)
    u_all = torch.cat([all_src, all_tgt])
    v_all = torch.cat([all_tgt, all_src])

    # Canonical order: min(u, v), max(u, v) for deduplication
    canon_src = torch.minimum(u_all, v_all)
    canon_tgt = torch.maximum(u_all, v_all)
    
    # Pack into int64 for unique
    pair_keys = canon_src * N + canon_tgt
    unique_keys = torch.unique(pair_keys)

    dedup_src = unique_keys // N
    dedup_tgt = unique_keys % N

    # Store both directions
    final_src = torch.cat([dedup_src, dedup_tgt])
    final_tgt = torch.cat([dedup_tgt, dedup_src])

    edge_index = torch.stack([final_src, final_tgt], dim=0)
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    return edge_index, edge_type
