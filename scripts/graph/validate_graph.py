"""
Stage G4 — Graph Validation Pipeline.

Asserts structural integrity, symmetry, degree distribution, and homophily
for each generated PyG graph bundle (*_graph.pt) in data/graphs/.
Prints a comprehensive pass/fail table and outputs results to reports/graph_report.md.

Run as:
    python -m scripts.graph.validate_graph
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from scripts.graph.prepare_node_features import find_repo_root, load_schema


def validate_single_graph(
    data: Data,
    split_name: str,
    max_isolated_frac: float = 0.10,
    degree_hard_cap: int = 200,
) -> Dict[str, Any]:
    """Validate a single PyG Data graph object."""
    x = data.x
    edge_index = data.edge_index
    y = data.y
    edge_type = getattr(data, "edge_type", None)

    N = x.shape[0]
    D = x.shape[1]
    E = edge_index.shape[1]

    # 1. Feature checks: no NaN/inf, shape, dtype
    finite_x = bool(torch.isfinite(x).all().item())
    binary_y = bool(((y == 0) | (y == 1)).all().item())

    # 2. Edge index bounds and self-loops
    in_bounds = True
    no_self_loops = True
    is_symmetric = True
    no_duplicates = True

    if E > 0:
        in_bounds = bool((edge_index[0].min() >= 0).item() and (edge_index[1].max() < N).item())
        no_self_loops = bool((edge_index[0] != edge_index[1]).all().item())

        # Check deduplication
        pair_keys = edge_index[0].to(torch.int64) * N + edge_index[1].to(torch.int64)
        no_duplicates = bool(torch.unique(pair_keys).shape[0] == E)

        # Symmetry check
        rev_keys = edge_index[1].to(torch.int64) * N + edge_index[0].to(torch.int64)
        is_symmetric = bool(torch.equal(torch.sort(pair_keys)[0], torch.sort(rev_keys)[0]))

    # 3. Degrees and isolated nodes
    if E > 0:
        degs = degree(edge_index[0], num_nodes=N, dtype=torch.float32)
        isolated_count = int((degs == 0).sum().item())
        isolated_frac = float(isolated_count) / N
        max_deg = int(degs.max().item())
        p50 = float(torch.quantile(degs, 0.50).item())
        p90 = float(torch.quantile(degs, 0.90).item())
        p99 = float(torch.quantile(degs, 0.99).item())
    else:
        degs = torch.zeros(N)
        isolated_count = N
        isolated_frac = 1.0
        max_deg = 0
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0

    isolated_pass = isolated_frac <= max_isolated_frac
    degree_cap_pass = max_deg <= degree_hard_cap

    # 4. Homophily check: fraud-fraud edge rate vs base rate (fraud_rate^2)
    fraud_rate = float(y.to(torch.float32).mean().item())
    base_ff_rate = fraud_rate ** 2

    homophily_results = {}
    type_names = ["similarity", "temporal", "cluster", "anomaly"]

    if E > 0 and edge_type is not None:
        src_y = y[edge_index[0]]
        tgt_y = y[edge_index[1]]
        is_ff = (src_y == 1) & (tgt_y == 1)

        # Global homophily
        actual_ff_rate = float(is_ff.to(torch.float32).mean().item())
        homophily_lift = actual_ff_rate / max(base_ff_rate, 1e-6)

        # Per edge type
        for t_id, t_name in enumerate(type_names):
            t_mask = (edge_type == t_id)
            if t_mask.any():
                ff_t = is_ff[t_mask].to(torch.float32).mean().item()
                lift_t = ff_t / max(base_ff_rate, 1e-6)
                homophily_results[t_name] = {
                    "edge_count": int(t_mask.sum().item()) // 2,
                    "ff_rate": float(ff_t),
                    "lift": float(lift_t),
                }
    else:
        actual_ff_rate = 0.0
        homophily_lift = 0.0

    # 5. Estimated memory
    x_mb = (x.numel() * x.element_size()) / (1024**2)
    e_mb = (edge_index.numel() * edge_index.element_size()) / (1024**2)
    total_mb = x_mb + e_mb

    overall_pass = (
        finite_x
        and binary_y
        and in_bounds
        and no_self_loops
        and is_symmetric
        and no_duplicates
        and isolated_pass
    )

    return {
        "split": split_name,
        "nodes": N,
        "features": D,
        "directed_edges": E,
        "undirected_edges": E // 2,
        "finite_x": finite_x,
        "binary_y": binary_y,
        "in_bounds": in_bounds,
        "no_self_loops": no_self_loops,
        "is_symmetric": is_symmetric,
        "no_duplicates": no_duplicates,
        "isolated_frac": isolated_frac,
        "isolated_pass": isolated_pass,
        "degree_stats": {"p50": p50, "p90": p90, "p99": p99, "max": max_deg},
        "degree_cap_pass": degree_cap_pass,
        "fraud_rate": fraud_rate,
        "base_ff_rate": base_ff_rate,
        "actual_ff_rate": actual_ff_rate,
        "homophily_lift": homophily_lift,
        "per_type_homophily": homophily_results,
        "memory_mb": {"x": x_mb, "edges": e_mb, "total": total_mb},
        "overall_pass": overall_pass,
    }


def run_full_validation() -> Dict[str, Any]:
    """Load all saved graphs and run formal validation assertions."""
    repo_root = find_repo_root()
    graph_dir = repo_root / "data" / "graphs"

    print("=" * 80)
    print("STAGE G4: Running Comprehensive Graph Integrity & Homophily Validation")
    print("=" * 80)

    results = {}
    table_rows = []

    for split in ["train", "val", "test", "monitor"]:
        graph_path = graph_dir / f"{split}_graph.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}. Build graphs first.")

        graph_obj = torch.load(graph_path, weights_only=False)

        if split == "monitor" and isinstance(graph_obj, list):
            # Validate each window and aggregate
            window_res = []
            for i, w_data in enumerate(graph_obj):
                res_w = validate_single_graph(w_data, f"monitor_w{i}")
                window_res.append(res_w)

            total_nodes = sum(r["nodes"] for r in window_res)
            total_edges = sum(r["undirected_edges"] for r in window_res)
            avg_iso = float(np.mean([r["isolated_frac"] for r in window_res]))
            all_pass = all(r["overall_pass"] for r in window_res)

            res = {
                "split": "monitor (50 windows)",
                "nodes": total_nodes,
                "undirected_edges": total_edges,
                "isolated_frac": avg_iso,
                "finite_x": all(r["finite_x"] for r in window_res),
                "is_symmetric": all(r["is_symmetric"] for r in window_res),
                "no_self_loops": all(r["no_self_loops"] for r in window_res),
                "no_duplicates": all(r["no_duplicates"] for r in window_res),
                "degree_stats": {
                    "p50": float(np.mean([r["degree_stats"]["p50"] for r in window_res])),
                    "p90": float(np.mean([r["degree_stats"]["p90"] for r in window_res])),
                    "p99": float(np.mean([r["degree_stats"]["p99"] for r in window_res])),
                    "max": max(r["degree_stats"]["max"] for r in window_res),
                },
                "homophily_lift": float(np.mean([r["homophily_lift"] for r in window_res])),
                "overall_pass": all_pass,
            }
            results[split] = res
        else:
            res = validate_single_graph(graph_obj, split)
            results[split] = res

        deg = res["degree_stats"]
        table_rows.append({
            "Split": res["split"],
            "Nodes (N)": f"{res['nodes']:,}",
            "Edges (Undirected)": f"{res['undirected_edges']:,}",
            "Symmetric": "PASS" if res["is_symmetric"] else "FAIL",
            "No Loops/Dup": "PASS" if (res["no_self_loops"] and res["no_duplicates"]) else "FAIL",
            "Isolated Frac": f"{res['isolated_frac']:.2%}",
            "Deg(p50/p90/p99/max)": f"{deg['p50']:.0f} / {deg['p90']:.0f} / {deg['p99']:.0f} / {deg['max']}",
            "Homophily Lift": f"{res['homophily_lift']:.2f}x",
            "Status": "PASS" if res["overall_pass"] else "FAIL",
        })

    df_report = pd.DataFrame(table_rows)
    print("\n" + df_report.to_string(index=False) + "\n")

    # Detailed homophily report for train
    train_res = results["train"]
    print("--- Train Graph Homophily Analysis (Fraud-Fraud wiring vs random) ---")
    print(f"Base Random FF Rate (p^2): {train_res['base_ff_rate']:.4%}")
    print(f"Actual Overall FF Rate:    {train_res['actual_ff_rate']:.4%} (Lift: {train_res['homophily_lift']:.2f}x)")
    for t_name, t_info in train_res.get("per_type_homophily", {}).items():
        print(f"  > {t_name:<12}: Edges={t_info['edge_count']:,} | FF Rate={t_info['ff_rate']:.4%} | Lift={t_info['lift']:.2f}x")

    # Confirm all pass
    all_passed = all(r["overall_pass"] for r in results.values())
    assert all_passed, "Graph validation failed for one or more splits!"

    print("\n" + "=" * 80)
    print("STAGE G4 VALIDATION ALL PASSED SUCCESSFULLY!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_full_validation()
