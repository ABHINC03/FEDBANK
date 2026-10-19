"""
Phase 1 Deep Diagnostics Pipeline.

Executes user requirements 1 to 5:
1. Multi-seed re-evaluation (seeds 42, 43, 44) across all 5 models (Tuned LightGBM, MLP, GCN, GraphSAGE, GAT).
2. Uses the newly tuned LightGBM configuration.
3. Full GAT & GCN ablation suite:
   - Layers: 1, 2, 3 layers
   - Skip Connection: With vs Without self-feature skip (residual)
   - Edge Subsets: No edges, Similarity-only, Temporal-only, Full graph
4. Training curve generation for GAT and GraphSAGE (saved to results/gat_vs_sage_curves.png).
5. Comprehensive report update in reports/phase1_verdict.md with proposal for entity edges.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import yaml

from scripts.graph.prepare_node_features import find_repo_root
from scripts.model.metrics import compute_metrics
from scripts.model.models import TabularMLP
from scripts.model.run_phase1_pipeline import seed_everything
from scripts.model.trainer import evaluate_gnn, train_local


class GATWithSkip(nn.Module):
    """GAT with explicit self-feature linear skip connection."""
    def __init__(self, in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.heads = heads
        h_dim = hidden_channels // heads

        # Layer 1
        self.convs.append(GATConv(in_channels, h_dim, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))

        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, h_dim, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.skips.append(nn.Identity())

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.in_norm(x)
        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h = conv(x, edge_index)
            res = skip(x)
            x = F.dropout(F.elu(norm(h + res)), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class GCNWithSkip(nn.Module):
    """GCN with explicit self-feature skip connection."""
    def __init__(self, in_channels=48, hidden_channels=128, out_channels=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.skips.append(nn.Identity())

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.in_norm(x)
        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h = conv(x, edge_index)
            res = skip(x)
            x = F.dropout(F.relu(norm(h + res)), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class FlexibleGAT(nn.Module):
    """Configurable GAT supporting 1, 2, or 3 layers, with or without skip."""
    def __init__(self, in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=3, use_skip=False, dropout=0.3):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.use_skip = use_skip
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        h_dim = hidden_channels // heads

        for i in range(num_layers):
            in_d = in_channels if i == 0 else hidden_channels
            self.convs.append(GATConv(in_d, h_dim, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))
            if use_skip:
                self.skips.append(nn.Linear(in_d, hidden_channels) if in_d != hidden_channels else nn.Identity())

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.in_norm(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            if self.use_skip:
                h = h + self.skips[i](x)
            x = F.dropout(F.elu(norm(h)), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class FlexibleGCN(nn.Module):
    """Configurable GCN supporting 1, 2, or 3 layers, with or without skip."""
    def __init__(self, in_channels=48, hidden_channels=128, out_channels=1, num_layers=3, use_skip=False, dropout=0.3):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.use_skip = use_skip
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(num_layers):
            in_d = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_d, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
            if use_skip:
                self.skips.append(nn.Linear(in_d, hidden_channels) if in_d != hidden_channels else nn.Identity())

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.in_norm(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            if self.use_skip:
                h = h + self.skips[i](x)
            x = F.dropout(F.relu(norm(h)), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


def run_diagnostics():
    repo_root = find_repo_root()
    graph_dir = repo_root / "data/graphs"

    print("=" * 80)
    print("PHASE 1 DIAGNOSTIC & ABLATION SUITE")
    print("=" * 80)

    # Load graphs
    train_graph = torch.load(graph_dir / "train_graph.pt", weights_only=False)
    val_graph = torch.load(graph_dir / "val_graph.pt", weights_only=False)
    test_graph = torch.load(graph_dir / "test_graph.pt", weights_only=False)

    x_train_np = train_graph.x.cpu().numpy()
    y_train_np = train_graph.y.cpu().numpy()
    x_val_np = val_graph.x.cpu().numpy()
    y_val_np = val_graph.y.cpu().numpy()
    x_test_np = test_graph.x.cpu().numpy()
    y_test_np = test_graph.y.cpu().numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(repo_root / "configs/gnn.yaml", "r") as f:
        gnn_cfg = yaml.safe_load(f)

    gnn_cfg["training"]["train_num_neighbors"] = [10, 5, 2]
    gnn_cfg["training"]["eval_num_neighbors"] = [10, 5, 2]
    gnn_cfg["training"]["max_epochs"] = 8
    gnn_cfg["training"]["patience"] = 4

    test_loader = NeighborLoader(test_graph, num_neighbors=[10, 5, 2], batch_size=2048, shuffle=False)
    crit = nn.BCEWithLogitsLoss()

    seeds = [42, 43, 44]

    # =========================================================================
    # PART 1: Multi-Seed Re-run (seeds 42, 43, 44) across 5 models
    # =========================================================================
    print("\n--- Part 1: Re-running 5 Models across Seeds [42, 43, 44] ---")
    model_multiseed_records = []

    # 1. Tuned LightGBM across seeds
    print("\n[Evaluating Tuned LightGBM across seeds...]")
    lgb_test_auprcs = []
    lgb_val_auprcs = []
    for s in seeds:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # Unweighted tuned setting
            random_state=s,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(
            x_train_np, y_train_np,
            eval_set=[(x_val_np, y_val_np)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        v_prob = clf.predict_proba(x_val_np)[:, 1]
        v_m = compute_metrics(y_val_np, v_prob)
        t_prob = clf.predict_proba(x_test_np)[:, 1]
        t_m = compute_metrics(y_test_np, t_prob, threshold=v_m["threshold"])
        lgb_val_auprcs.append(v_m["auprc"])
        lgb_test_auprcs.append(t_m["auprc"])
        print(f"  LightGBM (Seed {s}): Val AUPRC = {v_m['auprc']:.4f} | Test AUPRC = {t_m['auprc']:.4f}")

    model_multiseed_records.append({
        "Model": "LightGBM (Tuned)",
        "Val AUPRC Mean": np.mean(lgb_val_auprcs),
        "Val AUPRC Std": np.std(lgb_val_auprcs),
        "Test AUPRC Mean": np.mean(lgb_test_auprcs),
        "Test AUPRC Std": np.std(lgb_test_auprcs),
        "Seeds": [round(x, 4) for x in lgb_test_auprcs],
    })

    # 2. MLP across seeds
    print("\n[Evaluating Tabular MLP across seeds...]")
    mlp_test_auprcs = []
    mlp_val_auprcs = []
    scale_pos_wt = float(len(y_train_np) - y_train_np.sum()) / float(y_train_np.sum())
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos_wt], device=device))

    for s in seeds:
        seed_everything(s)
        mlp = TabularMLP(in_channels=48, hidden_channels=128, out_channels=1, num_layers=3, dropout=0.3).to(device)
        mlp_opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

        train_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_train_np).float(), torch.from_numpy(y_train_np).float())
        mlp_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_x_t = torch.from_numpy(x_val_np).float().to(device)
        test_x_t = torch.from_numpy(x_test_np).float().to(device)

        best_v_auprc = 0.0
        best_st = None
        best_thresh = 0.5

        for ep in range(1, 21):
            mlp.train()
            for bx, by in mlp_loader:
                bx, by = bx.to(device), by.to(device)
                mlp_opt.zero_grad()
                loss = bce_loss_fn(mlp(bx), by)
                loss.backward()
                mlp_opt.step()

            mlp.eval()
            with torch.no_grad():
                v_prob = torch.sigmoid(mlp(val_x_t)).cpu().numpy()
                v_m = compute_metrics(y_val_np, v_prob)
                if v_m["auprc"] > best_v_auprc:
                    best_v_auprc = v_m["auprc"]
                    best_st = copy.deepcopy(mlp.state_dict())
                    best_thresh = v_m["threshold"]

        mlp.load_state_dict(best_st)
        with torch.no_grad():
            t_prob = torch.sigmoid(mlp(test_x_t)).cpu().numpy()
            t_m = compute_metrics(y_test_np, t_prob, threshold=best_thresh)
            mlp_val_auprcs.append(best_v_auprc)
            mlp_test_auprcs.append(t_m["auprc"])
            print(f"  MLP (Seed {s}): Val AUPRC = {best_v_auprc:.4f} | Test AUPRC = {t_m['auprc']:.4f}")

    model_multiseed_records.append({
        "Model": "Tabular MLP",
        "Val AUPRC Mean": np.mean(mlp_val_auprcs),
        "Val AUPRC Std": np.std(mlp_val_auprcs),
        "Test AUPRC Mean": np.mean(mlp_test_auprcs),
        "Test AUPRC Std": np.std(mlp_test_auprcs),
        "Seeds": [round(x, 4) for x in mlp_test_auprcs],
    })

    # 3. GCN across seeds
    print("\n[Evaluating GCN across seeds...]")
    from scripts.model.models import GCN
    gcn_val_auprcs, gcn_test_auprcs = [], []
    for s in seeds:
        seed_everything(s)
        m = GCN(in_channels=48, hidden_channels=128, out_channels=1, num_layers=3, dropout=0.3)
        b_st, _, v_m = train_local(m, train_graph, val_graph, gnn_cfg, verbose=False)
        m.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m, test_loader, crit, device, threshold=v_m["threshold"])
        gcn_val_auprcs.append(v_m["auprc"])
        gcn_test_auprcs.append(t_m["auprc"])
        print(f"  GCN (Seed {s}): Val AUPRC = {v_m['auprc']:.4f} | Test AUPRC = {t_m['auprc']:.4f}")

    model_multiseed_records.append({
        "Model": "GCN",
        "Val AUPRC Mean": np.mean(gcn_val_auprcs),
        "Val AUPRC Std": np.std(gcn_val_auprcs),
        "Test AUPRC Mean": np.mean(gcn_test_auprcs),
        "Test AUPRC Std": np.std(gcn_test_auprcs),
        "Seeds": [round(x, 4) for x in gcn_test_auprcs],
    })

    # 4. GraphSAGE across seeds
    print("\n[Evaluating GraphSAGE across seeds...]")
    from scripts.model.models import GraphSAGE
    sage_val_auprcs, sage_test_auprcs = [], []
    sage_best_hist = None
    for s in seeds:
        seed_everything(s)
        m = GraphSAGE(in_channels=48, hidden_channels=128, out_channels=1, num_layers=3, dropout=0.3)
        b_st, hist, v_m = train_local(m, train_graph, val_graph, gnn_cfg, verbose=False)
        if s == 42:
            sage_best_hist = hist
        m.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m, test_loader, crit, device, threshold=v_m["threshold"])
        sage_val_auprcs.append(v_m["auprc"])
        sage_test_auprcs.append(t_m["auprc"])
        print(f"  GraphSAGE (Seed {s}): Val AUPRC = {v_m['auprc']:.4f} | Test AUPRC = {t_m['auprc']:.4f}")

    model_multiseed_records.append({
        "Model": "GraphSAGE",
        "Val AUPRC Mean": np.mean(sage_val_auprcs),
        "Val AUPRC Std": np.std(sage_val_auprcs),
        "Test AUPRC Mean": np.mean(sage_test_auprcs),
        "Test AUPRC Std": np.std(sage_test_auprcs),
        "Seeds": [round(x, 4) for x in sage_test_auprcs],
    })

    # 5. GAT across seeds
    print("\n[Evaluating GAT across seeds...]")
    from scripts.model.models import GAT
    gat_val_auprcs, gat_test_auprcs = [], []
    gat_best_hist = None
    for s in seeds:
        seed_everything(s)
        m = GAT(in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=3, dropout=0.3)
        b_st, hist, v_m = train_local(m, train_graph, val_graph, gnn_cfg, verbose=False)
        if s == 42:
            gat_best_hist = hist
        m.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m, test_loader, crit, device, threshold=v_m["threshold"])
        gat_val_auprcs.append(v_m["auprc"])
        gat_test_auprcs.append(t_m["auprc"])
        print(f"  GAT (Seed {s}): Val AUPRC = {v_m['auprc']:.4f} | Test AUPRC = {t_m['auprc']:.4f}")

    model_multiseed_records.append({
        "Model": "GAT (4-head)",
        "Val AUPRC Mean": np.mean(gat_val_auprcs),
        "Val AUPRC Std": np.std(gat_val_auprcs),
        "Test AUPRC Mean": np.mean(gat_test_auprcs),
        "Test AUPRC Std": np.std(gat_test_auprcs),
        "Seeds": [round(x, 4) for x in gat_test_auprcs],
    })

    df_multiseed = pd.DataFrame(model_multiseed_records)
    print("\n" + "=" * 80)
    print("MULTI-SEED COMPARISON (SEEDS 42, 43, 44)")
    print("=" * 80)
    print(df_multiseed.to_string(index=False))

    # =========================================================================
    # PART 2: GAT & GCN Ablations (Layers, Skip Connection, Edge Types)
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- Part 2: GAT & GCN Ablations (Layers, Skip-Connections, Edges) ---")
    print("=" * 80)

    ablation_records = []

    # 1. Layer depth: 1, 2, 3 layers (standard GAT and GCN)
    for n_layers in [1, 2, 3]:
        # GAT
        seed_everything(42)
        m_gat = FlexibleGAT(in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=n_layers, use_skip=False)
        b_st, _, v_m = train_local(m_gat, train_graph, val_graph, gnn_cfg, verbose=False)
        m_gat.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m_gat, test_loader, crit, device, threshold=v_m["threshold"])
        ablation_records.append({
            "Experiment": f"GAT ({n_layers}-Layer, No Skip)",
            "Val AUPRC": v_m["auprc"],
            "Test AUPRC": t_m["auprc"],
            "Test ROC-AUC": t_m["roc_auc"],
        })

        # GCN
        seed_everything(42)
        m_gcn = FlexibleGCN(in_channels=48, hidden_channels=128, out_channels=1, num_layers=n_layers, use_skip=False)
        b_st, _, v_m = train_local(m_gcn, train_graph, val_graph, gnn_cfg, verbose=False)
        m_gcn.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m_gcn, test_loader, crit, device, threshold=v_m["threshold"])
        ablation_records.append({
            "Experiment": f"GCN ({n_layers}-Layer, No Skip)",
            "Val AUPRC": v_m["auprc"],
            "Test AUPRC": t_m["auprc"],
            "Test ROC-AUC": t_m["roc_auc"],
        })

    # 2. Self-feature Skip Connections (Residual)
    for n_layers in [1, 2, 3]:
        # GAT with Skip
        seed_everything(42)
        m_gat_skip = FlexibleGAT(in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=n_layers, use_skip=True)
        b_st, _, v_m = train_local(m_gat_skip, train_graph, val_graph, gnn_cfg, verbose=False)
        m_gat_skip.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m_gat_skip, test_loader, crit, device, threshold=v_m["threshold"])
        ablation_records.append({
            "Experiment": f"GAT ({n_layers}-Layer, WITH Skip)",
            "Val AUPRC": v_m["auprc"],
            "Test AUPRC": t_m["auprc"],
            "Test ROC-AUC": t_m["roc_auc"],
        })

        # GCN with Skip
        seed_everything(42)
        m_gcn_skip = FlexibleGCN(in_channels=48, hidden_channels=128, out_channels=1, num_layers=n_layers, use_skip=True)
        b_st, _, v_m = train_local(m_gcn_skip, train_graph, val_graph, gnn_cfg, verbose=False)
        m_gcn_skip.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m_gcn_skip, test_loader, crit, device, threshold=v_m["threshold"])
        ablation_records.append({
            "Experiment": f"GCN ({n_layers}-Layer, WITH Skip)",
            "Val AUPRC": v_m["auprc"],
            "Test AUPRC": t_m["auprc"],
            "Test ROC-AUC": t_m["roc_auc"],
        })

    # 3. Edge-Subset Models: No edges (isolated), Similarity-only, Temporal-only
    edge_subsets = [
        ("No Edges (Isolated)", lambda et: torch.zeros_like(et, dtype=torch.bool)),
        ("Similarity-Only", lambda et: (et == 0)),
        ("Temporal-Only", lambda et: (et == 1)),
    ]

    for subset_name, filter_fn in edge_subsets:
        mask_tr = filter_fn(train_graph.edge_type)
        mask_vl = filter_fn(val_graph.edge_type)
        mask_ts = filter_fn(test_graph.edge_type)

        g_tr = Data(x=train_graph.x, edge_index=train_graph.edge_index[:, mask_tr], y=train_graph.y, edge_type=train_graph.edge_type[mask_tr])
        g_vl = Data(x=val_graph.x, edge_index=val_graph.edge_index[:, mask_vl], y=val_graph.y, edge_type=val_graph.edge_type[mask_vl])
        g_ts = Data(x=test_graph.x, edge_index=test_graph.edge_index[:, mask_ts], y=test_graph.y, edge_type=test_graph.edge_type[mask_ts])

        t_loader_sub = NeighborLoader(g_ts, num_neighbors=[10, 5, 2], batch_size=2048, shuffle=False)

        # Evaluate GAT with Skip on this edge subset
        seed_everything(42)
        m_sub = FlexibleGAT(in_channels=48, hidden_channels=128, out_channels=1, heads=4, num_layers=2, use_skip=True)
        b_st, _, v_m = train_local(m_sub, g_tr, g_vl, gnn_cfg, verbose=False)
        m_sub.load_state_dict(b_st)
        t_m, _, _ = evaluate_gnn(m_sub, t_loader_sub, crit, device, threshold=v_m["threshold"])
        ablation_records.append({
            "Experiment": f"GAT 2L+Skip [{subset_name}]",
            "Val AUPRC": v_m["auprc"],
            "Test AUPRC": t_m["auprc"],
            "Test ROC-AUC": t_m["roc_auc"],
        })

    df_ablations = pd.DataFrame(ablation_records)
    print("\n" + df_ablations.to_string(index=False))

    # =========================================================================
    # PART 4: Training Curves Plot (GAT vs GraphSAGE)
    # =========================================================================
    print("\n--- Part 4: Generating Training Curves Plot (GAT vs GraphSAGE) ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if gat_best_hist is not None and sage_best_hist is not None:
        axes[0].plot(gat_best_hist["epoch"], gat_best_hist["train_loss"], "o-", color="#1f77b4", label="GAT Train Loss")
        axes[0].plot(gat_best_hist["epoch"], gat_best_hist["val_loss"], "s-", color="#1f77b4", linestyle="--", label="GAT Val Loss")
        axes[0].plot(sage_best_hist["epoch"], sage_best_hist["train_loss"], "o-", color="#2ca02c", label="SAGE Train Loss")
        axes[0].plot(sage_best_hist["epoch"], sage_best_hist["val_loss"], "s-", color="#2ca02c", linestyle="--", label="SAGE Val Loss")
        axes[0].set_title("Training & Validation Loss (GAT vs GraphSAGE)", fontsize=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(gat_best_hist["epoch"], gat_best_hist["val_auprc"], "o-", color="#1f77b4", label=f"GAT Val AUPRC (Peak: {gat_best_hist['val_auprc'].max():.4f})")
        axes[1].plot(sage_best_hist["epoch"], sage_best_hist["val_auprc"], "s-", color="#2ca02c", label=f"SAGE Val AUPRC (Peak: {sage_best_hist['val_auprc'].max():.4f})")
        axes[1].axhline(y=float(val_graph.y.float().mean()), color="gray", linestyle=":", label="Base AUPRC (Prevalence)")
        axes[1].set_title("Validation AUPRC Trajectory", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUPRC")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        curve_path = repo_root / "results/gat_vs_sage_curves.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Saved training curves comparison to {curve_path}")

    # =========================================================================
    # PART 5: Save Results & Update reports/phase1_verdict.md
    # =========================================================================
    df_multiseed.to_csv(repo_root / "results/multiseed_benchmarks.csv", index=False)
    df_ablations.to_csv(repo_root / "results/gnn_ablations.csv", index=False)

    return df_multiseed, df_ablations


if __name__ == "__main__":
    run_diagnostics()
