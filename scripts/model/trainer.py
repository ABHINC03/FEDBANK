"""
Federation-Ready Training Module.

Provides train_local(...) conforming to the FEDBANK specification:
- Mini-batch NeighborLoader training.
- LayerNorm compatible (no batch statistics).
- Returns pure state_dict and metric history.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from scripts.model.metrics import compute_metrics


class FocalLoss(nn.Module):
    """Focal loss with alpha balancing and gamma focusing for extreme class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


def evaluate_gnn(
    model: nn.Module,
    loader: NeighborLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate GNN model over a NeighborLoader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            # Evaluate only on seed nodes in the batch (batch.batch_size)
            batch_size = getattr(batch, "batch_size", batch.num_nodes)
            batch_logits = out[:batch_size]
            batch_y = batch.y[:batch_size].to(torch.float32)

            loss = criterion(batch_logits, batch_y)
            probs = torch.sigmoid(batch_logits).cpu().numpy()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_true_list.append(batch_y.cpu().numpy())
            y_prob_list.append(probs)

    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    avg_loss = total_loss / max(total_samples, 1)

    metrics = compute_metrics(y_true_all, y_prob_all, threshold=threshold)
    metrics["loss"] = avg_loss

    return metrics, y_true_all, y_prob_all


def train_local(
    model: nn.Module,
    train_data: Data,
    val_data: Data,
    cfg: Dict[str, Any],
    live_plotter: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, Dict[str, float]]:
    """
    Federation-ready training function for a single node / bank.

    Args:
        model: GNN or MLP PyTorch model.
        train_data: PyG Data object for training.
        val_data: PyG Data object for validation.
        cfg: Training hyperparameter dictionary.
        live_plotter: Optional callback called each epoch with history and current PR curve.
        verbose: If True, prints per-epoch metrics.

    Returns:
        best_state_dict: Best model state dict on validation AUPRC.
        history_df: Training and validation metric history per epoch.
        best_val_metrics: Validation metrics at optimal epoch.
    """
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device)

    # 1. Hyperparameters
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 2048)
    num_neighbors = train_cfg.get("train_num_neighbors", [15, 10, 5])
    eval_num_neighbors = train_cfg.get("eval_num_neighbors", [15, 10, 5])
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_epochs = train_cfg.get("max_epochs", 100)
    patience = train_cfg.get("patience", 10)
    loss_type = train_cfg.get("loss", "bce_with_logits")
    focal_gamma = train_cfg.get("focal_gamma", 2.0)

    # 2. Compute pos_weight from train labels
    y_train = train_data.y.to(torch.float32)
    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    # 3. Loss function
    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. Data Loaders
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        val_data,
        num_neighbors=eval_num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_auprc = -1.0
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        num_batches = len(train_loader)
        for b_idx, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            batch_size_actual = getattr(batch, "batch_size", batch.num_nodes)
            batch_logits = out[:batch_size_actual]
            batch_y = batch.y[:batch_size_actual].to(torch.float32)

            loss = criterion(batch_logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size_actual
            train_samples += batch_size_actual

            if verbose and (b_idx % 25 == 0 or b_idx == num_batches):
                print(f"  [Epoch {epoch:02d}] Batch {b_idx:03d}/{num_batches:03d} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / max(train_samples, 1)
        epoch_time = time.time() - epoch_start

        # Evaluate on validation
        val_metrics, val_true, val_probs = evaluate_gnn(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        peak_gpu_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_auprc": val_metrics["auprc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["best_f1"],
            "val_recall_1pct_fpr": val_metrics["recall_at_1pct_fpr"],
            "val_threshold": val_metrics["threshold"],
            "epoch_time_s": epoch_time,
            "peak_gpu_mb": peak_gpu_mb,
        }
        history.append(record)

        # Check early stopping on val AUPRC
        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            improved_str = " *"
        else:
            epochs_no_improve += 1
            improved_str = ""

        if verbose:
            print(
                f"Epoch {epoch:02d}/{max_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUPRC: {val_metrics['auprc']:.4f} | "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"F1: {val_metrics['best_f1']:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"VRAM: {peak_gpu_mb:.0f}MB{improved_str}"
            )

        if live_plotter is not None:
            live_plotter({
                "history": pd.DataFrame(history),
                "val_true": val_true,
                "val_probs": val_probs,
                "best_epoch": best_epoch,
                "peak_gpu_mb": peak_gpu_mb,
            })

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, AUPRC: {best_val_auprc:.4f})")
            break

    history_df = pd.DataFrame(history)
    return best_state_dict, history_df, best_val_metrics
