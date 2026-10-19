"""
Evaluation Metrics for FEDBANK Fraud Detection.

Implements all required evaluation functions:
- AUPRC (primary metric)
- ROC-AUC
- Best F1 score and threshold selection (threshold chosen on validation only)
- Recall at 1% FPR
- Precision at top-k
- Baseline AUPRC (positive class prevalence)
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Compute full evaluation metrics suite.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted fraud probabilities in [0, 1].
        threshold: Decision threshold. If None, computes optimal threshold maximizing F1.
        split_name: Name of split being evaluated.

    Returns:
        Dictionary of metric scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # 1. Primary metrics: AUPRC and ROC-AUC
    auprc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    base_auprc = float(np.mean(y_true))

    # 2. Optimal F1 and threshold selection
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid zero division
    denom = precisions + recalls
    denom[denom == 0] = 1.0
    f1_scores = 2 * (precisions * recalls) / denom

    best_idx = np.argmax(f1_scores)
    best_f1_val = float(f1_scores[best_idx])
    
    # Threshold array is 1 element shorter than precision/recall
    if best_idx < len(thresholds):
        optimal_thresh = float(thresholds[best_idx])
    else:
        optimal_thresh = 0.5

    applied_thresh = threshold if threshold is not None else optimal_thresh

    # 3. Binary predictions using threshold
    y_pred = (y_prob >= applied_thresh).astype(int)
    f1_at_thresh = float(f1_score(y_true, y_pred, zero_division=0))

    # 4. Recall at 1% FPR
    fpr, tpr, roc_threshs = roc_curve(y_true, y_prob)
    idx_1pct = np.searchsorted(fpr, 0.01, side="right") - 1
    idx_1pct = max(0, min(idx_1pct, len(tpr) - 1))
    recall_at_1pct_fpr = float(tpr[idx_1pct])

    # 5. Precision at top-1% and top-5%
    N = len(y_true)
    top_1pct_k = max(1, int(0.01 * N))
    top_5pct_k = max(1, int(0.05 * N))

    top_1pct_indices = np.argsort(y_prob)[-top_1pct_k:]
    precision_top_1pct = float(np.mean(y_true[top_1pct_indices]))

    top_5pct_indices = np.argsort(y_prob)[-top_5pct_k:]
    precision_top_5pct = float(np.mean(y_true[top_5pct_indices]))

    return {
        "auprc": auprc,
        "base_auprc": base_auprc,
        "auprc_lift": auprc / max(base_auprc, 1e-6),
        "roc_auc": roc_auc,
        "f1": f1_at_thresh,
        "best_f1": best_f1_val,
        "threshold": applied_thresh,
        "recall_at_1pct_fpr": recall_at_1pct_fpr,
        "precision_top_1pct": precision_top_1pct,
        "precision_top_5pct": precision_top_5pct,
    }
