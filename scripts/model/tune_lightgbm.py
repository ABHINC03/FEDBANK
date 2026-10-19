"""
Diagnostic Step 2: Re-tune LightGBM on Validation AUPRC.

Systematically evaluates:
- scale_pos_weight: [1.0, 5.39 (sqrt), 29.0 (full)]
- eval_metric: 'average_precision' (AUPRC)
- num_leaves / max_depth
- learning_rate
Reports the exact configuration that maximizes validation AUPRC.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from scripts.graph.prepare_node_features import find_repo_root
from scripts.model.metrics import compute_metrics


def tune_lightgbm():
    repo_root = find_repo_root()
    train_graph = torch.load(repo_root / "data/graphs/train_graph.pt", weights_only=False)
    val_graph = torch.load(repo_root / "data/graphs/val_graph.pt", weights_only=False)
    test_graph = torch.load(repo_root / "data/graphs/test_graph.pt", weights_only=False)

    x_train = train_graph.x.cpu().numpy()
    y_train = train_graph.y.cpu().numpy()
    x_val = val_graph.x.cpu().numpy()
    y_val = val_graph.y.cpu().numpy()
    x_test = test_graph.x.cpu().numpy()
    y_test = test_graph.y.cpu().numpy()

    full_pos_weight = float(len(y_train) - y_train.sum()) / float(y_train.sum())
    sqrt_pos_weight = float(np.sqrt(full_pos_weight))

    candidate_configs = [
        # 1. Default scale_pos_weight=1.0 with average_precision metric
        {"name": "unweighted_depth6", "scale_pos_weight": 1.0, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05},
        {"name": "unweighted_depth8", "scale_pos_weight": 1.0, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.05},
        {"name": "unweighted_lr03", "scale_pos_weight": 1.0, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.03},
        # 2. Sqrt scale_pos_weight (~5.4)
        {"name": "sqrt_pos_weight", "scale_pos_weight": sqrt_pos_weight, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05},
        {"name": "sqrt_pos_weight_depth8", "scale_pos_weight": sqrt_pos_weight, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.05},
        # 3. Full scale_pos_weight (~29.0)
        {"name": "full_pos_weight", "scale_pos_weight": full_pos_weight, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05},
        {"name": "full_pos_weight_depth8", "scale_pos_weight": full_pos_weight, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.05},
    ]

    print("=" * 80)
    print("STEP 2: Tuning LightGBM with Early Stopping on Validation AUPRC")
    print("=" * 80)

    results = []
    best_val_auprc = -1.0
    best_cfg = None

    for cfg in candidate_configs:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            num_leaves=cfg["num_leaves"],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_probs = clf.predict_proba(x_val)[:, 1]
        val_m = compute_metrics(y_val, val_probs)

        test_probs = clf.predict_proba(x_test)[:, 1]
        test_m = compute_metrics(y_test, test_probs, threshold=val_m["threshold"])

        res = {
            "Config": cfg["name"],
            "PosWeight": f"{cfg['scale_pos_weight']:.2f}",
            "Depth/Leaves": f"{cfg['max_depth']}/{cfg['num_leaves']}",
            "LR": cfg["learning_rate"],
            "Best Trees": clf.best_iteration_,
            "Val AUPRC": val_m["auprc"],
            "Val ROC-AUC": val_m["roc_auc"],
            "Test AUPRC": test_m["auprc"],
            "Test ROC-AUC": test_m["roc_auc"],
        }
        results.append(res)
        print(f"[{cfg['name']:<22}] Trees: {clf.best_iteration_:3d} | Val AUPRC: {val_m['auprc']:.4f} | Test AUPRC: {test_m['auprc']:.4f}")

        if val_m["auprc"] > best_val_auprc:
            best_val_auprc = val_m["auprc"]
            best_cfg = cfg
            best_cfg["best_iteration"] = clf.best_iteration_

    df_res = pd.DataFrame(results)
    print("\n" + df_res.to_string(index=False))
    print(f"\nBest Config on Validation AUPRC: {best_cfg}")
    return df_res, best_cfg


if __name__ == "__main__":
    tune_lightgbm()
