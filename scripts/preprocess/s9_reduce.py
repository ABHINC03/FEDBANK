"""
s9_reduce.py — Dimensionality reduction (S9).
FEDBANK_preprocessing_spec.md Sections 8.1–8.6.
Steps: hard prune → correlation pruning → importance selection → PSI screen → PCA track.
All fitted on TRAIN.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import json
import time
import numpy as np
import cudf
import cupy as cp
from cuml.decomposition import PCA

from .gpu_env import vram, free_gpu, check_vram


_META_COLS = {"TransactionID", "TransactionDT", "isFraud", "split", "window_id",
              "card_key_id", "transaction_month", "transaction_year"}


def _feature_cols(df: cudf.DataFrame) -> list[str]:
    """All non-meta columns eligible as model features."""
    exclude_time = {"transaction_month", "transaction_year"}
    return [
        c for c in df.columns
        if c not in _META_COLS and c not in exclude_time
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8.1 Hard prune
# ─────────────────────────────────────────────────────────────────────────────

def hard_prune(df: cudf.DataFrame, train_mask, feature_cols: list[str],
               protect: list[str]) -> tuple[list[str], list[str]]:
    drop = []
    kept = []
    SAMPLE = 20_000
    n_train = int(train_mask.sum())
    cp.random.seed(42)
    sample_idx = cp.random.choice(n_train, min(SAMPLE, n_train), replace=False)
    train_pos   = cp.where(train_mask.to_cupy())[0]
    sample_pos  = train_pos[sample_idx]

    for col in feature_cols:
        if col in protect:
            kept.append(col)
            continue
        try:
            tc = df[col][train_mask]
            std  = float(tc.std())
            # Drop near-zero std
            if std < 1e-6:
                drop.append(col)
                continue
            # Drop if ≥ 99.5% one value
            vc = tc.value_counts()
            top_share = float(vc.iloc[0]) / len(tc) if len(vc) > 0 else 0.0
            if top_share >= 0.995:
                drop.append(col)
                continue
        except Exception:
            pass
        kept.append(col)

    print(f"[S9.8.1] Hard prune: drop {len(drop)}, keep {len(kept)}")
    return kept, drop


# ─────────────────────────────────────────────────────────────────────────────
# 8.2 Correlation pruning
# ─────────────────────────────────────────────────────────────────────────────

def _mann_whitney_auc(train_vals: cp.ndarray, labels: cp.ndarray) -> float:
    """Univariate AUC via Mann-Whitney rank sum."""
    try:
        n = len(train_vals)
        order = cp.argsort(train_vals)
        ranks = cp.empty(n, dtype=cp.float32)
        ranks[order] = cp.arange(1, n + 1, dtype=cp.float32)
        pos_mask = labels == 1
        n_pos = int(pos_mask.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        rank_sum_pos = float(ranks[pos_mask].sum())
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(max(auc, 1 - auc))
    except Exception:
        return 0.5


def correlation_prune(df: cudf.DataFrame, train_mask, feature_cols: list[str],
                      protect: list[str], corr_thr: float = 0.95) -> tuple[list[str], list[str]]:
    print(f"[S9.8.2] Correlation pruning on {len(feature_cols)} cols …")
    train_labels = df["isFraud"][train_mask].to_cupy().astype("float32")

    # Compute AUC for each feature (on train)
    aucs: dict[str, float] = {}
    for col in feature_cols:
        arr = df[col][train_mask].to_cupy(dtype="float32", na_value=0.0)
        arr = cp.nan_to_num(arr, nan=0.0)
        aucs[col] = _mann_whitney_auc(arr, train_labels)

    # Sort by AUC desc (best first)
    sorted_cols = sorted(feature_cols, key=lambda c: aucs.get(c, 0.5), reverse=True)

    # Compute correlation matrix in blocks (handle up to ~300 cols at once on 6 GB)
    MAX_FULL = 250
    n_cols = len(sorted_cols)

    # Build full float32 train matrix (column-major is better for corrcoef)
    print(f"[S9.8.2] Building {n_cols}-col train matrix …")
    X_parts = []
    VBLOCK = 64
    for b in range(0, n_cols, VBLOCK):
        block = sorted_cols[b:b+VBLOCK]
        cols_cp = [df[c][train_mask].to_cupy(dtype="float32", na_value=0.0) for c in block]
        X_parts.append(cp.stack(cols_cp, axis=1))
    X = cp.concatenate(X_parts, axis=1)   # (n_train, n_cols)
    del X_parts
    free_gpu()

    # Greedy correlation pruning
    drop_corr = set()
    kept_mask  = [True] * n_cols

    # Process in block pairs if > MAX_FULL cols; otherwise full matrix
    if n_cols <= MAX_FULL:
        R = cp.corrcoef(X.T)   # (n_cols, n_cols)
        for i in range(n_cols):
            if not kept_mask[i]:
                continue
            for j in range(i + 1, n_cols):
                if not kept_mask[j]:
                    continue
                if abs(float(R[i, j])) >= corr_thr:
                    # Drop lower-AUC (j, since sorted_cols is AUC-desc)
                    c_j = sorted_cols[j]
                    if c_j not in protect:
                        kept_mask[j] = False
                        drop_corr.add(c_j)
        del R
    else:
        # Block-pair approach
        print("[S9.8.2] Large matrix: block-pair correlation …")
        for bi in range(0, n_cols, MAX_FULL):
            for bj in range(bi, n_cols, MAX_FULL):
                Xi = X[:, bi:bi+MAX_FULL]
                Xj = X[:, bj:bj+MAX_FULL]
                # Cross-correlation: (Xi.T @ Xj) / (n * std_i * std_j)
                n_tr = X.shape[0]
                mu_i = Xi.mean(axis=0, keepdims=True)
                mu_j = Xj.mean(axis=0, keepdims=True)
                std_i = Xi.std(axis=0) + 1e-8
                std_j = Xj.std(axis=0) + 1e-8
                Xi_c = (Xi - mu_i) / std_i
                Xj_c = (Xj - mu_j) / std_j
                R_block = (Xi_c.T @ Xj_c) / n_tr   # (len_i, len_j)

                for li in range(R_block.shape[0]):
                    gi = bi + li
                    if not kept_mask[gi]:
                        continue
                    start_lj = (li + 1) if bi == bj else 0
                    for lj in range(start_lj, R_block.shape[1]):
                        gj = bj + lj
                        if not kept_mask[gj]:
                            continue
                        if abs(float(R_block[li, lj])) >= corr_thr:
                            c_j = sorted_cols[gj]
                            if c_j not in protect:
                                kept_mask[gj] = False
                                drop_corr.add(c_j)
                del R_block
                free_gpu()

    kept_after_corr = [c for c, k in zip(sorted_cols, kept_mask) if k]
    drop_corr_list = list(drop_corr)
    print(f"[S9.8.2] Correlation prune: drop {len(drop_corr_list)}, keep {len(kept_after_corr)}")
    del X
    free_gpu()
    return kept_after_corr, drop_corr_list, aucs


# ─────────────────────────────────────────────────────────────────────────────
# 8.3 Importance-based selection
# ─────────────────────────────────────────────────────────────────────────────

def importance_select(df: cudf.DataFrame, train_mask, val_mask,
                      feature_cols: list[str], protect: list[str],
                      k: int, cfg) -> tuple[list[str], dict]:
    """
    Try GPU XGBoost; fallback to univariate AUC.
    Returns (selected_cols, importance_dict).
    """
    aucs = {}
    importance = {}

    try:
        import xgboost as xgb
        print("[S9.8.3] Fitting XGBoost for feature importance …")

        train_labels = df["isFraud"][train_mask].to_cupy().astype("float32")
        val_labels   = df["isFraud"][val_mask].to_cupy().astype("float32")

        VBLOCK = 64
        X_train_parts, X_val_parts = [], []
        for b in range(0, len(feature_cols), VBLOCK):
            block = feature_cols[b:b+VBLOCK]
            Xt = cp.stack([df[c][train_mask].to_cupy(dtype="float32", na_value=0.0) for c in block], axis=1)
            Xv = cp.stack([df[c][val_mask].to_cupy(dtype="float32", na_value=0.0)   for c in block], axis=1)
            X_train_parts.append(Xt)
            X_val_parts.append(Xv)
        X_train = cp.concatenate(X_train_parts, axis=1)
        X_val   = cp.concatenate(X_val_parts, axis=1)
        del X_train_parts, X_val_parts
        free_gpu()

        n_pos = float(train_labels.sum())
        n_neg = float(len(train_labels)) - n_pos

        try:
            dtrain = xgb.QuantileDMatrix(X_train, label=train_labels, max_bin=128)
            dval   = xgb.QuantileDMatrix(X_val,   label=val_labels,   max_bin=128)
        except Exception:
            dtrain = xgb.DMatrix(X_train, label=train_labels)
            dval   = xgb.DMatrix(X_val,   label=val_labels)

        del X_train, X_val
        free_gpu()

        params = {
            "tree_method": "hist", "device": "cuda",
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "scale_pos_weight": n_neg / max(n_pos, 1),
            "eval_metric": "aucpr", "random_state": cfg.SEED,
        }
        bst = xgb.train(
            params, dtrain,
            num_boost_round=400,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=50,
        )
        scores = bst.get_score(importance_type="gain")
        importance = {feature_cols[i]: scores.get(f"f{i}", 0.0) for i in range(len(feature_cols))}
        path_used = "xgboost"
        del bst, dtrain, dval
        free_gpu()

    except Exception as e:
        print(f"[S9.8.3] XGBoost failed ({e}); falling back to univariate AUC …")
        train_labels = df["isFraud"][train_mask].to_cupy().astype("float32")
        for col in feature_cols:
            arr = df[col][train_mask].to_cupy(dtype="float32", na_value=0.0)
            arr = cp.nan_to_num(arr, nan=0.0)
            importance[col] = _mann_whitney_auc(arr, train_labels)
        path_used = "univariate_auc"

    # Select top K (with protect)
    sorted_imp = sorted(feature_cols, key=lambda c: importance.get(c, 0.0), reverse=True)
    selected = []
    for c in protect:
        if c in feature_cols and c not in selected:
            selected.append(c)
    for c in sorted_imp:
        if len(selected) >= k:
            break
        if c not in selected:
            selected.append(c)

    print(f"[S9.8.3] {path_used}: selected {len(selected)} features (target K={k})")
    return selected, importance, path_used


# ─────────────────────────────────────────────────────────────────────────────
# 8.4 PSI screen
# ─────────────────────────────────────────────────────────────────────────────

def psi(train_x: cp.ndarray, other_x: cp.ndarray,
        bins: int = 10, eps: float = 1e-6) -> float:
    train_x = train_x[~cp.isnan(train_x)]
    other_x = other_x[~cp.isnan(other_x)]
    if len(train_x) == 0 or len(other_x) == 0:
        return 0.0
    qs = cp.unique(cp.quantile(train_x, cp.linspace(0, 1, bins + 1)))
    if len(qs) < 2:
        return 0.0
    qs[0], qs[-1] = -cp.inf, cp.inf
    a = cp.histogram(train_x, qs)[0] / len(train_x) + eps
    b = cp.histogram(other_x, qs)[0] / len(other_x) + eps
    return float(cp.sum((a - b) * cp.log(a / b)))


def psi_screen(df: cudf.DataFrame, train_mask, val_mask,
               feature_cols: list[str], drift_filter: bool = False,
               protect: list[str] = []) -> tuple[list[str], dict]:
    psi_vals: dict[str, float] = {}
    train_labels = df["isFraud"][train_mask].to_cupy().astype("float32")
    for col in feature_cols:
        try:
            tx = df[col][train_mask].to_cupy(dtype="float32", na_value=float("nan"))
            vx = df[col][val_mask].to_cupy(dtype="float32", na_value=float("nan"))
            tx = tx[~cp.isnan(tx)]
            vx = vx[~cp.isnan(vx)]
            psi_vals[col] = psi(tx, vx)
        except Exception:
            psi_vals[col] = 0.0

    high_psi = [c for c, v in psi_vals.items() if v > 0.25]
    print(f"[S9.8.4] PSI > 0.25: {len(high_psi)} features")

    if drift_filter and high_psi:
        drop = [c for c in high_psi if c not in protect]
        kept = [c for c in feature_cols if c not in drop]
        print(f"[S9.8.4] DRIFT_FILTER=True: dropping {len(drop)} high-PSI features")
    else:
        kept = feature_cols

    return kept, psi_vals


# ─────────────────────────────────────────────────────────────────────────────
# 8.5 PCA track for V-blocks
# ─────────────────────────────────────────────────────────────────────────────

def pca_v_blocks(df: cudf.DataFrame, train_mask,
                 v_cols: list[str], cfg) -> tuple[list[str], dict]:
    """
    Group V-features by identical train null-count (same block).
    Run cuML PCA per block. Return new PCA column names and the component store.
    """
    from collections import defaultdict

    # Group by null count
    null_count: dict[str, int] = {}
    for c in v_cols:
        null_count[c] = int(df[c][train_mask].isna().sum())

    groups: dict[int, list[str]] = defaultdict(list)
    for c in v_cols:
        groups[null_count[c]].append(c)

    pca_cols_added = []
    pca_store: dict[str, dict] = {}   # block_id → {components, mean, std, n_components}

    for block_id, (nc, bcols) in enumerate(sorted(groups.items())):
        n_c = len(bcols)
        if n_c < 2:
            continue

        # Build train matrix
        Xt = cp.stack(
            [df[c][train_mask].to_cupy(dtype="float32", na_value=0.0) for c in bcols],
            axis=1
        )
        Xt = cp.ascontiguousarray(Xt.astype("float32"))

        # Standardize on train
        mu  = Xt.mean(axis=0)
        std = Xt.std(axis=0) + 1e-8
        Xt_std = (Xt - mu) / std

        # PCA
        n_comp = min(10, n_c)
        try:
            pca = PCA(n_components=n_comp, svd_solver="full")
            pca.fit(Xt_std)
            evr = cp.cumsum(cp.array(pca.explained_variance_ratio_))
            # Keep smallest n with cumvar >= PCA_VAR; hard cap 5
            n_keep = int((evr < cfg.PCA_VAR).sum()) + 1
            n_keep = min(n_keep, cfg.PCA_MAX_PER_BLOCK, n_comp)
        except Exception as e:
            print(f"[S9.8.5] PCA failed for block nc={nc}: {e}; skipping")
            del Xt, Xt_std
            free_gpu()
            continue

        # Store parameters
        components_np = cp.asnumpy(pca.components_[:n_keep])   # (n_keep, n_c)
        mu_np  = cp.asnumpy(mu)
        std_np = cp.asnumpy(std)
        pca_store[f"block_{block_id}"] = {
            "cols": bcols, "components": components_np,
            "mean": mu_np, "std": std_np, "n_components": n_keep,
        }

        # Transform all splits
        for split_val, split_name in [(0,"tr"),(1,"va"),(2,"te"),(3,"mo")]:
            mask = df["split"] == split_val
            X_sp = cp.stack(
                [df[c][mask].to_cupy(dtype="float32", na_value=0.0) for c in bcols],
                axis=1
            )
            X_sp_std = (X_sp - mu) / std
            proj = X_sp_std @ cp.array(components_np.T)   # (n_rows, n_keep)
            for ki in range(n_keep):
                cname = f"Vpca_b{block_id}_{ki}"
                if cname not in df.columns:
                    df[cname] = cudf.Series(
                        cp.zeros(len(df), dtype=cp.float32), index=df.index
                    )
                df.loc[mask, cname] = cudf.Series(
                    proj[:, ki].astype(cp.float32), index=df[mask].index
                )
            del X_sp, X_sp_std, proj

        del Xt, Xt_std
        free_gpu()

        for ki in range(n_keep):
            pca_cols_added.append(f"Vpca_b{block_id}_{ki}")

    total_pca = len(pca_cols_added)
    if total_pca > 20:
        print(f"[S9.8.5] WARNING: {total_pca} PCA components > 20; consider raising PCA threshold.")

    return pca_cols_added, pca_store


# ─────────────────────────────────────────────────────────────────────────────
# Main S9 run
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg, df: cudf.DataFrame, state: dict) -> tuple[cudf.DataFrame, dict, dict]:
    check_vram(cfg.VRAM_MIN_MB, tag=" S9-start")
    t0 = time.time()

    train_mask = df["split"] == 0
    val_mask   = df["split"] == 1

    feature_cols_init = _feature_cols(df)
    print(f"[S9] Starting with {len(feature_cols_init)} feature columns")

    # 8.1 Hard prune
    kept, drop_hard = hard_prune(df, train_mask, feature_cols_init, cfg.PROTECT)
    free_gpu()

    # 8.2 Correlation pruning
    kept, drop_corr, aucs = correlation_prune(df, train_mask, kept, cfg.PROTECT, cfg.CORR_THR)
    free_gpu()

    # 8.3 Importance selection
    selected_track_a, importance, imp_path = importance_select(
        df, train_mask, val_mask, kept, cfg.PROTECT, cfg.K_SEL, cfg
    )
    free_gpu()

    # 8.4 PSI screen
    selected_track_a, psi_vals = psi_screen(
        df, train_mask, val_mask, selected_track_a,
        drift_filter=cfg.DRIFT_FILTER, protect=cfg.PROTECT
    )
    top10_psi = sorted(psi_vals.items(), key=lambda x: x[1], reverse=True)[:10]
    free_gpu()

    # 8.6 Final checks for Track A
    # No nulls, no inf, all float32
    for col in selected_track_a:
        nc = int(df[col].isna().sum())
        if nc > 0:
            print(f"[S9] WARNING: {col} has {nc} nulls in Track A — filling 0")
            df[col] = df[col].fillna(0.0)

    # Max pairwise |corr| on train < 0.95
    print(f"[S9] Track A: {len(selected_track_a)} features selected.")

    # 8.5 PCA track
    v_cols_sel = [c for c in kept if c.startswith("V") and not c.endswith("_missing")
                  and not c.startswith("Vpca")]
    print(f"[S9.8.5] Running PCA on {len(v_cols_sel)} V-features …")
    pca_new_cols, pca_store = pca_v_blocks(df, train_mask, v_cols_sel, cfg)

    # Track B = Track A non-V features + PCA components
    non_v_sel = [c for c in selected_track_a if not c.startswith("V")]
    track_b = non_v_sel + pca_new_cols

    # Cap Track B at 64; drop lowest-importance non-protected if over
    if len(track_b) > 64:
        protected_in_b = [c for c in track_b if c in cfg.PROTECT]
        non_protected  = [c for c in track_b if c not in protected_in_b]
        sorted_by_imp  = sorted(non_protected, key=lambda c: importance.get(c, 0.0), reverse=True)
        budget = 64 - len(protected_in_b)
        track_b = protected_in_b + sorted_by_imp[:budget]

    print(f"[S9] Track A: {len(selected_track_a)} | Track B: {len(track_b)}")

    # Save PCA params
    pca_npz_path = cfg.ARTIFACTS_DIR / "pca_V.npz"
    save_dict = {}
    for block_id, bdata in pca_store.items():
        safe_id = block_id.replace("/", "_")
        save_dict[f"{safe_id}_components"] = bdata["components"]
        save_dict[f"{safe_id}_mean"]       = bdata["mean"]
        save_dict[f"{safe_id}_std"]        = bdata["std"]
    np.savez_compressed(pca_npz_path, **save_dict)

    state["s9"] = {
        "drop_hard": drop_hard,
        "drop_corr": drop_corr,
        "importance_path": imp_path,
        "track_a": selected_track_a,
        "track_b": track_b,
        "psi_vals": {k: float(v) for k, v in psi_vals.items()},
        "top10_psi": [(k, float(v)) for k, v in top10_psi],
        "importance": {k: float(v) for k, v in importance.items()},
        "aucs": {k: float(v) for k, v in aucs.items()},
        "pca_cols": pca_new_cols,
        "pca_store_keys": list(pca_store.keys()),
    }

    vram(tag=" S9-end")
    print(f"[S9] Done in {time.time()-t0:.1f}s")
    return df, state, {"track_a": selected_track_a, "track_b": track_b}
