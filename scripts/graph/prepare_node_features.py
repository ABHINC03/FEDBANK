"""
Stage G1 — Node Features Preparation.

Loads preprocessed tabular features and metadata, validates them against the frozen
FEDBANK Schema v1 (fedbank_schema_v1.json), checks row alignments, ensures zero NaN/Inf,
reorders columns if needed, and produces model-ready node feature tensors.

Run as:
    python -m scripts.graph.prepare_node_features
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Optional torch import with fallback
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


FORBIDDEN_COLUMNS = {
    "isFraud",
    "TransactionDT",
    "TransactionID",
    "card_key_id",
    "split",
    "window_id",
}

EXPECTED_SIZES = {
    "train": 273000,
    "val": 58500,
    "test": 58500,
    "monitor": 200540,
}

EXPECTED_FRAUD_RATES = {
    "train": 0.0333,
    "val": 0.0362,
    "test": 0.0370,
    "monitor": 0.0364,
}


@dataclass
class NodeFeaturesBundle:
    """Container for processed node features and aligned metadata."""
    x: Union[np.ndarray, "torch.Tensor"]
    y: Union[np.ndarray, "torch.Tensor"]
    transaction_dt: Union[np.ndarray, "torch.Tensor"]
    transaction_id: Union[np.ndarray, "torch.Tensor"]
    window_id: Optional[Union[np.ndarray, "torch.Tensor"]]
    feature_names: List[str]
    split: str

    def to_torch(self, device: str = "cpu") -> NodeFeaturesBundle:
        """Convert all arrays to torch Tensors."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed or failed to import.")
        
        x_t = torch.as_tensor(self.x, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(self.y, dtype=torch.int8, device=device)
        dt_t = torch.as_tensor(self.transaction_dt, dtype=torch.int64, device=device)
        tid_t = torch.as_tensor(self.transaction_id, dtype=torch.int64, device=device)
        wid_t = (
            torch.as_tensor(self.window_id, dtype=torch.int32, device=device)
            if self.window_id is not None
            else None
        )
        return NodeFeaturesBundle(
            x=x_t,
            y=y_t,
            transaction_dt=dt_t,
            transaction_id=tid_t,
            window_id=wid_t,
            feature_names=self.feature_names,
            split=self.split,
        )


def find_repo_root() -> Path:
    """Find repository root by looking for .git or specific markers."""
    p = Path(__file__).resolve().parent
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists() or (parent / "fedbank_schema_v1.json").exists():
            return parent
    return Path.cwd()


def load_schema(schema_path: Optional[Path] = None) -> List[str]:
    """Load ordered feature column names from frozen schema."""
    if schema_path is None:
        schema_path = find_repo_root() / "fedbank_schema_v1.json"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    return [col["name"] for col in schema["columns"]]


def prepare_node_features(
    features_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    split: str,
    schema_cols: Optional[List[str]] = None,
    track: str = "sel",
    to_torch_tensor: bool = False,
) -> NodeFeaturesBundle:
    """
    Federation-ready pure function: prepares node features from in-memory DataFrames.
    
    Args:
        features_df: Tabular feature DataFrame.
        meta_df: Metadata DataFrame containing TransactionID, TransactionDT, isFraud, etc.
        split: One of 'train', 'val', 'test', 'monitor'.
        schema_cols: Expected ordered column names (for Track A).
        track: 'sel' (Track A, 48 features) or 'pca' (Track B, 27 features).
        to_torch_tensor: If True, convert to PyTorch tensors.
    """
    if len(features_df) != len(meta_df):
        raise ValueError(
            f"Row count mismatch for split '{split}': "
            f"features={len(features_df)}, meta={len(meta_df)}"
        )
    
    # 1. Column validation and ordering
    if track == "sel":
        if schema_cols is None:
            schema_cols = load_schema()
        
        feature_cols_set = set(features_df.columns)
        schema_cols_set = set(schema_cols)
        
        missing = schema_cols_set - feature_cols_set
        extra = feature_cols_set - schema_cols_set
        
        if missing or extra:
            raise ValueError(
                f"Features for split '{split}' do not match schema.\n"
                f"Missing: {missing}\nExtra: {extra}"
            )
        
        # Reorder to strict schema order
        features_df = features_df[schema_cols]
        final_feature_names = list(schema_cols)
    elif track == "pca":
        final_feature_names = list(features_df.columns)
        if len(final_feature_names) != 27:
            raise ValueError(f"Expected 27 PCA features, got {len(final_feature_names)}")
    else:
        raise ValueError(f"Unknown track: {track}. Must be 'sel' or 'pca'.")

    # 2. Assert forbidden columns are never in features
    forbidden_present = FORBIDDEN_COLUMNS.intersection(set(features_df.columns))
    if forbidden_present:
        raise ValueError(
            f"Leakage check failed: Forbidden columns {forbidden_present} found in features!"
        )

    # 3. Assert row alignment: if TransactionID is in both, verify exact match
    if "TransactionID" in features_df.columns and "TransactionID" in meta_df.columns:
        if not (features_df["TransactionID"].values == meta_df["TransactionID"].values).all():
            raise ValueError(f"TransactionID alignment mismatch in split '{split}'!")
    elif features_df.index.name == "TransactionID" and "TransactionID" in meta_df.columns:
        if not (features_df.index.values == meta_df["TransactionID"].values).all():
            raise ValueError(f"TransactionID index alignment mismatch in split '{split}'!")

    # 4. Cast to float32 numpy array and verify finite
    x = features_df.to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        n_nans = np.isnan(x).sum()
        n_infs = np.isinf(x).sum()
        raise ValueError(
            f"Features contain non-finite values in split '{split}': NaNs={n_nans}, Infs={n_infs}"
        )

    # 5. Extract meta attributes
    if "isFraud" not in meta_df.columns:
        raise ValueError(f"'isFraud' column missing from meta DataFrame for split '{split}'")
    y = meta_df["isFraud"].to_numpy(dtype=np.int8)

    if "TransactionDT" not in meta_df.columns:
        raise ValueError(f"'TransactionDT' column missing from meta DataFrame for split '{split}'")
    transaction_dt = meta_df["TransactionDT"].to_numpy(dtype=np.int64)

    if "TransactionID" not in meta_df.columns:
        raise ValueError(f"'TransactionID' column missing from meta DataFrame for split '{split}'")
    transaction_id = meta_df["TransactionID"].to_numpy(dtype=np.int64)

    window_id = None
    if "window_id" in meta_df.columns:
        window_id = meta_df["window_id"].to_numpy(dtype=np.int32)

    bundle = NodeFeaturesBundle(
        x=x,
        y=y,
        transaction_dt=transaction_dt,
        transaction_id=transaction_id,
        window_id=window_id,
        feature_names=final_feature_names,
        split=split,
    )

    if to_torch_tensor and HAS_TORCH:
        return bundle.to_torch()
    return bundle


def load_split_node_features(
    split: str,
    repo_root: Optional[Path] = None,
    track: str = "sel",
    schema_cols: Optional[List[str]] = None,
    to_torch_tensor: bool = False,
) -> NodeFeaturesBundle:
    """Load parquet files from data/processed/ and run prepare_node_features."""
    if repo_root is None:
        repo_root = find_repo_root()
    
    processed_dir = repo_root / "data" / "processed"
    feat_path = processed_dir / f"features_{track}_{split}.parquet"
    meta_path = processed_dir / f"meta_{split}.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    # Read parquet tables into pandas
    features_df = pq.read_table(feat_path).to_pandas()
    meta_df = pq.read_table(meta_path).to_pandas()

    return prepare_node_features(
        features_df=features_df,
        meta_df=meta_df,
        split=split,
        schema_cols=schema_cols,
        track=track,
        to_torch_tensor=to_torch_tensor,
    )


def run_stage_g1_assertions(track: str = "sel") -> Dict[str, NodeFeaturesBundle]:
    """
    Run full Stage G1 validations across all splits.
    
    Assertions enforced:
    1. Shape: (N, 48) for 'sel' (or 27 for 'pca').
    2. Split sizes: exactly 273,000 / 58,500 / 58,500 / 200,540.
    3. Fraud rates close to paper:
       train ~3.33%, val ~3.62%, test ~3.70%, monitor ~3.64%.
    4. Forbidden columns (isFraud, TransactionDT, TransactionID, card_key_id, split, window_id)
       are never in x.
    5. No NaN or inf values in x.
    6. Dtype is float32 for x and int8 for y.
    """
    repo_root = find_repo_root()
    schema_cols = load_schema(repo_root / "fedbank_schema_v1.json") if track == "sel" else None
    expected_dim = 48 if track == "sel" else 27

    print("=" * 78)
    print(f"STAGE G1: Running Node Features Validation (Track: {track.upper()})")
    print("=" * 78)

    results = {}
    table_rows = []

    for split in ["train", "val", "test", "monitor"]:
        bundle = load_split_node_features(
            split=split,
            repo_root=repo_root,
            track=track,
            schema_cols=schema_cols,
            to_torch_tensor=False,
        )
        results[split] = bundle

        n_rows, n_cols = bundle.x.shape
        fraud_count = int(bundle.y.sum())
        fraud_rate = float(bundle.y.mean())
        expected_size = EXPECTED_SIZES[split]
        expected_fr = EXPECTED_FRAUD_RATES[split]

        # Assertions
        assert n_rows == expected_size, (
            f"Split {split}: expected {expected_size} rows, got {n_rows}"
        )
        assert n_cols == expected_dim, (
            f"Split {split}: expected {expected_dim} features, got {n_cols}"
        )
        assert bundle.x.dtype == np.float32, (
            f"Split {split}: expected float32 dtype, got {bundle.x.dtype}"
        )
        assert bundle.y.dtype == np.int8, (
            f"Split {split}: expected int8 label dtype, got {bundle.y.dtype}"
        )
        assert np.isfinite(bundle.x).all(), (
            f"Split {split}: non-finite values detected in x"
        )
        
        # Check forbidden columns
        feature_set = set(bundle.feature_names)
        leakage = FORBIDDEN_COLUMNS.intersection(feature_set)
        assert len(leakage) == 0, f"Split {split}: Forbidden leakage in features: {leakage}"

        # Fraud rate sanity (within 0.15% margin of paper)
        assert abs(fraud_rate - expected_fr) < 0.0015, (
            f"Split {split}: fraud rate {fraud_rate:.4%} deviates from expected {expected_fr:.4%}"
        )

        table_rows.append({
            "Split": split,
            "Rows (N)": f"{n_rows:,}",
            "Features (D)": n_cols,
            "Fraud Count": f"{fraud_count:,}",
            "Fraud Rate": f"{fraud_rate * 100:.2f}%",
            "Paper Target": f"{expected_fr * 100:.2f}%",
            "Finite (no NaN/inf)": "PASS",
            "No Leakage": "PASS",
            "Status": "PASS",
        })

    # Print formatted table
    df_report = pd.DataFrame(table_rows)
    print("\n" + df_report.to_string(index=False) + "\n")
    print("=" * 78)
    print("STAGE G1 ASSERTIONS ALL PASSED SUCCESSFULLY!")
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_stage_g1_assertions(track="sel")
