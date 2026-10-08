"""
config.py — Pipeline configuration dataclass.
All defaults from FEDBANK_preprocessing_spec.md Section 10.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ seeds
    SEED: int = 42

    # ------------------------------------------------------------------ paths
    ROOT: Path = Path("/home/abhin/mainproject/FEDBANK")
    RAW_DIR: Path = Path("/home/abhin/mainproject/FEDBANK/data")   # CSVs live here
    INTERIM_DIR: Path = Path("/home/abhin/mainproject/FEDBANK/data/interim")
    PROCESSED_DIR: Path = Path("/home/abhin/mainproject/FEDBANK/data/processed")
    ARTIFACTS_DIR: Path = Path("/home/abhin/mainproject/FEDBANK/artifacts")
    REPORTS_DIR: Path = Path("/home/abhin/mainproject/FEDBANK/reports")

    # ------------------------------------------------------------------ splits
    TRAIN_N: int = 273_000
    VAL_N: int = 58_500
    TEST_N: int = 58_500
    # monitor = remaining rows

    # ------------------------------------------------------------------ cleaning
    DROP_MISSING: float = 0.90        # drop if missing rate > this
    NEAR_CONST_SHARE: float = 0.995   # drop if top-value share >= this

    # ------------------------------------------------------------------ missing flags (S5)
    FLAG_MIN: float = 0.05            # flag if missing rate > this

    # ------------------------------------------------------------------ imputation (S7)
    KNN_MAX: float = 0.50             # KNN impute if missing_rate < this
    KNN_K: int = 5
    KNN_REF: int = 50_000
    KNN_BATCH: int = 20_000
    KNN_ANCHOR_MAX: int = 24

    # ------------------------------------------------------------------ winsorize / skew (S6)
    WINSOR: tuple = (0.01, 0.99)
    SKEW_ABS: float = 2.0
    SKEW_METHOD: str = "signed_log1p"   # or "yeo_johnson"

    # ------------------------------------------------------------------ dim reduction (S8–S9)
    CORR_THR: float = 0.95
    K_SEL: int = 48                   # top-K features for Track A
    PCA_VAR: float = 0.95             # cumulative variance threshold for V-PCA
    PCA_MAX_PER_BLOCK: int = 5        # hard cap per V-block
    DRIFT_FILTER: bool = False        # report only by default

    # ------------------------------------------------------------------ feature engineering (S3)
    USE_D1_UID: bool = True
    EXCLUDE_TIME_PROXIES: bool = True  # exclude month/year from model features

    # ------------------------------------------------------------------ VRAM guard
    VRAM_MIN_MB: int = 600            # abort if free VRAM < this at stage start

    # ------------------------------------------------------------------ sample run
    SAMPLE_N: int = 0                 # 0 = full run; set to 20000 for smoke test

    # ------------------------------------------------------------------ protected features (never drop in corr pruning)
    PROTECT: list = field(default_factory=lambda: [
        "TransactionAmt_log",
        "card_velocity",
        "amt_to_avg_card_ratio",
        "transaction_hour",
        "card1_freq",
    ])

    # ------------------------------------------------------------------ columns never dropped by missing-rate rule
    NEVER_DROP: list = field(default_factory=lambda: [
        "TransactionAmt", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "ProductCD",
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
        "D1", "D15",
        "P_emaildomain", "R_emaildomain",
        "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    ])

    def __post_init__(self):
        for attr in ("ROOT", "RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR",
                     "ARTIFACTS_DIR", "REPORTS_DIR"):
            v = getattr(self, attr)
            if not isinstance(v, Path):
                setattr(self, attr, Path(v))
        # ensure dirs exist
        for attr in ("INTERIM_DIR", "PROCESSED_DIR", "ARTIFACTS_DIR", "REPORTS_DIR"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)
