"""
preprocessing.py
================
Loading, cleaning, encoding and scaling. Encoders and scalers are fitted
only on the training split and reused for validation/test/inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from config import ADDRESS_COLUMN, BENIGN_LABEL_VALUE, TARGET_COLUMN
from utils import get_logger
log = get_logger(__name__)

@dataclass
class PreparedData:
    X: pd.DataFrame
    y: Optional[pd.Series]
    address: Optional[pd.Series]

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at {path}. Place your CSV(s) in `dataset/` or set SUPERVISED_DATA_PATH / UNLABELED_DATA_PATH.")
    df = pd.read_csv(path)
    log.info("Loaded %s -> shape=%s", path.name, df.shape)
    return df

def _binarize_label(series: pd.Series) -> pd.Series:
    return (series != BENIGN_LABEL_VALUE).astype(int)

class CategoricalEncoder:
    """Per-column label encoders with safe handling for unseen categories."""
    def __init__(self) -> None:
        self.encoders: dict[str, LabelEncoder] = {}
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame) -> "CategoricalEncoder":
        self.columns = list(X.select_dtypes(include="object").columns)
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            le = self.encoders[col]
            known = set(le.classes_)
            fallback = le.classes_[0]
            X[col] = X[col].astype(str).apply(lambda v: v if v in known else fallback)
            X[col] = le.transform(X[col])
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

def split_features_target(df: pd.DataFrame, is_supervised: bool) -> PreparedData:
    df = df.copy()
    address = df[ADDRESS_COLUMN] if ADDRESS_COLUMN in df.columns else None
    if is_supervised:
        y = _binarize_label(df[TARGET_COLUMN])
        X = df.drop(columns=[c for c in (TARGET_COLUMN, ADDRESS_COLUMN) if c in df.columns])
    else:
        y = None
        X = df.drop(columns=[c for c in (ADDRESS_COLUMN,) if c in df.columns])
    return PreparedData(X=X, y=y, address=address)

def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler

def transform_with_scaler(scaler: RobustScaler, X: pd.DataFrame) -> np.ndarray:
    return scaler.transform(X)
