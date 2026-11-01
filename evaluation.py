"""
evaluation.py
=============
Evaluation utilities for imbalanced ransomware detection. Reports
precision, recall, F1, MCC, balanced accuracy and ranking metrics.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from utils import get_logger
log = get_logger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0), "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0), "mcc": matthews_corrcoef(y_true, y_pred)}
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics

def confusion_matrix_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual: Non-malicious", "Actual: Malicious"], columns=["Pred: Non-malicious", "Pred: Malicious"])

def build_comparison_table(results: Dict[str, Dict[str, float]], training_times: Dict[str, float]) -> pd.DataFrame:
    rows = [{"model": name, **metrics, "train_seconds": training_times.get(name, float("nan"))} for name, metrics in results.items()]
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("f1", "pr_auc") if c in df.columns]
    return df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
