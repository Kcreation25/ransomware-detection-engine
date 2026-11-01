"""
ensemble.py
===========
Combines a supervised classifier probability with an Isolation Forest
anomaly score into a calibrated hybrid risk score.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from config import ISO_FOREST_PARAMS, THRESHOLD_SEARCH_STEPS, WEIGHT_SEARCH_STEPS
from utils import get_logger, min_max_scale

log = get_logger(__name__)

@dataclass
class HybridScoreStats:
    sup_min: float
    sup_max: float
    unsup_min: float
    unsup_max: float
    weight_supervised: float
    threshold: float

class HybridDetector:
    """Fitted supervised model + Isolation Forest with calibrated scoring."""
    def __init__(self, supervised_model, iso_forest_params: dict | None = None):
        self.supervised_model = supervised_model
        self.iso_forest = IsolationForest(**(iso_forest_params or ISO_FOREST_PARAMS))
        self.stats: HybridScoreStats | None = None

    def fit_isolation_forest(self, X_unsup_scaled: np.ndarray) -> "HybridDetector":
        self.iso_forest.fit(X_unsup_scaled)
        return self

    def _raw_scores(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sup_prob = self.supervised_model.predict_proba(X_scaled)[:, 1]
        unsup_raw = -self.iso_forest.decision_function(X_scaled)
        return sup_prob, unsup_raw

    def calibrate(self, X_val_scaled: np.ndarray, y_val: np.ndarray) -> HybridScoreStats:
        sup_prob, unsup_raw = self._raw_scores(X_val_scaled)
        sup_min, sup_max = float(sup_prob.min()), float(sup_prob.max())
        unsup_min, unsup_max = float(unsup_raw.min()), float(unsup_raw.max())
        sup_scaled = min_max_scale(sup_prob, sup_min, sup_max)
        unsup_scaled = min_max_scale(unsup_raw, unsup_min, unsup_max)
        best = {"f1": -1.0, "weight": 0.65, "threshold": 0.5}
        for w in np.linspace(0.0, 1.0, WEIGHT_SEARCH_STEPS):
            hybrid = w * sup_scaled + (1 - w) * unsup_scaled
            for t in np.linspace(0.01, 0.99, THRESHOLD_SEARCH_STEPS):
                f1 = f1_score(y_val, (hybrid > t).astype(int), zero_division=0)
                if f1 > best["f1"]:
                    best = {"f1": f1, "weight": float(w), "threshold": float(t)}
        log.info("Calibrated hybrid ensemble: weight=%.2f threshold=%.2f val_F1=%.4f", best["weight"], best["threshold"], best["f1"])
        self.stats = HybridScoreStats(sup_min, sup_max, unsup_min, unsup_max, best["weight"], best["threshold"])
        return self.stats

    def score(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.stats is None:
            raise RuntimeError("Call calibrate() before score() — no reference normalization statistics available yet.")
        sup_prob, unsup_raw = self._raw_scores(X_scaled)
        sup_scaled = min_max_scale(sup_prob, self.stats.sup_min, self.stats.sup_max)
        unsup_scaled = min_max_scale(unsup_raw, self.stats.unsup_min, self.stats.unsup_max)
        w = self.stats.weight_supervised
        return w * sup_scaled + (1 - w) * unsup_scaled

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        return (self.score(X_scaled) > self.stats.threshold).astype(int)
