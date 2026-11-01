"""
config.py
=========
Single source of truth for paths, hyperparameter search spaces, and
run-time constants. Nothing in this project should hard-code a path
or a magic number outside of this file — that's what made the
original notebook hard to reuse.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"

SUPERVISED_DATA_PATH = os.environ.get("SUPERVISED_DATA_PATH", str(DATASET_DIR / "BitcoinHeistData.csv.zip"))
UNLABELED_DATA_PATH = os.environ.get("UNLABELED_DATA_PATH", str(DATASET_DIR / "no_label.csv"))
MODEL_ARTIFACT_PATH = MODELS_DIR / "hybrid_model.pkl"
SCALER_ARTIFACT_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_ARTIFACT_PATH = MODELS_DIR / "encoder.pkl"
METADATA_ARTIFACT_PATH = MODELS_DIR / "metadata.json"

for _dir in (DATASET_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "label"
ADDRESS_COLUMN = "address"
BENIGN_LABEL_VALUE = "white"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

@dataclass(frozen=True)
class ImbalanceConfig:
    strategies: List[str] = field(default_factory=lambda: ["class_weight", "smote", "borderline_smote", "adasyn", "smoteenn", "smotetomek"])

@dataclass(frozen=True)
class ModelConfig:
    candidates: List[str] = field(default_factory=lambda: ["hist_gb", "random_forest", "extra_trees", "balanced_random_forest", "xgboost", "lightgbm", "catboost"])
    cv_folds: int = 5
    n_search_iter: int = 25
    n_optuna_trials: int = 30
    scoring_metric: str = "f1"

IMBALANCE_CONFIG = ImbalanceConfig()
MODEL_CONFIG = ModelConfig()
ISO_FOREST_PARAMS: Dict = {"n_estimators": 200, "contamination": "auto", "random_state": RANDOM_STATE, "n_jobs": -1}
INITIAL_SUPERVISED_WEIGHT = 0.65
WEIGHT_SEARCH_STEPS = 21
THRESHOLD_SEARCH_STEPS = 99
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
