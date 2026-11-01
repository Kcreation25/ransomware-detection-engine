"""
models.py
=========
Registries for classifiers and imbalance-handling strategies. Optional
libraries are imported lazily and skipped when unavailable.
"""
from __future__ import annotations
from typing import Callable, Dict
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from config import RANDOM_STATE
from utils import get_logger
log = get_logger(__name__)

def _build_hist_gb(**kw):
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **kw)

def _build_random_forest(**kw):
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **kw)

def _build_extra_trees(**kw):
    return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, **kw)

def _build_balanced_random_forest(**kw):
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
    except ImportError:
        raise ImportError("imbalanced-learn is required for balanced_random_forest")
    return BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **kw)

def _build_xgboost(**kw):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost is not installed")
    return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, **kw)

def _build_lightgbm(**kw):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("lightgbm is not installed")
    return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **kw)

def _build_catboost(**kw):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("catboost is not installed")
    return CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, **kw)

CLASSIFIER_BUILDERS: Dict[str, Callable] = {
    "hist_gb": _build_hist_gb, "random_forest": _build_random_forest,
    "extra_trees": _build_extra_trees, "balanced_random_forest": _build_balanced_random_forest,
    "xgboost": _build_xgboost, "lightgbm": _build_lightgbm, "catboost": _build_catboost,
}
SEARCH_SPACES: Dict[str, Dict] = {
    "hist_gb": {"hgb__max_iter": [100, 200, 300], "hgb__learning_rate": [0.03, 0.05, 0.1], "hgb__max_depth": [3, 5, 7, None]},
    "random_forest": {"rf__n_estimators": [200, 400, 600], "rf__max_depth": [8, 12, None], "rf__min_samples_leaf": [1, 2, 4]},
    "extra_trees": {"et__n_estimators": [200, 400, 600], "et__max_depth": [8, 12, None]},
    "balanced_random_forest": {"brf__n_estimators": [200, 400], "brf__max_depth": [8, 12, None]},
    "xgboost": {"xgb__n_estimators": [200, 400, 600], "xgb__max_depth": [3, 5, 7], "xgb__learning_rate": [0.03, 0.05, 0.1]},
    "lightgbm": {"lgbm__n_estimators": [200, 400, 600], "lgbm__num_leaves": [15, 31, 63], "lgbm__learning_rate": [0.03, 0.05, 0.1]},
    "catboost": {"cat__iterations": [200, 400, 600], "cat__depth": [4, 6, 8], "cat__learning_rate": [0.03, 0.05, 0.1]},
}

def get_available_classifiers(candidates: list[str]) -> Dict[str, Callable]:
    available = {}
    for name in candidates:
        builder = CLASSIFIER_BUILDERS.get(name)
        if builder is None:
            log.warning("Unknown classifier '%s' — skipping", name)
            continue
        try:
            builder()
            available[name] = builder
        except ImportError as exc:
            log.warning("Skipping '%s': %s", name, exc)
    return available

def get_sampler(strategy: str):
    if strategy == "class_weight":
        return None
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "borderline_smote":
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(random_state=RANDOM_STATE)
    if strategy == "adasyn":
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=RANDOM_STATE)
    if strategy == "smoteenn":
        from imblearn.combine import SMOTEENN
        return SMOTEENN(random_state=RANDOM_STATE)
    if strategy == "smotetomek":
        from imblearn.combine import SMOTETomek
        return SMOTETomek(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")
