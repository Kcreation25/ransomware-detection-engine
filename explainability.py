"""
explainability.py
=================
Optional SHAP and LIME explanations for the fitted supervised estimator.
These dependencies are loaded only when explanation functions are called.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from utils import get_logger
log = get_logger(__name__)

def _require_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Run `pip install shap` to enable explainability features.") from exc
    return shap

def build_explainer(model, background_data: pd.DataFrame):
    """Build a TreeExplainer for supported tree-based estimators."""
    shap = _require_shap()
    return shap.TreeExplainer(model, background_data)

def global_feature_importance(explainer, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Return mean absolute SHAP value per feature."""
    shap_values = explainer.shap_values(X_sample)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

def explain_instance(explainer, X_row: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature SHAP contributions for a single wallet."""
    shap_values = explainer.shap_values(X_row)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    return pd.DataFrame({"feature": X_row.columns, "shap_value": values[0]}).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)

def lime_explain_instance(model_predict_fn, X_train: pd.DataFrame, X_row: pd.DataFrame, class_names=("non_malicious", "malicious")) -> Optional[list]:
    """Optional LIME local explanation; returns None if LIME is unavailable."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        log.warning("lime is not installed; skipping LIME explanation")
        return None
    explainer = LimeTabularExplainer(training_data=X_train.values, feature_names=list(X_train.columns), class_names=list(class_names), mode="classification", random_state=42)
    exp = explainer.explain_instance(X_row.values[0], model_predict_fn, num_features=10)
    return exp.as_list()
