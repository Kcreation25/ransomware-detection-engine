"""
train.py
========
End-to-end training entry point.

Pipeline: load labeled and unlabeled datasets; create stratified train,
validation and held-out test splits; fit preprocessing on training data;
benchmark model/resampling combinations; calibrate the hybrid ensemble;
evaluate once on the held-out test set; save model artifacts and metadata.
"""
from __future__ import annotations
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.pipeline import Pipeline
import config
from ensemble import HybridDetector
from evaluation import build_comparison_table, compute_metrics, confusion_matrix_frame
from feature_engineering import add_engineered_features
from models import SEARCH_SPACES, get_available_classifiers, get_sampler
from preprocessing import CategoricalEncoder, fit_scaler, load_csv, split_features_target, transform_with_scaler
from utils import get_logger, save_json, timer
log = get_logger(__name__)

def _pipeline_step_name(model_name: str) -> str:
    return {"hist_gb":"hgb", "random_forest":"rf", "extra_trees":"et", "balanced_random_forest":"brf", "xgboost":"xgb", "lightgbm":"lgbm", "catboost":"cat"}[model_name]

def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    classifiers = get_available_classifiers(config.MODEL_CONFIG.candidates)
    if not classifiers:
        raise RuntimeError("No classifiers available — install scikit-learn at minimum.")
    results, times, fitted = {}, {}, {}
    for strategy in config.IMBALANCE_CONFIG.strategies:
        for model_name, builder in classifiers.items():
            key = f"{model_name}__{strategy}"
            step = _pipeline_step_name(model_name)
            try:
                sampler = get_sampler(strategy)
                use_class_weight = strategy == "class_weight"
                steps = []
                if sampler is not None:
                    steps.append(("sampler", sampler))
                model_kwargs = {"class_weight": "balanced"} if use_class_weight else {}
                try:
                    estimator = builder(**model_kwargs)
                except TypeError:
                    estimator = builder()
                steps.append((step, estimator))
                pipe = Pipeline(steps)
                search_space = dict(SEARCH_SPACES.get(model_name, {}))
                with timer(log, f"tuning {key}") as t:
                    if search_space:
                        search = RandomizedSearchCV(pipe, param_distributions=search_space, n_iter=min(config.MODEL_CONFIG.n_search_iter, _space_size(search_space)), scoring=config.MODEL_CONFIG.scoring_metric, cv=config.MODEL_CONFIG.cv_folds, random_state=config.RANDOM_STATE, n_jobs=-1)
                        search.fit(X_train, y_train)
                        best_pipe = search.best_estimator_
                    else:
                        pipe.fit(X_train, y_train)
                        best_pipe = pipe
                y_val_pred = best_pipe.predict(X_val)
                y_val_prob = best_pipe.predict_proba(X_val)[:, 1]
                results[key] = compute_metrics(y_val, y_val_pred, y_val_prob)
                times[key] = t["seconds"]
                fitted[key] = best_pipe
                log.info("%s -> val F1=%.4f", key, results[key]["f1"])
            except Exception as exc:
                log.warning("Skipping %s: %s", key, exc)
    if not results:
        raise RuntimeError("Every model/resampling combination failed. Check dataset schema.")
    comparison = build_comparison_table(results, times)
    best_key = comparison.iloc[0]["model"]
    log.info("Best combination on validation F1: %s", best_key)
    return {"comparison_table": comparison, "best_key": best_key, "best_pipeline": fitted[best_key]}

def _space_size(space: Dict) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size

def main() -> None:
    df_supervised = load_csv(config.SUPERVISED_DATA_PATH)
    df_unlabeled = load_csv(config.UNLABELED_DATA_PATH)
    prepared = split_features_target(df_supervised, is_supervised=True)
    X, y = prepared.X, prepared.y
    unlabeled_prepared = split_features_target(df_unlabeled, is_supervised=False)
    X_unlabeled = unlabeled_prepared.X

    # Prevent transductive leakage: remove addresses already present in the
    # labeled dataset before fitting the unsupervised anomaly detector.
    if prepared.address is not None and unlabeled_prepared.address is not None:
        known_addresses = set(prepared.address.astype(str))
        unseen_mask = ~unlabeled_prepared.address.astype(str).isin(known_addresses)
        X_unlabeled = X_unlabeled.loc[unseen_mask].copy()
        log.info("Unlabeled rows with unseen addresses: %d", len(X_unlabeled))
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    val_fraction = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_fraction, stratify=y_train_full, random_state=config.RANDOM_STATE)
    log.info("Split sizes -> train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))
    X_train = add_engineered_features(X_train)
    X_val = add_engineered_features(X_val)
    X_test = add_engineered_features(X_test)
    if X_unlabeled.empty:
        # With no independent unlabeled addresses available, train the
        # Isolation Forest on the training split only (never val/test rows).
        log.warning(
            "No independent unlabeled addresses remain after leakage check; "
            "fitting Isolation Forest on X_train only."
        )
        X_unlabeled = X_train.copy()
    X_unlabeled = add_engineered_features(X_unlabeled)
    encoder = CategoricalEncoder().fit(X_train)
    X_train_enc = encoder.transform(X_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)
    X_unlabeled_enc = encoder.transform(X_unlabeled)
    scaler = fit_scaler(X_train_enc)
    def scaled_df(X_enc: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(transform_with_scaler(scaler, X_enc), columns=X_enc.columns, index=X_enc.index)
    X_train_scaled, X_val_scaled = scaled_df(X_train_enc), scaled_df(X_val_enc)
    X_test_scaled, X_unlabeled_scaled = scaled_df(X_test_enc), scaled_df(X_unlabeled_enc)
    bench = benchmark_models(X_train_scaled, y_train, X_val_scaled, y_val)
    print("\n=== Model x Imbalance-Strategy Comparison (sorted by F1) ===")
    print(bench["comparison_table"].to_string(index=False))
    bench["comparison_table"].to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    detector = HybridDetector(supervised_model=bench["best_pipeline"])
    detector.fit_isolation_forest(X_unlabeled_scaled)
    stats = detector.calibrate(X_val_scaled, y_val.to_numpy())
    y_test_pred = detector.predict(X_test_scaled)
    y_test_score = detector.score(X_test_scaled)
    test_metrics = compute_metrics(y_test.to_numpy(), y_test_pred, y_test_score)
    print("\n=== Final Hybrid Model — Held-out Test Set ===")
    for key, value in test_metrics.items():
        print(f"{key:>18}: {value:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix_frame(y_test.to_numpy(), y_test_pred))
    joblib.dump(detector, config.MODEL_ARTIFACT_PATH)
    joblib.dump(scaler, config.SCALER_ARTIFACT_PATH)
    joblib.dump(encoder, config.ENCODER_ARTIFACT_PATH)
    save_json(config.METADATA_ARTIFACT_PATH, {"best_pipeline_key": bench["best_key"], "hybrid_weight_supervised": stats.weight_supervised, "hybrid_threshold": stats.threshold, "test_metrics": test_metrics, "feature_columns": list(X_train_enc.columns)})
    log.info("Saved model -> %s", config.MODEL_ARTIFACT_PATH)
    log.info("Saved scaler -> %s", config.SCALER_ARTIFACT_PATH)
    log.info("Saved encoder -> %s", config.ENCODER_ARTIFACT_PATH)

if __name__ == "__main__":
    main()
