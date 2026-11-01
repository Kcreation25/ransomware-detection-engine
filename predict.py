"""
predict.py
==========
Loads saved model/scaler/encoder artifacts and scores new wallet data.
Usage: python predict.py path/to/wallets.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import pandas as pd
import config
from feature_engineering import add_engineered_features
from preprocessing import split_features_target, transform_with_scaler
from utils import get_logger, load_json
log = get_logger(__name__)

class RansomwareDetector:
    """Loads trained artifacts and returns predictions with risk scores."""
    def __init__(self) -> None:
        missing = [p for p in (config.MODEL_ARTIFACT_PATH, config.SCALER_ARTIFACT_PATH, config.ENCODER_ARTIFACT_PATH, config.METADATA_ARTIFACT_PATH) if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing trained artifacts: {missing}. Run `python train.py` first.")
        self.detector = joblib.load(config.MODEL_ARTIFACT_PATH)
        self.scaler = joblib.load(config.SCALER_ARTIFACT_PATH)
        self.encoder = joblib.load(config.ENCODER_ARTIFACT_PATH)
        self.metadata = load_json(config.METADATA_ARTIFACT_PATH)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = split_features_target(df, is_supervised=False)
        X = add_engineered_features(prepared.X)
        X_enc = self.encoder.transform(X)
        for col in self.metadata["feature_columns"]:
            if col not in X_enc.columns:
                X_enc[col] = 0
        X_enc = X_enc[self.metadata["feature_columns"]]
        X_scaled = pd.DataFrame(transform_with_scaler(self.scaler, X_enc), columns=X_enc.columns, index=X_enc.index)
        risk_score = self.detector.score(X_scaled)
        prediction = self.detector.predict(X_scaled)
        out = pd.DataFrame({"risk_score": risk_score, "prediction": prediction, "label": pd.Series(prediction).map({0: "Non-malicious", 1: "Malicious"})})
        if prepared.address is not None:
            out.insert(0, "address", prepared.address.values)
        return out

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/wallets.csv")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    detector = RansomwareDetector()
    results = detector.predict(df)
    out_path = config.REPORTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    log.info("Wrote %d predictions -> %s", len(results), out_path)
    print(results.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
