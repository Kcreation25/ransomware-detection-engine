# Hybrid Ransomware Detection Engine

A machine-learning research project for identifying Bitcoin wallets associated with ransomware activity. The system combines a supervised classifier with an Isolation Forest anomaly detector and calibrates their scores on a validation split.

> **Scope:** This project is an academic/research prototype for blockchain-risk analysis. It is not a production security product, does not monitor or block ransomware on endpoints, and its predictions should not be treated as proof of criminal activity.

## Key features

- Supervised model benchmarking across tree-based classifiers.
- Imbalance-handling strategies including class weighting and SMOTE-family samplers.
- Unsupervised anomaly detection with Isolation Forest.
- Validation-based calibration of ensemble weight and decision threshold.
- Evaluation with precision, recall, F1, ROC-AUC, PR-AUC, MCC and balanced accuracy.
- CSV prediction workflow and multi-page Streamlit dashboard.
- Optional SHAP and LIME explainability dependencies.

## Architecture

```text
Labeled BitcoinHeist data ──> preprocessing ──> feature engineering
                                      │
                                      ├──> supervised model benchmarking
                                      │
Unlabeled wallet data ────────────────┴──> Isolation Forest
                                                   │
                           validation calibration ─┘
                                      │
                              hybrid risk score
                                      │
                        held-out evaluation / prediction
                                      │
                         CLI + Streamlit dashboard
```

## Repository layout

```text
.
├── app.py                  # CLI entry point
├── train.py                # Training, tuning and evaluation pipeline
├── predict.py              # Prediction interface
├── config.py               # Paths and model configuration
├── preprocessing.py        # Dataset loading, encoding and scaling
├── feature_engineering.py  # Wallet-behavior features
├── models.py               # Classifier and sampler registries
├── ensemble.py             # Supervised + anomaly-score ensemble
├── evaluation.py           # Classification metrics and comparison tables
├── explainability.py       # Optional SHAP/LIME explanations
├── utils.py                # Logging, timing and JSON helpers
├── streamlit_app/
│   └── main.py              # Interactive dashboard
├── dataset/                # Place datasets here (not committed)
├── models/                 # Generated model artifacts (not committed)
├── reports/                # Generated reports (not committed)
├── images/
├── notebooks/
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Dataset

The pipeline is configured for the [UCI BitcoinHeist Ransomware Address Dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset).

Place the required files in `dataset/`:

- `BitcoinHeistData.csv.zip` — labeled dataset.
- `no_label.csv` — unlabeled wallet records used to fit the anomaly detector.

The dataset is not included in this repository. Follow the dataset's terms and cite the source when using it. You can override the default paths with `SUPERVISED_DATA_PATH` and `UNLABELED_DATA_PATH` environment variables.

## Setup

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Kcreation25/ransomware-detection-engine.git
cd ransomware-detection-engine

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional libraries (XGBoost, LightGBM, CatBoost, Optuna, SHAP and LIME) are listed in `requirements.txt`. Some are used only for optional model families or explainability.

## Run

After placing both datasets in `dataset/`:

```bash
# Train models, calibrate the ensemble and save artifacts
python train.py

# Or use the CLI dispatcher
python app.py train

# Score a CSV containing wallet features
python predict.py path/to/wallets.csv

# Launch the dashboard
streamlit run streamlit_app/main.py
# Or:
python app.py dashboard
```

Training writes model artifacts under `models/` and evaluation outputs under `reports/`. These generated files are excluded from Git by `.gitignore`; reproduce them locally by running the training pipeline.

## Evaluation notes

The dataset is imbalanced, so accuracy alone can be misleading. Review precision, recall, F1, PR-AUC, ROC-AUC, MCC and the confusion matrix together. Reported metrics are data- and configuration-dependent; this repository does not claim benchmark results without a completed run.

## Responsible use

- Use only datasets and wallet information you are authorized to analyze.
- A model score is an indicator for further review, not a definitive attribution.
- Do not use predictions as the sole basis for accusations, sanctions or other consequential decisions.
- Keep datasets, credentials, environment files and generated sensitive artifacts out of version control.

## License

Released under the MIT License. See [LICENSE](LICENSE).
