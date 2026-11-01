# 🛡️ Hybrid Ransomware Detection Engine

A machine-learning research project for identifying Bitcoin wallet addresses associated with ransomware activity. The system combines supervised classification with unsupervised anomaly detection to analyze wallet transaction patterns and generate risk predictions.

> **Project scope:** This is an academic research prototype for blockchain-risk analysis. Predictions are indicators for further investigation, not proof of criminal activity or wallet ownership.

---

## 🚀 Features

- 🔍 **AI-based ransomware detection** — classifies Bitcoin wallet records as benign or potentially ransomware-associated.
- 🤖 **Hybrid machine-learning architecture** — combines supervised classification with an Isolation Forest anomaly detector.
- ⚖️ **Class-imbalance handling** — supports class weighting and resampling strategies such as SMOTE, ADASYN and related methods.
- 📊 **Model evaluation** — reports accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, balanced accuracy and MCC.
- 🧠 **Explainable AI support** — optional SHAP and LIME integrations for model interpretation.
- 📁 **CSV prediction workflow** — scores wallet records supplied in CSV format.
- 📈 **Interactive Streamlit dashboard** — presents prediction and evaluation insights.
- 💾 **Local execution** — run training and predictions on your own machine after installing dependencies and preparing the data.

## 📸 Screenshots

Add real screenshots from your running application in the following locations:

| Screen | Suggested file |
|---|---|
| Dashboard | `images/dashboard.png` |
| Model evaluation | `images/evaluation.png` |
| Prediction results | `images/predictions.png` |

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Imbalance handling | Imbalanced-learn |
| Anomaly detection | Isolation Forest |
| Optional model families | XGBoost, LightGBM, CatBoost |
| Hyperparameter tuning | Optuna |
| Explainable AI | SHAP, LIME |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Model persistence | Joblib |

## 📁 Project Structure

```text
ransomware-detection-engine/
├── app.py                      # Command-line entry point
├── train.py                    # Main model training pipeline
├── predict.py                  # CSV prediction entry point
├── config.py                   # Dataset paths and model configuration
├── preprocessing.py            # Data loading, encoding and scaling
├── feature_engineering.py       # Wallet-behavior features
├── models.py                   # Model and sampler definitions
├── ensemble.py                 # Supervised + anomaly ensemble
├── evaluation.py                # Metrics and comparison reports
├── explainability.py            # SHAP/LIME explanation support
├── utils.py                     # Shared utilities
├── streamlit_app/
│   └── main.py                  # Interactive web dashboard
├── dataset/                     # Input datasets (not committed)
├── models/                      # Generated model artifacts
├── reports/                     # Evaluation and prediction outputs
├── images/                      # Screenshots
├── notebooks/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Installation & Setup (Windows)

### 1. Install prerequisites

Install **Python 3.10 or newer** and Git. During Python installation, enable **Add Python to PATH**.

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads
- VS Code (recommended): https://code.visualstudio.com/

### 2. Clone the repository

Open PowerShell or the VS Code terminal and run:

```powershell
git clone https://github.com/Kcreation25/ransomware-detection-engine.git
cd ransomware-detection-engine
```

### 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Prepare the datasets

The pipeline is configured for the [UCI BitcoinHeist Ransomware Address Dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset).

Place the files at the paths configured in `config.py`:

```text
dataset/
├── BitcoinHeistData.csv.zip
└── no_label.csv
```

The labeled dataset is used for supervised training. The separate unlabeled data is intended for the Isolation Forest component. Use independent wallet records where possible; overlapping addresses can undermine the validity of the anomaly-training/evaluation setup. Do not upload datasets to a public repository unless their license and redistribution terms allow it.

### 6. Train the model

From the repository root:

```powershell
python train.py
```

Training may take considerable time and memory depending on dataset size, hardware and configured model search. Successful training writes model artifacts under `models/` and evaluation reports under `reports/`.

### 7. Run predictions

After training has created the required model artifacts:

```powershell
python predict.py path\to\wallets.csv
```

Replace the example path with a CSV containing the wallet features expected by the project.

### 8. Launch the dashboard

```powershell
streamlit run streamlit_app/main.py
```

Open the local address printed in the terminal, usually:

```text
http://localhost:8501
```

## 🧠 Main Code and Outputs

- **Main training code:** `train.py`
- **Prediction code:** `predict.py`
- **Dashboard code:** `streamlit_app/main.py`
- **Model definitions:** `models.py`
- **Hybrid detection logic:** `ensemble.py`
- **Data preparation:** `preprocessing.py` and `feature_engineering.py`
- **Configuration:** `config.py`

Expected generated artifacts after a successful training run:

```text
models/
├── hybrid_model.pkl
├── scaler.pkl
├── encoder.pkl
└── metadata.json

reports/
└── model_comparison.csv
```

Prediction runs may also create prediction result files under `reports/`, depending on the prediction command and configuration.

Generated model binaries, datasets and run-specific outputs may be excluded from Git by `.gitignore`. Recreate them locally by running the pipeline.

## 📊 Preliminary Evaluation

A preliminary prototype evaluation was performed on a 200,000-row sample. The recorded sample metrics were:

| Metric | Preview result |
|---|---:|
| Accuracy | 98.02% |
| Precision | 33.67% |
| Recall | 40.49% |
| F1-score | 36.77% |
| ROC-AUC | 86.95% |
| PR-AUC | 32.62% |

**Important:** These are preliminary sample results, not full-dataset benchmark results. The dataset is imbalanced, so accuracy should not be interpreted on its own. Consider precision, recall, F1-score, PR-AUC, MCC and the confusion matrix together. Re-run the full evaluation before presenting final performance claims.

See [the prototype run summary](reports/run_summary.md) and [model comparison report](reports/model_comparison.csv) for additional context.

## ⚠️ Disclaimer

This project is developed for academic research and educational purposes. A model prediction is a risk signal for further review, not definitive proof of ransomware involvement, criminal activity or wallet ownership. Do not use predictions as the sole basis for accusations, sanctions or other consequential decisions.

## 📄 License

This project is distributed under the MIT License. See [LICENSE](LICENSE).

## 👨‍💻 Author

**Komal Singh**  
Computer Science & Engineering Student  
AI & Machine Learning Enthusiast

Interested in Machine Learning, Artificial Intelligence and Data Science.

---

⭐ If you find this project useful, consider starring the repository.
