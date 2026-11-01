"""
streamlit_app/main.py
======================
Multi-page dashboard for the Hybrid Ransomware Detection System.
Run with: streamlit run streamlit_app/main.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from predict import RansomwareDetector
from utils import load_json

st.set_page_config(page_title="Hybrid Ransomware Detection", page_icon="🛡️", layout="wide")
DARK_CSS = """
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stMetricValue"] { color: #4fd1c5; }
.risk-badge { padding: 4px 14px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; }
.risk-high { background: rgba(220,38,38,0.15); color: #f87171; }
.risk-low { background: rgba(34,197,94,0.15); color: #4ade80; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

@st.cache_resource
def get_detector() -> RansomwareDetector | None:
    try:
        return RansomwareDetector()
    except FileNotFoundError:
        return None

def gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=round(score * 100, 1), title={"text": "Ransomware Risk Score"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#f87171" if score > 0.5 else "#4ade80"}, "steps": [{"range": [0, 50], "color": "rgba(74,222,128,0.2)"}, {"range": [50, 100], "color": "rgba(248,113,113,0.2)"}]}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#e6e6e6")
    return fig

def page_home() -> None:
    st.title("🛡️ Hybrid Ransomware Detection System")
    st.markdown("A hybrid **supervised + unsupervised** system for flagging ransomware-associated Bitcoin wallets, combining a tuned gradient-boosting classifier with an Isolation Forest anomaly detector.")
    detector = get_detector()
    c1, c2, c3 = st.columns(3)
    if detector:
        meta = detector.metadata
        c1.metric("Best model", meta["best_pipeline_key"])
        c2.metric("Test F1", f'{meta["test_metrics"].get("f1", 0):.3f}')
        c3.metric("Test ROC-AUC", f'{meta["test_metrics"].get("roc_auc", 0):.3f}')
    else:
        st.warning("No trained model found yet. Run `python train.py` first, then relaunch this app.")

def page_about() -> None:
    st.title("About This Project")
    st.markdown("""
### Problem
Ransomware operators launder ransom payments through Bitcoin wallets. Identifying malicious wallets from on-chain behavior can support investigation and risk analysis.

### Approach
- **Supervised leg:** tree and gradient-boosting classifiers benchmarked across class weighting and resampling strategies.
- **Unsupervised leg:** Isolation Forest trained on unlabeled wallet traffic to identify anomalous behavior.
- **Hybrid ensemble:** supervised and anomaly scores blended using validation-calibrated weights and a decision threshold.

### Evaluation
Because the dataset is imbalanced, accuracy is not sufficient on its own. The pipeline also reports precision, recall, F1, PR-AUC, ROC-AUC, MCC and balanced accuracy.
""")

def page_dataset_info() -> None:
    st.title("Dataset Information")
    st.markdown("Expected files in `dataset/`: `BitcoinHeistData.csv.zip` (labeled) and `no_label.csv` (unlabeled, for the anomaly-detection leg).")
    for label, path in (("Labeled dataset", config.SUPERVISED_DATA_PATH), ("Unlabeled dataset", config.UNLABELED_DATA_PATH)):
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p, nrows=500)
            st.subheader(label)
            st.write("Preview (first 500 rows loaded).")
            st.dataframe(df.head(20))
        else:
            st.info(f"{label} not found at `{path}`.")

def page_prediction() -> None:
    st.title("Prediction")
    detector = get_detector()
    if detector is None:
        st.warning("No trained model found. Run `python train.py` first.")
        return
    uploaded = st.file_uploader("Upload a CSV of wallets to score", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with the same columns as the training data (minus the label) to get predictions.")
        return
    df = pd.read_csv(uploaded)
    results = detector.predict(df)
    st.success(f"Scored {len(results)} wallets.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(gauge(float(results["risk_score"].mean())), use_container_width=True)
    with col2:
        st.dataframe(results, use_container_width=True, height=280)
    st.download_button("Download predictions as CSV", data=results.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

def page_model_comparison() -> None:
    st.title("Model Comparison")
    comp_path = config.REPORTS_DIR / "model_comparison.csv"
    if not comp_path.exists():
        st.warning("No comparison table yet — run `python train.py` first.")
        return
    df = pd.read_csv(comp_path)
    st.dataframe(df, use_container_width=True)
    if "f1" in df.columns:
        fig = px.bar(df, x="model", y="f1", title="Validation F1 by model x imbalance strategy", color="f1", color_continuous_scale="Teal")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e6e6", xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

def page_explainable_ai() -> None:
    st.title("Explainable AI")
    st.markdown("SHAP explanations require the trained pipeline and a background sample of training data. Run this page after `python train.py`, with `shap` installed (`pip install shap`).")
    try:
        import shap
    except ImportError:
        st.error("`shap` is not installed. Run `pip install shap` and reload.")
        return
    st.info("Use `explainability.py` to build global and per-wallet explanations once a trained pipeline and representative background sample are available.")

def page_analytics() -> None:
    st.title("Analytics Dashboard")
    meta_path = config.METADATA_ARTIFACT_PATH
    if not meta_path.exists():
        st.warning("No trained model found. Run `python train.py` first.")
        return
    meta = load_json(meta_path)
    st.subheader("Test-set metrics")
    metrics_df = pd.DataFrame([meta["test_metrics"]]).T.rename(columns={0: "value"})
    st.dataframe(metrics_df)
    st.subheader("Calibrated ensemble parameters")
    c1, c2 = st.columns(2)
    c1.metric("Supervised weight", f'{meta["hybrid_weight_supervised"]:.2f}')
    c2.metric("Decision threshold", f'{meta["hybrid_threshold"]:.2f}')

PAGES = {"Home": page_home, "About Project": page_about, "Dataset Info": page_dataset_info, "Prediction": page_prediction, "Model Comparison": page_model_comparison, "Explainable AI": page_explainable_ai, "Analytics Dashboard": page_analytics}

def main() -> None:
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()
