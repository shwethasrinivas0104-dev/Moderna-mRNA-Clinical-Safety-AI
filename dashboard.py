import streamlit as st
import pandas as pd
from pathlib import Path

# Project folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Dashboard configuration
st.set_page_config(
    page_title="Moderna mRNA Clinical Safety Dashboard",
    layout="wide"
)

# Dashboard title
st.title("Moderna mRNA Clinical Safety Dashboard")

st.write(
    "An educational AI/ML platform for exploring "
    "mRNA vaccine clinical safety information."
)

st.warning(
    "Educational use only. This dashboard is not intended "
    "for clinical diagnosis or treatment decisions."
)

st.divider()

# PANEL 1: ROC Curve
st.header("1. Machine Learning Model Evaluation")

roc_path = FIGURES_DIR / "roc_curve.png"

if roc_path.exists():
    st.image(
        str(roc_path),
        caption="ROC Curve - Model Performance",
        use_container_width=True
    )
else:
    st.warning("ROC curve image not found.")

st.divider()

# PANEL 2: Attention Heatmap
st.header("2. Deep Learning Attention Analysis")

attention_path = FIGURES_DIR / "attention_heatmap.png"

if attention_path.exists():
    st.image(
        str(attention_path),
        caption="Self-Attention Heatmap",
        use_container_width=True
    )
else:
    st.warning("Attention heatmap not found.")

st.divider()

# PANEL 3: RAG Retrieval Evaluation
st.header("3. RAG Retrieval Evaluation")

rag_path = OUTPUTS_DIR / "rag_evaluation.csv"

if rag_path.exists():
    rag_data = pd.read_csv(rag_path)

    st.subheader("Retrieval Results")
    st.dataframe(rag_data, use_container_width=True)

    if "retrieval_latency_seconds" in rag_data.columns:
        st.subheader("Retrieval Latency")

        st.bar_chart(
            rag_data["retrieval_latency_seconds"]
        )

else:
    st.warning("RAG evaluation file not found.")

st.divider()

# Footer
st.caption(
    "Moderna mRNA Vaccine Clinical Safety & Dosage "
    "Intelligence Platform | Educational Project"
)

