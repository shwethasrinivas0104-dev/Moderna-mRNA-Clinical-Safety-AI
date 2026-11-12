from pathlib import Path
import sys
import json
import joblib
import pandas as pd

from pydantic import BaseModel, Field
from typing import List

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "risk_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"

# Import RAG functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_search

# Patient features used by the ML model
FEATURE_COLUMNS = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History"
]


# Clinical safety report validation
class ClinicalSafetyReport(BaseModel):
    patient_id: str
    predicted_risk_band: str
    risk_probability: float = Field(ge=0, le=1)
    key_findings: List[str]
    supporting_evidence: List[str]
    safety_recommendations: List[str]
    limitations: List[str]
    disclaimer: str = (
        "Educational project only. Not for clinical decision-making."
    )


def predict_patient_risk(patient_data):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    patient_df = pd.DataFrame(
        [patient_data],
        columns=FEATURE_COLUMNS
    )

    prediction = int(model.predict(patient_df)[0])
    probability = float(model.predict_proba(patient_df)[0][1])

    risk_band = "Moderate" if prediction == 1 else "Low"

    return risk_band, probability


def generate_report(report_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate report; retry validation once
    try:
        report = ClinicalSafetyReport(**report_data)
    except Exception as first_error:
        print("Validation failed. Retrying once...")
        try:
            report = ClinicalSafetyReport(**report_data)
        except Exception:
            raise ValueError(
                f"Report validation failed: {first_error}"
            )

    # Save JSON
    json_path = OUTPUT_DIR / "clinical_safety_report.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report.model_dump(), file, indent=4)

    # Save Markdown
    markdown_path = OUTPUT_DIR / "clinical_safety_report.md"

    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write("# Clinical Safety Report\n\n")

        for key, value in report.model_dump().items():
            file.write(f"## {key.replace('_', ' ').title()}\n")

            if isinstance(value, list):
                for item in value:
                    file.write(f"- {item}\n")
            else:
                file.write(f"{value}\n")

            file.write("\n")

    print("\nClinical safety report generated successfully!")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    return report


if __name__ == "__main__":

    print("Initializing clinical safety platform...")

    # Initialize RAG database and embedding model
    rag_search.main()

    if rag_search.model is None or rag_search.collection is None:
        raise RuntimeError("RAG initialization failed.")

    # Demo patient data
    patient_data = {
        "Age": 35,
        "Biomarker_Level": 2.5,
        "Pre_Existing_Conditions": 1,
        "Vaccine_Dose_mcg": 100,
        "Prior_Reaction_History": 1
    }

    # ML prediction
    risk_band, probability = predict_patient_risk(patient_data)

    print(f"\nPredicted risk band: {risk_band}")
    print(f"Risk probability: {probability:.4f}")

    # Retrieve safety evidence
    question = (
        "What precautions apply to patients with prior allergic reactions?"
    )

    rag_results = rag_search.retrieve_documents(question, top_k=3)

    documents = rag_results.get("documents", [[]])[0]
    distances = rag_results.get("distances", [[]])[0]
    print("RAG distances:", distances)

    # Do not present weak retrievals as reliable evidence
    if documents and distances and min(distances) <= 1.35:
        supporting_evidence = documents
    else:
        supporting_evidence = [
            "No sufficiently relevant evidence was retrieved. "
            "Further review is required."
        ]

    # Build report
    report_data = {
        "patient_id": "DEMO-001",
        "predicted_risk_band": risk_band,
        "risk_probability": probability,
        "key_findings": [
            "Prediction generated using the trained ML model.",
            f"Predicted risk band: {risk_band}."
        ],
        "supporting_evidence": supporting_evidence,
        "safety_recommendations": [
            "Review the retrieved educational safety guidance.",
            "Seek qualified clinical review where appropriate."
        ],
        "limitations": [
            "Synthetic demonstration data.",
            "Model is not clinically validated.",
            "RAG relevance threshold is experimental.",
            "This report is not a medical recommendation."
        ]
    }

    generate_report(report_data)

