import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_patient_records.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
FIGURES_PATH = ROOT / "outputs" / "figures"


def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    features = [
        "Age",
        "Biomarker_Level",
        "Pre_Existing_Conditions",
        "Vaccine_Dose_mcg",
        "Prior_Reaction_History"
    ]

    X = df[features]
    y = df["Reaction_Score"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Model evaluation completed!")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    auc = roc_auc_score(y_test, probabilities)
    print(f"ROC AUC: {auc:.3f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Reaction Score Prediction")
    plt.legend()
    plt.tight_layout()

    roc_path = FIGURES_PATH / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Confusion matrix
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()

    matrix_path = FIGURES_PATH / "confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.close()

    print(f"\nROC curve saved to: {roc_path}")
    print(f"Confusion matrix saved to: {matrix_path}")
    print("\nEducational project only—not for clinical decisions.")


if __name__ == "__main__":
    main()

