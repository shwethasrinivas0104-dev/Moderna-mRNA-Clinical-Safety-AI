import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_patient_records.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    target = "Reaction_Score"
    features = [
        "Age",
        "Biomarker_Level",
        "Pre_Existing_Conditions",
        "Vaccine_Dose_mcg",
        "Prior_Reaction_History"
    ]

    X = df[features].copy()
    y = df[target].astype(int)

    numeric_features = [
        "Age",
        "Biomarker_Level",
        "Vaccine_Dose_mcg"
    ]
    categorical_features = [
        "Pre_Existing_Conditions",
        "Prior_Reaction_History"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    }

    results = []
    best_f1 = -1
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC AUC": roc_auc_score(y_test, probabilities)
        }
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("ROC AUC:", round(metrics["ROC AUC"], 4))

        if metrics["F1 Score"] > best_f1:
            best_f1 = metrics["F1 Score"]
            best_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    joblib.dump(best_pipeline, MODEL_DIR / "risk_model.joblib")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model by F1 Score: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'risk_model.joblib'}")
    print(f"Saved metrics to: {OUTPUT_DIR / 'model_comparison.csv'}")
    print("\nEducational project only—not for clinical decisions.")

if __name__ == "__main__":
    main()

