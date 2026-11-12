import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_patient_data():
    # Load the cleaned dataset
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "cleaned_patient_records.csv"
    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # 1. Check reaction score distribution
    print("\nReaction Score Distribution:")
    print(df["Reaction_Score"].value_counts().sort_index())

    # 2. Check numerical summary
    print("\nNumerical Summary:")
    print(df[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]].describe())

    # Create a folder for figures
    figures_path = project_root / "outputs" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # Visualization 1: Reaction score distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Reaction_Score")
    plt.title("Distribution of Reaction Scores")
    plt.xlabel("Reaction Score")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(figures_path / "reaction_score_distribution.png", dpi=150)
    plt.close()

    # Visualization 2: Numerical feature distributions
    df[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]].hist(
        figsize=(10, 6), bins=15
    )
    plt.suptitle("Patient Feature Distributions")
    plt.tight_layout()
    plt.savefig(figures_path / "feature_distributions.png", dpi=150)
    plt.close()

    # Visualization 3: Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[["Age", "Biomarker_Level", "Vaccine_Dose_mcg", "Reaction_Score"]]
        .corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_path / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nAnalysis completed!")
    print("Three visualizations saved in outputs/figures.")


if __name__ == "__main__":
    analyze_patient_data()
