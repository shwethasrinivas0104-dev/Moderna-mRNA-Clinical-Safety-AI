from pathlib import Path
import pandas as pd


def load_patient_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "patient_records.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_patient_data()
