from data_loader import load_patient_data


def clean_patient_data():
    df = load_patient_data()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Fill missing numerical values with the median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Vaccine_Dose_mcg"] = df["Vaccine_Dose_mcg"].fillna(
        df["Vaccine_Dose_mcg"].median()
    )

    df["Biomarker_Level"] = df["Biomarker_Level"].fillna(df["Biomarker_Level"].median())

    # Fill missing categorical values with the most common value
    df["Prior_Reaction_History"] = df["Prior_Reaction_History"].fillna(
        df["Prior_Reaction_History"].mode()[0]
    )

    print("\nData cleaning completed!")
    print("Cleaned dataset shape:", df.shape)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    df = clean_patient_data()
    df.to_csv("data/cleaned_patient_records.csv", index=False)
    print("Cleaned dataset saved successfully!")
