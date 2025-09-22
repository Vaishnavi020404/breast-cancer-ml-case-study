# src/clean_data.py

import pandas as pd
import os

def clean_dataset(input_path="../data/breast_cancer_upsampled.csv",
                  output_path="../data/breast_cancer_clean.csv"):
    """
    Cleans the breast cancer dataset.

    Steps:
    1. Loads the dataset from the specified input path.
    2. Converts the 'diagnosis' column to numeric; invalid entries are set to NaN.
    3. Removes rows with missing or invalid 'diagnosis'.
    4. Converts 'diagnosis' to integer (0 or 1).
    5. Saves the cleaned dataset to the specified output path.

    Parameters:
    - input_path: str, path to the input CSV file
    - output_path: str, path to save the cleaned CSV file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load dataset
    data = pd.read_csv(input_path)
    print(f"[INFO] Loaded dataset from '{input_path}' ({len(data)} rows)")

    # Convert diagnosis to numeric, invalid entries become NaN
    data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")

    # Drop rows with NaN diagnosis
    initial_rows = len(data)
    data = data.dropna(subset=["diagnosis"])
    dropped_rows = initial_rows - len(data)

    # Convert diagnosis to integer
    data["diagnosis"] = data["diagnosis"].astype(int)

    # Save cleaned dataset
    data.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned dataset saved to '{output_path}'")
    print(f"[INFO] Rows dropped due to invalid/missing diagnosis: {dropped_rows}")
    print(f"[INFO] Remaining rows in cleaned dataset: {len(data)}")


if __name__ == "__main__":
    clean_dataset()
