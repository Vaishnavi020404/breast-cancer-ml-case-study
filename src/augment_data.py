# src/augment_existing.py

"""
Augment Existing Breast Cancer Dataset
--------------------------------------
This script:
1. Loads the cleaned dataset
2. Duplicates rows with slight random noise to reach a target number of rows
3. Saves the augmented dataset (overwrites the cleaned dataset)
"""

import pandas as pd
import numpy as np
import os

# -----------------------
# 1. Load Cleaned Dataset
# -----------------------
file_path = "../data/breast_cancer_clean.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found: {file_path}")

data = pd.read_csv(file_path)
print(f"[INFO] Original dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# -----------------------
# 2. Define Target Rows
# -----------------------
target_rows = 5000
current_rows = data.shape[0]

if current_rows >= target_rows:
    print(f"[INFO] Dataset already has {current_rows} rows, no augmentation needed.")
else:
    multiplier = target_rows // current_rows + 1  # Number of times to duplicate
    print(f"[INFO] Augmenting dataset to reach {target_rows} rows...")

    # -----------------------
    # 3. Duplicate and Add Noise
    # -----------------------
    numeric_cols = data.drop("diagnosis", axis=1).columns
    augmented_list = []

    for i in range(multiplier):
        temp = data.copy()
        for col in numeric_cols:
            # Add small noise: ±1-2% of each numeric feature
            noise = temp[col] * np.random.uniform(-0.02, 0.02, size=temp.shape[0])
            temp[col] += noise
        augmented_list.append(temp)

    # -----------------------
    # 4. Concatenate and Trim
    # -----------------------
    augmented_data = pd.concat(augmented_list, ignore_index=True).sample(n=target_rows, random_state=42)
    print(f"[INFO] Augmented dataset shape: {augmented_data.shape[0]} rows")

    # -----------------------
    # 5. Save Augmented Dataset
    # -----------------------
    augmented_data.to_csv(file_path, index=False)
    print(f"[INFO] File overwritten: {file_path}")
