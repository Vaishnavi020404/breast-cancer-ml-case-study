# src/breast_cancer_case_study.py

"""
Breast Cancer ML Case Study
---------------------------
This script:
1. Loads the cleaned dataset
2. Splits data into features and target
3. Trains multiple ML models
4. Outputs detailed metrics for each model
5. Saves a summary table for reporting

Models included:
- Decision Tree
- Logistic Regression
- SVM
- Random Forest
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# -----------------------
# 1. Load Cleaned Dataset
# -----------------------
dataset_path = "../data/breast_cancer_clean.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

data = pd.read_csv(dataset_path)
print(f"[INFO] Dataset loaded successfully: {len(data)} rows, {data.shape[1]} columns\n")

# Dataset overview
print("[INFO] Dataset Info:")
print(data.info(), "\n")
print("[INFO] Class Distribution:")
print(data['diagnosis'].value_counts(), "\n")
print("[INFO] First 5 rows:")
print(data.head(), "\n")
print("[INFO] Descriptive statistics of features:")
print(data.describe(), "\n")

# -----------------------------
# 2. Split Features and Target
# -----------------------------
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

print(f"[INFO] Features and target separated")
print(f"Number of features: {X.shape[1]}, Number of samples: {X.shape[0]}\n")

# -----------------------
# 3. Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Train/Test split done")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")

# -----------------------
# 4. Define ML Models
# -----------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# -----------------------
# 5. Train and Evaluate
# -----------------------
results = []

for name, model in models.items():
    print(f"[INFO] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4)

    print(f"[RESULT] Model: {name}")
    print(f"Accuracy: {acc:.4%}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    print("-" * 50, "\n")

    # Save for summary
    results.append({
        "Model": name,
        "Accuracy": acc,
        "True Positives": cm[1, 1],
        "True Negatives": cm[0, 0],
        "False Positives": cm[0, 1],
        "False Negatives": cm[1, 0]
    })

# -----------------------
# 6. Summary Table
# -----------------------
summary = pd.DataFrame(results)
print("[INFO] Summary Table of all models:\n")
print(summary, "\n")

# Optional: Save summary to CSV
summary_output_path = "../data/model_summary.csv"
summary.to_csv(summary_output_path, index=False)
print(f"[INFO] Summary saved as '{summary_output_path}'")
