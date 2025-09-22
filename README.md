# Breast Cancer ML Case Study

## Overview

This project demonstrates a machine learning case study on **breast cancer diagnosis**.
It includes **data cleaning, augmentation, and model training**, using multiple ML algorithms to predict whether a tumor is malignant or benign based on numeric features.

---

## Features

* Clean raw dataset to ensure data quality
* Augment dataset to reach **5000 samples** for robust training
* Implement multiple ML models:

  * Decision Tree
  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest
* Detailed evaluation: accuracy, confusion matrix, classification report
* Summary of all models saved as CSV for reporting

---

## Project Structure


breast-cancer-ml-case-study/
├── data/
│   ├── Breast_cancer_data.csv       # Original raw dataset
│   └── breast_cancer_clean.csv      # Cleaned and augmented dataset
│   └──model_summary.csv             # Summary of trained models
├── src/
│   ├── venv/                        # Virtual environment
│   ├── clean_data.py                 # Cleans raw dataset
│   ├── augment_existing.py           # Augments dataset to 5000 rows
│   ├── breast_cancer_case_study.py  # ML models and evaluation
├── requirements.txt                  # Required Python packages
└── README.md


---

## Dependencies

The project requires **Python 3.10+** and the following packages:

pandas>=1.5.0
numpy>=1.25.0
scikit-learn>=1.2.0

Install them using:

pip install -r requirements.txt

 ⚠️ Make sure you are in the project root directory when running this command.



## How to Run

1. **Clean the dataset**

python src/clean_data.py


2. **Augment the dataset to 5000 rows**

python src/augment_existing.py

3. **Run ML models and get results**

python src/breast_cancer_case_study.py

* The script prints detailed metrics (accuracy, confusion matrix, classification report) for all models.
* A **summary CSV** of all models is saved at: `src/model_summary.csv`.

---

## Notes

* The **cleaned and augmented dataset** (`breast_cancer_clean.csv`) is used for modeling.
* Original raw data (`Breast_cancer_data.csv`) is preserved for reference.
* The `src/venv` contains the virtual environment for dependency management.
* Using a **virtual environment (`venv`)** is recommended to avoid conflicts.
* The **model summary** updates automatically every time `breast_cancer_case_study.py` is run.

---

## Author:
**Vaishnavi**

---
