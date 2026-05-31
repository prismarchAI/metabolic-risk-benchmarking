# Real-World Diabetes Diagnostic & Benchmarking Pipeline

<p align="left">
  <img src="https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Framework-XGBoost-111111?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Level%200%20Baseline-blue?style=flat-square" alt="Status">
</p>

An internal diagnostic engine designed to evaluate baseline machine learning models against a real-world diabetes dataset. This pipeline serves as a multi-model benchmarking environment, evaluating feature relationships across linear, baggings, and boosting topologies.

---

## Core Repository Structure

The current version of this project consists exclusively of the following data and source files:

```text
├── real_diabetes_data.csv    # Source clinical data records
├── logistic_Regression.py    # Baseline linear probabilistic model
├── random_forest.py         # 100-tree ensemble bagging model with feature scaling
└── xgboost_classifier.py    # Gradient boosted decision tree execution script

