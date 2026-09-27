# Customer Churn Prediction

End-to-end classification workflow for identifying customers at risk of churn using the IBM Telco Customer Churn dataset.

## Highlights
- Data cleaning and validation
- Stratified train/test split
- Leakage-safe preprocessing with `Pipeline`
- One-hot encoding and scaling
- Logistic Regression, Random Forest, Gradient Boosting
- Accuracy, Precision, Recall, F1, ROC-AUC
- Reusable Joblib model pipeline

## Dataset
IBM Telco Customer Churn: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The training script downloads the dataset when needed. No performance numbers are hard-coded.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

Then create `data/raw/sample.csv` with the same feature columns and run `python src/predict.py`.

## Methodology
Preprocessing is fitted on training data and then applied to held-out data. This prevents test information from influencing preprocessing.

## Author
Hassan Ali — Computer Science student focused on Machine Learning and AI Engineering.
