# Titanic ML Baseline (LogReg + Pipeline)

## Goal
Predict survival on Titanic using a baseline machine learning model.

## What I did
- Loaded dataset with pandas  
- Built preprocessing pipeline:
  - numeric features: median imputation
  - categorical features: most frequent imputation + one-hot encoding
- Trained Logistic Regression model
- Evaluated with Accuracy, ROC-AUC and Confusion Matrix
- Tested different probability thresholds (0.3 / 0.5 / 0.7)

## Results (example)
- Dummy Accuracy: ~0.61
- Logistic Regression Accuracy: ~0.80
- ROC-AUC: ~0.84

## How to run
```bash
pip install -r titanic-ml-baseline/requirements.txt
python titanic_model_v2.py
