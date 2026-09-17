"""
Model Training Module for Customer Churn Prediction Agent.

Trains a Random Forest Classifier using preprocessed features from src/data_processor.py,
evaluates performance metrics, prints detailed confusion matrix and customer count summaries,
and serializes model artifacts into models/churn_model.joblib.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.data_processor import prepare_data


def train_and_evaluate_model(data_filepath: str, model_save_path: str):
    """
    Loads preprocessed data, trains a Random Forest Classifier, evaluates metrics,
    and saves the model & preprocessor package to joblib file.

    Args:
        data_filepath (str): Path to dataset CSV file.
        model_save_path (str): Path to output joblib artifact file.

    Returns:
        dict: Trained model package.
    """
    print(f"Loading data and fitting preprocessor from: {data_filepath} ...")
    data_dict = prepare_data(data_filepath)

    X_train_trans = data_dict['X_train_transformed']
    X_test_trans = data_dict['X_test_transformed']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']
    feature_names = data_dict['feature_names']
    num_cols = data_dict['num_cols']
    cat_cols = data_dict['cat_cols']

    print("Training Random Forest Classifier (n_estimators=200, class_weight='balanced') ...")
    # Initialize Random Forest model with balanced class weight to address churn class imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Train model ONLY on training data
    model.fit(X_train_trans, y_train)

    # Generate predictions on test data
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1]

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    actual_churn_count = int(np.sum(y_test == 1))
    actual_retained_count = int(np.sum(y_test == 0))
    pred_churn_count = int(np.sum(y_pred == 1))
    pred_retained_count = int(np.sum(y_pred == 0))

    # Display clean report
    print("\n==================================================")
    print("        CUSTOMER CHURN MODEL EVALUATION           ")
    print("==================================================")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"[[ True Negative (Retained correctly)  : {cm[0][0]} | False Positive (False Alarm Churn) : {cm[0][1]} ]")
    print(f" [ False Negative (Missed Churn)      : {cm[1][0]} | True Positive  (Caught Churn)      : {cm[1][1]} ]]")
    print("\nClassification Report:")
    print(clf_report)
    print("--------------------------------------------------")
    print(f"Actual Churn Customers in Test Set    : {actual_churn_count}")
    print(f"Actual Retained Customers in Test Set : {actual_retained_count}")
    print(f"Predicted Churn Customers             : {pred_churn_count}")
    print(f"Predicted Retained Customers          : {pred_retained_count}")
    print("==================================================")

    # Package trained artifacts into a single object for downstream inference & explainability
    model_package = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        }
    }

    # Save to joblib
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_package, model_save_path)
    print(f"\nSUCCESS: Trained model and preprocessor saved to: {model_save_path}\n")

    return model_package


if __name__ == "__main__":
    data_filepath = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_save_path = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\models\churn_model.joblib"

    train_and_evaluate_model(data_filepath, model_save_path)
