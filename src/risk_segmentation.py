"""
Customer Risk Segmentation Module for Customer Churn Prediction Agent.

Loads serialized model package from models/churn_model.joblib,
computes churn probability for raw customer records, categorizes risk levels
(Low Risk, Medium Risk, High Risk, Critical Risk), and exports full risk predictions.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\models\churn_model.joblib"


def load_model_package(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads saved model, preprocessor, and metadata package from joblib file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run src/train_model.py first.")
    return joblib.load(model_path)


def get_risk_level(churn_probability: float) -> str:
    """
    Categorizes churn probability score into understandable risk tiers.

    Thresholds:
    - probability < 0.30          -> "Low Risk"
    - 0.30 <= probability < 0.60  -> "Medium Risk"
    - 0.60 <= probability < 0.80  -> "High Risk"
    - probability >= 0.80         -> "Critical Risk"
    """
    if churn_probability < 0.30:
        return "Low Risk"
    elif churn_probability < 0.60:
        return "Medium Risk"
    elif churn_probability < 0.80:
        return "High Risk"
    else:
        return "Critical Risk"


def predict_customer_risk(customer_data, model_package=None, model_path: str = DEFAULT_MODEL_PATH):
    """
    Predicts churn probability, churn status ('Yes'/'No'), and assigns risk level
    for one or multiple raw customer records.

    Args:
        customer_data (dict, list, or pd.DataFrame): Raw customer record(s).
        model_package (dict, optional): Pre-loaded model package.
        model_path (str, optional): File path to joblib artifact.

    Returns:
        pd.DataFrame or dict: Input customer data appended with prediction metrics.
    """
    if model_package is None:
        model_package = load_model_package(model_path)

    model = model_package['model']
    preprocessor = model_package['preprocessor']

    # Standardize input format into a pandas DataFrame
    is_single_dict = False
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
        is_single_dict = True
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    else:
        raise ValueError("customer_data must be a dict, list of dicts, or pandas DataFrame.")

    # Prepare features for preprocessor (exclude customerID and Churn if present)
    X = df.copy()
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])
    if 'Churn' in X.columns:
        X = X.drop(columns=['Churn'])

    # Coerce TotalCharges to numeric float64 (handling blank string values)
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

    # Transform features using pre-fitted ColumnTransformer (NO refitting)
    X_transformed = preprocessor.transform(X)

    # Predict churn probabilities (class 1) and churn status binary predictions
    churn_probs = model.predict_proba(X_transformed)[:, 1]
    churn_preds = model.predict(X_transformed)

    # Append prediction metadata columns
    result_df = df.copy()
    result_df['churn_probability'] = np.round(churn_probs, 4)
    result_df['churn_prediction'] = np.where(churn_preds == 1, 'Yes', 'No')
    result_df['risk_level'] = result_df['churn_probability'].apply(get_risk_level)

    if is_single_dict:
        return result_df.iloc[0].to_dict()

    return result_df


def process_full_dataset(input_filepath: str, output_filepath: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads full dataset, generates predictions, prints risk tier distribution,
    and saves output to CSV.
    """
    print(f"Loading complete dataset from: {input_filepath} ...")
    raw_df = pd.read_csv(input_filepath)

    model_package = load_model_package(model_path)
    result_df = predict_customer_risk(raw_df, model_package=model_package)

    # Compute & print risk distribution summary
    print("\n==================================================")
    print("      FULL DATASET CUSTOMER RISK DISTRIBUTION     ")
    print("==================================================")
    risk_order = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    risk_counts = result_df['risk_level'].value_counts()
    risk_pcts = result_df['risk_level'].value_counts(normalize=True) * 100

    for level in risk_order:
        count = risk_counts.get(level, 0)
        pct = risk_pcts.get(level, 0.0)
        print(f" - {level:<15} : {count:>5} customers ({pct:>6.2f}%)")
    print("==================================================")

    # Save output CSV preserving original data + prediction columns
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    result_df.to_csv(output_filepath, index=False)
    print(f"\nSUCCESS: Exported full predictions to:\n  {output_filepath}")

    return result_df


if __name__ == "__main__":
    input_csv = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_csv = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\customer_risk_predictions.csv"

    print("--- TESTING SAMPLE CUSTOMER RISK SEGMENTATION ---")
    sample_customers = [
        {
            'customerID': 'SAMPLE-LOW-01',
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
            'tenure': 65, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
            'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Two year', 'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)', 'MonthlyCharges': 60.50, 'TotalCharges': '3932.50'
        },
        {
            'customerID': 'SAMPLE-MID-02',
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 18, 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Mailed check', 'MonthlyCharges': 55.20, 'TotalCharges': '993.60'
        },
        {
            'customerID': 'SAMPLE-HIGH-03',
            'gender': 'Female', 'SeniorCitizen': 1, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.70, 'TotalCharges': '191.40'
        }
    ]

    sample_results = predict_customer_risk(sample_customers)
    print("\nSample Customer Risk Predictions:")
    print(sample_results[['customerID', 'churn_probability', 'churn_prediction', 'risk_level']].to_string(index=False))

    print("\n--- PROCESSING FULL DATASET ---")
    full_results = process_full_dataset(input_csv, output_csv)

    print("\n--- FIRST 10 CUSTOMER PREDICTIONS SAMPLE ---")
    print(full_results[['customerID', 'churn_probability', 'churn_prediction', 'risk_level']].head(10).to_string(index=False))
