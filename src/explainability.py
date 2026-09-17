"""
Feature Importance & Explainability Module for Customer Churn Prediction Agent.

Provides:
1. Global Feature Importance (Random Forest feature_importances_ + visualization)
2. Individual Customer SHAP Explainability (TreeExplainer + waterfall/bar visualizations)
3. Factor extraction with precise human-readable feature state mappings for SHAP drivers
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

DEFAULT_MODEL_PATH = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\models\churn_model.joblib"
DEFAULT_PRED_PATH = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\customer_risk_predictions.csv"


def load_model_package(model_path: str = DEFAULT_MODEL_PATH):
    """Loads saved model, preprocessor, and feature names package."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train_model.py first.")
    return joblib.load(model_path)


def get_global_feature_importance(model_package=None, top_n: int = 15, model_path: str = DEFAULT_MODEL_PATH, save_csv: bool = True, save_png: bool = True):
    """
    Computes global feature importance from Random Forest feature_importances_,
    saves result to CSV, and generates a visual bar chart.
    """
    if model_package is None:
        model_package = load_model_package(model_path)

    model = model_package['model']
    feature_names = model_package['feature_names']

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    if save_csv:
        csv_path = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\feature_importance.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        importance_df.to_csv(csv_path, index=False)
        print(f"Global feature importance CSV saved to: {csv_path}")

    if save_png:
        png_path = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\feature_importance.png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        top_df = importance_df.head(top_n).sort_values(by='importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(top_df['feature'], top_df['importance'], color='#2E86AB')
        plt.xlabel("Random Forest Feature Importance Score")
        plt.title(f"Top {top_n} Global Features Driving Customer Churn", fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Global feature importance plot saved to: {png_path}")

    return importance_df


def _extract_class1_shap(shap_values):
    """
    Helper function to safely extract Class 1 (Churn) SHAP values across various SHAP return formats.
    Handles list of arrays, 3D numpy arrays, and SHAP Explanation objects.
    """
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif hasattr(shap_values, "values"):
        vals = shap_values.values
        if isinstance(vals, list):
            vals = vals[1]
        elif vals.ndim == 3:
            vals = vals[:, :, 1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    return vals


def format_shap_explanation(feature_name: str, transformed_val: float, raw_customer: dict) -> str:
    """
    Maps feature name, transformed numeric value (0 or 1 for OHE, scaled for numerical),
    and actual raw customer attribute to an accurate human-readable description.
    """
    # Continuous / Numerical Features
    if feature_name == "tenure":
        t_val = raw_customer.get("tenure", 0)
        return f"Customer tenure is {t_val} month{'s' if t_val != 1 else ''}"

    if feature_name == "MonthlyCharges":
        val = float(raw_customer.get("MonthlyCharges", 0.0))
        return f"Monthly charges are ${val:.2f}"

    if feature_name == "TotalCharges":
        val_str = str(raw_customer.get("TotalCharges", "0")).strip()
        try:
            val = float(val_str)
            return f"Total accumulated charges are ${val:.2f}"
        except ValueError:
            return "Total charges accumulated"

    if feature_name == "SeniorCitizen":
        is_senior = int(raw_customer.get("SeniorCitizen", 0)) == 1
        return "Customer is a Senior Citizen" if is_senior else "Customer is not a Senior Citizen"

    # One-Hot Encoded Categorical Features
    if "_" in feature_name:
        category, option = feature_name.split("_", 1)
        is_active = (transformed_val > 0.5)

        if category == "Contract":
            if is_active:
                return f"Has a {option} contract"
            else:
                raw_c = raw_customer.get("Contract", "")
                return f"Lacks a {option} contract (Has {raw_c})"

        if category == "InternetService":
            if is_active:
                return f"Subscribed to {option} internet"
            else:
                raw_i = raw_customer.get("InternetService", "")
                return f"Does not have {option} internet (Uses {raw_i})"

        if category == "PaymentMethod":
            if is_active:
                return f"Pays via {option}"
            else:
                raw_p = raw_customer.get("PaymentMethod", "")
                return f"Does not pay via {option} (Uses {raw_p})"

        if category in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            service_title = {
                "OnlineSecurity": "Online Security",
                "OnlineBackup": "Online Backup",
                "DeviceProtection": "Device Protection",
                "TechSupport": "Tech Support",
                "StreamingTV": "Streaming TV",
                "StreamingMovies": "Streaming Movies"
            }.get(category, category)

            if option == "No":
                return f"Lacks {service_title} service" if is_active else f"Has {service_title} service"
            elif option == "Yes":
                return f"Has {service_title} service" if is_active else f"Lacks {service_title} service"
            else:
                return f"No internet for {service_title}" if is_active else f"Has internet for {service_title}"

        if category in ["Partner", "Dependents", "PaperlessBilling", "PhoneService"]:
            title = {
                "Partner": "Partner account",
                "Dependents": "Dependents",
                "PaperlessBilling": "Paperless billing",
                "PhoneService": "Phone service"
            }.get(category, category)

            if option == "Yes":
                return f"Has {title}" if is_active else f"Does not have {title}"
            elif option == "No":
                return f"Does not have {title}" if is_active else f"Has {title}"

        if category == "gender":
            return f"Gender is {option}" if is_active else f"Gender is not {option}"

        if category == "MultipleLines":
            if is_active:
                return f"Multiple lines status: {option}"
            else:
                raw_m = raw_customer.get("MultipleLines", "")
                return f"Multiple lines status: {raw_m}"

    return f"{feature_name} (Value: {transformed_val:.2f})"


def explain_customer(customer_data, model_package=None, model_path: str = DEFAULT_MODEL_PATH):
    """
    Generates SHAP explainability for a single raw customer record.
    Returns churn probability, risk level, top positive SHAP factors (increasing churn risk),
    top negative SHAP factors (reducing churn risk), and human-readable feature descriptions.
    """
    if model_package is None:
        model_package = load_model_package(model_path)

    model = model_package['model']
    preprocessor = model_package['preprocessor']
    feature_names = model_package['feature_names']

    # Standardize input to 1-row DataFrame
    if isinstance(customer_data, dict):
        raw_dict = customer_data.copy()
        raw_df = pd.DataFrame([raw_dict])
    elif isinstance(customer_data, pd.DataFrame):
        raw_df = customer_data.copy().head(1)
        raw_dict = raw_df.iloc[0].to_dict()
    else:
        raise ValueError("customer_data must be a dict or pandas DataFrame.")

    customer_id = raw_dict.get('customerID', 'UNKNOWN')

    # Prepare features for preprocessor (drop customerID, Churn, and prediction columns)
    X = raw_df.copy()
    drop_cols = [c for c in ['customerID', 'Churn', 'churn_probability', 'churn_prediction', 'risk_level'] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

    # Transform features using saved ColumnTransformer
    X_transformed = preprocessor.transform(X)

    # Predict churn probability & class
    churn_prob = float(model.predict_proba(X_transformed)[0][1])
    pred_class = "Yes" if churn_prob >= 0.5 else "No"

    # Risk level thresholding
    if churn_prob < 0.30:
        risk_level = "Low Risk"
    elif churn_prob < 0.60:
        risk_level = "Medium Risk"
    elif churn_prob < 0.80:
        risk_level = "High Risk"
    else:
        risk_level = "Critical Risk"

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_transformed)

    shap_class1 = _extract_class1_shap(raw_shap_values)
    single_shap = shap_class1[0] if shap_class1.ndim > 1 else shap_class1
    transformed_vals = X_transformed[0]

    # Build feature explanation table
    explanation_rows = []
    for f_name, t_val, s_val in zip(feature_names, transformed_vals, single_shap):
        h_desc = format_shap_explanation(f_name, t_val, raw_dict)
        explanation_rows.append({
            'feature': f_name,
            'transformed_val': t_val,
            'shap_value': float(s_val),
            'human_explanation': h_desc
        })

    explanation_df = pd.DataFrame(explanation_rows)

    # Extract top 5 positive and negative SHAP factors
    pos_df = explanation_df[explanation_df['shap_value'] > 0].sort_values(by='shap_value', ascending=False).head(5)
    neg_df = explanation_df[explanation_df['shap_value'] < 0].sort_values(by='shap_value', ascending=True).head(5)

    pos_list = [(row['feature'], row['shap_value'], row['human_explanation']) for _, row in pos_df.iterrows()]
    neg_list = [(row['feature'], row['shap_value'], row['human_explanation']) for _, row in neg_df.iterrows()]

    return {
        'customer_id': customer_id,
        'churn_probability': churn_prob,
        'predicted_class': pred_class,
        'risk_level': risk_level,
        'raw_customer': raw_dict,
        'top_positive_factors': pos_list,
        'top_negative_factors': neg_list,
        'shap_values': single_shap,
        'feature_names': feature_names,
        'X_transformed': X_transformed
    }


def save_customer_explanation_plot(explanation_result, save_path: str = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\sample_customer_explanation.png"):
    """
    Generates a clear bar plot of the top positive and negative SHAP factors for an individual customer.
    """
    customer_id = explanation_result['customer_id']
    prob = explanation_result['churn_probability'] * 100
    risk = explanation_result['risk_level']

    pos = explanation_result['top_positive_factors']
    neg = explanation_result['top_negative_factors']

    # Combine top positive and negative factors
    combined = sorted([(x[2], x[1]) for x in (pos + neg)], key=lambda x: x[1])

    features = [x[0] for x in combined]
    values = [x[1] for x in combined]
    colors = ['#E63946' if v > 0 else '#2A9D8F' for v in values]

    plt.figure(figsize=(11, 6))
    plt.barh(features, values, color=colors)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("SHAP Contribution to Churn Probability (+ Churn / - Retention)")
    plt.title(f"Customer {customer_id} SHAP Drivers ({prob:.1f}% Churn Prob - {risk})", fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Individual customer SHAP plot saved to: {save_path}")


def print_formatted_explanation(exp_result):
    """
    Prints audit explanation report for a customer showing raw features and SHAP drivers.
    """
    raw = exp_result['raw_customer']
    print("==================================================")
    print(f"Customer ID          : {exp_result['customer_id']}")
    print(f"Actual Contract      : {raw.get('Contract')}")
    print(f"Actual InternetServ  : {raw.get('InternetService')}")
    print(f"Actual OnlineSec     : {raw.get('OnlineSecurity')}")
    print(f"Actual OnlineBackup  : {raw.get('OnlineBackup')}")
    print(f"Actual TechSupport   : {raw.get('TechSupport')}")
    print(f"Actual PaymentMethod : {raw.get('PaymentMethod')}")
    print(f"Churn Probability    : {exp_result['churn_probability']*100:.2f}% ({exp_result['risk_level']})")

    print("\nTop 5 Positive SHAP Features (Increasing Churn Risk):")
    for idx, (feat, val, desc) in enumerate(exp_result['top_positive_factors'], 1):
        print(f"  {idx}. [{feat}] -> +{val:.4f}")
        print(f"     Explanation: {desc}")

    print("\nTop 5 Negative SHAP Features (Reducing Churn Risk):")
    for idx, (feat, val, desc) in enumerate(exp_result['top_negative_factors'], 1):
        print(f"  {idx}. [{feat}] -> {val:.4f}")
        print(f"     Explanation: {desc}")
    print("==================================================")


if __name__ == "__main__":
    print("--- 1. GENERATING GLOBAL FEATURE IMPORTANCE ---")
    model_pkg = load_model_package()
    global_importance = get_global_feature_importance(model_package=model_pkg, top_n=15)

    print("\nTop 15 Global Features:")
    print(global_importance.head(15).to_string(index=False))

    print("\n--- 2. SHAP AUDIT TEST FOR 4 REQUIRED CUSTOMERS ---")
    predictions_csv = DEFAULT_PRED_PATH
    if os.path.exists(predictions_csv):
        df_preds = pd.read_csv(predictions_csv)
        audit_ids = ['7590-VHVEG', '1452-KIOVK', '3668-QPYBK', '9237-HQITU']

        for cust_id in audit_ids:
            cust_rows = df_preds[df_preds['customerID'] == cust_id]
            if not cust_rows.empty:
                cust_dict = cust_rows.iloc[0].to_dict()
                exp_res = explain_customer(cust_dict, model_package=model_pkg)
                print_formatted_explanation(exp_res)
                print()

        # Save plot for critical risk customer 3668-QPYBK
        crit_rows = df_preds[df_preds['customerID'] == '3668-QPYBK']
        if not crit_rows.empty:
            crit_exp = explain_customer(crit_rows.iloc[0].to_dict(), model_package=model_pkg)
            save_customer_explanation_plot(crit_exp)
    else:
        print(f"Prediction file not found at {predictions_csv}. Run src/risk_segmentation.py first.")
