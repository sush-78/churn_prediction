"""
Model Trainer Module wrapper for Customer Churn Prediction Agent.
Imports core training logic from train_model.py.
"""

from src.train_model import train_and_evaluate_model

if __name__ == "__main__":
    data_filepath = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_save_path = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\models\churn_model.joblib"

    train_and_evaluate_model(data_filepath, model_save_path)
