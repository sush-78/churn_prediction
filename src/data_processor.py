"""
Data Processing Module for Customer Churn Prediction Agent.

Handles:
- Dataset loading (pandas.read_csv)
- TotalCharges string-to-numeric coercion & NaN handling
- Target encoding (Churn: Yes -> 1, No -> 0)
- Identifier removal (customerID)
- Train-test splitting with stratification
- Scikit-learn ColumnTransformer & Pipeline creation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_clean_data(filepath: str):
    """
    Loads dataset from CSV file, converts TotalCharges to numeric (setting blanks to NaN),
    encodes Churn target column, and drops customerID identifier.

    Args:
        filepath (str): Path to dataset CSV file.

    Returns:
        tuple: (X, y, raw_df)
    """
    raw_df = pd.read_csv(filepath)
    df = raw_df.copy()

    # Convert TotalCharges from string to float64 (11 blank space strings ' ' become NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Encode Churn target column: Yes -> 1, No -> 0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        y = df['Churn']
        df = df.drop(columns=['Churn'])
    else:
        y = None

    # Remove customerID identifier column from feature set X
    if 'customerID' in df.columns:
        X = df.drop(columns=['customerID'])
    else:
        X = df

    return X, y, raw_df


def create_preprocessing_pipeline(X: pd.DataFrame):
    """
    Automatically identifies numerical and categorical features and constructs a
    Scikit-learn ColumnTransformer preprocessing pipeline.

    Args:
        X (pd.DataFrame): Input features DataFrame.

    Returns:
        tuple: (preprocessor, num_cols, cat_cols)
    """
    # Automatically identify feature types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Preprocessing for numerical features (Imputation + Scaling)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features (Imputation + OneHotEncoder)
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor, num_cols, cat_cols


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Complete pipeline to load, split, fit preprocessor on train set, and transform train & test sets.

    Returns:
        dict: Preprocessed datasets, preprocessor object, feature names, and metadata.
    """
    X, y, raw_df = load_and_clean_data(filepath)

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Build preprocessing pipeline
    preprocessor, num_cols, cat_cols = create_preprocessing_pipeline(X)

    # Fit preprocessor ONLY on X_train to prevent data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    # Transform X_test using the fitted preprocessor
    X_test_transformed = preprocessor.transform(X_test)

    # Extract transformed feature names for explainability downstream
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_names

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "preprocessor": preprocessor,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": all_feature_names
    }


if __name__ == "__main__":
    filepath = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_dict = prepare_data(filepath)

    print("==================================================")
    print("      DATA PREPROCESSING PIPELINE REPORT          ")
    print("==================================================")
    print(f"1. Training Rows Count (X_train)   : {data_dict['X_train_raw'].shape[0]}")
    print(f"2. Testing Rows Count (X_test)     : {data_dict['X_test_raw'].shape[0]}")
    print(f"3. Numerical Features Count        : {len(data_dict['num_cols'])}")
    print(f"   List: {data_dict['num_cols']}")
    print(f"4. Categorical Features Count      : {len(data_dict['cat_cols'])}")
    print(f"   List: {data_dict['cat_cols']}")
    print(f"5. Transformed Feature Count       : {data_dict['X_train_transformed'].shape[1]}")
    print(f"6. Preprocessing Verification      : SUCCESS! Fitted on X_train, transformed X_test cleanly.")
    print("==================================================")
