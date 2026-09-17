"""
Feature Stickiness Matrix Module (Analytics)

Calculates observed churn rate differences between users with and without specific features/services.
Observational analysis ONLY -- associations do not imply causation.
"""

import pandas as pd
import numpy as np

SERVICES_TO_ANALYZE = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'MultipleLines',
    'PhoneService'
]

CAUSATION_DISCLAIMER = (
    "Note: Observational analysis only. "
    "Observed associations between service adoption and churn rates do not imply direct causation."
)


def calculate_feature_stickiness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes observed churn rate metrics for each key feature/service.

    Args:
        df (pd.DataFrame): Dataframe with service columns and 'Churn' (or 'churn_prediction'/'churn_probability').

    Returns:
        pd.DataFrame: Table containing:
            - feature: Service name
            - user_count: Number of customers adopting the feature
            - non_user_count: Number of customers without the feature
            - churn_rate_with: Actual churn rate among users (%)
            - churn_rate_without: Actual churn rate among non-users (%)
            - churn_reduction_abs: Absolute percentage point reduction in churn
            - churn_reduction_pct: Relative percentage reduction in churn (%)
            - stickiness_tier: Anchor classification (High Anchor, Moderate Anchor, Low Anchor)
    """
    data = df.copy()

    # Normalize Churn column to 0/1 binary numeric
    if 'Churn' in data.columns:
        churn_series = data['Churn'].astype(str).str.strip().eq('Yes').astype(int)
        # If all were False (e.g. Churn was already numeric 1/0 or 1.0/0.0), check numeric conversion
        if churn_series.sum() == 0 and data['Churn'].astype(str).str.contains('1').any():
            churn_series = pd.to_numeric(data['Churn'], errors='coerce').fillna(0).astype(int)
    elif 'churn_prediction' in data.columns:
        churn_series = pd.to_numeric(data['churn_prediction'], errors='coerce').fillna(0).astype(int)
    elif 'churn_probability' in data.columns:
        churn_series = (pd.to_numeric(data['churn_probability'], errors='coerce').fillna(0) >= 0.5).astype(int)
    else:
        raise ValueError("DataFrame must contain 'Churn', 'churn_prediction', or 'churn_probability' column.")

    data['binary_churn'] = churn_series

    rows = []
    for service in SERVICES_TO_ANALYZE:
        if service not in data.columns:
            continue

        # Users are those with 'Yes'
        is_user = data[service].astype(str).str.strip() == 'Yes'

        users = data[is_user]
        non_users = data[~is_user]

        user_count = len(users)
        non_user_count = len(non_users)

        churn_rate_with = (users['binary_churn'].mean() * 100) if user_count > 0 else 0.0
        churn_rate_without = (non_users['binary_churn'].mean() * 100) if non_user_count > 0 else 0.0

        abs_diff = churn_rate_without - churn_rate_with
        rel_diff = ((abs_diff / churn_rate_without) * 100) if churn_rate_without > 0 else 0.0

        # Determine stickiness tier
        if abs_diff >= 15.0:
            tier = "High Anchor Service"
        elif abs_diff >= 5.0:
            tier = "Moderate Anchor Service"
        else:
            tier = "Low Anchor Service"

        rows.append({
            'feature': service,
            'user_count': user_count,
            'non_user_count': non_user_count,
            'churn_rate_with': round(churn_rate_with, 2),
            'churn_rate_without': round(churn_rate_without, 2),
            'churn_reduction_abs': round(abs_diff, 2),
            'churn_reduction_pct': round(rel_diff, 2),
            'stickiness_tier': tier
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(by='churn_reduction_abs', ascending=False).reset_index(drop=True)

    return result_df


if __name__ == "__main__":
    # Test feature stickiness
    df_raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    res = calculate_feature_stickiness(df_raw)
    print(res.to_string())
    print("\nDisclaimer:", CAUSATION_DISCLAIMER)
    assert not res.empty, "Result should not be empty"
    print("SUCCESS: analytics/stickiness.py passed.")
