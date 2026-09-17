"""
Silent Drop-Off Detection Module (Analytics)

Calculates a Proxy Engagement Score based on value-added service adoption.
Categorizes customers into Healthy Engagement, Moderate Engagement, and Silent Drop-Off Risk.
"""

import pandas as pd
import numpy as np

VALUE_ADDED_SERVICES = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]


def engagement_score(customer) -> int:
    """
    Calculates Proxy Engagement Score (0 - 100) for a single customer record.

    Base score: 100.
    Subtracts 10 points for every value-added service whose value is 'No' or inactive.

    Args:
        customer (dict or pd.Series): Customer record.

    Returns:
        int: Engagement score between 0 and 100.
    """
    score = 100
    for service in VALUE_ADDED_SERVICES:
        val = str(customer.get(service, '')).strip()
        # If customer does not have the active service ('No' or 'No internet service' or missing)
        if val in ['No', 'No internet service', '', 'nan', 'None']:
            score -= 10

    return max(0, score)


def classify_engagement(score: float) -> str:
    """
    Classifies Engagement Score into categories:
    - 80 - 100: Healthy Engagement
    - 50 - 79 : Moderate Engagement
    - 0  - 49 : Silent Drop-Off Risk
    """
    if score >= 80:
        return "Healthy Engagement"
    elif score >= 50:
        return "Moderate Engagement"
    else:
        return "Silent Drop-Off Risk"


def calculate_engagement_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends 'engagement_score', 'engagement_category', and 'engagement_warning'
    columns to the customer DataFrame.

    Args:
        df (pd.DataFrame): Customer DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engagement columns added.
    """
    result_df = df.copy()

    scores = []
    categories = []
    warnings = []

    for _, row in result_df.iterrows():
        sc = engagement_score(row.to_dict())
        cat = classify_engagement(sc)

        if cat == "Silent Drop-Off Risk":
            warn = "Customer is subscribed but uses very few value-added services. Shows low engagement and may be at elevated churn risk."
        elif cat == "Moderate Engagement":
            warn = "Customer has partial service adoption. Opportunity to cross-sell value-added features."
        else:
            warn = "Customer shows healthy service adoption across value-added products."

        scores.append(sc)
        categories.append(cat)
        warnings.append(warn)

    result_df['engagement_score'] = scores
    result_df['engagement_category'] = categories
    result_df['engagement_warning'] = warnings

    return result_df


if __name__ == "__main__":
    # Test engagement scoring module
    sample_cust = {
        'customerID': 'TEST-01',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No'
    }

    sc = engagement_score(sample_cust)
    cat = classify_engagement(sc)
    print(f"Sample Customer Engagement Score: {sc} | Category: {cat}")
    assert sc == 40, f"Expected 40, got {sc}"
    assert cat == "Silent Drop-Off Risk", f"Expected Silent Drop-Off Risk, got {cat}"
    print("SUCCESS: engagement.py test passed.")
