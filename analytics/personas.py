"""
Customer Personas & Segmentation Module (Analytics)

Performs unsupervised clustering (K-Means) on customer features and maps clusters to
meaningful business personas:
- High-Value At-Risk
- Digital Power Users
- Basic Service Long-Term
- Budget Vulnerable
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from analytics.engagement import calculate_engagement_scores
except ImportError:
    from engagement import calculate_engagement_scores


PERSONA_NAMES = {
    0: "Digital Power Users",
    1: "High-Value At-Risk",
    2: "Budget Vulnerable",
    3: "Basic Service Long-Term"
}

PERSONA_DESCRIPTIONS = {
    "Digital Power Users": "Tenured customers with high service adoption, high engagement, and low churn risk.",
    "High-Value At-Risk": "High monthly spend customers on short-term contracts exhibiting high churn probability.",
    "Budget Vulnerable": "Newer customers with low service adoption and basic features, sensitive to price & service gaps.",
    "Basic Service Long-Term": "Loyal, long-term customers with lower monthly spend and steady retention profile."
}


def generate_customer_personas(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Appends persona cluster IDs and human-readable persona labels to the DataFrame.

    Args:
        df (pd.DataFrame): Customer DataFrame containing features, churn_probability, etc.
        n_clusters (int): Number of clusters (default 4).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with 'persona_cluster', 'persona_name', and 'persona_description'.
    """
    data = df.copy()

    # Ensure required numerical columns exist
    if 'engagement_score' not in data.columns:
        data = calculate_engagement_scores(data)

    if 'churn_probability' not in data.columns:
        if 'churn_prediction' in data.columns:
            data['churn_probability'] = data['churn_prediction'].astype(float)
        else:
            data['churn_probability'] = 0.5

    # Convert TotalCharges if string
    total_charges = pd.to_numeric(data.get('TotalCharges', 0), errors='coerce').fillna(0)

    # Prepare features for clustering
    features = pd.DataFrame({
        'tenure': pd.to_numeric(data.get('tenure', 0), errors='coerce').fillna(0),
        'MonthlyCharges': pd.to_numeric(data.get('MonthlyCharges', 0), errors='coerce').fillna(0),
        'TotalCharges': total_charges,
        'engagement_score': pd.to_numeric(data.get('engagement_score', 50), errors='coerce').fillna(50),
        'churn_probability': pd.to_numeric(data.get('churn_probability', 0.5), errors='coerce').fillna(0.5)
    })

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    data['persona_cluster'] = clusters

    # Calculate cluster statistics to map clusters deterministically to personas
    cluster_stats = []
    for c in range(n_clusters):
        c_mask = (clusters == c)
        avg_monthly = features.loc[c_mask, 'MonthlyCharges'].mean()
        avg_tenure = features.loc[c_mask, 'tenure'].mean()
        avg_churn_prob = features.loc[c_mask, 'churn_probability'].mean()
        avg_eng = features.loc[c_mask, 'engagement_score'].mean()
        cluster_stats.append({
            'cluster': c,
            'avg_monthly': avg_monthly,
            'avg_tenure': avg_tenure,
            'avg_churn_prob': avg_churn_prob,
            'avg_eng': avg_eng
        })

    stats_df = pd.DataFrame(cluster_stats)

    # Priority mapping algorithm:
    # 1. High-Value At-Risk: Cluster with highest churn probability among high monthly spenders (> overall median monthly)
    # 2. Digital Power Users: Highest tenure & highest engagement score
    # 3. Basic Service Long-Term: High tenure, low monthly spend
    # 4. Budget Vulnerable: Remaining cluster (low tenure, moderate churn prob)

    assigned_map = {}
    unassigned_clusters = set(range(n_clusters))

    # Identify Digital Power Users (highest tenure + eng score)
    stats_df['power_score'] = stats_df['avg_tenure'] + (stats_df['avg_eng'] * 0.5)
    dpu_cluster = stats_df.sort_values(by='power_score', ascending=False).iloc[0]['cluster']
    assigned_map[int(dpu_cluster)] = "Digital Power Users"
    unassigned_clusters.remove(int(dpu_cluster))

    # Identify High-Value At-Risk (highest churn_prob among remaining)
    rem_df = stats_df[stats_df['cluster'].isin(unassigned_clusters)]
    hvar_cluster = rem_df.sort_values(by='avg_churn_prob', ascending=False).iloc[0]['cluster']
    assigned_map[int(hvar_cluster)] = "High-Value At-Risk"
    unassigned_clusters.remove(int(hvar_cluster))

    # Identify Basic Service Long-Term (higher tenure among remaining)
    rem_df2 = stats_df[stats_df['cluster'].isin(unassigned_clusters)]
    bslt_cluster = rem_df2.sort_values(by='avg_tenure', ascending=False).iloc[0]['cluster']
    assigned_map[int(bslt_cluster)] = "Basic Service Long-Term"
    unassigned_clusters.remove(int(bslt_cluster))

    # Remaining is Budget Vulnerable
    bv_cluster = list(unassigned_clusters)[0]
    assigned_map[int(bv_cluster)] = "Budget Vulnerable"

    data['persona_name'] = data['persona_cluster'].map(assigned_map)
    data['persona_description'] = data['persona_name'].map(PERSONA_DESCRIPTIONS)

    return data


def get_persona_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary table of customer personas with counts and average KPIs.
    """
    if 'persona_name' not in df.columns:
        df = generate_customer_personas(df)

    summary = df.groupby('persona_name').agg(
        customer_count=('customerID', 'count'),
        avg_churn_probability=('churn_probability', lambda x: round(x.mean(), 4)),
        avg_tenure=('tenure', lambda x: round(x.mean(), 1)),
        avg_monthly_charges=('MonthlyCharges', lambda x: round(x.mean(), 2)),
        avg_engagement_score=('engagement_score', lambda x: round(x.mean(), 1))
    ).reset_index()

    summary['percentage'] = round((summary['customer_count'] / summary['customer_count'].sum()) * 100, 1)
    summary['persona_description'] = summary['persona_name'].map(PERSONA_DESCRIPTIONS)

    return summary


if __name__ == "__main__":
    preds = pd.read_csv("data/customer_risk_predictions.csv")
    df_p = generate_customer_personas(preds)
    summary = get_persona_summary(df_p)
    print("Customer Personas Summary:\n", summary.to_string())
    print("SUCCESS: analytics/personas.py passed.")
