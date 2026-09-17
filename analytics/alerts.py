"""
Smart Internal Alerts Module (Analytics)

Generates operational risk alerts for proactive customer success actions based on churn risk,
engagement score, contract type, and service coverage gaps.
"""

import pandas as pd
import numpy as np

try:
    from analytics.engagement import calculate_engagement_scores
except ImportError:
    from engagement import calculate_engagement_scores


def generate_smart_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates customer data and generates structured operational alerts.

    Args:
        df (pd.DataFrame): Customer DataFrame containing raw features and 'churn_probability'.

    Returns:
        pd.DataFrame: Table of generated alerts with columns:
            - customerID
            - severity (CRITICAL, WARNING, INFO)
            - alert_type
            - alert_title
            - churn_probability
            - engagement_score
            - MonthlyCharges
            - Contract
            - action_recommended
    """
    data = df.copy()

    # Ensure engagement scores exist
    if 'engagement_score' not in data.columns or 'engagement_category' not in data.columns:
        data = calculate_engagement_scores(data)

    # Ensure churn probability exists
    if 'churn_probability' not in data.columns:
        if 'churn_prediction' in data.columns:
            data['churn_probability'] = data['churn_prediction'].astype(float)
        else:
            data['churn_probability'] = 0.5

    alerts = []

    for _, row in data.iterrows():
        cid = str(row.get('customerID', 'Unknown'))
        prob = float(row.get('churn_probability', 0.0))
        eng_score = float(row.get('engagement_score', 100))
        eng_cat = str(row.get('engagement_category', ''))
        contract = str(row.get('Contract', ''))
        monthly = float(row.get('MonthlyCharges', 0.0)) if pd.notnull(row.get('MonthlyCharges')) else 0.0
        tenure = float(row.get('tenure', 0.0)) if pd.notnull(row.get('tenure')) else 0.0
        internet = str(row.get('InternetService', ''))
        sec = str(row.get('OnlineSecurity', ''))
        tech = str(row.get('TechSupport', ''))

        # Rule 1: Silent Drop-Off Risk (CRITICAL / WARNING)
        if eng_cat == "Silent Drop-Off Risk" and prob >= 0.60:
            alerts.append({
                'customerID': cid,
                'severity': 'CRITICAL' if prob >= 0.75 else 'WARNING',
                'alert_type': 'Silent Drop-Off Risk',
                'alert_title': f'Low Engagement with High Churn Risk ({prob*100:.1f}%)',
                'churn_probability': round(prob, 4),
                'engagement_score': eng_score,
                'MonthlyCharges': monthly,
                'Contract': contract,
                'action_recommended': 'Proactive CS outreach: Offer complementary onboarding & security feature bundle.'
            })

        # Rule 2: Unprotected High-Value Account (WARNING)
        elif internet == 'Fiber optic' and (sec in ['No', ''] and tech in ['No', '']) and monthly > 80:
            alerts.append({
                'customerID': cid,
                'severity': 'CRITICAL' if prob >= 0.70 else 'WARNING',
                'alert_type': 'Unprotected Fiber Account',
                'alert_title': f'High Monthly Spend (${monthly:.2f}) without Security & Tech Support',
                'churn_probability': round(prob, 4),
                'engagement_score': eng_score,
                'MonthlyCharges': monthly,
                'Contract': contract,
                'action_recommended': 'Recommend TechSupport + OnlineSecurity bundle discount to secure high-value fiber line.'
            })

        # Rule 3: Month-to-Month Contract Vulnerability (CRITICAL / WARNING)
        elif contract == 'Month-to-month' and prob >= 0.65:
            alerts.append({
                'customerID': cid,
                'severity': 'CRITICAL' if prob >= 0.80 else 'WARNING',
                'alert_type': 'Contract Expiration Risk',
                'alert_title': f'Month-to-Month High Churn Probability ({prob*100:.1f}%)',
                'churn_probability': round(prob, 4),
                'engagement_score': eng_score,
                'MonthlyCharges': monthly,
                'Contract': contract,
                'action_recommended': 'Propose 1-year contract migration with 15% annual discount incentive.'
            })

        # Rule 4: High Spend At-Risk (INFO / WARNING)
        elif monthly >= 90.0 and prob >= 0.50:
            alerts.append({
                'customerID': cid,
                'severity': 'WARNING',
                'alert_type': 'High Spend Vulnerability',
                'alert_title': f'Premium Tier Customer (${monthly:.2f}/mo) at Moderate Risk',
                'churn_probability': round(prob, 4),
                'engagement_score': eng_score,
                'MonthlyCharges': monthly,
                'Contract': contract,
                'action_recommended': 'Assign VIP Account Manager for proactive satisfaction call.'
            })

        # Rule 5: Long-Term Customer Engagement Drop (INFO)
        elif tenure >= 36 and eng_score <= 50 and prob >= 0.40:
            alerts.append({
                'customerID': cid,
                'severity': 'INFO',
                'alert_type': 'Tenured Engagement Loss',
                'alert_title': f'Tenured Customer ({int(tenure)} mos) exhibiting low feature usage',
                'churn_probability': round(prob, 4),
                'engagement_score': eng_score,
                'MonthlyCharges': monthly,
                'Contract': contract,
                'action_recommended': 'Send loyalty renewal reward and complimentary feature trial.'
            })

    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        # Sort by severity priority (CRITICAL -> WARNING -> INFO) and then churn_probability
        severity_map = {'CRITICAL': 1, 'WARNING': 2, 'INFO': 3}
        alerts_df['sev_rank'] = alerts_df['severity'].map(severity_map)
        alerts_df = alerts_df.sort_values(by=['sev_rank', 'churn_probability'], ascending=[True, False]).drop(columns=['sev_rank']).reset_index(drop=True)

    return alerts_df


def get_alerts_summary(alerts_df: pd.DataFrame) -> dict:
    """
    Computes alert count summary.
    """
    if alerts_df.empty:
        return {'total': 0, 'critical': 0, 'warning': 0, 'info': 0}

    counts = alerts_df['severity'].value_counts().to_dict()
    return {
        'total': len(alerts_df),
        'critical': counts.get('CRITICAL', 0),
        'warning': counts.get('WARNING', 0),
        'info': counts.get('INFO', 0)
    }


if __name__ == "__main__":
    preds = pd.read_csv("data/customer_risk_predictions.csv")
    alerts = generate_smart_alerts(preds)
    summary = get_alerts_summary(alerts)
    print("Generated Alerts Summary:", summary)
    print("\nTop 5 Alerts:\n", alerts.head(5).to_string())
    print("SUCCESS: analytics/alerts.py passed.")
