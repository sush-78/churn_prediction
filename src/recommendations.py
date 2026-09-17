"""
Retention Recommendation Engine Module for Customer Churn Prediction Agent.

Rule-based recommendation engine that yields personalized, explainable retention
actions for high-risk and critical-risk customers based on individual account attributes.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

DEFAULT_PRED_PATH = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\customer_risk_predictions.csv"
OUTPUT_HIGH_RISK_RECS = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\high_risk_recommendations.csv"


def get_recommendation_priority(risk_level: str) -> str:
    """Maps customer risk tier to recommendation urgency priority."""
    if risk_level == "Critical Risk":
        return "Urgent"
    elif risk_level == "High Risk":
        return "High"
    elif risk_level == "Medium Risk":
        return "Medium"
    else:
        return "Low"


def generate_recommendations(customer: dict) -> list:
    """
    Generates an ordered list of up to 5 personalized retention actions for a customer.

    Each recommendation dict contains:
    - 'action': Specific prescriptive intervention
    - 'reason': Explanation based on customer attribute
    - 'weight': Numeric weight for sorting relevance
    """
    recs = []
    seen_actions = set()

    def add_rec(action: str, reason: str, weight: int):
        if action not in seen_actions:
            seen_actions.add(action)
            recs.append({
                'action': action,
                'reason': reason,
                'weight': weight
            })

    # Rule 1: Contract Rules
    contract = str(customer.get('Contract', '')).strip()
    if contract == "Month-to-month":
        add_rec(
            "Offer a discounted annual or two-year contract upgrade incentive.",
            "Month-to-month customers have less contractual commitment.",
            100
        )
    elif contract == "One year":
        add_rec(
            "Offer an incentive to renew or upgrade to a two-year contract.",
            "Upgrade contract to extend customer commitment.",
            80
        )

    # Rule 2: Tenure Rules
    try:
        tenure = float(customer.get('tenure', 0))
    except (ValueError, TypeError):
        tenure = 0.0

    if tenure <= 6:
        add_rec(
            "Provide an early-tenure onboarding and engagement offer.",
            "New customers may need additional onboarding and engagement.",
            95
        )
    elif tenure <= 12:
        add_rec(
            "Schedule a proactive customer-success check-in.",
            "Ensure first-year satisfaction and service adoption.",
            75
        )

    # Rule 3: Monthly Charges Rules
    try:
        monthly_charges = float(customer.get('MonthlyCharges', 0))
    except (ValueError, TypeError):
        monthly_charges = 0.0

    if monthly_charges >= 80:
        add_rec(
            "Offer a personalized plan review or pricing/value option.",
            "High recurring charges may increase price sensitivity.",
            90
        )
    elif monthly_charges >= 60:
        add_rec(
            "Review the customer's current plan and available lower-cost alternatives.",
            "Moderate recurring cost review.",
            70
        )

    # Rule 4: Payment Method
    payment = str(customer.get('PaymentMethod', '')).strip()
    if payment == "Electronic check":
        add_rec(
            "Encourage migration to automatic bank transfer or credit card payment with a small incentive.",
            "Electronic check has high churn correlation compared to auto-pay.",
            85
        )

    # Rule 5: Technical Support
    tech_support = str(customer.get('TechSupport', '')).strip()
    if tech_support == "No":
        add_rec(
            "Offer a complimentary technical support trial or assisted setup.",
            "Providing support may improve the customer's service experience.",
            80
        )

    # Rule 6: Internet Service
    internet = str(customer.get('InternetService', '')).strip()
    if internet == "Fiber optic":
        add_rec(
            "Review fiber plan pricing and available value-added packages.",
            "Fiber optic accounts have higher monthly bills and churn sensitivity.",
            75
        )

    # Rule 7: Online Security
    sec = str(customer.get('OnlineSecurity', '')).strip()
    if sec == "No":
        add_rec(
            "Offer a trial or discounted online security add-on.",
            "Enhance account safety and value perception.",
            65
        )

    # Rule 8: Online Backup
    backup = str(customer.get('OnlineBackup', '')).strip()
    if backup == "No":
        add_rec(
            "Offer a trial or discounted online backup service.",
            "Protect customer data with cloud storage add-on.",
            60
        )

    # Rule 9: Device Protection
    device = str(customer.get('DeviceProtection', '')).strip()
    if device == "No":
        add_rec(
            "Offer a device protection trial or bundled discount.",
            "Provide hardware damage and replacement peace of mind.",
            55
        )

    # Rule 10: Partner / Dependents
    partner = str(customer.get('Partner', '')).strip()
    dependents = str(customer.get('Dependents', '')).strip()
    if partner == "No" and dependents == "No":
        add_rec(
            "Consider an individual-customer loyalty or engagement offer.",
            "Single-line accounts have lower switching friction.",
            50
        )

    # Sort recommendations by weight descending
    recs.sort(key=lambda x: x['weight'], reverse=True)

    # Limit to maximum 5 recommendations per customer
    return recs[:5]


def get_primary_recommendation(customer: dict):
    """
    Selects the single most relevant primary action and reason for a customer.

    Returns:
        tuple: (primary_action, primary_reason)
    """
    recs = generate_recommendations(customer)
    if recs:
        return recs[0]['action'], recs[0]['reason']
    else:
        return "Standard retention monitoring", "Customer has low risk factors."


def process_dataset_recommendations(filepath: str = DEFAULT_PRED_PATH):
    """
    Loads prediction dataset, generates up to 5 prioritized recommendations per customer,
    computes summary statistics, and exports high_risk_recommendations.csv.

    Returns:
        tuple: (full_df, high_risk_df)
    """
    print(f"Loading predictions from: {filepath} ...")
    df = pd.read_csv(filepath)

    rec_1_list = []
    rec_2_list = []
    rec_3_list = []
    rec_4_list = []
    rec_5_list = []
    priority_list = []

    for _, row in df.iterrows():
        cust_dict = row.to_dict()
        recs = generate_recommendations(cust_dict)
        priority = get_recommendation_priority(str(cust_dict.get('risk_level', 'Low Risk')))

        actions = [r['action'] for r in recs]

        r1 = actions[0] if len(actions) > 0 else "Standard retention monitoring"
        r2 = actions[1] if len(actions) > 1 else ""
        r3 = actions[2] if len(actions) > 2 else ""
        r4 = actions[3] if len(actions) > 3 else ""
        r5 = actions[4] if len(actions) > 4 else ""

        rec_1_list.append(r1)
        rec_2_list.append(r2)
        rec_3_list.append(r3)
        rec_4_list.append(r4)
        rec_5_list.append(r5)
        priority_list.append(priority)

    # Add recommendation columns to dataset
    df['primary_recommendation'] = rec_1_list
    df['recommendation_2'] = rec_2_list
    df['recommendation_3'] = rec_3_list
    df['recommendation_4'] = rec_4_list
    df['recommendation_5'] = rec_5_list
    df['recommendation_priority'] = priority_list

    # Filter High Risk & Critical Risk accounts
    high_risk_df = df[df['risk_level'].isin(['High Risk', 'Critical Risk'])].copy()

    # Map risk sorting rank (Critical Risk=1, High Risk=2)
    risk_rank_map = {'Critical Risk': 1, 'High Risk': 2, 'Medium Risk': 3, 'Low Risk': 4}
    high_risk_df['risk_rank'] = high_risk_df['risk_level'].map(risk_rank_map)

    # Sort by Critical Risk first, then churn_probability descending
    high_risk_df = high_risk_df.sort_values(by=['risk_rank', 'churn_probability'], ascending=[True, False])
    high_risk_df = high_risk_df.drop(columns=['risk_rank'])

    # Specify ordered output columns
    target_columns = [
        'customerID', 'churn_probability', 'churn_prediction', 'risk_level',
        'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'InternetService',
        'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'primary_recommendation',
        'recommendation_2', 'recommendation_3', 'recommendation_4', 'recommendation_5',
        'recommendation_priority'
    ]
    available_target_cols = [c for c in target_columns if c in high_risk_df.columns]
    high_risk_export = high_risk_df[available_target_cols]

    # Export to CSV
    os.makedirs(os.path.dirname(OUTPUT_HIGH_RISK_RECS), exist_ok=True)
    high_risk_export.to_csv(OUTPUT_HIGH_RISK_RECS, index=False)
    print(f"SUCCESS: High risk recommendations exported to:\n  {OUTPUT_HIGH_RISK_RECS}")

    return df, high_risk_export


def print_recommendation_report(full_df: pd.DataFrame, high_risk_df: pd.DataFrame):
    """Prints executive summary statistics of generated retention recommendations."""
    total_cust = len(full_df)
    high_risk_count = len(full_df[full_df['risk_level'] == 'High Risk'])
    crit_risk_count = len(full_df[full_df['risk_level'] == 'Critical Risk'])

    print("\n====================================")
    print("RETENTION RECOMMENDATION SUMMARY")
    print("====================================")
    print(f"Total Customers Processed : {total_cust}")
    print(f"High Risk Customers       : {high_risk_count}")
    print(f"Critical Risk Customers   : {crit_risk_count}")
    print(f"Combined High+Critical    : {len(high_risk_df)}")

    print("\nOverall Most Common Primary Recommendations:")
    top_primary_overall = full_df['primary_recommendation'].value_counts()
    for rec, count in top_primary_overall.items():
        pct = (count / total_cust) * 100
        print(f" - {rec:<65} : {count:>4} ({pct:.2f}%)")

    print("\nCritical Risk Primary Recommendations:")
    crit_df = full_df[full_df['risk_level'] == 'Critical Risk']
    crit_primary = crit_df['primary_recommendation'].value_counts()
    for rec, count in crit_primary.items():
        pct = (count / crit_risk_count) * 100 if crit_risk_count > 0 else 0
        print(f" - {rec:<65} : {count:>4} ({pct:.2f}%)")

    print("\nHigh Risk Primary Recommendations:")
    high_df = full_df[full_df['risk_level'] == 'High Risk']
    high_primary = high_df['primary_recommendation'].value_counts()
    for rec, count in high_primary.items():
        pct = (count / high_risk_count) * 100 if high_risk_count > 0 else 0
        print(f" - {rec:<65} : {count:>4} ({pct:.2f}%)")
    print("====================================")


def test_sample_customers(df: pd.DataFrame):
    """Tests recommendation engine on 5 distinct customer profiles from the dataset."""
    print("\n====================================")
    print("RETENTION RECOMMENDATION TEST")
    print("====================================")

    # Pick 1 Low Risk, 1 Medium Risk, 1 High Risk, 2 Critical Risk
    low_sample = df[df['risk_level'] == 'Low Risk'].iloc[0]
    med_sample = df[df['risk_level'] == 'Medium Risk'].iloc[0]
    high_sample = df[df['risk_level'] == 'High Risk'].iloc[0]
    crit_sample_1 = df[df['risk_level'] == 'Critical Risk'].iloc[0]
    crit_sample_2 = df[df['risk_level'] == 'Critical Risk'].iloc[1]

    test_samples = [low_sample, med_sample, high_sample, crit_sample_1, crit_sample_2]

    for sample in test_samples:
        cust_dict = sample.to_dict()
        cust_id = cust_dict.get('customerID', 'UNKNOWN')
        risk = cust_dict.get('risk_level', 'Low Risk')
        prob = cust_dict.get('churn_probability', 0.0) * 100

        recs = generate_recommendations(cust_dict)
        primary_act, primary_reason = get_primary_recommendation(cust_dict)

        print(f"\nCustomer ID: {cust_id}")
        print(f"Risk: {risk}")
        print(f"Churn Probability: {prob:.2f}%")
        print(f"\nPrimary Recommendation:\n  {primary_act}")
        print(f"Reason:\n  {primary_reason}")

        print("\nOther Recommended Actions:")
        other_recs = recs[1:]
        if other_recs:
            for idx, r in enumerate(other_recs, 1):
                print(f"  {idx}. {r['action']}")
        else:
            print("  None")
        print("-" * 50)


if __name__ == "__main__":
    full_df, high_risk_df = process_dataset_recommendations()
    print_recommendation_report(full_df, high_risk_df)
    test_sample_customers(full_df)
