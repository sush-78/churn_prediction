"""
High-Risk Customer Identification & Prioritization Module.

Loads predictions from data/customer_risk_predictions.csv,
extracts High Risk and Critical Risk accounts, produces executive summary statistics,
generates categorical breakdown insights, and exports CSV files & charts.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_PRED_PATH = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data\customer_risk_predictions.csv"

HIGH_RISK_COLUMNS = [
    'customerID',
    'churn_probability',
    'churn_prediction',
    'risk_level',
    'tenure',
    'Contract',
    'MonthlyCharges',
    'TotalCharges',
    'InternetService',
    'OnlineSecurity',
    'TechSupport',
    'PaymentMethod'
]


def load_predictions(filepath: str = DEFAULT_PRED_PATH) -> pd.DataFrame:
    """Loads prediction CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file not found at {filepath}. Please run src/risk_segmentation.py first.")
    return pd.read_csv(filepath)


def get_high_risk_customers(df: pd.DataFrame = None) -> pd.DataFrame:
    """Returns all customers categorized as High Risk or Critical Risk, sorted by churn probability."""
    if df is None:
        df = load_predictions()

    high_risk_df = df[df['risk_level'].isin(['High Risk', 'Critical Risk'])].copy()
    high_risk_df = high_risk_df.sort_values(by='churn_probability', ascending=False)

    available_cols = [c for c in HIGH_RISK_COLUMNS if c in high_risk_df.columns]
    return high_risk_df[available_cols]


def get_critical_customers(df: pd.DataFrame = None) -> pd.DataFrame:
    """Returns only customers categorized as Critical Risk (probability >= 0.80)."""
    if df is None:
        df = load_predictions()

    crit_df = df[df['risk_level'] == 'Critical Risk'].copy()
    crit_df = crit_df.sort_values(by='churn_probability', ascending=False)

    available_cols = [c for c in HIGH_RISK_COLUMNS if c in crit_df.columns]
    return crit_df[available_cols]


def get_top_risk_customers(df: pd.DataFrame = None, n: int = 20) -> pd.DataFrame:
    """Returns the top N customers sorted by highest churn probability."""
    if df is None:
        df = load_predictions()

    top_df = df.sort_values(by='churn_probability', ascending=False).head(n).copy()
    available_cols = [c for c in HIGH_RISK_COLUMNS if c in top_df.columns]
    return top_df[available_cols]


def generate_high_risk_report(df: pd.DataFrame = None):
    """
    Computes summary metrics, prints terminal report, exports CSV files & PNG charts.
    """
    if df is None:
        df = load_predictions()

    total_customers = len(df)

    low_risk = df[df['risk_level'] == 'Low Risk']
    med_risk = df[df['risk_level'] == 'Medium Risk']
    high_risk = df[df['risk_level'] == 'High Risk']
    crit_risk = df[df['risk_level'] == 'Critical Risk']

    combined_high_risk = df[df['risk_level'].isin(['High Risk', 'Critical Risk'])]

    low_count = len(low_risk)
    med_count = len(med_risk)
    high_count = len(high_risk)
    crit_count = len(crit_risk)
    combined_count = len(combined_high_risk)
    combined_pct = (combined_count / total_customers) * 100 if total_customers > 0 else 0.0

    # Compute Averages safely handling missing values
    avg_prob_high = high_risk['churn_probability'].mean() * 100 if high_count > 0 else 0.0
    avg_prob_crit = crit_risk['churn_probability'].mean() * 100 if crit_count > 0 else 0.0
    avg_charges_high = combined_high_risk['MonthlyCharges'].mean() if combined_count > 0 else 0.0
    avg_tenure_high = combined_high_risk['tenure'].mean() if combined_count > 0 else 0.0

    # Export CSV Files
    high_risk_export = get_high_risk_customers(df)
    crit_risk_export = get_critical_customers(df)
    top_20_export = get_top_risk_customers(df, n=20)

    csv_dir = r"c:\Users\Sushrutha Methuku\OneDrive\Desktop\Customer Churn Prediction Agent\data"
    high_csv_path = os.path.join(csv_dir, "high_risk_customers.csv")
    crit_csv_path = os.path.join(csv_dir, "critical_risk_customers.csv")
    top20_csv_path = os.path.join(csv_dir, "top_20_risk_customers.csv")

    high_risk_export.to_csv(high_csv_path, index=False)
    crit_risk_export.to_csv(crit_csv_path, index=False)
    top_20_export.to_csv(top20_csv_path, index=False)

    # Visualization 1: Risk Distribution Chart
    risk_img_path = os.path.join(csv_dir, "risk_distribution.png")
    plt.figure(figsize=(9, 5))
    categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    counts = [low_count, med_count, high_count, crit_count]
    colors = ['#2A9D8F', '#E9C46A', '#F4A261', '#E63946']

    bars = plt.bar(categories, counts, color=colors)
    plt.ylabel("Number of Customers")
    plt.title("Customer Risk Segmentation Distribution", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total_customers) * 100
        plt.text(bar.get_x() + bar.get_width() / 2., height + 50, f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(risk_img_path, dpi=300)
    plt.close()

    # Visualization 2: Top 20 Risk Customers Chart
    top20_img_path = os.path.join(csv_dir, "top_20_risk_customers.png")
    plt.figure(figsize=(12, 6))
    top20_sorted = top_20_export.sort_values(by='churn_probability', ascending=True)

    y_labels = top20_sorted['customerID']
    x_probs = top20_sorted['churn_probability'] * 100

    plt.barh(y_labels, x_probs, color='#E63946')
    plt.xlabel("Churn Probability (%)")
    plt.ylabel("Customer ID")
    plt.title("Top 20 Highest-Risk Customers", fontsize=14, fontweight='bold')
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for idx, prob in enumerate(x_probs):
        plt.text(prob + 0.5, idx, f"{prob:.1f}%", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(top20_img_path, dpi=300)
    plt.close()

    # Categorical Breakdowns
    contract_breakdown = combined_high_risk['Contract'].value_counts()
    internet_breakdown = combined_high_risk['InternetService'].value_counts()
    payment_breakdown = combined_high_risk['PaymentMethod'].value_counts()

    # Terminal Report
    print("====================================")
    print("CUSTOMER CHURN RISK REPORT")
    print("====================================")
    print(f"\nTotal Customers: {total_customers}")
    print(f"\nLow Risk      : {low_count}")
    print(f"Medium Risk   : {med_count}")
    print(f"High Risk     : {high_count}")
    print(f"Critical Risk : {crit_count}")
    print(f"\nHigh + Critical Risk Customers: {combined_count}")
    print(f"Percentage                    : {combined_pct:.2f}%")

    print("\n------------------------------------")
    print("TOP 10 HIGHEST-RISK CUSTOMERS")
    print("------------------------------------")
    print(f"{'Rank':<4} | {'Customer ID':<12} | {'Churn Prob':<10} | {'Risk':<13} | {'Contract':<15} | {'Tenure':<6} | {'Monthly Charges'}")
    print("-" * 85)

    top_10 = top_20_export.head(10)
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        prob_str = f"{row['churn_probability']*100:.2f}%"
        charge_str = f"${row['MonthlyCharges']:.2f}"
        print(f"{rank:<4} | {row['customerID']:<12} | {prob_str:<10} | {row['risk_level']:<13} | {row['Contract']:<15} | {row['tenure']:<6} | {charge_str}")

    print("\n------------------------------------")
    print("HIGH-RISK CUSTOMER INSIGHTS")
    print("------------------------------------")
    print("\nContract breakdown:")
    for k, v in contract_breakdown.items():
        pct = (v / combined_count) * 100
        print(f" - {k:<15} : {v:>4} ({pct:.2f}%)")

    print("\nInternet Service breakdown:")
    for k, v in internet_breakdown.items():
        pct = (v / combined_count) * 100
        print(f" - {k:<15} : {v:>4} ({pct:.2f}%)")

    print("\nPayment Method breakdown:")
    for k, v in payment_breakdown.items():
        pct = (v / combined_count) * 100
        print(f" - {k:<28} : {v:>4} ({pct:.2f}%)")

    print("\nSummary Statistics:")
    print(f" - Avg Churn Probability (High Risk)    : {avg_prob_high:.2f}%")
    print(f" - Avg Churn Probability (Critical Risk): {avg_prob_crit:.2f}%")
    print(f" - Avg Monthly Charges (High+Critical)  : ${avg_charges_high:.2f}")
    print(f" - Avg Tenure (High+Critical)          : {avg_tenure_high:.1f} months")
    print("====================================")

    print(f"\nSUCCESS Artifacts Created:")
    print(f" 1. {high_csv_path}")
    print(f" 2. {crit_csv_path}")
    print(f" 3. {top20_csv_path}")
    print(f" 4. {risk_img_path}")
    print(f" 5. {top20_img_path}")


if __name__ == "__main__":
    generate_high_risk_report()
