"""
Customer Churn Prediction Agent - Streamlit Interactive Dashboard
AI-powered churn risk prediction, explainability, retention recommendations, and advanced retention analytics.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import core intelligence modules
from src.explainability import explain_customer, format_shap_explanation
from src.recommendations import generate_recommendations, get_primary_recommendation

# Import advanced analytics modules
from analytics.engagement import calculate_engagement_scores
from analytics.stickiness import calculate_feature_stickiness, CAUSATION_DISCLAIMER
from analytics.alerts import generate_smart_alerts, get_alerts_summary
from analytics.personas import generate_customer_personas, get_persona_summary

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #1E40AF;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_csv_data(filepath: str) -> pd.DataFrame:
    """Loads CSV dataset with caching."""
    if not os.path.exists(filepath):
        st.error(f"Required data file missing: `{filepath}`. Please ensure pipeline artifacts are generated.")
        st.stop()
    return pd.read_csv(filepath)


@st.cache_resource
def load_churn_model(model_path: str = r"models/churn_model.joblib"):
    """Loads trained Random Forest model and preprocessor package with caching."""
    if not os.path.exists(model_path):
        st.error(f"Trained model artifact missing: `{model_path}`. Please run `src/train_model.py` first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def get_enriched_predictions():
    """Generates enriched customer dataset with engagement scores and persona clusters."""
    base_df = load_csv_data(PREDICTIONS_CSV)
    eng_df = calculate_engagement_scores(base_df)
    enriched_df = generate_customer_personas(eng_df)
    return enriched_df


# File paths
PREDICTIONS_CSV = r"data/customer_risk_predictions.csv"
HIGH_RISK_RECS_CSV = r"data/high_risk_recommendations.csv"
TOP_20_CSV = r"data/top_20_risk_customers.csv"
FEATURE_IMP_CSV = r"data/feature_importance.csv"
MODEL_JOBLIB = r"models/churn_model.joblib"

# Load core & enriched artifacts
predictions_df = get_enriched_predictions()
model_pkg = load_churn_model(MODEL_JOBLIB)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## Customer Churn Agent")

navigation_option = st.sidebar.radio(
    "Navigation",
    options=[
        "Executive Overview",
        "Customer Risk Explorer",
        "Customer 360 & AI Explanation",
        "High-Risk Retention Actions",
        "Engagement & Drop-Off Risk",
        "Feature Stickiness Matrix",
        "Smart Operational Alerts",
        "Customer Personas & Segments",
        "Model Insights"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Metadata")
st.sidebar.info(
    "**Model:** Random Forest Classifier\n\n"
    "**Explainability:** SHAP (TreeExplainer)\n\n"
    "**Analytics:** Engagement, Stickiness, Alerts & Clustering\n\n"
    "**Dataset:** Telco Customer Churn"
)

# Render Dashboard Header
st.markdown('<div class="main-header">Customer Churn Prediction Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered churn risk prediction, explainability and retention recommendations</div>', unsafe_allow_html=True)


# =============================================================================
# 1. EXECUTIVE OVERVIEW
# =============================================================================
if navigation_option == "Executive Overview":
    st.markdown("### Executive Overview")

    total_customers = len(predictions_df)
    actual_churn_rate = (predictions_df['Churn'].map({'Yes': 1, 'No': 0}).sum() / total_customers) * 100 if 'Churn' in predictions_df.columns else 26.54
    high_crit_count = len(predictions_df[predictions_df['risk_level'].isin(['High Risk', 'Critical Risk'])])
    high_crit_pct = (high_crit_count / total_customers) * 100
    avg_pred_prob = predictions_df['churn_probability'].mean() * 100

    # Analytics metrics
    avg_engagement = predictions_df['engagement_score'].mean()
    silent_drop_count = len(predictions_df[predictions_df['engagement_category'] == 'Silent Drop-Off Risk'])
    silent_drop_pct = (silent_drop_count / total_customers) * 100

    alerts_df = generate_smart_alerts(predictions_df)
    alert_summary = get_alerts_summary(alerts_df)

    # KPI Row 1
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Customers</div><div class="kpi-value">{total_customers:,}</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Actual Churn Rate</div><div class="kpi-value">{actual_churn_rate:.2f}%</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">High + Critical Risk</div><div class="kpi-value">{high_crit_count:,} ({high_crit_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Churn Probability</div><div class="kpi-value">{avg_pred_prob:.2f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Row 2 (Advanced Retention Intelligence)
    akpi_col1, akpi_col2, akpi_col3, akpi_col4 = st.columns(4)
    with akpi_col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Engagement Score</div><div class="kpi-value">{avg_engagement:.1f} / 100</div></div>', unsafe_allow_html=True)
    with akpi_col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Silent Drop-Off Risk</div><div class="kpi-value">{silent_drop_count:,} ({silent_drop_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    with akpi_col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Critical Operational Alerts</div><div class="kpi-value">{alert_summary["critical"]:,}</div></div>', unsafe_allow_html=True)
    with akpi_col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Customer Personas</div><div class="kpi-value">4 Clusters</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Risk Level Distribution")
        risk_counts = predictions_df['risk_level'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']).fillna(0)

        fig_donut = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.45,
            color=risk_counts.index,
            color_discrete_map={
                'Low Risk': '#2A9D8F',
                'Medium Risk': '#E9C46A',
                'High Risk': '#F4A261',
                'Critical Risk': '#E63946'
            }
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.markdown("#### Engagement Category Breakdown")
        eng_counts = predictions_df['engagement_category'].value_counts().fillna(0)

        fig_eng_pie = px.pie(
            values=eng_counts.values,
            names=eng_counts.index,
            hole=0.45,
            color=eng_counts.index,
            color_discrete_map={
                'Healthy Engagement': '#2A9D8F',
                'Moderate Engagement': '#E9C46A',
                'Silent Drop-Off Risk': '#E63946'
            }
        )
        fig_eng_pie.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_eng_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_eng_pie, use_container_width=True)

    # High-Risk Customer Profile Section
    st.markdown("---")
    st.markdown("### Executive Retention Insights")

    high_crit_df = predictions_df[predictions_df['risk_level'].isin(['High Risk', 'Critical Risk'])]
    avg_m_charges = high_crit_df['MonthlyCharges'].mean()
    avg_tenure = high_crit_df['tenure'].mean()
    m2m_pct = (len(high_crit_df[high_crit_df['Contract'] == 'Month-to-month']) / len(high_crit_df)) * 100
    fiber_pct = (len(high_crit_df[high_crit_df['InternetService'] == 'Fiber optic']) / len(high_crit_df)) * 100
    echeck_pct = (len(high_crit_df[high_crit_df['PaymentMethod'] == 'Electronic check']) / len(high_crit_df)) * 100

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.metric("Avg Monthly Charges", f"${avg_m_charges:.2f}")
    with p_col2:
        st.metric("Avg Tenure", f"{avg_tenure:.1f} mos")
    with p_col3:
        st.metric("Month-to-Month Contract", f"{m2m_pct:.1f}%")
    with p_col4:
        st.metric("Fiber Optic Internet", f"{fiber_pct:.1f}%")
    with p_col5:
        st.metric("Electronic Check Payment", f"{echeck_pct:.1f}%")

    st.info(
        "**Executive Summary & Strategic Priorities:**\n\n"
        f"1. **Churn Concentration:** Churn is heavily concentrated among accounts with **Month-to-Month contracts ({m2m_pct:.1f}%)**, "
        f"**Fiber Optic internet plans ({fiber_pct:.1f}%)**, and **Electronic Check payments ({echeck_pct:.1f}%)**.\n"
        f"2. **Silent Drop-Off Risk:** **{silent_drop_count:,} accounts ({silent_drop_pct:.1f}%)** exhibit low service adoption (<50 score). "
        "These customers are subscribed but inactive across security & tech support value-added features.\n"
        f"3. **Operational Alerts:** The agent generated **{alert_summary['critical']:,} Critical Alerts** requiring immediate outreach by Customer Success managers.\n"
        "4. **Feature Stickiness:** Observational analysis shows **Online Security** and **Tech Support** provide the highest churn reduction anchor (>16% absolute lower churn)."
    )


# =============================================================================
# 2. CUSTOMER RISK EXPLORER
# =============================================================================
elif navigation_option == "Customer Risk Explorer":
    st.markdown("### Customer Risk Explorer")

    # Filter Sidebar/Top Section
    st.markdown("#### Filter Customer Database")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        selected_risk = st.multiselect("Risk Level", options=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'], default=['High Risk', 'Critical Risk'])
    with f_col2:
        selected_contract = st.multiselect("Contract", options=sorted(predictions_df['Contract'].dropna().unique().tolist()))
    with f_col3:
        selected_internet = st.multiselect("Internet Service", options=sorted(predictions_df['InternetService'].dropna().unique().tolist()))
    with f_col4:
        selected_payment = st.multiselect("Payment Method", options=sorted(predictions_df['PaymentMethod'].dropna().unique().tolist()))

    s_col1, s_col2 = st.columns([2, 1])
    with s_col1:
        search_id = st.text_input("Search by Customer ID", "").strip()
    with s_col2:
        min_prob = st.slider("Min Churn Probability (%)", 0, 100, 0) / 100.0

    # Apply Filters
    filtered_df = predictions_df.copy()

    if selected_risk:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(selected_risk)]
    if selected_contract:
        filtered_df = filtered_df[filtered_df['Contract'].isin(selected_contract)]
    if selected_internet:
        filtered_df = filtered_df[filtered_df['InternetService'].isin(selected_internet)]
    if selected_payment:
        filtered_df = filtered_df[filtered_df['PaymentMethod'].isin(selected_payment)]
    if min_prob > 0:
        filtered_df = filtered_df[filtered_df['churn_probability'] >= min_prob]
    if search_id:
        filtered_df = filtered_df[filtered_df['customerID'].astype(str).str.contains(search_id, case=False, na=False)]

    filtered_df = filtered_df.sort_values(by='churn_probability', ascending=False)

    st.markdown(f"**Showing {len(filtered_df):,} matching customer records**")

    # Format probability as percentage for display
    display_df = filtered_df.copy()
    display_df['churn_probability'] = (display_df['churn_probability'] * 100).map('{:.2f}%'.format)

    display_cols = ['customerID', 'churn_probability', 'churn_prediction', 'risk_level', 'engagement_score', 'engagement_category', 'persona_name', 'tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'PaymentMethod']
    available_disp_cols = [c for c in display_cols if c in display_df.columns]

    st.dataframe(display_df[available_disp_cols], use_container_width=True, height=400)

    # Customer 360 Navigation Action
    st.markdown("---")
    st.markdown("#### Select Customer for Deep Dive")

    if not filtered_df.empty:
        cust_options = filtered_df['customerID'].tolist()
        chosen_cust = st.selectbox("Choose a customer to view in Customer 360:", options=cust_options)

        if st.button("View Customer 360", type="primary"):
            st.session_state['selected_customer_id'] = chosen_cust
            st.success(f"Customer `{chosen_cust}` selected! Switch to the **Customer 360 & AI Explanation** tab in the sidebar.")
    else:
        st.warning("No customers match the current filter selection.")


# =============================================================================
# 3. CUSTOMER 360 & AI EXPLANATION
# =============================================================================
elif navigation_option == "Customer 360 & AI Explanation":
    st.markdown("### Customer 360 & AI Explanation")

    # Retrieve selected customer or default
    default_id = st.session_state.get('selected_customer_id', '3668-QPYBK')

    all_ids = predictions_df['customerID'].tolist()
    if default_id not in all_ids:
        default_id = all_ids[0]

    select_index = all_ids.index(default_id)
    selected_id = st.selectbox("Select Customer ID:", options=all_ids, index=select_index)

    cust_row = predictions_df[predictions_df['customerID'] == selected_id].iloc[0]
    cust_dict = cust_row.to_dict()

    prob = float(cust_row['churn_probability'])
    prob_pct = prob * 100
    risk = str(cust_row['risk_level'])
    eng_score = float(cust_row.get('engagement_score', 50))
    eng_cat = str(cust_row.get('engagement_category', 'Moderate Engagement'))
    persona_name = str(cust_row.get('persona_name', 'Customer Segment'))

    # Customer Banner Row
    b_col1, b_col2, b_col3 = st.columns([1.5, 1, 1.5])

    with b_col1:
        st.markdown(f"### Customer ID: `{selected_id}`")
        if risk == "Critical Risk":
            st.markdown('<span class="badge-critical">CRITICAL RISK</span>', unsafe_allow_html=True)
        elif risk == "High Risk":
            st.markdown('<span class="badge-high">HIGH RISK</span>', unsafe_allow_html=True)
        elif risk == "Medium Risk":
            st.markdown('<span class="badge-medium">MEDIUM RISK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-low">LOW RISK</span>', unsafe_allow_html=True)

        st.markdown(f"**Persona:** `{persona_name}`")
        st.markdown(f"**Engagement Category:** `{eng_cat}`")

    with b_col2:
        st.metric("Churn Probability", f"{prob_pct:.2f}%", delta="Prediction: " + str(cust_row['churn_prediction']))
        st.metric("Engagement Score", f"{eng_score:.0f} / 100")

    with b_col3:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#1E293B'},
                'steps': [
                    {'range': [0, 30], 'color': '#D1FAE5'},
                    {'range': [30, 60], 'color': '#FEF3C7'},
                    {'range': [60, 80], 'color': '#FFEDD5'},
                    {'range': [80, 100], 'color': '#FEE2E2'}
                ]
            }
        ))
        fig_gauge.update_layout(height=180, margin=dict(t=10, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Customer Attributes Table
    st.markdown("#### Account & Service Profile")
    a_col1, a_col2, a_col3, a_col4 = st.columns(4)

    with a_col1:
        st.markdown(f"**Tenure:** {cust_row.get('tenure')} months")
        st.markdown(f"**Contract:** {cust_row.get('Contract')}")
        st.markdown(f"**Payment Method:** {cust_row.get('PaymentMethod')}")

    with a_col2:
        st.markdown(f"**Monthly Charges:** ${cust_row.get('MonthlyCharges')}")
        st.markdown(f"**Total Charges:** ${cust_row.get('TotalCharges')}")
        st.markdown(f"**Internet Service:** {cust_row.get('InternetService')}")

    with a_col3:
        st.markdown(f"**Online Security:** {cust_row.get('OnlineSecurity')}")
        st.markdown(f"**Online Backup:** {cust_row.get('OnlineBackup')}")
        st.markdown(f"**Tech Support:** {cust_row.get('TechSupport')}")

    with a_col4:
        st.markdown(f"**Device Protection:** {cust_row.get('DeviceProtection')}")
        st.markdown(f"**Partner:** {cust_row.get('Partner')}")
        st.markdown(f"**Dependents:** {cust_row.get('Dependents')}")

    st.markdown("---")

    # SHAP Explainability Section
    st.markdown("### Why is this customer at risk? (SHAP AI Explainability)")

    try:
        exp_result = explain_customer(cust_dict, model_package=model_pkg)
        pos_factors = exp_result['top_positive_factors']
        neg_factors = exp_result['top_negative_factors']

        shap_col1, shap_col2 = st.columns(2)

        with shap_col1:
            st.markdown("#### 🚨 Factors Increasing Churn Risk (+ SHAP)")
            for idx, (feat, val, desc) in enumerate(pos_factors, 1):
                st.markdown(f"**{idx}. {desc}**")
                st.caption(f"Feature: `{feat}` | Impact: `+{val:.4f}`")

        with shap_col2:
            st.markdown("#### 🛡️ Factors Reducing Churn Risk (- SHAP)")
            for idx, (feat, val, desc) in enumerate(neg_factors, 1):
                st.markdown(f"**{idx}. {desc}**")
                st.caption(f"Feature: `{feat}` | Impact: `{val:.4f}`")

        # SHAP Horizontal Bar Plot
        combined_factors = sorted([(x[2], x[1]) for x in (pos_factors + neg_factors)], key=lambda x: x[1])
        plot_features = [x[0] for x in combined_factors]
        plot_values = [x[1] for x in combined_factors]
        plot_colors = ['#E63946' if v > 0 else '#2A9D8F' for v in plot_values]

        fig_shap = px.bar(
            x=plot_values,
            y=plot_features,
            orientation='h',
            labels={'x': 'SHAP Contribution Value', 'y': 'Feature Description'},
            title=f"SHAP Feature Contribution Chart for Customer {selected_id}"
        )
        fig_shap.update_traces(marker_color=plot_colors)
        fig_shap.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_shap, use_container_width=True)

        st.caption("*Note: Positive SHAP contribution pushes prediction toward churn; negative contribution pushes prediction away from churn.*")

    except Exception as e:
        st.warning(f"Could not compute SHAP explanation for this customer: {e}")

    st.markdown("---")

    # Retention Recommendations Section
    st.markdown("### AI Retention Recommendations")

    recs = generate_recommendations(cust_dict)
    primary_act, primary_reason = get_primary_recommendation(cust_dict)

    rec_prio = exp_result['risk_level'] if 'exp_result' in locals() else risk
    prio_label = "URGENT" if rec_prio == "Critical Risk" else ("HIGH" if rec_prio == "High Risk" else ("MEDIUM" if rec_prio == "Medium Risk" else "LOW"))

    st.markdown(f"**Recommendation Urgency Priority:** `{prio_label}`")

    st.success(f"**Primary Action:** {primary_act}\n\n**Reason:** {primary_reason}")

    st.markdown("#### Additional Prescriptive Actions:")
    other_actions = recs[1:]
    if other_actions:
        for idx, r in enumerate(other_actions, 1):
            st.markdown(f"**{idx}. {r['action']}** — *{r['reason']}*")
    else:
        st.markdown("No additional action required. Standard retention monitoring.")

    st.caption("*Rule-based retention recommendations derived from customer attribute analysis.*")


# =============================================================================
# 4. HIGH-RISK RETENTION ACTIONS
# =============================================================================
elif navigation_option == "High-Risk Retention Actions":
    st.markdown("### High-Risk Retention Actions")

    high_recs_df = load_csv_data(HIGH_RISK_RECS_CSV)

    crit_cnt = len(high_recs_df[high_recs_df['risk_level'] == 'Critical Risk'])
    high_cnt = len(high_recs_df[high_recs_df['risk_level'] == 'High Risk'])
    comb_cnt = len(high_recs_df)
    avg_hr_prob = high_recs_df['churn_probability'].mean() * 100

    # KPIs
    hk1, hk2, hk3, hk4 = st.columns(4)
    with hk1:
        st.metric("Critical Customers", f"{crit_cnt:,}")
    with hk2:
        st.metric("High-Risk Customers", f"{high_cnt:,}")
    with hk3:
        st.metric("Combined Target Accounts", f"{comb_cnt:,}")
    with hk4:
        st.metric("Avg Target Churn Prob", f"{avg_hr_prob:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    st.markdown("#### Filter High-Risk Action List")
    hf1, hf2, hf3, hf4 = st.columns(4)
    with hf1:
        h_risk_sel = st.multiselect("Risk Tier", options=['Critical Risk', 'High Risk'], default=['Critical Risk', 'High Risk'])
    with hf2:
        h_contract_sel = st.multiselect("Contract Type", options=sorted(high_recs_df['Contract'].dropna().unique().tolist()))
    with hf3:
        h_internet_sel = st.multiselect("Internet Service Type", options=sorted(high_recs_df['InternetService'].dropna().unique().tolist()))
    with hf4:
        h_payment_sel = st.multiselect("Payment Method", options=sorted(high_recs_df['PaymentMethod'].dropna().unique().tolist()))

    filtered_high_df = high_recs_df.copy()
    if h_risk_sel:
        filtered_high_df = filtered_high_df[filtered_high_df['risk_level'].isin(h_risk_sel)]
    if h_contract_sel:
        filtered_high_df = filtered_high_df[filtered_high_df['Contract'].isin(h_contract_sel)]
    if h_internet_sel:
        filtered_high_df = filtered_high_df[filtered_high_df['InternetService'].isin(h_internet_sel)]
    if h_payment_sel:
        filtered_high_df = filtered_high_df[filtered_high_df['PaymentMethod'].isin(h_payment_sel)]

    # Format churn probability
    disp_high_df = filtered_high_df.copy()
    disp_high_df['churn_probability'] = (disp_high_df['churn_probability'] * 100).map('{:.2f}%'.format)

    st.dataframe(disp_high_df, use_container_width=True, height=350)

    # Download button
    csv_data = filtered_high_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk Retention Report (CSV)",
        data=csv_data,
        file_name="high_risk_retention_report.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("---")

    # Top 20 Section
    st.markdown("### Top 20 Highest-Risk Customers")
    top20_df = load_csv_data(TOP_20_CSV)

    fig_top20 = px.bar(
        top20_df.sort_values(by='churn_probability', ascending=True),
        x='churn_probability',
        y='customerID',
        orientation='h',
        labels={'churn_probability': 'Churn Probability', 'customerID': 'Customer ID'},
        title="Top 20 Accounts Requiring Immediate Retention Action",
        color='churn_probability',
        color_continuous_scale='Reds'
    )
    fig_top20.update_layout(height=450, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_top20, use_container_width=True)


# =============================================================================
# 5. ENGAGEMENT & DROP-OFF RISK
# =============================================================================
elif navigation_option == "Engagement & Drop-Off Risk":
    st.markdown("### Proxy Engagement & Silent Drop-Off Risk Analytics")

    st.markdown(
        "This module measures customer service adoption across 6 value-added products "
        "(`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) "
        "to compute a **Proxy Engagement Score (0 - 100)** and flag silent drop-off vulnerability."
    )

    total_c = len(predictions_df)
    eng_counts = predictions_df['engagement_category'].value_counts()
    healthy_c = eng_counts.get('Healthy Engagement', 0)
    mod_c = eng_counts.get('Moderate Engagement', 0)
    silent_c = eng_counts.get('Silent Drop-Off Risk', 0)

    # KPI Row
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.metric("Avg Engagement Score", f"{predictions_df['engagement_score'].mean():.1f} / 100")
    with ec2:
        st.metric("Healthy Engagement", f"{healthy_c:,} ({(healthy_c/total_c)*100:.1f}%)")
    with ec3:
        st.metric("Moderate Engagement", f"{mod_c:,} ({(mod_c/total_c)*100:.1f}%)")
    with ec4:
        st.metric("Silent Drop-Off Risk", f"{silent_c:,} ({(silent_c/total_c)*100:.1f}%)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    ech1, ech2 = st.columns(2)

    with ech1:
        st.markdown("#### Engagement Category Breakdown")
        fig_ec = px.pie(
            values=eng_counts.values,
            names=eng_counts.index,
            hole=0.45,
            color=eng_counts.index,
            color_discrete_map={
                'Healthy Engagement': '#2A9D8F',
                'Moderate Engagement': '#E9C46A',
                'Silent Drop-Off Risk': '#E63946'
            }
        )
        fig_ec.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_ec.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_ec, use_container_width=True)

    with ech2:
        st.markdown("#### Engagement Score vs. Churn Probability")
        fig_scat = px.scatter(
            predictions_df,
            x='engagement_score',
            y='churn_probability',
            color='engagement_category',
            hover_data=['customerID', 'Contract', 'MonthlyCharges'],
            labels={'engagement_score': 'Proxy Engagement Score', 'churn_probability': 'Churn Probability'},
            color_discrete_map={
                'Healthy Engagement': '#2A9D8F',
                'Moderate Engagement': '#E9C46A',
                'Silent Drop-Off Risk': '#E63946'
            },
            opacity=0.6
        )
        fig_scat.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("---")
    st.markdown("### Silent Drop-Off Risk Customer List")

    silent_df = predictions_df[predictions_df['engagement_category'] == 'Silent Drop-Off Risk'].sort_values(by='churn_probability', ascending=False)
    disp_silent = silent_df.copy()
    disp_silent['churn_probability'] = (disp_silent['churn_probability'] * 100).map('{:.2f}%'.format)

    st.dataframe(disp_silent[['customerID', 'engagement_score', 'churn_probability', 'risk_level', 'engagement_warning', 'Contract', 'MonthlyCharges', 'InternetService']], use_container_width=True, height=350)


# =============================================================================
# 6. FEATURE STICKINESS MATRIX
# =============================================================================
elif navigation_option == "Feature Stickiness Matrix":
    st.markdown("### Feature Stickiness Matrix")

    # Observational Disclaimer
    st.markdown(f'<div class="disclaimer-box">💡 <b>Observational Analysis Disclaimer:</b> {CAUSATION_DISCLAIMER}</div>', unsafe_allow_html=True)

    stickiness_df = calculate_feature_stickiness(predictions_df)

    st.markdown("#### Observed Churn Rates by Feature Adoption")

    # Metrics overview
    top_anchor = stickiness_df.iloc[0] if not stickiness_df.empty else None
    if top_anchor is not None:
        sk1, sk2, sk3, sk4 = st.columns(4)
        with sk1:
            st.metric("Top Anchor Feature", str(top_anchor['feature']))
        with sk2:
            st.metric("Top Churn Reduction (Abs)", f"{top_anchor['churn_reduction_abs']:.2f}% pts")
        with sk3:
            st.metric("Top Relative Churn Reduction", f"{top_anchor['churn_reduction_pct']:.1f}%")
        with sk4:
            st.metric("High Anchor Features", len(stickiness_df[stickiness_df['stickiness_tier'] == 'High Anchor Service']))

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar Chart comparison
    fig_stick = go.Figure()
    fig_stick.add_trace(go.Bar(
        name='Churn Rate WITH Feature (%)',
        x=stickiness_df['feature'],
        y=stickiness_df['churn_rate_with'],
        marker_color='#2A9D8F'
    ))
    fig_stick.add_trace(go.Bar(
        name='Churn Rate WITHOUT Feature (%)',
        x=stickiness_df['feature'],
        y=stickiness_df['churn_rate_without'],
        marker_color='#E63946'
    ))
    fig_stick.update_layout(
        barmode='group',
        title="Comparison of Churn Rates: Feature Adopters vs. Non-Adopters",
        xaxis_title="Feature / Service",
        yaxis_title="Observed Churn Rate (%)",
        height=380,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_stick, use_container_width=True)

    st.markdown("#### Stickiness Breakdown Table")
    st.dataframe(stickiness_df, use_container_width=True, height=320)


# =============================================================================
# 7. SMART OPERATIONAL ALERTS
# =============================================================================
elif navigation_option == "Smart Operational Alerts":
    st.markdown("### Smart Operational Risk Alerts")

    st.markdown(
        "Rule-based internal operational alerts automatically identify customer accounts requiring proactive interventions "
        "based on churn probability, engagement score, contract type, and service coverage gaps."
    )

    alerts_df = generate_smart_alerts(predictions_df)
    alert_summary = get_alerts_summary(alerts_df)

    # Alert KPIs
    al1, al2, al3, al4 = st.columns(4)
    with al1:
        st.metric("Total Operational Alerts", f"{alert_summary['total']:,}")
    with al2:
        st.metric("Critical Alerts", f"{alert_summary['critical']:,}")
    with al3:
        st.metric("Warning Alerts", f"{alert_summary['warning']:,}")
    with al4:
        st.metric("Info Alerts", f"{alert_summary['info']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    af1, af2 = st.columns(2)
    with af1:
        sel_sev = st.multiselect("Severity Tier", options=['CRITICAL', 'WARNING', 'INFO'], default=['CRITICAL', 'WARNING'])
    with af2:
        all_types = sorted(alerts_df['alert_type'].unique().tolist()) if not alerts_df.empty else []
        sel_types = st.multiselect("Alert Type", options=all_types)

    filt_alerts = alerts_df.copy()
    if sel_sev:
        filt_alerts = filt_alerts[filt_alerts['severity'].isin(sel_sev)]
    if sel_types:
        filt_alerts = filt_alerts[filt_alerts['alert_type'].isin(sel_types)]

    st.markdown(f"**Showing {len(filt_alerts):,} matching active alerts**")

    disp_alerts = filt_alerts.copy()
    if 'churn_probability' in disp_alerts.columns:
        disp_alerts['churn_probability'] = (disp_alerts['churn_probability'] * 100).map('{:.1f}%'.format)

    st.dataframe(disp_alerts[['customerID', 'severity', 'alert_type', 'alert_title', 'churn_probability', 'engagement_score', 'MonthlyCharges', 'Contract', 'action_recommended']], use_container_width=True, height=400)

    # Download Alerts CSV
    alert_csv = filt_alerts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Smart Risk Alerts Report (CSV)",
        data=alert_csv,
        file_name="smart_operational_alerts.csv",
        mime="text/csv",
        type="primary"
    )


# =============================================================================
# 8. CUSTOMER PERSONAS & SEGMENTS
# =============================================================================
elif navigation_option == "Customer Personas & Segments":
    st.markdown("### Customer Personas & Behavioral Segmentation")

    st.markdown(
        "Unsupervised K-Means clustering segments the customer base into 4 distinct business personas based on "
        "tenure, monthly spend, engagement score, total charges, and predicted churn risk."
    )

    persona_summary = get_persona_summary(predictions_df)

    # Display KPI Cards for each Persona
    p_cols = st.columns(4)
    for idx, (_, row) in enumerate(persona_summary.iterrows()):
        with p_cols[idx % 4]:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-title">{row["persona_name"]}</div>'
                f'<div class="kpi-value">{row["customer_count"]:,} ({row["percentage"]}%)</div>'
                f'<div style="font-size: 0.85rem; color: #64748B; margin-top: 0.3rem;">Avg Churn Prob: <b>{row["avg_churn_probability"]*100:.1f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Persona Scatter Plot
    st.markdown("#### Persona Clustering: Monthly Charges vs. Churn Probability")
    fig_p_scat = px.scatter(
        predictions_df,
        x='MonthlyCharges',
        y='churn_probability',
        color='persona_name',
        size='tenure',
        hover_data=['customerID', 'Contract', 'engagement_score'],
        labels={'MonthlyCharges': 'Monthly Charges ($)', 'churn_probability': 'Churn Probability', 'persona_name': 'Persona Segment'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_p_scat.update_layout(height=400, margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig_p_scat, use_container_width=True)

    st.markdown("#### Persona Summary Matrix")
    st.dataframe(persona_summary, use_container_width=True, height=250)


# =============================================================================
# 9. MODEL INSIGHTS
# =============================================================================
elif navigation_option == "Model Insights":
    st.markdown("### Model Insights & Technical Evaluation")

    metrics = model_pkg.get('metrics', {
        'accuracy': 0.7637,
        'precision': 0.5485,
        'recall': 0.6203,
        'f1_score': 0.5822,
        'roc_auc': 0.8225
    })

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with m2:
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
    with m3:
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
    with m4:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    with m5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")

    st.markdown("---")

    # Confusion Matrix & Global Importance Row
    col_cm, col_imp = st.columns([1, 1.5])

    with col_cm:
        st.markdown("#### Test Set Confusion Matrix")

        cm_data = np.array([[844, 191], [142, 232]])
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Class", y="Actual Class", color="Customer Count"),
            x=['Retained (0)', 'Churn (1)'],
            y=['Retained (0)', 'Churn (1)'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "**True Negatives (TN)**: 844 | **False Positives (FP)**: 191\n\n"
            "**False Negatives (FN)**: 142 | **True Positives (TP)**: 232"
        )

    with col_imp:
        st.markdown("#### Global Feature Importance (Top 15)")

        imp_df = load_csv_data(FEATURE_IMP_CSV).head(15).sort_values(by='importance', ascending=True)

        # Map human readable feature names
        name_map = {
            'TotalCharges': 'Total Charges ($)',
            'tenure': 'Tenure (Months)',
            'MonthlyCharges': 'Monthly Charges ($)',
            'Contract_Month-to-month': 'Month-to-Month Contract',
            'OnlineSecurity_No': 'No Online Security',
            'Contract_Two year': 'Two-Year Contract',
            'TechSupport_No': 'No Tech Support',
            'PaymentMethod_Electronic check': 'Electronic Check Payment',
            'InternetService_Fiber optic': 'Fiber Optic Internet',
            'OnlineBackup_No': 'No Online Backup',
            'gender_Female': 'Gender: Female',
            'gender_Male': 'Gender: Male',
            'SeniorCitizen': 'Senior Citizen Status',
            'PaperlessBilling_Yes': 'Paperless Billing',
            'DeviceProtection_No': 'No Device Protection'
        }
        imp_df['display_name'] = imp_df['feature'].map(lambda x: name_map.get(x, x))

        fig_imp = px.bar(
            imp_df,
            x='importance',
            y='display_name',
            orientation='h',
            labels={'importance': 'Random Forest Importance Score', 'display_name': 'Feature Name'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
