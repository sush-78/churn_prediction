# 🔮 Customer Churn Prediction & Retention Agent

An AI-powered system designed to predict customer churn, explain churn drivers using SHAP feature importance, segment customer risk profiles, identify high-risk individuals, and generate actionable retention recommendations.

## 🚀 Key Features & Hackathon Requirements
1. **Customer Data Processing**: Automated cleaning, encoding, and feature transformation.
2. **Churn Classification**: Random Forest model predicting customer churn status.
3. **Churn Probability**: Quantitative score (0–100%) indicating likelihood of leaving.
4. **Customer Risk Segmentation**: Risk tiering (`Low`, `Medium`, `High` risk groups).
5. **Feature Importance & Explainability**: Global and per-customer SHAP value analysis.
6. **High-Risk Identification**: Filterable dashboard highlighting critical accounts.
7. **Retention Recommendations**: Prescriptive strategies tailored to churn factors.

## 📁 Directory Structure
```text
Customer Churn Prediction Agent/
├── data/
│   ├── raw/                  # Raw input datasets (e.g., Telco Churn CSV)
│   └── processed/            # Cleaned & transformed datasets
├── models/                   # Serialized Random Forest model & scalers (.pkl)
├── src/                      # Modular Python package
│   ├── __init__.py
│   ├── data_processor.py     # Data loading, preprocessing, and encoding
│   ├── model_trainer.py      # Random Forest training & evaluation
│   ├── risk_segmentation.py  # Churn probability & risk tier assignment
│   ├── explainability.py     # SHAP calculations & explainability plots
│   └── recommender.py        # Automated retention action engine
├── app.py                    # Streamlit web interactive application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest Classifier)
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
