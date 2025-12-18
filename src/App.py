# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -----------------------
# Load Model & Preprocessors
# -----------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'workspace')

MODEL_PATH  = os.path.join(BASE_DIR, 'Best_Model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Scaler.pkl')
OHE_PATH    = os.path.join(BASE_DIR, 'One_Hot_Encoder.pkl')
OE_PATH     = os.path.join(BASE_DIR, 'Ordinal_Encoder.pkl')

# -----------------------
# Load Model & Preprocessors
# -----------------------
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
ohe    = joblib.load(OHE_PATH)
oe     = joblib.load(OE_PATH)
# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>Customer Churn Predictor</h1>
    <p style='text-align: center; font-size: 16px; color: #333;'>Predict if a customer is likely to churn based on their profile</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------
# Input Form
# -----------------------
st.subheader("Enter Customer Information")
with st.form(key='churn_form'):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=39)
    Usage_Frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=14)
    Support_Calls = st.number_input("Support Calls", min_value=0, max_value=50, value=5)
    Payment_Delay = st.number_input("Payment Delay (days)", min_value=0, max_value=365, value=18)
    Subscription_Type = st.selectbox("Subscription Type", ['Standard', 'Premium', 'Basic'])
    Contract_Length = st.selectbox("Contract Length", ['Monthly', 'Quarterly', 'Annual'])
    Total_Spend = st.number_input("Total Spend ($)", min_value=0, max_value=10000, value=932)
    Last_Interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=17)

    submit_button = st.form_submit_button(label='Predict Churn')

# -----------------------
# Prediction Function
# -----------------------
def predict_churn(input_data):
    df = pd.DataFrame([input_data])

    num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                'Payment Delay', 'Total Spend', 'Last Interaction']
    ordinal_cols = ['Subscription Type', 'Contract Length']
    onehot_cols = ['Gender']

    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])

    # Encode ordinal features
    df[ordinal_cols] = oe.transform(df[ordinal_cols])

    # Encode one-hot features
    ohe_df = pd.DataFrame(ohe.transform(df[onehot_cols]), 
                          columns=ohe.get_feature_names_out(onehot_cols))
    df = df.drop(columns=onehot_cols)
    df = pd.concat([df, ohe_df], axis=1)

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return prediction, proba

# -----------------------
# Run Prediction
# -----------------------
if submit_button:
    input_data = {
        'Age': Age,
        'Gender': Gender,
        'Tenure': Tenure,
        'Usage Frequency': Usage_Frequency,
        'Support Calls': Support_Calls,
        'Payment Delay': Payment_Delay,
        'Subscription Type': Subscription_Type,
        'Contract Length': Contract_Length,
        'Total Spend': Total_Spend,
        'Last Interaction': Last_Interaction
    }

    pred, proba = predict_churn(input_data)

    st.markdown("---")
    st.subheader("Prediction Result")
    
    if pred == 1:
        st.error(f"The customer is **likely to churn** 🛑")
    else:
        st.success(f"The customer is **likely to stay** ✅")

    st.markdown("**Prediction Probabilities:**")
    st.write(f"Stay: {proba[0]:.2f}, Churn: {proba[1]:.2f}")

    # Optional: Display a probability bar chart
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=['Stay', 'Churn'],
        y=[proba[0], proba[1]],
        marker_color=['green','red']
    ))
    fig.update_layout(title_text='Prediction Probabilities', yaxis=dict(range=[0,1]))
    st.plotly_chart(fig)
