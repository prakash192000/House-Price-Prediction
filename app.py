"""
House Price Prediction – Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Load model & encoders ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model.pkl")
    enc_data = joblib.load("encoders.pkl")
    return model, enc_data

model, enc = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='color:#1a73e8;'>🏠 House Price Predictor</h1>
        <p style='color:#555; font-size:16px;'>
            Estimate Indian residential property prices (in Lakhs ₹)
            using an XGBoost machine learning model.
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar – About ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 About This App")
    st.info(
        "This app uses an **XGBoost** model trained on ~29,000 Indian "
        "residential listings.\n\n"
        "- **R² Score:** 0.828\n"
        "- **MAE:** ~22 Lakhs\n"
        "- **RMSE:** ~45 Lakhs"
    )
    st.header("📌 How to Use")
    st.markdown(
        "1. Fill in the property details on the right.\n"
        "2. Click **Predict Price**.\n"
        "3. The estimated price range will appear below."
    )

# ── Input Form ─────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
    bhk_no    = st.slider("Number of BHK", min_value=1, max_value=10, value=2)
    bhk_or_rk = st.selectbox("Property Type", ["BHK", "RK"])
    square_ft = st.number_input("Area (Square Feet)", min_value=100.0,
                                max_value=50000.0, value=1200.0, step=50.0)

with col2:
    city_options = list(enc['top_cities']) + ["Other"]
    city         = st.selectbox("City", sorted(city_options))
    latitude     = st.number_input("Latitude",  min_value=8.0,  max_value=35.0, value=12.97, format="%.5f")
    longitude    = st.number_input("Longitude", min_value=68.0, max_value=98.0, value=77.59, format="%.5f")

st.subheader("🏗️ Property Status")
col3, col4, col5, col6 = st.columns(4)
under_construction = col3.checkbox("Under Construction")
rera               = col4.checkbox("RERA Approved")
ready_to_move      = col5.checkbox("Ready to Move", value=True)
resale             = col6.checkbox("Resale Property")

# ── Prediction ─────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    # Encode inputs
    posted_enc = enc['le_posted'].transform([posted_by])[0] \
        if posted_by in enc['le_posted'].classes_ else 0
    bhkor_enc  = enc['le_bhkor'].transform([bhk_or_rk])[0] \
        if bhk_or_rk in enc['le_bhkor'].classes_ else 0
    city_cat   = city if city in enc['top_cities'] else "Other"
    city_enc   = enc['le_city'].transform([city_cat])[0] \
        if city_cat in enc['le_city'].classes_ else 0

    input_df = pd.DataFrame([[
        posted_enc,
        int(under_construction),
        int(rera),
        bhk_no,
        bhkor_enc,
        square_ft,
        int(ready_to_move),
        int(resale),
        longitude,
        latitude,
        city_enc,
    ]], columns=enc['features'])

    prediction = model.predict(input_df)[0]
    low  = prediction * 0.90
    high = prediction * 1.10

    st.success("✅ Prediction Complete!")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Lower Estimate",  f"₹ {low:.1f} L")
    col_r2.metric("Predicted Price", f"₹ {prediction:.1f} L", delta="Estimate")
    col_r3.metric("Upper Estimate",  f"₹ {high:.1f} L")

    st.caption("Price range is ±10% of the model prediction. Results are estimates only.")

    # Summary card
    with st.expander("📄 View Input Summary"):
        summary = {
            "Posted By": posted_by,
            "BHK": bhk_no,
            "Type": bhk_or_rk,
            "Area (sq ft)": square_ft,
            "City": city,
            "Latitude": latitude,
            "Longitude": longitude,
            "Under Construction": under_construction,
            "RERA Approved": rera,
            "Ready to Move": ready_to_move,
            "Resale": resale,
        }
        st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:13px;'>"
    "House Price Predictor · Capstone Project · Powered by XGBoost + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
