import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="\U0001F3E1", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("house_price_model.pkl")
    scaler = joblib.load("house_price_scaler.pkl")
    features = joblib.load("house_price_features.pkl")
    return model, scaler, features

model, scaler, feature_order = load_artifacts()

st.title("House Price Prediction")
st.write("Enter the property details below to estimate its sale price (Ames, Iowa housing market).")

with st.form("house_form"):
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=200, value=1500)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
        year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)

    with col2:
        bedrooms = st.slider("Bedrooms Above Grade", 0, 8, 3)
        full_bath = st.slider("Full Bathrooms", 0, 4, 2)
        half_bath = st.slider("Half Bathrooms", 0, 2, 0)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=500, value=9000)
        year_sold = st.number_input("Year Sold", min_value=2006, max_value=2025, value=2010)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    total_sf = total_bsmt_sf + gr_liv_area
    house_age = year_sold - year_built
    total_bath = full_bath + 0.5 * half_bath

    # Build a single-row input aligned to the training feature order.
    # Any engineered/one-hot columns not explicitly set here default to 0 / typical values,
    # since a full 79-attribute form would be impractical for an end user.
    input_row = pd.DataFrame(0, index=[0], columns=feature_order)
    for col, val in {
        "OverallQual": overall_qual, "GrLivArea": gr_liv_area, "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars, "YearBuilt": year_built, "BedroomAbvGr": bedrooms,
        "FullBath": full_bath, "HalfBath": half_bath, "LotArea": lot_area, "YrSold": year_sold,
        "TotalSF": total_sf, "HouseAge": house_age, "TotalBath": total_bath,
    }.items():
        if col in input_row.columns:
            input_row[col] = val

    input_scaled = scaler.transform(input_row)
    pred_log = model.predict(input_row if hasattr(model, "feature_importances_") else input_scaled)[0]
    pred_price = np.expm1(pred_log)

    st.divider()
    st.success(f"Estimated Sale Price: **${pred_price:,.0f}**")
    st.caption("Model: trained on the Ames Housing Dataset. See the accompanying notebook for full methodology.")
    st.caption("Note: this simplified form covers the most influential features; the full model uses 79 attributes.")
