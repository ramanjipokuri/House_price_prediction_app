import streamlit as st
import numpy as np
import joblib
import json

# =====================================================
# Load model, scaler, feature schema
# =====================================================
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# =====================================================
# UI
# =====================================================
st.title("🏠 House Price Prediction")
st.caption("Clean ML pipeline • Stable predictions")

st.sidebar.header("Enter House Details")

# EXACT features used in training
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1.0, max_value=5.0, value=2.0)
sqft_living = st.sidebar.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500)
floors = st.sidebar.number_input("Floors", min_value=1.0, max_value=3.0, value=1.0)
grade = st.sidebar.slider("House Grade", min_value=1, max_value=13, value=7)

lat = st.sidebar.number_input("Latitude", value=47.6, format="%.4f")
long = st.sidebar.number_input("Longitude", value=-122.3, format="%.4f")

# =====================================================
# Build input feature vector (NO GUESSING)
# =====================================================
input_data = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "floors": floors,
    "grade": grade,
    "lat": lat,
    "long": long
}

# Arrange features exactly as training order
feature_vector = [input_data[col] for col in feature_names]
features = np.array([feature_vector])

# =====================================================
# Prediction
# =====================================================
features_scaled = scaler.transform(features)
predicted_price = model.predict(features_scaled)[0]

# =====================================================
# Output
# =====================================================
st.subheader("💰 Predicted House Price")
st.success(f"USD {predicted_price:,.2f}")

# =====================================================
# Debug (optional – remove later)
# =====================================================
with st.expander("🔍 Debug Info"):
    st.write("Feature order:", feature_names)
    st.write("Input values:", feature_vector)
