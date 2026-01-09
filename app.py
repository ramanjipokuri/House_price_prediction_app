import streamlit as st
import numpy as np
import pandas as pd
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
# Load dataset
# =====================================================
df = pd.read_csv("House_data.csv")

# =====================================================
# Currency rates (Base = USD)
# =====================================================
currency_rates = {
    "USD": 1,
    "INR": 90.22,
    "EUR": 0.86,
    "GBP": 0.75,
    "AUD": 1.50,
    "CAD": 1.39,
    "SGD": 1.29,
    "JPY": 157.72,
    "AED": 3.67,
    "CNY": 6.98
}

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# Background Video (Base64 – reliable everywhere)
# =====================================================
def get_video_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


st.markdown(f"""
<style>
.stApp {{ background: transparent; }}

video.bg {{
    position: fixed;
    top: 0; left: 0;
    min-width: 100%; min-height: 100%;
    object-fit: cover;
    z-index: -1;
}}

.block-container {{
    background: rgba(0,0,0,0.6);
    padding: 2rem;
    border-radius: 18px;
}}

section[data-testid="stSidebar"] {{
    background: rgba(15,15,15,0.95);
}}

h1,h2,h3,h4,p,span,label {{
    color: white !important;
}}
</style>

<video class="bg" autoplay muted loop>
  <source src="https://www.pexels.com/download/video/856661/" type="video/mp4">
</video>
""", unsafe_allow_html=True)

# =====================================================
# UI - Title
# =====================================================
st.title("🏠 House Price Prediction Dashboard")
st.caption("Prediction • Price Difference • Visualizations • Multi-currency")

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1.0, 5.0, 2.0)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 300, 10000, 1500)
floors = st.sidebar.number_input("Floors", 1.0, 3.0, 1.0)
grade = st.sidebar.slider("House Grade", 1, 13, 7)

lat = st.sidebar.number_input("Latitude", value=47.6, format="%.4f")
long = st.sidebar.number_input("Longitude", value=-122.3, format="%.4f")

# =====================================================
# Build input feature vector
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

feature_vector = [input_data[col] for col in feature_names]
features = np.array([feature_vector])

# =====================================================
# Prediction (User Input)
# =====================================================
features_scaled = scaler.transform(features)
current_price = model.predict(features_scaled)[0]

# =====================================================
# Baseline House (Reference)
# =====================================================
baseline_input = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1500,
    "floors": 1,
    "grade": 7,
    "lat": 47.6,
    "long": -122.3
}

baseline_vector = [baseline_input[col] for col in feature_names]
baseline_scaled = scaler.transform([baseline_vector])
baseline_price = model.predict(baseline_scaled)[0]

price_diff = current_price - baseline_price

# =====================================================
# MAIN OUTPUT
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Predicted Price (USD)")
    st.success(f"USD {current_price:,.2f}")

with col2:
    st.subheader("📈 Price Difference (vs Baseline)")
    st.info(f"USD {price_diff:,.2f}")

# =====================================================
# Multi-Currency Display
# =====================================================
st.subheader("🌍 Price in Multiple Currencies")

currency_df = []
for cur, rate in currency_rates.items():
    currency_df.append({
        "Currency": cur,
        "Price": current_price * rate
    })

currency_df = pd.DataFrame(currency_df)
st.dataframe(currency_df, use_container_width=True)

# =====================================================
# Visualizations
# =====================================================
st.subheader("📊 Data Visualizations")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.write("Price vs Living Area")
    st.scatter_chart(
        df[["sqft_living", "price"]].rename(
            columns={"sqft_living": "Living Area", "price": "Price"}
        )
    )

with viz_col2:
    st.write("Price vs Grade")
    st.bar_chart(
        df.groupby("grade")["price"].mean()
    )

# =====================================================
# Dataset Display
# =====================================================
with st.expander("📂 View Dataset"):
    st.dataframe(df.head(50), use_container_width=True)
