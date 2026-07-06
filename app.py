import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🏠 House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0F172A,#1E293B,#2563EB);
}

.main-title{
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#dbeafe;
    font-size:18px;
    margin-bottom:30px;
}

.block{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,.25);
}

.result{
    background:linear-gradient(90deg,#16a34a,#22c55e);
    color:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:34px;
    font-weight:bold;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("house_rent_prediction.pkl")

# ---------------- TITLE ----------------
st.markdown("<div class='main-title'>🏠 House Rent Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict house rent using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("<div class='block'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("🏡 BHK", 1, 10, 2)
    size = st.number_input("📐 Size (sq ft)", 100, 10000, 1000)
    bathroom = st.number_input("🚿 Bathrooms", 1, 10, 2)
    city = st.selectbox(
        "🌆 City",
        ["Bangalore","Chennai","Delhi","Hyderabad","Kolkata","Mumbai"]
    )

with col2:
    floor = st.text_input("🏢 Floor", "1 out of 3")

    area_type = st.selectbox(
        "📍 Area Type",
        ["Super Area","Carpet Area","Built Area"]
    )

    furnishing = st.selectbox(
        "🛋 Furnishing Status",
        ["Unfurnished","Semi-Furnished","Furnished"]
    )

    tenant = st.selectbox(
        "👨‍👩‍👧 Tenant Preferred",
        ["Bachelors","Family","Bachelors/Family"]
    )

st.write("")

if st.button("🚀 Predict Rent", use_container_width=True):

    input_df = pd.DataFrame({
        "BHK":[bhk],
        "Size":[size],
        "Floor":[floor],
        "Area Type":[area_type],
        "City":[city],
        "Furnishing Status":[furnishing],
        "Tenant Preferred":[tenant],
        "Bathroom":[bathroom]
    })

    prediction = model.predict(input_df)[0]

    st.markdown(
        f"<div class='result'>💰 Estimated Rent<br><br>₹ {prediction:,.0f} / Month</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption("Made with ❤️ using Streamlit • Scikit-learn • Random Forest")