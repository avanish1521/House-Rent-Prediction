import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return joblib.load("house_rent_prediction.pkl")

model = load_model()

# ------------------ CSS ------------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
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
    margin-bottom:20px;
}
div[data-testid="stForm"]{
    background:white;
    padding:25px;
    border-radius:15px;
}
.result{
    background:#16a34a;
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 class='main-title'>🏠 House Rent Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict Monthly House Rent using Machine Learning</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "4746 Houses")
col2.metric("Model", "Random Forest")
col3.metric("R² Score", "67.67%")

st.write("")

# ------------------ FORM ------------------
with st.form("predict_form"):

    c1, c2 = st.columns(2)

    with c1:
        bhk = st.number_input("BHK", 1, 10, 2)
        size = st.number_input("Size (sq.ft)", 100, 10000, 1000)
        bathroom = st.number_input("Bathroom", 1, 10, 2)
        city = st.selectbox(
            "City",
            [
                "Bangalore",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai"
            ]
        )

    with c2:

        floor = st.text_input("Floor", "1 out of 3")

        area_type = st.selectbox(
            "Area Type",
            [
                "Super Area",
                "Carpet Area",
                "Built Area"
            ]
        )

        furnishing = st.selectbox(
            "Furnishing Status",
            [
                "Unfurnished",
                "Semi-Furnished",
                "Furnished"
            ]
        )

        tenant = st.selectbox(
            "Tenant Preferred",
            [
                "Bachelors",
                "Family",
                "Bachelors/Family"
            ]
        )

    submit = st.form_submit_button("🚀 Predict Rent")

# ------------------ PREDICTION ------------------

if submit:

    input_df = pd.DataFrame({
        "BHK": pd.Series([bhk], dtype="int64"),
        "Size": pd.Series([size], dtype="int64"),
        "Floor": pd.Series([floor], dtype="string"),
        "Area Type": pd.Series([area_type], dtype="string"),
        "City": pd.Series([city], dtype="string"),
        "Furnishing Status": pd.Series([furnishing], dtype="string"),
        "Tenant Preferred": pd.Series([tenant], dtype="string"),
        "Bathroom": pd.Series([bathroom], dtype="int64")
    })

    try:
        with st.spinner("Predicting..."):

            prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class='result'>
            💰 Estimated Monthly Rent<br><br>
            ₹ {prediction:,.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)

        st.subheader("Debug Information")

        st.write(input_df)
        st.write(input_df.dtypes)

st.divider()

st.caption("Developed by Avanish Mishra ❤️")