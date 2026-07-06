import streamlit as st
import pandas as pd
import joblib
import numpy as np

from tensorflow.keras.models import load_model

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="🏠 House Rent Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

@st.cache_resource
def load_models():

    rf_model = joblib.load("house_rent_prediction.pkl")

    nn_model = load_model("house_rent_nn.keras")

    preprocessor = joblib.load("preprocessor.pkl")

    return rf_model, nn_model, preprocessor


rf_model, nn_model, preprocessor = load_models()

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
}

/* Hide Streamlit menu & footer */
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
header{
    visibility:hidden;
}

/* Main Title */

.title{
    text-align:center;
    color:white;
    font-size:52px;
    font-weight:800;
}

.subtitle{
    text-align:center;
    color:#dbeafe;
    font-size:20px;
    margin-bottom:30px;
}

/* Cards */

.card{

    background:white;

    padding:22px;

    border-radius:18px;

    box-shadow:0px 8px 25px rgba(0,0,0,.25);

}

/* Prediction Card */

.prediction{

    background:linear-gradient(90deg,#16a34a,#22c55e);

    color:white;

    padding:30px;

    border-radius:18px;

    text-align:center;

    font-size:34px;

    font-weight:bold;

}

/* Sidebar */

section[data-testid="stSidebar"]{

    background:#111827;

}

/* Metric Cards */

[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    padding:10px;

    box-shadow:0px 4px 12px rgba(0,0,0,.2);

}

/* Button */

.stButton>button{

    width:100%;

    border-radius:12px;

    height:55px;

    font-size:20px;

    font-weight:bold;

    background:#2563eb;

    color:white;

}

.stButton>button:hover{

    background:#1d4ed8;

}

/* Radio */

.stRadio label{

    font-size:18px;

    font-weight:600;

}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <h1 class="title">🏠 House Rent Prediction System</h1>
    <p class="subtitle">
        Predict Monthly House Rent using Machine Learning & Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "🏘 Dataset",
        "4746",
        "Houses"
    )

with k2:
    st.metric(
        "🌳 Random Forest",
        "R² 67.67%",
        "ML Model"
    )

with k3:
    st.metric(
        "🧠 Neural Network",
        "TensorFlow",
        "Deep Learning"
    )

with k4:
    st.metric(
        "⚡ Status",
        "Ready",
        "Deployment"
    )

st.write("")
st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🏠 House Rent Prediction")

st.sidebar.write(
    "Predict house rent using Machine Learning and Deep Learning."
)

st.sidebar.divider()

model_choice = st.sidebar.radio(
    "🤖 Select Prediction Model",
    [
        "Random Forest",
        "Neural Network",
        "Compare Both"
    ]
)

st.sidebar.divider()

st.sidebar.success(
    f"Selected Model:\n\n{model_choice}"
)

st.sidebar.info(
    """
Developed by

**Avanish Mishra**
"""
)
# ---------------------------------------------------
# PROPERTY DETAILS
# ---------------------------------------------------

st.markdown("## 🏠 Property Details")

with st.container(border=True):

    with st.form("prediction_form"):

        left, right = st.columns(2)

        # ---------------- LEFT ----------------

        with left:

            bhk = st.selectbox(
                "🏡 BHK",
                [1,2,3,4,5,6]
            )

            size = st.number_input(
                "📐 Size (Sq.ft)",
                min_value=100,
                max_value=10000,
                value=1200,
                step=50
            )

            bathroom = st.selectbox(
                "🚿 Bathrooms",
                [1,2,3,4,5,6]
            )

            floor = st.text_input(
                "🏢 Floor",
                value="1 out of 3"
            )

        # ---------------- RIGHT ----------------

        with right:

            city = st.selectbox(
                "🌆 City",
                [
                    "Bangalore",
                    "Chennai",
                    "Delhi",
                    "Hyderabad",
                    "Kolkata",
                    "Mumbai"
                ]
            )

            area_type = st.selectbox(
                "📍 Area Type",
                [
                    "Super Area",
                    "Carpet Area",
                    "Built Area"
                ]
            )

            furnishing = st.selectbox(
                "🛋 Furnishing",
                [
                    "Furnished",
                    "Semi-Furnished",
                    "Unfurnished"
                ]
            )

            tenant = st.selectbox(
                "👨‍👩‍👧 Tenant Preferred",
                [
                    "Family",
                    "Bachelors",
                    "Bachelors/Family"
                ]
            )

        st.write("")

        submit = st.form_submit_button(
            "🚀 Predict House Rent",
            use_container_width=True
        )
# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if submit:

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

    with st.spinner("🤖 Predicting House Rent..."):

        # -----------------------------
        # RANDOM FOREST
        # -----------------------------

        rf_prediction = rf_model.predict(input_df)[0]

        # -----------------------------
        # NEURAL NETWORK
        # -----------------------------

        X = preprocessor.transform(input_df)

        if hasattr(X, "toarray"):
            X = X.toarray()

        nn_prediction = nn_model.predict(
            X,
            verbose=0
        )[0][0]

    st.divider()

    # ===================================
    # RANDOM FOREST ONLY
    # ===================================

    if model_choice == "Random Forest":

        st.markdown(f"""
        <div class='prediction'>
        🌳 Random Forest Prediction<br><br>

        ₹ {rf_prediction:,.0f}
        </div>
        """, unsafe_allow_html=True)

    # ===================================
    # NEURAL NETWORK ONLY
    # ===================================

    elif model_choice == "Neural Network":

        st.markdown(f"""
        <div class='prediction'>
        🧠 Neural Network Prediction<br><br>

        ₹ {nn_prediction:,.0f}
        </div>
        """, unsafe_allow_html=True)

    # ===================================
    # COMPARE BOTH
    # ===================================

    else:

        st.subheader("📊 Model Comparison")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "🌳 Random Forest",
                f"₹ {rf_prediction:,.0f}"
            )

        with c2:

            st.metric(
                "🧠 Neural Network",
                f"₹ {nn_prediction:,.0f}"
            )

        st.write("")

        difference = abs(rf_prediction - nn_prediction)

        st.info(
            f"📈 Difference Between Models : ₹ {difference:,.0f}"
        )

        if difference < 1000:

            st.success(
                "✅ Both models are giving very similar predictions."
            )

        elif rf_prediction > nn_prediction:

            st.warning(
                "🌳 Random Forest predicted a higher rent."
            )

        else:

            st.warning(
                "🧠 Neural Network predicted a higher rent."
            )
# ---------------------------------------------------
# VISUAL ANALYTICS
# ---------------------------------------------------

if submit:

    st.divider()

    st.subheader("📊 Prediction Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        chart_df = pd.DataFrame({
            "Model": ["Random Forest", "Neural Network"],
            "Predicted Rent": [rf_prediction, nn_prediction]
        })

        st.bar_chart(
            chart_df.set_index("Model")
        )

    with chart_col2:

        st.dataframe(
            chart_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # -----------------------------------------------
    # DOWNLOAD RESULT
    # -----------------------------------------------

    result = pd.DataFrame({

        "BHK":[bhk],
        "Size":[size],
        "City":[city],
        "Bathroom":[bathroom],
        "Random Forest":[round(rf_prediction)],
        "Neural Network":[round(nn_prediction)]

    })

    csv = result.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction",

        data=csv,

        file_name="house_rent_prediction.csv",

        mime="text/csv"

    )

st.divider()

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
"""
<center>

### 👨‍💻 Developed by Avanish Mishra

House Rent Prediction using

🌳 Random Forest • 🧠 Neural Network • Streamlit

⭐ Thank you for visiting!

</center>
""",
unsafe_allow_html=True
)