import streamlit as st
import pandas as pd
import joblib

# ============================
# Page Config & Styling
# ============================
st.set_page_config(page_title="Cairo Weather Prediction", page_icon="🌸", layout="wide")

# Custom CSS for pink theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff0f6;
    }
    .stSidebar {
        background-color: #ffe6f0;
    }
    h1, h2, h3 {
        color: #cc0066;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #cc0066;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# Load Data & Model
# ============================
DATA_PATH = "Cairo-Weather.csv"
MODEL_PATH = "LinearRegression.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df = load_data()
model = load_model()

# ============================
# Sidebar for navigation
# ============================
st.sidebar.title("🌸 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Prediction"])

# ============================
# Overview Page
# ============================
if page == "Overview":
    st.title("🌸 Welcome to Cairo Weather Prediction App")
    st.write("This app predicts the average daily temperature in Cairo using weather-related features.")

    st.subheader("✨ Features Used in the Model")
    features = [
        'apparent_temperature_mean (°C)',
        'et0_fao_evapotranspiration (mm)',
        'daylight_duration (s)',
        'shortwave_radiation_sum (MJ/m²)',
        'dew_point_2m_mean (°C)',
        'sunshine_duration (s)'
    ]
    st.markdown("\n".join([f"- {f}" for f in features]))

    st.subheader("📊 Quick Insights")
    st.write("Here are some quick stats from the dataset:")
    st.write(df[features].describe())

    st.markdown('<div class="footer">🌸 Developed by Menna Mohamed Rady 🌸</div>', unsafe_allow_html=True)

# ============================
# Prediction Page
# ============================
elif page == "Prediction":
    st.title("🤖 Temperature Prediction")
    st.write("Enter feature values to predict the daily average temperature.")

    features = [
        'apparent_temperature_mean (°C)',
        'et0_fao_evapotranspiration (mm)',
        'daylight_duration (s)',
        'shortwave_radiation_sum (MJ/m²)',
        'dew_point_2m_mean (°C)',
        'sunshine_duration (s)'
    ]

    input_values = {}
    cols = st.columns(2)
    for i, col in enumerate(features):
        with cols[i % 2]:
            default_val = float(df[col].mean()) if col in df.columns else 0.0
            input_values[col] = st.number_input(col, value=default_val)

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded!")
        else:
            X_new = pd.DataFrame([input_values])
            try:
                pred = model.predict(X_new)[0]
                st.success(f"✅ Predicted temperature: {pred:.2f} °C")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
