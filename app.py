
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
page = st.sidebar.radio("Go to", ["Overview", "Prediction", "Advice", "Report (Interactive)"])

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

elif page == "Advice":
    st.title("💡 Interactive Weather Advice (Step-by-Step)")

    # User inputs
    temp = st.number_input("Temperature (°C):", value=30.0)
    radiation = st.number_input("Solar Radiation (MJ/m²):", value=25.0)
    et0 = st.number_input("Evapotranspiration (mm):", value=5.5)
    dew = st.number_input("Dew Point (°C):", value=12.0)
    daylight = st.number_input("Daylight Duration (seconds):", value=43000)

    tips = []

    # Temperature-based advice
    if temp < 12:
        tips.append(("🧥 Cold", "Wear layers + jacket, scarf at night.", "blue"))
    elif temp < 20:
        tips.append(("🧥 Cool", "T-shirt + light jacket.", "lightblue"))
    elif temp < 29:
        tips.append(("👕 Mild/Warm", "Light cotton, stay hydrated.", "green"))
    elif temp < 36:
        tips.append(("🔥 Hot", "Cotton/linen, cap, avoid midday sun.", "orange"))
    else:
        tips.append(("🥵 Extreme Heat", "Stay in shade/AC, hydrate with electrolytes, limit activity 11am–4pm.", "red"))

    # Additional factors
    if radiation >= 20:
        tips.append(("🕶 High UV", "Use SPF 50+ sunscreen + sunglasses.", "orange"))
    if et0 >= 6:
        tips.append(("💧 Rapid moisture loss", "Drink extra water.", "blue"))
    if dew >= 18:
        tips.append(("💦 High humidity", "Wear breathable fabrics, ensure ventilation.", "lightblue"))
    elif dew <= 5:
        tips.append(("🌵 Very dry", "Moisturize skin, drink water.", "brown"))
    if daylight >= 43000:
        tips.append(("☀ Long day", "Do outdoor activities before 11am or after 4pm.", "yellow"))

    # Initialize step in session_state
    if "advice_step" not in st.session_state:
        st.session_state.advice_step = 0

    # Navigation buttons
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅ Previous") and st.session_state.advice_step > 0:
            st.session_state.advice_step -= 1
    with col3:
        if st.button("Next ➡") and st.session_state.advice_step < len(tips)-1:
            st.session_state.advice_step += 1

    # Progress bar
    progress = (st.session_state.advice_step + 1) / max(len(tips),1)
    st.progress(progress)

    # Display current tip
    if tips:
        title, desc, color = tips[st.session_state.advice_step]
        with st.expander(f"{title}", expanded=True):
            st.markdown(f"<span style='color:{color}; font-weight:bold'>{desc}</span>", unsafe_allow_html=True)
        st.caption(f"Tip {st.session_state.advice_step + 1} of {len(tips)}")
    else:
        st.info("No tips available for the given inputs.")

# ============================
# Report (Interactive) Page (Professional Version)
# ============================
elif page == "Report (Interactive)":
    st.title("📑 Interactive Weather Report")

    # Initialize step
    if "step" not in st.session_state:
        st.session_state.step = 1

    # Navigation buttons with styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅ Previous", use_container_width=True) and st.session_state.step > 1:
            st.session_state.step -= 1
    with col3:
        if st.button("Next ➡", use_container_width=True) and st.session_state.step < 6:
            st.session_state.step += 1

    st.markdown("---")  # Divider line

    step = st.session_state.step

    # Slides content
    if step == 1:
        st.header("✨ Welcome")
        st.write("Welcome to the interactive weather report for *Cairo* 🌸")
        st.info("Use the navigation buttons at the top to move between slides ➡⬅")

    elif step == 2:
        st.header("📂 Dataset Snapshot")
        st.write("Here is a quick preview of the dataset (first 10 rows):")
        st.dataframe(df.head(10), use_container_width=True)

    elif step == 3:
        st.header("📊 Descriptive Statistics")
        st.write("Summary statistics for the dataset:")
        st.dataframe(df.describe(), use_container_width=True)

    elif step == 4:
        st.header("📈 Temperature Trend")
        st.write("Line chart showing the temperature trend over time:")
        st.line_chart(df['apparent_temperature_mean (°C)'])

    elif step == 5:
        st.header("📊 Advanced Visualization")
        st.subheader("Bar Chart - Average Temperature (First 20 Days)")
        st.bar_chart(df['apparent_temperature_mean (°C)'].head(20))

        st.subheader("Histogram - Temperature Distribution")
        fig = df['apparent_temperature_mean (°C)'].plot(
            kind='hist', bins=20, title="Temperature Distribution"
        ).get_figure()
        st.pyplot(fig)

    elif step == 6:
        st.header("⚡ Key Indicators & Advice")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Avg Temperature", f"{df['apparent_temperature_mean (°C)'].mean():.2f} °C")
        col2.metric("💦 Avg Humidity (Dew Point)", f"{df['dew_point_2m_mean (°C)'].mean():.2f} °C")
        col3.metric("☀ Sunshine Hours", f"{df['sunshine_duration (s)'].mean()/3600:.1f} h")

        st.markdown("""
        ### 💡 Quick Tips
        - 🧥 *Cold weather* → Wear multiple layers.  
        - 🔥 *Hot weather* → Avoid going out during midday.  
        - 🕶 *High UV index* → Use sunscreen + sunglasses.  
        """)
        
