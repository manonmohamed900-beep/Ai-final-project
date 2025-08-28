
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
    st.title("💡 Weather Advice")

    # إدخالات بسيطة
    temp = st.number_input("درجة الحرارة (°C):", value=30.0)
    radiation = st.number_input("إشعاع شمسي (MJ/m²):", value=25.0)
    et0 = st.number_input("Evapotranspiration (mm):", value=5.5)
    dew = st.number_input("Dew Point (°C):", value=12.0)
    daylight = st.number_input("مدة النهار (ثواني):", value=43000)

    tips = []

    # بناءً على الحرارة
    if temp < 12:
        tips.append("🧥 الجو بارد: البس طبقات + جاكيت، سكارف بالليل.")
    elif temp < 20:
        tips.append("🧥 لطيف مائل للبرودة: تيشيرت + جاكيت خفيف.")
    elif temp < 29:
        tips.append("👕 معتدل/دافي: قطن خفيف واشرب مية كويس.")
    elif temp < 36:
        tips.append("🔥 حار: قطن/لينين، كاب، قللي الخروج وقت الظهر.")
    else:
        tips.append("🥵 شديد الحرارة: ظل/تكييف، سوائل وإلكتروليتس، قللي المجهود 11ص–4م.")

    # عوامل إضافية
    if radiation >= 20:
        tips.append("🕶 إشعاع عالي: واقي شمس SPF 50+ ونضارة.")
    if et0 >= 6:
        tips.append("💧 الجو بيسحب رطوبة بسرعة: اشربي مية زيادة.")
    if dew >= 18:
        tips.append("💦 رطوبة عالية: اختاري أقمشة ماصّة للعرق وتهوية كويسة.")
    elif dew <= 5:
        tips.append("🌵 جفاف عالي: مرطّب للبشرة وشرب مية.")
    if daylight >= 43000:
        tips.append("☀ نهار طويل: المجهود يكون قبل 11ص أو بعد 4م.")

    st.subheader("✨ النصايح:")
    for t in tips:
        st.markdown(f"- {t}")

# ============================
# Report (Interactive) Page
# ============================
elif page == "Report (Interactive)":
    st.title("📑 Interactive Report")

    # تأكيد إن الحالة موجودة
    if "step" not in st.session_state:
        st.session_state.step = 1

    # أزرار التحكم
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬅ السابق") and st.session_state.step > 1:
            st.session_state.step -= 1
    with col2:
        if st.button("التالي ➡") and st.session_state.step < 5:
            st.session_state.step += 1

    step = st.session_state.step

    # عرض الشرائح
    if step == 1:
        st.header("✨ Welcome")
        st.write("مرحبًا بك في تقرير الطقس التفاعلي عن القاهرة 🌸")
        st.write("استخدم الأزرار للتنقل بين الشرائح ➡⬅")

    elif step == 2:
        st.header("📂 Dataset Snapshot")
        st.dataframe(df.head(10))

    elif step == 3:
        st.header("📊 Descriptive Statistics")
        st.write(df.describe())

    elif step == 4:
        st.header("📈 Temperature Trend")
        st.line_chart(df['apparent_temperature_mean (°C)'])

    elif step == 5:
        st.header("⚡ Key Indicators & Advice")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 متوسط الحرارة", f"{df['apparent_temperature_mean (°C)'].mean():.2f} °C")
        col2.metric("💦 متوسط الرطوبة (dew point)", f"{df['dew_point_2m_mean (°C)'].mean():.2f} °C")
        col3.metric("☀ ساعات شمس", f"{df['sunshine_duration (s)'].mean()/3600:.1f} h")

        st.markdown("""
        ### 💡 نصائح سريعة
        - 🧥 الجو برد = البس طبقات.  
        - 🔥 الجو حر = قلل الخروج وقت الظهر.  
        - 🕶 إشعاع عالي = واقي شمس + نضارة.  
        """)
        
