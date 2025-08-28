import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# =======================
# Load Data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("Cairo-Weather.csv")  # الملف موجود جنب app.py
    return df

df = load_data()

# =======================
# Sidebar Navigation
# =======================
st.sidebar.title("🌤 Cairo Weather App")
page = st.sidebar.radio("Go to", ["Overview", "Visualization", "Prediction", "Advice", "Report"])

# =======================
# Overview Page
# =======================
if page == "Overview":
    st.title("🌤 Cairo Weather Overview")
    st.write("Welcome to the Cairo Weather Dashboard. Explore metrics, charts, and predictions for Cairo's climate.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Temp", f"{df['Temperature'].max():.1f} °C")
    col2.metric("Min Temp", f"{df['Temperature'].min():.1f} °C")
    col3.metric("Avg Temp", f"{df['Temperature'].mean():.1f} °C")

    fig = px.line(df, x="Date", y="Temperature", title="Temperature Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

# =======================
# Visualization Page
# =======================
elif page == "Visualization":
    st.title("📊 Data Visualization")

    st.subheader("Temperature Distribution")
    fig1 = px.histogram(df, x="Temperature", nbins=20, title="Temperature Histogram")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Humidity vs Temperature")
    fig2 = px.scatter(df, x="Humidity", y="Temperature", color="Wind",
                      title="Humidity vs Temp (colored by Wind)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Wind Speed Over Time")
    fig3 = px.line(df, x="Date", y="Wind", title="Wind Speed Trend")
    st.plotly_chart(fig3, use_container_width=True)

# =======================
# Prediction Page
# =======================
elif page == "Prediction":
    st.title("🤖 Weather Prediction")

    features = ["Humidity", "Wind", "Radiation"]
    df = df.dropna(subset=features + ["Temperature"])

    X = df[features]
    y = df["Temperature"]

    # تحميل أو تدريب الموديل
    try:
        model = joblib.load("LinearRegression.pkl")
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, "LinearRegression.pkl")

    st.write("Enter weather conditions to predict temperature:")

    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
    radiation = st.slider("Solar Radiation", 0, 1000, 500)

    input_data = np.array([[humidity, wind, radiation]])
    prediction = model.predict(input_data)[0]

    st.success(f"🌡 Predicted Temperature: {prediction:.2f} °C")

# =======================
# Advice Page
# =======================
elif page == "Advice":
    st.title("💡 Weather Advice")

    avg_temp = df["Temperature"].mean()
    if avg_temp > 30:
        st.warning("☀ It's hot! Stay hydrated and avoid the sun at noon.")
    elif avg_temp < 15:
        st.info("🧥 It's cold! Wear warm clothes and take care of your health.")
    else:
        st.success("🌤 The weather is moderate. Enjoy your day outside!")

# =======================
# Report Page
# =======================
elif page == "Report":
    st.title("📑 Weather Report")

    st.write("Here’s a summary of Cairo’s climate based on the dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Humidity", f"{df['Humidity'].mean():.1f} %")
        st.metric("Max Wind Speed", f"{df['Wind'].max():.1f} km/h")
    with col2:
        st.metric("Average Radiation", f"{df['Radiation'].mean():.1f}")
        st.metric("Days Recorded", len(df))

    fig_report = px.box(df, y="Temperature", title="Temperature Variation")
    st.plotly_chart(fig_report, use_container_width=True)
        
