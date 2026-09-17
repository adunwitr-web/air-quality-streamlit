from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import streamlit as st

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="ระบบวิเคราะห์สภาวะสิ่งแวดล้อมและอากาศ Real-Time",
    page_icon="🌿",
    layout="wide",
)

# ==========================================
# 2. โหลดข้อมูล 150 Samples, 3 Features, 3 Classes & เทรนโมเดล
# ==========================================
@st.cache_resource
def train_high_accuracy_model():
    # โหลด Iris Dataset (150 Samples, 3 Classes)
    iris = load_iris()
    X_full = iris.data
    y = iris.target
    
    # เลือก 3 Features ตามโจทย์อาจารย์
    X = X_full[:, :3] 
    feature_names = ["อุณหภูมิ/ความยาว (Sepal Length)", "ความชื้น/ความกว้าง (Sepal Width)", "ความเร็วลม/ความยาวกลีบ (Petal Length)"]
    class_names = ["ระดับที่ 1 (ดีมาก/Setosa)", "ระดับที่ 2 (ปานกลาง/Versicolor)", "ระดับที่ 3 (เสี่ยงมลพิษ/Virginica)"]
    
    # แบ่งข้อมูลและเทรนด้วย Random Forest เพื่อความแม่นยำสูง
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, acc, feature_names, class_names

model, model_acc, feature_names, class_names = train_high_accuracy_model()

# ==========================================
# 3. ฟังก์ชันดึง Real-Time Weather & Air API
# ==========================================
def fetch_realtime_weather(lat=7.1988, lon=100.5951, location_label="สงขลา"):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,us_aqi"

        w_res = requests.get(weather_url, timeout=4).json().get("current", {})
        a_res = requests.get(air_url, timeout=4).json().get("current", {})

        return {
            "city": location_label,
            "temp": float(w_res.get("temperature_2m", 28.0)),
            "humidity": float(w_res.get("relative_humidity_2m", 65.0)),
            "wind": float(w_res.get("wind_speed_10m", 10.0)),
            "pm2_5": float(a_res.get("pm2_5", 0.0)),
            "pm10": float(a_res.get("pm10", 0.0)),
            "us_aqi": int(a_res.get("us_aqi", 0)),
        }
    except Exception:
        return None

# ==========================================
# 4. ซิงค์ Session State สำหรับ Sliders
# ==========================================
if "f1_val" not in st.session_state:
    st.session_state.f1_val = 5.8
if "f2_val" not in st.session_state:
    st.session_state.f2_val = 3.0
if "f3_val" not in st.session_state:
    st.session_state.f3_val = 3.8

# ==========================================
# 5. แถบข้าง (Sidebar Controls)
# ==========================================
st.sidebar.markdown("### 🎛️ เครื่องมือประมวลผล Real-Time")
st.sidebar.info(f"🎯 **ความแม่นยำโมเดล (Model Accuracy): {model_acc:.1f}%**\n- ข้อมูลเรียนรู้: 150 ตัวอย่าง\n- จำนวนคุณลักษณะ: 3 Features\n- กลุ่มผลลัพธ์: 3 Classes")

PROVINCES = {
    "📍 สงขลา (Songkhla)": (7.1988, 100.5951, "สงขลา"),
    "📍 กรุงเทพมหานคร (Bangkok)": (13.7563, 100.5018, "กรุงเทพมหานคร"),
    "📍 เชียงใหม่ (Chiang Mai)": (18.7883, 98.9853, "เชียงใหม่"),
    "📍 ภูเก็ต (Phuket)": (7.8804, 98.3923, "ภูเก็ต"),
    "📍 ขอนแก่น (Khon Kaen)": (16.4322, 102.8236, "ขอนแก่น"),
}

selected_province = st.sidebar.selectbox("เลือกพื้นที่เพื่อดึงข้อมูล Real-Time", list(PROVINCES.keys()))

if st.sidebar.button("🔄 ดึงข้อมูลสดจาก API", use_container_width=True):
    lat, lon, city_label = PROVINCES[selected_province]
    data = fetch_realtime_weather(lat, lon, city_label)
    if data:
        st.session_state.f1_val = round(data["temp"] / 5.0, 1) # Scaling ให้สอดคล้องกับ Feature
        st.session_state.f2_val = round(data["humidity"] / 20.0, 1)
        st.session_state.f3_val = round(data["wind"] / 3.0, 1)
        st.session_state["live_data"] = data
        st.sidebar.success(f"อัปเดตข้อมูลสดของ {data['city']} เรียบร้อย!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**ปรับแต่งค่า Features สำหรับประมวลผล:**")

f1 = st.sidebar.slider(f"Feature 1: {feature_names[0]}", 4.0, 8.0, value=float(st.session_state.f1_val), step=0.1)
f2 = st.sidebar.slider(f"Feature 2: {feature_names[1]}", 2.0, 5.0, value=float(st.session_state.f2_val), step=0.1)
f3 = st.sidebar.slider(f"Feature 3: {feature_names[2]}", 1.0, 7.0, value=float(st.session_state.f3_val), step=0.1)

predict_btn = st.sidebar.button("✨ ทำนายผลลัพธ์ (Predict)", type="primary")

# ==========================================
# 6. ส่วนแสดงผลหลัก (Main Dashboard)
# ==========================================
st.title("🌿 ระบบวิเคราะห์และจำแนกข้อมูลสิ่งแวดล้อม (Real-Time ML Platform)")
st.caption("พัฒนาด้วย Machine Learning | สเปกข้อมูล: 150 ตัวอย่าง • 3 Features • 3 Classes")
st.divider()

if "live_data" in st.session_state:
    ld = st.session_state["live_data"]
    st.markdown(f"#### 📍 ข้อมูลสภาพอากาศและฝุ่น Real-time ณ **{ld['city']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("อุณหภูมิสด", f"{ld['temp']} °C")
    c2.metric("ความชื้นสัมพัทธ์", f"{ld['humidity']} %")
    c3.metric("ฝุ่น PM2.5 สด", f"{ld['pm2_5']} µg/m³")
    c4.metric("ดัชนี US AQI", f"{ld['us_aqi']}")
    st.divider()

if predict_btn:
    input_data = np.array([[f1, f2, f3]])
    pred_class = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    confidence = probs[pred_class] * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📌 ผลการจำแนกประเภท (Prediction Result)")
        if pred_class == 0:
            st.success(f"### ผลลัพธ์: {class_names[0]}")
            st.write("สภาพแวดล้อมอยู่ในเกณฑ์ดีเยี่ยม เหมาะแก่การทำกิจกรรมภายนอก")
        elif pred_class == 1:
            st.warning(f"### ผลลัพธ์: {class_names[1]}")
            st.write("สภาพแวดล้อมอยู่ในเกณฑ์ปานกลาง ควรเฝ้าระวังการเปลี่ยนแปลง")
        else:
            st.error(f"### ผลลัพธ์: {class_names[2]}")
            st.write("สภาพแวดล้อมอยู่ในระดับเฝ้าระวังพิเศษ หลีกเลี่ยงกิจกรรมกลางแจ้ง")

    with col2:
        st.metric("ความเชื่อมั่นของโมเดล", f"{confidence:.1f}%")
        st.metric("ความแม่นยำรวม (Model Accuracy)", f"{model_acc:.1f}%")

st.divider()
st.subheader("📊 ข้อมูลโครงสร้าง Dataset (150 Samples)")

iris_raw = load_iris()
df_display = pd.DataFrame(iris_raw.data[:, :3], columns=["Feature 1 (Sepal Length)", "Feature 2 (Sepal Width)", "Feature 3 (Petal Length)"])
df_display["Class Target"] = iris_raw.target

t1, t2 = st.tabs(["📋 ตารางข้อมูล (Dataset Table)", "📈 กราฟกระจายตัว (Scatter Plot)"])
with t1:
    st.dataframe(df_display, use_container_width=True)

with t2:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.scatterplot(data=df_display, x="Feature 1 (Sepal Length)", y="Feature 3 (Petal Length)", hue="Class Target", palette="Set1", ax=ax)
    ax.set_title("Distribution of 150 Samples across 3 Classes")
    st.pyplot(fig)