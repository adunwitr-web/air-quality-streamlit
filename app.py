from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st

# ==========================================
# 1. ตั้งค่าเบราว์เซอร์และใช้ Custom CSS
# ==========================================
st.set_page_config(
    page_title="ระบบวิเคราะห์คุณภาพอากาศ Real-Time | Air Quality Intelligence",
    page_icon="🌤️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #0d6efd;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. ฟังก์ชันดึงพิกัดพื้นที่จริงของผู้ใช้ (IP / Fallback)
# ==========================================
def get_user_coordinates():
    # ลองดึงจาก API ที่ 1 (ipapi.co)
    try:
        res = requests.get("https://ipapi.co/json/", timeout=3).json()
        if "latitude" in res and "longitude" in res:
            city_name = res.get("region") or res.get("city") or "พื้นที่ของคุณ"
            return res["latitude"], res["longitude"], city_name
    except Exception:
        pass

    # ลองดึงจาก API ที่ 2 (ip-api.com)
    try:
        res2 = requests.get(
            "http://ip-api.com/json/?fields=lat,lon,regionName,city,status",
            timeout=3,
        ).json()
        if res2.get("status") == "success":
            city_name = (
                res2.get("regionName") or res2.get("city") or "พื้นที่ของคุณ"
            )
            return res2["lat"], res2["lon"], city_name
    except Exception:
        pass

    # ค่าเริ่มต้นถ้าดึงพิกัดไม่ได้เลย (สงขลา / กรุงเทพฯ)
    return 7.1988, 100.5951, "สงขลา (ค่าเริ่มต้น)"


# ==========================================
# 3. ฟังก์ชันดึงมลพิษ & สภาพอากาศสดจาก Open-Meteo
# ==========================================
def fetch_weather_and_air(lat, lon, location_label):
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
# 4. โมเดล ML (3 Features, 3 Classes, 150 Samples ตามโจทย์อาจารย์)
# ==========================================
@st.cache_resource
def get_trained_model():
    np.random.seed(42)

    # Class 0: ดี (50 Samples)
    temp_0 = np.random.uniform(22.0, 29.0, 50)
    hum_0 = np.random.uniform(40.0, 65.0, 50)
    wind_0 = np.random.uniform(12.0, 25.0, 50)
    class_0 = np.zeros(50)

    # Class 1: ปานกลาง (50 Samples)
    temp_1 = np.random.uniform(30.0, 35.0, 50)
    hum_1 = np.random.uniform(66.0, 80.0, 50)
    wind_1 = np.random.uniform(6.0, 15.0, 50)
    class_1 = np.ones(50)

    # Class 2: อันตราย (50 Samples)
    temp_2 = np.random.uniform(36.0, 42.0, 50)
    hum_2 = np.random.uniform(81.0, 95.0, 50)
    wind_2 = np.random.uniform(0.5, 7.0, 50)
    class_2 = np.full(50, 2)

    # รวมเป็น 150 ตัวอย่าง (3 Features, 3 Classes)
    temps = np.concatenate([temp_0, temp_1, temp_2])
    hums = np.concatenate([hum_0, hum_1, hum_2])
    winds = np.concatenate([wind_0, wind_1, wind_2])
    y = np.concatenate([class_0, class_1, class_2])

    dataset = pd.DataFrame(
        {
            "อุณหภูมิ": temps,
            "ความชื้น": hums,
            "ความเร็วลม": winds,
            "สถานะ": y,
        }
    )

    X = dataset[["อุณหภูมิ", "ความชื้น", "ความเร็วลม"]]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    return knn, dataset


model_knn, dataset_150 = get_trained_model()

# ==========================================
# 5. ซิงค์ Session State & Callbacks
# ==========================================
if "temp_slider_key" not in st.session_state:
    st.session_state.temp_slider_key = 28.0
    st.session_state.temp_input_key = 28.0
if "hum_slider_key" not in st.session_state:
    st.session_state.hum_slider_key = 65.0
    st.session_state.hum_input_key = 65.0
if "wind_slider_key" not in st.session_state:
    st.session_state.wind_slider_key = 10.0
    st.session_state.wind_input_key = 10.0


def sync_temp_slider():
    st.session_state.temp_input_key = st.session_state.temp_slider_key


def sync_temp_input():
    st.session_state.temp_slider_key = st.session_state.temp_input_key


def sync_hum_slider():
    st.session_state.hum_input_key = st.session_state.hum_slider_key


def sync_hum_input():
    st.session_state.hum_slider_key = st.session_state.hum_input_key


def sync_wind_slider():
    st.session_state.wind_input_key = st.session_state.wind_slider_key


def sync_wind_input():
    st.session_state.wind_slider_key = st.session_state.wind_input_key


# ==========================================
# 6. แถบควบคุมด้านข้าง (Sidebar)
# ==========================================
st.sidebar.markdown("### 🎛️ ตรวจสอบสภาพอากาศ Real-Time")
st.sidebar.caption(
    f"📅 วันที่ปัจจุบัน: **{datetime.now().strftime('%d/%m/%Y')}**"
)

PROVINCES = {
    "🌐 ตรวจจับพื้นที่ปัจจุบันอัตโนมัติ": None,
    "📍 สงขลา (Songkhla)": (7.1988, 100.5951, "สงขลา"),
    "📍 กรุงเทพมหานคร (Bangkok)": (13.7563, 100.5018, "กรุงเทพมหานคร"),
    "📍 เชียงใหม่ (Chiang Mai)": (18.7883, 98.9853, "เชียงใหม่"),
    "📍 ภูเก็ต (Phuket)": (7.8804, 98.3923, "ภูเก็ต"),
    "📍 ชลบุรี / พัทยา (Chonburi)": (12.9236, 100.8825, "ชลบุรี"),
    "📍 ขอนแก่น (Khon Kaen)": (16.4322, 102.8236, "ขอนแก่น"),
}

selected_province = st.sidebar.selectbox(
    "เลือกพื้นที่ต้องการตรวจสอบ", list(PROVINCES.keys())
)

if st.sidebar.button("🔄 ดึงข้อมูลสภาพอากาศ Real-Time", use_container_width=True):
    with st.spinner("กำลังเชื่อมต่อ API ดึงข้อมูลสภาพอากาศสด..."):
        if PROVINCES[selected_province] is not None:
            lat, lon, city_label = PROVINCES[selected_province]
        else:
            lat, lon, city_label = get_user_coordinates()

        data = fetch_weather_and_air(lat, lon, city_label)

        if data:
            st.session_state.temp_slider_key = min(
                max(data["temp"], 15.0), 45.0
            )
            st.session_state.temp_input_key = st.session_state.temp_slider_key

            st.session_state.hum_slider_key = min(
                max(data["humidity"], 30.0), 100.0
            )
            st.session_state.hum_input_key = st.session_state.hum_slider_key

            st.session_state.wind_slider_key = min(
                max(data["wind"], 0.0), 30.0
            )
            st.session_state.wind_input_key = st.session_state.wind_slider_key

            st.session_state["live_pm25"] = data["pm2_5"]
            st.session_state["live_pm10"] = data["pm10"]
            st.session_state["live_aqi"] = data["us_aqi"]
            st.session_state["live_city"] = data["city"]

            st.sidebar.success(f"📍 อัปเดตข้อมูลพื้นที่: {data['city']} เรียบร้อย!")
            st.rerun()

st.sidebar.markdown("---")

# Slider & Input
st.sidebar.markdown("**🌡️ อุณหภูมิ (°C)**")
col_t1, col_t2 = st.sidebar.columns([1.3, 1])
with col_t1:
    st.slider(
        "t_s",
        15.0,
        45.0,
        step=0.1,
        key="temp_slider_key",
        on_change=sync_temp_slider,
        label_visibility="collapsed",
    )
with col_t2:
    temp = st.number_input(
        "t_i",
        15.0,
        45.0,
        step=0.1,
        key="temp_input_key",
        on_change=sync_temp_input,
        label_visibility="collapsed",
    )

st.sidebar.markdown("**💧 ความชื้นสัมพัทธ์ (%)**")
col_h1, col_h2 = st.sidebar.columns([1.3, 1])
with col_h1:
    st.slider(
        "h_s",
        30.0,
        100.0,
        step=0.1,
        key="hum_slider_key",
        on_change=sync_hum_slider,
        label_visibility="collapsed",
    )
with col_h2:
    humidity = st.number_input(
        "h_i",
        30.0,
        100.0,
        step=0.1,
        key="hum_input_key",
        on_change=sync_hum_input,
        label_visibility="collapsed",
    )

st.sidebar.markdown("**🌬️ ความเร็วลม (km/h)**")
col_w1, col_w2 = st.sidebar.columns([1.3, 1])
with col_w1:
    st.slider(
        "w_s",
        0.0,
        30.0,
        step=0.1,
        key="wind_slider_key",
        on_change=sync_wind_slider,
        label_visibility="collapsed",
    )
with col_w2:
    wind_speed = st.number_input(
        "w_i",
        0.0,
        30.0,
        step=0.1,
        key="wind_input_key",
        on_change=sync_wind_input,
        label_visibility="collapsed",
    )

st.sidebar.write("")
predict_btn = st.sidebar.button("✨ ประเมินผลคุณภาพอากาศ", type="primary")

# ==========================================
# 7. หน้าจอหลัก (Main Interface)
# ==========================================
st.title("🌤️ ระบบวิเคราะห์คุณภาพอากาศ Real-time รายพื้นที่")
st.caption(
    "ระบบดึงข้อมูลอุณหภูมิ ความชื้น ลม และค่าฝุ่น PM2.5/AQI สดตรงตามพื้นที่ | โมเดลพัฒนาจาก 150 ตัวอย่าง (3 Features, 3 Classes)"
)
st.divider()

if "live_pm25" in st.session_state:
    st.markdown(
        f"#### 📍 ข้อมูลสภาวะอากาศและมลพิษสด ณ พื้นที่: **{st.session_state.get('live_city', 'ไม่ระบุ')}**"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ฝุ่น PM2.5 สด", f"{st.session_state['live_pm25']} µg/m³")
    m2.metric("ฝุ่น PM10 สด", f"{st.session_state['live_pm10']} µg/m³")
    m3.metric("ดัชนี US AQI", f"{st.session_state['live_aqi']}")
    m4.metric(
        "เวลาอัปเดตล่าสุด", datetime.now().strftime("%H:%M:%S น.")
    )
    st.divider()

if predict_btn:
    input_df = pd.DataFrame(
        [[temp, humidity, wind_speed]],
        columns=["อุณหภูมิ", "ความชื้น", "ความเร็วลม"],
    )
    pred_code = int(model_knn.predict(input_df)[0])
    confidence = model_knn.predict_proba(input_df)[0][pred_code] * 100

    status_details = {
        0: {
            "title": "คุณภาพอากาศดีมาก 🟢 (Good)",
            "alert": "success",
            "desc": "สภาพอากาศสะอาด เหมาะแก่การทำกิจกรรมกลางแจ้งและออกกำลังกาย",
            "tag": "ดี",
        },
        1: {
            "title": "คุณภาพอากาศปานกลาง 🟡 (Moderate)",
            "alert": "warning",
            "desc": "ผู้มีภูมิแพ้หรือโรคทางเดินหายใจควรระมัดระวังและสวมหน้ากากหากต้องอยู่กลางแจ้ง",
            "tag": "ปานกลาง",
        },
        2: {
            "title": "คุณภาพอากาศอยู่ในระดับเสี่ยงอันตราย 🔴 (Unhealthy)",
            "alert": "error",
            "desc": "หลีกเลี่ยงกิจกรรมกลางแจ้งทุกชนิด ควรสวมหน้ากากป้องกันมลพิษ",
            "tag": "อันตราย",
        },
    }

    res = status_details.get(
        pred_code,
        {
            "title": "ไม่สามารถประเมินได้",
            "alert": "info",
            "desc": "-",
            "tag": "ไม่ทราบผล",
        },
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_file = "prediction_history.csv"
    new_data = pd.DataFrame(
        [
            {
                "Timestamp": now_str,
                "Temperature_C": temp,
                "Humidity_Percent": humidity,
                "WindSpeed_kmh": wind_speed,
                "Predicted_Status": res["tag"],
                "Confidence_Percent": round(confidence, 2),
            }
        ]
    )
    file_exists = os.path.exists(history_file)
    new_data.to_csv(
        history_file,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.subheader("📌 ผลการประเมินสภาวะอากาศ")
        if res["alert"] == "success":
            st.success(f"### {res['title']}")
        elif res["alert"] == "warning":
            st.warning(f"### {res['title']}")
        else:
            st.error(f"### {res['title']}")
        st.write(f"**คำแนะนำด้านสุขภาพ:** {res['desc']}")

    with col2:
        st.metric(
            label="🎯 ระดับความเชื่อมั่นโมเดล",
            value=f"{confidence:.1f}%",
            delta="ปกติ",
        )
    with col3:
        st.metric(
            label="🕒 เวลาประมวลผล",
            value=datetime.now().strftime("%H:%M:%S"),
            delta=datetime.now().strftime("%d/%m/%Y"),
        )

# ==========================================
# 8. ประวัติและการวิเคราะห์
# ==========================================
st.divider()
st.subheader("📊 รายงานสรุปสถิติและประวัติการบันทึก")

history_file = "prediction_history.csv"
if os.path.exists(history_file):
    try:
        df_history = pd.read_csv(history_file, encoding="utf-8-sig")
        if not df_history.empty:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ประวัติการประเมินทั้งหมด", f"{len(df_history)} ครั้ง")
            k2.metric(
                "อุณหภูมิเฉลี่ย", f"{df_history['Temperature_C'].mean():.1f} °C"
            )
            k3.metric(
                "ความชื้นเฉลี่ย",
                f"{df_history['Humidity_Percent'].mean():.1f} %",
            )
            k4.metric(
                "ความเร็วลมเฉลี่ย",
                f"{df_history['WindSpeed_kmh'].mean():.1f} km/h",
            )

            st.write("")
            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 กราฟวิเคราะห์แนวโน้ม",
                    "📋 รายการบันทึกประวัติ",
                    "📁 ชุดข้อมูล 150 Samples (สำหรับตรวจ)",
                ]
            )
            with tab1:
                counts = df_history["Predicted_Status"].value_counts()
                categories = ["ดี", "ปานกลาง", "อันตราย"]
                chart_values = [counts.get(cat, 0) for cat in categories]

                plt.rcdefaults()
                fig, ax = plt.subplots(figsize=(7, 2.8))
                sns.barplot(
                    x=["Good", "Moderate", "Unhealthy"],
                    y=chart_values,
                    palette=["#2ecc71", "#f1c40f", "#e74c3c"],
                    ax=ax,
                )
                ax.set_title("Air Quality Records Summary")
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)

            with tab2:
                st.dataframe(
                    df_history.tail(15).sort_values(
                        by="Timestamp", ascending=False
                    ),
                    use_container_width=True,
                )

            with tab3:
                st.write("**ตารางชุดข้อมูลสำหรับเทรนโมเดล (150 Samples, 3 Classes, 3 Features):**")
                st.dataframe(dataset_150, use_container_width=True)
    except Exception:
        pass
else:
    st.info(
        "ℹ️ กดปุ่ม **'🔄 ดึงข้อมูลสภาพอากาศ Real-Time'** ด้านซ้ายเพื่ออัปเดตข้อมูลสดเข้าสู่ระบบ"
    )