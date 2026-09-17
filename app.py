from datetime import datetime
import os
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="ระบบประเมินคุณภาพอากาศ Real-Time แม่นยำสูง",
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
# 2. ฟังก์ชันแปลงพิกัดและดึงชื่อสถานที่ (High-Reliability Geocoding)
# ==========================================
def get_exact_location_name(lat, lon):
    # พยายามดึงชื่อสถานที่ผ่าน Open-Meteo Direct Reverse Geocoding ก่อน
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={lat},{lon}&count=1&language=th&format=json"
        res = requests.get(url, timeout=3).json()
        if "results" in res and len(res["results"]) > 0:
            loc = res["results"][0]
            name = loc.get("name", "")
            admin1 = loc.get("admin1", "")
            country = loc.get("country", "")
            return f"{name} {admin1} {country}".strip()
    except Exception:
        pass

    # สำรองด้วย Nominatim API
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=th"
        headers = {"User-Agent": "HighPrecisionAirMonitor/3.0"}
        res = requests.get(url, headers=headers, timeout=3).json()
        address = res.get("address", {})

        suburb = address.get("suburb") or address.get("village") or address.get("town") or ""
        district = address.get("amphoe") or address.get("district") or address.get("county") or ""
        province = address.get("province") or address.get("state") or address.get("city") or ""

        parts = [p for p in [suburb, district, province] if p]
        if parts:
            return " ".join(parts)
    except Exception:
        pass

    return f"พิกัดแม่นยำสูง ({lat:.5f}, {lon:.5f})"

# ==========================================
# 3. ฟังก์ชันดึงมลพิษและสภาพอากาศ Real-Time
# ==========================================
def fetch_exact_weather_and_air(lat, lon):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,us_aqi"

        w_res = requests.get(weather_url, timeout=5).json().get("current", {})
        a_res = requests.get(air_url, timeout=5).json().get("current", {})

        location_name = get_exact_location_name(lat, lon)

        return {
            "location_name": location_name,
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
# 4. โมเดล Machine Learning (150 Samples, 3 Features, 3 Classes)
# ==========================================
@st.cache_resource
def get_trained_model():
    np.random.seed(42)

    temp_0 = np.random.uniform(22.0, 29.0, 50)
    hum_0 = np.random.uniform(40.0, 65.0, 50)
    wind_0 = np.random.uniform(12.0, 25.0, 50)
    class_0 = np.zeros(50)

    temp_1 = np.random.uniform(30.0, 35.0, 50)
    hum_1 = np.random.uniform(66.0, 80.0, 50)
    wind_1 = np.random.uniform(6.0, 15.0, 50)
    class_1 = np.ones(50)

    temp_2 = np.random.uniform(36.0, 42.0, 50)
    hum_2 = np.random.uniform(81.0, 95.0, 50)
    wind_2 = np.random.uniform(0.5, 7.0, 50)
    class_2 = np.full(50, 2)

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
# 5. Session State Sync
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
# 6. Sidebar & Ultra-High Precision GPS JavaScript
# ==========================================
st.sidebar.markdown("### 🎛️ ระบบดึงพิกัด GPS อุปกรณ์สด")
st.sidebar.caption(f"📅 วันที่: **{datetime.now().strftime('%d/%m/%Y')}**")

# สคริปต์ GPS พิเศษ: เปิด High Accuracy + Timeout Guard + Auto Sync
high_precision_gps_html = """
<div style="text-align:center;">
    <button onclick="getExactGPS()" style="width:100%; height:45px; background-color:#198754; color:white; font-weight:bold; border:none; border-radius:8px; cursor:pointer; font-size:14px;">
        📡 กดปุ่มนี้เพื่อค้นหาพิกัด GPS สด (ตรงจุด 100%)
    </button>
    <p id="status" style="font-size:12px; color:#6c757d; margin-top:5px;"></p>
</div>

<script>
function getExactGPS() {
  const status = document.getElementById('status');
  status.innerText = "กำลังค้นหาสัญญาณดาวเทียม GPS...";

  if (navigator.geolocation) {
    const options = {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 0
    };

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        status.innerText = "ส่งค่าพิกัดสำเร็จ!";
        
        // ส่งพิกัดเข้า Streamlit URL Query Parameters
        const url = new URL(window.location.href);
        url.searchParams.set('lat', lat);
        url.searchParams.set('lon', lon);
        window.parent.location.href = url.href;
      },
      (error) => {
        status.innerText = "ข้อผิดพลาด: " + error.message + " (โปรดอนุญาตให้เบราว์เซอร์ใช้ Location)";
      },
      options
    );
  } else {
    status.innerText = "เบราว์เซอร์ไม่รองรับ GPS";
  }
}
</script>
"""
components.html(high_precision_gps_html, height=80)

query_params = st.query_params
gps_lat = query_params.get("lat", None)
gps_lon = query_params.get("lon", None)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 ทางเลือก: ป้อนพิกัดตรงด้วยตัวเอง")

if gps_lat and gps_lon:
    target_lat = float(gps_lat)
    target_lon = float(gps_lon)
    st.sidebar.success(f"🎯 พิกัด GPS สด: {target_lat:.5f}, {target_lon:.5f}")
else:
    st.sidebar.info("💡 หากไม่กดปุ่ม GPS สดด้านบน สามารถป้อนพิกัดด้านล่างนี้ได้:")
    target_lat = st.sidebar.number_input("ละติจูด (Latitude)", value=13.7563, format="%.5f")
    target_lon = st.sidebar.number_input("ลองจิจูด (Longitude)", value=100.5018, format="%.5f")

if st.sidebar.button("🔄 อัปเดตสภาพอากาศ ณ พิกัดปัจจุบัน", use_container_width=True):
    with st.spinner("กำลังดึงข้อมูลสภาพอากาศตรงจุด..."):
        data = fetch_exact_weather_and_air(target_lat, target_lon)

        if data:
            st.session_state.temp_slider_key = min(max(data["temp"], 15.0), 45.0)
            st.session_state.temp_input_key = st.session_state.temp_slider_key

            st.session_state.hum_slider_key = min(max(data["humidity"], 30.0), 100.0)
            st.session_state.hum_input_key = st.session_state.hum_slider_key

            st.session_state.wind_slider_key = min(max(data["wind"], 0.0), 50.0)
            st.session_state.wind_input_key = st.session_state.wind_slider_key

            st.session_state["live_pm25"] = data["pm2_5"]
            st.session_state["live_pm10"] = data["pm10"]
            st.session_state["live_aqi"] = data["us_aqi"]
            st.session_state["live_city"] = data["location_name"]

            st.sidebar.success(f"📍 สภาพอากาศ: {data['location_name']}")
            st.rerun()

# แถบควบคุมตัวแปร
st.sidebar.markdown("---")
st.sidebar.markdown("**🌡️ อุณหภูมิ (°C)**")
col_t1, col_t2 = st.sidebar.columns([1.3, 1])
with col_t1:
    st.slider("t_s", 15.0, 45.0, step=0.1, key="temp_slider_key", on_change=sync_temp_slider, label_visibility="collapsed")
with col_t2:
    temp = st.number_input("t_i", 15.0, 45.0, step=0.1, key="temp_input_key", on_change=sync_temp_input, label_visibility="collapsed")

st.sidebar.markdown("**💧 ความชื้นสัมพัทธ์ (%)**")
col_h1, col_h2 = st.sidebar.columns([1.3, 1])
with col_h1:
    st.slider("h_s", 30.0, 100.0, step=0.1, key="hum_slider_key", on_change=sync_hum_slider, label_visibility="collapsed")
with col_h2:
    humidity = st.number_input("h_i", 30.0, 100.0, step=0.1, key="hum_input_key", on_change=sync_hum_input, label_visibility="collapsed")

st.sidebar.markdown("**🌬️ ความเร็วลม (km/h)**")
col_w1, col_w2 = st.sidebar.columns([1.3, 1])
with col_w1:
    st.slider("w_s", 0.0, 50.0, step=0.1, key="wind_slider_key", on_change=sync_wind_slider, label_visibility="collapsed")
with col_w2:
    wind_speed = st.number_input("w_i", 0.0, 50.0, step=0.1, key="wind_input_key", on_change=sync_wind_input, label_visibility="collapsed")

st.sidebar.write("")
predict_btn = st.sidebar.button("✨ ประเมินผลคุณภาพอากาศ", type="primary")

# ==========================================
# 7. หน้าจอแสดงผลหลัก
# ==========================================
st.title("🌤️ ระบบวิเคราะห์คุณภาพอากาศ Real-Time แม่นยำสูง")
st.caption("ดึงพิกัดจากชิป GPS บนอุปกรณ์ตรงจุด เพื่อประเมินค่าฝุ่นและเข้าโมเดล Machine Learning (150 Samples, 3 Features, 3 Classes)")
st.divider()

if "live_pm25" in st.session_state:
    st.markdown(f"#### 📍 ข้อมูลสภาวะอากาศและฝุ่น ณ ตำแหน่ง: **{st.session_state.get('live_city', 'ไม่ระบุ')}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ฝุ่น PM2.5 สด", f"{st.session_state['live_pm25']} µg/m³")
    m2.metric("ฝุ่น PM10 สด", f"{st.session_state['live_pm10']} µg/m³")
    m3.metric("ดัชนี US AQI", f"{st.session_state['live_aqi']}")
    m4.metric("เวลาอัปเดต", datetime.now().strftime("%H:%M:%S น."))
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
            "desc": "สภาพอากาศสะอาด เหมาะแก่การทำกิจกรรมกลางแจ้งและการออกกำลังกาย",
            "tag": "ดี",
        },
        1: {
            "title": "คุณภาพอากาศปานกลาง 🟡 (Moderate)",
            "alert": "warning",
            "desc": "ผู้มีภูมิแพ้หรือโรคทางเดินหายใจควรสวมหน้ากากอนามัยเมื่ออยู่กลางแจ้ง",
            "tag": "ปานกลาง",
        },
        2: {
            "title": "คุณภาพอากาศอยู่ในระดับเสี่ยงอันตราย 🔴 (Unhealthy)",
            "alert": "error",
            "desc": "หลีกเลี่ยงกิจกรรมกลางแจ้งทุกชนิด และสวมหน้ากากป้องกันมลพิษ PM2.5 ทันที",
            "tag": "อันตราย",
        },
    }

    res = status_details.get(pred_code, {"title": "ไม่สามารถประเมินได้", "alert": "info", "desc": "-", "tag": "ไม่ทราบผล"})

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
    new_data.to_csv(history_file, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")

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
        st.metric(label="🎯 ความเชื่อมั่นโมเดล ML", value=f"{confidence:.1f}%", delta="ปกติ")
    with col3:
        st.metric(label="🕒 เวลาประมวลผล", value=datetime.now().strftime("%H:%M:%S"), delta=datetime.now().strftime("%d/%m/%Y"))

# ==========================================
# 8. รายงานสรุปส่งอาจารย์
# ==========================================
st.divider()
st.subheader("📊 ตารางชุดข้อมูล 150 Samples และประวัติการประเมิน")

tab1, tab2 = st.tabs(["📁 ชุดข้อมูล 150 Samples (สำหรับส่งอาจารย์)", "📋 บันทึกประวัติการวิเคราะห์"])

with tab1:
    st.write("**ตาราง Dataset (3 Features: อุณหภูมิ, ความชื้น, ความเร็วลม | 3 Classes: ดี, ปานกลาง, อันตราย | 150 ตัวอย่าง):**")
    st.dataframe(dataset_150, use_container_width=True)

with tab2:
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file, encoding="utf-8-sig")
        st.dataframe(df_history.tail(15).sort_values(by="Timestamp", ascending=False), use_container_width=True)
    else:
        st.info("ยังไม่มีประวัติการบันทึกข้อมูล")