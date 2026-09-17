from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="ระบบวิเคราะห์คุณภาพอากาศ Real-Time | รัตภูมิ สงขลา",
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
# 2. ฐานข้อมูล 77 จังหวัด (ค่าสำรอง)
# ==========================================
THAI_PROVINCES = {
    "สงขลา (รัตภูมิ / พิกัดศูนย์กลาง)": (7.1350, 100.2783),
    "สงขลา (ตัวเมือง)": (7.1988, 100.5951),
    "กรุงเทพมหานคร": (13.7563, 100.5018),
    "เชียงใหม่": (18.7883, 98.9853),
    "ภูเก็ต": (7.8804, 98.3923),
    "ขอนแก่น": (16.4322, 102.8236),
    "ชลบุรี": (13.3611, 100.9847),
}

# ==========================================
# 3. Reverse Geocoding (ค้นหา ตำบล/อำเภอ/จังหวัด จาก พิกัด GPS สด)
# ==========================================
def get_exact_address(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=th"
        headers = {"User-Agent": "RattaphumAirCheck_App/1.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        address = res.get("address", {})

        suburb = (
            address.get("suburb")
            or address.get("village")
            or address.get("town")
            or ""
        )
        district = (
            address.get("amphoe")
            or address.get("district")
            or address.get("county")
            or ""
        )
        province = (
            address.get("province")
            or address.get("state")
            or address.get("city")
            or ""
        )

        parts = [p for p in [suburb, district, province] if p]
        if parts:
            return " ".join(parts)
        return f"พิกัดดาวเทียม ({lat:.4f}, {lon:.4f})"
    except Exception:
        return f"พิกัดดาวเทียม ({lat:.4f}, {lon:.4f})"


# ==========================================
# 4. ดึงข้อมูลมลพิษระดับตาราง Grid 1 km จาก Open-Meteo
# ==========================================
def fetch_exact_weather_and_air(lat, lon):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,us_aqi"

        w_res = requests.get(weather_url, timeout=5).json().get("current", {})
        a_res = requests.get(air_url, timeout=5).json().get("current", {})

        location_name = get_exact_address(lat, lon)

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
# 5. โมเดล ML ตามโจทย์อาจารย์ (3 Features, 3 Classes, 150 Samples)
# ==========================================
@st.cache_resource
def get_trained_model():
    np.random.seed(42)

    # Class 0: ดี (50 ตัวอย่าง)
    temp_0 = np.random.uniform(22.0, 29.0, 50)
    hum_0 = np.random.uniform(40.0, 65.0, 50)
    wind_0 = np.random.uniform(12.0, 25.0, 50)
    class_0 = np.zeros(50)

    # Class 1: ปานกลาง (50 ตัวอย่าง)
    temp_1 = np.random.uniform(30.0, 35.0, 50)
    hum_1 = np.random.uniform(66.0, 80.0, 50)
    wind_1 = np.random.uniform(6.0, 15.0, 50)
    class_1 = np.ones(50)

    # Class 2: อันตราย (50 ตัวอย่าง)
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
# 6. Session State & Callbacks
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
# 7. Sidebar Controls & GPS Device Hook
# ==========================================
st.sidebar.markdown("### 🎛️ ตรวจสอบพิกัดอุปกรณ์สด (GPS)")
st.sidebar.caption(
    f"📅 วันที่ปัจจุบัน: **{datetime.now().strftime('%d/%m/%Y')}**"
)

# JavaScript สั่งดึงพิกัดจากชิป GPS บนมือถือ/คอมพิวเตอร์
gps_html = """
<script>
function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(showPosition, showError, {enableHighAccuracy: true});
  } else {
    alert("เบราว์เซอร์ไม่รองรับ GPS");
  }
}
function showPosition(position) {
  const lat = position.coords.latitude;
  const lon = position.coords.longitude;
  const urlParams = new URLSearchParams(window.location.search);
  window.parent.postMessage({
    type: 'streamlit:setQueryParams',
    queryParams: {lat: lat, lon: lon}
  }, '*');
}
function showError(error) {
  console.log("GPS Error: " + error.message);
}
</script>
<button onclick="getLocation()" style="width:100%; height:42px; background-color:#28a745; color:white; font-weight:bold; border:none; border-radius:8px; cursor:pointer;">
📡 กดเพื่อดึงพิกัด GPS จริงจากดาวเทียม
</button>
"""
components.html(gps_html, height=50)

# เช็คว่ามี query param จาก GPS หรือไม่
query_params = st.query_params
gps_lat = query_params.get("lat", None)
gps_lon = query_params.get("lon", None)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 หรือเลือกพื้นที่ / ป้อนพิกัดมือ")

selected_mode = st.sidebar.radio(
    "โหมดระบุพื้นที่:",
    ["พิกัดจากดาวเทียม GPS", "เลือกอำเภอ/จังหวัดในระบบ", "กรอก ละติจูด/ลองจิจูด เอง"],
)

target_lat, target_lon = 7.1350, 100.2783  # ค่าเริ่มต้น: อ.รัตภูมิ สงขลา

if selected_mode == "พิกัดจากดาวเทียม GPS":
    if gps_lat and gps_lon:
        target_lat = float(gps_lat)
        target_lon = float(gps_lon)
        st.sidebar.info(f"🎯 ได้รับ GPS สด: {target_lat:.4f}, {target_lon:.4f}")
    else:
        st.sidebar.warning(
            "⚠️ กรุณากดปุ่มสีเขียวด้านบนเพื่อส่งค่า GPS จากอุปกรณ์เข้าสู่ระบบ"
        )
elif selected_mode == "เลือกอำเภอ/จังหวัดในระบบ":
    prov_choice = st.sidebar.selectbox(
        "เลือกพื้นที่:", list(THAI_PROVINCES.keys())
    )
    target_lat, target_lon = THAI_PROVINCES[prov_choice]
else:
    target_lat = st.sidebar.number_input(
        "Latitude (รัตภูมิ ~ 7.1350)", value=7.1350, format="%.5f"
    )
    target_lon = st.sidebar.number_input(
        "Longitude (รัตภูมิ ~ 100.2783)", value=100.2783, format="%.5f"
    )

if st.sidebar.button("🔄 ดึงข้อมูลสภาพอากาศจุดนี้แบบตรงเป๊ะ", use_container_width=True):
    with st.spinner("กำลังเชื่อมต่อดาวเทียมและสถานีวัดอากาศใกล้ที่สุด..."):
        data = fetch_exact_weather_and_air(target_lat, target_lon)

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
            st.session_state["live_city"] = data["location_name"]

            st.sidebar.success(f"📍 ระบุตำแหน่ง: {data['location_name']}")
            st.rerun()

# Controls
st.sidebar.markdown("---")
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
# 8. Main Dashboard
# ==========================================
st.title("🌤️ ระบบวิเคราะห์คุณภาพอากาศแม่นยำสูง (อ.รัตภูมิ / ระบุพื้นที่สด)")
st.caption(
    "ใช้พิกัดชิป GPS จริงเพื่อดึงข้อมูลฝุ่น PM2.5 และสภาพอากาศระดับตำบล ส่งประมวลผลต่อโมเดล ML (3 Features, 3 Classes, 150 Samples)"
)
st.divider()

if "live_pm25" in st.session_state:
    st.markdown(
        f"#### 📍 ตำแหน่งพิกัดปัจจุบัน: **{st.session_state.get('live_city', 'ไม่ระบุ')}**"
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
            "desc": "อากาศสะอาด ปลอดภัยสำหรับชาวอำเภอรัตภูมิในการทำกิจกรรม outdoor",
            "tag": "ดี",
        },
        1: {
            "title": "คุณภาพอากาศปานกลาง 🟡 (Moderate)",
            "alert": "warning",
            "desc": "ผู้มีโรคประจำตัวหรือภูมิแพ้ควรสวมหน้ากากอนามัยเมื่ออยู่กลางแจ้ง",
            "tag": "ปานกลาง",
        },
        2: {
            "title": "คุณภาพอากาศอยู่ในระดับเสี่ยงอันตราย 🔴 (Unhealthy)",
            "alert": "error",
            "desc": "อันตรายจากมลพิษ! หลีกเลี่ยงกิจกรรมกลางแจ้ง และสวมหน้ากาก PM2.5 ทันที",
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
        st.write(f"**คำแนะนำสุขภาพ:** {res['desc']}")

    with col2:
        st.metric(
            label="🎯 ความเชื่อมั่นโมเดล ML",
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
# 9. รายงานสำหรับส่งอาจารย์
# ==========================================
st.divider()
st.subheader("📊 ตารางชุดข้อมูล 150 ตัวอย่าง และประวัติการวิเคราะห์")

tab1, tab2 = st.tabs(
    ["📁 ชุดข้อมูล 150 Samples (ตรงโจทย์อาจารย์)", "📋 บันทึกประวัติการวิเคราะห์"]
)

with tab1:
    st.write(
        "**โครงสร้าง Dataset: 3 Features (อุณหภูมิ, ความชื้น, ความเร็วลม) | 3 Classes (ดี, ปานกลาง, อันตราย) | 150 Samples**"
    )
    st.dataframe(dataset_150, use_container_width=True)

with tab2:
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file, encoding="utf-8-sig")
        st.dataframe(
            df_history.tail(10).sort_values(by="Timestamp", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("ยังไม่มีประวัติการกดประเมิน")