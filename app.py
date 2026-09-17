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
    page_title="ระบบวิเคราะห์คุณภาพอากาศ Real-Time รายอำเภอ | Air Quality Intelligence",
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
# 2. ฐานข้อมูล 77 จังหวัดในประเทศไทย (พิกัดศูนย์กลางจังหวัด)
# ==========================================
THAI_PROVINCES = {
    "กรุงเทพมหานคร": (13.7563, 100.5018),
    "กระบี่": (8.0863, 98.9063),
    "กาญจนบุรี": (14.0228, 99.5328),
    "กาฬสินธุ์": (16.4322, 103.5065),
    "กำแพงเพชร": (16.4828, 99.5228),
    "ขอนแก่น": (16.4322, 102.8236),
    "จันทบุรี": (12.6114, 102.1039),
    "ฉะเชิงเทรา": (13.6904, 101.0780),
    "ชลบุรี": (13.3611, 100.9847),
    "ชัยนาท": (15.1852, 100.1252),
    "ชัยภูมิ": (15.8105, 102.0288),
    "ชุมพร": (10.4930, 99.1800),
    "เชียงราย": (19.9105, 99.8406),
    "เชียงใหม่": (18.7883, 98.9853),
    "ตรัง": (7.5563, 99.6114),
    "ตราด": (12.2428, 102.5175),
    "ตาก": (16.8839, 99.1258),
    "นครนายก": (14.2069, 101.2131),
    "นครปฐม": (13.8196, 100.0444),
    "นครพนม": (17.3920, 104.7696),
    "นครราชสีมา": (14.9799, 102.0978),
    "นครศรีธรรมราช": (8.4304, 99.9631),
    "นครสวรรค์": (15.7023, 100.1372),
    "นนทบุรี": (13.8591, 100.5217),
    "นราธิวาส": (6.4255, 101.8253),
    "น่าน": (18.7830, 100.7782),
    "บึงกาฬ": (18.3617, 103.6464),
    "บุรีรัมย์": (14.9930, 103.1029),
    "ปทุมธานี": (14.0208, 100.5250),
    "ประจวบคีรีขันธ์": (11.8124, 99.7973),
    "ปราจีนบุรี": (14.0509, 101.3717),
    "ปัตตานี": (6.8663, 101.2501),
    "พระนครศรีอยุธยา": (14.3532, 100.5684),
    "พังงา": (8.4501, 98.5255),
    "พัทลุง": (7.6167, 100.0833),
    "พิจิตร": (16.4422, 100.3489),
    "พิษณุโลก": (16.8211, 100.2658),
    "เพชรบุรี": (13.1069, 99.9447),
    "เพชรบูรณ์": (16.4189, 101.1592),
    "แพร่": (18.1446, 100.1403),
    "พะเยา": (19.1658, 99.9019),
    "ภูเก็ต": (7.8804, 98.3923),
    "มหาสารคาม": (16.1852, 103.3006),
    "มุกดาหาร": (16.5453, 104.7233),
    "แม่ฮ่องสอน": (19.3021, 97.9654),
    "ยโสธร": (15.7924, 104.1453),
    "ยะลา": (6.5411, 101.2804),
    "ร้อยเอ็ด": (16.0538, 103.6520),
    "ระนอง": (9.9658, 98.6347),
    "ระยอง": (12.6814, 101.2814),
    "ราชบุรี": (13.5373, 99.8169),
    "ลพบุรี": (14.7995, 100.6534),
    "ลำปาง": (18.2888, 99.4923),
    "ลำพูน": (18.5744, 99.0087),
    "เลย": (17.4860, 101.7223),
    "ศรีสะเกษ": (15.1186, 104.3220),
    "สกลนคร": (17.1681, 104.1486),
    "สงขลา": (7.1988, 100.5951),
    "สตูล": (6.6238, 100.0674),
    "สมุทรปราการ": (13.5991, 100.5998),
    "สมุทรสงคราม": (13.4098, 100.0023),
    "สมุทรสาคร": (13.5475, 100.2744),
    "สระแก้ว": (13.8240, 102.0722),
    "สระบุรี": (14.5289, 100.9108),
    "สิงห์บุรี": (14.8936, 100.3967),
    "สุโขทัย": (17.0078, 99.8230),
    "สุพรรณบุรี": (14.4742, 100.1172),
    "สุราษฎร์ธานี": (9.1382, 99.3217),
    "สุรินทร์": (14.8818, 103.4936),
    "หนองคาย": (17.8783, 102.7413),
    "หนองบัวลำภู": (17.2038, 102.4403),
    "อ่างทอง": (14.5896, 100.4550),
    "อุดรธานี": (17.4156, 102.7859),
    "อุทัยธานี": (15.3831, 100.0247),
    "อุตรดิตถ์": (17.6256, 100.0992),
    "อุบลราชธานี": (15.2287, 104.8594),
    "อำนาจเจริญ": (15.8582, 104.6258),
}

# ==========================================
# 3. ฟังก์ชันระบุชื่อ อำเภอ/ตำบล จากพิกัดละติจูด-ลองจิจูด ย้อนกลับ (Reverse Geocoding)
# ==========================================
def get_detailed_location(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=th"
        headers = {"User-Agent": "AirQualityApp_Thailand/1.0"}
        res = requests.get(url, headers=headers, timeout=4).json()
        address = res.get("address", {})

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
        suburb = address.get("suburb") or address.get("village") or ""

        location_parts = [p for p in [suburb, district, province] if p]
        if location_parts:
            return " ".join(location_parts)
        return f"พิกัด ({lat:.4f}, {lon:.4f})"
    except Exception:
        return f"พิกัด ({lat:.4f}, {lon:.4f})"

# ==========================================
# 4. ฟังก์ชันดึงมลพิษ & สภาพอากาศสดจาก Open-Meteo API
# ==========================================
def fetch_weather_and_air_exact(lat, lon):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,us_aqi"

        w_res = requests.get(weather_url, timeout=4).json().get("current", {})
        a_res = requests.get(air_url, timeout=4).json().get("current", {})

        location_label = get_detailed_location(lat, lon)

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
# 5. โมเดล ML (3 Features, 3 Classes, 150 Samples)
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
# 6. ซิงค์ Session State & Callbacks
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
# 7. แถบควบคุมด้านข้าง (Sidebar Controls)
# ==========================================
st.sidebar.markdown("### 🎛️ ระบบค้นหาตำแหน่งรายอำเภอ")
st.sidebar.caption(
    f"📅 วันที่ปัจจุบัน: **{datetime.now().strftime('%d/%m/%Y')}**"
)

search_mode = st.sidebar.radio(
    "รูปแบบการระบุตำแหน่ง:",
    ["เลือกจาก 77 จังหวัดในไทย", "ระบุพิกัดตรง (Latitude/Longitude)"],
)

target_lat, target_lon = 7.1988, 100.5951  # Default: สงขลา

if search_mode == "เลือกจาก 77 จังหวัดในไทย":
    selected_prov = st.sidebar.selectbox(
        "เลือกจังหวัดที่ต้องการตรวจสอบ:",
        options=sorted(list(THAI_PROVINCES.keys())),
        index=sorted(list(THAI_PROVINCES.keys())).index("สงขลา"),
    )
    target_lat, target_lon = THAI_PROVINCES[selected_prov]
else:
    target_lat = st.sidebar.number_input("Latitude (ละติจูด)", value=7.1988, format="%.5f")
    target_lon = st.sidebar.number_input("Longitude (ลองจิจูด)", value=100.5951, format="%.5f")

if st.sidebar.button("🔄 ดึงข้อมูลอากาศ Real-Time แบบเจาะจงจุด", use_container_width=True):
    with st.spinner("กำลังระบุอำเภอ/ตำบล และดึงสภาพอากาศสดจาก API..."):
        data = fetch_weather_and_air_exact(target_lat, target_lon)

        if data:
            st.session_state.temp_slider_key = min(max(data["temp"], 15.0), 45.0)
            st.session_state.temp_input_key = st.session_state.temp_slider_key

            st.session_state.hum_slider_key = min(max(data["humidity"], 30.0), 100.0)
            st.session_state.hum_input_key = st.session_state.hum_slider_key

            st.session_state.wind_slider_key = min(max(data["wind"], 0.0), 30.0)
            st.session_state.wind_input_key = st.session_state.wind_slider_key

            st.session_state["live_pm25"] = data["pm2_5"]
            st.session_state["live_pm10"] = data["pm10"]
            st.session_state["live_aqi"] = data["us_aqi"]
            st.session_state["live_city"] = data["city"]

            st.sidebar.success("📍 ตรวจพบพื้นที่เรียบร้อย!")
            st.rerun()

st.sidebar.markdown("---")

# Slider & Input Control
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
    st.slider("w_s", 0.0, 30.0, step=0.1, key="wind_slider_key", on_change=sync_wind_slider, label_visibility="collapsed")
with col_w2:
    wind_speed = st.number_input("w_i", 0.0, 30.0, step=0.1, key="wind_input_key", on_change=sync_wind_input, label_visibility="collapsed")

st.sidebar.write("")
predict_btn = st.sidebar.button("✨ ประเมินผลคุณภาพอากาศ", type="primary")

# ==========================================
# 8. หน้าจอหลัก (Main Dashboard)
# ==========================================
st.title("🌤️ ระบบวิเคราะห์คุณภาพอากาศ Real-time รายอำเภอ/จังหวัด")
st.caption("ระบบระบุ อำเภอ/ตำบล จากพิกัดพิกัดละติจูด-ลองจิจูดและดึงสภาพอากาศสดส่งเข้าโมเดล ML (3 Features, 3 Classes, 150 Samples)")
st.divider()

if "live_pm25" in st.session_state:
    st.markdown(
        f"#### 📍 ข้อมูลสภาวะอากาศและมลพิษสด ณ พื้นที่: **{st.session_state.get('live_city', 'ไม่ระบุ')}**"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ฝุ่น PM2.5 สด", f"{st.session_state['live_pm25']} µg/m³")
    m2.metric("ฝุ่น PM10 สด", f"{st.session_state['live_pm10']} µg/m³")
    m3.metric("ดัชนี US AQI", f"{st.session_state['live_aqi']}")
    m4.metric("เวลาอัปเดตล่าสุด", datetime.now().strftime("%H:%M:%S น."))
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
# 9. รายงานและประวัติ
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
            k2.metric("อุณหภูมิเฉลี่ย", f"{df_history['Temperature_C'].mean():.1f} °C")
            k3.metric("ความชื้นเฉลี่ย", f"{df_history['Humidity_Percent'].mean():.1f} %")
            k4.metric("ความเร็วลมเฉลี่ย", f"{df_history['WindSpeed_kmh'].mean():.1f} km/h")

            st.write("")
            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 กราฟวิเคราะห์แนวโน้ม",
                    "📋 รายการบันทึกประวัติ",
                    "📁 ชุดข้อมูล 150 Samples (สำหรับส่งอาจารย์)",
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
    st.info("ℹ️ กดปุ่ม **'🔄 ดึงข้อมูลอากาศ Real-Time แบบเจาะจงจุด'** ด้านซ้ายเพื่อเริ่มดึงสภาพอากาศสด")