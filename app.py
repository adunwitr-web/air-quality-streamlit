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
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="Global Air Quality Intelligence System",
    page_icon="🌍",
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
# 2. Reverse Geocoding (Global Location Lookup)
# ==========================================
def get_global_location(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "GlobalAirQualitySystem/2.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        address = res.get("address", {})

        city = (
            address.get("city")
            or address.get("town")
            or address.get("suburb")
            or address.get("county")
            or ""
        )
        state = address.get("state") or address.get("region") or ""
        country = address.get("country") or ""

        parts = [p for p in [city, state, country] if p]
        if parts:
            return ", ".join(parts)
        return f"Coordinates ({lat:.4f}, {lon:.4f})"
    except Exception:
        return f"Coordinates ({lat:.4f}, {lon:.4f})"


# ==========================================
# 3. Global Weather & Air Quality API
# ==========================================
def fetch_global_weather_and_air(lat, lon):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,us_aqi"

        w_res = requests.get(weather_url, timeout=5).json().get("current", {})
        a_res = requests.get(air_url, timeout=5).json().get("current", {})

        location_name = get_global_location(lat, lon)

        return {
            "location_name": location_name,
            "temp": float(w_res.get("temperature_2m", 25.0)),
            "humidity": float(w_res.get("relative_humidity_2m", 50.0)),
            "wind": float(w_res.get("wind_speed_10m", 10.0)),
            "pm2_5": float(a_res.get("pm2_5", 0.0)),
            "pm10": float(a_res.get("pm10", 0.0)),
            "us_aqi": int(a_res.get("us_aqi", 0)),
        }
    except Exception:
        return None


# ==========================================
# 4. ML Model Training (150 Samples, 3 Features, 3 Classes)
# ==========================================
@st.cache_resource
def get_trained_model():
    np.random.seed(42)

    # Class 0: Good / Optimal (50 samples)
    temp_0 = np.random.uniform(18.0, 28.0, 50)
    hum_0 = np.random.uniform(30.0, 60.0, 50)
    wind_0 = np.random.uniform(12.0, 30.0, 50)
    class_0 = np.zeros(50)

    # Class 1: Moderate (50 samples)
    temp_1 = np.random.uniform(29.0, 34.0, 50)
    hum_1 = np.random.uniform(61.0, 75.0, 50)
    wind_1 = np.random.uniform(6.0, 15.0, 50)
    class_1 = np.ones(50)

    # Class 2: Unhealthy / Severe (50 samples)
    temp_2 = np.random.uniform(35.0, 45.0, 50)
    hum_2 = np.random.uniform(76.0, 95.0, 50)
    wind_2 = np.random.uniform(0.5, 7.0, 50)
    class_2 = np.full(50, 2)

    temps = np.concatenate([temp_0, temp_1, temp_2])
    hums = np.concatenate([hum_0, hum_1, hum_2])
    winds = np.concatenate([wind_0, wind_1, wind_2])
    y = np.concatenate([class_0, class_1, class_2])

    dataset = pd.DataFrame(
        {
            "Temperature_C": temps,
            "Humidity_Pct": hums,
            "WindSpeed_kmh": winds,
            "AirQuality_Class": y,
        }
    )

    X = dataset[["Temperature_C", "Humidity_Pct", "WindSpeed_kmh"]]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    return knn, dataset


model_knn, dataset_150 = get_trained_model()

# ==========================================
# 5. Session State Synchronization
# ==========================================
if "temp_slider_key" not in st.session_state:
    st.session_state.temp_slider_key = 25.0
    st.session_state.temp_input_key = 25.0
if "hum_slider_key" not in st.session_state:
    st.session_state.hum_slider_key = 50.0
    st.session_state.hum_input_key = 50.0
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
# 6. Sidebar & Geolocation Component
# ==========================================
st.sidebar.markdown("### 🌐 Global Location Selector")
st.sidebar.caption(f"📅 UTC Date: **{datetime.utcnow().strftime('%Y-%m-%d')}**")

# Standard Browser HTML5 Geolocation Hook
gps_html = """
<script>
function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(showPosition, showError, {enableHighAccuracy: true});
  } else {
    alert("Geolocation is not supported by this browser.");
  }
}
function showPosition(position) {
  const lat = position.coords.latitude;
  const lon = position.coords.longitude;
  window.parent.postMessage({
    type: 'streamlit:setQueryParams',
    queryParams: {lat: lat, lon: lon}
  }, '*');
}
function showError(error) {
  console.log("GPS Error: " + error.message);
}
</script>
<button onclick="getLocation()" style="width:100%; height:42px; background-color:#198754; color:white; font-weight:bold; border:none; border-radius:8px; cursor:pointer;">
📡 Detect Live GPS Location
</button>
"""
components.html(gps_html, height=50)

query_params = st.query_params
gps_lat = query_params.get("lat", None)
gps_lon = query_params.get("lon", None)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Location Mode")

selected_mode = st.sidebar.radio(
    "Select Positioning Method:",
    ["Browser Auto-GPS", "Global Capital Cities", "Manual Coordinates Input"],
)

# World Major Cities Sample (Global Standards)
GLOBAL_CITIES = {
    "London, United Kingdom": (51.5074, -0.1278),
    "New York, United States": (40.7128, -74.0060),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Paris, France": (48.8566, 2.3522),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Singapore": (1.3521, 103.8198),
}

target_lat, target_lon = 0.0, 0.0  # Neutral Default (Equator)

if selected_mode == "Browser Auto-GPS":
    if gps_lat and gps_lon:
        target_lat = float(gps_lat)
        target_lon = float(gps_lon)
        st.sidebar.info(
            f"🎯 Live Signal: {target_lat:.4f}, {target_lon:.4f}"
        )
    else:
        st.sidebar.warning(
            "⚠️ Click the green button above to grant browser location permission."
        )
elif selected_mode == "Global Capital Cities":
    city_choice = st.sidebar.selectbox(
        "Choose City:", list(GLOBAL_CITIES.keys())
    )
    target_lat, target_lon = GLOBAL_CITIES[city_choice]
else:
    target_lat = st.sidebar.number_input(
        "Latitude (-90.0 to 90.0)", value=13.7563, format="%.4f"
    )
    target_lon = st.sidebar.number_input(
        "Longitude (-180.0 to 180.0)", value=100.5018, format="%.4f"
    )

if st.sidebar.button("🔄 Sync Live Weather & Air Data", use_container_width=True):
    with st.spinner("Fetching global meteorological data..."):
        data = fetch_global_weather_and_air(target_lat, target_lon)

        if data:
            st.session_state.temp_slider_key = min(
                max(data["temp"], -20.0), 50.0
            )
            st.session_state.temp_input_key = st.session_state.temp_slider_key

            st.session_state.hum_slider_key = min(
                max(data["humidity"], 0.0), 100.0
            )
            st.session_state.hum_input_key = st.session_state.hum_slider_key

            st.session_state.wind_slider_key = min(
                max(data["wind"], 0.0), 100.0
            )
            st.session_state.wind_input_key = st.session_state.wind_slider_key

            st.session_state["live_pm25"] = data["pm2_5"]
            st.session_state["live_pm10"] = data["pm10"]
            st.session_state["live_aqi"] = data["us_aqi"]
            st.session_state["live_city"] = data["location_name"]

            st.sidebar.success(f"📍 Synchronized: {data['location_name']}")
            st.rerun()

# Environmental Feature Controls
st.sidebar.markdown("---")
st.sidebar.markdown("**🌡️ Temperature (°C)**")
col_t1, col_t2 = st.sidebar.columns([1.3, 1])
with col_t1:
    st.slider(
        "t_s",
        -20.0,
        50.0,
        step=0.1,
        key="temp_slider_key",
        on_change=sync_temp_slider,
        label_visibility="collapsed",
    )
with col_t2:
    temp = st.number_input(
        "t_i",
        -20.0,
        50.0,
        step=0.1,
        key="temp_input_key",
        on_change=sync_temp_input,
        label_visibility="collapsed",
    )

st.sidebar.markdown("**💧 Relative Humidity (%)**")
col_h1, col_h2 = st.sidebar.columns([1.3, 1])
with col_h1:
    st.slider(
        "h_s",
        0.0,
        100.0,
        step=0.1,
        key="hum_slider_key",
        on_change=sync_hum_slider,
        label_visibility="collapsed",
    )
with col_h2:
    humidity = st.number_input(
        "h_i",
        0.0,
        100.0,
        step=0.1,
        key="hum_input_key",
        on_change=sync_hum_input,
        label_visibility="collapsed",
    )

st.sidebar.markdown("**🌬️ Wind Speed (km/h)**")
col_w1, col_w2 = st.sidebar.columns([1.3, 1])
with col_w1:
    st.slider(
        "w_s",
        0.0,
        100.0,
        step=0.1,
        key="wind_slider_key",
        on_change=sync_wind_slider,
        label_visibility="collapsed",
    )
with col_w2:
    wind_speed = st.number_input(
        "w_i",
        0.0,
        100.0,
        step=0.1,
        key="wind_input_key",
        on_change=sync_wind_input,
        label_visibility="collapsed",
    )

st.sidebar.write("")
predict_btn = st.sidebar.button("✨ Evaluate Air Quality Index", type="primary")

# ==========================================
# 7. Main Dashboard Interface
# ==========================================
st.title("🌍 Global Air Quality Intelligence System")
st.caption(
    "Real-Time Meteorological Data Sync | Machine Learning Classification Model (150 Samples, 3 Features, 3 Classes)"
)
st.divider()

if "live_pm25" in st.session_state:
    st.markdown(
        f"#### 📍 Target Location: **{st.session_state.get('live_city', 'Unspecified')}**"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live PM2.5", f"{st.session_state['live_pm25']} µg/m³")
    m2.metric("Live PM10", f"{st.session_state['live_pm10']} µg/m³")
    m3.metric("US AQI Index", f"{st.session_state['live_aqi']}")
    m4.metric(
        "Last Updated (UTC)", datetime.utcnow().strftime("%H:%M:%S")
    )
    st.divider()

if predict_btn:
    input_df = pd.DataFrame(
        [[temp, humidity, wind_speed]],
        columns=["Temperature_C", "Humidity_Pct", "WindSpeed_kmh"],
    )
    pred_code = int(model_knn.predict(input_df)[0])
    confidence = model_knn.predict_proba(input_df)[0][pred_code] * 100

    status_details = {
        0: {
            "title": "Air Quality Status: GOOD 🟢",
            "alert": "success",
            "desc": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "tag": "Good",
        },
        1: {
            "title": "Air Quality Status: MODERATE 🟡",
            "alert": "warning",
            "desc": "Air quality is acceptable; however, sensitive groups may experience minor health effects.",
            "tag": "Moderate",
        },
        2: {
            "title": "Air Quality Status: UNHEALTHY 🔴",
            "alert": "error",
            "desc": "Health alert: Everyone may begin to experience health effects. Avoid outdoor activities.",
            "tag": "Unhealthy",
        },
    }

    res = status_details.get(
        pred_code,
        {
            "title": "Status Unknown",
            "alert": "info",
            "desc": "-",
            "tag": "Unknown",
        },
    )

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history_file = "prediction_history.csv"
    new_data = pd.DataFrame(
        [
            {
                "Timestamp_UTC": now_str,
                "Temperature_C": temp,
                "Humidity_Pct": humidity,
                "WindSpeed_kmh": wind_speed,
                "Predicted_Class": res["tag"],
                "Confidence_Pct": round(confidence, 2),
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
        st.subheader("📌 Classification Assessment")
        if res["alert"] == "success":
            st.success(f"### {res['title']}")
        elif res["alert"] == "warning":
            st.warning(f"### {res['title']}")
        else:
            st.error(f"### {res['title']}")
        st.write(f"**Health Guidelines:** {res['desc']}")

    with col2:
        st.metric(
            label="🎯 Model Confidence",
            value=f"{confidence:.1f}%",
            delta="Nominal",
        )
    with col3:
        st.metric(
            label="🕒 Processing Time (UTC)",
            value=datetime.utcnow().strftime("%H:%M:%S"),
            delta=datetime.utcnow().strftime("%Y-%m-%d"),
        )

# ==========================================
# 8. Model Dataset & Analytics Report
# ==========================================
st.divider()
st.subheader("📊 Model Dataset & Prediction History Log")

tab1, tab2 = st.tabs(
    ["📁 Training Dataset (150 Samples)", "📋 Historical Prediction Records"]
)

with tab1:
    st.write(
        "**Machine Learning Dataset Overview: 3 Features | 3 Classes (Good, Moderate, Unhealthy) | 150 Samples**"
    )
    st.dataframe(dataset_150, use_container_width=True)

with tab2:
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file, encoding="utf-8-sig")
        st.dataframe(
            df_history.tail(15).sort_values(by="Timestamp_UTC", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No assessment records captured yet.")