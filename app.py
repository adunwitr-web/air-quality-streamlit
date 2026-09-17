from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st

st.set_page_config(
    page_title="Air Quality Predictor", page_icon="🌤️", layout="wide"
)


# 1. โหลดข้อมูลและเทรนโมเดล KNN
@st.cache_resource
def train_knn_model():
    ตารางข้อมูล = pd.read_csv("AirQuality_150.csv")
    X = ตารางข้อมูล[["อุณหภูมิ", "ความชื้น", "ความเร็วลม"]]
    y = ตารางข้อมูล["สถานะคุณภาพอากาศ"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model


model_knn = train_knn_model()

# 2. แถบป้อนข้อมูลข้างจอ (Sidebar)
st.sidebar.header("⚙️ กรอกข้อมูลสภาพอากาศ")
temp = st.sidebar.slider("🌡️ อุณหภูมิ (°C)", 0.0, 50.0, 28.0, 0.5)
humidity = st.sidebar.slider("💧 ความชื้น (%)", 0.0, 100.0, 65.0, 0.5)
wind_speed = st.sidebar.slider("🌬️ ความเร็วลม (km/h)", 0.0, 50.0, 10.0, 0.5)
predict_btn = st.sidebar.button("🔍 ทำนายผลคุณภาพอากาศ", type="primary")

# 3. หน้าจอหลัก
st.title("🌤️ ระบบประเมินและทำนายคุณภาพอากาศ (KNN)")
st.write("แอปพลิเคชันสำหรับทำนายคุณภาพอากาศและบันทึกประวัติด้วย Machine Learning")

col1, col2 = st.columns([1, 1])

if predict_btn:
    input_df = pd.DataFrame(
        [[temp, humidity, wind_speed]],
        columns=["อุณหภูมิ", "ความชื้น", "ความเร็วลม"],
    )

    pred_code = model_knn.predict(input_df)[0]
    confidence = model_knn.predict_proba(input_df)[0][pred_code] * 100

    status_map = {
        0: ("ดี 🟢 (Good)", "success"),
        1: ("ปานกลาง 🟡 (Moderate)", "warning"),
        2: ("อันตราย 🔴 (Unhealthy)", "error"),
    }
    status_text, alert_type = status_map[pred_code]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_file = "prediction_history.csv"

    new_data = pd.DataFrame(
        [
            {
                "Timestamp": now_str,
                "Temperature_C": temp,
                "Humidity_Percent": humidity,
                "WindSpeed_kmh": wind_speed,
                "Predicted_Status": status_text.split(" ")[0],
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

    with col1:
        st.subheader("📊 ผลการทำนาย")
        if alert_type == "success":
            st.success(f"**สถานะ:** {status_text}")
        elif alert_type == "warning":
            st.warning(f"**สถานะ:** {status_text}")
        else:
            st.error(f"**สถานะ:** {status_text}")

        st.metric(
            label="🎯 ความมั่นใจของโมเดล", value=f"{confidence:.2f} %"
        )
        st.info(f"💾 บันทึกประวัติเมื่อ {now_str} เรียบร้อยแล้ว!")

# 4. ประวัติและกราฟ
st.divider()
st.subheader("📈 ประวัติการทำนายและสถิติสรุป")

history_file = "prediction_history.csv"
if os.path.exists(history_file):
    df_history = pd.read_csv(history_file)
    if not df_history.empty:
        tab1, tab2 = st.tabs(["📋 ตารางประวัติ", "📊 กราฟสรุป"])
        with tab1:
            st.dataframe(df_history.tail(10), use_container_width=True)
        with tab2:
            counts = df_history["Predicted_Status"].value_counts()
            categories = ["ดี", "ปานกลาง", "อันตราย"]
            chart_values = [counts.get(cat, 0) for cat in categories]

            plt.rcdefaults()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

            sns.barplot(
                x=["Good (0)", "Moderate (1)", "Unhealthy (2)"],
                y=chart_values,
                palette=colors,
                ax=ax,
            )
            ax.set_title(
                f"Prediction Summary (Total: {len(df_history)} records)"
            )
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
else:
    st.info(
        "ℹ️ ยังไม่มีประวัติการทำนาย ให้ลองเลือกค่าที่ sidebar ด้านซ้ายแล้วกดทำนายผล"
    )