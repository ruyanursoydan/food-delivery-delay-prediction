# -*- coding: utf-8 -*-
"""
Streamlit App - Yemek Teslimatı Gecikme Tahmini
=================================================
Kullanım:
    streamlit run app.py

Gereken dosyalar (aynı klasörde olmalı):
    - voting_reg_delay_model.pkl
    - preprocessor.pkl

Bu iki dosya, food_delivery_delay_prediction.py çalıştırıldıktan sonra
otomatik olarak üretilir (joblib.dump ile).
"""

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1) Model ve preprocessor'ı yükle (uygulama her açıldığında bir kez)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("voting_reg_delay_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_artifacts()

st.set_page_config(page_title="Teslimat Gecikme Tahmini", page_icon="🛵")
st.title("🛵 Yemek Teslimatı Gecikme Tahmini")
st.write(
    "Aşağıdaki bilgileri girerek siparişin tahmini süreye göre "
    "kaç dakika geç kalacağını (veya erken geleceğini) öğrenebilirsiniz."
)

# ---------------------------------------------------------------
# 2) Kullanıcıdan ham (encoding/scaling yapılmamış) veriyi al
#    NOT: Buradaki seçenek listeleri projedeki eğitim verisiyle
#    birebir aynı olmalı (aksi halde OneHotEncoder bilinmeyen
#    kategori olarak sıfır vektörle kodlar - handle_unknown='ignore')
# ---------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    day_of_week = st.selectbox(
        "Haftanın günü",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    city = st.selectbox("Şehir", ["City_A", "City_B", "City_C", "City_D"])

    customer_type = st.selectbox("Müşteri tipi", ["New", "Regular", "Premium"])

    restaurant_type = st.selectbox(
        "Restoran tipi", ["Fast Food", "Casual Dining", "Cafe", "Fine Dining"]
    )

    restaurant_primary_category = st.selectbox(
        "Restoran kategorisi",
        ["Burger", "Pizza", "Asian", "Dessert", "Local", "Healthy"],
    )

    weather = st.selectbox("Hava durumu", ["Clear", "Cloudy", "Rain", "Storm"])
    traffic_level = st.selectbox("Trafik seviyesi", ["Low", "Medium", "High"])

with col2:
    customer_age = st.slider("Müşteri yaşı", 18, 70, 30)
    restaurant_rating = st.slider("Restoran puanı", 1.0, 5.0, 4.2, step=0.1)
    items_count = st.slider("Ürün sayısı", 1, 15, 2)
    delivery_fee = st.number_input("Teslimat ücreti (₺)", 0.0, 50.0, 3.5)
    order_total = st.number_input("Sipariş tutarı (₺)", 1.0, 500.0, 40.0)
    distance_km = st.slider("Mesafe (km)", 0.1, 25.0, 5.0)
    delivery_partner_experience_months = st.slider("Kurye tecrübesi (ay)", 0, 72, 24)
    delivery_partner_rating = st.slider("Kurye puanı", 1.0, 5.0, 4.5, step=0.1)
    restaurant_preparation_time_minutes = st.slider("Hazırlık süresi (dk)", 1, 80, 15)
    estimated_delivery_time_minutes = st.slider("Sistem tahmini süre (dk)", 5, 130, 30)
    order_hour = st.slider("Sipariş saati", 0, 23, 19)

# ---------------------------------------------------------------
# 3) Feature engineering (eğitimdeki ile birebir aynı mantık)
# ---------------------------------------------------------------
import numpy as np

hour_sin = np.sin(2 * np.pi * order_hour / 24.0)
hour_cos = np.cos(2 * np.pi * order_hour / 24.0)
peak_hours = [12, 13, 14, 18, 19, 20, 21]
is_peak_hour = 1 if order_hour in peak_hours else 0

weather_traffic = f"{weather}_{traffic_level}"
weather_severity = {"Clear": 1, "Cloudy": 2, "Rain": 3, "Storm": 4}
weather_score = weather_severity[weather]
distance_x_weather = distance_km * weather_score

# ---------------------------------------------------------------
# 4) Tahmin butonu
# ---------------------------------------------------------------
if st.button("Tahmini Gecikmeyi Hesapla"):
    input_df = pd.DataFrame([{
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "city": city,
        "customer_age": customer_age,
        "customer_type": customer_type,
        "restaurant_type": restaurant_type,
        "restaurant_primary_category": restaurant_primary_category,
        "restaurant_rating": restaurant_rating,
        "items_count": items_count,
        "delivery_fee": delivery_fee,
        "order_total": order_total,
        "distance_km": distance_km,
        "delivery_partner_experience_months": delivery_partner_experience_months,
        "delivery_partner_rating": delivery_partner_rating,
        "restaurant_preparation_time_minutes": restaurant_preparation_time_minutes,
        "estimated_delivery_time_minutes": estimated_delivery_time_minutes,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_peak_hour": is_peak_hour,
        "weather_traffic": weather_traffic,
        "weather_score": weather_score,
        "distance_x_weather": distance_x_weather,
    }])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]

    st.subheader("Sonuç")
    if prediction > 0:
        st.error(f"Tahmini gecikme: **{prediction:.1f} dakika** (siparişin geç kalması bekleniyor)")
    else:
        st.success(f"Tahmini erken teslimat: **{abs(prediction):.1f} dakika** (siparişin erken gelmesi bekleniyor)")

    st.caption(
        "Not: Bu tahmin, geçmiş sipariş verileriyle eğitilmiş bir modele dayanır "
        "ve gerçek gecikmeyi garanti etmez (ortalama hata payı ~2 dakika)."
    )
