# 🛵 Food Delivery Delay Prediction

Yemek teslimat siparişlerinin, sistem tarafından verilen tahmini teslimat süresine göre **kaç dakika geç kalacağını** (veya erken geleceğini) tahmin eden uçtan uca bir regresyon projesi.

## 📌 Problem Tanımı

```
delay_min = actual_delivery_time_minutes - estimated_delivery_time_minutes
```

- `delay_min > 0` → sipariş geç kaldı
- `delay_min < 0` → sipariş erken geldi
- `delay_min ≈ 0` → sipariş tam zamanında geldi

## 🔍 Veri Seti

~50.000 satırlık sipariş verisi. İptal edilen siparişler (target hesaplanamadığı için) analiz öncesi çıkarıldı, kalan **49.132 sipariş** üzerinde çalışıldı.

## ⚙️ Kullanılan Yöntem

### 1) Veri Temizleme & Leakage Kontrolü
- Target'tan doğrudan türeyen / teslimat sonrası oluşan kolonlar çıkarıldı: `actual_delivery_time_minutes`, `late_delivery`, `customer_rating`, `tip_amount`
- ID kolonları (`order_id`, `customer_id`, `restaurant_id`, `driver_id`) ve redundant kolonlar (`order_timestamp`, `subtotal`, `tax_amount`, `service_fee` vb.) elendi

### 2) Keşifçi Veri Analizi (EDA)
- Kategorik ve numerik değişkenlerin `delay_min` ile ilişkisi incelendi
- **En güçlü sinyal: hava durumu** — Clear'da ortalama 2.55 dk gecikme, Storm'da 15.65 dk (~6 kat artış)
- `day_of_week`, `traffic_level` orta düzey etkili; `payment_method`, `restaurant_primary_category` zayıf sinyalli bulundu

### 3) Feature Engineering
| Değişken | Açıklama |
|---|---|
| `hour_sin`, `hour_cos` | Sipariş saatinin döngüsel (cyclical) encoding'i |
| `is_peak_hour` | Öğle/akşam yoğun saat bayrağı |
| `weather_traffic` | Hava durumu × trafik seviyesi kombinasyonu |
| `weather_score` | Hava şiddetinin ordinal skoru (Clear=1 → Storm=4) |
| `distance_x_weather` | Mesafe × hava şiddeti etkileşimi |

### 4) Ön İşleme
- Kategorik değişkenler: One-Hot Encoding
- Numerik değişkenler: StandardScaler
- **Leakage önleme:** encoder/scaler sadece train setine `fit` edildi, test setine sadece `transform` uygulandı

### 5) Modelleme
7 farklı regresyon modeli karşılaştırıldı (LR, KNN, CART, RF, GBM, XGBoost, LightGBM), en iyi 3 model (RF, XGBoost, LightGBM) üzerinde `GridSearchCV` ile hiperparametre optimizasyonu yapıldı.

### 6) Final Model — Voting Regressor
`LinearRegression + LightGBM + XGBoost` kombinasyonu, tekil modellerin hepsinden daha iyi sonuç verdi.

## 📊 Sonuçlar (Test Seti)

| Metrik | Değer |
|---|---|
| **MAE** | 2.07 dakika |
| **RMSE** | 2.59 dakika |
| **R²** | 0.785 |

Model, hiç görmediği test verisinde ortalama **~2 dakikalık sapmayla** gecikme tahmini yapabiliyor ve `delay_min`'deki varyansın **%78.5'ini** açıklayabiliyor.

## 🗂️ Proje Yapısı

```
├── food_delivery_delay_prediction.py   # Ana pipeline: EDA, feature engineering, modelleme
├── app.py                               # Streamlit tahmin uygulaması
├── voting_reg_delay_model.pkl           # Eğitilmiş final model
├── preprocessor.pkl                     # Fit edilmiş encoder + scaler
└── README.md
```

## 🚀 Çalıştırma

```bash
# Modeli eğitmek için
python food_delivery_delay_prediction.py

# Streamlit uygulamasını başlatmak için
streamlit run app.py
```

## 🛠️ Kullanılan Teknolojiler

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `seaborn` / `matplotlib` · `streamlit` · `joblib`

## 👥 Ekip

*(Bootcamp grup projesi)*
