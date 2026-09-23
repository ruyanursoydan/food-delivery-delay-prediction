# -*- coding: utf-8 -*-
"""
Food Delivery Delay Prediction
================================
Amaç: Yemek teslimatının tahmini süreye göre kaç dakika geç kalacağını
(delay_min = actual_delivery_time_minutes - estimated_delivery_time_minutes)
tahmin eden bir regresyon modeli kurmak.

Akış:
    1. Kütüphaneler ve ayarlar
    2. Yardımcı fonksiyonlar (EDA)
    3. Veri okuma ve temizleme (leakage kolonlarının çıkarılması)
    4. Keşifçi Veri Analizi (EDA)
    5. Feature Engineering
    6. Aykırı değer (outlier) düzenleme
    7. Train/Test split
    8. Encoding & Scaling (ColumnTransformer)
    9. Base model karşılaştırması
    10. Hiperparametre optimizasyonu
    11. Voting Regressor (final model)
    12. Test seti üzerinde final değerlendirme
    13. Model ve preprocessor'ı kaydetme (joblib)
"""

# ==============================================================
# 1) KÜTÜPHANELER VE AYARLAR
# ==============================================================
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Veri dosyasının yolu (kendi ortamınıza göre güncelleyin)
DATA_PATH = "food_delivery_orders_dataset.csv"
RANDOM_STATE = 42


# ==============================================================
# 2) YARDIMCI FONKSİYONLAR (EDA)
# ==============================================================
def check_df(dataframe, head=5):
    """DataFrame hakkında genel bir özet basar (shape, tip, head/tail, NA, quantile)."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numeric_df = dataframe.select_dtypes(include=["number"])
    if not numeric_df.empty:
        print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    """Kategorik bir değişkenin sınıf dağılımını ve oranını gösterir."""
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    """Numerik bir değişkenin describe + histogramını gösterir."""
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    """Kategorik değişken kırılımında target ortalamasını gösterir."""
    summary = dataframe.groupby(categorical_col)[target].agg(["mean", "count"]).reset_index()
    summary.columns = [categorical_col, "TARGET_MEAN", "COUNT"]
    print(summary.to_string(index=False))
    print("=" * 45)
    if plot:
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(x=categorical_col, y=target, data=dataframe, ci=None, palette="magma")
        plt.title(f"{categorical_col} vs {target} (Ortalama Target)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    """Numerik değişken ile target arasındaki korelasyonu ve trendi gösterir."""
    corr = dataframe[numerical_col].corr(dataframe[target])
    print(f"Korelasyon ({numerical_col} vs {target}): {corr:.4f}\n")
    if plot:
        plt.figure(figsize=(8, 4))
        sns.regplot(x=numerical_col, y=target, data=dataframe,
                    scatter_kws={"alpha": 0.1, "s": 15, "color": "navy"},
                    line_kws={"color": "red", "linewidth": 2})
        plt.title(f"{numerical_col} vs {target}")
        plt.tight_layout()
        plt.show(block=True)


def correlation_matrix(df, cols):
    """Numerik değişkenler arası korelasyon heatmap'i çizer."""
    plt.figure(figsize=(16, 10))
    sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5,
                annot_kws={"size": 10}, linecolor="w", cmap="RdBu")
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik-fakat-kardinal
    değişkenlerin isimlerini döndürür.

    Returns
    -------
    cat_cols, num_cols, cat_but_car : list, list, list
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


# ==============================================================
# 3) VERİ OKUMA VE TEMİZLEME
# ==============================================================
def load_and_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # İptal edilen siparişlerde actual_delivery_time_minutes NaN -> target hesaplanamaz
    df = df[df["actual_delivery_time_minutes"].notna()].copy()

    # Target: delay_min
    df["delay_min"] = df["actual_delivery_time_minutes"] - df["estimated_delivery_time_minutes"]

    # Leakage / redundant / kimlik kolonlarının çıkarılması
    drop_cols = [
        "actual_delivery_time_minutes",     # target'tan hesaplandı -> leakage
        "late_delivery",                    # target'ın binary hali -> leakage
        "customer_rating", "tip_amount",    # teslimat sonrası oluşuyor
        "order_id", "customer_id", "restaurant_id", "driver_id",  # id kolonları
        "order_timestamp", "order_date",    # order_hour/day_of_week zaten var
        "order_status", "cancellation_reason",  # filtre sonrası anlamsız/sabit
        "subtotal", "tax_amount", "service_fee",  # order_total zaten var (yüksek korelasyon)
        "discount_percent",                 # target ile ~sıfır korelasyon
        "payment_method",                   # target ile ayırt edici değil
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# ==============================================================
# 4) EDA (opsiyonel - interaktif inceleme için)
# ==============================================================
def run_eda(df):
    check_df(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    for col in cat_cols:
        cat_summary(df, col)
    for col in num_cols:
        num_summary(df, col, plot=False)

    correlation_matrix(df, num_cols)

    for col in num_cols:
        target_summary_with_num(df, "delay_min", col, plot=False)
    for col in cat_cols:
        target_summary_with_cat(df, "delay_min", col, plot=False)


# ==============================================================
# 5) FEATURE ENGINEERING
# ==============================================================
def feature_engineering(df):
    # Döngüsel (cyclical) saat encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24.0)

    # Yoğun saat flag'i (öğle + akşam)
    peak_hours = [12, 13, 14, 18, 19, 20, 21]
    df["is_peak_hour"] = df["order_hour"].isin(peak_hours).astype(int)
    df = df.drop(columns=["order_hour"])

    # Hava + trafik kombinasyonu
    df["weather_traffic"] = df["weather"] + "_" + df["traffic_level"]

    # Hava şiddeti (ordinal) ve mesafe etkileşimi
    weather_severity = {"Clear": 1, "Cloudy": 2, "Rain": 3, "Storm": 4}
    df["weather_score"] = df["weather"].map(weather_severity)
    df["distance_x_weather"] = df["distance_km"] * df["weather_score"]

    # delivery_area, city ile örtüştüğü için (her city'nin alt bölgesi) drop edildi
    df = df.drop(columns=["weather", "traffic_level", "delivery_area"])
    return df


def handle_outliers(df, num_cols):
    # Sadece feature'lardaki outlier'lar düzenlenir; target (delay_min) dokunulmaz
    for col in ["order_total", "restaurant_preparation_time_minutes"]:
        if col in num_cols:
            replace_with_thresholds(df, col)
    return df


# ==============================================================
# 6) PREPROCESSING (Encoding + Scaling)
# ==============================================================
ONEHOT_COLS = ["day_of_week", "city", "customer_type",
               "restaurant_type", "restaurant_primary_category",
               "weather_traffic"]

NUM_COLS = ["customer_age", "restaurant_rating", "items_count", "delivery_fee",
            "order_total", "distance_km", "delivery_partner_experience_months",
            "delivery_partner_rating", "restaurant_preparation_time_minutes",
            "estimated_delivery_time_minutes", "weather_score", "distance_x_weather"]

PASSTHROUGH_COLS = ["is_weekend", "is_peak_hour", "hour_sin", "hour_cos"]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
        ("scale", StandardScaler(), NUM_COLS),
        ("pass", "passthrough", PASSTHROUGH_COLS),
    ])


# ==============================================================
# 7) MODELLEME
# ==============================================================
def base_models(X, y, scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2")):
    print("Base Models....")
    regressors = [
        ("LR", LinearRegression()),
        ("KNN", KNeighborsRegressor()),
        ("CART", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ("RF", RandomForestRegressor(random_state=RANDOM_STATE)),
        ("GBM", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("XGBoost", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE)),
        ("LightGBM", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)),
    ]
    results = {}
    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=5, scoring=list(scoring))
        mae = -cv_results["test_neg_mean_absolute_error"].mean()
        rmse = -cv_results["test_neg_root_mean_squared_error"].mean()
        r2 = cv_results["test_r2"].mean()
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"{name:10s} -> MAE: {round(mae, 4)}  RMSE: {round(rmse, 4)}  R2: {round(r2, 4)}")
    return results


RF_PARAMS = {"max_depth": [8, 15, None], "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20], "n_estimators": [200, 300]}
XGBOOST_PARAMS = {"learning_rate": [0.1, 0.01], "max_depth": [5, 8], "n_estimators": [100, 200]}
LIGHTGBM_PARAMS = {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500]}


def hyperparameter_optimization(X, y, cv=3, scoring="neg_mean_absolute_error"):
    print("Hyperparameter Optimization....")
    regressors = [
        ("RF", RandomForestRegressor(random_state=RANDOM_STATE), RF_PARAMS),
        ("XGBoost", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE), XGBOOST_PARAMS),
        ("LightGBM", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1), LIGHTGBM_PARAMS),
    ]
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}\n")
        best_models[name] = final_model
    return best_models


def train_voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_reg = VotingRegressor(estimators=[
        ("LR", LinearRegression()),
        ("LightGBM", best_models["LightGBM"]),
        ("XGBoost", best_models["XGBoost"]),
    ]).fit(X, y)

    cv_results = cross_validate(voting_reg, X, y, cv=3,
                                 scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    mae = -cv_results["test_neg_mean_absolute_error"].mean()
    rmse = -cv_results["test_neg_root_mean_squared_error"].mean()
    r2 = cv_results["test_r2"].mean()
    print(f"MAE: {round(mae, 4)}  RMSE: {round(rmse, 4)}  R2: {round(r2, 4)}")
    return voting_reg


def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("=== Test Seti Sonuçları ===")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2  :", r2_score(y_test, y_pred))


# ==============================================================
# 8) ANA ÇALIŞTIRMA AKIŞI
# ==============================================================
def main():
    # Veri okuma ve temizleme
    df = load_and_clean_data(DATA_PATH)

    # EDA yapmak isterseniz açın (grafikler çok sayıda pencere açar):
    # run_eda(df)

    # Feature engineering
    df = feature_engineering(df)

    # Outlier düzenleme (sadece feature'larda, target'a dokunulmaz)
    _, num_cols, _ = grab_col_names(df, cat_th=5, car_th=20)
    df = handle_outliers(df, num_cols)

    # X / y ayır ve split
    y = df["delay_min"]
    X = df.drop(columns=["delay_min"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("Train/Test shape:", X_train.shape, X_test.shape)

    # Preprocessing (sadece train'e fit)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Base model karşılaştırması
    base_models(X_train_processed, y_train)

    # Hiperparametre optimizasyonu (RF, XGBoost, LightGBM)
    best_models = hyperparameter_optimization(X_train_processed, y_train)

    # Voting Regressor (final model): LR + LightGBM + XGBoost
    voting_reg = train_voting_regressor(best_models, X_train_processed, y_train)

    # Test seti üzerinde final değerlendirme
    evaluate_on_test(voting_reg, X_test_processed, y_test)

    # Model ve preprocessor'ı kaydet
    joblib.dump(voting_reg, "voting_reg_delay_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Model ve preprocessor kaydedildi.")


if __name__ == "__main__":
    main()
