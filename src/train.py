import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan las credenciales de Supabase en .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Definición de variables climáticas a entrenar con modelos individuales
TARGETS = {
    "temperature_c": {"label": "Temperatura", "unit": "°C"},
    "humidity_pct": {"label": "Humedad", "unit": "%"},
    "wind_speed_kmh": {"label": "Velocidad viento", "unit": "km/h"},
    "pressure_hpa": {"label": "Presión", "unit": "hPa"}
}

# Mapeo de la variable objetivo a su correspondiente predictor Lag-1 (Modelo Persistencia)
LAG1_COLS = {
    "temperature_c": "temp_lag_1",
    "humidity_pct": "humidity_lag_1",
    "wind_speed_kmh": "wind_lag_1",
    "pressure_hpa": "pressure_lag_1"
}

FEATURE_COLS = [
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "humidity_lag_1", "humidity_lag_2",
    "wind_lag_1", "wind_lag_2",
    "pressure_lag_1", "pressure_lag_2",
    "hour", "dayofyear"
]

# Lista de ciudades oficiales del proyecto
CITIES = [
    {"city": "New York", "country": "US"},
    {"city": "Miami", "country": "US"},
    {"city": "Los Angeles", "country": "US"},
    {"city": "Paris", "country": "FR"},
    {"city": "Toulouse", "country": "FR"},
    {"city": "Lyon", "country": "FR"},
]


def load_weather_for_city(city):
    # Descargar todos los registros históricos por páginas
    records = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase.table("weather_observations")
            .select("*")
            .eq("city", city)
            .order("timestamp")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        batch = response.data or []
        records.extend(batch)

        if len(batch) < page_size:
            break

        offset += page_size

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No hay datos históricos para {city}. "
            "Ejecuta primero etl_history.py."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    numeric_cols = list(TARGETS.keys())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Evitar entrenar con huecos en las horas
    gaps = df["timestamp"].diff().dropna()
    if (gaps != pd.Timedelta(hours=1)).any():
        print("Aviso: existen horas faltantes en el historial.")

    return df


def create_features(df, target):
    df = df.copy()

    # Predictores: desfases temporales (lags)
    df["temp_lag_1"] = df["temperature_c"].shift(1)
    df["temp_lag_2"] = df["temperature_c"].shift(2)
    df["temp_lag_3"] = df["temperature_c"].shift(3)

    df["humidity_lag_1"] = df["humidity_pct"].shift(1)
    df["humidity_lag_2"] = df["humidity_pct"].shift(2)

    df["wind_lag_1"] = df["wind_speed_kmh"].shift(1)
    df["wind_lag_2"] = df["wind_speed_kmh"].shift(2)

    df["pressure_lag_1"] = df["pressure_hpa"].shift(1)
    df["pressure_lag_2"] = df["pressure_hpa"].shift(2)

    # Variables temporales
    df["hour"] = df["timestamp"].dt.hour
    df["dayofyear"] = df["timestamp"].dt.dayofyear

    # Limpiar nulos generados por los shifts
    df = df.dropna(subset=FEATURE_COLS + [target])

    X = df[FEATURE_COLS]
    y = df[target]

    return X, y


def train_single_model(city, target, df):
    info = TARGETS[target]
    X, y = create_features(df, target)

    if len(X) < 100:
        raise ValueError(
            f"Solo hay {len(X)} filas útiles para {target}. "
            "Carga más datos antes de entrenar."
        )

    # Separación temporal: 80% entrenamiento, 20% prueba
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    
    # MODELO BASE 1: Persistencia (Lag-1 / Naive Forecast)
   
    lag1_col = LAG1_COLS[target]
    preds_base_persistence = X_test[lag1_col]
    mae_persistence = float(mean_absolute_error(y_test, preds_base_persistence))
    rmse_persistence = float(np.sqrt(mean_squared_error(y_test, preds_base_persistence)))

 
    # MODELO BASE 2: Regresión Lineal (Linear Regression)

    model_base_lr = LinearRegression()
    model_base_lr.fit(X_train, y_train)
    preds_base_lr = model_base_lr.predict(X_test)
    mae_lr = float(mean_absolute_error(y_test, preds_base_lr))
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, preds_base_lr)))


    # MODELO PRINCIPAL: Random Forest Regressor

    model_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train, y_train)

    preds_rf = model_rf.predict(X_test)
    mae_rf = float(mean_absolute_error(y_test, preds_rf))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, preds_rf)))

    # Cálculo del porcentaje de mejora de Random Forest frente a los modelos base
    improvement_vs_persistence = (
        ((mae_persistence - mae_rf) / mae_persistence) * 100
        if mae_persistence > 0 else 0.0
    )
    improvement_vs_lr = (
        ((mae_lr - mae_rf) / mae_lr) * 100
        if mae_lr > 0 else 0.0
    )

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_").lower()
    model_name = f"rf_{slug}_{target}"
    model_version = "latest"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")

    # Eliminar versiones obsoletas con timestamp de este mismo modelo para evitar desorden
    if os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.startswith(f"{model_name}_v_") and fname.endswith(".joblib"):
                try:
                    os.remove(os.path.join(MODEL_DIR, fname))
                except Exception:
                    pass

    # Guardar / sobrescribir siempre el modelo más actualizado localmente
    joblib.dump(model_rf, model_path)

    # Registro y sobrescritura en Supabase (model_registry)
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "city": city,
        "training_rows": len(X_train),
        "mae": mae_rf,
        "rmse": rmse_rf,
        "mae_persistence": mae_persistence,
        "rmse_persistence": rmse_persistence,
        "mae_linear_regression": mae_lr,
        "rmse_linear_regression": rmse_lr,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"RandomForest vs Baselines -> Mejora vs Lag-1: {improvement_vs_persistence:+.1f}%, "
            f"vs RegLineal: {improvement_vs_lr:+.1f}%"
        )
    }

    try:
        # Borrar registro previo con este model_name para garantizar 1 solo registro por modelo
        supabase.table("model_registry").delete().eq("model_name", model_name).execute()
        supabase.table("model_registry").insert(metadata).execute()
    except Exception:
        # Respaldo en caso de que aún no se hayan aplicado las columnas de baselines en Supabase
        fallback_metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "city": city,
            "training_rows": len(X_train),
            "mae": mae_rf,
            "rmse": rmse_rf,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "notes": (
                f"RF vs Baselines -> MAE Lag-1: {mae_persistence:.2f}, RegLineal: {mae_lr:.2f} "
                f"(Mejora vs Lag-1: {improvement_vs_persistence:+.1f}%, vs RegLineal: {improvement_vs_lr:+.1f}%)"
            )
        }
        try:
            supabase.table("model_registry").delete().eq("model_name", model_name).execute()
            supabase.table("model_registry").insert(fallback_metadata).execute()
        except Exception as e:
            print(f"    [AVISO] No se pudo registrar en Supabase: {e}")

    print(f"\n  Resultados para {info['label']} ({target}) - {city}:")
    print(f"    [Base 1 - Persistencia Lag-1]: MAE = {mae_persistence:6.2f} {info['unit']} | RMSE = {rmse_persistence:6.2f} {info['unit']}")
    print(f"    [Base 2 - Regresión Lineal ]: MAE = {mae_lr:6.2f} {info['unit']} | RMSE = {rmse_lr:6.2f} {info['unit']}")
    print(f"    [Principal - Random Forest ]: MAE = {mae_rf:6.2f} {info['unit']} | RMSE = {rmse_rf:6.2f} {info['unit']} -> Mejora vs Base 1: {improvement_vs_persistence:+.1f}% | vs Base 2: {improvement_vs_lr:+.1f}%")

    return model_path, model_name, model_version, mae_rf, rmse_rf


def train_city_models(city):
    print(f"\n" + "=" * 65)
    print(f" Entrenando 4 modelos individuales para: {city}")
    print("=" * 65)

    df = load_weather_for_city(city)

    for target in TARGETS:
        train_single_model(city, target, df)


if __name__ == "__main__":
    print("\nSelecciona ciudad para entrenar sus 4 modelos:")
    for i, c in enumerate(CITIES, start=1):
        print(f"  {i}. {c['city']} ({c['country']})")
    print("  a. Todas las ciudades")

    choice = input("\nIngresa número (o 'a' para todas): ").strip().lower()
    if choice == "a":
        for c in CITIES:
            try:
                train_city_models(c["city"])
            except Exception as e:
                print(f"[ERROR] No se pudo entrenar {c['city']}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(CITIES):
        c = CITIES[int(choice) - 1]
        train_city_models(c["city"])
    else:
        print("Opción inválida.")
