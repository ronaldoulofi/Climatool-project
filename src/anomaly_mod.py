import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from supabase import create_client

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

CITIES = [
    {"city": "New York", "country": "US"},
    {"city": "Miami", "country": "US"},
    {"city": "Los Angeles", "country": "US"},
    {"city": "Paris", "country": "FR"},
    {"city": "Toulouse", "country": "FR"},
    {"city": "Lyon", "country": "FR"},
]


def get_city_id_map():
    """Obtiene el mapa de {nombre_ciudad: uuid} desde la tabla 'cities' en Supabase."""
    response = supabase.table("cities").select("id, name").execute()
    records = response.data or []
    return {r["name"].strip().lower(): r["id"] for r in records}


def load_weather_for_city(city):
    """Descarga todos los registros históricos de clima para una ciudad."""
    records = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase.table("weather_observations")
            .select("timestamp, temperature_c")
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
        raise ValueError(f"No hay observaciones climáticas para {city}.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["temperature_c", "timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_anomaly_features(df):
    """
    Calcula variables temporales y el valor esperado habitual
    según la estacionalidad (mes y hora del día).
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear

    # Valor esperado típico (mediana climática histórica por mes y hora)
    expected_medians = (
        df.groupby(["month", "hour"])["temperature_c"]
        .median()
        .reset_index()
        .rename(columns={"temperature_c": "expected_value"})
    )

    df = df.merge(expected_medians, on=["month", "hour"], how="left")
    df["deviation"] = df["temperature_c"] - df["expected_value"]

    feature_cols = ["temperature_c", "deviation", "hour", "month"]
    df = df.dropna(subset=feature_cols)

    return df, feature_cols


def train_or_load_isolation_forest(city, df, feature_cols, contamination=0.015):
    """
    Entrena y guarda/actualiza el modelo Isolation Forest para la ciudad.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_").lower()
    model_path = os.path.join(MODEL_DIR, f"if_temperature_{slug}.joblib")

    X = df[feature_cols]

    # Modelo Isolation Forest optimizado para series meteorológicas
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    joblib.dump(model, model_path)
    return model, model_path


def classify_severity(deviation):
    """
    Clasifica la severidad respetando el check constraint de Supabase
    ('low', 'medium', 'high').
    """
    abs_dev = abs(deviation)
    if abs_dev >= 10.0:
        return "high"
    elif abs_dev >= 7.0:
        return "medium"
    else:
        return "low"


def build_description(temperature_c, expected_value, deviation):
    """Genera una descripción textual clara del evento anómalo."""
    if deviation > 0:
        tipo = "Temperatura anormalmente alta (ola de calor/pico inusual)"
        signo = f"+{deviation:.1f}"
    else:
        tipo = "Temperatura anormalmente baja (ola de frío/caída inusual)"
        signo = f"{deviation:.1f}"

    return f"{tipo}: observada {temperature_c:.1f}°C vs esperada {expected_value:.1f}°C ({signo}°C de desviación)"


def detect_temperature_anomalies(city, df=None, contamination=0.015, min_deviation=5.0):
    """
    Detecta anomalías de temperatura para una ciudad dada y las prepara para Supabase.
    """
    if df is None:
        df = load_weather_for_city(city)

    df_prepared, feature_cols = prepare_anomaly_features(df)
    X = df_prepared[feature_cols]

    model, model_path = train_or_load_isolation_forest(city, df_prepared, feature_cols, contamination)

    # Predicción: -1 indica anomalía, 1 indica normal
    predictions = model.predict(X)
    
    # decision_function: valores más negativos representan mayor grado de anomalía
    raw_scores = model.decision_function(X)
    # Convertimos a anomaly_score positivo escalado
    anomaly_scores = np.round(-raw_scores, 4)

    df_prepared["is_anomaly"] = predictions == -1
    df_prepared["anomaly_score"] = anomaly_scores

    # Filtro de anomalías: detectado por el modelo Y desviación meteorológica significativa
    anomalies_df = df_prepared[
        (df_prepared["is_anomaly"]) & (df_prepared["deviation"].abs() >= min_deviation)
    ].copy()

    # Asignar severidad y descripción
    anomalies_df["severity"] = anomalies_df["deviation"].apply(classify_severity)
    anomalies_df["description"] = anomalies_df.apply(
        lambda row: build_description(
            row["temperature_c"], row["expected_value"], row["deviation"]
        ),
        axis=1
    )

    return anomalies_df, model_path


def save_anomalies_to_supabase(city, anomalies_df):

    city_map = get_city_id_map()
    city_key = city.strip().lower()
    city_id = city_map.get(city_key)

    if not city_id:
        raise ValueError(f"No se encontró el UUID de '{city}' en la tabla 'cities'.")

    # Limpiar anomalías previas para esta ciudad para evitar registros duplicados
    try:
        supabase.table("weather_anomalies").delete().eq("city_id", city_id).execute()
    except Exception as e:
        print(f"  [AVISO] No se pudo limpiar registros previos: {e}")

    if anomalies_df.empty:
        print(f"  [INFO] No se detectaron anomalías significativas para {city}.")
        return 0

    records = []
    for _, row in anomalies_df.iterrows():
        records.append({
            "city_id": city_id,
            "timestamp": row["timestamp"].isoformat(),
            "variable": "temperature",
            "observed_value": round(float(row["temperature_c"]), 2),
            "expected_value": round(float(row["expected_value"]), 2),
            "deviation": round(float(row["deviation"]), 2),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "severity": row["severity"],
            "description": row["description"]
        })

    # Inserción en bloques de 500 registros para optimizar payload de Supabase
    batch_size = 500
    total_saved = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("weather_anomalies").insert(batch).execute()
        total_saved += len(batch)

    return total_saved


def run_anomaly_detection_pipeline(city):
    """Flujo completo para una ciudad: carga datos, entrena IF, detecta y guarda."""
    print(f"\n" + "=" * 70)
    print(f" SISTEMA DE DETECCIÓN DE ANOMALÍAS (Isolation Forest) - {city}")
    print("=" * 70)

    print(" 1. Descargando historial climático...")
    df = load_weather_for_city(city)
    print(f"    Total observaciones analizadas: {len(df):,}")

    print(" 2. Entrenando modelo Isolation Forest y detectando patrones inusuales...")
    anomalies_df, model_path = detect_temperature_anomalies(city, df)
    print(f"    Modelo guardado en: {model_path}")
    print(f"    Anomalías detectadas: {len(anomalies_df):,}")

    if not anomalies_df.empty:
        sev_counts = anomalies_df["severity"].value_counts().to_dict()
        print(f"    Desglose por severidad: {sev_counts}")

        print("\n  Top 3 anomalías más severas detectadas:")
        top_anomalies = anomalies_df.sort_values(by="deviation", key=abs, ascending=False).head(3)
        for _, r in top_anomalies.iterrows():
            fecha = r["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
            print(f"    * [{r['severity'].upper():<6}] {fecha} | {r['description']}")

    print("\n 3. Sincronizando resultados con Supabase (weather_anomalies)...")
    total_saved = save_anomalies_to_supabase(city, anomalies_df)
    print(f"    Exitosamente guardados {total_saved} registros en 'weather_anomalies'.")

    return anomalies_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" DETECCIÓN DE ANOMALÍAS CLIMÁTICAS CON ISOLATION FOREST")
    print("=" * 70)
    print("Selecciona una opción:")
    for idx, c in enumerate(CITIES, start=1):
        print(f"  {idx}. {c['city']} ({c['country']})")
    print("  a. Analizar y sincronizar todas las ciudades")

    choice = input("\nIngresa número (o 'a' para todas): ").strip().lower()

    if choice == "a":
        for c in CITIES:
            try:
                run_anomaly_detection_pipeline(c["city"])
            except Exception as e:
                print(f"[ERROR] No se pudo procesar anomalías en {c['city']}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(CITIES):
        selected_city = CITIES[int(choice) - 1]["city"]
        run_anomaly_detection_pipeline(selected_city)
    else:
        print("Opción inválida.")

