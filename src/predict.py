import sys
import os
import re
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
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

TARGETS = {
    "temperature_c": {"label": "Temperatura", "unit": "°C", "field": "predicted_temperature_c"},
    "humidity_pct": {"label": "Humedad", "unit": "%", "field": "predicted_humidity_pct"},
    "wind_speed_kmh": {"label": "Velocidad viento", "unit": "km/h", "field": "predicted_wind_speed_kmh"},
    "pressure_hpa": {"label": "Presión atmosférica", "unit": "hPa", "field": "predicted_pressure_hpa"}
}

FEATURES = [
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "humidity_lag_1", "humidity_lag_2",
    "wind_lag_1", "wind_lag_2",
    "pressure_lag_1", "pressure_lag_2",
    "hour", "dayofyear"
]

CITIES = [
    {"city": "New York", "country": "US", "latitude": 40.7128, "longitude": -74.0060},
    {"city": "Miami", "country": "US", "latitude": 25.7617, "longitude": -80.1918},
    {"city": "Los Angeles", "country": "US", "latitude": 34.0522, "longitude": -118.2437},
    {"city": "Paris", "country": "FR", "latitude": 48.8566, "longitude": 2.3522},
    {"city": "Toulouse", "country": "FR", "latitude": 43.6047, "longitude": 1.4442},
    {"city": "Lyon", "country": "FR", "latitude": 45.7640, "longitude": 4.8357},
]

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]


def load_target_model(city, target):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_").lower()
    target_pattern = f"rf_{slug}_{target}"

    # 1. Cargar directamente el modelo más actualizado (sin timestamp)
    exact_path = os.path.join(MODEL_DIR, f"{target_pattern}.joblib")
    if os.path.isfile(exact_path):
        meta = {"model_name": target_pattern, "model_version": "latest"}
        return joblib.load(exact_path), meta

    # 2. Respaldo: buscar archivo local con versión previa/timestamp en carpeta models
    if os.path.isdir(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR), reverse=True):
            if f.startswith(target_pattern) and f.endswith(".joblib"):
                model_path = os.path.join(MODEL_DIR, f)
                meta = {"model_name": target_pattern, "model_version": "local"}
                return joblib.load(model_path), meta

    # 3. Respaldo adicional: modelo legado unificado rf_{slug}_v_*.joblib
    legacy_pattern = f"rf_{slug}_v_"
    if os.path.isdir(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR), reverse=True):
            if f.startswith(legacy_pattern) and f.endswith(".joblib"):
                model_path = os.path.join(MODEL_DIR, f)
                meta = {"model_name": f"rf_{slug}", "model_version": "legacy"}
                return joblib.load(model_path), meta

    raise FileNotFoundError(
        f"No se encontró un modelo para '{target}' en {city}. "
        "Ejecuta train.py primero."
    )


def load_seed_observations(city, country="US", latitude=None, longitude=None):

    now = datetime.now(timezone.utc)
    
    # 1. Intentar cargar desde Supabase
    try:
        response = (
            supabase.table("weather_observations")
            .select("*")
            .eq("city", city)
            .order("timestamp", desc=True)
            .limit(10)
            .execute()
        )
        data = response.data or []
        if len(data) >= 3:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            for col in TARGETS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            last3 = df.tail(3)
            diffs = last3["timestamp"].diff().dropna()
            # Si los datos son horarios consecutivos y relativamente recientes (menos de 48h)
            if (diffs == pd.Timedelta(hours=1)).all():
                gap_hours = (now - last3.iloc[-1]["timestamp"]).total_seconds() / 3600
                if gap_hours <= 48:
                    return last3
    except Exception:
        pass

    # 2. Respaldo en Open-Meteo para asegurar datos reales frescos del día de consulta
    if latitude is None or longitude is None:
        city_info = next((c for c in CITIES if c["city"].lower() == city.lower()), None)
        if city_info:
            latitude, longitude = city_info["latitude"], city_info["longitude"]
        else:
            latitude, longitude = 40.7128, -74.0060

    try:
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
                "past_days": 2,
                "forecast_days": 1,
                "timezone": "UTC"
            },
            timeout=15
        )
        res.raise_for_status()
        h = res.json().get("hourly", {})
        times = h.get("time", [])
        temps = h.get("temperature_2m", [])
        hums = h.get("relative_humidity_2m", [])
        winds = h.get("wind_speed_10m", [])
        press = h.get("pressure_msl", [])

        records = []
        for i, ts in enumerate(times):
            if i < len(temps) and temps[i] is not None:
                records.append({
                    "city": city,
                    "timestamp": pd.to_datetime(f"{ts}:00+00:00", utc=True),
                    "temperature_c": float(temps[i]),
                    "humidity_pct": float(hums[i]) if i < len(hums) and hums[i] is not None else 50.0,
                    "wind_speed_kmh": float(winds[i]) if i < len(winds) and winds[i] is not None else 10.0,
                    "pressure_hpa": float(press[i]) if i < len(press) and press[i] is not None else 1013.25
                })

        df_fresh = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        if len(df_fresh) >= 3:
            return df_fresh.tail(3)
    except Exception as e:
        print(f"  [Aviso al consultar Open-Meteo: {e}]")

    # 3. Si hay datos en Supabase aunque sean más antiguos, usarlos como semilla
    try:
        if 'df' in locals() and len(df) >= 3:
            return df.tail(3)
    except Exception:
        pass

    raise ValueError(f"No fue posible cargar observaciones climáticas base para {city}.")


def predict_city(city):

    # 1. Identificar fecha, día de la semana y semana de la consulta
    now = datetime.now(timezone.utc)
    day_name = DIAS_SEMANA[now.weekday()]
    month_name = MESES[now.month - 1]
    consult_str = f"{day_name}, {now.day} de {month_name} de {now.year} a las {now.strftime('%H:%M')} UTC"

    iso_year, iso_week, _ = now.isocalendar()
    week_monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_sunday = week_monday + timedelta(days=6, hours=23, minutes=59)
    week_str = f"Semana {iso_week} ({week_monday.strftime('%d/%m/%Y')} al {week_sunday.strftime('%d/%m/%Y')})"

    city_meta = next((c for c in CITIES if c["city"].lower() == city.lower()), {"city": city, "country": "Global"})
    country = city_meta.get("country", "")

    # 2. Cargar modelos para cada variable objetivo
    models = {}
    models_meta = {}
    for target in TARGETS:
        model, meta = load_target_model(city, target)
        models[target] = model
        models_meta[target] = meta.get("model_name", f"rf_{target}")

    # 3. Cargar observaciones base para los lags iniciales
    seed_df = load_seed_observations(
        city=city,
        country=country,
        latitude=city_meta.get("latitude"),
        longitude=city_meta.get("longitude")
    )

    latest_actual = seed_df.iloc[-1]
    curr_temp = latest_actual.get("temperature_c")
    curr_hum = latest_actual.get("humidity_pct")
    curr_wind = latest_actual.get("wind_speed_kmh")
    curr_press = latest_actual.get("pressure_hpa")

    # 4. Proyección multi-paso recursiva de 7 días (168 horas)
    print(f"\n[INFO] Ejecutando predicción climática de 7 días (168 horas) para {city}...")
    
    # Inicializar matriz de variables predictoras
    # Orden FEATURES: temp_lag_1, temp_lag_2, temp_lag_3, humidity_lag_1, humidity_lag_2, wind_lag_1, wind_lag_2, pressure_lag_1, pressure_lag_2, hour, dayofyear
    start_forecast_time = now.replace(minute=0, second=0, microsecond=0)
    arr = np.zeros((1, len(FEATURES)), dtype=np.float64)
    arr[0, 0] = seed_df.iloc[-1]["temperature_c"]
    arr[0, 1] = seed_df.iloc[-2]["temperature_c"]
    arr[0, 2] = seed_df.iloc[-3]["temperature_c"]

    arr[0, 3] = seed_df.iloc[-1]["humidity_pct"]
    arr[0, 4] = seed_df.iloc[-2]["humidity_pct"]

    arr[0, 5] = seed_df.iloc[-1]["wind_speed_kmh"]
    arr[0, 6] = seed_df.iloc[-2]["wind_speed_kmh"]

    arr[0, 7] = seed_df.iloc[-1]["pressure_hpa"]
    arr[0, 8] = seed_df.iloc[-2]["pressure_hpa"]

    first_target_time = start_forecast_time + timedelta(hours=1)
    arr[0, 9] = first_target_time.hour
    arr[0, 10] = first_target_time.timetuple().tm_yday

    hourly_predictions = []

    for h in range(168):
        current_step_time = start_forecast_time + timedelta(hours=h + 1)
        feat_df = pd.DataFrame(arr, columns=FEATURES)

        p_temp = float(models["temperature_c"].predict(feat_df)[0])
        p_hum = float(models["humidity_pct"].predict(feat_df)[0])
        p_wind = float(models["wind_speed_kmh"].predict(feat_df)[0])
        p_press = float(models["pressure_hpa"].predict(feat_df)[0])

        # Asegurar límites físicos razonables en variables climáticas
        p_hum = max(5.0, min(100.0, p_hum))
        p_wind = max(0.0, p_wind)

        hourly_predictions.append({
            "timestamp": current_step_time,
            "date": current_step_time.date(),
            "temperature_c": p_temp,
            "humidity_pct": p_hum,
            "wind_speed_kmh": p_wind,
            "pressure_hpa": p_press
        })

        # Actualizar lags para la siguiente hora
        arr[0, 2] = arr[0, 1]
        arr[0, 1] = arr[0, 0]
        arr[0, 0] = p_temp

        arr[0, 4] = arr[0, 3]
        arr[0, 3] = p_hum

        arr[0, 6] = arr[0, 5]
        arr[0, 5] = p_wind

        arr[0, 8] = arr[0, 7]
        arr[0, 7] = p_press

        next_time = current_step_time + timedelta(hours=1)
        arr[0, 9] = next_time.hour
        arr[0, 10] = next_time.timetuple().tm_yday

    df_preds = pd.DataFrame(hourly_predictions)

    # 5. Agrupación por días de la semana
    daily_summary = []
    unique_dates = sorted(df_preds["date"].unique())

    for target_date in unique_dates:
        sub = df_preds[df_preds["date"] == target_date]
        w_day = target_date.weekday()
        d_name = DIAS_SEMANA[w_day]
        is_today = (target_date == now.date())
        label = f"{d_name} (Hoy)" if is_today else d_name

        daily_summary.append({
            "date": target_date,
            "day_name": d_name,
            "label": label,
            "temp_min": float(sub["temperature_c"].min()),
            "temp_max": float(sub["temperature_c"].max()),
            "temp_avg": float(sub["temperature_c"].mean()),
            "hum_avg": float(sub["humidity_pct"].mean()),
            "wind_avg": float(sub["wind_speed_kmh"].mean()),
            "press_avg": float(sub["pressure_hpa"].mean())
        })

    # Resumen general consolidado de toda la semana
    week_temp_avg = float(df_preds["temperature_c"].mean())
    week_temp_min = float(df_preds["temperature_c"].min())
    week_temp_max = float(df_preds["temperature_c"].max())
    week_hum_avg = float(df_preds["humidity_pct"].mean())
    week_wind_avg = float(df_preds["wind_speed_kmh"].mean())
    week_press_avg = float(df_preds["pressure_hpa"].mean())

    # 6. Mostrar informe en consola
    print("\n" + "=" * 90)
    print(f" PREDICCIÓN CLIMÁTICA SEMANAL: {city.upper()} ({country})")
    print("=" * 90)
    print(f"  * Día de la consulta    : {consult_str}")
    print(f"  * Semana de la consulta : {week_str}")
    print(f"  * Horizonte proyectado  : 7 días ({unique_dates[0].strftime('%d/%m/%Y')} al {unique_dates[-1].strftime('%d/%m/%Y')})")
    
    curr_t_str = f"{curr_temp:.2f} °C" if curr_temp is not None else "N/D"
    curr_h_str = f"{curr_hum:.1f} %" if curr_hum is not None else "N/D"
    curr_w_str = f"{curr_wind:.1f} km/h" if curr_wind is not None else "N/D"
    curr_p_str = f"{curr_press:.1f} hPa" if curr_press is not None else "N/D"
    print(f"  * Condición actual      : Temp: {curr_t_str} | Humedad: {curr_h_str} | Viento: {curr_w_str} | Presión: {curr_p_str}")
    print("-" * 90)
    print(" PRONÓSTICO DIARIO PARA LA SEMANA:")
    print(f" {'Día':<16} {'Fecha':<12} {'Temp (°C) [Mín / Máx / Prom]':<30} {'Humedad':<11} {'Viento':<12} {'Presión':<10}")
    print("-" * 90)

    for d in daily_summary:
        temp_str = f"{d['temp_min']:5.1f} / {d['temp_max']:5.1f} ({d['temp_avg']:5.1f} °C)"
        hum_str = f"{d['hum_avg']:5.1f} %"
        wind_str = f"{d['wind_avg']:5.1f} km/h"
        press_str = f"{d['press_avg']:6.1f} hPa"
        date_str = d["date"].strftime("%d/%m/%Y")
        print(f" {d['label']:<16} {date_str:<12} {temp_str:<30} {hum_str:<11} {wind_str:<12} {press_str:<10}")

    print("-" * 90)
    print(" RESUMEN GENERAL DE LA SEMANA:")
    print(f"  - Temperatura promedio semanal : {week_temp_avg:5.1f} °C  (Mín: {week_temp_min:5.1f} °C, Máx: {week_temp_max:5.1f} °C)")
    print(f"  - Humedad promedio semanal     : {week_hum_avg:5.1f} %")
    print(f"  - Velocidad de viento promedio : {week_wind_avg:5.1f} km/h")
    print(f"  - Presión atmosférica promedio : {week_press_avg:6.1f} hPa")
    print("=" * 90)

    # 7. Guardar predicciones en Supabase (weather_predictions)
    version_str = datetime.now(timezone.utc).strftime("v_%Y%m%d%H%M")
    prediction_run_str = now.isoformat()

    records_to_insert = []
    for d in daily_summary:
        target_ts = datetime(d["date"].year, d["date"].month, d["date"].day, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        notes_str = (
            f"Predicción semanal ({d['label']} {d['date'].strftime('%d/%m')}): "
            f"Temp {d['temp_min']:.1f}-{d['temp_max']:.1f}°C (prom {d['temp_avg']:.1f}°C), "
            f"Hum {d['hum_avg']:.1f}%, Viento {d['wind_avg']:.1f}km/h, Presión {d['press_avg']:.1f}hPa"
        )
        records_to_insert.append({
            "city": city,
            "model_name": models_meta.get("temperature_c", "rf_models"),
            "model_version": version_str,
            "prediction_run": prediction_run_str,
            "target_timestamp": target_ts,
            "predicted_temperature_c": round(d["temp_avg"], 2),
            "predicted_humidity_pct": round(d["hum_avg"], 2),
            "predicted_wind_speed_kmh": round(d["wind_avg"], 2),
            "predicted_pressure_hpa": round(d["press_avg"], 2),
            "notes": notes_str
        })

    try:
        supabase.table("weather_predictions").insert(records_to_insert).execute()
        print(f" [INFO] Pronóstico de los {len(records_to_insert)} días de la semana guardado exitosamente en Supabase.")
    except Exception as e:
        # Respaldo si faltan algunas columnas en la tabla weather_predictions
        try:
            fallbacks = [{
                "city": r["city"],
                "model_name": r["model_name"],
                "model_version": r["model_version"],
                "prediction_run": r["prediction_run"],
                "target_timestamp": r["target_timestamp"],
                "predicted_temperature_c": r["predicted_temperature_c"],
                "notes": r["notes"]
            } for r in records_to_insert]
            supabase.table("weather_predictions").insert(fallbacks).execute()
            print(" [INFO] Guardado en Supabase con columnas base (temperatura y notas).")
        except Exception as e2:
            print(f" [AVISO] No se pudo guardar en Supabase: {e2}")

    return {
        "city": city,
        "consultation_time": consult_str,
        "week": week_str,
        "daily_summary": daily_summary,
        "weekly_metrics": {
            "temp_avg": week_temp_avg,
            "temp_min": week_temp_min,
            "temp_max": week_temp_max,
            "humidity_avg": week_hum_avg,
            "wind_speed_avg": week_wind_avg,
            "pressure_avg": week_press_avg
        }
    }


# Alias para compatibilidad de nomenclatura
predict_week_for_city = predict_city


if __name__ == "__main__":
    print("\nSelecciona ciudad para predecir los datos climáticos de la semana:")
    for i, c in enumerate(CITIES, start=1):
        print(f"  {i}. {c['city']} ({c['country']})")
    print("  a. Todas las ciudades")

    choice = input("\nIngresa número (o 'a' para todas): ").strip().lower()
    if choice == "a":
        for c in CITIES:
            print(f"\n=== Generando predicciones de la semana para {c['city']} ===")
            try:
                predict_city(c["city"])
            except Exception as e:
                print(f"[ERROR] No se pudo predecir {c['city']}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(CITIES):
        c = CITIES[int(choice) - 1]
        predict_city(c["city"])
    else:
        print("Opción inválida.")
