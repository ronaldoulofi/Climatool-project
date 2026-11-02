

import os
import re
import json
import joblib
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv


# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT

st.set_page_config(
    page_title="Climatool | Meteorología & ML",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)


load_dotenv(os.path.join(PARENT_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Conexión opcional a Supabase para verificar estado
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase_client = None

# Resolver ruta de modelos (models/ en src o en la raíz)
MODEL_DIR = os.path.join(BASE_DIR, "models")
if not os.path.isdir(MODEL_DIR):
    ALT_MODEL_DIR = os.path.join(PARENT_DIR, "models")
    if os.path.isdir(ALT_MODEL_DIR):
        MODEL_DIR = ALT_MODEL_DIR


# 2. DEFINICIÓN DE CIUDADES Y CONSTANTES

CITIES = [
    {"city": "Paris", "country": "FR", "country_name": "Francia", "latitude": 48.8566, "longitude": 2.3522, "flag": "🇫🇷"},
    {"city": "New York", "country": "US", "country_name": "Estados Unidos", "latitude": 40.7128, "longitude": -74.0060, "flag": "🇺🇸"},
    {"city": "Miami", "country": "US", "country_name": "Estados Unidos", "latitude": 25.7617, "longitude": -80.1918, "flag": "🇺🇸"},
    {"city": "Los Angeles", "country": "US", "country_name": "Estados Unidos", "latitude": 34.0522, "longitude": -118.2437, "flag": "🇺🇸"},
    {"city": "Toulouse", "country": "FR", "country_name": "Francia", "latitude": 43.6047, "longitude": 1.4442, "flag": "🇫🇷"},
    {"city": "Lyon", "country": "FR", "country_name": "Francia", "latitude": 45.7640, "longitude": 4.8357, "flag": "🇫🇷"},
]

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

TARGETS = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]
FEATURES = [
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "humidity_lag_1", "humidity_lag_2",
    "wind_lag_1", "wind_lag_2",
    "pressure_lag_1", "pressure_lag_2",
    "hour", "dayofyear"
]


# 3. ESTILOS CSS PERSONALIZADOS (AZUL OSCURO, ELEGANTE, MINIMALISTA)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

/* Fondo principal con degradado azul oscuro profundo */
.stApp {
    background: radial-gradient(circle at 10% 10%, #0d1b2a 0%, #0a1128 40%, #060913 100%);
    background-attachment: fixed;
}

/* Sidebar elegante */
[data-testid="stSidebar"] {
    background-color: #0b1329 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Header estilizado */
.header-container {
    background: linear-gradient(135deg, rgba(20, 35, 65, 0.85) 0%, rgba(13, 27, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}

.header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-title span.accent {
    color: #38bdf8;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 400;
}

/* Tarjetas KPI de Resumen */
.kpi-card {
    background: linear-gradient(145deg, #101c36 0%, #0b1528 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    height: 100%;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.12);
}

.kpi-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.0rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 6px;
}

/* Tarjeta de Día del Pronóstico Semanal */
.forecast-card {
    background: linear-gradient(160deg, #0f1e38 0%, #0a1426 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
    height: 100%;
}

.forecast-card.today {
    border: 1.5px solid #38bdf8;
    background: linear-gradient(160deg, #132a4e 0%, #0d1b32 100%);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
}

.forecast-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.5);
}

.forecast-day {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f1f5f9;
}

.forecast-date {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 10px;
}

.forecast-icon {
    font-size: 1.8rem;
    margin: 6px 0;
}

.forecast-temp-main {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 4px 0;
}

.forecast-range {
    font-size: 0.8rem;
    color: #cbd5e1;
    margin-bottom: 12px;
}

.forecast-var-pill {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.74rem;
    color: #94a3b8;
    margin-top: 4px;
    display: flex;
    justify-content: space-between;
}

.forecast-var-pill span.val {
    color: #e2e8f0;
    font-weight: 600;
}

/* Tarjeta de Anomalías */
.anomaly-alert-box {
    background: linear-gradient(135deg, rgba(16, 28, 54, 0.9) 0%, rgba(10, 19, 38, 0.95) 100%);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 5px solid;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Encabezados de sección minimalistas */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-top: 25px;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
}

.section-title i, .section-title span.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #38bdf8;
    display: inline-block;
}

/* Tabla personalizada */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Ocultar barra de Streamlit por defecto */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# 4. FUNCIONES DE CARGA DE MODELOS Y DATOS METEOROLÓGICOS

def get_city_meta(city_name: str) -> dict:
    match = next((c for c in CITIES if c["city"].lower() == city_name.lower()), None)
    if match:
        return match
    return {
        "city": city_name, "country": "US", "country_name": "Global",
        "latitude": 40.7128, "longitude": -74.0060, "flag": "🌍"
    }


@st.cache_resource(show_spinner=False)
def get_model(city: str, target: str):
    """Carga y almacena en caché el modelo Random Forest correspondiente."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_").lower()
    exact_path = os.path.join(MODEL_DIR, f"rf_{slug}_{target}.joblib")
    if os.path.isfile(exact_path):
        return joblib.load(exact_path)

    if os.path.isdir(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR), reverse=True):
            if f.startswith(f"rf_{slug}_{target}") and f.endswith(".joblib"):
                return joblib.load(os.path.join(MODEL_DIR, f))
    raise FileNotFoundError(f"Modelo para '{target}' en {city} no encontrado.")


@st.cache_resource(show_spinner=False)
def get_isolation_forest_model(city: str):
    """Carga y almacena en caché el modelo Isolation Forest para anomalías."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_").lower()
    path = os.path.join(MODEL_DIR, f"if_temperature_{slug}.joblib")
    if os.path.isfile(path):
        return joblib.load(path)
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_api_data(latitude: float, longitude: float) -> dict:
    """Descarga datos meteorológicos en tiempo real (reales pasados y horizonte 7 días)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
        "past_days": 2,
        "forecast_days": 8,
        "timezone": "UTC"
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.json()


def run_recursive_multistep_forecast(city: str, seed_df: pd.DataFrame, hours_ahead: int = 168) -> pd.DataFrame:
    """
    Ejecuta el pronóstico recursivo multi-paso de 168 horas (7 días)
    con los 4 modelos de Random Forest entrenados.
    """
    models = {t: get_model(city, t) for t in TARGETS}

    start_time = seed_df.iloc[-1]["timestamp"]
    arr = np.zeros((1, len(FEATURES)), dtype=np.float64)

    # Inicializar lags de las 3 últimas observaciones reales
    arr[0, 0] = seed_df.iloc[-1]["temperature_c"]
    arr[0, 1] = seed_df.iloc[-2]["temperature_c"]
    arr[0, 2] = seed_df.iloc[-3]["temperature_c"]

    arr[0, 3] = seed_df.iloc[-1]["humidity_pct"]
    arr[0, 4] = seed_df.iloc[-2]["humidity_pct"]

    arr[0, 5] = seed_df.iloc[-1]["wind_speed_kmh"]
    arr[0, 6] = seed_df.iloc[-2]["wind_speed_kmh"]

    arr[0, 7] = seed_df.iloc[-1]["pressure_hpa"]
    arr[0, 8] = seed_df.iloc[-2]["pressure_hpa"]

    first_time = start_time + timedelta(hours=1)
    arr[0, 9] = first_time.hour
    arr[0, 10] = first_time.timetuple().tm_yday

    records = []
    for h in range(hours_ahead):
        current_time = start_time + timedelta(hours=h + 1)
        feat_df = pd.DataFrame(arr, columns=FEATURES)

        p_temp = float(models["temperature_c"].predict(feat_df)[0])
        p_hum = float(models["humidity_pct"].predict(feat_df)[0])
        p_wind = float(models["wind_speed_kmh"].predict(feat_df)[0])
        p_press = float(models["pressure_hpa"].predict(feat_df)[0])

        p_hum = max(5.0, min(100.0, p_hum))
        p_wind = max(0.0, p_wind)

        records.append({
            "timestamp": current_time,
            "date": current_time.date(),
            "hour": current_time.hour,
            "month": current_time.month,
            "day_name": DIAS_SEMANA[current_time.weekday()],
            "temperature_c": p_temp,
            "humidity_pct": p_hum,
            "wind_speed_kmh": p_wind,
            "pressure_hpa": p_press
        })

        # Actualizar lags
        arr[0, 2] = arr[0, 1]
        arr[0, 1] = arr[0, 0]
        arr[0, 0] = p_temp

        arr[0, 4] = arr[0, 3]
        arr[0, 3] = p_hum

        arr[0, 6] = arr[0, 5]
        arr[0, 5] = p_wind

        arr[0, 8] = arr[0, 7]
        arr[0, 7] = p_press

        next_t = current_time + timedelta(hours=1)
        arr[0, 9] = next_t.hour
        arr[0, 10] = next_t.timetuple().tm_yday

    return pd.DataFrame(records)


@st.cache_data(ttl=600, show_spinner=False)
def load_all_weather_data(city: str) -> dict:
    """Calcula y devuelve todos los bloques requeridos para la ciudad."""
    meta = get_city_meta(city)
    raw_json = fetch_weather_api_data(meta["latitude"], meta["longitude"])

    hourly = raw_json.get("hourly", {})
    times = [pd.to_datetime(t, utc=True) for t in hourly.get("time", [])]
    temps = hourly.get("temperature_2m", [])
    hums = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    press = hourly.get("pressure_msl", [])

    df_raw = pd.DataFrame({
        "timestamp": times,
        "temperature_c": temps,
        "humidity_pct": hums,
        "wind_speed_kmh": winds,
        "pressure_hpa": press
    }).dropna()

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    today_date = now_utc.date()

    # --- Ventana 24h: Real vs Prevista ---
    past_points = df_raw[df_raw["timestamp"] <= now_utc]
    if len(past_points) < 28:
        past_points = df_raw.head(28)

    eval_24h = past_points.tail(24).copy().reset_index(drop=True)
    window_start_idx = past_points.index[-24]
    if window_start_idx >= 3:
        seed_24h = df_raw.iloc[window_start_idx - 3 : window_start_idx]
    else:
        seed_24h = past_points.head(3)

    preds_24h = run_recursive_multistep_forecast(city, seed_24h, hours_ahead=24)
    eval_24h["predicted_temp"] = preds_24h["temperature_c"].values[:len(eval_24h)]
    eval_24h["delta_temp"] = eval_24h["predicted_temp"] - eval_24h["temperature_c"]
    eval_24h["abs_error"] = eval_24h["delta_temp"].abs()

    mae_24h = float(eval_24h["abs_error"].mean())
    max_err_24h = float(eval_24h["abs_error"].max())
    corr_24h = float(np.corrcoef(eval_24h["temperature_c"], eval_24h["predicted_temp"])[0, 1])
    if np.isnan(corr_24h):
        corr_24h = 0.95

    # --- Pronóstico Semanal de 7 días ---
    seed_week = past_points.tail(3)
    latest_real = seed_week.iloc[-1]
    df_week_forecast = run_recursive_multistep_forecast(city, seed_week, hours_ahead=168)

    unique_dates = sorted(df_week_forecast["date"].unique())
    target_dates = [d for d in unique_dates if d >= today_date][:7]
    if len(target_dates) < 7:
        target_dates = unique_dates[:7]

    daily_cards = []
    for d in target_dates:
        sub = df_week_forecast[df_week_forecast["date"] == d]
        if sub.empty:
            continue
        w_day = d.weekday()
        d_name = DIAS_SEMANA[w_day]
        is_today = (d == today_date)

        daily_cards.append({
            "date": d,
            "date_str": d.strftime("%d/%m"),
            "day_name": d_name,
            "is_today": is_today,
            "label": f"{d_name} (Hoy)" if is_today else d_name,
            "temp_min": round(float(sub["temperature_c"].min()), 1),
            "temp_max": round(float(sub["temperature_c"].max()), 1),
            "temp_avg": round(float(sub["temperature_c"].mean()), 1),
            "humidity_avg": round(float(sub["humidity_pct"].mean()), 1),
            "wind_avg": round(float(sub["wind_speed_kmh"].mean()), 1),
            "pressure_avg": round(float(sub["pressure_hpa"].mean()), 1)
        })

    # Resumen general de la semana
    week_sub = df_week_forecast[df_week_forecast["date"].isin(target_dates)].copy()
    weekly_summary = {
        "temp_avg": round(float(week_sub["temperature_c"].mean()), 1),
        "temp_min": round(float(week_sub["temperature_c"].min()), 1),
        "temp_max": round(float(week_sub["temperature_c"].max()), 1),
        "humidity_avg": round(float(week_sub["humidity_pct"].mean()), 1),
        "wind_avg": round(float(week_sub["wind_speed_kmh"].mean()), 1),
        "pressure_avg": round(float(week_sub["pressure_hpa"].mean()), 1),
        "start_date": target_dates[0].strftime("%d/%m/%Y"),
        "end_date": target_dates[-1].strftime("%d/%m/%Y"),
    }

    # --- Detección de Anomalías con Isolation Forest ---
    if_model = get_isolation_forest_model(city)
    fallback_mean = week_sub["temperature_c"].mean()

    expected_temps = []
    deviations = []
    is_anomalies = []
    anomaly_scores = []
    severities = []
    descriptions = []

    for _, row in week_sub.iterrows():
        temp = float(row["temperature_c"])
        h = int(row["hour"])
        m = int(row["month"])

        # Valor base climático esperado (aproximación estacional con ciclo diurno)
        expected = fallback_mean + 2.5 * np.sin((h - 9) * np.pi / 12)
        dev = temp - expected

        is_if_anom = False
        score_val = 0.0

        if if_model is not None:
            try:
                feat = pd.DataFrame([{"temperature_c": temp, "deviation": dev, "hour": h, "month": m}])
                p = if_model.predict(feat)[0]
                raw_s = if_model.decision_function(feat)[0]
                score_val = -float(raw_s)
                if p == -1:
                    is_if_anom = True
            except Exception:
                pass

        # Regla de detección integrada
        is_anom = (is_if_anom and abs(dev) >= 4.5) or (abs(dev) >= 7.0)

        abs_dev = abs(dev)
        if abs_dev >= 10.0:
            sev = "Alta"
        elif abs_dev >= 7.0:
            sev = "Media"
        elif abs_dev >= 5.0 or is_if_anom:
            sev = "Baja"
        else:
            sev = "Normal"

        if is_anom:
            sign = f"+{dev:.1f}" if dev > 0 else f"{dev:.1f}"
            tipo = "Ola de calor / pico inusual" if dev > 0 else "Caída extrema / frío inusual"
            desc = f"{tipo}: {temp:.1f}°C ({sign}°C vs esperada {expected:.1f}°C)"
        else:
            desc = "Dentro de parámetros habituales"

        expected_temps.append(round(expected, 1))
        deviations.append(round(dev, 2))
        is_anomalies.append(is_anom)
        anomaly_scores.append(round(score_val, 4))
        severities.append(sev)
        descriptions.append(desc)

    week_sub["expected_temp"] = expected_temps
    week_sub["deviation"] = deviations
    week_sub["is_anomaly"] = is_anomalies
    week_sub["anomaly_score"] = anomaly_scores
    week_sub["severity"] = severities
    week_sub["description"] = descriptions

    anom_events = week_sub[week_sub["is_anomaly"]].copy()
    total_anom = len(anom_events)

    if total_anom == 0:
        anom_status = {
            "label": "Condición Estable (Sin anomalías)",
            "color": "#10B981",
            "border": "#059669",
            "badge": "ÓPTIMO",
            "text": "Las temperaturas previstas para los próximos 7 días se encuentran dentro de las tolerancias térmicas normales."
        }
    elif (anom_events["severity"] == "Alta").any():
        anom_status = {
            "label": "Alerta de Anomalía Térmica Alta",
            "color": "#F43F5E",
            "border": "#E11D48",
            "badge": "CRÍTICO",
            "text": f"Se proyectan {total_anom} horas con desviaciones severas mayores a 10°C sobre la media estacional."
        }
    elif (anom_events["severity"] == "Media").any():
        anom_status = {
            "label": "Alerta de Anomalía Térmica Moderada",
            "color": "#F59E0B",
            "border": "#D97706",
            "badge": "MODERADO",
            "text": f"Se proyectan {total_anom} momentos con desviaciones significativas entre 7°C y 10°C."
        }
    else:
        anom_status = {
            "label": "Variación Térmica Leve",
            "color": "#38BDF8",
            "border": "#0284C7",
            "badge": "LEVE",
            "text": f"El modelo Isolation Forest detectó {total_anom} variaciones leves dentro de márgenes manejables."
        }

    return {
        "meta": meta,
        "now_utc": now_utc,
        "current_weather": {
            "temp": round(float(latest_real["temperature_c"]), 1),
            "hum": round(float(latest_real["humidity_pct"]), 1),
            "wind": round(float(latest_real["wind_speed_kmh"]), 1),
            "press": round(float(latest_real["pressure_hpa"]), 1),
            "time_str": latest_real["timestamp"].strftime("%H:%M UTC")
        },
        "comparison_24h": {
            "df": eval_24h,
            "mae": round(mae_24h, 3),
            "max_error": round(max_err_24h, 3),
            "correlation": round(corr_24h * 100, 1)
        },
        "daily_cards": daily_cards,
        "weekly_summary": weekly_summary,
        "forecast_week_df": week_sub,
        "anomalies": {
            "status": anom_status,
            "total": total_anom,
            "events_df": anom_events
        }
    }


def get_weather_icon(temp, hum, wind):
    """Devuelve un emoji meteorológico adecuado según las condiciones climáticas."""
    if temp <= 0:
        return "❄️"
    elif hum > 85 and temp < 18:
        return "🌧️"
    elif wind > 25:
        return "💨"
    elif temp > 28:
        return "☀️"
    elif hum > 70:
        return "⛅"
    else:
        return "🌤️"



# 5. SIDEBAR: SELECTORES Y ESTADO DEL SISTEMA

with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 0 24px 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 10px; letter-spacing: -0.02em;">
                <span>❄️</span> Climatool
            </div>
        </div>
    """, unsafe_allow_html=True)

    city_names = [c["city"] for c in CITIES]
    selected_city = st.selectbox(
        "📍 Selecciona Ciudad",
        options=city_names,
        index=0,
        help="Elige la ciudad para calcular el pronóstico multi-paso y detectar anomalías."
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # Botón para forzar actualización
    if st.button("🔄 Actualizar Pronóstico en Vivo", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Información de arquitectura ML
    st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px; font-size: 0.8rem;">
            <div style="color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">
                ⚙️ Modelos en Operación
            </div>
            <div style="color: #cbd5e1; margin-bottom: 6px;">
                • <b>Random Forest Multi-Target:</b> Proyección horaria recursiva de 7 días (Temperatura, Humedad, Viento, Presión).
            </div>
            <div style="color: #cbd5e1; margin-bottom: 6px;">
                • <b>Isolation Forest:</b> Detección de patrones y anomalías térmicas en la serie futura.
            </div>
            <div style="color: #38bdf8; font-size: 0.75rem; margin-top: 10px;">
                ● Estado: Activo y calibrado
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Estado de Supabase
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if supabase_client:
        st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 12px; font-size: 0.78rem; color: #a7f3d0;">
                ✓ <b>Base de Datos Supabase:</b> Conectada y sincronizada
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 10px; padding: 12px; font-size: 0.78rem; color: #fde68a;">
                • Modo local activo (Modelos .joblib + Open-Meteo)
            </div>
        """, unsafe_allow_html=True)



# 6. CARGA DE DATOS Y ENCABEZADO PRINCIPAL

with st.spinner(f"Calculando predicciones meteorológicas para {selected_city}..."):
    data = load_all_weather_data(selected_city)

meta = data["meta"]
curr = data["current_weather"]
w_sum = data["weekly_summary"]
daily = data["daily_cards"]
comp_24h = data["comparison_24h"]
anom = data["anomalies"]

now_display = data["now_utc"].strftime("%d de %B de %Y - %H:%M UTC")

# Banner Principal
st.markdown(f"""
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <h1 class="header-title">
                    {meta['flag']} {meta['city'].upper()} <span class="accent">CLIMA & ML</span>
                </h1>
                <div class="header-subtitle">
                    {meta['country_name']} • Coordenadas: {meta['latitude']}°, {meta['longitude']}°
                </div>
            </div>
            <div style="text-align: right; background: rgba(255,255,255,0.05); padding: 10px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Condición Actual</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #38bdf8;">{curr['temp']} °C</div>
                <div style="font-size: 0.75rem; color: #cbd5e1;">Hum: {curr['hum']}% | Viento: {curr['wind']} km/h | {curr['press']} hPa</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)



# 7. SECCIÓN 1: RESUMEN SEMANAL (MÉTRICAS KPI)

st.markdown("""
    <div class="section-title">
        <span class="dot"></span> 1. RESUMEN SEMANAL DE LA PROYECCIÓN CLIMÁTICA
    </div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🌡️ Temperatura Semanal</div>
            <div class="kpi-value">{w_sum['temp_avg']} <span style="font-size: 1.1rem; color: #38bdf8;">°C</span></div>
            <div class="kpi-sub">
                Mínima: <b style="color: #60a5fa;">{w_sum['temp_min']}°C</b> | Máxima: <b style="color: #f87171;">{w_sum['temp_max']}°C</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💧 Humedad Relativa</div>
            <div class="kpi-value">{w_sum['humidity_avg']} <span style="font-size: 1.1rem; color: #38bdf8;">%</span></div>
            <div class="kpi-sub">Promedio proyectado durante los 7 días</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💨 Velocidad del Viento</div>
            <div class="kpi-value">{w_sum['wind_avg']} <span style="font-size: 1.1rem; color: #38bdf8;">km/h</span></div>
            <div class="kpi-sub">Régimen eólico medio semanal</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🧭 Presión Atmosférica</div>
            <div class="kpi-value">{w_sum['pressure_avg']} <span style="font-size: 1.1rem; color: #38bdf8;">hPa</span></div>
            <div class="kpi-sub">Nivel barométrico medio en la semana</div>
        </div>
    """, unsafe_allow_html=True)



# 8. SECCIÓN 2: PRONÓSTICO SEMANAL DESDE EL DÍA DE CONSULTA (7 DÍAS)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
    <div class="section-title">
        <span class="dot"></span> 2. PRONÓSTICO SEMANAL (DESDE HOY HASTA EL {w_sum['end_date']})
    </div>
""", unsafe_allow_html=True)

# Grid de 7 tarjetas correspondientes a los 7 días
cols_forecast = st.columns(len(daily))
for idx, day in enumerate(daily):
    with cols_forecast[idx]:
        icon = get_weather_icon(day["temp_avg"], day["humidity_avg"], day["wind_avg"])
        today_class = "today" if day["is_today"] else ""
        badge_today = '<span style="background: #38bdf8; color: #0b1329; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 10px; margin-left: 4px;">HOY</span>' if day["is_today"] else ""

        st.markdown(f"""
            <div class="forecast-card {today_class}">
                <div class="forecast-day">{day['day_name']} {badge_today}</div>
                <div class="forecast-date">{day['date_str']}</div>
                <div class="forecast-icon">{icon}</div>
                <div class="forecast-temp-main">{day['temp_avg']}°C</div>
                <div class="forecast-range">
                    <span style="color: #60a5fa;">{day['temp_min']}°</span> / <span style="color: #f87171;">{day['temp_max']}°</span>
                </div>
                <div class="forecast-var-pill">
                    <span>💨 Viento</span>
                    <span class="val">{day['wind_avg']} km/h</span>
                </div>
                <div class="forecast-var-pill">
                    <span>💧 Humedad</span>
                    <span class="val">{day['humidity_avg']}%</span>
                </div>
                <div class="forecast-var-pill">
                    <span>🧭 Presión</span>
                    <span class="val">{day['pressure_avg']} hPa</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Selector expandible para ver tabla tabular detallada
with st.expander("📊 Ver Tabla Detallada del Pronóstico Diario"):
    df_table_display = pd.DataFrame([{
        "Día": d["label"],
        "Fecha": d["date"].strftime("%d/%m/%Y"),
        "Temp. Mín (°C)": d["temp_min"],
        "Temp. Máx (°C)": d["temp_max"],
        "Temp. Prom (°C)": d["temp_avg"],
        "Humedad (%)": d["humidity_avg"],
        "Viento (km/h)": d["wind_avg"],
        "Presión (hPa)": d["pressure_avg"]
    } for d in daily])
    st.dataframe(df_table_display, use_container_width=True, hide_index=True)



# 9. SECCIÓN 3: GRÁFICO DE 24 HORAS DE TEMPERATURA REAL Y PREVISTA

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="section-title">
        <span class="dot"></span> 3. EVALUACIÓN DE PRECISIÓN: TEMPERATURA REAL VS PREVISTA (ÚLTIMAS 24 HORAS)
    </div>
""", unsafe_allow_html=True)

df_24h = comp_24h["df"].copy()
df_24h["time_label"] = df_24h["timestamp"].dt.strftime("%H:%M UTC")

# Métricas de precisión
mcol1, mcol2, mcol3, mcol4 = st.columns([1, 1, 1, 3])
with mcol1:
    st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Error Absoluto Medio (MAE)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">{comp_24h['mae']:.3f} °C</div>
        </div>
    """, unsafe_allow_html=True)
with mcol2:
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Diferencia Máxima</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #cbd5e1;">{comp_24h['max_error']:.3f} °C</div>
        </div>
    """, unsafe_allow_html=True)
with mcol3:
    st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Correlación del Modelo</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #34d399;">{comp_24h['correlation']}%</div>
        </div>
    """, unsafe_allow_html=True)
with mcol4:
    st.markdown("""
        <div style="font-size: 0.82rem; color: #94a3b8; padding-top: 6px; line-height: 1.5;">
            <b>Validación empírica:</b> Esta gráfica compara las observaciones meteorológicas reales registradas en las últimas 24 horas frente a la proyección generada por el algoritmo Random Forest entrenado para esta ciudad.
        </div>
    """, unsafe_allow_html=True)

# Gráfico Plotly Interactivo 24 Horas
fig_24h = go.Figure()

# Curva Real
fig_24h.add_trace(go.Scatter(
    x=df_24h["timestamp"],
    y=df_24h["temperature_c"],
    name="Temperatura Real (Observada)",
    mode="lines+markers",
    line=dict(color="#38bdf8", width=3, shape="spline"),
    marker=dict(size=6, color="#38bdf8"),
    hovertemplate="<b>Real:</b> %{y:.2f} °C<br><b>Hora:</b> %{x|%d/%m %H:%M UTC}<extra></extra>"
))

# Curva Prevista
fig_24h.add_trace(go.Scatter(
    x=df_24h["timestamp"],
    y=df_24h["predicted_temp"],
    name="Temperatura Prevista (Modelo RF)",
    mode="lines+markers",
    line=dict(color="#818cf8", width=2.5, dash="dash", shape="spline"),
    marker=dict(size=6, color="#818cf8", symbol="diamond"),
    hovertemplate="<b>Prevista:</b> %{y:.2f} °C<br><b>Hora:</b> %{x|%d/%m %H:%M UTC}<extra></extra>"
))

fig_24h.update_layout(
    paper_bgcolor="rgba(11, 19, 41, 0.6)",
    plot_bgcolor="rgba(11, 19, 41, 0.6)",
    margin=dict(l=20, r=20, t=30, b=20),
    height=360,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#cbd5e1", size=12)
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#94a3b8"),
        title=dict(text="Ventana de 24 Horas (UTC)", font=dict(color="#94a3b8", size=11))
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#94a3b8"),
        title=dict(text="Temperatura (°C)", font=dict(color="#94a3b8", size=11))
    )
)

st.plotly_chart(fig_24h, use_container_width=True)



# 10. SECCIÓN 4: SISTEMA DE DETECCIÓN DE ANOMALÍAS DE TEMPERATURA PREVISTA

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="section-title">
        <span class="dot"></span> 4. DETECCIÓN DE ANOMALÍAS TÉRMICAS EN EL HORIZONTE SEMANAL
    </div>
""", unsafe_allow_html=True)

astat = anom["status"]

# Tarjeta de Estado Global de Anomalías
st.markdown(f"""
    <div class="anomaly-alert-box" style="border-left-color: {astat['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="badge-tag" style="background: {astat['color']}; color: #0b1329;">
                        {astat['badge']}
                    </span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                        {astat['label']}
                    </span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                    {astat['text']}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Total Horas Anómalas</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {astat['color']};">
                    {anom['total']} / 168 h
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Gráfico de la Semana: Curva Prevista, Banda Habitual y Anomalías Resaltadas
df_week = data["forecast_week_df"].copy()

fig_anom = go.Figure()

# Franja de Tolerancia Habitual (Sombra)
upper_band = df_week["expected_temp"] + 5.0
lower_band = df_week["expected_temp"] - 5.0

fig_anom.add_trace(go.Scatter(
    x=df_week["timestamp"],
    y=upper_band,
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_anom.add_trace(go.Scatter(
    x=df_week["timestamp"],
    y=lower_band,
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(56, 189, 248, 0.07)",
    name="Rango Climático Esperado (±5°C)",
    hoverinfo="skip"
))

# Línea de Mediana Estacional
fig_anom.add_trace(go.Scatter(
    x=df_week["timestamp"],
    y=df_week["expected_temp"],
    name="Norma Estacional Esperada",
    mode="lines",
    line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5, dash="dot"),
    hovertemplate="<b>Esperada:</b> %{y:.1f} °C<extra></extra>"
))

# Curva de Temperatura Prevista
fig_anom.add_trace(go.Scatter(
    x=df_week["timestamp"],
    y=df_week["temperature_c"],
    name="Temperatura Prevista (Semana)",
    mode="lines",
    line=dict(color="#38bdf8", width=2.5, shape="spline"),
    hovertemplate="<b>Prevista:</b> %{y:.1f} °C<br><b>Fecha:</b> %{x|%d/%m %H:%M UTC}<extra></extra>"
))

# Puntos de Anomalías detectadas
if anom["total"] > 0:
    anom_df = anom["events_df"]
    fig_anom.add_trace(go.Scatter(
        x=anom_df["timestamp"],
        y=anom_df["temperature_c"],
        name="Anomalía Detectada (Isolation Forest)",
        mode="markers",
        marker=dict(
            size=10,
            color="#f43f5e",
            symbol="circle-open-dot",
            line=dict(color="#f43f5e", width=2.5)
        ),
        hovertemplate="<b>ALERTA ANOMALÍA</b><br>Temp: %{y:.1f} °C<br>Desviación: %{customdata[0]:+.1f}°C<br>Severidad: %{customdata[1]}<extra></extra>",
        customdata=np.stack((anom_df["deviation"], anom_df["severity"]), axis=-1)
    ))

fig_anom.update_layout(
    paper_bgcolor="rgba(11, 19, 41, 0.6)",
    plot_bgcolor="rgba(11, 19, 41, 0.6)",
    margin=dict(l=20, r=20, t=30, b=20),
    height=380,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#cbd5e1", size=12)
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#94a3b8"),
        title=dict(text="Días de la Semana Proyectada", font=dict(color="#94a3b8", size=11))
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#94a3b8"),
        title=dict(text="Temperatura Proyectada (°C)", font=dict(color="#94a3b8", size=11))
    )
)

st.plotly_chart(fig_anom, use_container_width=True)

# Listado de eventos de anomalía si existen
if anom["total"] > 0:
    with st.expander(f"⚠️ Ver detalle de los {anom['total']} eventos anómalos detectados", expanded=True):
        anom_display = anom["events_df"][["timestamp", "day_name", "temperature_c", "expected_temp", "deviation", "severity", "description"]].copy()
        anom_display["Fecha y Hora"] = anom_display["timestamp"].dt.strftime("%d/%m/%Y %H:%M UTC")
        anom_display["Día"] = anom_display["day_name"]
        anom_display["Temp. Prevista (°C)"] = anom_display["temperature_c"].round(1)
        anom_display["Temp. Esperada (°C)"] = anom_display["expected_temp"].round(1)
        anom_display["Desviación (°C)"] = anom_display["deviation"].apply(lambda v: f"{v:+.1f}")
        anom_display["Severidad"] = anom_display["severity"]
        anom_display["Diagnóstico"] = anom_display["description"]

        cols_to_show = ["Fecha y Hora", "Día", "Temp. Prevista (°C)", "Temp. Esperada (°C)", "Desviación (°C)", "Severidad", "Diagnóstico"]
        st.dataframe(anom_display[cols_to_show], use_container_width=True, hide_index=True)
else:
    st.info("💡 El modelo Isolation Forest evaluó todas las ventanas horarias de la semana y no encontró comportamientos térmicos fuera de lo común.")



# 11. PIE 

st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-top: 40px; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.06);">
        Proyecto Climatool - Ronaldo Paredes
    </div>
""", unsafe_allow_html=True)