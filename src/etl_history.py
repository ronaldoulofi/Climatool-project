import sys
import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def load_history(city, country=None):
    # Buscar coordenadas de la ciudad
    params = {"name": city, "count": 1, "language": "en"}
    if country:
        params["countryCode"] = country

    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=params,
        timeout=30
    )
    geo.raise_for_status()
    results = geo.json().get("results", [])

    if not results:
        raise ValueError(f"No se encontró la ciudad: {city}")

    place = results[0]
    latitude = place["latitude"]
    longitude = place["longitude"]
    city_name = city
    country_name = place.get("country", country or "")

    # Descargar 2 años de historial (~730 días) hasta la fecha actual
    end_date = date.today()
    start_date = end_date - timedelta(days=730)

    print(f"Extrayendo historial de 2 años ({start_date.isoformat()} a {end_date.isoformat()}) para {city_name}...")

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "wind_speed_10m,"
                "pressure_msl"
            ),
            "wind_speed_unit": "kmh",
            "timezone": "UTC"
        },
        timeout=120
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    times = hourly.get("time", [])
    rows = []
    for i, timestamp in enumerate(times):
        def value(key):
            values = hourly.get(key)
            return values[i] if values and i < len(values) else None

        temp = value("temperature_2m")
        if temp is None:
            continue

        rows.append({
            "city": city_name,
            "country": country_name,
            "timestamp": f"{timestamp}:00+00:00",
            "temperature_c": temp,
            "humidity_pct": value("relative_humidity_2m"),
            "wind_speed_kmh": value("wind_speed_10m"),
            "pressure_hpa": value("pressure_msl"),
            "latitude": latitude,
            "longitude": longitude,
            "source": "open-meteo-archive"
        })

    # Guardar por lotes mediante UPSERT (on_conflict="city,timestamp")
    batch_size = 500
    total_batches = (len(rows) + batch_size - 1) // batch_size
    print(f"Guardando {len(rows)} registros en Supabase en {total_batches} lotes (conservando datos existentes)...")

    for idx, i in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[i:i + batch_size]
        supabase.table("weather_observations").upsert(
            batch,
            on_conflict="city,timestamp"
        ).execute()
        if idx % 5 == 0 or idx == total_batches:
            print(f"  Progreso: lote {idx}/{total_batches} guardado...")

    print(f"Éxito: {len(rows)} registros de 2 años procesados para {city_name}. Los registros existentes se mantuvieron íntegros.")

# Lista de ciudades del proyecto
CITIES = [
    {"city": "New York", "country": "US"},
    {"city": "Miami", "country": "US"},
    {"city": "Los Angeles", "country": "US"},
    {"city": "Paris", "country": "FR"},
    {"city": "Toulouse", "country": "FR"},
    {"city": "Lyon", "country": "FR"},
]

if __name__ == "__main__":
    print("\nSelecciona ciudad para descargar historial:")
    for i, c in enumerate(CITIES, start=1):
        print(f"  {i}. {c['city']} ({c['country']})")
    print("  a. Todas las ciudades")

    choice = input("\nIngresa número (o 'a' para todas): ").strip().lower()
    if choice == "a":
        for c in CITIES:
            print(f"\n=== Descargando historial para {c['city']} ({c['country']}) ===")
            load_history(c["city"], c["country"])
    elif choice.isdigit() and 1 <= int(choice) <= len(CITIES):
        c = CITIES[int(choice) - 1]
        print(f"\n=== Descargando historial para {c['city']} ({c['country']}) ===")
        load_history(c["city"], c["country"])
    else:
        print("Opción inválida.")