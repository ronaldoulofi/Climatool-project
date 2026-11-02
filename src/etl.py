
import os
import time
import requests
from datetime import datetime
from supabase import create_client
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Define SUPABASE_URL y SUPABASE_KEY en env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# CONFIG: lista de ciudades del proyecto
CITIES = [
    {"city": "New York", "country": "US"},
    {"city": "Miami", "country": "US"},
    {"city": "Los Angeles", "country": "US"},
    {"city": "Paris", "country": "FR"},
    {"city": "Toulouse", "country": "FR"},
    {"city": "Lyon", "country": "FR"},
]

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def get_coordinates(city: str, country: str, retries=2):
    params = {"name": city, "count": 1, "language": "en", "format": "json", "countryCode": country}
    for attempt in range(retries + 1):
        try:
            res = requests.get(GEOCODING_URL, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data.get("results"):
                return None
            place = data["results"][0]
            return {
                "name": place.get("name"),
                "country": place.get("country"),
                "latitude": float(place.get("latitude")),
                "longitude": float(place.get("longitude"))
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            raise

def upsert_city_cache(city_name: str, country: str, latitude: float, longitude: float):
    payload = {
        "name": city_name,
        "country": country,
        "latitude": latitude,
        "longitude": longitude
    }
    # upsert based on unique (name,country)
    resp = supabase.table("cities").upsert(payload, on_conflict="name,country").execute()
    return resp

def get_cached_city(city_name: str, country: str):
    r = supabase.table("cities").select("*").ilike("name", city_name).execute()
    if r.data and len(r.data) > 0:
        if country:
            country_clean = country.strip().lower()
            iso_map = {"us": "united states", "fr": "france", "es": "spain", "uk": "united kingdom", "de": "germany"}
            expected = iso_map.get(country_clean, country_clean)
            for row in r.data:
                c_country = (row.get("country") or "").strip().lower()
                if c_country == country_clean or c_country == expected or country_clean in c_country or c_country in country_clean:
                    return row
        return r.data[0]
    return None

def get_weather(latitude: float, longitude: float, retries=2):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
        "forecast_days": 7,
        "timezone": "UTC"
    }
    for attempt in range(retries + 1):
        try:
            res = requests.get(WEATHER_URL, params=params, timeout=15)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            raise

def prepare_records_from_weather(city:str, country:str, latitude:float, longitude:float, weather_json:dict, city_id:str = None) -> List[Dict]:
    hourly = weather_json.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    hums = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    press = hourly.get("pressure_msl", [])

    records = []
    for i, ts in enumerate(times):
        rec = {
            "city_id": city_id,
            "city": city,
            "country": country,
            "timestamp": ts,  # ISO string, timezone UTC
            "temperature_c": float(temps[i]) if i < len(temps) else None,
            "humidity_pct": float(hums[i]) if i < len(hums) else None,
            "wind_speed_kmh": float(winds[i]) if i < len(winds) else None,
            "pressure_hpa": float(press[i]) if i < len(press) else None,
            "latitude": latitude,
            "longitude": longitude,
            "source": "open-meteo"
        }
        records.append(rec)
    return records

def insert_weather_records(records: List[Dict]):
    if not records:
        return None
    
    try:
        resp = supabase.table("weather_raw").upsert(records, on_conflict="city,timestamp").execute()
        return resp
    except Exception as e:
      
        try:
            resp = supabase.table("weather_raw").insert(records).execute()
            return resp
        except Exception as e2:
 
            inserted = []
            for r in records:
                try:
                    supabase.table("weather_raw").insert(r).execute()
                    inserted.append(r)
                except Exception:
                    continue
            return {"inserted_count": len(inserted)}

def run_etl_for_city(city_name: str, country: str):
    # 1) check cache
    city_id = None
    cached = get_cached_city(city_name, country)
    if cached:
        city_id = cached.get("id")
        latitude = cached["latitude"]
        longitude = cached["longitude"]
    else:
        coords = get_coordinates(city_name, country)
        if not coords:
            print(f"[WARN] No coords for {city_name}")
            return
        latitude = coords["latitude"]; longitude = coords["longitude"]
        resp = upsert_city_cache(coords["name"], coords["country"], latitude, longitude)
        if resp.data and len(resp.data) > 0:
            city_id = resp.data[0].get("id")
        else:
            cached_refreshed = get_cached_city(coords["name"], coords["country"])
            city_id = cached_refreshed.get("id") if cached_refreshed else None

    # 2) get weather
    weather = get_weather(latitude, longitude)

    # 3) prepare records
    records = prepare_records_from_weather(city_name, country, latitude, longitude, weather, city_id=city_id)

    # 4) insert into supabase (upsert to avoid duplicates)
    result = insert_weather_records(records)
    print(f"[INFO] Insert result for {city_name}: {result}")
    return result

if __name__ == "__main__":
    print("\nSelecciona ciudad para ejecutar ETL:")
    for i, c in enumerate(CITIES, start=1):
        print(f"  {i}. {c['city']} ({c['country']})")
    print("  a. Todas las ciudades")

    choice = input("\nIngresa número (o 'a' para todas): ").strip().lower()
    if choice == "a":
        for c in CITIES:
            print(f"\n=== Ejecutando ETL para {c['city']} ({c['country']}) ===")
            run_etl_for_city(c["city"], c["country"])
    elif choice.isdigit() and 1 <= int(choice) <= len(CITIES):
        c = CITIES[int(choice) - 1]
        print(f"\n=== Ejecutando ETL para {c['city']} ({c['country']}) ===")
        run_etl_for_city(c["city"], c["country"])
    else:
        print("Opción inválida.")