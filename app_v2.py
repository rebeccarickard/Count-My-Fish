import math
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

import joblib
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_geolocation import streamlit_geolocation
import plotly.express as px

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="NC Fishing Score", page_icon="🎣", layout="centered")

page = st.sidebar.radio(
    "Navigation",
    ["Fishing Score", "Log Catch", "My Catch Log", "Challenges", "Community", "Map"]
)

APP_MODE = st.sidebar.radio(
    "Connection Mode",
    ["Auto", "Online", "Offline"],
    help="Auto tries live APIs/PostgreSQL first, then falls back to local offline data."
)

LOCAL_DB_PATH = Path("offline_fishing_cache.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")
DEMO_USER_ID = os.getenv("DEMO_USER_ID", "demo_user")


# -------------------------------------------------
# LOAD TRAINED MODEL + TRAINING COLUMNS
# -------------------------------------------------
@st.cache_resource
def load_model_assets():
    model = joblib.load("xgb_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns


# -------------------------------------------------
# DISPLAY COLUMN NAMES
# -------------------------------------------------
DISPLAY_COLUMN_NAMES = {
    "timestamp": "Date/Time",
    "species": "Species",
    "fish_count": "Fish Caught",
    "avg_size": "Average Size (in)",
    "bait": "Bait",
    "technique": "Technique",
    "notes": "Notes",
    "water_body": "Water Body",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "location_label": "Location",
    "resolved_location": "Resolved Location",
    "total_fish_caught": "Total Fish Caught",
}


# -------------------------------------------------
# ONLINE/OFFLINE + DATABASE HELPERS
# -------------------------------------------------
def network_available() -> bool:
    """Lightweight online check used by Auto mode."""
    if APP_MODE == "Offline":
        return False
    if APP_MODE == "Online":
        return True
    try:
        requests.get("https://api.open-meteo.com", timeout=3)
        return True
    except Exception:
        return False


def postgres_available() -> bool:
    return bool(DATABASE_URL) and psycopg2 is not None and network_available()


@st.cache_resource
def get_local_conn():
    conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_postgres_conn():
    if not postgres_available():
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_local_db():
    conn = get_local_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS catches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT,
            species TEXT,
            fish_count INTEGER,
            avg_size REAL,
            bait TEXT,
            technique TEXT,
            notes TEXT,
            water_body TEXT,
            latitude REAL,
            longitude REAL,
            resolved_location TEXT,
            pending_sync INTEGER DEFAULT 1
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conditions_cache (
            cache_key TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            weather_json TEXT,
            tide_json TEXT,
            cached_at TEXT
        )
    """)
    conn.commit()


def init_postgres_db():
    conn = get_postgres_conn()
    if conn is None:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS catches (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT,
                    species TEXT,
                    fish_count INTEGER,
                    avg_size DOUBLE PRECISION,
                    bait TEXT,
                    technique TEXT,
                    notes TEXT,
                    water_body TEXT,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    resolved_location TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conditions_cache (
                    cache_key TEXT PRIMARY KEY,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    weather_json JSONB,
                    tide_json JSONB,
                    cached_at TEXT
                )
            """)
    conn.close()


def init_storage():
    init_local_db()
    init_postgres_db()


def cache_key_for_location(lat: float, lon: float) -> str:
    return f"{round(float(lat), 2)}_{round(float(lon), 2)}"


def cache_conditions(lat: float, lon: float, weather: dict, tide: dict):
    key = cache_key_for_location(lat, lon)
    cached_at = datetime.now().isoformat(timespec="minutes")

    local = get_local_conn()
    local.execute(
        """
        INSERT OR REPLACE INTO conditions_cache
        (cache_key, latitude, longitude, weather_json, tide_json, cached_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (key, lat, lon, json.dumps(weather), json.dumps(tide), cached_at),
    )
    local.commit()

    conn = get_postgres_conn()
    if conn is not None:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conditions_cache
                    (cache_key, latitude, longitude, weather_json, tide_json, cached_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key)
                    DO UPDATE SET
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        weather_json = EXCLUDED.weather_json,
                        tide_json = EXCLUDED.tide_json,
                        cached_at = EXCLUDED.cached_at
                    """,
                    (key, lat, lon, json.dumps(weather), json.dumps(tide), cached_at),
                )
        conn.close()


def get_cached_conditions(lat: float, lon: float):
    key = cache_key_for_location(lat, lon)

    conn = get_postgres_conn()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM conditions_cache WHERE cache_key = %s", (key,))
                row = cur.fetchone()
            conn.close()
            if row:
                return dict(row["weather_json"]), dict(row["tide_json"]), row["cached_at"]
        except Exception:
            pass

    local = get_local_conn()
    row = local.execute("SELECT * FROM conditions_cache WHERE cache_key = ?", (key,)).fetchone()
    if row:
        return json.loads(row["weather_json"]), json.loads(row["tide_json"]), row["cached_at"]
    return None


def save_catch_entry(entry: dict):
    """Save to PostgreSQL when online; always keep a local offline copy too."""
    fields = [
        "user_id", "timestamp", "species", "fish_count", "avg_size", "bait", "technique",
        "notes", "water_body", "latitude", "longitude", "resolved_location"
    ]
    record = {field: entry.get(field, "") for field in fields}
    record["user_id"] = DEMO_USER_ID

    saved_to_postgres = False
    conn = get_postgres_conn()
    if conn is not None:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO catches
                        (user_id, timestamp, species, fish_count, avg_size, bait, technique, notes,
                         water_body, latitude, longitude, resolved_location)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        tuple(record[f] for f in fields),
                    )
            saved_to_postgres = True
        except Exception as e:
            st.warning(f"PostgreSQL save failed, so this catch was stored offline instead: {e}")
        finally:
            conn.close()

    local = get_local_conn()
    local.execute(
        """
        INSERT INTO catches
        (user_id, timestamp, species, fish_count, avg_size, bait, technique, notes,
         water_body, latitude, longitude, resolved_location, pending_sync)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(record[f] for f in fields) + (0 if saved_to_postgres else 1,),
    )
    local.commit()


def load_catches() -> pd.DataFrame:
    conn = get_postgres_conn()
    if conn is not None:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM catches WHERE user_id = %s ORDER BY timestamp DESC",
                conn,
                params=(DEMO_USER_ID,),
            )
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass

    local = get_local_conn()
    return pd.read_sql_query(
        "SELECT * FROM catches WHERE user_id = ? ORDER BY timestamp DESC",
        local,
        params=(DEMO_USER_ID,),
    )


def delete_catch(catch_id):
    local = get_local_conn()
    local.execute("DELETE FROM catches WHERE id = ? AND user_id = ?", (int(catch_id), DEMO_USER_ID))
    local.commit()

    conn = get_postgres_conn()
    if conn is not None:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM catches WHERE id = %s AND user_id = %s", (int(catch_id), DEMO_USER_ID))
        conn.close()


def clear_all_catches():
    local = get_local_conn()
    local.execute("DELETE FROM catches WHERE user_id = ?", (DEMO_USER_ID,))
    local.commit()

    conn = get_postgres_conn()
    if conn is not None:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM catches WHERE user_id = %s", (DEMO_USER_ID,))
        conn.close()


def sync_offline_catches_to_postgres():
    if not postgres_available():
        return 0

    local = get_local_conn()
    rows = local.execute(
        "SELECT * FROM catches WHERE user_id = ? AND pending_sync = 1",
        (DEMO_USER_ID,),
    ).fetchall()

    if not rows:
        return 0

    conn = get_postgres_conn()
    if conn is None:
        return 0

    synced_ids = []
    with conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO catches
                    (user_id, timestamp, species, fish_count, avg_size, bait, technique, notes,
                     water_body, latitude, longitude, resolved_location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["user_id"], row["timestamp"], row["species"], row["fish_count"], row["avg_size"],
                        row["bait"], row["technique"], row["notes"], row["water_body"], row["latitude"],
                        row["longitude"], row["resolved_location"],
                    ),
                )
                synced_ids.append(row["id"])
    conn.close()

    local.executemany("UPDATE catches SET pending_sync = 0 WHERE id = ?", [(i,) for i in synced_ids])
    local.commit()
    return len(synced_ids)


# -------------------------------------------------
# LIVE WEATHER
# -------------------------------------------------
def get_live_weather(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,weather_code"
        f"&temperature_unit=fahrenheit"
        f"&wind_speed_unit=mph"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()["current"]

    temperature = data["temperature_2m"]
    wind_speed = data["wind_speed_10m"]
    weather_code = data["weather_code"]

    weather_map = {
        0: "sunny",
        1: "mostly_clear",
        2: "partly_cloudy",
        3: "cloudy",
        61: "rainy",
        63: "rainy",
        65: "rainy",
    }

    weather = weather_map.get(weather_code, "unknown")

    return {
        "temperature": temperature,
        "wind_speed": wind_speed,
        "weather": weather,
    }


# -------------------------------------------------
# NOAA TIDE HELPERS
# -------------------------------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return r * c


def get_nearest_noaa_station(lat: float, lon: float) -> dict:
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    stations = data["stations"]

    closest_station = None
    closest_distance = float("inf")

    for station in stations:
        station_lat = station["lat"]
        station_lon = station["lng"]
        distance = haversine_miles(lat, lon, station_lat, station_lon)

        if distance < closest_distance:
            closest_distance = distance
            closest_station = {
                "id": station["id"],
                "name": station["name"],
                "lat": station_lat,
                "lon": station_lon,
                "distance_miles": distance,
            }

    return closest_station


def get_latest_tide_level(station_id: str) -> dict:
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=water_level"
        f"&application=fish_app"
        f"&date=latest"
        f"&station={station_id}"
        f"&datum=MLLW"
        f"&units=english"
        f"&time_zone=lst_ldt"
        f"&format=json"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    latest = data["data"][0]

    return {
        "t": latest["t"],
        "v": float(latest["v"]),
    }


def get_today_high_low_predictions(station_id: str) -> list[dict]:
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=predictions"
        f"&application=fish_app"
        f"&date=today"
        f"&station={station_id}"
        f"&datum=MLLW"
        f"&interval=hilo"
        f"&units=english"
        f"&time_zone=lst_ldt"
        f"&format=json"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    preds = data["predictions"]

    parsed = []
    for item in preds:
        parsed.append(
            {
                "time": datetime.strptime(item["t"], "%Y-%m-%d %H:%M"),
                "type": item["type"],
                "value": float(item["v"]),
            }
        )

    return parsed


def infer_tide_stage(predictions: list[dict], current_time: datetime | None = None) -> str:
    if current_time is None:
        current_time = datetime.now()

    previous_event = None
    next_event = None

    for event in predictions:
        if event["time"] <= current_time:
            previous_event = event
        elif event["time"] > current_time and next_event is None:
            next_event = event

    if previous_event is None or next_event is None:
        return "unknown"

    if previous_event["type"] == "L" and next_event["type"] == "H":
        return "rising"
    if previous_event["type"] == "H" and next_event["type"] == "L":
        return "falling"
    return "unknown"


def get_realtime_tide(lat: float, lon: float) -> dict:
    station = get_nearest_noaa_station(lat, lon)
    latest_level = get_latest_tide_level(station["id"])
    predictions = get_today_high_low_predictions(station["id"])
    tide_stage = infer_tide_stage(predictions)

    return {
        "noaa_station_id": station["id"],
        "noaa_station_name": station["name"],
        "station_distance_miles": station["distance_miles"],
        "tide_level": latest_level["v"],
        "tide_time": latest_level["t"],
        "tide_stage": tide_stage,
    }


# -------------------------------------------------
# GEOCODING
# -------------------------------------------------
def geocode_location_name(location_name: str):
    geolocator = Nominatim(user_agent="nc_fishing_score_app")
    location = geolocator.geocode(location_name)

    if location is None:
        return None

    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "address": location.address,
    }


# -------------------------------------------------
# FISHING SCORE
# -------------------------------------------------
def get_fishing_score(
    model,
    model_columns,
    lat: float,
    lon: float,
    species: str,
    time_of_day: str,
    season: str,
    user_id: int = 999,
) -> dict:
    cached = get_cached_conditions(lat, lon)

    if network_available():
        try:
            live_weather = get_live_weather(lat, lon)
            live_tide = get_realtime_tide(lat, lon)
            cache_conditions(lat, lon, live_weather, live_tide)
        except Exception:
            if cached is not None:
                live_weather, live_tide, cached_at = cached
                st.warning(f"Live conditions were unavailable, so the app used cached conditions from {cached_at}.")
            else:
                raise
    else:
        if cached is not None:
            live_weather, live_tide, cached_at = cached
            st.info(f"Offline mode: using cached weather/tide data from {cached_at}.")
        else:
            st.warning("Offline mode: no cached weather/tide data was found for this location, so default conditions were used.")
            live_weather = {"temperature": 72, "wind_speed": 8, "weather": "unknown"}
            live_tide = {
                "noaa_station_id": "offline",
                "noaa_station_name": "Offline estimate",
                "station_distance_miles": None,
                "tide_level": 0.0,
                "tide_time": "offline",
                "tide_stage": "unknown",
            }

    sample = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "month": datetime.now().month,
                "latitude": lat,
                "longitude": lon,
                "species": species,
                "temperature": live_weather["temperature"],
                "weather": live_weather["weather"],
                "wind_speed": live_weather["wind_speed"],
                "time_of_day": time_of_day,
                "season": season,
                "tide_level": live_tide["tide_level"],
                "tide_stage": live_tide["tide_stage"],
                "noaa_station_id": live_tide["noaa_station_id"],
            }
        ]
    )

    sample_encoded = pd.get_dummies(sample)
    sample_encoded = sample_encoded.reindex(columns=model_columns, fill_value=0)

    prob_success = model.predict_proba(sample_encoded)[0][1]
    fishing_score = round(prob_success * 100)
    chance_of_success = round(prob_success * 100)

    if fishing_score < 40:
        rating = "Poor"
        emoji = "🔴"
    elif fishing_score < 70:
        rating = "Fair"
        emoji = "🟡"
    elif fishing_score < 85:
        rating = "Good"
        emoji = "🟢"
    else:
        rating = "Excellent"
        emoji = "⭐"

    explanation = []
    if live_weather["wind_speed"] < 10:
        explanation.append("low wind")
    if live_weather["weather"] in ["cloudy", "partly_cloudy", "mostly_clear"]:
        explanation.append("favorable sky conditions")
    if live_tide["tide_stage"] == "rising":
        explanation.append("rising tide")
    if time_of_day == "morning":
        explanation.append("morning conditions")

    if not explanation:
        explanation_text = "Current conditions are mixed, so fishing success may be less predictable."
    else:
        explanation_text = "Today's score is stronger because of " + ", ".join(explanation) + "."

    return {
        "fishing_score": fishing_score,
        "chance_of_success": chance_of_success,
        "rating": rating,
        "emoji": emoji,
        "species": species,
        "time_of_day": time_of_day,
        "season": season,
        "live_weather": live_weather,
        "live_tide": live_tide,
        "explanation": explanation_text,
    }


# -------------------------------------------------
# LOAD MODEL + SESSION STATE
# -------------------------------------------------
try:
    xgb_model, model_columns = load_model_assets()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure xgb_model.pkl and model_columns.pkl are in the same folder as app.py."
    )
    st.stop()

init_storage()

synced_count = sync_offline_catches_to_postgres()
if synced_count:
    st.sidebar.success(f"Synced {synced_count} offline catch(es) to PostgreSQL.")

st.session_state.catch_log = load_catches().to_dict("records")

if APP_MODE == "Offline":
    st.sidebar.info("Offline mode: catch logs are saved locally and will sync later.")
elif postgres_available():
    st.sidebar.success("PostgreSQL connected.")
else:
    st.sidebar.warning("Using local offline storage. Add DATABASE_URL to enable PostgreSQL.")

if "selected_challenges" not in st.session_state:
    st.session_state.selected_challenges = [
        "Catch 3 Different Species",
        "Catch 10 Fish Total",
        "Catch a Fish Over 20 Inches"
    ]

if "selected_social_features" not in st.session_state:
    st.session_state.selected_social_features = [
        "Weekly Leaderboard",
        "Fishing Streak",
        "Recent Activity Feed",
        "Friends / Rivals",
        "Species Leaderboards",
        "Hotspot Leaderboard",
        "Fishing Score Sharing",
        "Titles / Skill Levels",
        "Brag Card",
    ]

def build_community_leaderboard(user_catch_df):
    # Your real stats
    your_total_fish = int(user_catch_df["fish_count"].sum()) if not user_catch_df.empty else 0
    your_species_logged = int(user_catch_df["species"].nunique()) if not user_catch_df.empty else 0
    your_largest_fish = float(user_catch_df["avg_size"].max()) if not user_catch_df.empty else 0.0

    # Mock community users
    community_data = [
        {"Angler": "BayCaster22", "Most Fish Caught": 18, "Most Species Logged": 4, "Largest Fish": 24.5},
        {"Angler": "DrumHunterNC", "Most Fish Caught": 12, "Most Species Logged": 3, "Largest Fish": 27.0},
        {"Angler": "SpeckPro", "Most Fish Caught": 9, "Most Species Logged": 2, "Largest Fish": 21.5},
        {"Angler": "FlounderFan", "Most Fish Caught": 15, "Most Species Logged": 3, "Largest Fish": 19.0},
        {"Angler": "You", "Most Fish Caught": your_total_fish, "Most Species Logged": your_species_logged, "Largest Fish": round(your_largest_fish, 1)},
    ]

    return pd.DataFrame(community_data)

def highlight_user_row(row):
    if row["Angler"] == "You":
        return ["background-color: #d0ebff; font-weight: bold; color: #4682b4"] * len(row)
    return [""] * len(row)

def build_community_leaderboard(user_catch_df):
    your_total_fish = int(user_catch_df["fish_count"].sum()) if not user_catch_df.empty else 0
    your_species_logged = int(user_catch_df["species"].nunique()) if not user_catch_df.empty else 0
    your_largest_fish = float(user_catch_df["avg_size"].max()) if not user_catch_df.empty else 0.0
    your_locations_logged = int(
        user_catch_df["water_body"].replace("", pd.NA).dropna().nunique()
    ) if "water_body" in user_catch_df.columns else 0

    community_data = [
        {
            "Angler": "BayCaster22",
            "Most Fish Caught": 18,
            "Most Species Logged": 4,
            "Largest Fish": 24.5,
            "Locations Logged": 3,
        },
        {
            "Angler": "DrumHunterNC",
            "Most Fish Caught": 12,
            "Most Species Logged": 3,
            "Largest Fish": 27.0,
            "Locations Logged": 2,
        },
        {
            "Angler": "SpeckPro",
            "Most Fish Caught": 9,
            "Most Species Logged": 2,
            "Largest Fish": 21.5,
            "Locations Logged": 2,
        },
        {
            "Angler": "FlounderFan",
            "Most Fish Caught": 15,
            "Most Species Logged": 3,
            "Largest Fish": 19.0,
            "Locations Logged": 4,
        },
        {
            "Angler": "You",
            "Most Fish Caught": your_total_fish,
            "Most Species Logged": your_species_logged,
            "Largest Fish": round(your_largest_fish, 1),
            "Locations Logged": your_locations_logged,
        },
    ]

    return pd.DataFrame(community_data)


def highlight_user_row(row):
    if row["Angler"] == "You":
        return ["background-color: #d0ebff; font-weight: bold; color: #4682b4"] * len(row)
    return [""] * len(row)


def add_location_label(df):
    out = df.copy()
    out["location_label"] = out.apply(
        lambda row: row["water_body"]
        if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != ""
        else f"{row['latitude']}, {row['longitude']}",
        axis=1,
    )
    return out


def get_weekly_catch_df(catch_df):
    weekly_df = catch_df.copy()
    weekly_df["timestamp"] = pd.to_datetime(weekly_df["timestamp"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    return weekly_df[weekly_df["timestamp"] >= cutoff]


def get_user_title(total_fish_caught, unique_species_caught, largest_fish):
    if total_fish_caught >= 25 or largest_fish >= 25:
        return "🏆 Trophy Hunter"
    elif total_fish_caught >= 15 or unique_species_caught >= 4:
        return "🎣 Coastal Explorer"
    elif total_fish_caught >= 8:
        return "🐟 Active Angler"
    return "🌊 Beginner Angler"


def build_activity_feed(catch_df):
    feed_df = catch_df.copy()
    feed_df["timestamp"] = pd.to_datetime(feed_df["timestamp"], errors="coerce")
    feed_df = feed_df.sort_values("timestamp", ascending=False).head(5)

    activities = []
    for _, row in feed_df.iterrows():
        location_text = (
            row["water_body"]
            if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != ""
            else f"{row['latitude']}, {row['longitude']}"
        )
        timestamp_text = (
            row["timestamp"].strftime("%m/%d/%Y %I:%M %p")
            if pd.notnull(row["timestamp"])
            else "Unknown time"
        )

        activities.append(
            f"🎣 {timestamp_text} — Caught {int(row['fish_count'])} {row['species']} at {location_text} "
            f"using {row['technique']}"
        )
    return activities

SPECIES_COLOR_MAP = {
    "Red Drum": "#dc143c",          # crimson
    "Flounder": "#1e90ff",          # blue
    "Weakfish": "#8a2be2",          # purple
    "Spotted Seatrout": "#228b22",  # green
    "Striped Bass": "#ff8c00",      # orange
}

# -------------------------------------------------
# PAGE 1: FISHING SCORE
# -------------------------------------------------
if page == "Fishing Score":
    st.title("🎣 NC Fishing Score")
    st.write("Get a live fishing score based on current weather and tide conditions.")
    st.caption("You can use your current location, enter a place name, or enter latitude and longitude manually.")

    st.markdown("### 📍 Location")
    location_mode = st.radio(
        "Location input method",
        ["Use my current location", "Enter a place name", "Enter latitude and longitude manually"],
        key="score_location_mode"
    )

    detected_lat = None
    detected_lon = None
    detected_address = None

    if location_mode == "Use my current location":
        location = streamlit_geolocation()

        if location and location.get("latitude") is not None and location.get("longitude") is not None:
            detected_lat = float(location["latitude"])
            detected_lon = float(location["longitude"])
            st.success("Location detected successfully.")
            st.write(f"Detected Latitude: {round(detected_lat, 4)}")
            st.write(f"Detected Longitude: {round(detected_lon, 4)}")
        else:
            st.info("Location not available yet. You can switch to place name or manual entry if needed.")

    with st.form("fishing_score_form"):
        species = st.selectbox(
            "Species",
            [
                "red_drum",
                "flounder",
                "weakfish",
                "spotted_seatrout",
                "striped_bass",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
            key="score_species"
        )

        if location_mode == "Enter a place name":
            place_name = st.text_input(
                "Enter a town, beach, inlet, or body of water",
                value="Wilmington, NC",
                key="score_place_name"
            )
        else:
            place_name = None

        col1, col2 = st.columns(2)
        with col1:
            if location_mode == "Enter latitude and longitude manually":
                lat = st.number_input("Latitude", value=34.2257, format="%.4f", key="score_lat")
            elif location_mode == "Use my current location" and detected_lat is not None:
                lat = detected_lat
                st.text_input("Latitude", value=str(round(lat, 4)), disabled=True, key="score_lat_display")
            else:
                lat = None

            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"], key="score_time_of_day")

        with col2:
            if location_mode == "Enter latitude and longitude manually":
                lon = st.number_input("Longitude", value=-77.9447, format="%.4f", key="score_lon")
            elif location_mode == "Use my current location" and detected_lon is not None:
                lon = detected_lon
                st.text_input("Longitude", value=str(round(lon, 4)), disabled=True, key="score_lon_display")
            else:
                lon = None

            season = st.selectbox("Season", ["spring", "summer", "fall", "winter"], key="score_season")

        submitted = st.form_submit_button("Get Fishing Score")

    if submitted:
        try:
            if location_mode == "Enter a place name":
                geocode_result = geocode_location_name(place_name)

                if geocode_result is None:
                    st.error("Could not find that location. Try a more specific place name.")
                    st.stop()

                lat = geocode_result["latitude"]
                lon = geocode_result["longitude"]
                detected_address = geocode_result["address"]

            if location_mode == "Use my current location" and (lat is None or lon is None):
                st.error("Current location is unavailable. Please allow location access or use another location option.")
                st.stop()

            result = get_fishing_score(
                model=xgb_model,
                model_columns=model_columns,
                lat=lat,
                lon=lon,
                species=species,
                time_of_day=time_of_day,
                season=season,
            )

            st.subheader("Fishing Conditions Report")
            st.markdown(f"## {result['emoji']} {result['fishing_score']}/100")
            st.write(f"**Rating:** {result['rating']}")
            st.write(f"**Chance of Success:** {result['chance_of_success']}%")
            st.write(f"**Species:** {result['species'].replace('_', ' ').title()}")

            if location_mode == "Enter a place name" and detected_address is not None:
                st.write(f"**Resolved Location:** {detected_address}")

            st.write(f"**Latitude:** {round(lat, 4)}")
            st.write(f"**Longitude:** {round(lon, 4)}")

            st.markdown("### 🌤 Current Conditions")
            st.write(f"**Weather:** {result['live_weather']['weather'].replace('_', ' ').title()}")
            st.write(f"**Temperature:** {result['live_weather']['temperature']}°F")
            st.write(f"**Wind Speed:** {result['live_weather']['wind_speed']} mph")

            st.markdown("### 🌊 Tide Conditions")
            st.write(f"**Tide Stage:** {result['live_tide']['tide_stage'].title()}")
            st.write(f"**Tide Level:** {result['live_tide']['tide_level']} ft")
            st.write(f"**NOAA Station:** {result['live_tide']['noaa_station_id']}")

            st.markdown("### 💡 Why This Score?")
            st.info(result["explanation"])

        except Exception as e:
            st.error(f"Something went wrong while generating the fishing score: {e}")


# -------------------------------------------------
# PAGE 2: LOG CATCH
# -------------------------------------------------
if page == "Log Catch":
    st.title("📝 Log a Catch")
    st.write("Save details about your trip and catch.")

    st.markdown("### 📍 Catch Location")
    log_location_mode = st.radio(
        "Catch location input method",
        ["Use my current location", "Enter a place name", "Enter latitude and longitude manually"],
        key="log_location_mode"
    )

    log_detected_lat = None
    log_detected_lon = None
    log_detected_address = None

    if log_location_mode == "Use my current location":
        log_location = streamlit_geolocation()

        if log_location and log_location.get("latitude") is not None and log_location.get("longitude") is not None:
            log_detected_lat = float(log_location["latitude"])
            log_detected_lon = float(log_location["longitude"])
            st.success("Catch location detected successfully.")
            st.write(f"Detected Latitude: {round(log_detected_lat, 4)}")
            st.write(f"Detected Longitude: {round(log_detected_lon, 4)}")
        else:
            st.info("Catch location not available yet. You can switch to place name or manual entry if needed.")

    with st.form("log_catch_form"):
        species = st.selectbox(
            "Species",
            [
                "red_drum",
                "flounder",
                "weakfish",
                "spotted_seatrout",
                "striped_bass",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
            key="log_species"
        )

        fish_count = st.number_input("Number Caught", min_value=0, step=1, key="log_fish_count")
        avg_size = st.number_input("Average Size (inches)", min_value=0.0, step=0.5, key="log_avg_size")
        bait = st.text_input("Bait Used", key="log_bait")
        technique = st.selectbox(
            "Technique",
            ["Casting", "Bottom Fishing", "Trolling", "Jigging", "Fly Fishing", "Other"],
            key="log_technique"
        )
        notes = st.text_area("Notes", key="log_notes")

        if log_location_mode == "Enter a place name":
            water_body = st.text_input(
                "Water Body / Town / Location Name",
                value="Wilmington, NC",
                key="log_place_name"
            )
        else:
            water_body = st.text_input("Water Body / Location Name", key="log_water_body")

        col1, col2 = st.columns(2)
        with col1:
            if log_location_mode == "Enter latitude and longitude manually":
                lat = st.number_input("Latitude", value=34.2257, format="%.4f", key="log_lat")
            elif log_location_mode == "Use my current location" and log_detected_lat is not None:
                lat = log_detected_lat
                st.text_input("Latitude", value=str(round(lat, 4)), disabled=True, key="log_lat_display")
            else:
                lat = None

        with col2:
            if log_location_mode == "Enter latitude and longitude manually":
                lon = st.number_input("Longitude", value=-77.9447, format="%.4f", key="log_lon")
            elif log_location_mode == "Use my current location" and log_detected_lon is not None:
                lon = log_detected_lon
                st.text_input("Longitude", value=str(round(lon, 4)), disabled=True, key="log_lon_display")
            else:
                lon = None

        submitted_log = st.form_submit_button("Save Catch")

    if submitted_log:
        if log_location_mode == "Enter a place name":
            geocode_result = geocode_location_name(water_body)

            if geocode_result is None:
                st.error("Could not find that location. Try a more specific place name.")
                st.stop()

            lat = geocode_result["latitude"]
            lon = geocode_result["longitude"]
            log_detected_address = geocode_result["address"]

        if log_location_mode == "Use my current location" and (lat is None or lon is None):
            st.error("Current location is unavailable. Please allow location access or use another location option.")
            st.stop()

        catch_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "species": species.replace("_", " ").title(),
            "fish_count": fish_count,
            "avg_size": avg_size,
            "bait": bait,
            "technique": technique,
            "notes": notes,
            "water_body": water_body if water_body else "",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
        }

        if log_detected_address is not None:
            catch_entry["resolved_location"] = log_detected_address

        save_catch_entry(catch_entry)
        st.session_state.catch_log = load_catches().to_dict("records")
        st.success("Catch saved successfully!")


# -------------------------------------------------
# PAGE 3: MY CATCH LOG
# -------------------------------------------------
if page == "My Catch Log":
    st.title("📖 My Catch Log")

    if len(st.session_state.catch_log) == 0:
        st.info("No catches logged yet.")
    else:
        catch_df = pd.DataFrame(st.session_state.catch_log)

        catch_df["location_label"] = catch_df.apply(
            lambda row: row["water_body"] if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != ""
            else f"{row['latitude']}, {row['longitude']}",
            axis=1
        )

        with st.expander("📊 Catch Statistics", expanded=True):
            total_logs = len(catch_df)
            total_fish = catch_df["fish_count"].sum()
            avg_size = catch_df["avg_size"].mean()

            if {"species", "fish_count"}.issubset(catch_df.columns) and not catch_df.empty:
                species_totals = catch_df.groupby("species")["fish_count"].sum()
                most_common_species = species_totals.idxmax()
            else:
                most_common_species = "N/A"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Catch Logs", total_logs)
                st.metric("Total Fish Caught", int(total_fish))
            with col2:
                st.metric("Average Fish Size (in)", round(avg_size, 1) if pd.notnull(avg_size) else 0)
                st.metric("Most Caught Species", most_common_species)

        # overall best catch locations
        overall_location_totals = (
            catch_df.groupby("location_label", as_index=False)["fish_count"]
            .sum()
            .sort_values(by="fish_count", ascending=False)
        )
        top_overall_locations = overall_location_totals.head(5)

        st.subheader("📍 Best Catch Locations Overall")
        top_overall_locations = top_overall_locations.rename(
            columns={"fish_count": "total_fish_caught"}
        )
        top_overall_locations_display = top_overall_locations.rename(columns=DISPLAY_COLUMN_NAMES)
        st.dataframe(top_overall_locations_display, use_container_width=True)

        # best catch location by species
        species_location_totals = (
            catch_df.groupby(["species", "location_label"], as_index=False)["fish_count"]
            .sum()
            .sort_values(by=["species", "fish_count"], ascending=[True, False])
        )

        top_species_locations = (
            species_location_totals.groupby("species", as_index=False)
            .first()
        )

        st.subheader("🐟 Best Catch Location by Species")
        top_species_locations = top_species_locations.rename(
            columns={"fish_count": "total_fish_caught"}
        )
        top_species_locations_display = top_species_locations.rename(columns=DISPLAY_COLUMN_NAMES)
        st.dataframe(top_species_locations_display, use_container_width=True)

        st.subheader("🔎 Filter Catch Log")

        col1, col2, col3 = st.columns(3)

        with col1:
            species_options = ["All"] + sorted(catch_df["species"].dropna().unique().tolist())
            selected_species = st.selectbox("Species", species_options)

        with col2:
            technique_options = ["All"] + sorted(catch_df["technique"].dropna().unique().tolist())
            selected_technique = st.selectbox("Technique", technique_options)

        with col3:
            bait_options = sorted([b for b in catch_df["bait"].dropna().unique().tolist() if b != ""])
            selected_bait = st.selectbox("Bait", ["All"] + bait_options if bait_options else ["All"])

        filtered_df = catch_df.copy()

        if selected_species != "All":
            filtered_df = filtered_df[filtered_df["species"] == selected_species]

        if selected_technique != "All":
            filtered_df = filtered_df[filtered_df["technique"] == selected_technique]

        if selected_bait != "All":
            filtered_df = filtered_df[filtered_df["bait"] == selected_bait]

        
        st.subheader("📋 Logged Catches")

        filtered_df_display = filtered_df.copy()

        # convert timestamp to datetime for sorting
        filtered_df_display["timestamp"] = pd.to_datetime(
            filtered_df_display["timestamp"],
            errors="coerce"
        )

        # sort most recent first
        filtered_df_display = filtered_df_display.sort_values(
            by="timestamp",
            ascending=False
        )

        # format for display
        filtered_df_display["timestamp"] = filtered_df_display["timestamp"].dt.strftime("%b %d, %Y %I:%M %p")

        # rename columns for display
        filtered_df_display = filtered_df_display.rename(columns=DISPLAY_COLUMN_NAMES)

        st.dataframe(filtered_df_display, use_container_width=True)

        st.subheader("🗑 Manage Catch Log")

        delete_options_df = filtered_df.copy()

        delete_options_df["timestamp"] = pd.to_datetime(
            delete_options_df["timestamp"],
            errors="coerce"
        )

        delete_options_df = delete_options_df.sort_values(
            by="timestamp",
            ascending=False
        ).reset_index()

        def build_delete_label(row):
            if pd.notnull(row["timestamp"]):
                timestamp_str = row["timestamp"].strftime("%b %d, %Y %I:%M %p")
            else:
                timestamp_str = "Unknown Date"

            if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != "":
                location_str = row["water_body"]
            else:
                location_str = f"{row['latitude']}, {row['longitude']}"

            return f"{timestamp_str} | {row['species']} | {int(row['fish_count'])} fish | {location_str}"

        delete_options_df["delete_label"] = delete_options_df.apply(build_delete_label, axis=1)

        if not delete_options_df.empty:
            selected_delete_label = st.selectbox(
                "Select a catch to delete",
                delete_options_df["delete_label"].tolist(),
                key="delete_catch_select"
            )

            if st.button("Delete Selected Catch", key="delete_selected_catch_btn"):
                selected_row = delete_options_df.loc[
                    delete_options_df["delete_label"] == selected_delete_label
                ].iloc[0]

                if "id" in selected_row and pd.notnull(selected_row["id"]):
                    delete_catch(selected_row["id"])
                    st.session_state.catch_log = load_catches().to_dict("records")
                    st.success("Selected catch deleted.")
                    st.rerun()
                else:
                    st.error("This catch could not be deleted because it has no database ID.")

        st.markdown("### ⚠ Clear Entire Catch Log")

        confirm_clear = st.checkbox(
            "I understand this will permanently remove all logged catches for this session.",
            key="confirm_clear_catch_log"
        )

        if st.button("Clear All Catches", key="clear_all_catches_btn"):
            if confirm_clear:
                clear_all_catches()
                st.session_state.catch_log = []
                st.success("All catches cleared.")
                st.rerun()
            else:
                st.warning("Please confirm before clearing the catch log.")


# -------------------------------------------------
# PAGE 4: CHALLENGES
# -------------------------------------------------
if page == "Challenges":
    st.title("🏆 Challenges")

    if len(st.session_state.catch_log) == 0:
        st.info("Log some catches first to see challenge progress.")
    else:
        catch_df = pd.DataFrame(st.session_state.catch_log)

        total_fish_caught = catch_df["fish_count"].sum()
        unique_species_caught = catch_df["species"].nunique()
        unique_locations = catch_df["water_body"].replace("", pd.NA).dropna().nunique()
        unique_techniques = catch_df["technique"].dropna().nunique()
        largest_fish = catch_df["avg_size"].max() if "avg_size" in catch_df.columns else 0

        if "bait" in catch_df.columns and not catch_df["bait"].dropna().empty:
            bait_totals = catch_df.groupby("bait")["fish_count"].sum()
            top_bait_total = bait_totals.max() if not bait_totals.empty else 0
            top_bait_name = bait_totals.idxmax() if not bait_totals.empty else "N/A"
        else:
            top_bait_total = 0
            top_bait_name = "N/A"

        all_challenges = [
            {
                "title": "Catch 3 Different Species",
                "progress": min(unique_species_caught / 3, 1.0),
                "completed": unique_species_caught >= 3,
                "message_done": f"Completed! You logged {unique_species_caught} species.",
                "message_progress": f"{unique_species_caught} / 3 species logged",
            },
            {
                "title": "Catch 10 Fish Total",
                "progress": min(total_fish_caught / 10, 1.0),
                "completed": total_fish_caught >= 10,
                "message_done": f"Completed! You logged {int(total_fish_caught)} fish.",
                "message_progress": f"{int(total_fish_caught)} / 10 fish caught",
            },
            {
                "title": "Catch a Fish Over 20 Inches",
                "progress": min(largest_fish / 20, 1.0) if pd.notnull(largest_fish) else 0,
                "completed": largest_fish >= 20,
                "message_done": f"Completed! Largest logged fish: {largest_fish:.1f} inches",
                "message_progress": f"Current best: {largest_fish:.1f} / 20 inches",
            },
            {
                "title": "Log Catches from 2 Different Locations",
                "progress": min(unique_locations / 2, 1.0),
                "completed": unique_locations >= 2,
                "message_done": f"Completed! You logged catches from {unique_locations} locations.",
                "message_progress": f"{unique_locations} / 2 locations logged",
            },
            {
                "title": "Use 2 Different Techniques",
                "progress": min(unique_techniques / 2, 1.0),
                "completed": unique_techniques >= 2,
                "message_done": f"Completed! You used {unique_techniques} techniques.",
                "message_progress": f"{unique_techniques} / 2 techniques used",
            },
            {
                "title": "Catch 5 Fish Using the Same Bait",
                "progress": min(top_bait_total / 5, 1.0) if top_bait_total > 0 else 0,
                "completed": top_bait_total >= 5,
                "message_done": f"Completed! Best bait: {top_bait_name} with {int(top_bait_total)} fish.",
                "message_progress": f"Current best bait: {top_bait_name} ({int(top_bait_total)} / 5 fish)",
            },
        ]

        st.subheader("⚙️ Choose Your Challenges")

        challenge_titles = [c["title"] for c in all_challenges]

        selected_titles = st.multiselect(
            "Select the challenges you want to track",
            options=challenge_titles,
            default=st.session_state.selected_challenges,
            key="challenge_selector"
        )

        st.session_state.selected_challenges = selected_titles

        show_completed = st.toggle(
            "Show completed selected challenges",
            value=True,
            key="show_completed_selected_challenges"
        )

        selected_challenges = [
            c for c in all_challenges
            if c["title"] in st.session_state.selected_challenges
        ]

        if not show_completed:
            selected_challenges = [c for c in selected_challenges if not c["completed"]]

        st.subheader("🎯 My Challenges")

        if len(selected_challenges) == 0:
            st.info("No challenges selected. Choose one or more challenges above.")
        else:
            for challenge in selected_challenges:
                st.write(f"**{challenge['title']}**")
                st.progress(challenge["progress"])

                if challenge["completed"]:
                    st.success(challenge["message_done"])
                else:
                    st.write(challenge["message_progress"])

        st.subheader("🏅 Achievements")

        achievements = []

        if unique_species_caught >= 3:
            achievements.append("🐟 Multi-Species Angler")
        if total_fish_caught >= 10:
            achievements.append("🎣 Double-Digit Catcher")
        if largest_fish >= 20:
            achievements.append("📏 Trophy Catch")
        if unique_locations >= 2:
            achievements.append("📍 Explorer")
        if unique_techniques >= 2:
            achievements.append("🧠 Versatile Fisher")
        if top_bait_total >= 5:
            achievements.append("🪱 Bait Specialist")

        if achievements:
            for badge in achievements:
                st.success(badge)
        else:
            st.info("No achievements unlocked yet. Keep logging catches!")

        st.subheader("🥇 Personal Leaderboard")

        leaderboard_data = {
            "Category": [
                "Total Fish Caught",
                "Largest Fish (in)",
                "Species Logged",
                "Locations Logged",
                "Techniques Used"
            ],
            "Your Progress": [
                int(total_fish_caught),
                round(largest_fish, 1),
                unique_species_caught,
                unique_locations,
                unique_techniques
            ]
        }

        leaderboard_df = pd.DataFrame(leaderboard_data)
        st.dataframe(leaderboard_df, use_container_width=True)
        

# -------------------------------------------------
# PAGE 5: Community & Social
# -------------------------------------------------        
if page == "Community":
    st.title("🌎 Community & Social")

    if len(st.session_state.catch_log) == 0:
        st.info("Log some catches first to unlock the community and social features.")
    else:
        catch_df = pd.DataFrame(st.session_state.catch_log)
        catch_df = add_location_label(catch_df)

        catch_df["timestamp"] = pd.to_datetime(catch_df["timestamp"], errors="coerce")

        total_fish_caught = int(catch_df["fish_count"].sum())
        unique_species_caught = int(catch_df["species"].nunique())
        largest_fish = float(catch_df["avg_size"].max()) if not catch_df.empty else 0.0
        unique_locations = int(catch_df["location_label"].nunique())

        if "bait" in catch_df.columns and not catch_df["bait"].dropna().empty:
            bait_totals = catch_df.groupby("bait")["fish_count"].sum()
            top_bait_total = bait_totals.max() if not bait_totals.empty else 0
        else:
            top_bait_total = 0

        st.subheader("⚙️ Choose Social Features")
        social_feature_options = [
            "Weekly Leaderboard",
            "Fishing Streak",
            "Recent Activity Feed",
            "Friends / Rivals",
            "Species Leaderboards",
            "Hotspot Leaderboard",
            "Fishing Score Sharing",
            "Titles / Skill Levels",
            "Brag Card",
        ]

        selected_social_features = st.multiselect(
            "Select the social features you want to show",
            options=social_feature_options,
            default=st.session_state.selected_social_features,
            key="social_features_selector"
        )

        st.session_state.selected_social_features = selected_social_features

        # 1. Weekly / Monthly Competitions
        if "Weekly Leaderboard" in selected_social_features:
            st.subheader("📅 Weekly Leaderboard")
            weekly_df = get_weekly_catch_df(catch_df)

            weekly_total_fish = int(weekly_df["fish_count"].sum()) if not weekly_df.empty else 0
            weekly_species = int(weekly_df["species"].nunique()) if not weekly_df.empty else 0
            weekly_largest = float(weekly_df["avg_size"].max()) if not weekly_df.empty else 0.0

            weekly_leaderboard = pd.DataFrame([
                {"Angler": "BayCaster22", "Most Fish Caught": 7, "Most Species Logged": 2, "Largest Fish": 21.0},
                {"Angler": "DrumHunterNC", "Most Fish Caught": 5, "Most Species Logged": 2, "Largest Fish": 24.0},
                {"Angler": "SpeckPro", "Most Fish Caught": 4, "Most Species Logged": 1, "Largest Fish": 18.0},
                {"Angler": "You", "Most Fish Caught": weekly_total_fish, "Most Species Logged": weekly_species, "Largest Fish": round(weekly_largest, 1)},
            ])

            weekly_metric = st.selectbox(
                "Weekly leaderboard category",
                ["Most Fish Caught", "Most Species Logged", "Largest Fish"],
                key="weekly_leaderboard_metric"
            )

            weekly_sorted = weekly_leaderboard.sort_values(
                by=weekly_metric,
                ascending=False
            ).reset_index(drop=True)

            weekly_sorted.insert(
                0,
                "Badge",
                ["🥇", "🥈", "🥉"] + [""] * max(0, len(weekly_sorted) - 3)
            )
            weekly_sorted.index = weekly_sorted.index + 1
            weekly_sorted.index.name = "Rank"

            styled_weekly = weekly_sorted.style.apply(highlight_user_row, axis=1)
            st.dataframe(styled_weekly, use_container_width=True)

        # 2. Fishing Streaks
        if "Fishing Streak" in selected_social_features:
            st.subheader("🔥 Fishing Streak")
            streak_days = catch_df["timestamp"].dt.date.nunique()
            st.metric("Fishing Days Logged", streak_days)

        # 3. Recent Activity Feed
        if "Recent Activity Feed" in selected_social_features:
            st.subheader("📰 Recent Activity Feed")
            activity_feed = build_activity_feed(catch_df)
            for item in activity_feed:
                st.write(item)

        # 4. Friends / Rivals
        if "Friends / Rivals" in selected_social_features:
            st.subheader("🤝 Friends / Rivals")
            community_df = build_community_leaderboard(catch_df)
            avg_fish = community_df["Most Fish Caught"].mean()
            diff = total_fish_caught - avg_fish
            st.metric("You vs Community Average", total_fish_caught, delta=round(diff, 1))

        # 5. Species Leaderboards
        if "Species Leaderboards" in selected_social_features:
            st.subheader("🐟 Species Leaderboards")
            species_choice = st.selectbox(
                "Choose a species leaderboard",
                sorted(catch_df["species"].dropna().unique().tolist()),
                key="species_leaderboard_choice"
            )

            species_df = catch_df[catch_df["species"] == species_choice]
            species_total = int(species_df["fish_count"].sum())
            species_largest = float(species_df["avg_size"].max()) if not species_df.empty else 0.0

            species_leaderboard = pd.DataFrame([
                {"Angler": "BayCaster22", "Fish Caught": 5, "Largest Fish": 20.0},
                {"Angler": "DrumHunterNC", "Fish Caught": 4, "Largest Fish": 24.0},
                {"Angler": "SpeckPro", "Fish Caught": 3, "Largest Fish": 18.0},
                {"Angler": "You", "Fish Caught": species_total, "Largest Fish": round(species_largest, 1)},
            ])

            species_metric = st.selectbox(
                "Species leaderboard category",
                ["Fish Caught", "Largest Fish"],
                key="species_leaderboard_metric"
            )

            species_sorted = species_leaderboard.sort_values(
                by=species_metric,
                ascending=False
            ).reset_index(drop=True)

            species_sorted.insert(
                0,
                "Badge",
                ["🥇", "🥈", "🥉"] + [""] * max(0, len(species_sorted) - 3)
            )
            species_sorted.index = species_sorted.index + 1
            species_sorted.index.name = "Rank"

            styled_species = species_sorted.style.apply(highlight_user_row, axis=1)
            st.dataframe(styled_species, use_container_width=True)

        # 6. Hotspot Leaderboard
        if "Hotspot Leaderboard" in selected_social_features:
            st.subheader("📍 Hotspot Leaderboard")
            hotspot_df = (
                catch_df.groupby("location_label", as_index=False)["fish_count"]
                .sum()
                .sort_values(by="fish_count", ascending=False)
                .head(5)
                .rename(columns={"location_label": "Location", "fish_count": "Total Fish Caught"})
            )
            st.dataframe(hotspot_df, use_container_width=True)

        # 7. Fishing Conditions Score Sharing
        if "Fishing Score Sharing" in selected_social_features:
            st.subheader("🌊 Best Fishing Conditions")
            st.info("A future version can compare logged fishing scores across trips. For now, this can be mocked or tied to saved score snapshots.")

        # 8. Titles / Skill Levels
        if "Titles / Skill Levels" in selected_social_features:
            st.subheader("🏅 Skill Level")
            title = get_user_title(total_fish_caught, unique_species_caught, largest_fish)
            st.success(f"Your current title: {title}")

        # 9. Brag Card
        if "Brag Card" in selected_social_features:
            st.subheader("📣 Brag Card")
            title = get_user_title(total_fish_caught, unique_species_caught, largest_fish)
            brag_card = f'''
🎣 Your Fishing Stats
{title}
🐟 Total Fish Caught: {total_fish_caught}
📍 Locations Explored: {unique_locations}
📏 Largest Fish: {largest_fish:.1f} inches
🐠 Species Logged: {unique_species_caught}
'''
            st.code(brag_card)



# -------------------------------------------------
# PAGE 6: MAP
# -------------------------------------------------
if page == "Map":
    st.title("🗺️ Fishing Map")

    if len(st.session_state.catch_log) == 0:
        st.info("Log some catches first to see them on the map.")
    else:
        catch_df = pd.DataFrame(st.session_state.catch_log)

        # make sure coordinates are numeric
        catch_df["latitude"] = pd.to_numeric(catch_df["latitude"], errors="coerce")
        catch_df["longitude"] = pd.to_numeric(catch_df["longitude"], errors="coerce")

        map_df = catch_df.dropna(subset=["latitude", "longitude"]).copy()

        if map_df.empty:
            st.warning("No valid coordinates available to display on the map.")
        else:
            # -----------------------------------
            # 1. FILTER FIRST
            # -----------------------------------
            st.subheader("🔎 Filter Map")

            species_options = ["All"] + sorted(map_df["species"].dropna().unique().tolist())
            selected_species = st.selectbox(
                "Filter by Species",
                species_options,
                key="map_species_filter"
            )

            if selected_species != "All":
                map_df = map_df[map_df["species"] == selected_species]

            if map_df.empty:
                st.info("No catches match the selected species.")
            else:
                # -----------------------------------
                # prepare data
                # -----------------------------------
                map_df["location_label"] = map_df.apply(
                    lambda row: row["water_body"]
                    if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != ""
                    else f"{row['latitude']}, {row['longitude']}",
                    axis=1
                )

                # format timestamp for tooltip and display
                map_df["timestamp_display"] = pd.to_datetime(
                    map_df["timestamp"],
                    errors="coerce"
                ).dt.strftime("%m/%d/%Y %I:%M %p")

                # -----------------------------------
                # 2. MAP SECOND
                # -----------------------------------
                st.subheader("📍 Logged Catch Locations")

                fig = px.scatter_map(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    color="species",
                    size="fish_count",
                    size_max=7,
                    color_discrete_map=SPECIES_COLOR_MAP,
                    hover_name="species",
                    hover_data={
                        "fish_count": True,
                        "avg_size": True,
                        "bait": True,
                        "technique": True,
                        "location_label": True,
                        "timestamp_display": True,
                        "latitude": False,
                        "longitude": False,
                    },
                    zoom=6,
                    height=550
                )

                fig.update_traces(
                    cluster=dict(
                        enabled=True,
                        maxzoom=10,
                        step=20
                    )
                )

                fig.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0}
                )

                st.plotly_chart(fig, use_container_width=True)

                # -----------------------------------
                # 3. TOP LOCATIONS AFTER MAP
                # -----------------------------------
                st.subheader("🏆 Top Locations on Map")

                top_map_locations = (
                    map_df.groupby("location_label", as_index=False)["fish_count"]
                    .sum()
                    .sort_values(by="fish_count", ascending=False)
                    .head(5)
                    .rename(columns={
                        "location_label": "Location",
                        "fish_count": "Total Fish Caught"
                    })
                )

                st.dataframe(top_map_locations, use_container_width=True)

                # -----------------------------------
                # 4. MAP DATA LAST
                # -----------------------------------
                st.subheader("📋 Map Data")

                map_display_df = map_df.copy()
                map_display_df["timestamp"] = pd.to_datetime(
                    map_display_df["timestamp"],
                    errors="coerce"
                ).dt.strftime("%m/%d/%Y %I:%M %p")

                map_display_df = map_display_df.drop(columns=["timestamp_display"], errors="ignore")
                map_display_df = map_display_df.rename(columns=DISPLAY_COLUMN_NAMES)

                st.dataframe(map_display_df, use_container_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Prototype app using PostgreSQL for deployed multi-user storage, with local offline caching for catch logs, weather, and tide data.")
