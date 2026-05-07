import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_geolocation import streamlit_geolocation
import plotly.express as px
import base64
import hashlib

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "underwater.jpg"
bg_image = get_base64_image(image_path)

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="Count My Fish", page_icon="🎣", layout="centered")

st.markdown("""
<div style='text-align: center; padding-top: 0.5rem; padding-bottom: 1rem;'>
    <h1 style='color:#174A5C; margin-bottom:0;'>🎣 Count My Fish</h1>
    <p style='font-size:18px; color:#2F5D6B; margin-top:0;'>
        Fish smarter - your NC fishing companion
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.60);
    z-index: 0;
}}

h1, h2, h3 {{
    color: #174A5C;
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
    position: relative;
    z-index: 1;
}}

.stButton > button {{
    background-color: #1F7A8C;
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 600;
}}

.stButton > button:hover {{
    background-color: #145C6B;
    color: white;
}}

section[data-testid="stSidebar"] {{
    background-color: rgba(180, 235, 230, 0.55);
    border-right: 1px solid rgba(31, 122, 140, 0.4);
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {{
    color: #0F3D4C;
}}

section[data-testid="stSidebar"] [role="radiogroup"] label {{
    background-color: rgba(255, 255, 255, 0.55);
    border-radius: 10px;
    padding: 0.35rem 0.5rem;
    margin-bottom: 0.25rem;
}}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background-color: rgba(31, 122, 140, 0.12);
}}

div[data-testid="stMetric"] {{
    background-color: white;
    border: 1px solid #dbe9ee;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}
</style>
""", unsafe_allow_html=True)

pages = [
    "Profile",
    "Fishing Report",
    "Log Catch",
    "My Catch Log",
    "Challenges",
    "Community",
    "Map",
    "Agency Report"
]

default_page = st.session_state.get("current_page", "Profile")

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(default_page) if default_page in pages else 0
)

if st.session_state.get("login_success"):
    st.success(f"Successfully logged in as {st.session_state['username']} 🎣")
    st.session_state["login_success"] = False

st.sidebar.markdown("---")
connection_mode = st.sidebar.radio(
    "Connection Mode",
    ["Auto", "Online", "Offline"],
    help="Auto tries live APIs first, then uses cached offline data if needed."
)


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
    "id": "ID",
    "timestamp": "Date/Time",
    "species": "Species",
    "fish_count": "Fish Caught",
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
# SQLITE DATABASE + OFFLINE CACHE
# -------------------------------------------------
DB_PATH = Path("fishing_app.sqlite")
CACHE_MAX_AGE_MINUTES = 180


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password, password_hash):
    return hash_password(password) == password_hash


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                angler_id INTEGER,
                timestamp TEXT NOT NULL,
                species TEXT NOT NULL,
                fish_count INTEGER NOT NULL,
                bait TEXT,
                technique TEXT,
                notes TEXT,
                water_body TEXT,
                latitude REAL,
                longitude REAL,
                resolved_location TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catch_lengths (
                length_id INTEGER PRIMARY KEY AUTOINCREMENT,
                catch_id INTEGER NOT NULL,
                fish_length REAL NOT NULL,
                FOREIGN KEY (catch_id) REFERENCES catches(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_cache (
                cache_key TEXT PRIMARY KEY,
                cached_at TEXT NOT NULL,
                temperature REAL,
                wind_speed REAL,
                weather TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tide_cache (
                cache_key TEXT PRIMARY KEY,
                cached_at TEXT NOT NULL,
                noaa_station_id TEXT,
                noaa_station_name TEXT,
                station_distance_miles REAL,
                tide_level REAL,
                tide_time TEXT,
                tide_stage TEXT
            )
            """
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS anglers (
                angler_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT,     
                zip_code TEXT NOT NULL,
                angler_type TEXT NOT NULL DEFAULT 'Recreational',
                is_minor TEXT NOT NULL DEFAULT 'N',
                guardian_angler_id INTEGER
            )
        """)
        try:
            conn.execute("ALTER TABLE anglers ADD COLUMN password_hash TEXT")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fishing_licenses (
                license_id INTEGER PRIMARY KEY AUTOINCREMENT,
                angler_id INTEGER NOT NULL,
                license_number TEXT,
                license_type TEXT,
                is_lifetime TEXT NOT NULL DEFAULT 'N',
                expiration_date TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fishing_reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                angler_id INTEGER,
                generated_at TEXT,
                species TEXT,
                latitude REAL,
                longitude REAL,
                weather TEXT,
                temperature REAL,
                wind_speed REAL,
                tide_stage TEXT,
                tide_level REAL,
                bait_type TEXT,
                technique TEXT,
                water_type TEXT,
                report_score INTEGER
            )
        """)

        conn.commit()



def rounded_location_key(lat: float, lon: float) -> str:
    return f"{round(float(lat), 2)}_{round(float(lon), 2)}"


def is_cache_fresh(cached_at: str, max_age_minutes: int = CACHE_MAX_AGE_MINUTES) -> bool:
    try:
        cached_time = datetime.fromisoformat(cached_at)
        return datetime.now() - cached_time <= timedelta(minutes=max_age_minutes)
    except Exception:
        return False


def is_online() -> bool:
    if connection_mode == "Offline":
        return False
    if connection_mode == "Online":
        return True

    try:
        requests.get("https://api.open-meteo.com", timeout=3)
        return True
    except requests.RequestException:
        return False


def save_catch(catch_entry: dict):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO catches (
                angler_id, timestamp, species, fish_count, bait, technique, notes,
                water_body, latitude, longitude, resolved_location
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catch_entry.get("angler_id"),
                catch_entry.get("timestamp"),
                catch_entry.get("species"),
                int(catch_entry.get("fish_count", 0)),
                catch_entry.get("bait", ""),
                catch_entry.get("technique", ""),
                catch_entry.get("notes", ""),
                catch_entry.get("water_body", ""),
                float(catch_entry.get("latitude")) if catch_entry.get("latitude") is not None else None,
                float(catch_entry.get("longitude")) if catch_entry.get("longitude") is not None else None,
                catch_entry.get("resolved_location", ""),
            ),
        )
        catch_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for length in catch_entry.get("fish_lengths", []):
            conn.execute(
                """
                INSERT INTO catch_lengths (catch_id, fish_length)
                VALUES (?, ?)
                """,
                (catch_id, float(length))
            )

        conn.commit()


def get_catches_df() -> pd.DataFrame:
    with get_db_connection() as conn:
        return pd.read_sql_query("SELECT * FROM catches ORDER BY timestamp DESC", conn)

def get_user_catches_df(angler_id) -> pd.DataFrame:
    if angler_id is None:
        return pd.DataFrame()

    with get_db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM catches
            WHERE angler_id = ?
            ORDER BY timestamp DESC
            """,
            conn,
            params=(angler_id,)
        )

def get_user_fish_lengths_df(angler_id) -> pd.DataFrame:
    if angler_id is None:
        return pd.DataFrame()

    with get_db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT 
                cl.length_id,
                cl.catch_id,
                cl.fish_length,
                c.species,
                c.angler_id,
                c.timestamp
            FROM catch_lengths cl
            LEFT JOIN catches c ON cl.catch_id = c.id
            WHERE c.angler_id = ?
            """,
            conn,
            params=(angler_id,)
        )

def get_today_species_total(angler_id, species):
    today = datetime.now().strftime("%Y-%m-%d")

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(fish_count), 0) AS total_today
            FROM catches
            WHERE angler_id = ?
              AND species = ?
              AND DATE(timestamp) = ?
            """,
            (angler_id, species, today)
        ).fetchone()

    return int(row["total_today"]) if row else 0

def get_fish_lengths_df() -> pd.DataFrame:
    with get_db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT 
                cl.length_id,
                cl.catch_id,
                cl.fish_length,
                c.species,
                c.angler_id,
                c.timestamp
            FROM catch_lengths cl
            LEFT JOIN catches c ON cl.catch_id = c.id
            """,
            conn
        )

def get_catches_with_anglers_df() -> pd.DataFrame:
    with get_db_connection() as conn:
        return pd.read_sql_query("""
            SELECT c.*, a.full_name, a.username, a.angler_type
            FROM catches c
            LEFT JOIN anglers a ON c.angler_id = a.angler_id
            ORDER BY c.timestamp DESC
        """, conn)

def get_agency_report_df():
    with get_db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT 
                c.id AS catch_id,
                c.timestamp,
                a.full_name,
                a.username,
                a.zip_code,
                a.angler_type,
                a.is_minor,
                c.species,
                c.fish_count,
                GROUP_CONCAT(cl.fish_length, ', ') AS individual_fish_lengths,
                c.bait,
                c.technique,
                c.water_body,
                c.latitude,
                c.longitude,
                c.resolved_location,
                c.notes
            FROM catches c
            LEFT JOIN anglers a ON c.angler_id = a.angler_id
            LEFT JOIN catch_lengths cl ON c.id = cl.catch_id
            GROUP BY c.id
            ORDER BY c.timestamp DESC
            """,
            conn
        )

def summarize_suggestions(df, species, use_public_area=False):
    if df.empty:
        return {
            "top_bait": None,
            "top_technique": None,
            "top_location": None,
            "summary": "No matching catch history yet."
        }

    df = df[df["species"].str.lower().str.replace(" ", "_") == species]

    if df.empty:
        return {
            "top_bait": None,
            "top_technique": None,
            "top_location": None,
            "summary": "No matching catch history yet for this species."
        }

    top_bait = (
        df.groupby("bait")["fish_count"].sum().sort_values(ascending=False).index[0]
        if "bait" in df.columns and not df["bait"].dropna().empty else None
    )

    top_technique = (
        df.groupby("technique")["fish_count"].sum().sort_values(ascending=False).index[0]
        if "technique" in df.columns and not df["technique"].dropna().empty else None
    )

    if use_public_area:
        df = add_public_area_label(df)
        location_col = "public_area"
    else:
        location_col = "water_body"

    top_location = (
        df.groupby(location_col)["fish_count"].sum().sort_values(ascending=False).index[0]
        if location_col in df.columns and not df[location_col].dropna().empty else None
    )

    parts = []
    if top_bait:
        parts.append(f"Best bait: **{top_bait}**.")
    if top_technique:
        parts.append(f"Best technique: **{top_technique}**.")
    if top_location:
        if use_public_area:
            parts.append(f"Hot general area: **{top_location}**.")
        else:
            parts.append(f"Your top saved spot: **{top_location}**.")

    return {
        "top_bait": top_bait,
        "top_technique": top_technique,
        "top_location": top_location,
        "summary": " ".join(parts) if parts else "No strong pattern found yet."
    }


def get_condition_suggestions(species, selected_angler_id=None):
    catch_df = get_catches_with_anglers_df()

    if catch_df.empty:
        return {
            "personal": "No catch history yet. Log more catches to unlock personalized suggestions.",
            "global": "No global catch history yet."
        }

    if selected_angler_id is not None:
        personal_df = catch_df[catch_df["angler_id"] == selected_angler_id]
    else:
        personal_df = pd.DataFrame()

    global_df = add_public_area_label(catch_df.copy())

    personal = summarize_suggestions(personal_df, species, use_public_area=False)
    global_suggestions = summarize_suggestions(global_df, species, use_public_area=True)

    return {
        "personal": personal["summary"],
        "global": global_suggestions["summary"]
    }

def delete_catch(catch_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM catches WHERE id = ?", (int(catch_id),))
        conn.commit()


def clear_all_catches():
    with get_db_connection() as conn:
        conn.execute("DELETE FROM catch_lengths")
        conn.execute("DELETE FROM catches")
        conn.commit()

def save_angler(full_name, username, password, zip_code, angler_type, is_minor, guardian_angler_id=None):
    hashed_password = hash_password(password)

    with get_db_connection() as conn:
        cur = conn.execute("""
            INSERT INTO anglers 
            (full_name, username, password_hash, zip_code, angler_type, is_minor, guardian_angler_id)
            VALUES (?, ?, ?, ?, ?, ?, ?) 
        """, (full_name, username, hashed_password, zip_code, angler_type, is_minor, guardian_angler_id))

        conn.commit()
        return cur.lastrowid


def get_anglers_df(username):
    conn = get_db_connection()

    df = pd.read_sql_query(
        "SELECT * FROM anglers WHERE username = ?",
        conn,
        params=(username,)
    )

    conn.close()
    return df


def save_license(angler_id, license_number, license_type, is_lifetime, expiration_date):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO fishing_licenses
            (angler_id, license_number, license_type, is_lifetime, expiration_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            angler_id,
            license_number,
            license_type,
            "Y" if is_lifetime else "N",
            expiration_date
        ))
        conn.commit()

def get_cached_weather(lat: float, lon: float) -> dict | None:
    cache_key = rounded_location_key(lat, lon)
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM weather_cache WHERE cache_key = ?", (cache_key,)).fetchone()

    if row is None:
        return None

    return {
        "temperature": row["temperature"],
        "wind_speed": row["wind_speed"],
        "weather": row["weather"],
        "cached_at": row["cached_at"],
    }


def save_cached_weather(lat: float, lon: float, weather_data: dict):
    cache_key = rounded_location_key(lat, lon)
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO weather_cache
            (cache_key, cached_at, temperature, wind_speed, weather)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                datetime.now().isoformat(timespec="seconds"),
                weather_data["temperature"],
                weather_data["wind_speed"],
                weather_data["weather"],
            ),
        )
        conn.commit()


def get_cached_tide(lat: float, lon: float) -> dict | None:
    cache_key = rounded_location_key(lat, lon)
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM tide_cache WHERE cache_key = ?", (cache_key,)).fetchone()

    if row is None:
        return None

    return {
        "noaa_station_id": row["noaa_station_id"],
        "noaa_station_name": row["noaa_station_name"],
        "station_distance_miles": row["station_distance_miles"],
        "tide_level": row["tide_level"],
        "tide_time": row["tide_time"],
        "tide_stage": row["tide_stage"],
        "cached_at": row["cached_at"],
    }


def save_cached_tide(lat: float, lon: float, tide_data: dict):
    cache_key = rounded_location_key(lat, lon)
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tide_cache
            (cache_key, cached_at, noaa_station_id, noaa_station_name,
             station_distance_miles, tide_level, tide_time, tide_stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                datetime.now().isoformat(timespec="seconds"),
                tide_data.get("noaa_station_id"),
                tide_data.get("noaa_station_name"),
                tide_data.get("station_distance_miles"),
                tide_data.get("tide_level"),
                tide_data.get("tide_time"),
                tide_data.get("tide_stage"),
            ),
        )
        conn.commit()


# -------------------------------------------------
# LIVE WEATHER WITH OFFLINE CACHE
# -------------------------------------------------
def fetch_live_weather_from_api(lat: float, lon: float) -> dict:
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

    weather_map = {
        0: "sunny",
        1: "mostly_clear",
        2: "partly_cloudy",
        3: "cloudy",
        61: "rainy",
        63: "rainy",
        65: "rainy",
    }

    return {
        "temperature": data["temperature_2m"],
        "wind_speed": data["wind_speed_10m"],
        "weather": weather_map.get(data["weather_code"], "unknown"),
        "source": "live",
    }


def get_live_weather(lat: float, lon: float) -> dict:
    cached = get_cached_weather(lat, lon)

    if is_online():
        try:
            live_weather = fetch_live_weather_from_api(lat, lon)
            save_cached_weather(lat, lon, live_weather)
            return live_weather
        except requests.RequestException:
            pass

    if cached is not None:
        cached["source"] = "cached"
        return cached

    # Last-resort defaults keep the report page usable even with no internet and no cache yet.
    return {
        "temperature": 72,
        "wind_speed": 8,
        "weather": "unknown",
        "source": "offline_default",
    }

# -------------------------------------------------
# WEATHER FORECASTING
# -------------------------------------------------
def fetch_hourly_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,wind_speed_10m,weather_code"
        f"&temperature_unit=fahrenheit"
        f"&wind_speed_unit=mph"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()["hourly"]

    weather_map = {
        0: "sunny",
        1: "mostly_clear",
        2: "partly_cloudy",
        3: "cloudy",
        61: "rainy",
        63: "rainy",
        65: "rainy",
    }

    hourly_df = pd.DataFrame({
        "datetime": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"],
        "wind_speed": data["wind_speed_10m"],
        "weather": [
            weather_map.get(code, "unknown")
            for code in data["weather_code"]
        ]
    })

    hourly_df["hour"] = hourly_df["datetime"].dt.hour

    return hourly_df

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
    cached = get_cached_tide(lat, lon)

    if is_online():
        try:
            station = get_nearest_noaa_station(lat, lon)
            latest_level = get_latest_tide_level(station["id"])
            predictions = get_today_high_low_predictions(station["id"])
            tide_stage = infer_tide_stage(predictions)

            live_tide = {
                "noaa_station_id": station["id"],
                "noaa_station_name": station["name"],
                "station_distance_miles": station["distance_miles"],
                "tide_level": latest_level["v"],
                "tide_time": latest_level["t"],
                "tide_stage": tide_stage,
                "source": "live",
            }
            save_cached_tide(lat, lon, live_tide)
            return live_tide
        except requests.RequestException:
            pass

    if cached is not None:
        cached["source"] = "cached"
        return cached

    # Last-resort defaults keep the app usable offline before a cache exists.
    return {
        "noaa_station_id": "unknown",
        "noaa_station_name": "unknown",
        "station_distance_miles": None,
        "tide_level": 0.0,
        "tide_time": "unknown",
        "tide_stage": "unknown",
        "source": "offline_default",
    }


# -------------------------------------------------
# GEOCODING
# -------------------------------------------------
def geocode_location_name(location_name: str):
    try:
        if location_name is None or str(location_name).strip() == "":
            return None

        query = str(location_name).strip()

        # Add NC only if the user did not already include it
        if "nc" not in query.lower() and "north carolina" not in query.lower():
            query = f"{query}, NC"

        geolocator = Nominatim(user_agent="count_my_fish_app")

        location = geolocator.geocode(
            query,
            language="en",
            timeout=10,
            exactly_one=True
        )

        if location is None:
            return None

        return {
            "latitude": float(location.latitude),
            "longitude": float(location.longitude),
            "address": location.address,
        }

    except Exception as e:
        st.error(f"Geocoder error: {e}")
        return None


MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def format_species_name(species):
    species_map = {
        "red_drum": "Red Drum",
        "flounder": "Flounder",
        "weakfish": "Weakfish/Gray Trout",
        "spotted_seatrout": "Spotted Seatrout/Speckled Trout",
        "striped_bass": "Striped Bass",
    }

    if pd.isna(species):
        return ""

    species_key = str(species).strip().lower().replace(" ", "_")
    return species_map.get(species_key, str(species).replace("_", " ").title())

import difflib

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def clean_display_text(value):
    if not value:
        return ""
    return str(value).strip().title()


def fuzzy_match(value, valid_options, cutoff=0.7):
    """
    Matches misspellings to closest valid option.
    """
    if not value:
        return value

    value = normalize_text(value)
    matches = difflib.get_close_matches(value, valid_options, n=1, cutoff=cutoff)

    return matches[0] if matches else value


def get_temp_bucket(temp):
    if temp < 50:
        return "cold"
    elif temp < 65:
        return "cool"
    elif temp < 75:
        return "ideal"
    elif temp < 85:
        return "warm"
    return "hot"


def get_wind_category(wind_speed):
    if wind_speed < 5:
        return "calm"
    elif wind_speed < 10:
        return "light"
    elif wind_speed < 15:
        return "moderate"
    return "strong"


def get_location_region(lat, lon):
    """
    Simple prototype logic:
    Treat far offshore/east coastal coordinates as possible federal/offshore area.
    This is only an approximation for app warnings.
    """
    if lon > -76.0:
        return "offshore"
    return "inshore"

def detect_striped_bass_area(lat, lon):
    """
    Prototype area detection based on approximate coordinates.
    This is not a legal boundary tool.
    User should verify current NC DEQ / NC Wildlife rules.
    """

    if lat is None or lon is None:
        return "not_sure"

    lat = float(lat)
    lon = float(lon)

    # Offshore / Atlantic Ocean approximation
    if lon > -76.0:
        return "atlantic_ocean"

    # Roanoke River / northeastern NC approximation
    if 35.7 <= lat <= 36.6 and -78.0 <= lon <= -76.4:
        return "roanoke_management_area"

    # Albemarle Sound / northern coastal sound approximation
    if 35.7 <= lat <= 36.3 and -76.7 <= lon <= -75.5:
        return "albemarle_management_area"

    # Cape Fear River / Wilmington area approximation
    if 33.8 <= lat <= 34.5 and -78.2 <= lon <= -77.5:
        return "cape_fear_river"

    # General central/southern coastal area
    if 33.8 <= lat <= 35.7 and -78.5 <= lon <= -75.5:
        return "central_southern_management_area"

    return "not_sure"

def get_regulation_location_zone(lat, lon):
    return get_location_region(lat, lon)

def get_time_period(hour):
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"


# -------------------------------------------------
# FISHING REGULATIONS
# -------------------------------------------------

REGULATIONS = {
    "red_drum": {
        "display": "Red Drum",
        "nc_min_size": 18,
        "nc_max_size": 27,
        "nc_bag_limit": 1,
        "nc_season": "Open year-round",
        "nc_restriction": "Must be 18–27 inches. Fish over 27 inches must be released.",
        "federal_eez_rule": "Harvest or possession of Atlantic red drum is prohibited in the EEZ. Release immediately if caught offshore.",
    },
    "flounder": {
        "display": "Flounder",
        "nc_min_size": None,
        "nc_max_size": None,
        "nc_bag_limit": 0,
        "nc_season": "Currently closed unless NC DEQ announces an open season.",
        "nc_restriction": "Check current NC DEQ flounder proclamations before keeping.",
        "federal_eez_rule": None,
    },
    "weakfish": {
        "display": "Weakfish / Gray Trout",
        "nc_min_size": 12,
        "nc_max_size": None,
        "nc_bag_limit": 1,
        "nc_season": "Open unless changed by proclamation",
        "nc_restriction": "Minimum 12 inches; 1 fish per person per day.",
        "federal_eez_rule": "Weakfish/Gray Trout under 12 inches may not be possessed in the EEZ.",
    },
    "spotted_seatrout": {
        "display": "Spotted Seatrout / Speckled Trout",
        "nc_min_size": None,
        "nc_max_size": None,
        "nc_bag_limit": 0,
        "nc_season": "Currently closed unless NC DEQ announces an open season.",
        "nc_restriction": "Check current NC DEQ proclamations before keeping.",
        "federal_eez_rule": None,
    },
    "striped_bass": {
        "display": "Striped Bass",
        "nc_min_size": None,
        "nc_max_size": None,
        "nc_bag_limit": None,
        "nc_season": "Varies by management area",
        "nc_restriction": (
            "Striped bass rules vary by area. Select the correct management area "
            "and verify current NC DEQ or NC Wildlife rules before keeping fish."
        ),
        "federal_eez_rule": (
            "Fishing for, harvesting, or possessing Atlantic striped bass in the EEZ is prohibited."
        ),
        "area_rules": {
            "atlantic_ocean": {
                "label": "Atlantic Ocean",
                "min_size": 28,
                "max_size": 31,
                "bag_limit": 1,
                "season": "Year-round",
                "restriction": "Harvest slot limit of 28–31 inches total length; 1 striped bass per person per day; circle hook required when fishing with natural bait."
            },
            "roanoke_management_area": {
                "label": "Roanoke River Management Area",
                "min_size": 18,
                "max_size": 22,
                "bag_limit": None,
                "season": "Contact NC Wildlife Resources Commission for current season.",
                "restriction": "18–22 inches total length. Possession and season information should be verified with NC Wildlife Resources Commission."
            },
            "albemarle_management_area": {
                "label": "Albemarle Sound Management Area",
                "min_size": 18,
                "max_size": 25,
                "bag_limit": 0,
                "season": "Season closed; no harvest allowed.",
                "restriction": "Slot limit listed as 18–25 inches total length, but possession is currently illegal/no harvest allowed."
            },
            "central_southern_management_area": {
                "label": "Central Southern Management Area",
                "min_size": None,
                "max_size": None,
                "bag_limit": None,
                "season": "Area-specific; verify current proclamation.",
                "restriction": "Rules vary by inland, joint, coastal, Tar-Pamlico, Neuse, and Cape Fear waters. Check current NC DEQ proclamations."
            },
            "cape_fear_river": {
                "label": "Cape Fear River",
                "min_size": None,
                "max_size": None,
                "bag_limit": None,
                "season": "Area-specific; verify current proclamation.",
                "restriction": "Check NC DEQ striped bass information and current proclamations for Cape Fear River rules."
            },
            "not_sure": {
                "label": "Unknown Striped Bass Area",
                "min_size": None,
                "max_size": None,
                "bag_limit": None,
                "season": "Unknown",
                "restriction": (
                    "Striped bass rules vary significantly by area. If you are unsure which "
                    "management area you are in, do not keep striped bass until you verify the "
                    "current NC DEQ or NC Wildlife rule for your location."
                )
            },
        }
    }
}

def get_regulation_info(species):
    return REGULATIONS.get(species)

def get_striped_bass_area_rule(striped_bass_area):
    reg = get_regulation_info("striped_bass")
    if reg is None:
        return None
    return reg.get("area_rules", {}).get(striped_bass_area)

def get_daily_bag_limit(species, striped_bass_area=None):
    reg = get_regulation_info(species)

    if reg is None:
        return None

    if species == "striped_bass" and striped_bass_area is not None:
        area_rule = get_striped_bass_area_rule(striped_bass_area)
        if area_rule and area_rule["bag_limit"] is not None:
            return area_rule["bag_limit"]

    return reg["nc_bag_limit"]

def check_regulations(species, fish_count=None, fish_lengths=None, location_region=None, striped_bass_area=None):
    reg = get_regulation_info(species)

    if reg is None:
        return ["No regulation information available."]

    warnings = []

    if species == "striped_bass" and striped_bass_area is not None:
        area_rule = get_striped_bass_area_rule(striped_bass_area)

        if area_rule:
            if area_rule["bag_limit"] == 0:
                warnings.append(f"⚠️ {area_rule['label']}: {area_rule['restriction']}")

            if fish_count is not None and area_rule["bag_limit"] is not None:
                if fish_count > area_rule["bag_limit"]:
                    warnings.append(
                        f"⚠️ {area_rule['label']} bag limit: {area_rule['bag_limit']} per person/day."
                    )

            if fish_lengths:
                for idx, length in enumerate(fish_lengths, start=1):
                    if area_rule["min_size"] is not None and length < area_rule["min_size"]:
                        warnings.append(
                            f"⚠️ Fish #{idx}: {area_rule['label']} minimum size is {area_rule['min_size']} inches."
                        )

                    if area_rule["max_size"] is not None and length > area_rule["max_size"]:
                        warnings.append(
                            f"⚠️ Fish #{idx}: {area_rule['label']} maximum keeper size is {area_rule['max_size']} inches."
                        )

    if location_region == "offshore" and reg.get("federal_eez_rule"):
        warnings.append(f"⚠️ Federal offshore rule: {reg['federal_eez_rule']}")

    if reg["nc_bag_limit"] == 0:
        warnings.append(f"⚠️ {reg['display']} may currently be closed or unlawful to possess.")

    if fish_count is not None and reg["nc_bag_limit"] is not None:
        if fish_count > reg["nc_bag_limit"]:
            warnings.append(
                f"⚠️ Bag limit: {reg['display']} limit is {reg['nc_bag_limit']} per person/day."
            )

    if fish_lengths:
        for idx, length in enumerate(fish_lengths, start=1):
            if reg["nc_min_size"] is not None and length < reg["nc_min_size"]:
                warnings.append(
                    f"⚠️ Fish #{idx}: {reg['display']} must be at least {reg['nc_min_size']} inches."
                )

            if reg["nc_max_size"] is not None and length > reg["nc_max_size"]:
                warnings.append(
                    f"⚠️ Fish #{idx}: {reg['display']} must be no larger than {reg['nc_max_size']} inches to keep."
                )

    if not warnings:
        warnings.append("✅ No obvious regulation issues based on the information entered.")

    return warnings


# -------------------------------------------------
# FISHING REPORT
# -------------------------------------------------

def get_fishing_report(
    model,
    model_columns,
    lat: float,
    lon: float,
    species: str,
    month: int,
    hour: int,
    bait_type: str,
    technique: str,
    water_type: str,
    user_skill_level: str,
    user_id: int = 999,
) -> dict:
    live_weather = get_live_weather(lat, lon)
    live_tide = get_realtime_tide(lat, lon)
    time_of_day = get_time_period(hour)
    temp_bucket = get_temp_bucket(live_weather["temperature"])
    wind_category = get_wind_category(live_weather["wind_speed"])
    tide_strength = abs(float(live_tide["tide_level"]))
    location_region = get_location_region(lat, lon)

    sample = pd.DataFrame(
        [
            {
                "user_id": user_id,

                # 1. More specific date feature
                "month": month,

                # 2. More specific time feature
                "hour": hour,
                "time_of_day": time_of_day,

                # 3. Location features
                "latitude": lat,
                "longitude": lon,
                "location_region": location_region,

                # 4. Species
                "species": species,

                # 5. Weather features
                "temperature": live_weather["temperature"],
                "temp_bucket": temp_bucket,
                "weather": live_weather["weather"],

                # 6. Wind features
                "wind_speed": live_weather["wind_speed"],
                "wind_category": wind_category,

                # 7. Tide features
                "tide_level": live_tide["tide_level"],
                "tide_stage": live_tide["tide_stage"],
                "tide_strength": tide_strength,
                "noaa_station_id": live_tide["noaa_station_id"],

                # 8. Fishing method features
                "bait_type": bait_type,
                "technique": technique,
                "water_type": water_type,

                # 9. User behavior / skill feature
                "user_skill_level": user_skill_level,
            }
        ]
    )

    sample_encoded = pd.get_dummies(sample)
    sample_encoded = sample_encoded.reindex(columns=model_columns, fill_value=0)

    prob_success = model.predict_proba(sample_encoded)[0][1]
    fishing_report = round(prob_success * 100)
    chance_of_success = round(prob_success * 100)

    if fishing_report < 40:
        rating = "Poor"
        emoji = "🔴"
    elif fishing_report < 70:
        rating = "Fair"
        emoji = "🟡"
    elif fishing_report < 85:
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
    if hour in [5, 6, 7, 8, 18, 19, 20]:
        explanation.append("prime fishing hour")
    if temp_bucket == "ideal":
        explanation.append("ideal temperature range")
    if wind_category in ["calm", "light"]:
        explanation.append("manageable wind conditions")

    if not explanation:
        explanation_text = "Current conditions are mixed, so fishing success may be less predictable."
    else:
        explanation_text = "Today's report is stronger because of " + ", ".join(explanation) + "."

    return {
        "fishing_report": fishing_report,
        "chance_of_success": chance_of_success,
        "rating": rating,
        "emoji": emoji,
        "species": species,
        "time_of_day": time_of_day,
        "month": month,
        "hour": hour,
        "bait_type": bait_type,
        "technique": technique,
        "water_type": water_type,
        "user_skill_level": user_skill_level,
        "temp_bucket": temp_bucket,
        "wind_category": wind_category,
        "location_region": location_region,
        "live_weather": live_weather,
        "live_tide": live_tide,
        "explanation": explanation_text,
    }

def get_best_times_today(
    model,
    model_columns,
    lat,
    lon,
    species,
    month,
    bait_type,
    technique,
    water_type,
    user_skill_level,
):
    try:
        hourly_weather = fetch_hourly_weather_forecast(lat, lon)
    except Exception:
        hourly_weather = pd.DataFrame()

    live_tide = get_realtime_tide(lat, lon)
    possible_hours = list(range(5, 21))
    results = []

    for test_hour in possible_hours:
        if not hourly_weather.empty and test_hour in hourly_weather["hour"].values:
            weather_row = hourly_weather[hourly_weather["hour"] == test_hour].iloc[0]
            temperature = weather_row["temperature"]
            wind_speed = weather_row["wind_speed"]
            weather = weather_row["weather"]
        else:
            fallback_weather = get_live_weather(lat, lon)
            temperature = fallback_weather["temperature"]
            wind_speed = fallback_weather["wind_speed"]
            weather = fallback_weather["weather"]

        time_of_day = get_time_period(test_hour)
        temp_bucket = get_temp_bucket(temperature)
        wind_category = get_wind_category(wind_speed)
        tide_strength = abs(float(live_tide["tide_level"]))
        location_region = get_location_region(lat, lon)

        sample = pd.DataFrame([{
            "user_id": 999,
            "month": month,
            "hour": test_hour,
            "time_of_day": time_of_day,
            "latitude": lat,
            "longitude": lon,
            "location_region": location_region,
            "species": species,
            "temperature": temperature,
            "temp_bucket": temp_bucket,
            "weather": weather,
            "wind_speed": wind_speed,
            "wind_category": wind_category,
            "tide_level": live_tide["tide_level"],
            "tide_stage": live_tide["tide_stage"],
            "tide_strength": tide_strength,
            "noaa_station_id": live_tide["noaa_station_id"],
            "bait_type": bait_type,
            "technique": technique,
            "water_type": water_type,
            "user_skill_level": user_skill_level,
        }])

        sample_encoded = pd.get_dummies(sample)
        sample_encoded = sample_encoded.reindex(columns=model_columns, fill_value=0)

        prob_success = model.predict_proba(sample_encoded)[0][1]
        report_score = round(prob_success * 100)

        if report_score < 40:
            rating = "Poor"
        elif report_score < 70:
            rating = "Fair"
        elif report_score < 85:
            rating = "Good"
        else:
            rating = "Excellent"

        results.append({
            "Time": datetime.strptime(str(test_hour), "%H").strftime("%I %p").lstrip("0"),
            "Fishing Report Score": report_score,
            "Rating": rating,
            "Temp": round(temperature, 1),
            "Wind": round(wind_speed, 1),
            "Weather": str(weather).replace("_", " ").title()
        })

    return pd.DataFrame(results).sort_values(
        by="Fishing Report Score",
        ascending=False
    )

def save_fishing_report(report):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO fishing_reports (
                angler_id,
                generated_at,
                species,
                latitude,
                longitude,
                weather,
                temperature,
                wind_speed,
                tide_stage,
                tide_level,
                bait_type,
                technique,
                water_type,
                report_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report["angler_id"],
            report["generated_at"],
            report["species"],
            report["latitude"],
            report["longitude"],
            report["weather"],
            report["temperature"],
            report["wind_speed"],
            report["tide_stage"],
            report["tide_level"],
            report["bait_type"],
            report["technique"],
            report["water_type"],
            report["report_score"],
        ))

        conn.commit()

def get_fishing_reports_df():
    with get_db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM fishing_reports
            ORDER BY generated_at DESC
            """,
            conn
        )

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

init_db()

if connection_mode == "Offline":
    st.sidebar.info("Offline mode: catches save locally, and weather/tide use cache when available.")
elif connection_mode == "Auto":
    st.sidebar.caption("Auto mode: tries live data first, then falls back to SQLite cache.")
else:
    st.sidebar.caption("Online mode: tries live data and saves a local SQLite cache.")

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
        "Fishing Report Sharing",
        "Titles / Skill Levels",
        "Brag Card",
    ]



def build_community_leaderboard(catch_df):
    if catch_df.empty:
        return pd.DataFrame()

    lengths_df = get_fish_lengths_df()

    leaderboard = (
        catch_df.groupby("username")
        .agg(
            **{
                "Most Fish Caught": ("fish_count", "sum"),
                "Most Species Logged": ("species", "nunique"),
                "Locations Logged": ("water_body", "nunique"),
            }
        )
        .reset_index()
        .rename(columns={"username": "Angler"})
    )

    largest_lengths = (
        lengths_df.merge(
            catch_df[["id", "username"]],
            left_on="catch_id",
            right_on="id",
            how="inner"
        )
        .groupby("username")["fish_length"]
        .max()
        .reset_index()
        .rename(columns={
            "username": "Angler",
            "fish_length": "Largest Fish"
        })
    )

    leaderboard = leaderboard.merge(largest_lengths, on="Angler", how="left")
    leaderboard["Largest Fish"] = leaderboard["Largest Fish"].fillna(0).round(1)

    return leaderboard

def get_display_location(row):
    if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != "":
        return row["water_body"]

    if pd.notnull(row.get("resolved_location")) and str(row.get("resolved_location")).strip() != "":
        return row["resolved_location"]

    if pd.notnull(row.get("latitude")) and pd.notnull(row.get("longitude")):
        return f"{round(row['latitude'], 2)}, {round(row['longitude'], 2)}"

    return "Unknown Location"


def clean_location_name(name):
    if not name:
        return ""
    return name.split(",")[0]

def highlight_user_row(row):
    current_user = st.session_state.get("username")

    if row["Angler"] == current_user:
        return ["background-color: #d0ebff; font-weight: bold; color: #4682b4"] * len(row)

    return [""] * len(row)


def add_location_label(df):
    out = df.copy()
    out["location_label"] = out.apply(get_display_location, axis=1)
    return out

def get_public_area_label(lat, lon):
    """
    Creates a generalized public area label.
    Rounding to 0.1 degrees gives an approximate area, not an exact spot.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Area"

    lat_area = round(float(lat), 1)
    lon_area = round(float(lon), 1)

    return f"General area near {lat_area}, {lon_area}"


def add_public_area_label(df):
    out = df.copy()
    out["public_area"] = out.apply(
        lambda row: get_public_area_label(row.get("latitude"), row.get("longitude")),
        axis=1
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
            f"🎣 {timestamp_text} — Caught {int(row['fish_count'])} {format_species_name(row['species'])} at {location_text} "
            f"using {row['technique']}"
        )
    return activities

SPECIES_COLOR_MAP = {
    "Red Drum": "#dc143c",          # crimson
    "Flounder": "#1e90ff",          # blue
    "Weakfish/Gray Trout": "#8a2be2",          # purple
    "Spotted Seatrout/Speckled Trout": "#228b22",  # green
    "Striped Bass": "#ff8c00",      # orange
}


# -------------------------------------------------
# PAGE 0: PROFILE
# -------------------------------------------------
if page == "Profile":
    st.markdown("""
    # 👤 Angler Profile
    ### Set up angler information once for easier catch logging and reporting.
    """)

    # LOGIN WITH PASSWORD FOR EXISTING PROFILES
    st.subheader("Log In With Existing Profile")

    login_username = st.text_input("Username", key="login_username")
    login_password = st.text_input("Password", type="password", key="login_password")

    if st.button("Log In"):
        existing_profile = get_anglers_df(login_username)

        if existing_profile.empty:
            st.error("No profile found with that username.")
        else:
            stored_hash = existing_profile.iloc[0]["password_hash"]

            if stored_hash is None or not check_password(login_password, stored_hash):
                st.error("Incorrect username or password.")
            else:
                st.session_state["username"] = login_username
                st.session_state["angler_id"] = int(existing_profile.iloc[0]["angler_id"])
                st.session_state["login_success"] = True
                st.session_state["current_page"] = "Fishing Report"
                st.rerun()

    with st.form("profile_form"):
        full_name = st.text_input("Full Name")
        username = st.text_input("Username / Display Name")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        zip_code = st.text_input("ZIP Code of Residence", max_chars=5)

        angler_type = st.selectbox(
            "Angler Type",
            ["Recreational", "Commercial"]
        )

        is_child = st.checkbox("This profile is for a child/dependent under 18")

        add_license = st.checkbox("Add fishing license information?")

        license_number = ""
        license_type = "Annual"
        is_lifetime = False
        expiration_date = None

        if add_license:
            license_number = st.text_input("Fishing License Number")
            license_type = st.selectbox("License Type", ["Annual", "Lifetime", "Other"])
            is_lifetime = license_type == "Lifetime"

            if not is_lifetime:
                expiration_date = st.date_input("License Expiration Date")

        submitted_profile = st.form_submit_button("Save Profile")

    if submitted_profile:
        if not full_name or not zip_code or not username or not password:
            st.error("Full name, username, ZIP code, and password are required.")
        elif not zip_code.isdigit() or len(zip_code) != 5:
            st.error("ZIP code must be exactly 5 digits.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                angler_id = save_angler(
                    full_name=full_name,
                    username=username,
                    password=password,
                    zip_code=zip_code,
                    angler_type=angler_type,
                    is_minor="Y" if is_child else "N",
                    guardian_angler_id=None
                )

                if add_license:
                    save_license(
                        angler_id=angler_id,
                        license_number=license_number,
                        license_type=license_type,
                        is_lifetime=is_lifetime,
                        expiration_date=str(expiration_date) if expiration_date else None
                    )

                st.session_state["username"] = username
                st.session_state["angler_id"] = angler_id

                st.session_state["login_success"] = True
                st.session_state["current_page"] = "Fishing Report"

                st.rerun()

            except sqlite3.IntegrityError:
                st.error("That username is already taken. Please choose another username.")

    st.subheader("Saved Profiles")
    
    current_user = st.session_state.get("username")

    if not current_user:
        st.warning("Please log in to view saved profiles.")
        st.stop()

    anglers_df = get_anglers_df(current_user)

    if anglers_df.empty:
        st.info("No profiles saved yet.")
    else:
        display_profiles = anglers_df.copy()

        display_profiles["Minor"] = display_profiles["is_minor"].map({
            "Y": "Yes",
            "N": "No"
        })

        display_profiles = display_profiles.rename(columns={
            "username": "Username",
            "angler_type": "Angler Type"
        })

        st.dataframe(
            display_profiles[["Username", "Angler Type", "Minor"]],
            use_container_width=True,
            hide_index=True
        )



# -------------------------------------------------
# PAGE 1: FISHING REPORT
# -------------------------------------------------
if page == "Fishing Report":
    st.markdown("""
    # 🎣 NC Fishing Report
    ### Smarter fishing conditions, safer catch logging, and better harvest reporting.
    """)
    st.write("Get a real-time fishing report based on current weather and tide conditions.")
    st.caption("You can use your current location, enter a place name, or enter latitude and longitude manually.")

    current_user = st.session_state.get("username")
    anglers_df = get_anglers_df(current_user)

    selected_angler_id = None
    selected_angler_name = "Guest"

    if not anglers_df.empty:
        angler_options = dict(zip(anglers_df["username"], anglers_df["angler_id"]))

        selected_angler_name = st.selectbox(
            "Who is checking this fishing report?",
            list(angler_options.keys()),
            key="report_angler_select"
        )

        selected_angler_id = angler_options[selected_angler_name]
    else:
        st.info("No angler profile selected. The app will show global suggestions only.")

    st.markdown("### 📍 Location")
    location_mode = st.radio(
        "Location input method",
        ["Use my current location", "Enter a place name", "Enter latitude and longitude manually"],
        key="report_location_mode"
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

    with st.form("fishing_report_form"):
        species = st.selectbox(
            "Species",
            [
                "red_drum",
                "flounder",
                "weakfish",
                "spotted_seatrout",
                "striped_bass",
            ],
            format_func=format_species_name,
            key="report_species"
        )

        striped_bass_area = None

        if species == "striped_bass":
            striped_bass_area = st.selectbox(
                "Where are you fishing for striped bass?",
                [
                    "not_sure",
                    "atlantic_ocean",
                    "roanoke_management_area",
                    "albemarle_management_area",
                    "central_southern_management_area",
                    "cape_fear_river",
                ],
                format_func=lambda x: {
                    "not_sure": "I'm not sure — show me the safest warning",
                    "atlantic_ocean": "Atlantic Ocean / offshore",
                    "roanoke_management_area": "Roanoke River area",
                    "albemarle_management_area": "Albemarle Sound area",
                    "central_southern_management_area": "Central/Southern coastal rivers",
                    "cape_fear_river": "Cape Fear River area",
                }[x],
                key="report_striped_bass_area"
            )

        if location_mode == "Enter a place name":
            place_name = st.text_input(
                "Enter a town, beach, inlet, or body of water",
                value="Wilmington, NC",
                key="report_place_name"
            )
        else:
            place_name = None

        col1, col2 = st.columns(2)

        with col1:
            if location_mode == "Enter latitude and longitude manually":
                lat = st.number_input("Latitude", value=34.2257, format="%.4f", key="report_lat")
            elif location_mode == "Use my current location" and detected_lat is not None:
                lat = detected_lat
                st.text_input("Latitude", value=str(round(lat, 4)), disabled=True, key="report_lat_display")
            else:
                lat = None

            selected_date = st.date_input(
                "Date",
                value=datetime.now(),
                key="report_date"
            )

            month = selected_date.month

            day_of_week = selected_date.weekday()

            bait_type = st.selectbox(
                "Bait Type",
                ["live_bait", "cut_bait", "artificial_lure", "shrimp", "mullet", "other"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="report_bait_type"
            )

            water_type = st.selectbox(
                "Water Type",
                ["inshore", "nearshore", "offshore", "pier", "surf", "river", "sound", "unknown"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="report_water_type"
            )

        with col2:
            if location_mode == "Enter latitude and longitude manually":
                lon = st.number_input("Longitude", value=-77.9447, format="%.4f", key="report_lon")
            elif location_mode == "Use my current location" and detected_lon is not None:
                lon = detected_lon
                st.text_input("Longitude", value=str(round(lon, 4)), disabled=True, key="report_lon_display")
            else:
                lon = None

            time_options = []
            for h in range(24):
                display = datetime.strptime(str(h), "%H").strftime("%I %p")
                time_options.append((display, h))

            # Show dropdown to user
            selected_time = st.selectbox(
                "Time of Day",
                time_options,
                format_func=lambda x: x[0],
                index=datetime.now().hour,
                key="report_time"
            )

            # Extract hour for model
            hour = selected_time[1]

            technique = st.selectbox(
                "Technique",
                ["casting", "bottom_fishing", "trolling", "jigging", "fly_fishing", "other"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="report_technique"
            )

            user_skill_level = st.selectbox(
                "Angler Experience Level",
                ["beginner", "intermediate", "advanced"],
                format_func=lambda x: x.title(),
                key="report_user_skill_level"
            )

        submitted = st.form_submit_button("Get Fishing Report")

    if submitted:
        try:
            if location_mode == "Enter a place name":
                geocode_result = geocode_location_name(place_name)

                if geocode_result is None:
                    st.error("Could not find that location in North Carolina. Try adding NC, a nearby town, or a more specific place name.")
                    st.stop()

                lat = geocode_result["latitude"]
                lon = geocode_result["longitude"]
                detected_address = geocode_result["address"]

            if location_mode == "Use my current location" and (lat is None or lon is None):
                st.error("Current location is unavailable. Please allow location access or use another location option.")
                st.stop()

            result = get_fishing_report(
                model=xgb_model,
                model_columns=model_columns,
                lat=lat,
                lon=lon,
                species=species,
                month=month,
                hour=hour,
                bait_type=bait_type,
                technique=technique,
                water_type=water_type,
                user_skill_level=user_skill_level,
            )

            save_fishing_report({
                "angler_id": selected_angler_id,
                "generated_at": datetime.now().isoformat(),
                "species": species,
                "latitude": lat,
                "longitude": lon,
                "weather": result["live_weather"]["weather"],
                "temperature": result["live_weather"]["temperature"],
                "wind_speed": result["live_weather"]["wind_speed"],
                "tide_stage": result["live_tide"]["tide_stage"],
                "tide_level": result["live_tide"]["tide_level"],
                "bait_type": bait_type,
                "technique": technique,
                "water_type": water_type,
                "report_score": result["fishing_report"],
            })


            suggestions = get_condition_suggestions(
                species=species,
                selected_angler_id=selected_angler_id,
            )

            st.subheader("Fishing Conditions Report")
            st.markdown(f"## {result['emoji']} {result['fishing_report']}/100")
            st.write(f"**Rating:** {result['rating']}")
            st.write(f"**Chance of Success:** {result['chance_of_success']}%")
            st.write(f"**Species:** {format_species_name(result['species'])}")
            reg = get_regulation_info(result['species'])
            reg_location_zone = get_regulation_location_zone(lat, lon)
            striped_area_rule = None
            if result['species'] == "striped_bass" and striped_bass_area is not None:
                striped_area_rule = get_striped_bass_area_rule(striped_bass_area)

            if reg:
                st.markdown("### ⚖️ Regulation Reminder")
                st.info(
                    f"**{reg['display']}**\n\n"
                    f"**NC rule:** {reg['nc_season']}\n\n"
                    f"{reg['nc_restriction']}\n\n"
                    f"**Federal/offshore note:** {reg['federal_eez_rule'] if reg['federal_eez_rule'] else 'No federal EEZ rule found in the uploaded document for this species.'}\n\n"
                    "Always verify current NC DEQ rules before keeping fish."
                )

                if reg_location_zone == "offshore" and reg.get("federal_eez_rule"):
                    st.warning(f"⚠️ Offshore warning: {reg['federal_eez_rule']}")
                if striped_area_rule:
                    st.warning(
                        f"**Striped Bass Area Rule — {striped_area_rule['label']}**\n\n"
                        f"**Season:** {striped_area_rule['season']}\n\n"
                        f"{striped_area_rule['restriction']}"
                    )
            st.write(f"**Date:** {selected_date.strftime('%B %d, %Y')}")
            formatted_time = datetime.strptime(str(result["hour"]), "%H").strftime("%I %p").lstrip("0")
            st.write(f"**Time:** {formatted_time}")
            st.write(f"**Bait Type:** {result['bait_type'].replace('_', ' ').title()}")
            st.write(f"**Technique:** {result['technique'].replace('_', ' ').title()}")
            st.write(f"**Water Type:** {result['water_type'].replace('_', ' ').title()}")
            st.write(f"**Experience Level:** {result['user_skill_level'].title()}")
            st.write(f"**Location Region:** {result['location_region'].title()}")

            if location_mode == "Enter a place name" and detected_address is not None:
                st.write(f"**Resolved Location:** {clean_display_text(detected_address)}")

            st.write(f"**Latitude:** {round(lat, 4)}")
            st.write(f"**Longitude:** {round(lon, 4)}")

            st.markdown("### 🌤 Current Conditions")
            st.write(f"**Weather:** {result['live_weather']['weather'].replace('_', ' ').title()}")
            st.write(f"**Temperature:** {result['live_weather']['temperature']}°F")
            st.write(f"**Wind Speed:** {result['live_weather']['wind_speed']} mph")
            st.write(f"**Temperature Category:** {result['temp_bucket'].title()}")
            st.write(f"**Wind Category:** {result['wind_category'].title()}")
            st.caption(f"Weather source: {result['live_weather'].get('source', 'unknown')}")

            st.markdown("### 🌊 Tide Conditions")
            st.write(f"**Tide Stage:** {result['live_tide']['tide_stage'].title()}")
            st.write(f"**Tide Level:** {result['live_tide']['tide_level']} ft")
            st.write(f"**NOAA Station:** {result['live_tide']['noaa_station_id']}")
            st.caption(f"Tide source: {result['live_tide'].get('source', 'unknown')}")

            st.markdown("### 💡 Why This Report?")
            st.info(result["explanation"])

            st.markdown("### 🕒 Best Times Today")

            best_times_df = get_best_times_today(
                model=xgb_model,
                model_columns=model_columns,
                lat=lat,
                lon=lon,
                species=species,
                month=month,
                bait_type=bait_type,
                technique=technique,
                water_type=water_type,
                user_skill_level=user_skill_level,
            )

            best_time = best_times_df.iloc[0]

            st.success(
                f"Best predicted time today: **{best_time['Time']}** "
                f"with a report score of **{best_time['Fishing Report Score']}/100** "
                f"({best_time['Rating']})."
            )

            st.dataframe(
                best_times_df.head(5),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 🎯 Personalized Suggestions")
            st.info(suggestions["personal"])

            st.markdown("### 🌎 Global Suggestions")
            st.info(suggestions["global"])
            
        except Exception as e:
            st.error(f"Something went wrong while generating the fishing report: {e}")


# -------------------------------------------------
# PAGE 2: LOG CATCH
# -------------------------------------------------
if page == "Log Catch":
    st.markdown("""
    # 📝 Log a Catch
    ### Record catch details, fish lengths, locations, and regulation checks.
    """)

    current_user = st.session_state.get("username")
    anglers_df = get_anglers_df(current_user)

    if anglers_df.empty:
        st.warning("Please create an angler profile before logging a catch.")
        st.stop()

    angler_options = dict(zip(anglers_df["username"], anglers_df["angler_id"]))

    selected_angler_name = st.selectbox(
        "Who is this catch for?",
        list(angler_options.keys())
    )

    selected_angler_id = angler_options[selected_angler_name]

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

    striped_bass_area = None

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
            format_func=format_species_name,
            key="log_species"
        )

        fish_count = st.number_input("Number Caught", min_value=0, step=1, key="log_fish_count")

        fish_lengths_text = st.text_input(
            "Fish Lengths (inches)",
            placeholder="Example: 18.5, 21, 24",
            help="Enter one length per fish, separated by commas.",
            key="log_fish_lengths_text"
        )

        fish_lengths = []

        if fish_lengths_text.strip():
            try:
                fish_lengths = [
                    float(x.strip())
                    for x in fish_lengths_text.split(",")
                    if x.strip() != ""
                ]
            except ValueError:
                st.error("Please enter fish lengths as numbers separated by commas.")

        bait = st.selectbox(
            "Bait Used",
            ["live bait", "cut bait", "shrimp", "mullet", "artificial lure", "other"],
            format_func=lambda x: x.title(),
            key="log_bait"
        )
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
                st.error("Could not find that location in North Carolina. Try adding NC, a nearby town, or a more specific place name.")
                st.stop()

            lat = geocode_result["latitude"]
            lon = geocode_result["longitude"]
            log_detected_address = geocode_result["address"]

        if log_location_mode == "Use my current location" and (lat is None or lon is None):
            st.error("Current location is unavailable. Please allow location access or use another location option.")
            st.stop()
        
        auto_striped_bass_area = None

        if species == "striped_bass":
            auto_striped_bass_area = detect_striped_bass_area(lat, lon)

            st.info(
                f"Auto-detected striped bass area: "
                f"{REGULATIONS['striped_bass']['area_rules'][auto_striped_bass_area]['label']}"
            )

            striped_bass_area = auto_striped_bass_area
            st.caption(
                "Area detection is approximate. Always verify current NC DEQ or NC Wildlife rules before keeping striped bass."
            )


        reg_location_zone = get_regulation_location_zone(lat, lon)

        if len(fish_lengths) != int(fish_count):
            st.error(
                f"You entered {len(fish_lengths)} fish lengths, but Number Caught is {int(fish_count)}. "
                "Please enter one length for each fish."
            )
            st.stop()

        daily_bag_limit = get_daily_bag_limit(
            species=species,
            striped_bass_area=striped_bass_area
        )

        already_logged_today = get_today_species_total(
            angler_id=selected_angler_id,
            species=species
        )

        new_total_today = already_logged_today + int(fish_count)

        if daily_bag_limit is not None:
            if daily_bag_limit == 0:
                st.error(
                    f"{format_species_name(species)} is currently closed or has a 0 fish daily limit. "
                    "This catch cannot be saved."
                )
                st.stop()

            if already_logged_today >= daily_bag_limit:
                st.error(
                    f"You have already logged the daily limit for {format_species_name(species)} today "
                    f"({already_logged_today}/{daily_bag_limit}). This catch cannot be saved."
                )
                st.stop()

            if new_total_today > daily_bag_limit:
                remaining = daily_bag_limit - already_logged_today
                st.error(
                    f"This entry would exceed the daily limit for {format_species_name(species)}. "
                    f"You have already logged {already_logged_today}, and the daily limit is {daily_bag_limit}. "
                    f"You can only log {remaining} more today."
                )
                st.stop()
        
        reg_warnings = check_regulations(
            species=species,
            fish_count=fish_count,
            fish_lengths=fish_lengths,
            location_region=reg_location_zone,
            striped_bass_area=striped_bass_area
        )

        for warning in reg_warnings:
            if warning.startswith("⚠️"):
                st.markdown(f"""
                <div style="
                    background-color: rgba(255, 165, 0, 0.12);
                    border: 1px solid rgba(255, 140, 0, 0.4);
                    color: #CC7000;
                    padding: 0.8rem;
                    border-radius: 12px;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                ">
                ⚠️ {warning.replace("⚠️", "").strip()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background-color: rgba(255, 215, 0, 0.15);
                    box-shadow: 0 2px 8px rgba(255, 215, 0, 0.2);        
                    border: 1px solid rgba(184, 134, 11, 0.4);
                    color: #B8860B;
                    padding: 0.8rem;
                    border-radius: 12px;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                ">
                🏆 {warning.replace("✅", "").strip()}
                </div>
                """, unsafe_allow_html=True)    


# -------------------------------------------------
# REVIEW BEFORE SAVING
# -------------------------------------------------
        st.markdown("### ✅ Review Catch Before Saving")

        st.write(f"**Species:** {format_species_name(species)}")
        st.write(f"**Number Caught:** {fish_count}")
        st.write(f"**Fish Lengths:** {fish_lengths}")
        st.write(f"**Bait:** {bait}")
        st.write(f"**Technique:** {technique}")
        st.write(f"**Location:** {water_body}")

        water_body_clean = clean_display_text(water_body)
        
        catch_entry = {
            "angler_id": selected_angler_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "species": species,
            "fish_count": fish_count,
            "fish_lengths": fish_lengths,
            "bait": bait,
            "technique": technique,
            "notes": notes,
            "water_body": water_body_clean,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "resolved_location": log_detected_address if log_detected_address else None,
        }

        if log_detected_address is not None:
            catch_entry["resolved_location"] = log_detected_address

        save_catch(catch_entry)
        st.success("Catch saved successfully!")
        st.rerun()


# -------------------------------------------------
# PAGE 3: MY CATCH LOG
# -------------------------------------------------
if page == "My Catch Log":
    st.markdown("""
    # 📖 My Catch Log
    ### Review your catch history, statistics, locations, and saved entries.
    """)

    current_angler_id = st.session_state.get("angler_id")

    if current_angler_id is None:
        st.warning("Please log in to view your catch log.")
        st.stop()

    catch_df = get_user_catches_df(current_angler_id)

    if catch_df.empty:
        st.info("No catches logged yet.")
    else:
        catch_df["location_label"] = catch_df.apply(
            lambda row: row["water_body"]
            if pd.notnull(row.get("water_body")) and str(row.get("water_body")).strip() != ""
            else f"{row['latitude']}, {row['longitude']}",
            axis=1
        )

        with st.expander("📊 Catch Statistics", expanded=True):
            total_logs = len(catch_df)
            total_fish = catch_df["fish_count"].sum()
            lengths_df = get_user_fish_lengths_df(current_angler_id)

            avg_length = (
                lengths_df["fish_length"].mean()
                if not lengths_df.empty
                else 0
            )

            largest_fish = (
                lengths_df["fish_length"].max()
                if not lengths_df.empty
                else 0
            )

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
                st.metric("Average Fish Length (in)", round(avg_length, 1) if pd.notnull(avg_length) else 0)
                st.metric("Largest Fish (in)", round(largest_fish, 1) if pd.notnull(largest_fish) else 0)
                st.metric("Most Caught Species", format_species_name(most_common_species))

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
        
        top_species_locations["species"] = top_species_locations["species"].apply(format_species_name)
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
        filtered_df_display["species"] = filtered_df_display["species"].apply(format_species_name)
        
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
                location_str = clean_display_text(row["water_body"])
            else:
                location_str = f"{row['latitude']}, {row['longitude']}"

            return f"{timestamp_str} | {format_species_name(row['species'])} | {int(row['fish_count'])} fish | {location_str}"

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

                original_index = selected_row["index"]
                delete_catch(selected_row["id"])
                st.success("Selected catch deleted.")
                st.rerun()

        st.markdown("### ⚠ Clear Entire Catch Log")

        confirm_clear = st.checkbox(
            "I understand this will permanently remove all logged catches for this session.",
            key="confirm_clear_catch_log"
        )

        if st.button("Clear All Catches", key="clear_all_catches_btn"):
            if confirm_clear:
                clear_all_catches()
                st.success("All catches cleared.")
                st.rerun()
            else:
                st.warning("Please confirm before clearing the catch log.")


# -------------------------------------------------
# PAGE 4: CHALLENGES
# -------------------------------------------------
if page == "Challenges":
    st.markdown("""
    # 🏆 Challenges
    ### Track progress, earn milestones, and stay motivated to log catches.
    """)

    current_angler_id = st.session_state.get("angler_id")

    if current_angler_id is None:
        st.warning("Please log in to view your challenges.")
        st.stop()

    catch_df = get_user_catches_df(current_angler_id)

    if catch_df.empty:
        st.info("Log some catches first to see challenge progress.")
    else:

        total_fish_caught = catch_df["fish_count"].sum()
        unique_species_caught = catch_df["species"].nunique()
        unique_locations = catch_df["water_body"].replace("", pd.NA).dropna().nunique()
        unique_techniques = catch_df["technique"].dropna().nunique()
        lengths_df = get_user_fish_lengths_df(current_angler_id)

        largest_fish = (
            lengths_df["fish_length"].max()
            if not lengths_df.empty
            else 0
        )

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
                    st.markdown(f"""
                        <div style="
                            background-color: rgba(31, 122, 140, 0.50);
                            border: 1px solid rgba(31, 122, 140, 0.95);
                            color: #123F4D;
                            padding: 0.8rem;
                            border-radius: 12px;
                            font-weight: 600;
                            margin-bottom: 0.75rem;
                        ">
                        ✔ {challenge["message_done"]}
                        </div>
                        """, unsafe_allow_html=True)
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
                st.markdown(f"""
                    <div style="
                        background-color: rgba(31, 122, 140, 0.50);
                        border: 1px solid rgba(31, 122, 140, 0.95);
                        color: #123F4D;
                        padding: 0.8rem;
                        border-radius: 12px;
                        font-weight: 600;
                        margin-bottom: 0.6rem;
                    ">
                    🏅 {badge}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No achievements unlocked yet. Keep logging catches!")

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
    st.markdown("""
    # 🌎 Community
    ### Compare progress fairly with other anglers while protecting exact fishing spots.
    """)

    if get_catches_df().empty:
        st.info("Log some catches first to unlock the community and social features.")
    else:
        catch_df = get_catches_with_anglers_df()
        division = st.selectbox(
            "Leaderboard Division",
            ["Recreational", "Commercial"],
            key="community_division"
        )

        catch_df = catch_df[catch_df["angler_type"] == division]
        if catch_df.empty:
            st.info(f"No {division.lower()} catches logged yet.")
            st.stop()
        catch_df = add_location_label(catch_df)

        catch_df["timestamp"] = pd.to_datetime(catch_df["timestamp"], errors="coerce")

        total_fish_caught = int(catch_df["fish_count"].sum())
        unique_species_caught = int(catch_df["species"].nunique())

        # Only use fish lengths from the selected leaderboard division
        division_catch_ids = catch_df["id"].tolist()

        lengths_df = get_fish_lengths_df()
        division_lengths_df = lengths_df[lengths_df["catch_id"].isin(division_catch_ids)]

        largest_fish = (
            division_lengths_df["fish_length"].max()
            if not division_lengths_df.empty
            else 0
        )
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
            "Fishing Report Sharing",
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
            st.subheader(f"📅 Weekly Leaderboard — {division}")
            weekly_df = get_weekly_catch_df(catch_df)

            weekly_total_fish = int(weekly_df["fish_count"].sum()) if not weekly_df.empty else 0
            weekly_species = int(weekly_df["species"].nunique()) if not weekly_df.empty else 0
            weekly_catch_ids = weekly_df["id"].tolist()

            weekly_lengths_df = division_lengths_df[
                division_lengths_df["catch_id"].isin(weekly_catch_ids)
            ]

            weekly_largest = (
                weekly_lengths_df["fish_length"].max()
                if not weekly_lengths_df.empty
                else 0
            )

            if weekly_df.empty:
                weekly_leaderboard = pd.DataFrame()
            else:
                lengths_df = get_fish_lengths_df()

                # Basic weekly stats
                weekly_leaderboard = (
                    weekly_df.groupby("username")
                    .agg(
                        **{
                            "Most Fish Caught": ("fish_count", "sum"),
                            "Most Species Logged": ("species", "nunique"),
                        }
                    )
                    .reset_index()
                    .rename(columns={"username": "Angler"})
                ) 

                # Largest fish per user (weekly)
                weekly_lengths = lengths_df.merge(
                    weekly_df[["id", "username"]],
                    left_on="catch_id",
                    right_on="id",
                    how="inner"
                )

                largest_weekly = (
                    weekly_lengths.groupby("username")["fish_length"]
                    .max()
                    .reset_index()
                    .rename(columns={
                        "username": "Angler",
                        "fish_length": "Largest Fish"
                    })
                )

                weekly_leaderboard = weekly_leaderboard.merge(
                    largest_weekly,
                    on="Angler",
                    how="left"
                )

                weekly_leaderboard["Largest Fish"] = weekly_leaderboard["Largest Fish"].fillna(0).round(1)


            weekly_metric = st.selectbox(
                "Weekly leaderboard category",
                ["Most Fish Caught", "Most Species Logged", "Largest Fish"],
                key="weekly_leaderboard_metric"
            )

            weekly_sorted = weekly_leaderboard.sort_values(
                by=weekly_metric,
                ascending=False
            ).reset_index(drop=True)

            badges = ["🥇", "🥈", "🥉"]

            weekly_sorted.insert(
                0,
                "Badge",
                [badges[i] if i < len(badges) else "" for i in range(len(weekly_sorted))]
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
                format_func=format_species_name,
                key="species_leaderboard_choice"
            )

            species_df = catch_df[catch_df["species"] == species_choice]
            species_total = int(species_df["fish_count"].sum())
            species_lengths_df = division_lengths_df[
                division_lengths_df["species"] == species_choice
            ]

            species_largest = (
                float(species_lengths_df["fish_length"].max())
                if not species_lengths_df.empty
                else 0.0
            )

            species_leaderboard = (
                species_df.groupby("username")["fish_count"]
                .sum()
                .reset_index()
                .rename(columns={
                    "username": "Angler",
                    "fish_count": "Fish Caught"
                })
            )

            species_largest_by_user = (
                species_lengths_df.merge(
                    species_df[["id", "username"]],
                    left_on="catch_id",
                    right_on="id",
                    how="inner"
                )
                .groupby("username")["fish_length"]
                .max()
                .reset_index()
                .rename(columns={
                    "username": "Angler",
                    "fish_length": "Largest Fish"
                })
            )

            species_leaderboard = species_leaderboard.merge(
                species_largest_by_user,
                on="Angler",
                how="left"
            )

            species_leaderboard["Largest Fish"] = species_leaderboard["Largest Fish"].fillna(0).round(1)

            species_metric = st.selectbox(
                "Species leaderboard category",
                ["Fish Caught", "Largest Fish"],
                key="species_leaderboard_metric"
            )

            species_sorted = species_leaderboard.sort_values(
                by=species_metric,
                ascending=False
            ).reset_index(drop=True)

            badges = ["🥇", "🥈", "🥉"]

            species_sorted.insert(
                0,
                "Badge",
                [badges[i] if i < len(badges) else "" for i in range(len(species_sorted))]
            )
            species_sorted.index = species_sorted.index + 1
            species_sorted.index.name = "Rank"

            styled_species = species_sorted.style.apply(highlight_user_row, axis=1)
            st.dataframe(styled_species, use_container_width=True)

        # 6. Hotspot Leaderboard
        if "Hotspot Leaderboard" in selected_social_features:
            st.subheader("📍 Hotspot Leaderboard")

            hotspot_source = catch_df.copy()

            hotspot_source["location_name"] = hotspot_source.apply(get_display_location, axis=1)
            hotspot_source["location_name"] = hotspot_source["location_name"].apply(clean_location_name)

            hotspot_df = (
                hotspot_source.groupby("location_name", as_index=False)["fish_count"]
                .sum()
                .sort_values(by="fish_count", ascending=False)
                .head(5)
                .rename(columns={
                    "location_name": "Location",
                    "fish_count": "Total Fish Caught"
                })
            )

            st.dataframe(hotspot_df, use_container_width=True)

        # 7. Fishing Conditions Report Sharing
        if "Fishing Report Sharing" in selected_social_features:
            st.subheader("🌊 Best Fishing Conditions")
            
            reports_df = get_fishing_reports_df()

            if reports_df.empty:
                st.info("Not enough fishing reports have been generated yet.")
            else:
                weather_summary = (
                    reports_df.groupby("weather")["report_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                weather_summary.columns = [
                    "Weather Condition",
                    "Average Report Score"
                ]

                weather_summary["Weather Condition"] = (
                    weather_summary["Weather Condition"]
                    .str.replace("_", " ")
                    .str.title()
                )
                
                weather_summary["Average Report Score"] = (
                    weather_summary["Average Report Score"]
                    .round(1)
                )

                best_weather = weather_summary.iloc[0]

                st.success(
                    f"Best average conditions so far: "
                    f"**{best_weather['Weather Condition']}** "
                    f"with an average report score of "
                    f"**{best_weather['Average Report Score']}**."
                )

                st.dataframe(
                    weather_summary,
                    use_container_width=True,
                    hide_index=True
                )

        # 8. Titles / Skill Levels
        if "Titles / Skill Levels" in selected_social_features:
            st.subheader("🏅 Skill Level")
            title = get_user_title(total_fish_caught, unique_species_caught, largest_fish)
            st.markdown(f"""
            <div style="
                background-color: rgba(31, 122, 140, 0.25);
                border: 1px solid rgba(31, 122, 140, 0.50);
                color: #0F3D4C;
                padding: 0.8rem;
                border-radius: 12px;
                font-weight: 600;
                margin-bottom: 0.5rem;
            ">
            🏅 Your current title: {title}
            </div>
            """, unsafe_allow_html=True)

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
    st.markdown("""
    # 🗺️ Catch Map
    ### Explore logged catches by species and general fishing areas.
    """)

    current_angler_id = st.session_state.get("angler_id")

    if current_angler_id is None:
        st.warning("Please log in to view your catch map.")
        st.stop()

    catch_df = get_user_catches_df(current_angler_id)

    if catch_df.empty:
        st.info("Log some catches first to see them on the map.")
    else:

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

                map_df["public_latitude"] = map_df["latitude"].round(1)
                map_df["public_longitude"] = map_df["longitude"].round(1)
                map_df["public_area"] = map_df.apply(
                    lambda row: get_public_area_label(row["latitude"], row["longitude"]),
                    axis=1
                )

                fig = px.scatter_map(
                    map_df,
                    lat="public_latitude",
                    lon="public_longitude",
                    color="species",
                    size="fish_count",
                    size_max=7,
                    color_discrete_map=SPECIES_COLOR_MAP,
                    hover_name="species",
                    hover_data={
                        "fish_count": True,
                        "bait": True,
                        "technique": True,
                        "public_area": True,
                        "timestamp_display": True,
                        "latitude": False,
                        "longitude": False,
                        "public_latitude": False,
                        "public_longitude": False,
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
# PAGE 7: AGENCY REPORT
# -------------------------------------------------
if page == "Agency Report":
    st.markdown("""
    # 📊 Agency Harvest Report
    ### Export structured harvest data for fisheries and wildlife review.
    """)

    agency_df = get_agency_report_df()

    if agency_df.empty:
        st.info("No catch data available yet.")
    else:
        st.dataframe(agency_df, use_container_width=True)

        csv = agency_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Agency Report CSV",
            data=csv,
            file_name="agency_harvest_report.csv",
            mime="text/csv"
        )




# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Prototype app using SQLite local storage, offline cache, live weather/NOAA tide data, and an XGBoost fishing success model.")
