from datetime import datetime, timedelta

import pandas as pd
import requests


DISTRICT_LOCATIONS = {
    "Andhra Pradesh": {
        "Guntur": (16.3067, 80.4365),
        "Krishna": (16.6100, 80.7214),
        "Kurnool": (15.8281, 78.0373),
        "Nellore": (14.4426, 79.9865),
        "Visakhapatnam": (17.6868, 83.2185),
    },
    "Karnataka": {
        "Belagavi": (15.8497, 74.4977),
        "Bengaluru Urban": (12.9716, 77.5946),
        "Dharwad": (15.4589, 75.0078),
        "Mysuru": (12.2958, 76.6394),
        "Tumakuru": (13.3409, 77.1010),
    },
    "Maharashtra": {
        "Aurangabad": (19.8762, 75.3433),
        "Mumbai": (19.0760, 72.8777),
        "Nagpur": (21.1458, 79.0882),
        "Nashik": (19.9975, 73.7898),
        "Pune": (18.5204, 73.8567),
    },
    "Tamil Nadu": {
        "Chennai": (13.0827, 80.2707),
        "Coimbatore": (11.0168, 76.9558),
        "Madurai": (9.9252, 78.1198),
        "Salem": (11.6643, 78.1460),
        "Tiruchirappalli": (10.7905, 78.7047),
    },
    "Telangana": {
        "Hyderabad": (17.3850, 78.4867),
        "Karimnagar": (18.4386, 79.1288),
        "Khammam": (17.2473, 80.1514),
        "Nalgonda": (17.0575, 79.2684),
        "Warangal": (17.9689, 79.5941),
    },
}


SEASON_CODES = {
    "Monsoon": 0,
    "Summer": 1,
    "Winter": 2,
}

# Availability districts that are stored under a different name in the quality file.
QUALITY_DISTRICT_ALIASES = {
    ("Andhra Pradesh", "Krishna"): "Vijayawada",
    ("Karnataka", "Bengaluru Urban"): "Bengaluru",
    ("Karnataka", "Dharwad"): "Hubballi",
}


def get_season(month):
    month = int(month)
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (3, 4, 5):
        return "Summer"
    return "Winter"


def get_location(state, district):
    try:
        latitude, longitude = DISTRICT_LOCATIONS[state][district]
    except KeyError:
        raise ValueError("Please select a supported state and district.")

    return {
        "state": state,
        "district": district,
        "latitude": latitude,
        "longitude": longitude,
    }


def _recent_rainfall_mm(hourly, observed_at, hours=24):
    """Sum observed hourly rain ending at the station's current local time."""
    times = hourly.get("time") or []
    amounts = hourly.get("precipitation") or []
    if not times or not amounts or not observed_at:
        return None

    end = datetime.fromisoformat(observed_at)
    start = end - timedelta(hours=hours)
    total = 0.0
    counted = 0

    for stamp, amount in zip(times, amounts):
        if amount is None:
            continue
        observed = datetime.fromisoformat(stamp)
        if start < observed <= end:
            total += float(amount)
            counted += 1

    if counted == 0:
        return None
    return total


def fetch_open_meteo_weather(latitude, longitude):
    """Fetch live weather for this coordinate pair.

    Temperature and humidity are the current station readings.
    Rainfall is the past 24 hours of observed precipitation. The current-hour
    amount is often 0 mm even when the district had rain earlier today, which
    made every district look the same.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "hourly": "precipitation",
        "timezone": "auto",
        "past_days": 1,
        "forecast_days": 1,
    }

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current") or {}

    if current.get("temperature_2m") is None or current.get("relative_humidity_2m") is None:
        raise ValueError("Open-Meteo returned no current weather values.")

    observed_at = current.get("time") or datetime.now().strftime("%Y-%m-%dT%H:%M")
    observed_date = observed_at[:10]
    month = int(observed_date[5:7])
    season = get_season(month)
    rainfall = _recent_rainfall_mm(payload.get("hourly") or {}, observed_at)
    if rainfall is None:
        rainfall = float(current.get("precipitation") or 0.0)

    return {
        "Date": observed_date,
        "Temperature_C": round(float(current["temperature_2m"]), 2),
        "Rainfall_mm": round(float(rainfall), 2),
        "Humidity_pct": round(float(current["relative_humidity_2m"]), 2),
        "Current_Precipitation_mm": round(float(current.get("precipitation") or 0.0), 2),
        "Season": season,
        "Season_Code": SEASON_CODES[season],
        "observed_at": observed_at,
        "rainfall_window": "past 24 hours",
        "source": "Open-Meteo",
        "latitude": round(float(payload.get("latitude", latitude)), 4),
        "longitude": round(float(payload.get("longitude", longitude)), 4),
    }


def _district_names(dataframe, state):
    if "State" not in dataframe.columns or "District" not in dataframe.columns:
        return set()
    return set(dataframe.loc[dataframe["State"] == state, "District"].astype(str).unique())


def _numeric_series(dataframe, field):
    if field not in dataframe.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(dataframe[field], errors="coerce").dropna()


def latest_numeric_defaults(dataframe, state, district, fields, aliases=None):
    """Latest non-empty values for one district.

    A missing district used to fall back to the last row of the whole file, so
    Krishna, Mumbai, Nalgonda and the other unmatched names all received the
    same numbers. Unmatched districts now use a same-place alias when one
    exists, otherwise the state median.
    """
    aliases = aliases or {}
    names = _district_names(dataframe, state)
    source_district = district if district in names else aliases.get((state, district))
    match = "exact"

    if source_district in names:
        filtered = dataframe[
            (dataframe["State"] == state) &
            (dataframe["District"] == source_district)
        ].copy()
        if source_district != district:
            match = "alias"
    elif names:
        filtered = dataframe[dataframe["State"] == state].copy()
        source_district = None
        match = "state_median"
    else:
        filtered = dataframe.copy()
        source_district = None
        match = "dataset_median"

    source_date = None
    if "Date" in filtered.columns and not filtered.empty:
        filtered = filtered.copy()
        filtered["Date"] = filtered["Date"].astype(str)
        filtered = filtered.sort_values("Date")
        if match in ("exact", "alias"):
            source_date = str(filtered["Date"].iloc[-1])

    defaults = {}
    for field in fields:
        series = _numeric_series(filtered, field)
        if series.empty:
            value = 0.0
        elif match in ("exact", "alias"):
            value = float(series.iloc[-1])
        else:
            value = float(series.median())
        defaults[field] = round(value, 4)

    if match == "exact":
        note = f"Latest {district} record ({source_date})."
    elif match == "alias":
        note = (
            f"No {district} rows in this dataset. "
            f"Using {source_district}, the matching district ({source_date})."
        )
    elif match == "state_median":
        note = (
            f"No {district} rows in this dataset. "
            f"Using the {state} median so this district is not copied from another district."
        )
    else:
        note = "No matching state rows. Using the dataset median."

    return defaults, {
        "match": match,
        "district": district,
        "source_district": source_district or district,
        "source_date": source_date,
        "note": note,
    }
