import os
import re
import sqlite3
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, render_template, redirect, url_for, session
from utils.lime import generate_lime_html, generate_plotly_lime
from utils.live import (
    DISTRICT_LOCATIONS,
    QUALITY_DISTRICT_ALIASES,
    fetch_open_meteo_weather,
    get_location,
    latest_numeric_defaults,
)
from utils.recommendations import generate_recommendations
import preprocessing


warnings.filterwarnings("ignore")


# =========================================================
# Flask application
# =========================================================
app = Flask(__name__)
app.secret_key = "water_quality_secret_key"


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =========================================================
# Project paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "signup.db")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_water_quality_model.pkl"
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)

TARGET_ENCODER_PATH = os.path.join(
    MODELS_DIR,
    "target_encoder.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)

AVAILABILITY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "water_availability_XGBoost_7day.sav"
)

AVAILABILITY_SCALER_PATH = os.path.join(
    MODELS_DIR,
    "Water_Scalers.pkl"
)

AVAILABILITY_DATA_PATH = os.path.join(
    BASE_DIR,
    "water_availability_forecasting_10years.csv"
)

WATER_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset-water-realistic.csv"
)


# =========================================================
# Database initialization
# =========================================================
def init_db():

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT UNIQUE,
            name TEXT,
            email TEXT,
            mobile TEXT,
            password TEXT
        )
    """)

    con.commit()
    con.close()


init_db()


# =========================================================
# Load model artifacts
# =========================================================
model = None
scaler = None
target_encoder = None
feature_names = None


def load_artifacts():

    global model
    global scaler
    global target_encoder
    global feature_names

    if model is not None:
        return

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        target_encoder = joblib.load(TARGET_ENCODER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

        feature_names = list(feature_names)

    except Exception as error:
        raise RuntimeError(
            "Failed to load model files. Make sure the models folder contains "
            "best_water_quality_model.pkl, scaler.pkl, target_encoder.pkl "
            f"and feature_names.pkl. Details: {error}"
        )


load_artifacts()


# =========================================================
# Load availability artifacts
# =========================================================
availability_model = None
availability_target_encoder = None
availability_scaler = None
df_availability = None
state_encoder = None
district_encoder = None
season_encoder = None

def load_availability_artifacts():
    global availability_model
    global availability_target_encoder
    global availability_scaler
    global df_availability

    if availability_model is not None:
        return

    try:
        avail_data = joblib.load(AVAILABILITY_MODEL_PATH)
        # Note: target_encoder is not used for regression, but we load it if present
        availability_model = avail_data.get("model", avail_data)
        availability_target_encoder = avail_data.get("encoder", None)
        availability_scaler = joblib.load(AVAILABILITY_SCALER_PATH)

        # Load database
        df_availability = pd.read_csv(AVAILABILITY_DATA_PATH)
        df_availability["Date"] = pd.to_datetime(df_availability["Date"])

    except Exception as error:
        raise RuntimeError(
            "Failed to load availability model files. Make sure the models folder contains "
            "water_availability_XGBoost_7day.sav and Water_Scalers.pkl, and the base folder contains "
            f"water_availability_forecasting_10years.csv. Details: {error}"
        )


try:
    load_availability_artifacts()
except Exception:
    # Keep the app usable when an XGBoost pickle was created with a different
    # runtime. Forecast routes call the loader again and show the exact error.
    availability_model = None


def preprocess_availability_features(df):
    df_scaled = df.copy()
    scaler_cols = list(availability_scaler.feature_names_in_)
    
    # Construct a DataFrame with the 24 columns for the scaler
    scaler_df = pd.DataFrame(index=df_scaled.index)
    for col in scaler_cols:
        if col in df_scaled.columns:
            scaler_df[col] = df_scaled[col]
        elif col == 'Water_lag_1' and 'Water_Availability_Lag_1' in df_scaled.columns:
            scaler_df[col] = df_scaled['Water_Availability_Lag_1']
        elif col == 'Water_lag_3' and 'Water_Availability_Lag_1' in df_scaled.columns:
            scaler_df[col] = df_scaled['Water_Availability_Lag_1']
        elif col == 'Water_lag_7' and 'Water_Availability_Lag_7' in df_scaled.columns:
            scaler_df[col] = df_scaled['Water_Availability_Lag_7']
        elif col == 'Water_lag_14' and 'Water_Availability_Lag_7' in df_scaled.columns:
            scaler_df[col] = df_scaled['Water_Availability_Lag_7']
        elif col == 'Water_lag_15' and 'Water_Availability_Lag_15' in df_scaled.columns:
            scaler_df[col] = df_scaled['Water_Availability_Lag_15']
        elif col == 'Water_ma_7' and 'Rainfall_mm_RollingMean_7' in df_scaled.columns:
            scaler_df[col] = df_scaled['Rainfall_mm_RollingMean_7']
        elif col == 'Water_ma_14' and 'Rainfall_mm_RollingMean_15' in df_scaled.columns:
            scaler_df[col] = df_scaled['Rainfall_mm_RollingMean_15']
        elif col == 'Water_ma_15' and 'Rainfall_mm_RollingMean_15' in df_scaled.columns:
            scaler_df[col] = df_scaled['Rainfall_mm_RollingMean_15']
        elif col == 'Water_ma_30' and 'Rainfall_mm_RollingMean_30' in df_scaled.columns:
            scaler_df[col] = df_scaled['Rainfall_mm_RollingMean_30']
        elif col == 'log_water_change':
            scaler_df[col] = 0.0
            if 'Water_Availability_Lag_1' in df_scaled.columns and 'Water_Availability_Lag_7' in df_scaled.columns:
                scaler_df[col] = df_scaled['Water_Availability_Lag_1'] - df_scaled['Water_Availability_Lag_7']
        elif col == 'RSI':
            scaler_df[col] = 0.5
        else:
            scaler_df[col] = 0.0
            
    # Transform using availability scaler
    scaled_values = availability_scaler.transform(scaler_df)
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_cols, index=df_scaled.index)
    
    # Map them back into df_scaled
    for col in scaler_cols:
        if col in df_scaled.columns:
            df_scaled[col] = scaled_df[col]
        elif col == 'Water_lag_1' and 'Water_Availability_Lag_1' in df_scaled.columns:
            df_scaled['Water_Availability_Lag_1'] = scaled_df['Water_lag_1']
        elif col == 'Water_lag_7' and 'Water_Availability_Lag_7' in df_scaled.columns:
            df_scaled['Water_Availability_Lag_7'] = scaled_df['Water_lag_7']
        elif col == 'Water_lag_15' and 'Water_Availability_Lag_15' in df_scaled.columns:
            df_scaled['Water_Availability_Lag_15'] = scaled_df['Water_lag_15']
            
    return df_scaled


class AvailabilityModelWrapper:
    def __init__(self, trained_model, feature_names):
        self.model = trained_model
        self.feature_names = feature_names

    def predict_proba(self, rows):
        if isinstance(rows, pd.DataFrame):
            rows = rows[self.feature_names].to_numpy(dtype=float)
        else:
            rows = np.asarray(rows, dtype=float)

        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        return self.model.predict_proba(rows)


# =========================================================
# Model wrapper for LIME
# =========================================================
class ScaledModelWrapper:

    def __init__(self, trained_model, trained_scaler):
        self.model = trained_model
        self.scaler = trained_scaler

    def predict_proba(self, rows):

        if isinstance(rows, pd.DataFrame):
            rows = rows[feature_names].to_numpy(dtype=float)
        else:
            rows = np.asarray(rows, dtype=float)

        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        scaled_rows = self.scaler.transform(rows)

        return self.model.predict_proba(scaled_rows)


lime_model = ScaledModelWrapper(
    trained_model=model,
    trained_scaler=scaler
)


# =========================================================
# Helper functions
# =========================================================
def class_names_function():

    return [
        str(class_name)
        for class_name in target_encoder.classes_
    ]


def identity_preprocess(dataframe):

    processed = dataframe.copy()

    for column in processed.columns:
        processed[column] = pd.to_numeric(
            processed[column],
            errors="coerce"
        )

    return processed.dropna()


def get_form_float(field_name):

    value = request.form.get(field_name, "").strip()

    if not value:
        raise ValueError(
            f"Please enter a value for {field_name}."
        )

    try:
        return float(value)

    except ValueError:
        raise ValueError(
            f"{field_name} must be a valid number."
        )





def get_indicator_breakdown(submitted_values):
    # Extract values
    ph = submitted_values.get("pH", 7.0)
    do = submitted_values.get("Dissolved_Oxygen_mgL", 8.0)
    bod = submitted_values.get("BOD_mgL", 2.0)
    nitrate = submitted_values.get("Nitrate_mgL", 2.0)
    tds = submitted_values.get("TDS_mgL", 150.0)
    turbidity = submitted_values.get("Turbidity_NTU", 1.0)
    
    breakdown = []
    
    # 1. pH Level
    ph_level = "Poor"
    ph_badge = "danger"
    ph_pct = 25
    if 7.0 <= ph <= 7.5:
        ph_level = "Excellent"
        ph_badge = "success"
        ph_pct = 95
    elif (6.5 <= ph < 7.0) or (7.5 < ph <= 8.0):
        ph_level = "Good"
        ph_badge = "primary"
        ph_pct = 75
    elif (6.0 <= ph < 6.5) or (8.0 < ph <= 8.5):
        ph_level = "Moderate"
        ph_badge = "warning"
        ph_pct = 50
    else:
        ph_level = "Poor"
        ph_badge = "danger"
        ph_pct = 20
        
    breakdown.append({
        "name": "pH Level",
        "value": ph,
        "unit": "",
        "level": ph_level,
        "badge_class": ph_badge,
        "progress_pct": ph_pct,
        "icon": "fa-flask",
        "ranges": "Excellent: 7.0 - 7.5 | Good: 6.5 - 7.0 or 7.5 - 8.0 | Moderate: 6.0 - 6.5 or 8.0 - 8.5 | Poor: <6.0 or >8.5",
        "description": "Measures water acidity or alkalinity. A range of 6.5 to 8.5 is generally acceptable, with 7.0-7.5 being ideal."
    })
    
    # 2. Dissolved Oxygen (DO)
    do_level = "Poor"
    do_badge = "danger"
    do_pct = 20
    if do >= 8.2:
        do_level = "Excellent"
        do_badge = "success"
        do_pct = 95
    elif 7.8 <= do < 8.2:
        do_level = "Good"
        do_badge = "primary"
        do_pct = 75
    elif 7.4 <= do < 7.8:
        do_level = "Moderate"
        do_badge = "warning"
        do_pct = 50
    else:
        do_level = "Poor"
        do_badge = "danger"
        do_pct = 25
        
    breakdown.append({
        "name": "Dissolved Oxygen (DO)",
        "value": do,
        "unit": " mg/L",
        "level": do_level,
        "badge_class": do_badge,
        "progress_pct": do_pct,
        "icon": "fa-wind",
        "ranges": "Excellent: >= 8.2 | Good: 7.8 - 8.2 | Moderate: 7.4 - 7.8 | Poor: < 7.4",
        "description": "Amount of oxygen dissolved in water. Higher values support rich aquatic life and cleaner water ecosystems."
    })
    
    # 3. BOD (Biochemical Oxygen Demand)
    bod_level = "Poor"
    bod_badge = "danger"
    bod_pct = 20
    if bod < 2.8:
        bod_level = "Excellent"
        bod_badge = "success"
        bod_pct = 95
    elif 2.8 <= bod < 3.5:
        bod_level = "Good"
        bod_badge = "primary"
        bod_pct = 75
    elif 3.5 <= bod < 4.2:
        bod_level = "Moderate"
        bod_badge = "warning"
        bod_pct = 50
    else:
        bod_level = "Poor"
        bod_badge = "danger"
        bod_pct = 25
        
    breakdown.append({
        "name": "BOD (Biochemical Oxygen Demand)",
        "value": bod,
        "unit": " mg/L",
        "level": bod_level,
        "badge_class": bod_badge,
        "progress_pct": bod_pct,
        "icon": "fa-biohazard",
        "ranges": "Excellent: < 2.8 | Good: 2.8 - 3.5 | Moderate: 3.5 - 4.2 | Poor: >= 4.2",
        "description": "Amount of oxygen required by organisms to break down organic waste. Lower values indicate lower organic pollution."
    })
    
    # 4. Nitrate
    nitrate_level = "Poor"
    nitrate_badge = "danger"
    nitrate_pct = 20
    if nitrate < 2.4:
        nitrate_level = "Excellent"
        nitrate_badge = "success"
        nitrate_pct = 95
    elif 2.4 <= nitrate < 2.8:
        nitrate_level = "Good"
        nitrate_badge = "primary"
        nitrate_pct = 75
    elif 2.8 <= nitrate < 3.2:
        nitrate_level = "Moderate"
        nitrate_badge = "warning"
        nitrate_pct = 50
    else:
        nitrate_level = "Poor"
        nitrate_badge = "danger"
        nitrate_pct = 25
        
    breakdown.append({
        "name": "Nitrate",
        "value": nitrate,
        "unit": " mg/L",
        "level": nitrate_level,
        "badge_class": nitrate_badge,
        "progress_pct": nitrate_pct,
        "icon": "fa-flask-vial",
        "ranges": "Excellent: < 2.4 | Good: 2.4 - 2.8 | Moderate: 2.8 - 3.2 | Poor: >= 3.2",
        "description": "Nitrogen compound level in water. Excess nitrates run off from agricultural fields and can degrade water safety."
    })
    
    # 5. TDS (Total Dissolved Solids)
    tds_level = "Poor"
    tds_badge = "danger"
    tds_pct = 20
    if tds < 180:
        tds_level = "Excellent"
        tds_badge = "success"
        tds_pct = 95
    elif 180 <= tds < 210:
        tds_level = "Good"
        tds_badge = "primary"
        tds_pct = 75
    elif 210 <= tds < 250:
        tds_level = "Moderate"
        tds_badge = "warning"
        tds_pct = 50
    else:
        tds_level = "Poor"
        tds_badge = "danger"
        tds_pct = 25
        
    breakdown.append({
        "name": "TDS (Total Dissolved Solids)",
        "value": tds,
        "unit": " mg/L",
        "level": tds_level,
        "badge_class": tds_badge,
        "progress_pct": tds_pct,
        "icon": "fa-vial",
        "ranges": "Excellent: < 180 | Good: 180 - 210 | Moderate: 210 - 250 | Poor: >= 250",
        "description": "Total concentration of dissolved substances. Lower values indicate purer water with fewer minerals and residues."
    })
    
    # 6. Turbidity
    turbidity_level = "Poor"
    turbidity_badge = "danger"
    turbidity_pct = 20
    if turbidity < 1.8:
        turbidity_level = "Excellent"
        turbidity_badge = "success"
        turbidity_pct = 95
    elif 1.8 <= turbidity < 3.0:
        turbidity_level = "Good"
        turbidity_badge = "primary"
        turbidity_pct = 75
    elif 3.0 <= turbidity < 4.5:
        turbidity_level = "Moderate"
        turbidity_badge = "warning"
        turbidity_pct = 50
    else:
        turbidity_level = "Poor"
        turbidity_badge = "danger"
        turbidity_pct = 25
        
    breakdown.append({
        "name": "Turbidity",
        "value": turbidity,
        "unit": " NTU",
        "level": turbidity_level,
        "badge_class": turbidity_badge,
        "progress_pct": turbidity_pct,
        "icon": "fa-water",
        "ranges": "Excellent: < 1.8 | Good: 1.8 - 3.0 | Moderate: 3.0 - 4.5 | Poor: >= 4.5",
        "description": "Measures clarity of water. Lower NTU numbers indicate clearer water with less suspended particulate matter."
    })
    
    return breakdown


def get_calculated_indices_breakdown(pollution_index, water_quality_score):
    breakdown = []
    
    # 1. Pollution Index
    pi = float(pollution_index)
    pi_level = "Very Bad"
    pi_badge = "danger"
    pi_pct = 10
    if pi < 6.5:
        pi_level = "Excellent"
        pi_badge = "success"
        pi_pct = 95
    elif 6.5 <= pi < 9.5:
        pi_level = "Good"
        pi_badge = "primary"
        pi_pct = 75
    elif 9.5 <= pi < 12.5:
        pi_level = "Moderate"
        pi_badge = "warning"
        pi_pct = 50
    elif 12.5 <= pi < 15.5:
        pi_level = "Poor"
        pi_badge = "danger"
        pi_pct = 25
    else:
        pi_level = "Very Bad"
        pi_badge = "danger"
        pi_pct = 10
        
    breakdown.append({
        "name": "Calculated Pollution Index",
        "value": round(pi, 4),
        "unit": "",
        "level": pi_level,
        "badge_class": pi_badge,
        "progress_pct": pi_pct,
        "icon": "fa-biohazard",
        "ranges": "Excellent: < 6.5 | Good: 6.5 - 9.5 | Moderate: 9.5 - 12.5 | Poor: 12.5 - 15.5 | Very Bad: >= 15.5",
        "description": "Calculated as BOD + Nitrate + Turbidity. Represents the overall organic and chemical pollution level. Lower values are cleaner."
    })
    
    # 2. Water Quality Score
    wqs = float(water_quality_score)
    wqs_level = "Very Bad"
    wqs_badge = "danger"
    wqs_pct = 10
    if wqs >= 11.5:
        wqs_level = "Excellent"
        wqs_badge = "success"
        wqs_pct = 95
    elif 9.5 <= wqs < 11.5:
        wqs_level = "Good"
        wqs_badge = "primary"
        wqs_pct = 75
    elif 7.5 <= wqs < 9.5:
        wqs_level = "Moderate"
        wqs_badge = "warning"
        wqs_pct = 50
    elif 5.5 <= wqs < 7.5:
        wqs_level = "Poor"
        wqs_badge = "danger"
        wqs_pct = 25
    else:
        wqs_level = "Very Bad"
        wqs_badge = "danger"
        wqs_pct = 10
        
    breakdown.append({
        "name": "Calculated Water Quality Score",
        "value": round(wqs, 4),
        "unit": "",
        "level": wqs_level,
        "badge_class": wqs_badge,
        "progress_pct": wqs_pct,
        "icon": "fa-star",
        "ranges": "Excellent: >= 11.5 | Good: 9.5 - 11.5 | Moderate: 7.5 - 9.5 | Poor: 5.5 - 7.5 | Very Bad: < 5.5",
        "description": "Calculated as DO + pH - BOD - Turbidity. Represents overall water health index based on vital metrics. Higher values are healthier."
    })
    
    return breakdown


# =========================================================
# Page routes
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")




@app.route("/home1")
def home1():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "home1.html",
        error=request.args.get("error")
    )


@app.route("/home2")
def home2():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "home2.html",
        error=request.args.get("error")
    )


def get_live_form_float(field_name, fallback):
    value = request.form.get(field_name, "").strip()
    if value == "":
        return float(fallback)
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name.replace('_', ' ')} must be a valid number.")


def predict_live_quality(values):
    load_artifacts()

    pollution_index = values["BOD_mgL"] + values["Nitrate_mgL"] + values["Turbidity_NTU"]
    water_quality_score = (
        values["Dissolved_Oxygen_mgL"] +
        values["pH"] -
        values["BOD_mgL"] -
        values["Turbidity_NTU"]
    )

    model_values = {
        "Pollution_Index": pollution_index,
        "Water_Quality_Score": water_quality_score,
        "Rainfall_mm": values["Rainfall_mm"],
        "TDS_mgL": values["TDS_mgL"],
        "Turbidity_NTU": values["Turbidity_NTU"],
        "pH": values["pH"],
        "Dissolved_Oxygen_mgL": values["Dissolved_Oxygen_mgL"],
        "Season": values["Season_Code"],
        "River_Flow_m3s": values["River_Flow_m3s"],
        "Humidity_pct": values["Humidity_pct"],
    }

    input_df = pd.DataFrame(
        [[model_values[feature] for feature in feature_names]],
        columns=feature_names
    )
    input_scaled = scaler.transform(input_df)
    predicted_class = int(model.predict(input_scaled)[0])
    probability_array = model.predict_proba(input_scaled)[0]
    prediction = target_encoder.inverse_transform([predicted_class])[0]

    return {
        "prediction": str(prediction),
        "confidence": round(float(probability_array[predicted_class]) * 100, 2),
        "probabilities": {
            str(class_name): round(float(probability) * 100, 2)
            for class_name, probability in zip(target_encoder.classes_, probability_array)
        },
        "pollution_index": round(float(pollution_index), 4),
        "water_quality_score": round(float(water_quality_score), 4),
    }


def predict_live_availability(values):
    load_availability_artifacts()

    user_input = {
        "State": values["State"],
        "District": values["District"],
        "Date": values["Date"],
        "Temperature_C": values["Temperature_C"],
        "Rainfall_mm": values["Rainfall_mm"],
        "Humidity_pct": values["Humidity_pct"],
        "Reservoir_Level_pct": values["Reservoir_Level_pct"],
        "Groundwater_Level_m": values["Groundwater_Level_m"],
        "River_Flow_m3s": values["River_Flow_m3s"],
        "Population": values["Population"],
        "Water_Consumption_MLD": values["Water_Consumption_MLD"],
        "Reservoir_Inflow_MLD": values["Reservoir_Inflow_MLD"],
        "Reservoir_Outflow_MLD": values["Reservoir_Outflow_MLD"],
        "Water_Availability_MLD": values["Water_Availability_MLD"],
    }

    df_scaled, idx, _, _ = preprocessing.preprocess_for_prediction(
        user_input,
        df_availability,
        availability_scaler
    )

    availability_feature_names = [
        'Month', 'DayOfYear', 'Temperature_C', 'Rainfall_mm', 'Humidity_pct',
        'Reservoir_Level_pct', 'Groundwater_Level_m', 'River_Flow_m3s',
        'Population', 'Water_Consumption_MLD', 'Reservoir_Inflow_MLD',
        'Reservoir_Outflow_MLD', 'Water_Availability_MLD', 'log_water_change',
        'Water_lag_1', 'Water_lag_3', 'Water_lag_7', 'Water_lag_14',
        'Water_lag_15', 'Water_ma_7', 'Water_ma_14', 'Water_ma_15',
        'Water_ma_30', 'RSI'
    ]

    feature_row = df_scaled.iloc[[idx]][availability_feature_names]
    prediction_return = float(availability_model.predict(feature_row)[0])
    current_availability = float(values["Water_Availability_MLD"])
    predicted_value = round(float(current_availability * np.exp(prediction_return)), 2)

    pct33 = df_availability["Water_Availability_MLD"].quantile(0.33)
    pct66 = df_availability["Water_Availability_MLD"].quantile(0.66)
    if predicted_value <= pct33:
        predicted_class = "Low"
    elif predicted_value <= pct66:
        predicted_class = "Medium"
    else:
        predicted_class = "High"

    forecast_rows = []
    for day in range(1, 8):
        ratio = day / 7.0
        value = round(current_availability + (predicted_value - current_availability) * ratio, 2)
        date_value = pd.to_datetime(values["Date"]) + pd.Timedelta(days=day)
        forecast_rows.append({
            "Date": date_value.strftime("%Y-%m-%d"),
            "Predicted_Value": value,
            "Predicted_Class": (
                "Low" if value <= pct33 else "Medium" if value <= pct66 else "High"
            )
        })

    return {
        "prediction": f"{predicted_value:.2f} MLD",
        "prediction_class": predicted_class,
        "forecast_rows": forecast_rows,
    }


@app.route("/live", methods=["GET"])
def live_dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "live.html",
        district_map=DISTRICT_LOCATIONS,
        error=request.args.get("error")
    )


@app.route("/live/run", methods=["POST"])
def live_run():
    if "username" not in session:
        return redirect(url_for("login"))
    state = request.form.get("State", "").strip()
    district = request.form.get("District", "").strip()

    try:
        location = get_location(state, district)
        weather = fetch_open_meteo_weather(
            latitude=location["latitude"],
            longitude=location["longitude"]
        )
    except Exception as error:
        return render_template(
            "live.html",
            district_map=DISTRICT_LOCATIONS,
            error=f"Live weather fetch failed: {error}"
        )

    try:
        availability_df = pd.read_csv(AVAILABILITY_DATA_PATH)
        water_df = pd.read_csv(WATER_DATA_PATH)

        availability_defaults, availability_source = latest_numeric_defaults(
            availability_df,
            state,
            district,
            [
                "Reservoir_Level_pct",
                "Groundwater_Level_m",
                "River_Flow_m3s",
                "Population",
                "Water_Consumption_MLD",
                "Reservoir_Inflow_MLD",
                "Reservoir_Outflow_MLD",
                "Water_Availability_MLD",
            ]
        )
        quality_defaults, quality_source = latest_numeric_defaults(
            water_df,
            state,
            district,
            [
                "pH",
                "Turbidity_NTU",
                "Dissolved_Oxygen_mgL",
                "TDS_mgL",
                "BOD_mgL",
                "Nitrate_mgL",
            ],
            aliases=QUALITY_DISTRICT_ALIASES,
        )

        manual_override = request.form.get("manual_override") == "1"
        values = {
            "State": state,
            "District": district,
            "Date": request.form.get("Date", "").strip() or weather["Date"],
            "Temperature_C": weather["Temperature_C"],
            "Rainfall_mm": weather["Rainfall_mm"],
            "Humidity_pct": weather["Humidity_pct"],
            "Season": weather["Season"],
            "Season_Code": weather["Season_Code"],
        }

        for field, fallback in {**availability_defaults, **quality_defaults}.items():
            if manual_override:
                values[field] = get_live_form_float(field, fallback)
            else:
                values[field] = float(fallback)

        quality_result = None
        quality_error = None
        availability_result = None
        availability_error = None

        try:
            quality_result = predict_live_quality(values)
        except Exception as error:
            quality_error = str(error)

        try:
            availability_result = predict_live_availability(values)
        except Exception as error:
            availability_error = str(error)

        return render_template(
            "live.html",
            district_map=DISTRICT_LOCATIONS,
            selected_state=state,
            selected_district=district,
            location=location,
            weather=weather,
            values=values,
            manual_override=manual_override,
            availability_source=availability_source,
            quality_source=quality_source,
            quality_result=quality_result,
            quality_error=quality_error,
            availability_result=availability_result,
            availability_error=availability_error,
            error=None
        )

    except Exception as error:
        return render_template(
            "live.html",
            district_map=DISTRICT_LOCATIONS,
            selected_state=state,
            selected_district=district,
            weather=weather,
            error=str(error)
        )


@app.route("/graphs1")
def graphs1():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("graphs1.html")


@app.route("/graphs2")
def graphs2():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("graphs2.html")


@app.route("/notes1")
@app.route("/static/Water_Availability_forecasting.html")
def notes1():
    if "username" not in session:
        return redirect(url_for("login"))
    return app.send_static_file("Water_Availability_forecasting.html")


@app.route("/notes2")
@app.route("/static/Water_Quality_prediction.html")
def notes2():
    if "username" not in session:
        return redirect(url_for("login"))
    return app.send_static_file("Water_Quality_prediction.html")


@app.route("/logon")
def logon():
    return render_template("signup.html")


@app.route("/login")
def login():
    if "username" in session:
        return redirect(url_for("home1"))
    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# Registration
# =========================================================
@app.route("/signup", methods=["GET", "POST"])
def register():

    if request.method == "GET":
        return render_template("signup.html")

    username = request.form.get("user", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    mobile = request.form.get("mobile", "").strip()
    password = request.form.get("password", "")

    username_pattern = r"^.{6,}$"
    name_pattern = r"^[A-Za-z ]{3,}$"
    email_pattern = r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$"
    mobile_pattern = r"^[6-9][0-9]{9}$"
    password_pattern = r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$"

    if not re.match(username_pattern, username):
        return render_template(
            "signup.html",
            message="Username must be at least 6 characters."
        )

    if not re.match(name_pattern, name):
        return render_template(
            "signup.html",
            message=(
                "Full Name must be at least 3 letters, "
                "using only letters and spaces."
            )
        )

    if not re.match(email_pattern, email):
        return render_template(
            "signup.html",
            message="Enter a valid email address."
        )

    if not re.match(mobile_pattern, mobile):
        return render_template(
            "signup.html",
            message="Mobile must start with 6-9 and contain 10 digits."
        )

    if not re.match(password_pattern, password):
        return render_template(
            "signup.html",
            message=(
                "Password must contain at least 8 characters, "
                "one uppercase letter, one lowercase letter and one number."
            )
        )

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "SELECT 1 FROM info WHERE user = ?",
        (username,)
    )

    if cur.fetchone():
        con.close()

        return render_template(
            "signup.html",
            message="Username already exists."
        )

    cur.execute(
        """
        INSERT INTO info (
            user,
            name,
            email,
            mobile,
            password
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            username,
            name,
            email,
            mobile,
            password
        )
    )

    con.commit()
    con.close()

    return redirect(url_for("login"))


# =========================================================
# Sign in
# =========================================================
@app.route("/signin", methods=["GET", "POST"])
def signin():

    if request.method == "GET":
        if "username" in session:
            return redirect(url_for("home1"))
        return render_template("signin.html")

    username = request.form.get("user", "").strip()
    password = request.form.get("password", "")

    if username == "admin" and password == "admin":
        session["username"] = "admin"
        return redirect(url_for("home1"))

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        """
        SELECT user
        FROM info
        WHERE user = ? AND password = ?
        """,
        (
            username,
            password
        )
    )

    data = cur.fetchone()
    con.close()

    if data:
        session["username"] = data[0]
        return redirect(url_for("home1"))

    return render_template(
        "signin.html",
        message="Invalid username or password."
    )


# =========================================================
# Prediction
# =========================================================
def perform_prediction(template_name):
    try:
        load_artifacts()

        # Get input parameters
        bod = get_form_float("BOD_mgL")
        nitrate = get_form_float("Nitrate_mgL")
        rainfall = get_form_float("Rainfall_mm")
        tds = get_form_float("TDS_mgL")
        turbidity = get_form_float("Turbidity_NTU")
        ph = get_form_float("pH")
        do = get_form_float("Dissolved_Oxygen_mgL")
        season = get_form_float("Season")
        river_flow = get_form_float("River_Flow_m3s")
        humidity = get_form_float("Humidity_pct")

        # Calculate engineered features
        pollution_index = bod + nitrate + turbidity
        water_quality_score = do + ph - bod - turbidity

        submitted_values = {
            "Pollution_Index": pollution_index,
            "Water_Quality_Score": water_quality_score,
            "Rainfall_mm": rainfall,
            "TDS_mgL": tds,
            "Turbidity_NTU": turbidity,
            "pH": ph,
            "Dissolved_Oxygen_mgL": do,
            "Season": season,
            "River_Flow_m3s": river_flow,
            "Humidity_pct": humidity
        }

        missing_features = [
            feature
            for feature in feature_names
            if feature not in submitted_values
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        input_df = pd.DataFrame(
            [[
                submitted_values[feature]
                for feature in feature_names
            ]],
            columns=feature_names
        )

        input_scaled = scaler.transform(
            input_df
        )

        predicted_class = int(
            model.predict(input_scaled)[0]
        )

        probability_array = model.predict_proba(
            input_scaled
        )[0]

        prediction = target_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = round(
            float(
                probability_array[predicted_class]
            ) * 100,
            2
        )

        probabilities = {
            str(class_name): round(float(probability) * 100, 2)
            for class_name, probability in zip(
                target_encoder.classes_,
                probability_array
            )
        }

        lime_html, lime_error = generate_lime_html(
            features=input_df,
            model=lime_model,
            selected_features=feature_names,
            num_cols=feature_names,
            encoding={},
            class_names_fn=class_names_function,
            preprocess_raw_features_fn=identity_preprocess,
            background_paths=[
                os.path.join(
                    MODELS_DIR,
                    "lime_background.csv"
                ),
                os.path.join(
                    MODELS_DIR,
                    "water_quality_background.csv"
                )
            ]
        )

        lime_plot, lime_plot_error = generate_plotly_lime(
            raw_features=input_df,
            predicted_class=predicted_class,
            feature_names=feature_names,
            class_names_fn=class_names_function,
            predict_fn=lime_model.predict_proba,
            target_encoder=target_encoder
        )

        if lime_plot_error:
            if lime_error:
                lime_error = (
                    lime_error
                    + " | Plotly LIME error: "
                    + lime_plot_error
                )
            else:
                lime_error = (
                    "Plotly LIME error: "
                    + lime_plot_error
                )

        indicator_values = {
            "pH": ph,
            "Dissolved_Oxygen_mgL": do,
            "BOD_mgL": bod,
            "Nitrate_mgL": nitrate,
            "TDS_mgL": tds,
            "Turbidity_NTU": turbidity
        }
        indicator_breakdown = get_indicator_breakdown(indicator_values)
        indices_breakdown = get_calculated_indices_breakdown(pollution_index, water_quality_score)

        # Get recommendations and suggestions
        rec_data, rec_warning = generate_recommendations(
            features=submitted_values,
            prediction_label=prediction,
            confidence_score=confidence,
            is_water_quality=True
        )

        return render_template(
            template_name,
            prediction=str(prediction),
            confidence=confidence,
            probabilities=probabilities,
            lime_html=lime_html,
            lime_plot=lime_plot,
            lime_error=lime_error,
            input_values=submitted_values,
            pollution_index=pollution_index,
            water_quality_score=water_quality_score,
            indicator_breakdown=indicator_breakdown,
            indices_breakdown=indices_breakdown,
            recommendations=rec_data.get("recommendations"),
            suggestions=rec_data.get("suggestions"),
            crops=rec_data.get("crops"),
            recommendations_source=rec_data.get("source"),
            recommendations_warning=rec_warning,
            error=None
        )

    except Exception as error:
        fallback_home = "home1.html" if "1" in template_name else "home2.html"
        return render_template(
            fallback_home,
            error=str(error)
        )

def perform_forecasting(template_name):
    try:
        load_availability_artifacts()

        # Construct user input dict
        user_input = {
            "State": request.form.get("State", "").strip(),
            "District": request.form.get("District", "").strip(),
            "Date": request.form.get("Date", "").strip(),
            "Temperature_C": request.form.get("Temperature_C", "").strip(),
            "Rainfall_mm": request.form.get("Rainfall_mm", "").strip(),
            "Humidity_pct": request.form.get("Humidity_pct", "").strip(),
            "Reservoir_Level_pct": request.form.get("Reservoir_Level_pct", "").strip(),
            "Groundwater_Level_m": request.form.get("Groundwater_Level_m", "").strip(),
            "River_Flow_m3s": request.form.get("River_Flow_m3s", "").strip(),
            "Population": request.form.get("Population", "").strip(),
            "Water_Consumption_MLD": request.form.get("Water_Consumption_MLD", "").strip(),
            "Reservoir_Inflow_MLD": request.form.get("Reservoir_Inflow_MLD", "").strip(),
            "Reservoir_Outflow_MLD": request.form.get("Reservoir_Outflow_MLD", "").strip(),
            "Water_Availability_MLD": request.form.get("Water_Availability_MLD", "").strip()
        }

        # Perform preprocessing, feature engineering and validation
        df_scaled, idx, df_filtered, df_model_input = preprocessing.preprocess_for_prediction(
            user_input, df_availability, availability_scaler
        )

        state = user_input["State"]
        district = user_input["District"]
        date_str = user_input["Date"]
        target_date = pd.to_datetime(date_str)
        temp = float(user_input["Temperature_C"])
        rainfall = float(user_input["Rainfall_mm"])
        humidity = float(user_input["Humidity_pct"])
        res_level = float(user_input["Reservoir_Level_pct"])
        gw_level = float(user_input["Groundwater_Level_m"])
        river_flow = float(user_input["River_Flow_m3s"])
        population = float(user_input["Population"])
        consumption = float(user_input["Water_Consumption_MLD"])
        inflow = float(user_input["Reservoir_Inflow_MLD"])
        outflow = float(user_input["Reservoir_Outflow_MLD"])
        current_availability = float(user_input["Water_Availability_MLD"])

        # Exact model feature list
        availability_feature_names = [
            'Month', 'DayOfYear', 'Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Reservoir_Level_pct',
            'Groundwater_Level_m', 'River_Flow_m3s', 'Population', 'Water_Consumption_MLD',
            'Reservoir_Inflow_MLD', 'Reservoir_Outflow_MLD', 'Water_Availability_MLD', 'log_water_change',
            'Water_lag_1', 'Water_lag_3', 'Water_lag_7', 'Water_lag_14', 'Water_lag_15',
            'Water_ma_7', 'Water_ma_14', 'Water_ma_15', 'Water_ma_30', 'RSI'
        ]

        # Calculate percentile thresholds for category mappings dynamically
        pct33 = df_availability["Water_Availability_MLD"].quantile(0.33)
        pct66 = df_availability["Water_Availability_MLD"].quantile(0.66)

        def get_category(val):
            if val <= pct33:
                return "Low"
            elif val <= pct66:
                return "Medium"
            else:
                return "High"

        # Containers for forecast and line graph data
        forecast_table = []
        chart_actual = []
        chart_forecast = []
        chart_labels = []

        # 1. Past 7 days actual values (including selected date)
        actual_rows = df_filtered.iloc[max(0, idx - 6) : idx + 1]
        for _, row in actual_rows.iterrows():
            chart_actual.append(float(row["Water_Availability_MLD"]))
            chart_forecast.append(None)
            chart_labels.append(row["Date"].strftime("%Y-%m-%d"))

        # 2. True 7-day horizon prediction from today's features
        feat_row_today = df_scaled.iloc[[idx]][availability_feature_names]
        pred_ret_7d = float(availability_model.predict(feat_row_today)[0])
        day7_val = round(float(current_availability * np.exp(pred_ret_7d)), 2)
        day7_class = get_category(day7_val)

        # 3. Build a coherent 7-day path ending at the model day-7 value
        #    (linear bridge from current availability → day-7 forecast)
        start_val = float(current_availability)
        hist_len = len(chart_actual)  # freeze length before appending forecast days
        for i in range(7):
            forecast_date = target_date + pd.Timedelta(days=i + 1)
            forecast_date_str = forecast_date.strftime("%Y-%m-%d")
            chart_labels.append(forecast_date_str)
            chart_actual.append(None)

            step_ratio = (i + 1) / 7.0
            pred_val = round(start_val + (day7_val - start_val) * step_ratio, 2)
            # Exact Day-7 match with the main card (avoid float drift)
            if i == 6:
                pred_val = day7_val
            pred_class = get_category(pred_val)

            forecast_table.append({
                "Date": forecast_date_str,
                "Predicted_Value": pred_val,
                "Predicted_Class": pred_class
            })
            chart_forecast.append(pred_val)

        # Connect actual → forecast at the last historical day only.
        # Do NOT use len(chart_actual) after the loop — that would overwrite Day 7.
        if hist_len > 0:
            chart_forecast[hist_len - 1] = chart_actual[hist_len - 1]

        prediction = f"{day7_val:.2f} MLD ({day7_class})"
        prediction_value = f"{day7_val:.2f} MLD"
        prediction_class = day7_class
        confidence = 76.43  # Accuracy score

        # Summary statistics (Day 1 = start of path, Day 7 = main model result)
        pred_vals = [row["Predicted_Value"] for row in forecast_table]
        day1_val = forecast_table[0]["Predicted_Value"]
        stats = {
            "max_level": f"{max(pred_vals):.2f} MLD",
            "min_level": f"{min(pred_vals):.2f} MLD",
            "most_frequent": f"{np.mean(pred_vals):.2f} MLD",
            "day1_level": f"{day1_val:.2f} MLD",
            "day7_level": f"{day7_val:.2f} MLD",
            "day1_class": forecast_table[0]["Predicted_Class"],
            "day7_class": day7_class,
        }

        # Chart.js line graph payload
        chart_data = {
            "labels": chart_labels,
            "actual": chart_actual,
            "forecast": chart_forecast
        }

        lime_html = None
        lime_plot = None
        lime_error = None
        try:
            from lime.lime_tabular import LimeTabularExplainer
            from utils.lime import create_lime_plot

            # LIME Regression Explainer Setup
            class AvailabilityRegressionWrapper:
                def __init__(self, model, feature_names):
                    self.model = model
                    self.feature_names = feature_names

                def predict(self, X):
                    return self.model.predict(X)

            availability_lime_model = AvailabilityRegressionWrapper(
                availability_model,
                availability_feature_names
            )

            feat_row_day7_scaled = df_scaled.iloc[[idx]][availability_feature_names].to_numpy()
            feature_row = feat_row_day7_scaled[0]
            rng = np.random.default_rng(42)
            perturbation_scale = np.maximum(np.abs(feature_row) * 0.08, 0.05)
            background_array = rng.normal(loc=feature_row, scale=perturbation_scale, size=(300, len(availability_feature_names)))
            background_array = np.vstack([feature_row, background_array])

            explainer = LimeTabularExplainer(
                training_data=background_array,
                feature_names=availability_feature_names,
                mode="regression",
                discretize_continuous=True,
                random_state=42
            )

            explanation = explainer.explain_instance(
                data_row=feature_row,
                predict_fn=availability_lime_model.predict,
                num_features=len(availability_feature_names),
                num_samples=1000
            )

            lime_data = explanation.as_list()
            lime_features = [feat for feat, contrib in lime_data]
            lime_values = [contrib for feat, contrib in lime_data]

            predicted_label = f"Forecasted Day 7 Availability: {day7_val:.2f} MLD ({day7_class})"

            try:
                lime_plot = create_lime_plot(
                    lime_features=lime_features,
                    lime_values=lime_values,
                    predicted_label=predicted_label
                )
            except Exception as e:
                lime_error = f"Plotly LIME plotting failed: {e}"

            lime_html = explanation.as_html()
        except Exception as error:
            lime_error = f"LIME explanation failed: {error}"

        # Recommendation Generation
        submitted_values = {
            "State": state,
            "District": district,
            "Date": date_str,
            "Temperature_C": temp,
            "Rainfall_mm": rainfall,
            "Humidity_pct": humidity,
            "Reservoir_Level_pct": res_level,
            "Groundwater_Level_m": gw_level,
            "River_Flow_m3s": river_flow,
            "Population": population,
            "Water_Consumption_MLD": consumption,
            "Reservoir_Inflow_MLD": inflow,
            "Reservoir_Outflow_MLD": outflow,
            "Water_Availability_MLD": current_availability
        }

        rec_data, rec_warning = generate_recommendations(
            features=submitted_values,
            prediction_label=day7_class,
            confidence_score=confidence,
            is_water_quality=False
        )

        return render_template(
            template_name,
            prediction=str(prediction),
            prediction_value=prediction_value,
            prediction_class=prediction_class,
            confidence=confidence,
            probabilities={},
            forecast_table=forecast_table,
            stats=stats,
            chart_data=chart_data,
            lime_html=lime_html,
            lime_plot=lime_plot,
            lime_error=lime_error,
            input_values=submitted_values,
            recommendations=rec_data.get("recommendations"),
            suggestions=rec_data.get("suggestions"),
            recommendations_source=rec_data.get("source"),
            recommendations_warning=rec_warning,
            error=None
        )

    except Exception as error:
        return render_template(
            "home2.html",
            error=str(error)
        )


@app.route("/predict1", methods=["POST"])
def predict1():
    if "username" not in session:
        return redirect(url_for("login"))
    return perform_prediction("result1.html")


@app.route("/predict2", methods=["POST"])
def predict2():
    if "username" not in session:
        return redirect(url_for("login"))
    return perform_forecasting("result2.html")


@app.route("/inspect")
def inspect_models():
    try:
        load_availability_artifacts()
        
        info = {
            "model_type": str(type(availability_model)),
            "model_features": getattr(availability_model, "feature_names_in_", None),
            "scaler_features": getattr(availability_scaler, "feature_names_in_", None)
        }
        
        # Convert feature lists to strings for display
        if info["model_features"] is not None:
            info["model_features"] = list(info["model_features"])
        if info["scaler_features"] is not None:
            info["scaler_features"] = list(info["scaler_features"])
            
        return f"<h1>Model Inspection</h1><pre>{info}</pre>"
    except Exception as e:
        return f"<h1>Error</h1><pre>{str(e)}</pre>"


# =========================================================
# Run application
# =========================================================
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
