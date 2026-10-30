import numpy as np
import pandas as pd

def validate_inputs(user_input):
    """
    Validates form inputs for water availability forecasting.
    Raises ValueError with a friendly error message if validation fails.
    """
    required_fields = [
        "State", "District", "Date", "Temperature_C", "Rainfall_mm", "Humidity_pct",
        "Reservoir_Level_pct", "Groundwater_Level_m", "River_Flow_m3s", "Population",
        "Water_Consumption_MLD", "Reservoir_Inflow_MLD", "Reservoir_Outflow_MLD",
        "Water_Availability_MLD"
    ]
    for field in required_fields:
        if field not in user_input or user_input[field] is None or str(user_input[field]).strip() == "":
            raise ValueError(f"Please enter a value for {field.replace('_', ' ')}.")

    # Validate numbers
    numeric_fields = [
        "Temperature_C", "Rainfall_mm", "Humidity_pct", "Reservoir_Level_pct",
        "Groundwater_Level_m", "River_Flow_m3s", "Population", "Water_Consumption_MLD",
        "Reservoir_Inflow_MLD", "Reservoir_Outflow_MLD", "Water_Availability_MLD"
    ]
    for field in numeric_fields:
        try:
            val = float(user_input[field])
        except (ValueError, TypeError):
            raise ValueError(f"{field.replace('_', ' ')} must be a valid number.")
            
        # Non-negative checks
        if field in ["Rainfall_mm", "Groundwater_Level_m", "River_Flow_m3s", "Population",
                    "Water_Consumption_MLD", "Reservoir_Inflow_MLD", "Reservoir_Outflow_MLD"]:
            if val < 0:
                raise ValueError(f"{field.replace('_', ' ')} cannot be negative.")
        
        # Strictly positive checks
        if field == "Water_Availability_MLD":
            if val <= 0:
                raise ValueError("Current Water Availability must be a strictly positive number.")
                
        # Percentage checks
        if field in ["Humidity_pct", "Reservoir_Level_pct"]:
            if not (0 <= val <= 100):
                raise ValueError(f"{field.replace('_', ' ')} must be between 0 and 100.")

def build_model_features(user_input, historical_data):
    """
    Validates inputs, appends the user input row to the historical dataset,
    and engineers all lag, moving average, RSI, and time features.
    """
    validate_inputs(user_input)

    state = user_input.get("State", "").strip()
    district = user_input.get("District", "").strip()
    date_str = user_input.get("Date", "").strip()
    
    try:
        target_date = pd.to_datetime(date_str)
    except Exception:
        raise ValueError("Invalid Date format. Use YYYY-MM-DD.")
        
    # Filter the dataset for the selected State and District, before target_date
    df_filtered = historical_data[
        (historical_data["State"] == state) &
        (historical_data["District"] == district) &
        (historical_data["Date"] < target_date)
    ].sort_values("Date").reset_index(drop=True)
    
    if len(df_filtered) < 35:
        raise ValueError(
            f"Insufficient historical data before {date_str}. "
            f"At least 35 days of historical data are required to generate forecasts. "
            f"Only {len(df_filtered)} records found."
        )
        
    # Take the last 35 rows of history
    df_history = df_filtered.tail(35).copy()
    
    # Re-assign dates consecutively ending at target_date - 1 day
    history_dates = [target_date - pd.Timedelta(days=i) for i in range(35, 0, -1)]
    df_history["Date"] = history_dates
    
    # Construct today's row
    new_row = {
        "Date": target_date,
        "State": state,
        "District": district,
        "Month": float(target_date.month),
        "DayOfYear": float(target_date.timetuple().tm_yday),
        "Temperature_C": float(user_input["Temperature_C"]),
        "Rainfall_mm": float(user_input["Rainfall_mm"]),
        "Humidity_pct": float(user_input["Humidity_pct"]),
        "Reservoir_Level_pct": float(user_input["Reservoir_Level_pct"]),
        "Groundwater_Level_m": float(user_input["Groundwater_Level_m"]),
        "River_Flow_m3s": float(user_input["River_Flow_m3s"]),
        "Population": float(user_input["Population"]),
        "Water_Consumption_MLD": float(user_input["Water_Consumption_MLD"]),
        "Reservoir_Inflow_MLD": float(user_input["Reservoir_Inflow_MLD"]),
        "Reservoir_Outflow_MLD": float(user_input["Reservoir_Outflow_MLD"]),
        "Water_Availability_MLD": float(user_input["Water_Availability_MLD"]),
        "Water_Availability": "Medium"
    }
    
    # Combine history and today's row
    df_combined = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)
    df_combined = df_combined.sort_values("Date").reset_index(drop=True)
    
    # Today's row will be the last row (index 35)
    idx = 35
    
    # Feature engineering
    df_model_input = df_combined.copy()
    
    # Lags
    for lag in [1, 3, 7, 14, 15]:
        df_model_input[f"Water_lag_{lag}"] = df_model_input["Water_Availability_MLD"].shift(lag)
        
    # Moving Averages
    for w in [7, 14, 15, 30]:
        df_model_input[f"Water_ma_{w}"] = df_model_input["Water_Availability_MLD"].rolling(w).mean()
        
    # RSI
    delta = df_model_input["Water_Availability_MLD"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df_model_input["RSI"] = 100 - (100 / (1 + rs))
    
    # log_water_change
    df_model_input["log_water_change"] = np.log(df_model_input["Water_Availability_MLD"] / df_model_input["Water_lag_1"])
    
    return df_model_input, idx, df_combined

def preprocess_for_prediction(user_input, historical_data, scaler):
    """
    Fully prepares and scales the feature dataframe for forecasting.
    Returns the scaled features, target row index, and base filtered dataframe.
    """
    df_model_input, idx, df_filtered = build_model_features(user_input, historical_data)
    
    availability_feature_names = [
        'Month', 'DayOfYear', 'Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Reservoir_Level_pct',
        'Groundwater_Level_m', 'River_Flow_m3s', 'Population', 'Water_Consumption_MLD',
        'Reservoir_Inflow_MLD', 'Reservoir_Outflow_MLD', 'Water_Availability_MLD', 'log_water_change',
        'Water_lag_1', 'Water_lag_3', 'Water_lag_7', 'Water_lag_14', 'Water_lag_15',
        'Water_ma_7', 'Water_ma_14', 'Water_ma_15', 'Water_ma_30', 'RSI'
    ]
    
    # Ensure all required features are selected in correct order
    df_features = df_model_input[availability_feature_names].copy()
    
    # Scale features
    scaled_values = scaler.transform(df_features)
    df_scaled = pd.DataFrame(scaled_values, columns=availability_feature_names, index=df_features.index)
    
    return df_scaled, idx, df_filtered, df_model_input
