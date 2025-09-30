import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data_processor import celsius_to_fahrenheit, process_weather_data, run_quality_checks

def test_celsius_to_fahrenheit():
    # Test known conversion
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212

def test_process_weather_data_conversion():
    # Simulate NOAA raw data in Celsius
    raw_data = {
        "results": [
            {"date": "2025-07-29T00:00:00", "datatype": "TMAX", "value": 36.1},
            {"date": "2025-07-29T00:00:00", "datatype": "TMIN", "value": 25.6}
        ]
    }
    df = process_weather_data(raw_data, "New York", "2025-07-29", "2025-07-29")
    # Check conversion
    assert abs(df.loc[0, "temp_max_f"] - 97) < 1
    assert abs(df.loc[0, "temp_min_f"] - 78) < 1
    assert "temp_avg_f" in df.columns

def test_run_quality_checks_missing_and_outliers():
    # Create a DataFrame with missing and outlier values
    data = {
        "date": [pd.Timestamp("2025-07-29")],
        "temp_max_f": [200],  # Outlier
        "temp_min_f": [-50],  # Outlier
        "city": ["New York"],
        "temp_avg_f": [75],
        "energy_demand_gwh": [1000]
    }
    df = pd.DataFrame(data)
    config = {
        "data_quality": {
            "temp_outlier_fahrenheit": {"max": 130, "min": -30}
        }
    }
    report = run_quality_checks(df, config)
    assert report["temp_outliers_count"] == 1
    assert report["missing_values"]["temp_max_f"] == 0