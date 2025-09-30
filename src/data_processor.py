import os
import sys
import pandas as pd
import logging

def celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def ensure_complete_date_range(df, start_date, end_date, city_name):
    """Ensures the DataFrame covers the complete date range."""
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Drop duplicate dates to ensure unique index before reindexing
    if not df.empty:
        df = df.drop_duplicates(subset='date', keep='first')
    
    # Reindex to include all dates in the range
    df = df.set_index('date').reindex(all_dates).reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    df['city'] = city_name
    return df

def process_weather_data(raw_data, city_name, start_date_str, end_date_str):
    """Processes raw NOAA weather data into a clean DataFrame."""
    if not raw_data or 'results' not in raw_data:
        logging.warning(f"No weather data to process for {city_name}.")
        return pd.DataFrame()

    records = []
    for item in raw_data['results']:
        date = item['date'].split('T')[0]
        datatype = item['datatype']
        # Value is in tenths of a degree C, convert to C then F
        value_c = item['value'] / 10
        value_f = celsius_to_fahrenheit(value_c)
        records.append({'date': date, 'datatype': datatype, 'value': value_f})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # A more robust way to pivot the data using groupby and unstack
    # This handles potential duplicate entries for a date/datatype pair
    df = df.groupby(['date', 'datatype'])['value'].mean().unstack().reset_index()
    df.columns.name = None # Remove the name of the columns index
    
    df.rename(columns={'TMAX': 'temp_max_f', 'TMIN': 'temp_min_f'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['city'] = city_name

    # Ensure complete date range using passed dates
    df = ensure_complete_date_range(df, start_date_str, end_date_str, city_name)
    return df

def process_energy_data(raw_data, city_name, start_date_str, end_date_str):
    """Processes raw EIA energy data into a clean DataFrame."""
    if not raw_data or 'response' not in raw_data or 'data' not in raw_data['response']:
        logging.warning(f"No energy data to process for {city_name}.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data['response']['data'])
    if df.empty:
        return df

    df.rename(columns={'period': 'date', 'value': 'energy_demand_gwh'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['energy_demand_gwh'] = pd.to_numeric(df['energy_demand_gwh'], errors='coerce')
    
    # Resample hourly data to daily sums and keep only the date part
    df = df.set_index('date').resample('D').sum().reset_index()
    
    df['city'] = city_name

    # Ensure complete date range using passed dates
    df = ensure_complete_date_range(df, start_date_str, end_date_str, city_name)
    return df[['date', 'city', 'energy_demand_gwh']]

def run_quality_checks(df, config):
    """Runs data quality checks and returns a report."""
    report = {}
    
    # 1. Missing values
    report['missing_values'] = df.isnull().sum().to_dict()

    # 2. Outliers
    max_temp = config['data_quality']['temp_outlier_fahrenheit']['max']
    min_temp = config['data_quality']['temp_outlier_fahrenheit']['min']
    temp_outliers = df[(df['temp_max_f'] > max_temp) | (df['temp_min_f'] < min_temp)]
    energy_outliers = df[df['energy_demand_gwh'] < 0]
    report['temp_outliers_count'] = len(temp_outliers)
    report['negative_energy_count'] = len(energy_outliers)

    # 3. Data freshness
    if not df.empty:
        latest_date = df['date'].max()
        report['latest_data_date'] = latest_date.strftime('%Y-%m-%d')
        report['days_since_latest_data'] = (pd.Timestamp.now() - latest_date).days
    else:
        report['latest_data_date'] = 'N/A'
        report['days_since_latest_data'] = 'N/A'
        
    logging.info(f"Data Quality Report: {report}")
    return report