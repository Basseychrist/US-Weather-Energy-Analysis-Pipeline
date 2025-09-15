import requests
import logging
from datetime import datetime
import json
import os
import time

def _fetch_with_retries(url, params, headers, max_retries=3, backoff_factor=2):
    """Generic fetch function with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Request failed: {e}. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed after {max_retries} retries: {e}")
                return None

def fetch_weather_data(config, city, start_date, end_date):
    """Fetches daily temperature data (TMAX, TMIN) from the NOAA API."""
    logging.info(f"Fetching weather data for {city['name']} from {start_date} to {end_date}")
    headers = {'token': config['noaa']['token']}
    params = {
        'datasetid': 'GHCND',
        'stationid': city['noaa_station_id'],
        'startdate': start_date,
        'enddate': end_date,
        'datatypeid': 'TMAX,TMIN',
        'limit': 1000,
        'units': 'metric' # Fetch in Celsius, convert later
    }
    
    data = _fetch_with_retries(config['noaa']['base_url'], params, headers)

    if data:
        # Save raw data
        raw_path = os.path.join(config['paths']['raw_data'], f"weather_{city['name']}_{start_date}_{end_date}.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, 'w') as f:
            json.dump(data, f)
        logging.info(f"Successfully fetched weather data for {city['name']}.")
    return data

def fetch_energy_data(config, city, start_date, end_date):
    """Fetches daily energy consumption data from the EIA API."""
    logging.info(f"Fetching energy data for {city['eia_region_code']} from {start_date} to {end_date}")
    
    url = f"{config['eia']['base_url']}"
    params = {
        "api_key": config['eia']['api_key'],
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": city['eia_region_code'],
        "start": start_date,
        "end": end_date,
    }
    headers = {'Accept': 'application/json'}

    data = _fetch_with_retries(url, params, headers)

    if data:
        # Save raw data
        raw_path = os.path.join(config['paths']['raw_data'], f"energy_{city['name']}_{start_date}_{end_date}.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, 'w') as f:
            json.dump(data, f)
        logging.info(f"Successfully fetched energy data for {city['eia_region_code']}.")
    return data