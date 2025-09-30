import requests
import logging
import time
import math
import json
import os

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
    """Fetches weather data from the NOAA API."""
    logging.info(f"Fetching weather data for {city['name']} from {start_date} to {end_date}")
    api_key = config['noaa']['token']
    base_url = config['noaa']['base_url']
    station_id = city.get('noaa_station_id')

    if not station_id:
        logging.warning(f"No NOAA station ID for {city['name']}.")
        return None

    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": ["TMAX", "TMIN"],
        "limit": 1000,
        "units": "metric"
    }
    headers = {"token": api_key}

    data = _fetch_with_retries(base_url, params, headers)

    if data:
        # Save raw data
        raw_path = os.path.join(config['paths']['raw_data'], f"weather_{city['name']}_{start_date}_{end_date}.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, 'w') as f:
            json.dump(data, f)
        logging.info(f"Successfully fetched weather data for {city['name']}.")
    return data

def fetch_energy_data(config, city, start_date, end_date):
    """
    Fetches energy data from the EIA API, handling pagination to retrieve all data.
    """
    api_key = config['eia']['api_key']
    base_url = config['eia']['base_url']
    region_code = city.get('eia_region_code')

    if not region_code:
        logging.warning(f"No EIA region code configured for city: {city['name']}")
        return None

    all_data = []
    limit = 5000  # EIA API row limit

    # Initial request to get the total count
    initial_params = {
        'api_key': api_key,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': region_code,
        'start': start_date,
        'end': end_date,
        'length': 0
    }

    try:
        response = requests.get(base_url, params=initial_params, timeout=30)
        response.raise_for_status()
        json_response = response.json()
        total_rows = int(json_response.get('response', {}).get('total', 0))

        if total_rows == 0:
            logging.warning(f"No energy data found for {region_code} in the specified date range.")
            return None

        # Loop through all pages
        for i in range(math.ceil(total_rows / limit)):
            offset = i * limit
            params = {
                'api_key': api_key,
                'frequency': 'hourly',
                'data[0]': 'value',
                'facets[respondent][]': region_code,
                'start': start_date,
                'end': end_date,
                'sort[0][column]': 'period',
                'sort[0][direction]': 'asc',
                'offset': offset,
                'length': limit
            }
            
            page_response = requests.get(base_url, params=params, timeout=30)
            page_response.raise_for_status()
            page_json = page_response.json()

            if 'response' in page_json and 'data' in page_json['response']:
                data = page_json['response']['data']
                if data:
                    all_data.extend(data)
            else:
                logging.warning(f"EIA API response for page {i+1} did not contain 'data'.")
                break
        
        return {'response': {'data': all_data}}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching EIA data for {region_code}: {e}")
        return None