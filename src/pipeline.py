import yaml
import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

from .data_fetcher import fetch_weather_data, fetch_energy_data
from .data_processor import process_weather_data, process_energy_data, run_quality_checks

def run_pipeline(mode):
    """Runs the full data pipeline."""
    # Logging is now configured in main.py
    logging.info("--- Pipeline run started. ---")

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if mode == 'historical':
        # Fetch exactly 180 days of data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=179)  # Adjust to ensure 180 days inclusive
        logging.info(f"Running HISTORICAL data load from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
    else: # forecast
        # Fetch yesterday's data (for daily run)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date
        logging.info(f"Running DAILY data load for {start_date.strftime('%Y-%m-%d')}.")

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    all_weather_dfs = []
    for city in config['cities']:
        logging.info(f"Fetching weather data for {city['name']} from {start_date_str} to {end_date_str}")
        weather_data = fetch_weather_data(config, city, start_date_str, end_date_str)
        if weather_data:
            logging.info(f"Successfully fetched weather data for {city['name']}.")
            # Pass raw weather data and city metadata to process_weather_data
            weather_df = process_weather_data(weather_data, city['name'], start_date_str, end_date_str)
            # Ensure the processed DataFrame covers the full date range
            expected_days = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
            missing_days = set(expected_days) - set(weather_df['date'])
            if missing_days:
                logging.warning(f"{city['name']} missing {len(missing_days)} days in weather data.")
            all_weather_dfs.append(weather_df)
        else:
            logging.warning(f"No weather data found for {city['name']}.")

    all_energy_dfs = []
    for city in config['cities']:
        logging.info(f"Fetching energy data for {city['name']} from {start_date_str} to {end_date_str}")
        energy_data = fetch_energy_data(config, city, start_date_str, end_date_str)
        if energy_data:
            logging.info(f"Successfully fetched energy data for {city['name']}.")
            # Pass raw energy data and city metadata to process_energy_data
            energy_df = process_energy_data(energy_data, city['name'], start_date_str, end_date_str)
            # Ensure the processed DataFrame covers the full date range
            expected_days = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
            missing_days = set(expected_days) - set(energy_df['date'])
            if missing_days:
                logging.warning(f"{city['name']} missing {len(missing_days)} days in energy data.")
            all_energy_dfs.append(energy_df)
        else:
            logging.warning(f"No energy data found for {city['name']}.")

    if not all_weather_dfs:
        logging.error("No weather data fetched. Halting pipeline.")
        return
        
    if not all_energy_dfs:
        logging.warning("No energy data fetched. Proceeding with weather data only.")
        # Combine and save only weather data
        combined_weather = pd.concat(all_weather_dfs, ignore_index=True)
        output_path = os.path.join(config['paths']['processed_data'], 'weather_data.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_weather.to_csv(output_path, index=False)
        logging.info(f"Pipeline finished. Weather data saved to {output_path}")
        return

    # Combine data from all cities
    combined_weather = pd.concat(all_weather_dfs, ignore_index=True)
    combined_energy = pd.concat(all_energy_dfs, ignore_index=True)

    # Merge weather and energy data
    final_df = pd.merge(combined_weather, combined_energy, on=['date', 'city'], how='inner')
    
    # Add average temperature
    final_df['temp_avg_f'] = (final_df['temp_max_f'] + final_df['temp_min_f']) / 2

    # Run data quality checks
    quality_report = run_quality_checks(final_df, config)
    logging.info(f"Final Data Quality Report: {quality_report}")

    # Save processed data
    output_path = os.path.join(config['paths']['processed_data'], 'weather_energy_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Pipeline finished. Processed data saved to {output_path}")
    # Add average temperature
    final_df['temp_avg_f'] = (final_df['temp_max_f'] + final_df['temp_min_f']) / 2

    # Run data quality checks
    quality_report = run_quality_checks(final_df, config)
    logging.info(f"Final Data Quality Report: {quality_report}")

    # Save processed data
    output_path = os.path.join(config['paths']['processed_data'], 'weather_energy_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Pipeline finished. Processed data saved to {output_path}")

