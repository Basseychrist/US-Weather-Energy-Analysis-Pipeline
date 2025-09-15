import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd
import os

from .data_fetcher import fetch_weather_data, fetch_energy_data
from .data_processor import process_weather_data, process_energy_data, run_quality_checks

def setup_logging(log_path):
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def run_pipeline(mode):
    """Runs the full data pipeline."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config['paths']['log_file'])
    
    if mode == 'historical':
        # Fetch last 90 days
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=90)
        logging.info("Running HISTORICAL data load for the last 90 days up to yesterday.")
    else: # forecast
        # Fetch yesterday's data (for daily run)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date
        logging.info("Running DAILY data load for yesterday.")

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    all_weather_dfs = []
    for city in config['cities']:
        weather_data = fetch_weather_data(config, city, start_date_str, end_date_str)
        if weather_data:
            weather_df = process_weather_data(weather_data, city['name'])
            all_weather_dfs.append(weather_df)

    all_energy_dfs = []
    for city in config['cities']:
        energy_data = fetch_energy_data(config, city, start_date_str, end_date_str)
        if energy_data:
            energy_df = process_energy_data(energy_data, city['name'])
            all_energy_dfs.append(energy_df)
    
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

