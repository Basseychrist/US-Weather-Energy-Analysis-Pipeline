# Project 1: US Weather + Energy Analysis Pipeline

This project demonstrates a production-ready data pipeline and analysis dashboard that explores the relationship between weather patterns and energy consumption in major US cities.

## Business Context

Energy companies can significantly improve demand forecasting by incorporating weather data. This pipeline automates the process of fetching, cleaning, and analyzing weather and energy data to uncover critical patterns, helping to optimize power generation and reduce costs.

## Features

- **Automated Data Pipeline**: Daily and historical data fetching from NOAA (weather) and EIA (energy) APIs.
- **Robust Error Handling**: The pipeline logs errors and continues processing available data.
- **Data Quality Checks**: Automated checks for missing values, outliers, and data freshness.
- **Interactive Dashboard**: A Streamlit application for visualizing the data with four key analyses.
- **Modular & Configurable**: Code is organized into modules with a central `config.yaml` for easy management.

## Repository Structure

project1-energy-analysis/
├── README.md # This file
├── pyproject.toml # Project dependencies
├── config/
│ └── config.yaml # API keys, cities list, paths
├── src/ # Source code for the data pipeline
│ ├── data_fetcher.py # API interaction
│ ├── data_processor.py # Data cleaning and quality checks
│ ├── analysis.py # Functions for statistical analysis
│ └── pipeline.py # Main pipeline orchestration script
├── dashboards/
│ └── app.py # Streamlit dashboard application
└── data/
├── raw/ # Raw JSON data from APIs
└── processed/ # Clean, analysis-ready CSV data

## Step1: Set up a virtual environment (recommended): Using uv:

- **Bash**
  uv venv
  source .venv/Scripts/activate

## Step2: To install dependencies:

- **Bash**
  uv pip install -r pyproject.toml

## Step3: Configure API Keys:

Open config/config.yaml and add your personal API keys for NOAA and EIA.

## How to Run

### Prerequisites

- Python 3.8+
- An environment with the packages from `pyproject.toml` installed.

### Running the Pipeline

To run the data pipeline, use the `main.py` script from the root directory:

```bash
# For a historical load (last 90 days)
python main.py historical

# For a daily update (yesterday's data)
python main.py forecast
```

### Viewing the Dashboard

To launch the Streamlit dashboard, run the following command from the root directory:

```bash
streamlit run dashboards/app.py
```

## Next Steps

- Integrate a formal data quality framework like Great Expectations.
- Add unit and integration tests for the pipeline.
- Expand the dashboard with more advanced analytics (e.g., time-series forecasting).
