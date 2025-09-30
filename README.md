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
# For a historical load (last 180 days)
python main.py historical

# For a daily update (yesterday's data)
python main.py forecast
```

### Viewing the Dashboard

To launch the Streamlit dashboard, run the following command from the root directory:

```bash
streamlit run dashboards/app.py
```

## Analysis Insights

The visualizations provided in the dashboard reveal several key patterns in the relationship between weather and energy consumption.

### Heatmap Analysis (Usage Patterns)

The heatmap breaks down average energy usage by temperature range and day of the week. For a city like New York, the following insights can be drawn:

- **Temperature Impact**: As temperatures rise, energy consumption consistently increases. This is shown by the colors shifting from blue (low usage) at the bottom of the chart to red (high usage) at the top. This strong positive correlation is typical for locations with high air conditioning usage during warmer weather.

- **Weekly Cycle**: There is a clear difference in energy consumption between weekdays and weekends. Usage is significantly lower on Saturdays and Sundays (darker blue cells) compared to weekdays (Monday-Friday). This pattern reflects reduced commercial and industrial activity over the weekend.

- **Peak Demand**: The highest energy demand occurs when high temperatures coincide with a weekday. For example, the top-left cells (hot weekdays) are the reddest, indicating peak consumption.

- **Missing Data (`NaN`)**: The white cells marked `NaN` indicate that for the selected time period, there were no recorded days with that specific combination of temperature and day of the week.

**In summary, the heatmap clearly shows that energy demand in New York is driven by both weather (temperature) and economic activity (weekday vs. weekend).**

### Correlation Analysis (Temperature vs. Energy Usage)

The correlation analysis provides insights into how temperature changes are related to energy usage variations. Key points include:

- **Positive Correlation**: There is a noticeable positive correlation between temperature and energy usage. As temperatures increase, energy usage tends to increase as well, likely due to higher air conditioning demand.

- **R-squared Value**: The R-squared value of the linear regression line indicates a strong fit, meaning temperature is a good predictor of energy usage in this dataset.

- **Outliers**: Some outliers exist where energy usage is higher or lower than expected for a given temperature. These may warrant further investigation to understand underlying causes.

- **Data Distribution**: The scatter plot shows the distribution of energy usage values at different temperatures, with a clear trend of increasing usage with temperature.

**Conclusion**: The correlation analysis confirms that temperature significantly impacts energy usage, validating the need for weather data in energy demand forecasting.

### Intelligent Warning System

To prevent misinterpretation, the dashboard now includes an intelligent warning system. If you select a date range where the temperature variation is too small (e.g., less than 20°F), a warning message will appear above the correlation plot, as you have seen.

**What to do when you see the warning:**

The warning is your guide to a better analysis. To resolve it and see a meaningful correlation, you should:

1.  Go to the **"Filters"** section in the sidebar of the dashboard.
2.  Use the **"Select Date Range"** calendar to choose a wider range of dates (e.g., 90 days).
3.  The correlation plot will automatically update. If the new temperature range is sufficient, the warning will disappear, and the R-squared and correlation values will become more meaningful.

This feature ensures that any conclusions drawn from the chart are based on a statistically sound sample of data.

## Next Steps

- Integrate a formal data quality framework like Great Expectations.
- Add unit and integration tests for the pipeline.
- Expand the dashboard with more advanced analytics (e.g., time-series forecasting).
