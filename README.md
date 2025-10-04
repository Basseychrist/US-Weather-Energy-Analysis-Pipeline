# Project 1: US Weather + Energy Analysis Pipeline

This project automates the process of fetching, cleaning, and analyzing weather and energy data for major US cities. It helps energy companies understand how weather impacts energy demand.

---

## Quick Start Guide (for Junior Developers)

### 1. **Set up your Python environment**

> **Why?**  
> Isolates project dependencies so you don't break your system Python.

```bash
# Create a virtual environment using uv (recommended)
uv venv
source .venv/Scripts/activate  # On Windows
# On Mac/Linux: source .venv/bin/activate
```

### 2. **Install dependencies**

> **Why?**  
> Installs all required Python packages for the pipeline and dashboard.

```bash
uv pip install -r pyproject.toml
```

### 3. **Configure API Keys**

> **Why?**  
> The pipeline needs access to NOAA (weather) and EIA (energy) APIs.

- Open `config/config.yaml`
- Add your NOAA and EIA API keys under the appropriate fields.

### 4. **Run the Data Pipeline**

> **Why?**  
> Fetches and processes historical or daily weather/energy data for all cities.

```bash
# For a historical load (last 180 days)
python main.py historical
```

- **Business Rule:**
  - Historical mode fetches and processes the last 180 days for all cities.
  - Forecast mode fetches only yesterday's data.

### 5. **Launch the Dashboard**

> **Why?**  
> Visualizes the processed data and provides interactive analysis.

```bash
streamlit run dashboards/app.py
```

- Open the link shown in your terminal to view the dashboard in your browser.

---

## Business Logic & Complex Rules

- **Weather Data Conversion:**  
  NOAA API returns temperatures in degrees Celsius.  
  The pipeline converts these to Fahrenheit using:  
  `F = (C * 9/5) + 32`

- **Data Quality:**  
  The pipeline checks for missing values, outliers, and data freshness.  
  Outliers are flagged if temperatures are outside expected ranges.

- **Date Range Handling:**  
  The pipeline ensures every day in the requested range is present in the output, even if some days are missing from the raw data.

- **Correlation Analysis Warning:**  
  If the selected date range in the dashboard has less than 20°F temperature variation, a warning is shown.

  > **Tip:** Select a wider date range for meaningful analysis.

- **Heatmap Binning:**  
  The dashboard bins average daily temperatures into ranges (e.g., `<50°F`, `50-60°F`, etc.) for usage pattern analysis.

---

## Troubleshooting

- **API errors:**  
  Check your API keys and internet connection.
- **Missing data:**  
  Some days may be missing if the API did not return data.
- **Temperature values look wrong:**  
  Make sure the conversion from Celsius to Fahrenheit is applied (see above).
- **Dashboard shows warning:**  
  Widen your date range in the dashboard filters.

---

## How to Run Tests

> **Why?**  
> Automated tests help ensure your pipeline and data processing are working correctly.

1. Make sure you have installed all dependencies (including `pytest`).
2. From your project root directory, run:

```bash
pytest tests/
```

- This will automatically discover and run all test files in the `tests` folder.
- Do **not** run test files directly with `python tests/test_pipeline.py`.

## Data Quality Report (in-app)

The Streamlit app includes an automated "Data Quality Report" (toggle in the sidebar). It runs these checks:

- Missing Values

  - What: Counts nulls per column (temp_max_f, temp_min_f, energy_demand_gwh).
  - Why: Missing data can bias analyses, break plots, and hide trends. The report shows where and how many missing values exist and lists example rows.

- Temperature Outliers

  - What: Flags daily records where temp_max_f > 130°F or temp_min_f < -50°F.
  - Why: These readings are usually erroneous (sensor/ingestion issues). Outliers can distort averages and correlations.

- Negative Energy Consumption

  - What: Flags rows where energy_demand_gwh < 0.
  - Why: Energy demand should be non-negative; negative values indicate unit or ingestion errors.

- Data Freshness
  - What: Shows the most recent date in the dataset and how many days have passed since then.
  - Why: Stale data can cause incorrect operational insights; a freshness flag helps detect pipeline failures.

How to use:

1. Open the dashboard: streamlit run dashboards/app.py
2. In the sidebar, select your date range and cities.
3. Check "Show Data Quality Report" to view missing-value tables, outlier counts, sample problematic rows, and a time-series chart of quality metrics.

Business rules:

- Temperature thresholds are configurable in code; adjust if you need different outlier bounds.
- The report aggregates metrics daily to help detect systemic issues over time.

## Next Steps

- Add automated tests for pipeline and dashboard.
- Integrate a formal data quality framework (e.g., Great Expectations).
- Expand dashboard with time-series forecasting.

---

## Questions?

If you get stuck, ask a senior developer or check the comments in the code for explanations of business rules and logic.
The warning is your guide to a better analysis. To resolve it and see a meaningful correlation, you should:

1.  Go to the **"Filters"** section in the sidebar of the dashboard.
2.  Use the **"Select Date Range"** calendar to choose a wider range of dates (e.g., 90 days).
3.  The correlation plot will automatically update. If the new temperature range is sufficient, the warning will disappear, and the R-squared and correlation values will become more meaningful.

This feature ensures that any conclusions drawn from the chart are based on a statistically sound sample of data.

## Auto-refresh behavior (app runs pipeline)

- On startup, the Streamlit app checks data/processed/weather_energy_data.csv.
  - If the file is missing or older than 24 hours the app will automatically run the pipeline:
    python main.py realtime
  - Auto-run can be disabled in the app sidebar ("Auto-run pipeline on startup").
- You can force a pipeline run from the dashboard sidebar using "Run pipeline now".
- If automatic runs fail, the UI will display a warning with basic error information.
- Note: main.py must exist at the project root and implement a 'realtime' mode that creates/updates data/processed/weather_energy_data.csv.
- If your pipeline requires secrets (API keys), configure them via Streamlit Cloud Secrets before deploying.

## Running Historical Loads from the Dashboard

You can trigger a historical full load (equivalent to `python main.py historical`) directly from the Streamlit dashboard:

- Open the dashboard: `streamlit run dashboards/app.py`
- In the sidebar find "Pipeline / Data Refresh"
- Click "Run historical load (full)"

Notes and caveats:

- Historical loads can be long-running (minutes to hours). The dashboard will run the pipeline synchronously in the current session — do not close the browser or stop the Streamlit process while it runs.
- On Streamlit Cloud, ensure secrets (NOAA/EIA API keys) are configured in the app's Secrets so main.py can access them.
- If your pipeline is heavy or requires more time than allowed by the environment, prefer running `python main.py historical` outside the dashboard (CI job or server) and let the dashboard read the produced CSV.
- The dashboard exposes both a realtime/manual small-run button and the historical full-load button; use the latter only when you need to reprocess the full historical window.

## Auto-run historical on startup

You can configure the dashboard to automatically run the full historical pipeline when the app opens.

- Enable via the dashboard:

  - Open the app and check "Auto-run historical on startup (full reprocess)" in the sidebar.

- Enable via environment variable (useful for Streamlit Cloud):
  - Set AUTO_RUN_HISTORICAL=true in your environment or Streamlit Cloud Secrets.
  - On Streamlit Cloud: add a secret named AUTO_RUN_HISTORICAL with value true.

Notes and caveats:

- Historical runs can be long-running (minutes to hours). Keep the session open while it runs.
- On Streamlit Cloud ensure NOAA/EIA API keys are available via Secrets so the pipeline can access them.
- If you prefer not to run heavy jobs inside the dashboard, run locally or via CI: `python main.py historical`.
