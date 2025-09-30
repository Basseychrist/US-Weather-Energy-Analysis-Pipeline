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

## Next Steps

- Integrate a formal data quality framework like Great Expectations.
- Add unit and integration tests for the pipeline.
- Expand the dashboard with more advanced analytics (e.g., time-series forecasting).
