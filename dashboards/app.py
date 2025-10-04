import subprocess
import time
import os
import sys
import streamlit as st
import pandas as pd
import yaml

# --- Add missing visualization imports ---
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root and src folder to sys.path so "import src.*" works reliably
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
for _p in (project_root, src_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Try to import analysis/data helpers from src, provide safe fallbacks if missing ---
try:
    from src.analysis import get_correlation_stats, prepare_heatmap_data
    from src.data_processor import run_quality_checks
except Exception as e:
    # Fallback implementations so the app doesn't crash if src.* is unavailable
    def get_correlation_stats(df):
        """
        Fallback: compute Pearson correlation, R^2 and linear fit.
        Returns: correlation, r_squared, (slope, intercept), plot_df
        """
        import numpy as np
        if df is None or df.empty:
            return 0.0, 0.0, (0.0, 0.0), pd.DataFrame()
        x = pd.to_numeric(df.get('temp_avg_f', pd.Series(dtype=float)), errors='coerce')
        y = pd.to_numeric(df.get('energy_demand_gwh', pd.Series(dtype=float)), errors='coerce')
        valid = x.notnull() & y.notnull()
        if valid.sum() < 2:
            return 0.0, 0.0, (0.0, 0.0), df.copy()
        x = x[valid].astype(float)
        y = y[valid].astype(float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            correlation = float(x.corr(y))
        except Exception:
            slope, intercept, r_squared, correlation = 0.0, 0.0, 0.0, 0.0
        return correlation, r_squared, (float(slope), float(intercept)), df.copy()

    def prepare_heatmap_data(df):
        """
        Fallback: pivot average energy_demand_gwh by coarse temp bins and weekday.
        """
        import numpy as np
        if df is None or df.empty:
            return pd.DataFrame()
        df2 = df.copy()
        bins = [0, 50, 60, 70, 80, 90, np.inf]
        labels = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        df2['temp_range'] = pd.cut(df2.get('temp_avg_f', pd.Series()), bins=bins, labels=labels, right=False)
        df2['day_of_week'] = df2['date'].dt.day_name()
        pivot = df2.groupby(['temp_range', 'day_of_week'])['energy_demand_gwh'].mean().unstack(fill_value=0)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = pivot.reindex(columns=[c for c in day_order if c in pivot.columns])
        return pivot

    def run_quality_checks(df, config):
        """
        Fallback: basic quality report with missing values, outlier and freshness counts.
        """
        report = {}
        if df is None or df.empty:
            report['missing_values'] = {}
            report['temp_outliers_count'] = 0
            report['negative_energy_count'] = 0
            report['latest_data_date'] = 'N/A'
            report['days_since_latest_data'] = 'N/A'
            return report
        report['missing_values'] = df.isnull().sum().to_dict()
        max_temp = config.get('data_quality', {}).get('temp_outlier_fahrenheit', {}).get('max', 130) if config else 130
        min_temp = config.get('data_quality', {}).get('temp_outlier_fahrenheit', {}).get('min', -50) if config else -50
        temp_outliers = df[(df.get('temp_max_f', pd.Series()) > max_temp) | (df.get('temp_min_f', pd.Series()) < min_temp)]
        energy_outliers = df[df.get('energy_demand_gwh', pd.Series()) < 0]
        report['temp_outliers_count'] = len(temp_outliers)
        report['negative_energy_count'] = len(energy_outliers)
        latest_date = df['date'].max() if 'date' in df.columns else None
        report['latest_data_date'] = latest_date.strftime('%Y-%m-%d') if pd.notnull(latest_date) else 'N/A'
        report['days_since_latest_data'] = (pd.Timestamp.now() - latest_date).days if pd.notnull(latest_date) else 'N/A'
        return report

# --- Pipeline runner helpers (works locally & on Streamlit Cloud) ---
def run_pipeline(mode='realtime', timeout_seconds=600):
    """
    Run main.py with the given mode using same Python executable.
    Returns dict with keys: ran (bool), stdout, stderr, reason.
    """
    main_py = os.path.join(project_root, 'main.py')
    if not os.path.exists(main_py):
        return {'ran': False, 'reason': f"main.py not found at {main_py}"}
    cmd = [sys.executable, main_py, mode]
    try:
        proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=timeout_seconds)
        if proc.returncode == 0:
            return {'ran': True, 'stdout': proc.stdout}
        else:
            return {'ran': False, 'stderr': proc.stderr or proc.stdout, 'returncode': proc.returncode}
    except subprocess.TimeoutExpired as te:
        return {'ran': False, 'reason': 'timeout', 'detail': str(te)}
    except Exception as e:
        return {'ran': False, 'reason': 'exception', 'detail': str(e)}

def run_pipeline_if_needed(processed_rel_path='data/processed/weather_energy_data.csv',
                           max_age_hours=24,
                           force=False,
                           mode='realtime'):
    """
    Check processed CSV; run pipeline if missing/stale or if force=True.
    Records result in st.session_state for cache busting.
    """
    processed_path = os.path.join(project_root, processed_rel_path)
    needs_run = force or (not os.path.exists(processed_path))
    if not needs_run:
        try:
            age_hours = (time.time() - os.path.getmtime(processed_path)) / 3600.0
            if age_hours > max_age_hours:
                needs_run = True
        except Exception:
            needs_run = True

    if not needs_run:
        return {'ran': False, 'reason': 'up-to-date'}

    result = run_pipeline(mode=mode)
    if result.get('ran'):
        st.session_state['_pipeline_last_run'] = time.time()
    st.session_state['_pipeline_last_result'] = result
    return result

# --- Cached loader with safe retry and regeneration attempt ---
@st.cache_data
def load_data(reload_token=None):
    """
    Load processed CSV and config. reload_token busts cache after pipeline runs.
    Returns (df, city_df, config) or (None, None, None) on failure.
    """
    processed_path = os.path.join(project_root, 'data', 'processed', 'weather_energy_data.csv')

    # If missing: attempt to run pipeline once
    if not os.path.exists(processed_path):
        # do not call st.* here (cache functions should be pure), return None to let caller run pipeline
        return None, None, None

    # If file exists but is empty, indicate failure to caller
    try:
        if os.path.getsize(processed_path) == 0:
            return None, None, None
    except Exception:
        return None, None, None

    # Try reading CSV
    try:
        df = pd.read_csv(processed_path, parse_dates=['date'])
    except Exception:
        return None, None, None

    # Load config
    try:
        with open(os.path.join(project_root, 'config', 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        city_df = pd.DataFrame(config.get('cities', []))
    except Exception:
        return None, None, None

    # Merge lat/lon if available
    if not city_df.empty and 'name' in city_df.columns:
        try:
            df = pd.merge(df, city_df[['name', 'lat', 'lon']], left_on='city', right_on='name', how='left')
        except Exception:
            pass

    return df, city_df, config

# --- Sidebar controls for auto-run and manual run (unique keys) ---
st.sidebar.header("Pipeline / Data Refresh")
st.sidebar.write("Auto-run will attempt to fetch & process latest data if missing or stale.")

# Read environment variable to allow enabling in deployed environments (Streamlit Cloud secrets)
env_auto_hist = os.getenv("AUTO_RUN_HISTORICAL", "false").lower() in ("1", "true", "yes")

# Checkbox to auto-run realtime (existing) — keep as-is
auto_run_enabled = st.sidebar.checkbox("Auto-run pipeline on startup (realtime)", value=True, key="auto_run_enabled_v3")

# NEW: checkbox to auto-run full historical load on startup (default from env)
auto_run_historical = st.sidebar.checkbox(
    "Auto-run historical on startup (full reprocess)", 
    value=env_auto_hist, 
    key="auto_run_historical_v1"
)

manual_run = st.sidebar.button("Run pipeline now", key="run_pipeline_manual_v3")
# New historical button (keep for manual trigger)
run_historical = st.sidebar.button("Run historical load (full)", key="run_historical_full_v1")

# Manual realtime run handling (existing)
if manual_run:
    with st.spinner("Running pipeline (realtime)..."):
        res = run_pipeline_if_needed(force=True) if 'run_pipeline_if_needed' in globals() else run_pipeline(mode='realtime')
    if res.get('ran'):
        st.sidebar.success("Realtime pipeline completed and data refreshed.")
        st.session_state['_pipeline_last_run'] = time.time()
    else:
        st.sidebar.error(f"Realtime pipeline failed: {res.get('reason') or res.get('stderr') or res.get('detail')}")

# Manual historical run handling (existing)
if run_historical:
    st.sidebar.warning("Historical load can be long-running. Keep this session open until it finishes.")
    with st.spinner("Running full historical pipeline (this may take a while)..."):
        # Prefer helper; fallback to run_pipeline
        try:
            res = run_pipeline_if_needed(force=True, mode='historical')
        except TypeError:
            res = run_pipeline(mode='historical', timeout_seconds=3600)
        except NameError:
            res = run_pipeline(mode='historical', timeout_seconds=3600)
    if res.get('ran'):
        st.sidebar.success("Historical pipeline completed. Processed CSV updated.")
        st.session_state['_pipeline_last_run'] = time.time()
    else:
        st.sidebar.error(f"Historical pipeline failed: {res.get('reason') or res.get('stderr') or res.get('detail')}")
        if res.get('stderr'):
            st.sidebar.text(res.get('stderr')[:200])

# NEW: Auto-run historical on startup if enabled (run once per session)
if auto_run_historical and not st.session_state.get('_auto_hist_checked'):
    st.session_state['_auto_hist_checked'] = True
    st.info("Auto-run historical is enabled — the app will run a full historical pipeline now.")
    with st.spinner("Auto-running full historical pipeline (this may take a while)..."):
        try:
            res = run_pipeline_if_needed(force=True, mode='historical')
        except TypeError:
            res = run_pipeline(mode='historical', timeout_seconds=3600)
        except NameError:
            res = run_pipeline(mode='historical', timeout_seconds=3600)
    if res.get('ran'):
        st.success("Auto historical run completed. Processed CSV updated.")
        st.session_state['_pipeline_last_run'] = time.time()
    else:
        st.error(f"Auto historical run failed: {res.get('reason') or res.get('stderr') or res.get('detail')}")

# --- Cached data loading and pipeline triggering (existing logic) ---
# --- Attempt to load data, if load_data returns None then try one regen run and retry load ---
reload_token = st.session_state.get('_pipeline_last_run', None)
df, city_df, config = load_data(reload_token)

if df is None:
    # Attempt one regeneration run (only if main.py exists)
    regen_msg = ""
    regen_result = None
    main_py = os.path.join(project_root, 'main.py')
    if os.path.exists(main_py):
        with st.spinner("Processed data missing or invalid: running pipeline to generate it..."):
            regen_result = run_pipeline(mode='realtime')
        if regen_result.get('ran'):
            st.success("Pipeline finished; attempting to load processed data.")
            st.session_state['_pipeline_last_run'] = time.time()
            # reload cached data by calling load_data with updated token
            df, city_df, config = load_data(st.session_state['_pipeline_last_run'])
        else:
            regen_msg = regen_result.get('reason') or regen_result.get('stderr') or regen_result.get('detail') or "unknown error"

    # If still no data, show actionable message and stop
    if df is None:
        st.error("Processed data not available. The app attempted to generate it but failed.")
        if regen_result:
            st.error(f"Pipeline attempt failed: {regen_msg}")
        else:
            st.info("No pipeline entrypoint (main.py) found to auto-generate data. Run pipeline locally: python main.py realtime")
        st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="US Weather & Energy Analysis",
    page_icon="⚡",
    layout="wide"
)

# --- Force light (white) theme via CSS override ---
def apply_force_light_theme():
    """
    Inject CSS to force a light/white theme for Streamlit UI.
    This complements .streamlit/config.toml and ensures the app remains white
    even if viewer/system/browser prefers dark mode.
    """
    light_theme_css = """
    <style>
    :root { color-scheme: light; }
    html, body, .stApp, .reportview-container, .main, .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"], .sidebar .css-1d391kg {
        background-color: #f6f8fb !important;
        color: #000000 !important;
    }
    /* Header / top bar */
    header, [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Buttons and primary accents */
    .stButton>button, button[kind="primary"] {
        background-color: #0055aa !important;
        color: #ffffff !important;
        border: none !important;
    }
    /* Tables and dataframes */
    .stDataFrame, .element-container, .stMarkdown {
        color: #000000 !important;
        background-color: transparent !important;
    }
    /* Ensure widgets text visible */
    .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {
        color: #000000 !important;
    }
    /* Remove any dark backgrounds applied by theme toggles */
    [data-testid="stAppViewContainer"] .main .block-container {
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(light_theme_css, unsafe_allow_html=True)

# Call the theming helper immediately so the UI renders light
apply_force_light_theme()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
    
# Date Range Selector
min_date = df['date'].min().date()
max_date = df['date'].max().date()
default_start_date = max_date - pd.Timedelta(days=90)  # Default to the last 90 days
default_start_date = max(default_start_date, min_date)  # Ensure default start date is within range
start_date, end_date = st.sidebar.date_input(
    "Select Date Range (Max: 90 days)",
    value=(default_start_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Ensure the selected range does not exceed 90 days
if (end_date - start_date).days > 90:  # Limit to 90 days
    st.sidebar.error("Please select a date range of 90 days or less.")
    st.stop()

# City Filter (Multiselect)
all_cities = df['city'].unique()
selected_cities = st.sidebar.multiselect(
    "Select Cities",
    options=all_cities,
    default=all_cities
)

# Add Data Quality toggle in sidebar
show_quality = st.sidebar.checkbox("Show Data Quality Report", value=False)

# --- Filter Data Based on Selections ---
filtered_df = df[
    (df['date'].dt.date >= start_date) &
    (df['date'].dt.date <= end_date) &
    (df['city'].isin(selected_cities))
]

# --- Main Dashboard ---
st.title("⚡ US Weather & Energy Consumption Dashboard")
st.markdown(f"Data last updated: **{max_date.strftime('%Y-%m-%d')}**")

# --- Data Quality Helpers ---
def compute_quality_timeseries(df_in):
    """Return per-day quality metrics: missing count, temp outliers, negative energy."""
    if df_in.empty:
        return pd.DataFrame(columns=['date','missing_total','temp_outliers','negative_energy'])
    d = df_in.copy()
    # Identify outliers and missing indicators
    d['temp_outlier'] = ((d['temp_max_f'] > 130) | (d['temp_min_f'] < -50)).fillna(False).astype(int)
    d['negative_energy'] = (d['energy_demand_gwh'] < 0).fillna(False).astype(int)
    d['missing_count'] = d[['temp_max_f','temp_min_f','energy_demand_gwh']].isnull().sum(axis=1)
    ts = d.groupby(d['date'].dt.date).agg(
        missing_total=('missing_count','sum'),
        temp_outliers=('temp_outlier','sum'),
        negative_energy=('negative_energy','sum')
    ).reset_index().rename(columns={'date':'date'})
    ts['date'] = pd.to_datetime(ts['date'])
    return ts

# --- Show Data Quality Report if requested ---
if show_quality:
    st.header("Data Quality Report")
    # Run single-run quality checks (summary)
    report = run_quality_checks(filtered_df, config)

    # Missing values by column
    st.subheader("Missing Values")
    missing_dict = report.get('missing_values', {})
    # Show table of missing counts (column, count)
    missing_df = pd.DataFrame(list(missing_dict.items()), columns=['column','missing_count'])
    st.table(missing_df)

    # Summary metrics
    temp_outliers_count = report.get('temp_outliers_count', 0)
    negative_energy_count = report.get('negative_energy_count', 0)
    latest_data_date = report.get('latest_data_date', 'N/A')
    days_since_latest_data = report.get('days_since_latest_data', 'N/A')

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Temp Outliers", f"{temp_outliers_count}")
    col_b.metric("Negative Energy Rows", f"{negative_energy_count}")
    col_c.metric("Latest Data Date", f"{latest_data_date}", delta=f"{days_since_latest_data} days since latest")

    # Detailed rows with examples (show a few rows that are problematic)
    st.subheader("Example Problematic Rows")
    # show rows with any missing or outlier flags
    problems = filtered_df[
        filtered_df[['temp_max_f','temp_min_f','energy_demand_gwh']].isnull().any(axis=1) |
        (filtered_df['temp_max_f'] > 130) |
        (filtered_df['temp_min_f'] < -50) |
        (filtered_df['energy_demand_gwh'] < 0)
    ]
    if not problems.empty:
        st.dataframe(problems.head(50))
    else:
        st.info("No problematic rows found in the current filter.")

    # Show time series of quality metrics over the selected date range
    st.subheader("Quality Metrics Over Time")
    ts = compute_quality_timeseries(filtered_df)
    if not ts.empty:
        # Use line chart for trends
        st.line_chart(ts.set_index('date')[['missing_total','temp_outliers','negative_energy']])
    else:
        st.info("Not enough data to compute time series metrics.")

    # Documentation / explanations for each check
    st.subheader("What each check does and why it matters")
    st.markdown("""
    - Missing Values: counts nulls in temp_max_f, temp_min_f, energy_demand_gwh. Missing data can bias analysis and break visualizations.
    - Temperature Outliers: flags temperatures > 130°F or < -50°F. These are physically implausible for most US locations and usually indicate sensor or ingestion issues.
    - Negative Energy Consumption: flags negative values which are invalid for demand (indicates data or unit errors).
    - Data Freshness: reports most recent date and days since latest. Stale data may indicate upstream failures or delays.
    
    Business rules:
    - Temperature thresholds (130°F / -50°F) are conservative bounds; adjust in config if needed for your domain.
    - We show per-day aggregated counts so you can detect systemic/data-source issues over time.
    """)

# --- Visualization 1: Geographic Overview ---
st.header("Geographic Overview")
    
# Prepare data for map
if not filtered_df.empty:
    # Get the latest data for each city
    latest_data = filtered_df.sort_values('date').groupby('city').last().reset_index()

    # Calculate % change from the previous day for each city
    prev_day_data = filtered_df[filtered_df['date'].isin(latest_data['date'] - pd.Timedelta(days=1))]
    if not prev_day_data.empty:
        prev_day_grouped = prev_day_data.groupby('city')['energy_demand_gwh'].first().rename('energy_demand_gwh_prev')
        latest_data = pd.merge(latest_data, prev_day_grouped, on='city', how='left')
        # Use assignment to avoid FutureWarning
        latest_data['energy_demand_gwh_prev'] = latest_data['energy_demand_gwh_prev'].fillna(0)
        
        # Avoid division by zero
        latest_data['energy_change_pct'] = latest_data.apply(
            lambda row: ((row['energy_demand_gwh'] - row['energy_demand_gwh_prev']) / row['energy_demand_gwh_prev']) * 100 if row['energy_demand_gwh_prev'] > 0 else 0,
            axis=1
        )
    else:
        latest_data['energy_change_pct'] = 0.0

    # --- Create formatted columns for the tooltip ---
    latest_data['tooltip_temp'] = latest_data['temp_avg_f'].map('{:.1f}°F'.format)
    latest_data['tooltip_energy'] = latest_data['energy_demand_gwh'].map('{:,.0f} GWh'.format)
    latest_data['tooltip_pct_change'] = latest_data['energy_change_pct'].map('{:.1f}%'.format)

    # --- Color coding: red for high energy usage, green for low ---
    median_energy = latest_data['energy_demand_gwh'].median()
    latest_data['color'] = latest_data['energy_demand_gwh'].apply(
        lambda x: [255, 0, 0] if x > median_energy else [0, 255, 0]
    )
    
    # --- Sizing Logic with a Minimum Size ---
    min_size_radius = 30000  # Increased from 15000
    max_size_radius = 120000 # Increased from 60000
    max_energy = latest_data['energy_demand_gwh'].max()
    if max_energy > 0:
        latest_data['size'] = min_size_radius + (latest_data['energy_demand_gwh'] / max_energy) * (max_size_radius - min_size_radius)
    else:
        latest_data['size'] = min_size_radius


    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=39.8283,
            longitude=-98.5795,
            zoom=3,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=latest_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                opacity=0.6,
                pickable=True,
            ),
            pdk.Layer(
                'TextLayer',
                data=latest_data,
                get_position='[lon, lat]',
                get_text='city',
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
        ],
        tooltip={
            "html": "<b>{city}</b><br/>"
                    "Temp (Avg): {tooltip_temp}<br/>"
                    "Energy Usage: {tooltip_energy}<br/>"
                    "% Change from Yesterday: {tooltip_pct_change}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))
else:
    st.warning("No data available for the selected date range to display the map.")

# --- Visualization 2: Time Series Analysis ---
st.header("Time Series Analysis")
    
col1, col2 = st.columns([3, 1])
with col1:
    selected_ts_city = st.selectbox("Select a City for Time Series View", options=["All Cities"] + list(all_cities))
with col2:
    make_stationary = st.checkbox("Make data stationary (apply differencing)", value=False)

if selected_ts_city == "All Cities":
    ts_df = filtered_df.groupby('date').agg({
        'temp_avg_f': 'mean',
        'energy_demand_gwh': 'sum'
    }).reset_index()
else:
    ts_df = filtered_df[filtered_df['city'] == selected_ts_city]
    
if not ts_df.empty:
    y_temp, y_energy = 'temp_avg_f', 'energy_demand_gwh'
    title_prefix = ""
    yaxis_temp_title, yaxis_energy_title = "Avg Temperature (°F)", "Energy Consumption (GWh)"

    if make_stationary:
        ts_df['temp_avg_f_diff'] = ts_df['temp_avg_f'].diff()
        ts_df['energy_demand_gwh_diff'] = ts_df['energy_demand_gwh'].diff()
        ts_df.dropna(inplace=True)
        
        y_temp, y_energy = 'temp_avg_f_diff', 'energy_demand_gwh_diff'
        title_prefix = "Daily Change in "
        yaxis_temp_title, yaxis_energy_title = "Daily Temperature Change (°F)", "Daily Energy Change (GWh)"

if not ts_df.empty:
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Temperature Line
    fig_ts.add_trace(
        go.Scatter(x=ts_df['date'], y=ts_df[y_temp], name=yaxis_temp_title, line=dict(color='orange')),
        secondary_y=False,
    )
    
    # Add Energy Consumption Line
    fig_ts.add_trace(
        go.Scatter(x=ts_df['date'], y=ts_df[y_energy], name=yaxis_energy_title, line=dict(color='blue', dash='dot')),
        secondary_y=True,
    )

    # --- Robust Weekend Highlighting ---
    # Find all Saturdays in the dataframe's date range
    saturdays = ts_df[ts_df['date'].dt.dayofweek == 5]
    for sat in saturdays['date']:
        # For each Saturday, add a shaded rectangle covering the next 48 hours
        fig_ts.add_vrect(
            x0=sat, 
            x1=sat + pd.Timedelta(days=2),
            fillcolor="rgba(200, 200, 200, 0.2)", 
            line_width=0, 
            layer="below"
        )
    
    fig_ts.update_layout(
        title_text=f"{title_prefix}Temperature vs. Energy Consumption in {selected_ts_city}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_ts.update_yaxes(title_text=yaxis_temp_title, secondary_y=False)
    fig_ts.update_yaxes(title_text=yaxis_energy_title, secondary_y=True)
    
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.warning("No time series data to display for the selected filters.")


# --- Visualization 3: Correlation Analysis ---
st.header("Correlation Analysis")
    
if not filtered_df.empty:
    # --- Clean data for correlation analysis ---
    corr_df = filtered_df.dropna(subset=['temp_avg_f', 'energy_demand_gwh'])

    if not corr_df.empty and len(corr_df) > 2:
        # --- Add a warning for narrow temperature ranges ---
        temp_range = corr_df['temp_avg_f'].max() - corr_df['temp_avg_f'].min()
        if temp_range < 20:  # Define a threshold (e.g., 20°F) for a "narrow" range
            st.warning(
                f"**Warning:** The temperature range in the selected data ({temp_range:.1f}°F) is very narrow. "
                "The correlation results may be misleading. Try selecting a wider date range for a more meaningful analysis."
            )

        fig_corr = px.scatter(
            corr_df,
            x='temp_avg_f',
            y='energy_demand_gwh',
            color='city',
            trendline='ols',
            trendline_scope='overall',
            hover_data=['date', 'temp_avg_f', 'energy_demand_gwh']
        )
        
        # Correctly unpack all 4 returned values
        correlation, r_squared, (slope, intercept), plot_df = get_correlation_stats(corr_df)

        # Format the regression equation string
        intercept_sign = '+' if intercept >= 0 else '-'
        equation = f"y = {slope:.2f}x {intercept_sign} {abs(intercept):.2f}"

        fig_corr.update_layout(
            title=f"Temperature vs. Energy Consumption (All Cities)<br><b>{equation}</b> | R² = {r_squared:.3f} | Correlation = {correlation:.3f}",
            xaxis_title="Average Temperature (°F)",
            yaxis_title="Energy Demand (GWh)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough complete data to perform correlation analysis for the selected filters.")
else:
    st.warning("No data available for correlation analysis.")


# --- Visualization 4: Usage Patterns Heatmap ---
st.header("Usage Patterns Heatmap")
    
heatmap_city = st.selectbox("Select City for Heatmap", options=list(all_cities))
    
heatmap_df = filtered_df[filtered_df['city'] == heatmap_city]
heatmap_data = prepare_heatmap_data(heatmap_df)

if not heatmap_data.empty:
    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu_r',
        text=heatmap_data.values,
        texttemplate="%{text:,.0f}"
    ))
    
    fig_hm.update_layout(
        title=f'Average Energy Usage (GWh) by Temp & Day of Week in {heatmap_city}',
        xaxis_title='Day of Week',
        yaxis_title='Temperature Range'
    )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.warning(f"Not enough data to generate a heatmap for {heatmap_city}.")


# Add a small UI control to run/force the pipeline before loading data
st.sidebar.header("Pipeline / Data Refresh")
st.sidebar.write("The app can auto-run the pipeline to produce data/processed/weather_energy_data.csv.")
force_run = st.sidebar.button("Run pipeline now")

# If the user forced a run show a spinner
if force_run:
    with st.spinner("Running pipeline to refresh processed data..."):
        result = run_pipeline_if_needed(force=True)
    if result.get('ran'):
        st.sidebar.success("Pipeline completed and data refreshed.")
    else:
        st.sidebar.error(f"Pipeline did not run: {result.get('reason') or result.get('stderr') or result.get('detail')}")

# On app start attempt automatic refresh if file missing or stale (max_age_hours=24)
with st.spinner("Checking processed data and running pipeline if needed..."):
    auto_result = run_pipeline_if_needed(max_age_hours=24)
    if auto_result.get('ran'):
        st.info("Pipeline run completed to refresh processed data.")
    elif auto_result.get('ran') is False and auto_result.get('reason') == 'up-to-date':
        # nothing to do
        pass
    else:
        # show a non-blocking warning so user knows why auto-run did not complete
        st.warning(f"Pipeline auto-run skipped or failed: {auto_result.get('reason') or auto_result.get('stderr') or auto_result.get('detail')}")