import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yaml
import sys
import numpy as np
from src.analysis import get_correlation_stats, prepare_heatmap_data
from src.data_processor import run_quality_checks  # added import

# To deploy on Streamlit Cloud:
# - Ensure all dependencies are listed in pyproject.toml
# - Make sure 'data/processed/weather_energy_data.csv' exists and is up-to-date
# - Make sure 'config/config.yaml' exists and contains city metadata and API keys
# - All imports and file paths should be relative to the project root

# Add src to path to import modules
# Make sure the app is run from the root of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.analysis import get_correlation_stats, prepare_heatmap_data

# --- Page Configuration ---
st.set_page_config(
    page_title="US Weather & Energy Analysis",
    page_icon="⚡",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    processed_path = 'data/processed/weather_energy_data.csv'
    if not os.path.exists(processed_path):
        st.error(f"Processed data not found at {processed_path}. Please run the pipeline first.")
        return None, None, None
    
    df = pd.read_csv(processed_path, parse_dates=['date'])
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    city_df = pd.DataFrame(config['cities'])
    
    # Merge city metadata (lat/lon)
    df = pd.merge(df, city_df[['name', 'lat', 'lon']], left_on='city', right_on='name', how='left')
    return df, city_df, config  # return config as well

df, city_df, config = load_data()

if df is not None:
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