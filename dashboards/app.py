import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yaml
import sys

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
        return None, None
    
    df = pd.read_csv(processed_path, parse_dates=['date'])
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    city_df = pd.DataFrame(config['cities'])
    
    # Merge city metadata (lat/lon)
    df = pd.merge(df, city_df[['name', 'lat', 'lon']], left_on='city', right_on='name', how='left')
    return df, city_df

df, city_df = load_data()

if df is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Date Range Selector
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # City Filter (Multiselect)
    all_cities = df['city'].unique()
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        options=all_cities,
        default=all_cities
    )

    # --- Filter Data Based on Selections ---
    filtered_df = df[
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date) &
        (df['city'].isin(selected_cities))
    ]

    # --- Main Dashboard ---
    st.title("⚡ US Weather & Energy Consumption Dashboard")
    st.markdown(f"Data last updated: **{max_date.strftime('%Y-%m-%d')}**")

    # --- Visualization 1: Geographic Overview ---
    st.header("Geographic Overview")
    
    # Prepare data for map
    latest_data = filtered_df.loc[filtered_df.groupby('city')['date'].idxmax()].copy()
    
    # Calculate % change from yesterday
    yesterday_data = filtered_df[filtered_df['date'] == (latest_data['date'].max() - pd.Timedelta(days=1))]
    if not yesterday_data.empty:
        latest_data = pd.merge(latest_data, yesterday_data[['city', 'energy_demand_gwh']], on='city', suffixes=('', '_prev'))
        latest_data['energy_change_pct'] = ((latest_data['energy_demand_gwh'] - latest_data['energy_demand_gwh_prev']) / latest_data['energy_demand_gwh_prev']) * 100
    else:
        latest_data['energy_change_pct'] = 0.0

    if not latest_data.empty:
        latest_data['color'] = latest_data['energy_demand_gwh'].apply(lambda x: [255, 0, 0] if x > latest_data['energy_demand_gwh'].median() else [0, 255, 0])
        latest_data['size'] = latest_data['energy_demand_gwh'] / latest_data['energy_demand_gwh'].max() * 50000

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
            ],
            tooltip={
                "html": "<b>{city}</b><br/>"
                        "Temp (Avg): {temp_avg_f:.1f}°F<br/>"
                        "Energy Usage: {energy_demand_gwh:,.0f} GWh<br/>"
                        "% Change from Yesterday: {energy_change_pct:.1f}%",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        ))
    else:
        st.warning("No data available for the selected date range to display the map.")

    # --- Visualization 2: Time Series Analysis ---
    st.header("Time Series Analysis")
    city_options = ["All Cities"] + list(all_cities)
    selected_ts_city = st.selectbox("Select a City for Time Series View", options=city_options)

    if selected_ts_city == "All Cities":
        ts_df = filtered_df.groupby('date').agg({
            'temp_avg_f': 'mean',
            'energy_demand_gwh': 'sum'
        }).reset_index()
    else:
        ts_df = filtered_df[filtered_df['city'] == selected_ts_city]
    
    if not ts_df.empty:
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Temperature Line
        fig_ts.add_trace(
            go.Scatter(x=ts_df['date'], y=ts_df['temp_avg_f'], name="Avg Temperature (°F)", line=dict(color='orange')),
            secondary_y=False,
        )
        
        # Add Energy Consumption Line
        fig_ts.add_trace(
            go.Scatter(x=ts_df['date'], y=ts_df['energy_demand_gwh'], name="Energy Consumption (GWh)", line=dict(color='blue', dash='dot')),
            secondary_y=True,
        )

        # Highlight weekends
        weekends = ts_df[ts_df['date'].dt.dayofweek >= 5]
        for i in range(0, len(weekends), 2):
             fig_ts.add_vrect(
                 x0=weekends['date'].iloc[i], 
                 x1=weekends['date'].iloc[i+1] if i+1 < len(weekends) else weekends['date'].iloc[i] + pd.Timedelta(days=1),
                 fillcolor="rgba(200, 200, 200, 0.2)", line_width=0, layer="below"
             )
        
        fig_ts.update_layout(
            title_text=f"Temperature vs. Energy Consumption in {selected_ts_city}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_ts.update_yaxes(title_text="Avg Temperature (°F)", secondary_y=False)
        fig_ts.update_yaxes(title_text="Energy Consumption (GWh)", secondary_y=True)
        
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("No time series data to display for the selected filters.")


    # --- Visualization 3: Correlation Analysis ---
    st.header("Correlation Analysis")
    
    if not filtered_df.empty:
        fig_corr = px.scatter(
            filtered_df,
            x='temp_avg_f',
            y='energy_demand_gwh',
            color='city',
            trendline='ols',
            trendline_scope='overall',
            hover_data=['date']
        )
        
        correlation, r_squared, _ = get_correlation_stats(filtered_df)

        fig_corr.update_layout(
            title=f"Temperature vs. Energy Consumption (All Cities)<br>R² = {r_squared:.3f} | Correlation = {correlation:.3f}",
            xaxis_title="Average Temperature (°F)",
            yaxis_title="Energy Demand (GWh)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
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