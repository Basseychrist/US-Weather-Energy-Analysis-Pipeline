import pandas as pd
from scipy.stats import linregress
import statsmodels.api as sm
import numpy as np

def get_correlation_stats(df):
    """
    Calculates correlation, R-squared, and regression line with confidence intervals
    for temperature vs. energy demand.
    """
    # Ensure there's data to analyze
    df_cleaned = df.dropna(subset=['temp_avg_f', 'energy_demand_gwh'])
    if len(df_cleaned) < 2:
        return 0, 0, (0, 0), None  # correlation, r_squared, (slope, intercept), plot_df

    # Use scipy for basic stats
    slope, intercept, r_value, _, _ = linregress(
        df_cleaned['temp_avg_f'], df_cleaned['energy_demand_gwh']
    )
    correlation = r_value
    r_squared = r_value**2

    # Use statsmodels for confidence intervals
    X = sm.add_constant(df_cleaned['temp_avg_f'])
    y = df_cleaned['energy_demand_gwh']
    model = sm.OLS(y, X).fit()
    
    # Get predictions and confidence intervals
    predictions = model.get_prediction(X).summary_frame(alpha=0.05)
    
    # Combine with original x values for plotting
    plot_df = pd.DataFrame({
        'x': df_cleaned['temp_avg_f'],
        'y_hat': predictions['mean'],
        'ci_lower': predictions['mean_ci_lower'],
        'ci_upper': predictions['mean_ci_upper']
    }).sort_values(by='x')

    return correlation, r_squared, (slope, intercept), plot_df

def prepare_heatmap_data(df):
    """Prepares data for the usage patterns heatmap using specific temperature bins."""
    if df.empty or 'temp_avg_f' not in df.columns or df['temp_avg_f'].isnull().all():
        return pd.DataFrame()
        
    # Define custom temperature bins and labels as requested
    bins = [50, 60, 70, 80, 90, float('inf')]
    labels = ['50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
    
    df = df.copy()
    # Fill missing temp_avg_f with the mean of available values to avoid dropping all data
    if df['temp_avg_f'].isnull().any():
        df['temp_avg_f'] = df['temp_avg_f'].fillna(df['temp_avg_f'].mean())
    df['temp_range'] = pd.cut(df['temp_avg_f'], bins=bins, labels=labels, right=False)
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Fill missing energy_demand_gwh with 0 for heatmap visualization
    df['energy_demand_gwh'] = df['energy_demand_gwh'].fillna(0)
    heatmap_data = df.groupby(['temp_range', 'day_of_week'], observed=False)['energy_demand_gwh'].mean().unstack(fill_value=0)
    
    # Order columns by day of the week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(columns=day_order)
    
    # Order rows by temperature range
    heatmap_data = heatmap_data.reindex(labels).sort_index(ascending=False)
    
    return heatmap_data