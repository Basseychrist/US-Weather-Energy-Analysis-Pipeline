import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def get_correlation_stats(df, x_col='temp_avg_f', y_col='energy_demand_gwh'):
    """Calculates correlation coefficient and R-squared value."""
    if df.empty or df[[x_col, y_col]].isnull().values.any():
        return 0, 0, (0, 0) # correlation, r_squared, (slope, intercept)

    correlation = df[x_col].corr(df[y_col])
    
    X = df[[x_col]].values
    y = df[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return correlation, r_squared, (slope, intercept)

def prepare_heatmap_data(df):
    """Prepares data for the energy usage heatmap."""
    if df.empty:
        return pd.DataFrame()
        
    df['temp_avg_f'] = (df['temp_max_f'] + df['temp_min_f']) / 2
    temp_bins = [-np.inf, 50, 60, 70, 80, 90, np.inf]
    temp_labels = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
    df['temp_range'] = pd.cut(df['temp_avg_f'], bins=temp_bins, labels=temp_labels, right=False)
    
    df['day_of_week'] = df['date'].dt.day_name()
    
    heatmap_data = df.groupby(['temp_range', 'day_of_week'])['energy_demand_gwh'].mean().unstack()
    
    # Order days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(columns=day_order)
    
    return heatmap_data