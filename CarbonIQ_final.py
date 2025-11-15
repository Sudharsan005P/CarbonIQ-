import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import itertools

warnings.filterwarnings("ignore")
# %matplotlib inline # This line is specific to Jupyter and might not be needed in other environments

# Load the dataset
df = pd.read_csv('co2_emissions_kt_by_country.csv')

# --- Data Exploration ---

# Describe the dataframe
def display_feature_list(features, feature_type):
    print(f"\n{feature_type} Features: ")
    print(', '.join(features) if features else 'None')

def describe_df(df):
    global categorical_features, continuous_features
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']

    print(f"\n{type(df).__name__} shape: {df.shape}")
    print('-'*20)
    print(f"\n{df.shape[0]:,.0f} samples")
    print('-'*20)
    print(f"\n{df.shape[1]:,.0f} attributes")
    print('-'*20)

    print(f'\nMissing Data: \n{df.isnull().sum()}')
    print('-'*20)
    print(f'\nDuplicates: {df.duplicated().sum()}')
    print('-'*20)
    print(f'\nData Types: \n{df.dtypes}')
    print('-'*20)
    
    display_feature_list(categorical_features, 'Categorical')
    print('-'*20)
    print(df.describe())
    print(f'\n{type(df).__name__} Head: \n')
    try:
        # prefer IPython.display when available, fall back to print
        from IPython.display import display as _display
        _display(df.head(5))
    except Exception:
        print(df.head(5))
    print(f'\n{type(df).__name__} Tail: \n')
    try:
        from IPython.display import display as _display
        _display(df.tail(5))
    except Exception:
        print(df.tail(5))

describe_df(df)

# Time Series Visualization
fig = px.line(df, x='year', y='value', color='country_name', 
              title='CO2 Emissions Over Time for All Countries', labels={'value': 'CO2 Emissions (kt)', 'country_name': 'Country'})
fig.show()

# --- Data-Preprocessing ---

# Detect and remove negative values
negative_values = df[df['value'] < 0]
print("Negative values found:")
print(negative_values)
df = df[df['value'] >= 0]

# Show Unique Countries
unique_country = df['country_name'].unique()
print(f'There are {len(unique_country)} unique countries.')

# Remove non-countries and region names
region_name = [
    'Africa Eastern and Southern', 'Africa Western and Central', 'Arab World',
    'Caribbean small states', 'Central Europe and the Baltics',
    'East Asia & Pacific', 
    'Europe & Central Asia', 'Euro area', 
    'Latin America & Caribbean', 
     'Middle East & North Africa', 
    'North America', 'Other small states', 
    'West Bank and Gaza', 'Pacific island small states', 
    'South Asia', 'Sub-Saharan Africa',
    'Small states', 'World'
]
# if remove_name isn't defined elsewhere, default to empty list
try:
    remove_name
except NameError:
    remove_name = []

df = df[~df['country_name'].isin(remove_name)]
country_df = df[~df['country_name'].isin(region_name)]
region_df = df[df['country_name'].isin(region_name)]

region_df.rename(columns={'country_name': 'continent_name'}, inplace=True)

# derive continent_type from the region_df we just created
try:
    continent_type = region_df['continent_name'].unique()
except Exception:
    continent_type = []

# Plotting each continent CO2 emission from 1960 to 2019
fig = px.line(region_df, x='year', y='value', color='continent_name', 
              title='CO2 Emissions Over Time for all Continents', labels={'value': 'CO2 Emissions (kt)', 'continent_name': 'Continent'})

fig.show()

# --- ARIMA Forecasting ---

# Find the Hyperparameter for ARIMA
def check_stationary(continent, df, d=0):
    series = df.loc[df['continent_name'] == continent, 'value'].values
    if d > 0:
        for _ in range(d):
            series = np.diff(series, n=1)
    result = adfuller(series)
    labels = ['ADF Test statistic', 'p-value', 'Lags used', 'No. of observations']
    
    print(f'Adfuller test for {continent} (d={d})')
    for value, label in zip(result, labels):
        print(f'{label} : {value}')
    
    if result[1] <= 0.05:
        print(f'\033[1m{continent} exhibits stationary time series after {d} differencing\033[0;0m')
        print('-'*50)
        return True
    else:
        return False
max_d = 3
initial_d = {}

for x in continent_type:
    is_stationary = False
    for d in range(max_d + 1):
        is_stationary = check_stationary(x, region_df, d=d)
        if is_stationary:
            initial_d[x] = d
            break
    if not is_stationary:
        print(f'{x} did not become stationary even after {max_d} differencing')
        initial_d[x] = None
        print('-'*50)

def determine_p(continent, df, d):
    series = df.loc[df['continent_name'] == continent, 'value'].values
    for _ in range(d):
        series = np.diff(series, n=1)
    
    plt.figure(figsize=(10, 6))
    plot_pacf(series, lags=12, method='ywm')
    plt.title(f'PACF for {continent} after {d} differencing')
    plt.show()

for continent, d_value in initial_d.items():
    if d_value is not None:
        determine_p(continent, region_df, d_value)

def determine_q(continent, df, d):
    series = df.loc[df['continent_name'] == continent, 'value'].values
    for _ in range(d):
        series = np.diff(series, n=1)
    
    plt.figure(figsize=(10, 6))
    plot_acf(series, lags=12)
    plt.title(f'ACF for {continent} after {d} differencing')
    plt.show()

for continent, d_value in initial_d.items():
    if d_value is not None:
        determine_q(continent, region_df, d_value)

# Forecasting with ARIMA
def arima_forecast(continent, df, order, forecast_years=10):
    series = df.loc[df['continent_name'] == continent, ['year', 'value']].set_index('year')
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=forecast_years)
    
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series['value'], label='Historical')
    plt.plot(range(series.index[-1] + 1, series.index[-1] + 1 + forecast_years), forecast, label='Forecast')
    plt.title(f'CO2 Emissions Forecast for {continent}')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (kt)')
    plt.legend()
    plt.show()
    
    return forecast

# Example usage with placeholder p,d,q values. 
# Replace with values determined from ACF/PACF plots
arima_orders = {
    'Africa Eastern and Southern': (1, 1, 1),
    'Africa Western and Central': (1, 1, 1),
    'Arab World': (1, 1, 1),
    'Central Europe and the Baltics': (2, 2, 2),
    'Caribbean small states': (1, 1, 1),
    'East Asia & Pacific': (1, 1, 1),
    'Europe & Central Asia': (1, 1, 1),
    'Euro area': (1, 1, 1),
    'Latin America & Caribbean': (1, 1, 1),
    'Middle East & North Africa': (1, 1, 1),
    'North America': (2, 2, 2),
    'Other small states': (1, 1, 1),
    'West Bank and Gaza': (1, 1, 1),
    'Pacific island small states': (1, 1, 1),
    'South Asia': (2, 2, 2),
    'Sub-Saharan Africa': (1, 1, 1),
    'Small states': (2, 2, 2),
    'World': (1, 1, 1)
}

for continent, order in arima_orders.items():
    arima_forecast(continent, region_df, order)

# Generate synthetic AQI data
np.random.seed(42)
dates = pd.to_datetime(pd.date_range(start='2025-01-01', end='2025-01-31'))
data = {
    'Date': dates,
    'AQI': np.random.randint(50, 200, size=len(dates)),
    'PM2.5': np.random.uniform(10, 120, len(dates)),
    'PM10': np.random.uniform(20, 200, len(dates)),
    'NO2': np.random.uniform(5, 80, len(dates)),
    'SO2': np.random.uniform(2, 40, len(dates)),
    'CO': np.random.uniform(0.2, 2.0, len(dates)),
    'O3': np.random.uniform(10, 100, len(dates))
}
aqi_df = pd.DataFrame(data)
print(aqi_df.head())



