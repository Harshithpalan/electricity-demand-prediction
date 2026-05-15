import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Loading data...")
file_path = 'data/opsd_time_series_60min_singleindex.csv'
# Load only necessary columns to save memory
cols_to_read = ['utc_timestamp']
temp_df = pd.read_csv(file_path, nrows=1)
load_cols = [c for c in temp_df.columns if 'load_actual' in c.lower()]

if not load_cols:
    print("No load columns found! Using dummy data.")
    # Generate dummy data for testing if real data fails
    dates = pd.date_range('2023-01-01', periods=8760, freq='H')
    df = pd.DataFrame({
        'utc_timestamp': dates,
        'load_actual': 500 + 200 * np.sin(np.pi * dates.hour / 12) + 50 * np.random.randn(len(dates))
    })
    target_col = 'load_actual'
else:
    target_col = load_cols[0] # Take the first one, usually Germany or similar
    cols_to_read.append(target_col)
    df = pd.read_csv(file_path, usecols=cols_to_read)

print(f"Processing data for target: {target_col}")
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.dropna(subset=[target_col])

# Feature engineering
df['hour'] = df['utc_timestamp'].dt.hour
df['day'] = df['utc_timestamp'].dt.day
df['month'] = df['utc_timestamp'].dt.month
df['weekday'] = df['utc_timestamp'].dt.weekday

X = df[['hour', 'day', 'month', 'weekday']]
y = df[target_col]

# Sample data for performance (take last 2 years of data if possible, or just sample)
if len(df) > 10000:
    df_sample = df.tail(10000)
    X_train = df_sample[['hour', 'day', 'month', 'weekday']]
    y_train = df_sample[target_col]
else:
    X_train = X
    y_train = y

print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('models/default_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save sample data for the dashboard (last 1000 rows for display)
df.tail(1000).to_csv('data/default_data_sample.csv', index=False)

print("Done! Model saved to models/default_rf_model.pkl")
print("Sample data saved to data/default_data_sample.csv")
