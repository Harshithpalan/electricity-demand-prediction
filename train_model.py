import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train():
    # 1. Load Dataset
    print("Loading dataset...")
    df = pd.read_csv('electricity_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 2. Feature Engineering
    print("Extracting features...")
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['day_of_month'] = df['Datetime'].dt.day
    
    # Define features and target
    X = df[['hour', 'day_of_week', 'month', 'day_of_month']]
    y = df['Consumption_MW']
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Training Complete!")
    print(f"Mean Absolute Error: {mae:.2f} MW")
    print(f"R2 Score: {r2:.2f}")
    
    # 6. Save Model
    joblib.dump(model, 'model.joblib')
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train()
