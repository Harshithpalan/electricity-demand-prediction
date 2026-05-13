import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(years=2):
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=365 * years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    data = []
    for dt in date_range:
        # Base consumption
        base = 300
        
        # Daily pattern: higher in evening (6 PM - 10 PM), lower in late night
        hour_effect = 50 * np.sin((dt.hour - 6) * 2 * np.pi / 24) + 50
        if 18 <= dt.hour <= 22:
            hour_effect += 100
            
        # Weekly pattern: higher on weekdays
        day_effect = 30 if dt.weekday() < 5 else 0
        
        # Seasonal pattern: higher in summer (Jun-Aug) and winter (Dec-Feb)
        month_effect = 80 * np.cos((dt.month - 1) * 2 * np.pi / 12) + 50
        
        # Random noise
        noise = np.random.normal(0, 15)
        
        consumption = base + hour_effect + day_effect + month_effect + noise
        data.append({
            'Datetime': dt,
            'Consumption_MW': round(max(50, consumption), 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv('electricity_data.csv', index=False)
    print(f"Generated {len(df)} rows of data and saved to electricity_data.csv")

if __name__ == "__main__":
    generate_data()
