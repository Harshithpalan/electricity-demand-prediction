import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide"
)

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load Model and Data
@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    df = pd.read_csv('electricity_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return model, df

model, df = load_assets()

# --- HEADER ---
st.title("⚡ AI-Based Electricity Demand System")
st.markdown("#### Smart Monitoring & Predictive Forecasting for Cities")

# --- SIDEBAR ---
st.sidebar.header("Control Panel")
st.sidebar.markdown("---")
prediction_date = st.sidebar.date_input("Select Prediction Date", datetime.now() + timedelta(days=1))
prediction_hour = st.sidebar.slider("Select Hour", 0, 23, 12)

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)

current_val = df.iloc[-1]['Consumption_MW']
avg_val = df['Consumption_MW'].mean()
peak_val = df['Consumption_MW'].max()

with col1:
    st.metric("Current Demand", f"{current_val:.1f} MW", delta_color="inverse")
with col2:
    st.metric("Avg Consumption", f"{avg_val:.1f} MW")
with col3:
    st.metric("Peak Demand", f"{peak_val:.1f} MW")
with col4:
    # Prediction for the selected time
    pred_features = pd.DataFrame([[prediction_hour, prediction_date.weekday(), prediction_date.month, prediction_date.day]], 
                                columns=['hour', 'day_of_week', 'month', 'day_of_month'])
    predicted_val = model.predict(pred_features)[0]
    st.metric("Predicted Demand", f"{predicted_val:.1f} MW", delta=f"{predicted_val - current_val:.1f} MW")

# --- ALERTS ---
if predicted_val > 500:
    st.markdown(f"""
        <div class="high-demand-alert">
            ⚠️ <strong>HIGH DEMAND ALERT:</strong> Predicted demand of <strong>{predicted_val:.1f} MW</strong> 
            on {prediction_date.strftime('%Y-%m-%d')} at {prediction_hour}:00 exceeds safety threshold (500 MW).
        </div>
    """, unsafe_allow_html=True)

# --- CHARTS ---
st.markdown("### Consumption Trends")

tab1, tab2 = st.tabs(["📈 Historical Trends", "🔮 24-Hour Forecast"])

with tab1:
    # Last 7 days view
    last_7_days = df.tail(24 * 7)
    fig_hist = px.line(last_7_days, x='Datetime', y='Consumption_MW', 
                      title='Electricity Consumption (Last 7 Days)',
                      color_discrete_sequence=['#38bdf8'])
    fig_hist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        xaxis_gridcolor='rgba(255,255,255,0.1)',
        yaxis_gridcolor='rgba(255,255,255,0.1)'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    # Generate 24-hour forecast
    # Convert date to datetime to avoid 'date' object having no 'hour' attribute
    prediction_dt = datetime.combine(prediction_date, datetime.min.time())
    forecast_dates = [prediction_dt + timedelta(hours=i) for i in range(24)]
    forecast_df = pd.DataFrame({
        'Datetime': forecast_dates,
        'hour': [d.hour for d in forecast_dates],
        'day_of_week': [d.weekday() for d in forecast_dates],
        'month': [d.month for d in forecast_dates],
        'day_of_month': [d.day for d in forecast_dates]
    })
    forecast_df['Predicted_MW'] = model.predict(forecast_df[['hour', 'day_of_week', 'month', 'day_of_month']])
    
    fig_forecast = px.area(forecast_df, x='Datetime', y='Predicted_MW', 
                         title=f'24-Hour Forecast for {prediction_date.strftime("%Y-%m-%d")}',
                         color_discrete_sequence=['#818cf8'])
    fig_forecast.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        xaxis_gridcolor='rgba(255,255,255,0.1)',
        yaxis_gridcolor='rgba(255,255,255,0.1)'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("Developed for **Smart City Internship Project** | Powered by Random Forest Regressor")
