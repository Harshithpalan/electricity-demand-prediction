import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Styling ---
def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        /* Main background */
        .stApp {
            background: radial-gradient(circle at top right, #1a1f2c, #0b0e14);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Glassmorphism Container */
        .glass-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f1218 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            font-weight: 500;
        }
        
        /* Custom Radio Buttons (Sidebar Menu) */
        .stRadio > div {
            gap: 10px;
        }
        .stRadio label {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 15px !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            cursor: pointer;
        }
        .stRadio label:hover {
            background: rgba(0, 212, 255, 0.1) !important;
            border-color: #00d4ff !important;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
            padding: 24px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin-bottom: 1.5rem;
        }
        .metric-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: #00d4ff;
            box-shadow: 0 15px 40px rgba(0, 212, 255, 0.2);
        }
        
        /* Titles & Text */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        .gradient-text {
            background: linear-gradient(90deg, #00d4ff, #0083fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Status Pulse */
        .pulse {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            box-shadow: 0 0 0 rgba(0, 212, 255, 0.4);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 212, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 10px 10px 0 0;
            padding: 0 20px;
            color: #8892b0;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 212, 255, 0.1) !important;
            color: #00d4ff !important;
            border-bottom: 2px solid #00d4ff !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #00d4ff 0%, #0083fe 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 700;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
            transform: scale(1.02);
            color: white !important;
        }
        
        /* Alerts */
        .stAlert {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            color: #e0e0e0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def animated_metric(label, value, delta=None, prefix="", suffix=""):
    delta_html = ""
    if delta:
        color = "#ff4b4b" if "-" in str(delta) else "#00c853"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 4px;">{delta}</div>'
    
    st.markdown(f"""
        <div class="metric-card animate-fade-in">
            <div style="color: #8892b0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
            <div style="color: #00d4ff; font-size: 1.8rem; font-weight: 700; margin-top: 8px;">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# --- Data & Model Helpers ---
@st.cache_resource
def load_default_model():
    try:
        with open('models/default_rf_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv('data/default_data_sample.csv')
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        return df
    except:
        return None

def train_custom_model(df, features, target):
    # Drop rows with NaN in features or target
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Handle categorical
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, (X_test, y_test, y_pred)

def get_monitoring_status(value, threshold_high, threshold_mod):
    if value >= threshold_high:
        return "Critical", "🔴 High Usage Expected", "#ff4b4b"
    elif value >= threshold_mod:
        return "Warning", "🟠 Moderate Usage Expected", "#ffa600"
    else:
        return "Normal", "🟢 Normal Load Status", "#00c853"
