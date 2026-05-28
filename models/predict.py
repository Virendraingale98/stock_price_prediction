import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from backend import crud

# Suppress TensorFlow logs for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from features.engineer import engineer_features

def generate_next_day_prediction(ticker: str, model_name: str, db: Session) -> float:
    """
    Loads the saved model and scaler for the ticker, engineers features on the
    most recent data, and predicts the next day's closing price.
    """
    # 1. Fetch latest prices from DB
    db_prices = crud.get_historical_prices(db, ticker)
    if not db_prices or len(db_prices) < 60:
        raise ValueError(f"Insufficient price data found in DB for {ticker}. Need at least 60 days.")
        
    # Convert database objects to DataFrame (we only need the last 100 days for inference)
    data = [{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in db_prices[-100:]]
    
    df = pd.DataFrame(data)
    
    # 2. Engineer features (exclude target since we're predicting the unknown next day)
    df_feats = engineer_features(df, include_target=False)
    
    # Extract the last row which contains features for the current day
    # Check if there are any NaNs in the final row. If so, forward-fill or handle it.
    df_feats = df_feats.ffill().bfill()
    
    # Define the 27 feature columns used in training
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'daily_return',
        'sma_7', 'sma_21', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
        'adx_14', 'obv',
        'close_lag_1', 'close_lag_3', 'close_lag_7',
        'day_of_week', 'month', 'is_month_end'
    ]
    
    # Check that we have the last row with no NaNs
    latest_features = df_feats[feature_cols].tail(1)
    
    # 3. Load the pre-trained scaler
    scaler_path = f"models/saved_models/scaler_{ticker}.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler for {ticker} not found. Please train models first.")
        
    scaler = joblib.load(scaler_path)
    
    # 4. Standardize features
    scaled_feats = scaler.transform(df_feats[feature_cols])
    
    # 5. Predict using selected model
    if model_name == "Linear Regression":
        model_path = f"models/saved_models/model_lr_{ticker}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Linear Regression model for {ticker} not found.")
        model = joblib.load(model_path)
        
        # Predict using the last scaled row
        X_last = scaled_feats[-1].reshape(1, -1)
        pred = model.predict(X_last)[0]
        
    elif model_name == "Random Forest":
        model_path = f"models/saved_models/model_rf_{ticker}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Random Forest model for {ticker} not found.")
        model = joblib.load(model_path)
        
        X_last = scaled_feats[-1].reshape(1, -1)
        pred = model.predict(X_last)[0]
        
    elif model_name == "XGBoost":
        model_path = f"models/saved_models/model_xgb_{ticker}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model for {ticker} not found.")
        model = joblib.load(model_path)
        
        X_last = scaled_feats[-1].reshape(1, -1)
        pred = model.predict(X_last)[0]
        
    elif model_name == "LSTM":
        model_path = f"models/saved_models/model_lstm_{ticker}.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model for {ticker} not found.")
            
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        # Prepare 60-day sequence from scaled features
        if len(scaled_feats) < 60:
            raise ValueError(f"Insufficient scaled data points ({len(scaled_feats)}) to create 60-day sequence.")
            
        X_seq_last = scaled_feats[-60:].reshape(1, 60, -1)
        pred = model.predict(X_seq_last, verbose=0)[0][0]
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return float(pred)
