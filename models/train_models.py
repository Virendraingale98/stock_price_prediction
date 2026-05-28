import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Suppress TensorFlow logs for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from backend import crud, schemas
from features.engineer import engineer_features
from utils.metrics import calculate_metrics

# Try importing deep learning and boosting models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    tf = None

try:
    import shap
except ImportError:
    shap = None


def train_and_save_all(ticker: str, db: Session) -> list:
    """
    Fetches stock data, trains 4 models, saves them to disk,
    writes metrics to database, and saves SHAP explanation plot.
    """
    # 1. Fetch historical prices from DB
    db_prices = crud.get_historical_prices(db, ticker)
    if not db_prices:
        raise ValueError(f"No price data found in database for ticker {ticker}.")
        
    data = [{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in db_prices]
    
    df_raw = pd.DataFrame(data)
    
    # 2. Engineer features
    df_feats = engineer_features(df_raw, include_target=True)
    
    # Drop rows with NaNs (from rolling indicators and target shift)
    df_feats = df_feats.dropna().reset_index(drop=True)
    
    if len(df_feats) < 100:
        raise ValueError(f"Insufficient clean records ({len(df_feats)}) after feature engineering for {ticker}.")
        
    # Feature columns (27 features)
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'daily_return',
        'sma_7', 'sma_21', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
        'adx_14', 'obv',
        'close_lag_1', 'close_lag_3', 'close_lag_7',
        'day_of_week', 'month', 'is_month_end'
    ]
    
    X = df_feats[feature_cols].copy()
    y = df_feats['target'].copy()
    close_current = df_feats['close'].copy()  # For directional accuracy
    
    # 3. Train/Test Split (Time-based, 80% train, 20% test)
    split_idx = int(len(df_feats) * 0.8)
    
    X_train_raw = X.iloc[:split_idx]
    X_test_raw = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx].values
    y_test = y.iloc[split_idx:].values
    close_current_test = close_current.iloc[split_idx:].values
    
    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Scale full X for sequencing
    X_scaled = np.vstack([X_train_scaled, X_test_scaled])
    
    # Save Scaler
    os.makedirs("models/saved_models", exist_ok=True)
    scaler_path = f"models/saved_models/scaler_{ticker}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Prepare LSTM sequences (60 days)
    seq_length = 60
    X_seq, y_seq, close_seq_test = [], [], []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i : i + seq_length])
        y_seq.append(y.iloc[i + seq_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Align LSTM train/test splits with the standard split
    # Since we need 60 days history, test sequences are where target is >= split_idx
    # which corresponds to indices starting from split_idx - seq_length
    train_seq_end = split_idx - seq_length
    
    X_seq_train = X_seq[:train_seq_end]
    y_seq_train = y_seq[:train_seq_end]
    X_seq_test = X_seq[train_seq_end:]
    y_seq_test = y_seq[train_seq_end:]
    
    # Verify alignment: y_seq_test should match y_test exactly
    assert len(y_seq_test) == len(y_test), f"LSTM test target length ({len(y_seq_test)}) does not match ML test target length ({len(y_test)})!"
    
    trained_metrics = []
    
    # --- MODEL 1: Linear Regression ---
    print("Training Linear Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    
    metrics_lr = calculate_metrics(y_test, y_pred_lr, close_current_test)
    metrics_lr['model_name'] = "Linear Regression"
    metrics_lr['ticker'] = ticker
    crud.save_model_metric(db, schemas.ModelMetricBase(**metrics_lr))
    trained_metrics.append(metrics_lr)
    
    # Save Linear Regression Model
    joblib.dump(model_lr, f"models/saved_models/model_lr_{ticker}.pkl")
    
    # --- MODEL 2: Random Forest ---
    print("Training Random Forest...")
    rf_base = RandomForestRegressor(n_estimators=200, random_state=42)
    # Tune max_depth using grid search
    param_grid = {"max_depth": [5, 10, 15, 20]}
    grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    model_rf = grid_search.best_estimator_
    
    y_pred_rf = model_rf.predict(X_test_scaled)
    metrics_rf = calculate_metrics(y_test, y_pred_rf, close_current_test)
    metrics_rf['model_name'] = "Random Forest"
    metrics_rf['ticker'] = ticker
    crud.save_model_metric(db, schemas.ModelMetricBase(**metrics_rf))
    trained_metrics.append(metrics_rf)
    
    # Save Random Forest Model
    joblib.dump(model_rf, f"models/saved_models/model_rf_{ticker}.pkl")
    print(f"RF Best Max Depth: {grid_search.best_params_['max_depth']}")
    
    # --- MODEL 3: XGBoost ---
    if xgb is not None:
        print("Training XGBoost...")
        # Create validation split for early stopping
        val_split = int(len(X_train_scaled) * 0.9)
        X_tr, y_tr = X_train_scaled[:val_split], y_train[:val_split]
        X_val, y_val = X_train_scaled[val_split:], y_train[val_split:]
        
        model_xgb = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
            early_stopping_rounds=15
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred_xgb = model_xgb.predict(X_test_scaled)
        metrics_xgb = calculate_metrics(y_test, y_pred_xgb, close_current_test)
        metrics_xgb['model_name'] = "XGBoost"
        metrics_xgb['ticker'] = ticker
        crud.save_model_metric(db, schemas.ModelMetricBase(**metrics_xgb))
        trained_metrics.append(metrics_xgb)
        
        # Save XGBoost Model
        joblib.dump(model_xgb, f"models/saved_models/model_xgb_{ticker}.pkl")
    else:
        print("Skipping XGBoost (package not installed).")
        
    # --- MODEL 4: LSTM ---
    if tf is not None:
        print("Training LSTM (2 layers)...")
        # Build sequence model
        num_features = len(feature_cols)
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model_lstm.compile(optimizer="adam", loss="mean_squared_error")
        
        # Early stopping callback
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        model_lstm.fit(
            X_seq_train, y_seq_train,
            validation_split=0.1,
            epochs=15,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred_lstm = model_lstm.predict(X_seq_test, verbose=0).flatten()
        metrics_lstm = calculate_metrics(y_test, y_pred_lstm, close_current_test)
        metrics_lstm['model_name'] = "LSTM"
        metrics_lstm['ticker'] = ticker
        crud.save_model_metric(db, schemas.ModelMetricBase(**metrics_lstm))
        trained_metrics.append(metrics_lstm)
        
        # Save LSTM Model in native Keras format
        model_lstm.save(f"models/saved_models/model_lstm_{ticker}.keras")
    else:
        print("Skipping LSTM (TensorFlow not installed).")
        
    # --- SHAP EXPLAINABILITY (using Random Forest) ---
    if shap is not None:
        try:
            print("Generating SHAP Feature Importance Plot...")
            explainer = shap.TreeExplainer(model_rf)
            
            # Select up to 100 samples from the test set for SHAP analysis (speed efficiency)
            X_shap = X_test_raw.iloc[:100] if len(X_test_raw) > 100 else X_test_raw
            X_shap_scaled = scaler.transform(X_shap)
            
            shap_values = explainer(X_shap_scaled)
            
            # Plot and save
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap_scaled, feature_names=feature_cols, max_display=15, show=False)
            plt.title(f"SHAP Feature Importance (Top 15) - {ticker}", fontsize=14, pad=15)
            plt.tight_layout()
            
            os.makedirs("models/shap_plots", exist_ok=True)
            shap_plot_path = f"models/shap_plots/shap_importance_{ticker}.png"
            plt.savefig(shap_plot_path, dpi=300)
            plt.close()
            print(f"SHAP plot saved to {shap_plot_path}")
        except Exception as e:
            print(f"Warning: Failed to generate SHAP plot: {str(e)}")
    else:
        print("Skipping SHAP Plot (shap not installed).")
        
    return trained_metrics
