import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from datetime import date
from typing import Dict, Any

from backend import crud, models
from features.engineer import engineer_features

def get_recommendation(
    ticker: str, 
    current_price: float, 
    predicted_price: float, 
    model_name: str, 
    db: Session
) -> Dict[str, Any]:
    """
    Computes expected return, checks RSI condition, generates BUY/SELL/HOLD signal,
    and calculates confidence score based on model RMSE vs stock volatility.
    """
    # 1. Calculate predicted return percentage
    predicted_return = ((predicted_price - current_price) / current_price) * 100
    
    # 2. Get latest RSI (14) by engineering features on the most recent 50 days of data
    rsi = 50.0  # Default neutral RSI if calculation fails
    volatility = current_price * 0.02  # Default volatility (2% of price) if calculation fails
    
    try:
        # Fetch last 60 prices to make sure we have enough data for 50-day rolling indicators
        db_prices = crud.get_historical_prices(db, ticker)
        if len(db_prices) >= 20:
            # Convert to DataFrame
            data = [{
                'date': p.date,
                'open': p.open,
                'high': p.high,
                'low': p.low,
                'close': p.close,
                'volume': p.volume
            } for p in db_prices[-60:]]
            
            df = pd.DataFrame(data)
            df_feats = engineer_features(df, include_target=False)
            
            # Extract latest RSI and 30-day close price volatility
            if not df_feats.empty:
                latest_row = df_feats.iloc[-1]
                if 'rsi_14' in latest_row and not pd.isna(latest_row['rsi_14']):
                    rsi = float(latest_row['rsi_14'])
                    
                # Calculate 30-day standard deviation of closing prices
                closes = df_feats['close'].tail(30).values
                if len(closes) >= 5:
                    volatility = float(np.std(closes))
    except Exception as e:
        print(f"Warning: Failed to calculate RSI/volatility for recommendation: {str(e)}")
        
    # 3. Decision Rules
    # BUY if expected return > 1.5% and RSI < 60
    # SELL if expected return < -1.5% or RSI > 75
    # HOLD otherwise
    if predicted_return > 1.5 and rsi < 60:
        signal = "BUY"
    elif predicted_return < -1.5 or rsi > 75:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    # 4. Confidence Score (0-100%)
    # confidence = 100 / (1 + RMSE / Volatility)
    # Fetch RMSE for the model on this stock
    rmse = None
    try:
        metrics = crud.get_model_metrics(db, ticker)
        for m in metrics:
            if m.model_name == model_name:
                rmse = m.rmse
                break
    except Exception as e:
        print(f"Warning: Failed to fetch model RMSE: {str(e)}")
        
    if rmse is None:
        # Default fallback RMSE (estimate: 1.5% of current price)
        rmse = current_price * 0.015
        
    # Prevent division by zero
    if volatility <= 0:
        volatility = current_price * 0.02
        
    confidence = 100.0 / (1.0 + (rmse / volatility))
    
    # Ensure confidence stays within [0, 100]
    confidence = max(0.0, min(100.0, float(confidence)))
    
    return {
        "predicted_return": float(predicted_return),
        "rsi": rsi,
        "signal": signal,
        "confidence": float(confidence),
        "rmse": float(rmse),
        "volatility": float(volatility)
    }
