import pandas as pd
import numpy as np
from typing import Tuple, List

def engineer_features(df_raw: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Engineers technical indicators, lag features, and date features from raw stock prices.
    
    Parameters:
        df_raw: Pandas DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
        include_target: If True, shifts the Close price to create 'target' (next-day closing price).
        
    Returns:
        DataFrame with all engineered features.
    """
    df = df_raw.copy()
    
    # Ensure sorted by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Basic Price Features
    df['daily_return'] = df['close'].pct_change() * 100
    
    # Try importing 'ta' library, fallback to manual calculations if it fails
    try:
        import ta
        print("Using 'ta' library for feature engineering...")
        
        # Moving Averages
        df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['sma_21'] = ta.trend.sma_indicator(df['close'], window=21)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Momentum
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_hist'] = ta.trend.macd_diff(df['close'])
        
        # Volatility
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Trend
        df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
    except ImportError:
        print("Warning: 'ta' library not found. Falling back to manual feature engineering...")
        
        # Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_21'] = df['close'].rolling(window=21).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Momentum: RSI (14)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # Smoothed Wilder's method for RSI
        for i in range(14, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Momentum: MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility: Bollinger Bands (20, 2)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility: Average True Range (14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        for i in range(14, len(df)):
            df.loc[i, 'atr_14'] = (df.loc[i-1, 'atr_14'] * 13 + tr.iloc[i]) / 14
            
        # Trend: OBV
        df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
        
        # Trend: ADX (14)
        up_move = df['high'].diff()
        down_move = df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_rolling = tr.rolling(window=14).mean()
        plus_dm_rolling = pd.Series(plus_dm).rolling(window=14).mean()
        minus_dm_rolling = pd.Series(minus_dm).rolling(window=14).mean()
        
        for i in range(14, len(df)):
            tr_rolling.iloc[i] = (tr_rolling.iloc[i-1] * 13 + tr.iloc[i]) / 14
            plus_dm_rolling.iloc[i] = (plus_dm_rolling.iloc[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_rolling.iloc[i] = (minus_dm_rolling.iloc[i-1] * 13 + minus_dm[i]) / 14
            
        plus_di = 100 * (plus_dm_rolling / tr_rolling.replace(0, 1e-9))
        minus_di = 100 * (minus_dm_rolling / tr_rolling.replace(0, 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        
        df['adx_14'] = dx.rolling(window=14).mean()
        for i in range(28, len(df)):
            df.loc[i, 'adx_14'] = (df.loc[i-1, 'adx_14'] * 13 + dx.iloc[i]) / 14

    # Lag features
    df['close_lag_1'] = df['close'].shift(1)
    df['close_lag_3'] = df['close'].shift(3)
    df['close_lag_7'] = df['close'].shift(7)
    
    # Date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Target Variable: next-day closing price
    if include_target:
        df['target'] = df['close'].shift(-1)
        
    return df
