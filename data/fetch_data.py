import yfinance as yf
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional

def fetch_yfinance_data(
    ticker: str, start_date: str = "2018-01-01", end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Downloads historical stock data from Yahoo Finance for a given ticker and date range.
    Returns a list of dictionaries ready for insertion into SQLite via SQLAlchemy.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # Download data from yfinance
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        print(f"Warning: No data returned for ticker {ticker}")
        return []
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Handle MultiIndex columns that newer yfinance versions return (e.g. [('Close', 'RELIANCE.NS')])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    # Rename the first column (which was the index) to 'date'
    df = df.rename(columns={df.columns[0]: 'date'})
        
    # Standardize column names (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    records = []
    for _, row in df.iterrows():
        # Handle date extraction from Timestamp
        row_date = row['date']
        if isinstance(row_date, pd.Timestamp):
            row_date = row_date.to_pydatetime().date()
        elif isinstance(row_date, str):
            row_date = datetime.strptime(row_date, "%Y-%m-%d").date()
            
        # Clean and type-cast the columns
        records.append({
            "ticker": ticker,
            "date": row_date,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        })
        
    print(f"Successfully fetched {len(records)} records for {ticker}")
    return records
