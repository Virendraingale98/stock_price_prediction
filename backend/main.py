import os
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta
from typing import List, Optional

from backend import database, models, schemas, crud
from data.fetch_data import fetch_yfinance_data
from recommendation.engine import get_recommendation

# Create FastAPI app
app = FastAPI(
    title="Stock Prediction and Recommendation System API",
    description="Backend API for fetching data, training ML/DL models, and generating stock recommendations.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables on startup
@app.on_event("startup")
def startup_event():
    models.Base.metadata.create_all(bind=database.engine)
    print("Database tables initialized.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API. Go to /docs for Swagger documentation."}

# --- Data Endpoints ---

@app.post("/data/sync", response_model=schemas.SyncResponse)
def sync_data(request: schemas.SyncRequest, db: Session = Depends(database.get_db)):
    """
    Syncs historical stock data from yfinance and stores it in the database.
    Also updates any pending historical predictions with actual close prices.
    """
    tickers = request.tickers
    if not tickers:
        # Default target tickers if none provided
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "SBIN.NS", "TMPV.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"
        ]
        
    total_records = 0
    for ticker in tickers:
        try:
            # Sync historical prices
            records = fetch_yfinance_data(ticker, start_date=request.start_date, end_date=request.end_date)
            if records:
                inserted = crud.bulk_insert_prices(db, records)
                total_records += inserted
                
                # Proactively update actual prices in previous predictions
                date_prices = {r['date']: r['close'] for r in records}
                updated_preds = crud.update_actual_prices(db, ticker, date_prices)
                if updated_preds > 0:
                    print(f"Updated {updated_preds} predictions with actual prices for {ticker}")
        except Exception as e:
            print(f"Error syncing {ticker}: {str(e)}")
            continue
            
    return {
        "status": "success",
        "message": f"Sync completed. Added/updated prices in database.",
        "synced_records": total_records
    }

@app.get("/data/prices/{ticker}", response_model=List[schemas.HistoricalPrice])
def get_prices(
    ticker: str, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None, 
    db: Session = Depends(database.get_db)
):
    """Retrieves historical prices from the SQLite database."""
    prices = crud.get_historical_prices(db, ticker, start_date, end_date)
    if not prices:
        raise HTTPException(status_code=404, detail=f"No historical prices found for {ticker}. Run /data/sync first.")
    return prices

# --- Model Endpoints ---

@app.post("/models/train", response_model=schemas.TrainResponse)
def train_models_endpoint(
    request: schemas.TrainRequest, 
    db: Session = Depends(database.get_db)
):
    """
    Trains all 4 models (Linear Regression, Random Forest, XGBoost, LSTM) for a ticker.
    Saves metrics to the database and saves model files to disk.
    """
    # Import train function dynamically to avoid circular dependencies or loading TF unnecessarily early
    try:
        from models.train_models import train_and_save_all
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to import training script. Are requirements fully installed? Error: {str(e)}"
        )
        
    ticker = request.ticker
    
    # Check if we have historical data first
    prices = crud.get_historical_prices(db, ticker)
    if len(prices) < 120:  # Need sufficient historical data (sequence_length=60 + training)
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient historical data for {ticker} (found {len(prices)} rows, need at least 120). Sync data first."
        )
        
    try:
        metrics = train_and_save_all(ticker, db)
        return {
            "status": "success",
            "message": f"Successfully trained all 4 models for {ticker}.",
            "metrics": metrics
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/models/metrics/{ticker}", response_model=List[schemas.ModelMetric])
def get_metrics(ticker: str, db: Session = Depends(database.get_db)):
    """Retrieves saved evaluation metrics for a ticker."""
    metrics = crud.get_model_metrics(db, ticker)
    if not metrics:
        raise HTTPException(
            status_code=404, 
            detail=f"No trained model metrics found for {ticker}. Train models first."
        )
    return metrics

# --- Prediction & Screener Endpoints ---

@app.post("/predictions/generate", response_model=schemas.Prediction)
def generate_prediction(
    request: schemas.PredictionRequest, 
    db: Session = Depends(database.get_db)
):
    """
    Generates a next-day price prediction using a selected model,
    runs the recommendation engine, logs prediction, and returns the result.
    """
    ticker = request.ticker
    model_name = request.model_name
    
    # 1. Load prediction function
    try:
        from models.predict import generate_next_day_prediction
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to import prediction module. Error: {str(e)}"
        )
        
    # 2. Get latest price from database
    latest_record = crud.get_latest_price(db, ticker)
    if not latest_record:
        raise HTTPException(
            status_code=400, 
            detail=f"No price data found for {ticker}. Please sync data first."
        )
        
    current_price = latest_record.close
    current_date = latest_record.date
    
    # Determine the target date (next business day approx)
    # If Friday, next day is Monday (current_date + 3)
    target_date = current_date + timedelta(days=1)
    if current_date.weekday() == 4:  # Friday
        target_date = current_date + timedelta(days=3)
    elif current_date.weekday() == 5:  # Saturday
        target_date = current_date + timedelta(days=2)
        
    try:
        # 3. Generate prediction using saved model
        predicted_close = generate_next_day_prediction(ticker, model_name, db)
        
        # 4. Generate BUY/SELL/HOLD and confidence
        rec = get_recommendation(ticker, current_price, predicted_close, model_name, db)
        
        # 5. Build prediction record
        pred_schema = schemas.PredictionBase(
            ticker=ticker,
            prediction_date=current_date,
            target_date=target_date,
            model_name=model_name,
            predicted_close=predicted_close,
            actual_close=None,
            predicted_return=rec['predicted_return'],
            signal=rec['signal'],
            confidence=rec['confidence']
        )
        
        # 6. Save to DB and return
        db_pred = crud.save_prediction(db, pred_schema)
        return db_pred
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Saved model '{model_name}' for {ticker} not found. Please train models first."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction generation failed: {str(e)}")

@app.get("/predictions/history/{ticker}", response_model=List[schemas.Prediction])
def get_prediction_history(
    ticker: str, 
    model_name: str = "Random Forest", 
    db: Session = Depends(database.get_db)
):
    """Retrieves all historical predictions and actual closing prices for overlay charts."""
    history = crud.get_predictions_history(db, ticker, model_name)
    return history

@app.get("/screener", response_model=schemas.ScreenerResponse)
def get_screener(
    model_name: str = "Random Forest", 
    db: Session = Depends(database.get_db)
):
    """Runs predictions for all 10 stock tickers simultaneously using the specified model."""
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "SBIN.NS", "TMPV.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"
    ]
    
    from models.predict import generate_next_day_prediction
    
    results = []
    target_date = date.today()
    
    for ticker in tickers:
        try:
            latest_record = crud.get_latest_price(db, ticker)
            if not latest_record:
                continue
                
            current_price = latest_record.close
            current_date = latest_record.date
            
            # Predict
            pred_close = generate_next_day_prediction(ticker, model_name, db)
            
            # Rec
            rec = get_recommendation(ticker, current_price, pred_close, model_name, db)
            
            # Next day target
            target_date = current_date + timedelta(days=1)
            if current_date.weekday() == 4:  # Friday
                target_date = current_date + timedelta(days=3)
            elif current_date.weekday() == 5:
                target_date = current_date + timedelta(days=2)
                
            results.append(schemas.ScreenerRow(
                ticker=ticker,
                current_price=current_price,
                predicted_price=pred_close,
                predicted_return=rec['predicted_return'],
                signal=rec['signal'],
                confidence=rec['confidence']
            ))
            
            # Also save to prediction database for logging
            pred_schema = schemas.PredictionBase(
                ticker=ticker,
                prediction_date=current_date,
                target_date=target_date,
                model_name=model_name,
                predicted_close=pred_close,
                actual_close=None,
                predicted_return=rec['predicted_return'],
                signal=rec['signal'],
                confidence=rec['confidence']
            )
            crud.save_prediction(db, pred_schema)
            
        except Exception as e:
            print(f"Screener warning: Failed to predict for {ticker} using {model_name}: {str(e)}")
            continue
            
    return schemas.ScreenerResponse(
        model_name=model_name,
        target_date=target_date,
        results=results
    )
