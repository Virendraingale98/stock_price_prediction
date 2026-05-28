from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import date
from typing import List, Dict, Any, Optional
from backend import models, schemas

# --- Historical Prices CRUD ---

def get_historical_prices(
    db: Session, ticker: str, start_date: Optional[date] = None, end_date: Optional[date] = None
) -> List[models.HistoricalPrice]:
    """Retrieve historical prices for a given ticker and date range."""
    query = db.query(models.HistoricalPrice).filter(models.HistoricalPrice.ticker == ticker)
    if start_date:
        query = query.filter(models.HistoricalPrice.date >= start_date)
    if end_date:
        query = query.filter(models.HistoricalPrice.date <= end_date)
    return query.order_by(models.HistoricalPrice.date.asc()).all()

def get_latest_price(db: Session, ticker: str) -> Optional[models.HistoricalPrice]:
    """Retrieve the most recent historical price record for a ticker."""
    return db.query(models.HistoricalPrice)\
        .filter(models.HistoricalPrice.ticker == ticker)\
        .order_by(desc(models.HistoricalPrice.date))\
        .first()

def bulk_insert_prices(db: Session, prices: List[Dict[str, Any]]) -> int:
    """Insert a list of historical prices in bulk, ignoring duplicates."""
    if not prices:
        return 0
    
    # Use SQLite-specific insert statement to handle conflicts (ON CONFLICT DO NOTHING)
    stmt = sqlite_insert(models.HistoricalPrice).values(prices)
    stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'date'])
    
    result = db.execute(stmt)
    db.commit()
    return result.rowcount

# --- Model Metrics CRUD ---

def save_model_metric(db: Session, metric: schemas.ModelMetricBase) -> models.ModelMetric:
    """Save model evaluation metrics to the database."""
    # Delete old metrics for this model and ticker before inserting new ones
    db.query(models.ModelMetric).filter(
        models.ModelMetric.ticker == metric.ticker,
        models.ModelMetric.model_name == metric.model_name
    ).delete()
    
    db_metric = models.ModelMetric(
        ticker=metric.ticker,
        model_name=metric.model_name,
        mae=metric.mae,
        rmse=metric.rmse,
        mape=metric.mape,
        r2=metric.r2,
        directional_accuracy=metric.directional_accuracy
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric

def get_model_metrics(db: Session, ticker: str) -> List[models.ModelMetric]:
    """Retrieve all model metrics for a ticker."""
    return db.query(models.ModelMetric).filter(models.ModelMetric.ticker == ticker).all()

# --- Predictions CRUD ---

def save_prediction(db: Session, pred: schemas.PredictionBase) -> models.Prediction:
    """Save a next-day stock price prediction."""
    # Upsert: check if prediction for ticker, target_date, and model_name already exists
    db_pred = db.query(models.Prediction).filter(
        models.Prediction.ticker == pred.ticker,
        models.Prediction.target_date == pred.target_date,
        models.Prediction.model_name == pred.model_name
    ).first()
    
    if db_pred:
        # Update existing
        db_pred.prediction_date = pred.prediction_date
        db_pred.predicted_close = pred.predicted_close
        db_pred.predicted_return = pred.predicted_return
        db_pred.signal = pred.signal
        db_pred.confidence = pred.confidence
        if pred.actual_close is not None:
            db_pred.actual_close = pred.actual_close
    else:
        # Create new
        db_pred = models.Prediction(
            ticker=pred.ticker,
            prediction_date=pred.prediction_date,
            target_date=pred.target_date,
            model_name=pred.model_name,
            predicted_close=pred.predicted_close,
            actual_close=pred.actual_close,
            predicted_return=pred.predicted_return,
            signal=pred.signal,
            confidence=pred.confidence
        )
        db.add(db_pred)
        
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_latest_prediction(db: Session, ticker: str, model_name: str) -> Optional[models.Prediction]:
    """Retrieve the latest prediction for a ticker and model."""
    return db.query(models.Prediction)\
        .filter(models.Prediction.ticker == ticker, models.Prediction.model_name == model_name)\
        .order_by(desc(models.Prediction.prediction_date))\
        .first()

def get_predictions_history(db: Session, ticker: str, model_name: str) -> List[models.Prediction]:
    """Retrieve all historical predictions for a ticker and model, ordered by target date."""
    return db.query(models.Prediction)\
        .filter(models.Prediction.ticker == ticker, models.Prediction.model_name == model_name)\
        .order_by(models.Prediction.target_date.asc())\
        .all()

def update_actual_prices(db: Session, ticker: str, date_prices: Dict[date, float]) -> int:
    """Update existing predictions with actual close prices once they become available."""
    updated = 0
    predictions_to_update = db.query(models.Prediction).filter(
        models.Prediction.ticker == ticker,
        models.Prediction.actual_close == None,
        models.Prediction.target_date.in_(list(date_prices.keys()))
    ).all()
    
    for pred in predictions_to_update:
        if pred.target_date in date_prices:
            pred.actual_close = date_prices[pred.target_date]
            updated += 1
            
    if updated > 0:
        db.commit()
    return updated
