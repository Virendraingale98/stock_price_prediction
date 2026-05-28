from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Optional

# --- Historical Prices Schemas ---
class HistoricalPriceBase(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

class HistoricalPrice(HistoricalPriceBase):
    id: int

    class Config:
        from_attributes = True

# --- Model Metrics Schemas ---
class ModelMetricBase(BaseModel):
    ticker: str
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float

class ModelMetric(ModelMetricBase):
    id: int
    trained_at: datetime

    class Config:
        from_attributes = True

# --- Predictions Schemas ---
class PredictionBase(BaseModel):
    ticker: str
    prediction_date: date
    target_date: date
    model_name: str
    predicted_close: float
    actual_close: Optional[float] = None
    predicted_return: float
    signal: str
    confidence: float

class Prediction(PredictionBase):
    id: int

    class Config:
        from_attributes = True

# --- Request / Response Schemas ---
class SyncRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    start_date: Optional[str] = "2018-01-01"
    end_date: Optional[str] = None

class SyncResponse(BaseModel):
    status: str
    message: str
    synced_records: int

class TrainRequest(BaseModel):
    ticker: str

class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: List[ModelMetricBase]

class PredictionRequest(BaseModel):
    ticker: str
    model_name: str = "Random Forest"

class ScreenerRow(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predicted_return: float
    signal: str
    confidence: float

class ScreenerResponse(BaseModel):
    model_name: str
    target_date: date
    results: List[ScreenerRow]
