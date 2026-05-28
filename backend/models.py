from sqlalchemy import Column, Integer, Float, String, Date, DateTime, UniqueConstraint, BigInteger
from sqlalchemy.sql import func
from backend.database import Base

class HistoricalPrice(Base):
    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uix_ticker_date"),
    )

class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mape = Column(Float, nullable=False)
    r2 = Column(Float, nullable=False)
    directional_accuracy = Column(Float, nullable=False)
    trained_at = Column(DateTime, default=func.now(), nullable=False)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    prediction_date = Column(Date, index=True, nullable=False)
    target_date = Column(Date, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    predicted_close = Column(Float, nullable=False)
    actual_close = Column(Float, nullable=True)  # Updated once the actual price is known
    predicted_return = Column(Float, nullable=False)
    signal = Column(String, nullable=False)       # BUY, HOLD, SELL
    confidence = Column(Float, nullable=False)    # 0.0 to 100.0
