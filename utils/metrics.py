import numpy as np

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray) -> dict:
    """
    Calculates regression metrics and directional accuracy.
    
    Parameters:
        y_true: Actual next-day closing prices (shape: N,)
        y_pred: Predicted next-day closing prices (shape: N,)
        y_current: Current day closing prices (shape: N,)
        
    Returns:
        Dictionary containing MAE, RMSE, MAPE, R2, and Directional Accuracy.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_current = np.array(y_current)
    
    # Regression metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Directional Accuracy
    # Actual direction of price change: Close_t+1 - Close_t
    actual_change = y_true - y_current
    # Predicted direction of price change: Pred_Close_t+1 - Close_t
    predicted_change = y_pred - y_current
    
    # Sign comparison (1 for positive/neutral change, -1 for negative, 0 for no change)
    actual_sign = np.sign(actual_change)
    predicted_sign = np.sign(predicted_change)
    
    # Treat 0 (no change) as positive for directional comparison, or exact match
    directional_match = actual_sign == predicted_sign
    directional_accuracy = np.mean(directional_match) * 100
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "directional_accuracy": float(directional_accuracy)
    }
