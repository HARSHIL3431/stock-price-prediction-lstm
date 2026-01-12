"""
Model Utilities Module
Handles model loading, prediction, and evaluation metrics
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_trained_model(model_path="lstm_stock_model.h5", scaler_path="scaler.pkl"):
    """
    Load pre-trained LSTM model and scaler
    
    Args:
        model_path: str, path to saved model
        scaler_path: str, path to saved scaler
        
    Returns:
        tuple of (model, scaler)
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def prepare_sequence(data, scaler, time_steps=60):
    """
    Prepare input sequence for LSTM prediction
    
    Args:
        data: numpy array or pandas Series of close prices
        scaler: fitted MinMaxScaler
        time_steps: int, number of time steps for LSTM
        
    Returns:
        numpy array shaped for LSTM input
    """
    if isinstance(data, pd.Series):
        data = data.values.reshape(-1, 1)
    elif isinstance(data, np.ndarray) and len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    scaled = scaler.transform(data)
    last_seq = scaled[-time_steps:]
    X_input = np.array(last_seq).reshape(1, time_steps, 1)
    return X_input


def predict_next_price(model, scaler, close_prices, time_steps=60):
    """
    Predict the next closing price
    
    Args:
        model: trained Keras model
        scaler: fitted MinMaxScaler
        close_prices: array-like of historical close prices
        time_steps: int, number of time steps for LSTM
        
    Returns:
        float: predicted next closing price
    """
    X_input = prepare_sequence(close_prices, scaler, time_steps)
    pred_scaled = model.predict(X_input, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    return pred_price


def predict_multiple_days(model, scaler, close_prices, time_steps=60, days=5):
    """
    Predict multiple days ahead (iterative approach)
    
    Args:
        model: trained Keras model
        scaler: fitted MinMaxScaler
        close_prices: array-like of historical close prices
        time_steps: int, number of time steps for LSTM
        days: int, number of days to predict
        
    Returns:
        list of predicted prices
    """
    predictions = []
    current_sequence = close_prices.copy()
    
    for _ in range(days):
        next_pred = predict_next_price(model, scaler, current_sequence, time_steps)
        predictions.append(next_pred)
        # Append prediction to sequence for next iteration
        current_sequence = np.append(current_sequence, next_pred)
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for predictions
    
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
        
    Returns:
        dict with RMSE, MAE, and MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy (how often we predicted the right direction)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = None
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }


def evaluate_model_on_recent_data(model, scaler, df, time_steps=60, test_days=30):
    """
    Evaluate model performance on recent historical data
    
    Args:
        model: trained Keras model
        scaler: fitted MinMaxScaler
        df: pandas DataFrame with Close prices
        time_steps: int, number of time steps for LSTM
        test_days: int, number of recent days to test on
        
    Returns:
        dict with metrics and predictions
    """
    if len(df) < time_steps + test_days:
        return None
    
    close_prices = df['Close'].values
    y_true = []
    y_pred = []
    
    # Test on last N days
    for i in range(-test_days, 0):
        historical_data = close_prices[:i]
        if len(historical_data) >= time_steps:
            pred = predict_next_price(model, scaler, historical_data, time_steps)
            actual = close_prices[i]
            y_pred.append(pred)
            y_true.append(actual)
    
    if len(y_true) > 0:
        metrics = calculate_metrics(y_true, y_pred)
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_true
        }
    
    return None


def format_metrics_display(metrics):
    """
    Format metrics for display
    
    Args:
        metrics: dict with metric values
        
    Returns:
        str: formatted metrics display
    """
    if metrics is None:
        return "Metrics not available"
    
    display = "**Model Performance Metrics:**\n\n"
    
    if 'RMSE' in metrics:
        display += f"- **RMSE**: ${metrics['RMSE']:.2f}\n"
    if 'MAE' in metrics:
        display += f"- **MAE**: ${metrics['MAE']:.2f}\n"
    if 'MAPE' in metrics:
        display += f"- **MAPE**: {metrics['MAPE']:.2f}%\n"
    if 'Directional_Accuracy' in metrics and metrics['Directional_Accuracy'] is not None:
        display += f"- **Directional Accuracy**: {metrics['Directional_Accuracy']:.2f}%\n"
    
    display += "\n*RMSE = Root Mean Squared Error*\n"
    display += "*MAE = Mean Absolute Error*\n"
    display += "*MAPE = Mean Absolute Percentage Error*\n"
    display += "*Directional Accuracy = % of time model predicted correct price direction*"
    
    return display
