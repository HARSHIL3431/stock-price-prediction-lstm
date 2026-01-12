"""
Technical Indicators Module
Computes various technical indicators for stock analysis
"""

import pandas as pd
import numpy as np


def calculate_sma(data, window=20):
    """
    Calculate Simple Moving Average
    
    Args:
        data: pandas Series of prices
        window: int, window size for moving average
        
    Returns:
        pandas Series with SMA values
    """
    return data.rolling(window=window).mean()


def calculate_ema(data, span=20):
    """
    Calculate Exponential Moving Average
    
    Args:
        data: pandas Series of prices
        span: int, span for EMA calculation
        
    Returns:
        pandas Series with EMA values
    """
    return data.ewm(span=span, adjust=False).mean()


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: pandas Series of prices
        period: int, RSI period (typically 14)
        
    Returns:
        pandas Series with RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: pandas Series of prices
        fast: int, fast EMA period
        slow: int, slow EMA period
        signal: int, signal line period
        
    Returns:
        tuple of (MACD line, Signal line, MACD Histogram)
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram


def calculate_volume_ma(volume, window=20):
    """
    Calculate Volume Moving Average
    
    Args:
        volume: pandas Series of volume data
        window: int, window size for moving average
        
    Returns:
        pandas Series with volume MA
    """
    return volume.rolling(window=window).mean()


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: pandas Series of prices
        window: int, window size for moving average
        num_std: int, number of standard deviations
        
    Returns:
        tuple of (upper band, middle band, lower band)
    """
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band


def add_all_indicators(df):
    """
    Add all technical indicators to a DataFrame containing OHLCV data
    
    Args:
        df: pandas DataFrame with columns: Open, High, Low, Close, Volume
        
    Returns:
        pandas DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = calculate_sma(df['Close'], window=20)
    df['SMA_50'] = calculate_sma(df['Close'], window=50)
    df['EMA_20'] = calculate_ema(df['Close'], span=20)
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    
    # MACD
    macd_line, signal_line, macd_histogram = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = macd_histogram
    
    # Volume MA
    df['Volume_MA'] = calculate_volume_ma(df['Volume'], window=20)
    
    # Bollinger Bands
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper_bb
    df['BB_Middle'] = middle_bb
    df['BB_Lower'] = lower_bb
    
    return df


def get_indicator_summary(df):
    """
    Get a summary of the latest indicator values
    
    Args:
        df: pandas DataFrame with calculated indicators
        
    Returns:
        dict with latest indicator values
    """
    if df.empty or len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    
    summary = {
        'Close': latest['Close'],
        'SMA_20': latest.get('SMA_20', None),
        'SMA_50': latest.get('SMA_50', None),
        'EMA_20': latest.get('EMA_20', None),
        'RSI': latest.get('RSI', None),
        'MACD': latest.get('MACD', None),
        'MACD_Signal': latest.get('MACD_Signal', None),
        'Volume': latest['Volume'],
        'Volume_MA': latest.get('Volume_MA', None),
    }
    
    return {k: v for k, v in summary.items() if v is not None and not pd.isna(v)}
