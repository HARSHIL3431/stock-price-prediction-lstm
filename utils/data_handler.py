"""
Data Handler Module
Handles data fetching, preprocessing, and validation
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(symbol, period="2y", interval="1d"):
    """
    Fetch OHLCV stock data from Yahoo Finance
    
    Args:
        symbol: str, stock ticker symbol
        period: str, data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: str, data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None
        
        # Remove rows with missing data
        data = data[required_cols].dropna()
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def validate_stock_symbol(symbol):
    """
    Validate if a stock symbol exists and has data
    
    Args:
        symbol: str, stock ticker symbol
        
    Returns:
        tuple: (bool, str) - (is_valid, message)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Check if we got valid info
        if not info or 'regularMarketPrice' not in info:
            # Try fetching recent data as fallback
            data = fetch_stock_data(symbol, period="5d")
            if data is not None and len(data) > 0:
                return True, "Valid symbol"
            return False, f"No data available for symbol '{symbol}'"
        
        return True, "Valid symbol"
    
    except Exception as e:
        return False, f"Invalid symbol '{symbol}': {str(e)}"


def get_stock_info(symbol):
    """
    Get basic information about a stock
    
    Args:
        symbol: str, stock ticker symbol
        
    Returns:
        dict with stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'market_cap': info.get('marketCap', None),
            'exchange': info.get('exchange', 'N/A')
        }
    except:
        return {
            'symbol': symbol,
            'name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'currency': 'USD',
            'market_cap': None,
            'exchange': 'N/A'
        }


def prepare_data_for_display(df, last_n_days=None):
    """
    Prepare data for display and visualization
    
    Args:
        df: pandas DataFrame with stock data
        last_n_days: int, optional, show only last N days
        
    Returns:
        pandas DataFrame
    """
    if df is None or df.empty:
        return None
    
    df_display = df.copy()
    
    if last_n_days is not None and last_n_days > 0:
        df_display = df_display.tail(last_n_days)
    
    return df_display


def calculate_price_change(df):
    """
    Calculate price change statistics
    
    Args:
        df: pandas DataFrame with Close prices
        
    Returns:
        dict with price change statistics
    """
    if df is None or df.empty or len(df) < 2:
        return None
    
    current_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2]
    
    # Day change
    day_change = current_price - previous_price
    day_change_pct = (day_change / previous_price) * 100
    
    # Period change
    first_price = df['Close'].iloc[0]
    period_change = current_price - first_price
    period_change_pct = (period_change / first_price) * 100
    
    # High and Low in period
    period_high = df['High'].max()
    period_low = df['Low'].min()
    
    return {
        'current_price': current_price,
        'day_change': day_change,
        'day_change_pct': day_change_pct,
        'period_change': period_change,
        'period_change_pct': period_change_pct,
        'period_high': period_high,
        'period_low': period_low,
        'avg_volume': df['Volume'].mean()
    }


def handle_missing_data(df):
    """
    Handle missing data in DataFrame
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame with handled missing data
    """
    if df is None or df.empty:
        return df
    
    # Forward fill then backward fill for any remaining NaNs
    df = df.ffill().bfill()
    
    # If still has NaNs, drop those rows
    df = df.dropna()
    
    return df


def get_data_quality_report(df):
    """
    Generate a data quality report
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict with data quality metrics
    """
    if df is None or df.empty:
        return {'status': 'No data'}
    
    total_rows = len(df)
    missing_values = df.isnull().sum().to_dict()
    
    return {
        'status': 'OK' if total_rows > 0 else 'No data',
        'total_rows': total_rows,
        'missing_values': missing_values,
        'date_range': {
            'start': df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None,
            'end': df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else None
        }
    }
