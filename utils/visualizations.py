"""
Visualization Module
Creates professional candlestick charts and other visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_candlestick_chart(df, stock_symbol, show_volume=True, show_ma=True):
    """
    Create a professional candlestick chart with volume and moving averages
    
    Args:
        df: pandas DataFrame with OHLCV data and indicators
        stock_symbol: str, stock ticker symbol
        show_volume: bool, whether to show volume subplot
        show_ma: bool, whether to show moving averages
        
    Returns:
        plotly Figure object
    """
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{stock_symbol} Price Chart', 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Add candlestick chart
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add moving averages if available and requested
    if show_ma:
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        if 'EMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA_20'],
                    name='EMA 20',
                    line=dict(color='purple', width=1.5, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # Add volume bars if requested
    if show_volume:
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Add volume moving average if available
        if 'Volume_MA' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Volume_MA'],
                    name='Volume MA',
                    line=dict(color='orange', width=1.5),
                    opacity=0.7
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'{stock_symbol} - Market Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=700 if show_volume else 600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Update x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='#1e1e1e',
            activecolor='#2e2e2e'
        )
    )
    
    return fig


def create_rsi_chart(df):
    """
    Create RSI indicator chart
    
    Args:
        df: pandas DataFrame with RSI column
        
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, 
                     annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5,
                     annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_macd_chart(df):
    """
    Create MACD indicator chart
    
    Args:
        df: pandas DataFrame with MACD columns
        
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='orange', width=2)
            )
        )
        
        if 'MACD_Histogram' in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5
                )
            )
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        xaxis_title='Date',
        yaxis_title='MACD',
        template='plotly_dark',
        height=300
    )
    
    return fig


def create_prediction_chart(df, predicted_price, predicted_days=None):
    """
    Create a chart showing actual prices and prediction
    
    Args:
        df: pandas DataFrame with historical close prices
        predicted_price: float, predicted next price
        predicted_days: list, optional multi-day predictions
        
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Historical Price',
            line=dict(color='#26a69a', width=2),
            mode='lines'
        )
    )
    
    # Next day prediction
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    
    fig.add_trace(
        go.Scatter(
            x=[last_date, next_date],
            y=[df['Close'].iloc[-1], predicted_price],
            name='Next Day Prediction',
            line=dict(color='orange', width=3, dash='dash'),
            mode='lines+markers',
            marker=dict(size=10)
        )
    )
    
    # Multi-day predictions if provided
    if predicted_days is not None and len(predicted_days) > 0:
        future_dates = pd.date_range(start=next_date, periods=len(predicted_days), freq='D')
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predicted_days,
                name=f'{len(predicted_days)}-Day Forecast',
                line=dict(color='red', width=2, dash='dot'),
                mode='lines+markers',
                marker=dict(size=8)
            )
        )
    
    fig.update_layout(
        title='Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_actual_vs_predicted_chart(actuals, predictions):
    """
    Create a chart comparing actual vs predicted prices
    
    Args:
        actuals: array-like of actual prices
        predictions: array-like of predicted prices
        
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    x_range = list(range(len(actuals)))
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=actuals,
            name='Actual',
            line=dict(color='green', width=2),
            mode='lines+markers'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=predictions,
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash'),
            mode='lines+markers'
        )
    )
    
    fig.update_layout(
        title='Model Performance: Actual vs Predicted',
        xaxis_title='Test Sample',
        yaxis_title='Price',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig
