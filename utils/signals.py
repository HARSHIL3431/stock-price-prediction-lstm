"""
Trading Signal Logic Module
Generates BUY/SELL/HOLD signals with explainability
"""

import pandas as pd
import numpy as np


def calculate_confidence_level(signals_count):
    """
    Calculate confidence level based on number of aligned signals
    
    Args:
        signals_count: int, number of positive signals
        
    Returns:
        str: 'High', 'Medium', or 'Low'
    """
    if signals_count >= 3:
        return "High"
    elif signals_count >= 2:
        return "Medium"
    else:
        return "Low"


def generate_trading_signal(indicators, predicted_price, current_price):
    """
    Generate trading signal based on indicators and AI prediction
    
    Args:
        indicators: dict with indicator values
        predicted_price: float, AI predicted next price
        current_price: float, current stock price
        
    Returns:
        dict with signal, confidence, and explanation
    """
    buy_signals = []
    sell_signals = []
    explanations = []
    
    # Get indicator values safely
    rsi = indicators.get('RSI', None)
    sma_20 = indicators.get('SMA_20', None)
    sma_50 = indicators.get('SMA_50', None)
    macd = indicators.get('MACD', None)
    macd_signal = indicators.get('MACD_Signal', None)
    
    # 1. RSI Analysis
    if rsi is not None:
        if rsi < 30:
            buy_signals.append("RSI oversold")
            explanations.append(f"📊 RSI is {rsi:.2f} (below 30), indicating oversold conditions - potential buying opportunity")
        elif rsi > 70:
            sell_signals.append("RSI overbought")
            explanations.append(f"📊 RSI is {rsi:.2f} (above 70), indicating overbought conditions - consider taking profits")
        else:
            explanations.append(f"📊 RSI is {rsi:.2f} (neutral zone 30-70)")
    
    # 2. Moving Average Analysis
    if sma_50 is not None and current_price is not None:
        if current_price > sma_50:
            buy_signals.append("Price above SMA-50")
            explanations.append(f"📈 Price ${current_price:.2f} is above 50-day MA ${sma_50:.2f} (bullish trend)")
        else:
            sell_signals.append("Price below SMA-50")
            explanations.append(f"📉 Price ${current_price:.2f} is below 50-day MA ${sma_50:.2f} (bearish trend)")
    
    # 3. Golden Cross / Death Cross (SMA 20 vs SMA 50)
    if sma_20 is not None and sma_50 is not None:
        if sma_20 > sma_50:
            buy_signals.append("Golden Cross pattern")
            explanations.append(f"✨ Golden Cross: 20-day MA above 50-day MA (bullish signal)")
        else:
            sell_signals.append("Death Cross pattern")
            explanations.append(f"⚠️ Death Cross: 20-day MA below 50-day MA (bearish signal)")
    
    # 4. MACD Analysis
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            buy_signals.append("MACD bullish")
            explanations.append(f"📊 MACD ({macd:.2f}) above signal line ({macd_signal:.2f}) - bullish momentum")
        else:
            sell_signals.append("MACD bearish")
            explanations.append(f"📊 MACD ({macd:.2f}) below signal line ({macd_signal:.2f}) - bearish momentum")
    
    # 5. AI Prediction Analysis
    if predicted_price is not None and current_price is not None:
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        if predicted_price > current_price * 1.01:  # More than 1% increase
            buy_signals.append("AI predicts upward movement")
            explanations.append(f"🤖 AI predicts price increase to ${predicted_price:.2f} (+{price_change_pct:.2f}%)")
        elif predicted_price < current_price * 0.99:  # More than 1% decrease
            sell_signals.append("AI predicts downward movement")
            explanations.append(f"🤖 AI predicts price decrease to ${predicted_price:.2f} ({price_change_pct:.2f}%)")
        else:
            explanations.append(f"🤖 AI predicts minimal change to ${predicted_price:.2f} ({price_change_pct:+.2f}%)")
    
    # Determine final signal
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    
    if buy_count > sell_count and buy_count >= 2:
        signal = "BUY"
        confidence = calculate_confidence_level(buy_count)
        summary = f"**{buy_count}** bullish signals detected vs **{sell_count}** bearish signals."
    elif sell_count > buy_count and sell_count >= 2:
        signal = "SELL"
        confidence = calculate_confidence_level(sell_count)
        summary = f"**{sell_count}** bearish signals detected vs **{buy_count}** bullish signals."
    else:
        signal = "HOLD"
        confidence = "Low" if abs(buy_count - sell_count) <= 1 else "Medium"
        summary = f"Mixed signals: **{buy_count}** bullish, **{sell_count}** bearish. Wait for clearer trend."
    
    return {
        'signal': signal,
        'confidence': confidence,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'explanations': explanations,
        'summary': summary,
        'buy_count': buy_count,
        'sell_count': sell_count
    }


def get_signal_color(signal):
    """
    Get color code for signal type
    
    Args:
        signal: str, 'BUY', 'SELL', or 'HOLD'
        
    Returns:
        str: color name
    """
    colors = {
        'BUY': 'green',
        'SELL': 'red',
        'HOLD': 'orange'
    }
    return colors.get(signal, 'gray')


def get_signal_emoji(signal):
    """
    Get emoji for signal type
    
    Args:
        signal: str, 'BUY', 'SELL', or 'HOLD'
        
    Returns:
        str: emoji
    """
    emojis = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡'
    }
    return emojis.get(signal, '⚪')


def format_signal_display(signal_data):
    """
    Format signal data for display
    
    Args:
        signal_data: dict from generate_trading_signal
        
    Returns:
        str: formatted signal display
    """
    signal = signal_data['signal']
    confidence = signal_data['confidence']
    emoji = get_signal_emoji(signal)
    
    display = f"{emoji} **{signal}** Signal - Confidence: **{confidence}**\n\n"
    display += signal_data['summary'] + "\n\n"
    display += "**Detailed Analysis:**\n"
    
    for explanation in signal_data['explanations']:
        display += f"- {explanation}\n"
    
    return display
