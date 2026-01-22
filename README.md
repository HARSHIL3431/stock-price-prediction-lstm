# 📈 AI Stock Market Analysis & Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![AI](https://img.shields.io/badge/AI-Explainable-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)

> **Educational AI-powered decision support system for stock market analysis**  
> ⚠️ *For learning and research purposes only – Not financial advice*

---

## 🚀 Project Overview

This project is a **professional-grade stock market analysis and prediction dashboard** that combines **deep learning (LSTM)** with **technical analysis** to provide **explainable Buy / Sell / Hold insights**.

Unlike typical student projects that predict prices for a single stock using only closing prices, this system focuses on:

- Real-world market visualization (candlestick charts)
- Explainable AI-driven signals
- Model performance evaluation
- Ethical, non-autonomous decision support

Inspired by platforms like **TradingView** and **Yahoo Finance**, this project is built **strictly for educational purposes**.

---

## 🎯 Key Features

### 📊 Market Visualization
- Professional **candlestick charts**
- Volume bars
- Moving averages overlay (SMA 20, SMA 50)
- Clean TradingView-style layout

### 🤖 AI Price Prediction
- LSTM-based time series forecasting
- Next-day closing price prediction
- Optional multi-day forecast
- Actual vs predicted visualization

### 📈 Technical Indicators
- Relative Strength Index (RSI)
- Moving Averages (SMA, EMA)
- MACD
- Volume Moving Average

### 🎯 Explainable Trading Signals
- Rule-based **BUY / SELL / HOLD** signals
- Confidence levels (Low / Medium / High)
- Human-readable explanations
- Indicator + AI reasoning (no black-box output)

### 📊 Model Evaluation
- RMSE
- MAE
- Directional Accuracy
- Visual comparison of predictions vs actual prices

### ⚠️ Ethical AI & Safety
- Clear disclaimer
- No automated trading
- No profit guarantees
- Decision-support only

---

## 🧠 System Architecture

| Component | Description |
|---------|------------|
| `app.py` | Streamlit dashboard and UI |
| `train_model.py` | LSTM model training |
| `requirements.txt` | Python dependencies |
| `scaler.pkl` | Saved data scaler |
| `lstm_stock_model.h5` | Trained LSTM model |
| `utils/data_handler.py` | Stock data fetching & preprocessing |
| `utils/indicators.py` | Technical indicator calculations |
| `utils/signals.py` | Buy / Sell / Hold logic |
| `utils/model_utils.py` | Predictions & evaluation |
| `utils/visualizations.py` | Charts & plots |


---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Deep Learning | TensorFlow (LSTM) |
| Data Source | yfinance |
| ML Utilities | scikit-learn |
| Visualization | Plotly, Matplotlib |
| Web App | Streamlit |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


Open the URL shown in the terminal (usually http://localhost:8501).

🧪 Model Training (Optional)

To retrain the model:

python train_model.py


This will:

Download historical stock data

Train an LSTM model

Save the trained model and scaler

📌 How Trading Signals Work
BUY Signal

RSI indicates oversold conditions

Price above key moving averages

Bullish MACD crossover

AI predicts upward movement

SELL Signal

RSI indicates overbought conditions

Price below key moving averages

Bearish MACD crossover

AI predicts downward movement

HOLD

Mixed or weak signals

Unclear market trend

Each signal includes:

Confidence level

Indicator-based explanation

AI reasoning

📊 Model Evaluation Metrics

RMSE – prediction error magnitude

MAE – average absolute error

Directional Accuracy – correctness of price movement direction

Metrics are displayed directly in the dashboard.

⚠️ Disclaimer

This project is intended strictly for educational and research purposes.
It does not provide financial advice and must not be used for real trading decisions.
Stock markets involve risk, and past performance does not guarantee future results.

🚧 Current Limitations

Model trained on limited stock data

Short-term prediction focus

No live trading integration

Does not consider news or macroeconomic events

🔮 Future Improvements

Multi-stock & multi-market training

Directional prediction (up/down/flat)

Walk-forward validation

Backtesting engine

Market regime detection

Attention-based deep learning models

👨‍🎓 Author

Harshil Thakkar
B.Tech – Artificial Intelligence & Machine Learning

⭐ Why This Project Matters

This project demonstrates:

Time-series machine learning

Explainable AI principles

Responsible AI usage

Clean software engineering

Real-world dashboard design

Suitable for:

College evaluation

GitHub portfolio

ML internships & entry-level roles
