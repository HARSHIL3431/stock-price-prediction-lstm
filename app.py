# app.py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

TIME_STEPS = 60

@st.cache_resource
def load_trained_model():
    model = load_model("lstm_stock_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def fetch_data(symbol, period="2y"):
    data = yf.download(symbol, period=period)
    data = data[["Close"]].dropna() #st.line_chart(data[["Close"]])

    return data

def prepare_last_sequence(close_prices, scaler, time_steps=TIME_STEPS):
    scaled = scaler.transform(close_prices)
    last_seq = scaled[-time_steps:]
    X_input = np.array(last_seq).reshape(1, time_steps, 1)
    return X_input

def main():
    st.title("📈 LSTM Stock Price Prediction App")
    st.write("Predict next-day closing price using a pre-trained LSTM model.")

    stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE.NS, TCS.NS, AAPL):", "RELIANCE.NS")

    period = st.selectbox(
        "Select data period for display (model was trained on historical data separately):",
        ["6mo", "1y", "2y", "5y"],
        index=2,
    )

    if st.button("Predict"):
        with st.spinner("Loading model and fetching data..."):
            model, scaler = load_trained_model()
            data = fetch_data(stock_symbol, period=period)

        if len(data) < TIME_STEPS:
            st.error(f"Not enough data points ({len(data)}) for TIME_STEPS={TIME_STEPS}. Try a different period or stock.")
            return

        st.subheader(f"Historical Close Price for {stock_symbol}")
        st.line_chart(data["Close"])

        X_input = prepare_last_sequence(data[["Close"]].values, scaler, TIME_STEPS)
        pred_scaled = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        last_actual_price = data["Close"].iloc[-1].item()

        st.subheader("Prediction Result")
        st.write(f"**Last Available Close Price:** {float(last_actual_price):.2f}")
        st.write(f"**Predicted Next Close Price:** {pred_price:.2f}")

        plot_df = pd.DataFrame(
            {
                "Last Price": [last_actual_price],
                "Predicted Next Price": [pred_price],
            },
            index=["Price"],
        )
        st.bar_chart(plot_df.T)

        st.info(
            "Note: This model is for educational and analytical purposes only, "
            "not for real-money trading decisions."
        )

if __name__ == "__main__":
    main()
