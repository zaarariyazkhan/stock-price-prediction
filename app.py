import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction using Machine Learning")
st.markdown("Predicting stock closing prices using Linear Regression, Random Forest, LSTM and GRU models.")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Stock", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "LSTM", "GRU"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

if st.sidebar.button("Run Prediction"):
    with st.spinner("Downloading data and training model..."):

        # Download data
        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        # Feature engineering
        df['SMA_7'] = SMAIndicator(close=df['Close'], window=7).sma_indicator()
        df['SMA_21'] = SMAIndicator(close=df['Close'], window=21).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Close_lag5'] = df['Close'].shift(5)
        df['Return'] = df['Close'].pct_change()
        df['Target_Return'] = df['Return'].shift(-1)
        df.dropna(inplace=True)

        # Show raw data
        st.subheader(f"{ticker} — Closing Price")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df['Close'], color='steelblue')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (INR)")
        ax1.set_title(f"{ticker} Closing Price")
        st.pyplot(fig1)

        actual_prices = None
        predicted_prices = None
        rmse = None
        r2 = None

        if model_choice in ["Linear Regression", "Random Forest"]:
            features = ['SMA_7', 'SMA_21', 'RSI', 'MACD', 'MACD_Signal',
                        'BB_High', 'BB_Low', 'Close_lag1', 'Close_lag2',
                        'Close_lag5', 'Return']

            X = df[features].values
            y_price = df['Close'].values

            scaler_X = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)

            split = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train = df['Target_Return'].values[:split]

            actual_prices = y_price[split:]
            prev_prices = y_price[split-1:-1]

            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

            model.fit(X_train, y_train)
            ret_pred = model.predict(X_test)
            predicted_prices = prev_prices * (1 + ret_pred)

        else:
            lookback = 60
            scaler = MinMaxScaler()
            close_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

            X_seq, y_seq = [], []
            for i in range(lookback, len(close_scaled)):
                X_seq.append(close_scaled[i-lookback:i])
                y_seq.append(close_scaled[i])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            split = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]

            if model_choice == "LSTM":
                nn_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])
            else:
                nn_model = Sequential([
                    GRU(50, return_sequences=True, input_shape=(lookback, 1)),
                    Dropout(0.2),
                    GRU(50),
                    Dropout(0.2),
                    Dense(1)
                ])

            nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            nn_model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.1, verbose=0)

            pred_scaled = nn_model.predict(X_test)
            predicted_prices = scaler.inverse_transform(pred_scaled)
            actual_prices = scaler.inverse_transform(y_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        r2 = r2_score(actual_prices, predicted_prices)

        # Results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")

        # Plot predictions
        st.subheader("Predicted vs Actual Price")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(actual_prices, label='Actual', color='black')
        ax2.plot(predicted_prices, label=f'{model_choice} Predicted', color='red', alpha=0.8)
        ax2.set_xlabel("Trading Days")
        ax2.set_ylabel("Price (INR)")
        ax2.legend()
        st.pyplot(fig2)

        st.success("Prediction complete!")
