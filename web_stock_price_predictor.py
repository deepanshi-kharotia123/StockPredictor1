import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 📌 Stock Price Predictor App
st.title("📈 Stock Price Predictor App")

# 🔹 User Input for Stock Symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# 🔹 Fetch and Display Historical Stock Data (20 Years)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

# ✅ Validate Data Retrieval
if google_data.empty:
    st.error("⚠️ No stock data retrieved. Please check the stock ticker or try again later.")
    st.stop()
if 'Close' not in google_data.columns:
    st.error("⚠️ 'Close' column missing in the dataset. Please check the data format.")
    st.stop()

# 🔹 Load Pretrained Stock Prediction Model
model = load_model("Latest_stock_price_model.keras")

# 📊 Display Retrieved Stock Data
st.subheader("📊 Historical Stock Data from yfinance")
st.write(google_data)

# 🔹 Train-Test Data Splitting
splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']].iloc[splitting_len:].copy()

if x_test.empty:
    st.error("⚠️ Test data is empty after splitting. Check your dataset.")
    st.stop()

# 📌 Function to Plot Graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label='Moving Average')
    plt.plot(full_data['Close'], 'b', label='Original Close Price')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, label='Extra Data', linestyle='dashed')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Trends')
    return fig

# 🔹 Moving Averages Analysis
st.subheader('📈 Stock Price Trends with Moving Averages')
for days in [250, 200, 100]:
    google_data[f'MA_for_{days}_days'] = google_data['Close'].rolling(days).mean()
    st.pyplot(plot_graph((15, 6), google_data[f'MA_for_{days}_days'], google_data))

# 🔹 Comparison of MA (100 & 250 Days)
st.subheader('📊 Moving Averages (100 vs 250 Days)')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# 🔹 Data Preprocessing: MinMax Scaling
#st.subheader("⚙️ Data Preprocessing")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

# 🔹 Prepare Data for Model Prediction
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# 📌 Stock Price Prediction Using Model
st.subheader("🔮 Stock Price Prediction")
predictions = model.predict(x_data)

# 🔹 Transform Predictions Back to Original Scale
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# 📊 Create DataFrame for Original vs Predicted Values
st.subheader("📊 Original vs Predicted Stock Prices")
ploting_data = pd.DataFrame(
    {'Original Test Data': inv_y_test.reshape(-1), 'Predictions': inv_pre.reshape(-1)},
    index=google_data.index[splitting_len + 100:]
)
st.write(ploting_data)

fig = plt.figure(figsize=(15, 6))
plt.plot(google_data.index[:splitting_len+100], google_data['Close'][:splitting_len+100], label="Data - Not Used")
plt.plot(ploting_data.index, ploting_data['Original Test Data'], label="Original Test Data", linestyle='dashed')
plt.plot(ploting_data.index, ploting_data['Predictions'], label="Predicted Test Data", linestyle='dotted')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Comparison of Original vs Predicted Stock Prices')
st.pyplot(fig)

# 🔮 Future Stock Price Predictions
st.subheader("📈 Forecasting Future Stock Prices")
num_future_days = 30
last_100_days = scaled_data[-100:]
future_predictions = []

for i in range(num_future_days):
    X_future = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    future_price = model.predict(X_future)
    future_predictions.append(future_price[0, 0])
    last_100_days = np.append(last_100_days[1:], future_price, axis=0)

# 🔹 Convert Future Predictions Back to Original Scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 📊 Create Future DataFrame
future_dates = pd.date_range(google_data.index[-1], periods=num_future_days+1)[1:]
future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

# 📉 Visualizing Future Stock Prices
fig1 = plt.figure(figsize=(15, 6))
plt.plot(future_data.index, future_data['Predicted Close'], color='green', linestyle='-', marker='o')
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.title('Predicted Future Stock Prices')
st.pyplot(fig1)

# 🔹 Display Predicted Future Prices
st.subheader("📊 Predicted Stock Prices for Next 30 Days")
st.write(future_data)

