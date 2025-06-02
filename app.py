import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import os

# Title
st.title('📈 Stock Trend Prediction')

# Inputs
user_input = st.text_input('Which stock are you tracking? (Enter ticker)', 'AAPL')

# Date picker for end date - default to today
start = '2010-01-01'
end = st.date_input('Select the end date for historical data', datetime.date.today())

# Convert end date to string format for yf.download
end_str = end.strftime('%Y-%m-%d')

# Download data
df = yf.download(user_input, start=start, end=end_str)

if df.empty:
    st.warning("No data found for the selected stock ticker and date range.")
else:
    # Show data description
    st.subheader(f'Data from {start} to {end_str}')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'{user_input} Closing Price')
    st.pyplot(fig)

    st.subheader('Closing Price with 100-Day Moving Average')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Close Price')
    plt.plot(ma100, label='100-day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price with 100-Day and 200-Day Moving Averages')
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Close Price')
    plt.plot(ma100, label='100-day MA')
    plt.plot(ma200, label='200-day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    # Split data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

    # Scaling training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'keras_model.h5')
    model = load_model(model_path)

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)  # Use transform, not fit_transform on test data

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    # Scale back to original values
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Accuracy metrics
    mae = mean_absolute_error(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)

    st.subheader('📊 Model Accuracy Metrics')
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"R-squared (R²): {r2:.4f}")

    # Final prediction plot
    st.subheader('📈 Predictions vs Original Prices')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
