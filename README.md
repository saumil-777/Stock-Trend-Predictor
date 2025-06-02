# 📈 Stock Trend Predictor (Basic Version)

A beginner-friendly Streamlit-based web application that predicts stock market trends using historical price data and a pre-trained LSTM (Long Short-Term Memory) model.

This app enables users to visualize past stock trends, analyze moving averages, and view predictions — all through a clean and interactive interface. It’s an excellent demonstration of combining machine learning, financial data, and web deployment using Python.

---

## 🚀 Features

* 🔎 Input any stock ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)
* 📊 Visualize historical **closing prices** from 2010 to present
* 📈 Plot 100-day and 200-day **moving averages**
* 🤖 Predict future price trends using a pre-trained **LSTM model**
* 🌐 Simple, interactive **web UI powered by Streamlit**

---

## ⚙️ How It Works

1. Takes user input for a stock ticker symbol.
2. Fetches real-time data using the `yfinance` API.
3. Visualizes historical trends with `matplotlib`.
4. Uses a trained LSTM model (`keras_model.h5`) to predict future stock prices based on recent data.
5. Displays a comparison of **actual vs. predicted prices**.

---

## 🧰 Tech Stack

* Python 🐍
* Streamlit
* Keras (TensorFlow backend)
* NumPy & Pandas
* yFinance
* Matplotlib

---

## 🛠️ Setup Instructions

1. **Clone this repository:**

   ```bash
   git clone https://github.com/saumil-777/stock-trend-predictor-basic.git
   cd stock-trend-predictor-basic
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have a `requirements.txt`, manually install: `streamlit`, `keras`, `pandas`, `numpy`, `matplotlib`, `yfinance`)*

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. **Model file:**

   Make sure `keras_model.h5` is present in the same folder as `app.py`.

---

## 🖼️ Screenshots

## 📸 Screenshots

### 🔹 Stock Input Section
![Input Section](https://github.com/saumil-777/Stock-Trend-Predictor/blob/188c773cc728ebedcd6d3bf27b4a51d4b3f1fae3/Screenshot%202025-05-26%20013437.png)

###    Closing Price vs Time Chart
![Chart](https://github.com/saumil-777/Stock-Trend-Predictor/blob/09cb87a89d7b5e12a0d95eb2f6aa001e54e60bc8/Screenshot%202025-05-26%20013452.png)

### 🔹 Moving Averages Chart
![MA Chart](https://github.com/saumil-777/Stock-Trend-Predictor/blob/23e9118382146d910f9c180292fe0f02e2d6dd20/Screenshot%202025-05-26%20013504.png)

### 🔹 Prediction Graph (Original vs Predicted)
![Prediction Graph](https://github.com/saumil-777/Stock-Trend-Predictor/blob/60af9fd1d6ddb3444547472f6dc87e7a3b89b04e/Screenshot%202025-05-26%20013513.png)


---

## 💡 Use Cases

* Learn basics of time series forecasting with LSTM
* Visualize trends and understand stock behavior
* Great for students, beginners, or finance enthusiasts
* Serve as a base template for more complex ML-based financial tools

---

## 🙏 Credits & Acknowledgements

* LSTM implementation inspired by various open-source tutorials
* Financial data sourced using the excellent [`yfinance`](https://github.com/ranaroussi/yfinance) library
* UI built with [Streamlit](https://streamlit.io/)

