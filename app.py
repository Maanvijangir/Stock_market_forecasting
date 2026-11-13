import os
import streamlit as st
import pandas as pd
import yfinance as yf 
import joblib
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import urllib.request
import xml.etree.ElementTree as ET

# Page Configuration
st.set_page_config(
    page_title="AI Stock Market Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to Polish UI
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Dashboard Controls")

# Expanded Nifty 50 Stock Universe
stocks = [
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "JIOFIN.NS",
    "SBIN.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "POWERGRID.NS"
]

selected_stock = st.sidebar.selectbox("Select Stock Ticker", stocks)
timeframe = st.sidebar.selectbox("Select Chart Timeframe", ["1mo", "3mo", "6mo", "1y", "ytd"], index=3)

st.title("📈 AI Stock Market Analytics Platform")
st.markdown(f"Real-time tracking, news sentiment, and deep-learning price projections for **{selected_stock}**.")

# LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = StockLSTM()

# Check model availability safely
model_path = f"models/{selected_stock}_LSTM.pth"
scaler_path = f"scaler/{selected_stock}_scaler.pkl"
has_ai_model = os.path.exists(model_path) and os.path.exists(scaler_path)

if has_ai_model:
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        scaler = joblib.load(scaler_path)
        model.eval()
    except Exception as e:
        has_ai_model = False

# Fetching Data via yfinance with error checking
@st.cache_data(ttl=300)
def load_stock_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

stock_data = load_stock_data(selected_stock, timeframe)
latest_data = load_stock_data(selected_stock, "3mo")

if stock_data.empty or len(stock_data) == 0:
    st.error(f"❌ Could not fetch historical data for **{selected_stock}** from Yahoo Finance. Please choose another stock.")
    st.stop()

current_price = float(stock_data["Close"].iloc[-1].item())
prev_close = float(stock_data["Close"].iloc[-2].item()) if len(stock_data) > 1 else current_price
daily_change = current_price - prev_close
daily_change_pct = (daily_change / prev_close) * 100

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 Overview & AI Prediction", "📰 Live News & Sentiment", "📈 Advanced Charting"])

with tab1:
    st.subheader("Price Metrics & LSTM Forecasting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price", 
            f"₹{current_price:.2f}", 
            f"{daily_change_pct:.2f}% today"
        )
        
    if has_ai_model:
        try:
            close_prices = latest_data[["Close"]]
            scaled_data = scaler.transform(close_prices)
            last_60 = scaled_data[-60:]

            X = torch.FloatTensor(np.array([last_60]))
            with torch.no_grad():
                prediction = model(X)

            predicted_price = scaler.inverse_transform(prediction.numpy())[0][0]
            change_pct = ((predicted_price - current_price) / current_price) * 100

            with col2:
                st.metric(
                    "AI Predicted Price", 
                    f"₹{predicted_price:.2f}", 
                    f"{change_pct:.2f}% expected"
                )
            with col3:
                signal = "Bullish 📈" if change_pct >= 0 else "Bearish 📉"
                st.metric("AI Model Signal", signal)
        except Exception as e:
            with col2:
                st.error("Prediction calculation failed.")
    else:
        with col2:
            st.info("LSTM Model weights not trained for this specific stock yet.")
        with col3:
            st.metric("AI Model Signal", "Not Available (Live Mode)")

    st.divider()
    
    # Model Performance Panel (only for fully trained models)
    model_metrics = {
        "TCS.NS": {"R2": 0.9002, "MAE": 65.48, "RMSE": 86.97},
        "INFY.NS": {"R2": 0.8181, "MAE": 37.30, "RMSE": 47.56},
        "RELIANCE.NS": {"R2": 0.8817, "MAE": 29.53, "RMSE": 37.45},
        "HDFCBANK.NS": {"R2": 0.8203, "MAE": 21.50, "RMSE": 24.75},
        "JIOFIN.NS": {"R2": 0.5683, "MAE": 5.14, "RMSE": 6.74}
    }

    if selected_stock in model_metrics:
        st.subheader("🎯 Model Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R² Score", model_metrics[selected_stock]["R2"])
        with m2:
            st.metric("Mean Absolute Error (MAE)", model_metrics[selected_stock]["MAE"])
        with m3:
            st.metric("Root Mean Squared Error (RMSE)", model_metrics[selected_stock]["RMSE"])
            
        r2 = model_metrics[selected_stock]["R2"]
        if r2 >= 0.85:
            st.success("✅ High Prediction Confidence Model")
        elif r2 >= 0.70:
            st.warning("⚠️ Medium Prediction Confidence Model")
        else:
            st.error("❌ Low Prediction Confidence Model")

with tab2:
    st.subheader("📰 Sentiment Analysis & Live News")
    
    try:
        sentiment_df = pd.read_csv("sentiment_results.csv")
        row = sentiment_df[sentiment_df["Ticker"] == selected_stock]
        
        if not row.empty:
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Positive Sentiment", f"{row['Positive %'].values[0]}%")
            with s_col2:
                st.metric("Neutral Sentiment", f"{row['Neutral %'].values[0]}%")
            with s_col3:
                st.metric("Negative Sentiment", f"{row['Negative %'].values[0]}%")
                
            st.info(f"Stored Market Sentiment Signal: **{row['Signal'].values[0]}**")
    except Exception:
        st.caption("No static `sentiment_results.csv` data mapped for this stock.")

    st.markdown("### Real-Time RSS Headlines")
    try:
        clean_ticker = selected_stock.replace(".NS", "")
        rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        root = ET.fromstring(response.read())
        items = root.findall('./channel/item')
        
        if items:
            for item in items[:6]:
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                st.markdown(f"• **[{title}]({link})** <small style='color:gray;'>({pub_date[:16]})</small>", unsafe_allow_html=True)
        else:
            st.info("No recent Google news items found.")
    except Exception:
        st.warning("Could not fetch real-time headlines right now.")

with tab3:
    st.subheader(f"Interactive Price Chart ({selected_stock})")
    
    chart_type = st.radio("Chart Style", ["Line Chart", "Candlestick (OHLC)"], horizontal=True)
    
    fig = go.Figure()
    
    if chart_type == "Line Chart":
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#00FFA3", width=2)
            )
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
                name="OHLC"
            )
        )

    fig.update_layout(
        title=f"{selected_stock} Performance History",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=550,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Raw Historical Dataset"):
        st.dataframe(stock_data, use_container_width=True)