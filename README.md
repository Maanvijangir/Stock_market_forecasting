# 📈 Stock Market Intelligence & Forecasting Pipeline

An end-to-end financial analytics and machine learning pipeline combining **Natural Language Processing (NLP)** for real-time news sentiment analysis and **Deep Learning (PyTorch LSTM)** for multi-stock time-series price forecasting across major Indian equities (NSE).

---

## Project Architecture

This project is structured into two modular workflows:

### 1. Financial News Sentiment Analysis (`NLP_Sentiment.ipynb`)
* **Data Ingestion:** Fetches live financial news headlines for target equities using the **News API** (`newsapi-python`).
* **Sentiment Scoring:** Analyzes headline polarity using **VADER** (`vaderSentiment`), computing positive, neutral, negative, and normalized compound scores.
* **Signal Generation:** Aggregates headline distributions into percentage metrics and classifies directional market sentiment (`Bullish`, `Bearish`, or `Neutral`), exporting results to `sentiment_results.csv`.

### 2. Multi-Stock Price Forecasting (`LSTM_Forecasting.ipynb`)
* **Data Engineering:** Retrieves historical daily OHLCV data (2021–2026) via `yfinance` and persists structured records into an **SQLite** relational database (`stck_data.db`).
* **Time-Series Preprocessing:** Normalizes closing prices using `MinMaxScaler` ($[0, 1]$) and generates 60-day sliding windows to predict the 61st day's closing price.
* **Deep Learning Model:** Trains a 2-layer **Long Short-Term Memory (LSTM)** neural network built in **PyTorch** using Mean Squared Error (MSE) loss and the Adam optimizer.
* **Artifact Persistence:** Saves trained model weights (`.pth`) and fitted scalers (`.pkl`) per ticker for downstream production inference.

---

## Tracked Equities & Model Performance

The pipeline evaluates five major NSE-listed companies (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `JIOFIN.NS`):

| Ticker | Company | RMSE (₹) | MAE (₹) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **TCS.NS** | Tata Consultancy Services | 92.58 | 69.88 | **0.8869** |
| **RELIANCE.NS** | Reliance Industries | 38.93 | 29.58 | **0.8722** |
| **HDFCBANK.NS** | HDFC Bank | 29.91 | 26.45 | **0.7377** |
| **INFY.NS** | Infosys | 49.85 | 39.19 | **0.8002** |
| **JIOFIN.NS** | Jio Financial Services | 6.33 | 5.04 | **0.6202** |

---

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch (`torch.nn`)
* **NLP:** VADER Sentiment (`vaderSentiment`), News API (`newsapi-python`)
* **Data & Storage:** Pandas, NumPy, SQLite3, `yfinance`
* **Machine Learning & Serialization:** Scikit-Learn (`MinMaxScaler`), Joblib

---

## Repository Structure

```text
├── NLP_Sentiment.ipynb          # News scraping & VADER sentiment classification
├── LSTM_Forecasting.ipynb       # SQLite ingestion, PyTorch LSTM training & evaluation
├── sentiment_results.csv        # Generated market sentiment & trading signals
├── stck_data.db                 # SQLite database storing historical OHLCV data
├── *_LSTM.pth                   # Saved PyTorch model state dictionaries per stock
├── *_scaler.pkl                 # Saved MinMaxScaler artifacts per stock
└── README.md                    # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Maanvijangir/Stock_market_forecasting.git
   cd stock-intelligence-pipeline
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy torch scikit-learn yfinance newsapi-python vaderSentiment matplotlib joblib
   ```

3. **Configure API Keys:**
   * Obtain a free API key from [NewsAPI.org](https://newsapi.org/).
   * Replace `API_KEY` in `NLP_Sentiment.ipynb` (or set it as an environment variable `NEWS_API_KEY`).

4. **Run the Notebooks:**
   * Execute `NLP_Sentiment.ipynb` to generate latest headline sentiment signals.
   * Execute `LSTM_Forecasting.ipynb` to populate the SQLite database, train models, and visualize actual vs. predicted closing prices.

---

## Future Improvements

* **Universal Stock Market Support:** Scale the pipeline beyond regional equities (like Indian NSE stocks) to accept dynamic global tickers from major stock exchanges worldwide (NYSE, NASDAQ, LSE, crypto markets, etc.).
* **React-Based User Interface:** Build a modern, interactive web frontend (using React or Next.js) where users can search any ticker, visualize live candlestick/line charts, view real-time VADER news sentiment cards, and see model price predictions.
* **Cloud Deployment & Automation:** Containerize the backend and inference engine using Docker, deploy the application to cloud infrastructure (AWS/GCP/Render), and set up automated cron jobs to fetch daily news and prices.
