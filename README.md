# AlphaPredict: End-to-End Stock Price Prediction & Recommendation System

AlphaPredict is a complete quantitative trading dashboard and machine learning system that fetches historical data for Indian NSE stocks, engineers technical features, trains predictive models (including LSTMs), generates BUY/SELL/HOLD recommendations, and logs prediction metrics inside a local SQLite database. The system is split into a **FastAPI backend** and a **Streamlit frontend**.

---

## 📈 System Architecture

```mermaid
graph TD
    subgraph Data Layer
        YF[Yahoo Finance API] -->|Download OHLCV| DB[(SQLite Database)]
    end

    subgraph Backend API [FastAPI - Port 8000]
        SYNC[POST /data/sync] -->|Writes to| DB
        TRAIN[POST /models/train] -->|Fit & Save| MODELS[Model Artifacts]
        TRAIN -->|Write Metrics| DB
        PRED[POST /predictions/generate] -->|Load Model| MODELS
        PRED -->|Calculate Signals| REC[Recommendation Engine]
        PRED -->|Log Prediction| DB
        SCREEN[GET /screener] -->|All Tickers| PRED
    end

    subgraph Frontend [Streamlit - Port 8501]
        DASH[📈 Dashboard Page] -->|Query Prices & Predictions| PRED
        ANALYSIS[🔍 Model Analysis Page] -->|Query Metrics & SHAP Plots| TRAIN
        SCREENER[📊 Screener Page] -->|Query Screener| SCREEN
    end

    DB -->|Read Prices/Metrics| Backend API
```

---

## 🛠️ Tech Stack & Features

- **Backend API**: FastAPI, SQLAlchemy (Object-Relational Mapping), Pydantic (validation), SQLite.
- **Data & Features**: `yfinance` API, `ta` (Technical Analysis library with manual fallbacks), Lag features (1, 3, 7 days), and Calendar effects.
- **Machine Learning Models**:
  1. **Linear Regression** (Baseline benchmarker)
  2. **Random Forest Regressor** (tuned via Grid Search cross-validation)
  3. **XGBoost Regressor** (utilizing early stopping on validation split)
  4. **LSTM (2-Layer Neural Network)** (60-day window sequences processed via Keras/TensorFlow)
- **Explainability**: SHAP (SHapley Additive exPlanations) values computed on Random Forest models.
- **UI Dashboard**: Streamlit incorporating interactive Plotly charts, residual error distributions, time-series splits, and a CSV screener exporter.

---

## 🚀 Setup & Installation

### 1. Pre-requisites
Ensure Python 3.12+ and Git are installed on your machine.

### 2. Clone and Install Dependencies
```bash
# Clone the repository
git clone <your-repo-link>
cd stock_price_prediction

# Install packages
pip install -r requirements.txt
```

### 3. Run the Backend API Server
Start the FastAPI backend (runs on port 8000 by default):
```bash
python -m uvicorn backend.main:app --reload
```
You can view the interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Run the Streamlit Frontend App
Start the Streamlit dashboard (runs on port 8501 by default):
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Exposing to Cloudflare Tunnel (cloudflared)

Cloudflare Tunnel lets you securely expose your local Streamlit frontend and FastAPI backend to the public internet under your custom subdomains without opening firewall ports.

### Step 1: Install cloudflared on Windows
1. Download the `cloudflared` installer for Windows from the [Cloudflare website](https://github.com/cloudflare/cloudflared/releases).
2. Alternatively, install it via **winget** in PowerShell:
   ```powershell
   winget install Cloudflare.cloudflared
   ```

### Step 2: Login to Cloudflare Account
Run the login command. It will open a browser window for you to select your domain:
```bash
cloudflared tunnel login
```

### Step 3: Create a Secure Tunnel
Create a new tunnel named `stock-predictor`:
```bash
cloudflared tunnel create stock-predictor
```
This generates a Tunnel ID and creates a credentials JSON file in `C:\Users\<YourUsername>\.cloudflare\`.

### Step 4: Configure the Ingress Rules
Create a `config.yml` file in `C:\Users\<YourUsername>\.cloudflare\` (or in the project root) to route two subdomains to your local ports:
```yaml
tunnel: <Your-Tunnel-ID>
credentials-file: C:\Users\<YourUsername>\.cloudflare\<Your-Tunnel-ID>.json

ingress:
  # Route Streamlit UI
  - hostname: stock.yourdomain.com
    service: http://localhost:8501
  # Route FastAPI Backend API
  - hostname: api-stock.yourdomain.com
    service: http://localhost:8000
  # Default catch-all (returns 404)
  - service: http_status:404
```

### Step 5: Route DNS Settings
Create DNS CNAME records on Cloudflare dashboard for both hostnames, or run:
```bash
cloudflared tunnel route dns stock-predictor stock.yourdomain.com
cloudflared tunnel route dns stock-predictor api-stock.yourdomain.com
```

### Step 6: Start the Tunnel
Run your tunnel to establish the secure connection:
```bash
cloudflared tunnel --config config.yml run
```
Now, users can visit `https://stock.yourdomain.com` from anywhere in the world to access your Streamlit dashboard!

---

## 📅 Quantitative Recommendation Logic

The system utilizes both model predictions and momentum indicators to compute trading signals:
- **BUY**: Expected Return $> +1.5\%$ AND Relative Strength Index (RSI) $< 60$
- **SELL**: Expected Return $< -1.5\%$ OR Relative Strength Index (RSI) $> 75$
- **HOLD**: When neither criteria are met.

**Confidence Score Formula:**
$$\text{Confidence (\%)} = \frac{100}{1 + \frac{\text{Model RMSE}}{\text{30-day Price Volatility}}}$$
If the model's test RMSE is low compared to the stock's natural price movements, the confidence score approaches $100\%$. If the model has high noise relative to the stock's movement, the score degrades toward $0\%$.
