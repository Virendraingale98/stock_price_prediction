import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime, timedelta
import io
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="AlphaPredict - Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI Backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Dark glassmorphism header */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* KPI Card styling */
    .kpi-container {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 1.2rem;
        border: 1px solid #f1f2f6;
        text-align: center;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 600;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    /* Recommendations Signals Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        color: white;
    }
    .badge-buy {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
    }
    .badge-sell {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }
    .badge-hold {
        background: linear-gradient(135deg, #f1c40f, #f39c12);
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# List of Tickers and Display Names
TICKER_MAP = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "TMPV.NS": "Tata Motors (TMPV)",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Ltd.",
    "LT.NS": "Larsen & Toubro"
}

# --- Sidebar Widgets ---
st.sidebar.markdown("### 🎛️ Controls")

# Navigation Menu
page = st.sidebar.radio(
    "Navigation", 
    ["📈 Stock Dashboard", "🔍 Model Analysis", "📊 Multi-Stock Screener"]
)

# Common settings depending on page
if page in ["📈 Stock Dashboard", "🔍 Model Analysis"]:
    selected_ticker_symbol = st.sidebar.selectbox(
        "Select Stock Ticker", 
        list(TICKER_MAP.keys()),
        format_func=lambda x: f"{x} ({TICKER_MAP[x]})"
    )
    
    model_name = st.sidebar.selectbox(
        "Select ML Model", 
        ["Random Forest", "Linear Regression", "XGBoost", "LSTM"]
    )

# Ticker data sync action in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Data Management")
if st.sidebar.button("Sync database with live yfinance data"):
    with st.spinner("Syncing data from yfinance..."):
        try:
            res = requests.post(f"{BACKEND_URL}/data/sync", json={"tickers": list(TICKER_MAP.keys())})
            if res.status_code == 200:
                st.sidebar.success(f"Synced successfully! Records: {res.json()['synced_records']}")
                st.rerun()
            else:
                st.sidebar.error(f"Sync failed: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Cannot connect to backend: {str(e)}")


# ==========================================
# PAGE 1: STOCK DASHBOARD
# ==========================================
if page == "📈 Stock Dashboard":
    st.markdown(f"<div class='main-title'>{TICKER_MAP[selected_ticker_symbol]} ({selected_ticker_symbol})</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Live historical price charts, next-day ML predictions, and quantitative trade recommendations</div>", unsafe_allow_html=True)
    
    # 1. Fetch data from backend
    try:
        prices_res = requests.get(f"{BACKEND_URL}/data/prices/{selected_ticker_symbol}")
        
        if prices_res.status_code == 404:
            st.warning("No price data found in SQLite. Syncing database first...")
            with st.spinner("Initial data download..."):
                sync_res = requests.post(f"{BACKEND_URL}/data/sync", json={"tickers": [selected_ticker_symbol]})
                if sync_res.status_code == 200:
                    prices_res = requests.get(f"{BACKEND_URL}/data/prices/{selected_ticker_symbol}")
                else:
                    st.error("Failed to sync initial data.")
                    st.stop()
                    
        prices_data = prices_res.json()
        df = pd.DataFrame(prices_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Get next-day prediction and recommendation
        pred_res = requests.post(
            f"{BACKEND_URL}/predictions/generate", 
            json={"ticker": selected_ticker_symbol, "model_name": model_name}
        )
        
        if pred_res.status_code == 404:
            st.info(f"Model files for {selected_ticker_symbol} do not exist yet. Training models...")
            with st.spinner("Training models for the first time (may take 30-60s)..."):
                train_res = requests.post(f"{BACKEND_URL}/models/train", json={"ticker": selected_ticker_symbol})
                if train_res.status_code == 200:
                    pred_res = requests.post(
                        f"{BACKEND_URL}/predictions/generate", 
                        json={"ticker": selected_ticker_symbol, "model_name": model_name}
                    )
                else:
                    st.error(f"Failed to train models: {train_res.text}")
                    st.stop()
                    
        if pred_res.status_code != 200:
            st.error(f"Error generating prediction: {pred_res.text}")
            st.stop()
            
        prediction = pred_res.json()
        
        # --- KPI Cards Row ---
        kpi_cols = st.columns(5)
        
        # KPI 1: Current Close Price
        current_close = df.iloc[-1]['close']
        kpi_cols[0].markdown(
            f"<div class='kpi-container'><div class='kpi-label'>Current Close Price</div><div class='kpi-value'>₹{current_close:,.2f}</div></div>", 
            unsafe_allow_html=True
        )
        
        # KPI 2: Predicted Price
        pred_price = prediction['predicted_close']
        kpi_cols[1].markdown(
            f"<div class='kpi-container'><div class='kpi-label'>Predicted Next-Day</div><div class='kpi-value'>₹{pred_price:,.2f}</div></div>", 
            unsafe_allow_html=True
        )
        
        # KPI 3: Expected Return %
        exp_return = prediction['predicted_return']
        color = "#2ecc71" if exp_return > 0 else "#e74c3c"
        sign = "+" if exp_return > 0 else ""
        kpi_cols[2].markdown(
            f"<div class='kpi-container'><div class='kpi-label'>Expected Return</div><div class='kpi-value' style='color:{color}'>{sign}{exp_return:.2f}%</div></div>", 
            unsafe_allow_html=True
        )
        
        # KPI 4: Recommendation Badge
        signal = prediction['signal']
        badge_class = "badge-buy" if signal == "BUY" else "badge-sell" if signal == "SELL" else "badge-hold"
        kpi_cols[3].markdown(
            f"<div class='kpi-container'><div class='kpi-label'>Recommendation</div><div style='margin-top:0.4rem;'><span class='badge {badge_class}'>{signal}</span></div></div>", 
            unsafe_allow_html=True
        )
        
        # KPI 5: Confidence Score
        confidence = prediction['confidence']
        kpi_cols[4].markdown(
            f"<div class='kpi-container'><div class='kpi-label'>Confidence Score</div><div class='kpi-value'>{confidence:.1f}%</div></div>", 
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Charts Section ---
        chart_tab1, chart_tab2 = st.tabs(["🕯️ Candlestick & Volume", "📈 Actual vs Predicted Overlay"])
        
        # Tab 1: Candlestick
        with chart_tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC",
                increasing_line_color='#2ecc71',
                decreasing_line_color='#e74c3c'
            ), row=1, col=1)
            
            # Volume bars
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['volume'],
                name="Volume",
                marker_color='#34495e',
                opacity=0.4
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"Historical Price & Volume for {selected_ticker_symbol}",
                yaxis_title="Price (₹)",
                yaxis2_title="Volume",
                xaxis_rangeslider_visible=False,
                height=600,
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Tab 2: Prediction Overlay
        with chart_tab2:
            # Fetch prediction history
            hist_res = requests.get(f"{BACKEND_URL}/predictions/history/{selected_ticker_symbol}?model_name={model_name}")
            hist_data = hist_res.json()
            
            if len(hist_data) < 5:
                st.info("Insufficient prediction history generated yet. Walkthrough predictions will populate this graph over time.")
                
                # Fallback: Show a test prediction plot
                st.markdown("#### Model Performance (Test Split Preview)")
                st.markdown("We can visualize this on the Model Analysis tab after training.")
            else:
                hist_df = pd.DataFrame(hist_data)
                hist_df['target_date'] = pd.to_datetime(hist_df['target_date'])
                
                # Filter out values where actual is known
                overlay_df = hist_df.dropna(subset=['actual_close']).sort_values('target_date')
                
                if overlay_df.empty:
                    st.info("Predictions saved, but waiting for next-day actual prices to log. Update prices using 'Sync database' sidebar button once new market data is live.")
                else:
                    fig_overlay = go.Figure()
                    fig_overlay.add_trace(go.Scatter(
                        x=overlay_df['target_date'],
                        y=overlay_df['actual_close'],
                        mode='lines+markers',
                        name='Actual Close Price',
                        line=dict(color='#2c3e50', width=2)
                    ))
                    fig_overlay.add_trace(go.Scatter(
                        x=overlay_df['target_date'],
                        y=overlay_df['predicted_close'],
                        mode='lines+markers',
                        name=f'Predicted Close ({model_name})',
                        line=dict(color='#3498db', width=2, dash='dash')
                    ))
                    
                    fig_overlay.update_layout(
                        title=f"Actual vs Predicted Closing Price ({model_name})",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_overlay, use_container_width=True)
                    
    except Exception as e:
        st.error(f"Connection Error: Could not load dashboard components. Is the backend server running? Details: {str(e)}")


# ==========================================
# PAGE 2: MODEL ANALYSIS
# ==========================================
elif page == "🔍 Model Analysis":
    st.markdown(f"<div class='main-title'>Model Performance Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Evaluate metrics, feature importances, and distribution errors for {selected_ticker_symbol}</div>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([0.45, 0.55])
    
    # Check if models are trained and fetch metrics
    try:
        metrics_res = requests.get(f"{BACKEND_URL}/models/metrics/{selected_ticker_symbol}")
        
        # If not trained
        if metrics_res.status_code == 404:
            st.warning("Models are not trained for this stock ticker yet.")
            if st.button("🚀 Train All 4 Models Now"):
                with st.spinner("Training Linear Regression, Random Forest, XGBoost, and LSTM (approx 30-45s)..."):
                    train_res = requests.post(f"{BACKEND_URL}/models/train", json={"ticker": selected_ticker_symbol})
                    if train_res.status_code == 200:
                        st.success("Trained successfully!")
                        st.rerun()
                    else:
                        st.error(f"Training failed: {train_res.text}")
            st.stop()
            
        metrics_data = metrics_res.json()
        metrics_df = pd.DataFrame(metrics_data)
        
        # Format the metrics table for display
        display_df = metrics_df[['model_name', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']].copy()
        display_df.columns = ["Model", "RMSE", "MAE", "MAPE (%)", "R² Score", "Directional Accuracy (%)"]
        
        # Left column: Metrics table and train/test split
        with col_left:
            st.subheader("📊 Regression Metrics Comparison")
            st.markdown("Metrics evaluated on the 20% validation/test split period (unseen data).")
            st.dataframe(
                display_df.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE (%)"], color="#d4edda")
                               .highlight_max(axis=0, subset=["R² Score", "Directional Accuracy (%)"], color="#d4edda")
                               .format({
                                   "RMSE": "₹{:.2f}",
                                   "MAE": "₹{:.2f}",
                                   "MAPE (%)": "{:.2f}%",
                                   "R² Score": "{:.4f}",
                                   "Directional Accuracy (%)": "{:.1f}%"
                               }),
                use_container_width=True,
                hide_index=True
            )
            
            # Train / Test split plot
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📅 Train / Test Time-Series Split")
            
            prices_res = requests.get(f"{BACKEND_URL}/data/prices/{selected_ticker_symbol}")
            prices_df = pd.DataFrame(prices_res.json())
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            current_close = float(prices_df.iloc[-1]['close'])
            
            # Time split indices
            split_idx = int(len(prices_df) * 0.8)
            train_df = prices_df.iloc[:split_idx]
            test_df = prices_df.iloc[split_idx:]
            
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(
                x=train_df['date'], y=train_df['close'],
                mode='lines', name='Train Set (80%)',
                line=dict(color='#2980b9')
            ))
            fig_split.add_trace(go.Scatter(
                x=test_df['date'], y=test_df['close'],
                mode='lines', name='Test/Val Set (20%)',
                line=dict(color='#e67e22')
            ))
            fig_split.add_vline(
                x=test_df.iloc[0]['date'], 
                line_width=2, line_dash="dash", line_color="gray"
            )
            fig_split.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_split, use_container_width=True)
            
        # Right column: SHAP plot and residual histogram
        with col_right:
            st.subheader("💡 SHAP Feature Importance (Explainable AI)")
            st.markdown("Features pushing predictions UP or DOWN (TreeExplainer on Random Forest model).")
            
            shap_image_path = f"models/shap_plots/shap_importance_{selected_ticker_symbol}.png"
            if os.path.exists(shap_image_path):
                st.image(shap_image_path, use_container_width=True)
            else:
                st.info("SHAP PNG plot not found on disk. Ensure model training was completed with `shap` library installed.")
                
            # Residual Histogram
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📉 Residual Error Distribution")
            
            hist_res = requests.get(f"{BACKEND_URL}/predictions/history/{selected_ticker_symbol}?model_name={model_name}")
            hist_data = hist_res.json()
            
            if len(hist_data) >= 5:
                hist_df = pd.DataFrame(hist_data).dropna(subset=['actual_close'])
                if not hist_df.empty:
                    residuals = hist_df['actual_close'] - hist_df['predicted_close']
                    
                    fig_res, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(residuals, kde=True, ax=ax, color='#8e44ad')
                    ax.set_title(f"Prediction Residuals ({model_name})", fontsize=10)
                    ax.set_xlabel("Error (Actual - Predicted) in ₹", fontsize=8)
                    ax.set_ylabel("Count", fontsize=8)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_res)
                else:
                    st.info("No predictions with logged actuals yet. Visualizing placeholder test residuals.")
            else:
                # Plot standard placeholder normal residuals
                fig_res, ax = plt.subplots(figsize=(6, 3))
                residuals = np.random.normal(0, current_close * 0.015, 100)
                sns.histplot(residuals, kde=True, ax=ax, color='#95a5a6')
                ax.set_title(f"Residual distribution placeholder (simulated)", fontsize=10)
                ax.set_xlabel("Simulated Error in ₹", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_res)
                
            # Button to retrain
            st.markdown("---")
            if st.button("🔄 Retrain All Models for This Stock"):
                with st.spinner("Retraining..."):
                    train_res = requests.post(f"{BACKEND_URL}/models/train", json={"ticker": selected_ticker_symbol})
                    if train_res.status_code == 200:
                        st.success("Retrained successfully!")
                        st.rerun()
                    else:
                        st.error(f"Retraining failed: {train_res.text}")
                        
    except Exception as e:
        st.error(f"Error loading model analysis. Is the FastAPI backend running? Details: {str(e)}")


# ==========================================
# PAGE 3: MULTI-STOCK SCREENER
# ==========================================
elif page == "📊 Multi-Stock Screener":
    st.markdown("<div class='main-title'>Multi-Stock Screener Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Evaluate and rank all 10 Indian NSE target stocks simultaneously</div>", unsafe_allow_html=True)
    
    selected_screener_model = st.selectbox(
        "Screener Model for Predictions", 
        ["Random Forest", "Linear Regression", "XGBoost", "LSTM"],
        key="screener_model_select"
    )
    
    st.markdown("Click the button below to fetch the latest closing prices and generate next-day predictions, returns, confidence scores, and signals for all 10 stocks.")
    
    if st.button("🚀 Run Multi-Stock Screener"):
        with st.spinner("Executing predictions across all 10 target tickers. This may take 5-10 seconds..."):
            try:
                screener_res = requests.get(f"{BACKEND_URL}/screener?model_name={selected_screener_model}")
                
                if screener_res.status_code == 200:
                    data = screener_res.json()
                    results = data['results']
                    target_date = data['target_date']
                    
                    if not results:
                        st.warning("No screener results returned. Make sure the database is synchronized and models are trained.")
                    else:
                        # Convert to DataFrame
                        scr_df = pd.DataFrame(results)
                        
                        # Add Ticker Company Name
                        scr_df['company_name'] = scr_df['ticker'].map(TICKER_MAP)
                        
                        # Reorder and format columns
                        scr_df = scr_df[['ticker', 'company_name', 'current_price', 'predicted_price', 'predicted_return', 'signal', 'confidence']]
                        scr_df.columns = ["Ticker", "Company Name", "Current Price", "Predicted Price", "Expected Return (%)", "Signal", "Confidence Score (%)"]
                        
                        # Format dataframe styling
                        def style_signal(val):
                            if val == 'BUY':
                                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif val == 'SELL':
                                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                            else:
                                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                        
                        st.markdown(f"### Screener Output for Target Date: **{target_date}** (Model: {selected_screener_model})")
                        
                        st.dataframe(
                            scr_df.style.applymap(style_signal, subset=["Signal"])
                                       .format({
                                           "Current Price": "₹{:.2f}",
                                           "Predicted Price": "₹{:.2f}",
                                           "Expected Return (%)": "{:+.2f}%",
                                           "Confidence Score (%)": "{:.1f}%"
                                       }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # CSV Download button
                        csv_buffer = io.StringIO()
                        scr_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Screener Results as CSV",
                            data=csv_data,
                            file_name=f"screener_results_{date.today()}_{selected_screener_model.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                else:
                    st.error(f"Screener execution failed: {screener_res.text}")
                    st.info("Hint: Make sure at least one model has been trained for each stock in the 'Model Analysis' tab, and live prices are synced.")
            except Exception as e:
                st.error(f"Connection Error: Backend did not respond. Details: {str(e)}")
