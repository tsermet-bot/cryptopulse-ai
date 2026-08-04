import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="CryptoPulse Institutional Terminal", layout="wide")

from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Αυτόματη ανανέωση κάθε 30 δευτερόλεπτα
count = st_autorefresh(interval=30000, limit=None, key="datarefresh")

# --- GLOBAL CUSTOM CSS ---
st.markdown("""
    <style>
    header[data-testid="stHeader"] { background-color: transparent !important; }
    header { visibility: hidden !important; }
    .stApp { background-color: #0b0e14 !important; color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #12161f !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    div[data-baseweb="select"] > div {
        background-color: #1e222d !important;
        color: #ffffff !important;
        border: 1px solid #363c4e !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] span { color: #ffffff !important; }

    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: bold; font-size: 1.3rem !important; }
    
    .signal-card { 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 1.2rem; 
        margin-bottom: 15px; 
    }
    </style>
""", unsafe_allow_html=True)

# --- FETCH BINANCE SYMBOLS ---
st.sidebar.header("⚙️ Terminal Settings")
symbol_input = st.sidebar.text_input("Binance Pair (π.χ. BTCUSDT):", value="BTCUSDT").upper()

# --- BINANCE REAL-TIME DATA FETCHERS ---
@st.cache_data(ttl=15)
def get_binance_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            df = pd.DataFrame(res.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Technical Indicators
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.00001)
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            return df
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=10)
def get_orderbook_and_funding(symbol):
    depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50"
    funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
    
    ob_ratio = 1.0
    funding_rate = 0.0
    
    try:
        res_ob = requests.get(depth_url, timeout=5)
        if res_ob.status_code == 200:
            data = res_ob.json()
            bids_vol = sum([float(b[1]) for b in data['bids']])
            asks_vol = sum([float(a[1]) for a in data['asks']])
            ob_ratio = bids_vol / asks_vol if asks_vol > 0 else 1.0
            
        res_f = requests.get(funding_url, timeout=5)
        if res_f.status_code == 200:
            funding_rate = float(res_f.json().get('lastFundingRate', 0.0)) * 100
    except Exception: pass
    
    return ob_ratio, funding_rate

# --- MAIN EXECUTION ---
st.title(f"🏛️ Institutional Terminal — {symbol_input}")

df_15m = get_binance_klines(symbol_input, "15m")
df_4h = get_binance_klines(symbol_input, "4h")
df_1d = get_binance_klines(symbol_input, "1d")

ob_ratio, funding_rate = get_orderbook_and_funding(symbol_input)

if not df_15m.empty and not df_4h.empty and not df_1d.empty:
    
    # Analyze Timeframes
    tf_15m_bull = df_15m.iloc[-1]['close'] > df_15m.iloc[-1]['EMA20']
    tf_4h_bull = df_4h.iloc[-1]['EMA20'] > df_4h.iloc[-1]['EMA50']
    tf_1d_bull = df_1d.iloc[-1]['close'] > df_1d.iloc[-1]['EMA50']

    # Weighted Institutional Scoring
    score = 0
    
    # 1. Multi-Timeframe Confluence (40%)
    if tf_1d_bull: score += 20
    if tf_4h_bull: score += 12
    if tf_15m_bull: score += 8
    
    # 2. Order Book Imbalance (30%)
    if ob_ratio > 1.2: score += 30  # Ισχυρή αγοραστική πίεση
    elif ob_ratio > 1.0: score += 15
    
    # 3. Derivatives Funding Rate Filter (30%)
    if funding_rate < 0.01 and funding_rate > -0.02: score += 30 # Υγιής αγορά
    elif funding_rate >= 0.03: score -= 15 # Over-leveraged Longs (Κίνδυνος Squeeze)

    confidence = max(0, min(100, score))

    if confidence >= 70:
        signal_label = f"🔥 INSTITUTIONAL BUY (Confluence: {confidence}%)"
        signal_color = "#00c853"
    elif confidence >= 45:
        signal_label = f"⚖️ NEUTRAL / WAIT (Confluence: {confidence}%)"
        signal_color = "#ffee58"
    else:
        signal_label = f"🚨 INSTITUTIONAL BEARISH / SELL (Confluence: {confidence}%)"
        signal_color = "#ff1744"

    # Display Institutional Signal Banner
    st.markdown(f'''
        <div class="signal-card" style="background-color: {signal_color}22; border: 2px solid {signal_color}; color: {signal_color};">
            🧠 <strong>Multi-Timeframe & Orderbook Matrix:</strong> {signal_label}
        </div>
    ''', unsafe_allow_html=True)

    # Multi-Timeframe Dashboard
    st.subheader("⏱️ Multi-Timeframe Analysis (MTF)")
    c1, c2, c3 = st.columns(3)
    c1.metric("15-Min (Scalp/Entry)", "Bullish 🟢" if tf_15m_bull else "Bearish 🔴")
    c2.metric("4-Hour (Mid Trend)", "Bullish 🟢" if tf_4h_bull else "Bearish 🔴")
    c3.metric("1-Day (Macro Trend)", "Bullish 🟢" if tf_1d_bull else "Bearish 🔴")

    st.markdown("---")

    # Order Book & Market Microstructure
    st.subheader("💧 Order Book & Futures Market Data")
    m1, m2, m3 = st.columns(3)
    m1.metric("Order Book Bid/Ask Ratio", f"{ob_ratio:.2f}x", "Buyers Dominant 🟢" if ob_ratio > 1.0 else "Sellers Dominant 🔴")
    m2.metric("Futures Funding Rate", f"{funding_rate:.4f}%", "Healthy 🟢" if funding_rate < 0.02 else "High Risk 🔴")
    m3.metric("1D RSI", f"{df_1d.iloc[-1]['RSI']:.1f}")

    # TradingView Interactive Widget
    st.markdown("---")
    st.subheader(f"📈 Real-Time Chart ({symbol_input})")
    tv_widget = f"""
        <div class="tradingview-widget-container" style="height:450px;width:100%">
          <div id="tradingview_1" style="height:450px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "BINANCE:{symbol_input}",
            "interval": "D",
            "theme": "dark",
            "style": "1",
            "container_id": "tradingview_1"
          }});
          </script>
        </div>
    """
    components.html(tv_widget, height=460)