import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="CryptoPulse AI - Institutional Terminal", layout="wide")

from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Αυτόματη ανανέωση
count = st_autorefresh(interval=30000, limit=None, key="datarefresh")

# --- INITIALIZE NLTK ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

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

    button[data-baseweb="tab"] { background-color: transparent !important; padding: 8px 16px !important; }
    button[data-baseweb="tab"] div p, button[data-baseweb="tab"] p { color: #9da8b6 !important; font-weight: 600 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #1a202c !important; border-bottom: 3px solid #58a6ff !important; }
    button[data-baseweb="tab"][aria-selected="true"] div p { color: #ffffff !important; font-weight: bold !important; }

    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: bold; font-size: 1.4rem !important; }
    
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

# --- FETCH TOP 100 COINS ---
@st.cache_data(ttl=3600)
def get_top_100_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    fallback_dict = {"Bitcoin (BTC)": {"id": "bitcoin", "symbol": "BTCUSD", "raw_symbol": "BTC", "name": "Bitcoin"}}
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            coins = res.json()
            return {f"{c['name']} ({c['symbol'].upper()})": {"id": c['id'], "symbol": f"{c['symbol'].upper()}USD", "raw_symbol": c['symbol'].upper(), "name": c['name']} for c in coins}
    except Exception: pass
    return fallback_dict

st.sidebar.header("⚙️ Ρυθμίσεις")
coin_options = get_top_100_coins()
selected_coin_label = st.sidebar.selectbox("Επίλεξε Νόμισμα (Top 100):", list(coin_options.keys()), index=0)
selected_coin_info = coin_options.get(selected_coin_label, list(coin_options.values())[0])

crypto_id = selected_coin_info["id"]
tv_symbol = selected_coin_info["symbol"]
selected_coin_name = selected_coin_info["name"]

# --- ADVANCED TECHNICAL ANALYSIS ENGINE ---
@st.cache_data(ttl=300)
def get_advanced_ohlc(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days=90"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json(), columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 1. RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().replace(0, 0.00001)
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))

            # 2. Stochastic RSI
            min_rsi = df['RSI'].rolling(window=14, min_periods=1).min()
            max_rsi = df['RSI'].rolling(window=14, min_periods=1).max()
            df['Stoch_RSI'] = ((df['RSI'] - min_rsi) / (max_rsi - min_rsi).replace(0, 0.00001)) * 100

            # 3. EMAs
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

            # 4. MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # 5. Bollinger Bands
            df['SMA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
            df['BB_Upper'] = df['SMA20'] + (std * 2)
            df['BB_Lower'] = df['SMA20'] - (std * 2)

            # 6. ROC
            df['ROC'] = df['close'].pct_change(periods=9) * 100

            return df
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_crypto_data(coin):
    try:
        res = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}", timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

data = get_crypto_data(crypto_id)
ohlc_df = get_advanced_ohlc(crypto_id)

st.title(f"⚡ CryptoPulse AI — {selected_coin_name}")

if data and 'market_data' in data and not ohlc_df.empty:
    market_data = data['market_data']
    price = market_data['current_price']['usd']
    
    last = ohlc_df.iloc[-1]
    rsi = last['RSI']
    stoch_rsi = last['Stoch_RSI']
    ema20 = last['EMA20']
    ema50 = last['EMA50']
    ema200 = last['EMA200'] if not pd.isna(last['EMA200']) else ema50
    macd = last['MACD']
    macd_sig = last['MACD_Signal']
    bb_lower = last['BB_Lower']
    bb_upper = last['BB_Upper']
    roc = last['ROC'] if not pd.isna(last['ROC']) else 0.0

    # SCORE
    score = 0
    max_score = 8

    if rsi < 35: score += 1
    elif rsi < 55: score += 0.5

    if stoch_rsi < 20: score += 1
    elif stoch_rsi < 50: score += 0.5

    if ema20 > ema50: score += 1
    if price > ema200: score += 1
    if price > ema20: score += 1
    if macd > macd_sig: score += 1

    if price <= bb_lower * 1.02: score += 1
    elif price >= bb_upper * 0.98: score -= 0.5

    if roc > 0: score += 1

    confidence = int((score / max_score) * 100)
    confidence = max(0, min(100, confidence))

    if confidence >= 70:
        signal_label = f"🔥 STRONG BUY (Εμπιστοσύνη: {confidence}%)"
        signal_color = "#00c853"
    elif confidence >= 50:
        signal_label = f"📈 WEAK BUY / BULLISH (Εμπιστοσύνη: {confidence}%)"
        signal_color = "#64dd17"
    elif confidence >= 35:
        signal_label = f"⚖️ NEUTRAL / HOLD (Εμπιστοσύνη: {confidence}%)"
        signal_color = "#ffee58"
    else:
        signal_label = f"🚨 SELL / BEARISH (Εμπιστοσύνη: {confidence}%)"
        signal_color = "#ff1744"

    # Display Multi-Indicator Signal Banner
    st.markdown(f'''
        <div class="signal-card" style="background-color: {signal_color}22; border: 2px solid {signal_color}; color: {signal_color};">
            🤖 <strong>Multi-Indicator AI Consensus:</strong> {signal_label}
        </div>
    ''', unsafe_allow_html=True)

    # Grid 8 Δεικτών
    st.subheader("📊 Αναλυτικός Πίνακας Τεχνικών Δεικτών")
    
    r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
    r1_col1.metric("RSI (14)", f"{rsi:.1f}", "🟢 Buy" if rsi < 40 else ("🔴 Sell" if rsi > 70 else "🟡 Neutral"))
    r1_col2.metric("Stoch RSI", f"{stoch_rsi:.1f}", "🟢 Oversold" if stoch_rsi < 20 else ("🔴 Overbought" if stoch_rsi > 80 else "🟡 Neutral"))
    r1_col3.metric("EMA Trend (20/50)", "Bullish 🟢" if ema20 > ema50 else "Bearish 🔴")
    r1_col4.metric("EMA 200 (Macro)", "Bull Market 🟢" if price > ema200 else "Bear Market 🔴")

    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    r2_col1.metric("MACD Cross", "Bullish 🟢" if macd > macd_sig else "Bearish 🔴")
    r2_col2.metric("Bollinger Position", "Lower Band 🟢" if price <= bb_lower*1.02 else ("Upper Band 🔴" if price >= bb_upper*0.98 else "Middle 🟡"))
    r2_col3.metric("Momentum (ROC)", f"{roc:+.2f}%", "Positive 🟢" if roc > 0 else "Negative 🔴")
    r2_col4.metric("Price vs EMA20", f"${ema20:,.2f}", "Above 🟢" if price > ema20 else "Below 🔴")

    st.markdown("---")

    # Interactive Chart
    st.subheader(f"📈 Διαδραστικό Γράφημα TradingView ({selected_coin_name})")
    tv_widget = f"""
        <div class="tradingview-widget-container" style="height:450px;width:100%">
          <div id="tradingview_1" style="height:450px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "BINANCE:{tv_symbol}",
            "interval": "D",
            "theme": "dark",
            "style": "1",
            "container_id": "tradingview_1"
          }});
          </script>
        </div>
    """
    components.html(tv_widget, height=460)
else:
    st.warning("⚠️ Φόρτωση δεδομένων... Παρακαλώ ανανεώστε τη σελίδα σε λίγα δευτερόλεπτα.")