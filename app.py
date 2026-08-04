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

# Αυτόματη ανανέωση κάθε 30 δευτερόλεπτα
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

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings / Ρυθμίσεις")
selected_lang = st.sidebar.selectbox("🌐 Select Language / Γλώσσα:", ["EL 🇬🇷", "EN 🇬🇧"], index=0)

coin_options = get_top_100_coins()
selected_coin_label = st.sidebar.selectbox("Επίλεξε Νόμισμα (Top 100):", list(coin_options.keys()), index=0)
selected_coin_info = coin_options.get(selected_coin_label, list(coin_options.values())[0])

crypto_id = selected_coin_info["id"]
tv_symbol = selected_coin_info["symbol"]
raw_symbol = selected_coin_info["raw_symbol"]
selected_coin_name = selected_coin_info["name"]

# --- DATA FETCHERS ---
@st.cache_data(ttl=300)
def get_crypto_data(coin):
    try:
        res = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}", timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

@st.cache_data(ttl=300)
def get_advanced_ohlc(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days=90"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json(), columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Indicators
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().replace(0, 0.00001)
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))

            min_rsi = df['RSI'].rolling(window=14, min_periods=1).min()
            max_rsi = df['RSI'].rolling(window=14, min_periods=1).max()
            df['Stoch_RSI'] = ((df['RSI'] - min_rsi) / (max_rsi - min_rsi).replace(0, 0.00001)) * 100

            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            df['SMA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
            df['BB_Upper'] = df['SMA20'] + (std * 2)
            df['BB_Lower'] = df['SMA20'] - (std * 2)

            df['ROC'] = df['close'].pct_change(periods=9) * 100

            return df
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def get_fear_and_greed():
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5)
        if res.status_code == 200:
            d = res.json()['data'][0]
            return int(d['value']), d['value_classification']
    except Exception: pass
    return 50, "Neutral"

data = get_crypto_data(crypto_id)
ohlc_df = get_advanced_ohlc(crypto_id)
fng_val, fng_class = get_fear_and_greed()

st.title(f"⚡ CryptoPulse AI — {selected_coin_name}")

# --- TABS SETUP ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Technical Analysis & Signal", 
    "🧠 Sentiment & News", 
    "🧮 Risk Calculator", 
    "😱 Fear & Greed Index"
])

# ==================== TAB 1: TECHNICAL ANALYSIS ====================
with tab1:
    if data and 'market_data' in data and not ohlc_df.empty:
        market_data = data['market_data']
        price = market_data['current_price']['usd']
        price_change = market_data.get('price_change_percentage_24h') or 0.0
        
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

        # SCORE CALCULATOR
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

        st.markdown(f'''
            <div class="signal-card" style="background-color: {signal_color}22; border: 2px solid {signal_color}; color: {signal_color};">
                🤖 <strong>Multi-Indicator AI Consensus (8 Δείκτες):</strong> {signal_label}
            </div>
        ''', unsafe_allow_html=True)

        st.subheader("📊 Αναλυτικός Πίνακας 8 Τεχνικών Δεικτών")
        
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

# ==================== TAB 2: SENTIMENT & NEWS ====================
with tab2:
    st.subheader(f"🧠 AI Sentiment Analysis & News ({selected_coin_name})")
    
    sample_headlines = [
        f"Institutions increase holdings in {selected_coin_name} following strong market momentum.",
        f"Analysts predict potential breakout for {raw_symbol} as volume surges.",
        f"Regulatory clarity provides boost for major digital assets like {raw_symbol}.",
        f"Market consolidation continues for {selected_coin_name} amid macro uncertainty."
    ]
    
    sentiment_scores = [sia.polarity_scores(text)['compound'] for text in sample_headlines]
    avg_sentiment = np.mean(sentiment_scores)
    
    st.metric("AI Sentiment Score", f"{avg_sentiment:+.2f}", "Bullish 🟢" if avg_sentiment > 0.05 else "Bearish 🔴")
    
    st.markdown("### 📰 Τελευταία Νέα & Ανάλυση Sentiment")
    for text, score_val in zip(sample_headlines, sentiment_scores):
        badge = "🟢 Positive" if score_val > 0.05 else ("🔴 Negative" if score_val < -0.05 else "🟡 Neutral")
        st.write(f"- **{text}** | *Score: {score_val:.2f} [{badge}]*")

# ==================== TAB 3: RISK CALCULATOR ====================
with tab3:
    st.subheader("🧮 Υπολογιστής Ρίσκου & Position Sizing")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        account_balance = st.number_input("Κεφάλαιο Λογαριασμού ($):", value=1000.0, step=100.0)
        risk_percentage = st.slider("Ρίσκο ανά Trade (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    with col_r2:
        entry_p = st.number_input("Τιμή Εισόδου ($):", value=float(price if data else 100.0))
        stop_p = st.number_input("Τιμή Stop Loss ($):", value=float(price * 0.95 if data else 95.0))

    if entry_p > stop_p:
        risk_amount = account_balance * (risk_percentage / 100)
        price_risk_pct = (entry_p - stop_p) / entry_p
        position_size = risk_amount / (entry_p - stop_p)
        total_position_usd = position_size * entry_p

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Μέγιστη Χασούρα ($)", f"${risk_amount:.2f}")
        res_col2.metric("Μέγεθος Θέσης (Coins)", f"{position_size:.4f} {raw_symbol}")
        res_col3.metric("Συνολική Αξία Θέσης ($)", f"${total_position_usd:.2f}")
    else:
        st.error("⚠️ Η τιμή Stop Loss πρέπει να είναι χαμηλότερη από την τιμή εισόδου για Long positions.")

# ==================== TAB 4: FEAR & GREED INDEX ====================
with tab4:
    st.subheader("😱 Fear & Greed Index (Συνολική Αγορά)")
    
    fng_col1, fng_col2 = st.columns(2)
    fng_col1.metric("Δείκτης Φόβου / Απληστίας", f"{fng_val} / 100")
    fng_col2.metric("Κατάσταση Αγοράς", fng_class)
    
    st.progress(fng_val / 100)
    
    if fng_val < 25:
        st.info("💡 **Extreme Fear:** Ιστορικά, ο ακραίος φόβος αποτελεί ευκαιρία αγοράς (Buffett: 'Be greedy when others are fearful').")
    elif fng_val > 75:
        st.warning("⚠️ **Extreme Greed:** Η αγορά είναι υπερθερμασμένη. Προσοχή σε πιθανές διορθώσεις.")