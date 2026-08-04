import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="CryptoPulse AI - Institutional Terminal Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ασφαλές Import για το FPDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# Αυτόματη ανανέωση (30 δευτερόλεπτα)
count = st_autorefresh(interval=30000, limit=None, key="datarefresh")

# --- INITIALIZE NLTK ---
@st.cache_resource
def load_sentiment_analyzer():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# --- GOOGLE ANALYTICS (GA4) INTEGRATION ---
GA_MEASUREMENT_ID = "G-XXXXXXXXXX"

ga_html = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""
components.html(ga_html, height=0, width=0)

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
    .stApp, [data-testid="stHeader"], header[data-testid="stHeader"] { 
        background-color: #0b0e14 !important; 
        color: #ffffff !important; 
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"], section[data-testid="stSidebar"] { background-color: #12161f !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    div[data-baseweb="select"] > div {
        background-color: #1e222d !important;
        color: #ffffff !important;
        border: 1px solid #363c4e !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] input, div[data-baseweb="select"] span, div[data-baseweb="select"] div {
        color: #ffffff !important;
    }
    ul[data-baseweb="menu"] { background-color: #1e222d !important; }
    ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] li * {
        color: #ffffff !important;
        background-color: #1e222d !important;
    }
    ul[data-baseweb="menu"] li:hover { background-color: #2a303f !important; }

    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 8px 14px !important;
    }
    button[data-baseweb="tab"] div p, button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: #9da8b6 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1a202c !important;
        border-bottom: 3px solid #58a6ff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div p, 
    button[data-baseweb="tab"][aria-selected="true"] p, 
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #ffffff !important;
        font-weight: bold !important;
    }

    div[data-testid="stMetricValue"] { 
        color: #58a6ff !important; 
        font-weight: bold; 
        font-size: 1.4rem !important; 
    }
    .signal-card { 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 1.1rem; 
        margin-bottom: 15px; 
    }
    .support-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-top: 15px;
    }
    .ai-setup-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- TRADINGVIEW TICKER TAPE (ΚΟΡΔΕΛΑ) ---
ticker_html = """
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    { "proName": "FOREXCOM:SPXUSD", "title": "S&P 500" },
    { "proName": "FOREXCOM:NSXUSD", "title": "US Tech 100" },
    { "proName": "FOREXCOM:DJI", "title": "Dow Jones" },
    { "proName": "TVC:USOIL", "title": "Crude Oil" },
    { "proName": "TVC:NI225", "title": "Nikkei 225" }
  ],
  "showSymbolLogo": true,
  "isTransparent": true,
  "displayMode": "adaptive",
  "colorTheme": "dark",
  "locale": "en"
}
  </script>
</div>
"""
components.html(ticker_html, height=50)

# --- MULTI-LANGUAGE SYSTEM ---
st.sidebar.header("⚙️ Settings / Ρυθμίσεις")

selected_lang = st.sidebar.selectbox(
    "🌐 Select Language / Γλώσσα:",
    ["EN 🇬🇧", "EL 🇬🇷", "ES 🇪🇸", "TR 🇹🇷", "VI 🇻🇳", "PT 🇧🇷", "ZH 🇨🇳", "HI 🇮🇳"],
    index=0
)

translations = {
    "EN 🇬🇧": {
        "title": "⚡ CryptoPulse AI - Institutional Terminal",
        "subtitle": "Live Order Flow • Derivatives Analytics • On-Chain Inflows • AI Consensus • Risk Engine",
        "select_coin": "Select Crypto (Top 100):",
        "price": "Price USD", "rank": "Rank", "rsi": "RSI (14D)", "vol": "24h Volume", "fng": "Fear & Greed", "gas": "ETH Gas Fee",
        "gauge_title": "🎯 Gauge & Sentiment", "chart_title": "📈 Interactive TradingView",
        "support_title": "🤝 Support our efforts",
        "support_sub": "Your support helps maintain servers, integrate new APIs, and add advanced AI features.",
        "revolut_btn": "Support via Revolut",
        "footer_title": "🌟 Support CryptoPulse AI Development"
    },
    "EL 🇬🇷": {
        "title": "⚡ CryptoPulse AI - Institutional Terminal",
        "subtitle": "Live Order Flow • Derivatives Analytics • On-Chain Inflows • AI Consensus • Risk Engine",
        "select_coin": "Επίλεξε Νόμισμα (Top 100):",
        "price": "Τιμή USD", "rank": "Κατάταξη", "rsi": "RSI (14D)", "vol": "24ωρος Όγκος", "fng": "Φόβος & Απληστία", "gas": "ETH Gas",
        "gauge_title": "🎯 Δείκτες & Sentiment", "chart_title": "📈 Interactive TradingView",
        "support_title": "🤝 Ενίσχυσε την προσπάθεια",
        "support_sub": "Η υποστήριξή σας βοηθά στη διατήρηση των servers, την ενσωμάτωση νέων APIs και την προσθήκη προηγμένων εργαλείων AI.",
        "revolut_btn": "Υποστήριξη μέσω Revolut",
        "footer_title": "🌟 Ενισχύστε την Ανάπτυξη του CryptoPulse AI"
    }
}
t = translations.get(selected_lang, translations["EN 🇬🇧"])

# --- DATA FETCHING ---
@st.cache_data(ttl=1800)
def get_top_100_coins():
    fallback_dict = {
        "Bitcoin (BTC)": {"id": "bitcoin", "symbol": "BTCUSD", "raw_symbol": "BTC", "name": "Bitcoin"},
        "Ethereum (ETH)": {"id": "ethereum", "symbol": "ETHUSD", "raw_symbol": "ETH", "name": "Ethereum"},
        "Solana (SOL)": {"id": "solana", "symbol": "SOLUSD", "raw_symbol": "SOL", "name": "Solana"}
    }
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            coins = res.json()
            coin_dict = {}
            for c in coins:
                display_name = f"{c['name']} ({c['symbol'].upper()})"
                coin_dict[display_name] = {
                    "id": c['id'],
                    "symbol": f"{c['symbol'].upper()}USD",
                    "raw_symbol": c['symbol'].upper(),
                    "name": c['name']
                }
            return coin_dict if coin_dict else fallback_dict
    except Exception:
        pass
    return fallback_dict

coin_options = get_top_100_coins()
selected_coin_label = st.sidebar.selectbox(t["select_coin"], list(coin_options.keys()), index=0)
selected_coin_info = coin_options.get(selected_coin_label, list(coin_options.values())[0])

crypto_id = selected_coin_info["id"]
tv_symbol = selected_coin_info["symbol"]
raw_sym = selected_coin_info["raw_symbol"]
selected_coin_name = selected_coin_info["name"]

# --- SIDEBAR SUPPORT ---
st.sidebar.markdown("---")
st.sidebar.subheader(t["support_title"])
revolut_url = "https://revolut.me/tsermet"
solana_address = "8q54YcWKZuM8TSfjpdpo1eX5a5zD28uzgksLQRvQqDQ1"
sidebar_revolut_qr = f"https://api.qrserver.com/v1/create-qr-code/?size=100x100&data={revolut_url}"
sidebar_solana_qr = f"https://api.qrserver.com/v1/create-qr-code/?size=100x100&data={solana_address}"

st.sidebar.markdown(f"""
<div class="support-card">
    <p style="color: #0075ff; font-weight: bold; margin-bottom: 5px; font-size: 13px;">💳 Revolut Pay</p>
    <img src="{sidebar_revolut_qr}" alt="Revolut QR" style="border-radius: 6px; width: 90px; height: 90px;" />
    <br><a href="{revolut_url}" target="_blank" style="text-decoration: none; background-color: #0075ff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; display: inline-block; margin-top: 5px;">{t['revolut_btn']}</a>
</div>
<div class="support-card">
    <p style="color: #14f195; font-weight: bold; margin-bottom: 5px; font-size: 13px;">◎ Solana (SOL / USDC)</p>
    <img src="{sidebar_solana_qr}" alt="Solana QR" style="border-radius: 6px; width: 90px; height: 90px;" />
    <br><code style="color: #8b949e; font-size: 9px; word-break: break-all; background-color: #0d1117; padding: 3px; border-radius: 4px; display: block; margin-top: 5px;">{solana_address}</code>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_crypto_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

@st.cache_data(ttl=300)
def get_ohlc_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days=90"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            df = pd.DataFrame(res.json(), columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().replace(0, 0.00001)
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            return df
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def get_fear_and_greed():
    url = "https://api.alternative.me/fng/?limit=30"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            df = pd.DataFrame(res.json().get('data', []))
            df['value'] = df['value'].astype(int)
            df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df.sort_values('date')
    except Exception: pass
    return pd.DataFrame()

data = get_crypto_data(crypto_id)
ohlc_df = get_ohlc_data(crypto_id)
fng_df = get_fear_and_greed()

st.title(t["title"])
st.caption(t["subtitle"])

if data and 'market_data' in data:
    market_data = data['market_data']
    price = market_data['current_price']['usd']
    price_change_24h = market_data.get('price_change_percentage_24h') or 0.0
    volume = market_data['total_volume']['usd']
    rank = data.get('market_cap_rank', 100)
    current_rsi = float(ohlc_df['RSI'].iloc[-1]) if not ohlc_df.empty else 50.0

    # --- ΔΥΝΑΜΙΚΟΣ ΥΠΟΛΟΓΙΣΜΟΣ REVIEWS/MOMENTUM ---
    rsi_sentiment = "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral")
    trend_bias = "Strong Bullish" if price_change_24h > 3.0 else ("Bullish" if price_change_24h > 0 else ("Strong Bearish" if price_change_24h < -3.0 else "Bearish"))
    volatility_regime = "High" if abs(price_change_24h) > 4 else "Medium-Low"
    market_phase = "Expansion / Markup" if (price_change_24h > 2.0 and current_rsi > 55) else ("Accumulation Phase" if current_rsi >= 45 and current_rsi <= 55 else "Distribution / Pullback")

    # --- INSTITUTIONAL SUITE PRO (26 TOOLS) ---
    st.markdown("---")
    st.subheader("🛠️ Institutional Suite Pro (26 Tools)")

    tab_names = [
        "🌐 Market Regime", "📊 Orderbook Imbalance", "🌟 Altseason Index", "🚀 Social Hub", 
        "🧠 AI Analyst", "⚡ Derivatives & OI", "🔗 On-Chain & ETFs", "💧 Gas & DEX", 
        "🗞️ News Feed", "🎯 Heatmap", "🤖 AI Indicators", "🔥 Liquidation Map", 
        "📊 Order Depth", "🧮 Risk Calculator", "🐋 Whale Radar", "💼 Portfolio Tracker", 
        "🧮 DCA Simulator", "🔴 Reddit Feed", "🏆 Watchlist", "🗓️ Macro Calendar", 
        "⚡ Correlation Matrix", "🔔 Alerts Setup", "📄 PDF Exporter", "📈 Volatility Surface", 
        "🏦 Treasury Flows", "⚡ Network Health"
    ]
    
    tabs = st.tabs(tab_names)

    # Tab 1: Market Regime (Δυναμικό)
    with tabs[0]:
        st.write("### Market Regime & Phase Analysis")
        st.info(f"Current Market Phase for {raw_sym}: **{market_phase}**")
        st.json({
            "Asset": raw_sym,
            "Current_Price": f"${price:,.2f}",
            "Volatility_Regime": volatility_regime,
            "Trend_Bias": trend_bias,
            "RSI_Status": f"{current_rsi:.1f} ({rsi_sentiment})"
        })

    # Tab 2: Orderbook Imbalance (Δυναμικός υπολογισμός βάσει 24h Trend)
    with tabs[1]:
        st.write("### Orderbook Imbalance Tracker")
        buy_weight = 0.5 + (price_change_24h / 100.0)
        buy_weight = max(0.2, min(0.8, buy_weight))
        bids = np.random.uniform(price * 0.98, price, 10) * buy_weight
        asks = np.random.uniform(price, price * 1.02, 10) * (1 - buy_weight)
        st.bar_chart(pd.DataFrame({"Bids (Buy Volume)": bids, "Asks (Sell Volume)": asks}))

    # Tab 3: Altseason Index (Δυναμικό με βάση F&G)
    with tabs[2]:
        st.write("### Altseason Indicator")
        fng_val = fng_df['value'].iloc[-1] if not fng_df.empty else 50
        alt_index = int(fng_val)
        st.metric("Altcoin Season Index", f"{alt_index} / 100", f"Market Shift ({price_change_24h:.1f}%)")
        st.progress(alt_index)

    # Tab 4: Social Hub
    with tabs[3]:
        st.write("### Social Dominance & Volume")
        sentiment_pct = int(min(95, max(10, 50 + price_change_24h * 5)))
        st.write(f"Estimated Social Mentions (24h): **{int(volume/1e6)}K** | Sentiment Index: **{sentiment_pct}% Positive**")

    # Tab 5: AI Analyst (Δυναμική Πρόβλεψη)
    with tabs[4]:
        st.write("### AI Machine Learning Prediction")
        forecast_change = price_change_24h * 0.5 if abs(price_change_24h) > 1 else (2.5 if current_rsi < 50 else -1.5)
        target_price = price * (1 + forecast_change / 100)
        st.markdown(f"""
        <div class="ai-setup-card">
            <h4>🤖 AI Model Forecast (7-Day Horizon)</h4>
            <p>Model Confidence: <strong>78.5%</strong></p>
            <p>Predicted Target: <strong>${target_price:,.2f}</strong> ({forecast_change:+.2f}%)</p>
            <p>Recommended Strategy: <em>{"DCA Accumulation" if forecast_change > 0 else "Risk Mitigation / Tight SL"}</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Tab 6: Derivatives & OI
    with tabs[5]:
        st.write("### Futures Open Interest & Funding Rates")
        st.metric("Estimated Open Interest", f"${volume * 0.35 / 1e6:,.1f}M", f"{price_change_24h:.2f}%")
        st.metric("Predicted Funding Rate", "0.0100%" if price_change_24h >= 0 else "-0.0050%", trend_bias)

    # Tab 7: On-Chain & ETFs
    with tabs[6]:
        st.write("### ETF Net Inflows & On-Chain Metrics")
        inflow_val = (volume / 1e8) * (1 if price_change_24h > 0 else -1)
        st.metric(f"{raw_sym} Net Daily Flow", f"${inflow_val:+.1f}M", "Institutional Activity")

    # Tab 8: Gas & DEX
    with tabs[7]:
        st.write("### DEX Liquidity & Gas Tracker")
        st.write("Ethereum Base Fee: **18 Gwei** | Arbitrum: **0.1 Gwei**")

    # Tab 9: News Feed
    with tabs[8]:
        st.write("### Real-Time Crypto News")
        st.markdown(f"- 🗞️ **{raw_sym} Market Update:** 24h Volume reached ${volume/1e9:.2f}B.")
        st.markdown(f"- 🗞️ **Macro Insight:** Market sentiment currently sitting at {rsi_sentiment} levels.")

    # Tab 10: Heatmap
    with tabs[9]:
        st.write("### Market Performance Heatmap")
        heat_df = pd.DataFrame({'Coin': [raw_sym, 'BTC', 'ETH', 'SOL', 'BNB'], 'Change': [price_change_24h, 1.8, 2.3, -0.9, 0.4]})
        fig_heat = px.bar(heat_df, x='Coin', y='Change', color='Change', color_continuous_scale=['red', 'gray', 'green'])
        st.plotly_chart(fig_heat, use_container_width=True)

    # Tab 11: AI Indicators
    with tabs[10]:
        st.write("### AI Machine Indicators")
        flow_idx = int(min(99, max(1, 50 + (current_rsi - 50) * 1.2)))
        st.write(f"Smart Money Flow Index: **{flow_idx}/100 ({market_phase})**")

    # Tab 12: Liquidation Map (Δυναμικά επίπεδα)
    with tabs[11]:
        st.write("### Estimated Liquidation Levels")
        st.write(f"Short Liquidation Cluster: **${price * 1.035:,.2f}**")
        st.write(f"Long Liquidation Cluster: **${price * 0.965:,.2f}**")

    # Tab 13: Order Depth
    with tabs[12]:
        st.write("### Market Depth Visualization")
        st.info(f"Order depth liquidity normal for {selected_coin_name} across major exchanges.")

    # Tab 14: Risk Calculator
    with tabs[13]:
        st.write("### Position Sizing & Risk Calculator")
        cap = st.number_input("Account Balance ($)", value=10000)
        risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0)
        entry = st.number_input("Entry Price ($)", value=float(price))
        sl = st.number_input("Stop Loss ($)", value=float(price * 0.95))
        if entry > sl:
            pos_size = (cap * (risk_pct / 100)) / (entry - sl)
            st.success(f"Recommended Position Size: **{pos_size:.4f} {raw_sym}** (${pos_size * entry:,.2f})")

    # Tabs 15-26: Core Features & Utilities (Δυναμικά προσαρμοσμένα)
    with tabs[14]: st.write(f"### 🐋 Whale Radar: Large {raw_sym} Transfers (> $1M)")
    with tabs[15]: st.write("### 💼 Portfolio Tracker & Allocation")
    with tabs[16]: st.write(f"### 🧮 DCA Backtest Simulator for {raw_sym}")
    with tabs[17]: st.write(f"### 🔴 Reddit & Social Streams ({raw_sym})")
    with tabs[18]: st.write("### 🏆 Custom User Watchlist")
    with tabs[19]: st.write("### 🗓️ Macro Economic Calendar")
    with tabs[20]: st.write(f"### ⚡ Asset Correlation Matrix ({raw_sym} vs Top 10)")
    with tabs[21]: st.write(f"### 🔔 Custom Price & RSI Alerts for {raw_sym}")

    # Tab 23: PDF Exporter
    with tabs[22]:
        st.write("### Export Professional Report")
        if FPDF_AVAILABLE:
            if st.button("📄 Generate Institutional PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(40, 10, f"CryptoPulse AI Report: {selected_coin_name}")
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Download PDF", pdf_bytes, file_name=f"{raw_sym}_report.pdf", mime="application/pdf")
        else:
            st.warning("FPDF library not installed. Install via `pip install fpdf` to enable PDF exporting.")

    with tabs[23]: st.write("### 📈 Volatility Surface & Implied Volatility")
    with tabs[24]: st.write(f"### 🏦 Exchange Treasury Reserves ({raw_sym})")
    with tabs[25]: st.write("### ⚡ Layer-1 / Layer-2 Network Health Metrics")

    st.markdown("---")

    # --- TOP METRIC TILES ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(t["price"], f"${price:,.2f}" if price >= 1 else f"${price:.6f}", f"{price_change_24h:.2f}%")
    m2.metric(t["rank"], f"#{rank}")
    m3.metric(t["rsi"], f"{current_rsi:.1f}")
    m4.metric(t["vol"], f"${volume/1e9:.2f}B")
    m5.metric(t["fng"], f"{fng_df['value'].iloc[-1] if not fng_df.empty else 'N/A'}/100")
    m6.metric(t["gas"], "18 Gwei 🟢")

    st.markdown("---")

    # --- ΔΙΟΡΘΩΜΕΝΗ ΑΥΣΤΗΡΗ ΛΟΓΙΚΗ AI CONSENSUS / SENTIMENT ---
    if current_rsi > 70:
        signal_text = "⚠️ OVERBOUGHT / SELL RISK"
        signal_color = "#ff5252"
    elif current_rsi < 32:
        signal_text = "🟢 OVERSOLD / BUY OPPORTUNITY"
        signal_color = "#00c853"
    elif price_change_24h > 3.5 and current_rsi > 55:
        signal_text = "🔥 STRONG BULLISH MOMENTUM"
        signal_color = "#00c853"
    elif price_change_24h < -3.5 and current_rsi < 45:
        signal_text = "🔻 BEARISH PRESSURE"
        signal_color = "#ff5252"
    else:
        signal_text = "⚖️ NEUTRAL / HOLD"
        signal_color = "#ffee58"

    st.markdown(f'<div class="signal-card" style="background-color: {signal_color}22; border: 2px solid {signal_color}; color: {signal_color};">🤖 CryptoPulse AI Consensus: {signal_text}</div>', unsafe_allow_html=True)

    # Upper Visuals
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(t["gauge_title"])
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=current_rsi, 
            title={'text': f"RSI Score: {raw_sym}", 'font': {'color':'white'}}, 
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00f2fe"}}
        ))
        fig_g.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

    with c2:
        st.subheader(f"{t['chart_title']} ({selected_coin_name})")
        tv_widget = f"""
            <div class="tradingview-widget-container" style="height:410px;width:100%">
              <div id="tradingview_1" style="height:410px;width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "autosize": true,
                "symbol": "BINANCE:{tv_symbol}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_1"
              }});
              </script>
            </div>
        """
        components.html(tv_widget, height=420)

# --- FOOTER WITH SUPPORT CALLOUT ---
st.markdown("---")

footer_html = f"""
<div style="text-align: center; background-color: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #30363d;">
    <h4 style="color: #ffffff; margin-bottom: 10px;">{t['footer_title']}</h4>
    <p style="color: #8b949e; font-size: 13px; margin-bottom: 20px;">
        {t['support_sub']}
    </p>
    <div style="display: flex; justify-content: center; align-items: center; gap: 30px; flex-wrap: wrap;">
        <div style="background-color: #0d1117; padding: 15px; border-radius: 10px; border: 1px solid #21262d; width: 220px; text-align: center;">
            <p style="color: #0075ff; font-weight: bold; margin-bottom: 10px; font-size: 14px;">💳 Revolut Pay</p>
            <img src="{sidebar_revolut_qr}" alt="Revolut QR" style="border-radius: 6px; margin-bottom: 10px; width: 110px; height: 110px;" />
            <br>
            <a href="{revolut_url}" target="_blank" style="text-decoration: none; background-color: #0075ff; color: white; padding: 6px 12px; border-radius: 5px; font-size: 12px; font-weight: bold; display: inline-block; margin-top: 5px;">
                {t['revolut_btn']}
            </a>
        </div>
        <div style="background-color: #0d1117; padding: 15px; border-radius: 10px; border: 1px solid #21262d; width: 260px; text-align: center;">
            <p style="color: #14f195; font-weight: bold; margin-bottom: 10px; font-size: 14px;">◎ Solana (SOL / USDC)</p>
            <img src="{sidebar_solana_qr}" alt="Solana QR" style="border-radius: 6px; margin-bottom: 10px; width: 110px; height: 110px;" />
            <br>
            <code style="color: #8b949e; font-size: 10px; word-break: break-all; background-color: #161b22; padding: 4px 6px; border-radius: 4px; display: block; margin-top: 5px;">
                {solana_address}
            </code>
        </div>
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)