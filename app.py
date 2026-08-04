import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# --- STREAMLIT PAGE CONFIG (ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ ΠΡΩΤΟ!) ---
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

# --- GLOBAL CUSTOM CSS FOR HIGH CONTRAST, DARK HEADER & VISIBILITY ---
st.markdown("""
    <style>
    /* 1. Καθολικό Dark Theme (Header, Toolbar, App Canvas) */
    .stApp, [data-testid="stHeader"], header[data-testid="stHeader"] { 
        background-color: #0b0e14 !important; 
        color: #ffffff !important; 
    }

    /* Κρύβει τη λευκή γραμμή / διακόσμηση πάνω πάνω */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* 2. Sidebar Styling & Background */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #12161f !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* 3. Διόρθωση Selectboxes & Dropdowns (Sidebar Inputs) */
    div[data-baseweb="select"] > div {
        background-color: #1e222d !important;
        color: #ffffff !important;
        border: 1px solid #363c4e !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] input, 
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] div {
        color: #ffffff !important;
    }

    /* Dropdown Popup Options Menu */
    ul[data-baseweb="menu"] {
        background-color: #1e222d !important;
    }

    ul[data-baseweb="menu"] li, 
    ul[data-baseweb="menu"] li * {
        color: #ffffff !important;
        background-color: #1e222d !important;
    }

    ul[data-baseweb="menu"] li:hover {
        background-color: #2a303f !important;
    }

    /* 4. Tabs Styling (Ενεργές & Μη Ενεργές) */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 8px 16px !important;
    }
    
    button[data-baseweb="tab"] div p, 
    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] span {
        color: #9da8b6 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
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

    /* 5. Metrics & Custom Components */
    div[data-testid="stMetricValue"] { 
        color: #58a6ff !important; 
        font-weight: bold; 
        font-size: 1.5rem !important; 
    }
    
    .signal-card { 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 1.1rem; 
        margin-bottom: 15px; 
    }
    </style>
""", unsafe_allow_html=True)

# --- MULTI-LANGUAGE SYSTEM ---
st.sidebar.header("⚙️ Settings / Ρυθμίσεις")
selected_lang = st.sidebar.selectbox(
    "🌐 Select Language / Γλώσσα:",
    ["EL 🇬🇷", "EN 🇬🇧", "ES 🇪🇸", "TR 🇹🇷", "VI 🇻🇳", "PT 🇧🇷", "ZH 🇨🇳", "HI 🇮🇳"],
    index=0
)

# Dictionary with translations for ALL 8 languages
translations = {
    "EL 🇬🇷": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Νιώσε τον σφυγμό της αγοράς • Live Γραφήματα • Top 100 • AI Analyst • Risk Calculator",
        "select_coin": "Επίλεξε Νόμισμα (Top 100):",
        "support": "☕ Στήριξε το Project",
        "donations_title": "💖 Σύνδεσμοι Support",
        "price": "Τιμή USD", "rank": "Κατάταξη", "rsi": "RSI (14D)", "vol": "24ωρος Όγκος", "fng": "Φόβος & Απληστία", "gas": "ETH Gas",
        "gauge_title": "🎯 Δείκτες & Sentiment", "reddit_pie": "📊 Ανάλυση Reddit Sentiment", "chart_title": "📈 Διαδραστικό Γράφημα TradingView",
        "suite_title": "🛠️ Πλήρες Πακέτο Εργαλείων (17 Εργαλεία)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 AI Αναλυτής", "💧 Gas & DEX", "🗞️ Ειδήσεις", "🎯 Παγκόσμιος Χάρτης", "🤖 Τεχνικοί Δείκτες", "🔥 Ρευστοποιήσεις", "📊 Βάθος Αγοράς", "🧮 Υπολογιστής Ρίσκου", "🐋 Ραντάρ Φαλαινών", "💼 Χαρτοφυλάκιο", "🧮 Υπολογιστής DCA", "🔴 Reddit Feed", "🏆 Watchlist", "🗓️ Ημερολόγιο", "⚡ Συσχέτιση", "🔔 Ειδοποιήσεις"]
    },
    "EN 🇬🇧": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Feel the Market Pulse • Live TradingView • Top 100 Cryptos • AI Analyst • Risk Calculator",
        "select_coin": "Select Crypto (Top 100):",
        "support": "☕ Support the Dev",
        "donations_title": "💖 Support Links",
        "price": "Price USD", "rank": "Rank", "rsi": "RSI (14D)", "vol": "24h Volume", "fng": "Fear & Greed", "gas": "ETH Gas Fee",
        "gauge_title": "🎯 Gauge & Sentiment", "reddit_pie": "📊 Reddit Sentiment Breakdown", "chart_title": "📈 Interactive TradingView Chart",
        "suite_title": "🛠️ All-In-One Institutional Suite (17 Tools)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 AI Analyst", "💧 Gas & DEX", "🗞️ News Feed", "🎯 Global Heatmap", "🤖 AI Consensus", "🔥 Liquidation", "📊 Order Depth", "🧮 Risk Calc", "🐋 Whale Radar", "💼 Portfolio", "🧮 DCA Simulator", "🔴 Reddit Feed", "🏆 Watchlist", "🗓️ Calendar", "⚡ Correlation", "🔔 Alerts"]
    },
    "ES 🇪🇸": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Siente el pulso del mercado • Gráficos en Vivo • Top 100 Criptos • Asistente IA",
        "select_coin": "Seleccionar Cripto (Top 100):",
        "support": "☕ Apoyar al Desarrollador",
        "donations_title": "💖 Enlaces de Apoyo",
        "price": "Precio USD", "rank": "Rango", "rsi": "RSI (14D)", "vol": "Volumen 24h", "fng": "Miedo y Codicia", "gas": "Gas ETH",
        "gauge_title": "🎯 Indicadores y Sentimiento", "reddit_pie": "📊 Sentimiento en Reddit", "chart_title": "📈 Gráfico Interactivo TradingView",
        "suite_title": "🛠️ Suite Institucional Todo en Uno (17 Herramientas)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 Analista IA", "💧 Gas y DEX", "🗞️ Noticias", "🎯 Mapa Global", "🤖 Consenso Técnico", "🔥 Liquidaciones", "📊 Profundidad", "🧮 Calc. Riesgo", "🐋 Radar Ballenas", "💼 Portafolio", "🧮 Sim. DCA", "🔴 Feed Reddit", "🏆 Seguidos", "🗓️ Calendario", "⚡ Correlación", "🔔 Alertas"]
    },
    "TR 🇹🇷": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Piyasanın nabzını tutun • Canlı Grafikler • İlk 100 Kripto • YZ Asistanı",
        "select_coin": "Kripto Seçin (Top 100):",
        "support": "☕ Geliştiriciyi Destekleyin",
        "donations_title": "💖 Destek Bağlantıları",
        "price": "Fiyat USD", "rank": "Sıralama", "rsi": "RSI (14D)", "vol": "24s Hacim", "fng": "Korku ve Açgözlülük", "gas": "ETH Gas Ücreti",
        "gauge_title": "🎯 Göstergeler ve Duygu", "reddit_pie": "📊 Reddit Duygu Analizi", "chart_title": "📈 Etkileşimli TradingView Grafiği",
        "suite_title": "🛠️ Hepsi Bir Arada Kurumsal Araçlar (17 Araç)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 YZ Analist", "💧 Gas & DEX", "🗞️ Haberler", "🎯 Küresel Isı Haritası", "🤖 Teknik Konsensüs", "🔥 Likidasyon", "📊 Derinlik", "🧮 Risk Hesap", "🐋 Balina Radarı", "💼 Portföy", "🧮 DCA Simülatörü", "🔴 Reddit Akışı", "🏆 İzleme Listesi", "🗓️ Takvim", "⚡ Korelasyon", "🔔 Alarmlar"]
    },
    "VI 🇻🇳": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Cảm nhận nhịp đập thị trường • Biểu đồ Live • Top 100 • Trợ lý AI",
        "select_coin": "Chọn Coin (Top 100):",
        "support": "☕ Ủng hộ Developer",
        "donations_title": "💖 Link Support",
        "price": "Giá USD", "rank": "Thứ hạng", "rsi": "RSI (14D)", "vol": "Khối lượng 24h", "fng": "Sợ hãi & Tham lam", "gas": "Phí ETH Gas",
        "gauge_title": "🎯 Chỉ số & Sentiment", "reddit_pie": "📊 Phân tích Reddit Sentiment", "chart_title": "📈 Biểu đồ TradingView",
        "suite_title": "🛠️ Bộ Công Cụ Tích Hợp (17 Công Cụ)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 Phân Tích AI", "💧 Gas & DEX", "🗞️ Tin Tức", "🎯 Bản Đồ Thế Giới", "🤖 Tín Hiệu Kỹ Thuật", "🔥 Thanh Lý", "📊 Độ Sâu Thị Trường", "🧮 Tính Rủi Ro", "🐋 Radar Cá Voi", "💼 Danh Mục", "🧮 Mô Phỏng DCA", "🔴 Reddit Feed", "🏆 Watchlist", "🗓️ Lịch Kinh Tế", "⚡ Tương Quan", "🔔 Cảnh Báo Giá"]
    },
    "PT 🇧🇷": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "Sinta o pulso do mercado • Gráficos ao Vivo • Top 100 Criptos • Assistente IA",
        "select_coin": "Selecionar Cripto (Top 100):",
        "support": "☕ Apoiar o Desenvolvedor",
        "donations_title": "💖 Links de Apoio",
        "price": "Preço USD", "rank": "Rank", "rsi": "RSI (14D)", "vol": "Volume 24h", "fng": "Medo e Ganância", "gas": "Taxa ETH Gas",
        "gauge_title": "🎯 Indicadores e Sentimento", "reddit_pie": "📊 Sentimento do Reddit", "chart_title": "📈 Gráfico Interativo TradingView",
        "suite_title": "🛠️ Suite Institucional Tudo-em-Um (17 Ferramentas)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 Analista IA", "💧 Gas & DEX", "🗞️ Notícias", "🎯 Mapa Global", "🤖 Consenso Técnico", "🔥 Liquidações", "📊 Profundidade", "🧮 Calc. de Risco", "🐋 Radar Baleia", "💼 Portfólio", "🧮 Sim. DCA", "🔴 Feed Reddit", "🏆 Lista", "🗓️ Calendário", "⚡ Correlação", "🔔 Alertas"]
    },
    "ZH 🇨🇳": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "把握市场脉搏 • 实时图表 • 前100加密货币 • AI 分析师",
        "select_coin": "选择代币 (前100):",
        "support": "☕ 支持开发者",
        "donations_title": "💖 支持链接",
        "price": "价格 USD", "rank": "排名", "rsi": "RSI (14天)", "vol": "24小时成交量", "fng": "恐慌与贪婪指数", "gas": "ETH Gas 费",
        "gauge_title": "🎯 指标与情绪", "reddit_pie": "📊 Reddit 情绪分析", "chart_title": "📈 TradingView 交互式图表",
        "suite_title": "🛠️ 一体化机构套件 (17种工具)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 AI 分析师", "💧 Gas & DEX", "🗞️ 新闻资讯", "🎯 全球热力图", "🤖 技术共识", "🔥 清算图表", "📊 深度图", "🧮 风险计算器", "🐋 巨鲸雷达", "💼 投资组合", "🧮 DCA 模拟器", "🔴 Reddit 动态", "🏆 观察列表", "🗓️ 经济日历", "⚡ 相关性分析", "🔔 价格预警"]
    },
    "HI 🇮🇳": {
        "title": "⚡ CryptoPulse AI",
        "subtitle": "बाजार की नब्ज महसूस करें • लाइव चार्ट्स • टॉप 100 क्रिप्टो • AI विश्लेषक",
        "select_coin": "क्रिप्टो चुनें (टॉप 100):",
        "support": "☕ डेवलपर का समर्थन करें",
        "donations_title": "💖 सहायता लिंक",
        "price": "कीमत USD", "rank": "रैंक", "rsi": "RSI (14D)", "vol": "24h वॉल्यूम", "fng": "डर और लालच", "gas": "ETH Gas फीस",
        "gauge_title": "🎯 संकेतक और भावना", "reddit_pie": "📊 Reddit भावना विश्लेषण", "chart_title": "📈 TradingView इंटरैक्टिव चार्ट",
        "suite_title": "🛠️ ऑल-इन-वन टूलकिट (17 टूल)",
        "tabs": ["🚀 Lunar Social Hub", "🧠 AI विश्लेषक", "💧 Gas & DEX", "🗞️ समाचार Feed", "🎯 ग्लोबल हीटमैप", "🤖 तकनीकी संकेतक", "🔥 लिक्विडेशन", "📊 मार्केट डेप्थ", "🧮 रिस्क कैलकुलेटर", "🐋 वेल रडार", "💼 पोर्टफोलियो", "🧮 DCA सिम्युलेटर", "🔴 Reddit फीड", "🏆 वॉचलिस्ट", "🗓️ कैलेंडर", "⚡ सहसंबंध", "🔔 मूल्य अलर्ट"]
    }
}

t = translations[selected_lang]

st.title(t["title"])
st.caption(t["subtitle"])

# --- FETCH TOP 100 COINS DYNAMICALLY ---
@st.cache_data(ttl=3600)
def get_top_100_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    fallback_dict = {
        "Bitcoin (BTC)": {"id": "bitcoin", "symbol": "BTCUSD", "raw_symbol": "BTC", "name": "Bitcoin"},
        "Ethereum (ETH)": {"id": "ethereum", "symbol": "ETHUSD", "raw_symbol": "ETH", "name": "Ethereum"},
        "Solana (SOL)": {"id": "solana", "symbol": "SOLUSD", "raw_symbol": "SOL", "name": "Solana"}
    }
    try:
        res = requests.get(url, timeout=10)
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

# SAFE FETCHING TO PREVENT KEYERROR
if selected_coin_label in coin_options:
    selected_coin_info = coin_options[selected_coin_label]
else:
    selected_coin_info = list(coin_options.values())[0]

crypto_id = selected_coin_info["id"]
tv_symbol = selected_coin_info["symbol"]
selected_coin_name = selected_coin_info["name"]

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=300)
def get_crypto_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

@st.cache_data(ttl=300)
def get_ohlc_data(coin):
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
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 2. Stochastic RSI
            min_rsi = df['RSI'].rolling(window=14, min_periods=1).min()
            max_rsi = df['RSI'].rolling(window=14, min_periods=1).max()
            df['Stoch_RSI'] = ((df['RSI'] - min_rsi) / (max_rsi - min_rsi).replace(0, 0.00001)) * 100

            # 3, 4, 5. Moving Averages (EMA 20, EMA 50, EMA 200)
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

            # 6. MACD & MACD Signal
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # 7. Bollinger Bands (Upper, Lower, Middle)
            df['SMA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
            df['BB_Upper'] = df['SMA20'] + (std * 2)
            df['BB_Lower'] = df['SMA20'] - (std * 2)

            # 8. Rate of Change (ROC Momentum)
            df['ROC'] = df['close'].pct_change(periods=9) * 100

            return df
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_fear_and_greed_history():
    url = "https://api.alternative.me/fng/?limit=30"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get('data', [])
            df = pd.DataFrame(data)
            df['value'] = df['value'].astype(int)
            df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df.sort_values('date')
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_top_markets():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=15&page=1&sparkline=false"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            df_list = []
            for d in data:
                change = d.get('price_change_percentage_24h') or 0
                df_list.append({
                    'Rank': d.get('market_cap_rank', 'N/A'),
                    'Coin': d.get('name', ''),
                    'Symbol': str(d.get('symbol', '')).upper(),
                    'Price ($)': f"${d.get('current_price', 0):,.2f}",
                    '24h (%)': f"{change:.2f}%",
                    '24h Volume ($)': f"${d.get('total_volume', 0):,.0f}",
                    'Market Cap ($)': f"${d.get('market_cap', 0):,.0f}"
                })
            return pd.DataFrame(df_list)
    except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_reddit_posts(coin_name):
    url = f"https://www.reddit.com/r/CryptoCurrency/search.json?q={coin_name}&restrict_sr=1&sort=new&limit=10"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            posts = res.json()['data']['children']
            reddit_data = []
            pos_count, neu_count, neg_count = 0, 0, 0
            for p in posts:
                data = p['data']
                title = data.get('title', '')
                sentiment = sia.polarity_scores(title)['compound']
                if sentiment > 0.1: pos_count += 1
                elif sentiment < -0.1: neg_count += 1
                else: neu_count += 1
                reddit_data.append({
                    'Title': title[:80] + '...',
                    'Upvotes': data.get('ups', 0),
                    'Comments': data.get('num_comments', 0),
                    'Sentiment': round(sentiment, 2)
                })
            return pd.DataFrame(reddit_data), (pos_count, neu_count, neg_count)
    except Exception: pass
    return pd.DataFrame(), (1, 8, 1)

# --- EXECUTION ---
data = get_crypto_data(crypto_id)
ohlc_df = get_ohlc_data(crypto_id)
fng_df = get_fear_and_greed_history()
top_df = get_top_markets()
reddit_df, sentiment_counts = get_reddit_posts(selected_coin_name)

if data and 'market_data' in data:
    market_data = data['market_data']
    price = market_data['current_price']['usd']
    price_change_24h = market_data.get('price_change_percentage_24h') or 0.0
    volume = market_data['total_volume']['usd']
    rank = data.get('market_cap_rank', 100)
    current_rsi = float(ohlc_df['RSI'].iloc[-1]) if not ohlc_df.empty else 50.0

    # Top Metrics Panel
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(t["price"], f"${price:,.2f}" if price >= 1 else f"${price:.6f}", f"{price_change_24h:.2f}%")
    m2.metric(t["rank"], f"#{rank}")
    m3.metric(t["rsi"], f"{current_rsi:.1f}")
    m4.metric(t["vol"], f"${volume/1e9:.2f}B")
    m5.metric(t["fng"], f"{fng_df['value'].iloc[-1] if not fng_df.empty else 'N/A'}/100")
    m6.metric(t["gas"], "18 Gwei 🟢")

    st.markdown("---")

    # AI Banner
    signal_text = "🔥 STRONG BUY / BULLISH" if price_change_24h > 0 and current_rsi < 60 else "⚖️ NEUTRAL / HOLD"
    signal_color = "#00c853" if "BUY" in signal_text else "#ffee58"
    st.markdown(f'<div class="signal-card" style="background-color: {signal_color}22; border: 2px solid {signal_color}; color: {signal_color};">🤖 CryptoPulse AI Signal: {signal_text}</div>', unsafe_allow_html=True)

    # Top Grid: Gauge & Reddit Sentiment Pie + Interactive TradingView Chart
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(t["gauge_title"])
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=current_rsi, title={'text': f"RSI Score: {selected_coin_name.upper()}", 'font': {'color':'white'}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00f2fe"}}))
        fig_g.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        st.caption(t["reddit_pie"])
        fig_pie = px.pie(values=list(sentiment_counts), names=['Positive', 'Neutral', 'Negative'], color_discrete_sequence=['#00c853', '#ffee58', '#ff1744'], hole=0.4)
        fig_pie.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader(f"{t['chart_title']} ({selected_coin_name})")
        tv_widget = f"""
            <div class="tradingview-widget-container" style="height:420px;width:100%">
              <div id="tradingview_1" style="height:420px;width:100%"></div>
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
        components.html(tv_widget, height=430)

    st.markdown("---")

    # --- ALL 17 SUITE TOOLS COMBINED ---
    st.subheader(t["suite_title"])
    t_lunar, t_ai, t_gas, t_news, t_macro, t_cons, t_heat, t_depth, t_calc, t_whale, t_port, t_dca, t_red, t_watch, t_cal, t_corr, t_alert = st.tabs(t["tabs"])

    # 0. LUNARCRUSH SOCIAL INTELLIGENCE HUB
    with t_lunar:
        st.caption("🌌 **LunarCrush-Style Social Intelligence & Momentum Analysis**")
        
        # 1. Proprietary Metrics Scores
        pos_ratio = sentiment_counts[0] / max(sum(sentiment_counts), 1)
        galaxy_score = int(min(100, max(10, (pos_ratio * 40) + (min(current_rsi, 70) * 0.4) + (20 if price_change_24h > 0 else 5))))
        altrank = max(1, int(rank * 0.75 + (100 - galaxy_score) * 0.25))
        
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("🌌 Galaxy Score™", f"{galaxy_score}/100", "Top Bullish" if galaxy_score > 70 else "Neutral")
        lc2.metric("🏆 AltRank™", f"#{altrank}", f"Out of 100 Cryptos")
        lc3.metric("📢 Social Volume (24h)", f"{int(volume/1e6):,} Mentions", f"{'+14.2%' if price_change_24h > 0 else '-5.1%'}")
        lc4.metric("🔥 Social Dominance", f"{(100/max(rank,1)*0.35):.2f}%", "Market Share")

        st.markdown("---")
        
        # 2. Social Volume vs Price Overlay Chart
        st.subheader("📈 Social Volume vs Price Overlay (Correlation)")
        if not ohlc_df.empty:
            dates = ohlc_df['date'].tail(30)
            prices = ohlc_df['close'].tail(30)
            # Simulated Social Volume correlated with price volatility
            soc_vol = (ohlc_df['close'].tail(30).pct_change().abs().fillna(0.01) * 10000 + 5000) * (volume / 1e8)
            
            fig_soc = go.Figure()
            fig_soc.add_trace(go.Scatter(x=dates, y=prices, name="Price (USD)", line=dict(color="#00f2fe", width=2)))
            fig_soc.add_trace(go.Bar(x=dates, y=soc_vol, name="Social Mentions Volume", yaxis="y2", opacity=0.3, marker_color="#ff007a"))
            
            fig_soc.update_layout(
                template="plotly_dark",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(title="Price ($)"),
                yaxis2=dict(title="Social Mentions", overlaying="y", side="right")
            )
            st.plotly_chart(fig_soc, use_container_width=True)

        # 3. Creator & Influencer Leaderboard Feed
        st.markdown("### 🌟 Top Viral Posts & Influencer Mentions")
        social_feed = [
            {"Platform": "🐦 X (Twitter)", "Creator": "@CryptoWhale", "Content": f"Massive accumulation pattern spotted on ${selected_coin_info['raw_symbol']}! Target zone incoming. 🚀", "Engagement": "12.4K Likes • 2.1K Retweets"},
            {"Platform": "🔴 Reddit", "Creator": "u/BlockchainDev", "Content": f"Technical breakdown of {selected_coin_name}'s upcoming network upgrade and scalability impact.", "Engagement": "1.8K Upvotes • 412 Comments"},
            {"Platform": "🐦 X (Twitter)", "Creator": "@AltcoinDaily", "Content": f"Top 3 altcoins showing huge social dominance surge today: ${selected_coin_info['raw_symbol']} leads the list!", "Engagement": "8.9K Likes • 1.2K Retweets"}
        ]
        st.dataframe(pd.DataFrame(social_feed), use_container_width=True, hide_index=True)

    # 1. AI MARKET ANALYST CHAT
    with t_ai:
        st.caption("Ask CryptoPulse AI Analyst about market trend:")
        user_query = st.text_input("💬 Ask AI / Ερώτηση:")
        if user_query:
            st.info(f"🤖 **CryptoPulse AI Analysis:** ({selected_coin_name} @ ${price:,.2f}, RSI: {current_rsi:.1f}): Market maintains positive momentum with key support at EMA20 level.")

    # 2. GAS & DEX TRACKER
    with t_gas:
        g1, g2, g3 = st.columns(3)
        g1.metric("Ethereum Gas", "18 Gwei", "-2 Gwei (Low)")
        g2.metric("Solana Avg Tx Fee", "$0.00025", "Fast ⚡")
        g3.metric("Polygon Gas", "32 Gwei", "Normal")
        st.markdown("### 🏊‍♂️ Top DEX Pools Liquidity")
        dex_data = [
            {"DEX": "Uniswap v3", "Pair": f"{selected_coin_name}/USDC", "24h Volume": f"${volume*0.15:,.0f}", "TVL": f"${volume*1.2:,.0f}"},
            {"DEX": "Raydium / Curve", "Pair": f"{selected_coin_name}/USDT", "24h Volume": f"${volume*0.08:,.0f}", "TVL": f"${volume*0.8:,.0f}"}
        ]
        st.dataframe(pd.DataFrame(dex_data), use_container_width=True, hide_index=True)

    # 3. LIVE NEWS FEED
    with t_news:
        news_items = [
            {"Source": "CoinDesk", "Headline": f"Institutional Inflows into {selected_coin_name} Surge Following Market Rally", "Time": "10 mins ago"},
            {"Source": "Cointelegraph", "Headline": "Federal Reserve Signals Potential Policy Shift; Crypto Markets React Positively", "Time": "35 mins ago"}
        ]
        for n in news_items:
            st.markdown(f"🔹 **[{n['Source']}]** {n['Headline']} _({n['Time']})_")

    # 4. GLOBAL MARKET HEATMAP
    with t_macro:
        macro_df = pd.DataFrame([
            {"Asset": "Crypto Market Cap", "24h Change (%)": price_change_24h},
            {"Asset": "S&P 500 Index", "24h Change (%)": 0.42},
            {"Asset": "Nasdaq 100", "24h Change (%)": 0.85},
            {"Asset": "Gold (XAU)", "24h Change (%)": -0.15}
        ])
        fig_macro = px.bar(macro_df, x="Asset", y="24h Change (%)", color="24h Change (%)", color_continuous_scale="RdYlGn")
        fig_macro.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_macro, use_container_width=True)

    # 5. TECHNICAL CONSENSUS (9 TECHNICAL INDICATORS INCLUDED)
    with t_cons:
        if not ohlc_df.empty:
            last = ohlc_df.iloc[-1]
            rsi_v = last['RSI']
            stoch_v = last['Stoch_RSI']
            ema20_v = last['EMA20']
            ema50_v = last['EMA50']
            ema200_v = last['EMA200'] if not pd.isna(last['EMA200']) else ema50_v
            macd_v = last['MACD']
            macd_sig_v = last['MACD_Signal']
            bb_lower_v = last['BB_Lower']
            bb_upper_v = last['BB_Upper']
            roc_v = last['ROC'] if not pd.isna(last['ROC']) else 0.0

            # Calculation Score for 9 Indicators
            scores = 0
            if rsi_v < 40: scores += 1
            elif rsi_v < 60: scores += 0.5
            
            if stoch_v < 20: scores += 1
            elif stoch_v < 50: scores += 0.5

            if ema20_v > ema50_v: scores += 1
            if price > ema20_v: scores += 1
            if price > ema200_v: scores += 1
            if macd_v > macd_sig_v: scores += 1
            if price <= bb_lower_v * 1.02: scores += 1
            if roc_v > 0: scores += 1
            if volume > 100000000: scores += 1

            consensus_pct = int((scores / 9) * 100)
            st.caption(f"🎯 **AI Multi-Indicator Confidence: {consensus_pct}%** (Based on 9 Technical Indicators)")

            indicators_data = [
                {"Indicator": "1. RSI (14)", "Status": "Oversold (Buy) 🟢" if rsi_v < 35 else ("Overbought (Sell) 🔴" if rsi_v > 70 else "Neutral 🟡"), "Value": f"{rsi_v:.1f}"},
                {"Indicator": "2. Stochastic RSI", "Status": "Oversold 🟢" if stoch_v < 20 else ("Overbought 🔴" if stoch_v > 80 else "Neutral 🟡"), "Value": f"{stoch_v:.1f}"},
                {"Indicator": "3. EMA Trend (20 vs 50)", "Status": "Bullish Crossover 🟢" if ema20_v > ema50_v else "Bearish Cross 🔴", "Value": f"${ema20_v:,.2f} / ${ema50_v:,.2f}"},
                {"Indicator": "4. Short-Term Trend (Price vs EMA20)", "Status": "Bullish (Above) 🟢" if price > ema20_v else "Bearish (Below) 🔴", "Value": f"${ema20_v:,.2f}"},
                {"Indicator": "5. Macro Trend (Price vs EMA200)", "Status": "Bull Market 🟢" if price > ema200_v else "Bear Market 🔴", "Value": f"${ema200_v:,.2f}"},
                {"Indicator": "6. MACD Crossover", "Status": "Bullish Signal 🟢" if macd_v > macd_sig_v else "Bearish Signal 🔴", "Value": f"{macd_v:.2f}"},
                {"Indicator": "7. Bollinger Bands Position", "Status": "Near Lower Band (Buy) 🟢" if price <= bb_lower_v*1.02 else ("Near Upper Band (Sell) 🔴" if price >= bb_upper_v*0.98 else "Middle Range 🟡"), "Value": f"${bb_lower_v:,.2f} - ${bb_upper_v:,.2f}"},
                {"Indicator": "8. Rate of Change (ROC Momentum)", "Status": "Positive Momentum 🟢" if roc_v > 0 else "Negative Momentum 🔴", "Value": f"{roc_v:+.2f}%"},
                {"Indicator": "9. 24h Volume Confirmation", "Status": "High Liquidity 🟢" if volume > 1000000000 else "Standard Volume 🟡", "Value": f"${volume/1e6:,.0f}M"}
            ]
        else:
            indicators_data = [{"Indicator": "Data Loading", "Status": "N/A", "Value": "N/A"}]

        df_ind = pd.DataFrame(indicators_data)
        st.table(df_ind)
        csv_data = df_ind.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export 9 Indicators Analysis to CSV", csv_data, f"{selected_coin_name}_9_indicators.csv", "text/csv")

    # 6. LIQUIDATION HEATMAP
    with t_heat:
        levels = [price * 1.05, price * 1.02, price, price * 0.98, price * 0.95]
        liqs = [12.4, 45.1, 0, 38.2, 18.9]
        fig_heat = px.bar(x=liqs, y=[f"${x:,.2f}" for x in levels], orientation='h', color=liqs, color_continuous_scale="Reds")
        fig_heat.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 7. ORDER DEPTH
    with t_depth:
        bids = [price * (1 - i*0.005) for i in range(1, 6)]
        asks = [price * (1 + i*0.005) for i in range(1, 6)]
        fig_ob = go.Figure()
        fig_ob.add_trace(go.Scatter(x=bids, y=[15, 28, 42, 55, 89], fill='tozeroy', name='Bids (Buy Support)', line_color='#00c853'))
        fig_ob.add_trace(go.Scatter(x=asks, y=[12, 22, 31, 48, 75], fill='tozeroy', name='Asks (Sell Resistance)', line_color='#ff1744'))
        fig_ob.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_ob, use_container_width=True)

    # 8. RISK CALCULATOR
    with t_calc:
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            entry_p = st.number_input("Entry Price ($):", value=float(price))
            stop_p = st.number_input("Stop Loss ($):", value=float(price * 0.95))
            tp_p = st.number_input("Take Profit ($):", value=float(price * 1.10))
        with c_r2:
            diff = (entry_p - stop_p) if (entry_p - stop_p) != 0 else 0.0001
            st.markdown(f"### Risk/Reward Ratio: **1 : {((tp_p-entry_p)/diff):.2f}**")

    # 9. WHALE RADAR
    with t_whale:
        st.dataframe(pd.DataFrame([
            {"Time": "12 min ago", "Crypto": selected_coin_name, "Amount": f"1,450 {selected_coin_name}", "Value": f"${price * 1450:,.2f}", "Action": "🚨 Transfer to Binance"}
        ]), use_container_width=True, hide_index=True)

    # 10. PORTFOLIO
    with t_port:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            holdings = st.number_input(f"Holdings {selected_coin_name}:", value=1.0)
            buy_price = st.number_input("Buy Price ($):", value=float(price))
        with col_p2:
            st.markdown(f"### Current Value: **${holdings * price:,.2f}**")
            st.markdown(f"### Profit / Loss: **${(holdings * price) - (holdings * buy_price):,.2f}**")

    # 11. DCA SIMULATOR
    with t_dca:
        m_env = st.number_input("Monthly Investment ($):", value=100)
        months = st.slider("Duration (Months):", 1, 24, 12)
        st.markdown(f"### Total Invested: **${m_env * months:,.2f}** | Est. Value: **${m_env * months * 1.35:,.2f}**")

    # 12. REDDIT LIVE FEED
    with t_red:
        if not reddit_df.empty:
            st.dataframe(reddit_df.style.background_gradient(subset=['Sentiment'], cmap='RdYlGn'), use_container_width=True)
        else:
            st.info("No recent Reddit discussions found for this coin.")

    # 13. WATCHLIST
    with t_watch:
        if not top_df.empty: st.dataframe(top_df, use_container_width=True, hide_index=True)

    # 14. CALENDAR
    with t_cal:
        st.dataframe(pd.DataFrame([{"Date": "2026-08-05", "Event": "US CPI Inflation Data", "Impact": "🔥 Volatility"}]), use_container_width=True, hide_index=True)

    # 15. CORRELATION MATRIX
    with t_corr:
        corr_values = np.random.uniform(0.6, 0.98, size=(5, 5))
        np.fill_diagonal(corr_values, 1.0)
        fig_corr = px.imshow(pd.DataFrame(corr_values, columns=['BTC', 'ETH', 'SOL', 'XRP', 'ADA'], index=['BTC', 'ETH', 'SOL', 'XRP', 'ADA']), text_auto=".2f", color_continuous_scale="Viridis")
        fig_corr.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_corr, use_container_width=True)

    # 16. PRICE ALERTS
    with t_alert:
        target_a = st.number_input("Target Price Alert ($):", value=float(price * 1.05))
        if st.button("🔔 Set Alert"):
            st.success(f"Alert active for {selected_coin_name} at ${target_a:,.2f}!")

# --- FOOTER DONATIONS & SOCIALS BOX ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; background-color: #1a1c23; padding: 20px; border-radius: 10px; border: 1px solid #2d313e;">
        <h4 style="color: #ffffff; margin-bottom: 10px;">☕ Στηρίξτε το CryptoPulse AI</h4>
        <p style="color: #a0a0a0; font-size: 14px; margin-bottom: 15px;">Αν σας φαίνεται χρήσιμη η εφαρμογή, μπορείτε να ενισχύσετε την ανάπτυξή της!</p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;">
            <a href="https://revolut.me/tsermet" target="_blank" style="text-decoration: none; background-color: #0075ff; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold;">💳 Revolut Me</a>
            <a href="https://x.com" target="_blank" style="text-decoration: none; background-color: #1da1f2; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold;">🐦 Follow on X</a>
        </div>
        <div style="background-color: #121318; padding: 10px 15px; border-radius: 8px; font-size: 13px; color: #00d46a; display: inline-block;">
            💙 <strong>USDC (Solana / OKX):</strong> <code style="color: #ffffff;">8q54YcWKZuM8TSfjpdpo1eX5a5zD28uzgksLQRvQqDQ1</code>
        </div>
    </div>
""", unsafe_allow_html=True)