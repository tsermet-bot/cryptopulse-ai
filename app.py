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

# --- GLOBAL CUSTOM CSS FOR READABILITY ---
st.markdown("""
    <style>
    /* Γενικό κείμενο και παράγραφοι */
    html, body, [class*="css"], p, span, label, div {
        color: #e0e6ed !important;
    }
    
    /* Τίτλοι και επικεφαλίδες */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: bold !important;
    }

    /* Πεδία εισαγωγής (Inputs / Text Areas / Selectboxes) */
    .stTextInput input, .stNumberInput input, .stSelectbox div {
        color: #ffffff !important;
        background-color: #1e222d !important;
        border: 1px solid #363c4e !important;
    }

    /* Υπότιτλοι στα πεδία */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #b2b9c0 !important;
        font-weight: 600 !important;
    }

    /* Tabs (Καρτέλες) */
    button[data-baseweb="tab"] {
        color: #a0aab5 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ΤΟ ΥΠΟΛΟΙΠΟ SCRIPT ΣΟΥ ΞΕΚΙΝΑΕΙ ΕΔΩ ---
