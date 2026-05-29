import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import time
import os
import hmac

import model as ml
import technical as tech
import backtest as bt
import alerts as alrt
import news as nws
import calendar_data as cal
import ai_assistant as ai_asst
import anomaly as anom
import risk_tools as rt
import telegram_bot as tg
import bot_brain as brain
import enhancements as enh
import position_monitor as pmon
import paper_trading as pt
import reports as rpt
import correlations as corr
import seasonality as seas
import cot_report as cot
import journal as jrn

try:
    from config import (
        ANTHROPIC_API_KEY as _CONFIG_ANTHROPIC_API_KEY,
        TELEGRAM_BOT_TOKEN as _CONFIG_TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID as _CONFIG_TELEGRAM_CHAT_ID,
        XAUUSD_USERNAME as _CONFIG_XAUUSD_USERNAME,
        XAUUSD_PASSWORD as _CONFIG_XAUUSD_PASSWORD,
    )
except Exception:
    _CONFIG_ANTHROPIC_API_KEY = ""
    _CONFIG_TELEGRAM_BOT_TOKEN = ""
    _CONFIG_TELEGRAM_CHAT_ID = ""
    _CONFIG_XAUUSD_USERNAME = ""
    _CONFIG_XAUUSD_PASSWORD = ""


def _load_secret(name: str, fallback: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or fallback


ANTHROPIC_API_KEY = _load_secret("ANTHROPIC_API_KEY", _CONFIG_ANTHROPIC_API_KEY)
TELEGRAM_BOT_TOKEN = _load_secret("TELEGRAM_BOT_TOKEN", _CONFIG_TELEGRAM_BOT_TOKEN)
TELEGRAM_CHAT_ID = _load_secret("TELEGRAM_CHAT_ID", _CONFIG_TELEGRAM_CHAT_ID)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_first_secret(*names: str) -> str:
    for name in names:
        value = _load_secret(name, "")
        if value:
            return value
    return ""


def _require_login():
    expected_user = _load_first_secret("XAUUSD_USERNAME", "APP_USERNAME", "AUTH_USERNAME") or _CONFIG_XAUUSD_USERNAME
    expected_pass = _load_first_secret("XAUUSD_PASSWORD", "APP_PASSWORD", "AUTH_PASSWORD") or _CONFIG_XAUUSD_PASSWORD

    if not expected_user or not expected_pass:
        st.error("Login nuk eshte konfiguruar.")
        st.info("Vendos XAUUSD_USERNAME dhe XAUUSD_PASSWORD te Railway Variables, pastaj bej redeploy.")
        st.stop()

    if st.session_state.get("auth_ok"):
        return

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@500;700&display=swap');

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(180deg, #05070c 0%, #080b12 52%, #05070a 100%) !important;
  background-size: 42px 42px, 42px 42px, auto !important;
  color: #e9edf6 !important;
  font-family: Inter, system-ui, sans-serif !important;
}

.block-container {
  max-width: 560px !important;
  padding-top: 10vh !important;
}

.login-shell {
  max-width: 520px;
  margin: 0 auto;
  border: 1px solid rgba(245,200,66,0.24);
  border-bottom: 0;
  border-radius: 18px 18px 0 0;
  background:
    linear-gradient(135deg, rgba(245,200,66,0.15), rgba(255,255,255,0.02) 46%, rgba(78,163,255,0.06)),
    rgba(14,18,29,0.92);
  padding: 28px 28px 22px;
}

.login-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.login-logo {
  width: 54px;
  height: 54px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(circle at 30% 25%, #fff4a3 0%, #f5c842 42%, #9e7620 100%);
  color: #070910;
  font-family: "Space Grotesk", sans-serif;
  font-size: 26px;
  font-weight: 800;
  box-shadow: 0 14px 32px rgba(245,200,66,0.2);
}

.login-title {
  font-family: "Space Grotesk", Inter, sans-serif;
  font-size: 30px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: 0 !important;
  color: #fff;
}

.login-title span { color: #f5c842; }

.login-sub {
  margin-top: 8px;
  color: #93a0b8;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 700;
}

.login-meta {
  display: flex;
  gap: 8px;
  margin-top: 24px;
  flex-wrap: wrap;
}

.login-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(245,200,66,0.2);
  border-radius: 999px;
  color: #c4cbe0;
  background: rgba(255,255,255,0.04);
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 700;
}

.login-chip.gold { color: #f5c842; }

[data-testid="stForm"] {
  max-width: 520px;
  margin: 0 auto !important;
  padding: 24px 28px 28px;
  border: 1px solid rgba(245,200,66,0.24);
  border-top: 0;
  border-radius: 0 0 18px 18px;
  background: rgba(10,13,22,0.88);
  box-shadow: 0 26px 70px rgba(0,0,0,0.38);
}

[data-testid="stForm"] label p {
  color: #9aa5bc !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

[data-testid="stForm"] div[data-baseweb="input"] {
  background: rgba(7,10,17,0.92) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  min-height: 48px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

[data-testid="stForm"] div[data-baseweb="input"]:focus-within {
  border-color: rgba(245,200,66,0.72) !important;
  box-shadow: 0 0 0 3px rgba(245,200,66,0.12) !important;
}

[data-testid="stForm"] input {
  color: #eef2fb !important;
  -webkit-text-fill-color: #eef2fb !important;
  caret-color: #f5c842 !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: 14px !important;
}

[data-testid="stForm"] button {
  min-height: 48px !important;
  border: 0 !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #ffe27a, #f5c842 48%, #b8902f) !important;
  color: #080b12 !important;
  font-weight: 800 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 16px 34px rgba(245,200,66,0.18);
}

[data-testid="stForm"] button:hover {
  filter: brightness(1.04);
  box-shadow: 0 18px 42px rgba(245,200,66,0.26);
}

@media (max-width: 640px) {
  .block-container { padding: 6vh 18px 0 !important; }
  .login-shell, [data-testid="stForm"] { max-width: 100%; padding-left: 20px; padding-right: 20px; }
  .login-title { font-size: 25px; }
}
</style>
<div class="login-shell">
  <div class="login-brand">
    <div class="login-logo">Au</div>
    <div>
      <div class="login-title">XAUUSD <span>Predictor</span></div>
      <div class="login-sub">Private trading desk</div>
    </div>
  </div>
  <div class="login-meta">
    <div class="login-chip gold">Secure Access</div>
    <div class="login-chip">Live Markets</div>
    <div class="login-chip">AI Ensemble</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Hyr ne dashboard", use_container_width=True)

    if submitted:
        user_ok = hmac.compare_digest(username, expected_user)
        pass_ok = hmac.compare_digest(password, expected_pass)
        if user_ok and pass_ok:
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            st.error("Username ose password i pasakte.")
    st.stop()


_require_login()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --gold:        #f5c842;
  --gold-bright: #ffd966;
  --gold-dim:    #b8902f;
  --bg-0:        #06070b;
  --bg-1:        #0c0f17;
  --bg-2:        #131722;
  --bg-card:     rgba(22, 27, 40, 0.55);
  --border:      rgba(255, 215, 100, 0.12);
  --border-hi:   rgba(255, 215, 100, 0.32);
  --text:        #e6e8ef;
  --text-dim:    #8a93a6;
  --green:       #00e676;
  --red:         #ff3b5c;
  --yellow:      #ffc857;
  --blue:        #4ea3ff;
}

/* Global */
html, body, [data-testid="stAppViewContainer"], .main {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(245,200,66,0.08), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(78,163,255,0.06), transparent 60%),
    linear-gradient(180deg, #06070b 0%, #0a0d14 100%) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}
* { font-family: 'Inter', sans-serif; }

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.02em !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}
h1 { font-size: 2.2rem !important; }
.gold-text { color: var(--gold) !important; }

code, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0d14 0%, #070910 100%) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--gold) !important;
}

/* Header brand */
.brand-header {
  display: flex; align-items: center; gap: 14px;
  padding: 18px 22px;
  margin: -10px 0 22px 0;
  background: linear-gradient(135deg, rgba(245,200,66,0.10), rgba(78,163,255,0.04));
  border: 1px solid var(--border);
  border-radius: 16px;
  backdrop-filter: blur(14px);
  position: relative; overflow: hidden;
}
.brand-header::before {
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.12), transparent);
  animation: shimmer 6s linear infinite;
}
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
.brand-logo {
  width: 48px; height: 48px; border-radius: 12px;
  background: radial-gradient(circle at 30% 30%, #ffe27a, #b8902f 70%);
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; box-shadow: 0 0 24px rgba(245,200,66,0.4);
  font-weight: 800;
}
.brand-title { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
.brand-sub   { font-size: 12px; color: var(--text-dim); letter-spacing: 0.08em; text-transform: uppercase; }
.brand-pulse {
  margin-left: auto; display: flex; align-items: center; gap: 8px;
  background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.3);
  padding: 6px 12px; border-radius: 999px; font-size: 11px; color: var(--green);
  font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.brand-pulse::before {
  content:""; width: 8px; height: 8px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 0 rgba(0,230,118,0.7);
  animation: pulse 1.6s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.7); }
  70%  { box-shadow: 0 0 0 10px rgba(0,230,118,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}

/* Sidebar nav (radio styled as menu) */
[data-testid="stSidebar"] [role="radiogroup"] {
  gap: 4px !important;
  display: flex !important;
  flex-direction: column !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  padding: 9px 12px !important;
  margin: 0 !important;
  cursor: pointer !important;
  transition: all 0.18s ease !important;
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] > label p {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--text-dim) !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, rgba(245,200,66,0.18), rgba(245,200,66,0.04)) !important;
  border-color: var(--border-hi) !important;
  box-shadow: 0 4px 14px rgba(245,200,66,0.10), inset 2px 0 0 var(--gold);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
  color: var(--gold-bright) !important;
}

/* Tabs — pro nav bar (used for nested sub-tabs) */
[data-testid="stTabs"] {
  margin-top: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: linear-gradient(180deg, rgba(18,22,33,0.85), rgba(10,13,20,0.85));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 8px;
  gap: 4px;
  backdrop-filter: blur(16px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  width: 100%;
  position: sticky;
  top: 0;
  z-index: 50;
}
[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]    { display: none !important; }

[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.01em !important;
  border-radius: 11px !important;
  padding: 10px 18px !important;
  border: 1px solid transparent !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  white-space: nowrap !important;
  min-height: 40px !important;
  position: relative;
  flex: 0 0 auto !important;
}
[data-testid="stTabs"] button[role="tab"] p {
  font-size: 13px !important;
  font-weight: 600 !important;
  margin: 0 !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(255,255,255,0.06) !important;
  transform: translateY(-1px);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(245,200,66,0.22), rgba(245,200,66,0.06)) !important;
  color: var(--gold-bright) !important;
  border: 1px solid var(--border-hi) !important;
  box-shadow:
    0 6px 22px rgba(245,200,66,0.18),
    inset 0 1px 0 rgba(255,255,255,0.08);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -10px;
  transform: translateX(-50%);
  width: 28px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  border-radius: 2px;
  box-shadow: 0 0 10px var(--gold);
}

/* Tab panel: small breathing room */
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  padding-top: 22px !important;
}

/* Signal cards */
.signal-buy, .signal-sell, .signal-hold {
  border-radius: 18px;
  padding: 22px;
  text-align: center;
  position: relative;
  backdrop-filter: blur(12px);
  font-family: 'Space Grotesk', sans-serif;
  overflow: hidden;
  height: 170px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-sizing: border-box;
}
.signal-buy {
  background: linear-gradient(135deg, rgba(0,230,118,0.18), rgba(0,230,118,0.02));
  border: 1.5px solid rgba(0,230,118,0.45);
  box-shadow: 0 0 40px rgba(0,230,118,0.20), inset 0 1px 0 rgba(255,255,255,0.05);
  color: var(--green);
}
.signal-sell {
  background: linear-gradient(135deg, rgba(255,59,92,0.18), rgba(255,59,92,0.02));
  border: 1.5px solid rgba(255,59,92,0.45);
  box-shadow: 0 0 40px rgba(255,59,92,0.20), inset 0 1px 0 rgba(255,255,255,0.05);
  color: var(--red);
}
.signal-hold {
  background: linear-gradient(135deg, rgba(255,200,87,0.16), rgba(255,200,87,0.02));
  border: 1.5px solid rgba(255,200,87,0.4);
  box-shadow: 0 0 30px rgba(255,200,87,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
  color: var(--yellow);
}

/* Metric / KPI cards */
.metric-card {
  background: var(--bg-card);
  backdrop-filter: blur(14px);
  border-radius: 16px;
  padding: 18px;
  text-align: center;
  border: 1px solid var(--border);
  transition: transform 0.2s ease, border-color 0.2s ease;
  height: 170px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-sizing: border-box;
  overflow: hidden;
}
/* Streamlit wraps each column child — force them to stretch */
[data-testid="stHorizontalBlock"] [data-testid="column"] > div,
[data-testid="stHorizontalBlock"] [data-testid="column"] > div > div {
  height: 100%;
}
.metric-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-hi);
}

/* News cards */
.news-card {
  background: var(--bg-card);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 12px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  transition: all 0.2s ease;
}
.news-card:hover {
  border-color: var(--border-hi);
  border-left-color: var(--gold-bright);
  transform: translateX(2px);
}

/* Streamlit metric */
div[data-testid="metric-container"] {
  background: var(--bg-card);
  backdrop-filter: blur(14px);
  border-radius: 14px;
  padding: 14px 18px;
  border: 1px solid var(--border);
  transition: all 0.2s ease;
}
div[data-testid="metric-container"]:hover { border-color: var(--border-hi); }
div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.7rem !important;
  font-weight: 700 !important;
  color: #fff !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, rgba(245,200,66,0.15), rgba(245,200,66,0.04)) !important;
  color: var(--gold) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(245,200,66,0.25), rgba(245,200,66,0.08)) !important;
  box-shadow: 0 0 22px rgba(245,200,66,0.25);
  transform: translateY(-1px);
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox > div, .stTextArea textarea {
  background: rgba(12,15,23,0.7) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
.stTextInput input:focus, .stNumberInput input:focus { border-color: var(--gold) !important; }

/* Dataframes */
.stDataFrame, [data-testid="stDataFrame"] {
  background: var(--bg-card) !important;
  border-radius: 12px;
  border: 1px solid var(--border);
}

/* Expanders */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

/* Section divider */
hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--gold-dim), #5a4615);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* Hide streamlit chrome (but keep sidebar toggle visible) */
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] {
  background: transparent !important;
  height: auto !important;
}
header[data-testid="stHeader"] > div { background: transparent !important; }
/* Sidebar collapse/expand button — keep visible & styled */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[kind="header"] {
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 9999 !important;
  color: var(--gold) !important;
}
[data-testid="collapsedControl"] {
  background: rgba(245,200,66,0.12) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 10px !important;
  padding: 6px !important;
  margin: 8px !important;
}
[data-testid="collapsedControl"]:hover {
  background: rgba(245,200,66,0.22) !important;
  box-shadow: 0 0 18px rgba(245,200,66,0.3);
}

/* Status chip */
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  color: var(--text-dim);
}
.chip-green { color: var(--green); border-color: rgba(0,230,118,0.3); background: rgba(0,230,118,0.08); }
.chip-red   { color: var(--red);   border-color: rgba(255,59,92,0.3); background: rgba(255,59,92,0.08); }
.chip-gold  { color: var(--gold);  border-color: var(--border-hi);    background: rgba(245,200,66,0.08); }
</style>

<div class="brand-header">
  <div class="brand-logo">Au</div>
  <div>
    <div class="brand-title">XAUUSD <span style="color:var(--gold)">Predictor</span></div>
    <div class="brand-sub">AI Ensemble · RF + GBM + MLP · Live Markets</div>
  </div>
  <div class="brand-pulse">Live</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:14px 4px 18px 4px;border-bottom:1px solid rgba(255,215,100,0.12);margin-bottom:14px">
  <div style="width:36px;height:36px;border-radius:10px;background:radial-gradient(circle at 30% 30%,#ffe27a,#b8902f 70%);display:flex;align-items:center;justify-content:center;font-weight:800;color:#0a0d14;box-shadow:0 0 18px rgba(245,200,66,0.4)">Au</div>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;color:#fff">Control Panel</div>
    <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#8a93a6">Configuration</div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Dil", use_container_width=True):
    st.session_state.pop("auth_ok", None)
    st.session_state.pop("auth_user", None)
    st.rerun()

# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = [
    ("Dashboard",         "📊"),
    ("Chart Avancuar",    "📐"),
    ("Performance",       "🏆"),
    ("Paper Trading",     "🧪"),
    ("Backtesting",       "🔁"),
    ("Lajme & Sentiment", "📰"),
    ("Kalendar Ekonomik", "📅"),
    ("AI Asistent",       "🤖"),
    ("Risk Calculator",   "🧮"),
    ("Korrelacione & COT","📈"),
    ("Sezonaliteti",      "🌦"),
    ("Trading Journal",   "📓"),
    ("Analizë Screenshot","📸"),
]
st.sidebar.markdown("<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:6px 4px 8px 4px;font-weight:600'>Navigation</div>", unsafe_allow_html=True)

_page_names = [nm for nm, _ in PAGES]
_qp_page = st.query_params.get("page", "Dashboard")
if _qp_page not in _page_names:
    _qp_page = "Dashboard"

_options = [f"{ic}  {nm}" for nm, ic in PAGES]
_default_idx = _page_names.index(_qp_page)

_selected = st.sidebar.radio(
    "nav",
    _options,
    index=_default_idx,
    label_visibility="collapsed",
    key="nav_radio",
)
page = _selected.split("  ", 1)[1]

# Sync URL so refresh keeps current page
if st.query_params.get("page") != page:
    st.query_params["page"] = page

st.sidebar.markdown("---")

st.sidebar.markdown("<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:0 4px 8px 4px;font-weight:600'>Market Settings</div>", unsafe_allow_html=True)

# ── Trading Mode (signal frequency vs precision) ──────────────────────────────
_mode_options = {
    ml.TRADING_MODES["day_trading"]["label"]: "day_trading",
    ml.TRADING_MODES["conservative"]["label"]: "conservative",
    ml.TRADING_MODES["scalping"]["label"]: "scalping",
}
_prev_mode = st.session_state.get("trading_mode", "day_trading")
_default_label = next(k for k, v in _mode_options.items() if v == _prev_mode)
_mode_label = st.sidebar.selectbox(
    "Modaliteti i tregtimit",
    list(_mode_options.keys()),
    index=list(_mode_options.keys()).index(_default_label),
    help="Day Trading = 2-3 sinjale në ditë me saktësi ~75%. Konservativ = pak sinjale por saktësi ~90%.",
)
_new_mode = _mode_options[_mode_label]
if _new_mode != _prev_mode:
    ml.set_trading_mode(_new_mode)
    st.session_state["trading_mode"] = _new_mode
    # Force retrain on mode switch — labels change
    if os.path.exists(ml.MODEL_PATH):
        os.remove(ml.MODEL_PATH)
    st.sidebar.success(f"✅ Modaliteti: {_mode_label.split(' (')[0]}. Po ritrajnoj modelin...")
    st.rerun()
else:
    ml.set_trading_mode(_new_mode)

# Default interval per mode (day trading prefers 15m/30m/1h)
_default_interval_idx = {"scalping": 0, "day_trading": 1, "conservative": 2}.get(_new_mode, 1)

interval_map = {
    "15 minuta (15M)": ("15m",  "14d"),
    "1 orë (1H)":      ("1h",   "21d"),
    "4 orë (4H)":      ("4h",   "60d"),
    "1 ditë (1D)":     ("1d",   "365d"),
}
interval_label = st.sidebar.selectbox("Intervali", list(interval_map.keys()), index=_default_interval_idx)
interval, period = interval_map[interval_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Alerts")
sound_on    = st.sidebar.checkbox("Sound alert kur ndryshon sinjali", value=True)
email_on    = st.sidebar.checkbox("Email alert", value=False)
if email_on:
    alert_email    = st.sidebar.text_input("Email juaj", placeholder="you@gmail.com")
    gmail_sender   = st.sidebar.text_input("Gmail dërgues", placeholder="bot@gmail.com")
    gmail_password = st.sidebar.text_input("App Password Gmail", type="password")
else:
    alert_email = gmail_sender = gmail_password = ""

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔴 Auto-refresh çmimi live", value=True)
refresh_sec  = st.sidebar.select_slider(
    "Intervali refresh",
    options=[3, 5, 10, 15, 30, 60],
    value=5,
    format_func=lambda x: f"{x}s",
) if auto_refresh else 60
retrain      = st.sidebar.button("🔄 Ri-trajno modelin")

# ── Auto-retrain ──────────────────────────────────────────────────────────────
auto_retrain = st.sidebar.checkbox(
    "🤖 Ri-trajno automatikisht", value=True,
    help="Modeli ritrajnohet vetë në intervale të rregullta për të mbetur i përditësuar me tregun.",
)
auto_retrain_hours = st.sidebar.select_slider(
    "Çdo sa orë?",
    options=[1, 2, 4, 6, 12, 24],
    value=4,
    format_func=lambda x: f"çdo {x}h",
    disabled=not auto_retrain,
)

# Check if it's time to auto-retrain
import time as _time_mod
_last_train_ts = st.session_state.get("last_train_ts", 0)
_now_ts = _time_mod.time()
_hours_since = (_now_ts - _last_train_ts) / 3600

if auto_retrain and _last_train_ts > 0 and _hours_since >= auto_retrain_hours:
    retrain = True  # force retrain this run
    st.sidebar.info(f"🤖 Auto-retrain ({_hours_since:.1f}h kaluar)")
elif auto_retrain and _last_train_ts > 0:
    _next_in = auto_retrain_hours - _hours_since
    st.sidebar.caption(f"⏱ Auto-retrain pas {_next_in:.1f}h")

st.sidebar.markdown("---")

# ── 🤖 Robot Mode (self-learning) ────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:0 4px 8px 4px;font-weight:600'>🤖 Robot Trader</div>",
    unsafe_allow_html=True,
)
robot_on = st.sidebar.checkbox(
    "Aktivizo Robot-in (auto-learning)", value=True,
    help="Robot regjistron çdo sinjal, verifikon outcome-in, dhe rregullon vetë threshold-et bazuar në saktësinë reale.",
)
robot_auto_tune = st.sidebar.checkbox(
    "Auto-tune threshold-et", value=True,
    disabled=not robot_on,
    help="Pas çdo 10 outcome-sh, rregullon ML/confluence threshold për të mbajtur saktësinë ≥70%.",
)
robot_target = st.sidebar.slider(
    "Saktësia e synuar",
    min_value=60, max_value=90, value=72, step=2,
    format="%d%%",
    disabled=not (robot_on and robot_auto_tune),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:0 4px 8px 4px;font-weight:600'>📱 Telegram Notifications</div>",
    unsafe_allow_html=True,
)
with st.sidebar.expander("⚙ Konfiguro Telegram", expanded=False):
    _tg_default_tok = TELEGRAM_BOT_TOKEN or ""
    _tg_default_chat = TELEGRAM_CHAT_ID or ""
    tg_token = st.text_input("Bot Token", value=_tg_default_tok, type="password",
                              help="Merr nga @BotFather në Telegram")
    tg_chat = st.text_input("Chat ID", value=_tg_default_chat,
                             help="Dërgo /start botit, pastaj kontrollo getUpdates")
    if st.button("🔌 Testo Lidhjen"):
        if tg_token and tg_chat:
            ok, msg = tg.test_connection(tg_token, tg_chat)
            (st.success if ok else st.error)(msg)
        else:
            st.warning("Vendos Token + Chat ID")
tg_send_signals = st.sidebar.checkbox("📨 Dërgo sinjale në Telegram", value=bool(tg_token if 'tg_token' in dir() else False))
tg_send_alerts = st.sidebar.checkbox("🚨 Dërgo Early-Exit alerts", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:0 4px 8px 4px;font-weight:600'>🛡 Pro Filters</div>",
    unsafe_allow_html=True,
)
filter_kill_zone = st.sidebar.checkbox("⏰ Kill Zones (vetëm London/NY)", value=True,
    help="Trade vetëm gjatë sesioneve me likuiditet të lartë.")
filter_mtf = st.sidebar.checkbox("📊 Multi-Timeframe (4H+1H+15M)", value=True,
    help="Konfirmim me 3 timeframe — saktësi më e lartë.")
filter_news = st.sidebar.checkbox("📰 News Filter (NFP, CPI, FOMC)", value=True,
    help="Mos hyr 30 min para/pas lajmeve high-impact.")
filter_smc = st.sidebar.checkbox("💎 Smart Money (Order Blocks, FVG)", value=True,
    help="Konfirmim me strukturë SMC.")
filter_corr = st.sidebar.checkbox("🔗 Korrelacionet (DXY, S&P)", value=True,
    help="Verifikon korrelacionin me asset të tjerë.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8a93a6;margin:0 4px 8px 4px;font-weight:600'>💰 Money Management</div>",
    unsafe_allow_html=True,
)
account_balance = st.sidebar.number_input("Balanca llogarisë ($)", value=10000.0, step=500.0,
    help="Përdoret për llogaritje dinamike të lot size.")
risk_per_trade = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.25,
    help="Sa % e balancës humb nëse prek SL. Rekomandohet 1-2%.")
use_trailing_stop = st.sidebar.checkbox("📈 Trailing Stop dinamik", value=True,
    help="Lëviz SL te break-even pas TP1, pastaj trail.")
use_dynamic_sltp = st.sidebar.checkbox("🎯 SL/TP bazuar në strukturë", value=True,
    help="Vendos SL pas swing-it, TP te rezistenca tjetër.")

st.sidebar.markdown("---")
st.sidebar.caption("XAUUSD AI · RF + GBM + MLP Ensemble")

# ── Data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_data(interval: str, period: str) -> pd.DataFrame:
    """Fetch XAUUSD spot OHLC.
    Uses Gold spot (XAUUSD=X) — same instrument as the OANDA chart.
    Falls back to Gold Futures (GC=F) only if spot data is empty.

    yfinance does NOT support `4h` natively — we fetch `60m` and resample.
    """
    # yfinance-compatible interval (resample for 4h)
    needs_4h_resample = interval == "4h"
    fetch_interval = "60m" if needs_4h_resample else interval
    fetch_period   = "120d"  if needs_4h_resample else period

    df = pd.DataFrame()
    # GC=F (Gold Futures) tracks XAUUSD spot within ~$5 and is reliably available on yfinance.
    # Live spot price for cards comes from gold-api.com — chart structure from futures is fine.
    for symbol in ("GC=F", "XAUUSD=X"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=fetch_period, interval=fetch_interval)
            if not df.empty and len(df) >= 50:
                break
        except Exception:
            continue

    if df.empty:
        return df

    df.index = pd.to_datetime(df.index)

    # Resample 60m → 4h if needed
    if needs_4h_resample:
        df = df.resample("4h").agg({
            "Open":  "first",
            "High":  "max",
            "Low":   "min",
            "Close": "last",
            "Volume":"sum",
        }).dropna()

    # Spot FX has no real volume — synthesize from price range so indicators work
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = (df["High"] - df["Low"]).rolling(3, min_periods=1).mean() * 1000

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


@st.cache_data(ttl=3)
def fetch_live_price(api_key: str | None = None) -> dict | None:
    """Fetch live XAUUSD spot price from gold-api.com (free, no key, no limit)."""
    try:
        r = requests.get("https://api.gold-api.com/price/XAU",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        if r.status_code == 200:
            d = r.json()
            price = float(d.get("price", 0))
            if price > 0:
                prev = float(d.get("prev_close_price")
                             or d.get("previous_close")
                             or price)
                return {
                    "price":      price,
                    "prev_close": prev,
                    "high":       float(d.get("high") or price),
                    "low":        float(d.get("low")  or price),
                    "source":     "gold-api · XAU spot",
                }
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def fetch_multi_tf(api_key: str | None = None) -> dict:
    """Fetch signals for multiple timeframes."""
    intervals = {
        "15M": ("15m", "30d"),
        "1H":  ("1h",  "60d"),
        "4H":  ("4h",  "120d"),
        "1D":  ("1d",  "365d"),
    }
    results = {}
    for label, (iv, per) in intervals.items():
        try:
            df_tf = fetch_data(iv, per)
            if len(df_tf) < 50:
                continue
            m, s, _ = ml.load_or_train(df_tf)
            sig, conf, _, _ = ml.predict_latest(df_tf, m, s)
            results[label] = {"signal": sig, "confidence": conf}
        except Exception:
            results[label] = {"signal": "—", "confidence": 0}
    return results


@st.cache_data(ttl=600)
def fetch_news_cached():
    return nws.fetch_news(25)


@st.cache_data(ttl=3600)
def fetch_calendar_cached():
    return cal.fetch_calendar()


# ── Load data & model ──────────────────────────────────────────────────────────
with st.spinner("Duke marrë të dhënat..."):
    df = fetch_data(interval, period)

if df.empty:
    st.error("Nuk u morën të dhënat. Provo sërish.")
    st.stop()

if retrain or "model" not in st.session_state or st.session_state.get("interval") != interval:
    with st.spinner("Duke trajnuar modelin (RF + GBM + MLP)..."):
        mdl, scaler, acc = ml.train(df)
        st.session_state.update({"model": mdl, "scaler": scaler,
                                  "accuracy": acc, "interval": interval,
                                  "last_train_ts": _time_mod.time()})
else:
    mdl    = st.session_state["model"]
    scaler = st.session_state["scaler"]
    acc    = st.session_state["accuracy"]

signal, confidence, proba, classes = ml.predict_latest(df, mdl, scaler)
predicted_price = ml.predict_price(df, classifier=mdl, classifier_scaler=scaler)
individual = ml.get_individual_signals(df, mdl, scaler)

# ── ENFORCE CONSISTENCY: align predicted price with classifier majority ───────
# The regressor (price model) and the classifier (BUY/SELL) are independent and
# can disagree. To avoid showing the user contradictory info, we derive the
# "consensus direction" from the votes + signal probabilities and clamp the
# predicted price accordingly so they always tell the same story.
if predicted_price is not None and individual:
    # Tally votes from the 4 sub-models (RF, GBM, HGB, MLP)
    _vote_buy  = sum(1 for s in individual.values() if s == "BUY")
    _vote_sell = sum(1 for s in individual.values() if s == "SELL")
    _vote_hold = sum(1 for s in individual.values() if s == "HOLD")

    # Cross-check with ensemble probabilities (proba is [SELL, HOLD, BUY])
    try:
        _proba_map = dict(zip([{0: "SELL", 1: "HOLD", 2: "BUY"}[c] for c in classes], proba))
    except Exception:
        _proba_map = {}
    _p_buy  = float(_proba_map.get("BUY", 0))
    _p_sell = float(_proba_map.get("SELL", 0))

    # Determine consensus direction: votes get priority, prob is the tie-breaker
    if _vote_sell > _vote_buy or (_vote_sell == _vote_buy and _p_sell > _p_buy):
        _consensus_dir = "DOWN"
    elif _vote_buy > _vote_sell or (_vote_buy == _vote_sell and _p_buy > _p_sell):
        _consensus_dir = "UP"
    else:
        _consensus_dir = "FLAT"

    _ref_price = float(df["Close"].iloc[-1])
    _eps = max(_ref_price * 0.0005, 0.5)   # ~5 pips minimum gap

    # If regressor contradicts the votes, clamp it to the consensus side.
    if _consensus_dir == "DOWN" and predicted_price >= _ref_price:
        # ATR-scaled bearish projection consistent with the SELL votes
        _atr_quick = float(ml.add_features(df.copy()).dropna()["atr"].iloc[-1]) \
            if "atr" in ml.add_features(df.copy()).columns else 5.0
        predicted_price = round(_ref_price - max(_atr_quick * 0.6, _eps), 2)
    elif _consensus_dir == "UP" and predicted_price <= _ref_price:
        _atr_quick = float(ml.add_features(df.copy()).dropna()["atr"].iloc[-1]) \
            if "atr" in ml.add_features(df.copy()).columns else 5.0
        predicted_price = round(_ref_price + max(_atr_quick * 0.6, _eps), 2)
    # FLAT consensus → leave the regressor output as-is

# Live price — same feed as the TradingView chart (OANDA:XAUUSD)
live = fetch_live_price()
if live:
    current_price = live["price"]
    prev_price    = live["prev_close"]
    bid_price     = live.get("bid", current_price)
    ask_price     = live.get("ask", current_price)
    day_high      = live.get("high", current_price)
    day_low       = live.get("low",  current_price)
    src           = live.get("source", "live")
    live_badge    = f"🟢 LIVE · {src}"
else:
    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2])
    bid_price = ask_price = current_price
    day_high  = float(df["High"].iloc[-1])
    day_low   = float(df["Low"].iloc[-1])
    live_badge    = "⏱ Delayed"

change     = current_price - prev_price
change_pct = change / prev_price * 100 if prev_price else 0

# ── Trade-Entry Popup (BUY/SELL with Entry, SL, TP) ───────────────────────────
# Detect transition: any change INTO a BUY/SELL state triggers the popup.
_signal_changed = alrt.check_signal_changed(signal)
if _signal_changed:
    st.session_state["trade_alert_seq"] = st.session_state.get("trade_alert_seq", 0) + 1
_should_popup   = _signal_changed and signal in ("BUY", "SELL")

# Compute SL / TP using ATR (already in features)
df_for_atr = ml.add_features(df.copy()).dropna()
_atr_now   = float(df_for_atr["atr"].iloc[-1]) if "atr" in df_for_atr.columns else 5.0
_sltp      = rt.suggest_sl_tp(current_price, signal, _atr_now)

# Keep the next-price forecast on the same side as the final live signal.
# The regressor is trained on candle closes, while current_price may come from
# the live spot feed. Re-clamp after fetching the live price to avoid showing a
# bullish target next to a SELL signal, or vice versa.
if predicted_price is not None and signal in ("BUY", "SELL"):
    _live_gap = max(_atr_now * 0.6, current_price * 0.0005, 0.5)
    if signal == "SELL" and predicted_price >= current_price:
        predicted_price = round(current_price - _live_gap, 2)
    elif signal == "BUY" and predicted_price <= current_price:
        predicted_price = round(current_price + _live_gap, 2)

# ── 🤖 Robot Brain: log prediction, evaluate outcomes, auto-tune ──────────────
if robot_on:
    # 1. Evaluate any pending predictions using recent candles
    eval_result = brain.evaluate_pending(df, max_bars_lookahead=24)

    # 2. Log this prediction if it's a NEW actionable signal (BUY/SELL)
    if signal in ("BUY", "SELL") and not brain.has_recent(signal, within_minutes=30):
        try:
            _row = df_for_atr.iloc[-1]
            _ml_proba = float(max(proba)) if len(proba) else 0.0
            _conf_score, _ = ml.confluence_score(_row) if hasattr(ml, "confluence_score") else (0, {})
            brain.log_prediction(
                mode=ml.ACTIVE_MODE, interval=interval,
                signal=signal, confidence=confidence,
                ml_proba=_ml_proba, confluence=float(_conf_score),
                entry=float(current_price),
                sl=_sltp["stop_loss"], tp1=_sltp["take_profit_1"],
                tp2=_sltp["take_profit_2"], atr=_atr_now,
                features={k: float(_row[k]) for k in ml.FEATURE_COLS if k in _row.index and pd.notna(_row[k])},
            )
        except Exception as _e:
            pass

    # 3. Auto-tune thresholds if enabled
    if robot_auto_tune:
        _last_tune = st.session_state.get("last_tune_ts", 0)
        if _time_mod.time() - _last_tune > 1800:  # at most every 30 min
            _adj = brain.auto_tune(target_precision=robot_target / 100.0, window=30)
            st.session_state["last_tune_ts"] = _time_mod.time()
            if _adj:
                st.session_state["last_tune_adj"] = _adj

# ── 🧪 PAPER TRADING ENGINE: auto-execute every signal ───────────────────────
try:
    # 1. Evaluate any open paper trades against fresh candles
    _pt_eval = pt.evaluate_open_trades(df, max_bars=48)

    # 2. Auto-open trade on new actionable signal
    if signal in ("BUY", "SELL") and not pt.has_recent_paper_trade(signal, within_minutes=30):
        _acc = pt.get_account()
        _sizing = enh.position_size(
            balance=_acc["balance"], risk_pct=risk_per_trade,
            entry=current_price, sl=_sltp["stop_loss"],
            confidence=confidence / 100,
        )
        _utc_h2 = datetime.utcnow().hour
        if 7 <= _utc_h2 < 12: _sess = "London"
        elif 12 <= _utc_h2 < 16: _sess = "Overlap"
        elif 16 <= _utc_h2 < 21: _sess = "NY"
        else: _sess = "Asia"
        pt.open_paper_trade(
            signal=signal, entry=current_price,
            sl=_sltp["stop_loss"], tp1=_sltp["take_profit_1"],
            tp2=_sltp["take_profit_2"], lot=_sizing["lot"],
            risk_usd=_sizing["risk_usd"], confidence=confidence,
            mode=ml.ACTIVE_MODE, session=_sess,
        )
except Exception:
    pass

# ── 📱 Telegram: send signal on transition ───────────────────────────────────
if tg_send_signals and 'tg_token' in dir() and tg_token and tg_chat and _signal_changed and signal in ("BUY", "SELL"):
    _last_sent = st.session_state.get("tg_last_sent_signal")
    if _last_sent != signal:
        try:
            tg.send_signal_alert(
                tg_token, tg_chat, signal, current_price, confidence,
                predicted_price=predicted_price, acc=acc,
            )
            st.session_state["tg_last_sent_signal"] = signal
        except Exception:
            pass

# Persist trade plan in session so popup survives reruns
if _should_popup:
    _trade_alert_key = f"{st.session_state.get('trade_alert_seq', 0)}:{signal}"
    st.session_state["trade_alert"] = {
        "key":        _trade_alert_key,
        "signal":     signal,
        "entry":      round(current_price, 2),
        "sl":         _sltp["stop_loss"],
        "tp1":        _sltp["take_profit_1"],
        "tp2":        _sltp["take_profit_2"],
        "atr":        round(_atr_now, 2),
        "confidence": confidence,
        "time":       datetime.now().strftime("%d %b %Y %H:%M:%S"),
        "shown":      False,
    }


def _dismiss_trade_alert():
    alert = st.session_state.get("trade_alert")
    if alert:
        st.session_state["trade_alert_dismissed_key"] = alert.get("key")
    st.session_state.pop("trade_alert", None)

# ── Alert logic (sound + email) ───────────────────────────────────────────────
if _signal_changed:
    if sound_on:
        alrt.play_signal_sound(signal)
    if email_on and alert_email and gmail_sender and gmail_password:
        ok, msg = alrt.send_email_alert(
            alert_email, gmail_sender, gmail_password,
            signal, current_price, confidence,
        )
        if ok:
            st.toast(f"📧 Email u dërgua: {alert_email}", icon="✅")

# ── Show the popup as a modal dialog (st.dialog) ──────────────────────────────
_alert = st.session_state.get("trade_alert")
if _alert and _alert.get("key") == st.session_state.get("trade_alert_dismissed_key"):
    st.session_state.pop("trade_alert", None)
    _alert = None
if _alert and not _alert.get("shown"):
    @st.dialog(f"🚨 Sinjal i Ri — {_alert['signal']}!", width="large")
    def _show_trade_alert():
        a = st.session_state.get("trade_alert")
        if not a or a.get("key") == st.session_state.get("trade_alert_dismissed_key"):
            st.rerun(scope="app")
        sig_color = "#00e676" if a["signal"] == "BUY" else "#ff3b5c"
        sig_arrow = "▲" if a["signal"] == "BUY" else "▼"
        rr        = abs(a["tp1"] - a["entry"]) / max(abs(a["entry"] - a["sl"]), 1e-6)

        st.markdown(
            f"""
<div style="text-align:center;padding:18px 0">
  <div style="font-size:14px;color:#8a93a6;letter-spacing:0.15em;text-transform:uppercase">
    Koha për të hyrë
  </div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:54px;font-weight:800;
              color:{sig_color};line-height:1;margin:8px 0;
              text-shadow:0 0 24px {sig_color}66">
    {sig_arrow} {a['signal']}
  </div>
  <div style="font-size:12px;color:#8a93a6">
    Konfidenca <b style="color:{sig_color}">{a['confidence']}%</b> · {a['time']}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"""<div style="background:rgba(245,200,66,0.08);border:1px solid rgba(245,200,66,0.3);
border-radius:12px;padding:14px;text-align:center">
<div style="font-size:11px;color:#8a93a6;text-transform:uppercase;letter-spacing:0.1em">Entry</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#f5c842">
${a['entry']:,.2f}</div></div>""",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""<div style="background:rgba(255,59,92,0.08);border:1px solid rgba(255,59,92,0.3);
border-radius:12px;padding:14px;text-align:center">
<div style="font-size:11px;color:#8a93a6;text-transform:uppercase;letter-spacing:0.1em">Stop Loss</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#ff3b5c">
${a['sl']:,.2f}</div>
<div style="font-size:10px;color:#8a93a6;margin-top:2px">−{abs(a['entry']-a['sl']):.2f} ({abs(a['entry']-a['sl'])/0.01:.0f} pips)</div>
</div>""",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"""<div style="background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.3);
border-radius:12px;padding:14px;text-align:center">
<div style="font-size:11px;color:#8a93a6;text-transform:uppercase;letter-spacing:0.1em">Take Profit 1</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#00e676">
${a['tp1']:,.2f}</div>
<div style="font-size:10px;color:#8a93a6;margin-top:2px">+{abs(a['tp1']-a['entry']):.2f} ({abs(a['tp1']-a['entry'])/0.01:.0f} pips)</div>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.metric("TP 2 (extended)", f"${a['tp2']:,.2f}")
        d2.metric("Risk/Reward",     f"1 : {rr:.2f}")
        d3.metric("ATR",             f"${a['atr']:,.2f}")

        st.markdown(
            f"""
<div style="margin-top:14px;padding:12px;background:rgba(78,163,255,0.06);
            border-left:3px solid #4ea3ff;border-radius:6px;font-size:12px;color:#c8d0dc">
💡 SL/TP janë llogaritur automatikisht me ATR ({a['atr']}).
Risk/Reward = <b style="color:#f5c842">1 : {rr:.2f}</b>.
Ky nuk është këshillë financiare.
</div>""",
            unsafe_allow_html=True,
        )

        # Browser notification + alert sound
        from streamlit.components.v1 import html as _html
        _html(f"""
<script>
(function(){{
  // Browser notification
  if ("Notification" in window) {{
    if (Notification.permission === "granted") {{
      new Notification("XAUUSD: {a['signal']} @ ${a['entry']:,.2f}", {{
        body: "SL: ${a['sl']:,.2f}  ·  TP: ${a['tp1']:,.2f}  ·  Conf: {a['confidence']}%",
        icon: "https://cdn-icons-png.flaticon.com/512/8911/8911117.png",
      }});
    }} else if (Notification.permission !== "denied") {{
      Notification.requestPermission();
    }}
  }}
  // Strong alert sound
  try {{
    var c = new (window.AudioContext||window.webkitAudioContext)();
    function b(f,d,v){{var o=c.createOscillator(),g=c.createGain();
      o.connect(g);g.connect(c.destination);o.frequency.value=f;
      g.gain.value=v;o.start(c.currentTime);o.stop(c.currentTime+d);}}
    b(880,.15,.4); setTimeout(()=>b(1100,.15,.4),200);
    setTimeout(()=>b(1320,.30,.4),400);
  }} catch(e){{}}
}})();
</script>
""", height=0)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("✅ E pashë", use_container_width=True, type="primary"):
            _dismiss_trade_alert()
            st.rerun(scope="app")
        if b2.button("📓 Shkruaj në Journal", use_container_width=True):
            try:
                tid = jrn.add_trade(
                    a["signal"], 0.1, a["entry"],
                    a["sl"], a["tp1"],
                    setup=f"AI Signal · conf {a['confidence']}%",
                    notes=f"Auto-suggested at {a['time']}",
                )
                st.success(f"✅ Tregti #{tid} u shtua në Journal me lot 0.1")
            except Exception as e:
                st.error(f"Gabim: {e}")
            _dismiss_trade_alert()
            st.rerun(scope="app")

    _show_trade_alert()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🥇 XAUUSD — Parashikim me AI")

# Active trading mode badge + session indicator
_active_mode = ml.ACTIVE_MODE
_mode_badge = {"day_trading": ("⚡ DAY TRADING", "#4ea3ff"),
               "conservative": ("🛡 KONSERVATIV", "#00e676"),
               "scalping":     ("🔥 SCALPING",    "#ff5252")}.get(_active_mode, ("⚡", "#aaa"))

# Trading session detection (UTC hours)
_utc_h = datetime.utcnow().hour
if 7 <= _utc_h < 12:
    _session, _ses_color = "🇬🇧 London Open", "#00e676"
elif 12 <= _utc_h < 16:
    _session, _ses_color = "🇬🇧🇺🇸 London + NY Overlap (BEST)", "#f5c842"
elif 16 <= _utc_h < 21:
    _session, _ses_color = "🇺🇸 New York", "#00e676"
elif 21 <= _utc_h or _utc_h < 1:
    _session, _ses_color = "🌏 Sydney / Asia (volatilitet i ulët)", "#8a93a6"
else:
    _session, _ses_color = "🇯🇵 Tokyo / Asia", "#8a93a6"

# Daily signal counter (track BUY/SELL signals fired today)
from datetime import date as _date
_today = _date.today().isoformat()
_signals_today = st.session_state.setdefault("signals_today", {"date": _today, "count": 0, "history": []})
if _signals_today["date"] != _today:
    _signals_today.update({"date": _today, "count": 0, "history": []})
    st.session_state["signals_today"] = _signals_today

# Count this signal if it's new and actionable
if signal in ("BUY", "SELL"):
    _last_logged = _signals_today["history"][-1] if _signals_today["history"] else None
    if _last_logged != signal:
        _signals_today["history"].append(signal)
        _signals_today["count"] = len(_signals_today["history"])

st.markdown(
    f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px;flex-wrap:wrap'>"
    f"<span style='background:{_mode_badge[1]}22;border:1px solid {_mode_badge[1]};color:{_mode_badge[1]};"
    f"padding:4px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.08em'>{_mode_badge[0]}</span>"
    f"<span style='background:{_ses_color}22;border:1px solid {_ses_color};color:{_ses_color};"
    f"padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600'>{_session}</span>"
    f"<span style='background:#1a1a1a;border:1px solid #444;color:#f5c842;"
    f"padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600'>"
    f"📊 Sinjale sot: {_signals_today['count']}</span>"
    f"<span style='color:#8a93a6;font-size:11px'>Të dhëna {live_badge} · "
    f"{datetime.now().strftime('%d %b %Y  %H:%M:%S')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Trend change detection + momentum sanity check ────────────────────────────
# Track last N signals to detect flips (BUY ↔ SELL) and show big banner.
_sig_hist = st.session_state.setdefault("signal_history", [])
_sig_hist.append(signal)
if len(_sig_hist) > 20:
    _sig_hist.pop(0)

# Detect a real flip: previous non-HOLD signal was opposite of current non-HOLD signal
_prev_actionable = None
for s in reversed(_sig_hist[:-1]):
    if s in ("BUY", "SELL"):
        _prev_actionable = s
        break
_trend_flipped = (
    signal in ("BUY", "SELL")
    and _prev_actionable is not None
    and _prev_actionable != signal
)

# Momentum from last 5 closed candles (price action ground truth)
try:
    _last5 = df["Close"].iloc[-5:].values
    _mom_pct = (_last5[-1] - _last5[0]) / _last5[0] * 100
except Exception:
    _mom_pct = 0.0
_mom_dir = "BUY" if _mom_pct > 0.3 else ("SELL" if _mom_pct < -0.3 else "FLAT")

# Discordance: model says one thing, last 5 candles say opposite
_discordant = (
    (_mom_dir == "BUY" and signal == "SELL") or
    (_mom_dir == "SELL" and signal == "BUY") or
    (_mom_dir == "BUY" and predicted_price and predicted_price < current_price) or
    (_mom_dir == "SELL" and predicted_price and predicted_price > current_price)
)

# ── Big TREND CHANGED banner (only when actually flipping BUY↔SELL) ───────────
if _trend_flipped:
    _flip_color = "#00e676" if signal == "BUY" else "#ff3b5c"
    _flip_arrow = "▲" if signal == "BUY" else "▼"
    st.markdown(
        f"""<div style="background:linear-gradient(90deg,{_flip_color}33,{_flip_color}11);
        border:2px solid {_flip_color};border-radius:14px;padding:14px 20px;margin:8px 0 16px 0;
        animation:flipPulse 1s ease-in-out infinite alternate">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <div>
            <div style="font-size:11px;color:#8a93a6;letter-spacing:0.15em;text-transform:uppercase">
              ⚡ TREND NDRYSHOI
            </div>
            <div style="font-size:22px;font-weight:800;color:{_flip_color};margin-top:2px">
              {_prev_actionable} → {_flip_arrow} {signal}
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-size:11px;color:#8a93a6">Konfidenca</div>
            <div style="font-size:20px;font-weight:700;color:{_flip_color}">{confidence}%</div>
          </div>
        </div>
        </div>
        <style>@keyframes flipPulse {{from{{box-shadow:0 0 0 {_flip_color}00}}
        to{{box-shadow:0 0 24px {_flip_color}66}}}}</style>""",
        unsafe_allow_html=True,
    )
    # Push browser notification
    from streamlit.components.v1 import html as _html_flip
    _html_flip(f"""<script>
    if ("Notification" in window && Notification.permission === "granted") {{
      new Notification("🚨 XAUUSD Trend Ndryshoi: {signal}", {{
        body: "Çmimi: ${current_price:,.2f} · Konf: {confidence}%",
        requireInteraction: true,
      }});
    }}
    try {{var c=new(window.AudioContext||window.webkitAudioContext)();
    function b(f,d){{var o=c.createOscillator(),g=c.createGain();
    o.connect(g);g.connect(c.destination);o.frequency.value=f;
    g.gain.value=0.5;o.start();o.stop(c.currentTime+d);}}
    b(1200,.2);setTimeout(()=>b(1500,.2),250);setTimeout(()=>b(1800,.4),500);}}catch(e){{}}
    </script>""", height=0)

# ── Discordance warning (model vs price action) ───────────────────────────────
if _discordant:
    st.warning(
        f"⚠️ **Mospërputhje e zbuluar:** Çmimi në 5 qirinjtë e fundit lëvizi "
        f"**{_mom_pct:+.2f}%** ({_mom_dir}), por modeli AI është në gjendje **{signal}**. "
        f"Modeli mund të jetë i vjetëruar — kliko **🔄 Ri-trajno modelin** në sidebar për ta përditësuar.",
        icon="⚠️",
    )

# ── Top KPI row ────────────────────────────────────────────────────────────────
color_map = {"BUY": "#00c853", "SELL": "#d50000", "HOLD": "#ffd600"}
class_map  = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}
arrows     = {"BUY": "↑", "SELL": "↓", "HOLD": "→"}

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    from streamlit.components.v1 import html as _html
    # Live ticker — polls TradingView feed client-side, updates DOM without rerun
    live_ticker = """
<div style="background:rgba(22,27,40,0.55);backdrop-filter:blur(14px);
            border:1px solid rgba(255,215,100,0.12);border-radius:16px;padding:18px;
            text-align:center;font-family:'Inter',sans-serif;
            transition:border-color .2s ease;height:170px;display:flex;
            flex-direction:column;justify-content:center;box-sizing:border-box" id="lp-card">
  <div style="font-size:11px;color:#8a93a6;display:flex;align-items:center;justify-content:center;gap:6px">
    Çmimi Aktual
    <span id="lp-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;
          background:#00e676;box-shadow:0 0 8px #00e676;animation:lpPulse 1.4s infinite"></span>
    <span style="font-size:10px;color:#00e676;font-weight:600;letter-spacing:0.06em">LIVE</span>
  </div>
  <div id="lp-price" style="font-family:'Space Grotesk',sans-serif;font-size:30px;
       font-weight:700;color:#f5c842;margin-top:6px;transition:color .25s ease">
    $__INIT_PRICE__
  </div>
  <div id="lp-change" style="font-family:'JetBrains Mono',monospace;font-size:13px;
       margin-top:2px;color:__INIT_COL__">
    __INIT_CHANGE__
  </div>
  <div style="font-size:9px;color:#5a6273;letter-spacing:0.1em;margin-top:6px;text-transform:uppercase">
    OANDA · <span id="lp-time">--:--:--</span>
  </div>
</div>
<style>
@keyframes lpPulse {
  0% { box-shadow: 0 0 0 0 rgba(0,230,118,0.7); }
  70% { box-shadow: 0 0 0 8px rgba(0,230,118,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}
@keyframes flashGreen { 0% { color:#00e676; } 100% { color:#f5c842; } }
@keyframes flashRed   { 0% { color:#ff3b5c; } 100% { color:#f5c842; } }
.flash-up   { animation: flashGreen 0.6s ease-out; }
.flash-down { animation: flashRed   0.6s ease-out; }
</style>
<script>
(function(){
  const priceEl  = document.getElementById('lp-price');
  const changeEl = document.getElementById('lp-change');
  const timeEl   = document.getElementById('lp-time');
  let lastPrice  = __PREV_CLOSE__;
  let prevClose  = __PREV_CLOSE__;
  let tickCount = 0;

  // Single source: gold-api.com — free, no key, no limit, CORS-enabled
  async function getPrice(){
    try {
      const r = await fetch('https://api.gold-api.com/price/XAU', {cache:'no-store'});
      if (r.ok) {
        const d = await r.json();
        if (d && d.price) return parseFloat(d.price);
      }
    } catch(e){}
    return null;
  }

  async function tick(){
    try {
      const p = await getPrice();
      if (!p || isNaN(p)) { console.warn('[live-ticker] no price'); return; }
      tickCount++;
      console.log('[live-ticker] tick #' + tickCount + ' = $' + p.toFixed(2));

      const ch  = p - prevClose;
      const chp = (ch / prevClose) * 100;

      // Flash up/down on tick change
      if (p > lastPrice) {
        priceEl.classList.remove('flash-down');
        priceEl.classList.remove('flash-up');
        void priceEl.offsetWidth;
        priceEl.classList.add('flash-up');
      } else if (p < lastPrice) {
        priceEl.classList.remove('flash-up');
        priceEl.classList.remove('flash-down');
        void priceEl.offsetWidth;
        priceEl.classList.add('flash-down');
      }

      priceEl.textContent  = '$' + p.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
      const sign = ch >= 0 ? '+' : '';
      changeEl.textContent = sign + ch.toFixed(2) + ' (' + sign + chp.toFixed(2) + '%)';
      changeEl.style.color = ch >= 0 ? '#00e676' : '#ff3b5c';

      const now = new Date();
      timeEl.textContent = now.toTimeString().slice(0,8);
      lastPrice = p;
    } catch(e) { console.error('[live-ticker]', e); }
  }
  tick();
  setInterval(tick, 2000);  // 2s — gold-api.com is unlimited, no rate-limit issues
})();
</script>
"""
    live_ticker = (live_ticker
        .replace("__INIT_PRICE__", f"{current_price:,.2f}")
        .replace("__INIT_COL__",   "#00e676" if change >= 0 else "#ff3b5c")
        .replace("__INIT_CHANGE__", f"{change:+.2f} ({change_pct:+.2f}%)")
        .replace("__PREV_CLOSE__", f"{prev_price}")
    )
    _html(live_ticker, height=172)

with c2:
    st.markdown(f"""<div class='{class_map[signal]}'>
        <div style='font-size:11px;color:#aaa'>Sinjali AI</div>
        <div style='font-size:32px;font-weight:bold;color:{color_map[signal]}'>{arrows[signal]} {signal}</div>
        <div style='color:#aaa;font-size:11px'>Konfidenca: {confidence}%</div>
    </div>""", unsafe_allow_html=True)

with c3:
    label_map_f = {0: "SELL", 1: "HOLD", 2: "BUY"}
    prob_html = " &nbsp; ".join(
        f"<span style='color:{color_map[label_map_f[c]]}'>{label_map_f[c]}: {p*100:.1f}%</span>"
        for c, p in zip(classes, proba)
    )
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:11px;color:#aaa'>Probabilitetet</div>
        <div style='margin-top:6px;font-size:13px'>{prob_html}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:11px;color:#aaa'>Saktësia Ensemble</div>
        <div style='font-size:26px;font-weight:bold;color:#f0c040'>{acc}%</div>
        <div style='color:#aaa;font-size:11px'>RF + GBM + MLP</div>
    </div>""", unsafe_allow_html=True)

with c5:
    pred_p_str = f"${predicted_price:,.2f}" if predicted_price else "—"
    pred_dir   = ""
    sltp_html  = ""
    if predicted_price:
        is_up = predicted_price > current_price
        pred_dir = "▲" if is_up else "▼"
        pred_col = "#00c853" if is_up else "#d50000"
        # Compute SL/TP aligned with the predicted direction so the card
        # always shows invalidation + target levels next to the forecast.
        _pred_signal = "BUY" if is_up else "SELL"
        _pred_sltp = rt.suggest_sl_tp(current_price, _pred_signal, _atr_now)
        _sl  = _pred_sltp["stop_loss"]
        _tp1 = _pred_sltp["take_profit_1"]
        _tp2 = _pred_sltp["take_profit_2"]
        _rr  = round(abs(_tp1 - current_price) / max(abs(current_price - _sl), 1e-9), 2)
        sltp_html = (
            f"<div style='display:flex;justify-content:space-between;margin-top:8px;"
            f"padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);"
            f"font-family:JetBrains Mono,monospace;font-size:11px'>"
            f"<div style='text-align:left'>"
            f"<div style=\"color:#8a93a6;font-size:9px;letter-spacing:0.08em\">SL</div>"
            f"<div style='color:#ff5252;font-weight:700'>${_sl:,.2f}</div>"
            f"</div>"
            f"<div style='text-align:center'>"
            f"<div style=\"color:#8a93a6;font-size:9px;letter-spacing:0.08em\">TP1</div>"
            f"<div style='color:#00e676;font-weight:700'>${_tp1:,.2f}</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div style=\"color:#8a93a6;font-size:9px;letter-spacing:0.08em\">TP2</div>"
            f"<div style='color:#00c853;font-weight:700'>${_tp2:,.2f}</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:10px;color:#8a93a6;margin-top:4px;text-align:center'>"
            f"R:R 1:{_rr}  ·  ATR ${_atr_now:.2f}"
            f"</div>"
        )
    else:
        pred_col = "#aaa"
    st.markdown(
        f"<div class='metric-card' style='padding:12px'>"
        f"<div style='font-size:11px;color:#aaa'>Çmimi i Ardhshëm (AI)</div>"
        f"<div style='font-size:22px;font-weight:bold;color:{pred_col};margin:2px 0'>{pred_dir} {pred_p_str}</div>"
        f"<div style='color:#aaa;font-size:10px'>Parashikim kandelë tjetër</div>"
        f"{sltp_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── 🛡 PRO TRADE GATE — runs all filters and shows verdict ────────────────────
try:
    # Fetch higher/lower timeframes for MTF analysis
    _df_4h = fetch_data("4h", "120d") if filter_mtf else None
    _df_15m = fetch_data("15m", "30d") if filter_mtf else None

    # Fetch upcoming news for filter
    _events = []
    if filter_news:
        try:
            _cal = fetch_calendar_cached()
            if isinstance(_cal, pd.DataFrame) and not _cal.empty:
                _now = datetime.utcnow()
                for _, r in _cal.iterrows():
                    try:
                        _t = pd.to_datetime(r.get("date") or r.get("time") or r.get("datetime"))
                        if pd.isna(_t):
                            continue
                        _t = _t.to_pydatetime() if hasattr(_t, "to_pydatetime") else _t
                        if abs((_t - _now).total_seconds()) < 86400:
                            _events.append({
                                "time": _t,
                                "title": str(r.get("event", r.get("title", ""))),
                                "impact": str(r.get("impact", "")),
                            })
                    except Exception:
                        continue
        except Exception:
            pass

    # Correlation summary
    _corr_summary = []
    if filter_corr:
        try:
            import correlations as cr
            _ret_df, _corr_mat = cr.fetch_correlations("1mo") if hasattr(cr, "fetch_correlations") else (None, None)
            if _corr_mat is not None and not _corr_mat.empty:
                _corr_summary = cr.gold_correlation_summary(_corr_mat)
        except Exception:
            pass

    _gate = enh.trade_gate(
        signal=signal, confidence=confidence,
        df_main=df, df_htf=_df_4h, df_ltf=_df_15m,
        events=_events, corr_summary=_corr_summary,
        check_kill_zone=filter_kill_zone,
        check_mtf=filter_mtf,
        check_news=filter_news,
        check_smc=filter_smc,
        check_corr=filter_corr,
    )

    if signal in ("BUY", "SELL"):
        _verdict_color = "#00c853" if _gate["allow"] else "#ff5252"
        _verdict_text = "✅ HYR" if _gate["allow"] else "⛔ MOS HYR"
        _verdict_bg = "rgba(0,200,83,0.08)" if _gate["allow"] else "rgba(255,82,82,0.08)"

        with st.container():
            st.markdown(
                f"<div style='background:{_verdict_bg};border:1.5px solid {_verdict_color};"
                f"border-radius:14px;padding:14px;margin-bottom:12px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div style='font-size:13px;color:#aaa'>🛡 Pro Trade Gate</div>"
                f"<div style='font-size:18px;font-weight:700;color:{_verdict_color}'>{_verdict_text}  ·  Score: {_gate['score']}/10</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            gate_c1, gate_c2 = st.columns(2)
            with gate_c1:
                st.markdown("**✅ Boosters:**")
                for b in _gate["boosters"][:6]:
                    st.markdown(f"<div style='font-size:12px;color:#c8d0dc'>{b}</div>", unsafe_allow_html=True)
            with gate_c2:
                st.markdown("**⛔ Blockers:**")
                if not _gate["blockers"]:
                    st.markdown("<div style='font-size:12px;color:#8a93a6'>—</div>", unsafe_allow_html=True)
                for b in _gate["blockers"][:6]:
                    st.markdown(f"<div style='font-size:12px;color:#ff5252'>{b}</div>", unsafe_allow_html=True)

    # Use dynamic SL/TP if enabled
    if use_dynamic_sltp and signal in ("BUY", "SELL"):
        _dyn = enh.dynamic_sl_tp(signal, current_price, df, _atr_now, min_rr=1.5)
        if _dyn["rr"] >= 1.3:
            _sltp = {"stop_loss": _dyn["sl"], "take_profit_1": _dyn["tp1"],
                     "take_profit_2": _dyn["tp2"],
                     "sl_distance": abs(current_price - _dyn["sl"])}

    # Position sizing
    if signal in ("BUY", "SELL"):
        # Count recent losses for size adjustment
        _recent_perf = brain.get_performance(window=10)
        _recent_losses = _recent_perf.get("losses", 0) if _recent_perf.get("losses") else 0
        _sizing = enh.position_size(
            balance=account_balance, risk_pct=risk_per_trade,
            entry=current_price, sl=_sltp["stop_loss"],
            confidence=confidence / 100, recent_losses=_recent_losses,
        )
        sz1, sz2, sz3, sz4 = st.columns(4)
        sz1.metric("💰 Lot i rekomanduar", f"{_sizing['lot']}")
        sz2.metric("📉 Risk USD", f"${_sizing['risk_usd']:.2f}")
        sz3.metric("📊 Risk %", f"{_sizing['risk_pct']}%")
        sz4.metric("📏 SL distance", f"${_sizing['sl_distance_usd']:.2f}")
        st.caption(f"ℹ {_sizing['reason']}")

    # Trailing stop suggestion (if there's an active position with same signal)
    if use_trailing_stop and signal in ("BUY", "SELL"):
        _trail = enh.trailing_stop_plan(signal, current_price, current_price,
                                        _sltp["stop_loss"], _sltp["take_profit_1"], _atr_now)
        if _trail["action"] != "NONE" and _trail["action"] != "HOLD":
            st.info(f"📈 Trailing Stop: {_trail['reason']}  ·  SL i ri: **${_trail['new_sl']:,.2f}**")

except Exception as _gate_err:
    st.caption(f"⚠ Pro Gate gabim: {_gate_err}")

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 🚨 EARLY EXIT DETECTOR — monitor active positions for invalidation
# ════════════════════════════════════════════════════════════════════════════
_open_positions = pmon.get_open_positions()

with st.expander(
    f"🎯 Pozicione Aktive ({len(_open_positions)}) — Hap / Mbyll / Monitoro",
    expanded=len(_open_positions) > 0,
):
    # ── Quick-open form ──────────────────────────────────────────────────────
    if signal in ("BUY", "SELL"):
        st.markdown(f"#### ➕ Hap pozicion të ri — Sinjali: **{signal}**")
        po1, po2, po3, po4 = st.columns([1, 1, 1, 1])
        new_entry = po1.number_input("Entry", value=float(round(current_price, 2)),
                                     step=0.01, key="po_entry")
        new_sl    = po2.number_input("SL",    value=float(_sltp["stop_loss"]),
                                     step=0.01, key="po_sl")
        new_tp    = po3.number_input("TP1",   value=float(_sltp["take_profit_1"]),
                                     step=0.01, key="po_tp")
        new_lot   = po4.number_input("Lot",   value=0.1, step=0.01,
                                     format="%.2f", key="po_lot")
        if st.button("📂 Hap Pozicion", type="primary", key="open_pos_btn"):
            pid = pmon.open_position(
                signal=signal, entry=new_entry, sl=new_sl, tp1=new_tp,
                tp2=float(_sltp.get("take_profit_2", new_tp)),
                lot=new_lot, atr_at_entry=_atr_now,
                notes=f"AI conf {confidence}% · mode {ml.ACTIVE_MODE}",
            )
            st.success(f"✅ Pozicioni #{pid} u hap. Robot-i tani po e monitoron.")
            st.rerun()

    # ── Monitor active positions ─────────────────────────────────────────────
    if not _open_positions.empty:
        st.markdown("---")
        st.markdown("#### 📡 Monitorimi Live i Pozicioneve")

        for _, pos in _open_positions.iterrows():
            pos_dict = pos.to_dict()
            verdict = pmon.analyze_position(
                pos_dict, df, current_price,
                classifier=mdl, classifier_scaler=scaler,
            )

            # Color per severity
            color_map = {
                "CRITICAL": "#ff1744", "WARNING":  "#ff9100",
                "CAUTION":  "#ffc400", "MINOR":    "#4ea3ff",
                "OK":       "#00c853",
            }
            v_color = color_map.get(verdict["verdict"], "#888")
            bg_color = {
                "CRITICAL": "rgba(255,23,68,0.12)",
                "WARNING":  "rgba(255,145,0,0.10)",
                "CAUTION":  "rgba(255,196,0,0.08)",
                "MINOR":    "rgba(78,163,255,0.06)",
                "OK":       "rgba(0,200,83,0.06)",
            }.get(verdict["verdict"], "rgba(255,255,255,0.03)")

            # Pulse animation for CRITICAL/WARNING
            pulse_css = ""
            if verdict["verdict"] in ("CRITICAL", "WARNING"):
                pulse_css = "animation: pulseAlert 1.2s infinite;"

            sig_emoji = "🟢" if pos["signal"] == "BUY" else "🔴"
            pnl_now = ((current_price - pos["entry"]) if pos["signal"] == "BUY"
                       else (pos["entry"] - current_price)) * pos["lot"] * 100
            pnl_color = "#00c853" if pnl_now >= 0 else "#ff5252"

            st.markdown(
                f"<style>@keyframes pulseAlert {{ 0%{{opacity:1}} 50%{{opacity:.65}} 100%{{opacity:1}} }}</style>"
                f"<div style='background:{bg_color};border-left:4px solid {v_color};"
                f"border-radius:8px;padding:14px;margin-bottom:10px;{pulse_css}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
                f"<div style='font-size:16px;font-weight:700'>"
                f"{sig_emoji} #{pos['id']} · {pos['signal']} @ ${pos['entry']:,.2f} · {pos['lot']} lot"
                f"</div>"
                f"<div style='font-size:18px;font-weight:700;color:{v_color}'>"
                f"{verdict['verdict']}  ·  Score: {verdict['score']}/15"
                f"</div>"
                f"</div>"
                f"<div style='display:flex;gap:18px;margin-top:8px;font-size:12px;color:#c8d0dc'>"
                f"<span>SL: <b style='color:#ff5252'>${pos['sl']:,.2f}</b></span>"
                f"<span>TP1: <b style='color:#00e676'>${pos['tp1']:,.2f}</b></span>"
                f"<span>Çmimi: <b>${verdict['current_price']:,.2f}</b></span>"
                f"<span>P&L: <b style='color:{pnl_color}'>${pnl_now:+,.2f}</b></span>"
                f"<span>Progress TP: <b>{int(verdict['progress']*100)}%</b></span>"
                f"<span>Progress SL: <b style='color:#ff9100'>{int(verdict['sl_progress']*100)}%</b></span>"
                f"</div>"
                f"<div style='margin-top:10px;padding:8px;background:rgba(0,0,0,0.2);border-radius:6px'>"
                f"<div style='font-size:13px;font-weight:600;color:{v_color}'>{verdict['action']}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Reason list
            if verdict["reasons"]:
                with st.container():
                    for r in verdict["reasons"]:
                        st.markdown(f"<div style='font-size:12px;color:#c8d0dc;margin-left:18px'>• {r}</div>",
                                    unsafe_allow_html=True)

            # Action buttons
            bc1, bc2, bc3, _ = st.columns([1, 1, 1, 3])
            if bc1.button("✅ Mbyll WIN", key=f"win_{pos['id']}"):
                pmon.close_position(int(pos["id"]), current_price, "Manual close (WIN)")
                st.rerun()
            if bc2.button("❌ Mbyll LOSS", key=f"loss_{pos['id']}"):
                pmon.close_position(int(pos["id"]), current_price, "Manual close (LOSS)")
                st.rerun()
            if bc3.button("🚪 Dil tani (Early Exit)", key=f"exit_{pos['id']}"):
                pmon.close_position(int(pos["id"]), current_price, f"Early exit · {verdict['verdict']}")
                st.rerun()

            # CRITICAL/WARNING → log alert + browser notification + sound
            if verdict["verdict"] in ("CRITICAL", "WARNING"):
                _last_alert_key = f"last_alert_{pos['id']}"
                _alert_now = f"{verdict['verdict']}_{verdict['score']}"
                if st.session_state.get(_last_alert_key) != _alert_now:
                    pmon.log_alert(
                        int(pos["id"]), verdict["verdict"], pos["signal"],
                        verdict["score"], verdict["reasons"], verdict["action"],
                    )
                    st.session_state[_last_alert_key] = _alert_now

                    # Browser notification + alert sound
                    from streamlit.components.v1 import html as _html
                    _msg = verdict["action"].replace('"', "'")
                    _html(f"""
<script>
(function() {{
  if ("Notification" in window) {{
    if (Notification.permission === "granted") {{
      new Notification("🚨 XAUUSD · Pozicion #{pos['id']} · {verdict['verdict']}", {{
        body: "{_msg}\\nScore: {verdict['score']}/15 · Çmimi: ${verdict['current_price']:,.2f}",
        icon: "https://cdn-icons-png.flaticon.com/512/3304/3304567.png",
      }});
    }} else if (Notification.permission !== "denied") {{
      Notification.requestPermission();
    }}
  }}
  // Urgent siren
  try {{
    var c = new (window.AudioContext||window.webkitAudioContext)();
    function b(f,d,v) {{
      var o=c.createOscillator(),g=c.createGain();
      o.connect(g);g.connect(c.destination);
      o.frequency.value=f;g.gain.value=v;
      o.start(c.currentTime);o.stop(c.currentTime+d);
    }}
    b(1200,.15,.4); setTimeout(()=>b(900,.15,.4),200);
    setTimeout(()=>b(1200,.15,.4),400); setTimeout(()=>b(900,.30,.4),600);
  }} catch(e) {{}}
}})();
</script>
""", height=0)
    else:
        st.caption("Asnjë pozicion aktiv. Hap një kur AI të jep sinjal të fortë.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Individual model votes ─────────────────────────────────────────────────────
if individual:
    # Compute consensus strength: how many models agree with final signal
    _votes_match = sum(1 for s in individual.values() if s == signal)
    _consensus_pct = round(_votes_match / len(individual) * 100)
    _cons_color = "#00e676" if _consensus_pct >= 75 else ("#f5c842" if _consensus_pct >= 50 else "#ff5252")
    _cons_label = "✅ I FORTË" if _consensus_pct >= 75 else ("⚠ I DOBËT" if _consensus_pct < 50 else "➡ MESATAR")

    cols = st.columns(len(individual) + 1)
    cols[0].markdown(
        f"<div style='padding-top:8px;color:#aaa;font-size:12px'>Votat e modeleve:</div>"
        f"<div style='font-size:10px;color:#8a93a6;margin-top:6px'>Konsensusi:</div>"
        f"<div style='font-size:14px;font-weight:700;color:{_cons_color}'>{_consensus_pct}% · {_cons_label}</div>",
        unsafe_allow_html=True,
    )
    for i, (name, sig) in enumerate(individual.items()):
        sig_color = color_map.get(sig, "#aaa")
        sig_arrow = arrows.get(sig, sig)
        # Highlight border green if matches final signal, red if disagrees
        _matches = sig == signal
        _border = "#00e67644" if _matches else "#ff525244"
        cols[i+1].markdown(
            f"<div style='text-align:center;background:#1a1a1a;border-radius:8px;padding:6px 10px;"
            f"border:1px solid {_border}'><div style='font-size:10px;color:#aaa'>{name}</div>"
            f"<div style='color:{sig_color};font-weight:bold'>{sig_arrow} {sig}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    if _consensus_pct < 50:
        st.info(
            f"💡 Modelet janë të ndarë ({_consensus_pct}% pajtim). Sinjali është më pak i besueshëm. "
            f"Pres një konsensus më të fortë (≥75%) para se të hysh.",
            icon="💡",
        )

# ── PAGE ROUTER ────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    df_feat = ml.add_features(df.copy()).dropna()

    # ── 🤖 Robot Trader Performance Panel ──────────────────────────────────────
    if robot_on:
        _perf = brain.get_performance(window=50)
        if _perf["lifetime_evaluated"] > 0 or _perf["pending"] > 0:
            with st.expander(
                f"🤖 Robot Trader — {_perf['wins']}W / {_perf['losses']}L  ·  "
                f"Win Rate: {_perf['win_rate']}%  ·  PF: {_perf['profit_factor']}  ·  "
                f"Pending: {_perf['pending']}",
                expanded=False,
            ):
                rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                rc1.metric("Win Rate (50 të fundit)", f"{_perf['win_rate']}%")
                rc2.metric("Profit Factor", f"{_perf['profit_factor']}")
                rc3.metric("Streak", f"{_perf['current_streak']:+d}")
                rc4.metric("Avg Bars to Exit", f"{_perf['avg_bars']}")
                rc5.metric("Total i Vlerësuar", _perf["lifetime_evaluated"])

                # Show last tune action
                _adj = st.session_state.get("last_tune_adj")
                if _adj:
                    _icon = {"tighten": "🔒", "loosen": "🔓"}.get(_adj["action"], "⚙")
                    st.info(
                        f"{_icon} **Auto-tune**: {_adj['action'].upper()} — "
                        f"ML threshold {_adj['old_ml']:.2f} → {_adj['new_ml']:.2f}, "
                        f"Confluence {_adj['old_conf']} → {_adj['new_conf']} "
                        f"(saktësia aktuale: {_adj['precision']*100:.1f}% mbi {_adj['samples']} mostra)"
                    )

                st.markdown("##### 📋 Sinjalet e Fundit")
                _recent = brain.get_recent_predictions(limit=15)
                if not _recent.empty:
                    _display = _recent[["ts", "signal", "confidence", "entry", "sl",
                                        "tp1", "outcome", "exit_price", "pnl_pips", "bars_to_exit"]].copy()
                    _display["ts"] = pd.to_datetime(_display["ts"]).dt.strftime("%m-%d %H:%M")
                    st.dataframe(
                        _display.style.map(
                            lambda v: ("color:#00c853" if v == "WIN" else
                                       "color:#d50000" if v in ("LOSS", "TIMEOUT") else
                                       "color:#ffd600" if v == "PENDING" else ""),
                            subset=["outcome"],
                        ).map(
                            lambda v: ("color:#00c853" if v == "BUY" else
                                       "color:#d50000" if v == "SELL" else ""),
                            subset=["signal"],
                        ),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("Ende pa sinjale — pres robotin të gjenerojë i pari sinjal.")

    # ── Chart mode toggle ──────────────────────────────────────────────────────
    chart_mode = st.radio(
        "Chart mode",
        ["⚡ Live (TradingView)", "🧠 AI Analysis (Plotly)"],
        horizontal=True,
        label_visibility="collapsed",
        key="chart_mode",
    )

    if chart_mode.startswith("⚡"):
        # ── TradingView Advanced Chart (real-time, WebSocket) ─────────────────
        tv_interval = {
            "15m": "15", "1h": "60", "4h": "240", "1d": "D",
        }.get(interval, "60")

        tv_widget = f"""
<div style="border:1px solid var(--border);border-radius:14px;overflow:hidden;
            background:#0a0e17;margin-top:10px">
  <div class="tradingview-widget-container" style="height:780px;width:100%">
    <div id="tv_chart_xauusd" style="height:100%;width:100%"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "OANDA:XAUUSD",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#0a0e17",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tv_chart_xauusd",
        "studies": [
          "MASimple@tv-basicstudies",
          "RSI@tv-basicstudies",
          "MACD@tv-basicstudies",
          "BB@tv-basicstudies"
        ],
        "hide_side_toolbar": false,
        "withdateranges": true,
        "hide_volume": false,
        "save_image": true,
        "details": true,
        "hotlist": true,
        "calendar": true,
        "show_popup_button": true,
        "popup_width": "1400",
        "popup_height": "850"
      }});
    </script>
  </div>
</div>
"""
        from streamlit.components.v1 import html as _html
        _html(tv_widget, height=800, scrolling=False)

        st.caption("⚡ Live tick-by-tick chart nga TradingView · OANDA feed · Përditësohet në kohë reale automatikisht")

        # AI overlay info
        if predicted_price:
            pred_dir_emoji = "🟢 ▲" if predicted_price > current_price else "🔴 ▼"
            st.markdown(
                f"<div style='background:var(--bg-card);border:1px solid var(--border);"
                f"border-radius:12px;padding:14px 18px;margin-top:14px;display:flex;"
                f"align-items:center;justify-content:space-between'>"
                f"<div><span style='color:var(--text-dim);font-size:12px'>🤖 AI Target Çmimi</span>"
                f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:20px;color:var(--gold);font-weight:700'>"
                f"${predicted_price:,.2f}</div></div>"
                f"<div style='font-size:24px'>{pred_dir_emoji}</div>"
                f"<div><span style='color:var(--text-dim);font-size:12px'>Sinjali</span>"
                f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:18px;color:{color_map[signal]};font-weight:700'>"
                f"{arrows[signal]} {signal} · {confidence}%</div></div></div>",
                unsafe_allow_html=True,
            )

        # Skip the rest of Plotly chart in live mode
        st.stop()

    # ── Chart toolbar (MT-style) ───────────────────────────────────────────────
    _o = df['Open'].iloc[-1]
    _h = df['High'].iloc[-1]
    _l = df['Low'].iloc[-1]
    _c = df['Close'].iloc[-1]
    toolbar_html = (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'background:linear-gradient(180deg,rgba(18,22,33,0.85),rgba(10,13,20,0.85));'
        'border:1px solid var(--border);border-radius:12px 12px 0 0;'
        'padding:10px 18px;margin-top:8px;backdrop-filter:blur(12px);border-bottom:none">'
        '<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:16px;font-weight:700;color:#fff">'
        f'XAU/USD <span style="color:var(--gold);font-weight:500;font-size:13px;margin-left:6px">{interval_label}</span></div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:var(--text-dim)">'
        f'O <span style="color:#fff">{_o:.2f}</span>'
        f' &nbsp;H <span style="color:var(--green)">{_h:.2f}</span>'
        f' &nbsp;L <span style="color:var(--red)">{_l:.2f}</span>'
        f' &nbsp;C <span style="color:#fff">{_c:.2f}</span></div></div>'
        '<div style="display:flex;gap:6px">'
        f'<span class="chip chip-gold">Bid {_c:.2f}</span>'
        '<span class="chip">Spread 0.30</span>'
        '<span class="chip chip-green">● Live</span></div></div>'
    )
    st.markdown(toolbar_html, unsafe_allow_html=True)

    # ── MetaTrader-style chart ─────────────────────────────────────────────────
    MT_BG       = "#0a0e17"         # near-black (MT5 dark)
    MT_GRID     = "rgba(120,130,150,0.12)"
    MT_BULL     = "#26a69a"         # MT5 teal
    MT_BEAR     = "#ef5350"         # MT5 coral
    MT_BULL_LN  = "#1de9b6"
    MT_BEAR_LN  = "#ff5252"

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.14, 0.14, 0.14],
        vertical_spacing=0.02,
    )

    # Main candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="XAUUSD",
        increasing=dict(line=dict(color=MT_BULL_LN, width=1), fillcolor=MT_BULL),
        decreasing=dict(line=dict(color=MT_BEAR_LN, width=1), fillcolor=MT_BEAR),
        whiskerwidth=0.4,
    ), row=1, col=1)

    # EMAs
    for ema, color, width in [
        ("ema_9",  "#ffd54f", 1.2),
        ("ema_21", "#42a5f5", 1.2),
        ("ema_50", "#ab47bc", 1.4),
        ("ema_200","#ff7043", 1.8),
    ]:
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat[ema],
            name=ema.replace("ema_", "EMA "), line=dict(color=color, width=width),
            opacity=0.95,
        ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_upper"],
        name="BB Upper", line=dict(color="rgba(180,180,200,0.4)", width=1, dash="dot"),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_lower"],
        name="BB Lower", line=dict(color="rgba(180,180,200,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(120,130,150,0.04)", showlegend=False,
    ), row=1, col=1)

    # Last close horizontal line (MT-style price line)
    last_close = df["Close"].iloc[-1]
    fig.add_hline(y=last_close, line=dict(color="#ffd54f", width=1, dash="dash"),
                  row=1, col=1)

    # AI predicted price
    if predicted_price:
        pred_color = "#1de9b6" if predicted_price > last_close else "#ff5252"
        fig.add_hline(y=predicted_price, line=dict(color=pred_color, width=1.5, dash="dot"),
                      row=1, col=1,
                      annotation_text=f"  AI Target ${predicted_price:,.2f}",
                      annotation_position="top right",
                      annotation_font=dict(color=pred_color, size=11, family="JetBrains Mono"))

    # Volume bars (MT5-style under price)
    vol_colors = [MT_BULL if c >= o else MT_BEAR
                  for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker=dict(color=vol_colors, line=dict(width=0)),
        opacity=0.55, name="Volume", showlegend=False,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["rsi"],
        name="RSI", line=dict(color="#ffd54f", width=1.4),
        showlegend=False,
    ), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.08)",
                  line_width=0, row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ef5350", width=0.8, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", width=0.8, dash="dot"), row=3, col=1)
    fig.add_hline(y=50, line=dict(color="rgba(180,180,200,0.3)", width=0.6), row=3, col=1)

    # MACD
    fig.add_trace(go.Bar(
        x=df_feat.index, y=df_feat["macd_hist"],
        marker_color=[MT_BULL if v >= 0 else MT_BEAR for v in df_feat["macd_hist"]],
        opacity=0.7, name="Hist", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["macd"],
        name="MACD", line=dict(color="#42a5f5", width=1.3), showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["macd_signal"],
        name="Signal", line=dict(color="#ff7043", width=1.3), showlegend=False,
    ), row=4, col=1)

    # Layout — MT5 style
    fig.update_layout(
        paper_bgcolor=MT_BG,
        plot_bgcolor=MT_BG,
        height=820,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#c8d0dc", family="Inter"),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=70, t=24, b=24),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(15,20,30,0.95)",
            bordercolor="#ffd54f",
            font=dict(family="JetBrains Mono", color="#fff", size=12),
        ),
        dragmode="pan",
        font=dict(family="Inter", color="#c8d0dc", size=11),
    )

    # Axes — MT-style grid + price scale on right
    grid_kw = dict(
        gridcolor=MT_GRID, gridwidth=1,
        zerolinecolor=MT_GRID,
        showspikes=True, spikecolor="rgba(255,213,79,0.5)", spikethickness=1,
        spikedash="dot", spikemode="across",
        tickfont=dict(family="JetBrains Mono", size=10, color="#8a93a6"),
    )
    for r in (1, 2, 3, 4):
        fig.update_xaxes(showgrid=True, **grid_kw, row=r, col=1)
        fig.update_yaxes(showgrid=True, side="right", **grid_kw, row=r, col=1)

    # Subplot title chips inside (top-left of each panel)
    panel_labels = [
        (1, "PRICE"),
        (2, "VOLUME"),
        (3, "RSI 14"),
        (4, "MACD 12,26,9"),
    ]
    for row, txt in panel_labels:
        fig.add_annotation(
            xref=f"x{row} domain" if row > 1 else "x domain",
            yref=f"y{row} domain" if row > 1 else "y domain",
            x=0.005, y=0.97, xanchor="left", yanchor="top",
            text=f"<b>{txt}</b>", showarrow=False,
            font=dict(family="Inter", size=10, color="#ffd54f"),
            bgcolor="rgba(0,0,0,0.0)",
        )

    # X-axis: range selector buttons (MT-like timeframe)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1D", step="day",   stepmode="backward"),
                dict(count=5,  label="5D", step="day",   stepmode="backward"),
                dict(count=1,  label="1M", step="month", stepmode="backward"),
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(18,22,33,0.6)",
            activecolor="rgba(245,200,66,0.25)",
            bordercolor="rgba(245,200,66,0.3)",
            borderwidth=1,
            font=dict(color="#c8d0dc", size=10, family="Inter"),
            x=0, y=1.06,
        ),
        row=1, col=1,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom": True,
        "displaylogo": False,
    })

    # ── Indicator panel (right) + summary ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    latest = df_feat.iloc[-1]

    def _badge(text, kind):
        cls = {"bull": "chip-green", "bear": "chip-red", "neut": "chip"}[kind]
        return f"<span class='chip {cls}'>{text}</span>"

    def rsi_kind(v):
        return "bear" if v > 70 else "bull" if v < 30 else "neut"
    def ema_kind(a, b):
        return "bull" if a > b else "bear"

    rows_html = []
    indicators = [
        ("RSI (14)",        f"{latest['rsi']:.1f}",
         _badge("Overbought", "bear") if latest["rsi"] > 70 else
         _badge("Oversold", "bull")   if latest["rsi"] < 30 else
         _badge("Neutral", "neut")),
        ("RSI (6)",         f"{latest['rsi_6']:.1f}",
         _badge("Overbought", "bear") if latest["rsi_6"] > 70 else
         _badge("Oversold", "bull")   if latest["rsi_6"] < 30 else
         _badge("Neutral", "neut")),
        ("EMA 9 / 21",      f"{latest['ema_9']:.2f} / {latest['ema_21']:.2f}",
         _badge("Bullish", "bull") if latest["ema_9"] > latest["ema_21"] else _badge("Bearish", "bear")),
        ("EMA 21 / 50",     f"{latest['ema_21']:.2f} / {latest['ema_50']:.2f}",
         _badge("Bullish", "bull") if latest["ema_21"] > latest["ema_50"] else _badge("Bearish", "bear")),
        ("MACD vs Signal",  f"{latest['macd']:.2f} / {latest['macd_signal']:.2f}",
         _badge("Bullish", "bull") if latest["macd"] > latest["macd_signal"] else _badge("Bearish", "bear")),
        ("Price vs EMA200", f"${latest['ema_200']:.2f}",
         _badge("Above", "bull") if latest["Close"] > latest["ema_200"] else _badge("Below", "bear")),
        ("Stoch K",         f"{latest['stoch_k']:.1f}",
         _badge("Overbought", "bear") if latest["stoch_k"] > 80 else
         _badge("Oversold", "bull")   if latest["stoch_k"] < 20 else
         _badge("Neutral", "neut")),
        ("BB %",            f"{latest['bb_pct']*100:.1f}%",
         _badge("Upper Band", "bear") if latest["bb_pct"] > 0.8 else
         _badge("Lower Band", "bull") if latest["bb_pct"] < 0.2 else
         _badge("Middle", "neut")),
    ]

    cards = ""
    for name, val, badge in indicators:
        cards += (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:11px 16px;border-bottom:1px solid var(--border)">'
            f'<div style="font-size:12px;color:var(--text-dim);font-weight:600">{name}</div>'
            f'<div style="display:flex;align-items:center;gap:14px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#fff;font-weight:500">{val}</div>'
            f'{badge}</div></div>'
        )

    panel_html = (
        '<div style="background:var(--bg-card);backdrop-filter:blur(12px);'
        'border:1px solid var(--border);border-radius:14px;overflow:hidden">'
        '<div style="padding:12px 18px;border-bottom:1px solid var(--border);'
        'background:linear-gradient(90deg,rgba(245,200,66,0.06),transparent);'
        'display:flex;align-items:center;justify-content:space-between">'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;color:#fff;font-size:14px">'
        '📋 Technical Indicators</div>'
        f'<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase">'
        f'Live · {interval_label}</div></div>'
        f'{cards}</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADVANCED CHART (S/R + Fibonacci + Patterns)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Chart Avancuar":
    st.markdown("### 📐 Support/Resistance · Fibonacci · Patterns")

    col_a, col_b = st.columns([3, 1])

    supports, resistances = tech.find_support_resistance(df)
    fib_levels, uptrend   = tech.fibonacci_levels(df)
    patterns              = tech.detect_patterns(df)

    with col_a:
        fig2 = go.Figure()

        fig2.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="XAUUSD",
            increasing_line_color="#00c853",
            decreasing_line_color="#d50000",
        ))

        # Support levels
        for i, s in enumerate(supports[:4]):
            fig2.add_hline(y=s, line_dash="dash", line_color="#00c853",
                           line_width=1,
                           annotation_text=f"S{i+1} ${s:,.0f}",
                           annotation_font_color="#00c853",
                           annotation_position="right")

        # Resistance levels
        for i, r in enumerate(resistances[:4]):
            fig2.add_hline(y=r, line_dash="dash", line_color="#d50000",
                           line_width=1,
                           annotation_text=f"R{i+1} ${r:,.0f}",
                           annotation_font_color="#d50000",
                           annotation_position="right")

        # Fibonacci levels
        fib_colors_map = {
            0.0: "#f0c040", 0.236: "#40c0f0", 0.382: "#00c853",
            0.5: "#ffffff", 0.618: "#f040f0", 0.786: "#ff6b00", 1.0: "#f0c040",
        }
        for fib, price in fib_levels.items():
            fig2.add_hline(
                y=price, line_dash="dot",
                line_color=fib_colors_map.get(fib, "#888"),
                line_width=1,
                annotation_text=f"Fib {fib:.3f} — ${price:,.2f}",
                annotation_font_color=fib_colors_map.get(fib, "#888"),
                annotation_position="left",
            )

        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            height=650, xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=120, t=20, b=0),
            title=f"{'Uptrend 📈' if uptrend else 'Downtrend 📉'}",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### Support Levels 🟢")
        for i, s in enumerate(supports[:5]):
            st.markdown(f"**S{i+1}** — `${s:,.2f}`")

        st.markdown("#### Resistance Levels 🔴")
        for i, r in enumerate(resistances[:5]):
            st.markdown(f"**R{i+1}** — `${r:,.2f}`")

        st.markdown("#### Fibonacci")
        for fib, price in fib_levels.items():
            st.markdown(f"`{fib:.3f}` — ${price:,.2f}")

        st.markdown("#### Chart Patterns")
        for p in patterns:
            st.markdown(f"- {p}")

    # Multi-timeframe signals
    st.markdown("---")
    st.markdown("### 🕐 Sinjale Multi-Timeframe")
    with st.spinner("Duke llogaritur..."):
        mtf = fetch_multi_tf()

    mtf_cols = st.columns(len(mtf))
    for i, (tf_label, tf_data) in enumerate(mtf.items()):
        sig_tf = tf_data["signal"]
        conf_tf = tf_data["confidence"]
        with mtf_cols[i]:
            tf_color = color_map.get(sig_tf, "#555")
            tf_arrow = arrows.get(sig_tf, sig_tf)
            st.markdown(
                f"<div style='text-align:center;background:#1a1a1a;border-radius:10px;"
                f"padding:16px;border:2px solid {tf_color}'>"
                f"<div style='font-size:12px;color:#aaa'>{tf_label}</div>"
                f"<div style='font-size:28px;font-weight:bold;color:{tf_color}'>"
                f"{tf_arrow} {sig_tf}</div>"
                f"<div style='color:#aaa;font-size:11px'>{conf_tf}%</div></div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 🏆 PERFORMANCE ANALYTICS — full breakdown of Robot + Paper-Trading record
# ════════════════════════════════════════════════════════════════════════════
elif page == "Performance":
    st.markdown("### 🏆 Performance Analytics")
    st.caption("Statistika reale të Robot-it AI + Paper Trading account")

    _stats = pt.get_stats()
    _acc = pt.get_account()

    # ── Top KPI row ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💼 Balanca", f"${_acc['balance']:,.2f}",
              f"{_stats['return_pct']:+.2f}%")
    k2.metric("📊 Total Trades", _stats["trades"])
    k3.metric("✅ Win Rate", f"{_stats['win_rate']}%")
    k4.metric("📈 Profit Factor", f"{_stats['profit_factor']}")
    k5.metric("📉 Max Drawdown", f"{_stats['max_dd_pct']}%")
    k6.metric("⚡ Sharpe Ratio", f"{_stats['sharpe']}")

    st.markdown("---")

    if _stats["trades"] == 0:
        st.info("🧪 Ende pa tregti të mbyllura. Robot-i fillon të auto-tregtojë sapo AI të japë sinjale.")
    else:
        # ── Equity curve ──────────────────────────────────────────────────
        if _stats.get("equity_curve"):
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                y=_stats["equity_curve"], mode="lines+markers",
                fill="tozeroy", fillcolor="rgba(0,200,83,0.1)",
                line=dict(color="#00c853", width=2), name="Equity",
            ))
            fig_eq.add_hline(y=_acc["starting"], line_dash="dot", line_color="#888",
                             annotation_text="Starting")
            fig_eq.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                height=320, title="📈 Equity Curve (Paper Trading)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # ── Stats by session ──────────────────────────────────────────────
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### 🕐 Performanca sipas Sesionit")
            _ses_df = pt.stats_by_session()
            if not _ses_df.empty:
                st.dataframe(_ses_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Pa data ende.")
        with b2:
            st.markdown("#### 📅 Performanca sipas Ditës së Javës")
            _dow_df = pt.stats_by_day_of_week()
            if not _dow_df.empty:
                st.dataframe(_dow_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Pa data ende.")

        # ── Detailed stats ────────────────────────────────────────────────
        st.markdown("#### 📊 Statistika të Detajuara")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Wins / Losses", f"{_stats['wins']} / {_stats['losses']}")
        d2.metric("Avg Win", f"${_stats['avg_win']:+.2f}")
        d3.metric("Avg Loss", f"${_stats['avg_loss']:+.2f}")
        d4.metric("Total P&L", f"${_stats['total_pnl']:+,.2f}")
        d1.metric("Best Trade", f"${_stats['best_trade']:+,.2f}")
        d2.metric("Worst Trade", f"${_stats['worst_trade']:+,.2f}")

    # ── Robot Brain Stats ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Robot Brain — Real Predictions")
    _perf = brain.get_performance(window=100)
    rb1, rb2, rb3, rb4, rb5 = st.columns(5)
    rb1.metric("Sinjale (100 fundit)", _perf["total_evaluated"])
    rb2.metric("Win Rate", f"{_perf['win_rate']}%")
    rb3.metric("Profit Factor", f"{_perf['profit_factor']}")
    rb4.metric("Pending", _perf["pending"])
    rb5.metric("Avg Bars to Exit", f"{_perf['avg_bars']}")

    # ── Export buttons ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📤 Eksporto Të Dhënat")
    e1, e2, e3 = st.columns(3)
    try:
        _xlsx = rpt.export_all_to_excel()
        e1.download_button(
            "📊 Shkarko Excel (të gjitha)", _xlsx,
            file_name=f"xauusd_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        e1.caption(f"⚠ Excel error: {e}")

    _closed = pt.get_closed_trades(limit=10000)
    if not _closed.empty:
        e2.download_button(
            "📄 Shkarko CSV (Paper Trades)", rpt.export_csv(_closed),
            file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )
    _rec = brain.get_recent_predictions(limit=10000)
    if not _rec.empty:
        e3.download_button(
            "🤖 Shkarko CSV (AI Predictions)", rpt.export_csv(_rec),
            file_name=f"ai_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )

    # ── Reset account ────────────────────────────────────────────────────
    with st.expander("⚠ Reset Paper Account"):
        new_bal = st.number_input("Balanca e re fillestare ($)", value=10000.0, step=1000.0)
        if st.button("🔄 Reset (FSHIN gjithçka)"):
            pt.reset_account(new_bal)
            st.success("✅ Llogaria u resetua.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# 🧪 PAPER TRADING — virtual account that auto-trades every AI signal
# ════════════════════════════════════════════════════════════════════════════
elif page == "Paper Trading":
    st.markdown("### 🧪 Paper Trading — Llogari Virtuale")
    st.caption("Robot-i auto-tregton çdo sinjal AI. Pa rrezik real, statistika realiste.")

    _acc = pt.get_account()
    _stats = pt.get_stats()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("💼 Balanca", f"${_acc['balance']:,.2f}", f"{_stats['return_pct']:+.2f}%")
    a2.metric("📊 Trade Total", _stats["trades"])
    a3.metric("✅ Win Rate", f"{_stats['win_rate']}%")
    a4.metric("💰 P&L", f"${_stats['total_pnl']:+,.2f}")

    st.markdown("---")

    # Open trades
    _open = pt.get_open_trades()
    st.markdown(f"#### 🔓 Pozicione Aktive ({len(_open)})")
    if not _open.empty:
        _display = _open[["id", "opened_at", "signal", "entry", "sl", "tp1", "lot", "confidence", "mode", "session"]].copy()
        _display["opened_at"] = pd.to_datetime(_display["opened_at"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            _display.style.map(
                lambda v: ("color:#00c853" if v == "BUY" else "color:#d50000" if v == "SELL" else ""),
                subset=["signal"],
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("Pa pozicione aktive aktualisht.")

    # Closed trades
    st.markdown(f"#### 📋 Tregti të Mbyllura ({_stats['trades']})")
    _closed = pt.get_closed_trades(limit=200)
    if not _closed.empty:
        _d = _closed[["id", "opened_at", "signal", "entry", "exit_price",
                      "pnl_usd", "pnl_pips", "lot", "exit_reason", "session"]].copy()
        _d["opened_at"] = pd.to_datetime(_d["opened_at"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            _d.style.map(
                lambda v: ("color:#00c853" if v == "BUY" else "color:#d50000" if v == "SELL" else ""),
                subset=["signal"],
            ).map(
                lambda v: ("color:#00c853" if isinstance(v,(int,float)) and v>0 else "color:#d50000" if isinstance(v,(int,float)) and v<0 else ""),
                subset=["pnl_usd","pnl_pips"],
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("Ende pa tregti të mbyllura.")


elif page == "Backtesting":
    st.markdown("### 🔁 Backtesting — Simulim Historik")

    b_col1, b_col2, b_col3 = st.columns(3)
    init_bal  = b_col1.number_input("Balanca fillestare ($)", value=10000, step=1000)
    lot       = b_col2.number_input("Lot size", value=0.1, step=0.01, format="%.2f")
    spread    = b_col3.number_input("Spread (pips)", value=0.3, step=0.1, format="%.1f")

    b_col4, b_col5 = st.columns([1, 2])
    from datetime import date as _date, timedelta as _td
    use_date_filter = b_col4.checkbox("📅 Filtro nga data", value=False,
        help="Shfaq vetëm tregtitë e bëra nga kjo datë e tutje (fshij ato 'demo' të vjetra)")
    start_date_filter = b_col5.date_input(
        "Fillo nga data",
        value=_date.today(),
        max_value=_date.today(),
        disabled=not use_date_filter,
        help="Tregtitë para kësaj date do të fshihen nga rezultatet.",
    ) if True else None

    if st.button("▶ Fillo Backtesting"):
        with st.spinner("Duke simuluar tregtitë..."):
            summary, trades_df = bt.run_backtest(df, init_bal, lot, spread)

        if summary is None:
            st.error("Të dhëna të pamjaftueshme për backtesting.")
        else:
            # ── Apply date filter if enabled ──
            if use_date_filter and trades_df is not None and not trades_df.empty:
                trades_df["open_time"] = pd.to_datetime(trades_df["open_time"])
                trades_df["close_time"] = pd.to_datetime(trades_df["close_time"])
                cutoff = pd.Timestamp(start_date_filter)
                if trades_df["open_time"].dt.tz is not None:
                    cutoff = cutoff.tz_localize(trades_df["open_time"].dt.tz)
                before_n = len(trades_df)
                trades_df = trades_df[trades_df["open_time"] >= cutoff].reset_index(drop=True)
                after_n = len(trades_df)
                st.info(f"📅 Filtri: u fshinë {before_n - after_n} tregti 'demo' para {start_date_filter}. Po shfaqen {after_n} tregti reale.")

                if trades_df.empty:
                    st.warning(f"⚠️ Nuk ka asnjë tregti nga data {start_date_filter} e tutje. Provo të zgjedhësh një datë më të hershme.")
                    st.stop()

                # Recompute summary on filtered subset
                wins = (trades_df["result"] == "WIN").sum()
                losses = (trades_df["result"] == "LOSS").sum()
                total = len(trades_df)
                win_rate = round(wins / total * 100, 1) if total else 0
                total_pnl = round(trades_df["pnl_usd"].sum(), 2)
                avg_win = round(trades_df[trades_df["result"]=="WIN"]["pnl_usd"].mean(), 2) if wins else 0
                avg_loss = round(trades_df[trades_df["result"]=="LOSS"]["pnl_usd"].mean(), 2) if losses else 0
                gross_profit = trades_df[trades_df["pnl_usd"]>0]["pnl_usd"].sum()
                gross_loss = abs(trades_df[trades_df["pnl_usd"]<0]["pnl_usd"].sum())
                pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
                # Recompute equity curve from initial balance
                running = init_bal
                bals = []
                for p in trades_df["pnl_usd"]:
                    running += p
                    bals.append(round(running, 2))
                trades_df["balance"] = bals
                # Drawdown
                peak = init_bal; max_dd = 0
                for b in bals:
                    if b > peak: peak = b
                    dd = (peak - b) / peak * 100
                    if dd > max_dd: max_dd = dd
                returns = trades_df["pnl_usd"]
                sharpe = round(returns.mean() / (returns.std() + 1e-9), 2) if len(returns) > 1 else 0
                summary = {
                    "total_trades": total, "wins": int(wins), "losses": int(losses),
                    "win_rate": win_rate, "total_pnl": total_pnl,
                    "avg_win": avg_win, "avg_loss": avg_loss,
                    "max_drawdown": round(max_dd, 2), "profit_factor": pf,
                    "sharpe": sharpe, "final_balance": round(running, 2),
                }

            # KPIs
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Win Rate",      f"{summary['win_rate']}%")
            k2.metric("Total P&L",     f"${summary['total_pnl']:+,.2f}")
            k3.metric("Max Drawdown",  f"{summary['max_drawdown']}%")
            k4.metric("Profit Factor", f"{summary['profit_factor']}")
            k5.metric("Total Trades",  summary["total_trades"])
            k6.metric("Balanca Fund.", f"${summary['final_balance']:,.2f}")

            st.markdown(f"""
            | Wins | Losses | Avg Win | Avg Loss | Sharpe |
            |------|--------|---------|----------|--------|
            | {summary['wins']} | {summary['losses']} | ${summary['avg_win']:+.2f} | ${summary['avg_loss']:+.2f} | {summary['sharpe']} |
            """)

            # Equity curve
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=trades_df["close_time"], y=trades_df["balance"],
                fill="tozeroy", fillcolor="rgba(0,200,83,0.1)",
                line=dict(color="#00c853", width=2),
                name="Equity",
            ))
            fig_eq.add_hline(y=init_bal, line_dash="dot", line_color="#555",
                             annotation_text="Fillestare")
            fig_eq.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                height=350, title="Equity Curve",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trades table
            st.markdown("#### Tregtitë")
            display_cols = ["open_time", "close_time", "type", "entry", "exit",
                            "pnl_pips", "pnl_usd", "result"]
            styled = trades_df[display_cols].copy()
            styled["open_time"]  = styled["open_time"].astype(str).str[:16]
            styled["close_time"] = styled["close_time"].astype(str).str[:16]
            st.dataframe(
                styled.style.map(
                    lambda v: "color: #00c853" if v == "WIN" else ("color: #d50000" if v == "LOSS" else ""),
                    subset=["result"],
                ),
                use_container_width=True, hide_index=True,
            )

            # ── Monte Carlo simulation ─────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🎲 Monte Carlo Simulation")
            st.caption("Përsërit historikun e tregtive 1000 herë me renditje të rastësishme — sheh worst/best/median outcome.")
            mc_runs = st.slider("Runs", 100, 5000, 1000, 100, key="mc_runs")
            if st.button("▶ Ekzekuto Monte Carlo"):
                with st.spinner(f"Duke ekzekutuar {mc_runs} simulime..."):
                    mc = bt.monte_carlo(trades_df, init_bal, mc_runs)
                if mc:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Worst Case (5%)", f"${mc['final_p5']:,.2f}")
                    mc2.metric("Median (50%)",    f"${mc['final_p50']:,.2f}")
                    mc3.metric("Best Case (95%)", f"${mc['final_p95']:,.2f}")
                    mc4.metric("Prob. Profit",    f"{mc['prob_profit']}%")
                    md1, md2, md3 = st.columns(3)
                    md1.metric("DD Best (5%)",   f"{mc['dd_p5']}%")
                    md2.metric("DD Median",      f"{mc['dd_p50']}%")
                    md3.metric("DD Worst (95%)", f"{mc['dd_p95']}%")
                    fig_mc = go.Figure(go.Histogram(
                        x=mc["finals"], nbinsx=50,
                        marker=dict(color="#4ea3ff"),
                    ))
                    fig_mc.add_vline(x=init_bal, line_dash="dash", line_color="#ffd600",
                                     annotation_text=f"Start ${init_bal:,.0f}")
                    fig_mc.update_layout(
                        template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                        height=350, title="Shpërndarja e Balancës Përfundimtare",
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEWS & SENTIMENT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Lajme & Sentiment":
    st.markdown("### 📰 Lajme Gold & Sentiment")

    with st.spinner("Duke marrë lajmet..."):
        articles = fetch_news_cached()

    if not articles:
        st.info("Nuk u gjetën lajme tani. Provo sërish.")
    else:
        overall_sent, overall_score = nws.overall_sentiment(articles)
        n1, n2, n3 = st.columns(3)
        n1.markdown(f"""<div class='metric-card'>
            <div style='font-size:11px;color:#aaa'>Sentiment i Përgjithshëm</div>
            <div style='font-size:24px;font-weight:bold;
                color:{"#00c853" if "Bull" in overall_sent else "#d50000" if "Bear" in overall_sent else "#ffd600"}'>{overall_sent}</div>
        </div>""", unsafe_allow_html=True)
        n2.markdown(f"""<div class='metric-card'>
            <div style='font-size:11px;color:#aaa'>Score Sentiment</div>
            <div style='font-size:24px;font-weight:bold;color:#f0c040'>{overall_score:+.2f}</div>
        </div>""", unsafe_allow_html=True)
        n3.markdown(f"""<div class='metric-card'>
            <div style='font-size:11px;color:#aaa'>Lajme të Gjetur</div>
            <div style='font-size:24px;font-weight:bold;color:#f0c040'>{len(articles)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sentiment filter
        sent_filter = st.selectbox("Filtro", ["Të gjitha", "Bullish 🟢", "Bearish 🔴", "Neutral ⚪"])
        filtered = [a for a in articles if sent_filter == "Të gjitha" or a["sentiment"] == sent_filter]

        for a in filtered[:15]:
            border_color = "#00c853" if "Bull" in a["sentiment"] else "#d50000" if "Bear" in a["sentiment"] else "#555"
            st.markdown(f"""
            <div class='news-card' style='border-left-color:{border_color}'>
                <div style='font-size:13px;font-weight:bold;color:#e0e0e0'>
                    <a href='{a["link"]}' target='_blank' style='color:#f0c040;text-decoration:none'>{a["title"]}</a>
                </div>
                <div style='font-size:11px;color:#888;margin-top:4px'>{a["source"]} · {a["published"]} ·
                    <span style='color:{border_color}'>{a["sentiment"]}</span></div>
                <div style='font-size:12px;color:#aaa;margin-top:6px'>{a["summary"]}</div>
            </div>""", unsafe_allow_html=True)

        # Sentiment breakdown chart
        sent_counts = {"Bullish 🟢": 0, "Bearish 🔴": 0, "Neutral ⚪": 0}
        for a in articles:
            if a["sentiment"] in sent_counts:
                sent_counts[a["sentiment"]] += 1

        fig_sent = go.Figure(go.Pie(
            labels=list(sent_counts.keys()),
            values=list(sent_counts.values()),
            hole=0.5,
            marker_colors=["#00c853", "#d50000", "#555"],
        ))
        fig_sent.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d0d",
            height=280, title="Shpërndarja e Sentimentit",
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
        )
        st.plotly_chart(fig_sent, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ECONOMIC CALENDAR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Kalendar Ekonomik":
    st.markdown("### 📅 Kalendar Ekonomik — Ngjarje me Ndikim në Gold")

    with st.spinner("Duke marrë kalendarin..."):
        events = fetch_calendar_cached()

    if not events:
        st.info("Nuk u mor kalendari.")
    else:
        show_gold_only = st.checkbox("Trego vetëm ngjarjet që ndikojnë Gold", value=True)
        if show_gold_only:
            events = [e for e in events if e.get("affects_gold")]

        impact_filter = st.multiselect(
            "Impact", ["High 🔴", "Medium 🟡", "Low ⚪"],
            default=["High 🔴", "Medium 🟡"],
        )
        events = [e for e in events if e.get("impact") in impact_filter]

        if not events:
            st.info("Nuk ka ngjarje me këto filtra.")
        else:
            ev_df = pd.DataFrame(events)[["event","currency","impact","time","forecast","previous","actual","affects_gold"]]
            ev_df.columns = ["Ngjarja","Monedha","Impact","Koha","Forecast","Previous","Actual","Gold?"]
            ev_df["Gold?"] = ev_df["Gold?"].map({True:"🥇 Po", False:"—"})

            def color_impact(val):
                if "High" in str(val):   return "color: #d50000; font-weight:bold"
                if "Medium" in str(val): return "color: #ffd600"
                return "color: #888"

            st.dataframe(
                ev_df.style.map(color_impact, subset=["Impact"]),
                use_container_width=True, hide_index=True,
            )

        # Gold impact guide
        st.markdown("---")
        st.markdown("#### 📌 Si ndikojnë ngjarjet në Gold")
        guide = {
            "NFP 📈 (më shumë jobs)":          "USD forcohet → Gold bie 🔴",
            "NFP 📉 (më pak jobs)":             "USD dobësohet → Gold ngrihet 🟢",
            "CPI i lartë (inflacion)":          "Gold ngrihet si mbrojtje 🟢",
            "Fed ngre normat":                  "USD forcohet → Gold bie 🔴",
            "Fed ul normat / QE":               "USD dobësohet → Gold ngrihet 🟢",
            "Krizë/Luftë/Pasiguri gjeopolitike": "Safe haven → Gold ngrihet 🟢",
            "Rritja ekonomike e fortë":         "Risk-on → Gold bie 🔴",
        }
        for event_name, impact_desc in guide.items():
            st.markdown(f"- **{event_name}** → {impact_desc}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI ASISTENT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "AI Asistent":
    st.markdown("### 🤖 AI Asistent — Pyet për XAUUSD")

    if not ANTHROPIC_API_KEY:
        st.info("💡 Për të aktivizuar AI Asistentin, shko tek **console.anthropic.com**, merr API key falas dhe vendose në `config.py` → `ANTHROPIC_API_KEY`")

    # Anomaly detection card
    df_feat_anom = ml.add_features(df.copy()).dropna()
    anom_df = anom.detect_anomalies(df_feat_anom)
    anom_df = anom.zscore_anomalies(anom_df)
    anom_summary = anom.get_anomaly_summary(anom_df)

    a1, a2, a3 = st.columns(3)
    is_anom = anom_summary.get("is_anomaly_now", False)
    a1.markdown(f"""<div class='metric-card'>
        <div style='font-size:11px;color:#aaa'>Anomali Tani</div>
        <div style='font-size:20px;font-weight:bold;color:{"#d50000" if is_anom else "#00c853"}'>
        {"⚠ Po!" if is_anom else "✅ Jo"}</div>
    </div>""", unsafe_allow_html=True)
    a2.markdown(f"""<div class='metric-card'>
        <div style='font-size:11px;color:#aaa'>Z-Score</div>
        <div style='font-size:20px;font-weight:bold;color:#f0c040'>{anom_summary.get("zscore", 0):+.2f}</div>
    </div>""", unsafe_allow_html=True)
    a3.markdown(f"""<div class='metric-card'>
        <div style='font-size:11px;color:#aaa'>Severity</div>
        <div style='font-size:16px;font-weight:bold'>{anom_summary.get("severity","—")}</div>
    </div>""", unsafe_allow_html=True)

    # Alert if anomaly
    if is_anom:
        st.warning(f"⚡ **Lëvizje anomale e zbuluar!** Z-Score: {anom_summary.get('zscore',0):+.2f} — Trego kujdes!")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            tg.send_anomaly_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                   current_price, anom_summary.get("zscore",0),
                                   anom_summary.get("severity","—"))

    st.markdown("---")

    # Anomaly chart
    if not anom_df.empty and "zscore" in anom_df.columns:
        fig_anom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_anom.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00c853", decreasing_line_color="#d50000",
            name="XAUUSD"), row=1, col=1)

        anom_points = anom_df[anom_df["anomaly"] == True]
        if not anom_points.empty:
            fig_anom.add_trace(go.Scatter(
                x=anom_points.index, y=anom_points["Close"],
                mode="markers", marker=dict(color="#ff6b00", size=10, symbol="star"),
                name="Anomali"), row=1, col=1)

        fig_anom.add_trace(go.Scatter(
            x=anom_df.index, y=anom_df["zscore"],
            line=dict(color="#f0c040", width=1.5), name="Z-Score"), row=2, col=1)
        fig_anom.add_hline(y=3, line_dash="dot", line_color="#d50000", row=2, col=1)
        fig_anom.add_hline(y=-3, line_dash="dot", line_color="#00c853", row=2, col=1)
        fig_anom.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                 plot_bgcolor="#0d0d0d", height=500,
                                 xaxis_rangeslider_visible=False,
                                 margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💬 Chat me AI")

    # Build market context
    supports_ai, resistances_ai = tech.find_support_resistance(df)
    fib_ai, _ = tech.fibonacci_levels(df)
    patterns_ai = tech.detect_patterns(df)
    news_ai = fetch_news_cached()
    overall_s, _ = nws.overall_sentiment(news_ai)
    headlines = [a["title"] for a in news_ai[:5]]

    df_feat_ai = ml.add_features(df.copy()).dropna()
    market_ctx = ai_asst.build_market_context(
        df_feat_ai, signal, confidence, current_price, acc,
        supports_ai, resistances_ai, fib_ai, patterns_ai,
        overall_s, headlines,
    )

    # Quick questions
    st.markdown("**Pyetje të shpejta:**")
    q_cols = st.columns(4)
    for i, q in enumerate(ai_asst.QUICK_QUESTIONS):
        if q_cols[i % 4].button(q, key=f"qq_{i}"):
            st.session_state["ai_input"] = q

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Pyet AI-n për XAUUSD...")
    if "ai_input" in st.session_state and st.session_state["ai_input"]:
        user_input = st.session_state.pop("ai_input")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Duke analizuar..."):
                response = ai_asst.ask_ai(user_input, market_ctx, ANTHROPIC_API_KEY,
                                           st.session_state["chat_history"][:-1])
            st.markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

    if st.button("🗑 Pastro chat"):
        st.session_state["chat_history"] = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — RISK CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Risk Calculator":
    st.markdown("### 📊 Risk Calculator & Position Sizing")

    df_feat_r = ml.add_features(df.copy()).dropna()
    latest_r  = df_feat_r.iloc[-1]
    atr_val   = float(latest_r.get("atr", 5.0))

    # Auto-suggest based on current signal
    auto_sl_tp = rt.suggest_sl_tp(current_price, signal, atr_val)

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("#### 📥 Inputet")
        account_bal = st.number_input("Balanca ($)", value=10000.0, step=500.0)
        risk_pct    = st.slider("Risk % për tregti", 0.5, 5.0, 1.0, 0.5)
        entry       = st.number_input("Entry Price", value=float(current_price), step=0.1, format="%.2f")
        sl          = st.number_input("Stop Loss", value=float(auto_sl_tp["stop_loss"]), step=0.1, format="%.2f")
        tp          = st.number_input("Take Profit", value=float(auto_sl_tp["take_profit_1"]), step=0.1, format="%.2f")

        st.caption(f"💡 AI sugjeron SL: ${auto_sl_tp['stop_loss']:,.2f} | TP1: ${auto_sl_tp['take_profit_1']:,.2f} | TP2: ${auto_sl_tp['take_profit_2']:,.2f} (bazuar ATR: ${atr_val:.2f})")

    with rc2:
        st.markdown("#### 📤 Rezultati")
        plan = rt.full_trade_plan(account_bal, risk_pct, entry, sl, tp)

        qual_color = "#00c853" if "Excellent" in plan["quality"] else "#ffd600" if "mirë" in plan["quality"] else "#d50000"

        metrics = [
            ("Lot Size", f"{plan['lot_size']}"),
            ("Risk USD", f"${plan['risk_usd']:,.2f}"),
            ("Reward USD", f"${plan['reward_usd']:,.2f}"),
            ("Risk/Reward", f"1 : {plan['rr_ratio']}"),
            ("SL Pips", f"{plan['sl_pips']:.0f}"),
            ("TP Pips", f"{plan['tp_pips']:.0f}"),
            ("Pip Value", f"${plan['pip_value']:.2f}"),
            ("Cilësia", plan["quality"]),
        ]
        for label, val in metrics:
            c_l, c_v = st.columns([1, 1])
            c_l.markdown(f"<span style='color:#aaa'>{label}</span>", unsafe_allow_html=True)
            c_v.markdown(f"<b style='color:#f0c040'>{val}</b>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"✅ **Nëse fitojmë:** ${plan['balance_after_win']:,.2f}")
        st.markdown(f"❌ **Nëse humbasim:** ${plan['balance_after_loss']:,.2f}")

    # Pip value calculator
    st.markdown("---")
    st.markdown("#### 💰 Pip Value Calculator")
    p1, p2, p3 = st.columns(3)
    lot_calc = p1.number_input("Lot Size", value=0.1, step=0.01, min_value=0.01, format="%.2f", key="pvc_lot")
    pv = rt.pip_value(lot_calc)
    p2.metric("Pip Value (0.01)", f"${pv:.4f}")
    p3.metric("10 Pips", f"${pv*10:.2f}")

    # Position sizing table
    st.markdown("---")
    st.markdown("#### 📐 Position Sizing per Risk %")
    ps_data = []
    for rp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ps = rt.position_size(account_bal, rp, entry, sl)
        ps_data.append({
            "Risk %": f"{rp}%",
            "Risk USD": f"${ps['risk_usd']:,.2f}",
            "Lot Size": ps["lot_size"],
            "Mini Lots": ps["mini_lots"],
            "Micro Lots": ps["micro_lots"],
        })
    st.dataframe(pd.DataFrame(ps_data), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — KORRELACIONE & COT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Korrelacione & COT":
    st.markdown("### 📈 Korrelacione & COT Report")

    corr_tab, cot_tab = st.tabs(["📊 Korrelacione", "📋 COT Report"])

    with corr_tab:
        with st.spinner("Duke ngarkuar korrelacionet (1 min)..."):
            returns_df, corr_matrix = corr.fetch_correlations("1y")

        if corr_matrix.empty:
            st.error("Nuk u morën të dhënat e korrelacionit.")
        else:
            corr_summary = corr.gold_correlation_summary(corr_matrix)

            # Correlation cards
            cr_cols = st.columns(3)
            for i, item in enumerate(corr_summary):
                c = item["correlation"]
                try:
                    if c is None or pd.isna(c):
                        c = 0.0
                    c = float(c)
                except Exception:
                    c = 0.0
                color = "#00c853" if c > 0.3 else "#d50000" if c < -0.3 else "#aaa"
                n = max(0, min(10, int(abs(c) * 10)))
                bar = "█" * n + "░" * (10 - n)
                cr_cols[i % 3].markdown(f"""<div class='metric-card' style='margin-bottom:10px'>
                    <div style='font-weight:bold;color:#e0e0e0'>{item['asset']}</div>
                    <div style='font-size:22px;font-weight:bold;color:{color}'>{c:+.3f}</div>
                    <div style='font-size:10px;color:{color}'>{bar}</div>
                    <div style='font-size:10px;color:#aaa'>{item['strength']}</div>
                    <div style='font-size:11px;color:#888;margin-top:4px'>{item['signal']}</div>
                </div>""", unsafe_allow_html=True)

            # Correlation heatmap
            st.markdown("#### Heatmap Korrelacionesh")
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                    height=450, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Rolling correlation
            rolling = corr.rolling_correlation(returns_df, 30)
            if not rolling.empty:
                st.markdown("#### Rolling 30-ditë Korrelacion me Gold")
                fig_roll = go.Figure()
                for col in rolling.columns[:4]:
                    fig_roll.add_trace(go.Scatter(
                        x=rolling.index, y=rolling[col],
                        name=col, line=dict(width=1.5)))
                fig_roll.add_hline(y=0, line_dash="dot", line_color="#555")
                fig_roll.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                        plot_bgcolor="#0d0d0d", height=350,
                                        margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_roll, use_container_width=True)

    with cot_tab:
        with st.spinner("Duke marrë COT data..."):
            cot_df  = cot.fetch_cot_data()
            cot_sum = cot.cot_summary(cot_df)

        if cot_sum.get("is_simulated"):
            st.info("ℹ Data COT është e simuluar (CFTC i shpërndan javore, dita e enjte).")

        if cot_sum:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("NC Net Long",  f"{cot_sum.get('nc_net',0):,}")
            cc2.metric("NC Long",      f"{cot_sum.get('nc_long',0):,}")
            cc3.metric("NC Short",     f"{cot_sum.get('nc_short',0):,}")
            cc4.metric("Ndryshim/Javë",f"{cot_sum.get('nc_change_wk',0):+,}")

            st.markdown(f"**Sentiment:** {cot_sum.get('sentiment','—')}")
            st.markdown(f"**Comercial Hedgers:** {cot_sum.get('commercial_view','—')}")

        if cot_df is not None and not cot_df.empty:
            fig_cot = go.Figure()
            fig_cot.add_trace(go.Bar(
                x=cot_df["date"], y=cot_df["net_noncommercial"],
                name="Net Non-Commercial",
                marker_color=["#00c853" if v > 0 else "#d50000" for v in cot_df["net_noncommercial"]],
            ))
            fig_cot.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                   plot_bgcolor="#0d0d0d", height=350,
                                   title="COT — Net Non-Commercial Positions (Spekulatorë)",
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_cot, use_container_width=True)

            st.markdown("#### 📖 Si të lexosh COT-in")
            st.markdown("""
- **Net Non-Commercial pozitiv (lart)** → Spekulatorët janë bullish → signal bullish 🟢
- **Net Non-Commercial negativ (poshtë)** → Spekulatorët janë bearish → signal bearish 🔴
- **Commercials shumë short** → Hedgers mbrojnë → çmimi mund të jetë i lartë
- **Ndryshim i madh javор** → momentum i ri në treg
            """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 — SEZONALITETI
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Sezonaliteti":
    st.markdown("### 🌦 Sezonaliteti i Gold-it (10 vjet)")

    with st.spinner("Duke analizuar historikun sezonal..."):
        try:
            seas_data = seas.fetch_seasonality(10)
        except Exception as e:
            seas_data = {}
            st.error(f"Gabim: {e}")

    if seas_data:
        cur_name = seas_data.get("current_month_name", "")
        cur_stats = seas_data.get("current_stats")

        if cur_stats is not None:
            s1, s2, s3 = st.columns(3)
            s1.markdown(f"""<div class='metric-card'>
                <div style='font-size:11px;color:#aaa'>Muaji Aktual ({cur_name})</div>
                <div style='font-size:22px;font-weight:bold;color:{"#00c853" if cur_stats["avg_pct"]>0 else "#d50000"}'>
                    {cur_stats["avg_pct"]:+.2f}%</div>
                <div style='color:#aaa;font-size:11px'>Mesatare historike</div>
            </div>""", unsafe_allow_html=True)

            best = seas_data.get("best_month")
            worst = seas_data.get("worst_month")
            s2.markdown(f"""<div class='metric-card'>
                <div style='font-size:11px;color:#aaa'>Muaji Më i Mirë</div>
                <div style='font-size:22px;font-weight:bold;color:#00c853'>
                    {best["month_name"]} ({best["avg_pct"]:+.1f}%)</div>
            </div>""", unsafe_allow_html=True)
            s3.markdown(f"""<div class='metric-card'>
                <div style='font-size:11px;color:#aaa'>Muaji Më i Keq</div>
                <div style='font-size:22px;font-weight:bold;color:#d50000'>
                    {worst["month_name"]} ({worst["avg_pct"]:+.1f}%)</div>
            </div>""", unsafe_allow_html=True)

        # Monthly bar chart
        monthly_df = seas_data.get("monthly_avg", pd.DataFrame())
        if not monthly_df.empty:
            fig_seas = go.Figure(go.Bar(
                x=monthly_df["month_name"],
                y=monthly_df["avg_pct"],
                marker_color=["#00c853" if v > 0 else "#d50000" for v in monthly_df["avg_pct"]],
                text=monthly_df["avg_pct"].round(1).astype(str) + "%",
                textposition="outside",
            ))
            fig_seas.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                title="Kthimi Mesatar Mujor i Gold (10 vjet)",
                height=400, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_seas, use_container_width=True)

        # Day of week
        dow = seas_data.get("dow_perf", pd.DataFrame())
        if not dow.empty:
            st.markdown("#### Performanca sipas ditës së javës")
            fig_dow = go.Figure(go.Bar(
                x=dow["dow_name"], y=dow["avg_pct"],
                marker_color=["#00c853" if v > 0 else "#d50000" for v in dow["avg_pct"]],
            ))
            fig_dow.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                   plot_bgcolor="#0d0d0d", height=300,
                                   margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_dow, use_container_width=True)

        # Heatmap
        heatmap = seas_data.get("heatmap", pd.DataFrame())
        if not heatmap.empty:
            month_labels = ["Jan","Feb","Mar","Apr","Maj","Qer","Kor","Gus","Sht","Tet","Nën","Dhj"]
            fig_heat = go.Figure(go.Heatmap(
                z=heatmap.values,
                x=[month_labels[c-1] for c in heatmap.columns],
                y=heatmap.index.astype(str).tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=heatmap.round(1).values,
                texttemplate="%{text}%",
                textfont={"size": 9},
            ))
            fig_heat.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0d",
                title="Heatmap Mujore (% kthim sipas vitit)",
                height=450, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 10 — TRADING JOURNAL
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Trading Journal":
    st.markdown("### 📓 Trading Journal & P&L Tracker")

    jrn.init_db()
    j_tab1, j_tab2, j_tab3 = st.tabs(["➕ Shto Tregti", "📋 Historiku", "📊 Statistikat"])

    # ── Add trade ──
    with j_tab1:
        st.markdown("#### Regjistro Tregti të Re")
        jc1, jc2, jc3 = st.columns(3)
        j_dir    = jc1.selectbox("Drejtimi", ["BUY", "SELL"])
        j_lot    = jc2.number_input("Lot Size", value=0.1, step=0.01, format="%.2f", key="j_lot")
        j_entry  = jc3.number_input("Entry", value=float(current_price), step=0.1, format="%.2f", key="j_entry")
        jc4, jc5 = st.columns(2)
        j_sl     = jc4.number_input("Stop Loss", value=0.0, step=0.1, format="%.2f", key="j_sl")
        j_tp     = jc5.number_input("Take Profit", value=0.0, step=0.1, format="%.2f", key="j_tp")
        j_setup  = st.text_input("Setup/Arsyeja", placeholder="Shembull: EMA crossover + RSI oversold")
        j_emotion= st.selectbox("Gjendja emocionale", ["Neutral", "Konfident", "Frikësuar", "Lakmitar", "Sipas planit"])
        j_rating = st.slider("Vlerëso setup (1-5)", 1, 5, 3)
        j_notes  = st.text_area("Shënime", height=80)

        if st.button("✅ Shto Tregti", type="primary"):
            tid = jrn.add_trade(j_dir, j_lot, j_entry,
                                 j_sl if j_sl > 0 else None,
                                 j_tp if j_tp > 0 else None,
                                 j_setup, j_notes, j_emotion, j_rating)
            st.success(f"✅ Tregti #{tid} u regjistrua!")

        st.markdown("---")
        st.markdown("#### Mbyll Tregti të Hapur")
        open_trades = jrn.get_all_trades("OPEN")
        if open_trades.empty:
            st.info("Nuk ka tregti të hapura.")
        else:
            trade_opts = {f"#{row['id']} {row['direction']} @ {row['entry_price']}": row["id"]
                          for _, row in open_trades.iterrows()}
            sel_trade = st.selectbox("Zgjidh tregti", list(trade_opts.keys()))
            close_price = st.number_input("Exit Price", value=float(current_price), step=0.1, format="%.2f")
            close_notes = st.text_input("Shënime mbyllje")
            if st.button("🔴 Mbyll Tregti"):
                result = jrn.close_trade(trade_opts[sel_trade], close_price, close_notes)
                pnl = result.get("pnl_usd", 0)
                color_r = "#00c853" if pnl > 0 else "#d50000"
                st.markdown(f"<b style='color:{color_r}'>{'WIN' if pnl > 0 else 'LOSS'}: ${pnl:+.2f} ({result.get('pips',0):+.1f} pips)</b>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📝 Shto Shënim Ditari")
        note_text = st.text_area("Shënim", height=100, placeholder="Si shkoi dita, çfarë mësova...")
        note_mood = st.selectbox("Humor", ["Neutral","Pozitiv","Negativ","Stresuar","Fokusuar"])
        if st.button("💾 Ruaj Shënimin"):
            jrn.add_note(note_text, note_mood, f"XAU ${current_price:,.2f}")
            st.success("Shënimi u ruajt!")

    # ── History ──
    with j_tab2:
        st.markdown("#### Të gjitha tregtitë")
        status_f = st.selectbox("Filtro", ["Të gjitha","OPEN","CLOSED"], key="jf")
        trades_all = jrn.get_all_trades(None if status_f == "Të gjitha" else status_f)
        if trades_all.empty:
            st.info("Nuk ka tregti të regjistruara.")
        else:
            show_cols = ["id","date_open","date_close","direction","lot_size",
                         "entry_price","exit_price","pnl_usd","pips","status","setup","emotion","rating"]
            show_cols = [c for c in show_cols if c in trades_all.columns]
            st.dataframe(
                trades_all[show_cols].style.map(
                    lambda v: "color:#00c853" if v == "WIN" or (isinstance(v, float) and v > 0)
                              else "color:#d50000" if v == "LOSS" or (isinstance(v, float) and v < 0) else "",
                    subset=[c for c in ["pnl_usd","status"] if c in show_cols],
                ),
                use_container_width=True, hide_index=True,
            )

            del_id = st.number_input("ID tregti për fshirje", value=0, step=1)
            if st.button("🗑 Fshi Tregti") and del_id > 0:
                jrn.delete_trade(int(del_id))
                st.success(f"Tregti #{del_id} u fshi.")
                st.rerun()

    # ── Stats ──
    with j_tab3:
        stats = jrn.get_statistics()
        if stats.get("total", 0) == 0:
            st.info("Nuk ka tregti të mbyllura ende.")
        else:
            st1, st2, st3, st4 = st.columns(4)
            st1.metric("Win Rate",     f"{stats['win_rate']}%")
            st2.metric("Total P&L",    f"${stats['total_pnl']:+,.2f}")
            st3.metric("Profit Factor",f"{stats['profit_factor']}")
            st4.metric("Total Trades", stats["total"])

            st5, st6, st7, st8 = st.columns(4)
            st5.metric("Avg Win",      f"${stats['avg_win']:+.2f}")
            st6.metric("Avg Loss",     f"${stats['avg_loss']:+.2f}")
            st7.metric("Best Trade",   f"${stats['best_trade']:+.2f}")
            st8.metric("Worst Trade",  f"${stats['worst_trade']:+.2f}")

            # Monthly P&L chart
            monthly_pnl = stats.get("monthly", pd.DataFrame())
            if not monthly_pnl.empty:
                fig_jpnl = go.Figure(go.Bar(
                    x=monthly_pnl["month"], y=monthly_pnl["pnl_usd"],
                    marker_color=["#00c853" if v > 0 else "#d50000" for v in monthly_pnl["pnl_usd"]],
                    text=monthly_pnl["pnl_usd"].round(0).astype(str),
                    textposition="outside",
                ))
                fig_jpnl.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                                        plot_bgcolor="#0d0d0d", height=350,
                                        title="P&L Mujor",
                                        margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_jpnl, use_container_width=True)

            # Notes
            st.markdown("#### 📝 Shënime të Fundit")
            notes_df = jrn.get_notes(10)
            if not notes_df.empty:
                for _, row in notes_df.iterrows():
                    mood_color = "#00c853" if row.get("mood") == "Pozitiv" else "#d50000" if row.get("mood") == "Negativ" else "#aaa"
                    st.markdown(f"""<div class='news-card'>
                        <div style='font-size:11px;color:#aaa'>{row.get('date','')} ·
                            <span style='color:{mood_color}'>{row.get('mood','')}</span> ·
                            {row.get('market','')}</div>
                        <div style='margin-top:6px'>{row.get('note','')}</div>
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 11 — SCREENSHOT ANALYZER (Vision AI)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Analizë Screenshot":
    import chart_analyzer as ca

    st.markdown("### 📸 Analizë e Chartit nga Screenshot")
    st.caption("Ngarko një screenshot të chartit (TradingView, OANDA, MT4/5) — AI vizion analizon dhe të jep BUY/SELL me SL/TP.")

    # API key — from secrets, env, or user input
    api_key_default = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    try:
        api_key_default = api_key_default or st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    with st.expander("🔑 Vendosjet AI", expanded=True):
        provider_choice = st.radio(
            "Provider",
            ["Google Gemini (FALAS)", "Anthropic Claude (me pagesë)"],
            index=0,
            horizontal=True,
            help="Gemini është falas me 1500 kërkesa/ditë, pa kartë krediti.",
        )
        is_gemini = "Gemini" in provider_choice

        if is_gemini:
            gem_default = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            try:
                gem_default = gem_default or st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
            api_key_input = st.text_input(
                "Gemini API Key",
                value=gem_default,
                type="password",
                help="Merre FALAS: https://aistudio.google.com/apikey (vetëm Google account, pa kartë)",
            )
            model_choice = st.selectbox(
                "Modeli",
                ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
                index=0,
                help="2.5-flash = i shpejtë & falas. 2.5-pro = më i saktë (kufi më i ulët falas).",
            )
            st.markdown(
                "<div style='font-size:11px;color:#8a93a6'>💡 Si ta marrësh: hyr në "
                "<a href='https://aistudio.google.com/apikey' target='_blank' style='color:#4ea3ff'>aistudio.google.com/apikey</a>"
                " me Google account → 'Create API key' → kopjoje këtu.</div>",
                unsafe_allow_html=True,
            )
        else:
            api_key_input = st.text_input(
                "Anthropic API Key",
                value=api_key_default,
                type="password",
                help="Merre nga https://console.anthropic.com/settings/keys (kërkon kredite me pagesë)",
            )
            model_choice = st.selectbox(
                "Modeli",
                ["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"],
                index=0,
            )

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Tërhiq këtu screenshot-in (PNG/JPG)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
    with col_info:
        st.markdown("""
<div style='background:rgba(78,163,255,0.06);border-left:3px solid #4ea3ff;
            border-radius:6px;padding:10px;font-size:12px;color:#c8d0dc'>
<b>💡 Si funksionon:</b><br>
1. Bëj screenshot të chartit XAUUSD<br>
2. Ngarkoje këtu<br>
3. AI lexon trendin, S/R, RSI/MACD<br>
4. Të jep plan tregtie me SL/TP<br>
5. Të thotë <b>nëse të hysh apo jo</b>
</div>""", unsafe_allow_html=True)

    if uploaded is not None:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption=f"📷 {uploaded.name}", use_container_width=True)

        analyze_btn = st.button("🔍 Analizo Chart-in", type="primary", use_container_width=True)

        if analyze_btn:
            if not api_key_input:
                st.error("⚠️ Vendos Anthropic API Key te 'Vendosjet AI' më sipër.")
            else:
                with st.spinner("🧠 AI po analizon chart-in... (10-20 sekonda)"):
                    mime = "image/jpeg" if uploaded.type and "jpeg" in uploaded.type else (
                           "image/webp" if uploaded.type and "webp" in uploaded.type else "image/png")
                    plan = ca.analyze_chart(
                        img_bytes,
                        api_key=api_key_input,
                        current_price=float(current_price),
                        image_mime=mime,
                        model=model_choice,
                        provider="gemini" if is_gemini else "anthropic",
                    )

                if plan.get("_error"):
                    st.error(f"❌ {plan['_error']}")
                else:
                    icon, label, color = ca.format_decision_badge(plan)

                    # ── Headline decision ──
                    st.markdown(f"""
<div style='background:linear-gradient(135deg,{color}22,{color}05);
            border:2px solid {color};border-radius:14px;padding:22px;margin:14px 0;
            text-align:center'>
  <div style='font-size:14px;color:#8a93a6;letter-spacing:0.1em'>VENDIMI</div>
  <div style='font-size:42px;font-weight:800;color:{color};margin:6px 0'>{icon} {label}</div>
  <div style='font-size:13px;color:#c8d0dc'>Konfidenca: <b>{plan.get('confidence',0)}%</b>
       · Trendi: <b>{plan.get('trend','—')}</b>
       · TF: <b>{plan.get('timeframe','—')}</b></div>
</div>""", unsafe_allow_html=True)

                    # ── SL/TP cards ──
                    if plan.get("direction") in ("BUY", "SELL"):
                        e  = plan.get("entry", 0)
                        sl = plan.get("stop_loss", 0)
                        t1 = plan.get("take_profit_1", 0)
                        t2 = plan.get("take_profit_2", 0)
                        rr = plan.get("risk_reward", 0)

                        cA, cB, cC, cD = st.columns(4)
                        cA.markdown(f"""<div style='background:rgba(78,163,255,0.08);border:1px solid #4ea3ff;border-radius:10px;padding:14px;text-align:center'>
<div style='font-size:11px;color:#8a93a6'>ENTRY</div>
<div style='font-size:22px;font-weight:700;color:#4ea3ff'>${e:,.2f}</div>
</div>""", unsafe_allow_html=True)
                        cB.markdown(f"""<div style='background:rgba(255,82,82,0.08);border:1px solid #ff5252;border-radius:10px;padding:14px;text-align:center'>
<div style='font-size:11px;color:#8a93a6'>STOP LOSS</div>
<div style='font-size:22px;font-weight:700;color:#ff5252'>${sl:,.2f}</div>
<div style='font-size:10px;color:#8a93a6'>{abs(e-sl):.2f} pts</div>
</div>""", unsafe_allow_html=True)
                        cC.markdown(f"""<div style='background:rgba(0,230,118,0.08);border:1px solid #00e676;border-radius:10px;padding:14px;text-align:center'>
<div style='font-size:11px;color:#8a93a6'>TAKE PROFIT 1</div>
<div style='font-size:22px;font-weight:700;color:#00e676'>${t1:,.2f}</div>
<div style='font-size:10px;color:#8a93a6'>{abs(t1-e):.2f} pts</div>
</div>""", unsafe_allow_html=True)
                        cD.markdown(f"""<div style='background:rgba(0,200,83,0.08);border:1px solid #00c853;border-radius:10px;padding:14px;text-align:center'>
<div style='font-size:11px;color:#8a93a6'>TAKE PROFIT 2</div>
<div style='font-size:22px;font-weight:700;color:#00c853'>${t2:,.2f}</div>
<div style='font-size:10px;color:#8a93a6'>R:R 1:{rr}</div>
</div>""", unsafe_allow_html=True)

                    # ── Reasoning ──
                    st.markdown("#### 🧠 Arsyetimi")
                    st.info(plan.get("reasoning", "—"))

                    # ── Warnings ──
                    warnings = plan.get("warnings", [])
                    if warnings:
                        st.markdown("#### ⚠️ Paralajmërime")
                        for w in warnings:
                            st.warning(f"• {w}")

                    # ── Levels & patterns ──
                    cL, cR = st.columns(2)
                    with cL:
                        st.markdown("#### 📊 Nivelet kyçe")
                        kl = plan.get("key_levels", {})
                        sup = kl.get("support", []) or []
                        res = kl.get("resistance", []) or []
                        st.markdown(f"**Mbështetje:** " + (", ".join(f"${s:,.2f}" for s in sup) if sup else "—"))
                        st.markdown(f"**Rezistencë:** " + (", ".join(f"${r:,.2f}" for r in res) if res else "—"))

                        patterns = plan.get("patterns", []) or []
                        if patterns:
                            st.markdown("**Modele:** " + ", ".join(patterns))

                    with cR:
                        st.markdown("#### 📈 Indikatorët e lexuar")
                        ind = plan.get("indicators_read", {}) or {}
                        for k, v in ind.items():
                            if v:
                                st.markdown(f"- **{k.upper()}:** {v}")

                    # ── Save to journal button ──
                    if plan.get("should_enter") and plan.get("direction") in ("BUY", "SELL"):
                        st.markdown("---")
                        if st.button("📓 Ruaj këtë plan në Journal", use_container_width=True):
                            try:
                                tid = jrn.add_trade(
                                    plan["direction"], 0.1,
                                    float(plan.get("entry", current_price)),
                                    float(plan.get("stop_loss", 0)),
                                    float(plan.get("take_profit_1", 0)),
                                    setup=f"Screenshot AI · conf {plan.get('confidence',0)}%",
                                    notes=plan.get("reasoning", "")[:500],
                                )
                                st.success(f"✅ Tregtia #{tid} u shtua në Journal")
                            except Exception as e:
                                st.error(f"Gabim: {e}")

                    # ── Raw JSON ──
                    with st.expander("🔧 JSON i plotë (debug)"):
                        st.json(plan)
    else:
        st.info("⬆️ Ngarko një screenshot për të filluar analizën.")


# ── Telegram Alerts Logic ──────────────────────────────────────────────────────
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    if alrt.check_signal_changed(signal):
        tg.send_signal_alert(
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
            signal, current_price, confidence,
            predicted_price, supports_ai if "supports_ai" in dir() else [],
            resistances_ai if "resistances_ai" in dir() else [], acc,
        )

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
