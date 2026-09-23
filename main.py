import os
import time
import base64
import json
import threading
import warnings
import math
from datetime import datetime, date
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
PAIRS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]
CAPITAL_INR = 100000.00
RISK_PER_TRADE_INR = 250.00

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
REPO_NAME = st.secrets.get("REPO_NAME", os.getenv("REPO_NAME", ""))
LEDGER_FILE = "indian_tw_tuned_ledger.csv"

TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=5)
    except: pass

# ==========================================
# GITHUB PERSISTENT SYNC (NO DATA LOSS)
# ==========================================
def load_ledger_from_github():
    if not GITHUB_TOKEN or not REPO_NAME:
        return pd.read_csv(LEDGER_FILE) if os.path.exists(LEDGER_FILE) else pd.DataFrame()
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{LEDGER_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()["content"]).decode("utf-8")
        from io import StringIO
        return pd.read_csv(StringIO(content))
    return pd.DataFrame()

def sync_trade_to_github(trade_data):
    df_existing = load_ledger_from_github()
    df_new = pd.DataFrame([trade_data])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True) if not df_existing.empty else df_new
    csv_str = df_combined.to_csv(index=False)
    
    # Save locally as fallback
    df_combined.to_csv(LEDGER_FILE, index=False)
    
    if not GITHUB_TOKEN or not REPO_NAME: return
    
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{LEDGER_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get("sha") if res.status_code == 200 else None
    
    payload = {
        "message": f"Auto-log trade {trade_data.get('Stock')} at {trade_data.get('Time')}",
        "content": base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
    }
    if sha: payload["sha"] = sha
    requests.put(url, headers=headers, json=payload, timeout=10)

# ==========================================
# MARKET DATA & COSTS
# ==========================================
def fetch_indian_data(ticker, interval="15m", period="5d"):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.rename(columns={'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except: return pd.DataFrame()

def calculate_indian_costs(entry_price, exit_price, quantity, side="LONG"):
    turnover = (entry_price + exit_price) * quantity
    sell_turnover = (exit_price if side == "LONG" else entry_price) * quantity
    brokerage = 0.0  # Kotak Neo Trade Free Plan
    stt = 0.00025 * sell_turnover          # 0.025% on sell side (MIS)
    exchange_charges = 0.0000345 * turnover
    stamp_duty = 0.00003 * (entry_price * quantity)
    sebi_fees = 0.000001 * turnover
    gst = 0.18 * (brokerage + exchange_charges + sebi_fees)
    return brokerage + stt + exchange_charges + stamp_duty + sebi_fees + gst

def calculate_ehma(series, length=16):
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(round(math.sqrt(length))))
    ema_half = series.ewm(span=half_len, adjust=False).mean()
    ema_full = series.ewm(span=length, adjust=False).mean()
    diff = 2 * ema_half - ema_full
    return diff.ewm(span=sqrt_len, adjust=False).mean()

# ==========================================
# TRADING BOT ENGINE
# ==========================================
class TWTunedIndianBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {p: {'status': 'NONE'} for p in pairs}
        self.current_date = date.today()
        self.daily_trades = 0
        self.daily_pnl_inr = 0.0
        self.max_daily_trades = 10 
        self.max_daily_loss = -2000.00 
        self.max_concurrent = 3

    def check_daily_reset(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_trades = 0
            self.daily_pnl_inr = 0.0
            for p in self.positions:
                if self.positions[p]['status'] == 'PENDING_ENTRY':
                    self.positions[p] = {'status': 'NONE'}

    def process_cycle(self):
        self.check_daily_reset()
        kill_active = self.daily_trades >= self.max_daily_trades or self.daily_pnl_inr <= self.max_daily_loss
        active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')

        for pair in self.pairs:
            df = fetch_indian_data(pair, "15m", "5d")
            time.sleep(1)
            if df.empty or len(df) < 105: continue

            df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
            df['MHULL'] = calculate_ehma(df['close'], 16)
            df['SHULL'] = df['MHULL'].shift(3) 
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            df['ATR_50'] = df['ATR'].rolling(50).mean()

            c_prev = df.iloc[-3]
            c_curr = df.iloc[-2]
            live_price = df.iloc[-1]['close']

            buy_sig = (c_prev['SHULL'] >= c_prev['MHULL']) and (c_curr['SHULL'] < c_curr['MHULL']) and \
                      (c_curr['close'] > c_curr['EMA100']) and (c_curr['close'] > c_curr['MHULL']) and \
                      (c_curr['ATR'] > c_curr['ATR_50'])
                         
            sell_sig = (c_prev['SHULL'] <= c_prev['MHULL']) and (c_curr['SHULL'] > c_curr['MHULL']) and \
                       (c_curr['close'] < c_curr['EMA100']) and (c_curr['close'] < c_curr['MHULL']) and \
                       (c_curr['ATR'] > c_curr['ATR_50'])

            pos = self.positions[pair]

            if pos['status'] == 'ACTIVE':
                ep = pos['limit_price']
                gross = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                net_floating = gross - calculate_indian_costs(ep, live_price, pos['size'], pos['side'])
                self.positions[pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), net_floating)

                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 3:
                    self.close_trade(pair, live_price, "Timeout")
                    continue
                if pos['side'] == 'LONG':
                    if live_price <= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market")
                    elif live_price >= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP")
                else:
                    if live_price >= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market")
                    elif live_price <= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP")
                continue

            if pos['status'] == 'PENDING_ENTRY':
                if df.iloc[-1]['low'] <= pos['limit_price'] if pos['side'] == 'LONG' else df.iloc[-1]['high'] >= pos['limit_price']:
                    self.activate_trade(pair)
                continue

            if kill_active or active_count >= self.max_concurrent: continue
            
            atr = c_curr['ATR']
            if buy_sig:
                sl, tp = live_price - (atr * 2.0), live_price + (atr * 4.0)
                size = max(1, int(RISK_PER_TRADE_INR / max(0.1, (live_price - sl))))
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'LONG', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': size, 'max_dd_inr': 0.0}
                send_telegram_alert(f"🇮🇳 <b>[NSE Tuned] BUY SIGNAL</b>\nStock: {pair}\nPrice: ₹{live_price:,.2f}")
            elif sell_sig:
                sl, tp = live_price + (atr * 2.0), live_price - (atr * 4.0)
                size = max(1, int(RISK_PER_TRADE_INR / max(0.1, (sl - live_price))))
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'SHORT', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': size, 'max_dd_inr': 0.0}
                send_telegram_alert(f"🇮🇳 <b>[NSE Tuned] SELL SIGNAL</b>\nStock: {pair}\nPrice: ₹{live_price:,.2f}")

    def activate_trade(self, pair):
        self.positions[pair].update({'status': 'ACTIVE', 'entry_time': datetime.now(), 'max_dd_inr': 0.0})
        self.daily_trades += 1
        send_telegram_alert(f"🟢 <b>[NSE Tuned] FILLED</b>\nStock: {pair}\nQty: {self.positions[pair]['size']} @ ₹{self.positions[pair]['limit_price']:,.2f}")

    def close_trade(self, pair, exec_price, reason):
        pos = self.positions[pair]
        ep, qty = pos['limit_price'], pos['size']
        gross = (exec_price - ep) * qty if pos['side'] == 'LONG' else (ep - exec_price) * qty
        costs = calculate_indian_costs(ep, exec_price, qty, pos['side'])
        net_inr = gross - costs
        self.daily_pnl_inr += net_inr

        trade_record = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Side': pos['side'], 'Qty': qty,
            'Entry': round(ep, 2), 'Exit': round(exec_price, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Kotak_Friction_INR': round(costs, 2), 
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        }
        sync_trade_to_github(trade_record)
        self.positions[pair] = {'status': 'NONE'}
        send_telegram_alert(f"🔔 <b>[NSE Tuned] CLOSED</b>\nStock: {pair}\nReason: {reason}\nNet: ₹{net_inr:,.2f}")

# ==========================================
# MASTER THREAD
# ==========================================
class MasterEngine:
    def __init__(self):
        self.bot = TWTunedIndianBot(PAIRS)

    def loop(self):
        time.sleep(5)
        send_telegram_alert("🚀 <b>NSE Tuned Engine Online (Streamlit Cloud)</b>\nGitHub Auto-Commit Active.")
        while True:
            try:
                self.bot.process_cycle()
                time.sleep(60)
            except:
                time.sleep(60)

@st.cache_resource
def start_engine():
    engine = MasterEngine()
    t = threading.Thread(target=engine.loop, daemon=True)
    t.start()
    return engine

master = start_engine()

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="NSE Quantitative Terminal", layout="wide")
st.title("🇮🇳 NSE Quantitative Terminal (Streamlit Cloud + GitHub Sync)")

df_led = load_ledger_from_github()
net_pnl = df_led['Net_PnL_INR'].sum() if not df_led.empty and 'Net_PnL_INR' in df_led.columns else 0.0
active_cnt = sum(1 for p in master.bot.positions.values() if p['status'] == 'ACTIVE')
pending_cnt = sum(1 for p in master.bot.positions.values() if p['status'] == 'PENDING_ENTRY')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_pnl):,.2f}", f"₹{net_pnl:,.2f}")
c2.metric("Market Exposure", f"{active_cnt} Active / {pending_cnt} Pending")
c3.metric("Today's Trades", f"{master.bot.daily_trades} / {master.bot.max_daily_trades}")
c4.metric("Today's PnL", f"₹{master.bot.daily_pnl_inr:,.2f}")

st.markdown("---")
st.subheader("📊 Live Chart & Indicators")
selected_pair = st.selectbox("Select Asset:", PAIRS)
df_chart = fetch_indian_data(selected_pair, "15m", "5d")

if not df_chart.empty and len(df_chart) >= 105:
    df_chart['EMA100'] = df_chart['close'].ewm(span=100, adjust=False).mean()
    df_chart['MHULL'] = calculate_ehma(df_chart['close'], 16)
    df_chart['SHULL'] = df_chart['MHULL'].shift(3)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name="15m Candles"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SHULL'], line=dict(color='#00E676', width=1.5, dash='dot'), name="SHULL (3 Lag)"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA100'], line=dict(color='#9C27B0', width=2), name="EMA 100"))

    pos = master.bot.positions[selected_pair]
    if pos['status'] == 'PENDING_ENTRY': fig.add_hline(y=pos['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text="Limit")
    elif pos['status'] == 'ACTIVE':
        fig.add_hline(y=pos['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text="Entry")
        fig.add_hline(y=pos['tp'], line_dash="dash", line_color="#00E676", annotation_text="TP")
        fig.add_hline(y=pos['sl'], line_dash="dash", line_color="#FF5252", annotation_text="SL")

    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📜 Indian Trade Ledger (Synced with GitHub)")
if not df_led.empty:
    st.download_button(label="📥 Download Ledger CSV", data=df_led.to_csv(index=False).encode('utf-8'), file_name=LEDGER_FILE, mime='text/csv')
    st.dataframe(df_led.sort_index(ascending=False), use_container_width=True)
else:
    st.info("No trades executed yet. The ledger is ready and will auto-commit upon trade completion.")
    return diff.ewm(span=sqrt_len, adjust=False).mean()

# ==========================================
# TRADING BOT ENGINE
# ==========================================
class TWTunedIndianBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {p: {'status': 'NONE'} for p in pairs}
        self.current_date = date.today()
        self.daily_trades = 0
        self.daily_pnl_inr = 0.0
        self.max_daily_trades = 10 
        self.max_daily_loss = -2000.00 
        self.max_concurrent = 3

    def check_daily_reset(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_trades = 0
            self.daily_pnl_inr = 0.0
            for p in self.positions:
                if self.positions[p]['status'] == 'PENDING_ENTRY':
                    self.positions[p] = {'status': 'NONE'}

    def process_cycle(self):
        self.check_daily_reset()
        kill_active = self.daily_trades >= self.max_daily_trades or self.daily_pnl_inr <= self.max_daily_loss
        active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')

        for pair in self.pairs:
            df = fetch_indian_data(pair, "15m", "5d")
            time.sleep(1)
            if df.empty or len(df) < 105: continue

            df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
            df['MHULL'] = calculate_ehma(df['close'], 16)
            df['SHULL'] = df['MHULL'].shift(3) 
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            df['ATR_50'] = df['ATR'].rolling(50).mean()

            c_prev = df.iloc[-3]
            c_curr = df.iloc[-2]
            live_price = df.iloc[-1]['close']

            buy_sig = (c_prev['SHULL'] >= c_prev['MHULL']) and (c_curr['SHULL'] < c_curr['MHULL']) and \
                      (c_curr['close'] > c_curr['EMA100']) and (c_curr['close'] > c_curr['MHULL']) and \
                      (c_curr['ATR'] > c_curr['ATR_50'])
                         
            sell_sig = (c_prev['SHULL'] <= c_prev['MHULL']) and (c_curr['SHULL'] > c_curr['MHULL']) and \
                       (c_curr['close'] < c_curr['EMA100']) and (c_curr['close'] < c_curr['MHULL']) and \
                       (c_curr['ATR'] > c_curr['ATR_50'])

            pos = self.positions[pair]

            if pos['status'] == 'ACTIVE':
                ep = pos['limit_price']
                gross = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                net_floating = gross - calculate_indian_costs(ep, live_price, pos['size'], pos['side'])
                self.positions[pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), net_floating)

                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 3:
                    self.close_trade(pair, live_price, "Timeout")
                    continue
                if pos['side'] == 'LONG':
                    if live_price <= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market")
                    elif live_price >= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP")
                else:
                    if live_price >= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market")
                    elif live_price <= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP")
                continue

            if pos['status'] == 'PENDING_ENTRY':
                if df.iloc[-1]['low'] <= pos['limit_price'] if pos['side'] == 'LONG' else df.iloc[-1]['high'] >= pos['limit_price']:
                    self.activate_trade(pair)
                continue

            if kill_active or active_count >= self.max_concurrent: continue
            
            atr = c_curr['ATR']
            if buy_sig:
                sl, tp = live_price - (atr * 2.0), live_price + (atr * 4.0)
                size = max(1, int(RISK_PER_TRADE_INR / max(0.1, (live_price - sl))))
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'LONG', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': size, 'max_dd_inr': 0.0}
                send_telegram_alert(f"🇮🇳 <b>[NSE Tuned] BUY SIGNAL</b>\nStock: {pair}\nPrice: ₹{live_price:,.2f}")
            elif sell_sig:
                sl, tp = live_price + (atr * 2.0), live_price - (atr * 4.0)
                size = max(1, int(RISK_PER_TRADE_INR / max(0.1, (sl - live_price))))
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'SHORT', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': size, 'max_dd_inr': 0.0}
                send_telegram_alert(f"🇮🇳 <b>[NSE Tuned] SELL SIGNAL</b>\nStock: {pair}\nPrice: ₹{live_price:,.2f}")

    def activate_trade(self, pair):
        self.positions[pair].update({'status': 'ACTIVE', 'entry_time': datetime.now(), 'max_dd_inr': 0.0})
        self.daily_trades += 1
        send_telegram_alert(f"🟢 <b>[NSE Tuned] FILLED</b>\nStock: {pair}\nQty: {self.positions[pair]['size']} @ ₹{self.positions[pair]['limit_price']:,.2f}")

    def close_trade(self, pair, exec_price, reason):
        pos = self.positions[pair]
        ep, qty = pos['limit_price'], pos['size']
        gross = (exec_price - ep) * qty if pos['side'] == 'LONG' else (ep - exec_price) * qty
        costs = calculate_indian_costs(ep, exec_price, qty, pos['side'])
        net_inr = gross - costs
        self.daily_pnl_inr += net_inr

        trade_record = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Side': pos['side'], 'Qty': qty,
            'Entry': round(ep, 2), 'Exit': round(exec_price, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Kotak_Friction_INR': round(costs, 2), 
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        }
        sync_trade_to_github(trade_record)
        self.positions[pair] = {'status': 'NONE'}
        send_telegram_alert(f"🔔 <b>[NSE Tuned] CLOSED</b>\nStock: {pair}\nReason: {reason}\nNet: ₹{net_inr:,.2f}")

# ==========================================
# MASTER THREAD
# ==========================================
class MasterEngine:
    def __init__(self):
        self.bot = TWTunedIndianBot(PAIRS)

    def loop(self):
        time.sleep(5)
        send_telegram_alert("🚀 <b>NSE Tuned Engine Online (Streamlit Cloud)</b>\nGitHub Auto-Commit Active.")
        while True:
            try:
                self.bot.process_cycle()
                time.sleep(60)
            except:
                time.sleep(60)

@st.cache_resource
def start_engine():
    engine = MasterEngine()
    t = threading.Thread(target=engine.loop, daemon=True)
    t.start()
    return engine

master = start_engine()

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="NSE Quantitative Terminal", layout="wide")
st.title("🇮🇳 NSE Quantitative Terminal (Streamlit Cloud + GitHub Sync)")

df_led = load_ledger_from_github()
net_pnl = df_led['Net_PnL_INR'].sum() if not df_led.empty and 'Net_PnL_INR' in df_led.columns else 0.0
active_cnt = sum(1 for p in master.bot.positions.values() if p['status'] == 'ACTIVE')
pending_cnt = sum(1 for p in master.bot.positions.values() if p['status'] == 'PENDING_ENTRY')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_pnl):,.2f}", f"₹{net_pnl:,.2f}")
c2.metric("Market Exposure", f"{active_cnt} Active / {pending_cnt} Pending")
c3.metric("Today's Trades", f"{master.bot.daily_trades} / {master.bot.max_daily_trades}")
c4.metric("Today's PnL", f"₹{master.bot.daily_pnl_inr:,.2f}")

st.markdown("---")
st.subheader("📊 Live Chart & Indicators")
selected_pair = st.selectbox("Select Asset:", PAIRS)
df_chart = fetch_indian_data(selected_pair, "15m", "5d")

if not df_chart.empty and len(df_chart) >= 105:
    df_chart['EMA100'] = df_chart['close'].ewm(span=100, adjust=False).mean()
    df_chart['MHULL'] = calculate_ehma(df_chart['close'], 16)
    df_chart['SHULL'] = df_chart['MHULL'].shift(3)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name="15m Candles"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SHULL'], line=dict(color='#00E676', width=1.5, dash='dot'), name="SHULL (3 Lag)"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA100'], line=dict(color='#9C27B0', width=2), name="EMA 100"))

    pos = master.bot.positions[selected_pair]
    if pos['status'] == 'PENDING_ENTRY': fig.add_hline(y=pos['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text="Limit")
    elif pos['status'] == 'ACTIVE':
        fig.add_hline(y=pos['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text="Entry")
        fig.add_hline(y=pos['tp'], line_dash="dash", line_color="#00E676", annotation_text="TP")
        fig.add_hline(y=pos['sl'], line_dash="dash", line_color="#FF5252", annotation_text="SL")

    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📜 Indian Trade Ledger (Synced with GitHub)")
if not df_led.empty:
    st.download_button(label="📥 Download Ledger CSV", data=df_led.to_csv(index=False).encode('utf-8'), file_name=LEDGER_FILE, mime='text/csv')
    st.dataframe(df_led.sort_index(ascending=False), use_container_width=True)
else:
    st.info("No trades executed yet. The ledger is ready and will auto-commit upon trade completion.")
          
