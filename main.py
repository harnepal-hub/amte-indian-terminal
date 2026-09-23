import os
import time
import base64
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

LEDGERS = {
    "AMTE": "indian_amte_ledger.csv",
    "TW_ORIG": "indian_tw_orig_ledger.csv",
    "TW_TUNED": "indian_tw_tuned_ledger.csv"
}

TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

LEDGER_COLUMNS = ['Time', 'Stock', 'Strategy', 'Side', 'Qty', 'Entry', 'Exit', 'Reason', 'Gross_INR', 'Kotak_Friction_INR', 'Net_PnL_INR', 'Max_DD_INR']

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=5)
    except: pass

# ==========================================
# GITHUB PERSISTENT SYNC 
# ==========================================
def load_ledger_from_github(strat_key):
    filename = LEDGERS[strat_key]
    if not GITHUB_TOKEN or not REPO_NAME:
        if os.path.exists(filename): return pd.read_csv(filename)
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()["content"]).decode("utf-8")
        from io import StringIO
        return pd.read_csv(StringIO(content))
    return pd.DataFrame(columns=LEDGER_COLUMNS)

def sync_trade_to_github(strat_key, trade_data):
    filename = LEDGERS[strat_key]
    df_existing = load_ledger_from_github(strat_key)
    df_new = pd.DataFrame([trade_data])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True) if not df_existing.empty else df_new
    csv_str = df_combined.to_csv(index=False)
    
    df_combined.to_csv(filename, index=False) 
    if not GITHUB_TOKEN or not REPO_NAME: return
    
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get("sha") if res.status_code == 200 else None
    
    payload = {
        "message": f"Auto-log {strat_key} trade {trade_data.get('Stock')} at {trade_data.get('Time')}",
        "content": base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
    }
    if sha: payload["sha"] = sha
    requests.put(url, headers=headers, json=payload, timeout=10)

# ==========================================
# MARKET DATA & COSTS
# ==========================================
def fetch_indian_data(ticker, interval, period="5d"):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df.rename(columns={'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
        return df
    except: return pd.DataFrame()

def calculate_indian_costs(entry_price, exit_price, quantity, side="LONG"):
    turnover = (entry_price + exit_price) * quantity
    sell_turnover = (exit_price if side == "LONG" else entry_price) * quantity
    brokerage = 0.0  
    stt = 0.00025 * sell_turnover          
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
# UNIFIED TRADING ENGINE (ALL 3 STRATEGIES)
# ==========================================
class UnifiedIndianEngine:
    def __init__(self, pairs):
        self.pairs = pairs
        self.strats = ["AMTE", "TW_ORIG", "TW_TUNED"]
        self.positions = {s: {p: {'status': 'NONE'} for p in pairs} for s in self.strats}
        self.current_date = date.today()
        self.daily_metrics = {s: {'trades': 0, 'pnl': 0.0} for s in self.strats}
        self.max_daily_trades = 10 
        self.max_daily_loss = -2000.00 
        self.max_concurrent = 3

    def check_daily_reset(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            for s in self.strats:
                self.daily_metrics[s] = {'trades': 0, 'pnl': 0.0}
                for p in self.pairs:
                    if self.positions[s][p]['status'] == 'PENDING_ENTRY':
                        self.positions[s][p] = {'status': 'NONE'}

    def process_cycle(self):
        self.check_daily_reset()
        
        for pair in self.pairs:
            df_15m = fetch_indian_data(pair, "15m", "5d")
            df_1h = fetch_indian_data(pair, "1h", "1mo")
            time.sleep(1) 
            
            if df_15m.empty or df_1h.empty or len(df_15m) < 105: continue

            df_1h['EMA20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
            df_1h['EMA50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
            macro_trend = "UP" if df_1h.iloc[-1]['EMA20'] > df_1h.iloc[-1]['EMA50'] else "DOWN"

            df_15m['EMA100'] = df_15m['close'].ewm(span=100, adjust=False).mean()
            df_15m['MHULL'] = calculate_ehma(df_15m['close'], 16)
            df_15m['SHULL_2'] = df_15m['MHULL'].shift(2)
            df_15m['SHULL_3'] = df_15m['MHULL'].shift(3)
            
            df_15m['SMA20'] = df_15m['close'].rolling(20).mean()
            df_15m['STD20'] = df_15m['close'].rolling(20).std()
            df_15m['Upper_BB'] = df_15m['SMA20'] + (df_15m['STD20'] * 1.5)
            df_15m['Lower_BB'] = df_15m['SMA20'] - (df_15m['STD20'] * 1.5)
            
            tr = pd.concat([df_15m['high'] - df_15m['low'], (df_15m['high'] - df_15m['close'].shift()).abs(), (df_15m['low'] - df_15m['close'].shift()).abs()], axis=1).max(axis=1)
            df_15m['ATR'] = tr.rolling(14).mean()
            df_15m['ATR_50'] = df_15m['ATR'].rolling(50).mean()

            c_prev = df_15m.iloc[-3]
            c_curr = df_15m.iloc[-2]
            live_price = df_15m.iloc[-1]['close']
            atr = c_curr['ATR']

            signals = {"AMTE": None, "TW_ORIG": None, "TW_TUNED": None}

            if macro_trend == "UP" and c_curr['low'] <= c_curr['Lower_BB']: signals["AMTE"] = "LONG"
            elif macro_trend == "DOWN" and c_curr['high'] >= c_curr['Upper_BB']: signals["AMTE"] = "SHORT"

            if (c_prev['SHULL_2'] >= c_prev['MHULL']) and (c_curr['SHULL_2'] < c_curr['MHULL']) and (c_curr['close'] > c_curr['EMA100']): signals["TW_ORIG"] = "LONG"
            elif (c_prev['SHULL_2'] <= c_prev['MHULL']) and (c_curr['SHULL_2'] > c_curr['MHULL']) and (c_curr['close'] < c_curr['EMA100']): signals["TW_ORIG"] = "SHORT"

            if (c_prev['SHULL_3'] >= c_prev['MHULL']) and (c_curr['SHULL_3'] < c_curr['MHULL']) and (c_curr['close'] > c_curr['EMA100']) and (c_curr['ATR'] > c_curr['ATR_50']): signals["TW_TUNED"] = "LONG"
            elif (c_prev['SHULL_3'] <= c_prev['MHULL']) and (c_curr['SHULL_3'] > c_curr['MHULL']) and (c_curr['close'] < c_curr['EMA100']) and (c_curr['ATR'] > c_curr['ATR_50']): signals["TW_TUNED"] = "SHORT"

            for strat in self.strats:
                pos = self.positions[strat][pair]
                
                if pos['status'] == 'ACTIVE':
                    ep = pos['limit_price']
                    gross = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                    net_floating = gross - calculate_indian_costs(ep, live_price, pos['size'], pos['side'])
                    self.positions[strat][pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), net_floating)

                    if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 3:
                        self.close_trade(strat, pair, live_price, "Timeout")
                        continue
                    if pos['side'] == 'LONG':
                        if live_price <= pos['sl']: self.close_trade(strat, pair, pos['sl'], "Stop Market")
                        elif live_price >= pos['tp']: self.close_trade(strat, pair, pos['tp'], "Limit TP")
                    else:
                        if live_price >= pos['sl']: self.close_trade(strat, pair, pos['sl'], "Stop Market")
                        elif live_price <= pos['tp']: self.close_trade(strat, pair, pos['tp'], "Limit TP")
                    continue

                if pos['status'] == 'PENDING_ENTRY':
                    if df_15m.iloc[-1]['low'] <= pos['limit_price'] if pos['side'] == 'LONG' else df_15m.iloc[-1]['high'] >= pos['limit_price']:
                        self.positions[strat][pair].update({'status': 'ACTIVE', 'entry_time': datetime.now(), 'max_dd_inr': 0.0})
                        self.daily_metrics[strat]['trades'] += 1
                        send_telegram_alert(f"🟢 <b>[NSE {strat}] FILLED</b>\nStock: {pair}\nPrice: ₹{pos['limit_price']:,.2f}")
                    continue

                kill_active = self.daily_metrics[strat]['trades'] >= self.max_daily_trades or self.daily_metrics[strat]['pnl'] <= self.max_daily_loss
                active_count = sum(1 for p in self.positions[strat].values() if p['status'] == 'ACTIVE')
                
                if not kill_active and active_count < self.max_concurrent and signals[strat]:
                    limit_p = c_curr['Lower_BB'] if signals[strat] == "LONG" and strat == "AMTE" else live_price
                    limit_p = c_curr['Upper_BB'] if signals[strat] == "SHORT" and strat == "AMTE" else limit_p
                    
                    sl = limit_p - (atr * 2.0) if signals[strat] == "LONG" else limit_p + (atr * 2.0)
                    tp = limit_p + (atr * 4.0) if signals[strat] == "LONG" else limit_p - (atr * 4.0)
                    size = max(1, int(RISK_PER_TRADE_INR / max(0.1, abs(limit_p - sl))))
                    
                    self.positions[strat][pair] = {'status': 'PENDING_ENTRY', 'side': signals[strat], 'limit_price': limit_p, 'sl': sl, 'tp': tp, 'size': size, 'max_dd_inr': 0.0}
                    send_telegram_alert(f"🇮🇳 <b>[NSE {strat}] {signals[strat]} SIGNAL</b>\nStock: {pair}\nPrice: ₹{limit_p:,.2f}")

    def close_trade(self, strat, pair, exec_price, reason):
        pos = self.positions[strat][pair]
        ep, qty = pos['limit_price'], pos['size']
        gross = (exec_price - ep) * qty if pos['side'] == 'LONG' else (ep - exec_price) * qty
        costs = calculate_indian_costs(ep, exec_price, qty, pos['side'])
        net_inr = gross - costs
        self.daily_metrics[strat]['pnl'] += net_inr

        trade_record = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Strategy': strat, 'Side': pos['side'], 'Qty': qty,
            'Entry': round(ep, 2), 'Exit': round(exec_price, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Kotak_Friction_INR': round(costs, 2), 
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        }
        sync_trade_to_github(strat, trade_record)
        self.positions[strat][pair] = {'status': 'NONE'}
        send_telegram_alert(f"🔔 <b>[NSE {strat}] CLOSED</b>\nStock: {pair}\nReason: {reason}\nNet: ₹{net_inr:,.2f}")

# ==========================================
# MASTER THREAD
# ==========================================
class AppRunner:
    def __init__(self):
        self.engine = UnifiedIndianEngine(PAIRS)

    def loop(self):
        time.sleep(5)
        send_telegram_alert("🚀 <b>NSE Multi-Model Engine Online</b>")
        while True:
            try:
                self.engine.process_cycle()
                time.sleep(60)
            except:
                time.sleep(60)

@st.cache_resource
def start_engine():
    runner = AppRunner()
    t = threading.Thread(target=runner.loop, daemon=True)
    t.start()
    return runner

runner = start_engine()
engine = runner.engine

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="NSE Multi-Model Terminal", layout="wide")
st.title("🇮🇳 NSE Multi-Model Terminal")

st.markdown("---")
st.subheader("🔎 Live Strategy Visualizer")
ui_pair = st.selectbox("Select Asset to Monitor:", PAIRS)
df_chart = fetch_indian_data(ui_pair, "15m", "5d")

if not df_chart.empty and len(df_chart) > 50:
    df_chart['EMA100'] = df_chart['close'].ewm(span=100, adjust=False).mean()
    df_chart['MHULL'] = calculate_ehma(df_chart['close'], 16)
    df_chart['SHULL_2'] = df_chart['MHULL'].shift(2)
    df_chart['SHULL_3'] = df_chart['MHULL'].shift(3)
    df_chart['SMA20'] = df_chart['close'].rolling(20).mean()
    df_chart['STD20'] = df_chart['close'].rolling(20).std()
    df_chart['Upper_BB'] = df_chart['SMA20'] + (df_chart['STD20'] * 1.5)
    df_chart['Lower_BB'] = df_chart['SMA20'] - (df_chart['STD20'] * 1.5)

tabs = st.tabs(["Strategy A (AMTE)", "Strategy B (TW Orig)", "Strategy C (TW Tuned)"])

for i, strat in enumerate(["AMTE", "TW_ORIG", "TW_TUNED"]):
    with tabs[i]:
        df_led = load_ledger_from_github(strat)
        net_pnl = df_led['Net_PnL_INR'].sum() if not df_led.empty else 0.0
        active_cnt = sum(1 for p in engine.positions[strat].values() if p['status'] == 'ACTIVE')
        pending_cnt = sum(1 for p in engine.positions[strat].values() if p['status'] == 'PENDING_ENTRY')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_pnl):,.2f}", f"₹{net_pnl:,.2f}")
        c2.metric("Market Exposure", f"{active_cnt} Active / {pending_cnt} Pending")
        c3.metric("Today's Trades", f"{engine.daily_metrics[strat]['trades']} / {engine.max_daily_trades}")
        c4.metric("Today's PnL", f"₹{engine.daily_metrics[strat]['pnl']:,.2f}")

        if not df_chart.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name="15m Candles"))
            
            if strat == "AMTE":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Upper_BB'], line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot'), name="Upper BB (1.5)"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Lower_BB'], line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dot'), name="Lower BB (1.5)"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA20'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name="SMA 20"))
            elif strat == "TW_ORIG":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SHULL_2'], line=dict(color='#00E676', width=1.5, dash='dot'), name="SHULL (2 Lag)"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA100'], line=dict(color='#9C27B0', width=2), name="EMA 100"))
            elif strat == "TW_TUNED":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SHULL_3'], line=dict(color='#00E676', width=1.5, dash='dot'), name="SHULL (3 Lag)"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA100'], line=dict(color='#9C27B0', width=2), name="EMA 100"))

            pos = engine.positions[strat][ui_pair]
            if pos['status'] == 'PENDING_ENTRY': 
                fig.add_hline(y=pos['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text="Limit")
            elif pos['status'] == 'ACTIVE':
                fig.add_hline(y=pos['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text="Entry")
                fig.add_hline(y=pos['tp'], line_dash="dash", line_color="#00E676", annotation_text="TP")
                fig.add_hline(y=pos['sl'], line_dash="dash", line_color="#FF5252", annotation_text="SL")

            fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), title=f"{ui_pair} - {strat} Overlays")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📜 {strat} Trade Ledger")
        if df_led.empty or len(df_led) == 0:
            st.dataframe(pd.DataFrame(columns=LEDGER_COLUMNS), use_container_width=True)
        else:
            st.download_button(label=f"📥 Download {strat} CSV", data=df_led.to_csv(index=False).encode('utf-8'), file_name=LEDGERS[strat], mime='text/csv')
            st.dataframe(df_led.sort_index(ascending=False), use_container_width=True)
                                                                                             
