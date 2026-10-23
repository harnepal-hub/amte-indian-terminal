import os
import time
import math
import threading
import warnings
from datetime import datetime, date, time as dtime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings('ignore')

# ==========================================
# GLOBAL CONFIGURATION (EXPANDED INDIAN EQUITIES)
# ==========================================
PAIRS = [
    "HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS",
    "ICICIBANK.NS", "TATAMOTORS.NS", "ITC.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]
CAPITAL_INR = 100000.00
RISK_PER_TRADE_INR = 250.00
USDT_INR_RATE = 1.00 

MAKER_FEE = 0.0005  
TAKER_FEE = 0.0005  
SLIPPAGE_RATE = 0.0005 

AMTE_LEDGER = "indian_amte_ledger.csv"
TW_LEDGER = "indian_tw_orig_ledger.csv"
TW_TUNED_LEDGER = "indian_tw_tuned_ledger.csv"
SUPERTREND_LEDGER = "indian_supertrend_ledger.csv"

# ==========================================
# CORE API FETCH (YFINANCE)
# ==========================================
def fetch_live_data(pair, interval, limit=120):
    try:
        valid_interval = "60m" if interval == "1h" else "15m"
        df = yf.download(tickers=pair, interval=valid_interval, period="5d", progress=False)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        df.rename(columns={'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df.tail(limit)
    except Exception as e: 
        return pd.DataFrame()

def log_trade(filename, trade_data):
    df_new = pd.DataFrame([trade_data])
    if os.path.exists(filename):
        try:
            df_old = pd.read_csv(filename)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv(filename, index=False)
        except Exception:
            df_new.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filename, index=False)

def is_market_open():
    """Strictly checks Indian market hours (9:15 AM to 3:30 PM IST, Monday-Friday)"""
    now = datetime.now()
    if now.weekday() >= 5: # Weekend check
        return False
    current_time = now.time()
    return dtime(9, 15) <= current_time <= dtime(15, 30)

def calculate_ehma(series, length=16):
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(round(math.sqrt(length))))
    ema_half = series.ewm(span=half_len, adjust=False).mean()
    ema_full = series.ewm(span=length, adjust=False).mean()
    diff = 2 * ema_half - ema_full
    return diff.ewm(span=sqrt_len, adjust=False).mean()

def calculate_pivots(df, left=33, right=21, quick_right=3):
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    last_p_high, last_p_low, last_q_high, last_q_low = np.nan, np.nan, np.nan, np.nan
    for i in range(left, n - right):
        if all(highs[i] >= highs[i - left:i]) and all(highs[i] >= highs[i+1:i + right + 1]):
            last_p_high = highs[i]
        if all(lows[i] <= lows[i - left:i]) and all(lows[i] <= lows[i+1:i + right + 1]):
            last_p_low = lows[i]
    for i in range(left, n - quick_right):
        if all(highs[i] >= highs[i - left:i]) and all(highs[i] >= highs[i+1:i + quick_right + 1]):
            last_q_high = highs[i]
        if all(lows[i] <= lows[i - left:i]) and all(lows[i] <= lows[i+1:i + quick_right + 1]):
            last_q_low = lows[i]
    return last_p_high, last_p_low, last_q_high, last_q_low

def apply_trailing_stop(pos, live_price):
    ep = pos['limit_price']
    side = pos['side']
    risk = abs(ep - pos['original_sl'])
    if side == 'LONG' and live_price >= (ep + risk):
        if pos['sl'] < ep: pos['sl'] = ep * (1 + MAKER_FEE)
    elif side == 'SHORT' and live_price <= (ep - risk):
        if pos['sl'] > ep: pos['sl'] = ep * (1 - MAKER_FEE)

# ==========================================
# STRATEGY 1: AMTE BB PULLBACK (WITH COOLDOWN)
# ==========================================
class AMTEBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {pair: {'status': 'NONE'} for pair in pairs}
        self.cooldowns = {pair: datetime.min for pair in pairs}
        self.current_date = date.today()
        self.daily_trades = 0
        self.daily_pnl_inr = 0.0
        self.max_daily_trades = 10 
        self.max_daily_loss = -2000.00
        self.max_concurrent = 2

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
        if not is_market_open(): return
        self.check_daily_reset()
        kill_active = self.daily_trades >= self.max_daily_trades or self.daily_pnl_inr <= self.max_daily_loss
        active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')

        for pair in self.pairs:
            if datetime.now() < self.cooldowns[pair]:
                continue

            df_macro = fetch_live_data(pair, "1h", 100)
            df_micro = fetch_live_data(pair, "15m", 100)
            if df_macro.empty or df_micro.empty: continue

            df_macro['EMA_20'] = df_macro['close'].ewm(span=20, adjust=False).mean()
            df_macro['EMA_50'] = df_macro['close'].ewm(span=50, adjust=False).mean()
            c_macro = df_macro.iloc[-2]
            is_bull = (c_macro['EMA_20'] > c_macro['EMA_50'])
            is_bear = (c_macro['EMA_20'] < c_macro['EMA_50'])

            df_micro['SMA20'] = df_micro['close'].rolling(20).mean()
            df_micro['STD20'] = df_micro['close'].rolling(20).std()
            df_micro['BBL'] = df_micro['SMA20'] - (df_micro['STD20'] * 1.5)
            df_micro['BBU'] = df_micro['SMA20'] + (df_micro['STD20'] * 1.5)
            tr = pd.concat([df_micro['high'] - df_micro['low'], (df_micro['high'] - df_micro['close'].shift()).abs(), (df_micro['low'] - df_micro['close'].shift()).abs()], axis=1).max(axis=1)
            df_micro['ATR'] = tr.rolling(14).mean()
            
            c_micro = df_micro.iloc[-2]
            live_price = df_micro.iloc[-1]['close']
            pos = self.positions[pair]

            if pos['status'] == 'ACTIVE':
                ep = pos['limit_price']
                floating_usd = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                floating_fees = (ep * pos['size'] * MAKER_FEE) + (live_price * pos['size'] * TAKER_FEE)
                self.positions[pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), (floating_usd - floating_fees) * USDT_INR_RATE)

                apply_trailing_stop(pos, live_price)

                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 4:
                    if (floating_usd - floating_fees) > 0:
                        self.close_trade(pair, live_price, "Timeout (Profit)", "TAKER")
                        continue
                
                if pos['side'] == 'LONG':
                    if live_price <= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market", "TAKER")
                    elif live_price >= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP", "MAKER")
                else:
                    if live_price >= pos['sl']: self.close_trade(pair, pos['sl'], "Stop Market", "TAKER")
                    elif live_price <= pos['tp']: self.close_trade(pair, pos['tp'], "Limit TP", "MAKER")
                continue

            if pos['status'] == 'PENDING_ENTRY':
                if (pos['side'] == 'LONG' and not is_bull) or (pos['side'] == 'SHORT' and not is_bear):
                    self.positions[pair] = {'status': 'NONE'}
                    continue
                if pos['side'] == 'LONG' and df_micro.iloc[-1]['low'] <= pos['limit_price']: self.activate_trade(pair)
                elif pos['side'] == 'SHORT' and df_micro.iloc[-1]['high'] >= pos['limit_price']: self.activate_trade(pair)
                continue

            if kill_active or active_count >= self.max_concurrent: continue
            
            risk_usd = RISK_PER_TRADE_INR / USDT_INR_RATE
            atr = c_micro['ATR']
            if is_bull and live_price > c_micro['BBL']:
                lp = c_micro['BBL']
                sl, tp = lp - (atr * 2.0), lp + (atr * 4.0)
                self.place_limit(pair, 'LONG', lp, sl, tp, risk_usd / max(0.0001, (lp - sl)))
            elif is_bear and live_price < c_micro['BBU']:
                lp = c_micro['BBU']
                sl, tp = lp + (atr * 2.0), lp - (atr * 4.0)
                self.place_limit(pair, 'SHORT', lp, sl, tp, risk_usd / max(0.0001, (sl - lp)))

    def place_limit(self, pair, side, lp, sl, tp, size):
        self.positions[pair] = {
            'status': 'PENDING_ENTRY', 'side': side, 'limit_price': lp, 'sl': sl, 'tp': tp, 
            'original_sl': sl, 'size': size, 'max_dd_inr': 0.0
        }

    def activate_trade(self, pair):
        self.positions[pair].update({'status': 'ACTIVE', 'entry_time': datetime.now(), 'max_dd_inr': 0.0})
        self.daily_trades += 1

    def close_trade(self, pair, exec_price, reason, order_type):
        pos = self.positions[pair]
        ep = pos['limit_price']
        exit_p = exec_price * (1 - (SLIPPAGE_RATE if order_type == "TAKER" else 0.0)) if pos['side'] == 'LONG' else exec_price * (1 + (SLIPPAGE_RATE if order_type == "TAKER" else 0.0))
        gross = (exit_p - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - exit_p) * pos['size']
        fees = (ep * pos['size'] * MAKER_FEE) + (exit_p * pos['size'] * (MAKER_FEE if order_type == "MAKER" else TAKER_FEE))
        net_inr = (gross - fees) * USDT_INR_RATE
        self.daily_pnl_inr += net_inr
        
        if "Stop" in reason:
            self.cooldowns[pair] = datetime.now() + pd.Timedelta(minutes=60)

        log_trade(AMTE_LEDGER, {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Strategy': 'AMTE', 'Side': pos['side'],
            'Entry': round(ep, 2), 'Exit': round(exit_p, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Friction_INR': round(fees, 2),
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        })
        self.positions[pair] = {'status': 'NONE'}

# ==========================================
# STRATEGY 2, 3 & 4 ENGINE (TW & SUPERTREND)
# ==========================================
class GeneralBotEngine:
    def __init__(self, pairs, strategy_type):
        self.pairs = pairs
        self.strategy_type = strategy_type
        self.positions = {pair: {'status': 'NONE'} for pair in pairs}
        self.current_date = date.today()
        self.daily_trades = 0
        self.daily_pnl_inr = 0.0
        self.max_daily_trades = 10 
        self.max_daily_loss = -2000.00 
        self.max_concurrent = 3
        
        if strategy_type == 'TW_ORIG': self.ledger = TW_LEDGER
        elif strategy_type == 'TW_TUN
