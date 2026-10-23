import os
import time
import math
import threading
import warnings
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings('ignore')

# ==========================================
# GLOBAL CONFIGURATION (INDIAN EQUITIES)
# ==========================================
PAIRS = ["HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS"]
CAPITAL_INR = 100000.00
RISK_PER_TRADE_INR = 250.00
USDT_INR_RATE = 1.00 # Set to 1 because we are directly trading in INR

# Friction estimates for Indian Equity Delivery/Intraday (STT, Brokerage, Exchange charges)
MAKER_FEE = 0.0005  
TAKER_FEE = 0.0005  
SLIPPAGE_RATE = 0.0005 

AMTE_LEDGER = "indian_amte_ledger.csv"
TW_LEDGER = "indian_tw_orig_ledger.csv"
TW_TUNED_LEDGER = "indian_tw_tuned_ledger.csv"

# ==========================================
# CORE API FETCH (UPGRADED TO YFINANCE)
# ==========================================
def fetch_live_data(pair, interval, limit=120):
    """Pulls live NSE data for free without requiring daily broker API tokens."""
    try:
        valid_interval = "60m" if interval == "1h" else "15m"
        df = yf.download(tickers=pair, interval=valid_interval, period="5d", progress=False)
        if df.empty: return pd.DataFrame()
        
        # Flatten MultiIndex columns if present (yfinance specific)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        df.rename(columns={'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        
        # Ensure correct types
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

# ==========================================
# STRATEGY 1: AMTE BB PULLBACK (LOOSENED FILTER)
# ==========================================
class AMTEBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {pair: {'status': 'NONE'} for pair in pairs}
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
        self.check_daily_reset()
        kill_active = self.daily_trades >= self.max_daily_trades or self.daily_pnl_inr <= self.max_daily_loss
        active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')

        for pair in self.pairs:
            df_macro = fetch_live_data(pair, "1h", 100)
            df_micro = fetch_live_data(pair, "15m", 100)
            if df_macro.empty or df_micro.empty: continue

            df_macro['EMA_20'] = df_macro['close'].ewm(span=20, adjust=False).mean()
            df_macro['EMA_50'] = df_macro['close'].ewm(span=50, adjust=False).mean()
            
            c_macro = df_macro.iloc[-2]
            # UPGRADE: Removed the strict 'spread > 0.05' requirement to catch more pullbacks
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

                # Smart Timeout Check
                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 4:
                    if (floating_usd - floating_fees) > 0:
                        self.close_trade(pair, live_price, "Timeout (Profit)", "TAKER")
                        continue
                
                if not pos.get('be_moved', False):
                    if (pos['side'] == 'LONG' and live_price >= pos['be_trig']) or (pos['side'] == 'SHORT' and live_price <= pos['be_trig']):
                        self.positions[pair]['sl'] = pos['be_sl']
                        self.positions[pair]['be_moved'] = True
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
                self.place_limit(pair, 'LONG', lp, sl, tp, risk_usd / max(0.0001, (lp - sl)), atr)
            elif is_bear and live_price < c_micro['BBU']:
                lp = c_micro['BBU']
                sl, tp = lp + (atr * 2.0), lp - (atr * 4.0)
                self.place_limit(pair, 'SHORT', lp, sl, tp, risk_usd / max(0.0001, (sl - lp)), atr)

    def place_limit(self, pair, side, lp, sl, tp, size, atr):
        be_trig = lp + (atr * 2.0) if side == 'LONG' else lp - (atr * 2.0)
        cost_buf = lp * ((MAKER_FEE * 2) + TAKER_FEE)
        self.positions[pair] = {
            'status': 'PENDING_ENTRY', 'side': side, 'limit_price': lp, 'sl': sl, 'tp': tp, 
            'size': size, 'be_trig': be_trig, 'be_sl': lp + cost_buf if side == 'LONG' else lp - cost_buf, 
            'be_moved': False, 'max_dd_inr': 0.0
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
        
        log_trade(AMTE_LEDGER, {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Strategy': 'AMTE', 'Side': pos['side'],
            'Entry': round(ep, 2), 'Exit': round(exit_p, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Friction_INR': round(fees, 2),
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        })
        self.positions[pair] = {'status': 'NONE'}

# ==========================================
# STRATEGY 2: TW ORIGINAL (FASTER EMA)
# ==========================================
class TWAllInOneBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {pair: {'status': 'NONE'} for pair in pairs}
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
            df = fetch_live_data(pair, "15m", 120)
            if df.empty or len(df) < 55: continue

            # UPGRADE: Changed from EMA100 to EMA50 for faster Indian Equity confirmation
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['MHULL'] = calculate_ehma(df['close'], length=16)
            df['SHULL'] = df['MHULL'].shift(2)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()

            c_prev = df.iloc[-3]
            c_curr = df.iloc[-2]
            live_price = df.iloc[-1]['close']

            # Use EMA50 instead of EMA100
            buy_signal = (c_prev['SHULL'] >= c_prev['MHULL']) and (c_curr['SHULL'] < c_curr['MHULL']) and (c_curr['close'] > c_curr['EMA50'])
            sell_signal = (c_prev['SHULL'] <= c_prev['MHULL']) and (c_curr['SHULL'] > c_curr['MHULL']) and (c_curr['close'] < c_curr['EMA50'])

            pos = self.positions[pair]

            if pos['status'] == 'ACTIVE':
                ep = pos['limit_price']
                floating_usd = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                floating_fees = (ep * pos['size'] * MAKER_FEE) + (live_price * pos['size'] * TAKER_FEE)
                self.positions[pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), (floating_usd - floating_fees) * USDT_INR_RATE)

                # Smart Timeout Check
                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 6:
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
                if df.iloc[-1]['low'] <= pos['limit_price'] if pos['side'] == 'LONG' else df.iloc[-1]['high'] >= pos['limit_price']:
                    self.activate_trade(pair)
                continue

            if kill_active or active_count >= self.max_concurrent: continue

            p_high, p_low, q_high, q_low = calculate_pivots(df)
            atr = c_curr['ATR']
            risk_usd = RISK_PER_TRADE_INR / USDT_INR_RATE

            if buy_signal:
                sl = q_low if (not np.isnan(q_low) and q_low < live_price) else (live_price - atr * 1.8)
                tp = p_high if (not np.isnan(p_high) and p_high > live_price) else (live_price + atr * 3.6)
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'LONG', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': risk_usd / max(0.0001, (live_price - sl)), 'max_dd_inr': 0.0}
            elif sell_signal:
                sl = q_high if (not np.isnan(q_high) and q_high > live_price) else (live_price + atr * 1.8)
                tp = p_low if (not np.isnan(p_low) and p_low < live_price) else (live_price - atr * 3.6)
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'SHORT', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': risk_usd / max(0.0001, (sl - live_price)), 'max_dd_inr': 0.0}

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

        log_trade(TW_LEDGER, {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Strategy': 'TW_ORIG', 'Side': pos['side'],
            'Entry': round(ep, 2), 'Exit': round(exit_p, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Friction_INR': round(fees, 2),
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        })
        self.positions[pair] = {'status': 'NONE'}

# ==========================================
# STRATEGY 3: TW FINE-TUNED (FASTER EMA)
# ==========================================
class TWTunedBot:
    def __init__(self, pairs):
        self.pairs = pairs
        self.positions = {pair: {'status': 'NONE'} for pair in pairs}
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
            df = fetch_live_data(pair, "15m", 120)
            if df.empty or len(df) < 55: continue

            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['MHULL'] = calculate_ehma(df['close'], length=16)
            df['SHULL'] = df['MHULL'].shift(3) 
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            df['ATR_50'] = df['ATR'].rolling(50).mean()

            c_prev = df.iloc[-3]
            c_curr = df.iloc[-2]
            live_price = df.iloc[-1]['close']

            buy_signal = (c_prev['SHULL'] >= c_prev['MHULL']) and (c_curr['SHULL'] < c_curr['MHULL']) and \
                         (c_curr['close'] > c_curr['EMA50']) and (c_curr['close'] > c_curr['MHULL']) and \
                         (c_curr['ATR'] > c_curr['ATR_50'])
                         
            sell_signal = (c_prev['SHULL'] <= c_prev['MHULL']) and (c_curr['SHULL'] > c_curr['MHULL']) and \
                          (c_curr['close'] < c_curr['EMA50']) and (c_curr['close'] < c_curr['MHULL']) and \
                          (c_curr['ATR'] > c_curr['ATR_50'])

            pos = self.positions[pair]

            if pos['status'] == 'ACTIVE':
                ep = pos['limit_price']
                floating_usd = (live_price - ep) * pos['size'] if pos['side'] == 'LONG' else (ep - live_price) * pos['size']
                floating_fees = (ep * pos['size'] * MAKER_FEE) + (live_price * pos['size'] * TAKER_FEE)
                self.positions[pair]['max_dd_inr'] = min(pos.get('max_dd_inr', 0.0), (floating_usd - floating_fees) * USDT_INR_RATE)

                if (datetime.now() - pos['entry_time']).total_seconds() / 3600 >= 6:
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
                if df.iloc[-1]['low'] <= pos['limit_price'] if pos['side'] == 'LONG' else df.iloc[-1]['high'] >= pos['limit_price']:
                    self.activate_trade(pair)
                continue

            if kill_active or active_count >= self.max_concurrent: continue

            p_high, p_low, q_high, q_low = calculate_pivots(df)
            atr = c_curr['ATR']
            risk_usd = RISK_PER_TRADE_INR / USDT_INR_RATE

            if buy_signal:
                sl = q_low if (not np.isnan(q_low) and q_low < live_price) else (live_price - atr * 1.8)
                tp = p_high if (not np.isnan(p_high) and p_high > live_price) else (live_price + atr * 3.6)
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'LONG', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': risk_usd / max(0.0001, (live_price - sl)), 'max_dd_inr': 0.0}
            elif sell_signal:
                sl = q_high if (not np.isnan(q_high) and q_high > live_price) else (live_price + atr * 1.8)
                tp = p_low if (not np.isnan(p_low) and p_low < live_price) else (live_price - atr * 3.6)
                self.positions[pair] = {'status': 'PENDING_ENTRY', 'side': 'SHORT', 'limit_price': live_price, 'sl': sl, 'tp': tp, 'size': risk_usd / max(0.0001, (sl - live_price)), 'max_dd_inr': 0.0}

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

        log_trade(TW_TUNED_LEDGER, {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Stock': pair, 'Strategy': 'TW_TUNED', 'Side': pos['side'],
            'Entry': round(ep, 2), 'Exit': round(exit_p, 2), 'Reason': reason,
            'Gross_INR': round(gross, 2), 'Friction_INR': round(fees, 2),
            'Net_PnL_INR': round(net_inr, 2), 'Max_DD_INR': round(pos.get('max_dd_inr', 0.0), 2)
        })
        self.positions[pair] = {'status': 'NONE'}

# ==========================================
# MASTER CONTINUOUS ENGINE RUNNER
# ==========================================
class MasterEngine:
    def __init__(self):
        self.amte = AMTEBot(PAIRS)
        self.tw_orig = TWAllInOneBot(PAIRS)
        self.tw_tuned = TWTunedBot(PAIRS)

    def loop(self):
        # Keep background loop running continuously
        while True:
            # Check if within Indian market hours (9:15 AM to 3:30 PM IST)
            current_time = datetime.now().time()
            if current_time >= datetime.strptime("09:15", "%H:%M").time() and current_time <= datetime.strptime("15:30", "%H:%M").time():
                try:
                    self.amte.process_cycle()
                    self.tw_orig.process_cycle()
                    self.tw_tuned.process_cycle()
                except Exception:
                    pass
            time.sleep(60)

@st.cache_resource
def start_master_engine():
    engine = MasterEngine()
    t = threading.Thread(target=engine.loop, daemon=True)
    t.start()
    return engine

master = start_master_engine()

# ==========================================
# STREAMLIT UI: 3 FULLY WORKING TABS
# ==========================================
st.set_page_config(page_title="Indian Equities Terminal", layout="wide")
st.title("⚡ Indian Equities Multi-Model Terminal")
st.caption("Engine is actively running in the background. Data provided securely by Yahoo Finance.")

tab1, tab2, tab3 = st.tabs(["⚡ Strategy A: AMTE BB Pullback", "🎯 Strategy B: TW Original", "🛠️ Strategy C: TW Fine-Tuned"])

# ------------------------------------------
# TAB 1: AMTE BB PULLBACK
# ------------------------------------------
with tab1:
    df_amte_led = pd.read_csv(AMTE_LEDGER) if os.path.exists(AMTE_LEDGER) else pd.DataFrame()
    net_amte = df_amte_led['Net_PnL_INR'].sum() if not df_amte_led.empty and 'Net_PnL_INR' in df_amte_led.columns else 0.0
    active_amte = sum(1 for p in master.amte.positions.values() if p['status'] == 'ACTIVE')
    pending_amte = sum(1 for p in master.amte.positions.values() if p['status'] == 'PENDING_ENTRY')
    
    win_rate_a, wins_a, losses_a = 0.0, 0, 0
    if not df_amte_led.empty and 'Net_PnL_INR' in df_amte_led.columns:
        wins_a = len(df_amte_led[df_amte_led['Net_PnL_INR'] > 0])
        losses_a = len(df_amte_led[df_amte_led['Net_PnL_INR'] <= 0])
        total_a = wins_a + losses_a
        win_rate_a = (wins_a / total_a * 100) if total_a > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_amte):,.2f}", f"₹{net_amte:,.2f}")
    c2.metric("Market Exposure", f"{active_amte} Active / {pending_amte} Pending")
    c3.metric("Today's Trades", f"{master.amte.daily_trades} / {master.amte.max_daily_trades}")
    c4.metric("Today's PnL", f"₹{master.amte.daily_pnl_inr:,.2f}")
    c5.metric("Win Rate (All-Time)", f"{win_rate_a:.1f}%", f"{wins_a}W / {losses_a}L")

    live_trades_a = []
    for p, pos in master.amte.positions.items():
        if pos['status'] != 'NONE':
            live_trades_a.append({
                "Stock": p, "Status": pos['status'], "Side": pos.get('side', '-'),
                "Target Entry": f"₹{pos.get('limit_price', 0):,.2f}",
                "Take Profit": f"₹{pos.get('tp', 0):,.2f}",
                "Stop Loss": f"₹{pos.get('sl', 0):,.2f}"
            })
    if live_trades_a:
        st.write("🟢 **Currently Running & Pending Trades**")
        st.dataframe(pd.DataFrame(live_trades_a), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 15m Bollinger Band Pullback Routing")
    pair_a = st.selectbox("Select Asset Pair (Strategy A):", PAIRS, key="pair_a")
    df_chart_a = fetch_live_data(pair_a, "15m", 100)
    
    if not df_chart_a.empty:
        df_chart_a['SMA20'] = df_chart_a['close'].rolling(20).mean()
        df_chart_a['STD20'] = df_chart_a['close'].rolling(20).std()
        df_chart_a['BBL'] = df_chart_a['SMA20'] - (df_chart_a['STD20'] * 1.5)
        df_chart_a['BBU'] = df_chart_a['SMA20'] + (df_chart_a['STD20'] * 1.5)

        fig_a = go.Figure()
        fig_a.add_trace(go.Candlestick(x=df_chart_a.index, open=df_chart_a['open'], high=df_chart_a['high'], low=df_chart_a['low'], close=df_chart_a['close'], name="15m Candles"))
        fig_a.add_trace(go.Scatter(x=df_chart_a.index, y=df_chart_a['BBU'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1), name="Upper BB"))
        fig_a.add_trace(go.Scatter(x=df_chart_a.index, y=df_chart_a['BBL'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1), fill='tonexty', fillcolor='rgba(100, 100, 255, 0.05)', name="Lower BB"))
        
        pos_a = master.amte.positions[pair_a]
        if pos_a['status'] == 'PENDING_ENTRY':
            fig_a.add_hline(y=pos_a['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text=f"Limit: ₹{pos_a['limit_price']:,.2f}")
        elif pos_a['status'] == 'ACTIVE':
            fig_a.add_hline(y=pos_a['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text=f"Entry: ₹{pos_a['limit_price']:,.2f}")
            fig_a.add_hline(y=pos_a['tp'], line_dash="dash", line_color="#00E676", annotation_text=f"TP: ₹{pos_a['tp']:,.2f}")
            fig_a.add_hline(y=pos_a['sl'], line_dash="dash", line_color="#FF5252", annotation_text=f"SL: ₹{pos_a['sl']:,.2f}")

        fig_a.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_a, use_container_width=True)

    if not df_amte_led.empty:
        st.download_button(label="📥 Download AMTE Ledger", data=df_amte_led.to_csv(index=False).encode('utf-8'), file_name='indian_amte_ledger.csv', mime='text/csv')
        st.dataframe(df_amte_led.sort_index(ascending=False), use_container_width=True)

# ------------------------------------------
# TAB 2: TW ALL-IN-ONE (ORIGINAL)
# ------------------------------------------
with tab2:
    df_tw_led = pd.read_csv(TW_LEDGER) if os.path.exists(TW_LEDGER) else pd.DataFrame()
    net_tw = df_tw_led['Net_PnL_INR'].sum() if not df_tw_led.empty and 'Net_PnL_INR' in df_tw_led.columns else 0.0
    active_tw = sum(1 for p in master.tw_orig.positions.values() if p['status'] == 'ACTIVE')
    pending_tw = sum(1 for p in master.tw_orig.positions.values() if p['status'] == 'PENDING_ENTRY')
    
    win_rate_b, wins_b, losses_b = 0.0, 0, 0
    if not df_tw_led.empty and 'Net_PnL_INR' in df_tw_led.columns:
        wins_b = len(df_tw_led[df_tw_led['Net_PnL_INR'] > 0])
        losses_b = len(df_tw_led[df_tw_led['Net_PnL_INR'] <= 0])
        total_b = wins_b + losses_b
        win_rate_b = (wins_b / total_b * 100) if total_b > 0 else 0.0

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_tw):,.2f}", f"₹{net_tw:,.2f}")
    t2.metric("Market Exposure", f"{active_tw} Active / {pending_tw} Pending")
    t3.metric("Today's Trades", f"{master.tw_orig.daily_trades} / {master.tw_orig.max_daily_trades}")
    t4.metric("Today's PnL", f"₹{master.tw_orig.daily_pnl_inr:,.2f}")
    t5.metric("Win Rate (All-Time)", f"{win_rate_b:.1f}%", f"{wins_b}W / {losses_b}L")

    live_trades_b = []
    for p, pos in master.tw_orig.positions.items():
        if pos['status'] != 'NONE':
            live_trades_b.append({
                "Stock": p, "Status": pos['status'], "Side": pos.get('side', '-'),
                "Target Entry": f"₹{pos.get('limit_price', 0):,.2f}",
                "Take Profit": f"₹{pos.get('tp', 0):,.2f}",
                "Stop Loss": f"₹{pos.get('sl', 0):,.2f}"
            })
    if live_trades_b:
        st.write("🟢 **Currently Running & Pending Trades**")
        st.dataframe(pd.DataFrame(live_trades_b), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 TW Original (2-Bar Lag)")
    pair_b = st.selectbox("Select Asset Pair (Strategy B):", PAIRS, key="pair_b")
    df_chart_b = fetch_live_data(pair_b, "15m", 120)

    if not df_chart_b.empty and len(df_chart_b) >= 55:
        df_chart_b['EMA50'] = df_chart_b['close'].ewm(span=50, adjust=False).mean()
        df_chart_b['MHULL'] = calculate_ehma(df_chart_b['close'], 16)
        df_chart_b['SHULL'] = df_chart_b['MHULL'].shift(2)

        fig_b = go.Figure()
        fig_b.add_trace(go.Candlestick(x=df_chart_b.index, open=df_chart_b['open'], high=df_chart_b['high'], low=df_chart_b['low'], close=df_chart_b['close'], name="15m Candles"))
        fig_b.add_trace(go.Scatter(x=df_chart_b.index, y=df_chart_b['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
        fig_b.add_trace(go.Scatter(x=df_chart_b.index, y=df_chart_b['SHULL'], line=dict(color='#FF4B4B', width=1.5, dash='dot'), name="SHULL (2 Lag)"))
        fig_b.add_trace(go.Scatter(x=df_chart_b.index, y=df_chart_b['EMA50'], line=dict(color='#9C27B0', width=2), name="EMA 50"))

        pos_b = master.tw_orig.positions[pair_b]
        if pos_b['status'] == 'PENDING_ENTRY': fig_b.add_hline(y=pos_b['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text="Limit")
        elif pos_b['status'] == 'ACTIVE':
            fig_b.add_hline(y=pos_b['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text="Entry")
            fig_b.add_hline(y=pos_b['tp'], line_dash="dash", line_color="#00E676", annotation_text="TP")
            fig_b.add_hline(y=pos_b['sl'], line_dash="dash", line_color="#FF5252", annotation_text="SL")

        fig_b.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    if not df_tw_led.empty:
        st.download_button(label="📥 Download TW Orig Ledger", data=df_tw_led.to_csv(index=False).encode('utf-8'), file_name='indian_tw_orig_ledger.csv', mime='text/csv')
        st.dataframe(df_tw_led.sort_index(ascending=False), use_container_width=True)

# ------------------------------------------
# TAB 3: TW FINE-TUNED (STRATEGY C)
# ------------------------------------------
with tab3:
    df_tuned_led = pd.read_csv(TW_TUNED_LEDGER) if os.path.exists(TW_TUNED_LEDGER) else pd.DataFrame()
    net_tuned = df_tuned_led['Net_PnL_INR'].sum() if not df_tuned_led.empty and 'Net_PnL_INR' in df_tuned_led.columns else 0.0
    active_tuned = sum(1 for p in master.tw_tuned.positions.values() if p['status'] == 'ACTIVE')
    pending_tuned = sum(1 for p in master.tw_tuned.positions.values() if p['status'] == 'PENDING_ENTRY')

    win_rate_c, wins_c, losses_c = 0.0, 0, 0
    if not df_tuned_led.empty and 'Net_PnL_INR' in df_tuned_led.columns:
        wins_c = len(df_tuned_led[df_tuned_led['Net_PnL_INR'] > 0])
        losses_c = len(df_tuned_led[df_tuned_led['Net_PnL_INR'] <= 0])
        total_c = wins_c + losses_c
        win_rate_c = (wins_c / total_c * 100) if total_c > 0 else 0.0

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Capital Balance", f"₹{(CAPITAL_INR + net_tuned):,.2f}", f"₹{net_tuned:,.2f}")
    f2.metric("Market Exposure", f"{active_tuned} Active / {pending_tuned} Pending")
    f3.metric("Today's Trades", f"{master.tw_tuned.daily_trades} / {master.tw_tuned.max_daily_trades}")
    f4.metric("Today's PnL", f"₹{master.tw_tuned.daily_pnl_inr:,.2f}")
    f5.metric("Win Rate (All-Time)", f"{win_rate_c:.1f}%", f"{wins_c}W / {losses_c}L")

    live_trades_c = []
    for p, pos in master.tw_tuned.positions.items():
        if pos['status'] != 'NONE':
            live_trades_c.append({
                "Stock": p, "Status": pos['status'], "Side": pos.get('side', '-'),
                "Target Entry": f"₹{pos.get('limit_price', 0):,.2f}",
                "Take Profit": f"₹{pos.get('tp', 0):,.2f}",
                "Stop Loss": f"₹{pos.get('sl', 0):,.2f}"
            })
    if live_trades_c:
        st.write("🟢 **Currently Running & Pending Trades**")
        st.dataframe(pd.DataFrame(live_trades_c), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 TW Fine-Tuned (3-Bar Lag + ATR Volatility Filter)")
    pair_c = st.selectbox("Select Asset Pair (Strategy C):", PAIRS, key="pair_c")
    df_chart_c = fetch_live_data(pair_c, "15m", 120)

    if not df_chart_c.empty and len(df_chart_c) >= 55:
        df_chart_c['EMA50'] = df_chart_c['close'].ewm(span=50, adjust=False).mean()
        df_chart_c['MHULL'] = calculate_ehma(df_chart_c['close'], 16)
        df_chart_c['SHULL'] = df_chart_c['MHULL'].shift(3) 

        fig_c = go.Figure()
        fig_c.add_trace(go.Candlestick(x=df_chart_c.index, open=df_chart_c['open'], high=df_chart_c['high'], low=df_chart_c['low'], close=df_chart_c['close'], name="15m Candles"))
        fig_c.add_trace(go.Scatter(x=df_chart_c.index, y=df_chart_c['MHULL'], line=dict(color='#0018F3', width=2), name="MHULL"))
        fig_c.add_trace(go.Scatter(x=df_chart_c.index, y=df_chart_c['SHULL'], line=dict(color='#00E676', width=1.5, dash='dot'), name="SHULL (3 Lag)"))
        fig_c.add_trace(go.Scatter(x=df_chart_c.index, y=df_chart_c['EMA50'], line=dict(color='#9C27B0', width=2), name="EMA 50"))

        pos_c = master.tw_tuned.positions[pair_c]
        if pos_c['status'] == 'PENDING_ENTRY': fig_c.add_hline(y=pos_c['limit_price'], line_dash="dot", line_color="#BDBDBD", annotation_text="Limit")
        elif pos_c['status'] == 'ACTIVE':
            fig_c.add_hline(y=pos_c['limit_price'], line_dash="solid", line_color="#FF9800", annotation_text="Entry")
            fig_c.add_hline(y=pos_c['tp'], line_dash="dash", line_color="#00E676", annotation_text="TP")
            fig_c.add_hline(y=pos_c['sl'], line_dash="dash", line_color="#FF5252", annotation_text="SL")

        fig_c.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

    if not df_tuned_led.empty:
        st.download_button(label="📥 Download TW Tuned Ledger", data=df_tuned_led.to_csv(index=False).encode('utf-8'), file_name='indian_tw_tuned_ledger.csv', mime='text/csv')
        st.dataframe(df_tuned_led.sort_index(ascending=False), use_container_width=True)
