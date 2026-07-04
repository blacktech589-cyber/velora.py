# ==============================
# SYSTEM & ERROR HANDLING
# ==============================
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
import warnings
import hashlib
import json
import joblib
from pathlib import Path
warnings.filterwarnings("ignore")

def log_exception(exc_type, exc_value, exc_traceback):
    with open("hata_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

sys.excepthook = log_exception

# ==============================
# CORE LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import requests
from collections import deque

# Optional sentiment
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# Optional Binomo selenium reader
try:
    from binomo_scraper import fetch_binomo_prices
    BINOMO_READER_AVAILABLE = True
except Exception:
    BINOMO_READER_AVAILABLE = False
    fetch_binomo_prices = None

# ==============================
# TIMING & RISK CONFIG
# ==============================
SCAN_INTERVAL_SEC = 45
PREDICTION_LEAD_SEC = 7

PROTECTION_MODE = True
MIN_CONFIDENCE_TO_TRADE = 70
HIGH_VOL_THRESHOLD = 0.060
NEWS_TIMEOUT_SEC = 6
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
PRICE_TIMEOUT_SEC = 8
PRICE_INTERVAL = "1m"
PRICE_LIMIT = 120
USE_BINOMO_SCREEN = os.getenv("USE_BINOMO_SCREEN", "false").lower() == "true"

# ==============================
# CONFIRMATION CONFIG (12 + 4 CANDLE)
# BUY = 12 düşüş + 4 düşüş
# SELL = 12 yükseliş + 4 yükseliş
# ==============================
CONFIRM_WINDOW_FAST = 4
CONFIRM_WINDOW_SLOW = 12
MIN_FAST_COUNT = 4
MIN_SLOW_DOMINANCE = 1.0
RSI_BUY_MAX = 45
RSI_SELL_MIN = 55
ATR_MIN_RATIO = 0.001
ATR_MAX_RATIO = 0.080
EMA_STACK_REQUIRED = False
MACD_CONFIRM_REQUIRED = False

# ==============================
# MODEL STORAGE
# ==============================
MODEL_DIR = Path("velora_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "velora_700f_models.pkl"
SCALER_PATH = MODEL_DIR / "velora_700f_scaler.pkl"
META_PATH = MODEL_DIR / "velora_700f_meta.json"

# ==============================
# MARKET DATA CONFIG
# ==============================
BINANCE_SYMBOLS = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "Cardano": "ADAUSDT",
    "Solana": "SOLUSDT",
    "Chainlink": "LINKUSDT",
    "Bitcoin Cash": "BCHUSDT",
    "Kusama": "KSMUSDT",
    "Toncoin": "TONUSDT",
    "Aave": "AAVEUSDT",
    "Pancake Swap": "CAKEUSDT",
    "Uniswap": "UNIUSDT",
    "Crypto IDX": "BTCUSDT",
}
YAHOO_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/CHF": "GBPCHF=X",
    "AUD/CAD": "AUDCAD=X",
    "GBP/NZD": "GBPNZD=X",
    "CHF/JPY": "CHFJPY=X",
    "Nvidia": "NVDA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOG",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Yum Brands": "YUM",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "DAX40": "^GDAXI",
}

# ==============================
# SKLEARN IMPORTS
# ==============================
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ==============================
# SAFE XGBOOST & LGBM IMPORT
# ==============================
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
XGBClassifier = None
LGBMClassifier = None

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# ==============================
# SIGNAL GENERATOR / MARKET DATA
# ==============================
class SignalGenerator:
    def __init__(self, asset, time_seed=None):
        self.asset = asset
        self.time_seed = time_seed or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        np.random.seed(int(hashlib.md5(f"{asset}{self.time_seed}".encode()).hexdigest(), 16) % 2**32)

    def _fetch_binomo_screen_prices(self, limit=PRICE_LIMIT):
        if not USE_BINOMO_SCREEN or not BINOMO_READER_AVAILABLE or fetch_binomo_prices is None:
            return None, None
        try:
            result = fetch_binomo_prices(self.asset, limit=limit)
            if result and getattr(result, "prices", None):
                prices = np.array(result.prices, dtype=np.float64)
                if len(prices) >= 20:
                    return prices, result.source
        except Exception:
            return None, None
        return None, None

    def _fetch_binance_prices(self, symbol, interval=PRICE_INTERVAL, limit=PRICE_LIMIT):
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=PRICE_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            data = r.json()
            closes = [float(row[4]) for row in data if len(row) > 4]
            if len(closes) >= 30:
                return np.array(closes, dtype=np.float64)
        except Exception:
            return None
        return None

    def _fetch_yahoo_prices(self, symbol, interval=PRICE_INTERVAL, limit=PRICE_LIMIT):
        try:
            range_map = {"1m": "1d", "2m": "1d", "5m": "5d", "15m": "5d", "30m": "1mo", "60m": "1mo"}
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "interval": interval,
                "range": range_map.get(interval, "1d"),
                "includePrePost": "false",
                "events": "div,splits",
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=PRICE_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            data = r.json()
            result = (data.get("chart", {}) or {}).get("result", [])
            if not result:
                return None
            quote = (((result[0].get("indicators", {}) or {}).get("quote", [])) or [{}])[0]
            closes = quote.get("close", []) or []
            closes = [float(x) for x in closes if x is not None]
            if len(closes) >= 30:
                return np.array(closes[-limit:], dtype=np.float64)
        except Exception:
            return None
        return None

    def fetch_live_prices(self, length=PRICE_LIMIT):
        prices, source = self._fetch_binomo_screen_prices(limit=length)
        if prices is not None:
            return prices, source

        if self.asset in BINANCE_SYMBOLS:
            prices = self._fetch_binance_prices(BINANCE_SYMBOLS[self.asset], limit=length)
            if prices is not None:
                return prices, f"Binance {PRICE_INTERVAL}"

        if self.asset in YAHOO_SYMBOLS:
            prices = self._fetch_yahoo_prices(YAHOO_SYMBOLS[self.asset], limit=length)
            if prices is not None:
                return prices, f"Yahoo Finance {PRICE_INTERVAL}"

        return self.generate_realistic_prices(length), "Simulated"

    def generate_realistic_prices(self, length=120):
        if "EUR" in self.asset or "GBP" in self.asset or "USD" in self.asset:
            base = np.random.uniform(0.8, 2.0)
        elif any(x in self.asset for x in ["Bitcoin", "Ethereum"]):
            base = np.random.uniform(30000, 70000)
        elif any(x in self.asset for x in ["Gold", "Silver"]):
            base = np.random.uniform(1500, 2500)
        else:
            base = np.random.uniform(50, 500)

        mu = np.random.uniform(-0.005, 0.005)
        sigma = np.random.uniform(0.01, 0.08)
        dt = 1 / length

        price = base
        prices = [price]
        for _ in range(length - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            prices.append(price)

        return np.array(prices, dtype=np.float64)

    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0:
            return 100.0 if up > 0 else 0.0
        rs = up / down
        return float(100.0 - 100.0 / (1.0 + rs))

    def calculate_ema(self, prices, period=15):
        return float(pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1])

    def calculate_atr(self, prices, period=14):
        if len(prices) < period:
            d = np.diff(prices)
            return float(np.mean(np.abs(d))) if len(d) > 0 else 0.01
        trs = np.abs(np.diff(prices))
        return float(max(np.mean(trs[-period:]), 0.0001))

    def calculate_macd(self, prices):
        series = pd.Series(prices)
        ema12 = series.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = series.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26
        signal = series.ewm(span=9, adjust=False).mean().iloc[-1]
        hist = macd - signal
        return float(macd), float(signal), float(hist)

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        series = pd.Series(prices)
        sma = series.rolling(window=period).mean().iloc[-1]
        std = series.rolling(window=period).std().iloc[-1]
        if pd.isna(sma) or pd.isna(std):
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        pos = (prices[-1] - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        return float(sma), float(upper), float(lower), float(pos)

    def calculate_ultra_400_features(self, prices):
        eps = 1e-9
        p = np.array(prices, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        if len(p) < 120:
            pad = np.full(120 - len(p), p[-1] if len(p) else 1.0)
            p = np.concatenate([pad, p])

        rets = np.diff(p) / np.clip(p[:-1], eps, np.inf)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        feats = []

        rsi_periods = [2, 3, 5, 7, 9, 11, 14, 18, 21, 28, 35, 42, 50]
        ema_periods = [3, 5, 7, 9, 12, 15, 18, 21, 26, 30, 34, 40, 50, 60]
        atr_periods = [5, 7, 10, 14, 21, 28]

        rsi_vals = []
        for rp in rsi_periods:
            r = self.calculate_rsi(p, rp)
            rsi_vals.append(r)
            feats.append((r - 50.0) / 50.0)
            feats.append(r / 100.0)

        ema_vals = []
        for ep in ema_periods:
            e = self.calculate_ema(p, ep)
            ema_vals.append(e)
            feats.append((p[-1] - e) / (e + eps))

        for ap in atr_periods:
            a = self.calculate_atr(p, ap)
            feats.append(a / (p[-1] + eps))

        for i in range(len(rsi_vals) - 1):
            feats.append((rsi_vals[i] - rsi_vals[i + 1]) / 100.0)

        for i in range(len(ema_vals) - 1):
            feats.append((ema_vals[i] - ema_vals[i + 1]) / (ema_vals[i + 1] + eps))

        windows = [3, 5, 7, 9, 12, 15, 18, 21, 24, 28, 32, 36, 42, 50, 60, 72, 84, 96]
        for w in windows:
            seg = p[-w:]
            rseg = rets[-(w - 1):] if w > 1 else np.array([0.0])

            high = np.max(seg); low = np.min(seg); mean = np.mean(seg); std = np.std(seg)
            rng = high - low
            pos = (seg[-1] - low) / (rng + eps)

            feats.extend([
                (seg[-1] - seg[0]) / (seg[0] + eps),
                std / (mean + eps),
                pos,
                (high - seg[-1]) / (high + eps),
                (seg[-1] - low) / (low + eps),
                np.mean(rseg),
                np.std(rseg),
                np.max(rseg) if len(rseg) else 0.0,
                np.min(rseg) if len(rseg) else 0.0,
                np.median(rseg),
            ])

            for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                feats.append(np.quantile(rseg, q) if len(rseg) else 0.0)

            for lag in [1, 2, 3, 5, 8, 13]:
                if len(seg) > lag:
                    feats.append((seg[-1] - seg[-1 - lag]) / (seg[-1 - lag] + eps))
                else:
                    feats.append(0.0)

        macd, macd_sig, macd_hist = self.calculate_macd(p)
        feats.extend([macd / (p[-1] + eps), macd_sig / (p[-1] + eps), macd_hist / (p[-1] + eps)])

        sma20, up20, lo20, bb_pos20 = self.calculate_bollinger_bands(p, 20, 2)
        sma50, up50, lo50, bb_pos50 = self.calculate_bollinger_bands(p, 50, 2)
        feats.extend([
            (p[-1] - sma20) / (sma20 + eps),
            (p[-1] - sma50) / (sma50 + eps),
            bb_pos20, bb_pos50,
            (up20 - lo20) / (sma20 + eps),
            (up50 - lo50) / (sma50 + eps),
        ])

        for w in [7, 14, 21, 28, 35, 42, 50, 60]:
            seg = p[-w:]
            x = np.arange(w)
            try:
                c1 = np.polyfit(x, seg, 1)[0]
            except Exception:
                c1 = 0.0
            feats.append(c1 / (np.mean(seg) + eps))

        sign_rets = np.sign(rets)
        for w in [5, 7, 10, 14, 21, 28, 35]:
            s = sign_rets[-w:] if len(sign_rets) >= w else sign_rets
            if len(s) == 0:
                feats.extend([0.0, 0.0, 0.0])
            else:
                feats.extend([np.mean(s > 0), np.mean(s < 0), np.mean(s == 0)])

        rsi14 = self.calculate_rsi(p, 14)
        ema15 = self.calculate_ema(p, 15)
        ema7 = self.calculate_ema(p, 7)
        ema30 = self.calculate_ema(p, 30)
        atr14 = self.calculate_atr(p, 14)

        core = [
            (rsi14 - 50) / 50,
            (p[-1] - ema15) / (ema15 + eps),
            (ema7 - ema15) / (ema15 + eps),
            (ema15 - ema30) / (ema30 + eps),
            atr14 / (p[-1] + eps),
            macd / (p[-1] + eps),
            macd_hist / (p[-1] + eps),
            bb_pos20
        ]
        for i in range(len(core)):
            for j in range(i, len(core)):
                feats.append(core[i] * core[j])

        feats = np.array(feats, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.1, neginf=-0.1)
        feats = np.clip(feats, -10, 10)

        if len(feats) < 400:
            feats = np.concatenate([feats, np.zeros(400 - len(feats), dtype=np.float32)])
        elif len(feats) > 400:
            feats = feats[:400]

        return feats.astype(np.float32)

    def calculate_ultra_700_features(self, prices):
        f400 = self.calculate_ultra_400_features(prices)
        p = np.array(prices, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-9

        extra = []

        horizons = [2,3,4,5,6,8,10,12,15,18,21,24,30,36,42,50,60,72,84,96,110]
        for h in horizons:
            seg = p[-h:] if len(p) >= h else p
            if len(seg) < 2:
                extra.extend([0.0] * 8)
                continue
            r = np.diff(seg) / np.clip(seg[:-1], eps, np.inf)
            mu = float(np.mean(r))
            sd = float(np.std(r) + eps)
            z_last = float((r[-1] - mu) / sd)
            q10, q50, q90 = np.quantile(r, [0.1, 0.5, 0.9])
            extra.extend([
                float((seg[-1] - seg[0]) / (seg[0] + eps)),
                float(sd),
                float(mu),
                float(z_last),
                float(q10), float(q50), float(q90),
                float((np.max(seg) - np.min(seg)) / (np.mean(seg) + eps)),
            ])

        for lag in [1,2,3,5,8,13,21,34]:
            if len(p) > lag:
                extra.append(float((p[-1] - p[-1-lag]) / (p[-1-lag] + eps)))
                extra.append(float((p[-lag] - p[-1-lag]) / (p[-1-lag] + eps)))
            else:
                extra.extend([0.0, 0.0])

        for w in [10,14,20,28,36,50,70,90,120]:
            seg = p[-w:] if len(p) >= w else p
            if len(seg) < 3:
                extra.extend([0.0, 0.0]); continue
            x = np.arange(len(seg))
            c1 = np.polyfit(x, seg, 1)[0]
            c2 = np.polyfit(x, seg, 2)[0]
            extra.extend([float(c1 / (np.mean(seg) + eps)), float(c2 / (np.mean(seg) + eps))])

        f = np.concatenate([f400, np.array(extra, dtype=np.float32)])
        f = np.nan_to_num(f, nan=0.0, posinf=0.1, neginf=-0.1)
        f = np.clip(f, -10, 10)

        if len(f) < 700:
            f = np.concatenate([f, np.zeros(700 - len(f), dtype=np.float32)])
        elif len(f) > 700:
            f = f[:700]
        return f.astype(np.float32)

# ==============================
# NEWS ANALYZER
# ==============================
class NewsAnalyzer:
    def __init__(self, api_key=""):
        self.api_key = api_key.strip() if api_key else ""
        self.cache = {}

    def _asset_query(self, asset):
        mapping = {
            "Bitcoin": "Bitcoin OR BTC crypto",
            "Ethereum": "Ethereum OR ETH crypto",
            "Gold": "Gold commodity",
            "Silver": "Silver commodity",
            "Oil": "Crude oil WTI Brent",
            "Nvidia": "Nvidia stock",
            "Apple": "Apple stock",
            "Microsoft": "Microsoft stock",
            "Tesla": "Tesla stock",
            "SP500": "S&P 500 index",
            "NASDAQ100": "Nasdaq 100 index",
            "EUR/USD": "EUR USD forex",
            "GBP/USD": "GBP USD forex",
            "USD/JPY": "USD JPY forex",
        }
        return mapping.get(asset, f"{asset} market finance")

    def _fetch_news_newsapi(self, query):
        if not self.api_key:
            return []
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 10, "apiKey": self.api_key}
            r = requests.get(url, params=params, timeout=NEWS_TIMEOUT_SEC)
            if r.status_code != 200:
                return []
            data = r.json()
            arts = data.get("articles", [])
            return [f"{a.get('title','')} {a.get('description','')}" for a in arts]
        except Exception:
            return []
