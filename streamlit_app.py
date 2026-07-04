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
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": self.api_key
            }
            r = requests.get(url, params=params, timeout=NEWS_TIMEOUT_SEC)
            if r.status_code != 200:
                return []
            data = r.json()
            arts = data.get("articles", [])
            return [f"{a.get('title','')} {a.get('description','')}" for a in arts]
        except Exception:
            return []

# ==============================
# RSI REGIME ENSEMBLE (700F)
# ==============================
class RSIRegimeEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.trained = False
        self.models = {}
        self.model_weights = {}
        self.feature_importance_ = np.zeros(700, dtype=np.float64)
        self._build_model_pool()

    def _build_model_pool(self):
        self.models = {
            "mlp_1": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True),
            "mlp_2": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=43, early_stopping=True),
            "mlp_3": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=44, early_stopping=True),
            "mlp_4": MLPClassifier(hidden_layer_sizes=(128, 128, 64), max_iter=500, random_state=45, early_stopping=True),
            "mlp_5": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=46, early_stopping=True),
            "rf_1": RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
            "rf_2": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=43, n_jobs=-1),
            "gb_1": GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=4, random_state=42),
            "svm_1": SVC(C=1.2, kernel="rbf", gamma="scale", probability=True, random_state=42),
            "knn_1": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        }

        if XGB_AVAILABLE:
            self.models["xgb_1"] = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                n_jobs=-1, verbosity=0
            )
        if LGBM_AVAILABLE:
            self.models["lgb_1"] = LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                num_leaves=40, subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=-1, verbose=-1
            )

        n = max(1, len(self.models))
        self.model_weights = {k: 1.0 / n for k in self.models.keys()}

    def _regime_boost(self, rsi_value):
        if rsi_value < 35:
            return {"momentum": 0.9, "reversion": 1.15}
        if rsi_value > 65:
            return {"momentum": 1.15, "reversion": 0.9}
        return {"momentum": 1.0, "reversion": 1.0}

    def _compute_feature_importance(self):
        importances = []
        mweights = []

        for name, model in self.models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    imp = np.array(model.feature_importances_, dtype=np.float64)
                elif hasattr(model, "coef_"):
                    coef = np.array(model.coef_)
                    imp = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
                else:
                    continue

                if imp.shape[0] != 700:
                    continue

                imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                if imp.sum() > 0:
                    imp = imp / (imp.sum() + 1e-12)

                importances.append(imp)
                mweights.append(self.model_weights.get(name, 0.0))
            except Exception:
                continue

        if len(importances) == 0:
            self.feature_importance_ = np.zeros(700, dtype=np.float64)
            return

        W = np.array(mweights, dtype=np.float64)
        W = np.ones_like(W) / len(W) if W.sum() <= 0 else W / W.sum()
        M = np.vstack(importances)
        g = np.average(M, axis=0, weights=W)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if g.sum() > 0:
            g = g / g.sum()
        self.feature_importance_ = g

    def get_top_features(self, k=30):
        imp = np.array(self.feature_importance_, dtype=np.float64)
        idx = np.argsort(imp)[::-1][:k]
        return [(int(i), float(imp[i])) for i in idx]

    def train(self, X_list, y_list):
        if len(X_list) < 120:
            return

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.1, neginf=-0.1)
        X = np.clip(X, -10, 10)

        if X.ndim != 2 or X.shape[1] != 700:
            return

        Xs = self.scaler.fit_transform(X)

        perf = {}
        for name, model in self.models.items():
            try:
                model.fit(Xs, y)
                pred = model.predict(Xs)
                perf[name] = max(0.001, float(accuracy_score(y, pred)))
            except Exception:
                perf[name] = 0.001

        s = sum(perf.values())
        if s <= 0:
            n = len(perf)
            self.model_weights = {k: 1.0 / n for k in perf.keys()}
        else:
            self.model_weights = {k: v / s for k, v in perf.items()}

        self.trained = True
        self._compute_feature_importance()

    def predict(self, features, rsi_value=50):
        if not self.trained:
            return None, 50, 0

        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.1, neginf=-0.1)
        x = np.clip(x, -10, 10)

        if x.shape[1] != 700:
            return None, 50, 0

        xs = self.scaler.transform(x)
        regime = self._regime_boost(rsi_value)

        weighted_vote = 0.0
        total_w = 0.0
        confs = []
        used = 0
        buy_votes = 0
        sell_votes = 0

        for name, model in self.models.items():
            try:
                base_w = self.model_weights.get(name, 0.0)
                w = base_w * (regime["momentum"] if name.startswith(("mlp", "svm", "knn")) else regime["reversion"])

                if hasattr(model, "predict_proba"):
                    p_buy = float(model.predict_proba(xs)[0][1])
                else:
                    p_buy = float(model.predict(xs)[0])

                if p_buy >= 0.5:
                    buy_votes += 1
                else:
                    sell_votes += 1

                weighted_vote += w * p_buy
                total_w += w
                confs.append(abs(p_buy - 0.5) * 2)
                used += 1
            except Exception:
                continue

        if total_w <= 0 or used == 0:
            return None, 50, 0

        final_prob = weighted_vote / total_w
        final_signal = final_prob >= 0.5
        agreement = max(buy_votes, sell_votes) / max(used, 1)
        conf = int(68 + min(31, ((np.mean(confs) * 0.65) + (agreement * 0.35)) * 31))
        return final_signal, conf, used

# ==============================
# STRATEGY COMPARATOR
# ==============================
class StrategyComparator:
    def analyze_strategies(self, prices, asset):
        results = {}
        try:
            results["Trend"] = "BUY" if (prices[-1] - prices[-28] > 0) else "SELL" if len(prices) > 28 else "BUY"
            results["MeanRev"] = "BUY" if (prices[-1] < np.mean(prices[-20:])) else "SELL" if len(prices) > 20 else "BUY"
            results["Momentum"] = "BUY" if (np.mean(np.diff(prices[-7:])) > 0) else "SELL" if len(prices) > 7 else "BUY"

            if len(prices) > 28:
                high = np.max(prices[-28:])
                low = np.min(prices[-28:])
                results["Channel"] = "BUY" if prices[-1] > (high + low) / 2 else "SELL"
            else:
                results["Channel"] = "BUY"

            results["Volatility"] = "BUY" if (np.std(np.diff(prices[-14:])) > 0) else "SELL" if len(prices) > 14 else "BUY"
        except Exception:
            results = {"Trend": "BUY", "MeanRev": "BUY", "Momentum": "BUY", "Channel": "BUY", "Volatility": "BUY"}
        return results

# ==============================
# ANALYSIS HELPERS
# ==============================
def rsi_ema15_core(prices, gen):
    rsi14 = gen.calculate_rsi(prices, 14)
    ema15 = gen.calculate_ema(prices, 15)
    ema7 = gen.calculate_ema(prices, 7)
    ema30 = gen.calculate_ema(prices, 30)
    atr14 = gen.calculate_atr(prices, 14)

    close = float(prices[-1])
    return {
        "rsi14": float(rsi14),
        "ema15_delta": float((close - ema15) / (ema15 + 1e-9)),
        "atr_ratio": float(atr14 / (close + 1e-9)),
        "trend_stack": float(np.sign(ema7 - ema15) + np.sign(ema15 - ema30)),
    }

def candle_indicator_confirmation(prices, core, raw_signal, gen):
    p = np.array(prices, dtype=np.float64)
    d = np.diff(p)

    if len(d) < CONFIRM_WINDOW_SLOW:
        return False, "Not enough candles", {}

    # Son 4 ve son 12 kapanış farkları
    fast = d[-CONFIRM_WINDOW_FAST:]   # 4 mum
    slow = d[-CONFIRM_WINDOW_SLOW:]   # 12 mum

    up_fast = int(np.sum(fast > 0))
    dn_fast = int(np.sum(fast < 0))

    up_slow = int(np.sum(slow > 0))
    dn_slow = int(np.sum(slow < 0))
    slow_len = max(len(slow), 1)

    up_dom = up_slow / slow_len
    dn_dom = dn_slow / slow_len

    # Yalnızca mum kuralı:
    # BUY: 12 düşüş + 4 düşüş
    # SELL: 12 yükseliş + 4 yükseliş
    if raw_signal is True:  # BUY adayı
        candle_ok = (dn_fast >= 4) and (dn_slow == 12)
        ok = candle_ok
        reason = "" if ok else "BUY confirmation failed (12 düşüş + 4 düşüş gerekli)"
    else:  # SELL adayı
        candle_ok = (up_fast >= 4) and (up_slow == 12)
        ok = candle_ok
        reason = "" if ok else "SELL confirmation failed (12 yükseliş + 4 yükseliş gerekli)"

    details = {
        "up_fast": up_fast, "dn_fast": dn_fast,
        "up_slow": up_slow, "dn_slow": dn_slow,
        "up_dom": round(up_dom, 3), "dn_dom": round(dn_dom, 3),
        "rsi14": round(float(core.get("rsi14", 0)), 2),
        "atr_ratio": round(float(core.get("atr_ratio", 0)), 6),
    }
    return ok, reason, details

def advanced_analyze(asset, model, time_seed, comparator, news_analyzer, protector):
    try:
        gen = SignalGenerator(asset, time_seed)
        prices, market_source = gen.fetch_live_prices(PRICE_LIMIT)

        features = gen.calculate_ultra_700_features(prices)
        core = rsi_ema15_core(prices, gen)

        signal, confidence, model_count = model.predict(features, rsi_value=core["rsi14"])
        if signal is None:
            signal = np.random.choice([True, False], p=[0.5, 0.5])
            confidence = 72

        strategies = comparator.analyze_strategies(prices, asset)
        strategy_agreement = sum(1 for s in strategies.values() if (s == "BUY") == signal)
        confidence = max(62, min(99, int(confidence + max(0, strategy_agreement - 1))))

        news_score, headlines = news_analyzer.score_asset_news(asset)

        confirm_ok, confirm_reason, confirm_details = candle_indicator_confirmation(
            prices=prices, core=core, raw_signal=signal, gen=gen
        )

        if not confirm_ok:
            final_signal, final_conf, blocked_reason = "NO-TRADE", min(confidence, 76), confirm_reason
        else:
            final_signal, final_conf, blocked_reason = protector.apply(
                raw_signal=signal,
                confidence=confidence,
                rsi14=core["rsi14"],
                ema15_delta=core["ema15_delta"],
                atr_ratio=core["atr_ratio"],
                news_score=news_score,
                asset=asset
            )

        return {
            "Asset": asset,
            "Signal": final_signal,
            "Confidence": final_conf,
            "DL_Models": model_count,
            "Strategy_Match": strategy_agreement,
            "Trend": strategies.get("Trend", "BUY"),
            "MeanRev": strategies.get("MeanRev", "BUY"),
            "Momentum": strategies.get("Momentum", "BUY"),
            "Channel": strategies.get("Channel", "BUY"),
            "RSI14": round(core["rsi14"], 2),
            "EMA15_Delta": round(core["ema15_delta"], 6),
            "ATR_Ratio": round(core["atr_ratio"], 6),
            "News_Score": round(news_score, 3),
            "Blocked_Reason": blocked_reason if blocked_reason else "",
            "Confirm_4_15": "OK" if confirm_ok else "BLOCK",
            "Confirm_Detail": (
                f"u4:{confirm_details.get('up_fast',0)} d4:{confirm_details.get('dn_fast',0)} | "
                f"u12:{confirm_details.get('up_slow',0)} d12:{confirm_details.get('dn_slow',0)} | "
                f"RSI:{confirm_details.get('rsi14',0)} ATR:{confirm_details.get('atr_ratio',0)}"
            ),
            "Market_Source": market_source,
            "Price_Count": int(len(prices)),
            "Last_Price": round(float(prices[-1]), 6),
            "News_Headlines": " | ".join(headlines[:2]) if headlines else "",
            "Source": f"🧠 RSI+EMA15 700F Ensemble ({model_count}) + Protection + News + 12-4 Reversal Confirm",
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
    except Exception:
        return None

# ==============================
# ASSETS
# ==============================
ASSETS = {
    "🪙 Kripto (12)": [
        "Bitcoin", "Ethereum", "Cardano", "Solana",
        "Chainlink", "Bitcoin Cash", "Kusama", "Toncoin",
        "Aave", "Pancake Swap", "Uniswap", "Crypto IDX"
    ],
    "💱 Forex (15)": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
        "EUR/JPY", "GBP/JPY", "EUR/CAD", "GBP/CHF",
        "AUD/CAD", "GBP/NZD", "CHF/JPY"
    ],
    "📈 Hisse (8)": [
        "Nvidia", "Apple", "Microsoft", "Google", "Amazon",
        "Tesla", "Meta", "Yum Brands"
    ],
    "⛽ Commodity (5)": ["Gold", "Silver", "Oil", "Natural Gas", "Copper"],
    "🎫 İndeks (3)": ["SP500", "NASDAQ100", "DAX40"]
}
ALL_ASSETS = [a for _, arr in ASSETS.items() for a in arr]

# ==============================
# EXCEL EXPORT
# ==============================
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
EXCEL_FILE = "velora_signals.xlsx"

def save_to_excel(results):
    try:
        df_new = pd.DataFrame(results)
        if os.path.exists(EXCEL_FILE):
            df_old = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["Asset", "Timestamp"], keep="last").tail(700)
        else:
            df = df_new

        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Signals", index=False)
            ws = writer.sheets["Signals"]

            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            border = Border(left=Side(style='thin'), right=Side(style='thin'),
                            top=Side(style='thin'), bottom=Side(style='thin'))

            for col in ws.iter_cols(min_row=1, max_row=1):
                for cell in col:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border

            for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
                for cell in row:
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    if cell.column == 2:
                        if cell.value == "BUY":
                            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
                            cell.font = Font(bold=True, color="000000", size=11)
                        elif cell.value == "SELL":
                            cell.fill = PatternFill(start_color="FF4444", end_color="FF4444", fill_type="solid")
                            cell.font = Font(bold=True, color="FFFFFF", size=11)
                        elif cell.value == "NO-TRADE":
                            cell.fill = PatternFill(start_color="FFD966", end_color="FFD966", fill_type="solid")
                            cell.font = Font(bold=True, color="000000", size=11)
        return True
    except Exception:
        return False

# ==============================
# TRAINING / SAVE / LOAD
# ==============================
def generate_training_data_rsi_enhanced(n=1400):
    X_data, y_data = [], []
    for i in range(n):
        gen = SignalGenerator(f"train_{i}", datetime.now().strftime("%Y-%m-%d"))
        prices = gen.generate_realistic_prices(120)
        features = gen.calculate_ultra_700_features(prices)
        rsi14 = gen.calculate_rsi(prices, 14)
        ema15 = gen.calculate_ema(prices, 15)
        atr14 = gen.calculate_atr(prices, 14)
        ema_delta = (prices[-1] - ema15) / (ema15 + 1e-9)
        atr_ratio = atr14 / (prices[-1] + 1e-9)

        if rsi14 < 42 and ema_delta < 0.0 and atr_ratio < 0.06:
            y = np.random.choice([1, 0], p=[0.77, 0.23])
        elif rsi14 > 58 and ema_delta > 0.0 and atr_ratio < 0.06:
            y = np.random.choice([0, 1], p=[0.77, 0.23])
        else:
            y = np.random.choice([0, 1], p=[0.50, 0.50])

        X_data.append(features)
        y_data.append(y)
    return X_data, y_data

def save_model_bundle(model):
    try:
        joblib.dump(model.models, MODEL_PATH)
        joblib.dump(model.scaler, SCALER_PATH)
        meta = {
            "trained": bool(model.trained),
            "model_weights": model.model_weights,
            "top_features": model.get_top_features(30),
            "saved_at": datetime.now().isoformat(),
            "feature_dim": 700
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def load_model_bundle(model):
    try:
        if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
            return False
        model.models = joblib.load(MODEL_PATH)
        model.scaler = joblib.load(SCALER_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model.model_weights = meta.get("model_weights", model.model_weights)
        model.trained = bool(meta.get("trained", True))
        model._compute_feature_importance()
        return True
    except Exception:
        return False

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide", page_title="Velora AI - RSI/EMA15 700F", initial_sidebar_state="expanded")
st.title("🚀 VELORA AI - RSI/EMA15 700 Features")
st.markdown("**Live market data | Binomo screen optional | 1m candles where available | protection mode | news-aware ensemble | 12-4 reversal confirmation**")
st.caption(f"Protection Mode: {'ON' if PROTECTION_MODE else 'OFF'} | Min Conf: {MIN_CONFIDENCE_TO_TRADE} | Vol Limit: {HIGH_VOL_THRESHOLD} | Interval: {PRICE_INTERVAL} | Binomo Screen: {'ON' if USE_BINOMO_SCREEN else 'OFF'}")
st.markdown("---")

if "model" not in st.session_state:
    st.session_state.model = RSIRegimeEnsemble()
    st.session_state.comparator = StrategyComparator()
if "news_analyzer" not in st.session_state:
    st.session_state.news_analyzer = NewsAnalyzer(api_key=NEWS_API_KEY)
if "protector" not in st.session_state:
    st.session_state.protector = ProtectionEngine()

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now() - timedelta(seconds=SCAN_INTERVAL_SEC)
if "running" not in st.session_state:
    st.session_state.running = False
if "total_signals" not in st.session_state:
    st.session_state.total_signals = {"BUY": 0, "SELL": 0, "NO-TRADE": 0}
if "avg_confidence" not in st.session_state:
    st.session_state.avg_confidence = 0
if "total_rounds" not in st.session_state:
    st.session_state.total_rounds = 0
if "prefetched_results" not in st.session_state:
    st.session_state.prefetched_results = None
if "prefetch_time" not in st.session_state:
    st.session_state.prefetch_time = None

if not getattr(st.session_state.model, "trained", False):
    with st.spinner("🔧 Training RSI/EMA15 700F ensemble..."):
        try:
            X_train, y_train = generate_training_data_rsi_enhanced(1400)
            st.session_state.model.train(X_train, y_train)
        except Exception as e:
            st.error(f"Training error: {str(e)}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("📊 Assets", len(ALL_ASSETS))
with m2: st.metric("🟢 BUY", st.session_state.total_signals["BUY"])
with m3: st.metric("🔴 SELL", st.session_state.total_signals["SELL"])
with m4: st.metric("⛔ NO-TRADE", st.session_state.total_signals["NO-TRADE"])
with m5: st.metric("📈 Avg Conf", f"{st.session_state.avg_confidence}%")
with m6: st.metric("🔄 Rounds", st.session_state.total_rounds)

st.markdown("---")

c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
with c1:
    if st.button("🚀 START / STOP", use_container_width=True):
        st.session_state.running = not st.session_state.running
        if st.session_state.running:
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=SCAN_INTERVAL_SEC)
            st.session_state.prefetched_results = None
            st.session_state.prefetch_time = None
        st.rerun()
with c2:
    if st.button("🔄 SCAN NOW", use_container_width=True):
        st.session_state.last_refresh = datetime.now() - timedelta(seconds=SCAN_INTERVAL_SEC)
        st.session_state.prefetched_results = None
        st.session_state.prefetch_time = None
        st.rerun()
with c3:
    if st.button("💾 TRAIN+SAVE", use_container_width=True):
        with st.spinner("Training + saving 700F model..."):
            X_train, y_train = generate_training_data_rsi_enhanced(1400)
            st.session_state.model.train(X_train, y_train)
            ok = save_model_bundle(st.session_state.model)
            st.success("Saved ✅" if ok else "Save failed ❌")
with c4:
    if st.button("📂 LOAD MODEL", use_container_width=True):
        ok = load_model_bundle(st.session_state.model)
        st.success("Loaded ✅" if ok else "Load failed ❌")
with c5:
    if st.button("📥 DOWNLOAD", use_container_width=True):
        if os.path.exists(EXCEL_FILE):
            with open(EXCEL_FILE, "rb") as f:
                st.download_button(
                    "📊 Excel",
                    f,
                    f"Velora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

st.markdown("---")

def run_analysis_round(model, comparator, current_time, news_analyzer, protector):
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                advanced_analyze,
                asset,
                model,
                current_time,
                comparator,
                news_analyzer,
                protector
            ): asset
            for asset in ALL_ASSETS
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass
    return results

if st.session_state.running:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    remaining = SCAN_INTERVAL_SEC - elapsed

    if 0 < remaining <= PREDICTION_LEAD_SEC and st.session_state.prefetched_results is None:
        prefetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.spinner(f"🧠 Precomputing next signals... (T-{remaining:.1f}s)"):
            st.session_state.prefetched_results = run_analysis_round(
                st.session_state.model,
                st.session_state.comparator,
                prefetch_time,
                st.session_state.news_analyzer,
                st.session_state.protector
            )
            st.session_state.prefetch_time = prefetch_time

    if remaining <= 0:
        st.session_state.total_rounds += 1
        with st.spinner(f"🔄 Round {st.session_state.total_rounds}: Publishing signals..."):
            if st.session_state.prefetched_results is not None:
                results = st.session_state.prefetched_results
            else:
                seed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results = run_analysis_round(
                    st.session_state.model,
                    st.session_state.comparator,
                    seed,
                    st.session_state.news_analyzer,
                    st.session_state.protector
                )

        if results:
            df = pd.DataFrame(results)

            buy_count = len(df[df["Signal"] == "BUY"])
            sell_count = len(df[df["Signal"] == "SELL"])
            no_count = len(df[df["Signal"] == "NO-TRADE"])

            st.session_state.total_signals["BUY"] += buy_count
            st.session_state.total_signals["SELL"] += sell_count
            st.session_state.total_signals["NO-TRADE"] += no_count

            tradable = df[df["Signal"].isin(["BUY", "SELL"])]
            st.session_state.avg_confidence = int(tradable["Confidence"].mean()) if len(tradable) else 0
            st.session_state.last_refresh = datetime.now()
            save_to_excel(results)

            s1, s2, s3 = st.columns(3)
            with s1: st.success(f"✅ {len(results)} Signals")
            with s2: st.info(f"🟢 {buy_count} BUY | 🔴 {sell_count} SELL")
            with s3: st.warning(f"⛔ {no_count} NO-TRADE")

            st.markdown("---")
            st.subheader("🏆 Top Signals (Confidence)")
            top_df = df.nlargest(20, "Confidence")[[
                "Asset", "Signal", "Confidence", "DL_Models", "RSI14", "ATR_Ratio", "Confirm_4_15", "Market_Source", "Last_Price", "Blocked_Reason"
            ]].copy()
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🧬 Feature Importance (Top 30 / 700)")
            top_feats = st.session_state.model.get_top_features(k=30)
            if top_feats:
                fi_df = pd.DataFrame(top_feats, columns=["Feature_Index", "Importance"])
                fi_df["Importance"] = (fi_df["Importance"] * 100).round(4)
                st.dataframe(fi_df, use_container_width=True, hide_index=True)
                st.bar_chart(fi_df.set_index("Feature_Index")[["Importance"]])
            else:
                st.info("Feature importance henüz hazır değil.")

        st.session_state.prefetched_results = None
        st.session_state.prefetch_time = None
        time.sleep(0.2)
        st.rerun()
    else:
        progress = max(0.0, min(1.0, elapsed / SCAN_INTERVAL_SEC))
        st.progress(progress)
        st.info(f"⏱️ Next scan in {remaining:.1f} sec (precompute in last {PREDICTION_LEAD_SEC}s)")
        time.sleep(0.2)
        st.rerun()
else:
    st.info("👇 Click START to begin real-time analysis.")


















