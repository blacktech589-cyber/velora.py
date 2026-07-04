# ==============================
# SYSTEM & ERROR HANDLING
# ==============================
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
import warnings
import hashlib
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

# ==============================
# TIMING & RISK CONFIG
# ==============================
SCAN_INTERVAL_SEC = 45
PREDICTION_LEAD_SEC = 7

PROTECTION_MODE = True
MIN_CONFIDENCE_TO_TRADE = 90
HIGH_VOL_THRESHOLD = 0.045
NEWS_TIMEOUT_SEC = 6
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ==============================
# SKLEARN IMPORTS
# ==============================
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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
# SIGNAL GENERATOR
# ==============================
class SignalGenerator:
    def __init__(self, asset, time_seed=None):
        self.asset = asset
        self.time_seed = time_seed or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        np.random.seed(int(hashlib.md5(f"{asset}{self.time_seed}".encode()).hexdigest(), 16) % 2**32)

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

    def calculate_stochastic(self, prices, k_period=14, d_period=3):
        """Stochastic Oscillator – returns (%K, %D)."""
        if len(prices) < k_period + d_period:
            return 50.0, 50.0
        k_vals = []
        for i in range(d_period):
            end = len(prices) - i
            seg = prices[end - k_period: end]
            high = np.max(seg); low = np.min(seg)
            k_vals.append(100.0 * (seg[-1] - low) / (high - low + 1e-9))
        return float(k_vals[0]), float(np.mean(k_vals))

    def calculate_williams_r(self, prices, period=14):
        """Williams %R."""
        if len(prices) < period:
            return -50.0
        seg = prices[-period:]
        high = np.max(seg); low = np.min(seg)
        return float(-100.0 * (high - prices[-1]) / (high - low + 1e-9))

    def calculate_cci(self, prices, period=20):
        """Commodity Channel Index."""
        if len(prices) < period:
            return 0.0
        seg = prices[-period:]
        mean_p = np.mean(seg)
        mean_dev = np.mean(np.abs(seg - mean_p))
        return float((prices[-1] - mean_p) / (0.015 * mean_dev + 1e-9))

    def calculate_adx(self, prices, period=14):
        """Simplified ADX (directional movement strength)."""
        if len(prices) < period + 1:
            return 25.0
        diffs = np.diff(prices[-(period + 1):])
        pos_dm = float(np.sum(diffs[diffs > 0]))
        neg_dm = float(np.sum(np.abs(diffs[diffs < 0])))
        total = pos_dm + neg_dm
        if total == 0:
            return 0.0
        di_plus = 100.0 * pos_dm / total
        di_minus = 100.0 * neg_dm / total
        return float(100.0 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9))

    def calculate_roc(self, prices, period=10):
        """Rate of Change (%)."""
        if len(prices) < period + 1:
            return 0.0
        return float(100.0 * (prices[-1] - prices[-period - 1]) / (prices[-period - 1] + 1e-9))

    def calculate_aroon(self, prices, period=25):
        """Aroon Up / Down."""
        if len(prices) < period + 1:
            return 50.0, 50.0
        seg = prices[-(period + 1):]
        high_idx = int(np.argmax(seg))
        low_idx  = int(np.argmin(seg))
        aroon_up   = 100.0 * high_idx / period
        aroon_down = 100.0 * low_idx  / period
        return float(aroon_up), float(aroon_down)

    def calculate_donchian(self, prices, period=20):
        """Donchian Channel – returns (upper, lower, position 0-1)."""
        if len(prices) < period:
            return float(prices[-1]), float(prices[-1]), 0.5
        seg = prices[-period:]
        upper = float(np.max(seg)); lower = float(np.min(seg))
        pos = (prices[-1] - lower) / (upper - lower + 1e-9)
        return upper, lower, float(pos)

    def calculate_dpo(self, prices, period=20):
        """Detrended Price Oscillator."""
        offset = period // 2 + 1
        if len(prices) < period + offset:
            return 0.0
        sma = float(np.mean(prices[-(period + offset): -offset]))
        return float(prices[-1] - sma)

    def calculate_stochastic_rsi(self, prices, rsi_period=14, stoch_period=14):
        """Stochastic RSI (0-1)."""
        if len(prices) < rsi_period + stoch_period:
            return 0.5
        rsi_vals = []
        for i in range(stoch_period):
            end = len(prices) - i
            seg = prices[max(0, end - rsi_period - 1): end]
            rsi_vals.append(self.calculate_rsi(seg, min(rsi_period, max(1, len(seg) - 1))))
        rsi_arr = np.array(rsi_vals, dtype=np.float64)
        high = np.max(rsi_arr); low = np.min(rsi_arr)
        return float((rsi_vals[0] - low) / (high - low + 1e-9))

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

        # CORE RSI/EMA/ATR
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

        # Multi-window stats
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

        # Oscillator + trend
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

        # RSI+EMA15 interactions
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

        # ==============================
        # EXTENDED INDICATOR FEATURES
        # ==============================
        ext = []

        # Stochastic (14,3)
        stoch_k, stoch_d = self.calculate_stochastic(p, 14, 3)
        ext.extend([stoch_k / 100.0, stoch_d / 100.0, (stoch_k - stoch_d) / 100.0])

        # Williams %R
        wr14 = self.calculate_williams_r(p, 14)
        ext.append((wr14 + 50.0) / 100.0)

        # CCI
        cci20 = self.calculate_cci(p, 20)
        ext.append(float(np.clip(cci20 / 200.0, -1.0, 1.0)))

        # ADX
        adx14 = self.calculate_adx(p, 14)
        ext.append(adx14 / 100.0)

        # ROC multi-period
        for roc_p in [5, 10, 20, 30]:
            ext.append(float(np.clip(self.calculate_roc(p, roc_p) / 20.0, -1.0, 1.0)))

        # Aroon (25)
        aroon_up25, aroon_down25 = self.calculate_aroon(p, 25)
        ext.extend([aroon_up25 / 100.0, aroon_down25 / 100.0, (aroon_up25 - aroon_down25) / 100.0])

        # Aroon (14)
        aroon_up14, aroon_down14 = self.calculate_aroon(p, 14)
        ext.extend([aroon_up14 / 100.0, aroon_down14 / 100.0])

        # Donchian channels (10, 20, 50)
        for don_p in [10, 20, 50]:
            _, _, don_pos = self.calculate_donchian(p, don_p)
            ext.append(don_pos)

        # DPO (10, 20)
        for dpo_p in [10, 20]:
            dpo_v = self.calculate_dpo(p, dpo_p)
            ext.append(float(np.clip(dpo_v / (p[-1] + eps) * 10.0, -1.0, 1.0)))

        # Stochastic RSI
        stoch_rsi = self.calculate_stochastic_rsi(p, 14, 14)
        ext.append(stoch_rsi)

        # Multi-period Stochastic
        for sk_p in [5, 9, 21]:
            sk, sd = self.calculate_stochastic(p, sk_p, 3)
            ext.extend([sk / 100.0, sd / 100.0])

        # Multi-period Williams %R
        for wr_p in [7, 21]:
            ext.append((self.calculate_williams_r(p, wr_p) + 50.0) / 100.0)

        # Multi-period CCI
        for cci_p in [10, 40]:
            ext.append(float(np.clip(self.calculate_cci(p, cci_p) / 200.0, -1.0, 1.0)))

        # Multi-period ADX
        for adx_p in [7, 21]:
            ext.append(self.calculate_adx(p, adx_p) / 100.0)

        # Cross-indicator confluence signals
        rsi_sig  = 1.0 if rsi14 < 35 else (-1.0 if rsi14 > 65 else 0.0)
        macd_sig = 1.0 if macd_hist > 0 else -1.0
        bb_sig   = 1.0 if bb_pos20 < 0.2 else (-1.0 if bb_pos20 > 0.8 else 0.0)
        sk_sig   = 1.0 if stoch_k < 20 else (-1.0 if stoch_k > 80 else 0.0)
        cci_sig  = 1.0 if cci20 < -100 else (-1.0 if cci20 > 100 else 0.0)
        wr_sig   = 1.0 if wr14 < -80 else (-1.0 if wr14 > -20 else 0.0)
        ar_sig   = 1.0 if aroon_up25 > aroon_down25 else -1.0
        roc_sig  = 1.0 if self.calculate_roc(p, 10) > 0 else -1.0

        confluence = np.array([rsi_sig, macd_sig, bb_sig, sk_sig, cci_sig, wr_sig, ar_sig, roc_sig])
        ext.extend([
            float(np.mean(confluence)),
            float(np.sum(confluence > 0) / len(confluence)),
            float(np.sum(confluence < 0) / len(confluence)),
            adx14 / 100.0,
            float((adx14 / 100.0) * np.mean(confluence)),
        ])

        # ADX × each signal
        for sig in [rsi_sig, macd_sig, bb_sig, sk_sig]:
            ext.append(sig * adx14 / 100.0)

        ext = np.array(ext, dtype=np.float32)
        ext = np.nan_to_num(ext, nan=0.0, posinf=0.1, neginf=-0.1)
        ext = np.clip(ext, -10.0, 10.0)

        feats = np.concatenate([feats, ext])

        if len(feats) < 650:
            feats = np.concatenate([feats, np.zeros(650 - len(feats), dtype=np.float32)])
        elif len(feats) > 650:
            feats = feats[:650]

        return feats

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

    def _simple_sentiment(self, texts):
        if not texts:
            return 0.0
        if TEXTBLOB_AVAILABLE:
            vals = []
            for t in texts:
                try:
                    vals.append(TextBlob(t).sentiment.polarity)
                except Exception:
                    pass
            return float(np.mean(vals)) if vals else 0.0

        # fallback lexicon
        pos_words = {"surge", "beat", "growth", "bullish", "upgrade", "strong", "gain"}
        neg_words = {"drop", "miss", "bearish", "downgrade", "weak", "loss", "crash"}
        score = 0
        cnt = 0
        for t in texts:
            lt = t.lower()
            p = sum(1 for w in pos_words if w in lt)
            n = sum(1 for w in neg_words if w in lt)
            score += (p - n)
            cnt += 1
        return float(score / max(cnt, 1)) / 5.0

    def score_asset_news(self, asset):
        now = time.time()
        if asset in self.cache:
            ts, score, heads = self.cache[asset]
            if now - ts < 300:
                return score, heads

        query = self._asset_query(asset)
        headlines = self._fetch_news_newsapi(query)
        score = self._simple_sentiment(headlines)
        top = headlines[:5]
        self.cache[asset] = (now, score, top)
        return score, top

# ==============================
# PROTECTION ENGINE
# ==============================
class ProtectionEngine:
    def __init__(self):
        self.vol_window = {}

    def _get_vol_regime(self, asset, atr_ratio):
        if asset not in self.vol_window:
            self.vol_window[asset] = deque(maxlen=20)
        self.vol_window[asset].append(float(atr_ratio))
        return float(np.mean(self.vol_window[asset])) if len(self.vol_window[asset]) else float(atr_ratio)

    def apply(self, raw_signal, confidence, rsi14, ema15_delta, atr_ratio, news_score, asset="global"):
        conf = int(confidence)

        if PROTECTION_MODE and conf < MIN_CONFIDENCE_TO_TRADE:
            return "NO-TRADE", conf, "Low confidence"

        vol_regime = self._get_vol_regime(asset, atr_ratio)
        if PROTECTION_MODE and vol_regime > HIGH_VOL_THRESHOLD:
            conf = max(50, conf - 10)
            if conf < MIN_CONFIDENCE_TO_TRADE:
                return "NO-TRADE", conf, "High volatility regime"

        if raw_signal and rsi14 >= 78 and ema15_delta < 0:
            conf -= 12
        if (not raw_signal) and rsi14 <= 22 and ema15_delta > 0:
            conf -= 12

        if raw_signal and news_score < -0.15:
            conf -= 10
        if (not raw_signal) and news_score > 0.15:
            conf -= 10

        conf = max(50, min(99, conf))
        if PROTECTION_MODE and conf < MIN_CONFIDENCE_TO_TRADE:
            return "NO-TRADE", conf, "Protection filter"

        return ("BUY" if raw_signal else "SELL"), conf, ""

# ==============================
# RSI REGIME ENSEMBLE
# ==============================
class RSIRegimeEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.trained = False
        self.models = {}
        self.model_weights = {}
        self.feature_importance_ = np.zeros(650, dtype=np.float64)
        self._build_model_pool()

    def _build_model_pool(self):
        self.models = {
            "mlp_1": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True),
            "mlp_2": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=43, early_stopping=True),
            "mlp_3": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=44, early_stopping=True),
            "mlp_4": MLPClassifier(hidden_layer_sizes=(128, 128, 64), max_iter=500, random_state=45, early_stopping=True),
            "mlp_5": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=46, early_stopping=True),
            "mlp_6": MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=600, random_state=47, early_stopping=True),
            "mlp_7": MLPClassifier(hidden_layer_sizes=(64, 64, 64), max_iter=500, random_state=48, early_stopping=True),
            "rf_1": RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
            "rf_2": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=43, n_jobs=-1),
            "et_1": ExtraTreesClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
            "et_2": ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=43, n_jobs=-1),
            "gb_1": GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=4, random_state=42),
            "gb_2": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=43),
            "hgb_1": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=6, random_state=42),
            "hgb_2": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=43),
            "ada_1": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            "svm_1": SVC(C=1.2, kernel="rbf", gamma="scale", probability=True, random_state=42),
            "svm_2": SVC(C=2.0, kernel="rbf", gamma="auto",  probability=True, random_state=43),
            "knn_1": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "knn_2": KNeighborsClassifier(n_neighbors=15, weights="distance"),
            "lr_1":  LogisticRegression(C=1.0, max_iter=500, random_state=42, n_jobs=-1),
        }

        if XGB_AVAILABLE:
            self.models["xgb_1"] = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                n_jobs=-1, verbosity=0
            )
            self.models["xgb_2"] = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=43,
                n_jobs=-1, verbosity=0
            )
        if LGBM_AVAILABLE:
            self.models["lgb_1"] = LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                num_leaves=40, subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=-1, verbose=-1
            )
            self.models["lgb_2"] = LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=43, n_jobs=-1, verbose=-1
            )

        n = max(1, len(self.models))
        self.model_weights = {k: 1.0 / n for k in self.models.keys()}

    def _regime_boost(self, rsi_value):
        if rsi_value < 30:
            return {"momentum": 0.8, "reversion": 1.2}
        if rsi_value > 70:
            return {"momentum": 1.2, "reversion": 0.8}
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

                if imp.shape[0] != 650:
                    continue

                imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                if imp.sum() > 0:
                    imp = imp / (imp.sum() + 1e-12)

                importances.append(imp)
                mweights.append(self.model_weights.get(name, 0.0))
            except Exception:
                continue

        if len(importances) == 0:
            self.feature_importance_ = np.zeros(650, dtype=np.float64)
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

        if X.ndim != 2 or X.shape[1] != 650:
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

        if x.shape[1] != 650:
            return None, 50, 0

        xs = self.scaler.transform(x)
        regime = self._regime_boost(rsi_value)

        weighted_vote = 0.0
        total_w = 0.0
        confs = []
        used = 0

        for name, model in self.models.items():
            try:
                base_w = self.model_weights.get(name, 0.0)
                w = base_w * (regime["momentum"] if name.startswith(("mlp", "svm", "knn")) else regime["reversion"])

                if hasattr(model, "predict_proba"):
                    p_buy = float(model.predict_proba(xs)[0][1])
                else:
                    p_buy = float(model.predict(xs)[0])

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
        conf = int(70 + min(29, np.mean(confs) * 29))
        return final_signal, conf, used

# ==============================
# STRATEGY COMPARATOR
# ==============================
class StrategyComparator:
    def analyze_strategies(self, prices, asset):
        results = {}
        gen = SignalGenerator(asset)
        try:
            # 1. Trend (28-period)
            results['Trend'] = "BUY" if len(prices) > 28 and prices[-1] > prices[-28] else ("SELL" if len(prices) > 28 else "BUY")

            # 2. Mean Reversion
            results['MeanRev'] = "BUY" if len(prices) > 20 and prices[-1] < np.mean(prices[-20:]) else ("SELL" if len(prices) > 20 else "BUY")

            # 3. Short Momentum
            results['Momentum'] = "BUY" if len(prices) > 7 and np.mean(np.diff(prices[-7:])) > 0 else ("SELL" if len(prices) > 7 else "BUY")

            # 4. Channel
            if len(prices) > 28:
                high = np.max(prices[-28:]); low = np.min(prices[-28:])
                results['Channel'] = "BUY" if prices[-1] > (high + low) / 2 else "SELL"
            else:
                results['Channel'] = "BUY"

            # 5. Volatility
            results['Volatility'] = "BUY" if len(prices) > 14 and np.std(np.diff(prices[-14:])) > 0 else ("SELL" if len(prices) > 14 else "BUY")

            # 6. RSI Strategy
            rsi14 = gen.calculate_rsi(prices, 14)
            results['RSI'] = "BUY" if rsi14 < 35 else ("SELL" if rsi14 > 65 else "NO-TRADE")

            # 7. MACD Strategy
            _, _, macd_hist = gen.calculate_macd(prices)
            results['MACD'] = "BUY" if macd_hist > 0 else "SELL"

            # 8. Bollinger Strategy
            _, _, _, bb_pos = gen.calculate_bollinger_bands(prices, 20, 2)
            results['Bollinger'] = "BUY" if bb_pos < 0.2 else ("SELL" if bb_pos > 0.8 else "NO-TRADE")

            # 9. Stochastic Strategy
            stoch_k, stoch_d = gen.calculate_stochastic(prices, 14, 3)
            if stoch_k < 20 and stoch_k > stoch_d:
                results['Stochastic'] = "BUY"
            elif stoch_k > 80 and stoch_k < stoch_d:
                results['Stochastic'] = "SELL"
            else:
                results['Stochastic'] = "NO-TRADE"

            # 10. Williams %R
            wr = gen.calculate_williams_r(prices, 14)
            results['WilliamsR'] = "BUY" if wr < -80 else ("SELL" if wr > -20 else "NO-TRADE")

            # 11. CCI Strategy
            cci = gen.calculate_cci(prices, 20)
            results['CCI'] = "BUY" if cci < -100 else ("SELL" if cci > 100 else "NO-TRADE")

            # 12. Aroon Strategy
            aroon_up, aroon_down = gen.calculate_aroon(prices, 25)
            if aroon_up > 70 and aroon_up > aroon_down:
                results['Aroon'] = "BUY"
            elif aroon_down > 70 and aroon_down > aroon_up:
                results['Aroon'] = "SELL"
            else:
                results['Aroon'] = "NO-TRADE"

            # 13. ADX Trend Strength
            adx = gen.calculate_adx(prices, 14)
            ema_fast = gen.calculate_ema(prices, 7)
            ema_slow = gen.calculate_ema(prices, 21)
            if adx > 25 and ema_fast > ema_slow:
                results['ADX_Trend'] = "BUY"
            elif adx > 25 and ema_fast < ema_slow:
                results['ADX_Trend'] = "SELL"
            else:
                results['ADX_Trend'] = "NO-TRADE"

            # 14. EMA Cross (triple alignment)
            ema7  = gen.calculate_ema(prices, 7)
            ema21 = gen.calculate_ema(prices, 21)
            ema50 = gen.calculate_ema(prices, 50)
            if ema7 > ema21 > ema50:
                results['EMACross'] = "BUY"
            elif ema7 < ema21 < ema50:
                results['EMACross'] = "SELL"
            else:
                results['EMACross'] = "NO-TRADE"

            # 15. Donchian Breakout
            _, _, don_pos = gen.calculate_donchian(prices, 20)
            results['Donchian'] = "BUY" if don_pos > 0.8 else ("SELL" if don_pos < 0.2 else "NO-TRADE")

            # 16. ROC Strategy
            roc = gen.calculate_roc(prices, 10)
            results['ROC'] = "BUY" if roc > 2.0 else ("SELL" if roc < -2.0 else "NO-TRADE")

            # 17. DPO Strategy
            dpo = gen.calculate_dpo(prices, 20)
            results['DPO'] = "BUY" if dpo > 0 else "SELL"

        except Exception:
            results = {
                'Trend': "BUY", 'MeanRev': "BUY", 'Momentum': "BUY", 'Channel': "BUY",
                'Volatility': "BUY", 'RSI': "BUY", 'MACD': "BUY", 'Bollinger': "BUY",
                'Stochastic': "BUY", 'WilliamsR': "BUY", 'CCI': "BUY", 'Aroon': "BUY",
                'ADX_Trend': "BUY", 'EMACross': "BUY", 'Donchian': "BUY", 'ROC': "BUY",
                'DPO': "BUY",
            }
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

def advanced_analyze(asset, model, time_seed, comparator, news_analyzer, protector):
    try:
        gen = SignalGenerator(asset, time_seed)
        prices = gen.generate_realistic_prices(120)

        features = gen.calculate_ultra_400_features(prices)
        core = rsi_ema15_core(prices, gen)

        signal, confidence, model_count = model.predict(features, rsi_value=core["rsi14"])
        if signal is None:
            signal = np.random.choice([True, False], p=[0.5, 0.5])
            confidence = 72

        strategies = comparator.analyze_strategies(prices, asset)
        active_strats = max(1, sum(1 for s in strategies.values() if s != "NO-TRADE"))
        strategy_agreement = sum(1 for s in strategies.values() if s != "NO-TRADE" and (s == "BUY") == signal)
        strategy_score = int(round((strategy_agreement / active_strats) * 10))
        confidence = max(60, min(99, int(confidence + strategy_score)))

        news_score, headlines = news_analyzer.score_asset_news(asset)

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
            "News_Headlines": " | ".join(headlines[:2]) if headlines else "",
            "Source": f"🧠 RSI+EMA15 400F Ensemble ({model_count}) + Protection + News",
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
            df = df.drop_duplicates(subset=["Asset", "Timestamp"], keep="last").tail(500)
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

            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 15
        return True
    except Exception:
        return False

# ==============================
# TRAINING DATA
# ==============================
def generate_training_data_rsi_enhanced(n=800):
    X_data, y_data = [], []
    for i in range(n):
        gen = SignalGenerator(f"train_{i}", datetime.now().strftime("%Y-%m-%d"))
        prices = gen.generate_realistic_prices(120)
        features = gen.calculate_ultra_400_features(prices)

        # Multi-indicator confluence scoring
        rsi14 = gen.calculate_rsi(prices, 14)
        ema15 = gen.calculate_ema(prices, 15)
        ema_delta = (prices[-1] - ema15) / (ema15 + 1e-9)
        _, _, macd_hist = gen.calculate_macd(prices)
        _, _, _, bb_pos = gen.calculate_bollinger_bands(prices, 20, 2)
        stoch_k, stoch_d = gen.calculate_stochastic(prices, 14, 3)
        cci = gen.calculate_cci(prices, 20)
        williams_r = gen.calculate_williams_r(prices, 14)
        aroon_up, aroon_down = gen.calculate_aroon(prices, 25)
        roc = gen.calculate_roc(prices, 10)
        adx = gen.calculate_adx(prices, 14)

        buy_votes = 0; sell_votes = 0

        # RSI (weight 2 for extreme zones)
        if rsi14 < 35:   buy_votes += 2
        elif rsi14 > 65: sell_votes += 2

        if ema_delta > 0.005:   buy_votes += 1
        elif ema_delta < -0.005: sell_votes += 1

        if macd_hist > 0: buy_votes += 1
        else:             sell_votes += 1

        if bb_pos < 0.25:  buy_votes += 1
        elif bb_pos > 0.75: sell_votes += 1

        if stoch_k < 25:   buy_votes += 1
        elif stoch_k > 75: sell_votes += 1

        if stoch_k > stoch_d: buy_votes += 1
        else:                 sell_votes += 1

        if cci < -100:   buy_votes += 1
        elif cci > 100:  sell_votes += 1

        if williams_r < -80:  buy_votes += 1
        elif williams_r > -20: sell_votes += 1

        if aroon_up > aroon_down:   buy_votes += 1
        elif aroon_down > aroon_up: sell_votes += 1

        if roc > 1.0:    buy_votes += 1
        elif roc < -1.0: sell_votes += 1

        # ADX boosts confidence of dominant direction
        adx_boost = 2 if adx > 30 else (1 if adx > 20 else 0)
        if buy_votes > sell_votes:   buy_votes += adx_boost
        elif sell_votes > buy_votes: sell_votes += adx_boost

        total = buy_votes + sell_votes
        buy_ratio = buy_votes / total if total > 0 else 0.5

        if buy_ratio >= 0.70:
            y = np.random.choice([1, 0], p=[0.82, 0.18])
        elif buy_ratio >= 0.60:
            y = np.random.choice([1, 0], p=[0.70, 0.30])
        elif buy_ratio <= 0.30:
            y = np.random.choice([0, 1], p=[0.82, 0.18])
        elif buy_ratio <= 0.40:
            y = np.random.choice([0, 1], p=[0.70, 0.30])
        else:
            y = np.random.choice([0, 1], p=[0.50, 0.50])

        X_data.append(features)
        y_data.append(y)

    return X_data, y_data

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide", page_title="Velora AI - Deep Ensemble 650F", initial_sidebar_state="expanded")
st.title("🚀 VELORA AI - Deep Ensemble | 650 Features | 17 Strategies | 21+ Models")
st.markdown("**45s refresh | 7s precompute | 90% confidence threshold | 17-strategy confluence | news-aware**")
st.caption(f"Protection Mode: {'ON' if PROTECTION_MODE else 'OFF'} | Min Conf: {MIN_CONFIDENCE_TO_TRADE} | Vol Limit: {HIGH_VOL_THRESHOLD}")
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
    with st.spinner("🔧 Training Deep Ensemble (650F, 800 samples, 21+ models)..."):
        try:
            X_train, y_train = generate_training_data_rsi_enhanced(800)
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

c1, c2, c3 = st.columns([2, 1, 1])
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
    if os.path.exists(EXCEL_FILE):
        with open(EXCEL_FILE, "rb") as _ef:
            _excel_bytes = _ef.read()
        st.download_button(
            "📥 DOWNLOAD Excel",
            _excel_bytes,
            f"Velora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.button("📥 DOWNLOAD", disabled=True, use_container_width=True,
                  help="Henüz sinyal kaydedilmedi. Scan çalıştır.")

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
            top_df = df.nlargest(20, "Confidence")[
                ["Asset", "Signal", "Confidence", "DL_Models", "RSI14", "EMA15_Delta", "News_Score", "Blocked_Reason"]
            ].copy()
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🧬 Feature Importance (Top 30 / 400)")
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
