"""
Binomo 1M Signal Engine

UYARI:
- Bu kod finansal tavsiye degildir.
- Binomo'nun tum verilerini "bilmek" teknik ve hukuki olarak mumkun degildir.
- Algoritma sadece sizin sagladiginiz canli/CSV mum verisini analiz eder.
- 1 dakikalik islemler cok yuksek risklidir; varsayilan mod guven filtresiyle
  emin olmadigi yerde NO-TRADE uretir.

Calistirma:
    pip install streamlit pandas numpy requests scikit-learn openpyxl
    streamlit run binomo_1m_algo.py

Veri secenekleri:
1) CSV yukle: time,open,high,low,close,volume kolonlari tavsiye edilir.
2) HTTP endpoint kullan: JSON olarak mum listesi donen kendi veri servisiniz.
   Ornek mum:
   {"time":"2026-07-04 12:00:00","open":1.1,"high":1.2,"low":1.0,"close":1.15,"volume":10}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    MLPClassifier = None
    StandardScaler = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    keras = None
    layers = None


APP_TITLE = "Velora Enterprise 1M Intelligence"
SCAN_SECONDS = 90
MIN_CANDLES = 80
MAX_CANDLES = 500
TRAINING_CANDLES = 9000
LONG_TRAINING_CANDLES = 35040
DEFAULT_MIN_CONF = 68
DEFAULT_MAX_ATR = 0.008
EXPORT_FILE = Path("binomo_1m_signals.xlsx")
AUDIT_FILE = Path("velora_audit_log.jsonl")
CONFIG_FILE = Path("velora_enterprise_config.json")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RISK_PROFILES = {
    "Conservative": {"min_conf": 76, "max_atr": 0.0055, "allow_news_conflict": False},
    "Balanced": {"min_conf": 68, "max_atr": 0.0080, "allow_news_conflict": True},
    "Aggressive": {"min_conf": 60, "max_atr": 0.0140, "allow_news_conflict": True},
}

BINANCE_SYMBOLS = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT",
    "BNBUSDT": "BNBUSDT",
    "XRPUSDT": "XRPUSDT",
    "ADAUSDT": "ADAUSDT",
    "DOGEUSDT": "DOGEUSDT",
}

YAHOO_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "NASDAQ100": "^NDX",
    "SP500": "^GSPC",
}


@dataclass
class SignalResult:
    asset: str
    signal: str
    confidence: int
    reason: str
    last_price: float
    rsi: float
    ema_fast_delta: float
    ema_slow_delta: float
    atr_ratio: float
    trend_score: float
    pullback_score: float
    best_strategy: str
    strategy_score: float
    news_score: float
    telegram_score: float
    data_source: str
    model_probability: float
    risk_status: str
    action: str
    timestamp: str


def normalize_candles(df: pd.DataFrame, max_rows: int = MAX_CANDLES) -> pd.DataFrame:
    """Mum verisini standart OHLCV formatina getirir."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    aliases = {
        "timestamp": "time",
        "date": "time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})

    if "close" not in out.columns:
        raise ValueError("Veride 'close' kolonu bulunmali.")

    if "open" not in out.columns:
        out["open"] = out["close"].shift(1).fillna(out["close"])
    if "high" not in out.columns:
        out["high"] = out[["open", "close"]].max(axis=1)
    if "low" not in out.columns:
        out["low"] = out[["open", "close"]].min(axis=1)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "time" not in out.columns:
        out["time"] = pd.date_range(end=pd.Timestamp.now(), periods=len(out), freq="1min")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_values("time").tail(max_rows).reset_index(drop=True)
    return out[["time", "open", "high", "low", "close", "volume"]]


def fetch_http_candles(url: str, timeout: int = 8) -> pd.DataFrame:
    """Kendi Binomo veri koprunuzden mum JSON'u okur."""
    if not url.strip():
        return pd.DataFrame()

    r = requests.get(url.strip(), timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict):
        rows = payload.get("candles") or payload.get("data") or payload.get("result") or []
    else:
        rows = payload

    return normalize_candles(pd.DataFrame(rows))


def fetch_binance_candles(symbol: str, limit: int = MAX_CANDLES) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper().strip(), "interval": "1m", "limit": min(limit, 1000)}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    rows = []
    for item in r.json():
        rows.append(
            {
                "time": pd.to_datetime(int(item[0]), unit="ms"),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            }
        )
    return normalize_candles(pd.DataFrame(rows))


def fetch_binance_history(symbol: str, interval: str = "1h", limit: int = TRAINING_CANDLES) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    end_time = None
    while len(rows) < limit:
        params = {"symbol": symbol.upper().strip(), "interval": interval, "limit": min(1000, limit - len(rows))}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows = chunk + rows
        end_time = int(chunk[0][0]) - 1
        if len(chunk) < 1000:
            break
        time.sleep(0.05)
    parsed = [
        {
            "time": pd.to_datetime(int(item[0]), unit="ms"),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in rows[-limit:]
    ]
    return normalize_candles(pd.DataFrame(parsed), max_rows=limit)


def fetch_yahoo_candles(symbol: str, limit: int = MAX_CANDLES, interval: str = "1m", range_value: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": range_value, "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    data = r.json()
    result = (data.get("chart", {}) or {}).get("result", [])
    if not result:
        return pd.DataFrame()
    node = result[0]
    times = node.get("timestamp", []) or []
    quote = (((node.get("indicators", {}) or {}).get("quote", [])) or [{}])[0]
    rows = []
    for i, ts in enumerate(times):
        try:
            rows.append(
                {
                    "time": pd.to_datetime(int(ts), unit="s"),
                    "open": quote.get("open", [])[i],
                    "high": quote.get("high", [])[i],
                    "low": quote.get("low", [])[i],
                    "close": quote.get("close", [])[i],
                    "volume": quote.get("volume", [0])[i] if quote.get("volume") else 0,
                }
            )
        except Exception:
            pass
    return normalize_candles(pd.DataFrame(rows).tail(limit), max_rows=limit)


def fetch_yahoo_history(symbol: str, limit: int = TRAINING_CANDLES) -> pd.DataFrame:
    for interval, range_value in [("1h", "1y"), ("1d", "2y")]:
        try:
            df = fetch_yahoo_candles(symbol, limit=limit, interval=interval, range_value=range_value)
            if len(df) >= MIN_CANDLES:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def fetch_yahoo_long_history(symbol: str, limit: int = LONG_TRAINING_CANDLES) -> pd.DataFrame:
    for interval, range_value in [("1d", "5y"), ("1wk", "10y")]:
        try:
            df = fetch_yahoo_candles(symbol, limit=limit, interval=interval, range_value=range_value)
            if len(df) >= MIN_CANDLES:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def fetch_market_candles(source: str, asset: str, endpoint: str = "") -> tuple[pd.DataFrame, str]:
    source = source.lower()
    asset_key = asset.upper().strip()
    if source == "binomo/http":
        return fetch_http_candles(endpoint), "Binomo/HTTP"
    if source == "binance":
        symbol = BINANCE_SYMBOLS.get(asset_key, asset_key.replace("/", ""))
        return fetch_binance_candles(symbol), f"Binance {symbol} 1m"
    if source == "binance 1y":
        symbol = BINANCE_SYMBOLS.get(asset_key, asset_key.replace("/", ""))
        hist = fetch_binance_history(symbol)
        live = fetch_binance_candles(symbol)
        return normalize_candles(pd.concat([hist, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last"), max_rows=TRAINING_CANDLES + MAX_CANDLES), f"Binance {symbol} 1Y+1m"
    if source == "binance 4y":
        symbol = BINANCE_SYMBOLS.get(asset_key, asset_key.replace("/", ""))
        hist = fetch_binance_history(symbol, interval="1h", limit=LONG_TRAINING_CANDLES)
        live = fetch_binance_candles(symbol)
        return normalize_candles(pd.concat([hist, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last"), max_rows=LONG_TRAINING_CANDLES + MAX_CANDLES), f"Binance {symbol} 4Y+1m"
    if source == "yahoo":
        symbol = YAHOO_SYMBOLS.get(asset.upper().strip(), asset.strip())
        return fetch_yahoo_candles(symbol), f"Yahoo {symbol} 1m"
    if source == "yahoo 1y":
        symbol = YAHOO_SYMBOLS.get(asset.upper().strip(), asset.strip())
        hist = fetch_yahoo_history(symbol)
        live = fetch_yahoo_candles(symbol)
        return normalize_candles(pd.concat([hist, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last"), max_rows=TRAINING_CANDLES + MAX_CANDLES), f"Yahoo {symbol} 1Y+1m"
    if source == "yahoo 4y":
        symbol = YAHOO_SYMBOLS.get(asset.upper().strip(), asset.strip())
        hist = fetch_yahoo_long_history(symbol)
        live = fetch_yahoo_candles(symbol)
        return normalize_candles(pd.concat([hist, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last"), max_rows=LONG_TRAINING_CANDLES + MAX_CANDLES), f"Yahoo {symbol} 4Y+1m"
    return pd.DataFrame(), "No source"


def refresh_candles_for_scan(existing: pd.DataFrame, source: str, asset: str, endpoint: str = "") -> tuple[pd.DataFrame, str]:
    """90 sn taramada 1Y/4Y seti korur, yeni 1m veriyi ekler."""
    if existing is None or existing.empty:
        return fetch_market_candles(source, asset, endpoint)

    source_key = source.lower()
    if source_key.startswith("yahoo"):
        symbol = YAHOO_SYMBOLS.get(asset.upper().strip(), asset.strip())
        live = fetch_yahoo_candles(symbol)
        merged = pd.concat([existing, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last")
        limit = LONG_TRAINING_CANDLES + MAX_CANDLES if "4y" in source_key else TRAINING_CANDLES + MAX_CANDLES
        return normalize_candles(merged, max_rows=limit), f"Yahoo {symbol} history+live"
    if source_key.startswith("binance"):
        symbol = BINANCE_SYMBOLS.get(asset.upper().strip(), asset.upper().strip().replace("/", ""))
        live = fetch_binance_candles(symbol)
        merged = pd.concat([existing, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last")
        limit = LONG_TRAINING_CANDLES + MAX_CANDLES if "4y" in source_key else TRAINING_CANDLES + MAX_CANDLES
        return normalize_candles(merged, max_rows=limit), f"Binance {symbol} history+live"
    if source_key == "binomo/http" and endpoint.strip():
        live = fetch_http_candles(endpoint)
        merged = pd.concat([existing, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last")
        return normalize_candles(merged, max_rows=LONG_TRAINING_CANDLES + MAX_CANDLES), "Binomo/HTTP history+live"
    return existing, "CSV"


def text_sentiment_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    positive = [
        "rise", "rising", "bull", "bullish", "gain", "gains", "up", "breakout",
        "strong", "beat", "growth", "surge", "buy", "long", "support",
    ]
    negative = [
        "fall", "falling", "bear", "bearish", "loss", "losses", "down", "breakdown",
        "weak", "miss", "drop", "crash", "sell", "short", "resistance", "risk",
    ]
    blob = " ".join(str(t).lower() for t in texts)
    pos = sum(blob.count(w) for w in positive)
    neg = sum(blob.count(w) for w in negative)
    if pos + neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / (pos + neg), -1, 1))


def fetch_news(asset: str, api_key: str = "") -> tuple[float, list[str]]:
    api_key = (api_key or NEWS_API_KEY).strip()
    if not api_key:
        return 0.0, []
    query = asset.replace("/", " ")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        articles = r.json().get("articles", []) or []
        texts = [f"{a.get('title', '')} {a.get('description', '')}" for a in articles]
        headlines = [str(a.get("title", "")) for a in articles if a.get("title")]
        return text_sentiment_score(texts), headlines[:5]
    except Exception:
        return 0.0, []


def fetch_telegram_texts(bot_token: str = "", chat_id: str = "", bridge_url: str = "") -> list[str]:
    if bridge_url.strip():
        try:
            r = requests.get(bridge_url.strip(), timeout=7)
            r.raise_for_status()
            payload = r.json()
            rows = payload.get("messages", payload.get("data", payload)) if isinstance(payload, dict) else payload
            return [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in rows][-30:]
        except Exception:
            return []

    bot_token = (bot_token or TELEGRAM_BOT_TOKEN).strip()
    chat_id = str(chat_id or TELEGRAM_CHAT_ID).strip()
    if not bot_token:
        return []
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        r = requests.get(url, params={"limit": 40, "allowed_updates": json.dumps(["message", "channel_post"])}, timeout=7)
        r.raise_for_status()
        texts = []
        for item in r.json().get("result", []) or []:
            msg = item.get("message") or item.get("channel_post") or {}
            if chat_id and str((msg.get("chat") or {}).get("id", "")) != chat_id:
                continue
            text = msg.get("text") or msg.get("caption")
            if text:
                texts.append(str(text))
        return texts[-30:]
    except Exception:
        return []


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ret1"] = close.pct_change().fillna(0)
    out["ret2"] = close.pct_change(2).fillna(0)
    out["ret3"] = close.pct_change(3).fillna(0)
    out["ema5"] = close.ewm(span=5, adjust=False).mean()
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema15"] = close.ewm(span=15, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pos"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["rsi7"] = rsi(close, 7)
    out["rsi14"] = rsi(close, 14)
    out["rsi21"] = rsi(close, 21)
    out["atr14"] = atr(out, 14)
    out["atr_ratio"] = out["atr14"] / out["close"].replace(0, np.nan)
    atr_mean = out["atr_ratio"].rolling(40).mean()
    atr_std = out["atr_ratio"].rolling(40).std()
    out["atr_z"] = (out["atr_ratio"] - atr_mean) / atr_std.replace(0, np.nan)
    out["ema5_delta"] = (close - out["ema5"]) / close.replace(0, np.nan)
    out["ema9_delta"] = (close - out["ema9"]) / close.replace(0, np.nan)
    out["ema15_delta"] = (close - out["ema15"]) / close.replace(0, np.nan)
    out["ema21_delta"] = (close - out["ema21"]) / close.replace(0, np.nan)
    out["ema50_delta"] = (close - out["ema50"]) / close.replace(0, np.nan)
    out["ema5_15_spread"] = (out["ema5"] - out["ema15"]) / close.replace(0, np.nan)
    out["ema15_50_spread"] = (out["ema15"] - out["ema50"]) / close.replace(0, np.nan)
    out["mom3"] = close.pct_change(3)
    out["mom5"] = close.pct_change(5)
    out["mom10"] = close.pct_change(10)
    out["mom15"] = close.pct_change(15)
    out["vol10"] = out["ret1"].rolling(10).std()
    out["vol20"] = out["ret1"].rolling(20).std()
    out["range_pos20"] = range_position(close, 20)
    out["range_pos50"] = range_position(close, 50)
    out["body"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["wick_up"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["wick_down"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range
    out["candle_power"] = (out["close"] - out["open"]) / candle_range
    out["trend_10"] = rolling_slope(close, 10)
    out["trend_20"] = rolling_slope(close, 20)
    out["trend_50"] = rolling_slope(close, 50)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def range_position(close: pd.Series, window: int) -> pd.Series:
    low = close.rolling(window).min()
    high = close.rolling(window).max()
    return ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)


def rolling_slope(close: pd.Series, window: int) -> pd.Series:
    values = []
    x = np.arange(window)
    for i in range(len(close)):
        if i + 1 < window:
            values.append(0.0)
            continue
        seg = close.iloc[i + 1 - window : i + 1].to_numpy(dtype=float)
        mean_price = max(abs(float(np.mean(seg))), 1e-9)
        try:
            values.append(float(np.polyfit(x, seg, 1)[0] / mean_price))
        except Exception:
            values.append(0.0)
    return pd.Series(values, index=close.index)


BASE_FEATURES = [
    "ret1",
    "ret2",
    "ret3",
    "rsi7",
    "rsi14",
    "rsi21",
    "atr_ratio",
    "atr_z",
    "ema5_delta",
    "ema9_delta",
    "ema15_delta",
    "ema21_delta",
    "ema50_delta",
    "ema5_15_spread",
    "ema15_50_spread",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pos",
    "mom3",
    "mom5",
    "mom10",
    "mom15",
    "vol10",
    "vol20",
    "range_pos20",
    "range_pos50",
    "body",
    "wick_up",
    "wick_down",
    "candle_power",
    "trend_10",
    "trend_20",
    "trend_50",
]

MODEL_FEATURE_DIM = 700
MODEL_FEATURES = [f"f{i:03d}" for i in range(MODEL_FEATURE_DIM)]


def build_700_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derin ogrenme ve ensemble icin sabit 700 ozellik uretir."""
    feat = add_indicators(df)
    series_list = []

    def add_feature(values) -> None:
        s = pd.Series(values, index=feat.index)
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)
        series_list.append(s)

    for col in BASE_FEATURES:
        add_feature(feat[col])

    lag_cols = [
        "ret1", "ret2", "ret3", "rsi7", "rsi14", "rsi21", "atr_ratio", "atr_z",
        "ema5_delta", "ema9_delta", "ema15_delta", "ema21_delta", "ema50_delta",
        "ema5_15_spread", "ema15_50_spread", "macd", "macd_signal", "macd_hist", "bb_pos",
        "mom3", "mom5", "mom10", "mom15",
        "vol10", "vol20", "range_pos20", "range_pos50", "body", "wick_up",
        "wick_down", "candle_power", "trend_10", "trend_20", "trend_50",
    ]
    for col in lag_cols:
        for lag in [1, 2, 3, 4, 5, 8, 13, 21]:
            add_feature(feat[col].shift(lag))

    stat_cols = [
        "ret1", "body", "candle_power", "atr_ratio", "atr_z", "ema15_delta",
        "ema15_50_spread", "rsi14", "rsi21", "mom3", "mom10", "vol10",
        "range_pos20", "trend_20", "macd_hist", "bb_pos",
    ]
    for col in stat_cols:
        for window in [3, 5, 8, 13, 21, 34, 55]:
            roll = feat[col].rolling(window)
            add_feature(roll.mean())
            add_feature(roll.std())
            add_feature(roll.min())
            add_feature(roll.max())
            add_feature(feat[col] - roll.mean())

    close = feat["close"]
    for lag in range(1, 121):
        add_feature(close.pct_change(lag))

    interaction_cols = [
        "rsi14", "rsi21", "atr_ratio", "atr_z", "ema15_delta",
        "ema15_50_spread", "macd_hist", "bb_pos", "mom3", "mom10", "vol10", "range_pos20",
        "candle_power", "trend_20",
    ]
    for i, a in enumerate(interaction_cols):
        for b in interaction_cols[i:]:
            add_feature(feat[a] * feat[b])

    if len(series_list) < MODEL_FEATURE_DIM:
        for _ in range(MODEL_FEATURE_DIM - len(series_list)):
            add_feature(0.0)

    out = pd.concat(series_list[:MODEL_FEATURE_DIM], axis=1)
    out.columns = MODEL_FEATURES
    return out.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)


def make_supervised(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sonraki 1 mum yukari mi asagi mi etiketi olusturur."""
    feat = add_indicators(df)
    feature_frame = build_700_feature_frame(df)
    y = (feat["close"].shift(-1) > feat["close"]).astype(int)
    train = feature_frame.iloc[:-1].copy()
    y = y.iloc[:-1]
    X = train[MODEL_FEATURES].to_numpy(dtype=float)
    return X, y.to_numpy(dtype=int)


def strategy_votes(df: pd.DataFrame) -> dict[str, pd.Series]:
    feat = add_indicators(df)
    votes = {}

    votes["EMA15_Trend"] = np.sign(feat["ema5_15_spread"] + feat["ema15_50_spread"])
    votes["RSI_Reversal"] = pd.Series(
        np.where(feat["rsi14"] < 35, 1, np.where(feat["rsi14"] > 65, -1, 0)),
        index=feat.index,
    )
    votes["ATR_Momentum"] = np.sign(feat["mom3"] + feat["mom5"]) * (feat["atr_z"].abs() < 2.5).astype(int)
    votes["Pullback_EMA15"] = pd.Series(
        np.where(
            (feat["ema15_50_spread"] > 0) & (feat["ema15_delta"] < 0) & (feat["rsi14"] > 38),
            1,
            np.where(
                (feat["ema15_50_spread"] < 0) & (feat["ema15_delta"] > 0) & (feat["rsi14"] < 62),
                -1,
                0,
            ),
        ),
        index=feat.index,
    )
    votes["Range_Break"] = pd.Series(
        np.where(feat["range_pos20"] > 0.82, 1, np.where(feat["range_pos20"] < 0.18, -1, 0)),
        index=feat.index,
    )
    votes["MACD_Cross"] = np.sign(feat["macd_hist"])
    votes["Bollinger_Reversal"] = pd.Series(
        np.where(feat["bb_pos"] < 0.12, 1, np.where(feat["bb_pos"] > 0.88, -1, 0)),
        index=feat.index,
    )
    votes["Bollinger_Breakout"] = pd.Series(
        np.where((feat["bb_pos"] > 0.92) & (feat["mom3"] > 0), 1, np.where((feat["bb_pos"] < 0.08) & (feat["mom3"] < 0), -1, 0)),
        index=feat.index,
    )
    votes["Wick_Reversal"] = pd.Series(
        np.where((feat["wick_down"] > 0.55) & (feat["rsi14"] < 45), 1, np.where((feat["wick_up"] > 0.55) & (feat["rsi14"] > 55), -1, 0)),
        index=feat.index,
    )
    votes["Trend_Continuation"] = np.sign(feat["trend_10"] + feat["trend_20"] + feat["trend_50"])
    votes["RSI_EMA15_Filter"] = pd.Series(
        np.where((feat["rsi14"] < 40) & (feat["ema15_delta"] < 0), 1, np.where((feat["rsi14"] > 60) & (feat["ema15_delta"] > 0), -1, 0)),
        index=feat.index,
    )
    votes["Volatility_Break"] = pd.Series(
        np.where((feat["atr_z"] > 1.0) & (feat["mom5"] > 0), 1, np.where((feat["atr_z"] > 1.0) & (feat["mom5"] < 0), -1, 0)),
        index=feat.index,
    )

    return {name: pd.Series(vote, index=feat.index).fillna(0).clip(-1, 1) for name, vote in votes.items()}


def backtest_strategy_weights(df: pd.DataFrame, lookback: int = 180) -> dict:
    """Son veride hangi strateji daha iyi calismis, hizli agirlik hesaplar."""
    if len(df) < 40:
        return {"weights": {}, "best": "Yetersiz veri", "score": 0.0, "combined_vote": 0.0}

    feat = add_indicators(df).tail(lookback).reset_index(drop=True)
    votes = strategy_votes(feat)
    next_dir = np.sign(feat["close"].shift(-1) - feat["close"]).fillna(0)

    weights = {}
    best_name = "Yok"
    best_score = 0.0

    for name, vote in votes.items():
        aligned = pd.DataFrame({"vote": vote.iloc[:-1], "next": next_dir.iloc[:-1]}).replace(0, np.nan).dropna()
        if len(aligned) < 12:
            score = 0.0
        else:
            score = float((np.sign(aligned["vote"]) == np.sign(aligned["next"])).mean())
            score = max(0.0, score - 0.50) * 2.0
        weights[name] = score
        if score > best_score:
            best_name = name
            best_score = score

    latest_votes = {name: float(vote.iloc[-1]) for name, vote in votes.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        combined_vote = float(np.mean(list(latest_votes.values()))) if latest_votes else 0.0
    else:
        combined_vote = sum(latest_votes[name] * weights[name] for name in latest_votes) / total_weight

    return {
        "weights": weights,
        "latest_votes": latest_votes,
        "best": best_name,
        "score": best_score,
        "combined_vote": float(np.clip(combined_vote, -1, 1)),
    }


class OneMinuteModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler() if StandardScaler else None
        self.models = []
        self.deep_model = None
        self.keras_model = None
        self.trained = False
        self.deep_trained = False
        self.keras_trained = False

    def train(self, df: pd.DataFrame) -> str:
        if RandomForestClassifier is None:
            self.trained = False
            return "sklearn yok; sadece kural motoru kullaniliyor."

        if len(df) < MIN_CANDLES:
            self.trained = False
            return f"Model icin en az {MIN_CANDLES} mum gerekir; kural motoru kullaniliyor."

        X, y = make_supervised(df)
        if len(np.unique(y)) < 2:
            self.trained = False
            return "Etiketler tek yonlu; model egitilmedi, kural motoru kullaniliyor."

        Xs = self.scaler.fit_transform(X)
        self.models = [
            RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42, class_weight="balanced"),
            GradientBoostingClassifier(random_state=42),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ]
        for model in self.models:
            model.fit(Xs, y)

        if MLPClassifier is not None:
            self.deep_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0008,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            )
            self.deep_model.fit(Xs, y)
            self.deep_trained = True

        if keras is not None and layers is not None and len(Xs) >= 120:
            self.keras_model = keras.Sequential(
                [
                    layers.Input(shape=(MODEL_FEATURE_DIM,)),
                    layers.Dense(256, activation="relu"),
                    layers.Dropout(0.20),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.15),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )
            self.keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            self.keras_model.fit(Xs, y, epochs=25, batch_size=32, verbose=0, validation_split=0.15)
            self.keras_trained = True

        self.trained = True
        dl_parts = []
        if self.deep_trained:
            dl_parts.append("MLP")
        if self.keras_trained:
            dl_parts.append("Keras")
        dl = " + derin ogrenme aktif: " + ", ".join(dl_parts) if dl_parts else ""
        return f"{len(self.models)} klasik model{dl}, son {len(df)} mumla egitildi."

    def predict_proba_up(self, df: pd.DataFrame) -> Optional[float]:
        if not self.trained or not self.models:
            return None
        feat = build_700_feature_frame(df).iloc[-1:][MODEL_FEATURES].to_numpy(dtype=float)
        xs = self.scaler.transform(feat)
        probs = []
        for model in self.models:
            p = model.predict_proba(xs)[0]
            cls = list(model.classes_)
            probs.append(float(p[cls.index(1)] if 1 in cls else 0.5))

        if self.deep_trained and self.deep_model is not None:
            p = self.deep_model.predict_proba(xs)[0]
            cls = list(self.deep_model.classes_)
            deep_prob = float(p[cls.index(1)] if 1 in cls else 0.5)
            probs.extend([deep_prob, deep_prob])

        if self.keras_trained and self.keras_model is not None:
            keras_prob = float(self.keras_model.predict(xs, verbose=0)[0][0])
            probs.extend([keras_prob, keras_prob, keras_prob])

        return float(np.mean(probs))


def rule_scores(df: pd.DataFrame) -> dict:
    feat = add_indicators(df)
    last = feat.iloc[-1]
    prev = feat.iloc[-6:-1]

    ema_fast_delta = (last["ema5"] - last["ema15"]) / max(abs(last["close"]), 1e-9)
    ema_slow_delta = (last["ema15"] - last["ema50"]) / max(abs(last["close"]), 1e-9)
    trend_score = np.tanh((ema_fast_delta + ema_slow_delta) * 250)

    last5_down = int((prev["close"].diff().dropna() < 0).sum())
    last5_up = int((prev["close"].diff().dropna() > 0).sum())
    atr_normal = abs(float(last["atr_z"])) < 2.5
    pullback_buy = 1.0 if trend_score > 0 and last5_down >= 3 and last["rsi14"] < 58 and atr_normal else 0.0
    pullback_sell = 1.0 if trend_score < 0 and last5_up >= 3 and last["rsi14"] > 42 and atr_normal else 0.0

    momentum = np.tanh((last["mom3"] + last["mom5"] + last["mom10"]) * 90)
    rsi_bias = ((50 - last["rsi14"]) / 50 + (50 - last["rsi21"]) / 50) / 2

    return {
        "rsi": float(last["rsi14"]),
        "atr_ratio": float(last["atr_ratio"]),
        "ema_fast_delta": float(ema_fast_delta),
        "ema_slow_delta": float(ema_slow_delta),
        "trend_score": float(trend_score),
        "momentum": float(momentum),
        "rsi_bias": float(rsi_bias),
        "pullback_buy": float(pullback_buy),
        "pullback_sell": float(pullback_sell),
        "pullback_score": float(pullback_buy - pullback_sell),
    }


def generate_signal(
    asset: str,
    candles: pd.DataFrame,
    model: OneMinuteModel,
    min_conf: int,
    max_atr_ratio: float,
    news_score: float = 0.0,
    telegram_score: float = 0.0,
    data_source: str = "",
) -> SignalResult:
    if len(candles) < 30:
        return SignalResult(asset, "NO-TRADE", 0, "Yetersiz mum verisi", 0, 50, 0, 0, 0, 0, 0, "Yetersiz veri", 0, 0, 0, data_source, 0.5, "NO_DATA", "WAIT", now_str())

    df = normalize_candles(candles)
    scores = rule_scores(df)
    model_up = model.predict_proba_up(df)
    strategy = backtest_strategy_weights(df)

    if model_up is None:
        model_vote = 0.0
        model_probability = 0.5
    else:
        model_vote = (model_up - 0.5) * 2
        model_probability = float(model_up)

    raw = (
        0.38 * scores["trend_score"]
        + 0.24 * scores["momentum"]
        + 0.32 * scores["rsi_bias"]
        + 0.20 * scores["pullback_score"]
        + 0.35 * model_vote
        + 0.28 * strategy["combined_vote"]
        + 0.10 * news_score
        + 0.10 * telegram_score
    )

    if scores["rsi"] <= 30:
        raw = max(raw, 0.18)
    elif scores["rsi"] >= 70:
        raw = min(raw, -0.18)

    raw = float(np.clip(raw, -1, 1))

    confidence = int(round(50 + abs(raw) * 49))
    signal = "BUY" if raw > 0 else "SELL"
    reason = f"RSI zorunlu + EMA15/ATR + 700F deep + strateji testi ({strategy['best']})"

    if scores["atr_ratio"] > max_atr_ratio:
        signal = "NO-TRADE"
        confidence = min(confidence, 60)
        reason = "Volatilite filtresi: ATR yuksek"
    elif confidence < min_conf:
        signal = "NO-TRADE"
        reason = "Guven skoru dusuk"
    elif scores["rsi"] >= 78 and signal == "BUY":
        signal = "NO-TRADE"
        confidence = min(confidence, 62)
        reason = "RSI asiri alim bolgesinde BUY engellendi"
    elif scores["rsi"] <= 22 and signal == "SELL":
        signal = "NO-TRADE"
        confidence = min(confidence, 62)
        reason = "RSI asiri satim bolgesinde SELL engellendi"

    risk_status = "OK"
    if len(df) < MIN_CANDLES:
        risk_status = "LOW_DATA"
    elif scores["atr_ratio"] > max_atr_ratio:
        risk_status = "HIGH_VOL"
    elif abs(news_score) > 0.35 and np.sign(news_score) != np.sign(raw):
        risk_status = "NEWS_CONFLICT"
    elif abs(telegram_score) > 0.35 and np.sign(telegram_score) != np.sign(raw):
        risk_status = "TELEGRAM_CONFLICT"

    action = enterprise_action(signal, confidence, risk_status)

    return SignalResult(
        asset=asset,
        signal=signal,
        confidence=confidence,
        reason=reason,
        last_price=float(df["close"].iloc[-1]),
        rsi=scores["rsi"],
        ema_fast_delta=scores["ema_fast_delta"],
        ema_slow_delta=scores["ema_slow_delta"],
        atr_ratio=scores["atr_ratio"],
        trend_score=scores["trend_score"],
        pullback_score=scores["pullback_score"],
        best_strategy=strategy["best"],
        strategy_score=float(strategy["score"]),
        news_score=float(news_score),
        telegram_score=float(telegram_score),
        data_source=data_source,
        model_probability=model_probability,
        risk_status=risk_status,
        action=action,
        timestamp=now_str(),
    )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def audit_event(event: str, payload: dict) -> None:
    record = {"time": now_str(), "event": event, "payload": payload}
    try:
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def save_enterprise_config(config: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        audit_event("config_saved", config)
    except Exception:
        pass


def load_enterprise_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def health_report(model: OneMinuteModel, candles: pd.DataFrame, data_source: str) -> dict:
    return {
        "Data source": data_source or "Not loaded",
        "Candles": int(len(candles)),
        "Sklearn": "OK" if RandomForestClassifier is not None else "Missing",
        "MLP": "OK" if MLPClassifier is not None else "Missing",
        "TensorFlow": "OK" if keras is not None else "Optional missing",
        "Classic model": "Trained" if getattr(model, "trained", False) else "Not trained",
        "Deep MLP": "Trained" if getattr(model, "deep_trained", False) else "Not trained",
        "Keras": "Trained" if getattr(model, "keras_trained", False) else "Not trained",
        "Excel": str(EXPORT_FILE.resolve()),
        "Audit": str(AUDIT_FILE.resolve()),
    }


def enterprise_action(signal: str, confidence: int, risk_status: str) -> str:
    if signal == "NO-TRADE":
        return "WAIT"
    if risk_status != "OK":
        return "REVIEW"
    if confidence >= 82:
        return "PRIORITY_SIGNAL"
    return "STANDARD_SIGNAL"


def result_to_dict(r: SignalResult) -> dict:
    return {
        "Time": r.timestamp,
        "Asset": r.asset,
        "Signal": r.signal,
        "Confidence": r.confidence,
        "Reason": r.reason,
        "Last_Price": round(r.last_price, 8),
        "RSI14": round(r.rsi, 2),
        "EMA5_21_Delta": round(r.ema_fast_delta, 8),
        "EMA21_50_Delta": round(r.ema_slow_delta, 8),
        "ATR_Ratio": round(r.atr_ratio, 8),
        "Trend_Score": round(r.trend_score, 4),
        "Pullback_Score": round(r.pullback_score, 4),
        "Best_Strategy": r.best_strategy,
        "Strategy_Score": round(r.strategy_score, 4),
        "News_Score": round(r.news_score, 4),
        "Telegram_Score": round(r.telegram_score, 4),
        "Data_Source": r.data_source,
        "Model_Probability": round(r.model_probability, 4),
        "Risk_Status": r.risk_status,
        "Action": r.action,
    }


def append_excel(row: dict, strategies: Optional[pd.DataFrame] = None, health: Optional[dict] = None) -> None:
    new = pd.DataFrame([row])
    if EXPORT_FILE.exists():
        try:
            old = pd.read_excel(EXPORT_FILE, sheet_name="Signals")
        except Exception:
            old = pd.read_excel(EXPORT_FILE)
        out = pd.concat([old, new], ignore_index=True).tail(1000)
    else:
        out = new
    with pd.ExcelWriter(EXPORT_FILE, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Signals", index=False)
        if strategies is not None and not strategies.empty:
            strategies.to_excel(writer, sheet_name="Strategies", index=False)
        if health is not None:
            pd.DataFrame([health]).to_excel(writer, sheet_name="Health", index=False)
    audit_event("excel_saved", {"file": str(EXPORT_FILE), "asset": row.get("Asset"), "signal": row.get("Signal")})


def render_signal_badge(signal: str) -> None:
    if signal == "BUY":
        st.success("BUY")
    elif signal == "SELL":
        st.error("SELL")
    else:
        st.warning("NO-TRADE")


def ensure_model_fields(model: OneMinuteModel) -> OneMinuteModel:
    """Eski Streamlit session objeleri yeni alanlari tasimayabilir."""
    if not hasattr(model, "models"):
        model.models = []
    if not hasattr(model, "deep_model"):
        model.deep_model = None
    if not hasattr(model, "keras_model"):
        model.keras_model = None
    if not hasattr(model, "trained"):
        model.trained = False
    if not hasattr(model, "deep_trained"):
        model.deep_trained = False
    if not hasattr(model, "keras_trained"):
        model.keras_trained = False
    if not hasattr(model, "scaler"):
        model.scaler = StandardScaler() if StandardScaler else None
    return model


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem;}
        .metric-card {
            border: 1px solid rgba(120,120,120,.22);
            border-radius: 8px;
            padding: 14px 16px;
            background: rgba(255,255,255,.035);
        }
        .hero-title {font-size: 2.1rem; font-weight: 800; margin-bottom: .1rem;}
        .hero-sub {opacity: .78; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='hero-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>700 ozellikli deep ensemble, EMA15/RSI/ATR, haber, Telegram ve strateji backtest paneli.</div>",
        unsafe_allow_html=True,
    )

    if "model" not in st.session_state:
        st.session_state.model = OneMinuteModel()
    st.session_state.model = ensure_model_fields(st.session_state.model)
    if "candles" not in st.session_state:
        st.session_state.candles = pd.DataFrame()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_scan" not in st.session_state:
        st.session_state.last_scan = 0.0
    if "headlines" not in st.session_state:
        st.session_state.headlines = []
    if "telegram_texts" not in st.session_state:
        st.session_state.telegram_texts = []
    if "data_source" not in st.session_state:
        st.session_state.data_source = ""
    if "did_initial_load" not in st.session_state:
        st.session_state.did_initial_load = False

    with st.sidebar:
        st.header("Ayarlar")
        asset = st.text_input("Varlik adi", value="EUR/USD")
        source = st.selectbox("Veri kaynagi", ["Yahoo 4Y", "Yahoo 1Y", "Yahoo", "Binance 4Y", "Binance 1Y", "Binance", "Binomo/HTTP", "CSV"], index=0)
        min_conf = st.slider("Minimum guven", 50, 95, DEFAULT_MIN_CONF)
        max_atr = st.number_input("Maks ATR orani", min_value=0.0001, max_value=0.2, value=DEFAULT_MAX_ATR, step=0.0005, format="%.4f")
        endpoint = st.text_input("Binomo/HTTP mum endpoint", value="")
        uploaded = st.file_uploader("CSV mum verisi yukle", type=["csv"])
        st.divider()
        news_api_key = st.text_input("NewsAPI key", value=NEWS_API_KEY, type="password")
        use_news = st.toggle("Haber cek", value=False)
        st.divider()
        telegram_token = st.text_input("Telegram bot token", value=TELEGRAM_BOT_TOKEN, type="password")
        telegram_chat_id = st.text_input("Telegram chat id", value=TELEGRAM_CHAT_ID)
        telegram_bridge = st.text_input("Telegram bridge endpoint", value="")
        use_telegram = st.toggle("Telegram cek", value=False)
        st.divider()
        auto = st.toggle("90 saniyede otomatik tara", value=False)

        load_btn = st.button("Veriyi Yukle / Yenile", use_container_width=True)
        train_btn = st.button("Modeli Egit", use_container_width=True)
        scan_btn = st.button("Sinyal Uret", use_container_width=True)

    if uploaded is not None and load_btn and source == "CSV":
        st.session_state.candles = normalize_candles(pd.read_csv(uploaded))
        st.session_state.data_source = "CSV"
        st.success(f"CSV yuklendi: {len(st.session_state.candles)} mum")

    if load_btn and source != "CSV":
        try:
            st.session_state.candles, st.session_state.data_source = fetch_market_candles(source, asset, endpoint)
            st.success(f"{st.session_state.data_source} okundu: {len(st.session_state.candles)} mum")
            audit_event("data_loaded", {"source": st.session_state.data_source, "candles": len(st.session_state.candles)})
        except Exception as exc:
            st.error(f"Veri okunamadi: {exc}")

    if not st.session_state.did_initial_load and source != "CSV" and st.session_state.candles.empty:
        try:
            with st.spinner("Ilk veri seti yukleniyor..."):
                st.session_state.candles, st.session_state.data_source = fetch_market_candles(source, asset, endpoint)
            st.session_state.did_initial_load = True
            audit_event("initial_data_loaded", {"source": st.session_state.data_source, "candles": len(st.session_state.candles)})
        except Exception as exc:
            st.session_state.did_initial_load = True
            st.warning(f"Ilk veri yuklenemedi: {exc}")

    candles = st.session_state.candles

    if train_btn:
        msg = st.session_state.model.train(candles)
        st.info(msg)

    should_auto_scan = auto and (time.time() - st.session_state.last_scan >= SCAN_SECONDS)
    if scan_btn or should_auto_scan:
        st.session_state.last_scan = time.time()
        if source != "CSV":
            try:
                st.session_state.candles, st.session_state.data_source = refresh_candles_for_scan(st.session_state.candles, source, asset, endpoint)
                candles = st.session_state.candles
            except Exception as exc:
                st.warning(f"Yeni veri cekilemedi, mevcut veriyle devam: {exc}")
        news_score, headlines = fetch_news(asset, news_api_key) if use_news else (0.0, [])
        telegram_texts = fetch_telegram_texts(telegram_token, telegram_chat_id, telegram_bridge) if use_telegram else []
        telegram_score = text_sentiment_score(telegram_texts)
        st.session_state.headlines = headlines
        st.session_state.telegram_texts = telegram_texts
        result = generate_signal(
            asset,
            candles,
            st.session_state.model,
            min_conf,
            max_atr,
            news_score=news_score,
            telegram_score=telegram_score,
            data_source=st.session_state.data_source,
        )
        row = result_to_dict(result)
        st.session_state.history.insert(0, row)
        try:
            strategy_state = backtest_strategy_weights(candles) if not candles.empty else {"weights": {}, "latest_votes": {}}
            strategy_rows = [
                {
                    "Strategy": name,
                    "Weight": round(float(weight), 4),
                    "Latest_Vote": round(float(strategy_state.get("latest_votes", {}).get(name, 0)), 2),
                }
                for name, weight in sorted(strategy_state.get("weights", {}).items(), key=lambda x: x[1], reverse=True)
            ]
            append_excel(row, pd.DataFrame(strategy_rows), health_report(st.session_state.model, candles, st.session_state.data_source))
        except Exception as exc:
            st.warning(f"Excel yazilamadi: {exc}")

    top1, top2, top3, top4, top5 = st.columns(5)
    last_result = st.session_state.history[0] if st.session_state.history else None

    with top1:
        st.metric("Mum sayisi", len(candles))
    with top2:
        if getattr(st.session_state.model, "keras_trained", False):
            model_state = "Keras + MLP"
        elif getattr(st.session_state.model, "deep_trained", False):
            model_state = "MLP 700F"
        else:
            model_state = "Klasik aktif" if getattr(st.session_state.model, "trained", False) else "Kural motoru"
        st.metric("Model", model_state)
    with top3:
        st.metric("Son fiyat", "-" if candles.empty else f"{candles['close'].iloc[-1]:.8f}")
    with top4:
        st.metric("Son tarama", "-" if not last_result else last_result["Time"])
    with top5:
        st.metric("Kaynak", st.session_state.data_source or "-")

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Son Sinyal")
        if last_result:
            render_signal_badge(last_result["Signal"])
            st.metric("Guven", f"{last_result['Confidence']}%")
            st.write(last_result["Reason"])
            st.json({k: last_result[k] for k in ["RSI14", "ATR_Ratio", "Trend_Score", "Pullback_Score", "Best_Strategy", "Strategy_Score", "News_Score", "Telegram_Score"]})
        else:
            st.info("Veri yukleyip Sinyal Uret'e basin.")

    with right:
        st.subheader("Mum Grafiği")
        if not candles.empty:
            chart_df = candles.set_index("time")[["close"]].tail(160)
            st.line_chart(chart_df)
            st.dataframe(candles.tail(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Henuz mum verisi yok.")

    st.subheader("Strateji Testleri")
    if not candles.empty and len(candles) >= 40:
        strategy_state = backtest_strategy_weights(candles)
        weight_rows = [
            {
                "Strategy": name,
                "Weight": round(float(weight), 4),
                "Latest_Vote": round(float(strategy_state.get("latest_votes", {}).get(name, 0)), 2),
            }
            for name, weight in sorted(strategy_state.get("weights", {}).items(), key=lambda x: x[1], reverse=True)
        ]
        st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Strateji testi icin en az 40 mum gerekir.")

    info1, info2 = st.columns(2)
    with info1:
        st.subheader("Haber")
        if st.session_state.headlines:
            for h in st.session_state.headlines:
                st.write(f"- {h}")
        else:
            st.caption("Haber cekilmedi veya sonuc yok.")
    with info2:
        st.subheader("Telegram")
        if st.session_state.telegram_texts:
            for t in st.session_state.telegram_texts[-5:]:
                st.write(f"- {t[:220]}")
        else:
            st.caption("Telegram cekilmedi veya bot/bridge sonucu yok.")

    st.divider()
    st.subheader("Sinyal Gecmisi")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    else:
        st.info("Gecmis bos.")

    if auto:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
