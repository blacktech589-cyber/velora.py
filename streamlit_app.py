from __future__ import annotations

# ============================================================
# BINANCE AUTO MARKET DEEP-AI SCALPER — SINGLE FILE
# ============================================================
# Spot + USDⓈ-M Futures
# 200 features
# Multi-branch deep model:
#   TCN + BiLSTM/BiGRU + Transformer + self-attention + learned gating
# MC-dropout uncertainty
# Automatic Spot/Futures routing
# Percent-profit scalping take-profit
#
# Streamlit Cloud requirements.txt:
#   streamlit>=1.38
#   pandas>=2.1
#   numpy>=1.26
#   requests>=2.31
#   torch
#
# Run:
#   streamlit run streamlit_app.py
#
# IMPORTANT:
# - 451 = legal/regional HTTP restriction; this app does not bypass it.
# - Live trading is OFF by default.
# - Futures can liquidate leveraged positions.
# - AI does not guarantee profit.
# ============================================================

import json
import math
import time
import hmac
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------------- TORCH --------------------------
TORCH_OK = True
TORCH_ERROR = ""
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    TORCH_OK = False
    TORCH_ERROR = str(exc)

# -------------------------- CONFIG --------------------------
st.set_page_config(
    page_title="Binance Auto-Market Deep AI Scalper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COUNT = 200

SPOT_BASES = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
FUTURES_BASE = "https://fapi.binance.com"

RET_H = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
WIN = [5, 8, 13, 21, 34, 55, 89, 144]
RSI_P = [5, 7, 9, 14, 21, 28, 35, 50]
ATR_P = [5, 7, 10, 14, 21, 28, 35, 50]
STO_P = [5, 7, 9, 14, 21, 28, 35, 50]
BB_P = [10, 14, 20, 28, 35, 50, 75, 100]
VOL_P = [5, 8, 13, 21, 34, 55, 89, 144]

# -------------------------- STATE --------------------------
DEFAULTS = {
    "api_key": "",
    "api_secret": "",
    "api_ok": False,
    "spot_allowed": False,
    "futures_allowed": False,
    "api_permissions": {},
    "account": None,
    "paper_balance": 1000.0,
    "positions": {},
    "trade_log": [],
    "scan_rows": [],
    "training_log": [],
    "last_network_diag": {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------- STYLE --------------------------
st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;padding-bottom:2rem}
    [data-testid="stSidebar"]{background:#0b111c}
    .hero{
      padding:22px;border-radius:22px;
      background:linear-gradient(135deg,#0f172a,#172554);
      border:1px solid #334155;margin-bottom:14px
    }
    .hero h1{margin:0;font-size:31px}
    .hero p{margin:.35rem 0 0;opacity:.76}
    .ok{color:#66e6ae;font-weight:800}
    .no{color:#ff8fa3;font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HTTP / 451 HANDLING
# ============================================================

class Binance451Error(RuntimeError):
    pass


def _json_or_text(resp):
    try:
        return resp.json()
    except Exception:
        return resp.text


def _raise_for_binance(resp, label="Binance"):
    if resp.status_code == 451:
        raise Binance451Error(
            f"{label} HTTP 451: servis bu ağ/bölge için yasal nedenle erişimi reddetti. "
            "Bu bir API-key hatası değildir ve kod içinde bypass edilmez."
        )
    if not resp.ok:
        raise RuntimeError(
            f"{label} HTTP {resp.status_code}: {_json_or_text(resp)}"
        )


def spot_public_request(path: str, params=None, timeout=15):
    """
    Official Spot endpoint failover for connection/5xx issues.
    451 is treated as a legal restriction and is NOT bypassed.
    """
    last_error = None
    for base in SPOT_BASES:
        try:
            r = requests.get(base + path, params=params or {}, timeout=timeout)
            if r.status_code == 451:
                raise Binance451Error(
                    f"Spot HTTP 451 on {base}: legal/regional access restriction."
                )
            if r.status_code >= 500:
                last_error = RuntimeError(
                    f"{base} HTTP {r.status_code}"
                )
                continue
            _raise_for_binance(r, "Spot")
            return r.json(), base
        except Binance451Error:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Tüm resmi Spot uç noktaları başarısız: {last_error}"
    )


def futures_public_request(path: str, params=None, timeout=15):
    try:
        r = requests.get(
            FUTURES_BASE + path,
            params=params or {},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Futures bağlantı hatası: {exc}") from exc
    _raise_for_binance(r, "Futures")
    return r.json()

# ============================================================
# SIGNING / CLIENTS
# ============================================================

def sign_query(secret: str, payload: dict) -> str:
    query = urlencode(payload, doseq=True, safe="")
    signature = hmac.new(
        secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return query, signature


class SpotClient:
    def __init__(self, api_key="", api_secret=""):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        self.base = None

    def server_time(self):
        data, base = spot_public_request("/api/v3/time")
        self.base = base
        return int(data["serverTime"])

    def public(self, path, params=None):
        data, base = spot_public_request(path, params=params)
        self.base = base
        return data

    def signed(self, method, path, params=None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API Key / Secret Key eksik.")

        ts = self.server_time()
        payload = dict(params or {})
        payload["timestamp"] = ts
        payload["recvWindow"] = 5000

        query, signature = sign_query(self.api_secret, payload)
        url = f"{self.base}{path}?{query}&signature={signature}"

        r = requests.request(
            method,
            url,
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=20,
        )
        _raise_for_binance(r, "Spot signed API")
        return r.json()

    def account(self):
        return self.signed("GET", "/api/v3/account")

    def restrictions(self):
        return self.signed("GET", "/sapi/v1/account/apiRestrictions")

    def klines(self, symbol, interval="1m", limit=1000, end_time=None):
        p = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if end_time is not None:
            p["endTime"] = int(end_time)
        return self.public("/api/v3/klines", p)

    def ticker24(self, symbol):
        return self.public("/api/v3/ticker/24hr", {"symbol": symbol.upper()})

    def book_ticker(self, symbol):
        return self.public("/api/v3/ticker/bookTicker", {"symbol": symbol.upper()})

    def exchange_info(self, symbol):
        return self.public("/api/v3/exchangeInfo", {"symbol": symbol.upper()})

    def _symbol_filters(self, symbol):
        info = self.exchange_info(symbol)
        s = info["symbols"][0]
        return {f["filterType"]: f for f in s["filters"]}

    @staticmethod
    def _floor_step(v, step):
        return math.floor(v / step) * step if step > 0 else v

    def normalize_qty(self, symbol, qty):
        lot = self._symbol_filters(symbol).get("LOT_SIZE", {})
        step = float(lot.get("stepSize", "0.00000001"))
        min_qty = float(lot.get("minQty", "0"))
        q = self._floor_step(float(qty), step)
        if q < min_qty:
            raise RuntimeError(f"Spot qty minQty altında: {q} < {min_qty}")
        return q

    def market_buy_quote(self, symbol, usdt):
        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{float(usdt):.8f}",
            },
        )

    def market_sell_qty(self, symbol, qty):
        q = self.normalize_qty(symbol, qty)
        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{q:.12f}",
            },
        )


class FuturesClient:
    def __init__(self, api_key="", api_secret=""):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()

    def server_time(self):
        return int(futures_public_request("/fapi/v1/time")["serverTime"])

    def public(self, path, params=None):
        return futures_public_request(path, params=params)

    def signed(self, method, path, params=None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API Key / Secret Key eksik.")

        payload = dict(params or {})
        payload["timestamp"] = self.server_time()
        payload["recvWindow"] = 5000

        query, signature = sign_query(self.api_secret, payload)
        url = f"{FUTURES_BASE}{path}?{query}&signature={signature}"

        r = requests.request(
            method,
            url,
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=20,
        )
        _raise_for_binance(r, "USDⓈ-M Futures")
        return r.json()

    def klines(self, symbol, interval="1m", limit=1000, end_time=None):
        p = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if end_time is not None:
            p["endTime"] = int(end_time)
        return self.public("/fapi/v1/klines", p)

    def ticker24(self, symbol):
        return self.public("/fapi/v1/ticker/24hr", {"symbol": symbol.upper()})

    def book_ticker(self, symbol):
        return self.public("/fapi/v1/ticker/bookTicker", {"symbol": symbol.upper()})

    def premium_index(self, symbol):
        return self.public("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})

    def exchange_info(self):
        return self.public("/fapi/v1/exchangeInfo")

    def _symbol_filters(self, symbol):
        info = self.exchange_info()
        found = [s for s in info["symbols"] if s["symbol"] == symbol.upper()]
        if not found:
            raise RuntimeError(f"Futures symbol bulunamadı: {symbol}")
        s = found[0]
        return {f["filterType"]: f for f in s["filters"]}

    @staticmethod
    def _floor_step(v, step):
        return math.floor(v / step) * step if step > 0 else v

    def normalize_qty(self, symbol, qty):
        filters = self._symbol_filters(symbol)
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
        step = float(lot.get("stepSize", "0.001"))
        min_qty = float(lot.get("minQty", "0"))
        q = self._floor_step(float(qty), step)
        if q < min_qty:
            raise RuntimeError(f"Futures qty minQty altında: {q} < {min_qty}")
        return q

    def set_leverage(self, symbol, leverage):
        return self.signed(
            "POST",
            "/fapi/v1/leverage",
            {
                "symbol": symbol.upper(),
                "leverage": int(leverage),
            },
        )

    def market_open(self, symbol, direction, qty):
        side = "BUY" if direction == "LONG" else "SELL"
        q = self.normalize_qty(symbol, qty)
        return self.signed(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol.upper(),
                "side": side,
                "type": "MARKET",
                "quantity": f"{q:.12f}",
            },
        )

    def market_close(self, symbol, direction, qty):
        side = "SELL" if direction == "LONG" else "BUY"
        q = self.normalize_qty(symbol, qty)
        return self.signed(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol.upper(),
                "side": side,
                "type": "MARKET",
                "quantity": f"{q:.12f}",
                "reduceOnly": "true",
            },
        )

# ============================================================
# DATA
# ============================================================

def klines_to_df(rows):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base",
        "taker_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_base", "taker_quote",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("open_time")


def fetch_history(market, symbol, interval, total):
    client = SpotClient() if market == "SPOT" else FuturesClient()
    rows = []
    end_time = None

    while len(rows) < total:
        lim = min(1000, total - len(rows))
        batch = client.klines(symbol, interval, lim, end_time=end_time)
        if not batch:
            break
        rows = batch + rows
        end_time = int(batch[0][0]) - 1
        time.sleep(0.05)

    rows = rows[-total:]
    if not rows:
        raise RuntimeError("Geçmiş veri alınamadı.")
    return klines_to_df(rows)

# ============================================================
# 200 FEATURES
# ============================================================

def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    return a / b


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def rsi(s, p):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    rs = safe_div(up, dn)
    return 100 - 100 / (1 + rs)


def atr(df, p):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()


def zscore(s, p):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return safe_div(s - m, sd)


def build_features(df):
    req = {"open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    x = df.copy().sort_index()
    out = pd.DataFrame(index=x.index)

    o, h, l, c, v = x["open"], x["high"], x["low"], x["close"], x["volume"]
    qv = x["quote_volume"] if "quote_volume" in x.columns else c * v
    trades = x["trades"] if "trades" in x.columns else pd.Series(0.0, index=x.index)
    taker = x["taker_base"] if "taker_base" in x.columns else pd.Series(0.0, index=x.index)

    # 20 return features
    for n in RET_H:
        out[f"ret_{n}"] = c.pct_change(n)
    logc = np.log(c.replace(0, np.nan))
    for n in RET_H:
        out[f"logret_{n}"] = logc.diff(n)

    # 16 trend
    for w in WIN:
        out[f"sma_dev_{w}"] = safe_div(c, c.rolling(w).mean()) - 1
    for w in WIN:
        out[f"ema_dev_{w}"] = safe_div(c, ema(c, w)) - 1

    # 16 volatility / z
    r1 = c.pct_change()
    for w in WIN:
        out[f"ret_std_{w}"] = r1.rolling(w).std()
    for w in WIN:
        out[f"close_z_{w}"] = zscore(c, w)

    # 16 channel
    for w in WIN:
        lo = l.rolling(w).min()
        hi = h.rolling(w).max()
        width = (hi - lo).replace(0, np.nan)
        out[f"range_pos_{w}"] = (c - lo) / width
        out[f"range_width_{w}"] = width / c.replace(0, np.nan)

    # 8 RSI
    for p in RSI_P:
        out[f"rsi_{p}"] = rsi(c, p) / 100

    # 8 ATR
    for p in ATR_P:
        out[f"atr_pct_{p}"] = atr(x, p) / c.replace(0, np.nan)

    # 16 stochastic
    for p in STO_P:
        lo = l.rolling(p).min()
        hi = h.rolling(p).max()
        k = (c - lo) / (hi - lo).replace(0, np.nan)
        out[f"stoch_k_{p}"] = k
        out[f"stoch_d_{p}"] = k.rolling(3).mean()

    # 16 Bollinger
    for p in BB_P:
        m = c.rolling(p).mean()
        sd = c.rolling(p).std()
        up = m + 2*sd
        dn = m - 2*sd
        out[f"bb_pos_{p}"] = (c-dn)/(up-dn).replace(0, np.nan)
        out[f"bb_width_{p}"] = (up-dn)/m.replace(0, np.nan)

    # 24 volume
    for p in VOL_P:
        out[f"vol_z_{p}"] = zscore(v, p)
        out[f"vol_ratio_{p}"] = safe_div(v, v.rolling(p).mean())
        out[f"qvol_ratio_{p}"] = safe_div(qv, qv.rolling(p).mean())

    # 10 ROC
    for n in RET_H:
        out[f"roc_{n}"] = c/c.shift(n).replace(0, np.nan)-1

    # 10 candle geometry
    rng = (h-l).replace(0, np.nan)
    body = c-o
    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    min_oc = pd.concat([o, c], axis=1).min(axis=1)
    out["body_pct"] = body/o.replace(0, np.nan)
    out["body_to_range"] = body/rng
    out["abs_body_to_range"] = body.abs()/rng
    out["upper_wick_ratio"] = (h-max_oc)/rng
    out["lower_wick_ratio"] = (min_oc-l)/rng
    out["close_location"] = (c-l)/rng
    out["open_location"] = (o-l)/rng
    out["gap_pct"] = o/c.shift(1).replace(0, np.nan)-1
    out["hl_pct"] = (h-l)/c.replace(0, np.nan)
    out["oc_abs_pct"] = body.abs()/o.replace(0, np.nan)

    # 12 microstructure proxies
    signed = np.sign(c.diff()).fillna(0)
    obv = (signed*v).cumsum()
    for w in [5, 13, 21, 34]:
        out[f"obv_z_{w}"] = zscore(obv, w)
    for w in [5, 13, 21, 34]:
        out[f"trade_z_{w}"] = zscore(trades, w)
    taker_ratio = safe_div(taker, v)
    for w in [5, 13, 21, 34]:
        out[f"taker_ratio_ma_{w}"] = taker_ratio.rolling(w).mean()

    # 12 MACD
    for fast, slow in [(5,13),(8,21),(12,26),(13,34),(21,55),(34,89)]:
        macd = ema(c, fast) - ema(c, slow)
        sig = ema(macd, 9)
        out[f"macd_norm_{fast}_{slow}"] = macd/c.replace(0, np.nan)
        out[f"macd_hist_{fast}_{slow}"] = (macd-sig)/c.replace(0, np.nan)

    # 8 acceleration
    for n in [1,2,3,5,8,13,21,34]:
        rr = c.pct_change(n)
        out[f"accel_{n}"] = rr - rr.shift(n)

    # 6 time
    if isinstance(out.index, pd.DatetimeIndex):
        minute = out.index.minute + out.index.hour*60
        dow = out.index.dayofweek
        out["tod_sin"] = np.sin(2*np.pi*minute/1440)
        out["tod_cos"] = np.cos(2*np.pi*minute/1440)
        out["dow_sin"] = np.sin(2*np.pi*dow/7)
        out["dow_cos"] = np.cos(2*np.pi*dow/7)
        out["hour_sin"] = np.sin(2*np.pi*out.index.hour/24)
        out["hour_cos"] = np.cos(2*np.pi*out.index.hour/24)
    else:
        for name in ["tod_sin","tod_cos","dow_sin","dow_cos","hour_sin","hour_cos"]:
            out[name] = 0.0

    # two explicit interaction features
    out["trend_volume_interaction"] = out["ema_dev_21"] * out["vol_ratio_21"]
    out["momentum_vol_interaction"] = out["ret_5"] * out["atr_pct_14"]

    # exact 200 safety
    base_cols = list(out.columns)
    i = 0
    while out.shape[1] < FEATURE_COUNT:
        a = base_cols[i % len(base_cols)]
        b = base_cols[(i*7+11) % len(base_cols)]
        out[f"cross_{i:03d}"] = out[a] * out[b]
        i += 1

    out = out.iloc[:, :FEATURE_COUNT]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().fillna(0).clip(-1000, 1000).astype("float32")

    if out.shape[1] != FEATURE_COUNT:
        raise RuntimeError(f"Feature count mismatch: {out.shape[1]}")
    return out

# ============================================================
# SMART MULTI-BRANCH MODEL
# ============================================================

if TORCH_OK:

    class TemporalResidual(nn.Module):
        def __init__(self, channels, kernel, dilation, dropout):
            super().__init__()
            pad = ((kernel-1)*dilation)//2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.drop(self.net(x) + x)


    class SmartMarketAI(nn.Module):
        """
        Multi-branch temporal fusion:
        - CNN branch learns local/scalping patterns
        - BiLSTM/BiGRU branch learns sequence state
        - Transformer branch learns long-range relations
        - learned branch gate chooses which branch matters
        - final self-attention + pooled classifier
        """
        def __init__(
            self,
            n_features=FEATURE_COUNT,
            d_model=160,
            heads=8,
            dropout=0.20,
            classes=3,
        ):
            super().__init__()

            self.norm = nn.LayerNorm(n_features)
            self.project = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, d_model),
                nn.GELU(),
            )

            self.cnn = nn.Sequential(
                TemporalResidual(d_model, 3, 1, dropout),
                TemporalResidual(d_model, 5, 2, dropout),
                TemporalResidual(d_model, 7, 4, dropout),
            )

            self.lstm = nn.LSTM(
                d_model, 112, num_layers=2, batch_first=True,
                dropout=dropout, bidirectional=True,
            )
            self.lstm_proj = nn.Linear(224, d_model)

            self.gru = nn.GRU(
                d_model, 112, num_layers=2, batch_first=True,
                dropout=dropout, bidirectional=True,
            )
            self.gru_proj = nn.Linear(224, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model*4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

            self.branch_gate = nn.Sequential(
                nn.Linear(d_model*3, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, 3),
                nn.Softmax(dim=-1),
            )

            self.attn = nn.MultiheadAttention(
                d_model, heads, dropout=dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(d_model)

            self.pool_gate = nn.Sequential(
                nn.Linear(d_model*3, 128),
                nn.GELU(),
                nn.Linear(128, 3),
                nn.Softmax(dim=-1),
            )

            self.head = nn.Sequential(
                nn.Linear(d_model*3, 384),
                nn.GELU(),
                nn.LayerNorm(384),
                nn.Dropout(dropout),
                nn.Linear(384, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, 64),
                nn.GELU(),
                nn.Linear(64, classes),
            )

        def forward(self, x):
            x = self.project(self.norm(x))

            # branch 1: TCN
            cnn = self.cnn(x.transpose(1,2)).transpose(1,2)

            # branch 2: recurrent
            lstm, _ = self.lstm(x)
            rnn = self.lstm_proj(lstm)
            gru, _ = self.gru(rnn)
            rnn = rnn + self.gru_proj(gru)

            # branch 3: transformer
            tr = self.transformer(x)

            branch_summary = torch.cat(
                [cnn.mean(1), rnn.mean(1), tr.mean(1)], dim=-1
            )
            w = self.branch_gate(branch_summary)

            fused_seq = (
                cnn * w[:,0:1,None]
                + rnn * w[:,1:2,None]
                + tr * w[:,2:3,None]
            )

            a, _ = self.attn(fused_seq, fused_seq, fused_seq, need_weights=False)
            fused_seq = self.attn_norm(fused_seq + a)

            last = fused_seq[:,-1,:]
            mean = fused_seq.mean(1)
            mx = fused_seq.max(1).values

            pg = self.pool_gate(torch.cat([last, mean, mx], dim=-1))
            pooled = torch.cat(
                [
                    last * pg[:,0:1],
                    mean * pg[:,1:2],
                    mx * pg[:,2:3],
                ],
                dim=-1,
            )
            return self.head(pooled)


    class SeqDS(Dataset):
        def __init__(self, X, y, seq):
            self.X, self.y, self.seq = X, y, seq

        def __len__(self):
            return max(0, len(self.y)-self.seq+1)

        def __getitem__(self, i):
            j = i+self.seq-1
            return (
                torch.tensor(self.X[i:i+self.seq], dtype=torch.float32),
                torch.tensor(self.y[j], dtype=torch.long),
            )

# ============================================================
# TRAIN / LOAD / MC-DROPOUT INFERENCE
# ============================================================

def labels_from_close(close, horizon, threshold):
    future = close.shift(-horizon)/close - 1
    y = np.ones(len(close), dtype=np.int64)
    y[future < -threshold] = 0
    y[future > threshold] = 2
    return y


def class_weights(y):
    counts = np.bincount(y, minlength=3).astype(float)
    counts[counts == 0] = 1
    return torch.tensor(counts.sum()/(3*counts), dtype=torch.float32)


def model_dir_for(base_dir, market):
    return Path(base_dir) / market.lower()


def train_model(
    market,
    symbol,
    interval,
    bars,
    seq,
    horizon,
    threshold,
    epochs,
    batch,
    lr,
    base_dir,
):
    if not TORCH_OK:
        raise RuntimeError("PyTorch kurulu değil.")

    out = model_dir_for(base_dir, market)
    out.mkdir(parents=True, exist_ok=True)
    st.session_state.training_log = []

    def log(msg):
        st.session_state.training_log.append(str(msg))

    log(f"{market} {symbol}: geçmiş veri indiriliyor...")
    df = fetch_history(market, symbol, interval, bars)
    feats = build_features(df)
    y = labels_from_close(df["close"], horizon, threshold)

    usable = len(df)-horizon
    feats = feats.iloc[:usable]
    y = y[:usable]

    train_end = int(len(feats)*0.70)
    val_end = int(len(feats)*0.85)

    mean = feats.iloc[:train_end].mean().to_numpy(np.float32)
    std = feats.iloc[:train_end].std().replace(0,1).to_numpy(np.float32)

    X = ((feats.to_numpy(np.float32)-mean)/std).clip(-8,8)

    tr = DataLoader(
        SeqDS(X[:train_end], y[:train_end], seq),
        batch_size=batch, shuffle=True, drop_last=True
    )
    va = DataLoader(
        SeqDS(
            X[train_end-seq:val_end],
            y[train_end-seq:val_end],
            seq
        ),
        batch_size=batch, shuffle=False
    )
    te = DataLoader(
        SeqDS(X[val_end-seq:], y[val_end-seq:], seq),
        batch_size=batch, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    model = SmartMarketAI().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=.5, patience=1, min_lr=1e-6
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(y[:train_end]).to(device),
        label_smoothing=.03,
    )

    best = float("inf")
    patience = 0

    for epoch in range(1, epochs+1):
        model.train()
        tl, tc, tn = 0.0, 0, 0

        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += loss.item()
            tc += (logits.argmax(1) == yb).sum().item()
            tn += yb.numel()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vl += loss.item()
                vc += (logits.argmax(1) == yb).sum().item()
                vn += yb.numel()

        tl /= max(1, len(tr))
        vl /= max(1, len(va))
        ta = tc/max(1,tn)
        vaa = vc/max(1,vn)
        sched.step(vl)

        log(
            f"Epoch {epoch:02d} | train={tl:.4f}/{ta:.3f} "
            f"| val={vl:.4f}/{vaa:.3f} | lr={opt.param_groups[0]['lr']:.7f}"
        )

        if vl < best:
            best = vl
            patience = 0
            torch.save(model.state_dict(), out/"model.pt")
            np.savez(out/"scaler.npz", mean=mean, std=std)
            (out/"meta.json").write_text(
                json.dumps(
                    {
                        "market": market,
                        "symbol": symbol,
                        "interval": interval,
                        "seq": seq,
                        "horizon": horizon,
                        "threshold": threshold,
                        "feature_count": FEATURE_COUNT,
                        "features": list(feats.columns),
                        "classes": ["SELL","WAIT","BUY"],
                        "architecture": "TCN+BiLSTM+BiGRU+Transformer+Attention+Gating",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log("Yeni en iyi model kaydedildi.")
        else:
            patience += 1
            if patience >= 4:
                log("Early stopping.")
                break

    model.load_state_dict(torch.load(out/"model.pt", map_location=device))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for xb, yb in te:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

    acc = correct/max(1,total)
    log(f"Out-of-sample test accuracy: {acc:.4f}")
    return {"market":market, "test_accuracy":acc, "model_dir":str(out)}


_MODEL_CACHE = {}


def load_model(base_dir, market):
    if not TORCH_OK:
        raise RuntimeError("PyTorch kurulu değil.")

    p = model_dir_for(base_dir, market)
    key = str(p.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    for name in ["model.pt","meta.json","scaler.npz"]:
        if not (p/name).exists():
            raise RuntimeError(f"{market} model dosyası yok: {p/name}")

    meta = json.loads((p/"meta.json").read_text(encoding="utf-8"))
    sc = np.load(p/"scaler.npz")

    model = SmartMarketAI(n_features=int(meta["feature_count"]))
    model.load_state_dict(torch.load(p/"model.pt", map_location="cpu"))
    model.eval()

    obj = (model, meta, sc["mean"], sc["std"])
    _MODEL_CACHE[key] = obj
    return obj


def enable_mc_dropout(model):
    """
    Keep BatchNorm in eval, enable Dropout for MC sampling.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predict_market(client, market, symbol, interval, base_dir, mc_samples=8):
    model, meta, mean, std = load_model(base_dir, market)

    df = klines_to_df(client.klines(symbol, interval, limit=1000))
    feats = build_features(df).reindex(columns=meta["features"], fill_value=0.0)

    X = ((feats.to_numpy(np.float32)-mean)/std).clip(-8,8)
    seq = int(meta["seq"])
    if len(X) < seq:
        raise RuntimeError("Yetersiz sequence.")

    t = torch.tensor(X[-seq:], dtype=torch.float32).unsqueeze(0)

    # MC-dropout uncertainty
    enable_mc_dropout(model)
    draws = []
    with torch.no_grad():
        for _ in range(max(2, int(mc_samples))):
            draws.append(torch.softmax(model(t), dim=1).squeeze(0).numpy())
    model.eval()

    arr = np.stack(draws)
    mean_prob = arr.mean(axis=0)
    variance = float(arr.var(axis=0).mean())

    safe = np.clip(mean_prob.astype(float), 1e-9, 1.0)
    entropy = float(-np.sum(safe*np.log(safe))/np.log(3.0))
    uncertainty = min(100.0, 100.0*(0.70*entropy + 0.30*min(1.0, variance*100)))
    confidence = max(0.0, 100.0-uncertainty)

    return {
        "SELL": float(mean_prob[0]*100),
        "WAIT": float(mean_prob[1]*100),
        "BUY": float(mean_prob[2]*100),
        "CONFIDENCE": confidence,
        "UNCERTAINTY": uncertainty,
    }, df

# ============================================================
# ROUTER: SPOT OR FUTURES
# ============================================================

def spread_bps(book):
    bid = float(book["bidPrice"])
    ask = float(book["askPrice"])
    mid = (bid+ask)/2
    return ((ask-bid)/mid)*10000 if mid > 0 else 9999


def realized_vol_score(df):
    r = df["close"].pct_change().tail(40).std()
    # scalping likes some movement, but not extreme chaos
    pct = float(r*100)
    if pct <= 0:
        return 0.0
    if pct < 0.05:
        return pct/0.05*35
    if pct <= 0.8:
        return 35 + min(35, (pct-0.05)/0.75*35)
    return max(10, 70-(pct-0.8)*25)


def candidate_score(
    market,
    probs,
    spread,
    vol_score,
    funding_rate=0.0,
):
    direction_prob = max(probs["BUY"], probs["SELL"])
    wait_penalty = probs["WAIT"] * 0.20
    uncertainty_penalty = probs["UNCERTAINTY"] * 0.25
    spread_penalty = min(35.0, spread * 2.0)
    funding_penalty = min(15.0, abs(funding_rate)*10000*2.0)

    score = (
        direction_prob*0.62
        + probs["CONFIDENCE"]*0.23
        + vol_score*0.15
        - wait_penalty
        - uncertainty_penalty
        - spread_penalty
        - funding_penalty
    )

    if market == "SPOT" and probs["SELL"] > probs["BUY"]:
        score -= 30  # Spot has no native short in this bot

    return float(score)


def scan_one_symbol(
    symbol,
    interval,
    base_dir,
    spot_client,
    futures_client,
    allow_spot,
    allow_futures,
    mc_samples,
):
    rows = []

    # SPOT
    if allow_spot and (model_dir_for(base_dir, "SPOT")/"model.pt").exists():
        try:
            probs, df = predict_market(
                spot_client, "SPOT", symbol, interval, base_dir, mc_samples
            )
            book = spot_client.book_ticker(symbol)
            sp = spread_bps(book)
            vs = realized_vol_score(df)
            direction = "LONG" if probs["BUY"] >= probs["SELL"] else "NONE"
            score = candidate_score("SPOT", probs, sp, vs, 0)
            rows.append(
                {
                    "Symbol":symbol,
                    "Market":"SPOT",
                    "Direction":direction,
                    "BUY %":probs["BUY"],
                    "WAIT %":probs["WAIT"],
                    "SELL %":probs["SELL"],
                    "AI Güven %":probs["CONFIDENCE"],
                    "Belirsizlik %":probs["UNCERTAINTY"],
                    "Spread bps":sp,
                    "Funding %":0.0,
                    "Vol Score":vs,
                    "Router Score":score,
                }
            )
        except Exception as exc:
            rows.append({"Symbol":symbol,"Market":"SPOT","Error":str(exc)})

    # FUTURES
    if allow_futures and (model_dir_for(base_dir, "FUTURES")/"model.pt").exists():
        try:
            probs, df = predict_market(
                futures_client, "FUTURES", symbol, interval, base_dir, mc_samples
            )
            book = futures_client.book_ticker(symbol)
            sp = spread_bps(book)
            premium = futures_client.premium_index(symbol)
            funding = float(premium.get("lastFundingRate", 0.0))
            vs = realized_vol_score(df)
            direction = "LONG" if probs["BUY"] >= probs["SELL"] else "SHORT"
            score = candidate_score("FUTURES", probs, sp, vs, funding)
            rows.append(
                {
                    "Symbol":symbol,
                    "Market":"FUTURES",
                    "Direction":direction,
                    "BUY %":probs["BUY"],
                    "WAIT %":probs["WAIT"],
                    "SELL %":probs["SELL"],
                    "AI Güven %":probs["CONFIDENCE"],
                    "Belirsizlik %":probs["UNCERTAINTY"],
                    "Spread bps":sp,
                    "Funding %":funding*100,
                    "Vol Score":vs,
                    "Router Score":score,
                }
            )
        except Exception as exc:
            rows.append({"Symbol":symbol,"Market":"FUTURES","Error":str(exc)})

    return rows

# ============================================================
# POSITION / SCALPING
# ============================================================

def position_key(market, symbol):
    return f"{market}:{symbol}"


def market_price(market, symbol, spot, futures):
    client = spot if market == "SPOT" else futures
    return float(client.ticker24(symbol)["lastPrice"])


def pnl_percent(pos, current_price):
    sign = 1.0 if pos["direction"] == "LONG" else -1.0
    price_move = sign * (current_price-pos["entry"])/pos["entry"]*100
    return price_move * float(pos.get("leverage", 1))


def estimated_pnl_usdt(pos, current_price):
    roe = pnl_percent(pos, current_price)/100
    return float(pos["margin_usdt"]) * roe


def add_paper_position(
    market,
    symbol,
    direction,
    margin_usdt,
    entry,
    leverage,
    tp_pct,
):
    key = position_key(market, symbol)
    if key in st.session_state.positions:
        raise RuntimeError("Bu market/symbol için zaten açık takip pozisyonu var.")

    notional = margin_usdt * leverage
    qty = notional / entry

    st.session_state.positions[key] = {
        "market":market,
        "symbol":symbol,
        "direction":direction,
        "margin_usdt":float(margin_usdt),
        "entry":float(entry),
        "qty":float(qty),
        "leverage":int(leverage),
        "tp_pct":float(tp_pct),
        "mode":"PAPER",
        "opened_at":str(datetime.now()),
    }


def open_live_position(
    market,
    symbol,
    direction,
    margin_usdt,
    price,
    leverage,
    tp_pct,
    spot,
    futures,
):
    key = position_key(market, symbol)
    if key in st.session_state.positions:
        raise RuntimeError("Bu market/symbol zaten takip ediliyor.")

    if market == "SPOT":
        if direction != "LONG":
            raise RuntimeError("Spot modunda SHORT açılmaz.")

        result = spot.market_buy_quote(symbol, margin_usdt)
        executed_qty = float(result.get("executedQty", 0) or 0)
        quote_qty = float(result.get("cummulativeQuoteQty", 0) or 0)
        entry = quote_qty/executed_qty if executed_qty > 0 and quote_qty > 0 else price

        st.session_state.positions[key] = {
            "market":"SPOT",
            "symbol":symbol,
            "direction":"LONG",
            "margin_usdt":float(margin_usdt),
            "entry":float(entry),
            "qty":float(executed_qty),
            "leverage":1,
            "tp_pct":float(tp_pct),
            "mode":"LIVE",
            "opened_at":str(datetime.now()),
        }
        return result

    futures.set_leverage(symbol, leverage)
    qty = (margin_usdt*leverage)/price
    q = futures.normalize_qty(symbol, qty)
    result = futures.market_open(symbol, direction, q)

    st.session_state.positions[key] = {
        "market":"FUTURES",
        "symbol":symbol,
        "direction":direction,
        "margin_usdt":float(margin_usdt),
        "entry":float(price),
        "qty":float(q),
        "leverage":int(leverage),
        "tp_pct":float(tp_pct),
        "mode":"LIVE",
        "opened_at":str(datetime.now()),
    }
    return result


def close_position(key, current, spot, futures):
    pos = st.session_state.positions[key]

    if pos["mode"] == "LIVE":
        if pos["market"] == "SPOT":
            result = spot.market_sell_qty(pos["symbol"], pos["qty"])
        else:
            result = futures.market_close(
                pos["symbol"], pos["direction"], pos["qty"]
            )
    else:
        result = {"paper": True}

    pnl_pct_value = pnl_percent(pos, current)
    pnl_usdt_value = estimated_pnl_usdt(pos, current)

    if pos["mode"] == "PAPER":
        st.session_state.paper_balance += pnl_usdt_value

    st.session_state.trade_log.append(
        {
            "time":str(datetime.now()),
            "action":"CLOSE_TP",
            "market":pos["market"],
            "symbol":pos["symbol"],
            "direction":pos["direction"],
            "entry":pos["entry"],
            "exit":current,
            "pnl_pct":pnl_pct_value,
            "pnl_usdt_est":pnl_usdt_value,
            "mode":pos["mode"],
            "result":str(result),
        }
    )

    del st.session_state.positions[key]
    return result


def monitor_take_profit(spot, futures):
    events = []
    for key in list(st.session_state.positions.keys()):
        pos = st.session_state.positions.get(key)
        if not pos:
            continue
        try:
            current = market_price(pos["market"], pos["symbol"], spot, futures)
            roe = pnl_percent(pos, current)
            if roe >= float(pos["tp_pct"]):
                result = close_position(key, current, spot, futures)
                events.append(
                    f"{pos['market']} {pos['symbol']} {pos['direction']} "
                    f"TP kapandı: %{roe:.3f}"
                )
        except Exception as exc:
            events.append(f"{key} monitor error: {exc}")
    return events

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🔐 Binance API")

    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
    )
    api_secret = st.text_input(
        "Secret Key",
        value=st.session_state.api_secret,
        type="password",
    )

    if st.button("🔌 API Bağlantısını Test Et", use_container_width=True):
        diag = {}
        try:
            spot_test = SpotClient(api_key, api_secret)
            account = spot_test.account()
            perms = spot_test.restrictions()

            st.session_state.api_key = api_key
            st.session_state.api_secret = api_secret
            st.session_state.account = account
            st.session_state.api_permissions = perms
            st.session_state.spot_allowed = bool(
                perms.get("enableSpotAndMarginTrading", False)
            )
            st.session_state.futures_allowed = bool(
                perms.get("enableFutures", False)
            )
            st.session_state.api_ok = True

            diag["spot_endpoint"] = spot_test.base
            diag["spot_signed"] = "OK"
            diag["spot_trade_permission"] = st.session_state.spot_allowed
            diag["futures_permission"] = st.session_state.futures_allowed

            try:
                FuturesClient(api_key, api_secret).server_time()
                diag["futures_public"] = "OK"
            except Exception as exc:
                diag["futures_public"] = str(exc)

            st.session_state.last_network_diag = diag
            st.success("API imzalı bağlantı başarılı.")

        except Binance451Error as exc:
            st.session_state.api_ok = False
            st.session_state.last_network_diag = {"451": str(exc)}
            st.error(str(exc))
        except Exception as exc:
            st.session_state.api_ok = False
            st.session_state.last_network_diag = {"error": str(exc)}
            st.error(str(exc))

    if st.button("Anahtarları Temizle", use_container_width=True):
        for k in [
            "api_key","api_secret","account","api_permissions"
        ]:
            st.session_state[k] = "" if k in ["api_key","api_secret"] else None
        st.session_state.api_ok = False
        st.session_state.spot_allowed = False
        st.session_state.futures_allowed = False
        st.rerun()

    st.divider()
    st.header("⚡ Scalping")

    execution_mode = st.radio(
        "Emir modu",
        ["Paper", "Live"],
        index=0,
    )

    live_confirm = False
    if execution_mode == "Live":
        live_confirm = st.checkbox(
            "Gerçek Spot/Futures emirlerini aç",
            value=False,
        )

    symbols_text = st.text_area(
        "Pariteler",
        "BTCUSDT\nETHUSDT\nBNBUSDT\nSOLUSDT\nXRPUSDT",
        height=115,
    )
    symbols = [x.strip().upper() for x in symbols_text.splitlines() if x.strip()]

    interval = st.selectbox(
        "Scalping timeframe",
        ["1m","3m","5m","15m"],
        index=0,
    )

    margin_usdt = st.number_input(
        "İşlem başına USDT",
        min_value=5.0,
        value=20.0,
        step=5.0,
    )

    tp_pct = st.number_input(
        "Kârda kapat (%)",
        min_value=0.05,
        max_value=20.0,
        value=0.50,
        step=0.05,
        format="%.2f",
        help="Futures'ta bu değer yaklaşık ROE hedefidir (fiyat hareketi × kaldıraç).",
    )

    leverage = st.slider(
        "Futures kaldıraç",
        min_value=1,
        max_value=10,
        value=2,
    )

    min_router_score = st.slider(
        "Minimum Auto-Market skor",
        0.0,
        100.0,
        35.0,
        1.0,
    )

    mc_samples = st.slider(
        "AI belirsizlik örneklemesi",
        2,
        16,
        8,
    )

    base_model_dir = st.text_input(
        "Model ana klasörü",
        "models",
    )

spot = SpotClient(st.session_state.api_key, st.session_state.api_secret)
futures = FuturesClient(st.session_state.api_key, st.session_state.api_secret)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
      <h1>Binance Auto-Market Deep AI Scalper</h1>
      <p>
      Spot + USDⓈ-M Futures • 200 özellik • TCN + BiLSTM + BiGRU +
      Transformer + Attention + uncertainty • yüzde kârda otomatik kapanış
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TORCH_OK:
    st.error(
        f"PyTorch kurulamadı: {TORCH_ERROR}. Repo kökünde requirements.txt içine `torch` ekle."
    )

# ============================================================
# KPI / PERMISSIONS
# ============================================================

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("API", "Bağlı" if st.session_state.api_ok else "Bağlı değil")
k2.metric("Spot izin", "Açık" if st.session_state.spot_allowed else "Kapalı")
k3.metric("Futures izin", "Açık" if st.session_state.futures_allowed else "Kapalı")
k4.metric("Paper bakiye", f"{st.session_state.paper_balance:.2f} USDT")
k5.metric("Takip pozisyon", len(st.session_state.positions))

# ============================================================
# TABS
# ============================================================

tab_auto, tab_scalp, tab_train, tab_positions, tab_model, tab_diag = st.tabs(
    [
        "🤖 Otomatik Spot/Futures",
        "⚡ Scalper",
        "🏋 Eğitim",
        "💼 Pozisyonlar",
        "🧠 Model",
        "🔎 API/451",
    ]
)

# ============================================================
# AUTO ROUTER
# ============================================================

with tab_auto:
    st.subheader("AI hangi piyasayı kullanacağına kendisi karar versin")

    paper_router_spot = True if execution_mode == "Paper" else st.session_state.spot_allowed
    paper_router_fut = True if execution_mode == "Paper" else st.session_state.futures_allowed

    if st.button("🧠 Spot + Futures Tara", type="primary", disabled=not TORCH_OK):
        all_rows = []
        for symbol in symbols[:15]:
            try:
                all_rows.extend(
                    scan_one_symbol(
                        symbol=symbol,
                        interval=interval,
                        base_dir=base_model_dir,
                        spot_client=spot,
                        futures_client=futures,
                        allow_spot=paper_router_spot,
                        allow_futures=paper_router_fut,
                        mc_samples=mc_samples,
                    )
                )
            except Binance451Error as exc:
                st.error(str(exc))
                break
            except Exception as exc:
                all_rows.append({"Symbol":symbol, "Market":"?", "Error":str(exc)})

        st.session_state.scan_rows = all_rows

    valid = [
        r for r in st.session_state.scan_rows
        if "Router Score" in r and r.get("Direction") not in [None, "NONE"]
    ]

    if st.session_state.scan_rows:
        df_scan = pd.DataFrame(st.session_state.scan_rows)
        if "Router Score" in df_scan.columns:
            df_scan = df_scan.sort_values("Router Score", ascending=False)
        st.dataframe(df_scan, use_container_width=True, hide_index=True)

    if valid:
        best = sorted(valid, key=lambda x: x["Router Score"], reverse=True)[0]

        st.success(
            f"En yüksek router skoru: {best['Market']} / {best['Symbol']} / "
            f"{best['Direction']} — skor {best['Router Score']:.2f}"
        )

        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Market", best["Market"])
        b2.metric("Direction", best["Direction"])
        b3.metric("AI Güven", f"%{best['AI Güven %']:.1f}")
        b4.metric("Spread", f"{best['Spread bps']:.2f} bps")

        can_open = best["Router Score"] >= min_router_score

        if not can_open:
            st.warning(
                f"Skor {min_router_score:.1f} eşiğinin altında; bot işlem açmıyor."
            )

        if st.button(
            "🚀 AI Seçimini Aç",
            disabled=not can_open,
            type="primary",
        ):
            try:
                if execution_mode == "Live":
                    if not st.session_state.api_ok or not live_confirm:
                        raise RuntimeError("Live API bağlantısı ve onayı gerekli.")
                    if best["Market"] == "SPOT" and not st.session_state.spot_allowed:
                        raise RuntimeError("API anahtarında Spot işlem izni yok.")
                    if best["Market"] == "FUTURES" and not st.session_state.futures_allowed:
                        raise RuntimeError("API anahtarında Futures izni yok.")

                price = market_price(best["Market"], best["Symbol"], spot, futures)
                lev = 1 if best["Market"] == "SPOT" else leverage

                if execution_mode == "Paper":
                    add_paper_position(
                        best["Market"], best["Symbol"], best["Direction"],
                        margin_usdt, price, lev, tp_pct
                    )
                    st.session_state.paper_balance -= margin_usdt
                    result = {"paper":True}
                else:
                    result = open_live_position(
                        best["Market"], best["Symbol"], best["Direction"],
                        margin_usdt, price, lev, tp_pct, spot, futures
                    )

                st.session_state.trade_log.append(
                    {
                        "time":str(datetime.now()),
                        "action":"OPEN",
                        "market":best["Market"],
                        "symbol":best["Symbol"],
                        "direction":best["Direction"],
                        "entry":price,
                        "margin_usdt":margin_usdt,
                        "leverage":lev,
                        "tp_pct":tp_pct,
                        "router_score":best["Router Score"],
                        "mode":execution_mode.upper(),
                        "result":str(result),
                    }
                )
                st.success("Pozisyon açıldı ve yüzde-kâr TP monitörüne eklendi.")
            except Exception as exc:
                st.error(str(exc))

# ============================================================
# SCALPER MONITOR
# ============================================================

with tab_scalp:
    st.subheader("Yüzde kâra geldiğinde kapat")

    st.write(
        f"Aktif hedef: **%{tp_pct:.2f}**. "
        "Paper pozisyonlar ve bu panel üzerinden açılan Live pozisyonlar takip edilir."
    )

    if st.button("🔄 TP'leri Şimdi Kontrol Et", use_container_width=True):
        events = monitor_take_profit(spot, futures)
        if events:
            for event in events:
                st.info(event)
        else:
            st.success("Kontrol edildi; kapanan pozisyon yok.")

    live_auto_tp = st.checkbox(
        "Panel açıkken her 3 saniyede otomatik TP kontrolü",
        value=False,
        help="Bu bir arka plan servisi değildir; Streamlit oturumu aktifken çalışır.",
    )

    if live_auto_tp and hasattr(st, "fragment"):
        @st.fragment(run_every="3s")
        def auto_tp_fragment():
            events = monitor_take_profit(spot, futures)
            if events:
                for event in events:
                    st.write(event)
            st.caption(f"Son TP kontrolü: {datetime.now().strftime('%H:%M:%S')}")
        auto_tp_fragment()

# ============================================================
# TRAINING
# ============================================================

with tab_train:
    st.subheader("Spot ve Futures modellerini ayrı eğit")

    market_train = st.radio(
        "Eğitilecek piyasa",
        ["SPOT","FUTURES"],
        horizontal=True,
    )
    train_symbol = st.text_input("Eğitim symbol", "BTCUSDT")
    t1,t2,t3 = st.columns(3)

    with t1:
        train_interval = st.selectbox(
            "Eğitim timeframe",
            ["1m","3m","5m","15m"],
            index=2,
            key="train_tf",
        )
        bars = st.number_input(
            "Mum sayısı",
            3000, 100000, 15000, 1000
        )

    with t2:
        seq = st.number_input("Sequence", 32, 256, 96, 16)
        horizon = st.number_input("Horizon (mum)", 1, 24, 3)

    with t3:
        threshold = st.number_input(
            "Label threshold",
            0.0005, 0.05, 0.0025, 0.0005,
            format="%.4f"
        )
        epochs = st.number_input("Epoch", 1, 40, 8)

    batch = st.selectbox("Batch", [32,64,128,256], index=2)
    lr = st.number_input(
        "Learning rate",
        0.00001, 0.005, 0.0002, 0.00001,
        format="%.5f",
    )

    if st.button("🏋 Modeli Eğit", disabled=not TORCH_OK, type="primary"):
        try:
            with st.spinner("Model eğitiliyor..."):
                result = train_model(
                    market_train,
                    train_symbol.upper(),
                    train_interval,
                    int(bars),
                    int(seq),
                    int(horizon),
                    float(threshold),
                    int(epochs),
                    int(batch),
                    float(lr),
                    base_model_dir,
                )
                _MODEL_CACHE.clear()
            st.success("Eğitim tamamlandı.")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.training_log:
        st.text_area(
            "Eğitim logu",
            "\n".join(st.session_state.training_log),
            height=300,
        )

# ============================================================
# POSITIONS
# ============================================================

with tab_positions:
    st.subheader("Takip edilen pozisyonlar")

    rows = []
    for key, pos in st.session_state.positions.items():
        try:
            cur = market_price(pos["market"], pos["symbol"], spot, futures)
            roe = pnl_percent(pos, cur)
            est = estimated_pnl_usdt(pos, cur)
        except Exception:
            cur, roe, est = None, None, None

        rows.append(
            {
                "Key":key,
                "Market":pos["market"],
                "Symbol":pos["symbol"],
                "Direction":pos["direction"],
                "Mode":pos["mode"],
                "Entry":pos["entry"],
                "Current":cur,
                "Leverage":pos["leverage"],
                "TP %":pos["tp_pct"],
                "PnL %":roe,
                "PnL USDT est":est,
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Açık takip pozisyonu yok.")

    st.subheader("İşlem logu")
    if st.session_state.trade_log:
        st.dataframe(
            pd.DataFrame(st.session_state.trade_log).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# MODEL INFO
# ============================================================

with tab_model:
    st.subheader("Derin öğrenme mimarisi")
    st.code(
        """
200 causal features
        ↓
Feature Projection
        ↓
 ┌──────────────┬──────────────────┬───────────────────┐
 │ TCN branch   │ BiLSTM + BiGRU   │ Transformer × 3   │
 │ 3/5/7 kernel │ recurrent memory │ long-range context │
 └──────────────┴──────────────────┴───────────────────┘
        ↓
Learned Branch Gating
        ↓
8-head Self Attention
        ↓
Learned Last / Mean / Max Pooling
        ↓
Dense classifier
        ↓
SELL / WAIT / BUY
        ↓
Monte-Carlo Dropout
        ↓
Prediction uncertainty + confidence
        ↓
Spot/Futures Router
        """,
        language="text",
    )
    st.info(
        "Bu yapı güçlüdür ama 'en hızlı kâr' veya garantili kâr diye bir model yoktur. "
        "Router düşük güven, yüksek spread, yüksek funding veya aşırı belirsizlikte işlem puanını düşürür."
    )

# ============================================================
# DIAGNOSTICS / 451
# ============================================================

with tab_diag:
    st.subheader("API ve HTTP 451 teşhisi")

    if st.session_state.last_network_diag:
        st.json(st.session_state.last_network_diag)

    if st.button("🌐 Sadece ağ erişimini test et"):
        results = {}
        try:
            data, base = spot_public_request("/api/v3/time")
            results["Spot public"] = f"OK — {base} — {data.get('serverTime')}"
        except Exception as exc:
            results["Spot public"] = str(exc)

        try:
            data = futures_public_request("/fapi/v1/time")
            results["Futures public"] = f"OK — {data.get('serverTime')}"
        except Exception as exc:
            results["Futures public"] = str(exc)

        st.json(results)

    st.markdown(
        """
**451 ne demek?**

`/api/v3/time` API anahtarı istemeyen public bir endpointtir. Burada 451
alıyorsan sorun Secret Key değil; Binance isteği ağ/bölge düzeyinde
reddediyor. Bu uygulama coğrafi/yasal kısıtlamayı aşmaya çalışmaz.

**API izinleri**

- Spot için: `Enable Spot & Margin Trading`
- Futures için: `Enable Futures`
- Withdrawal kapalı tutulmalı.
        """
    )

st.caption(
    "Live Futures kaldıraç nedeniyle yüksek risk taşır. "
    "Önce Paper Mode ve out-of-sample test kullan."
)
