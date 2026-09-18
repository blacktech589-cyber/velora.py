from __future__ import annotations

# ============================================================
# BINANCE SPOT DEEP AI 200 - ALL IN ONE
# ============================================================
# Tek dosyada:
# - Binance Spot REST API client
# - API Key / Secret Key paneli
# - 200 causal feature
# - PyTorch deep learning model
# - Temporal CNN + BiLSTM + GRU + Transformer + Attention
# - Model training
# - BUY / WAIT / SELL inference
# - Paper trading
# - Live Spot MARKET BUY / SELL
# - Streamlit dashboard
#
# Kullanım:
#   pip install streamlit pandas numpy requests torch
#
# Eğitim:
#   python binance_spot_deep_ai_200_all_in_one.py --train \
#       --symbol BTCUSDT --interval 5m --bars 30000 --epochs 12
#
# Panel:
#   streamlit run binance_spot_deep_ai_200_all_in_one.py
#
# Güvenlik:
# - Withdraw/çekim API izni vermeyin.
# - Önce Paper Mode kullanın.
# - AI kâr garantisi vermez.
# ============================================================

import os
import sys
import time
import json
import math
import hmac
import hashlib
import argparse
import uuid
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        try:
            import streamlit as st
            st.error(
                "PyTorch (torch) kurulu değil. Repo köküne requirements.txt ekle ve içine torch yaz."
            )
            st.code(
                "streamlit>=1.38\npandas>=2.1\nnumpy>=1.26\nrequests>=2.31\ntorch\n",
                language="text"
            )
            st.stop()
        except ModuleNotFoundError:
            raise RuntimeError(
                "PyTorch kurulmamış. `pip install torch` çalıştırın."
            ) from exc
    raise


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SPOT_BASES = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
FUTURES_BASE = "https://fapi.binance.com"
FEATURE_COUNT = 200

RET_H = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
WIN = [5, 8, 13, 21, 34, 55, 89, 144]
RSI_P = [5, 7, 9, 14, 21, 28, 35, 50]
ATR_P = [5, 7, 10, 14, 21, 28, 35, 50]
STO_P = [5, 7, 9, 14, 21, 28, 35, 50]
BB_P = [10, 14, 20, 28, 35, 50, 75, 100]
VOL_P = [5, 8, 13, 21, 34, 55, 89, 144]


# ============================================================
# BINANCE CLIENT
# ============================================================

class Binance451Error(RuntimeError):
    pass


def _raise_binance(response, label="Binance"):
    if response.status_code == 451:
        raise Binance451Error(
            f"{label} HTTP 451: legal/regional access restriction. "
            "Bu hata API Key doğrulanmadan önce oluşabilir; kod bunu bypass etmez."
        )
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"{label} HTTP {response.status_code}: {detail}"
        )


def _spot_public(path: str, params=None):
    last_error = None
    for base in SPOT_BASES:
        try:
            response = requests.get(
                base + path,
                params=params or {},
                timeout=20,
            )
            if response.status_code == 451:
                raise Binance451Error(
                    f"Spot HTTP 451 on {base}: legal/regional access restriction."
                )
            if response.status_code >= 500:
                last_error = RuntimeError(
                    f"{base} HTTP {response.status_code}"
                )
                continue
            _raise_binance(response, "Spot")
            return response.json(), base
        except Binance451Error:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Tüm resmi Spot endpoint'leri başarısız: {last_error}"
    )


class BinanceSpotClient:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        self.base = None

    def public_get(self, path: str, params=None):
        data, base = _spot_public(path, params)
        self.base = base
        return data

    def server_time(self) -> int:
        data = self.public_get("/api/v3/time")
        return int(data["serverTime"])

    def signed(self, method: str, path: str, params=None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API Key / Secret Key eksik.")

        p = dict(params or {})
        p["recvWindow"] = int(p.get("recvWindow", 5000))
        p["timestamp"] = self.server_time()

        encoded_payload = urlencode(p, doseq=True, safe="")
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signed_query = f"{encoded_payload}&signature={signature}"
        url = f"{self.base}{path}"
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Global-DeepAI-Scalper/2.0",
        }

        if method.upper() == "GET":
            response = requests.get(
                url + "?" + signed_query,
                headers=headers,
                timeout=25,
            )
        else:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            response = requests.request(
                method.upper(),
                url,
                data=signed_query,
                headers=headers,
                timeout=25,
            )

        _raise_binance(response, "Spot signed API")
        return response.json()

    def account(self):
        return self.signed(
            "GET",
            "/api/v3/account",
            {"omitZeroBalances": "false"},
        )

    def api_restrictions(self):
        return self.signed(
            "GET",
            "/sapi/v1/account/apiRestrictions",
        )

    def account_info(self):
        return self.signed(
            "GET",
            "/sapi/v1/account/info",
        )

    def ticker24(self, symbol: str):
        return self.public_get(
            "/api/v3/ticker/24hr",
            {"symbol": symbol.upper()},
        )

    def ticker_price(self, symbol: str):
        return self.public_get(
            "/api/v3/ticker/price",
            {"symbol": symbol.upper()},
        )

    def book_ticker(self, symbol: str):
        return self.public_get(
            "/api/v3/ticker/bookTicker",
            {"symbol": symbol.upper()},
        )

    def exchange_info(self, symbol: str | None = None):
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self.public_get("/api/v3/exchangeInfo", params)

    def klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 1000,
        start_time=None,
        end_time=None,
    ):
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self.public_get("/api/v3/klines", params)

    def symbol_filters(self, symbol: str):
        info = self.exchange_info(symbol)
        if not info.get("symbols"):
            raise RuntimeError(f"Spot symbol bulunamadı: {symbol}")
        symbol_data = info["symbols"][0]
        filters = {
            f["filterType"]: f
            for f in symbol_data.get("filters", [])
        }
        return filters, symbol_data

    @staticmethod
    def floor_step(value: float, step: float) -> float:
        if step <= 0:
            return float(value)
        return math.floor(value / step) * step

    def normalize_quantity(self, symbol: str, qty: float) -> float:
        filters, _ = self.symbol_filters(symbol)
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
        step = float(lot.get("stepSize", "0.00000001"))
        min_qty = float(lot.get("minQty", "0"))
        max_qty = float(lot.get("maxQty", "0"))
        normalized = self.floor_step(qty, step)
        if min_qty > 0 and normalized < min_qty:
            raise RuntimeError(
                f"Spot miktar minQty altında: {normalized} < {min_qty}"
            )
        if max_qty > 0 and normalized > max_qty:
            raise RuntimeError(
                f"Spot miktar maxQty üstünde: {normalized} > {max_qty}"
            )
        return normalized

    def market_buy_quote(self, symbol: str, quote_amount: float):
        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{float(quote_amount):.8f}",
                "newClientOrderId": "ai_" + uuid.uuid4().hex[:20],
                "newOrderRespType": "FULL",
            },
        )

    def market_sell_qty(self, symbol: str, qty: float):
        normalized = self.normalize_quantity(symbol, qty)
        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{normalized:.12f}",
                "newClientOrderId": "ai_" + uuid.uuid4().hex[:20],
                "newOrderRespType": "FULL",
            },
        )


class BinanceFuturesClient:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()

    def public_get(self, path: str, params=None):
        response = requests.get(
            FUTURES_BASE + path,
            params=params or {},
            timeout=20,
        )
        _raise_binance(response, "USDⓈ-M Futures")
        return response.json()

    def server_time(self):
        return int(self.public_get("/fapi/v1/time")["serverTime"])

    def signed(self, method: str, path: str, params=None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API Key / Secret Key eksik.")
        p = dict(params or {})
        p["recvWindow"] = int(p.get("recvWindow", 5000))
        p["timestamp"] = self.server_time()
        query = urlencode(p, doseq=True, safe="")
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_query = f"{query}&signature={signature}"
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Global-DeepAI-Scalper/2.0",
        }
        url = FUTURES_BASE + path
        if method.upper() == "GET":
            response = requests.get(
                url + "?" + signed_query,
                headers=headers,
                timeout=25,
            )
        else:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            response = requests.request(
                method.upper(),
                url,
                data=signed_query,
                headers=headers,
                timeout=25,
            )
        _raise_binance(response, "USDⓈ-M Futures signed API")
        return response.json()

    def account(self):
        try:
            return self.signed("GET", "/fapi/v3/account")
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
        return self.signed("GET", "/fapi/v2/account")

    def ticker_price(self, symbol: str):
        return self.public_get(
            "/fapi/v1/ticker/price",
            {"symbol": symbol.upper()},
        )

    def book_ticker(self, symbol: str):
        return self.public_get(
            "/fapi/v1/ticker/bookTicker",
            {"symbol": symbol.upper()},
        )

    def premium_index(self, symbol: str):
        return self.public_get(
            "/fapi/v1/premiumIndex",
            {"symbol": symbol.upper()},
        )

    def exchange_info(self):
        return self.public_get("/fapi/v1/exchangeInfo")

    def klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 1000,
        start_time=None,
        end_time=None,
    ):
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self.public_get("/fapi/v1/klines", params)

    def symbol_filters(self, symbol: str):
        info = self.exchange_info()
        matches = [
            s for s in info.get("symbols", [])
            if s.get("symbol") == symbol.upper()
        ]
        if not matches:
            raise RuntimeError(f"Futures symbol bulunamadı: {symbol}")
        symbol_data = matches[0]
        filters = {
            f["filterType"]: f
            for f in symbol_data.get("filters", [])
        }
        return filters, symbol_data

    def normalize_quantity(self, symbol: str, qty: float) -> float:
        filters, _ = self.symbol_filters(symbol)
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
        step = float(lot.get("stepSize", "0.001"))
        min_qty = float(lot.get("minQty", "0"))
        max_qty = float(lot.get("maxQty", "0"))
        normalized = BinanceSpotClient.floor_step(qty, step)
        if min_qty > 0 and normalized < min_qty:
            raise RuntimeError(
                f"Futures miktar minQty altında: {normalized} < {min_qty}"
            )
        if max_qty > 0 and normalized > max_qty:
            raise RuntimeError(
                f"Futures miktar maxQty üstünde: {normalized} > {max_qty}"
            )
        return normalized

    def set_leverage(self, symbol: str, leverage: int):
        return self.signed(
            "POST",
            "/fapi/v1/leverage",
            {
                "symbol": symbol.upper(),
                "leverage": int(leverage),
            },
        )

    def open_market(self, symbol: str, direction: str, qty: float):
        side = "BUY" if direction == "LONG" else "SELL"
        normalized = self.normalize_quantity(symbol, qty)
        return self.signed(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol.upper(),
                "side": side,
                "type": "MARKET",
                "quantity": f"{normalized:.12f}",
                "newClientOrderId": "ai_" + uuid.uuid4().hex[:20],
                "newOrderRespType": "RESULT",
            },
        )

    def close_market(self, symbol: str, direction: str, qty: float):
        side = "SELL" if direction == "LONG" else "BUY"
        normalized = self.normalize_quantity(symbol, qty)
        return self.signed(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol.upper(),
                "side": side,
                "type": "MARKET",
                "quantity": f"{normalized:.12f}",
                "reduceOnly": "true",
                "newClientOrderId": "ai_" + uuid.uuid4().hex[:20],
                "newOrderRespType": "RESULT",
            },
        )


# ============================================================
# DATA HELPERS
# ============================================================

def klines_to_df(rows) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore"
    ]

    df = pd.DataFrame(rows, columns=columns)

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote"
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    return df.set_index("open_time")


def fetch_klines_history(
    symbol: str,
    interval: str,
    total: int,
    market: str = "SPOT",
) -> pd.DataFrame:
    client = (
        BinanceSpotClient()
        if market.upper() == "SPOT"
        else BinanceFuturesClient()
    )

    rows = []
    end_time = None

    while len(rows) < total:
        limit = min(1000, total - len(rows))

        batch = client.klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            end_time=end_time,
        )

        if not batch:
            break

        rows = batch + rows

        earliest_open = int(batch[0][0])
        end_time = earliest_open - 1

        time.sleep(0.05)

    rows = rows[-total:]

    if not rows:
        raise RuntimeError(
            f"{market} geçmiş mum verisi alınamadı."
        )

    return klines_to_df(rows)


# ============================================================
# FEATURE ENGINE - EXACTLY 200 FEATURES
# ============================================================

def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    return a / b


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()

    up = delta.clip(lower=0).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    down = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = safe_div(up, down)

    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return safe_div(series - mean, std)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    x = df.copy().sort_index()

    out = pd.DataFrame(index=x.index)

    o = x["open"]
    h = x["high"]
    l = x["low"]
    c = x["close"]
    v = x["volume"]

    qv = (
        x["quote_volume"]
        if "quote_volume" in x.columns
        else c * v
    )

    trades = (
        x["trades"]
        if "trades" in x.columns
        else pd.Series(0.0, index=x.index)
    )

    taker_base = (
        x["taker_base"]
        if "taker_base" in x.columns
        else pd.Series(0.0, index=x.index)
    )

    # --------------------------------------------------------
    # 1-20: returns + log returns
    # --------------------------------------------------------

    for n in RET_H:
        out[f"ret_{n}"] = c.pct_change(n)

    log_close = np.log(c.replace(0, np.nan))

    for n in RET_H:
        out[f"logret_{n}"] = log_close.diff(n)

    # --------------------------------------------------------
    # 21-36: SMA / EMA deviations
    # --------------------------------------------------------

    for w in WIN:
        sma = c.rolling(w).mean()
        out[f"sma_dev_{w}"] = safe_div(c, sma) - 1

    for w in WIN:
        em = ema(c, w)
        out[f"ema_dev_{w}"] = safe_div(c, em) - 1

    # --------------------------------------------------------
    # 37-52: volatility + close zscore
    # --------------------------------------------------------

    ret1 = c.pct_change()

    for w in WIN:
        out[f"ret_std_{w}"] = ret1.rolling(w).std()

    for w in WIN:
        out[f"close_z_{w}"] = zscore(c, w)

    # --------------------------------------------------------
    # 53-68: rolling high/low position
    # --------------------------------------------------------

    for w in WIN:
        rolling_min = l.rolling(w).min()
        rolling_max = h.rolling(w).max()

        width = (rolling_max - rolling_min).replace(0, np.nan)

        out[f"range_pos_{w}"] = (c - rolling_min) / width
        out[f"range_width_{w}"] = width / c.replace(0, np.nan)

    # --------------------------------------------------------
    # 69-76: RSI
    # --------------------------------------------------------

    for p in RSI_P:
        out[f"rsi_{p}"] = rsi(c, p) / 100.0

    # --------------------------------------------------------
    # 77-84: ATR
    # --------------------------------------------------------

    for p in ATR_P:
        out[f"atr_pct_{p}"] = atr(x, p) / c.replace(0, np.nan)

    # --------------------------------------------------------
    # 85-100: stochastic
    # --------------------------------------------------------

    for p in STO_P:
        low_roll = l.rolling(p).min()
        high_roll = h.rolling(p).max()

        k = (
            (c - low_roll)
            /
            (high_roll - low_roll).replace(0, np.nan)
        )

        out[f"stoch_k_{p}"] = k
        out[f"stoch_d_{p}"] = k.rolling(3).mean()

    # --------------------------------------------------------
    # 101-116: Bollinger
    # --------------------------------------------------------

    for p in BB_P:
        mean = c.rolling(p).mean()
        std = c.rolling(p).std()

        upper = mean + (2 * std)
        lower = mean - (2 * std)

        out[f"bb_pos_{p}"] = (
            (c - lower)
            /
            (upper - lower).replace(0, np.nan)
        )

        out[f"bb_width_{p}"] = (
            (upper - lower)
            /
            mean.replace(0, np.nan)
        )

    # --------------------------------------------------------
    # 117-140: volume
    # --------------------------------------------------------

    for p in VOL_P:
        out[f"vol_z_{p}"] = zscore(v, p)
        out[f"vol_ratio_{p}"] = safe_div(v, v.rolling(p).mean())
        out[f"qvol_ratio_{p}"] = safe_div(qv, qv.rolling(p).mean())

    # --------------------------------------------------------
    # 141-150: ROC
    # --------------------------------------------------------

    for n in RET_H:
        out[f"roc_{n}"] = (
            c
            /
            c.shift(n).replace(0, np.nan)
            - 1
        )

    # --------------------------------------------------------
    # 151-160: candle geometry
    # --------------------------------------------------------

    candle_range = (h - l).replace(0, np.nan)
    body = c - o

    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    min_oc = pd.concat([o, c], axis=1).min(axis=1)

    out["body_pct"] = body / o.replace(0, np.nan)
    out["body_to_range"] = body / candle_range
    out["abs_body_to_range"] = body.abs() / candle_range
    out["upper_wick_ratio"] = (h - max_oc) / candle_range
    out["lower_wick_ratio"] = (min_oc - l) / candle_range
    out["close_location"] = (c - l) / candle_range
    out["open_location"] = (o - l) / candle_range
    out["gap_pct"] = o / c.shift(1).replace(0, np.nan) - 1
    out["hl_pct"] = (h - l) / c.replace(0, np.nan)
    out["oc_abs_pct"] = (c - o).abs() / o.replace(0, np.nan)

    # --------------------------------------------------------
    # 161-172: microstructure proxies
    # --------------------------------------------------------

    signed = np.sign(c.diff()).fillna(0)
    obv = (signed * v).cumsum()

    for w in [5, 13, 21, 34]:
        out[f"obv_z_{w}"] = zscore(obv, w)

    for w in [5, 13, 21, 34]:
        out[f"trade_z_{w}"] = zscore(trades, w)

    taker_ratio = safe_div(taker_base, v)

    for w in [5, 13, 21, 34]:
        out[f"taker_ratio_ma_{w}"] = taker_ratio.rolling(w).mean()

    # --------------------------------------------------------
    # 173-184: MACD family
    # --------------------------------------------------------

    macd_pairs = [
        (5, 13),
        (8, 21),
        (12, 26),
        (13, 34),
        (21, 55),
        (34, 89)
    ]

    for fast, slow in macd_pairs:
        macd = ema(c, fast) - ema(c, slow)
        signal = ema(macd, 9)

        out[f"macd_norm_{fast}_{slow}"] = (
            macd
            /
            c.replace(0, np.nan)
        )

        out[f"macd_hist_{fast}_{slow}"] = (
            (macd - signal)
            /
            c.replace(0, np.nan)
        )

    # --------------------------------------------------------
    # 185-192: acceleration
    # --------------------------------------------------------

    for n in [1, 2, 3, 5, 8, 13, 21, 34]:
        r = c.pct_change(n)
        out[f"accel_{n}"] = r - r.shift(n)

    # --------------------------------------------------------
    # 193-198: cyclic time
    # --------------------------------------------------------

    if isinstance(out.index, pd.DatetimeIndex):
        minute = out.index.minute + out.index.hour * 60
        dow = out.index.dayofweek

        out["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
        out["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    else:
        for name in [
            "tod_sin",
            "tod_cos",
            "dow_sin",
            "dow_cos",
            "hour_sin",
            "hour_cos"
        ]:
            out[name] = 0.0

    # --------------------------------------------------------
    # 199-200 and fallback cross-features
    # --------------------------------------------------------

    out["trend_volume_interaction"] = (
        out["ema_dev_21"]
        *
        out["vol_ratio_21"]
    )

    out["momentum_volatility_interaction"] = (
        out["ret_5"]
        *
        out["atr_pct_14"]
    )

    # Safety: if feature count changes after edits, dynamically fill/truncate
    base_cols = list(out.columns)
    i = 0

    while out.shape[1] < FEATURE_COUNT:
        a = base_cols[i % len(base_cols)]
        b = base_cols[(i * 7 + 11) % len(base_cols)]

        out[f"cross_{i:03d}"] = out[a] * out[b]
        i += 1

    out = out.iloc[:, :FEATURE_COUNT]

    out = out.replace(
        [np.inf, -np.inf],
        np.nan
    )

    out = (
        out
        .ffill()
        .fillna(0.0)
        .clip(-1000, 1000)
        .astype("float32")
    )

    if out.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature count mismatch: {out.shape[1]}"
        )

    return out


# ============================================================
# DEEP LEARNING MODEL
# ============================================================

class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float
    ):
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x) + x)


class DeepSpotAI(nn.Module):
    """
    Daha güçlü sequence architecture:

    200 features
        ↓
    LayerNorm
        ↓
    Dense Projection
        ↓
    Residual Temporal CNN x3
        ↓
    BiLSTM x2
        ↓
    BiGRU x2
        ↓
    Transformer Encoder x2
        ↓
    Multi-Head Self Attention
        ↓
    Gated last/mean/max pooling
        ↓
    Dense classifier
        ↓
    SELL / WAIT / BUY
    """

    def __init__(
        self,
        n_features: int = FEATURE_COUNT,
        d_model: int = 160,
        lstm_hidden: int = 112,
        gru_hidden: int = 112,
        heads: int = 8,
        dropout: float = 0.18,
        classes: int = 3
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(n_features)

        self.project = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.GELU()
        )

        self.temporal = nn.Sequential(
            ResidualTemporalBlock(
                d_model,
                kernel_size=3,
                dilation=1,
                dropout=dropout
            ),
            ResidualTemporalBlock(
                d_model,
                kernel_size=5,
                dilation=2,
                dropout=dropout
            ),
            ResidualTemporalBlock(
                d_model,
                kernel_size=7,
                dilation=4,
                dropout=dropout
            ),
            ResidualTemporalBlock(
                d_model,
                kernel_size=9,
                dilation=8,
                dropout=dropout
            )
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.gru_proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )

        fused_dim = d_model * 3

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 320),
            nn.GELU(),
            nn.LayerNorm(320),
            nn.Dropout(dropout),
            nn.Linear(320, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, classes)
        )

    def forward(self, x):
        # x: [batch, sequence, 200]

        x = self.input_norm(x)
        x = self.project(x)

        # CNN expects [B,C,T]
        temporal = self.temporal(
            x.transpose(1, 2)
        ).transpose(1, 2)

        x = x + temporal

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)

        x = x + lstm_out

        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)

        x = x + gru_out

        x = self.transformer(x)

        attn_out, _ = self.attention(
            x,
            x,
            x,
            need_weights=False
        )

        x = self.attn_norm(x + attn_out)

        last_pool = x[:, -1, :]
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        concat_for_gate = torch.cat(
            [last_pool, mean_pool, max_pool],
            dim=-1
        )

        weights = self.gate(concat_for_gate)

        weighted_last = last_pool * weights[:, 0:1]
        weighted_mean = mean_pool * weights[:, 1:2]
        weighted_max = max_pool * weights[:, 2:3]

        fused = torch.cat(
            [
                weighted_last,
                weighted_mean,
                weighted_max
            ],
            dim=-1
        )

        return self.classifier(fused)


# ============================================================
# TRAINING DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length: int):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return max(
            0,
            len(self.y) - self.sequence_length + 1
        )

    def __getitem__(self, index):
        end = index + self.sequence_length - 1

        seq = self.X[
            index:
            index + self.sequence_length
        ]

        target = self.y[end]

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long)
        )


def build_labels(
    close: pd.Series,
    horizon: int,
    threshold: float
):
    future_return = (
        close.shift(-horizon)
        /
        close
        - 1
    )

    labels = np.ones(
        len(close),
        dtype=np.int64
    )

    # 0 SELL
    # 1 WAIT
    # 2 BUY

    labels[future_return < -threshold] = 0
    labels[
        (future_return >= -threshold)
        &
        (future_return <= threshold)
    ] = 1
    labels[future_return > threshold] = 2

    return labels


def compute_class_weights(y: np.ndarray):
    counts = np.bincount(
        y,
        minlength=3
    ).astype(np.float64)

    counts[counts == 0] = 1

    weights = counts.sum() / (3 * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def train_model(
    symbol: str,
    interval: str,
    bars: int,
    sequence_length: int,
    horizon: int,
    threshold: float,
    epochs: int,
    batch_size: int,
    out_dir: str,
    lr: float = 2e-4,
    market: str = "SPOT",
):
    out = Path(out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Binance geçmiş verisi indiriliyor...")

    df = fetch_klines_history(
        symbol,
        interval,
        bars,
        market=market,
    )

    print("200 özellik hesaplanıyor...")

    features = build_features(df)

    labels = build_labels(
        df["close"],
        horizon,
        threshold
    )

    usable = len(df) - horizon

    features = features.iloc[:usable]
    labels = labels[:usable]

    # chronological split
    train_end = int(len(features) * 0.70)
    val_end = int(len(features) * 0.85)

    train_features = features.iloc[:train_end]

    mean = train_features.mean().to_numpy(np.float32)

    std = (
        train_features
        .std()
        .replace(0, 1)
        .to_numpy(np.float32)
    )

    X = (
        (
            features.to_numpy(np.float32)
            - mean
        )
        / std
    ).clip(-8, 8)

    X_train = X[:train_end]
    y_train = labels[:train_end]

    X_val = X[
        train_end - sequence_length:
        val_end
    ]

    y_val = labels[
        train_end - sequence_length:
        val_end
    ]

    X_test = X[
        val_end - sequence_length:
    ]

    y_test = labels[
        val_end - sequence_length:
    ]

    train_ds = SequenceDataset(
        X_train,
        y_train,
        sequence_length
    )

    val_ds = SequenceDataset(
        X_val,
        y_val,
        sequence_length
    )

    test_ds = SequenceDataset(
        X_test,
        y_test,
        sequence_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = DeepSpotAI(
        n_features=FEATURE_COUNT
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    class_weights = compute_class_weights(
        y_train
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.03
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda")
    )

    best_val_loss = float("inf")
    patience = 0
    max_patience = 4

    for epoch in range(1, epochs + 1):
        model.train()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda")
            ):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            preds = logits.argmax(dim=1)

            train_correct += (
                preds == yb
            ).sum().item()

            train_total += yb.numel()

        model.eval()

        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)

                loss = loss_fn(
                    logits,
                    yb
                )

                total_val_loss += (
                    loss.item()
                )

                preds = logits.argmax(
                    dim=1
                )

                val_correct += (
                    preds == yb
                ).sum().item()

                val_total += yb.numel()

        train_loss = (
            total_train_loss
            /
            max(1, len(train_loader))
        )

        val_loss = (
            total_val_loss
            /
            max(1, len(val_loader))
        )

        train_acc = (
            train_correct
            /
            max(1, train_total)
        )

        val_acc = (
            val_correct
            /
            max(1, val_total)
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}"
            f" | train_loss={train_loss:.4f}"
            f" | train_acc={train_acc:.4f}"
            f" | val_loss={val_loss:.4f}"
            f" | val_acc={val_acc:.4f}"
            f" | lr={current_lr:.7f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0

            torch.save(
                model.state_dict(),
                out / "model.pt"
            )

            np.savez(
                out / "scaler.npz",
                mean=mean,
                std=std
            )

            metadata = {
                "symbol": symbol,
                "interval": interval,
                "sequence_length": sequence_length,
                "horizon": horizon,
                "threshold": threshold,
                "feature_count": FEATURE_COUNT,
                "features": list(features.columns),
                "classes": ["SELL", "WAIT", "BUY"],
                "architecture": (
                    "ResidualCNN+BiLSTM+BiGRU+"
                    "Transformer+MultiHeadAttention"
                )
            }

            (
                out / "meta.json"
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2
                ),
                encoding="utf-8"
            )

            print("Yeni en iyi model kaydedildi.")

        else:
            patience += 1

            if patience >= max_patience:
                print("Early stopping.")
                break

    # Test evaluation
    print("En iyi model test ediliyor...")

    model.load_state_dict(
        torch.load(
            out / "model.pt",
            map_location=device
        )
    )

    model.eval()

    test_correct = 0
    test_total = 0
    confusion = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            test_correct += (
                preds == yb
            ).sum().item()

            test_total += yb.numel()

            for true, pred in zip(
                yb.cpu().numpy(),
                preds.cpu().numpy()
            ):
                confusion[int(true), int(pred)] += 1

    test_acc = (
        test_correct
        /
        max(1, test_total)
    )

    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion matrix:")
    print(confusion)
    print(f"Model klasörü: {out.resolve()}")


# ============================================================
# MODEL LOADER + INFERENCE
# ============================================================

_MODEL_CACHE = {}


def load_model_artifacts(model_dir: str):
    key = str(Path(model_dir).resolve())

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    path = Path(model_dir)

    meta_path = path / "meta.json"
    model_path = path / "model.pt"
    scaler_path = path / "scaler.npz"

    if not meta_path.exists():
        raise RuntimeError(
            f"meta.json bulunamadı: {meta_path}"
        )

    if not model_path.exists():
        raise RuntimeError(
            f"model.pt bulunamadı: {model_path}"
        )

    if not scaler_path.exists():
        raise RuntimeError(
            f"scaler.npz bulunamadı: {scaler_path}"
        )

    metadata = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    scaler_data = np.load(
        scaler_path
    )

    mean = scaler_data["mean"]
    std = scaler_data["std"]

    model = DeepSpotAI(
        n_features=int(
            metadata["feature_count"]
        )
    )

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)
    model.eval()

    result = (
        model,
        metadata,
        mean,
        std
    )

    _MODEL_CACHE[key] = result

    return result


def predict_symbol(
    client: BinanceSpotClient,
    symbol: str,
    interval: str,
    model_dir: str
):
    (
        model,
        metadata,
        mean,
        std
    ) = load_model_artifacts(
        model_dir
    )

    rows = client.klines(
        symbol,
        interval,
        limit=1000
    )

    df = klines_to_df(rows)

    features = build_features(df)

    expected_features = metadata[
        "features"
    ]

    features = features.reindex(
        columns=expected_features,
        fill_value=0.0
    )

    values = features.to_numpy(
        np.float32
    )

    normalized = (
        (values - mean)
        /
        std
    ).clip(-8, 8)

    seq_len = int(
        metadata["sequence_length"]
    )

    if len(normalized) < seq_len:
        raise RuntimeError(
            f"Yetersiz mum: {len(normalized)} < {seq_len}"
        )

    sequence = normalized[
        -seq_len:
    ]

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)

        probs = torch.softmax(
            logits,
            dim=1
        ).squeeze(0).numpy()

    # MC-dropout uncertainty pass. BatchNorm remains eval; Dropout enabled.
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    draws = []
    with torch.no_grad():
        for _ in range(8):
            p = torch.softmax(
                model(tensor),
                dim=1
            ).squeeze(0).numpy()
            draws.append(p)

    model.eval()
    stacked = np.stack(draws, axis=0)
    mean_probs = stacked.mean(axis=0)
    variance = float(stacked.var(axis=0).mean())
    safe_probs = np.clip(mean_probs.astype(np.float64), 1e-9, 1.0)
    entropy = float(
        -np.sum(safe_probs * np.log(safe_probs)) / np.log(3.0)
    )
    uncertainty = min(
        100.0,
        100.0 * (0.75 * entropy + 0.25 * min(1.0, variance * 100.0))
    )
    confidence = max(0.0, 100.0 - uncertainty)

    result = {
        "SELL": float(mean_probs[0] * 100),
        "WAIT": float(mean_probs[1] * 100),
        "BUY": float(mean_probs[2] * 100),
        "CONFIDENCE": confidence,
        "UNCERTAINTY": uncertainty,
    }

    return result, df


# ============================================================
# CLI TRAIN MODE
# ============================================================

def cli_train():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT"
    )

    parser.add_argument(
        "--interval",
        default="5m"
    )

    parser.add_argument(
        "--bars",
        type=int,
        default=30000
    )

    parser.add_argument(
        "--seq",
        type=int,
        default=96
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=3
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0025
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=128
    )

    parser.add_argument(
        "--out",
        default="model_artifacts"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    args = parser.parse_args()

    train_model(
        symbol=args.symbol.upper(),
        interval=args.interval,
        bars=args.bars,
        sequence_length=args.seq,
        horizon=args.horizon,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch,
        out_dir=args.out,
        lr=args.lr
    )


# ============================================================
# STREAMLIT PANEL
# ============================================================

def run_streamlit_panel():
    import streamlit as st

    st.set_page_config(
        page_title="Binance Global Deep AI Scalper",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    defaults = {
        "api_key": "",
        "api_secret": "",
        "api_ok": False,
        "api_permissions": {},
        "spot_allowed": False,
        "futures_allowed": False,
        "spot_account": None,
        "futures_account": None,
        "paper_balance": 1000.0,
        "positions": {},
        "trade_log": [],
        "scan_rows": [],
        "diag": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown(
        """
        <style>
        .block-container {padding-top:1rem;padding-bottom:2rem}
        [data-testid="stSidebar"] {background:#0b111c}
        .hero {
            padding:22px;border-radius:22px;
            background:linear-gradient(135deg,#0f172a,#172554);
            border:1px solid #334155;margin-bottom:14px
        }
        .hero h1 {margin:0;font-size:31px}
        .hero p {margin:.35rem 0 0;opacity:.76}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def spread_bps(book):
        bid = float(book["bidPrice"])
        ask = float(book["askPrice"])
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 10000 if mid > 0 else 9999.0

    def detect_regime(df):
        close = df["close"]
        if len(close) < 60:
            return "UNKNOWN", 0.0, 0.0
        fast = ema(close, 12)
        slow = ema(close, 55)
        price = float(close.iloc[-1])
        trend_raw = (
            (float(fast.iloc[-1]) - float(slow.iloc[-1]))
            / max(price, 1e-12)
            * 100
        )
        trend_score = float(np.tanh(trend_raw * 8.0) * 100.0)
        vol = float(close.pct_change().tail(50).std() * 100)
        if abs(trend_score) >= 35:
            regime = "UPTREND" if trend_score > 0 else "DOWNTREND"
        elif vol >= 0.8:
            regime = "HIGH_VOL_RANGE"
        elif vol <= 0.05:
            regime = "LOW_VOL_RANGE"
        else:
            regime = "RANGE"
        return regime, trend_score, vol

    def vol_score(vol):
        if not np.isfinite(vol) or vol <= 0:
            return 0.0
        if vol < 0.02:
            return vol / 0.02 * 25.0
        if vol <= 0.8:
            return 25.0 + min(55.0, (vol - 0.02) / 0.78 * 55.0)
        return max(5.0, 80.0 - (vol - 0.8) * 25.0)

    def score_candidate(market, direction, probs, spread, regime, trend, vol, funding=0.0):
        directional = probs["BUY"] if direction == "LONG" else probs["SELL"]
        score = (
            directional * 0.48
            + probs.get("CONFIDENCE", 0.0) * 0.24
            + vol_score(vol) * 0.13
            - probs["WAIT"] * 0.10
            - probs.get("UNCERTAINTY", 100.0) * 0.14
            - min(35.0, spread * 2.2)
            - min(15.0, abs(funding) * 10000 * 2.0)
        )
        if direction == "LONG" and regime == "UPTREND":
            score += 10.0
        if direction == "SHORT" and regime == "DOWNTREND":
            score += 10.0
        if direction == "LONG" and regime == "DOWNTREND":
            score -= 12.0
        if direction == "SHORT" and regime == "UPTREND":
            score -= 12.0
        if market == "SPOT" and direction == "SHORT":
            score -= 100.0
        return float(score)

    def get_price(market, symbol, spot_client, futures_client):
        client = spot_client if market == "SPOT" else futures_client
        return float(client.ticker_price(symbol)["price"])

    def pnl_pct(position, current):
        sign = 1.0 if position["direction"] == "LONG" else -1.0
        raw = sign * (float(current) - position["entry"]) / position["entry"] * 100
        return raw * float(position.get("leverage", 1))

    def open_paper(market, symbol, direction, margin, price, leverage, tp_pct):
        key = f"{market}:{symbol}"
        if key in st.session_state.positions:
            raise RuntimeError("Bu market/symbol zaten açık.")
        if st.session_state.paper_balance < margin:
            raise RuntimeError("Paper bakiye yetersiz.")
        qty = (margin * leverage) / price
        st.session_state.paper_balance -= margin
        st.session_state.positions[key] = {
            "market": market,
            "symbol": symbol,
            "direction": direction,
            "mode": "PAPER",
            "entry": float(price),
            "qty": float(qty),
            "margin": float(margin),
            "leverage": int(leverage),
            "tp_pct": float(tp_pct),
            "opened_at": str(datetime.now()),
        }

    def open_live(market, symbol, direction, margin, price, leverage, tp_pct, spot_client, futures_client):
        key = f"{market}:{symbol}"
        if key in st.session_state.positions:
            raise RuntimeError("Bu market/symbol zaten açık.")

        if market == "SPOT":
            if direction != "LONG":
                raise RuntimeError("Spot bot SHORT açmaz.")
            result = spot_client.market_buy_quote(symbol, margin)
            qty = float(result.get("executedQty", 0) or 0)
            quote = float(result.get("cummulativeQuoteQty", 0) or 0)
            if qty <= 0:
                raise RuntimeError(f"Spot BUY executedQty alınamadı: {result}")
            actual_entry = quote / qty if quote > 0 else price
            st.session_state.positions[key] = {
                "market": "SPOT",
                "symbol": symbol,
                "direction": "LONG",
                "mode": "LIVE",
                "entry": float(actual_entry),
                "qty": float(qty),
                "margin": float(margin),
                "leverage": 1,
                "tp_pct": float(tp_pct),
                "opened_at": str(datetime.now()),
            }
            return result

        futures_client.set_leverage(symbol, leverage)
        qty = (margin * leverage) / price
        normalized = futures_client.normalize_quantity(symbol, qty)
        result = futures_client.open_market(symbol, direction, normalized)
        st.session_state.positions[key] = {
            "market": "FUTURES",
            "symbol": symbol,
            "direction": direction,
            "mode": "LIVE",
            "entry": float(price),
            "qty": float(normalized),
            "margin": float(margin),
            "leverage": int(leverage),
            "tp_pct": float(tp_pct),
            "opened_at": str(datetime.now()),
        }
        return result

    def close_position(key, current, spot_client, futures_client):
        pos = st.session_state.positions[key]
        if pos["mode"] == "LIVE":
            if pos["market"] == "SPOT":
                result = spot_client.market_sell_qty(pos["symbol"], pos["qty"])
            else:
                result = futures_client.close_market(
                    pos["symbol"], pos["direction"], pos["qty"]
                )
        else:
            result = {"paper": True}

        profit = pnl_pct(pos, current)
        pnl_usdt = pos["margin"] * (profit / 100.0)
        if pos["mode"] == "PAPER":
            st.session_state.paper_balance += pos["margin"] + pnl_usdt

        st.session_state.trade_log.append({
            "time": str(datetime.now()),
            "action": "CLOSE_TP",
            "market": pos["market"],
            "symbol": pos["symbol"],
            "direction": pos["direction"],
            "entry": pos["entry"],
            "exit": float(current),
            "leverage": pos["leverage"],
            "profit_pct": profit,
            "pnl_usdt_est": pnl_usdt,
            "mode": pos["mode"],
            "result": str(result),
        })
        del st.session_state.positions[key]
        return result

    def monitor_tp(spot_client, futures_client):
        events = []
        for key in list(st.session_state.positions.keys()):
            pos = st.session_state.positions.get(key)
            if not pos:
                continue
            try:
                current = get_price(pos["market"], pos["symbol"], spot_client, futures_client)
                profit = pnl_pct(pos, current)
                if profit >= pos["tp_pct"]:
                    close_position(key, current, spot_client, futures_client)
                    events.append(
                        f"{pos['market']} {pos['symbol']} {pos['direction']} TP kapandı: %{profit:.3f}"
                    )
            except Exception as exc:
                events.append(f"{key} TP kontrol hatası: {exc}")
        return events

    with st.sidebar:
        st.header("🔐 Binance Global API")
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

        if st.button("🔌 Global API'yi Test Et", use_container_width=True):
            diag = {}
            try:
                test_spot = BinanceSpotClient(api_key, api_secret)
                server_time = test_spot.server_time()
                diag["spot_public"] = {
                    "ok": True,
                    "endpoint": test_spot.base,
                    "serverTime": server_time,
                }
                account = test_spot.account()
                diag["spot_signed"] = {"ok": True, "canTrade": account.get("canTrade")}
                restrictions = test_spot.api_restrictions()
                diag["permissions"] = restrictions

                st.session_state.api_key = api_key
                st.session_state.api_secret = api_secret
                st.session_state.api_ok = True
                st.session_state.spot_account = account
                st.session_state.api_permissions = restrictions
                st.session_state.spot_allowed = bool(
                    restrictions.get("enableSpotAndMarginTrading", False)
                )
                st.session_state.futures_allowed = bool(
                    restrictions.get("enableFutures", False)
                )

                try:
                    fclient = BinanceFuturesClient(api_key, api_secret)
                    diag["futures_public"] = {
                        "ok": True,
                        "serverTime": fclient.server_time(),
                    }
                    if st.session_state.futures_allowed:
                        facc = fclient.account()
                        st.session_state.futures_account = facc
                        diag["futures_signed"] = {"ok": True, "canTrade": facc.get("canTrade")}
                except Exception as exc:
                    diag["futures"] = {"ok": False, "error": str(exc)}

                st.session_state.diag = diag
                st.success("Global API signed bağlantı başarılı.")

            except Exception as exc:
                st.session_state.api_ok = False
                st.session_state.diag = {"error": str(exc)}
                st.error(str(exc))

        if st.button("API anahtarlarını temizle", use_container_width=True):
            for key in [
                "api_key", "api_secret", "api_permissions", "spot_account",
                "futures_account", "diag"
            ]:
                if key in ["api_key", "api_secret"]:
                    st.session_state[key] = ""
                elif key in ["api_permissions", "diag"]:
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = None
            st.session_state.api_ok = False
            st.session_state.spot_allowed = False
            st.session_state.futures_allowed = False
            st.rerun()

        st.divider()
        st.header("⚡ Scalping")

        mode = st.radio("Emir modu", ["Paper", "Live"], index=0)
        live_confirm = False
        if mode == "Live":
            st.warning("Live mod gerçek Spot/Futures emir gönderir.")
            live_confirm = st.checkbox("Gerçek emirleri etkinleştir")

        symbols_text = st.text_area(
            "Pariteler",
            "BTCUSDT\nETHUSDT\nBNBUSDT\nSOLUSDT\nXRPUSDT",
            height=115,
        )
        symbols = [
            item.strip().upper()
            for item in symbols_text.splitlines()
            if item.strip()
        ]

        interval = st.selectbox(
            "AI timeframe",
            ["1m", "3m", "5m", "15m", "30m", "1h"],
            index=2,
        )
        spot_model_dir = st.text_input("Spot model klasörü", "models/spot")
        futures_model_dir = st.text_input("Futures model klasörü", "models/futures")
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
        )
        leverage = st.slider("Futures kaldıraç", 1, 10, 2)
        min_score = st.slider("Minimum router skoru", 0.0, 100.0, 35.0, 1.0)
        max_spread = st.number_input(
            "Maksimum spread (bps)",
            min_value=0.1,
            max_value=100.0,
            value=20.0,
            step=0.5,
        )

    spot = BinanceSpotClient(
        st.session_state.api_key,
        st.session_state.api_secret,
    )
    futures = BinanceFuturesClient(
        st.session_state.api_key,
        st.session_state.api_secret,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>Binance Global Deep AI Scalper</h1>
          <p>
          Spot + USDⓈ-M Futures • 200 özellik • Residual TCN ×4 • BiLSTM ×2 •
          BiGRU ×2 • Transformer ×4 • 8-head attention • MC-dropout uncertainty
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("API", "Bağlı" if st.session_state.api_ok else "Bağlı değil")
    k2.metric("Spot izin", "Açık" if st.session_state.spot_allowed else "Kapalı")
    k3.metric("Futures izin", "Açık" if st.session_state.futures_allowed else "Kapalı")
    k4.metric("Paper bakiye", f"{st.session_state.paper_balance:.2f} USDT")
    k5.metric("Açık pozisyon", len(st.session_state.positions))

    tab_auto, tab_tp, tab_train, tab_positions, tab_diag, tab_model = st.tabs([
        "🤖 Auto Spot/Futures",
        "⚡ TP Scalper",
        "🏋 Eğitim",
        "💼 Pozisyonlar",
        "🔎 API / 451",
        "🧠 Model",
    ])

    with tab_auto:
        st.subheader("AI Spot mı Futures mı karar versin")

        if st.button("🧠 Spot + Futures Tara", type="primary"):
            rows = []
            allow_spot = True if mode == "Paper" else st.session_state.spot_allowed
            allow_futures = True if mode == "Paper" else st.session_state.futures_allowed

            for symbol in symbols[:20]:
                if allow_spot and Path(spot_model_dir, "model.pt").exists():
                    try:
                        probs, df = predict_symbol(spot, symbol, interval, spot_model_dir)
                        book = spot.book_ticker(symbol)
                        sp = spread_bps(book)
                        regime, trend, vol = detect_regime(df)
                        direction = "LONG" if probs["BUY"] >= probs["SELL"] else "SHORT"
                        score = score_candidate(
                            "SPOT", direction, probs, sp, regime, trend, vol, 0.0
                        )
                        rows.append({
                            "Symbol": symbol,
                            "Market": "SPOT",
                            "Direction": "LONG" if direction == "LONG" else "NONE",
                            "BUY %": probs["BUY"],
                            "WAIT %": probs["WAIT"],
                            "SELL %": probs["SELL"],
                            "AI Güven %": probs.get("CONFIDENCE", 0.0),
                            "Belirsizlik %": probs.get("UNCERTAINTY", 100.0),
                            "Spread bps": sp,
                            "Funding %": 0.0,
                            "Regime": regime,
                            "Trend Score": trend,
                            "Vol %": vol,
                            "Router Score": score,
                        })
                    except Exception as exc:
                        rows.append({"Symbol": symbol, "Market": "SPOT", "Error": str(exc)})

                if allow_futures and Path(futures_model_dir, "model.pt").exists():
                    try:
                        probs, df = predict_symbol(
                            futures, symbol, interval, futures_model_dir
                        )
                        book = futures.book_ticker(symbol)
                        sp = spread_bps(book)
                        premium = futures.premium_index(symbol)
                        funding = float(premium.get("lastFundingRate", 0.0) or 0.0)
                        regime, trend, vol = detect_regime(df)
                        direction = "LONG" if probs["BUY"] >= probs["SELL"] else "SHORT"
                        score = score_candidate(
                            "FUTURES", direction, probs, sp, regime, trend, vol, funding
                        )
                        rows.append({
                            "Symbol": symbol,
                            "Market": "FUTURES",
                            "Direction": direction,
                            "BUY %": probs["BUY"],
                            "WAIT %": probs["WAIT"],
                            "SELL %": probs["SELL"],
                            "AI Güven %": probs.get("CONFIDENCE", 0.0),
                            "Belirsizlik %": probs.get("UNCERTAINTY", 100.0),
                            "Spread bps": sp,
                            "Funding %": funding * 100,
                            "Regime": regime,
                            "Trend Score": trend,
                            "Vol %": vol,
                            "Router Score": score,
                        })
                    except Exception as exc:
                        rows.append({"Symbol": symbol, "Market": "FUTURES", "Error": str(exc)})

            st.session_state.scan_rows = rows

        if st.session_state.scan_rows:
            df_scan = pd.DataFrame(st.session_state.scan_rows)
            if "Router Score" in df_scan.columns:
                df_scan = df_scan.sort_values("Router Score", ascending=False)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)

            valid = [
                row for row in st.session_state.scan_rows
                if "Router Score" in row
                and row.get("Direction") not in [None, "NONE"]
                and float(row.get("Spread bps", 9999)) <= max_spread
            ]

            if valid:
                best = max(valid, key=lambda row: row["Router Score"])
                st.success(
                    f"En iyi seçim: {best['Market']} / {best['Symbol']} / "
                    f"{best['Direction']} — skor {best['Router Score']:.2f}"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market", best["Market"])
                c2.metric("Direction", best["Direction"])
                c3.metric("AI Güven", f"%{best['AI Güven %']:.1f}")
                c4.metric("Regime", best["Regime"])

                can_open = best["Router Score"] >= min_score
                if not can_open:
                    st.warning("Router skoru eşik altında; işlem açılmıyor.")

                if st.button("🚀 AI Seçimini Aç", type="primary", disabled=not can_open):
                    try:
                        if mode == "Live":
                            if not st.session_state.api_ok or not live_confirm:
                                raise RuntimeError("Live API bağlantısı ve onayı gerekli.")
                            if best["Market"] == "SPOT" and not st.session_state.spot_allowed:
                                raise RuntimeError("API anahtarında Spot trading izni yok.")
                            if best["Market"] == "FUTURES" and not st.session_state.futures_allowed:
                                raise RuntimeError("API anahtarında Futures izni yok.")

                        price = get_price(best["Market"], best["Symbol"], spot, futures)
                        lev = 1 if best["Market"] == "SPOT" else leverage

                        if mode == "Paper":
                            open_paper(
                                best["Market"], best["Symbol"], best["Direction"],
                                margin_usdt, price, lev, tp_pct
                            )
                            result = {"paper": True}
                        else:
                            result = open_live(
                                best["Market"], best["Symbol"], best["Direction"],
                                margin_usdt, price, lev, tp_pct, spot, futures
                            )

                        st.session_state.trade_log.append({
                            "time": str(datetime.now()),
                            "action": "OPEN",
                            "market": best["Market"],
                            "symbol": best["Symbol"],
                            "direction": best["Direction"],
                            "entry": price,
                            "margin_usdt": margin_usdt,
                            "leverage": lev,
                            "tp_pct": tp_pct,
                            "router_score": best["Router Score"],
                            "mode": mode.upper(),
                            "result": str(result),
                        })
                        st.success("Pozisyon açıldı ve TP monitörüne eklendi.")
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.info("Filtreleri geçen işlem adayı yok.")

    with tab_tp:
        st.subheader("Yüzde kârda otomatik kapatma")
        st.write(f"Yeni pozisyon hedefi: **%{tp_pct:.2f}**")
        st.warning(
            "Bu sürüm otomatik stop-loss kullanmıyor. Futures'ta kaldıraç zarar riskini büyütür."
        )
        if st.button("🔄 TP'leri Şimdi Kontrol Et", use_container_width=True):
            events = monitor_tp(spot, futures)
            if events:
                for event in events:
                    st.info(event)
            else:
                st.success("Kontrol tamamlandı; kapanan pozisyon yok.")

        auto_tp = st.checkbox(
            "Panel açıkken her 3 saniyede TP kontrolü",
            value=False,
        )
        if auto_tp and hasattr(st, "fragment"):
            @st.fragment(run_every="3s")
            def auto_tp_fragment():
                events = monitor_tp(spot, futures)
                if events:
                    for event in events:
                        st.write(event)
                st.caption("Son kontrol: " + datetime.now().strftime("%H:%M:%S"))
            auto_tp_fragment()

    with tab_train:
        st.subheader("Spot / Futures ayrı model eğitimi")
        train_market = st.radio("Market", ["SPOT", "FUTURES"], horizontal=True)
        train_symbol = st.text_input("Eğitim symbol", "BTCUSDT")
        train_interval = st.selectbox(
            "Eğitim timeframe",
            ["1m", "3m", "5m", "15m", "30m", "1h"],
            index=2,
            key="train_interval_global",
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            bars = st.number_input("Geçmiş mum", 3000, 100000, 15000, 1000)
            seq = st.number_input("Sequence", 32, 256, 96, 16)
        with t2:
            horizon = st.number_input("Horizon", 1, 24, 3)
            threshold = st.number_input(
                "Label threshold", 0.0005, 0.05, 0.0025, 0.0005, format="%.4f"
            )
        with t3:
            epochs = st.number_input("Epoch", 1, 50, 10)
            batch = st.selectbox("Batch", [32, 64, 128, 256], index=2)
        lr = st.number_input(
            "Learning rate", 0.00001, 0.005, 0.00020, 0.00001, format="%.5f"
        )
        output_dir = st.text_input(
            "Eğitim çıktı klasörü",
            "models/spot" if train_market == "SPOT" else "models/futures",
        )

        if st.button("🏋 Modeli Eğit", type="primary"):
            try:
                with st.spinner("Derin öğrenme modeli eğitiliyor..."):
                    result = train_model(
                        symbol=train_symbol.upper(),
                        interval=train_interval,
                        bars=int(bars),
                        sequence_length=int(seq),
                        horizon=int(horizon),
                        threshold=float(threshold),
                        epochs=int(epochs),
                        batch_size=int(batch),
                        out_dir=output_dir,
                        lr=float(lr),
                        market=train_market,
                    )
                    clear_model_cache()
                st.success("Model eğitildi.")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

    with tab_positions:
        st.subheader("Açık pozisyonlar")
        rows = []
        for key, pos in st.session_state.positions.items():
            try:
                current = get_price(pos["market"], pos["symbol"], spot, futures)
                profit = pnl_pct(pos, current)
                pnl_usdt = pos["margin"] * (profit / 100.0)
            except Exception:
                current = None
                profit = None
                pnl_usdt = None
            rows.append({
                "Key": key,
                "Market": pos["market"],
                "Symbol": pos["symbol"],
                "Direction": pos["direction"],
                "Mode": pos["mode"],
                "Entry": pos["entry"],
                "Current": current,
                "Leverage": pos["leverage"],
                "TP %": pos["tp_pct"],
                "PnL %": profit,
                "PnL USDT est": pnl_usdt,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Açık pozisyon yok.")

        st.subheader("İşlem logu")
        if st.session_state.trade_log:
            log_df = pd.DataFrame(st.session_state.trade_log)
            st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True)
            st.download_button(
                "CSV indir",
                log_df.to_csv(index=False).encode("utf-8"),
                file_name="binance_global_ai_trades.csv",
                mime="text/csv",
            )

    with tab_diag:
        st.subheader("Global API ve HTTP 451 teşhisi")
        if st.session_state.diag:
            st.json(st.session_state.diag)

        if st.button("🌐 Sadece public ağ erişimini test et"):
            results = {}
            try:
                data, base = _spot_public("/api/v3/time")
                results["Spot"] = {
                    "ok": True,
                    "endpoint": base,
                    "serverTime": data.get("serverTime"),
                }
            except Exception as exc:
                results["Spot"] = {"ok": False, "error": str(exc)}
            try:
                fclient = BinanceFuturesClient()
                results["Futures"] = {
                    "ok": True,
                    "serverTime": fclient.server_time(),
                }
            except Exception as exc:
                results["Futures"] = {"ok": False, "error": str(exc)}
            st.json(results)

        st.markdown(
            """
`/api/v3/time` public endpointtir; API Key istemez. Burada HTTP 451 alıyorsan
API Key/Secret henüz doğrulanmamıştır. Sorun isteğin çıktığı ağ/sunucu erişimidir.
Bu uygulama yasal/bölgesel kısıtlamayı aşmaya çalışmaz.
            """
        )

        perms = st.session_state.api_permissions or {}
        if perms:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Reading", str(perms.get("enableReading")))
            p2.metric("Spot", str(perms.get("enableSpotAndMarginTrading")))
            p3.metric("Futures", str(perms.get("enableFutures")))
            p4.metric("IP Restrict", str(perms.get("ipRestrict")))
            if perms.get("enableWithdrawals", False):
                st.error("Withdrawal izni açık görünüyor. Trading botunda kapat.")

    with tab_model:
        st.subheader("Derin öğrenme mimarisi")
        st.code(
            """
200 causal features
        ↓
LayerNorm + Dense Projection
        ↓
Residual Temporal CNN ×4 (k=3/5/7/9, dilation=1/2/4/8)
        ↓
Bidirectional LSTM ×2
        ↓
Bidirectional GRU ×2
        ↓
Transformer Encoder ×4
        ↓
8-Head Self Attention
        ↓
Gated Last / Mean / Max Pooling
        ↓
Dense Classifier
        ↓
SELL / WAIT / BUY
        ↓
Monte-Carlo Dropout
        ↓
Confidence / Uncertainty
        ↓
Regime + Spread + Funding + Volatility Router
        ↓
Spot / Futures Selection
            """,
            language="text",
        )
        st.info(
            "Daha büyük model tek başına daha iyi değildir. "
            "Bu sürüm düşük güven, yüksek spread, ters rejim ve Futures funding maliyetini puanda cezalandırır."
        )

    st.caption(
        "Binance Global Deep AI Scalper • Kâr garantisi yoktur."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    if "--train" in sys.argv:
        cli_train()
    else:
        run_streamlit_panel()
