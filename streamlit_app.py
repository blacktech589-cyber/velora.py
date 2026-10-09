import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Hyperactive Spot Scanner",
    page_icon="⚡",
    layout="wide",
)

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
GATE = "https://api.gateio.ws/api/v4"
INTERVAL_SECONDS = 900
BARS_90_DAYS = 90 * 24 * 4
TIMEOUT = 18

session = requests.Session()
session.headers.update({"User-Agent": "HyperactiveSpotScanner/1.1"})


def binance_json(path, params=None):
    errors = []

    for host in BINANCE_HOSTS:
        try:
            response = session.get(
                host + path,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue

            response.raise_for_status()
            return response.json(), host

        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:100]}")

    raise RuntimeError(
        "Binance API adreslerine erişilemedi. " + " | ".join(errors)
    )


def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# BINANCE SPOT PİYASALARI
# ============================================================

def binance_markets():
    info, host = binance_json("/api/v3/exchangeInfo")
    markets = []

    for item in info.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol")

        if symbol:
            markets.append({
                "symbol": symbol,
                "market_id": symbol,
                "base_asset": item.get("baseAsset", ""),
                "provider": "Binance",
            })

    if not markets:
        raise RuntimeError(
            "Aktif Binance Spot USDT piyasası bulunamadı."
        )

    return markets, host


def binance_tickers():
    data, host = binance_json("/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        try:
            tickers[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h_pct": float(
                    item["priceChangePercent"]
                ),
                "trades_24h": int(item["count"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers, host


# ============================================================
# GATE.IO SPOT PİYASALARI
# ============================================================

def gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")
    markets = []

    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")

        if not market_id or not base:
            continue

        if item.get("quote") != "USDT":
            continue

        if str(item.get("trade_status", "")).lower() != "tradable":
            continue

        if item.get("delisted") is True:
            continue

        markets.append({
            "symbol": base + "USDT",
            "market_id": market_id,
            "base_asset": base,
            "provider": "Gate.io",
        })

    if not markets:
        raise RuntimeError(
            "Aktif Gate.io Spot USDT piyasası bulunamadı."
        )

    return markets


def gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")
    tickers = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            tickers[market_id] = {
                "price": float(item["last"]),
                "quote_volume": float(item["quote_volume"]),
                "change_24h_pct": float(
                    item["change_percentage"]
                ),
                "trades_24h": np.nan,
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


# ============================================================
# BORSAYI SEÇ
# ============================================================

def discover(exchange):
    if exchange == "Binance only":
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()

        return (
            markets,
            tickers,
            "Binance",
            f"{host1}; ticker={host2}",
        )

    if exchange == "Gate.io only":
        return (
            gate_markets(),
            gate_tickers(),
            "Gate.io",
            GATE,
        )

    try:
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()

        return (
            markets,
            tickers,
            "Binance",
            f"{host1}; ticker={host2}",
        )

    except Exception as exc:
        markets = gate_markets()
        tickers = gate_tickers()

        return (
            markets,
            tickers,
            "Gate.io",
            f"Gate.io fallback; Binance: {str(exc)[:150]}",
        )


# ============================================================
# MUM VERİSİNİ TEMİZLE
# ============================================================

def clean_candles(rows):
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.dropna(subset=cols[:6])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    # Henüz tamamlanmamış 15 dakikalık mumu kullanma.
    df = df[df["timestamp"] < current_start]

    df = df[
        (df["open"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["close"] > 0)
    ]

    return df.reset_index(drop=True)


# ============================================================
# BINANCE'TAN 90 GÜNLÜK 15 DAKİKALIK MUMLAR
# ============================================================

def binance_candles(symbol):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    start_ms = (
        current_start - (BARS_90_DAYS + 10) * INTERVAL_SECONDS
    ) * 1000

    end_ms = current_start * 1000
    cursor = start_ms
    rows = []

    while cursor < end_ms:
        data, _ = binance_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "15m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not data:
            break

        for candle in data:
            try:
                rows.append({
                    "timestamp": int
