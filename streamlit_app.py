import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Velora Scanner", page_icon="📊", layout="wide")

DB_PATH = "velora_history.sqlite3"
TARGET_CANDLES = 100_000
TIMEOUT = 15

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]
YAHOO_HOSTS = [
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
]

ASSETS = [
    {"name": "Bitcoin", "symbol": "BTCUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Ethereum", "symbol": "ETHUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Solana", "symbol": "SOLUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "XRP", "symbol": "XRPUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Dogecoin", "symbol": "DOGEUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Cardano", "symbol": "ADAUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Avalanche", "symbol": "AVAXUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "Chainlink", "symbol": "LINKUSDT", "category": "Crypto", "provider": "binance"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "category": "Forex", "provider": "yahoo"},
    {"name": "GBP/USD", "symbol": "GBPUSD=X", "category": "Forex", "provider": "yahoo"},
    {"name": "USD/JPY", "symbol": "JPY=X", "category": "Forex", "provider": "yahoo"},
    {"name": "AUD/USD", "symbol": "AUDUSD=X", "category": "Forex", "provider": "yahoo"},
    {"name": "USD/CAD", "symbol": "CAD=X", "category": "Forex", "provider": "yahoo"},
    {"name": "USD/CHF", "symbol": "CHF=X", "category": "Forex", "provider": "yahoo"},
    {"name": "NZD/USD", "symbol": "NZDUSD=X", "category": "Forex", "provider": "yahoo"},
    {"name": "EUR/GBP", "symbol": "EURGBP=X", "category": "Forex", "provider": "yahoo"},
    {"name": "Gold futures", "symbol": "GC=F", "category": "Other", "provider": "yahoo"},
    {"name": "Apple", "symbol": "AAPL", "category": "Other", "provider": "yahoo"},
    {"name": "Microsoft", "symbol": "MSFT", "category": "Other", "provider": "yahoo"},
    {"name": "NVIDIA", "symbol": "NVDA", "category": "Other", "provider": "yahoo"},
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 VeloraScanner/1.0",
    "Accept": "application/json",
})


# ---------------------------------------------------------
# SQLITE DATABASE
# ---------------------------------------------------------

def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                candle_time TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0,
                PRIMARY KEY(symbol, candle_time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_time
            ON candles(symbol, candle_time)
        """)
        conn.commit()


def load_cache(symbol):
    init_db()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            """
            SELECT candle_time AS time, open, high, low, close, volume
            FROM candles
            WHERE symbol=?
            ORDER BY candle_time
            """,
            conn,
            params=(symbol,),
        )

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["time", "open", "high", "low", "close"])

    return df


def save_cache(symbol, df):
    if df is None or df.empty:
        return

    rows = []

    for r in df.itertuples(index=False):
        ts = pd.Timestamp(r.time)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        vol = getattr(r, "volume", 0)

        rows.append((
            symbol,
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            0.0 if pd.isna(vol) else float(vol),
        ))

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


# ---------------------------------------------------------
# BINANCE PUBLIC API - CRYPTO
# ---------------------------------------------------------

def fetch_binance(symbol):
    """
    Fetch recent completed 1-minute Binance Spot candles.
    No API key is required.
    Binance returns at most 1,000 candles per request here.
    """

    errors = []

    for base in BINANCE_BASES:
        try:
            response = SESSION.get(
                base + "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": 1000,
                },
                timeout=TIMEOUT,
            )

            if response.status_code == 451:
                raise RuntimeError(
                    "HTTP 451: Binance API is unavailable from this "
                    "network or region."
                )

            if response.status_code == 429:
                raise RuntimeError(
                    "HTTP 429: Binance rate limit. Wait and retry."
                )

            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {response.status_code}: access denied "
                    "by provider or network."
                )

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list) or not payload:
                raise RuntimeError(
                    f"No Binance candles returned: {str(payload)[:250]}"
                )

            columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ]

            df = pd.DataFrame(payload, columns=columns)

            df["
