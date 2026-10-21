from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VELORA — 3-CANDLE REVERSAL SCANNER
# BUY: swing low, then 3 consecutive rising/green candles.
# SELL: swing high, then 3 consecutive falling/red candles.
# Auto-scan refresh interval: 40 seconds. No trade execution.
# ============================================================

APP_VERSION = "velora-reversal-40s-v1"
REFRESH_MS = 40_000
DEFAULT_HISTORY = 1_000
TARGET_HISTORY = 100_000
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VeloraScanner/4.0)",
    "Accept": "application/json,text/plain,*/*",
}

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

# Asset names and payout percentages supplied by the user.
# Public market data is not identical to Binomo's private OTC prices.
RAW_ASSETS = [
    ("Crypto IDX", "Crypto", "index", "CRYPTO_IDX", 82, 84),
    ("Bitcoin (OTC)", "Crypto", "crypto", "BTCUSDT", 80, 82),
    ("Ethereum (OTC)", "Crypto", "crypto", "ETHUSDT", 80, 82),
    ("Solana (OTC)", "Crypto", "crypto", "SOLUSDT", 80, 82),
    ("EUR/USD", "Forex", "fx", "EUR/USD", 80, 82),
    ("EUR/HUF", "Forex", "fx", "EUR/HUF", 80, 82),
    ("GBP/NOK", "Forex", "fx", "GBP/NOK", 80, 82),
    ("CAD/JPY", "Forex", "fx", "CAD/JPY", 80, 82),
    ("FC Barcelona Token (OTC)", "Crypto", "crypto", "BARUSDT", 80, 82),
    ("AUD/ZAR", "Forex", "fx", "AUD/ZAR", 80, 82),
    ("CAD/DKK", "Forex", "fx", "CAD/DKK", 80, 82),
    ("EUR/ILS", "Forex", "fx", "EUR/ILS", 80, 82),
    ("GBP/SGD", "Forex", "fx", "GBP/SGD", 80, 82),
    ("USD/MXN", "Forex", "fx", "USD/MXN", 80, 82),
    ("Cardano (OTC)", "Crypto", "crypto", "ADAUSDT", 80, 82),
    ("EUR/ZAR", "Forex", "fx", "EUR/ZAR", 80, 82),
    ("AUD/USD", "Forex", "fx", "AUD/USD", 80, 82),
    ("GBP/HKD", "Forex", "fx", "GBP/HKD", 80, 82),
    ("CAD/MXN", "Forex", "fx", "CAD/MXN", 80, 82),
    ("EUR/NOK", "Forex", "fx", "EUR/NOK", 80, 82),
    ("EUR/MXN", "Forex", "fx", "EUR/MXN", 80, 82),
    ("CHF/JPY", "Forex", "fx", "CHF/JPY", 80, 82),
    ("USD/ILS", "Forex", "fx", "USD/ILS", 80, 82),
    ("USD/SEK", "Forex", "fx", "USD/SEK", 80, 82),
    ("CAD/NOK", "Forex", "fx", "CAD/NOK", 80, 82),
    ("CAD/SEK", "Forex", "fx", "CAD/SEK", 80, 82),
    ("USD/PLN", "Forex", "fx", "USD/PLN", 80, 82),
    ("AUD/JPY", "Forex", "fx", "AUD/JPY", 80, 82),
    ("EUR/SEK", "Forex", "fx", "EUR/SEK", 80, 82),
    ("AUD/SGD", "Forex", "fx", "AUD/SGD", 80, 82),
    ("Chainlink (OTC)", "Crypto", "crypto", "LINKUSDT", 80, 82),
    ("GBP/MXN", "Forex", "fx", "GBP/MXN", 80, 82),
    ("CAD/SGD", "Forex", "fx", "CAD/SGD", 80, 82),
    ("AUD/DKK", "Forex", "fx", "AUD/DKK", 80, 82),
    ("EUR/SGD", "Forex", "fx", "EUR/SGD", 80, 82),
    ("CHF/DKK", "Forex", "fx", "CHF/DKK", 80, 82),
    ("GBP/PLN", "Forex", "fx", "GBP/PLN", 80, 82),
    ("GBP/CZK", "Forex", "fx", "GBP/CZK", 80, 82),
    ("EUR/NZD", "Forex", "fx", "EUR/NZD", 80, 82),
    ("AUD/HUF", "Forex", "fx", "AUD/HUF", 80, 82),
    ("EUR/JPY", "Forex", "fx", "EUR/JPY", 80, 82),
    ("USD/HUF", "Forex", "fx", "USD/HUF", 80, 82),
    ("GBP/DKK", "Forex", "fx", "GBP/DKK", 80, 82),
    ("AUD/NOK", "Forex", "fx", "AUD/NOK", 80, 82),
    ("NZD/JPY", "Forex", "fx", "NZD/JPY", 80, 82),
    ("CHF/NOK", "Forex", "fx", "CHF/NOK", 80, 82),
    ("USD/JPY", "Forex", "fx", "USD/JPY", 80, 82),
    ("CHF/PLN", "Forex", "fx", "CHF/PLN", 80, 82),
    ("GBP/TRY", "Forex", "fx", "GBP/TRY", 80, 82),
    ("USD/CAD", "Forex", "fx", "USD/CAD", 80, 82),
    ("AUD/SEK", "Forex", "fx", "AUD/SEK", 80, 82),
    ("CHF/SEK", "Forex", "fx", "CHF/SEK", 80, 82),
    ("CHF/SGD", "Forex", "fx", "CHF/SGD", 80, 82),
    ("NOK/JPY", "Forex", "fx", "NOK/JPY", 80, 82),
    ("NOK/SEK", "Forex", "fx", "NOK/SEK", 80, 82),
    ("NZD/CAD", "Forex", "fx", "NZD/CAD", 80, 82),
    ("NZD/DKK", "Forex", "fx", "NZD/DKK", 80, 82),
    ("GBP/NZD", "Forex", "fx", "GBP/NZD", 80, 82),
    ("NZD/USD", "Forex", "fx", "NZD/USD", 80, 82),
    ("NZD/HUF", "Forex", "fx", "NZD/HUF", 80, 82),
    ("NZD/NOK", "Forex", "fx", "NZD/NOK", 80, 82),
    ("NZD/SEK", "Forex", "fx", "NZD/SEK", 80, 82),
    ("NZD/SGD", "Forex", "fx", "NZD/SGD", 80, 82),
    ("SEK/JPY", "Forex", "fx", "SEK/JPY", 80, 82),
    ("SGD/JPY", "Forex", "fx", "SGD/JPY", 80, 82),
    ("SGD/HKD", "Forex", "fx", "SGD/HKD", 80, 82),
    ("USD/CHF", "Forex", "fx", "USD/CHF", 80, 82),
    ("AUD/CAD", "Forex", "fx", "AUD/CAD", 80, 82),
    ("USD/CZK", "Forex", "fx", "USD/CZK", 80, 82),
    ("Bitcoin Cash (OTC)", "Crypto", "crypto", "BCHUSDT", 80, 82),
    ("Kusama (OTC)", "Crypto", "crypto", "KSMUSDT", 80, 82),
    ("Aave (OTC)", "Crypto", "crypto", "AAVEUSDT", 80, 82),
    ("Pancake Swap (OTC)", "Crypto", "crypto", "CAKEUSDT", 80, 82),
    ("Uniswap (OTC)", "Crypto", "crypto", "UNIUSDT", 80, 82),
    ("NZD/CHF", "Forex", "fx", "NZD/CHF", 80, 82),
    ("GBP/CAD", "Forex", "fx", "GBP/CAD", 80, 82),
    ("AUD/NZD", "Forex", "fx", "AUD/NZD", 80, 82),
    ("USD/ZAR", "Forex", "fx", "USD/ZAR", 80, 82),
    ("AUD/CHF", "Forex", "fx", "AUD/CHF", 80, 82),
    ("GBP/USD", "Forex", "fx", "GBP/USD", 70, 72),
    ("Gold", "Other", "yahoo", "GC=F", 60, 62),
    ("GBP/HUF", "Forex", "fx", "GBP/HUF", 50, 52),
    ("EUR/PLN", "Forex", "fx", "EUR/PLN", 50, 52),
    ("ZAR/JPY", "Forex", "fx", "ZAR/JPY", 50, 52),
    ("CHF/HUF", "Forex", "fx", "CHF/HUF", 50, 52),
]

YAHOO_USD_QUOTE = {
    "USD/JPY": "JPY=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "USD/ZAR": "ZAR=X",
    "USD/SEK": "SEK=X",
    "USD/PLN": "PLN=X",
    "USD/MXN": "MXN=X",
    "USD/ILS": "ILS=X",
    "USD/HUF": "HUF=X",
    "USD/CZK": "CZK=X",
}

ASSETS: list[dict[str, Any]] = []
for name, category, kind, symbol, payout, vip_payout in RAW_ASSETS:
    data_symbol = symbol
    if kind == "fx":
        data_symbol = YAHOO_USD_QUOTE.get(
            symbol,
            symbol.replace("/", "") + "=X",
        )
    ASSETS.append({
        "name": name,
        "symbol": symbol,
        "data_symbol": data_symbol,
        "category": category,
        "kind": kind,
        "payout": payout,
        "vip_payout": vip_payout,
    })

COINBASE = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
    "ADAUSDT": "ADA-USD",
    "AVAXUSDT": "AVAX-USD",
    "LINKUSDT": "LINK-USD",
    "BCHUSDT": "BCH-USD",
    "KSMUSDT": "KSM-USD",
    "AAVEUSDT": "AAVE-USD",
    "CAKEUSDT": "CAKE-USD",
    "UNIUSDT": "UNI-USD",
}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def current_minute() -> pd.Timestamp:
    return now_utc().floor("min")


def http_get(url: str, params: dict | None = None) -> requests.Response:
    return requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )


def get_data_dir() -> Path:
    choices = (
        [Path(os.environ["VELORA_DATA_DIR"])]
        if os.environ.get("VELORA_DATA_DIR")
        else [Path.cwd(), Path("/tmp/velora_data")]
    )
    for folder in choices:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".velora_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return folder
        except Exception:
            continue
    fallback = Path(tempfile.gettempdir()) / "velora_data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DATA_DIR = get_data_dir()
DB_PATH = DATA_DIR / "velora_history.sqlite3"
EXCEL_PATH = DATA_DIR / "velora_scan_history.xlsx"


def init_db() -> None:
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='candles'"
        ).fetchone() is not None
        required = {"symbol", "time", "open", "high", "low", "close", "volume"}
        cols = (
            {row[1] for row in conn.execute("PRAGMA table_info(candles)")}
            if exists else set()
        )

        if exists and not required.issubset(cols):
            conn.execute("DROP INDEX IF EXISTS idx_symbol_time")
            conn.execute("ALTER TABLE candles RENAME TO candles_old")
            old_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(candles_old)")
            }
            conn.execute(
                "CREATE TABLE candles (symbol TEXT NOT NULL,time TEXT NOT NULL,"
                "open REAL NOT NULL,high REAL NOT NULL,low REAL NOT NULL,"
                "close REAL NOT NULL,volume REAL DEFAULT 0,PRIMARY KEY(symbol,time))"
            )
            time_col = next(
                (x for x in ("time", "candle_time", "timestamp") if x in old_cols),
                None,
            )
            if time_col and {
                "symbol", "open", "high", "low", "close"
            }.issubset(old_cols):
                volume_expr = "volume" if "volume" in old_cols else "0"
                conn.execute(
                    "INSERT OR REPLACE INTO candles(symbol,time,open,high,low,close,volume) "
                    f"SELECT symbol,{time_col},open,high,low,close,{volume_expr} "
                    "FROM candles_old "
                    f"WHERE symbol IS NOT NULL AND {time_col} IS NOT NULL"
                )
            conn.execute("DROP TABLE candles_old")

        elif not exists:
            conn.execute(
                "CREATE TABLE candles (symbol TEXT NOT NULL,time TEXT NOT NULL,"
                "open REAL NOT NULL,high REAL NOT NULL,low REAL NOT NULL,"
                "close REAL NOT NULL,volume REAL DEFAULT 0,PRIMARY KEY(symbol,time))"
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol,time)"
        )
        conn.commit()


def normalize_candles(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["time", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.copy()
    for col in cols:
        if col not in d.columns:
            d[col] = 0.0 if col == "volume" else np.nan

    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["volume"] = d["volume"].fillna(0)
    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    d = d[
        (d["open"] > 0)
        & (d["high"] > 0)
        & (d["low"] > 0)
        & (d["close"] > 0)
    ]

    # Only completed one-minute candles are used.
    d = d[d["time"] < current_minute()]

    return (
        d.drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)[cols]
    )


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    d = normalize_candles(df)
    if d.empty:
        return

    rows = []
    for row in d.itertuples(index=False):
        rows.append((
            symbol,
            pd.Timestamp(row.time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume or 0),
        ))

    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()


def load_cache(symbol: str) -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        d = pd.read_sql_query(
            "SELECT time,open,high,low,close,volume "
            "FROM candles WHERE symbol=? ORDER BY time",
            conn,
            params=(symbol,),
        )
    return normalize_candles(d)


def parse_binance(rows: list) -> pd.DataFrame:
    labels = [
        "stamp", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tb", "tq", "ignore",
    ]
    d = pd.DataFrame(rows, columns=labels)
    d["time"] = pd.to_datetime(
        d["stamp"], unit="ms", utc=True, errors="coerce"
    )
    return normalize_candles(
        d[["time", "open", "high", "low", "close", "volume"]]
    )


def fetch_binance(
    symbol: str,
    target: int,
) -> tuple[pd.DataFrame, str, str | None]:
    errors = []

    for base in BINANCE_ENDPOINTS:
        try:
            pages = []
            end_time = None

            while True:
                current_count = (
                    len
