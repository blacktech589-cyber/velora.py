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

APP_VERSION = "velora-reversal-40s-v1"
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

# Asset names and payout percentages provided by the user.
# Public market prices are not identical to Binomo OTC prices.
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

ASSETS = []
for name, category, kind, symbol, payout, vip_payout in RAW_ASSETS:
    data_symbol = symbol
    if kind == "fx":
        data_symbol = YAHOO_USD_QUOTE.get(
            symbol, symbol.replace("/", "") + "=X"
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
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD", "DOGEUSDT": "DOGE-USD", "ADAUSDT": "ADA-USD",
    "AVAXUSDT": "AVAX-USD", "LINKUSDT": "LINK-USD",
    "BCHUSDT": "BCH-USD", "KSMUSDT": "KSM-USD",
    "AAVEUSDT": "AAVE-USD", "CAKEUSDT": "CAKE-USD",
    "UNIUSDT": "UNI-USD",
}


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def current_minute():
    return now_utc().floor("min")


def http_get(url, params=None):
    return requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )


def get_data_dir():
    requested = os.environ.get("VELORA_DATA_DIR")
    candidates = (
        [Path(requested)]
        if requested
        else [Path.cwd(), Path("/tmp/velora_data")]
    )

    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".velora_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return folder
        except Exception:
            pass

    folder = Path(tempfile.gettempdir()) / "velora_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


DATA_DIR = get_data_dir()
DB_PATH = DATA_DIR / "velora_history.sqlite3"
EXCEL_PATH = DATA_DIR / "velora_scan_history.xlsx"


def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='candles'"
        ).fetchone() is not None

        required = {
            "symbol", "time", "open", "high", "low", "close", "volume"
        }

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
                "close REAL NOT NULL,volume REAL DEFAULT 0,"
                "PRIMARY KEY(symbol,time))"
            )
            time_col = next(
                (
                    x for x in ("time", "candle_time", "timestamp")
                    if x in old_cols
                ),
                None,
            )
            if time_col and {
                "symbol", "open", "high", "low", "close"
            }.issubset(old_cols):
                volume_expr = "volume" if "volume" in old_cols else "0"
                conn.execute(
                    "INSERT OR REPLACE INTO candles "
                    "(symbol,time,open,high,low,close,volume) "
                    f"SELECT symbol,{time_col},open,high,low,close,"
                    f"{volume_expr} FROM candles_old "
                    f"WHERE symbol IS NOT NULL AND {time_col} IS NOT NULL"
                )
            conn.execute("DROP TABLE candles_old")

        elif not exists:
            conn.execute(
                "CREATE TABLE candles (symbol TEXT NOT NULL,time TEXT NOT NULL,"
                "open REAL NOT NULL,high REAL NOT NULL,low REAL NOT NULL,"
                "close REAL NOT NULL,volume REAL DEFAULT 0,"
                "PRIMARY KEY(symbol,time))"
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol,time)"
        )
        conn.commit()


def normalize_candles(df):
    cols = ["time", "open", "high", "low", "close", "volume"]

    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.copy()

    for col in cols:
        if col not in d.columns:
            d[col] = 0.0 if col == "volume" else np.nan

    d["time"] = pd.to_datetime(
        d["time"], utc=True, errors="coerce"
    )

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

    d = d[d["time"] < current_minute()]

    return (
        d.drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)[cols]
    )


def save_cache(symbol, df):
    d = normalize_candles(df)
    if d.empty:
        return

    rows = []
    for row in d.itertuples(index=False):
        rows.append((
            symbol,
            pd.Timestamp(row.time).tz_convert("UTC").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
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


def load_cache(symbol):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        d = pd.read_sql_query(
            "SELECT time,open,high,low,close,volume "
            "FROM candles WHERE symbol=? ORDER BY time",
            conn,
            params=(symbol,),
        )
    return normalize_candles(d)


def parse_binance(rows):
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


def fetch_binance(symbol, target):
    errors = []

    for base in BINANCE_ENDPOINTS:
        try:
            pages = []
            end_time = None

            while True:
                current_count = (
                    len(normalize_candles(pd.concat(pages, ignore_index=True)))
                    if pages else 0
                )
                limit = min(1000, max(1, target - current_count))
                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                }

                if end_time is not None:
                    params["endTime"] = end_time

                response = http_get(base + "/api/v3/klines", params)

                if response.status_code in (403, 451):
                    raise RuntimeError(
                        f"HTTP {response.status_code}: Binance endpoint blocked"
                    )
                if response.status_code == 429:
                    raise RuntimeError("HTTP 429: Binance rate limit")

                response.raise_for_status()
                rows = response.json()

                if not isinstance(rows, list) or not rows:
                    raise RuntimeError(
                        f"No Binance candles returned: {str(rows)[:160]}"
                    )

                pages.append(parse_binance(rows))
                combined = normalize_candles(
                    pd.concat(pages, ignore_index=True)
                )

                if len(combined) >= target or target <= 1000:
                    break

                end_time = (
                    int(combined["time"].min().timestamp() * 1000) - 1
                )
                time.sleep(0.05)

                if len(pages) >= 110:
                    break

            return (
                normalize_candles(pd.concat(pages, ignore_index=True))
                .tail(target).reset_index(drop=True),
                base,
                None,
            )

        except Exception as exc:
            errors.append(f"{base}: {type(exc).__name__}: {exc}")

    raise RuntimeError("Binance failed: " + " | ".join(errors))


def fetch_bybit(symbol, target):
    pages, end_time, warning = [], None, None

    while True:
        have = (
            len(normalize_candles(pd.concat(pages, ignore_index=True)))
            if pages else 0
        )
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": "1",
            "limit": min(1000, max(40, target - have)),
        }

        if end_time is not None:
            params["end"] = end_time

        response = http_get(
            "https://api.bybit.com/v5/market/kline", params
        )
        response.raise_for_status()
        obj = response.json()

        if obj.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit {obj.get('retCode')}: {obj.get('retMsg')}"
            )

        rows = (obj.get("result") or {}).get("list") or []
        if not rows:
            break

        frame = pd.DataFrame(
            rows,
            columns=[
                "stamp", "open", "high", "low", "close",
                "volume", "turnover",
            ],
        )
        frame["time"] = pd.to_datetime(
            pd.to_numeric(frame["stamp"], errors="coerce"),
            unit="ms", utc=True,
        )
        pages.append(normalize_candles(
            frame[["time", "open", "high", "low", "close", "volume"]]
        ))

        merged = normalize_candles(pd.concat(pages, ignore_index=True))
        if len(merged) >= target or target <= 1000:
            break

        end_time = int(merged["time"].min().timestamp() * 1000) - 1
        time.sleep(0.05)

        if len(pages) >= 110:
            warning = "Bybit historical paging capped at 110 pages."
            break

    if not pages:
        raise RuntimeError("Bybit returned no candles")

    return (
        normalize_candles(pd.concat(pages, ignore_index=True))
        .tail(target).reset_index(drop=True),
        "Bybit public API",
        warning,
    )


def fetch_okx(symbol, target):
    inst = symbol[:-4] + "-USDT"
    pages, after, warning = [], None, None

    while True:
        have = (
            len(normalize_candles(pd.concat(pages, ignore_index=True)))
            if pages else 0
        )
        params = {
            "instId": inst,
            "bar": "1m",
            "limit": str(min(300, max(40, target - have))),
        }

        if after is not None:
            params["after"] = after

        response = http_get(
            "https://www.okx.com/api/v5/market/candles", params
        )
        response.raise_for_status()
        obj = response.json()

        if str(obj.get("code")) != "0":
            raise RuntimeError(f"OKX {obj.get('code')}: {obj.get('msg')}")

        rows = obj.get("data") or []
        if not rows:
            break

        frame = pd.DataFrame(
            rows,
            columns=[
                "stamp", "open", "high", "low", "close",
                "volume", "vol_ccy", "vol_quote", "confirm",
            ],
        )
        frame["time"] = pd.to_datetime(
            pd.to_numeric(frame["stamp"], errors="coerce"),
            unit="ms", utc=True,
        )
        pages.append(normalize_candles(
            frame[["time", "open", "high", "low", "close", "volume"]]
        ))

        merged = normalize_candles(pd.concat(pages, ignore_index=True))
        if len(merged) >= target or target <= 300:
            break

        new_after = str(
            int(merged["time"].min().timestamp() * 1000)
        )
        if after == new_after:
            warning = "OKX historical paging stopped without advancing."
            break

        after = new_after
        time.sleep(0.05)

        if len(pages) >= 340:
            warning = "OKX historical paging capped at 340 pages."
            break

    if not pages:
        raise RuntimeError("OKX returned no candles")

    return (
        normalize_candles(pd.concat(pages, ignore_index=True))
        .tail(target).reset_index(drop=True),
        "OKX public API",
        warning,
    )


def fetch_gate(symbol, target):
    pair = symbol[:-4] + "_USDT"
    response = http_get(
        "https://api.gateio.ws/api/v4/spot/candlesticks",
        {
            "currency_pair": pair,
            "interval": "1m",
            "limit": min(1000, target),
        },
    )
    response.raise_for_status()
    rows = response.json()

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Gate.io returned no candles")

    frame = pd.DataFrame(
        rows,
        columns=[
            "stamp", "volume", "close", "high",
            "low", "open", "quote_volume",
        ],
    )
    frame["time"] = pd.to_datetime(
        pd.to_numeric(frame["stamp"], errors="coerce"),
        unit="s", utc=True,
    )
    return (
        normalize_candles(
            frame[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Gate.io public API",
        None,
    )


def fetch_coinbase(symbol):
    product = COINBASE.get(symbol)
    if not product:
        raise RuntimeError(
            "Coinbase has no fallback mapping for this asset"
        )

    end = now_utc().floor("min")
    start = end - pd.Timedelta(minutes=299)
    params = {
        "granularity": 60,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    }

    response = http_get(
        f"https://api.exchange.coinbase.com/products/{product}/candles",
        params,
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise RuntimeError("Coinbase returned no candles")

    frame = pd.DataFrame(
        rows,
        columns=["stamp", "low", "high", "open", "close", "volume"],
    )
    frame["time"] = pd.to_datetime(
        frame["stamp"], unit="s", utc=True
    )

    return (
        normalize_candles(
            frame[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Coinbase public API",
        None,
    )


def fetch_yahoo(asset):
    symbol = str(asset["data_symbol"])
    errors = []

    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            encoded = requests.utils.quote(symbol, safe="")
            response = http_get(
                f"https://{host}/v8/finance/chart/{encoded}",
                {
                    "range": "7d",
                    "interval": "1m",
                    "includePrePost": "false",
                },
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "HTTP 429: Yahoo rate limit; reduce selected assets or wait"
                )

            response.raise_for_status()
            chart = response.json().get("chart", {})

            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))

            items = chart.get("result") or []
            if not items:
                raise RuntimeError(
                    f"Yahoo has no minute data for {symbol}"
                )

            item = items[0]
            timestamps = item.get("timestamp") or []
            quotes = item.get("indicators", {}).get("quote") or [{}]

            if not timestamps:
                raise RuntimeError(
                    f"Yahoo returned no timestamps for {symbol}"
                )

            quote = quotes[0]
            frame = pd.DataFrame({
                "time": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })

            data = normalize_candles(frame)
            if data.empty:
                raise RuntimeError(
                    f"Yahoo returned no valid completed candles for {symbol}"
                )

            return data, f"Yahoo Finance ({host})", None

        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}: {exc}")

    raise RuntimeError("Yahoo failed: " + " | ".join(errors))


def fetch_live(asset, target):
    kind = asset["kind"]
    symbol = asset["symbol"]
    warnings = []

    if kind == "index":
        raise RuntimeError(
            "Crypto IDX is a broker-specific index. "
            "No matching public minute feed is configured."
        )

    if kind == "crypto":
        errors = []
        providers = [
            ("Binance", lambda: fetch_binance(symbol, target)),
            ("Bybit", lambda: fetch_bybit(symbol, target)),
            ("OKX", lambda: fetch_okx(symbol, target)),
            ("Gate.io", lambda: fetch_gate(symbol, target)),
            ("Coinbase", lambda: fetch_coinbase(symbol)),
        ]

        for provider, function in providers:
            try:
                data, source, warning = function()
                if provider != "Binance":
                    warnings.append(
                        f"{provider} prices are public-market prices, "
                        "not Binomo OTC quotes."
                    )
                if warning:
                    warnings.append(warning)
                return data, source, warnings
            except Exception as exc:
                errors.append(
                    f"{provider}: {type(exc).__name__}: {exc}"
                )

        raise RuntimeError(
            "All public crypto sources failed: " + " | ".join(errors)
        )

    data, source, warning = fetch_yahoo(asset)
    if warning:
        warnings.append(warning)
    return data, source, warnings


def get_candles(asset, target, deep_history):
    symbol = str(asset["symbol"])
    cached = load_cache(symbol)

    fetch_target = target
    if deep_history and len(cached) >= target - 1000:
        fetch_target = 1000

    errors = []
    warnings = []
    live = pd.DataFrame()
    source = ""

    try:
        live, source, provider_warnings = fetch_live(asset, fetch_target)
        warnings.extend(provider_warnings)
        if not live.empty:
            save_cache(symbol, live)
    except Exception as exc:
        errors.append(f"Live data: {type(exc).__name__}: {exc}")

    cached = load_cache(symbol)
    frames = [
        data for data in (cached, live)
        if data is not None and not data.empty
    ]

    if not frames:
        raise RuntimeError(
            "No live or cached candles. " + " | ".join(errors)
        )

    combined = (
        normalize_candles(pd.concat(frames, ignore_index=True))
        .tail(target)
        .reset_index(drop=True)
    )

    if len(combined) < 25:
        raise RuntimeError(
            f"Only {len(combined)} completed candles; need at least 25. "
            + " | ".join(errors)
        )

    age_minutes = (
        now_utc() - combined["time"].iloc[-1]
    ).total_seconds() / 60

    if age_minutes > 10:
        raise RuntimeError(
            f"Latest candle is {age_minutes:.1f} minutes old; "
            "refusing to create stale signal. " + " | ".join(errors)
        )

    return (
        combined,
        source or "SQLite cache",
        " | ".join(warnings + errors) if warnings or errors else None,
    )


def add_indicators(df):
    data = df.copy()
    data["ema15"] = data["close"].ewm(
        span=15, adjust=False, min_periods=15
    ).mean()
    data["bb_mid"] = data["close"].rolling(
        20, min_periods=20
    ).mean()
    sd = data["close"].rolling(20, min_periods=20).std(ddof=0)
    data["bb_upper"] = data["bb_mid"] + 2 * sd
    data["bb_lower"] = data["bb_mid"] - 2 * sd
    return data


def signal_for(df, lookback):
    result = {
        "signal": "NO SIGNAL",
        "reason": "Waiting for a confirmed 3-candle reversal.",
        "change_pct": np.nan,
        "trend": "Unknown",
        "three_candle_pattern": "Not confirmed",
        "swing_point": "Not confirmed",
        "direction_up_ok": False,
        "direction_down_ok": False,
        "green_ratio": np.nan,
        "red_ratio": np.nan,
    }

    if df is None or len(df) < max(lookback, 25):
        result["reason"] = (
            f"Need at least {max(lookback, 25)} completed candles."
        )
        return result

    window = df.tail(int(lookback)).copy()
    previous = window.iloc[:-3]
    last3 = window.iloc[-3:]

    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])

    pct = (
        (last_close - first_close) / first_close * 100
        if first_close else np.nan
    )

    opens = last3["open"].astype(float).to_numpy()
    closes = last3["close"].astype(float).to_numpy()

    red = bool(np.all(closes < opens))
    green = bool(np.all(closes > opens))
    falling_closes = bool(closes[0] > closes[1] > closes[2])
    rising_closes = bool(closes[0] < closes[1] < closes[2])

    previous_peak_idx = previous["high"].astype(float).idxmax()
    previous_bottom_idx = previous["low"].astype(float).idxmin()

    top_before_three = (
        float(previous.loc[previous_peak_idx, "high"])
        > float(last3["high"].max())
    )
    bottom_before_three = (
        float(previous.loc[previous_bottom_idx, "low"])
        < float(last3["low"].min())
    )

    sell_ok = red and falling_closes and top_before_three
    buy_ok = green and rising_closes and bottom_before_three

    green_ratio = float((window["close"] > window["open"]).mean())
    red_ratio = float((window["close"] < window["open"]).mean())

    if sell_ok:
        signal = "SELL"
        pattern = "3 consecutive falling/red candles"
        swing = "Swing high before last 3 candles"
        reason = (
            f"SELL: swing high followed by 3 completed red candles "
            f"with successively lower closes. Change across {lookback} "
            f"candles: {pct:+.3f}%."
        )
    elif buy_ok:
        signal = "BUY"
        pattern = "3 consecutive rising/green candles"
        swing = "Swing low before last 3 candles"
        reason = (
            f"BUY: swing low followed by 3 completed green candles "
            f"with successively higher closes. Change across {lookback} "
            f"candles: {pct:+.3f}%."
        )
    else:
        signal = "NO SIGNAL"
        pattern = (
            "3 red candles" if red and falling_closes
            else "3 green candles" if green and rising_closes
            else "Last 3 candles do not form a consecutive 3-candle pattern"
        )

        if red and falling_closes and not top_before_three:
            swing = "No confirmed swing high before last 3 candles"
        elif green and rising_closes and not bottom_before_three:
            swing = "No confirmed swing low before last 3 candles"
        else:
            swing = "Swing reversal not confirmed"

        reason = (
            f"No confirmed reversal. Pattern: {pattern}; "
            f"swing check: {swing}."
        )

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": pct,
        "trend": (
            "Falling" if sell_ok
            else "Rising" if buy_ok
            else "No confirmed reversal"
        ),
        "three_candle_pattern": pattern,
        "swing_point": swing,
        "direction_up_ok": rising_closes,
        "direction_down_ok": falling_closes,
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
    }


def scan_asset(asset, lookback, target, deep_history):
    try:
        candles, source, warning = get_candles(
            asset, target, deep_history
        )
        data = add_indicators(candles)
        signal = signal_for(data, lookback)
        last = data.iloc[-1]

        return {
            **asset,
            **signal,
            "price": float(last["close"]),
            "ema15": (
                float(last["ema15"])
                if pd.notna(last["ema15"]) else np.nan
            ),
            "bb_upper": (
                float(last["bb_upper"])
                if pd.notna(last["bb_upper"]) else np.nan
            ),
            "bb_mid": (
                float(last["bb_mid"])
                if pd.notna(last["bb_mid"]) else np.nan
            ),
            "bb_lower": (
                float(last["bb_lower"])
                if pd.notna(last["bb_lower"]) else np.nan
            ),
            "count": len(data),
            "last_candle_utc": str(last["time"]),
            "source": source,
            "candles": data,
            "error": None,
            "warning": warning,
        }
    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "reason": "No signal because valid recent price data is unavailable.",
            "change_pct": np.nan,
            "trend": "Unknown",
            "three_candle_pattern": "Unavailable",
            "swing_point": "Unavailable",
            "direction_up_ok": False,
            "direction_down_ok": False,
            "green_ratio": np.nan,
            "red_ratio": np.nan,
            "price": np.nan,
            "ema15": np.nan,
            "bb_upper": np.nan,
            "bb_mid": np.nan,
            "bb_lower": np.nan,
            "count": 0,
            "last_candle_utc": None,
            "source": "Unavailable",
            "candles": None,
            "error": f"{type(exc).__name__}: {exc}",
            "warning": None,
        }


def safe_excel_value(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def export_row(result, scan_time):
    fields = [
        "name", "symbol", "category", "kind", "payout", "vip_payout",
        "signal", "reason", "trend", "three_candle_pattern", "swing_point",
        "change_pct", "green_ratio", "red_ratio", "price", "ema15",
        "bb_upper", "bb_mid", "bb_lower", "count", "last_candle_utc",
        "source", "warning", "error",
    ]
    row = {
        "scan_time_utc": scan_time,
        **{field: result.get(field) for field in fields},
    }
    return {
        key: safe_excel_value(value)
        for key, value in row.items()
    }


def save_excel_history(results, scan_time):
    result_table = pd.DataFrame([
        export_row(result, scan_time)
        for result in results
    ])

    error_table = result_table[
        result_table["error"].notna()
    ].copy()

    summary = pd.DataFrame([{
        "scan_time_utc": scan_time,
        "assets_scanned": len(results),
        "data_ok": sum(not bool(r.get("error")) for r in results),
        "data_errors": sum(bool(r.get("error")) for r in results),
        "BUY": sum(
            r.get("signal") == "BUY" and not r.get("error")
            for r in results
        ),
        "SELL": sum(
            r.get("signal") == "SELL" and not r.get("error")
            for r in results
        ),
        "NO_SIGNAL": sum(
            r.get("signal") == "NO SIGNAL" and not r.get("error")
            for r in results
        ),
    }])

    sheets = {
        "Scan History": summary,
        "Scan Results": result_table,
        "Data Errors": error_table,
    }

    if EXCEL_PATH.exists():
        try:
            with pd.ExcelFile(EXCEL_PATH) as book:
                for name, new_table in list(sheets.items()):
                    if name in book.sheet_names:
                        old_table = pd.read_excel(
                            book, sheet_name=name
                        )
                        sheets[name] = pd.concat(
                            [old_table, new_table],
                            ignore_index=True,
                        ).tail(1_000_000)
        except Exception:
            pass

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            prefix="velora_",
            dir=DATA_DIR,
            delete=False,
        ) as tmp:
            temporary_path = tmp.name

        with pd.ExcelWriter(
            temporary_path, engine="openpyxl"
        ) as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                for cells in ws.columns:
                    width = max(
                        (
                            len(str(cell.value))
                            if cell.value is not None else 0
                        )
                        for cell in cells[:1000]
                    )
                    ws.column_dimensions[
                        cells[0].column_letter
                    ].width = min(55, max(12, width + 2))

        os.replace(temporary_path, EXCEL_PATH)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# ============================================================
# STREAMLIT UI AND NATIVE AUTO-REFRESH
# No streamlit_autorefresh package required.
# ============================================================

st.set_page_config(
    page_title="Velora Scanner",
    page_icon="📊",
    layout="wide",
)

if st.session_state.get("_velora_version") != APP_VERSION:
    for key in (
        "results", "scan_time", "save_msg",
        "last_scan_epoch", "scan_config",
    ):
        st.session_state.pop(key, None)
    st.session_state["_velora_version"] = APP_VERSION

st.title("📊 Velora — 3-candle reversal scanner")

st.caption(
    "BUY: 3 rising candles after a swing low | "
    "SELL: 3 falling candles after a swing high | "
    "Refresh every 40 seconds | No trade execution"
)

st.warning(
    "Important: Binomo OTC quotes are not available from these public APIs. "
    "Crypto rows use public exchange Spot prices as proxies; Forex uses Yahoo "
    "Finance where supported. Payout/VIP payout are transcribed from your list "
    "and may change at the broker. Signals use completed one-minute candles."
)

st.sidebar.header("Scanner settings")

category = st.sidebar.selectbox(
    "Asset category",
    ["All", "Crypto", "Forex", "Other"],
    index=0,
)

lookback = st.sidebar.slider(
    "Swing-high / swing-low lookback (candles)",
    min_value=15,
    max_value=20,
    value=15,
    step=1,
)

deep_history = st.sidebar.checkbox(
    "Build history toward 100,000 candles (slower)",
    value=False,
)

show_no_signal = st.sidebar.checkbox(
    "Show NO SIGNAL results",
    value=True,
)

category_assets = [
    asset for asset in ASSETS
    if category == "All" or asset["category"] == category
]

default_names = [
    "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)",
    "EUR/USD", "AUD/USD", "GBP/USD", "USD/JPY",
    "USD/CAD", "USD/CHF", "Gold",
]

default_assets = [
    asset for asset in category_assets
    if asset["name"] in default_names
]

if not default_assets:
    default_assets = category_assets[:min(10, len(category_assets))]

chosen_names = st.multiselect(
    "Assets to scan (from your list)",
    options=[asset["name"] for asset in category_assets],
    default=[asset["name"] for asset in default_assets],
    help=(
        "Selecting many Yahoo Forex assets can trigger rate limits. "
        "Start with a smaller group. OTC prices may differ from public-market prices."
    ),
)

selected_assets = [
    asset for asset in category_assets
    if asset["name"] in chosen_names
]

target_count = (
    TARGET_HISTORY if deep_history else DEFAULT_HISTORY
)

st.write(f"**Assets selected:** {len(selected_assets)}")
st.write("**Scan refresh:** every 40 seconds")
st.write(f"**Swing lookback:** {lookback} completed one-minute candles")

st.caption(
    f"Excel history: `{EXCEL_PATH}`. The workbook is updated after each scan; "
    "download it to preserve a copy across redeployments."
)

st.markdown(f"""
### Signal rules
- **SELL:** the highest high in the selected last {lookback} completed
  candles occurred before the latest three candles, and those latest three
  candles are bearish with consecutively lower closes.
- **BUY:** the lowest low in the selected last {lookback} completed
  candles occurred before the latest three candles, and those latest three
  candles are bullish with consecutively higher closes.
- Otherwise the result is **NO SIGNAL**.
- EMA 15 and Bollinger Bands are shown for reference only.
- The unfinished current one-minute candle is excluded.
""")

if (
    "results" not in st.session_state
    or not isinstance(st.session_state.get("results"), list)
):
    st.session_state["results"] = []

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None


def render_scanner():
    """Run and display scan results; fragment reruns every 40 seconds."""
    manual_scan = st.button(
        "🔍 SCAN NOW",
        type="primary",
        use_container_width=True,
        key="velora_scan_now",
    )

    last_scan_epoch = float(
        st.session_state.get("last_scan_epoch", 0) or 0
    )

    current_config = (
        tuple(chosen_names),
        int(lookback),
        bool(deep_history),
        int(target_count),
    )

    config_changed = (
        st.session_state.get("scan_config") != current_config
    )

    timer_due = (
        time.time() - last_scan_epoch
    ) >= 39.0

    stored_before = st.session_state.get("results", [])
    no_previous_results = (
        not isinstance(stored_before, list)
        or len(stored_before) == 0
    )

    should_scan = (
        manual_scan
        or no_previous_results
        or config_changed
        or timer_due
    )

    if should_scan and selected_assets:
        scan_results = []
        progress = st.progress(0.0)
        status = st.empty()

        with ThreadPoolExecutor(
            max_workers=min(3, len(selected_assets))
        ) as executor:
            futures = {
                executor.submit(
                    scan_asset,
                    asset,
                    int(lookback),
                    target_count,
                    deep_history,
                ): asset
                for asset in selected_assets
            }

            for index, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                asset = futures[future]

                try:
                    scan_results.append(future.result())
                except Exception as exc:
                    scan_results.append({
                        **asset,
                        "signal": "NO SIGNAL",
                        "reason": f"Unexpected scan error: {exc}",
                        "candles": None,
                        "error": str(exc),
                    })

                progress.progress(
                    index / max(1, len(futures))
                )
                status.write(
                    f"Scanning {index}/{len(futures)} · "
                    "next automatic scan in 40 seconds"
                )

        order = {
            asset["name"]: index
            for index, asset in enumerate(selected_assets)
        }

        scan_results.sort(
            key=lambda result: order.get(
                result.get("name"), 9999
            )
        )

        scan_stamp = now_utc().strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        st.session_state["results"] = scan_results
        st.session_state["scan_time"] = scan_stamp
        st.session_state["last_scan_epoch"] = time.time()
        st.session_state["scan_config"] = current_config

        try:
            save_excel_history(scan_results, scan_stamp)
            st.session_state["save_msg"] = (
                f"Excel saved: {EXCEL_PATH.name}"
            )
        except Exception as exc:
            st.session_state["save_msg"] = (
                f"Excel save failed: {type(exc).__name__}: {exc}"
            )

        progress.empty()
        status.empty()

    elif should_scan and not selected_assets:
        st.info("Select at least one asset to start scanning.")

    stored = st.session_state.get("results", [])
    results = stored if isinstance(stored, list) else []

    if not isinstance(stored, list):
        st.session_state["results"] = []

    if st.session_state.get("save_msg"):
        message = st.session_state["save_msg"]
        if message.startswith("Excel saved:"):
            st.success(message)
        else:
            st.error(message)

    if len(results) == 0:
        st.info(
            "Select assets and wait for the initial scan or press SCAN NOW."
        )

        if EXCEL_PATH.exists():
            st.download_button(
                "📥 Download Excel history",
                EXCEL_PATH.read_bytes(),
                "velora_scan_history.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )
        return

    selected_name_set = set(chosen_names)
    results = [
        result for result in results
        if result.get("name") in selected_name_set
    ]

    valid = [
        result for result in results
        if result.get("candles") is not None
        and not result.get("error")
    ]

    errors = [
        result for result in results
        if result.get("error")
    ]

    buys = [
        result for result in valid
        if result.get("signal") == "BUY"
    ]

    sells = [
        result for result in valid
        if result.get("signal") == "SELL"
    ]

    st.caption(
        f"Last scan: {st.session_state.get('scan_time') or 'Not scanned yet'} · "
        "automatic refresh: 40 seconds"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assets scanned", len(results))
    m2.metric("BUY", len(buys))
    m3.metric("SELL", len(sells))
    m4.metric("Data errors", len(errors))

    st.subheader("🚨 BUY / SELL signals")

    signals = buys + sells

    if len(signals) > 0:
        st.dataframe(
            pd.DataFrame([{
                "Asset": result.get("name"),
                "Payout": f"{result.get('payout')}%",
                "VIP": f"{result.get('vip_payout')}%",
                "Signal": result.get("signal"),
                "Price": result.get("price"),
                "Symbol": result.get("symbol"),
                "Last candle (UTC)": result.get("last_candle_utc"),
                "Reason": result.get("reason"),
            } for result in signals]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No confirmed three-candle reversal right now. "
            "The scanner continues checking every 40 seconds."
        )

    st.subheader("📋 All scan results")

    shown = (
        valid
        if show_no_signal
        else [
            result for result in valid
            if result.get("signal") != "NO SIGNAL"
        ]
    )

    # Do not leave the table blank if there are no signals.
    if len(shown) == 0 and len(valid) > 0:
        shown = valid
        st.info(
            "There are currently no BUY/SELL signals. "
            "Showing valid NO SIGNAL rows so the table does not stay blank."
        )

    if len(shown) > 0:
        st.dataframe(
            pd.DataFrame([{
                "Asset": result.get("name"),
                "Payout": f"{result.get('payout')}%",
                "VIP payout": f"{result.get('vip_payout')}%",
                "Symbol": result.get("symbol"),
                "Category": result.get("category"),
                "Signal": result.get("signal"),
                "Price": result.get("price"),
                "Candles": result.get("count"),
                "3-candle pattern": result.get("three_candle_pattern"),
                "Swing check": result.get("swing_point"),
                "Source": result.get("source"),
                "Reason": result.get("reason"),
            } for result in shown]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No valid market data for the selected assets. "
            "Open Data errors below."
        )

    st.subheader("📈 Asset chart")

    chartable = [
        result for result in valid
        if isinstance(result.get("candles"), pd.DataFrame)
        and not result["candles"].empty
    ]

    if len(chartable) > 0:
        by_name = {
            result["name"]: result
            for result in chartable
        }

        chosen_name = st.selectbox(
            "Choose an asset",
            list(by_name.keys()),
            key="velora_chart_asset",
        )

        chosen = by_name[chosen_name]
        chart = chosen["candles"].tail(300).set_index("time")

        st.write(
            f"**{chosen['signal']}** — {chosen['reason']} · "
            f"{chosen.get('source', '')}"
        )

        if chosen.get("warning"):
            st.warning(chosen["warning"])

        st.markdown("**Close price and EMA 15**")
        st.line_chart(
            chart[["close", "ema15"]].rename(
                columns={"close": "Close", "ema15": "EMA 15"}
            )
        )

        st.markdown("**Bollinger Bands**")
        st.line_chart(
            chart[
                ["close", "bb_upper", "bb_mid", "bb_lower"]
            ].rename(
                columns={
                    "close": "Close",
                    "bb_upper": "Upper band",
                    "bb_mid": "Middle band",
                    "bb_lower": "Lower band",
                }
            )
        )

    else:
        st.info(
            "No chart available because no valid current candle data "
            "was retrieved."
        )

    st.subheader("⚠️ Data errors")

    if len(errors) > 0:
        error_text = "\n\n".join(
            f"ASSET: {result.get('name')} | "
            f"SYMBOL: {result.get('symbol')}\n"
            f"{result.get('error')}"
            for result in errors
        )

        st.text_area(
            "Copy error details",
            error_text,
            height=300,
        )
    else:
        st.success("No data errors in this scan.")

    st.subheader("💾 Automatic Excel history")

    if EXCEL_PATH.exists():
        st.download_button(
            "📥 Download Excel history (all scans)",
            EXCEL_PATH.read_bytes(),
            "velora_scan_history.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

        st.caption(
            "Workbook sheets: Scan History, Scan Results, Data Errors. "
            "Saved after each automatic or manual scan."
        )

    current_export = pd.DataFrame([
        export_row(
            result,
            st.session_state.get("scan_time") or "Unknown",
        )
        for result in results
    ])

    st.download_button(
        "📄 Download current scan CSV",
        current_export.to_csv(index=False).encode("utf-8-sig"),
        "velora_current_scan.csv",
        "text/csv",
        use_container_width=True,
    )

    st.caption(
        "Public-market prices are not the same as Binomo OTC quotes. "
        "Signals are informational and do not place orders. "
        "Refreshing runs a new scan every 40 seconds, but a new signal "
        "may wait until the next one-minute candle is completed."
    )


# Native Streamlit fragment; no streamlit_autorefresh package needed.
if hasattr(st, "fragment"):
    render_scanner = st.fragment(run_every="40s")(render_scanner)

render_scanner()
