from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

TITLE = "Velora — EMA 15 + Bollinger Bands"
TARGET = 100_000
LIMIT = 1_000
TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 VeloraScanner/3.0",
    "Accept": "application/json",
}

BINANCE = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

ASSETS = [
    {"name": "Bitcoin", "symbol": "BTCUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Ethereum", "symbol": "ETHUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Solana", "symbol": "SOLUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "XRP", "symbol": "XRPUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Dogecoin", "symbol": "DOGEUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Cardano", "symbol": "ADAUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Avalanche", "symbol": "AVAXUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "Chainlink", "symbol": "LINKUSDT", "category": "Crypto", "kind": "crypto"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "category": "Forex", "kind": "yahoo"},
    {"name": "GBP/USD", "symbol": "GBPUSD=X", "category": "Forex", "kind": "yahoo"},
    {"name": "USD/JPY", "symbol": "JPY=X", "category": "Forex", "kind": "yahoo"},
    {"name": "AUD/USD", "symbol": "AUDUSD=X", "category": "Forex", "kind": "yahoo"},
    {"name": "USD/CAD", "symbol": "CAD=X", "category": "Forex", "kind": "yahoo"},
    {"name": "USD/CHF", "symbol": "CHF=X", "category": "Forex", "kind": "yahoo"},
    {"name": "NZD/USD", "symbol": "NZDUSD=X", "category": "Forex", "kind": "yahoo"},
    {"name": "EUR/GBP", "symbol": "EURGBP=X", "category": "Forex", "kind": "yahoo"},
    {"name": "Gold futures", "symbol": "GC=F", "category": "Other", "kind": "yahoo"},
    {"name": "Apple", "symbol": "AAPL", "category": "Other", "kind": "yahoo"},
    {"name": "Microsoft", "symbol": "MSFT", "category": "Other", "kind": "yahoo"},
    {"name": "NVIDIA", "symbol": "NVDA", "category": "Other", "kind": "yahoo"},
]

COINBASE = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
    "ADAUSDT": "ADA-USD",
    "AVAXUSDT": "AVAX-USD",
    "LINKUSDT": "LINK-USD",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_utc():
    return pd.Timestamp.now(tz="UTC")


def minute_now():
    return now_utc().floor("min")


def get(url, params=None):
    return requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT,
    )


def data_dir():
    requested = os.environ.get("VELORA_DATA_DIR")

    candidates = (
        [Path(requested)]
        if requested
        else [Path.cwd(), Path("/tmp/velora_data")]
    )

    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            test_file = folder / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return folder
        except Exception:
            pass

    folder = Path(tempfile.gettempdir()) / "velora_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


DATA_DIR = data_dir()
DB = DATA_DIR / "velora_history.sqlite3"
XLSX = DATA_DIR / "velora_scan_history.xlsx"


# ============================================================
# SQLITE DATABASE AND OLD DATABASE MIGRATION
# ============================================================

def init_db():
    """Create the current schema and migrate older database schemas."""
    with sqlite3.connect(DB, timeout=30) as connection:
        exists = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='candles'"
        ).fetchone() is not None

        required = {
            "symbol", "time", "open", "high", "low", "close", "volume"
        }

        columns = (
            {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(candles)"
                ).fetchall()
            }
            if exists
            else set()
        )

        if exists and not required.issubset(columns):
            connection.execute("DROP INDEX IF EXISTS idx_symbol_time")
            connection.execute("ALTER TABLE candles RENAME TO candles_legacy")

            old_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(candles_legacy)"
                ).fetchall()
            }

            connection.execute(
                """
                CREATE TABLE candles (
                    symbol TEXT NOT NULL,
                    time TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    PRIMARY KEY(symbol, time)
                )
                """
            )

            time_column = next(
                (
                    name
                    for name in ("time", "candle_time", "timestamp")
                    if name in old_columns
                ),
                None,
            )

            needed = {"symbol", "open", "high", "low", "close"}

            if time_column and needed.issubset(old_columns):
                volume_expr = (
                    "volume" if "volume" in old_columns else "0"
                )

                connection.execute(
                    "INSERT OR REPLACE INTO candles "
                    "(symbol,time,open,high,low,close,volume) "
                    f"SELECT symbol,{time_column},open,high,low,close,"
                    f"{volume_expr} FROM candles_legacy "
                    f"WHERE symbol IS NOT NULL "
                    f"AND {time_column} IS NOT NULL"
                )

            connection.execute("DROP TABLE candles_legacy")

        elif not exists:
            connection.execute(
                """
                CREATE TABLE candles (
                    symbol TEXT NOT NULL,
                    time TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    PRIMARY KEY(symbol, time)
                )
                """
            )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_time "
            "ON candles(symbol,time)"
        )
        connection.commit()


def normalize(df):
    columns = ["time", "open", "high", "low", "close", "volume"]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    data = df.copy()

    for column in columns:
        if column not in data.columns:
            data[column] = (
                0.0 if column == "volume" else np.nan
            )

    data["time"] = pd.to_datetime(
        data["time"], utc=True, errors="coerce"
    )

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["volume"] = data["volume"].fillna(0)

    data = data.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["time"] < minute_now())
    ]

    return (
        data.drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)[columns]
    )


def save_cache(symbol, df):
    data = normalize(df)

    if data.empty:
        return

    rows = []

    for row in data.itertuples(index=False):
        timestamp = (
            pd.Timestamp(row.time)
            .tz_convert("UTC")
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        rows.append(
            (
                symbol,
                timestamp,
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume or 0),
            )
        )

    init_db()

    with sqlite3.connect(DB, timeout=30) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO candles "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        connection.commit()


def load_cache(symbol):
    init_db()

    with sqlite3.connect(DB, timeout=30) as connection:
        data = pd.read_sql_query(
            """
            SELECT time,open,high,low,close,volume
            FROM candles
            WHERE symbol=?
            ORDER BY time
            """,
            connection,
            params=(symbol,),
        )

    return normalize(data)


# ============================================================
# BINANCE PUBLIC CANDLES
# ============================================================

def parse_binance(rows):
    columns = [
        "stamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ]

    data = pd.DataFrame(rows, columns=columns)

    data["time"] = pd.to_datetime(
        data["stamp"], unit="ms", utc=True, errors="coerce"
    )

    return normalize(
        data[["time", "open", "high", "low", "close", "volume"]]
    )


def fetch_binance(symbol, target):
    errors = []

    for base in BINANCE:
        try:
            pages = []
            end_time = None

            while True:
                current_count = (
                    len(pd.concat(pages, ignore_index=True))
                    if pages
                    else 0
                )

                limit = min(1000, max(1, target - current_count))

                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                }

                if end_time is not None:
                    params["endTime"] = end_time

                response = get(base + "/api/v3/klines", params)

                if response.status_code in (403, 451):
                    raise RuntimeError(
                        f"HTTP {response.status_code}: endpoint blocked"
                    )

                if response.status_code == 429:
                    raise RuntimeError("HTTP 429: rate limited")

                response.raise_for_status()
                rows = response.json()

                if not isinstance(rows, list) or not rows:
                    raise RuntimeError(
                        f"No candles returned: {str(rows)[:150]}"
                    )

                pages.append(parse_binance(rows))

                combined = normalize(
                    pd.concat(pages, ignore_index=True)
                )

                if len(combined) >= target or target <= 1000:
                    break

                oldest = combined["time"].min()
                next_end = int(oldest.timestamp() * 1000) - 1

                if end_time is not None and next_end >= end_time:
                    break

                end_time = next_end
                time.sleep(0.05)

                if len(pages) >= 110:
                    break

            combined = (
                normalize(pd.concat(pages, ignore_index=True))
                .tail(target)
                .reset_index(drop=True)
            )

            return combined, base, None

        except Exception as error:
            errors.append(
                f"{base}: {type(error).__name__}: {error}"
            )

    raise RuntimeError("Binance failed: " + " | ".join(errors))


# ============================================================
# BYBIT PUBLIC CANDLES
# ============================================================

def fetch_bybit(symbol, target):
    if not symbol.endswith("USDT"):
        raise RuntimeError("Unsupported Bybit symbol")

    pages = []
    end_time = None
    warning = None

    while True:
        current_count = (
            len(pd.concat(pages, ignore_index=True))
            if pages
            else 0
        )

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": "1",
            "limit": min(1000, max(40, target - current_count)),
        }

        if end_time is not None:
            params["end"] = end_time

        response = get(
            "https://api.bybit.com/v5/market/kline",
            params,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit {payload.get('retCode')}: "
                f"{payload.get('retMsg')}"
            )

        rows = (payload.get("result") or {}).get("list") or []

        if not rows:
            break

        data = pd.DataFrame(
            rows,
            columns=[
                "stamp", "open", "high", "low",
                "close", "volume", "turnover",
            ],
        )

        data["time"] = pd.to_datetime(
            pd.to_numeric(data["stamp"], errors="coerce"),
            unit="ms",
            utc=True,
        )

        pages.append(
            normalize(
                data[
                    ["time", "open", "high", "low", "close", "volume"]
                ]
            )
        )

        combined = normalize(
            pd.concat(pages, ignore_index=True)
        )

        if len(combined) >= target or target <= 1000:
            break

        next_end = (
            int(combined["time"].min().timestamp() * 1000) - 1
        )

        if end_time is not None and next_end >= end_time:
            warning = "Bybit historical paging stopped."
            break

        end_time = next_end
        time.sleep(0.05)

        if len(pages) >= 110:
            warning = "Bybit sync limited to 110 pages."
            break

    if not pages:
        raise RuntimeError("Bybit returned no candles")

    return (
        normalize(pd.concat(pages, ignore_index=True))
        .tail(target)
        .reset_index(drop=True),
        "Bybit public API",
        warning,
    )


# ============================================================
# OKX PUBLIC CANDLES
# ============================================================

def fetch_okx(symbol, target):
    instrument = symbol[:-4] + "-USDT"
    pages = []
    after = None
    warning = None

    while True:
        current_count = (
            len(pd.concat(pages, ignore_index=True))
            if pages
            else 0
        )

        params = {
            "instId": instrument,
            "bar": "1m",
            "limit": str(min(300, max(40, target - current_count))),
        }

        if after is not None:
            params["after"] = after

        response = get(
            "https://www.okx.com/api/v5/market/candles",
            params,
        )
        response.raise_for_status()
        payload = response.json()

        if str(payload.get("code")) != "0":
            raise RuntimeError(
                f"OKX {payload.get('code')}: {payload.get('msg')}"
            )

        rows = payload.get("data") or []

        if not rows:
            break

        data = pd.DataFrame(
            rows,
            columns=[
                "stamp", "open", "high", "low", "close",
                "volume", "volume_currency", "volume_quote", "confirm",
            ],
        )

        data["time"] = pd.to_datetime(
            pd.to_numeric(data["stamp"], errors="coerce"),
            unit="ms",
            utc=True,
        )

        pages.append(
            normalize(
                data[
                    ["time", "open", "high", "low", "close", "volume"]
                ]
            )
        )

        combined = normalize(
            pd.concat(pages, ignore_index=True)
        )

        if len(combined) >= target or target <= 300:
            break

        next_after = str(
            int(combined["time"].min().timestamp() * 1000)
        )

        if after == next_after:
            warning = "OKX historical paging stopped."
            break

        after = next_after
        time.sleep(0.05)

        if len(pages) >= 340:
            warning = "OKX sync limited to 340 pages."
            break

    if not pages:
        raise RuntimeError("OKX returned no candles")

    return (
        normalize(pd.concat(pages, ignore_index=True))
        .tail(target)
        .reset_index(drop=True),
        "OKX public API",
        warning,
    )


# ============================================================
# GATE.IO AND COINBASE PUBLIC CANDLES
# ============================================================

def fetch_gate(symbol, target):
    pair = symbol[:-4] + "_USDT"

    response = get(
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

    data = pd.DataFrame(
        rows,
        columns=[
            "stamp", "volume", "close", "high",
            "low", "open", "quote_volume",
        ],
    )

    data["time"] = pd.to_datetime(
        pd.to_numeric(data["stamp"], errors="coerce"),
        unit="s",
        utc=True,
    )

    return (
        normalize(
            data[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Gate.io public API",
        None,
    )


def fetch_coinbase(symbol):
    product = COINBASE.get(symbol)

    if not product:
        raise RuntimeError("No Coinbase mapping for symbol")

    end = now_utc().floor("min")
    start = end - pd.Timedelta(minutes=299)

    params = {
        "granularity": 60,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    }

    response = get(
        f"https://api.exchange.coinbase.com/products/{product}/candles",
        params,
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise RuntimeError("Coinbase returned no candles")

    data = pd.DataFrame(
        rows,
        columns=["stamp", "low", "high", "open", "close", "volume"],
    )

    data["time"] = pd.to_datetime(
        data["stamp"], unit="s", utc=True
    )

    return (
        normalize(
            data[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Coinbase public API",
        None,
    )


# ============================================================
# YAHOO FINANCE
# ============================================================

def fetch_yahoo(asset):
    symbol = asset["symbol"]

    if asset["kind"] == "crypto":
        symbol = COINBASE.get(symbol, symbol)

    errors = []

    for host in [
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    ]:
        try:
            encoded = requests.utils.quote(symbol, safe="")
            url = f"https://{host}/v8/finance/chart/{encoded}"

            response = get(
                url,
                {
                    "range": "7d",
                    "interval": "1m",
                    "includePrePost": "false",
                },
            )
            response.raise_for_status()

            chart = response.json().get("chart", {})

            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))

            items = chart.get("result") or []

            if not items:
                raise RuntimeError("Yahoo returned no result")

            item = items[0]
            quotes = (
                item.get("indicators", {}).get("quote") or [{}]
            )
            quote = quotes[0]
            timestamps = item.get("timestamp") or []

            if not timestamps:
                raise RuntimeError("Yahoo returned no timestamps")

            data = pd.DataFrame({
                "time": pd.to_datetime(
                    timestamps, unit="s", utc=True
                ),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })

            return normalize(data), f"Yahoo Finance ({host})", None

        except Exception as error:
            errors.append(
                f"{host}: {type(error).__name__}: {error}"
            )

    raise RuntimeError("Yahoo failed: " + " | ".join(errors))


# ============================================================
# LOAD LIVE DATA AND CACHE
# ============================================================

def get_candles(asset, target):
    symbol = asset["symbol"]
    errors = []
    warnings = []
    recent = pd.DataFrame()
    source = ""

    if asset["kind"] == "crypto":
        providers = [
            ("Binance", lambda: fetch_binance(symbol, target)),
            ("Bybit", lambda: fetch_bybit(symbol, target)),
            ("OKX", lambda: fetch_okx(symbol, target)),
            ("Gate.io", lambda: fetch_gate(symbol, target)),
            ("Coinbase", lambda: fetch_coinbase(symbol)),
            ("Yahoo", lambda: fetch_yahoo(asset)),
        ]

        for name, fetcher in providers:
            try:
                recent, source, note = fetcher()

                if note:
                    warnings.append(str(note))

                if name != "Binance":
                    warnings.append(
                        f"Fallback provider {name} used; "
                        "prices may differ from Binance USDT Spot."
                    )

                break

            except Exception as error:
                errors.append(
                    f"{name}: {type(error).__name__}: {error}"
                )

    else:
        try:
            recent, source, note = fetch_yahoo(asset)

            if note:
                warnings.append(str(note))

        except Exception as error:
            errors.append(
                f"Yahoo: {type(error).__name__}: {error}"
            )

    recent = normalize(recent)

    if not recent.empty:
        save_cache(symbol, recent)

    try:
        cached = load_cache(symbol)
    except Exception as error:
        cached = pd.DataFrame()
        errors.append(f"SQLite: {error}")

    frames = [
        frame
        for frame in [cached, recent]
        if frame is not None and not frame.empty
    ]

    if not frames:
        raise RuntimeError(
            "No live or cached candles. "
            + (
                " | ".join(errors)
                if errors
                else "All providers returned empty data."
            )
        )

    data = (
        normalize(pd.concat(frames, ignore_index=True))
        .tail(target)
        .reset_index(drop=True)
    )

    if len(data) < 40:
        raise RuntimeError(
            f"Only {len(data)} completed candles; need at least 40. "
            + " | ".join(errors)
        )

    age_minutes = (
        now_utc() - data["time"].iloc[-1]
    ).total_seconds() / 60

    if age_minutes > 5:
        raise RuntimeError(
            f"Latest candle is {age_minutes:.1f} minutes old; "
            "no signal calculated. "
            + " | ".join(errors)
        )

    warning_text = (
        " | ".join(warnings + errors)
        if warnings or errors
        else None
    )

    return data, source, warning_text


# ============================================================
# EMA 15 AND BOLLINGER BANDS — DISPLAY ONLY FOR SIGNAL PURPOSES
# ============================================================

def indicators(df):
    data = df.copy()

    data["ema15"] = data["close"].ewm(
        span=15,
        adjust=False,
        min_periods=15,
    ).mean()

    data["bb_mid"] = data["close"].rolling(
        20, min_periods=20
    ).mean()

    deviation = data["close"].rolling(
        20, min_periods=20
    ).std(ddof=0)

    data["bb_upper"] = data["bb_mid"] + 2 * deviation
    data["bb_lower"] = data["bb_mid"] - 2 * deviation

    return data


# ============================================================
# SIGNAL RULE — USER REQUEST
#
# Falling across selected completed candles = BUY
# Rising across selected completed candles = SELL
# EMA and Bollinger do not block signals.
# ============================================================

def signal_for(df, lookback, mode=None):
    data = (
        df.copy()
        if {"ema15", "bb_mid", "bb_upper", "bb_lower"}.issubset(df.columns)
        else indicators(df)
    )

    empty = {
        "signal": "NO SIGNAL",
        "reason": f"Not enough completed candles; need {lookback}.",
        "change_pct": np.nan,
        "trend": "Unknown",
        "green_ratio": np.nan,
        "red_ratio": np.nan,
        "direction_up_ok": False,
        "direction_down_ok": False,
        "price_above_ema": False,
        "price_below_ema": False,
        "upper_band_ok": False,
        "lower_band_ok": False,
    }

    if len(data) < int(lookback):
        return empty

    segment = data.tail(int(lookback))
    first_close = float(segment["close"].iloc[0])
    last = data.iloc[-1]
    last_close = float(last["close"])

    if first_close <= 0:
        return {
            **empty,
            "reason": "Invalid starting close price.",
        }

    change = (last_close - first_close) / first_close * 100.0

    green = float(
        (segment["close"] > segment["open"]).mean()
    )
    red = float(
        (segment["close"] < segment["open"]).mean()
    )

    above = bool(
        pd.notna(last.get("ema15"))
        and last_close > float(last["ema15"])
    )
    below = bool(
        pd.notna(last.get("ema15"))
        and last_close < float(last["ema15"])
    )
    upper = bool(
        pd.notna(last.get("bb_upper"))
        and float(last["high"]) >= float(last["bb_upper"])
    )
    lower = bool(
        pd.notna(last.get("bb_lower"))
        and float(last["low"]) <= float(last["bb_lower"])
    )

    # Your rule: price falls -> BUY.
    if last_close < first_close:
        signal = "BUY"
        trend = "Falling"
        reason = (
            f"BUY: price fell {abs(change):.3f}% across the last "
            f"{lookback} completed 1-minute candles. "
            "EMA 15 and Bollinger Bands do not block this signal."
        )

    # Your rule: price rises -> SELL.
    elif last_close > first_close:
        signal = "SELL"
        trend = "Rising"
        reason = (
            f"SELL: price rose {change:.3f}% across the last "
            f"{lookback} completed 1-minute candles. "
            "EMA 15 and Bollinger Bands do not block this signal."
        )

    else:
        signal = "NO SIGNAL"
        trend = "Flat"
        reason = (
            f"First and last closing prices are equal across "
            f"the selected {lookback} candles."
        )

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": change,
        "trend": trend,
        "green_ratio": green,
        "red_ratio": red,
        "direction_up_ok": last_close > first_close,
        "direction_down_ok": last_close < first_close,
        "price_above_ema": above,
        "price_below_ema": below,
        "upper_band_ok": upper,
        "lower_band_ok": lower,
    }


# ============================================================
# SCAN ONE ASSET
# ============================================================

def scan(asset, lookback, mode, target):
    try:
        candles, source, warning = get_candles(asset, target)
        data = indicators(candles)
        signal = signal_for(data, lookback, mode)
        last = data.iloc[-1]

        return {
            **asset,
            **signal,
            "price": float(last["close"]),
            "ema15": (
                float(last["ema15"])
                if pd.notna(last["ema15"])
                else np.nan
            ),
            "bb_upper": (
                float(last["bb_upper"])
                if pd.notna(last["bb_upper"])
                else np.nan
            ),
            "bb_mid": (
                float(last["bb_mid"])
                if pd.notna(last["bb_mid"])
                else np.nan
            ),
            "bb_lower": (
                float(last["bb_lower"])
                if pd.notna(last["bb_lower"])
                else np.nan
            ),
            "count": len(data),
            "last_candle_utc": str(last["time"]),
            "source": source,
            "candles": data,
            "error": None,
            "warning": warning,
        }

    except Exception as error:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "reason": "Data unavailable; no signal calculated.",
            "change_pct": np.nan,
            "trend": "Unknown",
            "green_ratio": np.nan,
            "red_ratio": np.nan,
            "direction_up_ok": False,
            "direction_down_ok": False,
            "price_above_ema": False,
            "price_below_ema": False,
            "upper_band_ok": False,
            "lower_band_ok": False,
            "price": np.nan,
            "ema15": np.nan,
            "bb_upper": np.nan,
            "bb_mid": np.nan,
            "bb_lower": np.nan,
            "count": 0,
            "last_candle_utc": None,
            "source": "Unavailable",
            "candles": None,
            "error": str(error),
            "warning": None,
        }


# ============================================================
# EXCEL HISTORY — AUTOMATIC SAVE
# ============================================================

def excel_row(result, scan_time):
    fields = [
        "name", "symbol", "category", "kind", "signal",
        "reason", "trend", "change_pct", "green_ratio", "red_ratio",
        "price", "ema15", "bb_upper", "bb_mid", "bb_lower",
        "count", "last_candle_utc", "source", "warning", "error",
    ]

    output = {
        "scan_time_utc": scan_time,
        **{field: result.get(field) for field in fields},
    }

    for key, value in output.items():
        if (
            isinstance(value, (np.floating, float))
            and not np.isfinite(value)
        ):
            output[key] = None
        elif isinstance(value, np.integer):
            output[key] = int(value)

    return output


def save_excel(results, scan_time):
    result_table = pd.DataFrame([
        excel_row(result, scan_time)
        for result in results
    ])

    error_table = result_table[
        result_table["error"].notna()
    ].copy()

    summary_table = pd.DataFrame([{
        "scan_time_utc": scan_time,
        "assets_scanned": len(results),
        "data_ok": sum(
            not bool(result.get("error"))
            for result in results
        ),
        "data_errors": sum(
            bool(result.get("error"))
            for result in results
        ),
        "BUY": sum(
            result.get("signal") == "BUY"
            and not bool(result.get("error"))
            for result in results
        ),
        "SELL": sum(
            result.get("signal") == "SELL"
            and not bool(result.get("error"))
            for result in results
        ),
        "NO_SIGNAL": sum(
            result.get("signal") == "NO SIGNAL"
            and not bool(result.get("error"))
            for result in results
        ),
    }])

    sheets = {
        "Scan History": summary_table,
        "Scan Results": result_table,
        "Data Errors": error_table,
    }

    # Append new results without deleting earlier scans.
    if XLSX.exists():
        try:
            with pd.ExcelFile(XLSX) as workbook:
                for sheet_name, new_table in list(sheets.items()):
                    if sheet_name in workbook.sheet_names:
                        old_table = pd.read_excel(
                            workbook,
                            sheet_name=sheet_name,
                        )
                        sheets[sheet_name] = pd.concat(
                            [old_table, new_table],
                            ignore_index=True,
                        ).tail(1_000_000)
        except Exception:
            # If the existing workbook cannot be read, create a new one.
            pass

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            prefix="velora_",
            dir=DATA_DIR,
            delete=False,
        ) as file:
            temporary_path = file.name

        with pd.ExcelWriter(
            temporary_path,
            engine="openpyxl",
        ) as writer:
            for sheet_name, table in sheets.items():
                table.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions

                for cells in worksheet.columns:
                    width = max(
                        (
                            len(str(cell.value))
                            if cell.value is not None
                            else 0
                        )
                        for cell in cells[:1000]
                    )

                    worksheet.column_dimensions[
                        cells[0].column_letter
                    ].width = min(55, max(12, width + 2))

        os.replace(temporary_path, XLSX)

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="Velora Scanner",
    page_icon="📊",
    layout="wide",
)

APP_VERSION = "direction-only-15-20-v1"

# Clear old session data after this app version changes.
if st.session_state.get("_velora_app_version") != APP_VERSION:
    for key in ("results", "scan_time", "save_msg"):
        st.session_state.pop(key, None)

    st.session_state["_velora_app_version"] = APP_VERSION


st.title("📊 Velora — EMA 15 + Bollinger Bands")

st.caption(
    "1-minute market scanner | BUY / SELL signals only | "
    "No order execution"
)

st.warning(
    "Public market data only. Crypto tries Binance, Bybit, OKX, "
    "Gate.io, Coinbase and Yahoo Finance. Forex and stock assets use "
    "Yahoo Finance. Data availability varies. Yahoo prices are not "
    "Binomo OTC quotes. This app never places trades."
)

st.sidebar.header("Scanner settings")

category = st.sidebar.selectbox(
    "Asset category",
    ["All", "Crypto", "Forex", "Other"],
)

lookback = st.sidebar.select_slider(
    "Rise/fall candle count",
    options=list(range(15, 21)),
    value=15,
)

mode = st.sidebar.radio(
    "Bollinger display mode (informational only)",
    ["Near band (0.50%)", "Touch or cross", "Close outside band"],
    index=0,
    help=(
        "The Bollinger setting only changes diagnostic display. "
        "BUY/SELL signals are determined by the overall move "
        "across the selected completed candles."
    ),
)

asset_limit = int(
    st.sidebar.number_input(
        "Maximum assets per scan",
        min_value=1,
        max_value=len(ASSETS),
        value=10,
    )
)

deep = st.sidebar.checkbox(
    "Sync toward 100,000 candles (slower; crypto providers only)",
    value=False,
)

show_no = st.sidebar.checkbox(
    "Show NO SIGNAL results",
    value=True,
)

assets = [
    asset
    for asset in ASSETS
    if category == "All" or asset["category"] == category
][:asset_limit]

target = TARGET if deep else LIMIT

st.markdown(f"""
### Strategy rules

- **BUY:** The latest closing price is lower than the closing price
  at the start of the selected {lookback} completed 1-minute candles.
- **SELL:** The latest closing price is higher than the closing price
  at the start of the selected {lookback} completed 1-minute candles.
- **NO SIGNAL:** The first and last closing prices are equal, or data is unavailable.
- EMA 15 and Bollinger Bands remain on the chart for reference, but
  they do not block BUY/SELL signals.
- The unfinished current 1-minute candle is excluded.
""")

st.write(
    f"Assets selected: **{len(assets)}** · "
    f"History target: **{target:,} candles per asset**"
)

st.caption(
    f"Excel is automatically saved to `{XLSX}` after each scan. "
    "Download a copy to preserve it because hosted files may be deleted "
    "during redeployment."
)


# ============================================================
# SESSION STATE VALIDATION
# ============================================================

# Do not use `if results` on a possible pandas DataFrame.
if (
    "results" not in st.session_state
    or not isinstance(st.session_state.get("results"), list)
):
    st.session_state["results"] = []

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None


# ============================================================
# START SCAN
# ============================================================

if st.button(
    "🔍 START SCAN",
    type="primary",
    use_container_width=True,
):
    scan_results = []
    progress = st.progress(0.0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=min(3, max(1, len(assets)))
    ) as executor:
        futures = {
            executor.submit(
                scan,
                asset,
                int(lookback),
                mode,
                target,
            ): asset
            for asset in assets
        }

        for index, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            asset = futures[future]

            try:
                scan_results.append(future.result())

            except Exception as error:
                scan_results.append({
                    **asset,
                    "signal": "NO SIGNAL",
                    "reason": "Unexpected scan error.",
                    "candles": None,
                    "error": str(error),
                })

            progress.progress(
                index / max(1, len(futures))
            )
            status.write(
                f"Scanned {index}/{len(futures)} assets"
            )

    order = {
        asset["symbol"]: index
        for index, asset in enumerate(assets)
    }

    scan_results.sort(
        key=lambda result: order.get(
            result.get("symbol"), 999
        )
    )

    scan_time = now_utc().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    st.session_state["results"] = scan_results
    st.session_state["scan_time"] = scan_time

    try:
        save_excel(scan_results, scan_time)
        st.session_state["save_msg"] = (
            f"Excel history saved automatically: {XLSX.name}"
        )
    except Exception as error:
        st.session_state["save_msg"] = (
            f"Excel save failed: {error}"
        )

    progress.empty()
    status.empty()


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_value = st.session_state.get("results", [])

if isinstance(results_value, list):
    results = results_value
else:
    results = []
    st.session_state["results"] = []

scan_time = st.session_state.get("scan_time")

if len(results) == 0:
    st.info(
        "Click START SCAN to retrieve candles and calculate signals."
    )

    if XLSX.exists():
        st.download_button(
            "📥 Download Excel history",
            XLSX.read_bytes(),
            "velora_scan_history.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    st.stop()


if st.session_state.get("save_msg"):
    message = st.session_state["save_msg"]

    if message.startswith("Excel history saved"):
        st.success(message)
    else:
        st.error(message)


valid = [
    result
    for result in results
    if result.get("candles") is not None
    and not result.get("error")
]

errors = [
    result
    for result in results
    if result.get("error")
]

buys = [
    result
    for result in valid
    if result.get("signal") == "BUY"
]

sells = [
    result
    for result in valid
    if result.get("signal") == "SELL"
]

st.caption(f"Last scan: {scan_time}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Assets with 100K candles",
    sum(result["count"] >= TARGET for result in valid),
)
col2.metric("BUY", len(buys))
col3.metric("SELL", len(sells))
col4.metric("Data errors", len(errors))


# ============================================================
# BUY / SELL SIGNALS
# ============================================================

st.subheader("🚨 BUY / SELL signals")

signals = buys + sells

if len(signals) > 0:
    signal_table = pd.DataFrame([
        {
            "Asset": result.get("name"),
            "Symbol": result.get("symbol"),
            "Signal": result.get("signal"),
            "Price": result.get("price"),
            "Trend": result.get("trend"),
            "Change (%)": result.get("change_pct"),
            "EMA 15": result.get("ema15"),
            "BB upper": result.get("bb_upper"),
            "BB lower": result.get("bb_lower"),
            "Candles": result.get("count"),
            "Reason": result.get("reason"),
        }
        for result in signals
    ])

    st.table(signal_table)

else:
    st.info(
        "No direction signal: the first and last closes are equal, "
        "or valid current market data is unavailable."
    )


# ============================================================
# ALL RESULTS
# ============================================================

st.subheader("📋 All scan results")

shown = (
    valid
    if show_no
    else [
        result
        for result in valid
        if result.get("signal") != "NO SIGNAL"
    ]
)


def direction_status(result):
    if result.get("direction_up_ok"):
        return "PASS — price rose across selected candles"

    if result.get("direction_down_ok"):
        return "PASS — price fell across selected candles"

    return "NO MOVE — first and last closes are equal"


def ema_status(result):
    if result.get("price_above_ema"):
        return "INFO — close above EMA 15 (not a filter)"

    if result.get("price_below_ema"):
        return "INFO — close below EMA 15 (not a filter)"

    return "INFO — EMA 15 unavailable/equal (not a filter)"


def band_status(result):
    if result.get("upper_band_ok"):
        return "INFO — upper band touched (not a filter)"

    if result.get("lower_band_ok"):
        return "INFO — lower band touched (not a filter)"

    return "INFO — no band touched (not a filter)"


if len(shown) > 0:
    all_table = pd.DataFrame([
        {
            "Asset": result.get("name"),
            "Symbol": result.get("symbol"),
            "Category": result.get("category"),
            "Signal": result.get("signal"),
            "Price": result.get("price"),
            "Candles": result.get("count"),
            "Direction check": direction_status(result),
            "EMA check": ema_status(result),
            "Band check": band_status(result),
            "Source": result.get("source"),
            "Reason": result.get("reason"),
        }
        for result in shown
    ])

    st.table(all_table)

else:
    st.info(
        "No valid scan results. Review Data errors below."
    )


# ============================================================
# CHART
# ============================================================

st.subheader("📈 Asset chart")

chartable = [
    result
    for result in valid
    if isinstance(result.get("candles"), pd.DataFrame)
    and not result["candles"].empty
]

if len(chartable) > 0:
    chart_by_symbol = {
        result["symbol"]: result
        for result in chartable
    }

    selected_symbol = st.selectbox(
        "Choose an asset",
        options=list(chart_by_symbol.keys()),
        format_func=lambda symbol: chart_by_symbol[symbol]["name"],
        key="velora_chart_symbol",
    )

    chosen = chart_by_symbol[selected_symbol]
    chart = chosen["candles"].tail(500).set_index("time")

    st.write(
        f"**{chosen['signal']}** — {chosen['reason']} "
        f"· {chosen['source']}"
    )

    if chosen.get("warning"):
        st.warning(chosen["warning"])

    st.markdown("**Close price and EMA 15**")

    st.line_chart(
        chart[["close", "ema15"]].rename(
            columns={
                "close": "Close",
                "ema15": "EMA 15",
            }
        )
    )

    st.markdown("**Bollinger Bands**")

    st.line_chart(
        chart[["close", "bb_upper", "bb_mid", "bb_lower"]].rename(
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


# ============================================================
# DATA ERRORS
# ============================================================

st.subheader("⚠️ Data errors — readable details")

if len(errors) > 0:
    st.error(
        f"{len(errors)} asset(s) could not be scanned. "
        "Copy the messages below to diagnose them."
    )

    error_lines = []

    for result in errors:
        detail = str(
            result.get("error") or "No error text returned"
        )

        error_lines.append(
            f"ASSET: {result.get('name')} | "
            f"SYMBOL: {result.get('symbol')} | "
            f"CATEGORY: {result.get('category')}\n{detail}"
        )

    st.text_area(
        "Copy provider error messages",
        value="\n\n".join(error_lines),
        height=320,
    )

    for result in errors:
        with st.expander(
            f"{result.get('name')} ({result.get('symbol')})"
        ):
            st.code(
                str(result.get("error") or "No error text returned"),
                language="text",
            )

    st.caption(
        "If all providers fail with DNS, timeout, or connection errors, "
        "the hosting environment may be blocking external APIs."
    )

else:
    st.success("No data errors in this scan.")


# ============================================================
# EXCEL AND CSV DOWNLOADS
# ============================================================

st.subheader("💾 Automatic Excel history")

if XLSX.exists():
    st.download_button(
        "📥 Download Excel history (all scans)",
        XLSX.read_bytes(),
        "velora_scan_history.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    st.caption(
        "Workbook sheets: Scan History, Scan Results, Data Errors."
    )

current_export = pd.DataFrame([
    excel_row(result, scan_time or "Unknown")
    for result in results
])

st.download_button(
    "📄 Download current scan as CSV",
    current_export.to_csv(index=False).encode("utf-8-sig"),
    "velora_current_scan.csv",
    "text/csv",
    use_container_width=True,
)

st.divider()

st.caption(
    "Analysis signals only; no trade execution. "
    "100,000 candles are a target, not a guarantee. "
    "Download the Excel workbook to preserve a copy because hosted "
    "files may be deleted during redeployment."
)
