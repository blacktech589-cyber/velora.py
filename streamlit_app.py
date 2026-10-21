from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VELORA — EMA 15 + BOLLINGER BANDS
# BUY / SELL scanner; no automatic order execution.
# ============================================================

TARGET = 100_000
RECENT_LIMIT = 1_000
TIMEOUT = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 VeloraScanner/3.0",
    "Accept": "application/json",
}

BINANCE_ENDPOINTS = [
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

COINBASE_SYMBOLS = {
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


def current_minute():
    return now_utc().floor("min")


def http_get(url, params=None):
    return requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT,
    )


def get_data_directory():
    requested = os.environ.get("VELORA_DATA_DIR")

    if requested:
        candidates = [Path(requested)]
    else:
        candidates = [
            Path.cwd(),
            Path("/tmp/velora_data"),
        ]

    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            test_file = folder / ".velora_write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return folder
        except Exception:
            continue

    folder = Path(tempfile.gettempdir()) / "velora_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


DATA_DIR = get_data_directory()
DB_PATH = DATA_DIR / "velora_history.sqlite3"
EXCEL_PATH = DATA_DIR / "velora_scan_history.xlsx"


# ============================================================
# SQLITE CANDLE CACHE
# ============================================================

def init_database():
    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.execute(
            """
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
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbol_time
            ON candles(symbol, candle_time)
            """
        )
        connection.commit()


def normalize_candles(df):
    columns = [
        "time", "open", "high", "low", "close", "volume"
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    data = df.copy()

    for column in columns:
        if column not in data.columns:
            data[column] = 0.0 if column == "volume" else np.nan

    data["time"] = pd.to_datetime(
        data["time"],
        utc=True,
        errors="coerce",
    )

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
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
    ]

    # Do not use the unfinished current one-minute candle.
    data = data[data["time"] < current_minute()]

    data = (
        data.drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    return data[columns]


def save_candle_cache(symbol, df):
    data = normalize_candles(df)

    if data.empty:
        return

    init_database()
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
                float(row.volume),
            )
        )

    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO candles
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def load_candle_cache(symbol):
    init_database()

    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        data = pd.read_sql_query(
            """
            SELECT
                candle_time AS time,
                open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY candle_time
            """,
            connection,
            params=(symbol,),
        )

    return normalize_candles(data)


# ============================================================
# BINANCE PUBLIC CANDLE DATA
# ============================================================

def parse_binance_candles(rows):
    columns = [
        "stamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ]

    data = pd.DataFrame(rows, columns=columns)

    data["time"] = pd.to_datetime(
        data["stamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    return normalize_candles(
        data[["time", "open", "high", "low", "close", "volume"]]
    )


def fetch_binance(symbol, target):
    errors = []

    for endpoint in BINANCE_ENDPOINTS:
        try:
            pages = []
            end_time = None

            while True:
                current_count = (
                    len(pd.concat(pages, ignore_index=True))
                    if pages else 0
                )

                limit = min(
                    1000,
                    max(1, target - current_count),
                )

                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                }

                if end_time is not None:
                    params["endTime"] = end_time

                response = http_get(
                    endpoint + "/api/v3/klines",
                    params,
                )

                if response.status_code in (403, 451):
                    raise RuntimeError(
                        f"HTTP {response.status_code}: endpoint blocked"
                    )

                if response.status_code == 429:
                    raise RuntimeError(
                        "HTTP 429: rate limit reached"
                    )

                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, list) or not payload:
                    raise RuntimeError(
                        f"No Binance candles returned: {str(payload)[:150]}"
                    )

                page = parse_binance_candles(payload)
                pages.append(page)

                combined = normalize_candles(
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

            combined = normalize_candles(
                pd.concat(pages, ignore_index=True)
            )

            return (
                combined.tail(target).reset_index(drop=True),
                endpoint,
                None,
            )

        except Exception as error:
            errors.append(
                f"{endpoint}: {type(error).__name__}: {error}"
            )

    raise RuntimeError(
        "Binance endpoints failed: " + " | ".join(errors)
    )


# ============================================================
# BYBIT PUBLIC CANDLE DATA
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
            if pages else 0
        )

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": "1",
            "limit": min(1000, max(40, target - current_count)),
        }

        if end_time is not None:
            params["end"] = end_time

        response = http_get(
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
            normalize_candles(
                data[["time", "open", "high", "low", "close", "volume"]]
            )
        )

        combined = normalize_candles(
            pd.concat(pages, ignore_index=True)
        )

        if len(combined) >= target or target <= 1000:
            break

        next_end = int(
            combined["time"].min().timestamp() * 1000
        ) - 1

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
        normalize_candles(
            pd.concat(pages, ignore_index=True)
        ).tail(target).reset_index(drop=True),
        "Bybit public API",
        warning,
    )


# ============================================================
# OKX PUBLIC CANDLE DATA
# ============================================================

def fetch_okx(symbol, target):
    instrument = symbol[:-4] + "-USDT"
    pages = []
    after = None
    warning = None

    while True:
        current_count = (
            len(pd.concat(pages, ignore_index=True))
            if pages else 0
        )

        params = {
            "instId": instrument,
            "bar": "1m",
            "limit": str(min(300, max(40, target - current_count))),
        }

        if after is not None:
            params["after"] = after

        response = http_get(
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
            normalize_candles(
                data[["time", "open", "high", "low", "close", "volume"]]
            )
        )

        combined = normalize_candles(
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
        normalize_candles(
            pd.concat(pages, ignore_index=True)
        ).tail(target).reset_index(drop=True),
        "OKX public API",
        warning,
    )


# ============================================================
# GATE.IO AND COINBASE FALLBACKS
# ============================================================

def fetch_gateio(symbol, target):
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
        normalize_candles(
            data[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Gate.io public API",
        None,
    )


def fetch_coinbase(symbol):
    product = COINBASE_SYMBOLS.get(symbol)

    if not product:
        raise RuntimeError("No Coinbase mapping for this symbol")

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

    data = pd.DataFrame(
        rows,
        columns=["stamp", "low", "high", "open", "close", "volume"],
    )

    data["time"] = pd.to_datetime(
        data["stamp"],
        unit="s",
        utc=True,
    )

    return (
        normalize_candles(
            data[["time", "open", "high", "low", "close", "volume"]]
        ),
        "Coinbase public API",
        None,
    )


# ============================================================
# YAHOO FINANCE DATA
# ============================================================

def fetch_yahoo(asset):
    symbol = asset["symbol"]

    if asset["kind"] == "crypto":
        symbol = COINBASE_SYMBOLS.get(symbol, symbol)

    errors = []

    for host in [
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    ]:
        try:
            encoded = requests.utils.quote(symbol, safe="")
            url = f"https://{host}/v8/finance/chart/{encoded}"

            response = http_get(
                url,
                {
                    "range": "7d",
                    "interval": "1m",
                    "includePrePost": "false",
                },
            )
            response.raise_for_status()

            payload = response.json().get("chart", {})

            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))

            results = payload.get("result") or []

            if not results:
                raise RuntimeError("Yahoo returned no result")

            item = results[0]
            timestamps = item.get("timestamp") or []
            quotes = item.get("indicators", {}).get("quote") or []

            if not timestamps or not quotes:
                raise RuntimeError("Yahoo returned no candle timestamps")

            quote = quotes[0]

            data = pd.DataFrame({
                "time": pd.to_datetime(
                    timestamps,
                    unit="s",
                    utc=True,
                ),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })

            return (
                normalize_candles(data),
                f"Yahoo Finance ({host})",
                None,
            )

        except Exception as error:
            errors.append(
                f"{host}: {type(error).__name__}: {error}"
            )

    raise RuntimeError("Yahoo failed: " + " | ".join(errors))


# ============================================================
# DATA PROVIDER FALLBACK AND CACHE
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
            ("Gate.io", lambda: fetch_gateio(symbol, target)),
            ("Coinbase", lambda: fetch_coinbase(symbol)),
            ("Yahoo", lambda: fetch_yahoo(asset)),
        ]

        for provider_name, provider_function in providers:
            try:
                recent, source, warning = provider_function()

                if warning:
                    warnings.append(str(warning))

                if provider_name != "Binance":
                    warnings.append(
                        f"Fallback provider {provider_name} used; "
                        "prices may differ from Binance USDT Spot."
                    )

                break

            except Exception as error:
                errors.append(
                    f"{provider_name}: "
                    f"{type(error).__name__}: {error}"
                )
    else:
        try:
            recent, source, warning = fetch_yahoo(asset)
            if warning:
                warnings.append(str(warning))
        except Exception as error:
            errors.append(
                f"Yahoo: {type(error).__name__}: {error}"
            )

    recent = normalize_candles(recent)

    if not recent.empty:
        save_candle_cache(symbol, recent)

    try:
        cached = load_candle_cache(symbol)
    except Exception as error:
        cached = pd.DataFrame()
        errors.append(f"SQLite: {error}")

    parts = [
        frame for frame in [cached, recent]
        if frame is not None and not frame.empty
    ]

    if not parts:
        raise RuntimeError(
            "No live or cached candles. "
            + (" | ".join(errors) or "All providers returned empty data.")
        )

    combined = normalize_candles(
        pd.concat(parts, ignore_index=True)
    ).tail(TARGET).reset_index(drop=True)

    if len(combined) < 40:
        raise RuntimeError(
            f"Only {len(combined)} completed candles; "
            "at least 40 are needed. "
            + " | ".join(errors)
        )

    age_minutes = (
        now_utc() - combined["time"].iloc[-1]
    ).total_seconds() / 60

    if age_minutes > 5:
        raise RuntimeError(
            f"Latest candle is {age_minutes:.1f} minutes old. "
            "No signal is calculated from stale data. "
            + " | ".join(errors)
        )

    warning_text = (
        " | ".join(warnings + errors)
        if warnings or errors else None
    )

    return combined, source, warning_text


# ============================================================
# EMA 15 AND BOLLINGER BANDS
# ============================================================

def add_indicators(df):
    data = df.copy()

    data["ema15"] = data["close"].ewm(
        span=15,
        adjust=False,
        min_periods=15,
    ).mean()

    data["bb_mid"] = data["close"].rolling(
        20,
        min_periods=20,
    ).mean()

    deviation = data["close"].rolling(
        20,
        min_periods=20,
    ).std(ddof=0)

    data["bb_upper"] = data["bb_mid"] + 2 * deviation
    data["bb_lower"] = data["bb_mid"] - 2 * deviation

    return data


def calculate_signal(df, lookback, band_mode):
    data = add_indicators(df)
    last = data.iloc[-1]
    segment = data.tail(lookback)

    no_signal = {
        "signal": "NO SIGNAL",
        "reason": "Not enough indicator data",
        "change_pct": np.nan,
        "trend": "Unknown",
        "green_ratio": np.nan,
        "red_ratio": np.nan,
    }

    if (
        len(data) < max(40, lookback)
        or pd.isna(last["ema15"])
        or pd.isna(last["bb_upper"])
        or pd.isna(last["bb_lower"])
    ):
        return no_signal

    first_price = float(segment["close"].iloc[0])
    price = float(last["close"])

    if first_price <= 0:
        return {**no_signal, "reason": "Invalid price"}

    change_pct = (price - first_price) / first_price * 100

    green_ratio = float(
        (segment["close"] > segment["open"]).mean()
    )

    red_ratio = float(
        (segment["close"] < segment["open"]).mean()
    )

    if band_mode == "Touch or cross":
        upper_ok = float(last["high"]) >= float(last["bb_upper"])
        lower_ok = float(last["low"]) <= float(last["bb_lower"])
    else:
        upper_ok = price > float(last["bb_upper"])
        lower_ok = price < float(last["bb_lower"])

    rising = change_pct > 0 and green_ratio >= 0.5
    falling = change_pct < 0 and red_ratio >= 0.5

    above_ema = price > float(last["ema15"])
    below_ema = price < float(last["ema15"])

    if rising and above_ema and upper_ok:
        signal = "SELL"
        reason = (
            f"Last {lookback} completed candles rose "
            f"{change_pct:.3f}%; close is above EMA 15 "
            "and the upper Bollinger condition is met."
        )

    elif falling and below_ema and lower_ok:
        signal = "BUY"
        reason = (
            f"Last {lookback} completed candles fell "
            f"{change_pct:.3f}%; close is below EMA 15 "
            "and the lower Bollinger condition is met."
        )

    else:
        signal = "NO SIGNAL"
        reason = (
            f"Conditions incomplete: change={change_pct:.3f}%, "
            f"green={green_ratio:.0%}, red={red_ratio:.0%}, "
            f"above EMA={above_ema}, below EMA={below_ema}, "
            f"upper band={upper_ok}, lower band={lower_ok}."
        )

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": change_pct,
        "trend": (
            "Rising" if rising
            else "Falling" if falling
            else "Mixed/sideways"
        ),
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
    }


# ============================================================
# SCAN ONE ASSET
# ============================================================

def scan_asset(asset, lookback, band_mode, target):
    try:
        candles, source, warning = get_candles(asset, target)
        data = add_indicators(candles)
        signal = calculate_signal(data, lookback, band_mode)
        last = data.iloc[-1]

        return {
            **asset,
            **signal,
            "price": float(last["close"]),
            "ema15": float(last["ema15"]),
            "bb_upper": float(last["bb_upper"]),
            "bb_mid": float(last["bb_mid"]),
            "bb_lower": float(last["bb_lower"]),
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
# AUTOMATIC EXCEL HISTORY
# ============================================================

def excel_row(result, scan_time):
    fields = [
        "name", "symbol", "category", "kind",
        "signal", "reason", "trend",
        "change_pct", "green_ratio", "red_ratio",
        "price", "ema15", "bb_upper", "bb_mid", "bb_lower",
        "count", "last_candle_utc", "source", "warning", "error",
    ]

    row = {
        "scan_time_utc": scan_time,
        **{field: result.get(field) for field in fields},
    }

    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            row[key] = None
        elif isinstance(value, np.integer):
            row[key] = int(value)

    return row


def save_excel_history(results, scan_time):
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
        "Scan History": summary_table,
        "Scan Results": result_table,
        "Data Errors": error_table,
    }

    # Preserve previous scans if the workbook already exists.
    if EXCEL_PATH.exists():
        try:
            with pd.ExcelFile(EXCEL_PATH) as workbook:
                for name, new_table in list(sheets.items()):
                    if name in workbook.sheet_names:
                        old_table = pd.read_excel(
                            workbook,
                            sheet_name=name,
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
        ) as temporary_file:
            temporary_path = temporary_file.name

        with pd.ExcelWriter(
            temporary_path,
            engine="openpyxl",
        ) as writer:
            for name, table in sheets.items():
                table.to_excel(
                    writer,
                    sheet_name=name,
                    index=False,
                )

                worksheet = writer.sheets[name]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions

                for cells in worksheet.columns:
                    max_length = max(
                        (
                            len(str(cell.value))
                            if cell.value is not None else 0
                        )
                        for cell in cells[:1000]
                    )

                    worksheet.column_dimensions[
                        cells[0].column_letter
                    ].width = min(55, max(12, max_length + 2))

        os.replace(temporary_path, EXCEL_PATH)

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title="Velora Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Velora — EMA 15 + Bollinger Bands")
st.caption(
    "1-minute market scanner | BUY / SELL signals only | "
    "No order execution"
)

st.warning(
    "Crypto data tries Binance, Bybit, OKX, Gate.io, Coinbase and Yahoo. "
    "Forex and stock data uses Yahoo Finance. Availability varies. "
    "Yahoo prices are not Binomo OTC quotes. "
    "This app never places trades."
)

st.sidebar.header("Scanner settings")

category = st.sidebar.selectbox(
    "Asset category",
    ["All", "Crypto", "Forex", "Other"],
)

lookback = st.sidebar.select_slider(
    "Rise/fall candle count",
    options=[15, 16, 17],
    value=15,
)

band_mode = st.sidebar.radio(
    "Bollinger confirmation",
    ["Touch or cross", "Close outside band"],
)

asset_limit = int(
    st.sidebar.number_input(
        "Maximum assets per scan",
        min_value=1,
        max_value=len(ASSETS),
        value=10,
    )
)

deep_sync = st.sidebar.checkbox(
    "Sync toward 100,000 candles (slower)",
    value=False,
)

show_no_signal = st.sidebar.checkbox(
    "Show NO SIGNAL results",
    value=True,
)

selected_assets = [
    asset for asset in ASSETS
    if category == "All" or asset["category"] == category
][:asset_limit]

target = TARGET if deep_sync else RECENT_LIMIT

st.markdown(f"""
### Strategy rules

- **SELL:** The last {lookback} completed 1-minute candles rose,
  at least half were green, the latest close is above EMA 15,
  and the upper Bollinger condition is met.
- **BUY:** The last {lookback} completed 1-minute candles fell,
  at least half were red, the latest close is below EMA 15,
  and the lower Bollinger condition is met.
- **NO SIGNAL:** The conditions are not all met or current data is unavailable.
- The unfinished current candle is excluded.
""")

st.write(f"Assets selected: **{len(selected_assets)}**")
st.write(f"History target: **{target:,} candles per asset**")
st.caption(f"Excel history file: `{EXCEL_PATH}`")

if "results" not in st.session_state:
    st.session_state.results = []

if "scan_time" not in st.session_state:
    st.session_state.scan_time = None

if st.button(
    "🔍 START SCAN",
    type="primary",
    use_container_width=True,
):
    results = []
    progress = st.progress(0.0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=min(3, max(1, len(selected_assets)))
    ) as executor:

        futures = {
            executor.submit(
                scan_asset,
                asset,
                int(lookback),
                band_mode,
                target,
            ): asset
            for asset in selected_assets
        }

        for index, future in enumerate(as_completed(futures), start=1):
            asset = futures[future]

            try:
                results.append(future.result())

            except Exception as error:
                results.append({
                    **asset,
                    "signal": "NO SIGNAL",
                    "reason": "Unexpected scan error.",
                    "candles": None,
                    "error": str(error),
                })

            progress.progress(index / max(1, len(futures)))
            status.write(
                f"Scanned {index} / {len(futures)} assets"
            )

    order = {
        asset["symbol"]: index
        for index, asset in enumerate(selected_assets)
    }

    results.sort(
        key=lambda result: order.get(result.get("symbol"), 999)
    )

    scan_time = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")

    st.session_state.results = results
    st.session_state.scan_time = scan_time

    try:
        save_excel_history(results, scan_time)
        st.session_state.save_message = (
            f"Excel history saved automatically: {EXCEL_PATH.name}"
        )
    except Exception as error:
        st.session_state.save_message = (
            f"Excel save failed: {error}"
        )

    progress.empty()
    status.empty()


results = st.session_state.results
scan_time = st.session_state.scan_time

if not results:
    st.info("Click START SCAN to retrieve candles and calculate signals.")

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

    st.stop()


if hasattr(st.session_state, "save_message"):
    message = st.session_state.save_message

    if message.startswith("Excel history saved"):
        st.success(message)
    else:
        st.error(message)


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
    if result["signal"] == "BUY"
]

sells = [
    result for result in valid
    if result["signal"] == "SELL"
]

st.caption(f"Last scan: {scan_time}")

column1, column2, column3, column4 = st.columns(4)

column1.metric(
    "Assets with 100K candles",
    sum(result["count"] >= TARGET for result in valid),
)

column2.metric("BUY", len(buys))
column3.metric("SELL", len(sells))
column4.metric("Data errors", len(errors))


# ============================================================
# BUY / SELL SIGNALS
# ============================================================

st.subheader("🚨 BUY / SELL signals")

signals = buys + sells

if signals:
    signal_table = pd.DataFrame([
        {
            "Asset": result["name"],
            "Symbol": result["symbol"],
            "Signal": result["signal"],
            "Price": result["price"],
            "Trend": result["trend"],
            "Change (%)": result["change_pct"],
            "EMA 15": result["ema15"],
            "BB upper": result["bb_upper"],
            "BB lower": result["bb_lower"],
            "Candles": result["count"],
            "Reason": result["reason"],
        }
        for result in signals
    ])

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No BUY or SELL setup currently meets all conditions. "
        "The scanner will not force a signal."
    )


# ============================================================
# ALL SCAN RESULTS
# ============================================================

st.subheader("📋 All scan results")

shown_results = (
    valid
    if show_no_signal
    else [
        result for result in valid
        if result["signal"] != "NO SIGNAL"
    ]
)

if shown_results:
    st.dataframe(
        pd.DataFrame([
            {
                "Asset": result["name"],
                "Symbol": result["symbol"],
                "Category": result["category"],
                "Signal": result["signal"],
                "Price": result["price"],
                "Candles": result["count"],
                "Source": result["source"],
                "Reason": result["reason"],
            }
            for result in shown_results
        ]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No valid scan results. Review the Data errors section."
    )


# ============================================================
# CHARTS
# ============================================================

st.subheader("📈 Asset chart")

chartable = [
    result for result in valid
    if isinstance(result.get("candles"), pd.DataFrame)
    and not result["candles"].empty
]

if chartable:
    chosen = st.selectbox(
        "Choose an asset",
        chartable,
        format_func=lambda result: result["name"],
    )

    chart = (
        chosen["candles"]
        .tail(500)
        .set_index("time")
    )

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
        "could be retrieved."
    )


# ============================================================
# DETAILED DATA ERRORS
# ============================================================

st.subheader("⚠️ Data errors — details")

if errors:
    st.error(
        f"{len(errors)} asset(s) could not be scanned."
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Asset": result["name"],
                "Symbol": result["symbol"],
                "Provider": result["kind"],
                "Error details": result["error"],
            }
            for result in errors
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "If all providers fail with DNS or connection errors, "
        "the hosting environment may be blocking external APIs."
    )

else:
    st.success("No data errors in this scan.")


# ============================================================
# EXCEL DOWNLOAD
# ============================================================

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
        "Workbook sheets: Scan History, Scan Results, Data Errors."
    )

csv_table = pd.DataFrame([
    excel_row(result, scan_time or "Unknown")
    for result in results
])

st.download_button(
    "📄 Download current scan as CSV",
    csv_table.to_csv(index=False).encode("utf-8-sig"),
    "velora_current_scan.csv",
    "text/csv",
    use_container_width=True,
)

st.divider()

st.caption(
    "Velora provides analysis signals only and never executes trades. "
    "100,000 candles are a target, not a guarantee. "
    "Download the Excel workbook to keep a copy because hosted files "
    "may be deleted during redeployment."
)
