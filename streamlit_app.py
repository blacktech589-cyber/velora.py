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
# SQLITE CANDLE CACHE
# ============================================================

def init_db():
    with sqlite3.connect(DB, timeout=30) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                time TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(symbol, time)
            )
            """
        )
        connection.commit()


def normalize(df):
    columns = ["time", "open", "high", "low", "close", "volume"]

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
                float(row.volume),
            )
        )

    init_db()

    with sqlite3.connect(DB, timeout=30) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()


def load_cache(symbol):
    init_db()

    with sqlite3.connect(DB, timeout=30) as connection:
        data = pd.read_sql_query(
            """
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY time
            """,
            connection,
            params=(symbol,),
        )

    return normalize(data)


# ============================================================
# BINANCE PUBLIC DATA
# ============================================================

def parse_binance(rows):
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
                count = sum(len(page) for page in pages)
                limit = min(1000, max(1, target - count))

                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                }

                if end_time is not None:
                    params["endTime"] = end_time

                response = get(
                    base + "/api/v3/klines",
                    params,
                )

                if response.status_code in (403, 451):
                    raise RuntimeError(
                        f"HTTP {response.status_code}: endpoint blocked"
                    )

                if response.status_code == 429:
                    raise RuntimeError(
                        "HTTP 429: rate limited; try again later"
                    )

                response.raise_for_status()
                rows = response.json()

                if not isinstance(rows, list) or len(rows) == 0:
                    raise RuntimeError(
                        f"No Binance candles: {str(rows)[:150]}"
                    )

                page = parse_binance(rows)
                pages.append(page)

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

            combined = normalize(
                pd.concat(pages, ignore_index=True)
            ).tail(target).reset_index(drop=True)

            return combined, base, None

        except Exception as error:
            errors.append(
                f"{base}: {type(error).__name__}: {error}"
            )

    raise RuntimeError(
        "Binance failed: " + " | ".join(errors)
    )


# ============================================================
# BYBIT PUBLIC DATA
# ============================================================

def fetch_bybit(symbol, target):
    if not symbol.endswith("USDT"):
        raise RuntimeError("Unsupported Bybit symbol")

    pages = []
    end_time = None
    warning = None

    while True:
        count = sum(len(page) for page in pages)

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": "1",
            "limit": min(1000, max(40, target - count)),
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

        if len(rows) == 0:
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
                data[["time", "open", "high", "low", "close", "volume"]]
            )
        )

        combined = normalize(
            pd.concat(pages, ignore_index=True)
        )

        if len(combined) >= target or target <= 1000:
            break

        next_end = int(combined["time"].min().timestamp() * 1000) - 1

        if end_time is not None and next_end >= end_time:
            warning = "Bybit historical paging stopped."
            break

        end_time = next_end
        time.sleep(0.05)

        if len(pages) >= 110:
            warning = "Bybit sync limited to 110 pages."
            break

    if len(pages) == 0:
        raise RuntimeError("Bybit returned no candles")

    return (
        normalize(pd.concat(pages, ignore_index=True))
        .tail(target)
        .reset_index(drop=True),
        "Bybit public API",
        warning,
    )


# ============================================================
# OKX PUBLIC DATA
# ============================================================

def fetch_okx(symbol, target):
    instrument = symbol[:-4] + "-USDT"
    pages = []
    after = None
    warning = None

    while True:
        count = sum(len(page) for page in pages)

        params = {
            "instId": instrument,
            "bar": "1m",
            "limit": str(min(300, max(40, target - count))),
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

        if len(rows) == 0:
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
                data[["time", "open", "high", "low", "close", "volume"]]
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

    if len(pages) == 0:
        raise RuntimeError("OKX returned no candles")

    return (
        normalize(pd.concat(pages, ignore_index=True))
        .tail(target)
        .reset_index(drop=True),
        "OKX public API",
        warning,
    )


# ============================================================
# GATE.IO AND COINBASE
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

    if not isinstance(rows, list) or len(rows) == 0:
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

    if len(rows) == 0:
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

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
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

            if len(items) == 0:
                raise RuntimeError("Yahoo returned no result")

            item = items[0]
            quotes = item.get("indicators", {}).get("quote") or [{}]
            quote = quotes[0]
            stamps = item.get("timestamp") or []

            if len(stamps) == 0:
                raise RuntimeError("Yahoo returned no timestamps")

            data = pd.DataFrame({
                "time": pd.to_datetime(stamps, unit="s", utc=True),
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
# LOAD LIVE AND CACHED CANDLES
# ============================================================

def get_candles(asset, target):
    symbol = asset["symbol"]
    errors = []
    warnings = []
    recent = pd.DataFrame()
    source = ""

    if asset["kind"] == "crypto":
        fetchers = [
            ("Binance", lambda: fetch_binance(symbol, target)),
            ("Bybit", lambda: fetch_bybit(symbol, target)),
            ("OKX", lambda: fetch_okx(symbol, target)),
            ("Gate.io", lambda: fetch_gate(symbol, target)),
            ("Coinbase", lambda: fetch_coinbase(symbol)),
            ("Yahoo", lambda: fetch_yahoo(asset)),
        ]

        for provider_name, fetcher in fetchers:
            try:
                recent, source, note = fetcher()

                if note is not None:
                    warnings.append(str(note))

                if provider_name != "Binance":
                    warnings.append(
                        f"Fallback provider {provider_name} used; "
                        "prices may differ from Binance USDT Spot."
                    )

                break

            except Exception as error:
                errors.append(
                    f"{provider_name}: {type(error).__name__}: {error}"
                )

    else:
        try:
            recent, source, note = fetch_yahoo(asset)

            if note is not None:
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

    parts = [
        frame
        for frame in [cached, recent]
        if frame is not None and not frame.empty
    ]

    if len(parts) == 0:
        raise RuntimeError(
            "No live or cached candles. "
            + (
                " | ".join(errors)
                if len(errors) > 0
                else "All providers returned empty data."
            )
        )

    data = normalize(
        pd.concat(parts, ignore_index=True)
    ).tail(TARGET).reset_index(drop=True)

    if len(data) < 40:
        raise RuntimeError(
            f"Only {len(data)} completed candles; need at least 40. "
            + " | ".join(errors)
        )

    age = (
        now_utc() - data["time"].iloc[-1]
    ).total_seconds() / 60

    if age > 5:
        raise RuntimeError(
            f"Latest candle is {age:.1f} minutes old; "
            "no signal calculated. "
            + " | ".join(errors)
        )

    note = (
        " | ".join(warnings + errors)
        if len(warnings) > 0 or len(errors) > 0
        else None
    )

    return data, source, note


# ============================================================
# INDICATORS AND SIGNALS
# ============================================================

def indicators(df):
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


def signal_for(df, lookback, mode):
    data = indicators(df)

    empty = {
        "signal": "NO SIGNAL",
        "reason": "Not enough indicator data",
        "change_pct": np.nan,
        "trend": "Unknown",
        "green_ratio": np.nan,
        "red_ratio": np.nan,
    }

    if len(data) < max(40, lookback):
        return empty

    last = data.iloc[-1]
    segment = data.tail(lookback)

    if (
        pd.isna(last["ema15"])
        or pd.isna(last["bb_upper"])
        or pd.isna(last["bb_lower"])
    ):
        return empty

    first_price = float(segment["close"].iloc[0])
    price = float(last["close"])

    if first_price <= 0:
        return {**empty, "reason": "Invalid price"}

    change = (price - first_price) / first_price * 100

    green = float(
        (segment["close"] > segment["open"]).mean()
    )

    red = float(
        (segment["close"] < segment["open"]).mean()
    )

    if mode == "Touch or cross":
        upper = float(last["high"]) >= float(last["bb_upper"])
        lower = float(last["low"]) <= float(last["bb_lower"])
    else:
        upper = price > float(last["bb_upper"])
        lower = price < float(last["bb_lower"])

    rising = change > 0 and green >= 0.5
    falling = change < 0 and red >= 0.5

    above = price > float(last["ema15"])
    below = price < float(last["ema15"])

    if rising and above and upper:
        signal = "SELL"
        reason = (
            f"Last {lookback} candles rose {change:.3f}%; "
            "above EMA15 and upper Bollinger condition met."
        )

    elif falling and below and lower:
        signal = "BUY"
        reason = (
            f"Last {lookback} candles fell {change:.3f}%; "
            "below EMA15 and lower Bollinger condition met."
        )

    else:
        signal = "NO SIGNAL"
        reason = (
            f"Conditions incomplete: change={change:.3f}%, "
            f"green={green:.0%}, red={red:.0%}, "
            f"above EMA={above}, below EMA={below}, "
            f"upper band={upper}, lower band={lower}."
        )

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": change,
        "trend": (
            "Rising" if rising
            else "Falling" if falling
            else "Mixed/sideways"
        ),
        "green_ratio": green,
        "red_ratio": red,
    }


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
# EXCEL HISTORY
# ============================================================

def excel_row(result, scan_time):
    fields = [
        "name", "symbol", "category", "kind", "signal", "reason",
        "trend", "change_pct", "green_ratio", "red_ratio",
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
    rows = pd.DataFrame([
        excel_row(result, scan_time)
        for result in results
    ])

    errors = rows[rows["error"].notna()].copy()

    summary = pd.DataFrame([{
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
        "Scan History": summary,
        "Scan Results": rows,
        "Data Errors": errors,
    }

    if XLSX.exists():
        try:
            with pd.ExcelFile(XLSX) as workbook:
                for name, frame in list(sheets.items()):
                    if name in workbook.sheet_names:
                        old = pd.read_excel(
                            workbook,
                            sheet_name=name,
                        )

                        sheets[name] = pd.concat(
                            [old, frame],
                            ignore_index=True,
                        ).tail(1_000_000)

        except Exception:
            # If an existing workbook cannot be read, create a new workbook.
            pass

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            prefix="velora_",
            dir=DATA_DIR,
            delete=False,
        ) as temporary_file:
            temp_path = temporary_file.name

        with pd.ExcelWriter(
            temp_path,
            engine="openpyxl",
        ) as writer:
            for name, frame in sheets.items():
                frame.to_excel(
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
                            if cell.value is not None
                            else 0
                        )
                        for cell in cells[:1000]
                    )

                    worksheet.column_dimensions[
                        cells[0].column_letter
                    ].width = min(
                        55,
                        max(12, max_length + 2),
                    )

        os.replace(temp_path, XLSX)

    finally:
        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


# ============================================================
# STREAMLIT UI
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
    "Public market data only. Crypto tries Binance, Bybit, OKX, "
    "Gate.io, Coinbase and Yahoo. Forex/stock uses Yahoo Finance. "
    "Data availability varies; Yahoo prices are not Binomo OTC quotes. "
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

mode = st.sidebar.radio(
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

- **SELL:** The last {lookback} completed 1-minute candles rose,
  at least half were green, the latest close is above EMA 15,
  and the upper Bollinger condition is met.
- **BUY:** The last {lookback} completed 1-minute candles fell,
  at least half were red, the latest close is below EMA 15,
  and the lower Bollinger condition is met.
- **NO SIGNAL:** The conditions are not all met or reliable data is unavailable.
- The current unfinished candle is excluded.
""")

st.write(
    f"Assets selected: **{len(assets)}** · "
    f"History target: **{target:,} candles per asset**"
)

st.caption(
    f"Excel automatically saved to `{XLSX}` after each scan. "
    "Download it to preserve a copy; hosted local files may not "
    "survive redeployment."
)


# IMPORTANT:
# Streamlit can preserve session state across reruns. Older versions may
# have left a DataFrame in "results". A DataFrame cannot be evaluated as
# True/False. Reset incompatible state before using it.
if (
    "results" not in st.session_state
    or not isinstance(st.session_state.get("results"), list)
):
    st.session_state["results"] = []

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None


# ============================================================
# RUN SCAN
# ============================================================

if st.button(
    "🔍 START SCAN",
    type="primary",
    use_container_width=True,
):
    results = []
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
                results.append(future.result())

            except Exception as error:
                results.append({
                    **asset,
                    "signal": "NO SIGNAL",
                    "reason": "Unexpected error",
                    "candles": None,
                    "error": str(error),
                })

            progress.progress(index / max(1, len(futures)))
            status.write(
                f"Scanned {index}/{len(futures)} assets"
            )

    order = {
        asset["symbol"]: index
        for index, asset in enumerate(assets)
    }

    results.sort(
        key=lambda result: order.get(result.get("symbol"), 999)
    )

    scan_time = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")

    st.session_state["results"] = results
    st.session_state["scan_time"] = scan_time

    try:
        save_excel(results, scan_time)
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
# READ SESSION STATE SAFELY
# ============================================================

# Do not use `if results` on a potentially stored DataFrame.
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


if "save_msg" in st.session_state:
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
    if result["signal"] == "BUY"
]

sells = [
    result
    for result in valid
    if result["signal"] == "SELL"
]


# ============================================================
# SUMMARY METRICS
# ============================================================

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

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No BUY or SELL setup meets all selected conditions. "
        "No signal is forced."
    )


# ============================================================
# ALL SCAN RESULTS
# ============================================================

st.subheader("📋 All scan results")

if show_no:
    shown = valid
else:
    shown = [
        result
        for result in valid
        if result["signal"] != "NO SIGNAL"
    ]

if len(shown) > 0:
    all_table = pd.DataFrame([
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
        for result in shown
    ])

    st.dataframe(
        all_table,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No valid scan results. Review Data errors below."
    )


# ============================================================
# CHARTS
# ============================================================

st.subheader("📈 Asset chart")

chartable = [
    result
    for result in valid
    if isinstance(result.get("candles"), pd.DataFrame)
    and not result["candles"].empty
]

if len(chartable) > 0:
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
        "was retrieved."
    )


# ============================================================
# DATA ERRORS
# ============================================================

st.subheader("⚠️ Data errors — readable details")

if len(errors) > 0:
    st.error(
        f"{len(errors)} asset(s) could not be scanned. "
        "The messages below can be copied."
    )

    error_lines = []

    for result in errors:
        detail = str(
            result.get("error") or "No error text returned"
        )

        error_lines.append(
            f"ASSET: {result.get('name')} | "
            f"SYMBOL: {result.get('symbol')} | "
            f"CATEGORY: {result.get('category')}\n"
            f"{detail}"
        )

    readable_errors = "\n\n".join(error_lines)

    st.text_area(
        "Copy these provider error messages",
        value=readable_errors,
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
# EXCEL / CSV DOWNLOADS
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

export = pd.DataFrame([
    excel_row(result, scan_time or "Unknown")
    for result in results
])

st.download_button(
    "📄 Download current scan as CSV",
    export.to_csv(index=False).encode("utf-8-sig"),
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
