from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Velora: public-market-data scanner. It NEVER places trades.
# -----------------------------------------------------------------------------
APP_TITLE = "Velora — EMA 15 + Bollinger Bands"
TARGET_CANDLES = 100_000
RECENT_CANDLES = 1_000
REQUEST_TIMEOUT = 15
MAX_STALE_MINUTES = 5

BINANCE_BASES = [
    "https://data-api.binance.vision",  # public market-data endpoint
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
COINBASE_PRODUCTS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
    "ADAUSDT": "ADA-USD",
    "AVAXUSDT": "AVAX-USD",
    "LINKUSDT": "LINK-USD",
}
KRAKEN_PAIRS = {
    "BTCUSDT": "XBTUSD",
    "ETHUSDT": "ETHUSD",
    "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XRPUSD",
    "DOGEUSDT": "DOGEUSD",
    "ADAUSDT": "ADAUSD",
    "AVAXUSDT": "AVAXUSD",
    "LINKUSDT": "LINKUSD",
}
YAHOO_CRYPTO_SYMBOLS = {symbol: product.replace("-", "-") for symbol, product in COINBASE_PRODUCTS.items()}

ASSETS = [
    {"name": "Bitcoin", "symbol": "BTCUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Ethereum", "symbol": "ETHUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Solana", "symbol": "SOLUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "XRP", "symbol": "XRPUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Dogecoin", "symbol": "DOGEUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Cardano", "symbol": "ADAUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Avalanche", "symbol": "AVAXUSDT", "category": "Crypto", "provider": "crypto"},
    {"name": "Chainlink", "symbol": "LINKUSDT", "category": "Crypto", "provider": "crypto"},
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

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VeloraScanner/2.0)",
    "Accept": "application/json,text/plain,*/*",
}


def choose_data_dir() -> Path:
    """Pick a writable directory for SQLite and automatic Excel output."""
    requested = os.environ.get("VELORA_DATA_DIR")
    candidates = [Path(requested)] if requested else [Path.cwd(), Path("/tmp/velora_data")]
    for folder in candidates:
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


DATA_DIR = choose_data_dir()
DB_PATH = DATA_DIR / "velora_history.sqlite3"
EXCEL_PATH = DATA_DIR / "velora_scan_history.xlsx"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def current_minute() -> pd.Timestamp:
    return utc_now().floor("min")


def http_get(url: str, params: dict[str, Any] | None = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    return requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout)


def init_db() -> None:
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                candle_time TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0,
                PRIMARY KEY(symbol, candle_time)
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol, candle_time)")
        conn.commit()


def normalize_candles(df: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else np.nan
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    out = out[out["time"] < current_minute()]
    out = out.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    return out[columns]


def load_cache(symbol: str) -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            "SELECT candle_time AS time, open, high, low, close, volume "
            "FROM candles WHERE symbol=? ORDER BY candle_time",
            conn,
            params=(symbol,),
        )
    return normalize_candles(df)


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    clean = normalize_candles(df)
    if clean.empty:
        return
    rows = []
    for row in clean.itertuples(index=False):
        ts = pd.Timestamp(row.time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        volume = 0.0 if pd.isna(row.volume) else float(row.volume)
        rows.append((symbol, ts, float(row.open), float(row.high), float(row.low), float(row.close), volume))
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany("INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()


def parse_binance_klines(payload: list) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    raw = pd.DataFrame(payload, columns=cols)
    raw["time"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return normalize_candles(raw[["time", "open", "high", "low", "close", "volume"]])


def request_binance_page(base: str, symbol: str, limit: int, end_time_ms: int | None = None) -> pd.DataFrame:
    params: dict[str, Any] = {"symbol": symbol, "interval": "1m", "limit": min(1000, max(1, int(limit)))}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    response = http_get(f"{base}/api/v3/klines", params=params)
    if response.status_code == 451:
        raise RuntimeError("HTTP 451: Binance data endpoint blocked from this region/network.")
    if response.status_code == 429:
        raise RuntimeError("HTTP 429: provider rate limit; wait and retry.")
    if response.status_code in (401, 403):
        raise RuntimeError(f"HTTP {response.status_code}: provider/network access denied.")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"No Binance candles returned: {str(payload)[:180]}")
    return parse_binance_klines(payload)


def fetch_binance(symbol: str, target_count: int = RECENT_CANDLES) -> tuple[pd.DataFrame, str, str | None]:
    """Fetch recent candles; optionally page back toward the configured history target."""
    errors: list[str] = []
    for base in BINANCE_BASES:
        try:
            first = request_binance_page(base, symbol, min(1000, target_count))
            pages = [first]
            combined = normalize_candles(pd.concat(pages, ignore_index=True))
            warning = None
            if target_count > RECENT_CANDLES:
                while len(combined) < target_count:
                    # The API is paginated backward using the oldest known open time.
                    earliest = combined["time"].min()
                    end_ms = int(earliest.timestamp() * 1000) - 1
                    remaining = target_count - len(combined)
                    try:
                        older = request_binance_page(base, symbol, min(1000, remaining), end_time_ms=end_ms)
                    except Exception as page_exc:
                        warning = f"Historical sync stopped after {len(combined):,} candles: {page_exc}"
                        break
                    if older.empty:
                        warning = "Historical sync stopped: provider returned an empty page."
                        break
                    old_min = older["time"].min()
                    if old_min >= earliest:
                        warning = "Historical sync stopped: provider returned no older candles."
                        break
                    pages.append(older)
                    new_combined = normalize_candles(pd.concat(pages, ignore_index=True))
                    if len(new_combined) <= len(combined):
                        warning = "Historical sync stopped because the candle count did not increase."
                        break
                    combined = new_combined.tail(target_count).reset_index(drop=True)
                    time.sleep(0.04)  # gentle pacing for public endpoints
            if combined.empty:
                raise RuntimeError("No completed Binance candles available.")
            return combined.tail(target_count).reset_index(drop=True), base, warning
        except Exception as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("All Binance public endpoints failed. " + " | ".join(errors))


def fetch_coinbase(symbol: str) -> tuple[pd.DataFrame, str]:
    product = COINBASE_PRODUCTS.get(symbol)
    if not product:
        raise RuntimeError("No Coinbase fallback mapping for this symbol.")
    end = utc_now().floor("min")
    start = end - pd.Timedelta(minutes=299)
    params = {
        "granularity": 60,
        "start": start.to_pydatetime().isoformat(timespec="seconds").replace("+00:00", "Z"),
        "end": end.to_pydatetime().isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    response = http_get(f"https://api.exchange.coinbase.com/products/{product}/candles", params=params)
    if response.status_code == 429:
        raise RuntimeError("Coinbase HTTP 429: rate limit.")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Coinbase returned no candles.")
    # Coinbase candle order: time, low, high, open, close, volume.
    frame = pd.DataFrame(payload, columns=["stamp", "low", "high", "open", "close", "volume"])
    frame["time"] = pd.to_datetime(frame["stamp"], unit="s", utc=True, errors="coerce")
    return normalize_candles(frame[["time", "open", "high", "low", "close", "volume"]]), "Coinbase Exchange public API"


def fetch_kraken(symbol: str) -> tuple[pd.DataFrame, str]:
    pair = KRAKEN_PAIRS.get(symbol)
    if not pair:
        raise RuntimeError("No Kraken fallback mapping for this symbol.")
    response = http_get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": 1})
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError("Kraken API error: " + "; ".join(map(str, payload["error"])))
    result = payload.get("result") or {}
    key = next((k for k in result if k != "last"), None)
    if not key or not result.get(key):
        raise RuntimeError("Kraken returned no candles.")
    frame = pd.DataFrame(result[key], columns=["stamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["stamp"], errors="coerce"), unit="s", utc=True, errors="coerce")
    return normalize_candles(frame[["time", "open", "high", "low", "close", "volume"]]), "Kraken public OHLC API"


def yahoo_symbol(asset: dict[str, str]) -> str:
    if asset["provider"] == "crypto":
        return YAHOO_CRYPTO_SYMBOLS.get(asset["symbol"], asset["symbol"])
    return asset["symbol"]


def fetch_yahoo(asset: dict[str, str]) -> tuple[pd.DataFrame, str]:
    symbol = yahoo_symbol(asset)
    encoded = requests.utils.quote(symbol, safe="")
    errors: list[str] = []
    for host in YAHOO_HOSTS:
        try:
            response = http_get(
                f"https://{host}/v8/finance/chart/{encoded}",
                params={"range": "7d", "interval": "1m", "includePrePost": "false"},
            )
            if response.status_code == 429:
                raise RuntimeError("Yahoo HTTP 429: rate limit.")
            if response.status_code in (401, 403):
                raise RuntimeError(f"Yahoo HTTP {response.status_code}: access denied.")
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            rows = chart.get("result") or []
            if not rows:
                raise RuntimeError("Yahoo returned no chart data for this symbol.")
            item = rows[0]
            stamps = item.get("timestamp") or []
            quotes = item.get("indicators", {}).get("quote") or []
            if not stamps or not quotes:
                raise RuntimeError("Yahoo response is missing timestamps or OHLC values.")
            q = quotes[0]
            frame = pd.DataFrame({
                "time": pd.to_datetime(stamps, unit="s", utc=True),
                "open": q.get("open", []),
                "high": q.get("high", []),
                "low": q.get("low", []),
                "close": q.get("close", []),
                "volume": q.get("volume", []),
            })
            return normalize_candles(frame), f"Yahoo Finance ({host})"
        except Exception as exc:
            errors.append(f"{host}: {exc}")
    raise RuntimeError("Yahoo endpoints failed. " + " | ".join(errors))


def get_candles(asset: dict[str, str], target_count: int) -> tuple[pd.DataFrame, str, str | None]:
    symbol = asset["symbol"]
    recent = pd.DataFrame()
    source = ""
    warnings: list[str] = []
    errors: list[str] = []

    if asset["provider"] == "crypto":
        try:
            recent, host, warn = fetch_binance(symbol, target_count=target_count)
            source = f"Binance public market data ({host})"
            if warn:
                warnings.append(warn)
        except Exception as exc:
            errors.append(f"Binance: {exc}")
            for fetcher in (fetch_coinbase, fetch_kraken):
                try:
                    recent, source = fetcher(symbol)
                    warnings.append("Fallback source used. Price may differ from Binance USDT markets.")
                    break
                except Exception as fallback_exc:
                    errors.append(f"{fetcher.__name__}: {fallback_exc}")
            if recent.empty:
                try:
                    recent, source = fetch_yahoo(asset)
                    warnings.append("Fallback source used; Yahoo crypto data may differ from USDT markets.")
                except Exception as exc2:
                    errors.append(f"Yahoo fallback: {exc2}")
    else:
        try:
            recent, source = fetch_yahoo(asset)
        except Exception as exc:
            errors.append(f"Yahoo: {exc}")

    recent = normalize_candles(recent)
    if not recent.empty:
        save_cache(symbol, recent)
    cached = load_cache(symbol)
    frames = [part for part in (cached, recent) if part is not None and not part.empty]
    if not frames:
        raise RuntimeError("No live or cached candles. " + " | ".join(errors))

    combined = normalize_candles(pd.concat(frames, ignore_index=True)).tail(TARGET_CANDLES).reset_index(drop=True)
    if len(combined) < 40:
        raise RuntimeError(f"Only {len(combined)} completed candles available; at least 40 are needed. " + " | ".join(errors))

    age_minutes = (utc_now() - combined["time"].iloc[-1]).total_seconds() / 60
    if age_minutes > MAX_STALE_MINUTES:
        raise RuntimeError(
            f"Latest available candle is {age_minutes:.1f} minutes old; limit is {MAX_STALE_MINUTES} minutes. "
            "No current signal was calculated to avoid using stale data. Provider details: " + " | ".join(errors)
        )

    if not source:
        source = "SQLite cache (live provider unavailable)"
    warning = " | ".join(warnings + errors) if (warnings or errors) else None
    return combined, source, warning


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema15"] = out["close"].ewm(span=15, adjust=False, min_periods=15).mean()
    out["bb_mid"] = out["close"].rolling(window=20, min_periods=20).mean()
    sd = out["close"].rolling(window=20, min_periods=20).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + 2 * sd
    out["bb_lower"] = out["bb_mid"] - 2 * sd
    return out


def calculate_signal(df: pd.DataFrame, lookback: int, band_mode: str) -> dict[str, Any]:
    no_signal = {
        "signal": "NO SIGNAL", "reason": "Not enough valid data.",
        "change_pct": np.nan, "trend": "Unknown", "green_ratio": np.nan, "red_ratio": np.nan,
    }
    if df is None or len(df) < max(lookback, 40):
        return no_signal

    data = add_indicators(df)
    last = data.iloc[-1]
    segment = data.iloc[-lookback:]
    if pd.isna(last["ema15"]) or pd.isna(last["bb_upper"]) or pd.isna(last["bb_lower"]):
        return {**no_signal, "reason": "EMA/Bollinger indicators are not ready."}

    first_close = float(segment.iloc[0]["close"])
    price = float(last["close"])
    if first_close <= 0:
        return {**no_signal, "reason": "Invalid starting price."}

    change = (price - first_close) / first_close * 100.0
    green_ratio = float((segment["close"] > segment["open"]).mean())
    red_ratio = float((segment["close"] < segment["open"]).mean())
    if band_mode == "Touch or cross":
        upper_ok = float(last["high"]) >= float(last["bb_upper"])
        lower_ok = float(last["low"]) <= float(last["bb_lower"])
    else:
        upper_ok = price > float(last["bb_upper"])
        lower_ok = price < float(last["bb_lower"])

    rising = change > 0 and green_ratio >= 0.5
    falling = change < 0 and red_ratio >= 0.5
    above_ema = price > float(last["ema15"])
    below_ema = price < float(last["ema15"])

    if rising and above_ema and upper_ok:
        signal = "SELL"
        reason = f"Last {lookback} completed candles rose {change:.3f}%; close is above EMA 15 and upper Bollinger condition is met."
    elif falling and below_ema and lower_ok:
        signal = "BUY"
        reason = f"Last {lookback} completed candles fell {change:.3f}%; close is below EMA 15 and lower Bollinger condition is met."
    else:
        signal = "NO SIGNAL"
        reason = (
            f"Conditions incomplete: change {change:.3f}%, green candles {green_ratio:.0%}, "
            f"red candles {red_ratio:.0%}, above EMA={above_ema}, below EMA={below_ema}, "
            f"upper band={upper_ok}, lower band={lower_ok}."
        )
    trend = "Rising" if rising else "Falling" if falling else "Mixed / sideways"
    return {
        "signal": signal, "reason": reason, "change_pct": change, "trend": trend,
        "green_ratio": green_ratio, "red_ratio": red_ratio,
    }


def scan_asset(asset: dict[str, str], lookback: int, band_mode: str, target_count: int) -> dict[str, Any]:
    try:
        candles, source, warning = get_candles(asset, target_count=target_count)
        data = add_indicators(candles)
        signal_result = calculate_signal(data, lookback, band_mode)
        last = data.iloc[-1]
        return {
            **asset, **signal_result,
            "price": float(last["close"]),
            "ema15": float(last["ema15"]) if pd.notna(last["ema15"]) else np.nan,
            "bb_upper": float(last["bb_upper"]) if pd.notna(last["bb_upper"]) else np.nan,
            "bb_mid": float(last["bb_mid"]) if pd.notna(last["bb_mid"]) else np.nan,
            "bb_lower": float(last["bb_lower"]) if pd.notna(last["bb_lower"]) else np.nan,
            "count": len(data), "last_candle_utc": str(data["time"].iloc[-1]),
            "source": source, "candles": data, "error": None, "provider_warning": warning,
        }
    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL", "reason": "Data unavailable; no current signal calculated.",
            "change_pct": np.nan, "trend": "Unknown", "green_ratio": np.nan, "red_ratio": np.nan,
            "price": np.nan, "ema15": np.nan, "bb_upper": np.nan, "bb_mid": np.nan,
            "bb_lower": np.nan, "count": 0, "last_candle_utc": None,
            "source": "Unavailable", "candles": None, "error": str(exc), "provider_warning": None,
        }


def _clean_excel_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def result_export_row(result: dict[str, Any], scan_time: str) -> dict[str, Any]:
    keys = [
        "name", "symbol", "category", "provider", "signal", "reason", "trend", "change_pct",
        "green_ratio", "red_ratio", "price", "ema15", "bb_upper", "bb_mid", "bb_lower",
        "count", "last_candle_utc", "source", "provider_warning", "error",
    ]
    row = {"scan_time_utc": scan_time, **{key: result.get(key) for key in keys}}
    return {key: _clean_excel_value(value) for key, value in row.items()}


def append_excel_history(results: list[dict[str, Any]], scan_time: str) -> None:
    """Append this scan into three workbook sheets and atomically replace the file."""
    result_rows = [result_export_row(r, scan_time) for r in results]
    result_frame = pd.DataFrame(result_rows)
    errors_frame = result_frame[result_frame["error"].notna()].copy() if not result_frame.empty else pd.DataFrame()
    summary = pd.DataFrame([{
        "scan_time_utc": scan_time,
        "assets_scanned": len(results),
        "data_ok": sum(not bool(r.get("error")) for r in results),
        "data_errors": sum(bool(r.get("error")) for r in results),
        "BUY": sum(r.get("signal") == "BUY" and not r.get("error") for r in results),
        "SELL": sum(r.get("signal") == "SELL" and not r.get("error") for r in results),
        "NO_SIGNAL": sum(r.get("signal") == "NO SIGNAL" and not r.get("error") for r in results),
    }])

    sheets: dict[str, pd.DataFrame] = {
        "Scan History": summary,
        "Scan Results": result_frame,
        "Data Errors": errors_frame,
    }
    if EXCEL_PATH.exists():
        try:
            with pd.ExcelFile(EXCEL_PATH, engine="openpyxl") as workbook:
                for sheet_name, new_frame in list(sheets.items()):
                    if sheet_name in workbook.sheet_names:
                        old = pd.read_excel(workbook, sheet_name=sheet_name)
                        sheets[sheet_name] = pd.concat([old, new_frame], ignore_index=True)
        except Exception:
            # If a previous workbook was interrupted/corrupted, write a fresh valid workbook.
            pass

    # Respect Excel's row limit while retaining the newest records.
    max_data_rows = 1_000_000
    for sheet_name, frame in list(sheets.items()):
        if len(frame) > max_data_rows:
            sheets[sheet_name] = frame.tail(max_data_rows).reset_index(drop=True)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="velora_", suffix=".xlsx", dir=DATA_DIR, delete=False) as tmp:
            temp_path = tmp.name
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    max_len = min(max((len(str(cell.value)) if cell.value is not None else 0) for cell in col_cells[:1000]) + 2, 55)
                    ws.column_dimensions[col_cells[0].column_letter].width = max(12, max_len)
        os.replace(temp_path, EXCEL_PATH)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def results_frame(results: list[dict[str, Any]], scan_time: str) -> pd.DataFrame:
    return pd.DataFrame([result_export_row(r, scan_time) for r in results])


# -------------------------------- UI -----------------------------------------
st.set_page_config(page_title="Velora Scanner", page_icon="📊", layout="wide")
st.title("📊 Velora — EMA 15 + Bollinger Bands")
st.caption("1-minute market scanner | BUY / SELL signals only | No order execution")
st.warning(
    "Crypto data is requested from public market-data APIs. Forex and stock data uses Yahoo Finance. "
    "Provider access and historical coverage vary. Yahoo prices are not Binomo OTC quotes. "
    "This app does not place trades and signals do not guarantee profit."
)

st.sidebar.header("Scanner settings")
category = st.sidebar.selectbox("Asset category", ["All", "Crypto", "Forex", "Other"], index=0)
lookback = st.sidebar.select_slider("Rise/fall candle count", options=[15, 16, 17], value=15)
band_mode = st.sidebar.radio("Bollinger confirmation", ["Touch or cross", "Close outside band"], index=0)
asset_limit = st.sidebar.number_input("Maximum assets per scan", min_value=1, max_value=len(ASSETS), value=10, step=1)
deep_sync = st.sidebar.checkbox("Sync toward 100,000 candles (slower; Binance only)", value=False)
show_no_signal = st.sidebar.checkbox("Show NO SIGNAL results", value=True)

if category == "All":
    category_assets = ASSETS.copy()
else:
    category_assets = [asset for asset in ASSETS if asset["category"] == category]
selected_assets = category_assets[: int(asset_limit)]
target_count = TARGET_CANDLES if deep_sync else RECENT_CANDLES

st.markdown("### Strategy rules")
st.markdown(f"""
- **SELL:** The last {lookback} completed 1-minute candles rose, at least half were green, the latest close is above EMA 15, and the upper Bollinger condition is met.
- **BUY:** The last {lookback} completed 1-minute candles fell, at least half were red, the latest close is below EMA 15, and the lower Bollinger condition is met.
- **NO SIGNAL:** Not all conditions are met, or reliable current data is unavailable.
- The current unfinished candle is excluded. The app never opens or closes trades.
""")
st.write(f"Assets selected: **{len(selected_assets)}**")
st.write(f"History target for this scan: **{target_count:,} candles per asset**")
st.caption(f"Automatic Excel file: `{EXCEL_PATH.name}` in the app's data directory")

if "results" not in st.session_state:
    st.session_state.results = []
if "scan_time" not in st.session_state:
    st.session_state.scan_time = None
if "excel_save_message" not in st.session_state:
    st.session_state.excel_save_message = None

if st.button("🔍 START SCAN", type="primary", use_container_width=True):
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()
    max_workers = min(3, max(1, len(selected_assets)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(scan_asset, asset, int(lookback), band_mode, target_count): asset for asset in selected_assets}
        for i, future in enumerate(as_completed(futures), start=1):
            asset = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    **asset, "signal": "NO SIGNAL", "reason": "Unexpected scan error.", "trend": "Unknown",
                    "change_pct": np.nan, "green_ratio": np.nan, "red_ratio": np.nan,
                    "price": np.nan, "ema15": np.nan, "bb_upper": np.nan, "bb_mid": np.nan,
                    "bb_lower": np.nan, "count": 0, "last_candle_utc": None,
                    "source": "Unavailable", "candles": None, "error": str(exc), "provider_warning": None,
                })
            progress.progress(i / max(1, len(futures)))
            status.write(f"Scanned {i} / {len(futures)} assets")
    # Keep original asset-list order in the display and Excel rows.
    order = {a["symbol"]: i for i, a in enumerate(selected_assets)}
    results.sort(key=lambda r: order.get(r.get("symbol"), 9999))
    scan_time = utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state.results = results
    st.session_state.scan_time = scan_time
    try:
        append_excel_history(results, scan_time)
        st.session_state.excel_save_message = f"Excel history saved automatically to {EXCEL_PATH.name}."
    except Exception as exc:
        st.session_state.excel_save_message = f"Excel auto-save failed: {exc}"
    progress.empty()
    status.empty()

results = st.session_state.results
scan_time = st.session_state.scan_time
if not results:
    st.info("Click START SCAN to retrieve market data and calculate signals.")
    if EXCEL_PATH.exists():
        try:
            st.download_button(
                "📥 Download Excel history",
                data=EXCEL_PATH.read_bytes(),
                file_name="velora_scan_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Excel file exists but could not be read for download: {exc}")
    st.stop()

if st.session_state.excel_save_message:
    if st.session_state.excel_save_message.startswith("Excel history saved"):
        st.success(st.session_state.excel_save_message)
    else:
        st.error(st.session_state.excel_save_message)

valid = [r for r in results if r.get("candles") is not None and not r.get("error")]
errors = [r for r in results if r.get("error")]
buys = [r for r in valid if r.get("signal") == "BUY"]
sells = [r for r in valid if r.get("signal") == "SELL"]
st.caption(f"Last scan: {scan_time or 'Unknown'}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets with 100K candles", sum(int(r.get("count", 0)) >= TARGET_CANDLES for r in valid))
c2.metric("BUY", len(buys))
c3.metric("SELL", len(sells))
c4.metric("Data errors", len(errors))

st.subheader("🚨 BUY / SELL signals")
signals = buys + sells
if signals:
    signal_df = pd.DataFrame([{
        "Asset": r.get("name"), "Symbol": r.get("symbol"), "Signal": r.get("signal"),
        "Price": r.get("price"), "Trend": r.get("trend"), "Change (%)": r.get("change_pct"),
        "EMA 15": r.get("ema15"), "BB upper": r.get("bb_upper"), "BB lower": r.get("bb_lower"),
        "Candles": r.get("count"), "Last candle (UTC)": r.get("last_candle_utc"), "Reason": r.get("reason"),
    } for r in signals])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
else:
    st.info("No BUY or SELL setup currently meets all selected conditions. A signal is not forced when the rules are not met.")

st.subheader("📋 All scan results")
shown = valid if show_no_signal else [r for r in valid if r.get("signal") != "NO SIGNAL"]
if shown:
    st.dataframe(pd.DataFrame([{
        "Asset": r.get("name"), "Category": r.get("category"), "Symbol": r.get("symbol"),
        "Signal": r.get("signal"), "Price": r.get("price"), "Trend": r.get("trend"),
        "Candles": r.get("count"), "100K progress (%)": round(min(100.0, r.get("count", 0) / TARGET_CANDLES * 100), 2),
        "Source": r.get("source"), "Last candle (UTC)": r.get("last_candle_utc"), "Reason": r.get("reason"),
    } for r in shown]), use_container_width=True, hide_index=True)
else:
    st.info("No valid scan results to display. Open the Data errors section for the provider messages.")

st.subheader("📈 Asset chart")
chartable = [r for r in valid if isinstance(r.get("candles"), pd.DataFrame) and not r["candles"].empty]
if chartable:
    chosen_name = st.selectbox("Choose an asset", [r["name"] for r in chartable])
    chosen = next(r for r in chartable if r["name"] == chosen_name)
    chart = chosen["candles"].tail(500).set_index("time")
    st.write(f"**Signal:** {chosen['signal']} — {chosen['reason']}")
    st.write(f"**Available candles:** {chosen['count']:,} / {TARGET_CANDLES:,} | **Source:** {chosen['source']}")
    if chosen.get("provider_warning"):
        st.warning(f"Provider note: {chosen['provider_warning']}")
    st.markdown("**Close price and EMA 15**")
    st.line_chart(chart[["close", "ema15"]].rename(columns={"close": "Close", "ema15": "EMA 15"}))
    st.markdown("**Bollinger Bands**")
    st.line_chart(chart[["close", "bb_upper", "bb_mid", "bb_lower"]].rename(columns={
        "close": "Close", "bb_upper": "Upper band", "bb_mid": "Middle band", "bb_lower": "Lower band"
    }))
else:
    st.info("No chart available because no valid current candle data could be retrieved.")

st.subheader("⚠️ Data errors")
if errors:
    st.dataframe(pd.DataFrame([{
        "Asset": r.get("name"), "Symbol": r.get("symbol"), "Provider": r.get("provider"), "Error": r.get("error")
    } for r in errors]), use_container_width=True, hide_index=True)
else:
    st.success("No data errors in this scan.")

st.subheader("💾 Automatic Excel history")
if EXCEL_PATH.exists():
    try:
        xlsx_bytes = EXCEL_PATH.read_bytes()
        st.download_button(
            "📥 Download Excel history (all scans)",
            data=xlsx_bytes,
            file_name="velora_scan_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Workbook sheets: Scan History, Scan Results, Data Errors. The file is saved automatically after each completed scan.")
    except Exception as exc:
        st.warning(f"Could not read the saved Excel workbook: {exc}")
else:
    st.info("The Excel workbook will be created automatically after the first scan.")

export = results_frame(results, scan_time or "Unknown")
st.download_button(
    "📄 Download current scan as CSV",
    data=export.to_csv(index=False).encode("utf-8-sig"),
    file_name="velora_current_scan.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()
st.caption(
    "Velora provides analysis signals only; it does not execute trades. Public data coverage depends on the provider. "
    "100,000 candles are a target, not a guarantee. A full historical sync can take a long time and may be blocked by providers. "
    "On hosted platforms, local files may be deleted when the app restarts or is redeployed; download the Excel workbook to keep a durable copy."
)
