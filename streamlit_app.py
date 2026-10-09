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
                host + path, params=params, timeout=TIMEOUT
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
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


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
        raise RuntimeError("Aktif Binance Spot USDT piyasası bulunamadı.")

    return markets, host


def binance_tickers():
    data, host = binance_json("/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        try:
            tickers[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h_pct": float(item["priceChangePercent"]),
                "trades_24h": int(item["count"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers, host


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
        raise RuntimeError("Aktif Gate.io Spot USDT piyasası bulunamadı.")

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
                "change_24h_pct": float(item["change_percentage"]),
                "trades_24h": np.nan,
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


def discover(exchange):
    if exchange == "Binance only":
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()
        return markets, tickers, "Binance", f"{host1}; ticker={host2}"

    if exchange == "Gate.io only":
        return gate_markets(), gate_tickers(), "Gate.io", GATE

    try:
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()
        return markets, tickers, "Binance", f"{host1}; ticker={host2}"
    except Exception as exc:
        markets = gate_markets()
        tickers = gate_tickers()
        return (
            markets,
            tickers,
            "Gate.io",
            f"Gate.io fallback; Binance: {str(exc)[:150]}",
        )


def clean_candles(rows):
    cols = [
        "timestamp", "open", "high", "low",
        "close", "volume", "quote_volume",
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=cols[:6])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS
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
                    "timestamp": int(candle[0]) // 1000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "quote_volume": float(candle[7]),
                })
            except (ValueError, TypeError, IndexError):
                continue

        next_cursor = int(data[-1][0]) + INTERVAL_SECONDS * 1000
        if next_cursor <= cursor:
            break

        cursor = next_cursor
        if len(data) < 1000:
            break

        time.sleep(0.04)

    df = clean_candles(rows)
    if not df.empty:
        cutoff = current_start - BARS_90_DAYS * INTERVAL_SECONDS
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    return df


def gate_candles(market_id):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    cursor = (
        current_start - (BARS_90_DAYS + 10) * INTERVAL_SECONDS
    )
    end = current_start
    rows = []

    while cursor < end:
        page_end = min(cursor + 999 * INTERVAL_SECONDS, end)

        data = get_json(
            f"{GATE}/spot/candlesticks",
            {
                "currency_pair": market_id,
                "interval": "15m",
                "from": cursor,
                "to": page_end,
                "limit": 1000,
            },
        )

        if not data:
            break

        for candle in data:
            if len(candle) < 6:
                continue

            try:
                rows.append({
                    "timestamp": int(candle[0]),
                    "quote_volume": float(candle[1]),
                    "close": float(candle[2]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "open": float(candle[5]),
                    "volume": float(candle[6]) if len(candle) > 6 else 0.0,
                })
            except (ValueError, TypeError, IndexError):
                continue

        cursor = page_end + INTERVAL_SECONDS
        time.sleep(0.04)

    df = clean_candles(rows)
    if not df.empty:
        cutoff = current_start - BARS_90_DAYS * INTERVAL_SECONDS
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    return df


def load_candles(market):
    if market["provider"] == "Binance":
        return binance_candles(market["market_id"])
    return gate_candles(market["market_id"])


def rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def analyze(df, max_distance, min_volatility, crash_threshold=90.0):
    # Yeni listelenen aktif coinler için mevcut geçmişi de analiz et.
    if len(df) < 20:
        return {
            "valid": False,
            "reason": (
                f"Yalnızca {len(df)} tamamlanmış mum var. "
                "Analiz için en az 20 mum gerekli."
            ),
        }

    period = df.tail(BARS_90_DAYS).reset_index(drop=True)
    coverage_days = len(period) * 15 / 60 / 24

    peak_pos = int(period["high"].to_numpy().argmax())
    low_pos = int(period["low"].to_numpy().argmin())

    peak_price = float(period.iloc[peak_pos]["high"])
    low_price = float(period.iloc[low_pos]["low"])
    current_price = float(period.iloc[-1]["close"])

    if min(peak_price, low_price, current_price) <= 0:
        return {"valid": False, "reason": "Geçersiz fiyat verisi."}

    drop_from_peak_pct = (
        (peak_price - current_price) / peak_price * 100
    )
    distance_from_low_pct = (
        (current_price / low_price - 1) * 100
    )

    last3 = period.tail(3)
    closes = last3["close"].to_numpy()
    opens = last3["open"].to_numpy()

    rising3 = bool(closes[0] < closes[1] < closes[2])
    green_last = bool(closes[-1] > opens[-1])
    near_low = bool(0 <= distance_from_low_pct <= max_distance)
    crashed_enough = bool(drop_from_peak_pct >= crash_threshold)

    period["ret"] = period["close"].pct_change()
    period["atr"] = atr(period)
    period["rsi"] = rsi(period["close"])

    volume_24h = float(
        period["quote_volume"].tail(96).fillna(0).sum()
    )

    returns_24h = period["ret"].tail(96).dropna()
    volatility_pct = (
        float(returns_24h.std(ddof=1) * np.sqrt(96) * 100)
        if len(returns_24h) >= 48 else np.nan
    )

    atr_pct = (
        float(period["atr"].iloc[-1] / current_price * 100)
        if pd.notna(period["atr"].iloc[-1]) else np.nan
    )

    rsi_value = (
        float(period["rsi"].iloc[-1])
        if pd.notna(period["rsi"].iloc[-1]) else np.nan
    )

    volume_score = min(
        100.0, max(0.0, np.log10(max(volume_24h, 1.0)) * 10)
    )

    volatility_score = (
        min(100.0, max(0.0, volatility_pct * 5))
        if pd.notna(volatility_pct) else 0.0
    )

    activity_score = (volume_score + volatility_score) / 2

    hyperactive = bool(
        volume_24h > 0
        and pd.notna(volatility_pct)
        and volatility_pct >= min_volatility
    )

    crash_bottom_candidate = bool(
        crashed_enough and near_low and hyperactive
    )

    confirmed_reversal = bool(
        crash_bottom_candidate and rising3 and green_last
    )

    if confirmed_reversal:
        signal = "90% CRASH / LOW-AREA REVERSAL"
    elif crash_bottom_candidate and rising3:
        signal = "90% CRASH / POSSIBLE BOTTOM"
    elif crashed_enough and near_low:
        signal = "90% CRASH / NEAR 90-DAY LOW"
    elif crashed_enough:
        signal = "90% CRASH / NOT NEAR LOW"
    else:
        signal = "CRASH THRESHOLD NOT MET"

    peak_time = datetime.fromtimestamp(
        int(period.iloc[peak_pos]["timestamp"]), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    low_time = datetime.fromtimestamp(
        int(period.iloc[low_pos]["timestamp"]), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "valid": True,
        "current_close_15m": current_price,
        "90d_peak": peak_price,
        "90d_peak_time_utc": peak_time,
        "90d_low": low_price,
        "90d_low_time_utc": low_time,
        "drop_from_90d_peak_pct": drop_from_peak_pct,
        "distance_from_90d_low_pct": distance_from_low_pct,
        "crashed_90pct": crashed_enough,
        "near_90d_low": near_low,
        "three_rising_candles": rising3,
        "latest_candle_green": green_last,
        "volume_24h_candles_usdt": volume_24h,
        "realized_volatility_24h_pct": volatility_pct,
        "atr_pct": atr_pct,
        "rsi_14": rsi_value,
        "volume_score": volume_score,
        "volatility_score": volatility_score,
        "activity_score": activity_score,
        "high_activity": hyperactive,
        "crash_bottom_candidate": crash_bottom_candidate,
        "confirmed_reversal": confirmed_reversal,
        "signal": signal,
        "candles_analyzed": len(period),
        "history_days_analyzed": coverage_days,
        "full_90d_history": bool(len(period) >= BARS_90_DAYS),
    }


def scan_market(market, ticker, max_distance, min_volatility, crash_threshold):
    df = load_candles(market)
    analysis = analyze(df, max_distance, min_volatility, crash_threshold)

    if not analysis.get("valid"):
        raise RuntimeError(analysis.get("reason", "Analiz başarısız."))

    candle_price = float(analysis["current_close_15m"])
    ticker_price = float(ticker["price"])
    price_diff = abs(ticker_price - candle_price) / candle_price * 100

    return {
        "symbol": market["symbol"],
        "market_id":
