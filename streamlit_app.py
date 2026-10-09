import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Hyperactive Spot Scanner", page_icon="⚡", layout="wide")

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
BARS_30_DAYS = 30 * 24 * 4
TIMEOUT = 18

session = requests.Session()
session.headers.update({"User-Agent": "HyperactiveSpotScanner/1.0"})


def binance_json(path, params=None):
    errors = []
    for host in BINANCE_HOSTS:
        try:
            response = session.get(host + path, params=params, timeout=TIMEOUT)
            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue
            response.raise_for_status()
            return response.json(), host
        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:100]}")
    raise RuntimeError("Binance endpoints unavailable. " + " | ".join(errors))


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
                "symbol": symbol, "market_id": symbol,
                "base_asset": item.get("baseAsset", ""), "provider": "Binance"
            })
    if not markets:
        raise RuntimeError("No active Binance Spot USDT markets.")
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
            pass
    return tickers, host


def gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")
    markets = []
    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")
        if not market_id or not base or item.get("quote") != "USDT":
            continue
        if str(item.get("trade_status", "")).lower() != "tradable":
            continue
        if item.get("delisted") is True:
            continue
        markets.append({
            "symbol": base + "USDT", "market_id": market_id,
            "base_asset": base, "provider": "Gate.io"
        })
    if not markets:
        raise RuntimeError("No tradable Gate.io Spot USDT markets.")
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
            pass
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
        markets, tickers = gate_markets(), gate_tickers()
        return markets, tickers, "Gate.io", f"Gate.io fallback; Binance: {str(exc)[:150]}"


def clean_candles(rows):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=cols[:6]).drop_duplicates("timestamp").sort_values("timestamp")
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS
    df = df[df.timestamp < current_start]
    df = df[
        (df.open > 0) & (df.high >= df.low) &
        (df.high >= df.open) & (df.high >= df.close) &
        (df.low <= df.open) & (df.low <= df.close) &
        (df.close > 0)
    ]
    return df.reset_index(drop=True)


def binance_candles(symbol):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS
    start_ms = (current_start - (BARS_30_DAYS + 10) * INTERVAL_SECONDS) * 1000
    end_ms = current_start * 1000
    cursor = start_ms
    rows = []
    while cursor < end_ms:
        data, _ = binance_json("/api/v3/klines", {
            "symbol": symbol, "interval": "15m",
            "startTime": cursor, "endTime": end_ms, "limit": 1000
        })
        if not data:
            break
        for c in data:
            try:
                rows.append({
                    "timestamp": int(c[0]) // 1000,
                    "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]), "quote_volume": float(c[7])
                })
            except (ValueError, TypeError, IndexError):
                pass
        nxt = int(data[-1][0]) + INTERVAL_SECONDS * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        if len(data) < 1000:
            break
        time.sleep(0.04)
    return clean_candles(rows)


def gate_candles(market_id):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS
    cursor = current_start - (BARS_30_DAYS + 10) * INTERVAL_SECONDS
    end = current_start
    rows = []
    while cursor < end:
        page_end = min(cursor + 999 * INTERVAL_SECONDS, end)
        data = get_json(f"{GATE}/spot/candlesticks", {
            "currency_pair": market_id, "interval": "15m",
            "from": cursor, "to": page_end, "limit": 1000
        })
        if not data:
            break
        for c in data:
            if len(c) < 6:
                continue
            try:
                rows.append({
                    "timestamp": int(c[0]), "quote_volume": float(c[1]),
                    "close": float(c[2]), "high": float(c[3]),
                    "low": float(c[4]), "open": float(c[5]),
                    "volume": float(c[6]) if len(c) > 6 else 0.0
                })
            except (ValueError, TypeError, IndexError):
                pass
        cursor = page_end + INTERVAL_SECONDS
        time.sleep(0.04)
    return clean_candles(rows)


def load_candles(market):
    if market["provider"] == "Binance":
        return binance_candles(market["market_id"])
    return gate_candles(market["market_id"])


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, period=14):
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyze(df, max_distance, min_volatility):
    if len(df) < BARS_30_DAYS:
        return {"valid": False, "reason": f"Only {len(df)} completed candles; need {BARS_30_DAYS}."}
    month = df.tail(BARS_30_DAYS).reset_index(drop=True)
    bottom_pos = int(month.low.to_numpy().argmin())
    bottom = float(month.iloc[bottom_pos].low)
    price = float(month.iloc[-1].close)
    if bottom <= 0 or price <= 0:
        return {"valid": False, "reason": "Invalid prices."}
    distance = (price / bottom - 1) * 100
    last3 = month.tail(3)
    closes = last3.close.to_numpy()
    opens = last3.open.to_numpy()
    rising3 = bool(closes[0] < closes[1] < closes[2])
    green_last = bool(closes[-1] > opens[-1])
    bottom_before = bottom_pos < len(month) - 3
    near = bool(0 <= distance <= max_distance)

    month["ret"] = month.close.pct_change()
    month["atr"] = atr(month)
    month["rsi"] = rsi(month.close)
    atr_pct = float(month.atr.iloc[-1] / price * 100) if pd.notna(month.atr.iloc[-1]) else np.nan
    volume24 = float(month.quote_volume.tail(96).fillna(0).sum())
    rets = month.ret.tail(96).dropna()
    vol24 = float(rets.std(ddof=1) * np.sqrt(96) * 100) if len(rets) >= 48 else np.nan
    vol_score = min(100, max(0, vol24 * 5)) if pd.notna(vol24) else 0
    volume_score = min(100, max(0, np.log10(max(volume24, 1)) * 10))
    activity = (vol_score + volume_score) / 2
    hyperactive = bool(volume24 > 0 and pd.notna(vol24) and vol24 >= min_volatility)
    reversal = bool(near and bottom_before and rising3 and green_last and hyperactive)
    if reversal:
        signal = "HYPERACTIVE MONTHLY-LOW REVERSAL"
    elif near and hyperactive and rising3 and green_last:
        signal = "NEAR LOW / RISING"
    elif near and hyperactive:
        signal = "NEAR LOW / ACTIVE"
    elif rising3 and green_last:
        signal = "RISING MOMENTUM"
    else:
        signal = "OTHER"
    bottom_time = datetime.fromtimestamp(
        int(month.iloc[bottom_pos].timestamp), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "valid": True, "price_15m_close": price, "monthly_low": bottom,
        "monthly_low_time_utc": bottom_time, "distance_from_monthly_low_pct": distance,
        "three_rising_candles": rising3, "latest_candle_green": green_last,
        "near_monthly_low": near, "volume_24h_candles_usdt": volume24,
        "realized_volatility_24h_pct": vol24, "atr_pct": atr_pct,
        "rsi_14": float(month.rsi.iloc[-1]) if pd.notna(month.rsi.iloc[-1]) else np.nan,
        "volume_score": volume_score, "volatility_score": vol_score,
        "activity_score": activity, "high_activity": hyperactive,
        "monthly_low_reversal": reversal, "signal": signal, "candles_30d": len(month)
    }


def scan_market(market, ticker, max_distance, min_volatility):
    df = load_candles(market)
    a = analyze(df, max_distance, min_volatility)
    if not a.get("valid"):
        raise RuntimeError(a.get("reason", "Analysis failed."))
    candle_price = float(a["price_15m_close"])
    ticker_price = float(ticker["price"])
    diff = abs(ticker_price - candle_price) / candle_price * 100
    return {
        "symbol": market["symbol"], "market_id": market["market_id"],
        "provider": market["provider"], "ticker_price": ticker_price,
        "price_difference_pct": diff, "price_check_ok": diff <= 2,
        "ticker_quote_volume_24h_usdt": float(ticker["quote_volume"]),
        "change_24h_pct": float(ticker["change_24h_pct"]),
        "trades_24h": ticker.get("trades_24h", np.nan), **a
    }


# ------------------------------- UI -------------------------------

st.title("⚡ Hyperactive Spot Scanner")
st.write(
    "Finds high-volume, volatile Spot USDT markets near their 30-day low. "
    "A reversal candidate requires three consecutive rising completed "
    "15-minute closes and a green latest candle. No orders are placed."
)
st.warning("Volatility increases risk as well as opportunity. The activity score is not a probability of profit.")

with st.sidebar:
    exchange = st.selectbox("Exchange", [
        "Auto (Binance then Gate.io fallback)", "Binance only", "Gate.io only"
    ])
    max_distance = st.slider("Maximum distance from 30-day low (%)", 1.0, 10.0, 10.0, 0.5)
    min_volume = st.number_input("Minimum 24h volume (USDT)", min_value=0, value=100000, step=50000)
    min_volatility = st.number_input("Minimum 24h realized volatility (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
    scan_count = st.number_input("Markets to scan", min_value=1, max_value=2000, value=50, step=25)
    run = st.button("⚡ Start scan", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    progress = st.progress(0)
    status = st.empty()
    try:
        if exchange == "Binance only":
            markets, host1 = binance_markets()
            tickers, host2 = binance_tickers()
            provider, source = "Binance", f"{host1}; ticker={host2}"
        elif exchange == "Gate.io only":
            markets, tickers = gate_markets(), gate_tickers()
            provider, source = "Gate.io", GATE
        else:
            markets, tickers, provider, source = discover("Auto")

        candidates = []
        for market in markets:
            ticker = tickers.get(market["market_id"])
            if ticker is not None and float(ticker["quote_volume"]) >= min_volume:
                candidates.append({**market, **ticker})
        candidates.sort(key=lambda x: float(x["quote_volume"]), reverse=True)
        candidates = candidates[:int(scan_count)]
        if not candidates:
            raise RuntimeError("No markets passed the volume filter; lower it and retry.")

        st.info(f"Source: {provider} | Active markets: {len(markets)} | To scan: {len(candidates)} | API: {source}")
        results, errors = [], []
        for i, market in enumerate(candidates):
            status.write(f"{i+1}/{len(candidates)} — {market['symbol']} [{market['market_id']}]")
            try:
                results.append(scan_market(
                    market, tickers[market["market_id"]], max_distance, min_volatility
                ))
            except Exception as exc:
                errors.append({"symbol": market["symbol"], "market_id": market["market_id"], "provider": market["provider"], "error": str(exc)[:250]})
            progress.progress((i + 1) / len(candidates))
            time.sleep(0.05)

        st.session_state.results = pd.DataFrame(results)
        st.session_state.errors = pd.DataFrame(errors)
        st.session_state.provider = provider
        st.session_state.scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status.success(f"Done: {len(results)} analyzed, {len(errors)} errors.")
    except Exception as exc:
        st.error(f"Scan failed: {exc}")

results = st.session_state.get("results")
if results is not None:
    if results.empty:
        st.warning("No successful analyses. Inspect the error list or lower filters.")
    else:
        st.caption(f"Source: {st.session_state.get('provider', '-')} | {st.session_state.get('scan_time', '-')}")
        verified = results[results.price_check_ok == True].copy()
        hyper = verified[
            (verified.ticker_quote_volume_24h_usdt >= min_volume)
            & (verified.realized_volatility_24h_pct >= min_volatility)
            & (verified.high_activity == True)
        ].copy()
        reversals = hyper[hyper.monthly_low_reversal == True].sort_values("activity_score", ascending=False)
        rising = hyper[
            (hyper.near_monthly_low == True)
            & (hyper.three_rising_candles == True)
            & (hyper.latest_candle_green == True)
        ].sort_values("activity_score", ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzed", len(results))
        c2.metric("Price-verified", len(verified))
        c3.metric("Hyperactive", len(hyper))
        c4.metric("Monthly-low reversals", len(reversals))

        columns = [
            "symbol", "market_id", "provider", "price_15m_close", "ticker_price",
            "price_difference_pct", "monthly_low", "monthly_low_time_utc",
            "distance_from_monthly_low_pct", "ticker_quote_volume_24h_usdt",
            "realized_volatility_24h_pct", "atr_pct", "rsi_14", "trades_24h",
            "activity_score", "signal"
        ]

        st.subheader("🎯 Hyperactive monthly-low reversals")
        st.dataframe(reversals[columns], use_container_width=True, hide_index=True)
        st.download_button("Download reversals CSV", reversals.to_csv(index=False).encode("utf-8-sig"), "monthly_low_reversals.csv", "text/csv")

        st.subheader("📈 Rising near monthly low")
        st.dataframe(rising[columns], use_container_width=True, hide_index=True)
        st.download_button("Download rising CSV", rising.to_csv(index=False).encode("utf-8-sig"), "rising_near_monthly_low.csv", "text/csv")

        st.subheader("⚡ All hyperactive markets")
        st.dataframe(hyper.sort_values("activity_score", ascending=False)[columns], use_container_width=True, hide_index=True)
        st.download_button("Download hyperactive CSV", hyper.to_csv(index=False).encode("utf-8-sig"), "hyperactive_markets.csv", "text/csv")

        st.subheader("📋 All scan results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download all results CSV", results.to_csv(index=False).encode("utf-8-sig"), "all_scan_results.csv", "text/csv")

        errors = st.session_state.get("errors")
        if errors is not None and not errors.empty:
            with st.expander(f"Market errors ({len(errors)})"):
                st.dataframe(errors, use_container_width=True, hide_index=True)
                st.download_button("Download errors CSV", errors.to_csv(index=False).encode("utf-8-sig"), "scan_errors.csv", "text/csv")
else:
    st.info("Choose settings and click Start scan.")
