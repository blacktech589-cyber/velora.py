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
BINANCE_BASES = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"]
YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]

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
SESSION.headers.update({"User-Agent": "Mozilla/5.0 VeloraScanner/1.0", "Accept": "application/json"})


def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL, candle_time TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL DEFAULT 0,
            PRIMARY KEY(symbol, candle_time))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol, candle_time)")
        conn.commit()


def load_cache(symbol):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            "SELECT candle_time AS time, open, high, low, close, volume FROM candles WHERE symbol=? ORDER BY candle_time",
            conn, params=(symbol,))
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
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        vol = getattr(r, "volume", 0)
        rows.append((symbol, ts.strftime("%Y-%m-%d %H:%M:%S"), float(r.open), float(r.high), float(r.low), float(r.close), 0.0 if pd.isna(vol) else float(vol)))
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany("INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()


def fetch_binance(symbol):
    """Fetch the most recent 1,000 completed 1-minute spot candles without API credentials."""
    errors = []
    for base in BINANCE_BASES:
        try:
            response = SESSION.get(
                base + "/api/v3/klines",
                params={"symbol": symbol, "interval": "1m", "limit": 1000},
                timeout=TIMEOUT,
            )
            if response.status_code == 451:
                raise RuntimeError("HTTP 451: Binance API unavailable from this network/region.")
            if response.status_code == 429:
                raise RuntimeError("HTTP 429: Binance rate limit; wait and retry.")
            if response.status_code in (401, 403):
                raise RuntimeError(f"HTTP {response.status_code}: access denied by provider/network.")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"No Binance candles returned: {str(payload)[:250]}")
            df = pd.DataFrame(payload, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
            df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
            # A candle is complete only after its close time; remove the currently forming minute.
            current_minute = pd.Timestamp.now(tz="UTC").floor("min")
            df = df[df["time"] < current_minute].drop_duplicates("time").sort_values("time").reset_index(drop=True)
            if df.empty:
                raise RuntimeError("No completed Binance candles available.")
            return df, base
        except Exception as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("Binance endpoints failed. " + " | ".join(errors))


def fetch_yahoo(symbol):
    errors = []
    encoded = requests.utils.quote(symbol, safe="")
    for host in YAHOO_HOSTS:
        try:
            response = SESSION.get(
                f"https://{host}/v8/finance/chart/{encoded}",
                params={"range": "7d", "interval": "1m", "includePrePost": "false"},
                timeout=TIMEOUT,
            )
            if response.status_code == 429:
                raise RuntimeError("HTTP 429: Yahoo rate limit; wait and retry.")
            if response.status_code in (401, 403):
                raise RuntimeError(f"HTTP {response.status_code}: Yahoo access denied.")
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            results = chart.get("result") or []
            if not results:
                raise RuntimeError("Yahoo returned no chart data for this symbol.")
            item = results[0]
            stamps = item.get("timestamp") or []
            quotes = item.get("indicators", {}).get("quote") or []
            if not stamps or not quotes:
                raise RuntimeError("Yahoo response is missing timestamps or OHLC values.")
            q = quotes[0]
            df = pd.DataFrame({
                "time": pd.to_datetime(stamps, unit="s", utc=True),
                "open": q.get("open", []), "high": q.get("high", []),
                "low": q.get("low", []), "close": q.get("close", []),
                "volume": q.get("volume", []),
            })
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = df["volume"].fillna(0)
            df = df.dropna(subset=["time", "open", "high", "low", "close"])
            current_minute = pd.Timestamp.now(tz="UTC").floor("min")
            df = df[df["time"] < current_minute].drop_duplicates("time").sort_values("time").reset_index(drop=True)
            if df.empty:
                raise RuntimeError("No completed Yahoo one-minute candles returned.")
            return df, host
        except Exception as exc:
            errors.append(f"{host}: {exc}")
    raise RuntimeError("Yahoo endpoints failed. " + " | ".join(errors))


def get_candles(asset):
    symbol, provider = asset["symbol"], asset["provider"]
    recent, host, live_error = pd.DataFrame(), None, None
    try:
        if provider == "binance":
            recent, host = fetch_binance(symbol)
        else:
            recent, host = fetch_yahoo(symbol)
        save_cache(symbol, recent)
    except Exception as exc:
        live_error = str(exc)
    cached = load_cache(symbol)
    frames = [x for x in (cached, recent) if x is not None and not x.empty]
    if not frames:
        raise RuntimeError(live_error or "No live or cached candle data available.")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates("time", keep="last").sort_values("time").tail(TARGET_CANDLES).reset_index(drop=True)
    current_minute = pd.Timestamp.now(tz="UTC").floor("min")
    df = df[df["time"] < current_minute].reset_index(drop=True)
    if len(df) < 40:
        raise RuntimeError(f"Only {len(df)} completed candles available; need at least 40. " + (live_error or ""))
    source = f"{provider.upper()} ({host}) + SQLite cache" if host else "SQLite cache only; live provider unavailable"
    return df, source, live_error


def add_indicators(df):
    out = df.copy()
    out["ema15"] = out["close"].ewm(span=15, adjust=False, min_periods=15).mean()
    out["bb_mid"] = out["close"].rolling(20, min_periods=20).mean()
    sd = out["close"].rolling(20, min_periods=20).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + 2 * sd
    out["bb_lower"] = out["bb_mid"] - 2 * sd
    return out


def calculate_signal(df, lookback, band_mode):
    no_signal = {"signal": "NO SIGNAL", "reason": "Not enough valid data.", "change_pct": np.nan, "trend": "Unknown"}
    if df is None or len(df) < max(lookback + 25, 40):
        return no_signal
    data = add_indicators(df)
    last = data.iloc[-1]
    start = data.iloc[-(lookback + 1)]
    if pd.isna(last["ema15"]) or pd.isna(last["bb_upper"]) or pd.isna(last["bb_lower"]):
        return {**no_signal, "reason": "Indicators are not ready."}
    start_price, price = float(start["close"]), float(last["close"])
    if start_price == 0:
        return {**no_signal, "reason": "Invalid starting price."}
    change = (price - start_price) / start_price * 100
    segment = data.iloc[-(lookback + 1):-1]
    green_ratio = float((segment["close"] > segment["open"]).mean())
    red_ratio = float((segment["close"] < segment["open"]).mean())
    rising = change > 0 and green_ratio >= 0.5
    falling = change < 0 and red_ratio >= 0.5
    if band_mode == "Touch or cross":
        upper_ok = float(last["high"]) >= float(last["bb_upper"])
        lower_ok = float(last["low"]) <= float(last["bb_lower"])
    else:
        upper_ok = price >= float(last["bb_upper"])
        lower_ok = price <= float(last["bb_lower"])
    above_ema, below_ema = price > float(last["ema15"]), price < float(last["ema15"])
    if rising and above_ema and upper_ok:
        signal = "SELL"
        reason = f"Last {lookback} candles rose {change:.3f}%; close above EMA 15; upper Bollinger condition met."
    elif falling and below_ema and lower_ok:
        signal = "BUY"
        reason = f"Last {lookback} candles fell {change:.3f}%; close below EMA 15; lower Bollinger condition met."
    else:
        signal = "NO SIGNAL"
        reason = f"Conditions incomplete; change {change:.3f}%, green ratio {green_ratio:.0%}, red ratio {red_ratio:.0%}."
    trend = "Rising" if rising else "Falling" if falling else "Mixed / sideways"
    return {"signal": signal, "reason": reason, "change_pct": change, "trend": trend}


def scan_asset(asset, lookback, band_mode):
    try:
        candles, source, warning = get_candles(asset)
        data = add_indicators(candles)
        signal = calculate_signal(data, lookback, band_mode)
        last = data.iloc[-1]
        return {**asset, **signal, "price": float(last["close"]), "ema15": float(last["ema15"]),
                "bb_upper": float(last["bb_upper"]), "bb_lower": float(last["bb_lower"]),
                "count": len(data), "source": source, "candles": data, "error": None,
                "provider_warning": warning}
    except Exception as exc:
        return {**asset, "signal": "NO SIGNAL", "reason": "Data unavailable; signal not calculated.",
                "change_pct": np.nan, "trend": "Unknown", "price": np.nan, "ema15": np.nan,
                "bb_upper": np.nan, "bb_lower": np.nan, "count": 0, "source": "Unavailable",
                "candles": None, "error": str(exc), "provider_warning": None}


st.title("📊 Velora — EMA 15 + Bollinger Bands")
st.caption("1-minute market scanner | BUY / SELL signals only | No order execution")
st.warning("Crypto assets use Binance public Spot market data without API keys. Forex and stock assets use Yahoo Finance. Provider access and historical availability can vary. Yahoo prices are not Binomo OTC quotes.")
st.info("100,000 one-minute candles are a target, not a guarantee. Binance's public endpoint returns up to 1,000 recent candles per request; SQLite can accumulate data across scans, but cannot recreate missing history. Keep the SQLite file if you want to preserve cache.")

st.sidebar.header("Scanner settings")
category = st.sidebar.selectbox("Asset category", ["All", "Crypto", "Forex", "Other"])
lookback = st.sidebar.select_slider("Rise/fall candle count", options=[15, 16, 17], value=15)
band_mode = st.sidebar.radio("Bollinger confirmation", ["Touch or cross", "Close outside band"])
asset_limit = st.sidebar.number_input("Maximum assets per scan", min_value=1, max_value=len(ASSETS), value=10, step=1)
show_no_signal = st.sidebar.checkbox("Show NO SIGNAL results", value=True)
if category == "All":
    selected_assets = ASSETS.copy()
else:
    selected_assets = [a for a in ASSETS if a["category"] == category]
selected_assets = selected_assets[:int(asset_limit)]

st.markdown("### Strategy rules")
st.markdown(f"""
- **SELL:** Last {lookback} completed candles rose, at least half of the period candles were green, latest close is above EMA 15, and the upper Bollinger condition is met.
- **BUY:** Last {lookback} completed candles fell, at least half of the period candles were red, latest close is below EMA 15, and the lower Bollinger condition is met.
- **NO SIGNAL:** Conditions are not all met.
- Current unfinished one-minute candle is excluded. This app never places trades.
""")
st.write(f"Assets selected: **{len(selected_assets)}**")
st.write(f"History target: **{TARGET_CANDLES:,} candles per asset**")

if "results" not in st.session_state:
    st.session_state.results = []
if "scan_time" not in st.session_state:
    st.session_state.scan_time = None

if st.button("🔍 START SCAN", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0)
    status = st.empty()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(scan_asset, asset, lookback, band_mode): asset for asset in selected_assets}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                asset = futures[future]
                results.append({**asset, "signal": "NO SIGNAL", "reason": "Scanner error", "error": str(exc), "candles": None, "count": 0})
            progress.progress(i / len(futures))
            status.write(f"Scanned {i} / {len(futures)} assets")
    st.session_state.results = results
    st.session_state.scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    progress.empty()
    status.empty()

results = st.session_state.results
if not results:
    st.info("Click START SCAN to retrieve market data and calculate signals.")
    st.stop()

valid = [r for r in results if r.get("candles") is not None and not r.get("error")]
errors = [r for r in results if r.get("error")]
buys = [r for r in valid if r["signal"] == "BUY"]
sells = [r for r in valid if r["signal"] == "SELL"]
st.caption(f"Last scan: {st.session_state.scan_time or 'Unknown'}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets with 100K candles", sum(r["count"] >= TARGET_CANDLES for r in valid))
c2.metric("BUY", len(buys))
c3.metric("SELL", len(sells))
c4.metric("Data errors", len(errors))

st.subheader("🚨 BUY / SELL signals")
signals = buys + sells
if signals:
    st.dataframe(pd.DataFrame([{"Asset": r["name"], "Symbol": r["symbol"], "Signal": r["signal"], "Price": r["price"], "Trend": r["trend"], "Change (%)": r["change_pct"], "EMA 15": r["ema15"], "BB upper": r["bb_upper"], "BB lower": r["bb_lower"], "Candles": r["count"], "Reason": r["reason"]} for r in signals]), use_container_width=True, hide_index=True)
else:
    st.info("No BUY or SELL setup currently meets all selected conditions.")

st.subheader("📋 All scan results")
shown = valid if show_no_signal else [r for r in valid if r["signal"] != "NO SIGNAL"]
if shown:
    st.dataframe(pd.DataFrame([{"Asset": r["name"], "Category": r["category"], "Symbol": r["symbol"], "Signal": r["signal"], "Price": r["price"], "Trend": r["trend"], "Candles": r["count"], "100K progress (%)": round(min(100, r["count"] / TARGET_CANDLES * 100), 2), "Source": r["source"], "Reason": r["reason"]} for r in shown]), use_container_width=True, hide_index=True)

st.subheader("📈 Asset chart")
chartable = [r for r in valid if r.get("candles") is not None and not r["candles"].empty]
if chartable:
    name = st.selectbox("Choose an asset", [r["name"] for r in chartable])
    selected = next(r for r in chartable if r["name"] == name)
    chart = selected["candles"].tail(500).set_index("time")
    st.write(f"**Signal:** {selected['signal']} — {selected['reason']}")
    st.write(f"**Available candles:** {selected['count']:,} / {TARGET_CANDLES:,} | **Source:** {selected['source']}")
    st.markdown("**Close price and EMA 15**")
    st.line_chart(chart[["close", "ema15"]].rename(columns={"close": "Close", "ema15": "EMA 15"}))
    st.markdown("**Bollinger Bands**")
    st.line_chart(chart[["close", "bb_upper", "bb_mid", "bb_lower"]].rename(columns={"close": "Close", "bb_upper": "Upper band", "bb_mid": "Middle band", "bb_lower": "Lower band"}))
else:
    st.info("No chart available because no candle data could be retrieved.")

st.subheader("⚠️ Data errors")
if errors:
    st.error("Open the Error column below. It shows the provider response that prevented data loading.")
    st.dataframe(pd.DataFrame([{"Asset": r.get("name"), "Symbol": r.get("symbol"), "Provider": r.get("provider"), "Error": r.get("error")} for r in errors]), use_container_width=True, hide_index=True)
else:
    st.success("No data errors in this scan.")

if results:
    export = pd.DataFrame([{"asset": r.get("name"), "symbol": r.get("symbol"), "category": r.get("category"), "signal": r.get("signal"), "price": r.get("price"), "trend": r.get("trend"), "change_pct": r.get("change_pct"), "ema15": r.get("ema15"), "bb_upper": r.get("bb_upper"), "bb_lower": r.get("bb_lower"), "candle_count": r.get("count"), "source": r.get("source"), "reason": r.get("reason"), "error": r.get("error")} for r in results])
    st.download_button("📥 Download CSV results", export.to_csv(index=False).encode("utf-8-sig"), file_name="velora_scan_results.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("Velora provides analysis signals only and does not execute trades. Signals do not guarantee profit. Public market data may be delayed, restricted, or incomplete.")
