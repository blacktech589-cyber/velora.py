
import sqlite3
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VELORA — EMA 15 + BOLLINGER BANDS
# 1-minute scanner | BUY / SELL only | No order execution
# API key not required
# ============================================================

st.set_page_config(
    page_title="Velora — EMA15 Bollinger Scanner",
    page_icon="📊",
    layout="wide",
)

DB_PATH = "velora_history.sqlite3"
TARGET_CANDLES = 100_000
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
TIMEOUT = 20

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})

# Yahoo Finance symbols. These are market-data proxies,
# NOT actual Binomo OTC prices.
ASSETS = [
    {"name": "Bitcoin", "symbol": "BTC-USD", "category": "Crypto"},
    {"name": "Ethereum", "symbol": "ETH-USD", "category": "Crypto"},
    {"name": "Solana", "symbol": "SOL-USD", "category": "Crypto"},
    {"name": "Cardano", "symbol": "ADA-USD", "category": "Crypto"},
    {"name": "Chainlink", "symbol": "LINK-USD", "category": "Crypto"},
    {"name": "Bitcoin Cash", "symbol": "BCH-USD", "category": "Crypto"},
    {"name": "Avalanche", "symbol": "AVAX-USD", "category": "Crypto"},
    {"name": "Dogecoin", "symbol": "DOGE-USD", "category": "Crypto"},
    {"name": "XRP", "symbol": "XRP-USD", "category": "Crypto"},
    {"name": "Polkadot", "symbol": "DOT-USD", "category": "Crypto"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "category": "Forex"},
    {"name": "GBP/USD", "symbol": "GBPUSD=X", "category": "Forex"},
    {"name": "USD/JPY", "symbol": "JPY=X", "category": "Forex"},
    {"name": "USD/CHF", "symbol": "CHF=X", "category": "Forex"},
    {"name": "AUD/USD", "symbol": "AUDUSD=X", "category": "Forex"},
    {"name": "USD/CAD", "symbol": "CAD=X", "category": "Forex"},
    {"name": "NZD/USD", "symbol": "NZDUSD=X", "category": "Forex"},
    {"name": "EUR/GBP", "symbol": "EURGBP=X", "category": "Forex"},
    {"name": "Gold Futures (proxy)", "symbol": "GC=F", "category": "Other"},
    {"name": "Apple", "symbol": "AAPL", "category": "Other"},
    {"name": "Microsoft", "symbol": "MSFT", "category": "Other"},
    {"name": "NVIDIA", "symbol": "NVDA", "category": "Other"},
]


# --------------------------- DATABASE ---------------------------

def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                time TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0,
                PRIMARY KEY (symbol, time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
            ON candles(symbol, time)
        """)
        conn.commit()


def save_candles(symbol, df):
    if df is None or df.empty:
        return

    rows = []
    for row in df.itertuples(index=False):
        timestamp = pd.Timestamp(row.time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        rows.append((
            symbol,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        ))

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO candles
            (symbol, time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()


def load_candles(symbol):
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY time ASC
        """, conn, params=(symbol,))

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(
            subset=["time", "open", "high", "low", "close"]
        )

    return df


# --------------------------- DATA SOURCE ---------------------------

def fetch_yahoo_1m(symbol):
    """
    Fetch recent 1-minute candles without an API key.
    Yahoo's unofficial endpoint has history and symbol limitations.
    """

    url = f"{YAHOO_BASE}/{requests.utils.quote(symbol, safe='')}"

    response = SESSION.get(
        url,
        params={
            "range": "7d",
            "interval": "1m",
            "includePrePost": "false",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()
    chart = payload.get("chart", {})

    if chart.get("error"):
        message = chart["error"].get("description", "Yahoo data error")
        raise RuntimeError(message)

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("No data returned for this symbol.")

    item = results[0]
    timestamps = item.get("timestamp") or []

    quotes = item.get("indicators", {}).get("quote") or []
    if not timestamps or not quotes:
        raise RuntimeError("No candle data in provider response.")

    quote = quotes[0]

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    # Exclude the current, unfinished minute candle.
    current_minute = pd.Timestamp.now(tz="UTC").floor("min")
    df = df[df["time"] < current_minute]

    if df.empty:
        raise RuntimeError("No completed 1-minute candles available.")

    return df.reset_index(drop=True)


def get_history(asset):
    """
    Combine recently fetched candles with the local SQLite cache.
    100,000 candles is a target, not a guaranteed data-provider result.
    """

    init_db()
    symbol = asset["symbol"]

    recent = fetch_yahoo_1m(symbol)
    save_candles(symbol, recent)

    cached = load_candles(symbol)
    combined = pd.concat([cached, recent], ignore_index=True)

    if combined.empty:
        raise RuntimeError("No candle history available.")

    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)

    # Exclude unfinished candles even if they exist in the local cache.
    current_minute = pd.Timestamp.now(tz="UTC").floor("min")
    combined = combined[combined["time"] < current_minute]

    return combined.reset_index(drop=True)


# --------------------------- INDICATORS ---------------------------

def add_indicators(df):
    df = df.copy()

    # EMA 15
    df["ema15"] = df["close"].ewm(
        span=15,
        adjust=False,
        min_periods=15,
    ).mean()

    # Bollinger Bands: 20-period SMA, 2 standard deviations
    df["bb_mid"] = df["close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    rolling_std = df["close"].rolling(
        window=20,
        min_periods=20,
    ).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + (2 * rolling_std)
    df["bb_lower"] = df["bb_mid"] - (2 * rolling_std)

    return df


# --------------------------- SIGNAL LOGIC ---------------------------

def generate_signal(df, lookback=15, band_mode="Touch or cross"):
    """
    SELL:
      1. Price rose over the selected 15-17 candle period.
      2. Close is above EMA 15.
      3. Bollinger upper-band condition is satisfied.

    BUY:
      1. Price fell over the selected 15-17 candle period.
      2. Close is below EMA 15.
      3. Bollinger lower-band condition is satisfied.

    Signals are analysis outputs, not trade instructions or guarantees.
    """

    empty = {
        "signal": "NO SIGNAL",
        "reason": "Not enough data or conditions not met.",
        "change_pct": np.nan,
        "trend": "Unknown",
        "price": np.nan,
        "ema15": np.nan,
        "bb_upper": np.nan,
        "bb_lower": np.nan,
    }

    if df is None or len(df) < max(lookback + 20, 40):
        return empty

    data = add_indicators(df)
    last = data.iloc[-1]
    start = data.iloc[-(lookback + 1)]

    required = [
        last["close"], last["ema15"], last["bb_upper"],
        last["bb_lower"], start["close"],
    ]

    if any(pd.isna(x) for x in required):
        return empty

    start_price = float(start["close"])
    price = float(last["close"])
    ema15 = float(last["ema15"])
    upper = float(last["bb_upper"])
    lower = float(last["bb_lower"])

    if start_price == 0:
        return empty

    change_pct = (price - start_price) / start_price * 100

    rising = price > start_price
    falling = price < start_price

    if rising:
        trend = "Rising"
    elif falling:
        trend = "Falling"
    else:
        trend = "Flat"

    if band_mode == "Touch or cross":
        upper_condition = float(last["high"]) >= upper
        lower_condition = float(last["low"]) <= lower
    else:
        upper_condition = price >= upper
        lower_condition = price <= lower

    # SELL: rising period + EMA15 confirmation + upper band
    sell_condition = (
        rising
        and price > ema15
        and upper_condition
    )

    # BUY: falling period + EMA15 confirmation + lower band
    buy_condition = (
        falling
        and price < ema15
        and lower_condition
    )

    if sell_condition and not buy_condition:
        signal = "SELL"
        reason = (
            f"Last {lookback} candles rose by {change_pct:.3f}%; "
            "close is above EMA 15 and upper Bollinger condition is met."
        )
    elif buy_condition and not sell_condition:
        signal = "BUY"
        reason = (
            f"Last {lookback} candles fell by {change_pct:.3f}%; "
            "close is below EMA 15 and lower Bollinger condition is met."
        )
    else:
        signal = "NO SIGNAL"
        reason = (
            f"No complete setup. Trend={trend}; "
            f"change={change_pct:.3f}%; "
            f"close={price:.6g}; EMA15={ema15:.6g}."
        )

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": change_pct,
        "trend": trend,
        "price": price,
        "ema15": ema15,
        "bb_upper": upper,
        "bb_lower": lower,
    }


# --------------------------- SCANNER ---------------------------

def scan_asset(asset, lookback, band_mode):
    try:
        candles = get_history(asset)

        if len(candles) < 40:
            raise RuntimeError(
                f"Only {len(candles)} candles available; more data is needed."
            )

        indicators = add_indicators(candles)
        signal = generate_signal(indicators, lookback, band_mode)

        return {
            **asset,
            **signal,
            "count": len(indicators),
            "progress_pct": min(
                100.0,
                len(indicators) / TARGET_CANDLES * 100,
            ),
            "candles": indicators,
            "error": None,
            "source": "Yahoo Finance + local SQLite cache",
        }

    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "reason": "Signal not calculated because data was unavailable.",
            "change_pct": np.nan,
            "trend": "Unknown",
            "price": np.nan,
            "ema15": np.nan,
            "bb_upper": np.nan,
            "bb_lower": np.nan,
            "count": 0,
            "progress_pct": 0.0,
            "candles": None,
            "error": str(exc),
            "source": "Unavailable",
        }


# --------------------------- USER INTERFACE ---------------------------

st.title("📊 Velora — EMA 15 + Bollinger Bands")
st.caption(
    "1-minute market scanner | BUY / SELL signals only | "
    "No automatic order execution"
)

st.warning(
    "API key is not required. This app attempts to retrieve recent 1-minute "
    "candles from Yahoo Finance's unofficial chart endpoint. Data availability "
    "depends on the symbol and provider. These are not actual Binomo OTC quotes."
)

st.info(
    "100,000 one-minute candles represent about 69.4 days of continuous data. "
    "The free source usually cannot provide this much intraday history. "
    "The SQLite cache can accumulate data over time, but it cannot recreate "
    "historical candles that the provider never supplied. Cloud redeployments "
    "may also reset local files."
)

st.sidebar.header("Scanner settings")

category = st.sidebar.selectbox(
    "Asset category",
    ["All", "Crypto", "Forex", "Other"],
)

lookback = st.sidebar.select_slider(
    "Candles used to measure rise/fall",
    options=[15, 16, 17],
    value=15,
)

band_mode = st.sidebar.radio(
    "Bollinger confirmation",
    ["Touch or cross", "Close outside band"],
    index=0,
)

asset_limit = st.sidebar.number_input(
    "Maximum assets per scan",
    min_value=1,
    max_value=len(ASSETS),
    value=min(10, len(ASSETS)),
    step=1,
)

show_no_signal = st.sidebar.checkbox(
    "Show NO SIGNAL results",
    value=True,
)

if category == "Crypto":
    selected_assets = [a for a in ASSETS if a["category"] == "Crypto"]
elif category == "Forex":
    selected_assets = [a for a in ASSETS if a["category"] == "Forex"]
elif category == "Other":
    selected_assets = [a for a in ASSETS if a["category"] == "Other"]
else:
    selected_assets = list(ASSETS)

selected_assets = selected_assets[:int(asset_limit)]

st.sidebar.success("No API key required")

st.markdown("### Strategy rules")
st.markdown(
    f"""
- **SELL:** The last {lookback} completed candles rose, the close is above
  EMA 15, and the Bollinger upper-band condition is met.
- **BUY:** The last {lookback} completed candles fell, the close is below
  EMA 15, and the Bollinger lower-band condition is met.
- **NO SIGNAL:** The complete setup is not present.
- The current unfinished 1-minute candle is excluded.
"""
)

st.write(f"Assets selected: **{len(selected_assets)}**")
st.write(f"History target per asset: **{TARGET_CANDLES:,} completed candles**")

if st.button(
    "🔍 START SCAN",
    type="primary",
    use_container_width=True,
):
    results = []
    progress = st.progress(0)
    status = st.empty()

    # Low concurrency to reduce provider errors.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(scan_asset, asset, lookback, band_mode): asset
            for asset in selected_assets
        }

        total = len(futures)

        for index, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                asset = futures[future]
                results.append({
                    **asset,
                    "signal": "NO SIGNAL",
                    "reason": "Scanner error.",
                    "count": 0,
                    "progress_pct": 0.0,
                    "candles": None,
                    "error": str(exc),
                    "source": "Unavailable",
                })

            progress.progress(index / max(total, 1))
            status.write(f"Scanned {index} of {total} assets")

    st.session_state["velora_results"] = results
    st.session_state["velora_scan_time"] = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    progress.empty()
    status.empty()

results = st.session_state.get("velora_results", [])

if not results:
    st.info("Click START SCAN to retrieve data and calculate signals.")
    st.stop()

valid_results = [
    result for result in results
    if result.get("candles") is not None and not result.get("error")
]

errors = [result for result in results if result.get("error")]

buy_results = [
    result for result in valid_results if result["signal"] == "BUY"
]
sell_results = [
    result for result in valid_results if result["signal"] == "SELL"
]
no_signal_results = [
    result for result in valid_results if result["signal"] == "NO SIGNAL"
]

st.caption(
    "Last scan: " + st.session_state.get("velora_scan_time", "Unknown")
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Assets with 100K candles",
    sum(1 for r in valid_results if r["count"] >= TARGET_CANDLES),
)
col2.metric("BUY", len(buy_results))
col3.metric("SELL", len(sell_results))
col4.metric("Data errors", len(errors))

st.subheader("🚨 BUY / SELL signals")

signals = buy_results + sell_results

if signals:
    signal_table = pd.DataFrame([{
        "Asset": r["name"],
        "Signal": r["signal"],
        "Price": r["price"],
        f"{lookback}-candle trend": r["trend"],
        f"{lookback}-candle change (%)": r["change_pct"],
        "EMA 15": r["ema15"],
        "Bollinger upper": r["bb_upper"],
        "Bollinger lower": r["bb_lower"],
        "Candles stored": r["count"],
        "100K progress (%)": round(r["progress_pct"], 2),
        "Reason": r["reason"],
    } for r in signals])

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No BUY or SELL setups currently meet all selected conditions.")

st.subheader("📋 All scan results")

shown_results = (
    valid_results
    if show_no_signal
    else [r for r in valid_results if r["signal"] != "NO SIGNAL"]
)

if shown_results:
    all_table = pd.DataFrame([{
        "Asset": r["name"],
        "Category": r["category"],
        "Signal": r["signal"],
        "Price": r["price"],
        "Trend": r["trend"],
        "Candles stored": r["count"],
        "100K complete": r["count"] >= TARGET_CANDLES,
        "100K progress (%)": round(r["progress_pct"], 2),
        "Source": r["source"],
        "Reason": r["reason"],
    } for r in shown_results])

    st.dataframe(
        all_table,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("📈 Asset chart")

chartable = [
    r for r in valid_results
    if r.get("candles") is not None
    and len(r["candles"]) > 0
]

if chartable:
    selected_name = st.selectbox(
        "Choose an asset",
        [r["name"] for r in chartable],
    )

    selected = next(
        r for r in chartable if r["name"] == selected_name
    )

    chart_df = selected["candles"].tail(500).copy()
    chart_df = chart_df.set_index("time")

    st.write(f"**Signal:** {selected['signal']}")
    st.write(f"**Reason:** {selected['reason']}")
    st.write(
        f"**Candles available:** {selected['count']:,} / "
        f"{TARGET_CANDLES:,} target"
    )

    st.markdown("**Close price and EMA 15**")
    st.line_chart(
        chart_df[["close", "ema15"]].rename(columns={
            "close": "Close",
            "ema15": "EMA 15",
        })
    )

    st.markdown("**Bollinger Bands**")
    st.line_chart(
        chart_df[["close", "bb_upper", "bb_mid", "bb_lower"]].rename(
            columns={
                "close": "Close",
                "bb_upper": "Upper band",
                "bb_mid": "Middle band",
                "bb_lower": "Lower band",
            }
        )
    )
else:
    st.info("No chart available because candle data could not be retrieved.")

st.subheader("⚠️ Data errors")

if errors:
    st.dataframe(
        pd.DataFrame([{
            "Asset": r["name"],
            "Symbol": r["symbol"],
            "Error": r.get("error", "Unknown error"),
        } for r in errors]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No data errors in this scan.")

if results:
    export_df = pd.DataFrame([{
        "asset": r.get("name"),
        "symbol": r.get("symbol"),
        "category": r.get("category"),
        "signal": r.get("signal"),
        "price": r.get("price"),
        "trend": r.get("trend"),
        "change_pct": r.get("change_pct"),
        "ema15": r.get("ema15"),
        "bb_upper": r.get("bb_upper"),
        "bb_lower": r.get("bb_lower"),
        "candle_count": r.get("count"),
        "source": r.get("source"),
        "reason": r.get("reason"),
        "error": r.get("error"),
    } for r in results])

    st.download_button(
        "📥 Download CSV results",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="velora_scan_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Velora only analyzes market data and does not place trades. "
    "A signal is not a guarantee of a profitable outcome. Yahoo Finance "
    "intraday history is limited and may fail for some symbols. The 100,000 "
    "candle target is not guaranteed. Prices are not actual Binomo OTC quotes."
)
