
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VELORA - EMA 15 + BOLLINGER BANDS SCANNER
# 1-minute candles | BUY / SELL signals | No order execution
# No API key required
# ============================================================

st.set_page_config(
    page_title="Velora Scanner",
    page_icon="📊",
    layout="wide",
)

DB_PATH = "velora_history.sqlite3"
TARGET_CANDLES = 100_000
REQUEST_TIMEOUT = 20

ASSETS = [
    {"name": "Bitcoin", "symbol": "BTC-USD", "category": "Crypto"},
    {"name": "Ethereum", "symbol": "ETH-USD", "category": "Crypto"},
    {"name": "Solana", "symbol": "SOL-USD", "category": "Crypto"},
    {"name": "XRP", "symbol": "XRP-USD", "category": "Crypto"},
    {"name": "Dogecoin", "symbol": "DOGE-USD", "category": "Crypto"},
    {"name": "Cardano", "symbol": "ADA-USD", "category": "Crypto"},
    {"name": "Avalanche", "symbol": "AVAX-USD", "category": "Crypto"},
    {"name": "Chainlink", "symbol": "LINK-USD", "category": "Crypto"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "category": "Forex"},
    {"name": "GBP/USD", "symbol": "GBPUSD=X", "category": "Forex"},
    {"name": "USD/JPY", "symbol": "JPY=X", "category": "Forex"},
    {"name": "AUD/USD", "symbol": "AUDUSD=X", "category": "Forex"},
    {"name": "USD/CAD", "symbol": "CAD=X", "category": "Forex"},
    {"name": "USD/CHF", "symbol": "CHF=X", "category": "Forex"},
    {"name": "NZD/USD", "symbol": "NZDUSD=X", "category": "Forex"},
    {"name": "EUR/GBP", "symbol": "EURGBP=X", "category": "Forex"},
    {"name": "Gold futures proxy", "symbol": "GC=F", "category": "Other"},
    {"name": "Apple", "symbol": "AAPL", "category": "Other"},
    {"name": "Microsoft", "symbol": "MSFT", "category": "Other"},
    {"name": "NVIDIA", "symbol": "NVDA", "category": "Other"},
]


# ------------------------------------------------------------
# HTTP SESSION
# ------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
})


# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------

def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                candle_time TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0,
                PRIMARY KEY (symbol, candle_time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
            ON candles(symbol, candle_time)
        """)
        conn.commit()


def load_cached_candles(symbol):
    init_db()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            """
            SELECT candle_time AS time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY candle_time ASC
            """,
            conn,
            params=(symbol,),
        )

    if not df.empty:
        df["time"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=["time", "open", "high", "low", "close"]
        )

    return df


def save_cached_candles(symbol, df):
    if df is None or df.empty:
        return

    rows = []

    for row in df.itertuples(index=False):
        timestamp = pd.Timestamp(row.time)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        volume = getattr(row, "volume", 0)

        if pd.isna(volume):
            volume = 0

        rows.append((
            symbol,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(volume),
        ))

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO candles
            (symbol, candle_time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


# ------------------------------------------------------------
# YAHOO FINANCE DATA
# ------------------------------------------------------------

def fetch_yahoo_candles(symbol):
    """
    Try two Yahoo Finance hosts for recent one-minute OHLCV data.
    This endpoint requires no API key, but access and history
    availability are not guaranteed.
    """

    errors = []

    encoded_symbol = requests.utils.quote(symbol, safe="")

    hosts = [
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    ]

    for host in hosts:
        url = (
            f"https://{host}/v8/finance/chart/"
            f"{encoded_symbol}"
        )

        try:
            response = SESSION.get(
                url,
                params={
                    "range": "7d",
                    "interval": "1m",
                    "includePrePost": "false",
                },
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "HTTP 429: provider rate limit reached."
                )

            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {response.status_code}: access denied."
                )

            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError(
                    "Provider returned an invalid JSON response."
                )

            chart = payload.get("chart", {})

            if chart.get("error"):
                error_info = chart["error"] or {}
                raise RuntimeError(
                    str(
                        error_info.get("description")
                        or error_info.get("code")
                        or "Provider returned a chart error."
                    )
                )

            results = chart.get("result") or []

            if not results:
                raise RuntimeError(
                    "No chart data returned for this symbol."
                )

            result = results[0]
            timestamps = result.get("timestamp") or []

            quotes = (
                result.get("indicators", {}).get("quote") or []
            )

            if not timestamps or not quotes:
                raise RuntimeError(
                    "Response has no timestamp or OHLC data."
                )

            quote = quotes[0]

            df = pd.DataFrame({
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

            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            df["volume"] = df["volume"].fillna(0)

            df = df.dropna(
                subset=["time", "open", "high", "low", "close"]
            )

            df = (
                df.drop_duplicates(subset=["time"])
                .sort_values("time")
                .reset_index(drop=True)
            )

            # Remove the current unfinished one-minute candle.
            current_minute = pd.Timestamp.now(
                tz="UTC"
            ).floor("min")

            df = df[
                df["time"] < current_minute
            ].reset_index(drop=True)

            if df.empty:
                raise RuntimeError(
                    "No completed one-minute candles returned."
                )

            return df, host

        except Exception as exc:
            errors.append(f"{host}: {exc}")

    raise RuntimeError(
        "Both Yahoo Finance endpoints failed. "
        + " | ".join(errors)
    )


def get_candles(asset):
    symbol = asset["symbol"]

    recent = pd.DataFrame()
    provider_error = None
    host_used = None

    try:
        recent, host_used = fetch_yahoo_candles(symbol)
        save_cached_candles(symbol, recent)

    except Exception as exc:
        provider_error = str(exc)

    cached = load_cached_candles(symbol)

    frames = [
        frame
        for frame in [cached, recent]
        if frame is not None and not frame.empty
    ]

    if not frames:
        raise RuntimeError(
            provider_error or "No market data available."
        )

    df = pd.concat(frames, ignore_index=True)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    df = (
        df.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    current_minute = pd.Timestamp.now(
        tz="UTC"
    ).floor("min")

    df = df[df["time"] < current_minute]

    # Keep the most recent target candles in local history.
    df = df.tail(TARGET_CANDLES).reset_index(drop=True)

    if len(df) < 40:
        raise RuntimeError(
            f"Only {len(df)} completed candles are available. "
            "At least 40 are required. "
            + (provider_error or "")
        )

    if host_used:
        source = f"Yahoo Finance ({host_used}) + SQLite cache"
    else:
        source = "SQLite cache only; live provider unavailable"

    return df, source, provider_error


# ------------------------------------------------------------
# TECHNICAL INDICATORS
# ------------------------------------------------------------

def calculate_indicators(df):
    df = df.copy()

    # Exponential Moving Average, period 15
    df["ema15"] = df["close"].ewm(
        span=15,
        adjust=False,
        min_periods=15,
    ).mean()

    # Bollinger Bands: SMA 20, standard deviation x 2
    df["bb_mid"] = df["close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    standard_deviation = df["close"].rolling(
        window=20,
        min_periods=20,
    ).std(ddof=0)

    df["bb_upper"] = (
        df["bb_mid"] + 2 * standard_deviation
    )

    df["bb_lower"] = (
        df["bb_mid"] - 2 * standard_deviation
    )

    return df


# ------------------------------------------------------------
# BUY / SELL SIGNALS
# ------------------------------------------------------------

def calculate_signal(df, lookback=15, band_mode="Touch or cross"):
    """
    SELL:
      - Price rose over the selected completed-candle period.
      - At least half of the period's candles were green.
      - Latest close is above EMA 15.
      - Upper Bollinger condition is met.

    BUY:
      - Price fell over the selected completed-candle period.
      - At least half of the period's candles were red.
      - Latest close is below EMA 15.
      - Lower Bollinger condition is met.

    This app only calculates signals and never places orders.
    """

    empty = {
        "signal": "NO SIGNAL",
        "reason": "Not enough data to calculate a signal.",
        "change_pct": np.nan,
        "trend": "Unknown",
    }

    if df is None or len(df) < max(lookback + 25, 40):
        return empty

    data = calculate_indicators(df)

    latest = data.iloc[-1]
    starting = data.iloc[-(lookback + 1)]

    required = [
        latest["ema15"],
        latest["bb_upper"],
        latest["bb_lower"],
        starting["close"],
    ]

    if any(pd.isna(value) for value in required):
        return {
            **empty,
            "reason": "EMA or Bollinger indicators are not ready.",
        }

    start_price = float(starting["close"])
    current_price = float(latest["close"])

    if start_price == 0:
        return {
            **empty,
            "reason": "Starting price is zero.",
        }

    change_pct = (
        (current_price - start_price) / start_price
    ) * 100

    # Use the previous completed candles to assess candle direction.
    segment = data.iloc[-(lookback + 1):-1]

    green_ratio = float(
        (segment["close"] > segment["open"]).mean()
    )

    red_ratio = float(
        (segment["close"] < segment["open"]).mean()
    )

    rising = change_pct > 0 and green_ratio >= 0.5
    falling = change_pct < 0 and red_ratio >= 0.5

    if band_mode == "Touch or cross":
        upper_condition = (
            float(latest["high"]) >= float(latest["bb_upper"])
        )
        lower_condition = (
            float(latest["low"]) <= float(latest["bb_lower"])
        )
    else:
        upper_condition = (
            current_price >= float(latest["bb_upper"])
        )
        lower_condition = (
            current_price <= float(latest["bb_lower"])
        )

    above_ema = current_price > float(latest["ema15"])
    below_ema = current_price < float(latest["ema15"])

    if rising and above_ema and upper_condition:
        signal = "SELL"
        reason = (
            f"Last {lookback} candles rose by "
            f"{change_pct:.3f}%; close above EMA 15; "
            "upper Bollinger condition met."
        )

    elif falling and below_ema and lower_condition:
        signal = "BUY"
        reason = (
            f"Last {lookback} candles fell by "
            f"{change_pct:.3f}%; close below EMA 15; "
            "lower Bollinger condition met."
        )

    else:
        signal = "NO SIGNAL"
        reason = (
            f"Conditions incomplete. Change={change_pct:.3f}%; "
            f"green candle ratio={green_ratio:.0%}; "
            f"red candle ratio={red_ratio:.0%}."
        )

    if rising:
        trend = "Rising"
    elif falling:
        trend = "Falling"
    else:
        trend = "Mixed / sideways"

    return {
        "signal": signal,
        "reason": reason,
        "change_pct": change_pct,
        "trend": trend,
    }


# ------------------------------------------------------------
# SCAN ONE ASSET
# ------------------------------------------------------------

def scan_asset(asset, lookback, band_mode):
    try:
        candles, source, provider_error = get_candles(asset)
        data = calculate_indicators(candles)
        signal = calculate_signal(data, lookback, band_mode)
        latest = data.iloc[-1]

        return {
            **asset,
            **signal,
            "price": float(latest["close"]),
            "ema15": float(latest["ema15"]),
            "bb_upper": float(latest["bb_upper"]),
            "bb_lower": float(latest["bb_lower"]),
            "count": len(data),
            "source": source,
            "candles": data,
            "error": None,
            "provider_warning": provider_error,
        }

    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "reason": "No signal calculated because data is unavailable.",
            "change_pct": np.nan,
            "trend": "Unknown",
            "price": np.nan,
            "ema15": np.nan,
            "bb_upper": np.nan,
            "bb_lower": np.nan,
            "count": 0,
            "source": "Unavailable",
            "candles": None,
            "error": str(exc),
            "provider_warning": None,
        }


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.title("📊 Velora — EMA 15 + Bollinger Bands")

st.caption(
    "1-minute market scanner | BUY / SELL only | "
    "No automatic trading"
)

st.warning(
    "No API key is required. The app attempts to use Yahoo Finance's "
    "unofficial chart endpoint. Provider access and available history "
    "are not guaranteed. These prices are not actual Binomo OTC quotes."
)

st.info(
    "Target: 100,000 completed one-minute candles per asset. "
    "The free Yahoo Finance endpoint does not guarantee that history. "
    "SQLite can accumulate candles while the app retains its database, "
    "but it cannot recreate candles that were never downloaded."
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
    index=0,
    format_func=lambda value: (
        "Touch or cross"
        if value == "Touch or cross"
        else "Close outside band"
    ),
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

st.sidebar.success("No API key required")

if category == "Crypto":
    selected_assets = [
        asset for asset in ASSETS
        if asset["category"] == "Crypto"
    ]
elif category == "Forex":
    selected_assets = [
        asset for asset in ASSETS
        if asset["category"] == "Forex"
    ]
elif category == "Other":
    selected_assets = [
        asset for asset in ASSETS
        if asset["category"] == "Other"
    ]
else:
    selected_assets = ASSETS.copy()

selected_assets = selected_assets[:int(asset_limit)]

st.markdown("### Strategy rules")
st.markdown(
    f"""
- **SELL:** The last {lookback} completed candles rose,
  at least half of the period candles were green,
  the latest close is above EMA 15, and the upper Bollinger
  condition is met.
- **BUY:** The last {lookback} completed candles fell,
  at least half of the period candles were red,
  the latest close is below EMA 15, and the lower Bollinger
  condition is met.
- **NO SIGNAL:** The complete setup is not present.
- The unfinished current one-minute candle is excluded.
"""
)

st.write(f"Assets selected: **{len(selected_assets)}**")
st.write(f"History target per asset: **{TARGET_CANDLES:,} candles**")

if "velora_results" not in st.session_state:
    st.session_state["velora_results"] = []

if "velora_scan_time" not in st.session_state:
    st.session_state["velora_scan_time"] = None

if st.button(
    "🔍 START SCAN",
    type="primary",
    use_container_width=True,
):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                scan_asset,
                asset,
                lookback,
                band_mode,
            ): asset
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
                    "error": str(exc),
                    "count": 0,
                    "candles": None,
                })

            progress.progress(index / max(total, 1))
            status.write(f"Scanned {index} / {total} assets")

    st.session_state["velora_results"] = results
    st.session_state["velora_scan_time"] = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    progress.empty()
    status.empty()

results = st.session_state["velora_results"]

if not results:
    st.info("Click START SCAN to retrieve data and calculate signals.")
    st.stop()

valid_results = [
    result for result in results
    if result.get("candles") is not None
    and not result.get("error")
]

errors = [
    result for result in results
    if result.get("error")
]

buy_results = [
    result for result in valid_results
    if result.get("signal") == "BUY"
]

sell_results = [
    result for result in valid_results
    if result.get("signal") == "SELL"
]

no_signal_results = [
    result for result in valid_results
    if result.get("signal") == "NO SIGNAL"
]

st.caption(
    "Last scan: "
    + str(st.session_state.get("velora_scan_time") or "Not available")
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Assets with 100K candles",
    sum(
        result["count"] >= TARGET_CANDLES
        for result in valid_results
    ),
)

col2.metric("BUY", len(buy_results))
col3.metric("SELL", len(sell_results))
col4.metric("Data errors", len(errors))

st.subheader("🚨 BUY / SELL signals")

signals = buy_results + sell_results

if signals:
    signal_table = pd.DataFrame([
        {
            "Asset": result["name"],
            "Signal": result["signal"],
            "Price": result["price"],
            "Trend": result["trend"],
            "Period change (%)": result["change_pct"],
            "EMA 15": result["ema15"],
            "Bollinger upper": result["bb_upper"],
            "Bollinger lower": result["bb_lower"],
            "Candles stored": result["count"],
            "100K progress (%)": round(
                min(
                    100,
                    result["count"] / TARGET_CANDLES * 100,
                ),
                2,
            ),
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
        "No BUY or SELL setup currently meets all selected conditions."
    )

st.subheader("📋 All scan results")

shown_results = (
    valid_results
    if show_no_signal
    else [
        result for result in valid_results
        if result["signal"] != "NO SIGNAL"
    ]
)

if shown_results:
    all_table = pd.DataFrame([
        {
            "Asset": result["name"],
            "Category": result["category"],
            "Signal": result["signal"],
            "Price": result["price"],
            "Trend": result["trend"],
            "Candles": result["count"],
            "100K complete": result["count"] >= TARGET_CANDLES,
            "100K progress (%)": round(
                min(
                    100,
                    result["count"] / TARGET_CANDLES * 100,
                ),
                2,
            ),
            "Source": result["source"],
            "Reason": result["reason"],
        }
        for result in shown_results
    ])

    st.dataframe(
        all_table,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("📈 Asset chart")

chartable = [
    result for result in valid_results
    if result.get("candles") is not None
    and not result["candles"].empty
]

if chartable:
    selected_name = st.selectbox(
        "Choose an asset",
        [result["name"] for result in chartable],
    )

    selected = next(
        result for result in chartable
        if result["name"] == selected_name
    )

    chart_df = (
        selected["candles"]
        .tail(500)
        .set_index("time")
    )

    st.write(f"**Signal:** {selected['signal']}")
    st.write(f"**Reason:** {selected['reason']}")
    st.write(
        f"**Available candles:** {selected['count']:,} / "
        f"{TARGET_CANDLES:,}"
    )
    st.write(f"**Source:** {selected['source']}")

    st.markdown("**Close price and EMA 15**")
    st.line_chart(
        chart_df[["close", "ema15"]].rename(
            columns={
                "close": "Close",
                "ema15": "EMA 15",
            }
        )
    )

    st.markdown("**Bollinger Bands**")
    st.line_chart(
        chart_df[
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
        "No chart available because candle data could not be retrieved."
    )

st.subheader("⚠️ Data errors")

if errors:
    st.error(
        "Some assets could not be loaded. Read the Error column "
        "to diagnose the problem."
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Asset": result.get("name"),
                "Symbol": result.get("symbol"),
                "Error": result.get("error"),
            }
            for result in errors
        ]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No data errors in this scan.")

if results:
    export_df = pd.DataFrame([
        {
            "asset": result.get("name"),
            "symbol": result.get("symbol"),
            "category": result.get("category"),
            "signal": result.get("signal"),
            "price": result.get("price"),
            "trend": result.get("trend"),
            "change_pct": result.get("change_pct"),
            "ema15": result.get("ema15"),
            "bb_upper": result.get("bb_upper"),
            "bb_lower": result.get("bb_lower"),
            "candle_count": result.get("count"),
            "source": result.get("source"),
            "reason": result.get("reason"),
            "error": result.get("error"),
        }
        for result in results
    ])

    st.download_button(
        "📥 Download CSV results",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="velora_scan_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

st.caption(
    "Velora does not execute trades. Signals do not guarantee profit. "
    "Yahoo Finance's one-minute history is limited and may be unavailable. "
    "100,000 candles are a target, not a guarantee. "
    "Yahoo market data is not actual Binomo OTC pricing."
)
