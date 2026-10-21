
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# TensorFlow is optional. The technical scanner works without it.
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Velora AI Crypto Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Velora AI Crypto Scanner")
st.caption(
    "Analysis only — no order execution. Binance prices may differ "
    "from Binomo OTC prices. Signals do not guarantee profit."
)

API_BASE = "https://api.binance.com"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Velora-Crypto-Scanner/1.0"})

STABLECOINS = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE",
    "BUSD", "UST", "USTC", "EUR", "AEUR", "EURI",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]


# =========================================================
# BINANCE PUBLIC REST API — NO CCXT REQUIRED
# =========================================================

def api_get(path, params=None):
    """GET request to Binance public market-data endpoints."""
    url = API_BASE + path
    response = HTTP.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and "code" in data and data["code"] < 0:
        raise RuntimeError(data.get("msg", "Binance API error"))

    return data


def get_coin_universe(limit=50):
    """
    Find active Binance Spot USDT pairs and rank them by
    24-hour quote volume.
    """
    exchange_info = api_get("/api/v3/exchangeInfo")
    tickers = api_get("/api/v3/ticker/24hr")

    ticker_map = {
        item.get("symbol"): item
        for item in tickers
        if item.get("symbol")
    }

    rows = []

    for market in exchange_info.get("symbols", []):
        symbol = market.get("symbol", "")
        base = market.get("baseAsset", "")

        if market.get("status") != "TRADING":
            continue

        if market.get("isSpotTradingAllowed") is False:
            continue

        if market.get("quoteAsset") != "USDT":
            continue

        if base in STABLECOINS:
            continue

        if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
            continue

        ticker = ticker_map.get(symbol)
        if not ticker:
            continue

        try:
            price = float(ticker.get("lastPrice", 0))
            volume = float(ticker.get("quoteVolume", 0))
            change = float(ticker.get("priceChangePercent", 0))
        except (TypeError, ValueError):
            continue

        if not np.isfinite(price) or price <= 0:
            continue

        if not np.isfinite(volume) or volume <= 0:
            continue

        if not np.isfinite(change):
            change = 0.0

        rows.append({
            "Coin": base,
            "Symbol": symbol,
            "Price": price,
            "24h Volume (USDT)": volume,
            "24h Change (%)": change,
        })

    if not rows:
        raise RuntimeError(
            "Binance did not return eligible USDT Spot pairs."
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        "24h Volume (USDT)", ascending=False
    ).head(int(limit))

    return result.reset_index(drop=True)


def timeframe_milliseconds(timeframe):
    units = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
    }

    number = int(timeframe[:-1])
    unit = timeframe[-1]

    if unit not in units:
        raise ValueError("Unsupported timeframe")

    return number * units[unit]


def get_candles(symbol, timeframe="15m", limit=300):
    """
    Retrieve candles from Binance.
    Remove the last candle if it has not closed yet.
    """
    raw = api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(int(limit), 1000),
        },
    )

    if not raw:
        raise RuntimeError("No candles returned")

    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trade_count", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    df = pd.DataFrame(raw, columns=columns)

    df["timestamp"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["close_time"] = pd.to_numeric(
        df["close_time"], errors="coerce"
    )

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    now_ms = int(time.time() * 1000)

    # Binance closeTime is the last millisecond of a candle.
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise RuntimeError("Not enough completed candles")

    return df.reset_index(drop=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["EMA15"] = close.ewm(span=15, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    df["BB_MID"] = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)

    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std

    # RSI
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Stochastic
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    denominator = (highest - lowest).replace(0, np.nan)

    df["STOCH_K"] = (
        100 * (close - lowest) / denominator
    ).fillna(50)

    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean().fillna(50)

    # ATR
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # Directional movement and ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    safe_atr = df["ATR"].replace(0, np.nan)

    df["PLUS_DI"] = (
        100 * plus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean() / safe_atr
    )

    df["MINUS_DI"] = (
        100 * minus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean() / safe_atr
    )

    di_sum = (
        df["PLUS_DI"] + df["MINUS_DI"]
    ).replace(0, np.nan)

    dx = (
        100 * (df["PLUS_DI"] - df["MINUS_DI"]).abs() / di_sum
    )

    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["VOLUME_AVG20"] = volume.rolling(20).mean()

    return df.replace([np.inf, -np.inf], np.nan)


# =========================================================
# BUY / SELL SIGNAL ENGINE
# =========================================================

def analyze_signal(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    required = [
        "close", "EMA15", "EMA20", "EMA50",
        "BB_UPPER", "BB_LOWER", "RSI",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "STOCH_K", "STOCH_D", "PLUS_DI",
        "MINUS_DI", "ADX", "VOLUME_AVG20",
    ]

    if latest[required].isna().any():
        return {
            "Signal": "NO SIGNAL",
            "Buy Votes": 0,
            "Sell Votes": 0,
            "Confidence": 0,
            "RSI": np.nan,
            "ADX": np.nan,
            "ATR (%)": np.nan,
            "Reasons": "Indicator values incomplete",
        }

    price = float(latest["close"])
    buy = 0
    sell = 0
    buy_reasons = []
    sell_reasons = []

    # EMA trend
    if latest["EMA15"] > latest["EMA20"] > latest["EMA50"]:
        buy += 1
        buy_reasons.append("EMA bullish trend")

    elif latest["EMA15"] < latest["EMA20"] < latest["EMA50"]:
        sell += 1
        sell_reasons.append("EMA bearish trend")

    # Bollinger Band re-entry
    if (
        previous["close"] < previous["BB_LOWER"]
        and price > latest["BB_LOWER"]
    ):
        buy += 1
        buy_reasons.append("Bollinger lower-band recovery")

    if (
        previous["close"] > previous["BB_UPPER"]
        and price < latest["BB_UPPER"]
    ):
        sell += 1
        sell_reasons.append("Bollinger upper-band rejection")

    # RSI
    if (
        30 <= latest["RSI"] <= 45
        and latest["RSI"] > previous["RSI"]
    ):
        buy += 1
        buy_reasons.append("RSI recovering")

    elif (
        55 <= latest["RSI"] <= 70
        and latest["RSI"] < previous["RSI"]
    ):
        sell += 1
        sell_reasons.append("RSI weakening")

    # MACD
    if (
        latest["MACD"] > latest["MACD_SIGNAL"]
        and latest["MACD_HIST"] > previous["MACD_HIST"]
    ):
        buy += 1
        buy_reasons.append("MACD bullish momentum")

    elif (
        latest["MACD"] < latest["MACD_SIGNAL"]
        and latest["MACD_HIST"] < previous["MACD_HIST"]
    ):
        sell += 1
        sell_reasons.append("MACD bearish momentum")

    # Stochastic
    if (
        latest["STOCH_K"] > latest["STOCH_D"]
        and latest["STOCH_K"] < 35
    ):
        buy += 1
        buy_reasons.append("Stochastic bullish signal")

    elif (
        latest["STOCH_K"] < latest["STOCH_D"]
        and latest["STOCH_K"] > 65
    ):
        sell += 1
        sell_reasons.append("Stochastic bearish signal")

    # ADX directional confirmation
    if latest["ADX"] >= 18:
        if latest["PLUS_DI"] > latest["MINUS_DI"]:
            buy += 1
            buy_reasons.append("Positive directional trend")
        elif latest["MINUS_DI"] > latest["PLUS_DI"]:
            sell += 1
            sell_reasons.append("Negative directional trend")

    # Volume confirmation
    volume_confirmed = (
        latest["VOLUME_AVG20"] > 0
        and latest["volume"] >= latest["VOLUME_AVG20"]
    )

    if volume_confirmed and latest["close"] > latest["open"]:
        buy += 1
        buy_reasons.append("Green candle with strong volume")

    elif volume_confirmed and latest["close"] < latest["open"]:
        sell += 1
        sell_reasons.append("Red candle with strong volume")

    margin = abs(buy - sell)

    if buy >= 4 and margin >= 2:
        signal = "BUY"
        reasons = "; ".join(buy_reasons)
        confidence = min(95, 50 + buy * 7 + margin * 3)

    elif sell >= 4 and margin >= 2:
        signal = "SELL"
        reasons = "; ".join(sell_reasons)
        confidence = min(95, 50 + sell * 7 + margin * 3)

    else:
        signal = "NO SIGNAL"
        reasons = "Indicators do not agree sufficiently"
        confidence = 0

    atr_percent = (
        float(latest["ATR"]) / price * 100
        if price > 0 and pd.notna(latest["ATR"])
        else np.nan
    )

    return {
        "Signal": signal,
        "Buy Votes": int(buy),
        "Sell Votes": int(sell),
        "Confidence": int(confidence),
        "RSI": float(latest["RSI"]),
        "ADX": float(latest["ADX"]),
        "ATR (%)": atr_percent,
        "Reasons": reasons,
        "Last Candle (UTC)": str(latest["timestamp"]),
    }


# =========================================================
# OPTIONAL TENSORFLOW LSTM
# =========================================================

def predict_lstm(df, epochs=3):
    if not TF_AVAILABLE:
        return "UNAVAILABLE", np.nan

    prices = df["close"].values.astype(float).reshape(-1, 1)

    if len(prices) < 100:
        return "UNKNOWN", np.nan

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    window = 30
    X, y = [], []

    for i in range(window, len(scaled) - 1):
        X.append(scaled[i - window:i, 0])
        y.append(
            1 if scaled[i + 1, 0] > scaled[i, 0] else 0
        )

    if len(X) < 50:
        return "UNKNOWN", np.nan

    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32)

    split = int(len(X) * 0.85)

    if split < 30 or len(X) - split < 5:
        return "UNKNOWN", np.nan

    try:
        tf.keras.backend.clear_session()

        model = Sequential([
            Input(shape=(window, 1)),
            LSTM(16),
            Dropout(0.1),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X[:split],
            y[:split],
            epochs=int(epochs),
            batch_size=32,
            verbose=0,
            shuffle=False,
        )

        probability = float(
            model.predict(X[-1:], verbose=0)[0][0]
        )

        if probability >= 0.55:
            return "UP", probability

        if probability <= 0.45:
            return "DOWN", probability

        return "UNCERTAIN", probability

    except Exception:
        return "ERROR", np.nan


# =========================================================
# SCAN
# =========================================================

def scan_market(
    timeframe,
    coin_limit,
    candle_count,
    minimum_volume,
    use_lstm,
    lstm_epochs,
):
    universe = get_coin_universe(coin_limit)

    universe = universe[
        universe["24h Volume (USDT)"] >= minimum_volume
    ].copy()

    if universe.empty:
        return pd.DataFrame()

    records = universe.to_dict("records")
    results = []

    progress = st.progress(0)
    status = st.empty()

    lstm_count = 0
    max_lstm_models = 10

    for index, coin in enumerate(records, start=1):
        symbol = coin["Symbol"]

        status.write(
            f"Scanning {symbol} — {index}/{len(records)}"
        )

        row = dict(coin)
        row.update({
            "Signal": "NO SIGNAL",
            "Buy Votes": 0,
            "Sell Votes": 0,
            "Confidence": 0,
            "RSI": np.nan,
            "ADX": np.nan,
            "ATR (%)": np.nan,
            "LSTM": "OFF" if not use_lstm else "PENDING",
            "Reasons": "",
        })

        try:
            candles = get_candles(
                symbol,
                timeframe=timeframe,
                limit=candle_count,
            )

            indicators = add_indicators(candles)
            signal_data = analyze_signal(indicators)
            row.update(signal_data)

            if use_lstm and signal_data["Signal"] in ("BUY", "SELL"):
                if not TF_AVAILABLE:
                    row["LSTM"] = "UNAVAILABLE"
                    row["Reasons"] += "; TensorFlow unavailable"

                elif lstm_count < max_lstm_models:
                    lstm_count += 1

                    direction, probability = predict_lstm(
                        indicators,
                        epochs=lstm_epochs,
                    )

                    row["LSTM"] = direction
                    row["LSTM Probability"] = probability

                    # Do not accept a signal if the model confidently
                    # predicts the opposite direction.
                    if (
                        row["Signal"] == "BUY"
                        and direction == "DOWN"
                    ):
                        row["Signal"] = "NO SIGNAL"
                        row["Reasons"] += "; LSTM disagrees"

                    elif (
                        row["Signal"] == "SELL"
                        and direction == "UP"
                    ):
                        row["Signal"] = "NO SIGNAL"
                        row["Reasons"] += "; LSTM disagrees"

                else:
                    row["LSTM"] = "SKIPPED"

        except Exception as exc:
            row["Signal"] = "NO SIGNAL"
            row["Reasons"] = f"Data error: {str(exc)[:120]}"

        results.append(row)
        progress.progress(index / len(records))

        # Small delay to reduce request pressure.
        time.sleep(0.08)

    progress.empty()
    status.empty()

    result_df = pd.DataFrame(results)

    result_df["Vote Margin"] = (
        result_df["Buy Votes"].fillna(0)
        - result_df["Sell Votes"].fillna(0)
    ).abs()

    result_df["Priority"] = result_df["Signal"].map({
        "BUY": 0,
        "SELL": 1,
        "NO SIGNAL": 2,
    }).fillna(3)

    result_df = result_df.sort_values(
        [
            "Priority",
            "Confidence",
            "Vote Margin",
            "24h Volume (USDT)",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    return result_df


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Scanner Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    TIMEFRAMES,
    index=1,
)

coin_limit = st.sidebar.slider(
    "Coins to scan",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
)

candle_count = st.sidebar.selectbox(
    "Historical candles",
    [100, 200, 300, 500],
    index=2,
)

minimum_volume = st.sidebar.number_input(
    "Minimum 24h volume (USDT)",
    min_value=0,
    value=1_000_000,
    step=500_000,
)

use_lstm = st.sidebar.checkbox(
    "Enable TensorFlow LSTM",
    value=False,
)

lstm_epochs = st.sidebar.selectbox(
    "LSTM epochs",
    [2, 3, 5],
    index=1,
    disabled=not use_lstm,
)

scan_button = st.sidebar.button(
    "🔍 Scan coins",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "LSTM training can take additional time. "
    "The app never submits exchange orders."
)


# =========================================================
# MAIN DASHBOARD
# =========================================================

if "velora_results" not in st.session_state:
    st.session_state["velora_results"] = None

if scan_button:
    try:
        with st.spinner("Connecting to Binance public API..."):
            st.session_state["velora_results"] = scan_market(
                timeframe=timeframe,
                coin_limit=coin_limit,
                candle_count=candle_count,
                minimum_volume=float(minimum_volume),
                use_lstm=use_lstm,
                lstm_epochs=lstm_epochs,
            )

    except Exception as exc:
        st.error(
            "Unable to retrieve Binance data. Check your network, "
            "API access, and requirements.txt. "
            f"Details: {str(exc)}"
        )

results = st.session_state["velora_results"]

if results is None:
    st.info("Choose your settings and click **Scan coins**.")
    st.markdown(
        """
        **How signals are generated**

        - **BUY:** enough bullish indicator votes, with a clear margin.
        - **SELL:** enough bearish indicator votes, with a clear margin.
        - **NO SIGNAL:** indicators are mixed or insufficient.
        """
    )

elif results.empty:
    st.warning(
        "No eligible coins found. Lower the volume filter and scan again."
    )

else:
    buys = int((results["Signal"] == "BUY").sum())
    sells = int((results["Signal"] == "SELL").sum())
    neutral = int((results["Signal"] == "NO SIGNAL").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coins scanned", len(results))
    col2.metric("BUY", buys)
    col3.metric("SELL", sells)
    col4.metric("NO SIGNAL", neutral)

    st.subheader("⭐ Priority coins")

    priority = results[
        results["Signal"].isin(["BUY", "SELL"])
    ].copy()

    priority_columns = [
        "Coin",
        "Symbol",
        "Signal",
        "Price",
        "24h Change (%)",
        "24h Volume (USDT)",
        "Confidence",
        "Buy Votes",
        "Sell Votes",
        "RSI",
        "ADX",
        "LSTM",
    ]

    if priority.empty:
        st.warning(
            "No actionable signals were found in this scan. "
            "Do not force a trade when the conditions are unclear."
        )
    else:
        st.dataframe(
            priority[
                [c for c in priority_columns if c in priority.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("All scanned coins")

    selected_signals = st.multiselect(
        "Filter signals",
        ["BUY", "SELL", "NO SIGNAL"],
        default=["BUY", "SELL", "NO SIGNAL"],
    )

    filtered = results[
        results["Signal"].isin(selected_signals)
    ].copy()

    table_columns = [
        "Coin",
        "Symbol",
        "Signal",
        "Price",
        "24h Change (%)",
        "24h Volume (USDT)",
        "Confidence",
        "Buy Votes",
        "Sell Votes",
        "RSI",
        "ADX",
        "ATR (%)",
        "LSTM",
        "Reasons",
    ]

    st.dataframe(
        filtered[
            [c for c in table_columns if c in filtered.columns]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Selected coin details")

    selected_symbol = st.selectbox(
        "Select a coin",
        results["Symbol"].tolist(),
    )

    selected = results[
        results["Symbol"] == selected_symbol
    ].iloc[0]

    left, right = st.columns(2)

    with left:
        st.write(f"**Signal:** {selected['Signal']}")
        st.write(f"**Price:** {selected['Price']}")
        st.write(f"**RSI:** {selected.get('RSI', np.nan)}")
        st.write(f"**ADX:** {selected.get('ADX', np.nan)}")

    with right:
        st.write(
            f"**24h change:** {selected['24h Change (%)']:.2f}%"
        )
        st.write(
            f"**24h volume:** "
            f"{selected['24h Volume (USDT)']:,.0f} USDT"
        )
        st.write(
            f"**Confidence score:** {selected.get('Confidence', 0)}"
        )
        st.write(f"**LSTM:** {selected.get('LSTM', 'OFF')}")

    st.write("**Signal explanation**")
    st.write(selected.get("Reasons", ""))

    st.subheader("Price and indicator chart")

    try:
        chart = get_candles(
            selected_symbol,
            timeframe=timeframe,
            limit=candle_count,
        )

        chart = add_indicators(chart)
        chart = chart.set_index("timestamp")

        st.line_chart(
            chart[
                ["close", "EMA20", "EMA50", "BB_UPPER", "BB_LOWER"]
            ]
        )

    except Exception as exc:
        st.warning(f"Could not load chart: {exc}")

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download results CSV",
        data=csv_bytes,
        file_name="velora_crypto_signals.csv",
        mime="text/csv",
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Educational analysis only. This application does not execute trades "
    or connect to Binomo. Binance public prices can differ from Binomo OTC "
    prices. Technical analysis and machine-learning estimates can be wrong; "
    "no signal guarantees profit."
)
