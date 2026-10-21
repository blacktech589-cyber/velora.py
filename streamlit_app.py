
import time
import warnings

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# TensorFlow is optional at runtime.
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Crypto Signal Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Crypto Signal Scanner")
st.caption(
    "Analysis only — no order execution. Public Binance prices may differ "
    "from Binomo OTC prices. Signals are not guarantees of profit."
)


# ============================================================
# CONSTANTS
# ============================================================

STABLECOINS = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE",
    "BUSD", "UST", "USTC", "EUR", "AEUR", "EURI",
}

LEVERAGED_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR",
)

TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]


# ============================================================
# BINANCE CONNECTION
# ============================================================

@st.cache_resource
def get_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    return exchange


def get_spot_universe(exchange, limit=50):
    """Return liquid, active Binance USDT spot markets."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    rows = []

    for symbol, market in markets.items():
        if not market.get("active", True):
            continue

        if not market.get("spot", False):
            continue

        if not symbol.endswith("/USDT"):
            continue

        base = market.get("base", "")

        if base in STABLECOINS:
            continue

        if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
            continue

        ticker = tickers.get(symbol, {})
        last_price = ticker.get("last")
        quote_volume = ticker.get("quoteVolume")
        percentage = ticker.get("percentage")

        try:
            last_price = float(last_price)
            quote_volume = float(quote_volume or 0)
            percentage = float(percentage or 0)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(last_price) or last_price <= 0:
            continue

        if not np.isfinite(quote_volume) or quote_volume <= 0:
            continue

        if not np.isfinite(percentage):
            percentage = 0.0

        rows.append({
            "Coin": base,
            "Symbol": symbol,
            "Price": last_price,
            "24h Volume (USDT)": quote_volume,
            "24h Change (%)": percentage,
        })

    if not rows:
        raise RuntimeError(
            "No eligible Binance Spot USDT markets were returned."
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        "24h Volume (USDT)", ascending=False
    ).head(int(limit))

    return result.reset_index(drop=True)


# ============================================================
# MARKET DATA
# ============================================================

def fetch_candles(exchange, symbol, timeframe, requested=300):
    """Fetch OHLCV candles and remove the latest possibly incomplete candle."""
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=int(requested),
    )

    if not candles or len(candles) < 60:
        raise ValueError("Not enough candle data.")

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp", "open", "high", "low",
            "close", "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    numeric_columns = [
        "open", "high", "low", "close", "volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # The latest returned candle may still be forming.
    if len(df) > 60:
        df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise ValueError("Not enough completed candles.")

    return df.reset_index(drop=True)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Moving averages
    df["EMA15"] = close.ewm(span=15, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    df["BB_MID"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Stochastic oscillator
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)

    df["STOCH_K"] = (
        100 * (close - lowest_low) / denominator
    ).fillna(50)

    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean().fillna(50)

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # ADX and directional indicators
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

    atr_safe = df["ATR"].replace(0, np.nan)

    df["PLUS_DI"] = (
        100 * plus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean() / atr_safe
    )

    df["MINUS_DI"] = (
        100 * minus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean() / atr_safe
    )

    di_sum = (df["PLUS_DI"] + df["MINUS_DI"]).replace(0, np.nan)

    dx = (
        100 * (df["PLUS_DI"] - df["MINUS_DI"]).abs() / di_sum
    )

    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # Volume confirmation
    df["VOLUME_AVG20"] = volume.rolling(20).mean()

    # Recent candle direction
    df["CANDLE_GREEN"] = df["close"] > df["open"]
    df["CANDLE_RED"] = df["close"] < df["open"]

    return df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    required = [
        "close", "EMA15", "EMA20", "EMA50",
        "BB_UPPER", "BB_LOWER", "RSI",
        "MACD", "MACD_SIGNAL", "STOCH_K",
        "STOCH_D", "PLUS_DI", "MINUS_DI",
        "ADX", "VOLUME_AVG20",
    ]

    if latest[required].isna().any():
        return {
            "Signal": "NO SIGNAL",
            "Buy Votes": 0,
            "Sell Votes": 0,
            "Reasons": "Indicator data incomplete",
            "Confidence": 0,
        }

    price = float(latest["close"])
    buy_votes = 0
    sell_votes = 0
    buy_reasons = []
    sell_reasons = []

    # Trend
    if latest["EMA15"] > latest["EMA20"] > latest["EMA50"]:
        buy_votes += 1
        buy_reasons.append("EMA bullish alignment")

    elif latest["EMA15"] < latest["EMA20"] < latest["EMA50"]:
        sell_votes += 1
        sell_reasons.append("EMA bearish alignment")

    # Bollinger Bands: use a return from outside the band rather
    # than automatically selling/buying every band touch.
    if (
        previous["close"] < previous["BB_LOWER"]
        and price > latest["BB_LOWER"]
    ):
        buy_votes += 1
        buy_reasons.append("Price returned above lower Bollinger Band")

    if (
        previous["close"] > previous["BB_UPPER"]
        and price < latest["BB_UPPER"]
    ):
        sell_votes += 1
        sell_reasons.append("Price returned below upper Bollinger Band")

    # RSI
    if 30 <= latest["RSI"] <= 45 and latest["RSI"] > previous["RSI"]:
        buy_votes += 1
        buy_reasons.append("RSI recovering from lower zone")

    elif 55 <= latest["RSI"] <= 70 and latest["RSI"] < previous["RSI"]:
        sell_votes += 1
        sell_reasons.append("RSI weakening from upper zone")

    # MACD
    if (
        latest["MACD"] > latest["MACD_SIGNAL"]
        and latest["MACD_HIST"] > previous["MACD_HIST"]
    ):
        buy_votes += 1
        buy_reasons.append("MACD bullish momentum")

    elif (
        latest["MACD"] < latest["MACD_SIGNAL"]
        and latest["MACD_HIST"] < previous["MACD_HIST"]
    ):
        sell_votes += 1
        sell_reasons.append("MACD bearish momentum")

    # Stochastic
    if (
        latest["STOCH_K"] > latest["STOCH_D"]
        and latest["STOCH_K"] < 35
    ):
        buy_votes += 1
        buy_reasons.append("Stochastic bullish crossover")

    elif (
        latest["STOCH_K"] < latest["STOCH_D"]
        and latest["STOCH_K"] > 65
    ):
        sell_votes += 1
        sell_reasons.append("Stochastic bearish crossover")

    # Directional movement
    if latest["PLUS_DI"] > latest["MINUS_DI"] and latest["ADX"] >= 18:
        buy_votes += 1
        buy_reasons.append("Positive directional trend")

    elif latest["MINUS_DI"] > latest["PLUS_DI"] and latest["ADX"] >= 18:
        sell_votes += 1
        sell_reasons.append("Negative directional trend")

    # Candle and volume confirmation
    volume_confirmed = (
        latest["VOLUME_AVG20"] > 0
        and latest["volume"] >= latest["VOLUME_AVG20"]
    )

    if volume_confirmed and latest["CANDLE_GREEN"]:
        buy_votes += 1
        buy_reasons.append("Green candle with above-average volume")

    elif volume_confirmed and latest["CANDLE_RED"]:
        sell_votes += 1
        sell_reasons.append("Red candle with above-average volume")

    margin = abs(buy_votes - sell_votes)

    if buy_votes >= 4 and margin >= 2:
        signal = "BUY"
        reasons = "; ".join(buy_reasons)
        confidence = min(95, int(50 + buy_votes * 7 + margin * 3))

    elif sell_votes >= 4 and margin >= 2:
        signal = "SELL"
        reasons = "; ".join(sell_reasons)
        confidence = min(95, int(50 + sell_votes * 7 + margin * 3))

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
        "Buy Votes": int(buy_votes),
        "Sell Votes": int(sell_votes),
        "Reasons": reasons,
        "Confidence": confidence,
        "Price": price,
        "RSI": float(latest["RSI"]),
        "ADX": float(latest["ADX"]),
        "ATR (%)": atr_percent,
        "EMA20": float(latest["EMA20"]),
        "EMA50": float(latest["EMA50"]),
        "Last Candle (UTC)": str(latest["timestamp"]),
    }


# ============================================================
# OPTIONAL TENSORFLOW LSTM
# ============================================================

def lstm_predict_direction(df, epochs=3):
    """
    Small per-coin LSTM direction classifier.
    Returns UP, DOWN, or UNKNOWN.
    Uses historical close prices only; it is not a guarantee.
    """
    if not TF_AVAILABLE:
        return "UNKNOWN", np.nan

    prices = df["close"].astype(float).values.reshape(-1, 1)

    if len(prices) < 100:
        return "UNKNOWN", np.nan

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    window = 30
    X = []
    y = []

    for i in range(window, len(scaled) - 1):
        X.append(scaled[i - window:i, 0])
        y.append(
            1 if scaled[i + 1, 0] > scaled[i, 0] else 0
        )

    if len(X) < 40:
        return "UNKNOWN", np.nan

    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32)

    # Use a chronological split; do not shuffle future and past data.
    split = int(len(X) * 0.85)
    if split < 30 or len(X) - split < 5:
        return "UNKNOWN", np.nan

    X_train, y_train = X[:split], y[:split]

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
            X_train,
            y_train,
            epochs=int(epochs),
            batch_size=32,
            verbose=0,
        )

        prediction = float(
            model.predict(X[-1:].astype(np.float32), verbose=0)[0][0]
        )

        if prediction >= 0.55:
            return "UP", prediction

        if prediction <= 0.45:
            return "DOWN", prediction

        return "UNKNOWN", prediction

    except Exception:
        return "UNKNOWN", np.nan


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Scanner Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    TIMEFRAMES,
    index=1,
)

coin_limit = st.sidebar.slider(
    "Number of coins to scan",
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

use_lstm = st.sidebar.checkbox(
    "Enable TensorFlow LSTM confirmation",
    value=False,
)

lstm_epochs = st.sidebar.selectbox(
    "LSTM training epochs",
    [2, 3, 5],
    index=1,
    disabled=not use_lstm,
)

min_volume = st.sidebar.number_input(
    "Minimum 24h volume (USDT)",
    min_value=0,
    value=1_000_000,
    step=500_000,
)

scan_button = st.sidebar.button(
    "🔍 Scan coins",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "The LSTM option may significantly increase scan time. "
    "No exchange orders are placed."
)


# ============================================================
# SCAN FUNCTION
# ============================================================

def run_scan(exchange, timeframe, coin_limit, candle_count,
             min_volume, use_lstm, lstm_epochs):

    universe = get_spot_universe(exchange, limit=coin_limit)
    universe = universe[
        universe["24h Volume (USDT)"] >= float(min_volume)
    ].copy()

    if universe.empty:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    progress = st.progress(0)
    status = st.empty()

    records = universe.to_dict("records")
    total = len(records)

    # Limit expensive LSTM training to the first 10 coins by volume
    # that produce an actionable technical signal.
    lstm_candidates_used = 0
    max_lstm_candidates = 10

    for i, item in enumerate(records, start=1):
        symbol = item["Symbol"]

        status.write(f"Scanning {symbol} ({i}/{total})...")

        row = {
            "Coin": item["Coin"],
            "Symbol": symbol,
            "Price": item["Price"],
            "24h Volume (USDT)": item["24h Volume (USDT)"],
            "24h Change (%)": item["24h Change (%)"],
            "Signal": "NO SIGNAL",
            "Buy Votes": 0,
            "Sell Votes": 0,
            "Confidence": 0,
            "RSI": np.nan,
            "ADX": np.nan,
            "ATR (%)": np.nan,
            "LSTM": "OFF" if not use_lstm else "PENDING",
            "Reasons": "",
            "Last Candle (UTC)": "",
        }

        try:
            candles = fetch_candles(
                exchange,
                symbol,
                timeframe,
                requested=candle_count,
            )

            df = add_indicators(candles)
            signal_info = generate_signal(df)

            row.update(signal_info)

            # Keep original universe price and volume values.
            row["Coin"] = item["Coin"]
            row["Symbol"] = symbol
            row["24h Volume (USDT)"] = item["24h Volume (USDT)"]
            row["24h Change (%)"] = item["24h Change (%)"]

            if use_lstm and signal_info["Signal"] in ("BUY", "SELL"):
                if lstm_candidates_used < max_lstm_candidates:
                    lstm_candidates_used += 1

                    direction, probability = lstm_predict_direction(
                        df,
                        epochs=lstm_epochs,
                    )

                    row["LSTM"] = direction
                    row["LSTM Probability"] = probability

                    # Require model agreement if the model produced
                    # a confident direction.
                    if (
                        signal_info["Signal"] == "BUY"
                        and direction == "DOWN"
                    ):
                        row["Signal"] = "NO SIGNAL"
                        row["Reasons"] = (
                            str(row.get("Reasons", ""))
                            + "; LSTM disagrees"
                        )

                    elif (
                        signal_info["Signal"] == "SELL"
                        and direction == "UP"
                    ):
                        row["Signal"] = "NO SIGNAL"
                        row["Reasons"] = (
                            str(row.get("Reasons", ""))
                            + "; LSTM disagrees"
                        )

                    elif direction == "UNKNOWN":
                        row["Reasons"] = (
                            str(row.get("Reasons", ""))
                            + "; LSTM uncertain"
                        )
                else:
                    row["LSTM"] = "SKIPPED (limit)"

            results.append(row)

        except Exception as exc:
            row["Reasons"] = f"Data error: {str(exc)[:160]}"
            results.append(row)

        progress.progress(i / total)
        time.sleep(0.05)

    status.empty()
    progress.empty()

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return universe, results_df

    # Priority: actionable signals first, then stronger vote margin,
    # then higher 24h volume.
    results_df["Vote Margin"] = (
        results_df["Buy Votes"].fillna(0)
        - results_df["Sell Votes"].fillna(0)
    ).abs()

    results_df["Priority"] = results_df["Signal"].map({
        "BUY": 0,
        "SELL": 1,
        "NO SIGNAL": 2,
    }).fillna(3)

    results_df = results_df.sort_values(
        by=[
            "Priority",
            "Confidence",
            "Vote Margin",
            "24h Volume (USDT)",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    return universe, results_df


# ============================================================
# MAIN PANEL
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None

if "scan_universe" not in st.session_state:
    st.session_state["scan_universe"] = None

if scan_button:
    try:
        with st.spinner("Connecting to Binance public market data..."):
            exchange = get_exchange()

            universe, results = run_scan(
                exchange=exchange,
                timeframe=timeframe,
                coin_limit=coin_limit,
                candle_count=candle_count,
                min_volume=min_volume,
                use_lstm=use_lstm,
                lstm_epochs=lstm_epochs,
            )

            st.session_state["scan_universe"] = universe
            st.session_state["scan_results"] = results

    except Exception as exc:
        st.error(
            "Scan failed. Check your internet connection, package "
            "installation, and whether Binance is accessible from "
            f"your network. Details: {str(exc)}"
        )

results = st.session_state.get("scan_results")

if results is None:
    st.info(
        "Choose your settings in the sidebar and click **Scan coins** "
        "to begin."
    )

    st.markdown(
        """
        **Signal logic**
        - BUY: bullish technical votes exceed bearish votes sufficiently.
        - SELL: bearish technical votes exceed bullish votes sufficiently.
        - NO SIGNAL: indicators are mixed or the signal is too weak.
        """
    )

else:
    if results.empty:
        st.warning(
            "No results were returned. Lower the minimum volume filter "
            "or try another timeframe."
        )
    else:
        buy_count = int((results["Signal"] == "BUY").sum())
        sell_count = int((results["Signal"] == "SELL").sum())
        no_signal_count = int((results["Signal"] == "NO SIGNAL").sum())

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Coins scanned", len(results))
        c2.metric("BUY signals", buy_count)
        c3.metric("SELL signals", sell_count)
        c4.metric("NO SIGNAL", no_signal_count)

        st.subheader("⭐ Priority coins")

        actionable = results[
            results["Signal"].isin(["BUY", "SELL"])
        ].copy()

        if actionable.empty:
            st.warning(
                "No coins currently meet the signal threshold. "
                "This is not a reason to force a trade."
            )
        else:
            display_columns = [
                "Coin",
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

            display_columns = [
                col for col in display_columns
                if col in actionable.columns
            ]

            st.dataframe(
                actionable[display_columns],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("All scanned coins")

        filter_signal = st.multiselect(
            "Filter by signal",
            options=["BUY", "SELL", "NO SIGNAL"],
            default=["BUY", "SELL", "NO SIGNAL"],
        )

        filtered = results[
            results["Signal"].isin(filter_signal)
        ].copy()

        all_columns = [
            "Coin",
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
            "Last Candle (UTC)",
        ]

        all_columns = [
            col for col in all_columns if col in filtered.columns
        ]

        st.dataframe(
            filtered[all_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Coin details")

        selected_symbol = st.selectbox(
            "Choose a coin",
            options=results["Symbol"].tolist(),
        )

        selected_row = results[
            results["Symbol"] == selected_symbol
        ].iloc[0]

        left, right = st.columns(2)

        with left:
            st.write(f"**Signal:** {selected_row['Signal']}")
            st.write(f"**Price:** {selected_row['Price']}")
            st.write(f"**RSI:** {selected_row.get('RSI', np.nan)}")
            st.write(f"**ADX:** {selected_row.get('ADX', np.nan)}")

        with right:
            st.write(
                f"**24h change:** "
                f"{selected_row['24h Change (%)']:.2f}%"
            )
            st.write(
                f"**24h volume:** "
                f"{selected_row['24h Volume (USDT)']:,.0f} USDT"
            )
            st.write(
                f"**Confidence score:** "
                f"{selected_row.get('Confidence', 0)}"
            )

        st.write("**Signal explanation**")
        st.write(selected_row.get("Reasons", ""))

        st.subheader("Price chart")

        try:
            exchange = get_exchange()
            chart_df = fetch_candles(
                exchange,
                selected_symbol,
                timeframe,
                requested=candle_count,
            )

            chart_df = add_indicators(chart_df)
            chart_df = chart_df.set_index("timestamp")

            chart_columns = [
                "close", "EMA20", "EMA50",
                "BB_UPPER", "BB_LOWER",
            ]

            st.line_chart(chart_df[chart_columns])

        except Exception as exc:
            st.warning(f"Chart could not be loaded: {exc}")

        csv_data = results.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download scan results (CSV)",
            data=csv_data,
            file_name="crypto_signal_scan.csv",
            mime="text/csv",
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Important: This application is for informational and educational "
    "analysis only. It does not execute orders or connect to a Binomo "
    "account. Technical indicators and machine-learning predictions can "
    "be wrong. Public Binance market prices may differ from OTC prices, "
    "and no signal guarantees profit. Verify market conditions independently."
)
