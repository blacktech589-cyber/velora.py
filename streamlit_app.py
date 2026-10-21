
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Crypto Signal Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Crypto Signal Scanner")
st.caption("Binance Spot • BUY/SELL • Technical Analysis • Optional TensorFlow LSTM")
st.warning(
    "Analysis only: no order execution. Public Binance prices may differ "
    "from Binomo OTC prices. Signals are not guarantees of profit."
)

# -------------------------------------------------------
# OPTIONAL TENSORFLOW
# -------------------------------------------------------
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    Sequential = LSTM = Dense = Dropout = EarlyStopping = None
    TF_AVAILABLE = False

# -------------------------------------------------------
# BINANCE DATA VIA CCXT
# -------------------------------------------------------
try:
    import ccxt
except ImportError:
    st.error(
        "ccxt is not installed. Add ccxt to requirements.txt, "
        "then redeploy the app."
    )
    st.code(
        "streamlit\npandas\nnumpy\nccxt\nscikit-learn\ntensorflow",
        language="text",
    )
    st.stop()


@st.cache_resource
def create_exchange():
    return ccxt.binance({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })


exchange = create_exchange()


# -------------------------------------------------------
# UNIVERSE / COIN LIST
# -------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_top_coins(limit=50):
    ex = create_exchange()
    markets = ex.load_markets()
    tickers = ex.fetch_tickers()

    rows = []

    for symbol, ticker in tickers.items():
        market = markets.get(symbol, {})

        if not market:
            continue

        if (
            market.get("spot") is not True
            or market.get("active") is False
            or not symbol.endswith("/USDT")
        ):
            continue

        base = symbol.split("/")[0]

        # Exclude leveraged tokens and common stablecoins.
        excluded_suffixes = ("UP", "DOWN", "BULL", "BEAR")
        excluded_bases = {
            "USDC", "FDUSD", "TUSD", "USDP", "DAI",
            "EUR", "AEUR", "USTC",
        }

        if base in excluded_bases:
            continue

        if base.endswith(excluded_suffixes):
            continue

        price = ticker.get("last")
        quote_volume = ticker.get("quoteVolume")

        if price is None or quote_volume is None:
            continue

        if not np.isfinite(price) or not np.isfinite(quote_volume):
            continue

        if price <= 0 or quote_volume <= 0:
            continue

        rows.append({
            "Coin": symbol,
            "Price": float(price),
            "24h Volume (USDT)": float(quote_volume),
            "24h Change (%)": float(ticker.get("percentage") or 0),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result.sort_values("24h Volume (USDT)", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


# -------------------------------------------------------
# CANDLES
# -------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_candles(symbol, timeframe="15m", limit=300):
    ex = create_exchange()
    rows = ex.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit,
    )

    df = pd.DataFrame(
        rows,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    if df.empty:
        raise ValueError("Exchange returned no candles.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], unit="ms", utc=True
    )

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Exclude the currently forming candle where possible.
    if len(df) > 2:
        timeframe_ms = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }[timeframe]

        now_ms = int(time.time() * 1000)
        last_open_ms = int(df.iloc[-1]["Timestamp"].timestamp() * 1000)

        if now_ms < last_open_ms + timeframe_ms:
            df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise ValueError("Not enough completed candles.")

    return df.reset_index(drop=True)


# -------------------------------------------------------
# INDICATORS
# -------------------------------------------------------
def add_indicators(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["EMA15"] = close.ewm(span=15, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands 20, 2
    df["BB_Middle"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std

    # RSI 14
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
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)

    df["Stoch_K"] = (
        100 * (close - low14) / denom
    ).fillna(50)

    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean().fillna(50)

    # Volume confirmation
    df["Volume_MA20"] = volume.rolling(20).mean()

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

    df["ATR"] = tr.rolling(14).mean()

    # Directional movement / ADX
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
    )

    atr_smoothed = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean().replace(0, np.nan)

    plus_di = (
        100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smoothed
    )

    minus_di = (
        100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smoothed
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["Plus_DI"] = plus_di.fillna(0)
    df["Minus_DI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)

    return df


# -------------------------------------------------------
# SIGNAL ENGINE
# -------------------------------------------------------
def analyze_coin(df):
    row = df.iloc[-1]

    buy = 0
    sell = 0
    reasons = []

    def vote(name, is_buy, is_sell):
        nonlocal buy, sell

        if bool(is_buy):
            buy += 1
            reasons.append(f"{name}: BUY")
        elif bool(is_sell):
            sell += 1
            reasons.append(f"{name}: SELL")
        else:
            reasons.append(f"{name}: NEUTRAL")

    vote("EMA trend", row.EMA15 > row.EMA50, row.EMA15 < row.EMA50)

    vote(
        "Bollinger Bands",
        row.Close < row.BB_Lower,
        row.Close > row.BB_Upper,
    )

    vote("RSI", row.RSI < 30, row.RSI > 70)

    vote(
        "MACD",
        row.MACD > row.MACD_Signal,
        row.MACD < row.MACD_Signal,
    )

    vote(
        "Stochastic",
        row.Stoch_K < 20 and row.Stoch_K > row.Stoch_D,
        row.Stoch_K > 80 and row.Stoch_K < row.Stoch_D,
    )

    vote(
        "ADX + DI",
        row.ADX >= 20 and row.Plus_DI > row.Minus_DI,
        row.ADX >= 20 and row.Minus_DI > row.Plus_DI,
    )

    vote(
        "Volume",
        row.Volume > row.Volume_MA20 and row.Close > row.Open,
        row.Volume > row.Volume_MA20 and row.Close < row.Open,
    )

    if buy > sell:
        raw_signal = "BUY"
    elif sell > buy:
        raw_signal = "SELL"
    else:
        raw_signal = "NO SIGNAL"

    # Conservative default: require at least 4 votes and a margin.
    if max(buy, sell) < 4 or abs(buy - sell) < 2:
        signal = "NO SIGNAL"
    else:
        signal = raw_signal

    total = buy + sell
    strength = (
        round(100 * max(buy, sell) / total)
        if total > 0
        else 0
    )

    return {
        "Signal": signal,
        "BUY votes": buy,
        "SELL votes": sell,
        "Strength (%)": strength,
        "RSI": round(float(row.RSI), 2),
        "EMA15": float(row.EMA15),
        "EMA50": float(row.EMA50),
        "Price": float(row.Close),
        "Volume": float(row.Volume),
        "Reasons": reasons,
    }


# -------------------------------------------------------
# OPTIONAL TENSORFLOW LSTM
# -------------------------------------------------------
def lstm_predict(close_series, epochs=5, lookback=30):
    if not TF_AVAILABLE or MinMaxScaler is None:
        return None, (
            "TensorFlow or scikit-learn is unavailable. "
            "The technical scanner still works without LSTM."
        )

    values = np.asarray(close_series, dtype=float).reshape(-1, 1)

    if len(values) < lookback + 100:
        return None, "Not enough candles for LSTM training."

    split = int(len(values) * 0.8)

    scaler = MinMaxScaler()
    scaler.fit(values[:split])
    scaled = scaler.transform(values)

    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    train_end = split - lookback

    if train_end < 30:
        return None, "Not enough training samples."

    X_train = X[:train_end].reshape(-1, lookback, 1)
    y_train = y[:train_end]
    X_val = X[train_end:].reshape(-1, lookback, 1)
    y_val = y[train_end:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")

    fit_args = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": 32,
        "verbose": 0,
        "callbacks": [
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
    }

    if len(X_val):
        fit_args["validation_data"] = (X_val, y_val)

    model.fit(**fit_args)

    current = float(values[-1, 0])
    last = scaled[-lookback:].reshape(1, lookback, 1)
    prediction = float(
        scaler.inverse_transform(model.predict(last, verbose=0))[0, 0]
    )

    if prediction > current:
        direction = "BUY"
    elif prediction < current:
        direction = "SELL"
    else:
        direction = "NO SIGNAL"

    return {
        "Current price": current,
        "LSTM estimate": prediction,
        "LSTM direction": direction,
    }, None


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("⚙️ Scanner settings")

timeframe = st.sidebar.selectbox(
    "Candlestick timeframe",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=1,
)

coin_limit = st.sidebar.slider(
    "Coins to scan by 24h volume",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
)

min_volume = st.sidebar.number_input(
    "Minimum 24h quote volume (USDT)",
    min_value=0.0,
    value=1_000_000.0,
    step=500_000.0,
)

enable_lstm = st.sidebar.checkbox(
    "Use TensorFlow LSTM confirmation",
    value=False,
)

lstm_epochs = st.sidebar.slider(
    "LSTM training epochs",
    min_value=2,
    max_value=15,
    value=5,
    disabled=not enable_lstm,
)

st.sidebar.caption(
    "Top-volume coins are screened first. Signals can be wrong."
)


# -------------------------------------------------------
# INVESTMENT CALCULATOR
# -------------------------------------------------------
st.header("💰 Investment calculator")

cc1, cc2, cc3 = st.columns(3)

with cc1:
    currency = st.selectbox(
        "Display currency",
        ["USDT", "USD", "EUR", "TRY"],
    )

with cc2:
    stake = st.number_input(
        "Investment amount",
        min_value=0.01,
        value=1.25,
        step=0.25,
    )

with cc3:
    payout = st.number_input(
        "Payout (%)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )

c1, c2, c3 = st.columns(3)

c1.metric("Stake", f"{stake:.2f} {currency}")
c2.metric("Profit if correct", f"{stake * payout / 100:.2f} {currency}")
c3.metric("Loss if incorrect", f"{stake:.2f} {currency}")

st.caption(
    "This is a mathematical estimate only. Currency changes the display "
    "label and does not perform a real conversion."
)

st.divider()


# -------------------------------------------------------
# PRIORITY COIN PANEL
# -------------------------------------------------------
st.header("⭐ Priority coins")
st.write(
    "The panel ranks screened coins by signal quality and volume. "
    "A coin is not automatically a BUY just because it appears first."
)

if st.button("🔎 Scan coins now", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading Binance Spot markets and 24h volumes..."):
            universe = get_top_coins(coin_limit)

        if universe.empty:
            st.error("No eligible USDT Spot markets were returned.")
            st.stop()

        universe = universe[
            universe["24h Volume (USDT)"] >= min_volume
        ].copy()

        if universe.empty:
            st.warning(
                "No coins meet the volume threshold. Lower the minimum volume."
            )
            st.stop()

        results = []
        progress = st.progress(0)
        status = st.empty()

        total = len(universe)

        for i, item in enumerate(universe.itertuples(index=False), start=1):
            symbol = item.Coin
            status.write(f"Scanning {symbol} ({i}/{total})")

            try:
                candles = get_candles(symbol, timeframe, 300)
                df = add_indicators(candles)
                analysis = analyze_coin(df)

                lstm_direction = "NOT USED"
                lstm_estimate = np.nan

                if enable_lstm and TF_AVAILABLE:
                    try:
                        pred, err = lstm_predict(
                            df["Close"],
                            epochs=lstm_epochs,
                        )

                        if pred:
                            lstm_direction = pred["LSTM direction"]
                            lstm_estimate = pred["LSTM estimate"]
                        else:
                            lstm_direction = "UNAVAILABLE"
                    except Exception:
                        lstm_direction = "ERROR"

                # Conservative LSTM confirmation.
                final_signal = analysis["Signal"]

                if enable_lstm:
                    if lstm_direction in ("BUY", "SELL"):
                        if final_signal != lstm_direction:
                            final_signal = "NO SIGNAL"
                    else:
                        final_signal = "NO SIGNAL"

                results.append({
                    "Coin": symbol,
                    "Signal": final_signal,
                    "BUY votes": analysis["BUY votes"],
                    "SELL votes": analysis["SELL votes"],
                    "Strength (%)": analysis["Strength (%)"],
                    "Price": analysis["Price"],
                    "24h Volume (USDT)": item._2,
                    "24h Change (%)": item._3,
                    "RSI": analysis["RSI"],
                    "LSTM": lstm_direction,
                    "LSTM estimate": lstm_estimate,
                    "Reasons": " | ".join(analysis["Reasons"]),
                })

            except Exception as exc:
                results.append({
                    "Coin": symbol,
                    "Signal": "DATA ERROR",
                    "BUY votes": 0,
                    "SELL votes": 0,
                    "Strength (%)": 0,
                    "Price": item.Price,
                    "24h Volume (USDT)": item._2,
                    "24h Change (%)": item._3,
                    "RSI": np.nan,
                    "LSTM": "NOT AVAILABLE",
                    "LSTM estimate": np.nan,
                    "Reasons": str(exc)[:160],
                })

            progress.progress(i / total)

        status.empty()
        progress.empty()

        result_df = pd.DataFrame(results)

        # Prioritize actionable signals, then strength and liquidity.
        signal_order = {
            "BUY": 0,
            "SELL": 1,
            "NO SIGNAL": 2,
            "DATA ERROR": 3,
        }

        result_df["_order"] = result_df["Signal"].map(signal_order).fillna(4)

        result_df = result_df.sort_values(
            ["_order", "Strength (%)", "24h Volume (USDT)"],
            ascending=[True, False, False],
        ).drop(columns="_order")

        st.session_state["scan_results"] = result_df

    except Exception as exc:
        st.error(
            "Could not scan Binance markets. The service may be unavailable "
            "or access may be restricted from this network."
        )
        st.code(str(exc))


# -------------------------------------------------------
# DISPLAY SCAN RESULTS
# -------------------------------------------------------
if "scan_results" in st.session_state:
    result_df = st.session_state["scan_results"]

    st.subheader("🏆 First-priority coins")

    actionable = result_df[
        result_df["Signal"].isin(["BUY", "SELL"])
    ].head(10)

    if actionable.empty:
        st.info(
            "No coins currently meet the signal rules. "
            "The app will not force a BUY or SELL signal."
        )
    else:
        st.dataframe(
            actionable[
                [
                    "Coin",
                    "Signal",
                    "Strength (%)",
                    "BUY votes",
                    "SELL votes",
                    "Price",
                    "24h Volume (USDT)",
                    "24h Change (%)",
                    "RSI",
                    "LSTM",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 All scanned coins")

    signal_filter = st.selectbox(
        "Filter results",
        ["ALL", "BUY", "SELL", "NO SIGNAL", "DATA ERROR"],
    )

    display_df = result_df.copy()

    if signal_filter != "ALL":
        display_df = display_df[
            display_df["Signal"] == signal_filter
        ]

    st.dataframe(
        display_df.drop(columns=["Reasons", "LSTM estimate"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

    selected_coin = st.selectbox(
        "View signal details",
        result_df["Coin"].tolist(),
    )

    selected_row = result_df[
        result_df["Coin"] == selected_coin
    ].iloc[0]

    st.write(f"**Coin:** {selected_coin}")
    st.write(f"**Signal:** {selected_row['Signal']}")
    st.write(f"**Reason details:** {selected_row['Reasons']}")

    st.download_button(
        "⬇️ Download scan results CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="crypto_signal_scan.csv",
        mime="text/csv",
    )

st.divider()

st.caption(
    "Public market data only. No Binance keys are required. No orders are "
    "placed. TensorFlow estimates are experimental and are not proof of "
    "future price direction."
)
