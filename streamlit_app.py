
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Binomo Signal Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Currency Pair Signal Scanner")
st.caption("Currency pairs • Technical analysis • Optional LSTM")
st.warning(
    "Analysis only. This app does not connect to your Binomo account "
    "or execute trades. External market prices may differ from Binomo OTC."
)

# --------------------------------------------------
# OPTIONAL LIBRARIES
# --------------------------------------------------
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    Sequential = None
    LSTM = Dense = Dropout = EarlyStopping = None


# --------------------------------------------------
# CURRENCY PAIRS
# --------------------------------------------------
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/CHF": "GBPCHF=X",
    "GBP/CAD": "GBPCAD=X",
    "GBP/AUD": "GBPAUD=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "AUD/CHF": "AUDCHF=X",
    "AUD/NZD": "AUDNZD=X",
    "CAD/JPY": "CADJPY=X",
    "CAD/CHF": "CADCHF=X",
    "NZD/JPY": "NZDJPY=X",
    "NZD/CAD": "NZDCAD=X",
    "NZD/CHF": "NZDCHF=X",
}

CURRENCIES = ["USD", "EUR", "GBP", "TRY", "CAD", "AUD", "JPY", "CHF"]


# --------------------------------------------------
# FETCH MARKET DATA
# --------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker, interval, period):
    if yf is None:
        raise ImportError(
            "yfinance is missing. Install it using: "
            "python -m pip install yfinance"
        )

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(
            "No market data returned. Try another currency pair, "
            "timeframe, or historical period."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    if len(data) < 60:
        raise ValueError(
            f"Only {len(data)} candles are available. "
            "At least 60 are needed."
        )

    return data


# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------
def add_indicators(data):
    df = data.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Exponential moving averages
    df["EMA15"] = close.ewm(span=15, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands
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

    # Stochastic oscillator
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    denom = (highest - lowest).replace(0, np.nan)

    df["Stoch_K"] = (
        100 * (close - lowest) / denom
    ).fillna(50)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean().fillna(50)

    # True range and ATR
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

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

    atr_smooth = true_range.ewm(
        alpha=1 / 14, adjust=False
    ).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr_smooth

    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr_smooth

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["Plus_DI"] = plus_di.fillna(0)
    df["Minus_DI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(
        alpha=1 / 14, adjust=False
    ).mean().fillna(0)

    # CCI
    typical = (high + low + close) / 3
    typical_mean = typical.rolling(20).mean()

    mean_deviation = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    df["CCI"] = (
        (typical - typical_mean)
        / (0.015 * mean_deviation.replace(0, np.nan))
    ).fillna(0)

    # Williams %R
    rolling_high = high.rolling(14).max()
    rolling_low = low.rolling(14).min()
    wr_range = (rolling_high - rolling_low).replace(0, np.nan)

    df["Williams_R"] = (
        -100 * (rolling_high - close) / wr_range
    ).fillna(-50)

    # Candle direction
    df["Candle"] = np.where(
        close > df["Open"],
        "GREEN",
        np.where(close < df["Open"], "RED", "DOJI"),
    )

    return df


# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------
def get_signal(row):
    buy = 0
    sell = 0
    details = []

    def add_vote(label, buy_condition, sell_condition):
        nonlocal buy, sell

        if buy_condition:
            buy += 1
            details.append(f"{label}: BUY")
        elif sell_condition:
            sell += 1
            details.append(f"{label}: SELL")
        else:
            details.append(f"{label}: NEUTRAL")

    add_vote(
        "EMA 15/50",
        row["EMA15"] > row["EMA50"],
        row["EMA15"] < row["EMA50"],
    )

    add_vote(
        "Bollinger Bands",
        row["Close"] < row["BB_Lower"],
        row["Close"] > row["BB_Upper"],
    )

    add_vote(
        "RSI",
        row["RSI"] < 30,
        row["RSI"] > 70,
    )

    add_vote(
        "MACD",
        row["MACD"] > row["MACD_Signal"],
        row["MACD"] < row["MACD_Signal"],
    )

    add_vote(
        "Stochastic",
        row["Stoch_K"] < 20 and row["Stoch_K"] > row["Stoch_D"],
        row["Stoch_K"] > 80 and row["Stoch_K"] < row["Stoch_D"],
    )

    add_vote(
        "ADX + DI",
        row["ADX"] >= 20 and row["Plus_DI"] > row["Minus_DI"],
        row["ADX"] >= 20 and row["Minus_DI"] > row["Plus_DI"],
    )

    add_vote(
        "CCI",
        row["CCI"] < -100,
        row["CCI"] > 100,
    )

    add_vote(
        "Williams %R",
        row["Williams_R"] < -80,
        row["Williams_R"] > -20,
    )

    if buy > sell:
        signal = "BUY"
    elif sell > buy:
        signal = "SELL"
    else:
        signal = "NO SIGNAL"

    return signal, buy, sell, details


# --------------------------------------------------
# OPTIONAL LSTM PREDICTION
# --------------------------------------------------
def run_lstm(close_series, lookback=30, epochs=5):
    if Sequential is None or MinMaxScaler is None:
        return None, (
            "LSTM unavailable. Install compatible versions of "
            "tensorflow and scikit-learn."
        )

    values = close_series.astype(float).values.reshape(-1, 1)

    if len(values) < lookback + 100:
        return None, "Not enough historical candles for LSTM."

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

    model = Sequential(
        [
            LSTM(32, return_sequences=True,
                 input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    fit_args = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": 32,
        "verbose": 0,
        "callbacks": callbacks,
    }

    if len(X_val) > 0:
        fit_args["validation_data"] = (X_val, y_val)

    model.fit(**fit_args)

    last_sequence = scaled[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction = float(
        scaler.inverse_transform(prediction_scaled)[0, 0]
    )
    current = float(values[-1, 0])

    if prediction > current:
        signal = "BUY"
    elif prediction < current:
        signal = "SELL"
    else:
        signal = "NO SIGNAL"

    return {
        "current": current,
        "prediction": prediction,
        "signal": signal,
    }, None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

pair = st.sidebar.selectbox("Currency pair", list(PAIRS.keys()))

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1h", "1d"],
)

period_map = {
    "5m": ["5d", "1mo"],
    "15m": ["5d", "1mo", "3mo"],
    "30m": ["5d", "1mo", "3mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
}

period = st.sidebar.selectbox(
    "Historical period",
    period_map[timeframe],
)

min_votes = st.sidebar.slider(
    "Minimum votes for signal",
    min_value=1,
    max_value=8,
    value=4,
)

enable_lstm = st.sidebar.checkbox("Enable optional LSTM", value=False)

lstm_epochs = st.sidebar.slider(
    "LSTM epochs",
    min_value=2,
    max_value=20,
    value=5,
    disabled=not enable_lstm,
)


# --------------------------------------------------
# INVESTMENT CALCULATOR
# --------------------------------------------------
st.header("💰 Investment calculator")

col1, col2, col3 = st.columns(3)

with col1:
    account_currency = st.selectbox(
        "Account currency",
        CURRENCIES,
        index=0,
    )

with col2:
    investment = st.number_input(
        "Investment amount",
        min_value=0.01,
        value=1.25,
        step=0.25,
    )

with col3:
    payout = st.number_input(
        "Payout (%)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )

profit = investment * payout / 100
loss = investment

m1, m2, m3 = st.columns(3)

m1.metric(
    "Calculated stake",
    f"{investment:.2f} {account_currency}",
)

m2.metric(
    "Profit if correct",
    f"{profit:.2f} {account_currency}",
)

m3.metric(
    "Loss if incorrect",
    f"{loss:.2f} {account_currency}",
)

st.caption(
    "This is a mathematical estimate based on your entered payout. "
    "It is not a minimum deposit and does not guarantee profit. "
    "Currency selection changes the display label only; no conversion occurs."
)

st.divider()


# --------------------------------------------------
# ANALYSIS PANEL
# --------------------------------------------------
st.header("📈 Market analysis")

if yf is None:
    st.error(
        "yfinance is missing. Add it to requirements.txt and redeploy."
    )
    st.code(
        "streamlit\npandas\nnumpy\nyfinance\nscikit-learn\ntensorflow",
        language="text",
    )
    st.stop()

if st.button("🔍 Analyze market", type="primary", use_container_width=True):
    try:
        with st.spinner("Downloading market data..."):
            raw = fetch_data(PAIRS[pair], timeframe, period)
            df = add_indicators(raw)
            latest = df.iloc[-1]

        tech_signal, buy_votes, sell_votes, details = get_signal(latest)

        if buy_votes < min_votes and sell_votes < min_votes:
            final_signal = "NO SIGNAL"
        else:
            final_signal = tech_signal

        lstm_result = None
        lstm_error = None

        if enable_lstm:
            with st.spinner("Training LSTM model..."):
                lstm_result, lstm_error = run_lstm(
                    df["Close"],
                    lookback=30,
                    epochs=lstm_epochs,
                )

            if lstm_result is not None:
                if final_signal in ("BUY", "SELL"):
                    if lstm_result["signal"] != final_signal:
                        final_signal = "NO SIGNAL"
                else:
                    final_signal = "NO SIGNAL"

        st.subheader(f"{pair} | {timeframe}")

        a, b, c, d = st.columns(4)

        a.metric("Latest close", f"{float(latest['Close']):.6f}")
        b.metric("BUY votes", str(buy_votes))
        c.metric("SELL votes", str(sell_votes))
        d.metric("RSI (14)", f"{float(latest['RSI']):.2f}")

        if final_signal == "BUY":
            st.success("🟢 BUY")
        elif final_signal == "SELL":
            st.error("🔴 SELL")
        else:
            st.info("⚪ NO SIGNAL — wait for clearer conditions")

        st.write(f"**Last data timestamp:** {df.index[-1]}")
        st.write(
            "**Signal source:** Technical indicators"
            + (" + LSTM confirmation" if enable_lstm and lstm_result else "")
        )

        st.caption(
            "The latest available candle may not be closed. "
            "Signals can change as new data arrives."
        )

        st.subheader("Price chart")

        chart_columns = [
            "Close",
            "EMA15",
            "EMA20",
            "EMA50",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
        ]

        st.line_chart(df[chart_columns].tail(200))

        st.subheader("Technical indicator values")

        indicators = pd.DataFrame(
            {
                "Indicator": [
                    "EMA 15",
                    "EMA 20",
                    "EMA 50",
                    "Bollinger Upper",
                    "Bollinger Middle",
                    "Bollinger Lower",
                    "RSI",
                    "MACD",
                    "MACD Signal",
                    "Stochastic %K",
                    "Stochastic %D",
                    "ADX",
                    "+DI",
                    "-DI",
                    "CCI",
                    "Williams %R",
                    "ATR",
                    "Candle",
                ],
                "Value": [
                    latest["EMA15"],
                    latest["EMA20"],
                    latest["EMA50"],
                    latest["BB_Upper"],
                    latest["BB_Middle"],
                    latest["BB_Lower"],
                    latest["RSI"],
                    latest["MACD"],
                    latest["MACD_Signal"],
                    latest["Stoch_K"],
                    latest["Stoch_D"],
                    latest["ADX"],
                    latest["Plus_DI"],
                    latest["Minus_DI"],
                    latest["CCI"],
                    latest["Williams_R"],
                    latest["ATR"],
                    latest["Candle"],
                ],
            }
        )

        st.dataframe(
            indicators,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Signal breakdown")

        for detail in details:
            st.write(f"• {detail}")

        st.write(
            f"**BUY:** {buy_votes}/8 | "
            f"**SELL:** {sell_votes}/8 | "
            f"**Minimum votes:** {min_votes}"
        )

        if enable_lstm:
            st.subheader("🧠 LSTM estimate")

            if lstm_result is None:
                st.warning(lstm_error or "LSTM prediction unavailable.")
            else:
                x, y, z = st.columns(3)

                x.metric("Latest close", f"{lstm_result['current']:.6f}")
                y.metric("Model estimate", f"{lstm_result['prediction']:.6f}")
                z.metric("LSTM direction", lstm_result["signal"])

                st.caption(
                    "LSTM uses historical closing prices to estimate a value. "
                    "It is not a guarantee of the next candle's direction."
                )

        with st.expander("Show latest 30 candles"):
            st.dataframe(
                df.tail(30).sort_index(ascending=False),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            "Analysis failed. Check the internet connection, selected pair, "
            "data provider availability, and installed dependencies."
        )
        st.code(str(exc))

else:
    st.info(
        "Select a currency pair and timeframe, then click Analyze market."
    )

st.divider()

st.caption(
    "Risk notice: Signals may be wrong. Historical data and technical "
    "indicators do not guarantee future results. No trades are executed."
)
