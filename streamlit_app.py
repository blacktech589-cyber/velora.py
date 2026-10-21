
import sys
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Currency Pair Signal Scanner",
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
# DEPENDENCY CHECK
# --------------------------------------------------
st.sidebar.header("Environment")

st.sidebar.caption(f"Python: {sys.version.split()[0]}")

if importlib.util.find_spec("yfinance") is None:
    st.error(
        "The yfinance package is not installed in this Python environment. "
        "Add yfinance to requirements.txt and redeploy the app."
    )
    st.code(
        "streamlit\npandas\nnumpy\nyfinance\nscikit-learn\ntensorflow",
        language="text",
    )
    st.stop()

import yfinance as yf

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
# DATA DOWNLOAD
# --------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def download_data(ticker, interval, period):
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
            "No market data returned. Try another pair, timeframe, or period."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if not all(col in data.columns for col in required):
        raise ValueError("Market data does not contain OHLC columns.")

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    if len(data) < 60:
        raise ValueError(
            f"Only {len(data)} candles returned. At least 60 are required."
        )

    return data


# --------------------------------------------------
# INDICATORS
# --------------------------------------------------
def calculate_indicators(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
    df["EMA15"] = close.ewm(span=15, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    df["BB_Middle"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std

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
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # Stochastic
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    denom = (highest - lowest).replace(0, np.nan)

    df["Stoch_K"] = (100 * (close - lowest) / denom).fillna(50)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean().fillna(50)

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

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

    smoothed_tr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    smoothed_tr = smoothed_tr.replace(0, np.nan)

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / smoothed_tr
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / smoothed_tr
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["Plus_DI"] = plus_di.fillna(0)
    df["Minus_DI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)

    # CCI
    typical = (high + low + close) / 3
    typical_mean = typical.rolling(20).mean()

    deviation = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    df["CCI"] = (
        (typical - typical_mean)
        / (0.015 * deviation.replace(0, np.nan))
    ).fillna(0)

    # Williams %R
    rolling_high = high.rolling(14).max()
    rolling_low = low.rolling(14).min()
    wr_denom = (rolling_high - rolling_low).replace(0, np.nan)

    df["Williams_R"] = (
        -100 * (rolling_high - close) / wr_denom
    ).fillna(-50)

    df["Candle"] = np.where(
        close > df["Open"],
        "GREEN",
        np.where(close < df["Open"], "RED", "DOJI"),
    )

    return df


# --------------------------------------------------
# SIGNAL VOTING
# --------------------------------------------------
def technical_signal(row):
    buy = 0
    sell = 0
    reasons = []

    def vote(name, buy_condition, sell_condition):
        nonlocal buy, sell

        if buy_condition:
            buy += 1
            reasons.append(f"{name}: BUY")
        elif sell_condition:
            sell += 1
            reasons.append(f"{name}: SELL")
        else:
            reasons.append(f"{name}: NEUTRAL")

    vote(
        "EMA 15/50",
        row["EMA15"] > row["EMA50"],
        row["EMA15"] < row["EMA50"],
    )

    vote(
        "Bollinger Bands",
        row["Close"] < row["BB_Lower"],
        row["Close"] > row["BB_Upper"],
    )

    vote("RSI", row["RSI"] < 30, row["RSI"] > 70)

    vote(
        "MACD",
        row["MACD"] > row["MACD_Signal"],
        row["MACD"] < row["MACD_Signal"],
    )

    vote(
        "Stochastic",
        row["Stoch_K"] < 20 and row["Stoch_K"] > row["Stoch_D"],
        row["Stoch_K"] > 80 and row["Stoch_K"] < row["Stoch_D"],
    )

    vote(
        "ADX + DI",
        row["ADX"] >= 20 and row["Plus_DI"] > row["Minus_DI"],
        row["ADX"] >= 20 and row["Minus_DI"] > row["Plus_DI"],
    )

    vote("CCI", row["CCI"] < -100, row["CCI"] > 100)

    vote(
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

    return signal, buy, sell, reasons


# --------------------------------------------------
# OPTIONAL LSTM
# --------------------------------------------------
def predict_lstm(close_series, epochs=5, lookback=30):
    if Sequential is None or MinMaxScaler is None:
        return None, (
            "LSTM is unavailable. Check tensorflow and scikit-learn installation."
        )

    values = close_series.astype(float).values.reshape(-1, 1)

    if len(values) < lookback + 100:
        return None, "Not enough historical data for LSTM."

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
        return None, "Not enough LSTM training samples."

    X_train = X[:train_end].reshape(-1, lookback, 1)
    y_train = y[:train_end]

    X_val = X[train_end:].reshape(-1, lookback, 1)
    y_val = y[train_end:]

    model = Sequential(
        [
            LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
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

    kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": 32,
        "callbacks": callbacks,
        "verbose": 0,
    }

    if len(X_val) > 0:
        kwargs["validation_data"] = (X_val, y_val)

    model.fit(**kwargs)

    last_sequence = scaled[-lookback:].reshape(1, lookback, 1)
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted = float(scaler.inverse_transform(predicted_scaled)[0, 0])
    current = float(values[-1, 0])

    if predicted > current:
        direction = "BUY"
    elif predicted < current:
        direction = "SELL"
    else:
        direction = "NO SIGNAL"

    return {
        "current": current,
        "predicted": predicted,
        "signal": direction,
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

period_options = {
    "5m": ["5d", "1mo"],
    "15m": ["5d", "1mo", "3mo"],
    "30m": ["5d", "1mo", "3mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
}

period = st.sidebar.selectbox(
    "Historical period",
    period_options[timeframe],
)

min_votes = st.sidebar.slider(
    "Minimum votes",
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

c1, c2, c3 = st.columns(3)

with c1:
    currency = st.selectbox("Account currency", CURRENCIES, index=0)

with c2:
    stake = st.number_input(
        "Investment amount",
        min_value=0.01,
        value=1.25,
        step=0.25,
    )

with c3:
    payout = st.number_input(
        "Payout (%)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )

profit = stake * payout / 100
loss = stake

r1, r2, r3 = st.columns(3)
r1.metric("Calculated stake", f"{stake:.2f} {currency}")
r2.metric("Profit if correct", f"{profit:.2f} {currency}")
r3.metric("Loss if incorrect", f"{loss:.2f} {currency}")

st.caption(
    "Mathematical estimate based on the payout entered. "
    "Currency selection changes the display label only; it does not convert money."
)

st.divider()


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
st.header("📈 Market analysis")

if st.button("🔍 Analyze market", type="primary", use_container_width=True):
    try:
        with st.spinner("Downloading public market data..."):
            raw = download_data(PAIRS[pair], timeframe, period)
            df = calculate_indicators(raw)
            latest = df.iloc[-1]

        tech, buy_votes, sell_votes, reasons = technical_signal(latest)

        final_signal = tech
        if buy_votes < min_votes and sell_votes < min_votes:
            final_signal = "NO SIGNAL"

        lstm_result = None
        lstm_error = None

        if enable_lstm:
            with st.spinner("Training LSTM model..."):
                lstm_result, lstm_error = predict_lstm(
                    df["Close"],
                    epochs=lstm_epochs,
                )

            if lstm_result is not None:
                if final_signal in ("BUY", "SELL"):
                    if lstm_result["signal"] != final_signal:
                        final_signal = "NO SIGNAL"
                else:
                    final_signal = "NO SIGNAL"

        st.subheader(f"{pair} | {timeframe}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest close", f"{float(latest['Close']):.6f}")
        m2.metric("BUY votes", str(buy_votes))
        m3.metric("SELL votes", str(sell_votes))
        m4.metric("RSI", f"{float(latest['RSI']):.2f}")

        if final_signal == "BUY":
            st.success("🟢 BUY")
        elif final_signal == "SELL":
            st.error("🔴 SELL")
        else:
            st.info("⚪ NO SIGNAL")

        st.write(f"**Latest candle:** {df.index[-1]}")
        st.caption(
            "The latest candle may still be forming. Signals can change. "
            "Yahoo Finance prices may differ from Binomo OTC prices."
        )

        st.subheader("Price chart")
        chart_columns = [
            "Close", "EMA15", "EMA20", "EMA50",
            "BB_Upper", "BB_Middle", "BB_Lower",
        ]
        st.line_chart(df[chart_columns].tail(200))

        st.subheader("Technical indicators")
        table = pd.DataFrame(
            {
                "Indicator": [
                    "EMA 15", "EMA 20", "EMA 50",
                    "Bollinger Upper", "Bollinger Middle", "Bollinger Lower",
                    "RSI", "MACD", "MACD Signal",
                    "Stochastic %K", "Stochastic %D",
                    "ADX", "+DI", "-DI", "CCI", "Williams %R", "ATR",
                    "Candle",
                ],
                "Value": [
                    latest["EMA15"], latest["EMA20"], latest["EMA50"],
                    latest["BB_Upper"], latest["BB_Middle"], latest["BB_Lower"],
                    latest["RSI"], latest["MACD"], latest["MACD_Signal"],
                    latest["Stoch_K"], latest["Stoch_D"],
                    latest["ADX"], latest["Plus_DI"], latest["Minus_DI"],
                    latest["CCI"], latest["Williams_R"], latest["ATR"],
                    latest["Candle"],
                ],
            }
        )

        st.dataframe(table, use_container_width=True, hide_index=True)

        st.subheader("Signal breakdown")
        for reason in reasons:
            st.write("• " + reason)

        st.write(
            f"**BUY votes:** {buy_votes}/8 | "
            f"**SELL votes:** {sell_votes}/8 | "
            f"**Minimum votes:** {min_votes}"
        )

        if enable_lstm:
            st.subheader("🧠 LSTM prediction")

            if lstm_result is None:
                st.warning(lstm_error or "LSTM prediction unavailable.")
            else:
                x, y, z = st.columns(3)
                x.metric("Latest close", f"{lstm_result['current']:.6f}")
                y.metric("Model estimate", f"{lstm_result['predicted']:.6f}")
                z.metric("LSTM direction", lstm_result["signal"])

        with st.expander("Show last 30 candles"):
            st.dataframe(
                df.tail(30).sort_index(ascending=False),
                use_container_width=True,
            )

    except Exception as exc:
        st.error("Market analysis failed.")
        st.write(
            "Check the selected pair, data provider availability, "
            "internet connection, and installed dependencies."
        )
        st.code(str(exc))

else:
    st.info("Choose a currency pair and timeframe, then click Analyze market.")

st.divider()
st.caption(
    "Analysis only. Signals are estimates, not guarantees. "
    "No trades are executed."
)
