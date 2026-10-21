
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Currency Pair Signal Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Currency Pair Signal Scanner")
st.caption(
    "Currency pairs • Technical analysis • Optional LSTM"
)
st.warning(
    "Analysis only. This app does not connect to your Binomo account "
    "or execute trades. External market prices may differ from Binomo OTC."
)

# --------------------------------------------------
# OPTIONAL DEPENDENCIES
# --------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    MinMaxScaler = None
    SKLEARN_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    Sequential = None
    LSTM = Dense = Dropout = EarlyStopping = None
    TENSORFLOW_AVAILABLE = False


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
# MARKET DATA
# --------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(ticker, interval, period):
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Add yfinance to requirements.txt "
            "and redeploy the app."
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
            "No market data was returned. Try another pair or timeframe."
        )

    # yfinance can return MultiIndex columns for some ticker responses.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Market data is missing required columns: "
            + ", ".join(missing)
        )

    data = data[required].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    if len(data) < 60:
        raise ValueError(
            f"Only {len(data)} candles were returned. "
            "At least 60 candles are needed for this analysis."
        )

    return data


# --------------------------------------------------
# TECHNICAL INDICATORS
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

    # Bollinger Bands: 20-period, 2 standard deviations
    df["BB_Middle"] = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (2 * rolling_std)
    df["BB_Lower"] = df["BB_Middle"] - (2 * rolling_std)

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
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Stochastic oscillator
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)

    df["Stoch_K"] = (
        100 * (close - lowest_low) / denominator
    ).fillna(50)

    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean().fillna(50)

    # Average True Range
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

    atr_for_adx = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_for_adx.replace(0, np.nan)
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_for_adx.replace(0, np.nan)
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["Plus_DI"] = plus_di.fillna(0)
    df["Minus_DI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean().fillna(0)

    # CCI
    typical_price = (high + low + close) / 3
    tp_mean = typical_price.rolling(20).mean()

    mean_deviation = typical_price.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    df["CCI"] = (
        (typical_price - tp_mean)
        / (0.015 * mean_deviation.replace(0, np.nan))
    ).fillna(0)

    # Williams %R
    range_high = high.rolling(14).max()
    range_low = low.rolling(14).min()
    williams_range = (range_high - range_low).replace(0, np.nan)

    df["Williams_R"] = (
        -100 * (range_high - close) / williams_range
    ).fillna(-50)

    # Momentum
    df["Momentum"] = close - close.shift(10)

    # Candle direction
    df["Candle_Direction"] = np.where(
        close > df["Open"],
        "GREEN",
        np.where(close < df["Open"], "RED", "DOJI"),
    )

    return df


# --------------------------------------------------
# TECHNICAL SIGNALS
# --------------------------------------------------
def generate_technical_signal(row):
    buy_votes = 0
    sell_votes = 0
    reasons = []

    def vote(condition_buy, condition_sell, label):
        nonlocal buy_votes, sell_votes

        if condition_buy:
            buy_votes += 1
            reasons.append(f"{label}: BUY")
        elif condition_sell:
            sell_votes += 1
            reasons.append(f"{label}: SELL")
        else:
            reasons.append(f"{label}: NEUTRAL")

    # EMA trend
    vote(
        row["EMA15"] > row["EMA50"],
        row["EMA15"] < row["EMA50"],
        "EMA 15/50",
    )

    # Bollinger Bands: mean-reversion approach
    vote(
        row["Close"] < row["BB_Lower"],
        row["Close"] > row["BB_Upper"],
        "Bollinger Bands",
    )

    # RSI
    vote(
        row["RSI"] < 30,
        row["RSI"] > 70,
        "RSI",
    )

    # MACD
    vote(
        row["MACD"] > row["MACD_Signal"],
        row["MACD"] < row["MACD_Signal"],
        "MACD",
    )

    # Stochastic
    vote(
        row["Stoch_K"] < 20 and row["Stoch_K"] > row["Stoch_D"],
        row["Stoch_K"] > 80 and row["Stoch_K"] < row["Stoch_D"],
        "Stochastic",
    )

    # ADX direction only when trend strength is sufficient
    vote(
        row["ADX"] >= 20 and row["Plus_DI"] > row["Minus_DI"],
        row["ADX"] >= 20 and row["Minus_DI"] > row["Plus_DI"],
        "ADX + DI",
    )

    # CCI
    vote(
        row["CCI"] < -100,
        row["CCI"] > 100,
        "CCI",
    )

    # Williams %R
    vote(
        row["Williams_R"] < -80,
        row["Williams_R"] > -20,
        "Williams %R",
    )

    if buy_votes > sell_votes:
        signal = "BUY"
    elif sell_votes > buy_votes:
        signal = "SELL"
    else:
        signal = "NO SIGNAL"

    return signal, buy_votes, sell_votes, reasons


# --------------------------------------------------
# OPTIONAL LSTM
# --------------------------------------------------
def train_lstm_prediction(close_series, lookback=30, epochs=5):
    if not TENSORFLOW_AVAILABLE:
        return None, "TensorFlow is not installed."

    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn is not installed."

    values = close_series.astype(float).values.reshape(-1, 1)

    if len(values) < lookback + 80:
        return None, "Not enough candles for LSTM training."

    # Fit the scaler only on the training portion to reduce leakage.
    split_index = int(len(values) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(values[:split_index])

    scaled = scaler.transform(values)

    X = []
    y = []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    train_end = split_index - lookback

    if train_end < 30:
        return None, "Not enough training data."

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]

    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1], 1
    )
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1], 1
    )

    model = Sequential(
        [
            LSTM(
                32,
                input_shape=(lookback, 1),
                return_sequences=True,
            ),
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

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test)
        if len(X_test) > 0
        else None,
        callbacks=callbacks,
        verbose=0,
    )

    latest_sequence = scaled[-lookback:].reshape(
        1, lookback, 1
    )

    prediction_scaled = model.predict(
        latest_sequence, verbose=0
    )

    prediction = float(
        scaler.inverse_transform(prediction_scaled)[0, 0]
    )

    current_price = float(values[-1, 0])

    if prediction > current_price:
        prediction_signal = "BUY"
    elif prediction < current_price:
        prediction_signal = "SELL"
    else:
        prediction_signal = "NO SIGNAL"

    return {
        "prediction": prediction,
        "current_price": current_price,
        "signal": prediction_signal,
    }, None


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

pair = st.sidebar.selectbox(
    "Currency pair",
    list(PAIRS.keys()),
    index=0,
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1h", "1d"],
    index=0,
)

period_options = {
    "5m": ["5d", "1mo"],
    "15m": ["5d", "1mo", "3mo"],
    "30m": ["5d", "1mo", "3mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
}

period = st.sidebar.selectbox(
    "Historical data period",
    period_options[timeframe],
    index=0,
)

min_votes = st.sidebar.slider(
    "Minimum votes required",
    min_value=1,
    max_value=8,
    value=4,
)

enable_lstm = st.sidebar.checkbox(
    "Enable optional LSTM",
    value=False,
)

lstm_epochs = st.sidebar.slider(
    "LSTM training epochs",
    min_value=2,
    max_value=20,
    value=5,
    disabled=not enable_lstm,
)

st.sidebar.caption(
    "Signals are estimates, not guaranteed predictions."
)


# --------------------------------------------------
# INVESTMENT CALCULATOR
# --------------------------------------------------
st.header("💰 Investment calculator")

calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    account_currency = st.selectbox(
        "Account currency",
        CURRENCIES,
        index=0,
    )

with calc_col2:
    investment = st.number_input(
        "Investment amount",
        min_value=0.01,
        value=1.25,
        step=0.25,
    )

with calc_col3:
    payout_percent = st.number_input(
        "Payout (%)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )

profit_if_correct = investment * payout_percent / 100
loss_if_incorrect = investment

calc1, calc2, calc3 = st.columns(3)

calc1.metric(
    "Calculated stake",
    f"{investment:.2f} {account_currency}",
)

calc2.metric(
    "Profit if correct",
    f"{profit_if_correct:.2f} {account_currency}",
)

calc3.metric(
    "Loss if incorrect",
    f"{loss_if_incorrect:.2f} {account_currency}",
)

st.caption(
    "This is a mathematical estimate based on the payout you entered. "
    "It is not the platform's minimum deposit and does not guarantee profit. "
    "The currency selection changes the label only; it does not convert money."
)

st.divider()


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------
st.header("📈 Market analysis")

if not YFINANCE_AVAILABLE:
    st.error(
        "yfinance is missing. Add `yfinance` to requirements.txt "
        "and redeploy the app."
    )
    st.code(
        "streamlit\npandas\nnumpy\nyfinance\nscikit-learn\ntensorflow",
        language="text",
    )
    st.stop()

if st.button("🔍 Analyze market", type="primary", use_container_width=True):
    ticker = PAIRS[pair]

    try:
        with st.spinner(
            f"Fetching {pair} data and calculating indicators..."
        ):
            raw_data = fetch_market_data(
                ticker,
                timeframe,
                period,
            )

            analysis = calculate_indicators(raw_data)
            latest = analysis.iloc[-1]

            technical_signal, buy_votes, sell_votes, reasons = (
                generate_technical_signal(latest)
            )

            lstm_result = None
            lstm_error = None

            if enable_lstm:
                with st.spinner("Training LSTM model..."):
                    lstm_result, lstm_error = train_lstm_prediction(
                        analysis["Close"],
                        lookback=30,
                        epochs=lstm_epochs,
                    )

        # ------------------------------------------
        # SIGNAL DECISION
        # ------------------------------------------
        final_signal = technical_signal
        signal_source = "Technical analysis"

        if buy_votes < min_votes and sell_votes < min_votes:
            final_signal = "NO SIGNAL"

        if enable_lstm and lstm_result is not None:
            lstm_signal = lstm_result["signal"]

            # Only accept an LSTM confirmation when it agrees
            # with a valid technical signal.
            if final_signal in ("BUY", "SELL"):
                if lstm_signal != final_signal:
                    final_signal = "NO SIGNAL"
                    signal_source = "Technical/LSTM disagreement"
                else:
                    signal_source = "Technical analysis + LSTM confirmation"
            else:
                final_signal = "NO SIGNAL"
                signal_source = "Insufficient technical votes"

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------
        st.subheader(f"{pair} — {timeframe}")

        price = float(latest["Close"])

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Latest close", f"{price:.6f}")
        m2.metric("Technical BUY votes", str(buy_votes))
        m3.metric("Technical SELL votes", str(sell_votes))
        m4.metric("RSI", f"{float(latest['RSI']):.2f}")

        if final_signal == "BUY":
            st.success("🟢 BUY signal")
        elif final_signal == "SELL":
            st.error("🔴 SELL signal")
        else:
            st.info("⚪ NO SIGNAL — wait for clearer conditions")

        st.write(f"**Signal source:** {signal_source}")
        st.write(
            f"**Last candle timestamp:** {analysis.index[-1]}"
        )

        st.caption(
            "The last available candle may still be forming, depending on "
            "the data provider. A signal can change before the candle closes."
        )

        # ------------------------------------------
        # CHART
        # ------------------------------------------
        st.subheader("Price and moving averages")

        chart_data = analysis[
            ["Close", "EMA15", "EMA20", "EMA50",
             "BB_Upper", "BB_Middle", "BB_Lower"]
        ].tail(200)

        st.line_chart(chart_data)

        # ------------------------------------------
        # TECHNICAL DETAILS
        # ------------------------------------------
        st.subheader("Technical indicators")

        indicator_table = pd.DataFrame(
            {
                "Indicator": [
                    "EMA 15",
                    "EMA 20",
                    "EMA 50",
                    "Bollinger upper",
                    "Bollinger middle",
                    "Bollinger lower",
                    "RSI (14)",
                    "MACD",
                    "MACD signal",
                    "Stochastic %K",
                    "Stochastic %D",
                    "ADX",
                    "+DI",
                    "-DI",
                    "CCI",
                    "Williams %R",
                    "ATR (14)",
                    "Candle direction",
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
                    latest["Candle_Direction"],
                ],
            }
        )

        st.dataframe(
            indicator_table,
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------
        # SIGNAL REASONS
        # ------------------------------------------
        st.subheader("Signal breakdown")

        for reason in reasons:
            st.write(f"• {reason}")

        st.write(
            f"**BUY votes:** {buy_votes} / 8  |  "
            f"**SELL votes:** {sell_votes} / 8  |  "
            f"**Minimum votes setting:** {min_votes}"
        )

        # ------------------------------------------
        # LSTM RESULTS
        # ------------------------------------------
        if enable_lstm:
            st.subheader("🧠 LSTM prediction")

            if lstm_result is not None:
                lstm_col1, lstm_col2, lstm_col3 = st.columns(3)

                lstm_col1.metric(
                    "Latest close",
                    f"{lstm_result['current_price']:.6f}",
                )

                lstm_col2.metric(
                    "Model estimate",
                    f"{lstm_result['prediction']:.6f}",
                )

                lstm_col3.metric(
                    "LSTM direction",
                    lstm_result["signal"],
                )

                st.caption(
                    "LSTM is a statistical model trained on historical closing "
                    "prices. Its estimate is not a reliable guarantee of the "
                    "next candle's direction."
                )
            else:
                st.warning(
                    lstm_error
                    or "LSTM could not produce a prediction."
                )

        # ------------------------------------------
        # RECENT CANDLES
        # ------------------------------------------
        with st.expander("Show recent candles"):
            st.dataframe(
                analysis.tail(30).sort_index(ascending=False),
                use_container_width=True,
            )

    except ImportError as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error(
            "Analysis failed. The data provider may be unavailable, "
            "the selected ticker may not have data, or a dependency may "
            "be missing."
        )
        st.code(str(exc))

else:
    st.info(
        "Select a currency pair and timeframe, then click "
        "'Analyze market' to fetch public market data."
    )

st.divider()

st.caption(
    "Risk notice: Signals can be wrong. Historical data and technical "
    "indicators do not guarantee future results. This app does not place "
    "trades or connect to a brokerage account."
)
