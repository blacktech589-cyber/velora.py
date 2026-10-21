
import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Velora AI Signal Panel",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Velora Multi-Algorithm Signal Panel")
st.caption("EMA • Bollinger Bands • RSI • MACD • Stochastic • ADX • CCI • Williams %R • LSTM")
st.warning(
    "Research tool only. No Binomo connection or automatic trade execution. "
    "Public-market prices may differ from Binomo OTC prices."
)

# Optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ------------------------------------------------
# Asset mapping for public Yahoo Finance data
# These are NOT Binomo OTC feeds.
# ------------------------------------------------
ASSETS = {
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
    "AUD/JPY": "AUDJPY=X",
    "GBP/CHF": "GBPCHF=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/AUD": "GBPAUD=X",
}

# ------------------------------------------------
# Investment calculator - works without market data
# ------------------------------------------------
st.sidebar.header("Investment settings")

asset = st.sidebar.selectbox("Currency pair", list(ASSETS.keys()))

payout = st.sidebar.slider(
    "Payout (%)",
    min_value=50,
    max_value=100,
    value=80
)

target_profit = st.sidebar.number_input(
    "Target profit if successful",
    min_value=0.01,
    value=1.0,
    step=0.5
)

currency = st.sidebar.selectbox(
    "Currency",
    ["USD", "EUR", "TRY"]
)

max_investment = st.sidebar.number_input(
    "Maximum amount per trade",
    min_value=0.01,
    value=5.0,
    step=0.5
)

stake = target_profit / (payout / 100)

st.subheader("💰 Initial investment estimate")

a, b, c = st.columns(3)
a.metric("Calculated stake", f"{stake:.2f} {currency}")
b.metric("Profit if correct", f"{stake * payout / 100:.2f} {currency}")
c.metric("Loss if incorrect", f"{stake:.2f} {currency}")

if stake > max_investment:
    st.warning("Calculated stake exceeds your configured per-trade limit.")

st.caption(
    "This estimates a stake for the specified payout and target profit. "
    "It is not a minimum deposit, a recommendation, or a guarantee."
)

st.divider()

# ------------------------------------------------
# Data settings
# ------------------------------------------------
st.sidebar.header("Signal settings")

interval = st.sidebar.selectbox(
    "Candle timeframe",
    ["5m", "15m", "30m", "1h"],
    index=0
)

lookback_period = st.sidebar.selectbox(
    "Historical data period",
    ["5d", "1mo", "3mo"],
    index=0
)

minimum_votes = st.sidebar.slider(
    "Minimum agreeing algorithms",
    min_value=2,
    max_value=8,
    value=4
)

use_lstm = st.sidebar.checkbox(
    "Use LSTM if TensorFlow is available",
    value=False
)

refresh = st.button("🔄 Fetch data and calculate signals")

# ------------------------------------------------
# Indicators
# ------------------------------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False
    ).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(data):
    df = data.copy()

    # Normalize common yfinance column format
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    needed = ["open", "high", "low", "close"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed).copy()

    # EMA trend
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20, 2)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    # RSI
    df["rsi"] = rsi(df["close"])

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic oscillator
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denominator = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low14) / denominator
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # ADX
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=df.index
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

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # CCI
    typical = (df["high"] + df["low"] + df["close"]) / 3
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )
    df["cci"] = (
        (typical - typical_mean) /
        (0.015 * mean_deviation.replace(0, np.nan))
    )

    # Williams %R
    high14 = df["high"].rolling(14).max()
    low14 = df["low"].rolling(14).min()
    df["williams_r"] = (
        -100 * (high14 - df["close"]) /
        (high14 - low14).replace(0, np.nan)
    )

    # Momentum / candle return
    df["momentum"] = df["close"].pct_change(5) * 100
    df["return1"] = df["close"].pct_change()

    return df.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------
# Consensus algorithms
# ------------------------------------------------
def calculate_votes(row):
    votes = {}

    # 1. EMA crossover / trend
    votes["EMA trend"] = (
        1 if row["ema15"] > row["ema50"]
        else -1 if row["ema15"] < row["ema50"] else 0
    )

    # 2. Bollinger mean reversion
    if row["close"] < row["bb_lower"]:
        votes["Bollinger Bands"] = 1
    elif row["close"] > row["bb_upper"]:
        votes["Bollinger Bands"] = -1
    else:
        votes["Bollinger Bands"] = 0

    # 3. RSI
    if row["rsi"] < 30:
        votes["RSI"] = 1
    elif row["rsi"] > 70:
        votes["RSI"] = -1
    else:
        votes["RSI"] = 0

    # 4. MACD
    votes["MACD"] = (
        1 if row["macd"] > row["macd_signal"]
        else -1 if row["macd"] < row["macd_signal"] else 0
    )

    # 5. Stochastic
    if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
        votes["Stochastic"] = 1
    elif row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
        votes["Stochastic"] = -1
    else:
        votes["Stochastic"] = 0

    # 6. ADX directional trend
    if row["adx"] >= 20 and row["plus_di"] > row["minus_di"]:
        votes["ADX / DI"] = 1
    elif row["adx"] >= 20 and row["minus_di"] > row["plus_di"]:
        votes["ADX / DI"] = -1
    else:
        votes["ADX / DI"] = 0

    # 7. CCI
    if row["cci"] < -100:
        votes["CCI"] = 1
    elif row["cci"] > 100:
        votes["CCI"] = -1
    else:
        votes["CCI"] = 0

    # 8. Williams %R
    if row["williams_r"] < -80:
        votes["Williams %R"] = 1
    elif row["williams_r"] > -20:
        votes["Williams %R"] = -1
    else:
        votes["Williams %R"] = 0

    # 9. Momentum
    votes["Momentum"] = (
        1 if row["momentum"] > 0
        else -1 if row["momentum"] < 0 else 0
    )

    return votes


def try_lstm_signal(df):
    if not LSTM_AVAILABLE:
        return None, "TensorFlow is unavailable"

    features = ["return1", "rsi", "macd_hist", "momentum"]
    data = df[features + ["close"]].dropna().copy()

    if len(data) < 150:
        return None, "Not enough historical candles for LSTM"

    x_values = data[features].values
    close_values = data["close"].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_values)

    X, y = [], []
    window = 20

    for i in range(window, len(data) - 1):
        X.append(scaled[i - window:i])
        y.append(
            1 if close_values[i + 1] > close_values[i] else 0
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) < 80 or len(np.unique(y)) < 2:
        return None, "Not enough class diversity for LSTM"

    split = int(len(X) * 0.8)
    if split < 50 or split >= len(X):
        return None, "Insufficient training/test samples"

    model = Sequential([
        LSTM(24, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(12, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X[:split],
        y[:split],
        epochs=4,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    latest = scaled[-window:]
    probability_up = float(
        model.predict(latest[np.newaxis, ...], verbose=0)[0][0]
    )

    if probability_up >= 0.55:
        return 1, f"LSTM probability UP: {probability_up:.1%}"
    if probability_up <= 0.45:
        return -1, f"LSTM probability UP: {probability_up:.1%}"

    return 0, f"LSTM uncertain: {probability_up:.1%}"


# ------------------------------------------------
# Fetch market data and calculate signals
# ------------------------------------------------
if refresh:
    if not YFINANCE_AVAILABLE:
        st.error(
            "yfinance is not installed. Add yfinance to requirements.txt."
        )
    else:
        ticker = ASSETS[asset]

        with st.spinner(f"Fetching public data for {asset}..."):
            try:
                data = yf.download(
                    ticker,
                    period=lookback_period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if data is None or data.empty:
                    st.error(
                        "No data was returned for this asset/timeframe. "
                        "Try another timeframe or pair."
                    )
                else:
                    df = calculate_indicators(data)
                    df = df.dropna().copy()

                    if len(df) < 60:
                        st.error(
                            "Not enough valid candles to calculate all "
                            "indicators. Try a longer data period."
                        )
                    else:
                        row = df.iloc[-1]
                        votes = calculate_votes(row)

                        lstm_note = "LSTM not enabled"
                        if use_lstm:
                            with st.spinner("Evaluating LSTM..."):
                                lstm_vote, lstm_note = try_lstm_signal(df)
                            if lstm_vote is not None:
                                votes["LSTM"] = lstm_vote

                        buy_votes = sum(v == 1 for v in votes.values())
                        sell_votes = sum(v == -1 for v in votes.values())
                        active_votes = buy_votes + sell_votes

                        signal = "NO SIGNAL"
                        if buy_votes >= minimum_votes and buy_votes > sell_votes:
                            signal = "BUY"
                        elif sell_votes >= minimum_votes and sell_votes > buy_votes:
                            signal = "SELL"

                        st.subheader(f"{asset} — Analysis")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Latest public-market close", f"{row['close']:.6g}")
                        p2.metric("BUY votes", buy_votes)
                        p3.metric("SELL votes", sell_votes)

                        if signal == "BUY":
                            st.success("🟢 BUY signal — experimental consensus")
                        elif signal == "SELL":
                            st.error("🔴 SELL signal — experimental consensus")
                        else:
                            st.warning("⚪ NO SIGNAL — consensus threshold not reached")

                        st.caption(
                            f"Data source: Yahoo Finance ({ticker}); "
                            f"timeframe: {interval}. "
                            "This is not the Binomo OTC feed."
                        )

                        st.subheader("Algorithm vote table")
                        vote_df = pd.DataFrame([
                            {
                                "Algorithm": name,
                                "Vote": (
                                    "BUY" if value == 1 else
                                    "SELL" if value == -1 else
                                    "NEUTRAL"
                                )
                            }
                            for name, value in votes.items()
                        ])
                        st.dataframe(vote_df, use_container_width=True)

                        st.write("**Indicator values**")
                        st.json({
                            "EMA 15": float(row["ema15"]),
                            "EMA 50": float(row["ema50"]),
                            "RSI": float(row["rsi"]),
                            "MACD": float(row["macd"]),
                            "MACD signal": float(row["macd_signal"]),
                            "Bollinger upper": float(row["bb_upper"]),
                            "Bollinger lower": float(row["bb_lower"]),
                            "ADX": float(row["adx"]),
                            "ATR": float(row["atr"]),
                            "CCI": float(row["cci"]),
                            "Williams %R": float(row["williams_r"]),
                            "LSTM status": lstm_note
                        })

                        st.subheader("Recent price and indicators")
                        st.line_chart(
                            df[[
                                "close", "ema15", "ema50",
                                "bb_upper", "bb_lower"
                            ]].tail(200)
                        )

                        st.subheader("Recent candles")
                        st.dataframe(
                            df.tail(20),
                            use_container_width=True
                        )

                        st.caption(
                            "Technical indicators can disagree and may lag price. "
                            "A vote count is not a calibrated probability of winning. "
                            "Do not treat these signals as a guarantee or use them "
                            "without independent testing."
                        )

            except Exception as e:
                st.error(f"Could not fetch or analyze data: {e}")
                st.info(
                    "Check internet/data availability, selected ticker, "
                    "and installed package versions."
                )
else:
    st.info(
        "Investment calculator works immediately. "
        "Press 'Fetch data and calculate signals' to request public-market "
        "candles and calculate technical indicators."
    )
