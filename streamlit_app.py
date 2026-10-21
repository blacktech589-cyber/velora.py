
import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Velora | Binomo Signal Panel",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Velora — BUY / SELL Signal Panel")
st.caption("Currency pairs • Technical analysis • Optional LSTM")
st.warning(
    "Analysis only. This app does not connect to your Binomo account "
    "or execute trades. External market prices may differ from Binomo OTC."
)

# -------------------------------------------------
# Optional dependencies
# -------------------------------------------------
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# -------------------------------------------------
# Public-market ticker mapping
# These symbols are not Binomo OTC feeds.
# -------------------------------------------------
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
    "EUR/CAD": "EURCAD=X",
    "GBP/CHF": "GBPCHF=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/AUD": "GBPAUD=X",
    "AUD/CAD": "AUDCAD=X",
}

# -------------------------------------------------
# Sidebar: account and investment settings
# -------------------------------------------------
st.sidebar.header("Binomo-style settings")

asset = st.sidebar.selectbox(
    "Currency pair",
    list(ASSETS.keys())
)

account_currency = st.sidebar.selectbox(
    "Account currency",
    ["USD ($)", "EUR (€)", "TRY (₺)"]
)

payout = st.sidebar.slider(
    "Payout (%)",
    min_value=50,
    max_value=100,
    value=80
)

target_profit = st.sidebar.number_input(
    "Target profit if correct",
    min_value=0.01,
    value=1.0,
    step=0.5
)

max_stake = st.sidebar.number_input(
    "Maximum stake per trade",
    min_value=0.01,
    value=5.0,
    step=0.5
)

timeframe = st.sidebar.selectbox(
    "Candle timeframe",
    ["5m", "15m", "30m", "1h"],
    index=0
)

history_period = st.sidebar.selectbox(
    "Historical data period",
    ["5d", "1mo", "3mo"],
    index=0
)

min_votes = st.sidebar.slider(
    "Minimum agreeing algorithms",
    min_value=2,
    max_value=9,
    value=4
)

use_lstm = st.sidebar.checkbox(
    "Enable LSTM (requires TensorFlow)",
    value=False
)

# -------------------------------------------------
# Investment calculator — no CSV needed
# -------------------------------------------------
stake = target_profit / (payout / 100.0)

st.subheader("💰 Investment calculator")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Calculated stake",
    f"{stake:.2f} {account_currency}"
)

c2.metric(
    "Profit if correct",
    f"{stake * payout / 100.0:.2f} {account_currency}"
)

c3.metric(
    "Loss if incorrect",
    f"{stake:.2f} {account_currency}"
)

if stake > max_stake:
    st.warning(
        "The calculated stake is higher than your configured maximum. "
        "Reduce the target profit or review the settings."
    )

st.caption(
    "This is a mathematical estimate based on the payout you entered. "
    "It is not the platform's minimum deposit and does not guarantee profit."
)

st.divider()

# -------------------------------------------------
# Indicator calculations
# -------------------------------------------------
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(data):
    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).lower() for col in df.columns]

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()

    # EMA trend
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20, 2)
    middle = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)

    df["bb_mid"] = middle
    df["bb_upper"] = middle + 2 * std
    df["bb_lower"] = middle - 2 * std

    # RSI
    df["rsi"] = calculate_rsi(df["close"])

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    range14 = (high14 - low14).replace(0, np.nan)

    df["stoch_k"] = 100 * (df["close"] - low14) / range14
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    # ADX and directional indicators
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )

    atr_smooth = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean().replace(0, np.nan)

    df["plus_di"] = (
        100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smooth
    )
    df["minus_di"] = (
        100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smooth
    )

    di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    dx = 100 * (
        (df["plus_di"] - df["minus_di"]).abs() / di_sum
    )

    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # CCI
    typical = (df["high"] + df["low"] + df["close"]) / 3
    typical_mean = typical.rolling(20).mean()

    mean_deviation = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    df["cci"] = (
        (typical - typical_mean)
        / (0.015 * mean_deviation.replace(0, np.nan))
    )

    # Williams %R
    high14 = df["high"].rolling(14).max()
    low14 = df["low"].rolling(14).min()

    df["williams_r"] = (
        -100 * (high14 - df["close"])
        / (high14 - low14).replace(0, np.nan)
    )

    # Momentum and candle return
    df["momentum"] = df["close"].pct_change(5) * 100
    df["return1"] = df["close"].pct_change()

    return df.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------
# Algorithm votes
# 1 = BUY, -1 = SELL, 0 = NEUTRAL
# -------------------------------------------------
def algorithm_votes(row):
    votes = {}

    # 1. EMA trend
    if row["ema15"] > row["ema50"]:
        votes["EMA 15/50"] = 1
    elif row["ema15"] < row["ema50"]:
        votes["EMA 15/50"] = -1
    else:
        votes["EMA 15/50"] = 0

    # 2. Bollinger Bands mean-reversion condition
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
    if row["macd"] > row["macd_signal"]:
        votes["MACD"] = 1
    elif row["macd"] < row["macd_signal"]:
        votes["MACD"] = -1
    else:
        votes["MACD"] = 0

    # 5. Stochastic
    if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
        votes["Stochastic"] = 1
    elif row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
        votes["Stochastic"] = -1
    else:
        votes["Stochastic"] = 0

    # 6. ADX + directional movement
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
    if row["momentum"] > 0:
        votes["Momentum"] = 1
    elif row["momentum"] < 0:
        votes["Momentum"] = -1
    else:
        votes["Momentum"] = 0

    return votes


# -------------------------------------------------
# Optional LSTM model
# Uses chronological train/test separation.
# -------------------------------------------------
def lstm_vote(df):
    if not LSTM_AVAILABLE:
        return None, "TensorFlow is not installed."

    features = ["return1", "rsi", "macd_hist", "momentum"]
    data = df[features + ["close"]].dropna().copy()

    if len(data) < 180:
        return None, "Not enough historical candles for LSTM."

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features].values)

    window = 20
    X, y = [], []
    prices = data["close"].to_numpy()

    for i in range(window, len(data) - 1):
        X.append(scaled[i - window:i])
        y.append(1 if prices[i + 1] > prices[i] else 0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    split = int(len(X) * 0.8)

    if split < 50 or len(X) - split < 10:
        return None, "Insufficient train/test data."

    if len(np.unique(y[:split])) < 2:
        return None, "Training data contains only one class."

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

    probability = float(
        model.predict(X[-1:], verbose=0)[0][0]
    )

    if probability >= 0.55:
        return 1, f"Estimated upward probability: {probability:.1%}"
    elif probability <= 0.45:
        return -1, f"Estimated upward probability: {probability:.1%}"

    return 0, f"LSTM uncertain: {probability:.1%}"


# -------------------------------------------------
# Fetch and analyze data on button click
# -------------------------------------------------
if st.button("🔎 GET BUY / SELL SIGNAL", type="primary"):
    if not YF_AVAILABLE:
        st.error(
            "yfinance is missing. Add yfinance to requirements.txt "
            "and redeploy the app."
        )
        st.stop()

    ticker = ASSETS[asset]

    with st.spinner(f"Fetching public-market data for {asset}..."):
        try:
            raw = yf.download(
                ticker,
                period=history_period,
                interval=timeframe,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if raw is None or raw.empty:
                st.error(
                    "No public price data returned. Try a different pair, "
                    "timeframe, or historical period."
                )
                st.stop()

            df = calculate_indicators(raw).dropna().copy()

            if len(df) < 50:
                st.error("Not enough valid candles for analysis.")
                st.stop()

            latest = df.iloc[-1]
            votes = algorithm_votes(latest)
            lstm_message = "LSTM disabled"

            if use_lstm:
                with st.spinner("Running LSTM model..."):
                    extra_vote, lstm_message = lstm_vote(df)

                if extra_vote is not None:
                    votes["LSTM"] = extra_vote

            buy_count = sum(v == 1 for v in votes.values())
            sell_count = sum(v == -1 for v in votes.values())

            if buy_count >= min_votes and buy_count > sell_count:
                signal = "BUY"
            elif sell_count >= min_votes and sell_count > buy_count:
                signal = "SELL"
            else:
                signal = "NO SIGNAL"

            st.divider()
            st.subheader(f"{asset} — Current analysis")

            x1, x2, x3 = st.columns(3)
            x1.metric("Last public close", f"{latest['close']:.6g}")
            x2.metric("BUY votes", str(buy_count))
            x3.metric("SELL votes", str(sell_count))

            if signal == "BUY":
                st.success("🟢 BUY")
            elif signal == "SELL":
                st.error("🔴 SELL")
            else:
                st.warning("⚪ NO SIGNAL — WAIT")

            st.write(f"**Account currency selected:** {account_currency}")
            st.write(f"**Selected asset:** {asset}")
            st.write(f"**Payout entered:** {payout}%")
            st.write(f"**LSTM status:** {lstm_message}")

            vote_table = pd.DataFrame([
                {
                    "Algorithm": name,
                    "Signal": (
                        "BUY" if value == 1
                        else "SELL" if value == -1
                        else "NEUTRAL"
                    )
                }
                for name, value in votes.items()
            ])

            st.subheader("Algorithm results")
            st.dataframe(vote_table, use_container_width=True)

            st.subheader("Indicator values")
            st.dataframe(
                df[[
                    "close", "ema15", "ema50",
                    "bb_upper", "bb_lower", "rsi",
                    "macd", "macd_signal", "stoch_k",
                    "adx", "atr", "cci", "williams_r"
                ]].tail(10),
                use_container_width=True
            )

            st.subheader("Price and indicator chart")
            st.line_chart(
                df[[
                    "close", "ema15", "ema50",
                    "bb_upper", "bb_lower"
                ]].tail(200)
            )

            st.caption(
                "External public-market data is not the same as verified "
                "Binomo OTC data. Signals are experimental, may be wrong, "
                "and do not guarantee profitability. No trades are executed."
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check package installation, network access, and data availability."
            )

else:
    st.info(
        "The investment calculator is available without CSV. "
        "Press GET BUY / SELL SIGNAL to request public-market data."
    )
