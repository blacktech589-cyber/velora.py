
import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Velora AI Signal Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Velora AI Signal Scanner")
st.caption("EMA 15 + Bollinger Bands (20, 2) + LSTM | 5-minute candles")
st.warning(
    "This application provides experimental signals only. "
    "It does not connect to Binomo or execute trades."
)

# --------------------------------------------------
# Optional Plotly import
# --------------------------------------------------
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --------------------------------------------------
# LSTM import
# --------------------------------------------------
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    ML_AVAILABLE = True
    ML_ERROR = ""
except Exception as e:
    ML_AVAILABLE = False
    ML_ERROR = str(e)

# --------------------------------------------------
# Asset list
# --------------------------------------------------
ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "EUR/CHF", "AUD/JPY",
    "GBP/CHF", "CAD/JPY", "CHF/JPY", "EUR/AUD",
    "GBP/AUD", "AUD/CAD", "AUD/CHF", "EUR/CAD",
    "NZD/JPY", "GBP/CAD", "CAD/CHF", "NZD/CAD",
    "NZD/CHF", "EUR/NZD", "GBP/NZD", "AUD/NZD"
]

# --------------------------------------------------
# Indicators
# --------------------------------------------------
def calculate_indicators(df):
    df = df.copy()

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "timestamp": ["timestamp", "time", "date", "datetime"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "price"],
        "volume": ["volume", "vol", "v"]
    }

    rename = {}
    for target, choices in aliases.items():
        for choice in choices:
            if choice in df.columns:
                rename[choice] = target
                break

    df = df.rename(columns=rename)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV must contain Open, High, Low and Close columns. "
            f"Missing: {', '.join(missing)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "At least 100 valid candles are required. "
            "More historical candles are recommended."
        )

    # EMA 15
    df["ema15"] = df["close"].ewm(
        span=15, adjust=False
    ).mean()

    # Bollinger Bands: period 20, standard deviation 2
    df["bb_mid"] = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + 2 * rolling_std
    df["bb_lower"] = df["bb_mid"] - 2 * rolling_std

    # Model features
    df["return1"] = df["close"].pct_change()
    df["range_pct"] = (
        (df["high"] - df["low"]) /
        df["close"].replace(0, np.nan)
    )
    df["ema_gap"] = (
        (df["close"] - df["ema15"]) /
        df["close"].replace(0, np.nan)
    )
    band_width = (
        df["bb_upper"] - df["bb_lower"]
    ).replace(0, np.nan)

    df["bb_position"] = (
        (df["close"] - df["bb_lower"]) / band_width
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# --------------------------------------------------
# Prepare supervised sequences
# Target: direction of the next candle
# --------------------------------------------------
FEATURES = [
    "return1",
    "range_pct",
    "ema_gap",
    "bb_position"
]


def make_sequences(data, lookback=20):
    x_data = data[FEATURES].values
    close_values = data["close"].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(x_data)

    X, y = [], []

    for i in range(lookback, len(data) - 1):
        X.append(scaled[i - lookback:i])

        # Next candle closes above current candle = 1
        y.append(
            1 if close_values[i + 1] > close_values[i] else 0
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y), scaler


def build_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")

asset = st.sidebar.selectbox("Asset", ASSETS)
lookback = st.sidebar.slider(
    "LSTM lookback candles",
    min_value=10,
    max_value=60,
    value=20
)
epochs = st.sidebar.slider(
    "Training epochs",
    min_value=1,
    max_value=20,
    value=5
)
confidence_threshold = st.sidebar.slider(
    "Minimum signal confidence",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.05
)

uploaded_file = st.file_uploader(
    f"Upload {asset} 5-minute OHLCV CSV",
    type=["csv"]
)

st.info(
    "CSV format: Open, High, Low, Close. "
    "A Timestamp/Time column is optional. "
    "Use candles from the selected asset only."
)

# --------------------------------------------------
# Main application
# --------------------------------------------------
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        df = calculate_indicators(raw)

        st.subheader(f"{asset} — Data preview")
        st.dataframe(df.tail(10), use_container_width=True)

        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["close"],
                name="Close"
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["ema15"],
                name="EMA 15"
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                name="BB Upper"
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                name="BB Lower"
            ))

            fig.update_layout(
                title=f"{asset} — Indicators",
                xaxis_title="Candle",
                yaxis_title="Price",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Plotly is not installed. The scanner can still "
                "calculate indicators and signals without the chart."
            )
            st.line_chart(
                df[["close", "ema15", "bb_upper", "bb_lower"]].tail(200)
            )

        if not ML_AVAILABLE:
            st.error(
                "LSTM dependencies could not be loaded. "
                "Check requirements.txt and Streamlit Cloud logs."
            )
            st.code(ML_ERROR)
            st.stop()

        model_data = df.dropna(subset=FEATURES + ["close"]).reset_index(
            drop=True
        )

        if len(model_data) <= lookback + 50:
            st.error(
                "Not enough usable data after calculating indicators. "
                "Upload more 5-minute candles."
            )
            st.stop()

        X, y, scaler = make_sequences(model_data, lookback)

        if len(X) < 50 or len(np.unique(y)) < 2:
            st.error(
                "Not enough samples or only one target class. "
                "Upload a longer history with price movements."
            )
            st.stop()

        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(X_test) == 0:
            st.error("Not enough test samples.")
            st.stop()

        st.subheader("LSTM training")

        with st.spinner("Training LSTM model..."):
            model = build_model(
                (X_train.shape[1], X_train.shape[2])
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            validation_data = None
            callbacks = []

            if len(X_train) >= 20:
                validation_data = (X_test, y_test)
                callbacks.append(early_stop)

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=32,
                verbose=0,
                validation_data=validation_data,
                callbacks=callbacks
            )

            test_probs = model.predict(X_test, verbose=0).ravel()
            test_preds = (test_probs >= 0.5).astype(int)
            test_accuracy = accuracy_score(y_test, test_preds)

            # Latest available feature window
            latest_x = scaler.transform(
                model_data[FEATURES].values
            )[-lookback:]

            latest_x = np.expand_dims(
                latest_x.astype(np.float32), axis=0
            )

            probability_up = float(
                model.predict(latest_x, verbose=0)[0][0]
            )

        st.metric(
            "Chronological holdout accuracy",
            f"{test_accuracy * 100:.2f}%"
        )

        st.caption(
            "Historical test accuracy is not a guaranteed win rate "
            "and does not establish profitability."
        )

        last = model_data.iloc[-1]
        price = float(last["close"])

        # Indicator context plus LSTM direction
        above_ema = price > float(last["ema15"])
        below_ema = price < float(last["ema15"])
        near_lower = price <= float(last["bb_lower"])
        near_upper = price >= float(last["bb_upper"])

        signal = "NO SIGNAL"
        confidence = max(probability_up, 1 - probability_up)

        if confidence >= confidence_threshold:
            if probability_up >= confidence_threshold and above_ema:
                signal = "BUY"
            elif (
                probability_up <= 1 - confidence_threshold
                and below_ema
            ):
                signal = "SELL"

        st.subheader("Latest signal")

        col1, col2, col3 = st.columns(3)
        col1.metric("Last close", f"{price:.6g}")
        col2.metric("LSTM probability UP", f"{probability_up:.1%}")
        col3.metric("Signal confidence", f"{confidence:.1%}")

        if signal == "BUY":
            st.success("🟢 BUY — experimental model signal")
        elif signal == "SELL":
            st.error("🔴 SELL — experimental model signal")
        else:
            st.warning("⚪ NO SIGNAL — filters do not agree")

        st.write("**Indicator context**")
        st.write(f"- EMA 15: {float(last['ema15']):.6g}")
        st.write(f"- Bollinger upper: {float(last['bb_upper']):.6g}")
        st.write(f"- Bollinger lower: {float(last['bb_lower']):.6g}")
        st.write(f"- Price near lower band: {'Yes' if near_lower else 'No'}")
        st.write(f"- Price near upper band: {'Yes' if near_upper else 'No'}")

        st.caption(
            "This is an experimental analysis, not financial advice. "
            "Signals do not guarantee the next candle direction. "
            "The app does not connect to Binomo or place trades."
        )

    except Exception as e:
        st.error(f"Could not process the uploaded CSV: {e}")
        st.info(
            "Check the CSV column names and ensure OHLC values are numeric."
        )

else:
    st.info("Upload a CSV file containing historical 5-minute candles to begin.")
