
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Binomo VIP BUY SELL AI",
    page_icon="📈",
    layout="wide"
)

st.title("VIP | Binomo BUY / SELL AI Scanner")
st.caption("5M | EMA 15 | Bollinger Bands (20,2) | LSTM")

st.warning(
    "Bu uygulama sinyal üretir; otomatik emir açmaz ve Binomo'ya "
    "doğrudan bağlanmaz. Sonuçlar garanti değildir."
)

# Kullanıcının gönderdiği varlık listesi.
ASSETS = [
    ("Crypto IDX", 83, 85),
    ("Bitcoin (OTC)", 80, 82),
    ("Ethereum (OTC)", 80, 82),
    ("Solana (OTC)", 80, 82),
    ("EUR/USD", 80, 82),
    ("EUR/HUF", 80, 82),
    ("GBP/NOK", 80, 82),
    ("CAD/JPY", 80, 82),
    ("FC Barcelona Token (OTC)", 80, 82),
    ("AUD/ZAR", 80, 82),
    ("CAD/DKK", 80, 82),
    ("EUR/ILS", 80, 82),
    ("GBP/SGD", 80, 82),
    ("USD/MXN", 80, 82),
    ("Cardano (OTC)", 80, 82),
    ("EUR/ZAR", 80, 82),
    ("AUD/USD", 80, 82),
    ("GBP/HKD", 80, 82),
    ("CAD/MXN", 80, 82),
    ("EUR/NOK", 80, 82),
    ("EUR/MXN", 80, 82),
    ("CHF/JPY", 80, 82),
    ("USD/ILS", 80, 82),
    ("USD/SEK", 80, 82),
    ("CAD/NOK", 80, 82),
    ("CAD/SEK", 80, 82),
    ("USD/PLN", 80, 82),
    ("AUD/JPY", 80, 82),
    ("EUR/SEK", 80, 82),
    ("AUD/SGD", 80, 82),
    ("Chainlink (OTC)", 80, 82),
    ("GBP/MXN", 80, 82),
    ("CAD/SGD", 80, 82),
    ("AUD/DKK", 80, 82),
    ("EUR/SGD", 80, 82),
    ("CHF/DKK", 80, 82),
    ("GBP/PLN", 80, 82),
    ("GBP/CZK", 80, 82),
    ("EUR/NZD", 80, 82),
    ("AUD/HUF", 80, 82),
    ("USD/HUF", 80, 82),
    ("GBP/DKK", 80, 82),
    ("AUD/NOK", 80, 82),
    ("NZD/JPY", 80, 82),
    ("CHF/NOK", 80, 82),
    ("USD/JPY", 80, 82),
    ("CHF/PLN", 80, 82),
    ("GBP/TRY", 80, 82),
    ("USD/CAD", 80, 82),
    ("AUD/SEK", 80, 82),
    ("CHF/SEK", 80, 82),
    ("CHF/SGD", 80, 82),
    ("NOK/JPY", 80, 82),
    ("NOK/SEK", 80, 82),
    ("NZD/CAD", 80, 82),
    ("NZD/DKK", 80, 82),
    ("EUR/GBP (OTC)", 80, 82),
    ("GBP/NZD", 80, 82),
    ("NZD/USD", 80, 82),
    ("NZD/HUF", 80, 82),
    ("NZD/NOK", 80, 82),
    ("NZD/SEK", 80, 82),
    ("NZD/SGD", 80, 82),
    ("SEK/JPY", 80, 82),
    ("SGD/JPY", 80, 82),
    ("SGD/HKD", 80, 82),
    ("USD/CHF", 80, 82),
    ("AUD/CAD", 80, 82),
    ("USD/NOK", 80, 82),
    ("USD/CZK", 80, 82),
    ("Bitcoin Cash (OTC)", 80, 82),
    ("Kusama (OTC)", 80, 82),
    ("Aave (OTC)", 80, 82),
    ("GBP/JPY", 80, 82),
    ("Pancake Swap (OTC)", 80, 82),
    ("Uniswap (OTC)", 80, 82),
    ("EUR/CAD", 80, 82),
    ("NZD/CHF", 80, 82),
    ("GBP/CAD", 80, 82),
    ("AUD/NZD", 80, 82),
    ("USD/ZAR", 80, 82),
    ("AUD/CHF", 80, 82),
    ("French CAC 40 Index", 70, 72),
    ("GBP/USD", 70, 72),
    ("Vodafone", 65, 67),
    ("Gold", 60, 62),
    ("GBP/HUF", 50, 52),
    ("EUR/PLN", 50, 52),
    ("ZAR/JPY", 50, 52),
    ("CHF/HUF", 50, 52),
    ("Silver", 50, 52),
    ("Litecoin", 30, 32),
    ("USD/DKK (Gold)", 81, 82),
    ("GBP/CHF (Gold)", 81, 82),
    ("Yum Brands (Gold)", 76, 77),
    ("Nvidia (Gold)", 76, 77),
    ("Visa (Gold)", 71, 72),
    ("Starbucks (Gold)", 71, 72),
    ("Weibo (Gold)", 71, 72),
    ("Qualcomm (Gold)", 71, 72),
    ("Intel (Gold)", 71, 72),
    ("JPMorgan C (Gold)", 66, 67),
    ("Oil (Gold)", 66, 67),
    ("Ferrari (Gold)", 31, 32),
]

# Sidebar settings
with st.sidebar:
    st.header("VIP Settings")

    selected_asset = st.selectbox(
        "Varlık seç",
        [asset[0] for asset in ASSETS]
    )

    payout = next(
        (a[1:] for a in ASSETS if a[0] == selected_asset),
        (80, 82)
    )

    st.write(f"Paylaşılan oranlar: {payout[0]}% / {payout[1]}%")
    st.caption("Bunlar canlı doğrulanmış oranlar değildir.")

    lookback = st.number_input(
        "LSTM geçmiş mum sayısı",
        min_value=20,
        max_value=200,
        value=60,
        step=5
    )

    epochs = st.number_input(
        "Eğitim epoch",
        min_value=1,
        max_value=50,
        value=8
    )

    confidence = st.slider(
        "Minimum model olasılığı",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.01
    )

    min_rows = st.number_input(
        "Minimum veri satırı",
        min_value=150,
        max_value=10000,
        value=400,
        step=50
    )

st.subheader("Varlık listesi")

st.dataframe(
    pd.DataFrame(
        ASSETS,
        columns=["Varlık", "Oran A (%)", "Oran B (%)"]
    ),
    use_container_width=True,
    height=220
)

uploaded_file = st.file_uploader(
    f"{selected_asset} için 5 dakikalık CSV yükle",
    type=["csv"]
)

st.markdown("### CSV formatı")

st.code(
    "timestamp,open,high,low,close,volume\n"
    "2026-01-01 10:00:00,1.1000,1.1010,1.0990,1.1005,120",
    language="csv"
)

if uploaded_file is None:
    st.info(
        "Sinyal üretmek için seçtiğin varlığın geçmiş 5 dakikalık "
        "mum verilerini CSV olarak yükle."
    )
    st.stop()

# Read data
try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"CSV okunamadı: {exc}")
    st.stop()

df.columns = [str(c).strip().lower() for c in df.columns]

aliases = {
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "tick_volume": "volume"
}

rename_map = {
    old: new
    for old, new in aliases.items()
    if old in df.columns and new not in df.columns
}

df = df.rename(columns=rename_map)

required_columns = ["open", "high", "low", "close"]
missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    st.error(
        "CSV'de eksik sütunlar var: " + ", ".join(missing)
    )
    st.stop()

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

if "volume" not in df.columns:
    df["volume"] = 0.0
else:
    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    ).fillna(0)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )
    df = df.sort_values("timestamp")

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

if len(df) < max(int(min_rows), int(lookback) + 100):
    st.error(
        f"Yeterli veri yok. Kullanılabilir mum: {len(df)}. "
        f"En az {max(int(min_rows), int(lookback) + 100)} mum yükle."
    )
    st.stop()

if (df["close"] <= 0).any():
    st.error("Kapanış fiyatları pozitif olmalıdır.")
    st.stop()

# Indicators
df["ema15"] = df["close"].ewm(
    span=15,
    adjust=False
).mean()

df["bb_mid"] = df["close"].rolling(20).mean()

df["bb_std"] = df["close"].rolling(20).std(ddof=0)

df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]

df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

# LSTM features
df["return1"] = df["close"].pct_change()

df["range_pct"] = (
    (df["high"] - df["low"])
    / df["close"].replace(0, np.nan)
)

df["ema_gap"] = (
    (df["close"] - df["ema15"])
    / df["close"].replace(0, np.nan)
)

df["bb_position"] = (
    (df["close"] - df["bb_lower"])
    / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
)

# Target: whether the next candle closes higher
df["target"] = (
    df["close"].shift(-1) > df["close"]
).astype(int)

feature_columns = [
    "return1",
    "range_pct",
    "ema_gap",
    "bb_position"
]

data = df.dropna(
    subset=feature_columns
).copy()

if len(data) < int(lookback) + 80:
    st.error("Gösterge hesaplamasından sonra yeterli veri kalmadı.")
    st.stop()

# Load deep-learning dependencies
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

except Exception as exc:
    st.error(
        "TensorFlow yüklenemedi. Uyumlu Python sürümü kullanıp "
        "'pip install -r requirements.txt' komutunu çalıştır."
    )
    st.code(str(exc))
    st.stop()

# Train and generate a signal
if st.button(
    "LSTM Eğit ve BUY / SELL Sinyali Üret",
    type="primary"
):

    with st.spinner("LSTM modeli eğitiliyor..."):

        # The final candle has no known next-candle outcome.
        supervised = data.iloc[:-1].copy()

        split_index = int(len(supervised) * 0.80)

        if split_index <= int(lookback) + 20:
            st.error(
                "Eğitim ve test için daha fazla geçmiş veri yükle."
            )
            st.stop()

        # Fit scaler on the training period only.
        scaler = MinMaxScaler()

        scaler.fit(
            supervised[feature_columns].iloc[:split_index]
        )

        scaled_data = scaler.transform(
            data[feature_columns]
        )

        X, y, indices = [], [], []

        for i in range(int(lookback), len(data) - 1):
            X.append(
                scaled_data[i - int(lookback):i]
            )

            y.append(
                int(data["target"].iloc[i])
            )

            indices.append(i)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        indices = np.asarray(indices)

        train_mask = indices < split_index
        test_mask = indices >= split_index

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) < 50 or len(X_test) < 10:
            st.error(
                "Eğitim veya test dizisi yetersiz. Daha fazla mum yükle."
            )
            st.stop()

        # LSTM neural network
        model = Sequential([
            LSTM(
                32,
                input_shape=(
                    int(lookback),
                    len(feature_columns)
                )
            ),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=int(epochs),
            batch_size=32,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=0
        )

        # Historical holdout evaluation
        test_probabilities = model.predict(
            X_test,
            verbose=0
        ).reshape(-1)

        test_predictions = (
            test_probabilities >= 0.50
        ).astype(int)

        test_accuracy = accuracy_score(
            y_test.astype(int),
            test_predictions
        )

        # Predict direction for the next candle
        latest_sequence = scaled_data[
            -int(lookback):
        ].reshape(
            1,
            int(lookback),
            len(feature_columns)
        )

        probability_up = float(
            model.predict(
                latest_sequence,
                verbose=0
            )[0][0]
        )

        last = data.iloc[-1]

        close = float(last["close"])
        ema15 = float(last["ema15"])
        bb_mid = float(last["bb_mid"])
        bb_upper = float(last["bb_upper"])
        bb_lower = float(last["bb_lower"])

        # Indicator confirmation filters
        buy_context = (
            close > ema15
            and close > bb_mid
            and close < bb_upper
        )

        sell_context = (
            close < ema15
            and close < bb_mid
            and close > bb_lower
        )

        signal = "NO SIGNAL"

        reason = (
            "Model and indicators do not agree. "
            "No directional signal."
        )

        if (
            probability_up >= confidence
            and buy_context
        ):
            signal = "BUY"

            reason = (
                "LSTM upward probability passed the threshold; "
                "EMA 15 and Bollinger conditions confirm the direction."
            )

        elif (
            probability_up <= 1.0 - confidence
            and sell_context
        ):
            signal = "SELL"

            reason = (
                "LSTM downward probability passed the threshold; "
                "EMA 15 and Bollinger conditions confirm the direction."
            )

        st.session_state["vip_result"] = {
            "asset": selected_asset,
            "signal": signal,
            "reason": reason,
            "probability_up": probability_up,
            "accuracy": float(test_accuracy),
            "close": close,
            "ema15": ema15,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "data": data.copy()
        }

# Display latest signal
if (
    "vip_result" in st.session_state
    and st.session_state["vip_result"]["asset"] == selected_asset
):

    result = st.session_state["vip_result"]

    st.subheader(
        f"Son sinyal: {result['asset']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sinyal",
        result["signal"]
    )

    col2.metric(
        "LSTM yükseliş tahmini",
        f"{result['probability_up'] * 100:.1f}%"
    )

    col3.metric(
        "Geçmiş test doğruluğu",
        f"{result['accuracy'] * 100:.1f}%"
    )

    col4.metric(
        "Son kapanış",
        f"{result['close']:.8g}"
    )

    st.write(
        "**Sinyal nedeni:** " + result["reason"]
    )

    st.caption(
        "Model olasılığı kazanma oranı değildir. "
        "Geçmiş test doğruluğu gelecekteki kârlılığı garanti etmez."
    )

    # Price and indicator chart
    chart_data = result["data"].tail(180)

    if "timestamp" in chart_data.columns:
        x_axis = chart_data["timestamp"]
    else:
        x_axis = chart_data.index

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=x_axis,
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["ema15"],
            name="EMA 15"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["bb_upper"],
            name="BB Upper",
            line=dict(dash="dot")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["bb_mid"],
            name="BB Middle",
            line=dict(dash="dot")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["bb_lower"],
            name="BB Lower",
            line=dict(dash="dot")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=chart_data["volume"],
            name="Volume"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        legend_orientation="h"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Gösterge değerleri")

    st.dataframe(
        pd.DataFrame([{
            "Varlık": result["asset"],
            "Close": result["close"],
            "EMA 15": result["ema15"],
            "BB Middle": result["bb_mid"],
            "BB Upper": result["bb_upper"],
            "BB Lower": result["bb_lower"],
            "P(UP)": result["probability_up"],
            "Sinyal": result["signal"]
        }]),
        use_container_width=True
    )

st.subheader("Sinyal açıklamaları")

st.markdown("""
- **BUY:** LSTM yükseliş olasılığı eşiği geçer ve göstergeler yükselişi destekler.
- **SELL:** LSTM düşüş olasılığı eşiği geçer ve göstergeler düşüşü destekler.
- **NO SIGNAL:** Koşullar uyuşmuyorsa sinyal üretilmez.
- **5 dakika:** Yüklediğin mumlar 5 dakikalık olmalıdır.
- **Otomatik işlem:** Bu uygulama emir göndermez.
""")
