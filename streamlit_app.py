import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crypto Deep Learning Scanner",
    page_icon="🧠",
    layout="wide"
)

API_BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

INTERVAL = "15m"
CANDLE_LIMIT = 1000
LOOKBACK = 24
DEFAULT_EPOCHS = 300
MAX_WORKERS = 3

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40
REQUEST_TIMEOUT = 15


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧠 Crypto Deep Learning Scanner")

st.write(
    "Binance Spot USDT piyasaları için fiyat ve hacim "
    "verilerini kullanan makine öğrenmesi analiz paneli."
)

st.info(
    "Bu sürüm PyTorch gerektirmez. NumPy tabanlı lojistik "
    "regresyon modeli kullanır; gerçek LSTM değildir. "
    "Hiçbir alım satım emri göndermez."
)

st.warning(
    "BUY / SELL / HOLD yalnızca model tahminleridir. "
    "Yatırım tavsiyesi veya kâr garantisi değildir."
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Crypto-Scanner/1.0"
    })
    return session


def api_get(endpoint, params=None):
    errors = []

    for base in API_BASES:
        try:
            response = get_session().get(
                base + endpoint,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            errors.append(
                f"{base}: {str(exc)[:120]}"
            )

    raise RuntimeError(
        "Binance API adresleri başarısız oldu:\n"
        + "\n".join(errors)
    )


# ============================================================
# BINANCE SPOT SYMBOLS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_spot_symbols():
    data = api_get("/api/v3/exchangeInfo")
    symbols = []

    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            symbols.append(item["symbol"])

    return sorted(set(symbols))


# ============================================================
# MARKET TICKERS
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_tickers():
    data = api_get("/api/v3/ticker/24hr")
    rows = []

    for item in data:
        try:
            rows.append({
                "symbol": item["symbol"],
                "price": float(item["lastPrice"]),
                "change_24h": float(item["priceChangePercent"]),
                "volume_24h": float(item["quoteVolume"]),
                "high_24h": float(item["highPrice"]),
                "low_24h": float(item["lowPrice"]),
            })
        except (KeyError, ValueError, TypeError):
            continue

    return pd.DataFrame(rows)


# ============================================================
# CANDLE DATA
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def get_candles(symbol, count=1000):
    interval_ms = 15 * 60 * 1000
    now_ms = int(time.time() * 1000)

    # End before the current, unfinished 15-minute candle.
    end_ms = (now_ms // interval_ms) * interval_ms - 1

    rows = []

    while len(rows) < count:
        request_limit = min(1000, count - len(rows))

        page = api_get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": INTERVAL,
                "limit": request_limit,
                "endTime": end_ms,
            }
        )

        if not page:
            break

        rows = page + rows
        end_ms = int(page[0][0]) - 1

        if len(page) < request_limit or len(rows) >= count:
            break

        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_base", "taker_quote", "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]),
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_numeric(
        df["close_time"],
        errors="coerce"
    )

    for column in [
        "open", "high", "low", "close",
        "volume", "quote_volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df[df["close_time"] < now_ms]

    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .dropna(
            subset=[
                "open", "high", "low",
                "close", "volume", "quote_volume",
            ]
        )
        .tail(count)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# RAW PRICE AND VOLUME FEATURES
# No RSI / EMA / MACD / BOLLINGER BANDS
# ============================================================

def make_features(df):
    close = df["close"].to_numpy(dtype=np.float64)
    open_price = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    quote_volume = df["quote_volume"].to_numpy(dtype=np.float64)

    close = np.maximum(close, 1e-12)
    open_price = np.maximum(open_price, 1e-12)
    volume = np.maximum(volume, 1e-12)
    quote_volume = np.maximum(quote_volume, 1e-12)

    returns = np.zeros(len(close))
    volume_changes = np.zeros(len(volume))
    quote_changes = np.zeros(len(quote_volume))

    returns[1:] = np.log(close[1:] / close[:-1])
    volume_changes[1:] = np.log(volume[1:] / volume[:-1])
    quote_changes[1:] = np.log(
        quote_volume[1:] / quote_volume[:-1]
    )

    body = np.log(close / open_price)
    candle_range = (high - low) / close

    upper_wick = (
        high - np.maximum(open_price, close)
    ) / close

    lower_wick = (
        np.minimum(open_price, close) - low
    ) / close

    features = np.column_stack([
        returns,
        body,
        candle_range,
        upper_wick,
        lower_wick,
        volume_changes,
        quote_changes,
    ])

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# SEQUENCE DATASET
# ============================================================

def make_sequences(features, closes, lookback):
    X = []
    y = []
    target_positions = []

    for target_i in range(lookback, len(features)):
        sequence = features[target_i - lookback:target_i]

        # Summarize the recent raw-data sequence.
        # This is NOT an LSTM or a technical indicator.
        sequence_features = np.concatenate([
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence[-1],
        ])

        X.append(sequence_features)

        # Predict whether the target candle closed higher.
        y.append(
            1.0 if closes[target_i] > closes[target_i - 1]
            else 0.0
        )

        target_positions.append(target_i)

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(target_positions, dtype=np.int64),
    )


# ============================================================
# NUMPY LOGISTIC REGRESSION
# No external ML library required.
# ============================================================

def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_model(df, epochs=DEFAULT_EPOCHS):
    if len(df) < 200:
        raise ValueError(
            "En az 200 tamamlanmış mum gerekli."
        )

    features = make_features(df)
    closes = df["close"].to_numpy(dtype=np.float64)

    split = int(len(df) * 0.80)

    # Fit feature normalization on training candles only.
    feature_mean = features[:split].mean(axis=0)
    feature_std = features[:split].std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    scaled = (features - feature_mean) / feature_std

    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X, y, positions = make_sequences(
        scaled,
        closes,
        LOOKBACK
    )

    train_mask = positions < split
    val_mask = positions >= split

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    if len(X_train) < 50 or len(X_val) < 10:
        raise ValueError(
            "Eğitim veya doğrulama verisi yetersiz."
        )

    # Normalize sequence-summary features using training only.
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std

    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)

    # Class balancing.
    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = max(float(len(y_train) - y_train.sum()), 1.0)

    positive_weight = len(y_train) / (2.0 * positive_count)
    negative_weight = len(y_train) / (2.0 * negative_count)

    sample_weights = np.where(
        y_train >= 0.5,
        positive_weight,
        negative_weight
    )

    weights = np.zeros(X_train.shape[1], dtype=np.float64)
    bias = 0.0

    learning_rate = 0.03
    regularization = 0.002

    for _ in range(int(epochs)):
        probabilities = sigmoid(X_train @ weights + bias)

        errors = (
            (probabilities - y_train) * sample_weights
        )

        grad_w = (
            X_train.T @ errors / len(X_train)
            + regularization * weights
        )

        grad_b = float(errors.mean())

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    # Validation prediction.
    val_probabilities = sigmoid(X_val @ weights + bias)
    val_predictions = (val_probabilities >= 0.5).astype(float)

    accuracy = float(
        np.mean(val_predictions == y_val)
    )

    recalls = []

    for class_id in (0.0, 1.0):
        mask = y_val == class_id

        if mask.any():
            recalls.append(
                float(np.mean(val_predictions[mask] == y_val[mask]))
            )

    balanced_accuracy = (
        float(np.mean(recalls))
        if recalls else float("nan")
    )

    # Build the latest sequence with the same transformations.
    latest_sequence = scaled[-LOOKBACK:]

    latest_summary = np.concatenate([
        latest_sequence.mean(axis=0),
        latest_sequence.std(axis=0),
        latest_sequence[-1],
    ])

    latest_summary = (
        latest_summary - x_mean
    ) / x_std

    latest_summary = np.clip(
        latest_summary,
        -10,
        10
    )

    up_probability = float(
        sigmoid(latest_summary @ weights + bias)
    )

    down_probability = 1.0 - up_probability

    if up_probability >= BUY_THRESHOLD:
        signal = "BUY"
    elif up_probability <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "up_probability": up_probability,
        "down_probability": down_probability,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
    }


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(symbol, epochs):
    try:
        df = get_candles(symbol, CANDLE_LIMIT)

        if df.empty or len(df) < 200:
            raise ValueError(
                "Yeterli tamamlanmış mum bulunamadı."
            )

        prediction = train_model(df, epochs)

        return {
            "symbol": symbol,
            "signal": prediction["signal"],
            "price": float(df["close"].iloc[-1]),
            "up_probability": prediction["up_probability"],
            "down_probability": prediction["down_probability"],
            "accuracy": prediction["accuracy"],
            "balanced_accuracy": prediction["balanced_accuracy"],
            "candles": len(df),
            "train_samples": prediction["train_samples"],
            "validation_samples": prediction["validation_samples"],
            "error": None,
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "error": str(exc)[:400],
        }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Tarama ayarları")

scan_mode = st.sidebar.selectbox(
    "Parite kapsamı",
    [
        "En yüksek hacimli pariteler",
        "Tüm USDT Spot pariteleri",
    ]
)

coin_count = st.sidebar.slider(
    "Taranacak parite sayısı",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

minimum_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0.0,
    value=1_000_000.0,
    step=500_000.0,
)

epochs = st.sidebar.slider(
    "Eğitim turu",
    min_value=50,
    max_value=1000,
    value=DEFAULT_EPOCHS,
    step=50,
)

show_errors = st.sidebar.checkbox(
    "Hatalı pariteleri göster",
    value=True,
)

if st.sidebar.button("Önbelleği temizle"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD MARKET DATA
# ============================================================

try:
    with st.spinner("Binance Spot verileri alınıyor..."):
        spot_symbols = set(get_spot_symbols())
        tickers = get_tickers()

except Exception as exc:
    st.error("Binance verileri alınamadı.")
    st.code(str(exc))
    st.stop()

if tickers.empty:
    st.error("Binance ticker verisi boş döndü.")
    st.stop()

tickers = tickers[
    tickers["symbol"].isin(spot_symbols)
].copy()

tickers = tickers[
    tickers["volume_24h"] >= minimum_volume
].sort_values(
    "volume_24h",
    ascending=False
)

if scan_mode == "En yüksek hacimli pariteler":
    symbols_to_scan = (
        tickers.head(coin_count)["symbol"].tolist()
    )
else:
    symbols_to_scan = tickers["symbol"].tolist()

if not symbols_to_scan:
    st.warning(
        "Filtrelere uygun parite bulunamadı. "
        "Minimum hacmi düşürmeyi dene."
    )
    st.stop()

st.caption(
    f"{len(symbols_to_scan)} parite | "
    "15 dakika | Hedef: 1.000 tamamlanmış mum"
)


# ============================================================
# RUN SCAN
# ============================================================

if "results" not in st.session_state:
    st.session_state["results"] = []

if st.button(
    "🚀 Taramayı başlat",
    type="primary",
    use_container_width=True,
):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol,
                epochs
            ): symbol
            for symbol in symbols_to_scan
        }

        total = len(futures)

        for index, future in enumerate(
            as_completed(futures),
            start=1
        ):
            symbol = futures[future]

            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "symbol": symbol,
                    "error": str(exc)[:400],
                })

            progress.progress(index / total)
            status.write(
                f"Analiz ediliyor: {symbol} ({index}/{total})"
            )

    st.session_state["results"] = results
    progress.empty()
    status.empty()

    st.success("Tarama tamamlandı.")
    st.rerun()


# ============================================================
# DISPLAY RESULTS
# ============================================================

results = st.session_state["results"]

if results:
    result_df = pd.DataFrame(results)

    if "error" not in result_df.columns:
        result_df["error"] = None

    successful = result_df[
        result_df["error"].isna()
    ].copy()

    failed = result_df[
        result_df["error"].notna()
    ].copy()

    if not successful.empty:
        successful = successful.sort_values(
            ["up_probability", "accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)

        successful.insert(
            0,
            "rank",
            np.arange(1, len(successful) + 1),
        )

        successful["up_pct"] = (
            successful["up_probability"] * 100
        ).round(2)

        successful["down_pct"] = (
            successful["down_probability"] * 100
        ).round(2)

        successful["accuracy_pct"] = (
            successful["accuracy"] * 100
        ).round(2)

        successful["balanced_accuracy_pct"] = (
            successful["balanced_accuracy"] * 100
        ).round(2)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Başarılı analiz", len(successful))
        c2.metric(
            "BUY",
            int((successful["signal"] == "BUY").sum())
        )
        c3.metric(
            "SELL",
            int((successful["signal"] == "SELL").sum())
        )
        c4.metric(
            "HOLD",
            int((successful["signal"] == "HOLD").sum())
        )

        st.subheader("🏆 Model sıralaması")

        display_columns = [
            "rank",
            "symbol",
            "signal",
            "price",
            "up_pct",
            "down_pct",
            "accuracy_pct",
            "balanced_accuracy_pct",
            "candles",
            "train_samples",
            "validation_samples",
        ]

        display_df = successful[display_columns].rename(
            columns={
                "rank": "Sıra",
                "symbol": "Parite",
                "signal": "Sinyal",
                "price": "Son fiyat",
                "up_pct": "Yükseliş %",
                "down_pct": "Düşüş %",
                "accuracy_pct": "Doğruluk %",
                "balanced_accuracy_pct": "Dengeli doğruluk %",
                "candles": "Mum sayısı",
                "train_samples": "Eğitim örneği",
                "validation_samples": "Doğrulama örneği",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Sonuçları CSV indir",
            data=successful.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="crypto_scan_results.csv",
            mime="text/csv",
        )

        st.subheader("🔎 Parite detayı")

        selected_symbol = st.selectbox(
            "Grafiğini görüntüle",
            successful["symbol"].tolist(),
        )

        selected = successful[
            successful["symbol"] == selected_symbol
        ].iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Model sinyali", selected["signal"])
        b.metric(
            "Yükseliş olasılığı",
            f'{selected["up_probability"] * 100:.2f}%'
        )
        c.metric(
            "Düşüş olasılığı",
            f'{selected["down_probability"] * 100:.2f}%'
        )
        d.metric(
            "Doğrulama doğruluğu",
            f'{selected["accuracy"] * 100:.2f}%'
        )

        try:
            chart_df = get_candles(
                selected_symbol,
                CANDLE_LIMIT,
            )

            if not chart_df.empty:
                chart = chart_df.set_index("open_time")

                st.subheader(
                    f"{selected_symbol} — kapanış fiyatı"
                )

                st.line_chart(
                    chart[["close"]],
                    height=350,
                )

                st.subheader("Mumların yüksek/düşük aralığı")

                st.area_chart(
                    chart[["low", "high"]],
                    height=250,
                )

                with st.expander("Son 100 mum verisi"):
                    st.dataframe(
                        chart_df.tail(100).sort_values(
                            "open_time",
                            ascending=False,
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as exc:
            st.warning(
                f"Grafik yüklenemedi: {exc}"
            )

    else:
        st.warning(
            "Başarılı analiz bulunamadı. Hata tablosunu kontrol et."
        )

    if show_errors and not failed.empty:
        st.subheader("⚠️ Analiz hataları")

        st.dataframe(
            failed[["symbol", "error"]],
            use_container_width=True,
            hide_index=True,
        )

else:
    st.write(
        "Henüz tarama yapılmadı. Sol taraftan ayarları seç "
        "ve tarama düğmesine bas."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bu sürüm NumPy tabanlı lojistik regresyon kullanır; LSTM değildir. "
    "Doğrulama doğruluğu gelecekteki performansı garanti etmez."
)

st.caption(
    "Otomatik alım satım yoktur. UTC: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
)
