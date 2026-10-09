import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LSTM Crypto Scanner",
    page_icon="📈",
    layout="wide"
)

BINANCE_BASE_URL = "https://api.binance.com"
INTERVAL = "15m"
CANDLE_LIMIT = 1000
SEQUENCE_LENGTH = 48
EPOCHS = 8
BATCH_SIZE = 64
MAX_WORKERS = 5

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

DEVICE = "cpu"

if TORCH_AVAILABLE:
    torch.manual_seed(42)
    np.random.seed(42)


# =========================================================
# PAGE STYLE
# =========================================================

st.title("📈 LSTM Crypto Scanner")
st.caption(
    "Binance Spot • USDT pariteleri • 15 dakikalık mumlar • "
    "LSTM derin öğrenme • Sadece analiz, emir göndermez"
)

st.info(
    "Bu uygulama yalnızca herkese açık Binance piyasa verilerini okur. "
    "API anahtarı istemez ve hiçbir alım satım emri göndermez. "
    "Model tahminleri garanti değildir."
)


# =========================================================
# HTTP SESSION
# =========================================================

@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "LSTM-Crypto-Scanner/1.0"
    })
    return session


def binance_get(endpoint, params=None, timeout=20):
    url = BINANCE_BASE_URL + endpoint
    response = get_session().get(
        url,
        params=params,
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


# =========================================================
# BINANCE SPOT MARKETS
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_spot_symbols():
    data = binance_get("/api/v3/exchangeInfo")

    symbols = []

    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            symbols.append(item["symbol"])

    return sorted(set(symbols))


@st.cache_data(ttl=60, show_spinner=False)
def get_tickers():
    data = binance_get("/api/v3/ticker/24hr")

    rows = []

    for item in data:
        symbol = item.get("symbol", "")

        if not symbol.endswith("USDT"):
            continue

        try:
            rows.append({
                "symbol": symbol,
                "price": float(item["lastPrice"]),
                "change_24h": float(item["priceChangePercent"]),
                "quote_volume": float(item["quoteVolume"]),
                "high_24h": float(item["highPrice"]),
                "low_24h": float(item["lowPrice"])
            })
        except (TypeError, ValueError, KeyError):
            continue

    return pd.DataFrame(rows)


# =========================================================
# CANDLE DATA
# =========================================================

def fetch_candle_page(symbol, end_time=None, limit=1000):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": min(int(limit), 1000)
    }

    if end_time is not None:
        params["endTime"] = int(end_time)

    return binance_get(
        "/api/v3/klines",
        params=params
    )


@st.cache_data(ttl=120, show_spinner=False)
def get_candles(symbol, requested_count=1000):
    """
    Downloads up to requested_count completed candles.
    Uses pagination when necessary and excludes the
    currently forming candle.
    """

    requested_count = min(max(int(requested_count), 100), 1000)

    now_ms = int(time.time() * 1000)
    interval_ms = 15 * 60 * 1000

    # Last fully closed candle's close time.
    last_closed_boundary = (now_ms // interval_ms) * interval_ms - 1

    all_rows = []
    end_time = last_closed_boundary

    while len(all_rows) < requested_count:
        remaining = requested_count - len(all_rows)
        page_limit = min(remaining, 1000)

        page = fetch_candle_page(
            symbol,
            end_time=end_time,
            limit=page_limit
        )

        if not page:
            break

        all_rows = page + all_rows

        oldest_open_time = int(page[0][0])
        end_time = oldest_open_time - 1

        if len(page) < page_limit:
            break

        if len(all_rows) >= requested_count:
            break

        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(all_rows, columns=columns)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]),
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_numeric(
        df["close_time"],
        errors="coerce"
    )

    # Exclude any candle that has not fully closed.
    df = df[df["close_time"] < now_ms]

    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .tail(requested_count)
        .reset_index(drop=True)
    )

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    return df


# =========================================================
# DEEP LEARNING FEATURES
# No RSI, EMA, MACD, Bollinger Bands or indicator scoring.
# =========================================================

def make_features(df):
    """
    Features are derived directly from OHLCV price and volume
    data. No conventional technical indicators are calculated.
    """

    close = df["close"].to_numpy(dtype=np.float64)
    open_price = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    close_safe = np.maximum(close, 1e-12)
    open_safe = np.maximum(open_price, 1e-12)
    volume_safe = np.maximum(volume, 1e-12)

    log_return = np.zeros_like(close)
    log_return[1:] = np.log(
        close_safe[1:] / close_safe[:-1]
    )

    candle_body = np.log(close_safe / open_safe)

    candle_range = (
        high - low
    ) / close_safe

    upper_wick = (
        high - np.maximum(open_price, close)
    ) / close_safe

    lower_wick = (
        np.minimum(open_price, close) - low
    ) / close_safe

    volume_change = np.zeros_like(volume)
    volume_change[1:] = np.log(
        volume_safe[1:] / volume_safe[:-1]
    )

    quote_volume = df["quote_volume"].to_numpy(
        dtype=np.float64
    )

    quote_volume_safe = np.maximum(quote_volume, 1e-12)

    quote_volume_change = np.zeros_like(quote_volume)
    quote_volume_change[1:] = np.log(
        quote_volume_safe[1:] / quote_volume_safe[:-1]
    )

    features = np.column_stack([
        log_return,
        candle_body,
        candle_range,
        upper_wick,
        lower_wick,
        volume_change,
        quote_volume_change
    ])

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features.astype(np.float32)


# =========================================================
# LSTM MODEL
# =========================================================

if TORCH_AVAILABLE:

    class LSTMClassifier(nn.Module):
        def __init__(
            self,
            input_size,
            hidden_size=48,
            num_layers=2,
            dropout=0.20
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            self.dropout = nn.Dropout(dropout)

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 24),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(24, 2)
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            last_output = output[:, -1, :]
            last_output = self.dropout(last_output)
            return self.fc(last_output)


# =========================================================
# TRAINING AND VALIDATION
# =========================================================

def build_sequences(features, closes, sequence_length):
    """
    The label is whether the next candle closes higher than
    the current candle. Sequences never use future candles
    as input.
    """

    X = []
    y = []

    for i in range(sequence_length, len(features) - 1):
        sequence = features[i - sequence_length:i]
        next_close = closes[i + 1]
        current_close = closes[i]

        X.append(sequence)
        y.append(1 if next_close > current_close else 0)

    if not X:
        return None, None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


def scale_features_train_only(features, train_end):
    """
    Standardization parameters are calculated from the
    training period only to reduce leakage.
    """

    train_data = features[:train_end]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    std = np.where(std < 1e-8, 1.0, std)

    scaled = (features - mean) / std

    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return scaled.astype(np.float32), mean, std


def train_and_predict(df, epochs=EPOCHS):
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch kurulu değil. requirements.txt dosyasına "
            "'torch' ekleyip uygulamayı yeniden dağıt."
        )

    if len(df) < 200:
        raise ValueError(
            "Model eğitimi için yeterli mum verisi yok."
        )

    features = make_features(df)
    closes = df["close"].to_numpy(dtype=np.float64)

    sequence_length = SEQUENCE_LENGTH

    # Reserve the final portion for chronological validation.
    split_index = int(len(features) * 0.80)

    scaled_features, _, _ = scale_features_train_only(
        features,
        split_index
    )

    X, y = build_sequences(
        scaled_features,
        closes,
        sequence_length
    )

    if X is None or len(X) < 100:
        raise ValueError(
            "Model için yeterli eğitim dizisi oluşturulamadı."
        )

    # X[k] ends at candle i; y[k] describes candle i+1.
    label_positions = np.arange(
        sequence_length + 1,
        len(features)
    )

    train_mask = label_positions < split_index
    val_mask = label_positions >= split_index

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    if len(X_train) < 50 or len(X_val) < 10:
        raise ValueError(
            "Eğitim veya doğrulama verisi yetersiz."
        )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True
    )

    model = LSTMClassifier(
        input_size=features.shape[1]
    ).to(DEVICE)

    class_counts = np.bincount(
        y_train,
        minlength=2
    ).astype(np.float32)

    class_weights = len(y_train) / (
        2.0 * np.maximum(class_counts, 1.0)
    )

    weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    model.train()

    for epoch in range(int(epochs)):
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()

    # Chronological validation.
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_probabilities = torch.softmax(
            val_logits,
            dim=1
        ).cpu().numpy()

    val_predictions = np.argmax(
        val_probabilities,
        axis=1
    )

    accuracy = float(
        np.mean(val_predictions == y_val)
    )

    # Balanced accuracy helps when one class dominates.
    recalls = []

    for class_id in (0, 1):
        mask = y_val == class_id

        if np.any(mask):
            recalls.append(
                float(
                    np.mean(
                        val_predictions[mask] == class_id
                    )
                )
            )

    balanced_accuracy = (
        float(np.mean(recalls))
        if recalls else float("nan")
    )

    # Predict next-candle direction using the latest sequence.
    latest_sequence = scaled_features[
        -sequence_length:
    ]

    latest_tensor = torch.tensor(
        latest_sequence[np.newaxis, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        latest_logits = model(latest_tensor)
        latest_probabilities = torch.softmax(
            latest_logits,
            dim=1
        )[0].cpu().numpy()

    down_probability = float(latest_probabilities[0])
    up_probability = float(latest_probabilities[1])

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
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "last_close": float(closes[-1]),
        "last_candle": df["open_time"].iloc[-1],
        "model": model
    }


# =========================================================
# SINGLE COIN ANALYSIS
# =========================================================

def analyze_symbol(symbol):
    try:
        df = get_candles(
            symbol,
            requested_count=CANDLE_LIMIT
        )

        if df.empty or len(df) < 200:
            return {
                "symbol": symbol,
                "error": "Yeterli mum verisi bulunamadı."
            }

        result = train_and_predict(df)

        return {
            "symbol": symbol,
            "signal": result["signal"],
            "up_probability": result["up_probability"],
            "down_probability": result["down_probability"],
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "train_samples": result["train_samples"],
            "validation_samples": result["validation_samples"],
            "last_close": result["last_close"],
            "last_candle": result["last_candle"],
            "candles": len(df),
            "error": None
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "error": str(exc)[:250]
        }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Tarama Ayarları")

market_scope = st.sidebar.selectbox(
    "Parite kapsamı",
    [
        "En yüksek hacimli pariteler",
        "Tüm USDT Spot pariteleri"
    ]
)

top_n = st.sidebar.slider(
    "Tarama yapılacak parite sayısı",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Tüm pariteler seçilirse bu değer kullanılmaz."
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0.0,
    value=1_000_000.0,
    step=500_000.0
)

epochs = st.sidebar.slider(
    "LSTM eğitim turu",
    min_value=2,
    max_value=20,
    value=EPOCHS
)

show_all_results = st.sidebar.checkbox(
    "Hata veren pariteleri de göster",
    value=False
)

if st.sidebar.button("Veri önbelleğini temizle"):
    st.cache_data.clear()
    st.rerun()


# =========================================================
# MAIN DASHBOARD
# =========================================================

if not TORCH_AVAILABLE:
    st.error(
        "PyTorch bulunamadı. requirements.txt dosyasına "
        "'torch' ekleyin ve uygulamayı yeniden dağıtın."
    )
    st.stop()

try:
    with st.spinner("Binance Spot piyasaları alınıyor..."):
        valid_symbols = set(get_spot_symbols())
        tickers = get_tickers()

except Exception as exc:
    st.error(
        "Binance verisi alınamadı. Bağlantınızı veya Binance "
        f"erişimini kontrol edin. Ayrıntı: {exc}"
    )
    st.stop()

if tickers.empty:
    st.warning("Binance piyasa verisi boş döndü.")
    st.stop()

tickers = tickers[
    tickers["symbol"].isin(valid_symbols)
].copy()

tickers = tickers[
    tickers["quote_volume"] >= min_volume
].copy()

tickers = tickers.sort_values(
    "quote_volume",
    ascending=False
)

if market_scope == "En yüksek hacimli pariteler":
    selected_symbols = tickers.head(top_n)["symbol"].tolist()
else:
    selected_symbols = tickers["symbol"].tolist()

if not selected_symbols:
    st.warning(
        "Filtrelere uygun parite bulunamadı. Minimum hacim "
        "değerini düşürmeyi deneyin."
    )
    st.stop()

st.caption(
    f"Tarama listesi: {len(selected_symbols)} parite • "
    f"Zaman dilimi: {INTERVAL} • "
    f"Hedef mum sayısı: {CANDLE_LIMIT}"
)

run_scan = st.button(
    "🚀 LSTM taramasını başlat",
    type="primary",
    use_container_width=True
)

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

if run_scan:
    results = []
    progress = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                analyze_symbol,
                symbol
            ): symbol
            for symbol in selected_symbols
        }

        total = len(future_map)

        for completed, future in enumerate(
            as_completed(future_map),
            start=1
        ):
            symbol = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "symbol": symbol,
                    "error": str(exc)[:250]
                }

            results.append(result)

            progress.progress(
                completed / total
            )

            status_text.text(
                f"Analiz: {completed}/{total} — {symbol}"
            )

    st.session_state["scan_results"] = results

    progress.empty()
    status_text.empty()

    st.success(
        f"Tarama tamamlandı: {len(results)} parite işlendi."
    )


# =========================================================
# RESULTS TABLE
# =========================================================

results = st.session_state["scan_results"]

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
            ascending=[False, False]
        ).reset_index(drop=True)

        successful.insert(
            0,
            "rank",
            np.arange(1, len(successful) + 1)
        )

        successful["up_probability_pct"] = (
            successful["up_probability"] * 100
        ).round(2)

        successful["down_probability_pct"] = (
            successful["down_probability"] * 100
        ).round(2)

        successful["accuracy_pct"] = (
            successful["accuracy"] * 100
        ).round(2)

        successful["balanced_accuracy_pct"] = (
            successful["balanced_accuracy"] * 100
        ).round(2)

        buy_count = int(
            (successful["signal"] == "BUY").sum()
        )

        sell_count = int(
            (successful["signal"] == "SELL").sum()
        )

        hold_count = int(
            (successful["signal"] == "HOLD").sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Başarılı analiz",
            len(successful)
        )

        c2.metric(
            "BUY",
            buy_count
        )

        c3.metric(
            "SELL",
            sell_count
        )

        c4.metric(
            "HOLD",
            hold_count
        )

        st.subheader("🏆 LSTM Sinyal Sıralaması")

        display_columns = [
            "rank",
            "symbol",
            "signal",
            "last_close",
            "up_probability_pct",
            "down_probability_pct",
            "accuracy_pct",
            "balanced_accuracy_pct",
            "candles",
            "train_samples",
            "validation_samples"
        ]

        display_df = successful[display_columns].rename(
            columns={
                "rank": "Sıra",
                "symbol": "Parite",
                "signal": "Sinyal",
                "last_close": "Son fiyat",
                "up_probability_pct": "Yükseliş %",
                "down_probability_pct": "Düşüş %",
                "accuracy_pct": "Doğruluk %",
                "balanced_accuracy_pct": "Dengeli doğruluk %",
                "candles": "Mum sayısı",
                "train_samples": "Eğitim örneği",
                "validation_samples": "Doğrulama örneği"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = successful.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Sonuçları CSV olarak indir",
            data=csv_data,
            file_name="lstm_crypto_scan.csv",
            mime="text/csv"
        )

        st.subheader("🔎 Parite Detayı")

        available_symbols = successful["symbol"].tolist()

        selected_symbol = st.selectbox(
            "Grafiğini görmek istediğin parite",
            available_symbols
        )

        selected_row = successful[
            successful["symbol"] == selected_symbol
        ].iloc[0]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "LSTM sinyali",
            selected_row["signal"]
        )

        m2.metric(
            "Yükseliş olasılığı",
            f'{selected_row["up_probability"] * 100:.2f}%'
        )

        m3.metric(
            "Düşüş olasılığı",
            f'{selected_row["down_probability"] * 100:.2f}%'
        )

        m4.metric(
            "Doğrulama doğruluğu",
            f'{selected_row["accuracy"] * 100:.2f}%'
        )

        try:
            detail_df = get_candles(
                selected_symbol,
                requested_count=CANDLE_LIMIT
            )

            if not detail_df.empty:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=detail_df["open_time"],
                            open=detail_df["open"],
                            high=detail_df["high"],
                            low=detail_df["low"],
                            close=detail_df["close"],
                            name=selected_symbol
                        )
                    ]
                )

                fig.update_layout(
                    title=(
                        f"{selected_symbol} — "
                        "15 dakikalık mum grafiği"
                    ),
                    xaxis_title="Zaman (UTC)",
                    yaxis_title="Fiyat (USDT)",
                    xaxis_rangeslider_visible=False,
                    height=600
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    f"Son tamamlanmış mum: "
                    f"{detail_df['open_time'].iloc[-1]}"
                )

                with st.expander(
                    "Son mum verilerini görüntüle"
                ):
                    st.dataframe(
                        detail_df.tail(100).sort_values(
                            "open_time",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as exc:
            st.warning(
                f"Grafik verisi alınamadı: {exc}"
            )

    else:
        st.warning(
            "Başarılı analiz yok. Hata ayrıntılarını kontrol et."
        )

    if show_all_results and not failed.empty:
        st.subheader("⚠️ Analiz Hataları")

        st.dataframe(
            failed[["symbol", "error"]],
            use_container_width=True,
            hide_index=True
        )

else:
    st.write(
        "Tarama henüz başlatılmadı. Soldaki ayarları "
        "belirle ve **LSTM taramasını başlat** düğmesine bas."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Bilgilendirme: BUY/SELL/HOLD, modelin bir sonraki "
    "15 dakikalık mumun yönüne ilişkin sınıflandırmasıdır. "
    "Yatırım tavsiyesi değildir. Geçmiş doğruluk gelecekteki "
    "performansı garanti etmez. Bu uygulama emir göndermez."
)

st.caption(
    "Son arayüz zamanı: "
    + datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
)
