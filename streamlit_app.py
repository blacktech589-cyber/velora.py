```python
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# ============================================================
# AYARLAR
# ============================================================

st.set_page_config(
    page_title="LSTM Crypto Dashboard",
    page_icon="🧠",
    layout="wide",
)

CANDLE_COUNT = 1000
INTERVAL_SECONDS = 15 * 60
REQUEST_TIMEOUT = 20

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": "LSTM-Crypto-Dashboard/1.0"
})


# ============================================================
# BINANCE API
# ============================================================

def binance_get(path, params=None):
    errors = []

    for host in BINANCE_HOSTS:
        try:
            response = HTTP.get(
                host + path,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:120]}")

    raise RuntimeError(
        "Binance API bağlantısı başarısız: "
        + " | ".join(errors)
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_spot_symbols():
    data = binance_get("/api/v3/exchangeInfo")
    symbols = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        symbols.append(item["symbol"])

    return sorted(set(symbols))


@st.cache_data(ttl=60, show_spinner=False)
def get_tickers():
    data = binance_get("/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        try:
            tickers[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "change_24h_pct": float(
                    item["priceChangePercent"]
                ),
                "quote_volume": float(
                    item["quoteVolume"]
                ),
                "high_24h": float(item["highPrice"]),
                "low_24h": float(item["lowPrice"]),
            }
        except (KeyError, ValueError, TypeError):
            continue

    return tickers


@st.cache_data(ttl=60, show_spinner=False)
def get_candles(symbol, limit=CANDLE_COUNT):
    """
    Son 1000 tamamlanmış 15 dakikalık mumu alır.
    Devam eden mum analiz dışı bırakılır.
    """

    now = int(
        datetime.now(timezone.utc).timestamp()
    )

    current_candle_start = (
        now // INTERVAL_SECONDS
    ) * INTERVAL_SECONDS

    end_ms = current_candle_start * 1000

    start_ms = (
        end_ms
        - (limit + 5) * INTERVAL_SECONDS * 1000
    )

    data = binance_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "15m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": min(1000, limit + 5),
        },
    )

    rows = []

    for candle in data:
        try:
            rows.append({
                "timestamp": int(candle[0]) // 1000,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "quote_volume": float(candle[7]),
            })
        except (IndexError, TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"{symbol}: mum verisi alınamadı."
        )

    df = df.drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp")

    df = df[
        df["timestamp"] < current_candle_start
    ]

    numeric_columns = [
        "open", "high", "low", "close",
        "volume", "quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df[
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
    ]

    return (
        df.tail(limit)
        .reset_index(drop=True)
    )


# ============================================================
# DERİN ÖĞRENME GİRDİLERİ
# ============================================================

def make_features(df):
    """
    Ham mum ve hacim verilerinden model girdileri üretir.
    Teknik indikatör kullanılmaz.
    """

    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    quote_volume = df["quote_volume"].astype(float)

    features = pd.DataFrame(index=df.index)

    features["close_return"] = np.log(
        close / close.shift(1)
    )

    features["open_close_return"] = np.log(
        close / open_price
    )

    features["high_close_return"] = np.log(
        high / close
    )

    features["low_close_return"] = np.log(
        low / close
    )

    features["volume_change"] = np.log1p(
        volume.clip(lower=0)
    ).diff()

    features["quote_volume_change"] = np.log1p(
        quote_volume.clip(lower=0)
    ).diff()

    return features.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================
# LSTM MODELİ
# ============================================================

if TORCH_AVAILABLE:

    class PriceLSTM(nn.Module):
        def __init__(
            self,
            input_size=6,
            hidden_size=48,
            num_layers=2,
            dropout=0.15,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=(
                    dropout if num_layers > 1 else 0.0
                ),
            )

            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])


# ============================================================
# MODEL EĞİTİMİ VE TAHMİN
# ============================================================

def train_and_predict(
    candles,
    lookback=48,
    epochs=5,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch kurulu değil. "
            "pip install torch komutunu çalıştırın."
        )

    if len(candles) < lookback + 150:
        raise ValueError(
            "Model eğitimi için yeterli mum yok."
        )

    feature_df = make_features(candles)

    close = candles["close"].to_numpy(
        dtype=np.float64
    )

    # Bir sonraki mumun kapanışı yukarıda mı?
    labels = (
        close[1:] > close[:-1]
    ).astype(np.float32)

    # Her özellik satırının hedefi bir sonraki mumdur.
    feature_df = feature_df.iloc[:-1].copy()

    split = int(len(feature_df) * 0.8)

    if split <= lookback:
        raise ValueError(
            "Eğitim bölümü çok küçük."
        )

    # Ölçekleme parametreleri sadece eğitim verisinden.
    train_features = feature_df.iloc[:split]

    means = train_features.mean()

    stds = (
        train_features.std()
        .replace(0, 1)
        .fillna(1)
    )

    scaled = (
        (feature_df - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    latest_features = make_features(candles)

    latest_scaled = (
        (latest_features - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    sequences = []
    targets = []
    end_indices = []

    for end in range(lookback - 1, len(scaled)):
        sequences.append(
            scaled[end - lookback + 1:end + 1]
        )
        targets.append(labels[end])
        end_indices.append(end)

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    end_indices = np.asarray(end_indices)

    train_mask = end_indices < split
    validation_mask = ~train_mask

    if train_mask.sum() < 50:
        raise ValueError(
            "Yeterli eğitim örneği yok."
        )

    if validation_mask.sum() < 20:
        raise ValueError(
            "Yeterli doğrulama örneği yok."
        )

    torch.manual_seed(42)
    torch.set_num_threads(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = PriceLSTM(
        input_size=X.shape[2]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    loss_function = nn.BCEWithLogitsLoss()

    train_x = torch.tensor(
        X[train_mask],
        dtype=torch.float32,
        device=device,
    )

    train_y = torch.tensor(
        y[train_mask, None],
        dtype=torch.float32,
        device=device,
    )

    batch_size = min(128, len(train_x))

    model.train()

    for _ in range(int(epochs)):
        order = torch.randperm(
            len(train_x),
            device=device,
        )

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]

            optimizer.zero_grad()

            logits = model(train_x[indices])

            loss = loss_function(
                logits,
                train_y[indices],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

    model.eval()

    with torch.no_grad():
        validation_x = torch.tensor(
            X[validation_mask],
            dtype=torch.float32,
            device=device,
        )

        validation_y = y[validation_mask]

        validation_probabilities = torch.sigmoid(
            model(validation_x)
        ).cpu().numpy().reshape(-1)

        validation_predictions = (
            validation_probabilities >= 0.5
        ).astype(np.float32)

        validation_accuracy = float(
            np.mean(
                validation_predictions == validation_y
            ) * 100
        )

        latest_sequence = torch.tensor(
            latest_scaled[-lookback:][None, :, :],
            dtype=torch.float32,
            device=device,
        )

        probability_up = float(
            torch.sigmoid(
                model(latest_sequence)
            ).item()
        )

    return {
        "up_probability_pct": probability_up * 100,
        "down_probability_pct": (
            1 - probability_up
        ) * 100,
        "validation_accuracy_pct": validation_accuracy,
        "training_samples": int(train_mask.sum()),
        "validation_samples": int(
            validation_mask.sum()
        ),
    }


# ============================================================
# MODEL SİNYALİ
# ============================================================

def get_signal(probability):
    if not np.isfinite(probability):
        return "HOLD"

    if probability >= 60:
        return "BUY"

    if probability <= 40:
        return "SELL"

    return "HOLD"


# ============================================================
# TEK COIN ANALİZİ
# ============================================================

def analyze_coin(
    symbol,
    ticker,
    lookback,
    epochs,
):
    candles = get_candles(symbol, CANDLE_COUNT)

    prediction = train_and_predict(
        candles,
        lookback=lookback,
        epochs=epochs,
    )

    return {
        "symbol": symbol,
        "price": ticker["price"],
        "change_24h_pct": ticker["change_24h_pct"],
        "volume_24h_usdt": ticker["quote_volume"],
        "last_completed_close": float(
            candles.iloc[-1]["close"]
        ),
        "candles_used": len(candles),
        "history_days": round(
            len(candles) * 15 / 60 / 24,
            2,
        ),
        **prediction,
    }


# ============================================================
# GÖSTERGE PANELİ
# ============================================================

st.title("🧠 LSTM Crypto Dashboard")

st.write(
    "Binance Spot USDT piyasalarını derin öğrenme "
    "modeliyle analiz eden gösterge paneli."
)

st.info(
    "Yalnızca mum ve hacim verileri kullanılır. "
    "RSI, EMA, MACD, Bollinger Bands veya "
    "indikatör puanlaması yoktur."
)

st.warning(
    "BUY / SELL / HOLD tahminlerdir, kâr garantisi değildir. "
    "Bu uygulama yalnızca veri ve sinyal gösterir; "
    "hiçbir alım-satım emri göndermez."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Tarama ayarları")

    scan_count = st.number_input(
        "Taranacak coin sayısı",
        min_value=1,
        max_value=100,
        value=20,
        step=5,
    )

    lookback = st.selectbox(
        "LSTM geçmiş uzunluğu",
        [24, 48, 72, 96],
        index=1,
    )

    epochs = st.slider(
        "Eğitim turu",
        min_value=1,
        max_value=15,
        value=5,
    )

    start_scan = st.button(
        "Taramayı başlat",
        type="primary",
        use_container_width=True,
    )

    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch kurulu değil. "
            "Terminalde: python -m pip install torch"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_time" not in st.session_state:
    st.session_state.scan_time = None


# ============================================================
# SCAN
# ============================================================

if start_scan:
    if not TORCH_AVAILABLE:
        st.error(
            "Önce PyTorch kurun: python -m pip install torch"
        )
    else:
        progress = st.progress(0)
        status = st.empty()

        results = []
        errors = []

        try:
            status.write("Binance Spot piyasaları yükleniyor...")

            symbols = get_spot_symbols()
            tickers = get_tickers()

            candidates = []

            for symbol in symbols:
                ticker = tickers.get(symbol)

                if ticker is not None:
                    candidates.append({
                        "symbol": symbol,
                        **ticker,
                    })

            # Hacmi yüksek coinler önce taranır.
            candidates.sort(
                key=lambda row: row["quote_volume"],
                reverse=True,
            )

            candidates = candidates[:int(scan_count)]

            if not candidates:
                raise RuntimeError(
                    "Aktif USDT Spot piyasası bulunamadı."
                )

            for index, item in enumerate(candidates):
                symbol = item["symbol"]

                status.write(
                    f"{index + 1}/{len(candidates)} "
                    f"— {symbol} analiz ediliyor..."
                )

                try:
                    result = analyze_coin(
                        symbol,
                        item,
                        int(lookback),
                        int(epochs),
                    )

                    result["signal"] = get_signal(
                        result["up_probability_pct"]
                    )

                    results.append(result)

                except Exception as exc:
                    errors.append({
                        "symbol": symbol,
                        "error": str(exc)[:250],
                    })

                progress.progress(
                    (index + 1) / len(candidates)
                )

                time.sleep(0.05)

            st.session_state.results = pd.DataFrame(results)
            st.session_state.scan_errors = errors
            st.session_state.scan_time = (
                datetime.now(timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S UTC")
            )

            status.success(
                f"Tarama tamamlandı: {len(results)} başarılı, "
                f"{len(errors)} hatalı."
            )

        except Exception as exc:
            st.error(f"Tarama başarısız: {exc}")


# ============================================================
# SONUÇLAR
# ============================================================

results = st.session_state.results

if results is None:
    st.info(
        "Sol panelden ayarları seçip taramayı başlatın."
    )

elif results.empty:
    st.warning(
        "Sonuç bulunamadı. Binance bağlantısını ve "
        "tarama hatalarını kontrol edin."
    )

else:
    st.caption(
        f"Son tarama: {st.session_state.scan_time or '-'}"
    )

    if "up_probability_pct" not in results.columns:
        results["up_probability_pct"] = np.nan

    results["up_probability_pct"] = pd.to_numeric(
        results["up_probability_pct"],
        errors="coerce",
    )

    results["down_probability_pct"] = (
        100 - results["up_probability_pct"]
    )

    results["signal"] = results[
        "up_probability_pct"
    ].apply(get_signal)

    results = (
        results.sort_values(
            "up_probability_pct",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    st.session_state.results = results

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Analiz edilen coin",
        len(results),
    )

    col2.metric(
        "BUY",
        int((results["signal"] == "BUY").sum()),
    )

    col3.metric(
        "SELL",
        int((results["signal"] == "SELL").sum()),
    )

    col4.metric(
        "HOLD",
        int((results["signal"] == "HOLD").sum()),
    )

    st.subheader("Model sonuçları")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "CSV olarak indir",
        data=results.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="lstm_dashboard_results.csv",
        mime="text/csv",
    )

    st.subheader("En yüksek yükseliş olasılığı")

    st.dataframe(
        results.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Coin ayrıntısı")

    selected_symbol = st.selectbox(
        "Coin seç",
        results["symbol"].tolist(),
    )

    selected = results.loc[
        results["symbol"] == selected_symbol
    ].iloc[0]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Anlık fiyat",
        f"{selected['price']:.8g} USDT",
    )

    m2.metric(
        "Yükseliş olasılığı",
        f"{selected['up_probability_pct']:.2f}%",
    )

    m3.metric(
        "Düşüş olasılığı",
        f"{selected['down_probability_pct']:.2f}%",
    )

    st.metric(
        "Model sinyali",
        selected["signal"],
    )

    st.metric(
        "Kronolojik doğrulama doğruluğu",
        f"{selected['validation_accuracy_pct']:.2f}%",
    )

    candles = get_candles(
        selected_symbol,
        CANDLE_COUNT,
    )

    st.subheader(
        f"{selected_symbol} — son 1000 tamamlanmış mum"
    )

    chart_data = candles.copy()

    chart_data["datetime"] = pd.to_datetime(
        chart_data["timestamp"],
        unit="s",
        utc=True,
    )

    st.line_chart(
        chart_data.set_index("datetime")["close"],
        height=350,
    )

    st.caption(
        "Doğrulama doğruluğu geçmiş bir veri bölümünde ölçülür; "
        "gelecekteki performansın garantisi değildir. "
        "Bu gösterge paneli emir göndermez."
    )

    if st.session_state.scan_errors:
        with st.expander(
            f"Tarama hataları "
            f"({len(st.session_state.scan_errors)})"
        ):
            st.dataframe(
                pd.DataFrame(
                    st.session_state.scan_errors
                ),
                use_container_width=True,
                hide_index=True,
            )
```
