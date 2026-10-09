import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


st.set_page_config(
    page_title="LSTM Crypto Scanner",
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

CANDLE_LIMIT = 1000
INTERVAL = "15m"
LOOKBACK = 48
DEFAULT_EPOCHS = 5
MAX_WORKERS = 3
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40


# ---------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------

@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "LSTM-Crypto-Scanner/1.0"
    })
    return session


def api_get(endpoint, params=None):
    errors = []

    for base in API_BASES:
        try:
            response = get_session().get(
                base + endpoint,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            errors.append(
                f"{base}: {str(exc)[:100]}"
            )

    raise RuntimeError(
        "Binance veri adresleri başarısız:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------
# SPOT MARKETS
# ---------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def get_spot_symbols():
    data = api_get("/api/v3/exchangeInfo")
    result = []

    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            result.append(item["symbol"])

    return sorted(set(result))


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
                "low_24h": float(item["lowPrice"])
            })
        except (KeyError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# CANDLES
# ---------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_candles(symbol, count=1000):
    interval_ms = 15 * 60 * 1000
    now_ms = int(time.time() * 1000)
    end_ms = (now_ms // interval_ms) * interval_ms - 1

    rows = []

    while len(rows) < count:
        limit = min(1000, count - len(rows))

        page = api_get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": INTERVAL,
                "limit": limit,
                "endTime": end_ms
            }
        )

        if not page:
            break

        rows = page + rows
        end_ms = int(page[0][0]) - 1

        if len(page) < limit or len(rows) >= count:
            break

    if not rows:
        return pd.DataFrame()

    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_base", "taker_quote", "ignore"
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

    for col in [
        "open", "high", "low", "close",
        "volume", "quote_volume"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close_time"] < now_ms]

    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .dropna(
            subset=[
                "open", "high", "low",
                "close", "volume", "quote_volume"
            ]
        )
        .tail(count)
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------
# FEATURES: RAW PRICE AND VOLUME ONLY
# ---------------------------------------------------------

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
    volume_change = np.zeros(len(volume))
    quote_change = np.zeros(len(quote_volume))

    returns[1:] = np.log(close[1:] / close[:-1])
    volume_change[1:] = np.log(volume[1:] / volume[:-1])
    quote_change[1:] = np.log(
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
        volume_change,
        quote_change
    ])

    return np.nan_to_num(
        features,
        nan=0,
        posinf=0,
        neginf=0
    ).astype(np.float32)


# ---------------------------------------------------------
# LSTM
# ---------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        def __init__(self, input_size=7, hidden_size=48):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )

            self.output = nn.Sequential(
                nn.Linear(hidden_size, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(24, 2)
            )

        def forward(self, x):
            result, _ = self.lstm(x)
            return self.output(result[:, -1, :])


# ---------------------------------------------------------
# TRAINING AND PREDICTION
# ---------------------------------------------------------

def train_model(df, epochs):
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch kurulmamış. requirements.txt dosyasını kontrol et."
        )

    if len(df) < 200:
        raise ValueError("En az 200 mum gerekli.")

    features = make_features(df)
    closes = df["close"].to_numpy(dtype=np.float64)
    split = int(len(features) * 0.8)

    mean = features[:split].mean(axis=0)
    std = features[:split].std(axis=0)
    std[std < 1e-8] = 1.0

    scaled = (features - mean) / std
    scaled = np.nan_to_num(
        scaled,
        nan=0,
        posinf=0,
        neginf=0
    ).astype(np.float32)

    X, y, positions = [], [], []

    for i in range(LOOKBACK, len(df)):
        X.append(scaled[i - LOOKBACK:i])
        y.append(int(closes[i] > closes[i - 1]))
        positions.append(i)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    positions = np.asarray(positions)

    train_mask = positions < split
    val_mask = positions >= split

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_train) < 50 or len(X_val) < 10:
        raise ValueError("Eğitim/doğrulama verisi yetersiz.")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)

    model = LSTMModel(input_size=features.shape[1])

    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = len(y_train) / (2 * np.maximum(counts, 1))

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    loader = DataLoader(
        dataset,
        batch_size=min(64, len(dataset)),
        shuffle=True
    )

    model.train()

    for _ in range(int(epochs)):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

    model.eval()

    with torch.no_grad():
        val_logits = model(
            torch.tensor(X_val, dtype=torch.float32)
        )

        val_probs = torch.softmax(val_logits, dim=1).numpy()
        val_predictions = np.argmax(val_probs, axis=1)

        accuracy = float(np.mean(val_predictions == y_val))

        recalls = []

        for class_id in (0, 1):
            mask = y_val == class_id

            if mask.any():
                recalls.append(
                    float(np.mean(val_predictions[mask] == class_id))
                )

        balanced_accuracy = (
            float(np.mean(recalls))
            if recalls else float("nan")
        )

        latest_x = torch.tensor(
            scaled[-LOOKBACK:][None, :, :],
            dtype=torch.float32
        )

        probs = torch.softmax(model(latest_x), dim=1)[0].numpy()

    down_probability = float(probs[0])
    up_probability = float(probs[1])

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
        "validation_samples": len(X_val)
    }


# ---------------------------------------------------------
# ANALYZE SYMBOL
# ---------------------------------------------------------

def analyze_symbol(symbol, epochs):
    try:
        df = get_candles(symbol, CANDLE_LIMIT)

        if df.empty or len(df) < 200:
            raise ValueError("Yeterli tamamlanmış mum bulunamadı.")

        result = train_model(df, epochs)

        return {
            "symbol": symbol,
            "signal": result["signal"],
            "price": float(df["close"].iloc[-1]),
            "up_probability": result["up_probability"],
            "down_probability": result["down_probability"],
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "candles": len(df),
            "train_samples": result["train_samples"],
            "validation_samples": result["validation_samples"],
            "error": None
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "error": str(exc)[:500]
        }


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Tarama ayarları")

scan_mode = st.sidebar.selectbox(
    "Parite kapsamı",
    [
        "En yüksek hacimli pariteler",
        "Tüm USDT Spot pariteleri"
    ]
)

coin_count = st.sidebar.slider(
    "Taranacak parite sayısı",
    5, 100, 20, 5
)

minimum_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0.0,
    value=1_000_000.0,
    step=500_000.0
)

epochs = st.sidebar.slider(
    "LSTM eğitim turu",
    2, 15, DEFAULT_EPOCHS
)

show_errors = st.sidebar.checkbox(
    "Hatalı pariteleri göster",
    value=True
)

if st.sidebar.button("Önbelleği temizle"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------
# MARKET LOADING
# ---------------------------------------------------------

if not TORCH_AVAILABLE:
    st.error(
        "PyTorch kurulu değil. Önce requirements.txt dosyasını "
        "güncelle ve Streamlit Cloud'un paket kurulumunu tamamlamasını bekle."
    )
    st.code("torch")
    st.stop()

try:
    with st.spinner("Binance Spot verileri alınıyor..."):
        spot_symbols = set(get_spot_symbols())
        tickers = get_tickers()

except Exception as exc:
    st.error("Binance verileri alınamadı.")
    st.code(str(exc))
    st.stop()

if tickers.empty:
    st.error("Binance ticker verisi boş.")
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
    symbols_to_scan = tickers.head(coin_count)["symbol"].tolist()
else:
    symbols_to_scan = tickers["symbol"].tolist()

if not symbols_to_scan:
    st.warning("Filtrelere uygun parite yok. Minimum hacmi düşür.")
    st.stop()

st.caption(
    f"{len(symbols_to_scan)} parite | 15 dakika | "
    "Hedef: 1.000 tamamlanmış mum"
)


# ---------------------------------------------------------
# SCAN
# ---------------------------------------------------------

if "results" not in st.session_state:
    st.session_state["results"] = []

if st.button(
    "🚀 LSTM taramasını başlat",
    type="primary",
    use_container_width=True
):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(analyze_symbol, symbol, epochs): symbol
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
                    "error": str(exc)[:500]
                })

            progress.progress(index / total)
            status.write(
                f"Analiz: {symbol} ({index}/{total})"
            )

    st.session_state["results"] = results
    progress.empty()
    status.empty()

    st.success("Tarama tamamlandı.")
    st.rerun()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

results = st.session_state["results"]

if results:
    result_df = pd.DataFrame(results)

    if "error" not in result_df.columns:
        result_df["error"] = None

    successful = result_df[result_df["error"].isna()].copy()
    failed = result_df[result_df["error"].notna()].copy()

    if not successful.empty:
        successful = successful.sort_values(
            ["up_probability", "accuracy"],
            ascending=[False, False]
        ).reset_index(drop=True)

        successful.insert(0, "rank", np.arange(1, len(successful) + 1))

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

        st.subheader("🏆 LSTM sıralaması")

        columns = [
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
            "validation_samples"
        ]

        display = successful[columns].rename(
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
                "validation_samples": "Doğrulama örneği"
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 CSV indir",
            data=successful.to_csv(index=False).encode("utf-8-sig"),
            file_name="lstm_crypto_results.csv",
            mime="text/csv"
        )

        st.subheader("🔎 Parite detayı")

        selected_symbol = st.selectbox(
            "Parite seç",
            successful["symbol"].tolist()
        )

        selected = successful[
            successful["symbol"] == selected_symbol
        ].iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Sinyal", selected["signal"])
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
            chart_df = get_candles(selected_symbol, CANDLE_LIMIT)

            if not chart_df.empty:
                chart = chart_df.set_index("open_time")

                st.subheader(
                    f"{selected_symbol} — kapanış fiyatı"
                )

                st.line_chart(chart[["close"]], height=350)

                st.subheader("Yüksek/düşük fiyat aralığı")

                st.area_chart(
                    chart[["low", "high"]],
                    height=250
                )

                with st.expander("Son 100 mum verisi"):
                    st.dataframe(
                        chart_df.tail(100).sort_values(
                            "open_time",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as exc:
            st.warning(f"Grafik yüklenemedi: {exc}")

    else:
        st.warning("Başarılı analiz bulunamadı.")

    if show_errors and not failed.empty:
        st.subheader("⚠️ Analiz hataları")

        st.dataframe(
            failed[["symbol", "error"]],
            use_container_width=True,
            hide_index=True
        )

else:
    st.write(
        "Henüz tarama yapılmadı. Sol taraftaki ayarları seç "
        "ve taramayı başlat."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Model sonraki 15 dakikalık mumun yönünü sınıflandırmaya çalışır. "
    "Geçmiş doğruluk gelecekteki performansı garanti etmez."
)

st.caption(
    "Otomatik alım satım yoktur. UTC: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
)
