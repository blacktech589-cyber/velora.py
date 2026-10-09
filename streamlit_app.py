import os
import time
import hmac
import hashlib
from urllib.parse import urlencode
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
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deep Learning Crypto Scanner",
    page_icon="🧠",
    layout="wide",
)

BINANCE_API = "https://api.binance.com"
BINANCE_TESTNET = "https://testnet.binance.vision"

CANDLE_COUNT = 1000
INTERVAL_SECONDS = 900
REQUEST_TIMEOUT = 20

HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": "LSTM-Spot-Scanner/1.0"
})


# ============================================================
# BINANCE PUBLIC API
# ============================================================

def api_get(url, params=None):
    response = HTTP.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def binance_get(path, params=None):
    hosts = [
        BINANCE_API,
        "https://api-gcp.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
    ]

    errors = []

    for host in hosts:
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
            errors.append(f"{host}: {exc}")

    raise RuntimeError(
        "Binance API bağlantısı başarısız: "
        + " | ".join(errors)
    )


def get_spot_markets():
    data = binance_get("/api/v3/exchangeInfo")

    markets = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        markets.append(item["symbol"])

    return markets


def get_24h_tickers():
    data = binance_get("/api/v3/ticker/24hr")
    result = {}

    for item in data:
        try:
            result[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h_pct": float(
                    item["priceChangePercent"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return result


# ============================================================
# CANDLE DATA
# ============================================================

def get_candles(symbol, limit=CANDLE_COUNT):
    """
    Son tamamlanmış 15 dakikalık mumları getirir.
    Devam eden mum analiz dışında bırakılır.
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
        except (ValueError, TypeError, IndexError):
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"{symbol}: Mum verisi alınamadı."
        )

    df = df.drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp")

    # Sadece tamamlanmış mumlar.
    df = df[
        df["timestamp"] < current_candle_start
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "open", "high", "low",
            "close", "volume",
        ]
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
# DEEP LEARNING FEATURES
# ============================================================

def make_features(df):
    """
    Yalnızca mum ve hacim verilerinden özellikler oluşturur.
    RSI, EMA, MACD veya indikatör oylaması kullanılmaz.
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
# LSTM MODEL
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
                    dropout if num_layers > 1 else 0
                ),
            )

            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            sequence, _ = self.lstm(x)
            last_step = sequence[:, -1, :]
            return self.output_layer(last_step)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lstm(
    candles,
    lookback=48,
    epochs=5,
):
    """
    Sonraki mumun kapanışının yukarıda olma
    olasılığını tahmin eder.

    Veri kronolojik olarak eğitim ve doğrulama
    bölümlerine ayrılır.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch kurulu değil. "
            "python -m pip install torch"
        )

    if len(candles) < lookback + 150:
        raise ValueError(
            "Model eğitimi için yeterli mum yok."
        )

    features_df = make_features(candles)

    close = candles["close"].to_numpy(
        dtype=np.float64
    )

    # Hedef: sonraki kapanış mevcut kapanıştan yüksek mi?
    labels = (
        close[1:] > close[:-1]
    ).astype(np.float32)

    # Her hedef için yalnızca bilinen geçmiş özellikler.
    features_df = features_df.iloc[:-1].copy()

    split = int(len(features_df) * 0.8)

    if split <= lookback:
        raise ValueError(
            "Eğitim verisi yetersiz."
        )

    # Ölçekleyici yalnızca eğitim döneminde hesaplanır.
    means = features_df.iloc[:split].mean()

    stds = (
        features_df.iloc[:split]
        .std()
        .replace(0, 1)
        .fillna(1)
    )

    scaled = (
        (features_df - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    latest_features = make_features(
        candles
    )

    latest_scaled = (
        (latest_features - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    sequences = []
    targets = []
    end_indices = []

    for end in range(
        lookback - 1,
        len(scaled),
    ):
        sequences.append(
            scaled[
                end - lookback + 1:
                end + 1
            ]
        )

        targets.append(labels[end])
        end_indices.append(end)

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    end_indices = np.asarray(end_indices)

    train_mask = end_indices < split
    validation_mask = ~train_mask

    if train_mask.sum() < 50:
        raise ValueError(
            "Eğitim örnekleri yetersiz."
        )

    if validation_mask.sum() < 20:
        raise ValueError(
            "Doğrulama örnekleri yetersiz."
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

    batch_size = min(
        128,
        len(train_x),
    )

    model.train()

    for _ in range(int(epochs)):
        permutation = torch.randperm(
            len(train_x),
            device=device,
        )

        for start in range(
            0,
            len(permutation),
            batch_size,
        ):
            indices = permutation[
                start:start + batch_size
            ]

            optimizer.zero_grad()

            logits = model(
                train_x[indices]
            )

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

    # Kronolojik doğrulama.
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
# BUY / SELL / HOLD SIGNALS
# ============================================================

def get_signal(up_probability):
    if not np.isfinite(up_probability):
        return "HOLD"

    if up_probability >= 60:
        return "BUY"

    if up_probability <= 40:
        return "SELL"

    return "HOLD"


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_coin(
    symbol,
    ticker,
    lookback,
    epochs,
):
    candles = get_candles(
        symbol,
        CANDLE_COUNT,
    )

    prediction = train_lstm(
        candles,
        lookback=lookback,
        epochs=epochs,
    )

    close = float(
        candles.iloc[-1]["close"]
    )

    return {
        "symbol": symbol,
        "price": ticker["price"],
        "last_completed_close": close,
        "change_24h_pct": ticker["change_24h_pct"],
        "volume_24h_usdt": ticker["quote_volume"],
        "candles_used": len(candles),
        "up_probability_pct": prediction[
            "up_probability_pct"
        ],
        "down_probability_pct": prediction[
            "down_probability_pct"
        ],
        "validation_accuracy_pct": prediction[
            "validation_accuracy_pct"
        ],
        "training_samples": prediction[
            "training_samples"
        ],
        "validation_samples": prediction[
            "validation_samples"
        ],
    }


# ============================================================
# BINANCE SPOT TESTNET SIGNED REQUEST
# ============================================================

def testnet_signed_request(
    method,
    path,
    params,
):
    """
    Bu fonksiyon yalnızca Spot TESTNET adresini kullanır.
    Gerçek Binance API adresine emir göndermez.
    """

    api_key = os.getenv(
        "BINANCE_TESTNET_API_KEY",
        "",
    ).strip()

    api_secret = os.getenv(
        "BINANCE_TESTNET_API_SECRET",
        "",
    ).strip()

    if not api_key or not api_secret:
        raise RuntimeError(
            "Testnet API anahtarları tanımlı değil. "
            "BINANCE_TESTNET_API_KEY ve "
            "BINANCE_TESTNET_API_SECRET değişkenlerini ayarlayın."
        )

    payload = dict(params)

    payload["timestamp"] = int(
        time.time() * 1000
    )

    payload["recvWindow"] = 5000

    query_string = urlencode(payload)

    signature = hmac.new(
        api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    url = (
        f"{BINANCE_TESTNET}{path}"
        f"?{query_string}&signature={signature}"
    )

    response = HTTP.request(
        method,
        url,
        headers={
            "X-MBX-APIKEY": api_key,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def submit_testnet_order(
    symbol,
    side,
    usdt_amount=0,
    coin_quantity=0,
):
    if side not in ("BUY", "SELL"):
        raise ValueError(
            "Emir BUY veya SELL olmalıdır."
        )

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "newOrderRespType": "FULL",
    }

    if side == "BUY":
        if usdt_amount <= 0:
            raise ValueError(
                "USDT alım tutarı sıfırdan büyük olmalıdır."
            )

        params["quoteOrderQty"] = (
            f"{usdt_amount:.8f}"
            .rstrip("0")
            .rstrip(".")
        )

    else:
        if coin_quantity <= 0:
            raise ValueError(
                "Satılacak coin miktarı sıfırdan büyük olmalıdır."
            )

        params["quantity"] = (
            f"{coin_quantity:.8f}"
            .rstrip("0")
            .rstrip(".")
        )

    return testnet_signed_request(
        "POST",
        "/api/v3/order",
        params,
    )


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🧠 LSTM Deep Learning Crypto Scanner")

st.write(
    "Binance Spot USDT piyasaları üzerinde "
    "derin öğrenme tabanlı tahmin."
)

st.info(
    "Her coin için son 1.000 tamamlanmış 15 dakikalık "
    "mum kullanılır. RSI, EMA, MACD veya indikatör "
    "puanlaması kullanılmaz."
)

st.warning(
    "BUY / SELL / HOLD model sinyalleridir; kâr garantisi "
    "değildir. Emirler yalnızca Binance Spot Testnet'e "
    "gönderilir. Canlı işlemler bu uygulamada kapalıdır."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Tarama ayarları")

    scan_count = st.number_input(
        "Taranacak coin sayısı",
        min_value=1,
        max_value=200,
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

    st.divider()

    st.header("Testnet emir ayarları")

    usdt_amount = st.number_input(
        "BUY tutarı (USDT)",
        min_value=5.0,
        max_value=100000.0,
        value=10.0,
        step=5.0,
    )

    coin_quantity = st.number_input(
        "SELL miktarı (coin)",
        min_value=0.0,
        value=0.0,
        step=0.001,
        format="%.8f",
    )

    testnet_enabled = st.checkbox(
        "Testnet emirlerini etkinleştir",
        value=False,
    )

    st.caption(
        "Testnet emirleri varsayılan olarak kapalıdır. "
        "Gerçek API anahtarlarını kullanmayın."
    )

    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch kurulu değil. "
            "Terminalde: python -m pip install torch"
        )

    start_scan = st.button(
        "Derin öğrenme taramasını başlat",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# SESSION STATE
#
