import time
import os
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
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


st.set_page_config(
    page_title="Deep Learning Spot Scanner",
    page_icon="🧠",
    layout="wide"
)

BINANCE_TESTNET = "https://testnet.binance.vision"

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

GATE = "https://api.gateio.ws/api/v4"
TIMEOUT = 20
CANDLE_COUNT = 1000

session = requests.Session()
session.headers.update({
    "User-Agent": "DeepLearningSpotScanner/1.0"
})


# ============================================================
# API CONNECTIONS
# ============================================================

def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def binance_json(path, params=None):
    errors = []

    for host in BINANCE_HOSTS:
        try:
            response = session.get(
                host + path,
                params=params,
                timeout=TIMEOUT
            )

            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue

            response.raise_for_status()

            return response.json(), host

        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:100]}")

    raise RuntimeError(
        "Binance API erişimi başarısız: "
        + " | ".join(errors)
    )


def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()

    out = df.copy()
    seen = {}
    names = []

    for col in out.columns:
        name = str(col)
        n = seen.get(name, 0)
        seen[name] = n + 1

        names.append(
            name if n == 0 else f"{name}_{n}"
        )

    out.columns = names
    return out


# ============================================================
# MARKET DISCOVERY
# ============================================================

def binance_markets():
    data, host = binance_json(
        "/api/v3/exchangeInfo"
    )

    markets = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        markets.append({
            "symbol": item["symbol"],
            "market_id": item["symbol"],
            "provider": "Binance"
        })

    if not markets:
        raise RuntimeError(
            "Aktif Binance Spot USDT piyasası bulunamadı."
        )

    return markets, host


def binance_tickers():
    data, host = binance_json(
        "/api/v3/ticker/24hr"
    )

    out = {}

    for item in data:
        try:
            out[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(
                    item["quoteVolume"]
                ),
                "change_24h_pct": float(
                    item["priceChangePercent"]
                )
            }

        except (KeyError, TypeError, ValueError):
            continue

    return out, host


def gate_markets():
    data = get_json(
        f"{GATE}/spot/currency_pairs"
    )

    out = []

    for item in data:
        if item.get("quote") != "USDT":
            continue

        if item.get("trade_status") != "tradable":
            continue

        if item.get("delisted") is True:
            continue

        base = item.get("base", "")
        market_id = item.get("id", "")

        if base and market_id:
            out.append({
                "symbol": base + "USDT",
                "market_id": market_id,
                "provider": "Gate.io"
            })

    if not out:
        raise RuntimeError(
            "Aktif Gate.io Spot USDT piyasası bulunamadı."
        )

    return out


def gate_tickers():
    data = get_json(
        f"{GATE}/spot/tickers"
    )

    out = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            out[market_id] = {
                "price": float(item["last"]),
                "quote_volume": float(
                    item["quote_volume"]
                ),
                "change_24h_pct": float(
                    item["change_percentage"]
                )
            }

        except (KeyError, TypeError, ValueError):
            continue

    return out


def discover(exchange):
    if exchange == "Gate.io only":
        return (
            gate_markets(),
            gate_tickers(),
            "Gate.io"
        )

    try:
        markets, _ = binance_markets()
        tickers, _ = binance_tickers()

        return markets, tickers, "Binance"

    except Exception:
        if exchange == "Binance only":
            raise

        return (
            gate_markets(),
            gate_tickers(),
            "Gate.io (fallback)"
        )


# ============================================================
# CANDLE DATA
# ============================================================

def clean_candles(rows):
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=cols[:6]
    )

    df = df.drop_duplicates(
        "timestamp"
    ).sort_values("timestamp")

    now = int(
        datetime.now(timezone.utc).timestamp()
    )

    current_open = now // 900 * 900

    # Yalnızca tamamlanmış 15 dakikalık mumlar.
    df = df[
        df["timestamp"] < current_open
    ]

    df = df[
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
    ]

    return df.reset_index(drop=True)


def binance_candles(
    symbol,
    candle_count=CANDLE_COUNT
):
    """Binance'tan son tamamlanmış 15m mumlarını getirir."""

    now = int(
        datetime.now(timezone.utc).timestamp()
    )

    end_ms = (now // 900 * 900) * 1000

    start_ms = (
        end_ms
        - (int(candle_count) + 5) * 900 * 1000
    )

    data, _ = binance_json(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "15m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": min(
                1000,
                int(candle_count) + 5
            )
        }
    )

    rows = []

    for c in data:
        try:
            rows.append({
                "timestamp": int(c[0]) // 1000,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "quote_volume": float(c[7])
            })

        except (ValueError, TypeError, IndexError):
            continue

    return (
        clean_candles(rows)
        .tail(int(candle_count))
        .reset_index(drop=True)
    )


def gate_candles(
    market_id,
    candle_count=CANDLE_COUNT
):
    """Gate.io mumlarını getirir."""

    now = int(
        datetime.now(timezone.utc).timestamp()
    )

    end = now // 900 * 900
    start = end - (
        int(candle_count) + 5
    ) * 900

    data = get_json(
        f"{GATE}/spot/candlesticks",
        {
            "currency_pair": market_id,
            "interval": "15m",
            "from": start,
            "to": end,
            "limit": min(
                1000,
                int(candle_count) + 5
            )
        }
    )

    rows = []

    for c in data:
        try:
            rows.append({
                "timestamp": int(c[0]),
                "quote_volume": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "open": float(c[5]),
                "volume": (
                    float(c[6])
                    if len(c) > 6
                    else 0.0
                )
            })

        except (ValueError, TypeError, IndexError):
            continue

    return (
        clean_candles(rows)
        .tail(int(candle_count))
        .reset_index(drop=True)
    )


def load_candles(
    market,
    candle_count=CANDLE_COUNT
):
    if market["provider"] == "Binance":
        return binance_candles(
            market["market_id"],
            candle_count
        )

    return gate_candles(
        market["market_id"],
        candle_count
    )


# ============================================================
# DEEP LEARNING FEATURES
# ============================================================

def make_model_features(df):
    """
    Ham mum ve hacim verilerinden model girdileri üretir.
    RSI, EMA, MACD veya teknik indikatör stratejileri kullanılmaz.
    """

    close = (
        df["close"]
        .astype(float)
        .replace(0, np.nan)
    )

    open_ = (
        df["open"]
        .astype(float)
        .replace(0, np.nan)
    )

    high = (
        df["high"]
        .astype(float)
        .replace(0, np.nan)
    )

    low = (
        df["low"]
        .astype(float)
        .replace(0, np.nan)
    )

    volume = (
        df["volume"]
        .astype(float)
        .clip(lower=0)
    )

    quote_volume = (
        df["quote_volume"]
        .astype(float)
        .clip(lower=0)
    )

    features = pd.DataFrame(
        index=df.index
    )

    features["close_return"] = np.log(
        close / close.shift(1)
    )

    features["open_to_close"] = np.log(
        close / open_
    )

    features["high_to_close"] = np.log(
        high / close
    )

    features["low_to_close"] = np.log(
        low / close
    )

    features["volume_change"] = (
        np.log1p(volume).diff()
    )

    features["quote_volume_change"] = (
        np.log1p(quote_volume).diff()
    )

    return features.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# LSTM MODEL
# ============================================================

if TORCH_AVAILABLE:

    class PriceLSTM(nn.Module):
        def __init__(
            self,
            input_size=6,
            hidden_size=32,
            layers=2,
            dropout=0.15
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=layers,
                batch_first=True,
                dropout=(
                    dropout
                    if layers > 1
                    else 0.0
                )
            )

            self.head = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


# ============================================================
# TRAINING AND PREDICTION
# ============================================================

def train_and_predict(
    df,
    lookback=48,
    epochs=5,
    train_fraction=0.8
):
    """
    LSTM modelini eğitir.
    Eğitim ve doğrulama kronolojik olarak ayrılır.
    Tahmin, sonraki 15 dakikalık mumun yönü içindir.
    """

    if not TORCH_AVAILABLE:
        return {
            "up_probability_pct": np.nan,
            "model_signal": "PyTorch not installed",
            "validation_accuracy_pct": np.nan,
            "validation_samples": 0,
            "training_samples": 0
        }

    if len(df) < lookback + 150:
        return {
            "up_probability_pct": np.nan,
            "model_signal": "Insufficient history",
            "validation_accuracy_pct": np.nan,
            "validation_samples": 0,
            "training_samples": 0
        }

    all_features = make_model_features(df)

    close = df["close"].to_numpy(
        dtype=np.float64
    )

    # Bir sonraki kapanış daha yüksek mi?
    labels = (
        close[1:] > close[:-1]
    ).astype(np.float32)

    # Yalnızca hedefi bilinen satırlar.
    features = all_features.iloc[:-1].copy()

    split_row = int(
        len(features) * train_fraction
    )

    # Ölçekleyici yalnızca eğitim verisine fit edilir.
    means = features.iloc[:split_row].mean()

    stds = (
        features.iloc[:split_row]
        .std()
        .replace(0, 1)
        .fillna(1)
    )

    scaled = (
        (features - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    scaled_all = (
        (all_features - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(
        dtype=np.float32
    )

    X = []
    y = []
    end_indices = []

    for end in range(
        lookback - 1,
        len(scaled)
    ):
        X.append(
            scaled[
                end - lookback + 1:
                end + 1
            ]
        )

        y.append(labels[end])
        end_indices.append(end)

    if len(X) < 100:
        return {
            "up_probability_pct": np.nan,
            "model_signal": "Insufficient sequences",
            "validation_accuracy_pct": np.nan,
            "validation_samples": 0,
            "training_samples": 0
        }

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    end_indices = np.asarray(
        end_indices
    )

    train_mask = end_indices < split_row
    val_mask = ~train_mask

    if (
        train_mask.sum() < 50
        or val_mask.sum() < 20
    ):
        return {
            "up_probability_pct": np.nan,
            "model_signal": "Insufficient train/validation data",
            "validation_accuracy_pct": np.nan,
            "validation_samples": int(
                val_mask.sum()
            ),
            "training_samples": int(
                train_mask.sum()
            )
        }

    torch.manual_seed(42)
    torch.set_num_threads(1)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = PriceLSTM(
        input_size=X.shape[2]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.BCEWithLogitsLoss()

    train_x = torch.from_numpy(
        X[train_mask]
    ).to(device)

    train_y = torch.from_numpy(
        y[train_mask, None]
    ).to(device)

    model.train()

    batch_size = min(
        128,
        len(train_x)
    )

    for _ in range(int(epochs)):
        order = torch.randperm(
            len(train_x),
            device=device
        )

        for start in range(
            0,
            len(order),
            batch_size
        ):
            idx = order[
                start:start + batch_size
            ]

            optimizer.zero_grad()

            logits = model(
                train_x[idx]
            )

            loss = loss_fn(
                logits,
                train_y[idx]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

    model.eval()

    with torch.no_grad():
        vx = torch.from_numpy(
            X[val_mask]
        ).to(device)

        vy = y[val_mask]

        val_probs = torch.sigmoid(
            model(vx)
        ).cpu().numpy().reshape(-1)

        val_pred = (
            val_probs >= 0.5
        ).astype(np.float32)

        val_accuracy = float(
            (val_pred == vy).mean() * 100
        )

        # Son tamamlanmış mumların özellikleri.
        latest = torch.from_numpy(
            scaled_all[-lookback:][None, :, :]
        ).to(device)

        probability = float(
            torch.sigmoid(
                model(latest)
            ).item() * 100
        )

    if probability >= 55:
        signal = "MODEL PREDICTS UP"
    elif probability <= 45:
        signal = "MODEL PREDICTS DOWN"
    else:
        signal = "MODEL UNCERTAIN"

    return {
        "up_probability_pct": probability,
        "model_signal": signal,
        "validation_accuracy_pct": val_accuracy,
        "validation_samples": int(
            val_mask.sum()
        ),
        "training_samples": int(
            train_mask.sum()
        )
    }


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyze_market(
    market,
    ticker,
    lookback,
    epochs
):
    candles = load_candles(
        market,
        CANDLE_COUNT
    )

    if len(candles) < lookback + 150:
        raise ValueError(
            f"Yetersiz mum verisi: "
            f"{len(candles)} / {CANDLE_COUNT}"
        )

    prediction = train_and_predict(
        candles,
        lookback=lookback,
        epochs=epochs
    )

    close = float(
        candles.iloc[-1]["close"]
    )

    peak = float(
        candles["high"].max()
    )

    low = float(
        candles["low"].min()
    )

    drop = (
        (peak - close) / peak * 100
        if peak > 0
        else np.nan
    )

    from_low = (
        (close / low - 1) * 100
        if low > 0
        else np.nan
    )

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "exchange": market["provider"],
        "current_price": float(ticker["price"]),
        "last_completed_15m_close": close,
        "change_24h_pct": float(
            ticker["change_24h_pct"]
        ),
        "quote_volume_24h_usdt": float(
            ticker["quote_volume"]
        ),
        "drop_from_last_1000_candle_high_pct": drop,
        "distance_from_last_1000_candle_low_pct": from_low,
        "candles_used": len(candles),
        "history_hours_used": len(candles) * 0.25,
        **prediction
    }


# ============================================================
# BUY / SELL / HOLD
# ============================================================

def signal_from_probability(probability):
    """LSTM olasılığına göre muhafazakâr sinyal üretir."""

    if (
        probability is None
        or not np.isfinite(probability)
    ):
        return (
            "HOLD",
            "Model olasılığı hesaplanamadı"
        )

    if probability >= 60:
        return (
            "BUY",
            "LSTM yükseliş olasılığı en az %60"
        )

    if probability <= 40:
        return (
            "SELL",
            "LSTM yükseliş olasılığı en fazla %40"
        )

    return (
        "HOLD",
        "Model sinyali yeterince güçlü değil"
    )


# ============================================================
# BINANCE SPOT TESTNET ORDERS
# ============================================================

def signed_binance_testnet_request(
    method,
    path,
    params
):
    """
    İmzalı istek yalnızca Binance Spot TESTNET'e gider.
    Üretim API adresi burada kullanılmaz.
    """

    api_key = os.getenv(
        "BINANCE_TESTNET_API_KEY",
        ""
    ).strip()

    api_secret = os.getenv(
        "BINANCE_TESTNET_API_SECRET",
        ""
    ).strip()

    if not api_key or not api_secret:
        raise RuntimeError(
            "Testnet API anahtarı yok. "
            "BINANCE_TESTNET_API_KEY ve "
            "BINANCE_TESTNET_API_SECRET "
            "ortam değişkenlerini ayarlayın."
        )

    payload = dict(params)

    payload["timestamp"] = int(
        time.time() * 1000
    )

    payload.setdefault(
        "recvWindow",
        5000
    )

    query = urlencode(payload)

    signature = hmac.new(
        api_secret.encode(),
        query.encode(),
        hashlib.sha256
    ).hexdigest()

    url = (
        f"{BINANCE_TESTNET}{path}"
        f"?{query}&signature={signature}"
    )

    response = session.request(
        method,
        url,
        headers={
            "X-MBX-APIKEY": api_key
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def place_testnet_order(
    symbol,
    side,
    usdt_amount,
    base_quantity=None
):
    """Binance Spot TESTNET piyasa emri."""

    if side not in ("BUY", "SELL"):
        raise ValueError(
            "Emir yönü BUY veya SELL olmalı."
        )

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "newOrderRespType": "FULL"
    }

    if side == "BUY":
        if usdt_amount <= 0:
            raise ValueError(
                "Alım tutarı USDT olarak "
                "sıfırdan büyük olmalı."
            )

        params["quoteOrderQty"] = (
            f"{usdt_amount:.8f}"
            .rstrip("0")
            .rstrip(".")
        )

    else:
        if (
            base_quantity is None
            or base_quantity <= 0
        ):
            raise ValueError(
                "Satış için satılacak "
                "coin miktarını girin."
            )

        params["quantity"] = (
            f"{base_quantity:.8f}"
            .rstrip("0")
            .rstrip(".")
        )

    return signed_binance_testnet_request(
        "POST",
        "/api/v3/order",
        params
    )


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title(
    "🧠 Deep Learning Spot Scanner"
)

st.write(
    "Yalnızca LSTM derin öğrenme modeli kullanılır. "
    "Her coin için son 1000 tamamlanmış 15 dakikalık "
    "mum kullanılır. Teknik indikatör, strateji "
    "oylaması veya indikatör puanlaması yoktur."
)

st.warning(
    "BUY / SELL / HOLD bir model sinyalidir, "
    "kâr garantisi değildir. Varsayılan işlem modu "
    "TESTNET'tir: yalnızca Binance Spot Testnet'e "
    "emir gönderilir, gerçek para kullanılmaz. "
    "Canlı Binance emirleri bu sürümde kapalıdır."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    exchange =
