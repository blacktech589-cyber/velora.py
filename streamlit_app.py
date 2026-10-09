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
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

st.set_page_config(page_title="Deep Learning Spot Scanner", page_icon="🧠", layout="wide")

BINANCE_HOSTS = [
    "https://api.binance.com", "https://api-gcp.binance.com",
    "https://api1.binance.com", "https://api2.binance.com",
    "https://api3.binance.com", "https://api4.binance.com",
]
GATE = "https://api.gateio.ws/api/v4"
TIMEOUT = 20
CANDLE_COUNT = 1000  # Son 1000 tamamlanmış 15 dakikalık mum
session = requests.Session()
session.headers.update({"User-Agent": "DeepLearningSpotScanner/1.0"})


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def binance_json(path, params=None):
    errors = []
    for host in BINANCE_HOSTS:
        try:
            response = session.get(host + path, params=params, timeout=TIMEOUT)
            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue
            response.raise_for_status()
            return response.json(), host
        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:100]}")
    raise RuntimeError("Binance API erişimi başarısız: " + " | ".join(errors))


def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    seen, names = {}, []
    for col in out.columns:
        name = str(col)
        n = seen.get(name, 0)
        seen[name] = n + 1
        names.append(name if n == 0 else f"{name}_{n}")
    out.columns = names
    return out


def binance_markets():
    data, host = binance_json("/api/v3/exchangeInfo")
    markets = []
    for item in data.get("symbols", []):
        if item.get("status") != "TRADING" or item.get("quoteAsset") != "USDT":
            continue
        if item.get("isSpotTradingAllowed") is False:
            continue
        markets.append({"symbol": item["symbol"], "market_id": item["symbol"], "provider": "Binance"})
    if not markets:
        raise RuntimeError("Aktif Binance Spot USDT piyasası bulunamadı.")
    return markets, host


def binance_tickers():
    data, host = binance_json("/api/v3/ticker/24hr")
    out = {}
    for item in data:
        try:
            out[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h_pct": float(item["priceChangePercent"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return out, host


def gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")
    out = []
    for item in data:
        if item.get("quote") != "USDT" or item.get("trade_status") != "tradable" or item.get("delisted") is True:
            continue
        base, market_id = item.get("base", ""), item.get("id", "")
        if base and market_id:
            out.append({"symbol": base + "USDT", "market_id": market_id, "provider": "Gate.io"})
    if not out:
        raise RuntimeError("Aktif Gate.io Spot USDT piyasası bulunamadı.")
    return out


def gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")
    out = {}
    for item in data:
        market_id = item.get("currency_pair", "")
        if not market_id.endswith("_USDT"):
            continue
        try:
            out[market_id] = {
                "price": float(item["last"]),
                "quote_volume": float(item["quote_volume"]),
                "change_24h_pct": float(item["change_percentage"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return out


def discover(exchange):
    if exchange == "Gate.io only":
        return gate_markets(), gate_tickers(), "Gate.io"
    try:
        markets, _ = binance_markets()
        tickers, _ = binance_tickers()
        return markets, tickers, "Binance"
    except Exception:
        if exchange == "Binance only":
            raise
        return gate_markets(), gate_tickers(), "Gate.io (fallback)"


def clean_candles(rows):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=cols[:6]).drop_duplicates("timestamp").sort_values("timestamp")
    now = int(datetime.now(timezone.utc).timestamp())
    current_open = now // 900 * 900
    df = df[df["timestamp"] < current_open]
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["high"] >= df["low"])]
    return df.reset_index(drop=True)


def binance_candles(symbol, candle_count=CANDLE_COUNT):
    """Fetch up to the most recent 1000 completed 15-minute candles."""
    now = int(datetime.now(timezone.utc).timestamp())
    end_ms = (now // 900 * 900) * 1000
    start_ms = end_ms - (int(candle_count) + 5) * 900 * 1000
    data, _ = binance_json("/api/v3/klines", {
        "symbol": symbol, "interval": "15m", "startTime": start_ms,
        "endTime": end_ms, "limit": min(1000, int(candle_count) + 5),
    })
    rows = []
    for c in data:
        try:
            rows.append({"timestamp": int(c[0]) // 1000, "open": float(c[1]),
                         "high": float(c[2]), "low": float(c[3]),
                         "close": float(c[4]), "volume": float(c[5]),
                         "quote_volume": float(c[7])})
        except (ValueError, TypeError, IndexError):
            continue
    return clean_candles(rows).tail(int(candle_count)).reset_index(drop=True)


def gate_candles(market_id, candle_count=CANDLE_COUNT):
    """Fetch up to the most recent 1000 completed 15-minute candles."""
    now = int(datetime.now(timezone.utc).timestamp())
    end = now // 900 * 900
    start = end - (int(candle_count) + 5) * 900
    data = get_json(f"{GATE}/spot/candlesticks", {
        "currency_pair": market_id, "interval": "15m", "from": start,
        "to": end, "limit": min(1000, int(candle_count) + 5),
    })
    rows = []
    for c in data:
        try:
            rows.append({"timestamp": int(c[0]), "quote_volume": float(c[1]),
                         "close": float(c[2]), "high": float(c[3]),
                         "low": float(c[4]), "open": float(c[5]),
                         "volume": float(c[6]) if len(c) > 6 else 0.0})
        except (ValueError, TypeError, IndexError):
            continue
    return clean_candles(rows).tail(int(candle_count)).reset_index(drop=True)


def load_candles(market, candle_count=CANDLE_COUNT):
    if market["provider"] == "Binance":
        return binance_candles(market["market_id"], candle_count)
    return gate_candles(market["market_id"], candle_count)


# Only raw candle-derived features are used; no technical-indicator strategies.

def make_model_features(df):
    close = df["close"].astype(float).replace(0, np.nan)
    open_ = df["open"].astype(float).replace(0, np.nan)
    high = df["high"].astype(float).replace(0, np.nan)
    low = df["low"].astype(float).replace(0, np.nan)
    volume = df["volume"].astype(float).clip(lower=0)
    quote_volume = df["quote_volume"].astype(float).clip(lower=0)
    features = pd.DataFrame(index=df.index)
    features["close_return"] = np.log(close / close.shift(1))
    features["open_to_close"] = np.log(close / open_)
    features["high_to_close"] = np.log(high / close)
    features["low_to_close"] = np.log(low / close)
    features["volume_change"] = np.log1p(volume).diff()
    features["quote_volume_change"] = np.log1p(quote_volume).diff()
    return features.replace([np.inf, -np.inf], np.nan)


if TORCH_AVAILABLE:
    class PriceLSTM(nn.Module):
        def __init__(self, input_size=6, hidden_size=32, layers=2, dropout=0.15):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


def train_and_predict(df, lookback=48, epochs=5, train_fraction=0.8):
    """Chronological holdout; predicts next completed 15-minute candle direction."""
    if not TORCH_AVAILABLE:
        return {"up_probability_pct": np.nan, "model_signal": "PyTorch not installed",
                "validation_accuracy_pct": np.nan, "validation_samples": 0, "training_samples": 0}
    if len(df) < lookback + 150:
        return {"up_probability_pct": np.nan, "model_signal": "Insufficient history",
                "validation_accuracy_pct": np.nan, "validation_samples": 0, "training_samples": 0}

    all_features = make_model_features(df)
    # Target at row t is whether close[t+1] exceeds close[t].
    close = df["close"].to_numpy(dtype=np.float64)
    labels = (close[1:] > close[:-1]).astype(np.float32)
    # Training examples only have targets where the next close is already known.
    features = all_features.iloc[:-1].copy()

    # Only fit scaling parameters on the earlier training period.
    split_row = int(len(features) * train_fraction)
    means = features.iloc[:split_row].mean()
    stds = features.iloc[:split_row].std().replace(0, 1).fillna(1)
    scaled = ((features - means) / stds).clip(-8, 8).fillna(0).to_numpy(dtype=np.float32)
    scaled_all = ((all_features - means) / stds).clip(-8, 8).fillna(0).to_numpy(dtype=np.float32)

    X, y, end_indices = [], [], []
    for end in range(lookback - 1, len(scaled)):
        X.append(scaled[end - lookback + 1:end + 1])
        y.append(labels[end])
        end_indices.append(end)
    if len(X) < 100:
        return {"up_probability_pct": np.nan, "model_signal": "Insufficient sequences",
                "validation_accuracy_pct": np.nan, "validation_samples": 0, "training_samples": 0}

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    end_indices = np.asarray(end_indices)
    train_mask = end_indices < split_row
    val_mask = ~train_mask
    if train_mask.sum() < 50 or val_mask.sum() < 20:
        return {"up_probability_pct": np.nan, "model_signal": "Insufficient train/validation data",
                "validation_accuracy_pct": np.nan, "validation_samples": int(val_mask.sum()),
                "training_samples": int(train_mask.sum())}

    torch.manual_seed(42)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriceLSTM(input_size=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    train_x = torch.from_numpy(X[train_mask]).to(device)
    train_y = torch.from_numpy(y[train_mask, None]).to(device)

    model.train()
    batch_size = min(128, len(train_x))
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(train_x[idx])
            loss = loss_fn(logits, train_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        vx = torch.from_numpy(X[val_mask]).to(device)
        vy = y[val_mask]
        val_probs = torch.sigmoid(model(vx)).cpu().numpy().reshape(-1)
        val_pred = (val_probs >= 0.5).astype(np.float32)
        val_accuracy = float((val_pred == vy).mean() * 100)
        # Most recent window consists only of completed candle features.
        latest = torch.from_numpy(scaled_all[-lookback:][None, :, :]).to(device)
        probability = float(torch.sigmoid(model(latest)).item() * 100)

    if probability >= 55:
        signal = "MODEL PREDICTS UP"
    elif probability <= 45:
        signal = "MODEL PREDICTS DOWN"
    else:
        signal = "MODEL UNCERTAIN"
    return {"up_probability_pct": probability, "model_signal": signal,
            "validation_accuracy_pct": val_accuracy, "validation_samples": int(val_mask.sum()),
            "training_samples": int(train_mask.sum())}


def analyze_market(market, ticker, lookback, epochs):
    candles = load_candles(market, CANDLE_COUNT)
    if len(candles) < lookback + 150:
        raise ValueError(f"Yetersiz mum verisi: {len(candles)} / {CANDLE_COUNT}")
    prediction = train_and_predict(candles, lookback=lookback, epochs=epochs)
    close = float(candles.iloc[-1]["close"])
    peak = float(candles["high"].max())
    low = float(candles["low"].min())
    drop = (peak - close) / peak * 100 if peak > 0 else np.nan
    from_low = (close / low - 1) * 100 if low > 0 else np.nan
    return {
        "symbol": market["symbol"], "market_id": market["market_id"],
        "exchange": market["provider"], "current_price": float(ticker["price"]),
        "last_completed_15m_close": close,
        "change_24h_pct": float(ticker["change_24h_pct"]),
        "quote_volume_24h_usdt": float(ticker["quote_volume"]),
        "drop_from_last_1000_candle_high_pct": drop,
        "distance_from_last_1000_candle_low_pct": from_low,
        "candles_used": len(candles),
        "history_hours_used": len(candles) * 0.25,
        **prediction,
    }


st.title("🧠 Deep Learning Spot Scanner")
st.write("Yalnızca LSTM derin öğrenme modeli kullanılır. Her coin için son 1000 tamamlanmış 15 dakikalık mum kullanılır. Teknik indikatör, strateji oylaması veya indikatör puanlaması yoktur.")
st.warning("Model tahminleri olasılıksal değerlendirmedir; kâr garantisi değildir. Bu uygulama otomatik alım-satım emri göndermez.")

with st.sidebar:
    exchange = st.selectbox("Borsa", ["Auto (Binance then Gate.io)", "Binance only", "Gate.io only"])
    st.caption("Model her coin için son 1000 tamamlanmış 15 dakikalık mumu analiz eder (yaklaşık 10,4 gün).")
    scan_count = st.number_input("Taranacak piyasa sayısı (hacme göre)", min_value=1, max_value=500, value=30, step=10)
    lookback = st.selectbox("Modelin göreceği geçmiş mum", [24, 48, 72, 96], index=1)
    epochs = st.slider("Eğitim turu (epoch)", 1, 15, 5)
    if not TORCH_AVAILABLE:
        st.error("PyTorch bulunamadı. requirements.txt içindeki paketleri yükleyin.")
    run = st.button("Derin öğrenme taramasını başlat", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "errors" not in st.session_state:
    st.session_state.errors = pd.DataFrame()

if run:
    if not TORCH_AVAILABLE:
        st.error("Önce PyTorch kurun: python -m pip install torch")
    else:
        progress = st.progress(0)
        status = st.empty()
        try:
            markets, tickers, provider = discover(exchange)
            candidates = []
            for market in markets:
                ticker = tickers.get(market["market_id"])
                if ticker:
                    candidates.append({**market, **ticker})
            candidates.sort(key=lambda item: item["quote_volume"], reverse=True)
            candidates = candidates[:int(scan_count)]
            if not candidates:
                raise RuntimeError("Taranacak aktif Spot USDT piyasası bulunamadı.")
            st.info(f"Kaynak: {provider} | Analiz edilecek piyasa: {len(candidates)}")
            results, errors = [], []
            for i, market in enumerate(candidates):
                status.write(f"{i + 1}/{len(candidates)} — {market['symbol']}: LSTM eğitiliyor ve tahmin üretiliyor")
                try:
                    results.append(analyze_market(market, tickers[market["market_id"]], int(lookback), int(epochs)))
                except Exception as exc:
                    errors.append({"symbol": market["symbol"], "error": str(exc)[:250]})
                progress.progress((i + 1) / len(candidates))
                time.sleep(0.03)
            st.session_state.results = pd.DataFrame(results)
            st.session_state.errors = pd.DataFrame(errors)
            st.session_state.provider = provider
            st.session_state.scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            status.success(f"Tarama tamamlandı: {len(results)} sonuç, {len(errors)} hata.")
        except Exception as exc:
            st.error(f"Tarama başarısız: {exc}")

results = st.session_state.get("results")
if results is None:
    st.info("Sol panelden ayarları seçip taramayı başlatın.")
elif results.empty:
    st.warning("Başarılı sonuç yok. Hata listesini kontrol edin veya daha fazla piyasa tarayın.")
else:
    st.caption(f"Borsa: {st.session_state.get('provider', '-')} | Tarama zamanı: {st.session_state.get('scan_time', '-')}")
    ranked = results.sort_values("up_probability_pct", ascending=False, na_position="last").reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Analiz edilen coin", len(ranked))
    c2.metric("Model tahmini yukarı ≥ %55", int((ranked["up_probability_pct"] >= 55).sum()))
    c3.metric("Model tahmini aşağı ≤ %45", int((ranked["up_probability_pct"] <= 45).sum()))

    st.subheader("Model tahminine göre sıralama")
    st.dataframe(safe_dataframe(ranked), use_container_width=True, hide_index=True)
    st.download_button("Tüm tahminleri CSV indir", safe_dataframe(ranked).to_csv(index=False).encode("utf-8-sig"), "deep_learning_predictions.csv", "text/csv")

    st.subheader("Yükseliş olasılığı en yüksek coinler")
    st.dataframe(safe_dataframe(ranked.head(20)), use_container_width=True, hide_index=True)

    st.caption("Doğrulama başarısı, kronolojik olarak eğitimden sonra bırakılan veride ölçülür. Bu değer gerçek zamanlı gelecekteki başarıyı garanti etmez; ücretler, kayma ve piyasa rejimi ayrıca test edilmelidir.")
    errors = st.session_state.get("errors")
    if errors is not None and not errors.empty:
        with st.expander(f"Tarama hataları ({len(errors)})"):
            st.dataframe(safe_dataframe(errors), use_container_width=True, hide_index=True)
            st.download_button("Hataları CSV indir", errors.to_csv(index=False).encode("utf-8-sig"), "scan_errors.csv", "text/csv")
