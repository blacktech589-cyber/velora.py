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

st.set_page_config(page_title="Deep Learning Spot Scanner", page_icon="🧠", layout="wide")

BINANCE_TESTNET = "https://testnet.binance.vision"
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



def signal_from_probability(probability):
    """Convert the LSTM probability into a conservative directional signal."""
    if probability is None or not np.isfinite(probability):
        return "HOLD", "Model olasılığı hesaplanamadı"
    if probability >= 60:
        return "BUY", "LSTM yükseliş olasılığı en az %60"
    if probability <= 40:
        return "SELL", "LSTM yükseliş olasılığı en fazla %40"
    return "HOLD", "Model sinyali yeterince güçlü değil"


def signed_binance_testnet_request(method, path, params):
    """Send a signed request to Binance Spot TESTNET only; never production."""
    api_key = os.getenv("BINANCE_TESTNET_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Testnet API anahtarı yok. BINANCE_TESTNET_API_KEY ve BINANCE_TESTNET_API_SECRET ortam değişkenlerini ayarlayın.")
    payload = dict(params)
    payload["timestamp"] = int(time.time() * 1000)
    payload.setdefault("recvWindow", 5000)
    query = urlencode(payload)
    signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BINANCE_TESTNET}{path}?{query}&signature={signature}"
    response = session.request(method, url, headers={"X-MBX-APIKEY": api_key}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def place_testnet_order(symbol, side, usdt_amount, base_quantity=None):
    """Market order on Binance Spot TESTNET, not on the live exchange."""
    if side not in ("BUY", "SELL"):
        raise ValueError("Emir yönü BUY veya SELL olmalı.")
    params = {"symbol": symbol, "side": side, "type": "MARKET", "newOrderRespType": "FULL"}
    if side == "BUY":
        if usdt_amount <= 0:
            raise ValueError("Alım tutarı USDT olarak sıfırdan büyük olmalı.")
        params["quoteOrderQty"] = f"{usdt_amount:.8f}".rstrip("0").rstrip(".")
    else:
        if base_quantity is None or base_quantity <= 0:
            raise ValueError("Satış için satılacak coin miktarını girin.")
        params["quantity"] = f"{base_quantity:.8f}".rstrip("0").rstrip(".")
    return signed_binance_testnet_request("POST", "/api/v3/order", params)

st.title("🧠 Deep Learning Spot Scanner")
st.write("Yalnızca LSTM derin öğrenme modeli kullanılır. Her coin için son 1000 tamamlanmış 15 dakikalık mum kullanılır. Teknik indikatör, strateji oylaması veya indikatör puanlaması yoktur.")
st.warning("BUY / SELL / HOLD bir model sinyalidir, kâr garantisi değildir. Varsayılan işlem modu TESTNET'tir: yalnızca Binance Spot Testnet'e emir gönderilir, gerçek para kullanılmaz. Canlı Binance emirleri bu sürümde kapalıdır.")

with st.sidebar:
    exchange = "Binance only"
    st.caption("Borsa: Binance Spot")
    st.caption("Model her coin için son 1000 tamamlanmış 15 dakikalık mumu analiz eder (yaklaşık 10,4 gün).")
    scan_count = st.number_input("Taranacak piyasa sayısı (hacme göre)", min_value=1, max_value=500, value=30, step=10)
    lookback = st.selectbox("Modelin göreceği geçmiş mum", [24, 48, 72, 96], index=1)
    epochs = st.slider("Eğitim turu (epoch)", 1, 15, 5)
    st.subheader("Emir test ayarları")
    usdt_amount = st.number_input("Manuel BUY tutarı (USDT)", min_value=5.0, max_value=100000.0, value=10.0, step=5.0)
    sell_quantity = st.number_input("Manuel SELL miktarı (coin)", min_value=0.0, value=0.0, step=0.001, format="%.8f")
    enable_testnet = st.checkbox("Binance Spot TESTNET emirlerini etkinleştir", value=False)
    live_disabled = st.checkbox("Canlı Binance emirlerini etkinleştir (bu sürümde kullanılamaz)", value=False, disabled=True)
    st.caption("Testnet emirleri için BINANCE_TESTNET_API_KEY ve BINANCE_TESTNET_API_SECRET ortam değişkenlerini ayarlayın. Canlı işlem kodu bu sürümde özellikle kapalıdır.")
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

    # Compatibility guard: Streamlit may retain results from an older app version
    # in session_state. Normalize older model-probability column names before sorting.
    probability_aliases = (
        "dl_up_probability_pct",
        "prediction_probability_pct",
        "up_probability",
    )
    if "up_probability_pct" not in results.columns:
        alias = next((name for name in probability_aliases if name in results.columns), None)
        if alias is not None:
            results = results.rename(columns={alias: "up_probability_pct"})
        else:
            # Do not crash if a scan returned rows without the model output field.
            results["up_probability_pct"] = np.nan
            st.warning(
                "Sonuçlarda 'up_probability_pct' alanı bulunamadı. "
                "Bu tarama için yükseliş olasılığı mevcut değil; modeli yeniden çalıştırın."
            )

    results["up_probability_pct"] = pd.to_numeric(
        results["up_probability_pct"], errors="coerce"
    )
    signal_pairs = results["up_probability_pct"].apply(lambda p: signal_from_probability(p))
    results["signal"] = signal_pairs.apply(lambda pair: pair[0])
    results["signal_reason"] = signal_pairs.apply(lambda pair: pair[1])
    results["down_probability_pct"] = 100 - results["up_probability_pct"]
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

    st.subheader("Binance Spot Testnet — manuel emir")
    symbols = ranked["symbol"].dropna().astype(str).tolist()
    if symbols:
        selected_symbol = st.selectbox("Emir gönderilecek coin", symbols, key="order_symbol")
        selected_row = ranked.loc[ranked["symbol"].astype(str) == selected_symbol].iloc[0]
        st.write(f"Model sinyali: **{selected_row['signal']}** · Yükseliş olasılığı: **{selected_row['up_probability_pct']:.2f}%** · Düşüş olasılığı: **{selected_row['down_probability_pct']:.2f}%**" if pd.notna(selected_row["up_probability_pct"]) else "Model olasılığı yok; emir vermeden önce taramayı tekrar çalıştırın.")
        buy_col, sell_col = st.columns(2)
        with buy_col:
            if st.button(f"TESTNET BUY — {usdt_amount:.2f} USDT", type="primary", disabled=not enable_testnet, use_container_width=True):
                try:
                    order = place_testnet_order(selected_symbol, "BUY", float(usdt_amount))
                    st.success(f"Testnet BUY yanıtı alındı. Emir ID: {order.get('orderId', 'bilinmiyor')} · Durum: {order.get('status', 'bilinmiyor')}")
                    st.json(order)
                except Exception as exc:
                    st.error(f"Testnet BUY başarısız: {exc}")
        with sell_col:
            if st.button(f"TESTNET SELL — {sell_quantity:.8f} coin", disabled=(not enable_testnet or sell_quantity <= 0), use_container_width=True):
                try:
                    order = place_testnet_order(selected_symbol, "SELL", float(usdt_amount), float(sell_quantity))
                    st.success(f"Testnet SELL yanıtı alındı. Emir ID: {order.get('orderId', 'bilinmiyor')} · Durum: {order.get('status', 'bilinmiyor')}")
                    st.json(order)
                except Exception as exc:
                    st.error(f"Testnet SELL başarısız: {exc}")
        st.caption("Emir düğmeleri yalnızca tarama sonucu geldikten sonra kullanılabilir. Testnet API anahtarları üretim API anahtarlarıyla aynı değildir. BUY tutarı USDT; SELL miktarı coin birimindedir.")

    st.caption("Doğrulama başarısı, kronolojik olarak eğitimden sonra bırakılan veride ölçülür. Bu değer gerçek zamanlı gelecekteki başarıyı garanti etmez; ücretler, kayma ve piyasa rejimi ayrıca test edilmelidir.")
    errors = st.session_state.get("errors")
    if errors is not None and not errors.empty:
        with st.expander(f"Tarama hataları ({len(errors)})"):
            st.dataframe(safe_dataframe(errors), use_container_width=True, hide_index=True)
            st.download_button("Hataları CSV indir", errors.to_csv(index=False).encode("utf-8-sig"), "scan_errors.csv", "text/csv")
