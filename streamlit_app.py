import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# PyTorch is optional at import time so the dashboard can show a useful error
# instead of crashing if Streamlit Cloud has not installed it yet.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

BINANCE_URL = "https://api.binance.com"
INTERVAL = "15m"
TARGET_CANDLES = 500_000
LOOKAHEAD = 12
SEQ_LEN = 96

st.set_page_config(
    page_title="Binance Spot AI Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Binance Spot Deep Learning Scanner")
st.caption("15m • Rolling 500,000 candles • Top 10 BUY / SELL • No order execution")

if not TORCH_AVAILABLE:
    st.error(
        "PyTorch (torch) kurulu değil. Streamlit Cloud'da requirements.txt "
        "ve Python sürümünü kontrol edip uygulamayı yeniden deploy edin."
    )
    st.info(
        "Gerekli bağımlılıklar: streamlit, requests, pandas, numpy, torch. "
        "Bu uygulama emir göndermez; yalnızca analiz/öneri üretir."
    )

@st.cache_data(ttl=300)
def get_exchange_info():
    r = requests.get(f"{BINANCE_URL}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return r.json()

def get_spot_symbols():
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        permissions = s.get("permissions", [])
        if permissions and "SPOT" not in permissions:
            continue
        symbols.append(s["symbol"])
    return sorted(set(symbols))

def convert_klines(rows):
    if not rows:
        return pd.DataFrame()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def download_initial_500k(symbol, candle_count):
    rows = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while len(rows) < candle_count:
        limit = min(1000, candle_count - len(rows))
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit,
            "endTime": end_ms,
        }
        r = requests.get(f"{BINANCE_URL}/api/v3/klines", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        rows = batch + rows
        oldest = batch[0][0]
        end_ms = oldest - 1

        if len(batch) < limit:
            break
        time.sleep(0.04)

    df = convert_klines(rows)
    if len(df) > candle_count:
        df = df.tail(candle_count).reset_index(drop=True)
    return df

def download_new_candles(symbol, last_open_ms):
    rows = []
    start_ms = int(last_open_ms) + 1

    for _ in range(5):
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": 1000,
            "startTime": start_ms,
        }
        r = requests.get(f"{BINANCE_URL}/api/v3/klines", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
        time.sleep(0.04)

    return convert_klines(rows)

def update_symbol_cache(symbol, candle_count):
    cache = st.session_state.setdefault("candle_cache", {})

    if symbol not in cache or cache[symbol].empty:
        df = download_initial_500k(symbol, candle_count)
    else:
        old = cache[symbol]
        last_ms = int(old["open_time"].iloc[-1].timestamp() * 1000)
        new = download_new_candles(symbol, last_ms)
        if new.empty:
            df = old
        else:
            df = pd.concat([old, new], ignore_index=True)
            df = df.drop_duplicates("open_time").sort_values("open_time")
            df = df.tail(candle_count).reset_index(drop=True)

    cache[symbol] = df
    return df

def make_features(df):
    x = df.copy()
    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    for p in [5, 12, 20, 50, 100, 200, 500, 800]:
        x[f"ema{p}"] = close.ewm(span=p, adjust=False).mean() / close - 1

    delta = close.diff()
    for p in [7, 14, 21]:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / (loss + 1e-12)
        x[f"rsi{p}"] = 100 - (100 / (1 + rs))

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    x["bb_upper"] = (mid + 2 * std) / close - 1
    x["bb_lower"] = (mid - 2 * std) / close - 1

    x["vol20"] = close.pct_change().rolling(20).std()
    x["vol50"] = close.pct_change().rolling(50).std()
    x["volume_ratio"] = volume / (volume.rolling(50).mean() + 1e-12)

    x["candle_range"] = (high - low) / close
    x["candle_body"] = (x["close"] - x["open"]).abs() / close
    x["upper_wick"] = (high - x[["open", "close"]].max(axis=1)) / close
    x["lower_wick"] = (x[["open", "close"]].min(axis=1) - low) / close

    x["momentum12"] = close.pct_change(12)
    x["momentum48"] = close.pct_change(48)
    x["future_return"] = close.shift(-LOOKAHEAD) / close - 1

    return x.replace([np.inf, -np.inf], np.nan)

FEATURES = [
    "ema5", "ema12", "ema20", "ema50", "ema100", "ema200", "ema500", "ema800",
    "rsi7", "rsi14", "rsi21",
    "bb_upper", "bb_lower",
    "vol20", "vol50", "volume_ratio",
    "candle_range", "candle_body", "upper_wick", "lower_wick",
    "momentum12", "momentum48",
]

if TORCH_AVAILABLE:
    class TransformerAI(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            d = 96
            self.proj = nn.Linear(n_features, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=8,
                batch_first=True,
                dropout=0.10,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=3)
            self.cls = nn.Linear(d, 2)
            self.reg = nn.Linear(d, 1)

        def forward(self, x):
            z = self.proj(x)
            z = self.encoder(z)
            z = z[:, -1, :]
            return self.cls(z), self.reg(z)

def predict_ai(df):
    if not TORCH_AVAILABLE:
        return "N/A", 0.0, 0.0

    f = make_features(df).dropna(subset=FEATURES + ["future_return"]).copy()
    if len(f) < SEQ_LEN + 100:
        return "N/A", 0.0, 0.0

    vals = f[FEATURES].astype(np.float32).values
    mu = vals[:-LOOKAHEAD].mean(axis=0)
    sd = vals[:-LOOKAHEAD].std(axis=0) + 1e-6
    vals = (vals - mu) / sd

    yret = f["future_return"].values.astype(np.float32)
    max_samples = min(2500, len(f) - SEQ_LEN)
    starts = np.linspace(
        0, len(f) - SEQ_LEN - 1, max_samples, dtype=int
    )

    X = np.stack([vals[i:i+SEQ_LEN] for i in starts])
    y = (yret[starts + SEQ_LEN - 1] > 0).astype(np.int64)
    yr = yret[starts + SEQ_LEN - 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerAI(len(FEATURES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    model.train()
    batch_size = 64
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    yrt = torch.tensor(yr, dtype=torch.float32, device=device).unsqueeze(1)

    for _ in range(3):
        idx = torch.randperm(len(Xt), device=device)
        for start in range(0, len(Xt), batch_size):
            ii = idx[start:start+batch_size]
            logits, pred_r = model(Xt[ii])
            loss = ce(logits, yt[ii]) + 0.5 * mse(pred_r, yrt[ii])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    latest = torch.tensor(
        vals[-SEQ_LEN:], dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits, pred_r = model(latest)
        prob = torch.softmax(logits, dim=1)[0]
        buy_prob = float(prob[1].item())
        sell_prob = float(prob[0].item())
        expected = float(pred_r.item())

    if buy_prob >= sell_prob:
        return "BUY", buy_prob * 100, expected * 100
    return "SELL", sell_prob * 100, expected * 100

def calculate_score(df, signal, confidence, expected):
    f = make_features(df)
    last = f.iloc[-1]

    momentum = float(np.nan_to_num(last.get("momentum48", 0.0)))
    trend = float(np.nan_to_num(last.get("ema200", 0.0) - last.get("ema800", 0.0)))
    volatility = float(np.nan_to_num(last.get("vol50", 0.0)))
    quote_volume = float(np.nan_to_num(df["quote_volume"].tail(96).mean()))

    liquidity = min(np.log10(max(quote_volume, 1.0)) / 10.0, 1.0)
    risk_penalty = min(volatility * 10.0, 1.0)

    direction = 1 if signal == "BUY" else -1
    score = (
        expected * 0.45
        + (confidence / 100) * 0.25
        + direction * momentum * 0.12
        + direction * trend * 0.10
        + liquidity * 0.08
        - risk_penalty * 0.20
    )
    return float(score)

def scan_pair(symbol, candle_count):
    df = update_symbol_cache(symbol, candle_count)
    if df.empty or len(df) < SEQ_LEN + LOOKAHEAD + 100:
        return None

    avg_quote = float(df["quote_volume"].tail(96).mean())
    if avg_quote < 1_000_000:
        return None

    signal, confidence, expected = predict_ai(df)
    if signal not in ("BUY", "SELL"):
        return None

    score = calculate_score(df, signal, confidence, expected)
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(confidence, 2),
        "expected_return_%": round(expected, 4),
        "score": round(score, 4),
        "candles": len(df),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

def run_scan(symbols, candle_count):
    results = []
    for symbol in symbols:
        try:
            result = scan_pair(symbol, candle_count)
            if result:
                results.append(result)
        except Exception as e:
            st.session_state.setdefault("scan_errors", {})[symbol] = str(e)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    candle_count = st.number_input(
        "Mum sayısı",
        min_value=10_000,
        max_value=500_000,
        value=500_000,
        step=10_000,
    )
    pair_limit = st.number_input(
        "Taranacak maksimum parite",
        min_value=1,
        max_value=100,
        value=15,
        step=1,
    )
    auto_refresh = st.checkbox("15 dakikada otomatik yenile", value=True)
    st.caption("İlk taramada her parite için geçmiş veri indirilir.")

priority = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT"
]

@st.fragment(run_every="15m" if auto_refresh else None)
def scanner_fragment():
    c1, c2 = st.columns([1, 5])
    with c1:
        manual = st.button("🔄 Şimdi Tara", use_container_width=True)
    with c2:
        st.caption(
            "Tarama 15 dakikada bir yenilenir. "
            "Uygulama açık bir Streamlit oturumu sırasında çalışır."
        )

    if manual or "results" not in st.session_state:
        try:
            all_symbols = get_spot_symbols()
            ordered = [s for s in priority if s in all_symbols]
            ordered += [s for s in all_symbols if s not in ordered]
            symbols = ordered[:int(pair_limit)]

            with st.spinner(f"{len(symbols)} parite taranıyor..."):
                results = run_scan(symbols, int(candle_count))

            st.session_state["results"] = results
            st.session_state["last_scan"] = datetime.now(timezone.utc)
        except Exception as e:
            st.error(f"Tarama hatası: {e}")
            return

    results = st.session_state.get("results", pd.DataFrame())
    if results.empty:
        st.warning("Henüz uygun BUY/SELL sonucu oluşmadı.")
        return

    buys = results[results["signal"] == "BUY"].sort_values(
        ["score", "confidence"], ascending=False
    ).head(10)
    sells = results[results["signal"] == "SELL"].sort_values(
        ["score", "confidence"], ascending=False
    ).head(10)

    a, b = st.columns(2)
    with a:
        st.subheader("🟢 TOP 10 BUY")
        st.dataframe(buys, use_container_width=True, hide_index=True)
    with b:
        st.subheader("🔴 TOP 10 SELL")
        st.dataframe(sells, use_container_width=True, hide_index=True)

    st.subheader("Tüm sonuçlar")
    st.dataframe(results, use_container_width=True, hide_index=True)

    last_scan = st.session_state.get("last_scan")
    if last_scan:
        st.caption(f"Son tarama: {last_scan.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    errors = st.session_state.get("scan_errors", {})
    if errors:
        with st.expander("Parite hataları"):
            st.json(errors)

scanner_fragment()
