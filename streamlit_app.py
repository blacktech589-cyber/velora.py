import os
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VELORA 1M HISTORICAL SCANNER
# - Targets 100,000 completed one-minute candles per asset
# - BUY / SELL signals only (NO SIGNAL is used internally)
# - No trade execution
# - Does not depend on Binance
# - Does not claim proxy data is Binomo OTC data
# ============================================================

st.set_page_config(page_title="Velora 1M Scanner", page_icon="📊", layout="wide")

TARGET_CANDLES = 100_000
INTERVAL_MINUTES = 1
DB_PATH = "velora_history.sqlite3"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HTTP_TIMEOUT = 30

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
})


# Asset format: display name | category | provider symbol | OTC label
RAW_ASSETS = """
Crypto IDX|index||0
Bitcoin (OTC)|crypto|BTC/USD|1
Ethereum (OTC)|crypto|ETH/USD|1
Solana (OTC)|crypto|SOL/USD|1
FC Barcelona Token (OTC)|crypto|BAR/USD|1
Cardano (OTC)|crypto|ADA/USD|1
Chainlink (OTC)|crypto|LINK/USD|1
Bitcoin Cash (OTC)|crypto|BCH/USD|1
Kusama (OTC)|crypto|KSM/USD|1
Aave (OTC)|crypto|AAVE/USD|1
Pancake Swap (OTC)|crypto|CAKE/USD|1
Uniswap (OTC)|crypto|UNI/USD|1
EUR/USD|forex|EUR/USD|0
EUR/HUF|forex|EUR/HUF|0
GBP/NOK|forex|GBP/NOK|0
CAD/JPY|forex|CAD/JPY|0
AUD/ZAR|forex|AUD/ZAR|0
CAD/DKK|forex|CAD/DKK|0
EUR/ILS|forex|EUR/ILS|0
GBP/SGD|forex|GBP/SGD|0
USD/MXN|forex|USD/MXN|0
EUR/ZAR|forex|EUR/ZAR|0
AUD/USD|forex|AUD/USD|0
GBP/HKD|forex|GBP/HKD|0
CAD/MXN|forex|CAD/MXN|0
EUR/NOK|forex|EUR/NOK|0
EUR/MXN|forex|EUR/MXN|0
CHF/JPY|forex|CHF/JPY|0
USD/ILS|forex|USD/ILS|0
USD/SEK|forex|USD/SEK|0
CAD/NOK|forex|CAD/NOK|0
CAD/SEK|forex|CAD/SEK|0
USD/PLN|forex|USD/PLN|0
AUD/JPY|forex|AUD/JPY|0
EUR/SEK|forex|EUR/SEK|0
AUD/SGD|forex|AUD/SGD|0
GBP/MXN|forex|GBP/MXN|0
CAD/SGD|forex|CAD/SGD|0
AUD/DKK|forex|AUD/DKK|0
EUR/SGD|forex|EUR/SGD|0
CHF/DKK|forex|CHF/DKK|0
GBP/PLN|forex|GBP/PLN|0
GBP/CZK|forex|GBP/CZK|0
EUR/NZD|forex|EUR/NZD|0
AUD/HUF|forex|AUD/HUF|0
USD/HUF|forex|USD/HUF|0
GBP/DKK|forex|GBP/DKK|0
AUD/NOK|forex|AUD/NOK|0
NZD/JPY|forex|NZD/JPY|0
CHF/NOK|forex|CHF/NOK|0
USD/JPY|forex|USD/JPY|0
CHF/PLN|forex|CHF/PLN|0
GBP/TRY|forex|GBP/TRY|0
USD/CAD|forex|USD/CAD|0
AUD/SEK|forex|AUD/SEK|0
CHF/SEK|forex|CHF/SEK|0
CHF/SGD|forex|CHF/SGD|0
NOK/JPY|forex|NOK/JPY|0
NOK/SEK|forex|NOK/SEK|0
NZD/CAD|forex|NZD/CAD|0
NZD/DKK|forex|NZD/DKK|0
EUR/GBP (OTC)|forex|EUR/GBP|1
GBP/NZD|forex|GBP/NZD|0
NZD/USD|forex|NZD/USD|0
NZD/HUF|forex|NZD/HUF|0
NZD/NOK|forex|NZD/NOK|0
NZD/SEK|forex|NZD/SEK|0
NZD/SGD|forex|NZD/SGD|0
SEK/JPY|forex|SEK/JPY|0
SGD/JPY|forex|SGD/JPY|0
SGD/HKD|forex|SGD/HKD|0
USD/CHF|forex|USD/CHF|0
AUD/CAD|forex|AUD/CAD|0
USD/CZK|forex|USD/CZK|0
NZD/CHF|forex|NZD/CHF|0
GBP/CAD|forex|GBP/CAD|0
AUD/NZD|forex|AUD/NZD|0
USD/ZAR|forex|USD/ZAR|0
AUD/CHF|forex|AUD/CHF|0
GBP/USD|forex|GBP/USD|0
GBP/HUF|forex|GBP/HUF|0
EUR/PLN|forex|EUR/PLN|0
ZAR/JPY|forex|ZAR/JPY|0
CHF/HUF|forex|CHF/HUF|0
USD/DKK|forex|USD/DKK|0
GBP/CHF|forex|GBP/CHF|0
Gold|metal|XAU/USD|0
Yum Brands|stock|YUM|0
""".strip()

ASSETS = []
for line in RAW_ASSETS.splitlines():
    name, category, symbol, otc = line.split("|")
    ASSETS.append({
        "name": name,
        "category": category,
        "symbol": symbol or None,
        "otc": otc == "1",
    })


def get_yahoo_symbol(asset):
    """Convert the app's display symbol to a Yahoo Finance chart ticker."""
    symbol = asset.get("symbol")
    category = asset.get("category")

    if not symbol:
        return None

    if category == "crypto":
        return symbol.replace("/", "-")

    if category == "forex":
        # Yahoo FX ticker format, for example EURUSD=X
        return symbol.replace("/", "") + "=X"

    if category == "metal":
        # Gold futures is a proxy, not a spot quote and not Binomo OTC.
        return "GC=F" if symbol == "XAU/USD" else symbol

    if category == "stock":
        return symbol

    return None


def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                ts TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0,
                PRIMARY KEY(symbol, ts)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts
            ON candles(symbol, ts)
        """)
        conn.commit()


def db_read(symbol, start_iso, end_iso):
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            """
            SELECT ts AS time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            conn,
            params=(symbol, start_iso, end_iso),
        )
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def db_read_all(symbol):
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        df = pd.read_sql_query(
            """
            SELECT ts AS time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY ts ASC
            """,
            conn,
            params=(symbol,),
        )
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def db_write(symbol, df):
    if df is None or df.empty:
        return
    records = []
    for row in df.itertuples(index=False):
        records.append((
            symbol,
            pd.Timestamp(row.time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
            float(row.open), float(row.high), float(row.low),
            float(row.close), float(getattr(row, "volume", 0) or 0),
        ))
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO candles
            (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()


def yahoo_fetch_recent_1m(yahoo_symbol):
    """Fetch recent one-minute OHLCV bars from Yahoo's unauthenticated chart endpoint.

    This is an unofficial endpoint. Intraday history is limited and may be
    unavailable for some symbols/regions; there is no guarantee of 100K bars.
    """
    if not yahoo_symbol:
        raise RuntimeError("Bu varlık için desteklenen Yahoo sembolü yok.")

    url = f"{YAHOO_BASE}/{requests.utils.quote(yahoo_symbol, safe='')}"
    params = {
        "range": "7d",
        "interval": "1m",
        "includePrePost": "false",
        "events": "history",
    }
    response = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    chart = payload.get("chart", {})
    if chart.get("error"):
        error = chart["error"] or {}
        raise RuntimeError(
            str(error.get("description") or error.get("code") or "Yahoo Finance veri hatası")
        )

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo Finance bu sembol için veri döndürmedi.")

    item = results[0]
    timestamps = item.get("timestamp") or []
    indicators = item.get("indicators", {})
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        raise RuntimeError("Yanıtta mum verisi bulunamadı.")

    quote = quotes[0]
    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    # Current forming minute is excluded.
    cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(second=0, microsecond=0))
    df = df[df["time"] < cutoff]

    if df.empty:
        raise RuntimeError("Son 7 günlük aralıkta tamamlanmış 1 dakikalık mum bulunamadı.")

    return df[["time", "open", "high", "low", "close", "volume"]]


def fetch_history_100k(asset, progress_callback=None):
    """Collect recent no-key Yahoo 1m bars and combine with locally cached history.

    100,000 candles is the target, not a promise: a no-key source usually only
    provides a short recent intraday window. Repeated scans can accumulate bars
    in SQLite over time, but cannot backfill unavailable history.
    """
    init_db()
    yahoo_symbol = get_yahoo_symbol(asset)
    if not yahoo_symbol:
        raise RuntimeError("Bu varlık için veri sağlayıcı sembolü tanımlı değil.")

    recent = yahoo_fetch_recent_1m(yahoo_symbol)
    if not recent.empty:
        db_write(yahoo_symbol, recent)

    stored = db_read_all(yahoo_symbol)
    frames = [frame for frame in (stored, recent) if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("Anahtarsız veri kaynağından mum alınamadı.")

    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["time", "open", "high", "low", "close"])
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time")
    combined = combined[combined["time"] < pd.Timestamp(datetime.now(timezone.utc).replace(second=0, microsecond=0))]
    combined = combined.tail(TARGET_CANDLES).reset_index(drop=True)

    if len(combined) < 60:
        raise RuntimeError(
            f"Yalnızca {len(combined):,} tamamlanmış mum alındı; sinyal için en az 60 mum gerekli."
        )

    source = f"Yahoo Finance (anahtarsız, {yahoo_symbol}) + yerel SQLite önbelleği"
    return combined, source


def rsi_calc(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi


def calculate_indicators(df):
    df = df.copy()
    if len(df) < 60:
        return pd.DataFrame()

    close, high, low = df["close"], df["high"], df["low"]
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    df["rsi"] = rsi_calc(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = (close - low14) / denom * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / close.replace(0, np.nan) * 100

    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_sm = plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    atr_safe = df["atr"].replace(0, np.nan)
    df["plus_di"] = 100 * plus_sm / atr_safe
    df["minus_di"] = 100 * minus_sm / atr_safe
    di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["dx"] = 100 * (df["plus_di"] - df["minus_di"]).abs() / di_sum
    df["adx"] = df["dx"].ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    df["momentum_3m"] = (close / close.shift(3) - 1) * 100
    return df


def make_signal(df, min_votes=4, min_margin=2):
    if df is None or len(df) < 60:
        return {"signal": "NO SIGNAL", "buy_score": 0, "sell_score": 0, "score": 0, "reason": "Yetersiz mum"}

    row = df.iloc[-1]
    required = [
        "close", "ema20", "ema50", "rsi", "bb_upper", "bb_lower",
        "macd", "macd_signal", "stoch_k", "stoch_d", "adx",
        "plus_di", "minus_di",
    ]
    if any(pd.isna(row[k]) for k in required):
        return {"signal": "NO SIGNAL", "buy_score": 0, "sell_score": 0, "score": 0, "reason": "İndikatör eksik"}

    buy = sell = 0
    reasons = []
    close = float(row["close"])

    if close > row["ema20"] > row["ema50"]:
        buy += 1
        reasons.append("EMA trendi yukarı")
    elif close < row["ema20"] < row["ema50"]:
        sell += 1
        reasons.append("EMA trendi aşağı")

    if close <= row["bb_lower"] * 1.01:
        buy += 1
        reasons.append("Alt Bollinger bandı")
    elif close >= row["bb_upper"] * 0.99:
        sell += 1
        reasons.append("Üst Bollinger bandı")

    if row["rsi"] < 30:
        buy += 1
        reasons.append("RSI aşırı satım")
    elif row["rsi"] > 70:
        sell += 1
        reasons.append("RSI aşırı alım")
    elif 50 < row["rsi"] < 68:
        buy += 1
    elif 32 < row["rsi"] < 50:
        sell += 1

    if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
        buy += 1
        reasons.append("MACD pozitif")
    elif row["macd"] < row["macd_signal"] and row["macd_hist"] < 0:
        sell += 1
        reasons.append("MACD negatif")

    if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
        buy += 1
        reasons.append("Stochastic yukarı kesişim")
    elif row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
        sell += 1
        reasons.append("Stochastic aşağı kesişim")

    if row["adx"] >= 20:
        if row["plus_di"] > row["minus_di"]:
            buy += 1
            reasons.append("ADX yönü yukarı")
        elif row["minus_di"] > row["plus_di"]:
            sell += 1
            reasons.append("ADX yönü aşağı")

    if pd.notna(row.get("volume_ratio")) and row["volume_ratio"] >= 1.2:
        if row["close"] > row["open"]:
            buy += 1
            reasons.append("Hacimli yeşil mum")
        elif row["close"] < row["open"]:
            sell += 1
            reasons.append("Hacimli kırmızı mum")

    if pd.notna(row.get("momentum_3m")):
        if row["momentum_3m"] > 0:
            buy += 1
        elif row["momentum_3m"] < 0:
            sell += 1

    if buy >= min_votes and buy - sell >= min_margin:
        signal = "BUY"
    elif sell >= min_votes and sell - buy >= min_margin:
        signal = "SELL"
    else:
        signal = "NO SIGNAL"
        reasons.append("Gösterge uyumu yetersiz")

    score = min(100, round(max(buy, sell) / 8 * 100))
    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "score": score,
        "reason": "; ".join(reasons),
    }


def scan_one(asset, min_votes, min_margin):
    if asset["category"] == "index" or not asset["symbol"]:
        return {
            **asset, "signal": "NO SIGNAL", "score": 0,
            "buy_score": 0, "sell_score": 0, "price": np.nan,
            "rsi": np.nan, "adx": np.nan, "count": 0,
            "source": "Desteklenen fiyat kaynağı yok",
            "reason": "Crypto IDX için doğrudan sağlayıcı sembolü tanımlı değil.",
            "candles": None, "error": "Desteklenen sembol yok",
        }

    try:
        candles, source = fetch_history_100k(asset)
        indicators = calculate_indicators(candles)
        if indicators.empty:
            raise RuntimeError("İndikatör hesaplamak için yeterli veri yok.")

        signal = make_signal(indicators, min_votes, min_margin)
        last = indicators.iloc[-1]

        if asset["otc"]:
            source += " — OTС DEĞİL; temel piyasa proxy verisi"

        return {
            **asset,
            "signal": signal["signal"],
            "score": signal["score"],
            "buy_score": signal["buy_score"],
            "sell_score": signal["sell_score"],
            "price": float(last["close"]),
            "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else np.nan,
            "adx": float(last["adx"]) if pd.notna(last["adx"]) else np.nan,
            "momentum_3m": float(last["momentum_3m"]) if pd.notna(last["momentum_3m"]) else np.nan,
            "count": len(candles),
            "source": source,
            "reason": signal["reason"],
            "candles": indicators,
            "error": None,
        }
    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL", "score": 0,
            "buy_score": 0, "sell_score": 0,
            "price": np.nan, "rsi": np.nan, "adx": np.nan,
            "momentum_3m": np.nan, "count": 0,
            "source": "Veri alınamadı",
            "reason": "Sinyal oluşturulmadı; veri hatası.",
            "candles": None, "error": str(exc),
        }


# ================================ UI ================================

st.title("📊 Velora — 100K Mum / 1 Dakika")
st.caption("BUY / SELL teknik tarayıcı • Binance kullanmaz • Emir göndermez")

st.warning(
    "Bu sürüm API anahtarı istemez; Yahoo Finance'in anahtarsız, resmi olmayan "
    "grafik uç noktasından son 7 günlük 1 dakikalık veriyi dener. Bu kaynak "
    "100.000 tarihsel mumu tek seferde sağlayamaz ve tüm sembolleri desteklemez. "
    "Gerçek Binomo OTC mumları burada bulunmaz. OTC etiketi yalnızca gösterim etiketidir."
)

st.info(
    "100.000 adet 1 dakikalık mum yaklaşık 69,4 gün kesintisiz veridir. "
    "Anahtarsız kaynak sadece sınırlı yakın geçmiş sağlayabilir; SQLite önbelleği "
    "verileri taramalar arasında biriktirir, ancak geçmişte hiç sunulmayan mumları "
    "oluşturamaz. Streamlit Cloud yeniden dağıtımlarında yerel önbellek silinebilir."
)

st.sidebar.header("Tarama ayarları")
group = st.sidebar.selectbox("Varlık kapsamı", ["Tümü", "Kripto", "Forex", "Diğer"])
min_votes = st.sidebar.slider("Minimum gösterge puanı", 3, 7, 4)
min_margin = st.sidebar.slider("BUY/SELL puan farkı", 1, 4, 2)
asset_limit = st.sidebar.number_input(
    "Bu taramada en fazla kaç varlık?", min_value=1, max_value=len(ASSETS),
    value=min(10, len(ASSETS)), step=1,
)
include_no_signal = st.sidebar.checkbox("NO SIGNAL sonuçlarını göster", value=True)

if group == "Kripto":
    chosen = [a for a in ASSETS if a["category"] == "crypto"]
elif group == "Forex":
    chosen = [a for a in ASSETS if a["category"] == "forex"]
elif group == "Diğer":
    chosen = [a for a in ASSETS if a["category"] in ("metal", "stock", "index")]
else:
    chosen = list(ASSETS)
chosen = chosen[:int(asset_limit)]

st.sidebar.success("Anahtar gerekmiyor — Yahoo Finance anahtarsız veri kaynağı")
st.sidebar.caption(
    "Not: 1 dakikalık tarihçe sınırlıdır; tüm forex/kripto sembolleri bulunmayabilir."
)

if "velora_results" not in st.session_state:
    st.session_state.velora_results = []
if "velora_scan_time" not in st.session_state:
    st.session_state.velora_scan_time = None

st.write(f"Bu taramada seçilen varlık: **{len(chosen)}**")
st.write(f"Her varlık için hedef: **{TARGET_CANDLES:,} tamamlanmış 1 dakikalık mum**")

if st.button("🔍 100.000 MUM İLE TARAMAYI BAŞLAT", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0)
    status = st.empty()

    # Keep low concurrency because the history job can make many API requests.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(scan_one, asset, min_votes, min_margin): asset["name"]
            for asset in chosen
        }
        total = len(futures)

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "name": futures[future], "signal": "NO SIGNAL",
                    "error": str(exc), "reason": "Tarama hatası",
                    "candles": None, "count": 0,
                })
            progress.progress(i / max(total, 1))
            status.write(f"Tamamlanan varlık: {i}/{total}")

    st.session_state.velora_results = results
    st.session_state.velora_scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    progress.empty()
    status.empty()

results = st.session_state.velora_results
if not results:
    st.info("Tarama henüz başlatılmadı. Anahtarsız veri kaynağını denemek için taramayı başlat.")
    st.stop()

valid = [r for r in results if r.get("candles") is not None and not r.get("error")]
errors = [r for r in results if r.get("error")]
buy = [r for r in valid if r.get("signal") == "BUY"]
sell = [r for r in valid if r.get("signal") == "SELL"]
no_signal = [r for r in valid if r.get("signal") == "NO SIGNAL"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("100K geçmişi tamamlanan", sum(1 for r in valid if r.get("count") == TARGET_CANDLES))
c2.metric("BUY", len(buy))
c3.metric("SELL", len(sell))
c4.metric("Veri hatası / eksik geçmiş", len(errors))

if st.session_state.velora_scan_time:
    st.caption("Son tarama: " + st.session_state.velora_scan_time)

st.subheader("🚨 BUY / SELL sinyalleri")
signals = sorted(buy + sell, key=lambda r: r.get("score", 0), reverse=True)
if signals:
    st.dataframe(pd.DataFrame([{
        "Varlık": r.get("name"),
        "Sinyal": r.get("signal"),
        "Fiyat": r.get("price"),
        "Skor": r.get("score"),
        "BUY puanı": r.get("buy_score"),
        "SELL puanı": r.get("sell_score"),
        "RSI": r.get("rsi"),
        "ADX": r.get("adx"),
        "Mum sayısı": r.get("count"),
        "100K ilerleme %": round(min(100, r.get("count", 0) / TARGET_CANDLES * 100), 2),
        "Kaynak": r.get("source"),
        "Gerekçe": r.get("reason"),
    } for r in signals]), use_container_width=True, hide_index=True)
else:
    st.info("Eşik koşullarını sağlayan BUY/SELL sinyali yok.")

st.subheader("📋 Tüm analiz sonuçları")
shown = valid if include_no_signal else [r for r in valid if r.get("signal") != "NO SIGNAL"]
if shown:
    st.dataframe(pd.DataFrame([{
        "Varlık": r.get("name"),
        "Kategori": r.get("category"),
        "Sinyal": r.get("signal"),
        "Fiyat": r.get("price"),
        "Skor": r.get("score"),
        "Mum sayısı": r.get("count"),
        "100K tamamlandı": r.get("count") == TARGET_CANDLES,
        "100K ilerleme %": round(min(100, r.get("count", 0) / TARGET_CANDLES * 100), 2),
        "Kaynak": r.get("source"),
    } for r in shown]), use_container_width=True, hide_index=True)

st.subheader("🔬 Varlık grafiği")
chartable = [r for r in valid if r.get("candles") is not None]
if chartable:
    selected_name = st.selectbox("Detaylı inceleme", [r["name"] for r in chartable])
    selected = next(r for r in chartable if r["name"] == selected_name)
    candles = selected["candles"]
    st.write(f"**Veri kaynağı:** {selected['source']}")
    st.write(f"**Alınan tamamlanmış mum:** {selected['count']:,} / {TARGET_CANDLES:,}")
    st.write(f"**Sinyal:** {selected['signal']} — {selected['reason']}")
    st.line_chart(candles.set_index("time")[["close", "ema20", "ema50"]].rename(
        columns={"close": "Kapanış", "ema20": "EMA 20", "ema50": "EMA 50"}
    ))
    st.line_chart(candles.set_index("time")[["close", "bb_upper", "bb_mid", "bb_lower"]].rename(
        columns={"close": "Kapanış", "bb_upper": "BB üst", "bb_mid": "BB orta", "bb_lower": "BB alt"}
    ))
    st.line_chart(candles.set_index("time")[["rsi"]].rename(columns={"rsi": "RSI"}))
else:
    st.info("Grafik için yeterli tarihsel veri bulunamadı.")

st.subheader("⚠️ Verisi alınamayan veya 100K mum tamamlayamayan varlıklar")
if errors:
    st.dataframe(pd.DataFrame([{
        "Varlık": r.get("name"),
        "Alınan mum": r.get("count", 0),
        "Hata": r.get("error"),
    } for r in errors]), use_container_width=True, hide_index=True)
else:
    st.success("Bu taramada veri hatası yok.")

if results:
    export = pd.DataFrame([{
        "asset": r.get("name"),
        "symbol": r.get("symbol"),
        "category": r.get("category"),
        "signal": r.get("signal"),
        "price": r.get("price"),
        "score": r.get("score"),
        "buy_score": r.get("buy_score"),
        "sell_score": r.get("sell_score"),
        "candle_count": r.get("count"),
        "source": r.get("source"),
        "reason": r.get("reason"),
        "error": r.get("error"),
    } for r in results])
    st.download_button(
        "📥 CSV sonuçlarını indir",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="velora_100k_1m_scan.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Velora yalnızca teknik analiz yapar; emir göndermez. Anahtarsız intraday kaynak "
    "resmi olmayan ve sınırlı olabilir; 100.000 mum hedefi garanti değildir. "
    "Sinyal skoru kazanma olasılığı değildir. OTC etiketli varlıklarda gerçek "
    "Binomo OTC fiyatı kullanılmıyor."
)
