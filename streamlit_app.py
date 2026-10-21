
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
# VELORA — 100K MUM / 1 DAKİKA
# BUY / SELL teknik tarayıcı
# Binance kullanmaz, emir göndermez.
# Gerçek Binomo OTC verisi sağladığını iddia etmez.
# ============================================================

st.set_page_config(
    page_title="Velora 1M Scanner",
    page_icon="📊",
    layout="wide",
)

TARGET_CANDLES = 100_000
CHUNK_MINUTES = 3_000
DB_PATH = "velora_history.sqlite3"
TD_BASE = "https://api.twelvedata.com"
HTTP_TIMEOUT = 30

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "VeloraHistoricalScanner/1.0"
})

# Varlık adı | kategori | Twelve Data sembolü | OTC etiketi
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


# ============================================================
# API ANAHTARI
# ============================================================

def get_api_key():
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.environ.get("TWELVE_DATA_API_KEY", "").strip()


# ============================================================
# SQLITE ÖNBELLEK
# ============================================================

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
        df["time"] = pd.to_datetime(
            df["time"], utc=True, errors="coerce"
        )

    return df


def db_write(symbol, df):
    if df is None or df.empty:
        return

    records = []

    for row in df.itertuples(index=False):
        records.append((
            symbol,
            pd.Timestamp(row.time).tz_convert("UTC").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(getattr(row, "volume", 0) or 0),
        ))

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO candles
            (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()


# ============================================================
# TWELVE DATA MUM VERİSİ
# ============================================================

def td_fetch_chunk(symbol, start_dt, end_dt, api_key):
    params = {
        "symbol": symbol,
        "interval": "1min",
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "outputsize": 5000,
        "timezone": "UTC",
        "apikey": api_key,
        "format": "JSON",
    }

    response = SESSION.get(
        f"{TD_BASE}/time_series",
        params=params,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        message = (
            payload.get("message")
            or payload.get("code")
            or str(payload)
        )
        raise RuntimeError(str(message))

    df = pd.DataFrame(payload["values"])

    if df.empty:
        return pd.DataFrame(columns=[
            "time", "open", "high", "low", "close", "volume"
        ])

    df["time"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df:
        df["volume"] = pd.to_numeric(
            df["volume"], errors="coerce"
        ).fillna(0)
    else:
        df["volume"] = 0.0

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    return df[
        ["time", "open", "high", "low", "close", "volume"]
    ]


def fetch_history_100k(asset, progress_callback=None):
    symbol = asset.get("symbol")

    if not symbol:
        raise RuntimeError(
            "Bu varlık için veri sağlayıcı sembolü tanımlı değil."
        )

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY eksik. Streamlit Secrets'e ekleyin."
        )

    init_db()

    now = datetime.now(timezone.utc).replace(
        second=0, microsecond=0
    )

    # Forex, metal ve hisse senetleri için piyasa kapalı saatleri
    # hesaba katılarak daha geniş bir tarih aralığı istenir.
    calendar_factor = (
        1.6
        if asset.get("category") in ("forex", "metal", "stock")
        else 1.08
    )

    lookback_minutes = (
        int(TARGET_CANDLES * calendar_factor) + 3_000
    )
    start = now - timedelta(minutes=lookback_minutes)

    cached = db_read(
        symbol,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if len(cached) >= TARGET_CANDLES:
        return (
            cached.tail(TARGET_CANDLES).reset_index(drop=True),
            "SQLite cache",
        )

    cursor = start
    chunk_count = int(np.ceil(
        (now - start).total_seconds() / 60 / CHUNK_MINUTES
    ))

    completed_chunks = 0
    new_parts = []

    while cursor < now:
        chunk_end = min(
            cursor + timedelta(minutes=CHUNK_MINUTES),
            now,
        )

        part = td_fetch_chunk(
            symbol, cursor, chunk_end, api_key
        )

        if not part.empty:
            # Henüz tamamlanmamış güncel dakikayı dahil etme.
            part = part[part["time"] < pd.Timestamp(now)]

            if not part.empty:
                db_write(symbol, part)
                new_parts.append(part)

        completed_chunks += 1

        if progress_callback:
            progress_callback(
                completed_chunks / max(chunk_count, 1)
            )

        cursor = chunk_end
        time.sleep(0.15)

    frames = [cached] if not cached.empty else []
    frames.extend(new_parts)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(
            combined["time"], utc=True, errors="coerce"
        )
        combined = combined.dropna(subset=["time"])
        combined = combined.drop_duplicates(
            subset=["time"]
        ).sort_values("time")
    else:
        combined = pd.DataFrame(columns=[
            "time", "open", "high", "low", "close", "volume"
        ])

    # Önceki çalıştırmalarda kaydedilen satırları da dahil et.
    stored = db_read(
        symbol,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if not stored.empty:
        combined = pd.concat(
            [combined, stored], ignore_index=True
        )
        combined = combined.drop_duplicates(
            subset=["time"]
        ).sort_values("time")

    combined = combined[
        combined["time"] < pd.Timestamp(now)
    ]

    combined = combined.tail(
        TARGET_CANDLES
    ).reset_index(drop=True)

    if len(combined) < TARGET_CANDLES:
        raise RuntimeError(
            f"{len(combined):,}/{TARGET_CANDLES:,} mum alınabildi. "
            "Veri sağlayıcısının geçmiş veri planı, sembol desteği "
            "veya API kotası yetersiz olabilir."
        )

    return combined, "Twelve Data historical"


# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================

def rsi_calc(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    return rsi


def calculate_indicators(df):
    df = df.copy()

    if len(df) < 60:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    # Bollinger Bands
    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    # RSI
    df["rsi"] = rsi_calc(close, 14)

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)

    df["stoch_k"] = (
        (close - low14) / denom * 100
    )
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    df["atr_pct"] = (
        df["atr"] / close.replace(0, np.nan) * 100
    )

    # ADX ve yön göstergeleri
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0.0,
        ),
        index=df.index,
    )

    plus_sm = plus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    minus_sm = minus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    atr_safe = df["atr"].replace(0, np.nan)

    df["plus_di"] = 100 * plus_sm / atr_safe
    df["minus_di"] = 100 * minus_sm / atr_safe

    di_sum = (
        df["plus_di"] + df["minus_di"]
    ).replace(0, np.nan)

    df["dx"] = (
        100
        * (df["plus_di"] - df["minus_di"]).abs()
        / di_sum
    )

    df["adx"] = df["dx"].ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    # Hacim ve momentum
    df["volume_sma"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_sma"].replace(0, np.nan)
    )

    df["momentum_3m"] = (
        close / close.shift(3) - 1
    ) * 100

    return df


# ============================================================
# BUY / SELL SİNYALİ
# ============================================================

def make_signal(df, min_votes=4, min_margin=2):
    if df is None or len(df) < 60:
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "score": 0,
            "reason": "Yetersiz mum",
        }

    row = df.iloc[-1]

    required = [
        "close", "ema20", "ema50", "rsi",
        "bb_upper", "bb_lower", "macd",
        "macd_signal", "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
    ]

    if any(pd.isna(row[key]) for key in required):
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "score": 0,
            "reason": "İndikatör eksik",
        }

    buy = 0
    sell = 0
    reasons = []
    close = float(row["close"])

    # EMA trendi
    if close > row["ema20"] > row["ema50"]:
        buy += 1
        reasons.append("EMA trendi yukarı")

    elif close < row["ema20"] < row["ema50"]:
        sell += 1
        reasons.append("EMA trendi aşağı")

    # Bollinger Bands
    if close <= row["bb_lower"] * 1.01:
        buy += 1
        reasons.append("Alt Bollinger bandı")

    elif close >= row["bb_upper"] * 0.99:
        sell += 1
        reasons.append("Üst Bollinger bandı")

    # RSI
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

    # MACD
    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):
        buy += 1
        reasons.append("MACD pozitif")

    elif (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):
        sell += 1
        reasons.append("MACD negatif")

    # Stochastic
    if (
        row["stoch_k"] < 20
        and row["stoch_k"] > row["stoch_d"]
    ):
        buy += 1
        reasons.append("Stochastic yukarı kesişim")

    elif (
        row["stoch_k"] > 80
        and row["stoch_k"] < row["stoch_d"]
    ):
        sell += 1
        reasons.append("Stochastic aşağı kesişim")

    # ADX yön filtresi
    if row["adx"] >= 20:
        if row["plus_di"] > row["minus_di"]:
            buy += 1
            reasons.append("ADX yönü yukarı")

        elif row["minus_di"] > row["plus_di"]:
            sell += 1
            reasons.append("ADX yönü aşağı")

    # Hacim filtresi
    if (
        pd.notna(row.get("volume_ratio"))
        and row["volume_ratio"] >= 1.2
    ):
        if row["close"] > row["open"]:
            buy += 1
            reasons.append("Hacimli yeşil mum")

        elif row["close"] < row["open"]:
            sell += 1
            reasons.append("Hacimli kırmızı mum")

    # Son üç dakikalık momentum
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

    score = min(
        100,
        round(max(buy, sell) / 8 * 100),
    )

    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "score": score,
        "reason": "; ".join(reasons),
    }


# ============================================================
# TEK VARLIK TARAMASI
# ============================================================

def scan_one(asset, min_votes, min_margin):
    if asset["category"] == "index" or not asset["symbol"]:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "score": 0,
            "buy_score": 0,
            "sell_score": 0,
            "price": np.nan,
            "rsi": np.nan,
            "adx": np.nan,
            "count": 0,
            "source": "Desteklenen fiyat kaynağı yok",
            "reason": (
                "Crypto IDX için doğrudan sağlayıcı "
                "sembolü tanımlı değil."
            ),
            "candles": None,
            "error": "Desteklenen sembol yok",
        }

    try:
        candles, source = fetch_history_100k(asset)
        indicators = calculate_indicators(candles)

        if indicators.empty:
            raise RuntimeError(
                "İndikatör hesaplamak için yeterli veri yok."
            )

        signal = make_signal(
            indicators,
            min_votes,
            min_margin,
        )

        last = indicators.iloc[-1]

        if asset["otc"]:
            source += " — OTC DEĞİL; temel piyasa proxy verisi"

        return {
            **asset,
            "signal": signal["signal"],
            "score": signal["score"],
            "buy_score": signal["buy_score"],
            "sell_score": signal["sell_score"],
            "price": float(last["close"]),
            "rsi": (
                float(last["rsi"])
                if pd.notna(last["rsi"])
                else np.nan
            ),
            "adx": (
                float(last["adx"])
                if pd.notna(last["adx"])
                else np.nan
            ),
            "momentum_3m": (
                float(last["momentum_3m"])
                if pd.notna(last["momentum_3m"])
                else np.nan
            ),
            "count": len(candles),
            "source": source,
            "reason": signal["reason"],
            "candles": indicators,
            "error": None,
        }

    except Exception as exc:
        return {
            **asset,
            "signal": "NO SIGNAL",
            "score": 0,
            "buy_score": 0,
            "sell_score": 0,
            "price": np.nan,
            "rsi": np.nan,
            "adx": np.nan,
            "momentum_3m": np.nan,
            "count": 0,
            "source": "Veri alınamadı",
            "reason": "Sinyal oluşturulmadı; veri hatası.",
            "candles": None,
            "error": str(exc),
        }


# ============================================================
# STREAMLIT ARAYÜZÜ
# ============================================================

st.title("📊 Velora — 100K Mum / 1 Dakika")

st.caption(
    "BUY / SELL teknik tarayıcı • Binance kullanmaz • Emir göndermez"
)

st.warning(
    "Gerçek Binomo OTC mumları bu uygulamada bulunmuyor. "
    "OTC etiketi taşıyan varlıklarda sağlayıcıdan gelen temel piyasa "
    "verisi kullanılabilir; bu, Binomo OTC fiyatı değildir. Sağlayıcıda "
    "sembol veya 100.000 mum geçmişi yoksa uygulama hatayı gösterir."
)

st.info(
    "100.000 adet 1 dakikalık mum yaklaşık 69,4 günlük veri demektir. "
    "Her varlık için bu kadar geçmişi indirmek çok sayıda API çağrısı "
    "ve uygun veri planı gerektirebilir. İlk tarama uzun sürebilir; "
    "veriler SQLite yerel önbelleğinde saklanır."
)

st.sidebar.header("Tarama ayarları")

group = st.sidebar.selectbox(
    "Varlık kapsamı",
    ["Tümü", "Kripto", "Forex", "Diğer"],
)

min_votes = st.sidebar.slider(
    "Minimum gösterge puanı", 3, 7, 4
)

min_margin = st.sidebar.slider(
    "BUY/SELL puan farkı", 1, 4, 2
)

asset_limit = st.sidebar.number_input(
    "Bu taramada en fazla kaç varlık?",
    min_value=1,
    max_value=len(ASSETS),
    value=min(10, len(ASSETS)),
    step=1,
)

include_no_signal = st.sidebar.checkbox(
    "NO SIGNAL sonuçlarını göster",
    value=True,
)

if group == "Kripto":
    chosen = [
        a for a in ASSETS
        if a["category"] == "crypto"
    ]

elif group == "Forex":
    chosen = [
        a for a in ASSETS
        if a["category"] == "forex"
    ]

elif group == "Diğer":
    chosen = [
        a for a in ASSETS
        if a["category"] in ("metal", "stock", "index")
    ]

else:
    chosen = list(ASSETS)

chosen = chosen[:int(asset_limit)]

api_key = get_api_key()

if api_key:
    st.sidebar.success("Twelve Data API anahtarı bulundu.")
else:
    st.sidebar.error("TWELVE_DATA_API_KEY tanımlı değil.")

if "velora_results" not in st.session_state:
    st.session_state.velora_results = []

if "velora_scan_time" not in st.session_state:
    st.session_state.velora_scan_time = None

st.write(f"Bu taramada seçilen varlık: **{len(chosen)}**")

st.write(
    f"Her varlık için hedef: **{TARGET_CANDLES:,} "
    "tamamlanmış 1 dakikalık mum**"
)

if st.button(
    "🔍 100.000 MUM İLE TARAMAYI BAŞLAT",
    type="primary",
    use_container_width=True,
):
    if not api_key:
        st.error(
            "Önce Streamlit Secrets bölümüne "
            "TWELVE_DATA_API_KEY ekleyin."
        )
        st.stop()

    results = []
    progress = st.progress(0)
    status = st.empty()

    # API çağrılarını sınırlı eşzamanlılıkla çalıştır.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                scan_one,
                asset,
                min_votes,
                min_margin,
            ): asset["name"]
            for asset in chosen
        }

        total = len(futures)

        for i, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            try:
                results.append(future.result())

            except Exception as exc:
                results.append({
                    "name": futures[future],
                    "signal": "NO SIGNAL",
                    "error": str(exc),
                    "reason": "Tarama hatası",
                    "candles": None,
                    "count": 0,
                })

            progress.progress(i / max(total, 1))
            status.write(f"Tamamlanan varlık: {i}/{total}")

    st.session_state.velora_results = results

    st.session_state.velora_scan_time = (
        datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    progress.empty()
    status.empty()


# ============================================================
# SONUÇLAR
# ============================================================

results = st.session_state.velora_results

if not results:
    st.info(
        "Tarama başlatılmadı. API anahtarını ekledikten "
        "sonra taramayı başlat."
    )
    st.stop()

valid = [
    r for r in results
    if r.get("candles") is not None and not r.get("error")
]

errors = [r for r in results if r.get("error")]

buy = [r for r in valid if r.get("signal") == "BUY"]
sell = [r for r in valid if r.get("signal") == "SELL"]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "100K geçmişi tamamlanan",
    sum(
        1 for r in valid
        if r.get("count") == TARGET_CANDLES
    ),
)

c2.metric("BUY", len(buy))
c3.metric("SELL", len(sell))
c4.metric("Veri hatası / eksik geçmiş", len(errors))

if st.session_state.velora_scan_time:
    st.caption("Son tarama: " + st.session_state.velora_scan_time)


# BUY / SELL sinyalleri
st.subheader("🚨 BUY / SELL sinyalleri")

signals = sorted(
    buy + sell,
    key=lambda r: r.get("score", 0),
    reverse=True,
)

if signals:
    st.dataframe(
        pd.DataFrame([
            {
                "Varlık": r.get("name"),
                "Sinyal": r.get("signal"),
                "Fiyat": r.get("price"),
                "Skor": r.get("score"),
                "BUY puanı": r.get("buy_score"),
                "SELL puanı": r.get("sell_score"),
                "RSI": r.get("rsi"),
                "ADX": r.get("adx"),
                "Mum sayısı": r.get("count"),
                "Kaynak": r.get("source"),
                "Gerekçe": r.get("reason"),
            }
            for r in signals
        ]),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Eşik koşullarını sağlayan BUY/SELL sinyali yok.")


# Tüm analiz sonuçları
st.subheader("📋 Tüm analiz sonuçları")

shown = (
    valid
    if include_no_signal
    else [r for r in valid if r.get("signal") != "NO SIGNAL"]
)

if shown:
    st.dataframe(
        pd.DataFrame([
            {
                "Varlık": r.get("name"),
                "Kategori": r.get("category"),
                "Sinyal": r.get("signal"),
                "Fiyat": r.get("price"),
                "Skor": r.get("score"),
                "Mum sayısı": r.get("count"),
                "100K tamamlandı": (
                    r.get("count") == TARGET_CANDLES
                ),
                "Kaynak": r.get("source"),
            }
            for r in shown
        ]),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GRAFİKLER
# ============================================================

st.subheader("🔬 Varlık grafiği")

chartable = [
    r for r in valid
    if r.get("candles") is not None
]

if chartable:
    selected_name = st.selectbox(
        "Detaylı inceleme",
        [r["name"] for r in chartable],
    )

    selected = next(
        r for r in chartable
        if r["name"] == selected_name
    )

    candles = selected["candles"]

    st.write(f"**Veri kaynağı:** {selected['source']}")

    st.write(
        f"**Alınan tamamlanmış mum:** "
        f"{selected['count']:,} / {TARGET_CANDLES:,}"
    )

    st.write(
        f"**Sinyal:** {selected['signal']} — {selected['reason']}"
    )

    st.line_chart(
        candles.set_index("time")[["close", "ema20", "ema50"]].rename(
            columns={
                "close": "Kapanış",
                "ema20": "EMA 20",
                "ema50": "EMA 50",
            }
        )
    )

    st.line_chart(
        candles.set_index("time")[
            ["close", "bb_upper", "bb_mid", "bb_lower"]
        ].rename(
            columns={
                "close": "Kapanış",
                "bb_upper": "BB üst",
                "bb_mid": "BB orta",
                "bb_lower": "BB alt",
            }
        )
    )

    st.line_chart(
        candles.set_index("time")[["rsi"]].rename(
            columns={"rsi": "RSI"}
        )
    )

else:
    st.info("Grafik için yeterli tarihsel veri bulunamadı.")


# ============================================================
# VERİ HATALARI
# ============================================================

st.subheader(
    "⚠️ Verisi alınamayan veya 100K mum tamamlayamayan varlıklar"
)

if errors:
    st.dataframe(
        pd.DataFrame([
            {
                "Varlık": r.get("name"),
                "Alınan mum": r.get("count", 0),
                "Hata": r.get("error"),
            }
            for r in errors
        ]),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.success("Bu taramada veri hatası yok.")


# ============================================================
# CSV DIŞA AKTARMA
# ============================================================

if results:
    export = pd.DataFrame([
        {
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
        }
        for r in results
    ])

    st.download_button(
        "📥 CSV sonuçlarını indir",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="velora_100k_1m_scan.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()

st.caption(
    "Velora yalnızca teknik analiz yapar; emir göndermez. 100.000 mum hedefi, "
    "veri sağlayıcısının tarihçe, sembol ve API kotasıyla sınırlıdır. "
    "Sinyal skoru kazanma olasılığı değildir. OTC olarak işaretlenen varlıklarda "
    "gerçek Binomo OTC fiyatı kullanılmıyor."
)
