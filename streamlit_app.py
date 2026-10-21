
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# VELORA AI CRYPTO & FOREX SCANNER
# 1-minute analysis, BUY / SELL only, no trade execution
# =========================================================

st.set_page_config(
    page_title="Velora 1M Scanner",
    page_icon="📊",
    layout="wide",
)

KRAKEN = "https://api.kraken.com/0/public"
TWELVE_DATA = "https://api.twelvedata.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "VeloraScanner/1.0"})

INTERVAL = "1min"
CANDLE_LIMIT = 250

# =========================================================
# ASSETS FROM THE USER'S BINOMO LIST
# OTC prices are NOT claimed to be real Binomo OTC prices.
# =========================================================

ASSETS = [
    # Crypto and crypto index
    ("Crypto IDX", "index", None, False),
    ("Bitcoin (OTC)", "crypto", "BTC", True),
    ("Ethereum (OTC)", "crypto", "ETH", True),
    ("Solana (OTC)", "crypto", "SOL", True),
    ("FC Barcelona Token (OTC)", "crypto", "BAR", True),
    ("Cardano (OTC)", "crypto", "ADA", True),
    ("Chainlink (OTC)", "crypto", "LINK", True),
    ("Bitcoin Cash (OTC)", "crypto", "BCH", True),
    ("Kusama (OTC)", "crypto", "KSM", True),
    ("Aave (OTC)", "crypto", "AAVE", True),
    ("Pancake Swap (OTC)", "crypto", "CAKE", True),
    ("Uniswap (OTC)", "crypto", "UNI", True),

    # Forex
    ("EUR/USD", "forex", "EUR/USD", False),
    ("EUR/HUF", "forex", "EUR/HUF", False),
    ("GBP/NOK", "forex", "GBP/NOK", False),
    ("CAD/JPY", "forex", "CAD/JPY", False),
    ("AUD/ZAR", "forex", "AUD/ZAR", False),
    ("CAD/DKK", "forex", "CAD/DKK", False),
    ("EUR/ILS", "forex", "EUR/ILS", False),
    ("GBP/SGD", "forex", "GBP/SGD", False),
    ("USD/MXN", "forex", "USD/MXN", False),
    ("EUR/ZAR", "forex", "EUR/ZAR", False),
    ("AUD/USD", "forex", "AUD/USD", False),
    ("GBP/HKD", "forex", "GBP/HKD", False),
    ("CAD/MXN", "forex", "CAD/MXN", False),
    ("EUR/NOK", "forex", "EUR/NOK", False),
    ("EUR/MXN", "forex", "EUR/MXN", False),
    ("CHF/JPY", "forex", "CHF/JPY", False),
    ("USD/ILS", "forex", "USD/ILS", False),
    ("USD/SEK", "forex", "USD/SEK", False),
    ("CAD/NOK", "forex", "CAD/NOK", False),
    ("CAD/SEK", "forex", "CAD/SEK", False),
    ("USD/PLN", "forex", "USD/PLN", False),
    ("AUD/JPY", "forex", "AUD/JPY", False),
    ("EUR/SEK", "forex", "EUR/SEK", False),
    ("AUD/SGD", "forex", "AUD/SGD", False),
    ("GBP/MXN", "forex", "GBP/MXN", False),
    ("CAD/SGD", "forex", "CAD/SGD", False),
    ("AUD/DKK", "forex", "AUD/DKK", False),
    ("EUR/SGD", "forex", "EUR/SGD", False),
    ("CHF/DKK", "forex", "CHF/DKK", False),
    ("GBP/PLN", "forex", "GBP/PLN", False),
    ("GBP/CZK", "forex", "GBP/CZK", False),
    ("EUR/NZD", "forex", "EUR/NZD", False),
    ("AUD/HUF", "forex", "AUD/HUF", False),
    ("USD/HUF", "forex", "USD/HUF", False),
    ("GBP/DKK", "forex", "GBP/DKK", False),
    ("AUD/NOK", "forex", "AUD/NOK", False),
    ("NZD/JPY", "forex", "NZD/JPY", False),
    ("CHF/NOK", "forex", "CHF/NOK", False),
    ("USD/JPY", "forex", "USD/JPY", False),
    ("CHF/PLN", "forex", "CHF/PLN", False),
    ("GBP/TRY", "forex", "GBP/TRY", False),
    ("USD/CAD", "forex", "USD/CAD", False),
    ("AUD/SEK", "forex", "AUD/SEK", False),
    ("CHF/SEK", "forex", "CHF/SEK", False),
    ("CHF/SGD", "forex", "CHF/SGD", False),
    ("NOK/JPY", "forex", "NOK/JPY", False),
    ("NOK/SEK", "forex", "NOK/SEK", False),
    ("NZD/CAD", "forex", "NZD/CAD", False),
    ("NZD/DKK", "forex", "NZD/DKK", False),
    ("EUR/GBP (OTC)", "forex", "EUR/GBP", True),
    ("GBP/NZD", "forex", "GBP/NZD", False),
    ("NZD/USD", "forex", "NZD/USD", False),
    ("NZD/HUF", "forex", "NZD/HUF", False),
    ("NZD/NOK", "forex", "NZD/NOK", False),
    ("NZD/SEK", "forex", "NZD/SEK", False),
    ("NZD/SGD", "forex", "NZD/SGD", False),
    ("SEK/JPY", "forex", "SEK/JPY", False),
    ("SGD/JPY", "forex", "SGD/JPY", False),
    ("SGD/HKD", "forex", "SGD/HKD", False),
    ("USD/CHF", "forex", "USD/CHF", False),
    ("AUD/CAD", "forex", "AUD/CAD", False),
    ("USD/CZK", "forex", "USD/CZK", False),
    ("NZD/CHF", "forex", "NZD/CHF", False),
    ("GBP/CAD", "forex", "GBP/CAD", False),
    ("AUD/NZD", "forex", "AUD/NZD", False),
    ("USD/ZAR", "forex", "USD/ZAR", False),
    ("AUD/CHF", "forex", "AUD/CHF", False),
    ("GBP/USD", "forex", "GBP/USD", False),
    ("GBP/HUF", "forex", "GBP/HUF", False),
    ("EUR/PLN", "forex", "EUR/PLN", False),
    ("ZAR/JPY", "forex", "ZAR/JPY", False),
    ("CHF/HUF", "forex", "CHF/HUF", False),
    ("USD/DKK", "forex", "USD/DKK", False),
    ("GBP/CHF", "forex", "GBP/CHF", False),

    # Other assets
    ("Gold", "metal", "XAU/USD", False),
    ("Yum Brands", "stock", "YUM", False),
]

# =========================================================
# OPTIONAL TWELVE DATA API KEY
# Add TWELVE_DATA_API_KEY to Streamlit Secrets.
# Without it, FX/metals/stocks will be reported unavailable.
# =========================================================

def get_twelve_data_key():
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.environ.get("TWELVE_DATA_API_KEY", "").strip()


# =========================================================
# KRAKEN PUBLIC API
# =========================================================

def kraken_get(endpoint, params=None):
    response = SESSION.get(
        f"{KRAKEN}/{endpoint}",
        params=params or {},
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()
    errors = payload.get("error", [])

    if errors:
        raise RuntimeError("; ".join(errors))

    return payload.get("result", {})


@st.cache_data(ttl=300, show_spinner=False)
def get_kraken_pairs():
    result = kraken_get("AssetPairs")
    pairs = {}

    aliases = {
        "XXBT": "BTC",
        "XBT": "BTC",
        "XETH": "ETH",
        "XXDG": "DOGE",
        "XDG": "DOGE",
        "ZUSD": "USD",
        "ZEUR": "EUR",
    }

    for pair_key, info in result.items():
        if info.get("status", "online") != "online":
            continue

        wsname = info.get("wsname", "")

        if "/" not in wsname:
            continue

        base, quote = wsname.split("/", 1)
        base = aliases.get(base, base)
        quote = aliases.get(quote, quote)

        if quote not in ("USD", "USDT"):
            continue

        pairs.setdefault(base, []).append({
            "pair_key": pair_key,
            "quote": quote,
            "wsname": wsname,
        })

    return pairs


@st.cache_data(ttl=40, show_spinner=False)
def get_kraken_candles(pair_key, limit=250):
    result = kraken_get(
        "OHLC",
        {
            "pair": pair_key,
            "interval": 1,
        },
    )

    candle_key = next(
        (k for k in result if k != "last"),
        None,
    )

    if candle_key is None:
        return pd.DataFrame()

    raw = result[candle_key]

    if len(raw) > 1:
        raw = raw[:-1]  # Exclude current incomplete candle

    df = pd.DataFrame(
        raw,
        columns=[
            "time", "open", "high", "low", "close",
            "vwap", "volume", "count",
        ],
    )

    if df.empty:
        return df

    for col in [
        "open", "high", "low", "close", "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="s",
        utc=True,
    )

    return df.dropna(
        subset=["open", "high", "low", "close"]
    ).tail(limit).reset_index(drop=True)


# =========================================================
# TWELVE DATA FOR FOREX / METALS / STOCKS
# =========================================================

def get_twelve_data_candles(symbol, limit=250):
    key = get_twelve_data_key()

    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY tanımlı değil. "
            "Gerçek 1 dakikalık forex/metal/hisse verisi "
            "için bir Twelve Data API anahtarı ekleyin."
        )

    response = SESSION.get(
        f"{TWELVE_DATA}/time_series",
        params={
            "symbol": symbol,
            "interval": "1min",
            "outputsize": min(limit, 5000),
            "apikey": key,
            "format": "JSON",
        },
        timeout=25,
    )

    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        message = payload.get(
            "message",
            payload.get("status", "Veri alınamadı."),
        )
        raise RuntimeError(str(message))

    df = pd.DataFrame(payload["values"])

    if df.empty:
        return df

    df["time"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        ).fillna(0)
    else:
        df["volume"] = 0.0

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    df = df.sort_values("time")

    # Avoid using the still-forming one-minute candle.
    current_minute = pd.Timestamp.now(
        tz="UTC"
    ).floor("min")

    df = df[df["time"] < current_minute]

    return df.tail(limit).reset_index(drop=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

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
    result = 100 - 100 / (1 + rs)

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )

    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0,
    )

    return result


def calculate_indicators(df):
    df = df.copy()

    if len(df) < 60:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

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

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    df["atr_pct"] = (
        df["atr"] / close.replace(0, np.nan) * 100
    )

    # Directional movement / ADX
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

    atr_safe = df["atr"].replace(0, np.nan)

    plus_smoothed = plus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    df["plus_di"] = 100 * plus_smoothed / atr_safe
    df["minus_di"] = 100 * minus_smoothed / atr_safe

    di_total = (
        df["plus_di"] + df["minus_di"]
    ).replace(0, np.nan)

    df["dx"] = (
        100
        * (df["plus_di"] - df["minus_di"]).abs()
        / di_total
    )

    df["adx"] = df["dx"].ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    # Volume
    df["volume_sma"] = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / df["volume_sma"].replace(0, np.nan)
    )

    # Momentum
    df["momentum_3m"] = (
        (close / close.shift(3) - 1) * 100
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def make_signal(df, min_votes=4, min_margin=2):
    if df is None or len(df) < 60:
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "score": 0,
            "reason": "Yetersiz mum verisi",
        }

    row = df.iloc[-1]

    required = [
        "close", "ema20", "ema50", "rsi",
        "bb_upper", "bb_lower", "macd",
        "macd_signal", "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
    ]

    if any(pd.isna(row[x]) for x in required):
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "score": 0,
            "reason": "İndikatör verisi eksik",
        }

    buy = 0
    sell = 0
    reasons = []

    close = float(row["close"])

    # EMA trend
    if close > row["ema20"] > row["ema50"]:
        buy += 1
        reasons.append("EMA yükseliş trendi")

    elif close < row["ema20"] < row["ema50"]:
        sell += 1
        reasons.append("EMA düşüş trendi")

    # Bollinger Bands
    if close <= row["bb_lower"] * 1.01:
        buy += 1
        reasons.append("Alt Bollinger bandına yakın")

    elif close >= row["bb_upper"] * 0.99:
        sell += 1
        reasons.append("Üst Bollinger bandına yakın")

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

    # ADX directional movement
    if row["adx"] >= 20:
        if row["plus_di"] > row["minus_di"]:
            buy += 1
            reasons.append("ADX pozitif yön")

        elif row["minus_di"] > row["plus_di"]:
            sell += 1
            reasons.append("ADX negatif yön")

    # Volume confirmation
    volume_ratio = row["volume_ratio"]

    if pd.notna(volume_ratio) and volume_ratio >= 1.2:
        if close > row["open"]:
            buy += 1
            reasons.append("Yüksek hacimli yeşil mum")

        elif close < row["open"]:
            sell += 1
            reasons.append("Yüksek hacimli kırmızı mum")

    # Three-minute momentum
    if pd.notna(row["momentum_3m"]):
        if row["momentum_3m"] > 0:
            buy += 1
        elif row["momentum_3m"] < 0:
            sell += 1

    # Final decision
    if buy >= min_votes and buy - sell >= min_margin:
        signal = "BUY"

    elif sell >= min_votes and sell - buy >= min_margin:
        signal = "SELL"

    else:
        signal = "NO SIGNAL"
        reasons.append("Yeterli gösterge uyumu yok")

    score = round(max(buy, sell) / 8 * 100)

    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "score": min(score, 100),
        "reason": "; ".join(reasons),
    }


# =========================================================
# ASSET DATA ROUTER
# =========================================================

def fetch_asset_data(asset):
    name, category, symbol, is_otc = asset

    try:
        if category == "crypto":
            markets = get_kraken_pairs()
            choices = markets.get(symbol, [])

            if not choices:
                raise RuntimeError(
                    f"{symbol} için uygun Kraken spot paritesi yok."
                )

            # Prefer USD; fall back to USDT.
            choices = sorted(
                choices,
                key=lambda x: (
                    0 if x["quote"] == "USD" else 1
                ),
            )

            pair = choices[0]
            candles = get_kraken_candles(
                pair["pair_key"],
                limit=CANDLE_LIMIT,
            )

            source = "Kraken Spot proxy"

        elif category in ("forex", "metal", "stock"):
            candles = get_twelve_data_candles(
                symbol,
                limit=CANDLE_LIMIT,
            )

            source = "Twelve Data"

        else:
            raise RuntimeError(
                "Bu varlık için desteklenen 1 dakikalık "
                "piyasa verisi kaynağı tanımlı değil."
            )

        if candles.empty or len(candles) < 60:
            raise RuntimeError(
                "En az 60 tamamlanmış mum alınamadı."
            )

        indicators = calculate_indicators(candles)

        if indicators.empty:
            raise RuntimeError(
                "İndikatörler hesaplanamadı."
            )

        signal = make_signal(indicators)

        # Clearly identify non-Binomo prices.
        if is_otc:
            source_label = (
                source + " — OTC DEĞİL, YAKLAŞIK VERİ"
            )
        else:
            source_label = source

        return {
            "asset": name,
            "category": category,
            "symbol": symbol,
            "is_otc": is_otc,
            "source": source_label,
            "price": float(indicators.iloc[-1]["close"]),
            "signal": signal["signal"],
            "score": signal["score"],
            "buy_score": signal["buy_score"],
            "sell_score": signal["sell_score"],
            "rsi": float(indicators.iloc[-1]["rsi"]),
            "adx": float(indicators.iloc[-1]["adx"]),
            "momentum_3m": float(
                indicators.iloc[-1]["momentum_3m"]
            ),
            "reason": signal["reason"],
            "candles": indicators,
            "error": None,
        }

    except Exception as exc:
        return {
            "asset": name,
            "category": category,
            "symbol": symbol,
            "is_otc": is_otc,
            "source": "Veri yok",
            "price": np.nan,
            "signal": "NO SIGNAL",
            "score": 0,
            "buy_score": 0,
            "sell_score": 0,
            "rsi": np.nan,
            "adx": np.nan,
            "momentum_3m": np.nan,
            "reason": "Veri alınamadı",
            "candles": None,
            "error": str(exc),
        }


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("📊 Velora 1-Minute Scanner")

st.markdown(
    """
    **Zaman dilimi:** 1 dakika  
    **Sinyaller:** BUY / SELL / NO SIGNAL  
    **İşlem modu:** Yalnızca analiz — emir gönderilmez
    """
)

st.warning(
    "Gerçek Binomo OTC mum verisine doğrudan erişim "
    "bu uygulamada bulunmuyor. OTC olarak listelenen "
    "kripto varlıklar için destekleniyorsa Kraken spot "
    "verisi yalnızca yaklaşık gösterge kaynağı olarak "
    "kullanılır; Binomo OTC fiyatı değildir."
)

st.info(
    "Forex, Gold ve Yum Brands için Twelve Data API anahtarı "
    "gerekebilir. Ücretsiz API planı 1 dakikalık veri veya "
    "yüksek sayıda isteği kısıtlayabilir. Veri alınamayan "
    "varlıklara sinyal uydurulmaz."
)

# Sidebar
st.sidebar.header("Tarama ayarları")

asset_filter = st.sidebar.selectbox(
    "Varlık grubu",
    [
        "Tümü",
        "Kripto",
        "Forex",
        "Diğer",
    ],
)

min_votes = st.sidebar.slider(
    "Minimum gösterge puanı",
    min_value=3,
    max_value=7,
    value=4,
)

min_margin = st.sidebar.slider(
    "Minimum BUY/SELL puan farkı",
    min_value=1,
    max_value=4,
    value=2,
)

max_assets = st.sidebar.slider(
    "Tarama yapılacak varlık sayısı",
    min_value=10,
    max_value=len(ASSETS),
    value=len(ASSETS),
    step=5,
)

show_no_signal = st.sidebar.checkbox(
    "NO SIGNAL sonuçlarını göster",
    value=True,
)

# Filter assets
if asset_filter == "Kripto":
    filtered_assets = [
        x for x in ASSETS
        if x[1] == "crypto"
    ]

elif asset_filter == "Forex":
    filtered_assets = [
        x for x in ASSETS
        if x[1] == "forex"
    ]

elif asset_filter == "Diğer":
    filtered_assets = [
        x for x in ASSETS
        if x[1] in ("metal", "stock", "index")
    ]

else:
    filtered_assets = list(ASSETS)

filtered_assets = filtered_assets[:max_assets]

st.write(
    f"Seçili varlık sayısı: **{len(filtered_assets)}**"
)

# API key status
if get_twelve_data_key():
    st.sidebar.success("Twelve Data API anahtarı tanımlı.")
else:
    st.sidebar.warning(
        "Twelve Data API anahtarı yok. "
        "Forex/Gold/hisse verisi alınamayabilir."
    )

# Scan button
if "velora_results" not in st.session_state:
    st.session_state["velora_results"] = []

if "velora_scan_time" not in st.session_state:
    st.session_state["velora_scan_time"] = None

if st.button(
    "🔍 1 DAKİKALIK TARAMAYI BAŞLAT",
    type="primary",
    use_container_width=True,
):
    output = []
    progress = st.progress(0)
    status = st.empty()

    # Low concurrency helps reduce API pressure.
    workers = 2

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                fetch_asset_data,
                asset,
            ): asset[0]
            for asset in filtered_assets
        }

        total = len(futures)

        for index, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            try:
                result = future.result()

                if result:
                    output.append(result)

            except Exception as exc:
                output.append({
                    "asset": futures[future],
                    "signal": "NO SIGNAL",
                    "error": str(exc),
                    "reason": "Veri alınamadı",
                    "candles": None,
                })

            progress.progress(index / total)

            status.write(
                f"Taranıyor: {index}/{total}"
            )

            time.sleep(0.05)

    st.session_state["velora_results"] = output

    st.session_state["velora_scan_time"] = (
        pd.Timestamp.now(tz="UTC").strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    progress.empty()
    status.empty()


# =========================================================
# RESULTS
# =========================================================

results = st.session_state["velora_results"]

if not results:
    st.info(
        "Henüz tarama yapılmadı. "
        "Taramayı başlat düğmesine bas."
    )
    st.stop()

valid = [
    item for item in results
    if item.get("candles") is not None
    and item.get("error") is None
]

unavailable = [
    item for item in results
    if item.get("error")
]

buy_results = [
    item for item in valid
    if item["signal"] == "BUY"
]

sell_results = [
    item for item in valid
    if item["signal"] == "SELL"
]

no_signal_results = [
    item for item in valid
    if item["signal"] == "NO SIGNAL"
]

a, b, c, d = st.columns(4)

a.metric("Geçerli analiz", len(valid))
b.metric("BUY", len(buy_results))
c.metric("SELL", len(sell_results))
d.metric("Veri yok", len(unavailable))

if st.session_state["velora_scan_time"]:
    st.caption(
        "Son tarama: "
        + st.session_state["velora_scan_time"]
    )

# Priority signals first
st.subheader("🚨 BUY / SELL sinyalleri")

signals = buy_results + sell_results
signals.sort(
    key=lambda x: x.get("score", 0),
    reverse=True,
)

if not signals:
    st.info(
        "Bu taramada eşik koşullarını sağlayan BUY/SELL "
        "sinyali oluşmadı."
    )
else:
    signal_table = pd.DataFrame([
        {
            "Asset": x["asset"],
            "Signal": x["signal"],
            "Price": x["price"],
            "Score": x["score"],
            "BUY points": x["buy_score"],
            "SELL points": x["sell_score"],
            "RSI": x["rsi"],
            "ADX": x["adx"],
            "3m change %": x["momentum_3m"],
            "Data source": x["source"],
            "Reason": x["reason"],
        }
        for x in signals
    ])

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True,
    )

# All valid results
st.subheader("📋 Tüm geçerli analizler")

all_rows = valid.copy()

if not show_no_signal:
    all_rows = [
        x for x in all_rows
        if x["signal"] != "NO SIGNAL"
    ]

if all_rows:
    all_table = pd.DataFrame([
        {
            "Asset": x["asset"],
            "Category": x["category"],
            "Signal": x["signal"],
            "Price": x["price"],
            "Score": x["score"],
            "RSI": x["rsi"],
            "ADX": x["adx"],
            "ATR %": (
                float(x["candles"].iloc[-1]["atr_pct"])
                if pd.notna(
                    x["candles"].iloc[-1]["atr_pct"]
                )
                else np.nan
            ),
            "Source": x["source"],
        }
        for x in all_rows
    ])

    st.dataframe(
        all_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Gösterilecek geçerli analiz yok.")

# Detailed chart
st.subheader("🔬 Varlık detayları")

if valid:
    selected_name = st.selectbox(
        "Varlık seç",
        options=[x["asset"] for x in valid],
    )

    selected = next(
        x for x in valid
        if x["asset"] == selected_name
    )

    candles = selected["candles"]
    last = candles.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Son kapanış",
        f"{last['close']:.8g}",
    )

    m2.metric(
        "RSI",
        f"{last['rsi']:.2f}",
    )

    m3.metric(
        "ADX",
        f"{last['adx']:.2f}",
    )

    m4.metric(
        "Gösterge skoru",
        f"{selected['score']}%",
    )

    if selected["signal"] == "BUY":
        st.success("BUY — teknik göstergeler yükseliş yönünde uyumlu.")
    elif selected["signal"] == "SELL":
        st.error("SELL — teknik göstergeler düşüş yönünde uyumlu.")
    else:
        st.warning("NO SIGNAL — yeterli gösterge uyumu yok.")

    st.write("**Veri kaynağı:** " + selected["source"])
    st.write("**Gerekçe:** " + selected["reason"])

    chart = candles.set_index("time")[
        ["close", "ema20", "ema50"]
    ].rename(
        columns={
            "close": "Close",
            "ema20": "EMA 20",
            "ema50": "EMA 50",
        }
    )

    st.markdown("#### Fiyat ve EMA")
    st.line_chart(chart)

    bb_chart = candles.set_index("time")[
        ["close", "bb_upper", "bb_mid", "bb_lower"]
    ].rename(
        columns={
            "close": "Close",
            "bb_upper": "BB Upper",
            "bb_mid": "BB Middle",
            "bb_lower": "BB Lower",
        }
    )

    st.markdown("#### Bollinger Bands")
    st.line_chart(bb_chart)

    st.markdown("#### RSI")
    st.line_chart(
        candles.set_index("time")[["rsi"]]
    )

    st.markdown("#### MACD")
    st.line_chart(
        candles.set_index("time")[
            ["macd", "macd_signal", "macd_hist"]
        ]
    )

# Unavailable assets
st.subheader("⚠️ Verisi alınamayan varlıklar")

if unavailable:
    unavailable_table = pd.DataFrame([
        {
            "Asset": x.get("asset", "Unknown"),
            "Reason": x.get("error", "Unknown error"),
        }
        for x in unavailable
    ])

    st.dataframe(
        unavailable_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("Bu taramada veri hatası kaydedilmedi.")

# CSV download
st.subheader("📥 CSV dışa aktar")

if results:
    export_rows = []

    for x in results:
        export_rows.append({
            "asset": x.get("asset"),
            "symbol": x.get("symbol"),
            "signal": x.get("signal"),
            "price": x.get("price"),
            "score": x.get("score"),
            "buy_score": x.get("buy_score"),
            "sell_score": x.get("sell_score"),
            "rsi": x.get("rsi"),
            "adx": x.get("adx"),
            "source": x.get("source"),
            "reason": x.get("reason"),
            "error": x.get("error"),
        })

    export_df = pd.DataFrame(export_rows)

    st.download_button(
        "CSV dosyasını indir",
        data=export_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="velora_1m_scan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Yalnızca bilgilendirme amaçlıdır. Bu uygulama işlem "
    "gerçekleştirmez ve Binomo'ya bağlanmaz. Kraken ve "
    "Twelve Data fiyatları Binomo OTC fiyatlarıyla aynı "
    "olmayabilir. Gerçek Binomo OTC verisi mevcut değildir. "
    "Gösterge skoru kazanma olasılığı değildir. Kâr garantisi yoktur."
)
