import os
import io
import time
import math
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Spot AI Scanner"
TIMEFRAME = "15m"

DEFAULT_CANDLE_TARGET = 500_000
DEFAULT_SYMBOL_LIMIT = 20

FUTURE_BARS = 12

REQUEST_TIMEOUT = 30

BINANCE_BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
)

BINANCE_EXCHANGE_INFO = "/api/v3/exchangeInfo"
BINANCE_KLINES = "/api/v3/klines"

SUPPORTED_QUOTE_ASSETS = [
    "USDT",
    "USDC",
    "FDUSD",
    "BTC",
    "ETH",
    "BNB"
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Spot AI Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #888;
        margin-bottom: 20px;
    }

    .buy-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0,180,100,0.35);
        background: rgba(0,180,100,0.08);
    }

    .sell-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(220,60,60,0.35);
        background: rgba(220,60,60,0.08);
    }

    .metric-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(120,120,120,0.25);
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">📊 Spot AI Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">15m | 500K Target | MLP AI | No Order Execution</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "candle_cache" not in st.session_state:
    st.session_state.candle_cache = {}

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "last_source" not in st.session_state:
    st.session_state.last_source = None

if "scan_running" not in st.session_state:
    st.session_state.scan_running = False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

data_source = st.sidebar.selectbox(
    "Data Source",
    [
        "CSV / Uploaded Data",
        "CCXT Exchange",
        "Binance REST"
    ]
)

candle_target = st.sidebar.selectbox(
    "Candle Target",
    [
        10_000,
        50_000,
        100_000,
        250_000,
        500_000
    ],
    index=4
)

symbol_limit = st.sidebar.slider(
    "Maximum Symbols",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

future_bars = st.sidebar.slider(
    "Future Bars",
    min_value=3,
    max_value=48,
    value=12
)

train_limit = st.sidebar.slider(
    "Training Rows",
    min_value=2_000,
    max_value=30_000,
    value=10_000,
    step=1
)

min_confidence = st.sidebar.slider(
    "Minimum AI Confidence",
    min_value=0.30,
    max_value=0.95,
    value=0.55,
    step=0.01
)

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh Every 15 Minutes",
    value=True
)

st.sidebar.divider()

st.sidebar.info(
    "Bu uygulama emir göndermez. "
    "Sadece piyasa verisini analiz eder ve AI sinyali üretir."
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": "Spot-AI-Scanner/1.0",
            "Accept": "application/json"
        }
    )

    return session


HTTP = get_http_session()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:
        return float(value)

    except Exception:
        return default


def utc_now():

    return datetime.now(timezone.utc)


def normalize_symbol(symbol):

    return (
        str(symbol)
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )


def timeframe_to_minutes(timeframe):

    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }

    return mapping.get(timeframe, 15)


# ============================================================
# BINANCE ERROR
# ============================================================

class BinanceRestrictedError(Exception):
    pass


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_get(endpoint, params=None):

    url = BINANCE_BASE_URL.rstrip("/") + endpoint

    try:

        response = HTTP.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Network error: {exc}"
        )

    if response.status_code == 451:

        raise BinanceRestrictedError(
            "HTTP 451: Binance API bu çalışma ortamından "
            "erişime izin vermiyor."
        )

    if response.status_code == 429:

        raise RuntimeError(
            "HTTP 429: Binance rate limit."
        )

    if response.status_code >= 400:

        text = response.text[:500]

        raise RuntimeError(
            f"HTTP {response.status_code}: {text}"
        )

    try:

        return response.json()

    except Exception:

        raise RuntimeError(
            "API JSON cevabı okunamadı."
        )


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(ttl=900)
def get_binance_symbols():

    data = binance_get(
        BINANCE_EXCHANGE_INFO
    )

    symbols = []

    for item in data.get("symbols", []):

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        quote = item.get("quoteAsset", "")

        if quote not in SUPPORTED_QUOTE_ASSETS:
            continue

        symbol = item.get("symbol")

        if symbol:
            symbols.append(symbol)

    return sorted(symbols)


# ============================================================
# BINANCE KLINES
# ============================================================

def get_binance_klines(
    symbol,
    limit=1000,
    end_time=None
):

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "limit": min(int(limit), 1000)
    }

    if end_time is not None:
        params["endTime"] = int(end_time)

    data = binance_get(
        BINANCE_KLINES,
        params=params
    )

    if not data:
        return pd.DataFrame()

    rows = []

    for x in data:

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(x[0]),
                    unit="ms",
                    utc=True
                ),
                "open": safe_float(x[1]),
                "high": safe_float(x[2]),
                "low": safe_float(x[3]),
                "close": safe_float(x[4]),
                "volume": safe_float(x[5]),
            }
        )

    df = pd.DataFrame(rows)

    return clean_ohlcv(df)


# ============================================================
# DOWNLOAD LARGE HISTORY
# ============================================================

def download_binance_history(
    symbol,
    target=500_000,
    progress_callback=None
):

    target = int(target)

    all_chunks = []

    end_time = None

    collected = 0

    loops = math.ceil(target / 1000)

    for i in range(loops):

        try:

            chunk = get_binance_klines(
                symbol=symbol,
                limit=1000,
                end_time=end_time
            )

        except Exception:

            raise

        if chunk.empty:
            break

        all_chunks.append(chunk)

        collected += len(chunk)

        oldest = chunk["timestamp"].min()

        end_time = int(
            oldest.timestamp() * 1000
        ) - 1

        if progress_callback:

            progress_callback(
                min(collected / target, 1.0)
            )

        if len(chunk) < 1000:
            break

        if len(all_chunks) % 10 == 0:

            time.sleep(0.05)

    if not all_chunks:

        return pd.DataFrame()

    df = pd.concat(
        all_chunks,
        ignore_index=True
    )

    df = clean_ohlcv(df)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .tail(target)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CLEAN OHLCV
# ============================================================

def clean_ohlcv(df):

    if df is None:
        return pd.DataFrame()

    if df.empty:
        return df

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Eksik OHLCV kolonları: "
            + ", ".join(missing)
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    result = result.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    result = (
        result
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# CSV LOADING
# ============================================================

def read_uploaded_csv(uploaded_file):

    raw = uploaded_file.read()

    try:

        df = pd.read_csv(
            io.BytesIO(raw)
        )

    except Exception as exc:

        raise ValueError(
            f"CSV okunamadı: {exc}"
        )

    column_map = {}

    for col in df.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        column_map[normalized] = col

    rename = {}

    aliases = {

        "timestamp": [
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time"
        ],

        "open": [
            "open",
            "o"
        ],

        "high": [
            "high",
            "h"
        ],

        "low": [
            "low",
            "l"
        ],

        "close": [
            "close",
            "c"
        ],

        "volume": [
            "volume",
            "vol",
            "v"
        ]
    }

    for target, names in aliases.items():

        for name in names:

            if name in column_map:

                rename[column_map[name]] = target

                break

    df = df.rename(
        columns=rename
    )

    return clean_ohlcv(df)


# ============================================================
# CCXT
# ============================================================

@st.cache_resource
def get_ccxt():

    try:

        import ccxt

        return ccxt

    except ImportError:

        return None


def get_ccxt_exchange(exchange_id):

    ccxt = get_ccxt()

    if ccxt is None:

        raise RuntimeError(
            "CCXT kurulu değil. "
            "requirements.txt içine ccxt ekleyin."
        )

    if not hasattr(ccxt, exchange_id):

        raise ValueError(
            f"CCXT exchange bulunamadı: {exchange_id}"
        )

    exchange_class = getattr(
        ccxt,
        exchange_id
    )

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "timeout": 30000
        }
    )

    return exchange


def ccxt_symbols(exchange_id):

    exchange = get_ccxt_exchange(
        exchange_id
    )

    markets = exchange.load_markets()

    result = []

    for symbol, market in markets.items():

        try:

            if market.get("spot") is not True:
                continue

            if market.get("active") is False:
                continue

            quote = market.get("quote")

            if quote not in SUPPORTED_QUOTE_ASSETS:
                continue

            result.append(symbol)

        except Exception:
            continue

    return sorted(result)


def ccxt_history(
    exchange_id,
    symbol,
    target=500_000
):

    exchange = get_ccxt_exchange(
        exchange_id
    )

    all_rows = []

    limit = 1000

    since = None

    timeframe = TIMEFRAME

    estimated_ms = (
        timeframe_to_minutes(timeframe)
        * 60
        * 1000
    )

    loops = math.ceil(
        target / limit
    )

    for _ in range(loops):

        rows = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < limit:
            break

        last_ts = rows[-1][0]

        since = last_ts + estimated_ms

        if len(all_rows) >= target:
            break

    if not all_rows:

        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    return clean_ohlcv(
        df.tail(target)
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / (
        avg_loss + 1e-12
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - prev_close
    ).abs()

    tr3 = (
        low - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def macd(series):

    fast = ema(series, 12)

    slow = ema(series, 26)

    line = fast - slow

    signal = ema(line, 9)

    hist = line - signal

    return line, signal, hist


def bollinger(series, period=20):

    mid = series.rolling(
        period
    ).mean()

    std = series.rolling(
        period
    ).std()

    upper = mid + 2 * std

    lower = mid - 2 * std

    return mid, upper, lower


def adx(df, period=14):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move)
        & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move)
        & (down_move > 0),
        down_move,
        0
    )

    tr1 = high - low

    tr2 = (
        high - close.shift(1)
    ).abs()

    tr3 = (
        low - close.shift(1)
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr_value = tr.rolling(
        period
    ).mean()

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index
        ).rolling(period).mean()
        / (atr_value + 1e-12)
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index
        ).rolling(period).mean()
        / (atr_value + 1e-12)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di + 1e-12)
    )

    return dx.rolling(
        period
    ).mean()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    data = df.copy()

    close = data["close"]

    high = data["high"]

    low = data["low"]

    volume = data["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    data["ret_1"] = close.pct_change(1)

    data["ret_3"] = close.pct_change(3)

    data["ret_6"] = close.pct_change(6)

    data["ret_12"] = close.pct_change(12)

    data["ret_24"] = close.pct_change(24)

    data["ret_48"] = close.pct_change(48)

    data["ret_96"] = close.pct_change(96)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_periods = [
        5,
        20,
        50,
        100,
        200,
        500,
        800
    ]

    for p in ema_periods:

        e = ema(
            close,
            p
        )

        data[f"ema_{p}"] = e

        data[
            f"ema_dist_{p}"
        ] = (
            close / (e + 1e-12)
            - 1
        )

    data["ema_5_20"] = (
        data["ema_5"]
        / (data["ema_20"] + 1e-12)
        - 1
    )

    data["ema_20_50"] = (
        data["ema_20"]
        / (data["ema_50"] + 1e-12)
        - 1
    )

    data["ema_50_200"] = (
        data["ema_50"]
        / (data["ema_200"] + 1e-12)
        - 1
    )

    data["ema_200_800"] = (
        data["ema_200"]
        / (data["ema_800"] + 1e-12)
        - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    for p in [7, 14, 21]:

        data[
            f"rsi_{p}"
        ] = rsi(
            close,
            p
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_line, signal, hist = macd(
        close
    )

    data["macd"] = macd_line

    data["macd_signal"] = signal

    data["macd_hist"] = hist

    data["macd_norm"] = (
        macd_line
        / (close + 1e-12)
    )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb_mid, bb_upper, bb_lower = (
        bollinger(close, 20)
    )

    data["bb_mid"] = bb_mid

    data["bb_upper"] = bb_upper

    data["bb_lower"] = bb_lower

    data["bb_width"] = (
        (bb_upper - bb_lower)
        / (bb_mid + 1e-12)
    )

    data["bb_position"] = (
        (close - bb_lower)
        / (
            bb_upper
            - bb_lower
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["atr"] = atr(
        data,
        14
    )

    data["atr_pct"] = (
        data["atr"]
        / (close + 1e-12)
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    data["adx"] = adx(
        data,
        14
    )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    lowest = low.rolling(
        14
    ).min()

    highest = high.rolling(
        14
    ).max()

    data["stoch_k"] = (
        100
        * (close - lowest)
        / (
            highest
            - lowest
            + 1e-12
        )
    )

    data["stoch_d"] = (
        data["stoch_k"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # CANDLE GEOMETRY
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    body = (
        close - data["open"]
    )

    data["body_pct"] = (
        body
        / (close + 1e-12)
    )

    data["range_pct"] = (
        candle_range
        / (close + 1e-12)
    )

    data["upper_wick"] = (
        high
        - np.maximum(
            data["open"],
            close
        )
    )

    data["lower_wick"] = (
        np.minimum(
            data["open"],
            close
        )
        - low
    )

    data["body_to_range"] = (
        body.abs()
        / (
            candle_range
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = volume.rolling(
        20
    ).mean()

    volume_std = volume.rolling(
        20
    ).std()

    data["volume_ratio"] = (
        volume
        / (volume_mean + 1e-12)
    )

    data["volume_z"] = (
        volume
        - volume_mean
    ) / (
        volume_std + 1e-12
    )

    data["volume_change"] = (
        volume.pct_change()
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    data["volatility_12"] = (
        data["ret_1"]
        .rolling(12)
        .std()
    )

    data["volatility_24"] = (
        data["ret_1"]
        .rolling(24)
        .std()
    )

    data["volatility_48"] = (
        data["ret_1"]
        .rolling(48)
        .std()
    )

    data["volatility_96"] = (
        data["ret_1"]
        .rolling(96)
        .std()
    )

    # --------------------------------------------------------
    # HIGH / LOW DISTANCE
    # --------------------------------------------------------

    rolling_high = high.rolling(
        48
    ).max()

    rolling_low = low.rolling(
        48
    ).min()

    data["distance_high_48"] = (
        close
        / (rolling_high + 1e-12)
        - 1
    )

    data["distance_low_48"] = (
        close
        / (rolling_low + 1e-12)
        - 1
    )

    rolling_high_200 = high.rolling(
        200
    ).max()

    rolling_low_200 = low.rolling(
        200
    ).min()

    data["distance_high_200"] = (
        close
        / (rolling_high_200 + 1e-12)
        - 1
    )

    data["distance_low_200"] = (
        close
        / (rolling_low_200 + 1e-12)
        - 1
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    rolling_max = close.cummax()

    data["drawdown"] = (
        close
        / (rolling_max + 1e-12)
        - 1
    )

    # --------------------------------------------------------
    # PRICE Z-SCORE
    # --------------------------------------------------------

    mean_50 = close.rolling(
        50
    ).mean()

    std_50 = close.rolling(
        50
    ).std()

    data["price_z_50"] = (
        close
        - mean_50
    ) / (
        std_50 + 1e-12
    )

    mean_200 = close.rolling(
        200
    ).mean()

    std_200 = close.rolling(
        200
    ).std()

    data["price_z_200"] = (
        close
        - mean_200
    ) / (
        std_200 + 1e-12
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return data


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns():

    return [

        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_48",
        "ret_96",

        "ema_dist_5",
        "ema_dist_20",
        "ema_dist_50",
        "ema_dist_100",
        "ema_dist_200",
        "ema_dist_500",
        "ema_dist_800",

        "ema_5_20",
        "ema_20_50",
        "ema_50_200",
        "ema_200_800",

        "rsi_7",
        "rsi_14",
        "rsi_21",

        "macd_norm",
        "macd_signal",
        "macd_hist",

        "bb_width",
        "bb_position",

        "atr_pct",
        "adx",

        "stoch_k",
        "stoch_d",

        "body_pct",
        "range_pct",
        "upper_wick",
        "lower_wick",
        "body_to_range",

        "volume_ratio",
        "volume_z",
        "volume_change",

        "volatility_12",
        "volatility_24",
        "volatility_48",
        "volatility_96",

        "distance_high_48",
        "distance_low_48",
        "distance_high_200",
        "distance_low_200",

        "drawdown",

        "price_z_50",
        "price_z_200"
    ]


# ============================================================
# TARGET CREATION
# ============================================================

def create_targets(
    feature_df,
    future_bars=12
):

    data = feature_df.copy()

    future_return = (
        data["close"]
        .shift(-future_bars)
        / data["close"]
        - 1
    )

    data["future_return"] = (
        future_return
    )

    volatility = (
        data["volatility_24"]
        .fillna(
            data["ret_1"].std()
        )
    )

    threshold = (
        volatility
        * math.sqrt(
            max(future_bars, 1)
        )
    )

    threshold = threshold.clip(
        lower=0.002
    )

    data["target"] = 1

    data.loc[
        future_return > threshold,
        "target"
    ] = 2

    data.loc[
        future_return < -threshold,
        "target"
    ] = 0

    return data


# ============================================================
# MODEL
# ============================================================

def create_classifier():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(
                        128,
                        64,
                        32
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42
                )
            )
        ]
    )


def create_regressor():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=1.0
                )
            )
        ]
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_models(
    df,
    future_bars=12,
    train_limit=10_000
):

    features = build_features(
        df
    )

    features = create_targets(
        features,
        future_bars
    )

    columns = get_feature_columns()

    missing = [
        c
        for c in columns
        if c not in features.columns
    ]

    if missing:

        raise ValueError(
            "Eksik feature: "
            + ", ".join(missing)
        )

    usable = features.dropna(
        subset=columns
        + [
            "target",
            "future_return"
        ]
    ).copy()

    if len(usable) < 500:

        raise ValueError(
            f"Model eğitimi için yeterli veri yok. "
            f"Geçerli satır: {len(usable)}"
        )

    usable = usable.tail(
        min(
            train_limit,
            len(usable)
        )
    )

    X = usable[
        columns
    ].astype(float)

    y = usable[
        "target"
    ].astype(int)

    future_y = usable[
        "future_return"
    ].astype(float)

    unique_classes = sorted(
        y.unique()
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "Target sadece tek sınıf içeriyor."
        )

    classifier = create_classifier()

    regressor = create_regressor()

    classifier.fit(
        X,
        y
    )

    regressor.fit(
        X,
        future_y
    )

    predictions = classifier.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    return (
        classifier,
        regressor,
        features,
        columns,
        accuracy
    )


# ============================================================
# PREDICT
# ============================================================

def predict_latest(
    classifier,
    regressor,
    features,
    columns
):

    valid = features.dropna(
        subset=columns
    )

    if valid.empty:

        raise ValueError(
            "Son satırda yeterli feature yok."
        )

    latest = valid.iloc[-1]

    X = pd.DataFrame(
        [
            latest[columns]
            .astype(float)
        ]
    )

    probabilities = (
        classifier.predict_proba(X)[0]
    )

    classes = (
        classifier.classes_
    )

    prob_map = {
        int(c): float(p)
        for c, p in zip(
            classes,
            probabilities
        )
    }

    p_sell = prob_map.get(
        0,
        0.0
    )

    p_hold = prob_map.get(
        1,
        0.0
    )

    p_buy = prob_map.get(
        2,
        0.0
    )

    prediction = int(
        classifier.predict(X)[0]
    )

    expected_return = float(
        regressor.predict(X)[0]
    )

    if prediction == 2:

        signal = "BUY"

        confidence = p_buy

    elif prediction == 0:

        signal = "SELL"

        confidence = p_sell

    else:

        signal = "HOLD"

        confidence = p_hold

    close = float(
        latest["close"]
    )

    rsi14 = safe_float(
        latest.get(
            "rsi_14"
        )
    )

    bb_position = safe_float(
        latest.get(
            "bb_position"
        )
    )

    atr_pct = safe_float(
        latest.get(
            "atr_pct"
        )
    )

    adx_value = safe_float(
        latest.get(
            "adx"
        )
    )

    volume_ratio = safe_float(
        latest.get(
            "volume_ratio"
        )
    )

    momentum = safe_float(
        latest.get(
            "ret_24"
        )
    )

    trend = safe_float(
        latest.get(
            "ema_20_50"
        )
    )

    volatility = safe_float(
        latest.get(
            "volatility_24"
        )
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    direction_score = (
        p_buy - p_sell
    )

    momentum_score = np.tanh(
        momentum * 20
    )

    trend_score = np.tanh(
        trend * 20
    )

    volume_score = np.tanh(
        (volume_ratio - 1)
        / 2
    )

    if np.isnan(
        volatility
    ):

        risk_score = 0

    else:

        risk_score = (
            1
            - min(
                max(
                    volatility * 30,
                    0
                ),
                1
            )
        )

    score = (
        50
        + direction_score * 25
        + momentum_score * 10
        + trend_score * 7
        + volume_score * 4
        + risk_score * 4
    )

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    return {

        "signal": signal,

        "confidence": float(
            confidence
        ),

        "p_sell": float(
            p_sell
        ),

        "p_hold": float(
            p_hold
        ),

        "p_buy": float(
            p_buy
        ),

        "expected_return": (
            expected_return
        ),

        "price": close,

        "rsi14": rsi14,

        "bb_position": bb_position,

        "atr_pct": atr_pct,

        "adx": adx_value,

        "volume_ratio": volume_ratio,

        "momentum": momentum,

        "trend": trend,

        "volatility": volatility,

        "score": score
    }


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    df,
    future_bars,
    train_limit
):

    start = time.time()

    (
        classifier,
        regressor,
        features,
        columns,
        accuracy
    ) = train_models(
        df,
        future_bars=future_bars,
        train_limit=train_limit
    )

    prediction = predict_latest(
        classifier,
        regressor,
        features,
        columns
    )

    result = {
        "symbol": symbol,
        "signal": prediction["signal"],
        "score": prediction["score"],
        "confidence": prediction["confidence"],
        "expected_return": prediction["expected_return"],
        "price": prediction["price"],
        "p_buy": prediction["p_buy"],
        "p_hold": prediction["p_hold"],
        "p_sell": prediction["p_sell"],
        "rsi14": prediction["rsi14"],
        "bb_position": prediction["bb_position"],
        "atr_pct": prediction["atr_pct"],
        "adx": prediction["adx"],
        "volume_ratio": prediction["volume_ratio"],
        "momentum": prediction["momentum"],
        "trend": prediction["trend"],
        "volatility": prediction["volatility"],
        "training_accuracy": accuracy,
        "candles": len(df),
        "timestamp": df["timestamp"].iloc[-1],
        "processing_seconds": time.time() - start
    }

    return result


# ============================================================
# CSV RESULT EXPORT
# ============================================================

def dataframe_to_csv(df):

    if df is None or df.empty:

        return b""

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# DATA SOURCE UI
# ============================================================

uploaded_file = None

exchange_id = None

if data_source == "CSV / Uploaded Data":

    uploaded_file = st.sidebar.file_uploader(
        "OHLCV CSV yükle",
        type=[
            "csv"
        ]
    )

    st.sidebar.caption(
        "CSV kolonları: timestamp, open, high, low, close, volume"
    )


elif data_source == "CCXT Exchange":

    exchange_id = st.sidebar.selectbox(
        "Exchange",
        [
            "kraken",
            "coinbase",
            "kucoin",
            "okx",
            "bybit",
            "bitget"
        ]
    )


# ============================================================
# MANUAL SCAN
# ============================================================

manual_scan = st.sidebar.button(
    "🔄 Şimdi Tara",
    use_container_width=True
)


# ============================================================
# DATA LOADER
# ============================================================

def load_data_for_symbol(
    symbol,
    source,
    target,
    uploaded=None,
    exchange=None
):

    if source == "CSV / Uploaded Data":

        if uploaded is None:

            raise ValueError(
                "CSV yüklenmedi."
            )

        return read_uploaded_csv(
            uploaded
        ).tail(target)

    if source == "CCXT Exchange":

        return ccxt_history(
            exchange,
            symbol,
            target
        )

    if source == "Binance REST":

        return download_binance_history(
            symbol,
            target
        )

    raise ValueError(
        "Bilinmeyen veri kaynağı."
    )


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

def discover_symbols(
    source,
    limit,
    uploaded=None,
    exchange=None
):

    if source == "CSV / Uploaded Data":

        return [
            "UPLOADED_DATA"
        ]

    if source == "CCXT Exchange":

        symbols = ccxt_symbols(
            exchange
        )

        return symbols[:limit]

    if source == "Binance REST":

        symbols = get_binance_symbols()

        return symbols[:limit]

    return []


# ============================================================
# MAIN SCAN
# ============================================================

def run_scan():

    st.session_state.scan_errors = []

    st.session_state.scan_running = True

    results = []

    progress = st.progress(
        0
    )

    status = st.empty()

    try:

        status.info(
            "Semboller keşfediliyor..."
        )

        symbols = discover_symbols(
            data_source,
            symbol_limit,
            uploaded_file,
            exchange_id
        )

        if not symbols:

            raise ValueError(
                "Analiz edilecek sembol bulunamadı."
            )

        total = len(symbols)

        for index, symbol in enumerate(
            symbols
        ):

            status.info(
                f"Analiz: {symbol} "
                f"({index + 1}/{total})"
            )

            try:

                if (
                    data_source
                    == "CSV / Uploaded Data"
                ):

                    df = load_data_for_symbol(
                        symbol,
                        data_source,
                        candle_target,
                        uploaded_file,
                        exchange_id
                    )

                else:

                    cache_key = (
                        f"{data_source}:"
                        f"{exchange_id}:"
                        f"{symbol}"
                    )

                    cached = (
                        st.session_state
                        .candle_cache
                        .get(cache_key)
                    )

                    if (
                        cached is not None
                        and len(cached)
                        >= candle_target
                    ):

                        df = cached.tail(
                            candle_target
                        )

                    else:

                        df = load_data_for_symbol(
                            symbol,
                            data_source,
                            candle_target,
                            uploaded_file,
                            exchange_id
                        )

                    if not df.empty:

                        st.session_state.candle_cache[
                            cache_key
                        ] = df.tail(
                            candle_target
                        )

                if df.empty:

                    raise ValueError(
                        "OHLCV verisi boş."
                    )

                if len(df) < 500:

                    raise ValueError(
                        f"Yetersiz candle: "
                        f"{len(df)}"
                    )

                result = analyze_symbol(
                    symbol,
                    df,
                    future_bars,
                    train_limit
                )

                results.append(
                    result
                )

            except BinanceRestrictedError as exc:

                st.session_state.scan_errors.append(
                    {
                        "symbol": symbol,
                        "error": str(exc)
                    }
                )

                raise

            except Exception as exc:

                st.session_state.scan_errors.append(
                    {
                        "symbol": symbol,
                        "error": str(exc)
                    }
                )

            progress.progress(
                (index + 1)
                / total
            )

        result_df = pd.DataFrame(
            results
        )

        if not result_df.empty:

            result_df = (
                result_df
                .sort_values(
                    [
                        "signal",
                        "score"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
                .reset_index(drop=True)
            )

        st.session_state.scan_results = (
            result_df
        )

        st.session_state.last_scan = (
            utc_now()
        )

        st.session_state.last_source = (
            data_source
        )

    except BinanceRestrictedError as exc:

        st.error(
            "HTTP 451: Binance API bu çalışma "
            "ortamından erişime izin vermiyor."
        )

        st.info(
            "Bu bir Python/Streamlit hatası değildir. "
            "Data Source bölümünden CCXT veya CSV "
            "kaynağı kullanabilirsiniz."
        )

    except Exception as exc:

        st.error(
            f"Tarama hatası: {exc}"
        )

        st.session_state.scan_errors.append(
            {
                "symbol": "SYSTEM",
                "error": str(exc)
            }
        )

    finally:

        st.session_state.scan_running = False

        progress.empty()

        status.empty()


# ============================================================
# RUN CONDITIONS
# ============================================================

should_scan = False

if manual_scan:

    should_scan = True

if (
    auto_refresh
    and not st.session_state.scan_running
):

    should_scan = True


# ============================================================
# STREAMLIT FRAGMENT
# ============================================================

def scanner_fragment():

    if should_scan:

        run_scan()


try:

    fragment = st.fragment

except AttributeError:

    fragment = None


if fragment is not None:

    @st.fragment(
        run_every="15m"
        if auto_refresh
        else None
    )
    def auto_scanner():

        if should_scan:

            run_scan()

    auto_scanner()

else:

    if should_scan:

        run_scan()


# ============================================================
# CURRENT STATUS
# ============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Data Source",
        st.session_state.last_source
        or data_source
    )

with col2:

    st.metric(
        "Candle Target",
        f"{candle_target:,}"
    )

with col3:

    st.metric(
        "Symbols",
        symbol_limit
    )

with col4:

    if st.session_state.last_scan:

        st.metric(
            "Last Scan",
            st.session_state.last_scan.strftime(
                "%H:%M:%S"
            )
        )

    else:

        st.metric(
            "Last Scan",
            "-"
        )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.scan_results
)

if results.empty:

    st.warning(
        "Henüz sonuç yok."
    )

    st.markdown(
        """
        ### Veri kaynağı seçin

        **CSV / Uploaded Data**
        kullanarak OHLCV CSV yükleyebilirsiniz.

        **CCXT Exchange**
        seçerek desteklenen başka bir exchange
        üzerinden Spot verisi alabilirsiniz.

        **Binance REST**
        seçerseniz Binance HTTP 451 döndürdüğünde
        uygulama hata mesajını gösterecektir.
        """
    )

else:

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    buy = results[
        results["signal"] == "BUY"
    ].copy()

    sell = results[
        results["signal"] == "SELL"
    ].copy()

    hold = results[
        results["signal"] == "HOLD"
    ].copy()

    buy = buy[
        buy["confidence"]
        >= min_confidence
    ]

    sell = sell[
        sell["confidence"]
        >= min_confidence
    ]

    buy = buy.sort_values(
        [
            "score",
            "confidence"
        ],
        ascending=False
    ).head(10)

    sell = sell.sort_values(
        [
            "score",
            "confidence"
        ],
        ascending=[
            True,
            False
        ]
    ).head(10)

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    st.subheader(
        "🟢 Top 10 BUY"
    )

    if buy.empty:

        st.info(
            "Minimum confidence değerini geçen BUY yok."
        )

    else:

        buy_display = buy.copy()

        buy_display[
            "confidence"
        ] = (
            buy_display["confidence"]
            * 100
        ).round(2)

        buy_display[
            "expected_return"
        ] = (
            buy_display["expected_return"]
            * 100
        ).round(3)

        buy_display[
            "p_buy"
        ] = (
            buy_display["p_buy"]
            * 100
        ).round(2)

        buy_display[
            "score"
        ] = buy_display[
            "score"
        ].round(2)

        buy_display[
            "rsi14"
        ] = buy_display[
            "rsi14"
        ].round(2)

        st.dataframe(
            buy_display[
                [
                    "symbol",
                    "score",
                    "confidence",
                    "expected_return",
                    "p_buy",
                    "rsi14",
                    "adx",
                    "volume_ratio",
                    "price",
                    "candles"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    st.subheader(
        "🔴 Top 10 SELL"
    )

    if sell.empty:

        st.info(
            "Minimum confidence değerini geçen SELL yok."
        )

    else:

        sell_display = sell.copy()

        sell_display[
            "confidence"
        ] = (
            sell_display["confidence"]
            * 100
        ).round(2)

        sell_display[
            "expected_return"
        ] = (
            sell_display["expected_return"]
            * 100
        ).round(3)

        sell_display[
            "p_sell"
        ] = (
            sell_display["p_sell"]
            * 100
        ).round(2)

        sell_display[
            "score"
        ] = sell_display[
            "score"
        ].round(2)

        sell_display[
            "rsi14"
        ] = sell_display[
            "rsi14"
        ].round(2)

        st.dataframe(
            sell_display[
                [
                    "symbol",
                    "score",
                    "confidence",
                    "expected_return",
                    "p_sell",
                    "rsi14",
                    "adx",
                    "volume_ratio",
                    "price",
                    "candles"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    with st.expander(
        "📋 Tüm AI Sonuçları"
    ):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # DETAIL
    # --------------------------------------------------------

    st.subheader(
        "🔎 Symbol Detail"
    )

    selected_symbol = st.selectbox(
        "Symbol",
        results["symbol"].tolist()
    )

    selected = results[
        results["symbol"]
        == selected_symbol
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Signal",
            selected["signal"]
        )

    with c2:

        st.metric(
            "AI Confidence",
            f"{selected['confidence'] * 100:.2f}%"
        )

    with c3:

        st.metric(
            "Expected Return",
            f"{selected['expected_return'] * 100:.3f}%"
        )

    with c4:

        st.metric(
            "AI Score",
            f"{selected['score']:.2f}/100"
        )

    detail_data = pd.DataFrame(
        {
            "Metric": [
                "Price",
                "BUY Probability",
                "HOLD Probability",
                "SELL Probability",
                "RSI 14",
                "Bollinger Position",
                "ATR %",
                "ADX",
                "Volume Ratio",
                "Momentum",
                "Trend",
                "Volatility",
                "Training Accuracy",
                "Candles",
                "Processing Seconds"
            ],

            "Value": [
                selected["price"],
                f"{selected['p_buy'] * 100:.2f}%",
                f"{selected['p_hold'] * 100:.2f}%",
                f"{selected['p_sell'] * 100:.2f}%",
                selected["rsi14"],
                selected["bb_position"],
                selected["atr_pct"],
                selected["adx"],
                selected["volume_ratio"],
                selected["momentum"],
                selected["trend"],
                selected["volatility"],
                f"{selected['training_accuracy'] * 100:.2f}%",
                selected["candles"],
                selected["processing_seconds"]
            ]
        }
    )

    st.dataframe(
        detail_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

if not results.empty:

    st.divider()

    st.subheader(
        "💾 Export"
    )

    csv_bytes = dataframe_to_csv(
        results
    )

    st.download_button(
        "⬇️ Download AI Results CSV",
        data=csv_bytes,
        file_name=(
            "spot_ai_results.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# CACHE STATUS
# ============================================================

with st.expander(
    "🗄️ Candle Cache"
):

    if st.session_state.candle_cache:

        cache_rows = []

        for key, df in (
            st.session_state
            .candle_cache
            .items()
        ):

            if df is None or df.empty:
                continue

            cache_rows.append(
                {
                    "key": key,
                    "candles": len(df),
                    "first": str(
                        df["timestamp"].iloc[0]
                    ),
                    "last": str(
                        df["timestamp"].iloc[-1]
                    )
                }
            )

        if cache_rows:

            st.dataframe(
                pd.DataFrame(cache_rows),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Cache boş."
            )

    else:

        st.info(
            "Cache henüz oluşturulmadı."
        )


# ============================================================
# ERROR DETAILS
# ============================================================

if st.session_state.scan_errors:

    st.divider()

    st.subheader(
        "⚠️ Hata Detayları"
    )

    error_df = pd.DataFrame(
        st.session_state.scan_errors
    )

    st.dataframe(
        error_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Scanner | 15m | 500K target | "
    "MLP AI | No Order Execution"
)

st.caption(
    "Model eğitim amacıyla teknik göstergelerden "
    "BUY / HOLD / SELL sınıflandırması üretir. "
    "Bu uygulama yatırım tavsiyesi değildir."
)
