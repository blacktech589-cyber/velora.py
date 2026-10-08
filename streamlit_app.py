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
# APPLICATION CONFIG
# ============================================================

APP_NAME = "Spot AI Scanner"

TIMEFRAME = "15m"

DEFAULT_CANDLE_TARGET = 500_000

DEFAULT_SYMBOL_LIMIT = 10

DEFAULT_FUTURE_BARS = 12

DEFAULT_TRAIN_LIMIT = 10_000

REQUEST_TIMEOUT = 30

BINANCE_BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
)

BINANCE_EXCHANGE_INFO = "/api/v3/exchangeInfo"

BINANCE_KLINES = "/api/v3/klines"

SUPPORTED_QUOTES = [
    "USDT",
    "USDC",
    "FDUSD",
    "BTC",
    "ETH",
    "BNB"
]


# ============================================================
# PAGE CONFIG
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

    .scanner-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .scanner-subtitle {
        color: #888888;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .buy-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0, 180, 100, 0.35);
        background: rgba(0, 180, 100, 0.08);
    }

    .sell-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(220, 60, 60, 0.35);
        background: rgba(220, 60, 60, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="scanner-title">📊 Spot AI Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="scanner-subtitle">'
    '15m | 500K Target | MLP AI | No Order Execution'
    '</div>',
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

if "last_status" not in st.session_state:
    st.session_state.last_status = "READY"

if "force_scan" not in st.session_state:
    st.session_state.force_scan = False


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": "Spot-AI-Scanner/2.0",
            "Accept": "application/json"
        }
    )

    return session


HTTP = get_http_session()


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def safe_float(
    value,
    default=np.nan
):

    try:
        return float(value)
    except Exception:
        return default


def timeframe_minutes(
    timeframe
):

    values = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }

    return values.get(
        timeframe,
        15
    )


# ============================================================
# EXCEPTIONS
# ============================================================

class BinanceRestrictedError(
    Exception
):
    pass


class DataProviderError(
    Exception
):
    pass


# ============================================================
# BINANCE HTTP
# ============================================================

def binance_get(
    endpoint,
    params=None
):

    url = (
        BINANCE_BASE_URL.rstrip("/")
        + endpoint
    )

    try:

        response = HTTP.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

    except requests.RequestException as exc:

        raise DataProviderError(
            f"Network error: {exc}"
        )

    if response.status_code == 451:

        raise BinanceRestrictedError(
            "HTTP 451: Binance API bu çalışma "
            "ortamından erişime izin vermiyor."
        )

    if response.status_code == 429:

        raise DataProviderError(
            "HTTP 429: Binance rate limit."
        )

    if response.status_code >= 400:

        try:
            body = response.json()
        except Exception:
            body = response.text[:500]

        raise DataProviderError(
            f"HTTP {response.status_code}: {body}"
        )

    try:

        return response.json()

    except Exception as exc:

        raise DataProviderError(
            f"JSON parse error: {exc}"
        )


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_binance_symbols():

    data = binance_get(
        BINANCE_EXCHANGE_INFO
    )

    symbols = []

    for item in data.get(
        "symbols",
        []
    ):

        if item.get(
            "status"
        ) != "TRADING":
            continue

        if (
            item.get(
                "isSpotTradingAllowed"
            )
            is False
        ):
            continue

        quote = item.get(
            "quoteAsset",
            ""
        )

        if quote not in SUPPORTED_QUOTES:
            continue

        symbol = item.get(
            "symbol"
        )

        if symbol:
            symbols.append(
                symbol
            )

    return sorted(
        symbols
    )


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
        "limit": min(
            int(limit),
            1000
        )
    }

    if end_time is not None:

        params["endTime"] = int(
            end_time
        )

    data = binance_get(
        BINANCE_KLINES,
        params
    )

    if not data:

        return pd.DataFrame()

    rows = []

    for item in data:

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(item[0]),
                    unit="ms",
                    utc=True
                ),
                "open": safe_float(
                    item[1]
                ),
                "high": safe_float(
                    item[2]
                ),
                "low": safe_float(
                    item[3]
                ),
                "close": safe_float(
                    item[4]
                ),
                "volume": safe_float(
                    item[5]
                )
            }
        )

    return clean_ohlcv(
        pd.DataFrame(rows)
    )


# ============================================================
# BINANCE LARGE HISTORY
# ============================================================

def download_binance_history(
    symbol,
    target,
    progress_callback=None
):

    target = int(target)

    chunks = []

    end_time = None

    total = 0

    max_loops = math.ceil(
        target / 1000
    )

    for loop_index in range(
        max_loops
    ):

        df = get_binance_klines(
            symbol,
            1000,
            end_time
        )

        if df.empty:
            break

        chunks.append(
            df
        )

        total += len(df)

        oldest = df[
            "timestamp"
        ].min()

        end_time = (
            int(
                oldest.timestamp()
                * 1000
            )
            - 1
        )

        if progress_callback:

            progress_callback(
                min(
                    total / target,
                    1.0
                )
            )

        if len(df) < 1000:
            break

        if total >= target:
            break

        if (
            loop_index % 10
            == 0
        ):

            time.sleep(
                0.05
            )

    if not chunks:

        return pd.DataFrame()

    result = pd.concat(
        chunks,
        ignore_index=True
    )

    result = clean_ohlcv(
        result
    )

    result = (
        result
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
        )
        .tail(target)
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# CCXT
# ============================================================

@st.cache_resource
def import_ccxt():

    try:

        import ccxt

        return ccxt

    except ImportError:

        return None


def create_ccxt_exchange(
    exchange_id
):

    ccxt = import_ccxt()

    if ccxt is None:

        raise DataProviderError(
            "CCXT kurulu değil. "
            "requirements.txt kontrol edin."
        )

    if not hasattr(
        ccxt,
        exchange_id
    ):

        raise DataProviderError(
            f"Exchange bulunamadı: "
            f"{exchange_id}"
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


def get_ccxt_symbols(
    exchange_id
):

    exchange = create_ccxt_exchange(
        exchange_id
    )

    try:

        markets = (
            exchange.load_markets()
        )

    except Exception as exc:

        raise DataProviderError(
            f"{exchange_id} market listesi "
            f"alınamadı: {exc}"
        )

    symbols = []

    for symbol, market in markets.items():

        try:

            if market.get(
                "spot"
            ) is not True:

                continue

            if market.get(
                "active"
            ) is False:

                continue

            quote = market.get(
                "quote"
            )

            if quote not in SUPPORTED_QUOTES:

                continue

            symbols.append(
                symbol
            )

        except Exception:

            continue

    return sorted(
        symbols
    )


# ============================================================
# CCXT HISTORY
# ============================================================

def get_ccxt_history(
    exchange_id,
    symbol,
    target,
    progress_callback=None
):

    exchange = create_ccxt_exchange(
        exchange_id
    )

    limit = 1000

    rows = []

    timeframe_ms = (
        timeframe_minutes(
            TIMEFRAME
        )
        * 60
        * 1000
    )

    # Most exchanges allow historical
    # pagination with "since".

    since = None

    max_loops = math.ceil(
        target / limit
    )

    for index in range(
        max_loops
    ):

        try:

            batch = (
                exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=since,
                    limit=limit
                )
            )

        except Exception as exc:

            raise DataProviderError(
                f"{exchange_id} {symbol} "
                f"OHLCV hatası: {exc}"
            )

        if not batch:

            break

        rows.extend(
            batch
        )

        if progress_callback:

            progress_callback(
                min(
                    len(rows) / target,
                    1.0
                )
            )

        if len(batch) < limit:

            break

        last_timestamp = (
            batch[-1][0]
        )

        since = (
            last_timestamp
            + timeframe_ms
        )

        if len(rows) >= target:

            break

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
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

    df = clean_ohlcv(
        df
    )

    return (
        df
        .tail(target)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CSV
# ============================================================

def read_csv_ohlcv(
    uploaded_file
):

    raw = uploaded_file.read()

    try:

        df = pd.read_csv(
            io.BytesIO(raw)
        )

    except Exception as exc:

        raise DataProviderError(
            f"CSV okunamadı: {exc}"
        )

    original_columns = list(
        df.columns
    )

    normalized = {}

    for column in original_columns:

        key = (
            str(column)
            .strip()
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        normalized[key] = column

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

    rename = {}

    for target, names in aliases.items():

        for name in names:

            if name in normalized:

                rename[
                    normalized[name]
                ] = target

                break

    df = df.rename(
        columns=rename
    )

    return clean_ohlcv(
        df
    )


# ============================================================
# CLEAN OHLCV
# ============================================================

def clean_ohlcv(
    df
):

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
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise DataProviderError(
            "Eksik OHLCV kolonları: "
            + ", ".join(missing)
        )

    result = df.copy()

    result[
        "timestamp"
    ] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        result[column] = pd.to_numeric(
            result[column],
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
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(
    series,
    period=14
):

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

    return (
        100
        - 100 / (1 + rs)
    )


def calculate_atr(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous_close = close.shift(
        1
    )

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_macd(
    series
):

    fast = calculate_ema(
        series,
        12
    )

    slow = calculate_ema(
        series,
        26
    )

    line = fast - slow

    signal = calculate_ema(
        line,
        9
    )

    histogram = (
        line - signal
    )

    return (
        line,
        signal,
        histogram
    )


def calculate_bollinger(
    series,
    period=20
):

    middle = series.rolling(
        period
    ).mean()

    std = series.rolling(
        period
    ).std()

    upper = (
        middle
        + 2 * std
    )

    lower = (
        middle
        - 2 * std
    )

    return (
        middle,
        upper,
        lower
    )


def calculate_adx(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (
            up_move > down_move
        )
        & (
            up_move > 0
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            down_move > up_move
        )
        & (
            down_move > 0
        ),
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
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )

    atr_value = tr.rolling(
        period
    ).mean()

    plus_dm_series = pd.Series(
        plus_dm,
        index=df.index
    )

    minus_dm_series = pd.Series(
        minus_dm,
        index=df.index
    )

    plus_di = (
        100
        * plus_dm_series.rolling(
            period
        ).mean()
        / (
            atr_value + 1e-12
        )
    )

    minus_di = (
        100
        * minus_dm_series.rolling(
            period
        ).mean()
        / (
            atr_value + 1e-12
        )
    )

    dx = (
        100
        * (
            plus_di - minus_di
        ).abs()
        / (
            plus_di
            + minus_di
            + 1e-12
        )
    )

    return dx.rolling(
        period
    ).mean()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    df
):

    data = df.copy()

    close = data["close"]

    high = data["high"]

    low = data["low"]

    volume = data["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    for period in [
        1,
        3,
        6,
        12,
        24,
        48,
        96
    ]:

        data[
            f"ret_{period}"
        ] = close.pct_change(
            period
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for period in [
        5,
        20,
        50,
        100,
        200,
        500,
        800
    ]:

        value = calculate_ema(
            close,
            period
        )

        data[
            f"ema_{period}"
        ] = value

        data[
            f"ema_dist_{period}"
        ] = (
            close
            / (
                value + 1e-12
            )
            - 1
        )

    data["ema_5_20"] = (
        data["ema_5"]
        / (
            data["ema_20"]
            + 1e-12
        )
        - 1
    )

    data["ema_20_50"] = (
        data["ema_20"]
        / (
            data["ema_50"]
            + 1e-12
        )
        - 1
    )

    data["ema_50_200"] = (
        data["ema_50"]
        / (
            data["ema_200"]
            + 1e-12
        )
        - 1
    )

    data["ema_200_800"] = (
        data["ema_200"]
        / (
            data["ema_800"]
            + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    for period in [
        7,
        14,
        21
    ]:

        data[
            f"rsi_{period}"
        ] = calculate_rsi(
            close,
            period
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    (
        macd_line,
        macd_signal,
        macd_hist
    ) = calculate_macd(
        close
    )

    data["macd"] = macd_line

    data[
        "macd_signal"
    ] = macd_signal

    data[
        "macd_hist"
    ] = macd_hist

    data[
        "macd_norm"
    ] = (
        macd_line
        / (
            close + 1e-12
        )
    )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    (
        bb_middle,
        bb_upper,
        bb_lower
    ) = calculate_bollinger(
        close,
        20
    )

    data[
        "bb_middle"
    ] = bb_middle

    data[
        "bb_upper"
    ] = bb_upper

    data[
        "bb_lower"
    ] = bb_lower

    data[
        "bb_width"
    ] = (
        bb_upper
        - bb_lower
    ) / (
        bb_middle
        + 1e-12
    )

    data[
        "bb_position"
    ] = (
        close
        - bb_lower
    ) / (
        bb_upper
        - bb_lower
        + 1e-12
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data[
        "atr"
    ] = calculate_atr(
        data,
        14
    )

    data[
        "atr_pct"
    ] = (
        data["atr"]
        / (
            close + 1e-12
        )
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    data[
        "adx"
    ] = calculate_adx(
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

    data[
        "stoch_k"
    ] = (
        100
        * (
            close - lowest
        )
        / (
            highest
            - lowest
            + 1e-12
        )
    )

    data[
        "stoch_d"
    ] = (
        data["stoch_k"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    candle_body = (
        close
        - data["open"]
    )

    data[
        "body_pct"
    ] = (
        candle_body
        / (
            close + 1e-12
        )
    )

    data[
        "range_pct"
    ] = (
        candle_range
        / (
            close + 1e-12
        )
    )

    data[
        "upper_wick"
    ] = (
        high
        - np.maximum(
            data["open"],
            close
        )
    )

    data[
        "lower_wick"
    ] = (
        np.minimum(
            data["open"],
            close
        )
        - low
    )

    data[
        "body_to_range"
    ] = (
        candle_body.abs()
        / (
            candle_range
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = (
        volume
        .rolling(20)
        .mean()
    )

    volume_std = (
        volume
        .rolling(20)
        .std()
    )

    data[
        "volume_ratio"
    ] = (
        volume
        / (
            volume_mean
            + 1e-12
        )
    )

    data[
        "volume_z"
    ] = (
        volume
        - volume_mean
    ) / (
        volume_std
        + 1e-12
    )

    data[
        "volume_change"
    ] = volume.pct_change()

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    returns = data[
        "ret_1"
    ]

    for period in [
        12,
        24,
        48,
        96
    ]:

        data[
            f"volatility_{period}"
        ] = (
            returns
            .rolling(period)
            .std()
        )

    # --------------------------------------------------------
    # HIGH / LOW
    # --------------------------------------------------------

    for period in [
        48,
        200
    ]:

        rolling_high = (
            high
            .rolling(period)
            .max()
        )

        rolling_low = (
            low
            .rolling(period)
            .min()
        )

        data[
            f"distance_high_{period}"
        ] = (
            close
            / (
                rolling_high
                + 1e-12
            )
            - 1
        )

        data[
            f"distance_low_{period}"
        ] = (
            close
            / (
                rolling_low
                + 1e-12
            )
            - 1
        )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    running_high = (
        close
        .cummax()
    )

    data[
        "drawdown"
    ] = (
        close
        / (
            running_high
            + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # PRICE Z SCORE
    # --------------------------------------------------------

    for period in [
        50,
        200
    ]:

        mean = (
            close
            .rolling(period)
            .mean()
        )

        std = (
            close
            .rolling(period)
            .std()
        )

        data[
            f"price_z_{period}"
        ] = (
            close - mean
        ) / (
            std + 1e-12
        )

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return data


# ============================================================
# FEATURE LIST
# ============================================================

def feature_columns():

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
# TARGET
# ============================================================

def create_target(
    df,
    future_bars
):

    data = df.copy()

    future_return = (
        data["close"]
        .shift(-future_bars)
        / data["close"]
        - 1
    )

    data[
        "future_return"
    ] = future_return

    volatility = (
        data[
            "volatility_24"
        ]
        .fillna(
            data[
                "ret_1"
            ].std()
        )
    )

    threshold = (
        volatility
        * math.sqrt(
            future_bars
        )
    )

    threshold = threshold.clip(
        lower=0.002
    )

    # 0 = SELL
    # 1 = HOLD
    # 2 = BUY

    data[
        "target"
    ] = 1

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
# MLP
# ============================================================

def create_classifier():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "mlp",
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
                    max_iter=100,
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
                "ridge",
                Ridge(
                    alpha=1.0
                )
            )
        ]
    )


# ============================================================
# TRAIN
# ============================================================

def train_models(
    df,
    future_bars,
    train_limit
):

    features = build_features(
        df
    )

    features = create_target(
        features,
        future_bars
    )

    columns = feature_columns()

    usable = features.dropna(
        subset=columns
        + [
            "target",
            "future_return"
        ]
    ).copy()

    if len(usable) < 500:

        raise ValueError(
            "Model için yeterli temiz veri "
            f"yok. Temiz satır: {len(usable)}"
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

    if y.nunique() < 2:

        raise ValueError(
            "AI training target yalnızca "
            "tek sınıf içeriyor."
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

    predictions = (
        classifier.predict(X)
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
# PREDICTION
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
            "Son candle için feature "
            "oluşturulamadı."
        )

    latest = valid.iloc[-1]

    X = pd.DataFrame(
        [
            latest[
                columns
            ].astype(float)
        ]
    )

    probabilities = (
        classifier
        .predict_proba(X)[0]
    )

    classes = (
        classifier.classes_
    )

    probability_map = {
        int(
            class_id
        ): float(
            probability
        )
        for class_id,
        probability
        in zip(
            classes,
            probabilities
        )
    }

    p_sell = probability_map.get(
        0,
        0.0
    )

    p_hold = probability_map.get(
        1,
        0.0
    )

    p_buy = probability_map.get(
        2,
        0.0
    )

    prediction = int(
        classifier.predict(
            X
        )[0]
    )

    expected_return = float(
        regressor.predict(
            X
        )[0]
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

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    price = float(
        latest["close"]
    )

    rsi14 = safe_float(
        latest["rsi_14"]
    )

    bb_position = safe_float(
        latest["bb_position"]
    )

    atr_pct = safe_float(
        latest["atr_pct"]
    )

    adx = safe_float(
        latest["adx"]
    )

    volume_ratio = safe_float(
        latest["volume_ratio"]
    )

    momentum = safe_float(
        latest["ret_24"]
    )

    trend = safe_float(
        latest["ema_20_50"]
    )

    volatility = safe_float(
        latest["volatility_24"]
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    direction = (
        p_buy
        - p_sell
    )

    momentum_component = np.tanh(
        momentum * 20
    )

    trend_component = np.tanh(
        trend * 20
    )

    volume_component = np.tanh(
        (volume_ratio - 1)
        / 2
    )

    if np.isnan(
        volatility
    ):

        risk_component = 0.0

    else:

        risk_component = (
            1
            - np.clip(
                volatility * 30,
                0,
                1
            )
        )

    score = (
        50
        + direction * 25
        + momentum_component * 10
        + trend_component * 7
        + volume_component * 4
        + risk_component * 4
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

        "confidence": confidence,

        "p_buy": p_buy,

        "p_hold": p_hold,

        "p_sell": p_sell,

        "expected_return":
            expected_return,

        "price":
            price,

        "rsi14":
            rsi14,

        "bb_position":
            bb_position,

        "atr_pct":
            atr_pct,

        "adx":
            adx,

        "volume_ratio":
            volume_ratio,

        "momentum":
            momentum,

        "trend":
            trend,

        "volatility":
            volatility,

        "score":
            score
    }


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
    df,
    future_bars,
    train_limit
):

    start_time = time.time()

    (
        classifier,
        regressor,
        features,
        columns,
        accuracy
    ) = train_models(
        df,
        future_bars,
        train_limit
    )

    prediction = predict_latest(
        classifier,
        regressor,
        features,
        columns
    )

    return {

        "symbol":
            symbol,

        "signal":
            prediction["signal"],

        "score":
            prediction["score"],

        "confidence":
            prediction["confidence"],

        "expected_return":
            prediction[
                "expected_return"
            ],

        "price":
            prediction["price"],

        "p_buy":
            prediction["p_buy"],

        "p_hold":
            prediction["p_hold"],

        "p_sell":
            prediction["p_sell"],

        "rsi14":
            prediction["rsi14"],

        "bb_position":
            prediction["bb_position"],

        "atr_pct":
            prediction["atr_pct"],

        "adx":
            prediction["adx"],

        "volume_ratio":
            prediction["volume_ratio"],

        "momentum":
            prediction["momentum"],

        "trend":
            prediction["trend"],

        "volatility":
            prediction["volatility"],

        "training_accuracy":
            accuracy,

        "candles":
            len(df),

        "last_candle":
            df[
                "timestamp"
            ].iloc[-1],

        "processing_seconds":
            time.time()
            - start_time
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)


data_source = st.sidebar.selectbox(
    "Data Source",
    [
        "CCXT Exchange",
        "CSV / Uploaded Data",
        "Binance REST"
    ],
    index=0
)


if data_source == "CCXT Exchange":

    exchange_id = st.sidebar.selectbox(
        "Exchange",
        [
            "kraken",
            "coinbase",
            "kucoin",
            "okx",
            "bybit",
            "bitget"
        ],
        index=0
    )

else:

    exchange_id = None


uploaded_file = None

if data_source == "CSV / Uploaded Data":

    uploaded_file = (
        st.sidebar.file_uploader(
            "OHLCV CSV",
            type=["csv"]
        )
    )

    st.sidebar.caption(
        "Gerekli kolonlar: "
        "timestamp, open, high, low, close, volume"
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
    1,
    100,
    DEFAULT_SYMBOL_LIMIT
)


future_bars = st.sidebar.slider(
    "Future Bars",
    3,
    48,
    DEFAULT_FUTURE_BARS
)


train_limit = st.sidebar.slider(
    "Training Rows",
    2_000,
    30_000,
    DEFAULT_TRAIN_LIMIT,
    step=1_000
)


min_confidence = st.sidebar.slider(
    "Minimum AI Confidence",
    0.30,
    0.95,
    0.55,
    step=0.01
)


auto_refresh = st.sidebar.checkbox(
    "15 Dakikada Otomatik Tarama",
    value=True
)


st.sidebar.divider()


manual_scan = st.sidebar.button(
    "🔄 Şimdi Tara",
    use_container_width=True
)


if manual_scan:

    st.session_state.force_scan = True


st.sidebar.divider()


st.sidebar.info(
    "⚠️ Bu uygulama hiçbir emir göndermez. "
    "Sadece OHLCV verisini analiz eder."
)


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

def discover_symbols():

    if data_source == "CSV / Uploaded Data":

        if uploaded_file is None:

            raise DataProviderError(
                "Önce bir CSV dosyası yükleyin."
            )

        return [
            "UPLOADED_DATA"
        ]

    if data_source == "CCXT Exchange":

        symbols = get_ccxt_symbols(
            exchange_id
        )

        if not symbols:

            raise DataProviderError(
                f"{exchange_id} üzerinde "
                "Spot sembol bulunamadı."
            )

        return symbols[
            :symbol_limit
        ]

    if data_source == "Binance REST":

        try:

            symbols = (
                get_binance_symbols()
            )

        except BinanceRestrictedError:

            raise

        except Exception:

            raise

        return symbols[
            :symbol_limit
        ]

    raise DataProviderError(
        "Geçersiz data source."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_symbol_data(
    symbol
):

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if (
        data_source
        == "CSV / Uploaded Data"
    ):

        df = read_csv_ohlcv(
            uploaded_file
        )

        return (
            df
            .tail(
                candle_target
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # CCXT
    # --------------------------------------------------------

    if (
        data_source
        == "CCXT Exchange"
    ):

        cache_key = (
            "CCXT|"
            + str(exchange_id)
            + "|"
            + str(symbol)
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

            return (
                cached
                .tail(
                    candle_target
                )
                .reset_index(
                    drop=True
                )
            )

        df = get_ccxt_history(
            exchange_id,
            symbol,
            candle_target
        )

        if not df.empty:

            st.session_state.candle_cache[
                cache_key
            ] = df.tail(
                candle_target
            )

        return df

    # --------------------------------------------------------
    # BINANCE
    # --------------------------------------------------------

    if (
        data_source
        == "Binance REST"
    ):

        cache_key = (
            "BINANCE|"
            + str(symbol)
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

            return (
                cached
                .tail(
                    candle_target
                )
                .reset_index(
                    drop=True
                )
            )

        df = download_binance_history(
            symbol,
            candle_target
        )

        if not df.empty:

            st.session_state.candle_cache[
                cache_key
            ] = df.tail(
                candle_target
            )

        return df

    raise DataProviderError(
        "Data source hatası."
    )


# ============================================================
# SCAN
# ============================================================

def run_scan():

    st.session_state.scan_errors = []

    st.session_state.last_status = (
        "SCANNING"
    )

    results = []

    progress_bar = st.progress(
        0
    )

    status_box = st.empty()

    try:

        status_box.info(
            "Sembol listesi alınıyor..."
        )

        symbols = discover_symbols()

        total = len(symbols)

        if total == 0:

            raise DataProviderError(
                "Hiç sembol bulunamadı."
            )

        status_box.success(
            f"{total} sembol bulundu."
        )

        for index, symbol in enumerate(
            symbols
        ):

            status_box.info(
                f"Analiz ediliyor: "
                f"{symbol} "
                f"({index + 1}/{total})"
            )

            try:

                df = load_symbol_data(
                    symbol
                )

                if df.empty:

                    raise DataProviderError(
                        "OHLCV verisi boş."
                    )

                if len(df) < 500:

                    raise DataProviderError(
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

            except Exception as exc:

                error_message = (
                    f"{type(exc).__name__}: "
                    f"{str(exc)}"
                )

                st.session_state.scan_errors.append(
                    {
                        "symbol":
                            symbol,
                        "error":
                            error_message
                    }
                )

            progress_bar.progress(
                int(
                    (
                        index + 1
                    )
                    / total
                    * 100
                )
            )

        result_df = pd.DataFrame(
            results
        )

        if not result_df.empty:

            result_df = (
                result_df
                .sort_values(
                    "score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
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

        if result_df.empty:

            st.session_state.last_status = (
                "NO_RESULTS"
            )

        else:

            st.session_state.last_status = (
                "SUCCESS"
            )

    except BinanceRestrictedError as exc:

        message = (
            "HTTP 451: Binance API bu çalışma "
            "ortamından erişime izin vermiyor."
        )

        st.session_state.scan_errors.append(
            {
                "symbol": "BINANCE",
                "error": message
            }
        )

        st.session_state.last_status = (
            "BINANCE_451"
        )

        st.error(
            message
        )

        st.warning(
            "Bu durum Python veya Streamlit "
            "kod hatası değildir."
        )

        st.info(
            "Data Source bölümünden "
            "'CCXT Exchange' seçin. "
            "Varsayılan seçenek Kraken Spot'tur."
        )

    except Exception as exc:

        message = (
            f"{type(exc).__name__}: "
            f"{str(exc)}"
        )

        st.session_state.scan_errors.append(
            {
                "symbol": "SYSTEM",
                "error": message
            }
        )

        st.session_state.last_status = (
            "ERROR"
        )

        st.error(
            f"Tarama hatası: {message}"
        )

    finally:

        progress_bar.empty()

        status_box.empty()

        st.session_state.force_scan = (
            False
        )


# ============================================================
# SCAN TRIGGER
# ============================================================

def should_run_scan():

    if st.session_state.force_scan:

        return True

    if auto_refresh:

        return True

    return False


# ============================================================
# FRAGMENT SUPPORT
# ============================================================

if hasattr(
    st,
    "fragment"
):

    @st.fragment(
        run_every="15m"
        if auto_refresh
        else None
    )
    def scanner():

        if should_run_scan():

            run_scan()

    scanner()

else:

    if should_run_scan():

        run_scan()


# ============================================================
# STATUS
# ============================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Status",
        st.session_state.last_status
    )

with c2:

    st.metric(
        "Source",
        st.session_state.last_source
        or data_source
    )

with c3:

    st.metric(
        "Candle Target",
        f"{candle_target:,}"
    )

with c4:

    if st.session_state.last_scan:

        st.metric(
            "Last Scan",
            st.session_state.last_scan.strftime(
                "%Y-%m-%d %H:%M:%S"
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

    if (
        data_source
        == "Binance REST"
    ):

        st.info(
            "Binance REST seçildi. "
            "Eğer HTTP 451 alıyorsanız "
            "Binance bu çalışma ortamına "
            "API erişimi vermiyor."
        )

    elif (
        data_source
        == "CCXT Exchange"
    ):

        st.info(
            f"{exchange_id} Spot üzerinden "
            "veri alınmaya çalışılıyor."
        )

    else:

        st.info(
            "OHLCV CSV yükleyip "
            "'Şimdi Tara' butonuna basın."
        )


else:

    # ========================================================
    # BUY
    # ========================================================

    buy = results[
        results["signal"]
        == "BUY"
    ].copy()

    buy = buy[
        buy["confidence"]
        >= min_confidence
    ]

    buy = buy.sort_values(
        [
            "score",
            "confidence"
        ],
        ascending=False
    ).head(10)

    st.subheader(
        "🟢 TOP 10 BUY"
    )

    if buy.empty:

        st.info(
            "Confidence filtresini geçen "
            "BUY sinyali bulunamadı."
        )

    else:

        display = buy.copy()

        display[
            "confidence"
        ] = (
            display[
                "confidence"
            ] * 100
        ).round(2)

        display[
            "expected_return"
        ] = (
            display[
                "expected_return"
            ] * 100
        ).round(3)

        display[
            "p_buy"
        ] = (
            display[
                "p_buy"
            ] * 100
        ).round(2)

        display[
            "score"
        ] = display[
            "score"
        ].round(2)

        st.dataframe(
            display[
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

    # ========================================================
    # SELL
    # ========================================================

    sell = results[
        results["signal"]
        == "SELL"
    ].copy()

    sell = sell[
        sell["confidence"]
        >= min_confidence
    ]

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

    st.subheader(
        "🔴 TOP 10 SELL"
    )

    if sell.empty:

        st.info(
            "Confidence filtresini geçen "
            "SELL sinyali bulunamadı."
        )

    else:

        display = sell.copy()

        display[
            "confidence"
        ] = (
            display[
                "confidence"
            ] * 100
        ).round(2)

        display[
            "expected_return"
        ] = (
            display[
                "expected_return"
            ] * 100
        ).round(3)

        display[
            "p_sell"
        ] = (
            display[
                "p_sell"
            ] * 100
        ).round(2)

        display[
            "score"
        ] = display[
            "score"
        ].round(2)

        st.dataframe(
            display[
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

    # ========================================================
    # HOLD
    # ========================================================

    hold = results[
        results["signal"]
        == "HOLD"
    ].copy()

    st.subheader(
        "🟡 HOLD"
    )

    if hold.empty:

        st.info(
            "HOLD sonucu yok."
        )

    else:

        hold_display = hold.copy()

        hold_display[
            "confidence"
        ] = (
            hold_display[
                "confidence"
            ] * 100
        ).round(2)

        hold_display[
            "score"
        ] = hold_display[
            "score"
        ].round(2)

        st.dataframe(
            hold_display[
                [
                    "symbol",
                    "score",
                    "confidence",
                    "expected_return",
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

    # ========================================================
    # ALL RESULTS
    # ========================================================

    with st.expander(
        "📋 Tüm AI Sonuçları"
    ):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # SYMBOL DETAIL
    # ========================================================

    st.subheader(
        "🔎 Symbol Detail"
    )

    selected_symbol = st.selectbox(
        "Coin",
        results[
            "symbol"
        ].tolist()
    )

    selected = results[
        results["symbol"]
        == selected_symbol
    ].iloc[0]

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Signal",
            selected[
                "signal"
            ]
        )

    with d2:

        st.metric(
            "Confidence",
            f"{selected['confidence'] * 100:.2f}%"
        )

    with d3:

        st.metric(
            "Expected Return",
            f"{selected['expected_return'] * 100:.3f}%"
        )

    with d4:

        st.metric(
            "AI Score",
            f"{selected['score']:.2f}"
        )

    detail = pd.DataFrame(
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
                "Momentum 24",
                "EMA Trend 20/50",
                "Volatility",
                "Training Accuracy",
                "Candle Count",
                "Processing Time"
            ],

            "Value": [

                selected[
                    "price"
                ],

                f"{selected['p_buy'] * 100:.2f}%",

                f"{selected['p_hold'] * 100:.2f}%",

                f"{selected['p_sell'] * 100:.2f}%",

                selected[
                    "rsi14"
                ],

                selected[
                    "bb_position"
                ],

                selected[
                    "atr_pct"
                ],

                selected[
                    "adx"
                ],

                selected[
                    "volume_ratio"
                ],

                selected[
                    "momentum"
                ],

                selected[
                    "trend"
                ],

                selected[
                    "volatility"
                ],

                f"{selected['training_accuracy'] * 100:.2f}%",

                selected[
                    "candles"
                ],

                f"{selected['processing_seconds']:.2f}s"
            ]
        }
    )

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CSV EXPORT
# ============================================================

if not results.empty:

    st.divider()

    st.subheader(
        "💾 Export"
    )

    csv_data = results.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ AI Results CSV",
        data=csv_data,
        file_name=(
            "spot_ai_results.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# CACHE
# ============================================================

with st.expander(
    "🗄️ Candle Cache"
):

    if not st.session_state.candle_cache:

        st.info(
            "Candle cache boş."
        )

    else:

        cache_rows = []

        for key, df in (
            st.session_state
            .candle_cache
            .items()
        ):

            if (
                df is None
                or df.empty
            ):

                continue

            cache_rows.append(
                {
                    "source":
                        key,

                    "candles":
                        len(df),

                    "first":
                        str(
                            df[
                                "timestamp"
                            ].iloc[0]
                        ),

                    "last":
                        str(
                            df[
                                "timestamp"
                            ].iloc[-1]
                        )
                }
            )

        if cache_rows:

            st.dataframe(
                pd.DataFrame(
                    cache_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Cache kullanılabilir veri içermiyor."
            )


# ============================================================
# ERROR DETAILS
# ============================================================

if st.session_state.scan_errors:

    st.divider()

    st.subheader(
        "⚠️ Hata Detayları"
    )

    for error in (
        st.session_state.scan_errors
    ):

        if isinstance(
            error,
            dict
        ):

            symbol = str(
                error.get(
                    "symbol",
                    "SYSTEM"
                )
            )

            message = str(
                error.get(
                    "error",
                    "Bilinmeyen hata"
                )
            )

            st.error(
                f"{symbol}: {message}"
            )

        else:

            st.error(
                str(error)
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
    "Bu sistem yalnızca teknik verilerden "
    "istatistiksel BUY / HOLD / SELL tahmini "
    "üretir. Yatırım tavsiyesi değildir."
)
