import io
import time
import math
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
# CONFIG
# ============================================================

APP_NAME = "Spot AI Scanner"

TIMEFRAME = "15m"

DEFAULT_TARGET = 500_000

DEFAULT_SYMBOL_LIMIT = 10

DEFAULT_FUTURE_BARS = 12

DEFAULT_TRAIN_LIMIT = 10_000

REQUEST_TIMEOUT = 30

KRAKEN_BASE_URL = "https://api.kraken.com"

KRAKEN_ASSET_PAIRS = "/0/public/AssetPairs"

KRAKEN_OHLC = "/0/public/OHLC"

BINANCE_BASE_URL = "https://api.binance.com"

BINANCE_EXCHANGE_INFO = "/api/v3/exchangeInfo"

BINANCE_KLINES = "/api/v3/klines"

SUPPORTED_QUOTES = [
    "USDT",
    "USDC",
    "USD",
    "EUR"
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Spot AI Scanner",
    page_icon="📊",
    layout="wide"
)


st.markdown(
    """
    <style>
    .scanner-title {
        font-size: 32px;
        font-weight: 800;
    }

    .scanner-subtitle {
        color: #888;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

defaults = {

    "scan_results":
        pd.DataFrame(),

    "scan_errors":
        [],

    "candle_cache":
        {},

    "last_scan":
        None,

    "last_source":
        None,

    "last_status":
        "READY",

    "force_scan":
        False
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HTTP
# ============================================================

@st.cache_resource
def get_http():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "Spot-AI-Scanner/3.0",
            "Accept":
                "application/json"
        }
    )

    return session


HTTP = get_http()


# ============================================================
# EXCEPTIONS
# ============================================================

class DataProviderError(Exception):
    pass


class BinanceRestrictedError(Exception):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def safe_float(
    value,
    default=np.nan
):

    try:

        return float(value)

    except Exception:

        return default


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
        x for x in required
        if x not in df.columns
    ]

    if missing:

        raise DataProviderError(
            "Eksik OHLCV kolonları: "
            + ", ".join(missing)
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
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
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# KRAKEN REQUEST
# ============================================================

def kraken_get(
    endpoint,
    params=None
):

    url = (
        KRAKEN_BASE_URL
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
            f"Kraken network error: {exc}"
        )

    if response.status_code >= 400:

        raise DataProviderError(
            f"Kraken HTTP "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    try:

        data = response.json()

    except Exception as exc:

        raise DataProviderError(
            f"Kraken JSON hatası: {exc}"
        )

    errors = data.get(
        "error",
        []
    )

    if errors:

        raise DataProviderError(
            "Kraken API: "
            + str(errors)
        )

    return data.get(
        "result",
        {}
    )


# ============================================================
# KRAKEN SYMBOLS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_kraken_symbols():

    result = kraken_get(
        KRAKEN_ASSET_PAIRS
    )

    symbols = []

    for key, item in result.items():

        if not isinstance(
            item,
            dict
        ):

            continue

        status = item.get(
            "status",
            "online"
        )

        if status != "online":

            continue

        quote = item.get(
            "quote",
            ""
        )

        wsname = item.get(
            "wsname",
            ""
        )

        altname = item.get(
            "altname",
            key
        )

        # Kraken quote formats:
        # ZUSD / USD
        # USDT
        quote_clean = str(
            quote
        ).replace(
            "Z",
            ""
        ).replace(
            "X",
            ""
        )

        if (
            quote_clean not in
            SUPPORTED_QUOTES
            and not str(
                wsname
            ).endswith(
                "/USD"
            )
            and not str(
                wsname
            ).endswith(
                "/USDT"
            )
        ):

            continue

        # Do not include obvious
        # derivatives/futures.

        if ".d" in str(
            altname
        ).lower():

            continue

        symbols.append(
            {
                "api_symbol":
                    altname,

                "display_symbol":
                    wsname
                    or altname
            }
        )

    symbols = sorted(
        symbols,
        key=lambda x:
            x["display_symbol"]
    )

    return symbols


# ============================================================
# KRAKEN OHLC
# ============================================================

def get_kraken_ohlc(
    symbol,
    since=None
):

    params = {
        "pair": symbol,

        # Kraken OHLC uses interval
        # in minutes.
        "interval": 15
    }

    if since is not None:

        params["since"] = int(
            since
        )

    result = kraken_get(
        KRAKEN_OHLC,
        params
    )

    pair_key = None

    for key in result.keys():

        if key != "last":

            pair_key = key

            break

    if pair_key is None:

        return pd.DataFrame()

    rows = result[
        pair_key
    ]

    if not rows:

        return pd.DataFrame()

    output = []

    for row in rows:

        # Kraken:
        # time, open, high, low, close,
        # vwap, volume, count

        output.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(
                            row[0]
                        ),
                        unit="s",
                        utc=True
                    ),

                "open":
                    safe_float(
                        row[1]
                    ),

                "high":
                    safe_float(
                        row[2]
                    ),

                "low":
                    safe_float(
                        row[3]
                    ),

                "close":
                    safe_float(
                        row[4]
                    ),

                "volume":
                    safe_float(
                        row[6]
                    )
            }
        )

    return clean_ohlcv(
        pd.DataFrame(
            output
        )
    )


# ============================================================
# KRAKEN HISTORY
# ============================================================

def download_kraken_history(
    symbol,
    target
):

    # Kraken's public OHLC endpoint has
    # a limited amount of historical data.
    #
    # Therefore we request what is
    # actually available and use the
    # maximum available history.

    frames = []

    since = None

    previous_oldest = None

    max_iterations = min(
        math.ceil(
            target / 720
        ),
        100
    )

    for _ in range(
        max_iterations
    ):

        df = get_kraken_ohlc(
            symbol,
            since
        )

        if df.empty:

            break

        frames.append(
            df
        )

        oldest = int(
            df[
                "timestamp"
            ].min().timestamp()
        )

        if (
            previous_oldest
            is not None
            and oldest >= previous_oldest
        ):

            break

        previous_oldest = oldest

        # Kraken OHLC endpoint normally
        # returns recent candles. We stop
        # if target has already been met.

        combined = clean_ohlcv(
            pd.concat(
                frames,
                ignore_index=True
            )
        )

        if len(combined) >= target:

            break

        # Kraken public OHLC does not
        # provide unlimited pagination in
        # the same way as Binance.

        break

    if not frames:

        return pd.DataFrame()

    result = clean_ohlcv(
        pd.concat(
            frames,
            ignore_index=True
        )
    )

    return (
        result
        .tail(target)
        .reset_index(drop=True)
    )


# ============================================================
# BINANCE
# ============================================================

def binance_get(
    endpoint,
    params=None
):

    url = (
        BINANCE_BASE_URL
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
            f"Binance network error: {exc}"
        )

    if response.status_code == 451:

        raise BinanceRestrictedError(
            "HTTP 451: Binance API bu "
            "çalışma ortamından erişime "
            "izin vermiyor."
        )

    if response.status_code >= 400:

        raise DataProviderError(
            f"Binance HTTP "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    try:

        return response.json()

    except Exception as exc:

        raise DataProviderError(
            f"Binance JSON hatası: {exc}"
        )


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_binance_symbols():

    result = binance_get(
        BINANCE_EXCHANGE_INFO
    )

    symbols = []

    for item in result.get(
        "symbols",
        []
    ):

        if item.get(
            "status"
        ) != "TRADING":

            continue

        if item.get(
            "isSpotTradingAllowed"
        ) is False:

            continue

        if item.get(
            "quoteAsset"
        ) not in [
            "USDT",
            "USDC"
        ]:

            continue

        symbols.append(
            item["symbol"]
        )

    return sorted(
        symbols
    )


def get_binance_klines(
    symbol,
    limit=1000,
    end_time=None
):

    params = {
        "symbol":
            symbol,

        "interval":
            "15m",

        "limit":
            min(
                int(limit),
                1000
            )
    }

    if end_time is not None:

        params[
            "endTime"
        ] = int(
            end_time
        )

    data = binance_get(
        BINANCE_KLINES,
        params
    )

    rows = []

    for row in data:

        rows.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(
                            row[0]
                        ),
                        unit="ms",
                        utc=True
                    ),

                "open":
                    safe_float(row[1]),

                "high":
                    safe_float(row[2]),

                "low":
                    safe_float(row[3]),

                "close":
                    safe_float(row[4]),

                "volume":
                    safe_float(row[5])
            }
        )

    return clean_ohlcv(
        pd.DataFrame(rows)
    )


def download_binance_history(
    symbol,
    target
):

    frames = []

    end_time = None

    total = 0

    loops = math.ceil(
        target / 1000
    )

    for _ in range(
        loops
    ):

        df = get_binance_klines(
            symbol,
            1000,
            end_time
        )

        if df.empty:

            break

        frames.append(
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

        if total >= target:

            break

    if not frames:

        return pd.DataFrame()

    result = clean_ohlcv(
        pd.concat(
            frames,
            ignore_index=True
        )
    )

    return (
        result
        .tail(target)
        .reset_index(drop=True)
    )


# ============================================================
# CSV
# ============================================================

def read_csv(
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

    mapping = {}

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        mapping[
            normalized
        ] = column

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

            if name in mapping:

                rename[
                    mapping[name]
                ] = target

                break

    df = df.rename(
        columns=rename
    )

    return clean_ohlcv(
        df
    )


# ============================================================
# INDICATORS
# ============================================================

def ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def rsi(
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

    rs = (
        avg_gain
        / (
            avg_loss + 1e-12
        )
    )

    return (
        100
        - 100
        / (
            1 + rs
        )
    )


def atr(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high
                - previous
            ).abs(),
            (
                low
                - previous
            ).abs()
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


def macd(
    series
):

    fast = ema(
        series,
        12
    )

    slow = ema(
        series,
        26
    )

    line = fast - slow

    signal = ema(
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


def bollinger(
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


def adx(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    up = high.diff()

    down = -low.diff()

    plus_dm = np.where(
        (
            up > down
        )
        & (
            up > 0
        ),
        up,
        0
    )

    minus_dm = np.where(
        (
            down > up
        )
        & (
            down > 0
        ),
        down,
        0
    )

    tr = pd.concat(
        [
            high - low,
            (
                high
                - close.shift(1)
            ).abs(),
            (
                low
                - close.shift(1)
            ).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    atr_value = tr.rolling(
        period
    ).mean()

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index
        ).rolling(
            period
        ).mean()
        / (
            atr_value
            + 1e-12
        )
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index
        ).rolling(
            period
        ).mean()
        / (
            atr_value
            + 1e-12
        )
    )

    dx = (
        100
        * (
            plus_di
            - minus_di
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
# FEATURES
# ============================================================

def build_features(
    df
):

    data = df.copy()

    close = data["close"]

    high = data["high"]

    low = data["low"]

    volume = data["volume"]

    # RETURNS

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

    # EMA

    for period in [
        5,
        20,
        50,
        100,
        200,
        500,
        800
    ]:

        value = ema(
            close,
            period
        )

        data[
            f"ema_dist_{period}"
        ] = (
            close
            / (
                value
                + 1e-12
            )
            - 1
        )

    data[
        "ema_5_20"
    ] = (
        ema(close, 5)
        / (
            ema(close, 20)
            + 1e-12
        )
        - 1
    )

    data[
        "ema_20_50"
    ] = (
        ema(close, 20)
        / (
            ema(close, 50)
            + 1e-12
        )
        - 1
    )

    data[
        "ema_50_200"
    ] = (
        ema(close, 50)
        / (
            ema(close, 200)
            + 1e-12
        )
        - 1
    )

    data[
        "ema_200_800"
    ] = (
        ema(close, 200)
        / (
            ema(close, 800)
            + 1e-12
        )
        - 1
    )

    # RSI

    for period in [
        7,
        14,
        21
    ]:

        data[
            f"rsi_{period}"
        ] = rsi(
            close,
            period
        )

    # MACD

    (
        macd_line,
        macd_signal,
        macd_hist
    ) = macd(
        close
    )

    data[
        "macd_norm"
    ] = (
        macd_line
        / (
            close
            + 1e-12
        )
    )

    data[
        "macd_signal"
    ] = macd_signal

    data[
        "macd_hist"
    ] = macd_hist

    # BOLLINGER

    (
        middle,
        upper,
        lower
    ) = bollinger(
        close,
        20
    )

    data[
        "bb_width"
    ] = (
        upper - lower
    ) / (
        middle
        + 1e-12
    )

    data[
        "bb_position"
    ] = (
        close - lower
    ) / (
        upper - lower
        + 1e-12
    )

    # ATR

    data[
        "atr_pct"
    ] = (
        atr(data)
        / (
            close
            + 1e-12
        )
    )

    # ADX

    data[
        "adx"
    ] = adx(data)

    # STOCHASTIC

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
            close
            - lowest
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
        data[
            "stoch_k"
        ]
        .rolling(3)
        .mean()
    )

    # CANDLE

    candle_range = (
        high - low
    )

    body = (
        close
        - data["open"]
    )

    data[
        "body_pct"
    ] = (
        body
        / (
            close
            + 1e-12
        )
    )

    data[
        "range_pct"
    ] = (
        candle_range
        / (
            close
            + 1e-12
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
        body.abs()
        / (
            candle_range
            + 1e-12
        )
    )

    # VOLUME

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

    # VOLATILITY

    for period in [
        12,
        24,
        48,
        96
    ]:

        data[
            f"volatility_{period}"
        ] = (
            data["ret_1"]
            .rolling(period)
            .std()
        )

    # HIGH / LOW

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

    # DRAWDOWN

    running_high = (
        close.cummax()
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

    # Z SCORE

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
            std
            + 1e-12
        )

    return data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


# ============================================================
# FEATURE COLUMNS
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
    data,
    future_bars
):

    result = data.copy()

    future_return = (
        result["close"]
        .shift(
            -future_bars
        )
        / result["close"]
        - 1
    )

    result[
        "future_return"
    ] = future_return

    volatility = (
        result[
            "volatility_24"
        ]
        .fillna(
            result[
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

    # 0 SELL
    # 1 HOLD
    # 2 BUY

    result[
        "target"
    ] = 1

    result.loc[
        future_return
        > threshold,
        "target"
    ] = 2

    result.loc[
        future_return
        < -threshold,
        "target"
    ] = 0

    return result


# ============================================================
# MODELS
# ============================================================

def classifier_model():

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


def regression_model():

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

def train(
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

    usable = (
        features
        .dropna(
            subset=columns
            + [
                "target",
                "future_return"
            ]
        )
        .copy()
    )

    if len(usable) < 500:

        raise DataProviderError(
            f"AI için yeterli veri yok. "
            f"Temiz veri: {len(usable)}"
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

        raise DataProviderError(
            "AI target tek sınıf içeriyor."
        )

    classifier = (
        classifier_model()
    )

    regressor = (
        regression_model()
    )

    classifier.fit(
        X,
        y
    )

    regressor.fit(
        X,
        future_y
    )

    predictions = (
        classifier.predict(
            X
        )
    )

    accuracy = (
        accuracy_score(
            y,
            predictions
        )
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

def predict(
    classifier,
    regressor,
    features,
    columns
):

    valid = (
        features
        .dropna(
            subset=columns
        )
    )

    if valid.empty:

        raise DataProviderError(
            "Latest candle için feature yok."
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
        int(c):
            float(p)

        for c, p in zip(
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

    adx_value = safe_float(
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

    # SCORE

    direction = (
        p_buy
        - p_sell
    )

    momentum_score = np.tanh(
        momentum * 20
    )

    trend_score = np.tanh(
        trend * 20
    )

    volume_score = np.tanh(
        (
            volume_ratio
            - 1
        ) / 2
    )

    if np.isnan(
        volatility
    ):

        risk_score = 0

    else:

        risk_score = (
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

        "signal":
            signal,

        "confidence":
            confidence,

        "p_buy":
            p_buy,

        "p_hold":
            p_hold,

        "p_sell":
            p_sell,

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
            adx_value,

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
# ANALYZE
# ============================================================

def analyze(
    symbol,
    df,
    future_bars,
    train_limit
):

    started = time.time()

    (
        classifier,
        regressor,
        features,
        columns,
        accuracy
    ) = train(
        df,
        future_bars,
        train_limit
    )

    prediction = predict(
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
            prediction[
                "bb_position"
            ],

        "atr_pct":
            prediction["atr_pct"],

        "adx":
            prediction["adx"],

        "volume_ratio":
            prediction[
                "volume_ratio"
            ],

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
            - started
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
        "Kraken REST",
        "CSV / Uploaded Data",
        "Binance REST"
    ]
)


uploaded_file = None

if data_source == (
    "CSV / Uploaded Data"
):

    uploaded_file = (
        st.sidebar.file_uploader(
            "OHLCV CSV",
            type=["csv"]
        )
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
    10
)


future_bars = st.sidebar.slider(
    "Future Bars",
    3,
    48,
    12
)


train_limit = st.sidebar.slider(
    "Training Rows",
    2_000,
    30_000,
    10_000,
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
    True
)


if st.sidebar.button(
    "🔄 Şimdi Tara",
    use_container_width=True
):

    st.session_state.force_scan = True


st.sidebar.info(
    "No Order Execution: "
    "Bu uygulama hiçbir alım/satım emri göndermez."
)


# ============================================================
# DISCOVER
# ============================================================

def discover():

    if data_source == "Kraken REST":

        pairs = get_kraken_symbols()

        if not pairs:

            raise DataProviderError(
                "Kraken Spot sembol bulunamadı."
            )

        return pairs[
            :symbol_limit
        ]

    if data_source == (
        "CSV / Uploaded Data"
    ):

        if uploaded_file is None:

            raise DataProviderError(
                "CSV yüklenmedi."
            )

        return [
            {
                "api_symbol":
                    "UPLOADED_DATA",

                "display_symbol":
                    "UPLOADED_DATA"
            }
        ]

    if data_source == "Binance REST":

        return get_binance_symbols()[
            :symbol_limit
        ]

    raise DataProviderError(
        "Bilinmeyen data source."
    )


# ============================================================
# LOAD
# ============================================================

def load_symbol(
    pair
):

    if data_source == (
        "Kraken REST"
    ):

        api_symbol = pair[
            "api_symbol"
        ]

        display_symbol = pair[
            "display_symbol"
        ]

        cache_key = (
            "KRAKEN|"
            + api_symbol
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
                display_symbol,
                cached.tail(
                    candle_target
                )
            )

        df = download_kraken_history(
            api_symbol,
            candle_target
        )

        if not df.empty:

            st.session_state[
                "candle_cache"
            ][
                cache_key
            ] = df

        return (
            display_symbol,
            df
        )

    if data_source == (
        "CSV / Uploaded Data"
    ):

        df = read_csv(
            uploaded_file
        )

        return (
            "UPLOADED_DATA",
            df.tail(
                candle_target
            )
        )

    if data_source == (
        "Binance REST"
    ):

        symbol = pair

        cache_key = (
            "BINANCE|"
            + symbol
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
                symbol,
                cached.tail(
                    candle_target
                )
            )

        df = download_binance_history(
            symbol,
            candle_target
        )

        if not df.empty:

            st.session_state[
                "candle_cache"
            ][
                cache_key
            ] = df

        return (
            symbol,
            df
        )

    raise DataProviderError(
        "Data source bulunamadı."
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

    progress = st.progress(
        0
    )

    status = st.empty()

    try:

        status.info(
            "Spot sembolleri alınıyor..."
        )

        pairs = discover()

        total = len(
            pairs
        )

        if total == 0:

            raise DataProviderError(
                "Sembol bulunamadı."
            )

        for index, pair in enumerate(
            pairs
        ):

            if isinstance(
                pair,
                dict
            ):

                label = pair[
                    "display_symbol"
                ]

            else:

                label = str(
                    pair
                )

            status.info(
                f"Tarama: {label} "
                f"({index + 1}/{total})"
            )

            try:

                symbol, df = load_symbol(
                    pair
                )

                if df.empty:

                    raise DataProviderError(
                        "OHLCV veri seti boş."
                    )

                if len(df) < 1_000:

                    raise DataProviderError(
                        f"Yetersiz candle: "
                        f"{len(df)}. "
                        "En az 1,000 candle gerekli."
                    )

                result = analyze(
                    symbol,
                    df,
                    future_bars,
                    train_limit
                )

                results.append(
                    result
                )

            except Exception as exc:

                st.session_state[
                    "scan_errors"
                ].append(
                    {
                        "symbol":
                            label,

                        "error":
                            (
                                type(exc).__name__
                                + ": "
                                + str(exc)
                            )
                    }
                )

            progress.progress(
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

        st.session_state[
            "scan_results"
        ] = result_df

        st.session_state[
            "last_scan"
        ] = utc_now()

        st.session_state[
            "last_source"
        ] = data_source

        st.session_state[
            "last_status"
        ] = (
            "SUCCESS"
            if not result_df.empty
            else "NO_RESULTS"
        )

    except BinanceRestrictedError as exc:

        message = (
            "HTTP 451: Binance API bu "
            "çalışma ortamından erişime "
            "izin vermiyor."
        )

        st.session_state[
            "scan_errors"
        ].append(
            {
                "symbol":
                    "BINANCE",

                "error":
                    message
            }
        )

        st.session_state[
            "last_status"
        ] = "BINANCE_451"

        st.error(
            message
        )

        st.info(
            "Binance yerine sol menüden "
            "'Kraken REST' seçebilirsiniz."
        )

    except Exception as exc:

        message = (
            type(exc).__name__
            + ": "
            + str(exc)
        )

        st.session_state[
            "scan_errors"
        ].append(
            {
                "symbol":
                    "SYSTEM",

                "error":
                    message
            }
        )

        st.session_state[
            "last_status"
        ] = "ERROR"

        st.error(
            "Tarama hatası: "
            + message
        )

    finally:

        progress.empty()

        status.empty()

        st.session_state[
            "force_scan"
        ] = False


# ============================================================
# RUN
# ============================================================

should_run = (
    st.session_state.force_scan
    or auto_refresh
)


if hasattr(
    st,
    "fragment"
):

    @st.fragment(
        run_every=(
            "15m"
            if auto_refresh
            else None
        )
    )
    def scanner():

        if should_run:

            run_scan()

    scanner()

else:

    if should_run:

        run_scan()


# ============================================================
# STATUS
# ============================================================

st.divider()

c1, c2, c3, c4 = st.columns(
    4
)

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

    st.metric(
        "Last Scan",
        (
            st.session_state.last_scan
            .strftime(
                "%H:%M:%S"
            )
            if st.session_state.last_scan
            else "-"
        )
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

else:

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

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
            "BUY sinyali yok."
        )

    else:

        view = buy.copy()

        view[
            "confidence"
        ] *= 100

        view[
            "expected_return"
        ] *= 100

        view[
            "p_buy"
        ] *= 100

        view = view.round(
            {
                "confidence": 2,
                "expected_return": 3,
                "p_buy": 2,
                "score": 2,
                "rsi14": 2,
                "adx": 2,
                "volume_ratio": 2,
                "price": 8
            }
        )

        st.dataframe(
            view[
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
            "SELL sinyali yok."
        )

    else:

        view = sell.copy()

        view[
            "confidence"
        ] *= 100

        view[
            "expected_return"
        ] *= 100

        view[
            "p_sell"
        ] *= 100

        view = view.round(
            {
                "confidence": 2,
                "expected_return": 3,
                "p_sell": 2,
                "score": 2,
                "rsi14": 2,
                "adx": 2,
                "volume_ratio": 2,
                "price": 8
            }
        )

        st.dataframe(
            view[
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
    # HOLD
    # --------------------------------------------------------

    hold = results[
        results["signal"]
        == "HOLD"
    ].copy()

    st.subheader(
        "🟡 HOLD"
    )

    if hold.empty:

        st.info(
            "HOLD yok."
        )

    else:

        view = hold.copy()

        view[
            "confidence"
        ] *= 100

        view[
            "expected_return"
        ] *= 100

        view = view.round(
            3
        )

        st.dataframe(
            view[
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

    # --------------------------------------------------------
    # ALL
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
        results[
            "symbol"
        ].tolist()
    )

    row = results[
        results["symbol"]
        == selected_symbol
    ].iloc[0]

    a, b, c, d = st.columns(
        4
    )

    with a:

        st.metric(
            "Signal",
            row["signal"]
        )

    with b:

        st.metric(
            "Confidence",
            f"{row['confidence'] * 100:.2f}%"
        )

    with c:

        st.metric(
            "Expected Return",
            f"{row['expected_return'] * 100:.3f}%"
        )

    with d:

        st.metric(
            "Score",
            f"{row['score']:.2f}"
        )


# ============================================================
# EXPORT
# ============================================================

if not results.empty:

    st.divider()

    st.download_button(
        "⬇️ Download AI Results CSV",
        data=results.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
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
            "Candle Cache boş."
        )

    else:

        cache_data = []

        for key, df in (
            st.session_state
            .candle_cache
            .items()
        ):

            if df.empty:

                continue

            cache_data.append(
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

        if cache_data:

            st.dataframe(
                pd.DataFrame(
                    cache_data
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# ERRORS
# ============================================================

if st.session_state.scan_errors:

    st.divider()

    st.subheader(
        "⚠️ Hata Detayları"
    )

    for item in (
        st.session_state.scan_errors
    ):

        if isinstance(
            item,
            dict
        ):

            st.error(
                f"{item.get('symbol', 'SYSTEM')}: "
                f"{item.get('error', 'Bilinmeyen hata')}"
            )

        else:

            st.error(
                str(item)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Scanner | 15m | 500K Target | "
    "MLP AI | No Order Execution"
)

st.caption(
    "Bu sistem yalnızca teknik verilerden "
    "istatistiksel BUY / HOLD / SELL tahmini üretir. "
    "Yatırım tavsiyesi değildir."
)
