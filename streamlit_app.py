import io
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Spot Scalping AI Scanner"

BINANCE_URL = "https://api.binance.com"
KRAKEN_URL = "https://api.kraken.com"

REQUEST_TIMEOUT = 30

TIMEFRAME = "15m"

# 2 x 15 minutes = maximum 30 minutes
FUTURE_BARS = 2
HORIZON_MINUTES = 30

BUY_TARGET = 0.10
SELL_TARGET = -0.10

MIN_CANDLES = 300

MAX_CANDLES = 500_000

SUPPORTED_QUOTES = [
    "USDT",
    "USDC",
]

FEATURES = [
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

    "ema_5_20",
    "ema_20_50",
    "ema_50_200",

    "rsi_7",
    "rsi_14",
    "rsi_21",

    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",

    "bb_width",
    "bb_position",

    "atr_pct",
    "adx",

    "stoch_k",
    "stoch_d",

    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
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
    "price_z_200",
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)

st.title("📊 Spot Scalping AI Scanner")

st.caption(
    "15M • +10% BUY / -10% SELL • "
    "Maximum 30 Minutes • BUY/SELL Only • "
    "No Order Execution"
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "symbol_cache" not in st.session_state:
    st.session_state.symbol_cache = {}

if "force_scan" not in st.session_state:
    st.session_state.force_scan = False


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "Spot-AI-Scanner/2.0"
            ),
            "Accept": "application/json",
        }
    )

    return session


HTTP = get_http()


# ============================================================
# EXCEPTIONS
# ============================================================

class DataProviderError(Exception):
    pass


class Binance451Error(Exception):
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc():

    return datetime.now(
        timezone.utc
    )


def safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def clean_ohlcv(
    df,
):

    if df is None or df.empty:
        return pd.DataFrame()

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise DataProviderError(
            "Eksik OHLCV kolonları: "
            + ", ".join(missing)
        )

    x = df.copy()

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
        errors="coerce",
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    x = x.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    x = x[
        x["close"] > 0
    ]

    x = x[
        x["high"] >= x["low"]
    ]

    x = (
        x.sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    return x


# ============================================================
# GENERIC HTTP
# ============================================================

def http_get(
    url,
    params=None,
):

    try:

        response = HTTP.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise DataProviderError(
            f"Bağlantı hatası: {exc}"
        )

    return response


# ============================================================
# BINANCE
# ============================================================

def binance_get(
    endpoint,
    params=None,
):

    response = http_get(
        BINANCE_URL + endpoint,
        params,
    )

    if response.status_code == 451:

        raise Binance451Error(
            "Binance HTTP 451: Bu çalışma "
            "ortamından Binance API erişimi "
            "bölgesel olarak kısıtlanıyor."
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
    show_spinner=False,
)
def get_binance_symbols():

    data = binance_get(
        "/api/v3/exchangeInfo"
    )

    symbols = []

    for item in data.get(
        "symbols",
        [],
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
            "quoteAsset"
        )

        if quote not in SUPPORTED_QUOTES:

            continue

        symbol = item.get(
            "symbol"
        )

        if not symbol:

            continue

        symbols.append(
            symbol
        )

    return sorted(
        list(
            set(symbols)
        )
    )


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_binance_history(
    symbol,
    candle_count,
):

    candle_count = int(
        min(
            candle_count,
            MAX_CANDLES,
        )
    )

    frames = []

    remaining = candle_count

    end_time = None

    while remaining > 0:

        limit = min(
            remaining,
            1000,
        )

        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "limit": limit,
        }

        if end_time is not None:

            params[
                "endTime"
            ] = end_time

        data = binance_get(
            "/api/v3/klines",
            params,
        )

        if not data:
            break

        rows = []

        for item in data:

            rows.append(
                {
                    "timestamp":
                        pd.to_datetime(
                            int(item[0]),
                            unit="ms",
                            utc=True,
                        ),

                    "open":
                        float(item[1]),

                    "high":
                        float(item[2]),

                    "low":
                        float(item[3]),

                    "close":
                        float(item[4]),

                    "volume":
                        float(item[5]),
                }
            )

        chunk = clean_ohlcv(
            pd.DataFrame(rows)
        )

        if chunk.empty:
            break

        frames.append(
            chunk
        )

        remaining -= len(
            chunk
        )

        oldest = (
            chunk[
                "timestamp"
            ].min()
        )

        end_time = (
            int(
                oldest.timestamp()
                * 1000
            )
            - 1
        )

        if len(chunk) < limit:

            break

        time.sleep(
            0.05
        )

    if not frames:

        raise DataProviderError(
            f"{symbol}: Binance candle verisi yok."
        )

    result = clean_ohlcv(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    return (
        result
        .tail(candle_count)
        .reset_index(drop=True)
    )


# ============================================================
# KRAKEN
# ============================================================

def kraken_get(
    endpoint,
    params=None,
):

    response = http_get(
        KRAKEN_URL + endpoint,
        params,
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
        [],
    )

    if errors:

        raise DataProviderError(
            "Kraken API: "
            + str(errors)
        )

    return data.get(
        "result",
        {},
    )


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_kraken_symbols():

    result = kraken_get(
        "/0/public/AssetPairs"
    )

    output = []

    for key, item in result.items():

        if not isinstance(
            item,
            dict,
        ):

            continue

        status = item.get(
            "status",
            "online",
        )

        if status != "online":
            continue

        altname = str(
            item.get(
                "altname",
                key,
            )
        )

        wsname = str(
            item.get(
                "wsname",
                altname,
            )
        )

        upper = wsname.upper()

        if not (
            upper.endswith("/USD")
            or upper.endswith("/USDT")
            or upper.endswith("/USDC")
        ):

            continue

        if ".D" in upper:
            continue

        output.append(
            {
                "api": altname,
                "display": wsname,
            }
        )

    return sorted(
        output,
        key=lambda x: x["display"],
    )


def get_kraken_history(
    api_symbol,
):

    result = kraken_get(
        "/0/public/OHLC",
        {
            "pair": api_symbol,
            "interval": 15,
        },
    )

    pair_key = next(
        (
            key
            for key in result.keys()
            if key != "last"
        ),
        None,
    )

    if pair_key is None:

        raise DataProviderError(
            f"Kraken {api_symbol}: "
            "OHLC verisi yok."
        )

    rows = []

    for item in result[
        pair_key
    ]:

        rows.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(item[0]),
                        unit="s",
                        utc=True,
                    ),

                "open":
                    float(item[1]),

                "high":
                    float(item[2]),

                "low":
                    float(item[3]),

                "close":
                    float(item[4]),

                "volume":
                    float(item[6]),
            }
        )

    df = clean_ohlcv(
        pd.DataFrame(rows)
    )

    if len(df) < MIN_CANDLES:

        raise DataProviderError(
            f"Kraken {api_symbol}: "
            f"{len(df)} candle geldi. "
            f"Minimum {MIN_CANDLES} gerekli."
        )

    return df


# ============================================================
# CSV
# ============================================================

def load_csv(
    uploaded_file,
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

    normalized = {}

    for column in df.columns:

        normalized[
            str(column)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        ] = column

    aliases = {
        "timestamp": [
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time",
            "opentime",
        ],

        "open": [
            "open",
            "o",
        ],

        "high": [
            "high",
            "h",
        ],

        "low": [
            "low",
            "l",
        ],

        "close": [
            "close",
            "c",
        ],

        "volume": [
            "volume",
            "vol",
            "v",
        ],
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
# INDICATORS
# ============================================================

def EMA(
    series,
    period,
):

    return (
        series
        .ewm(
            span=period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )


def RSI(
    series,
    period=14,
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    rs = (
        avg_gain
        / (
            avg_loss
            + 1e-12
        )
    )

    return (
        100
        - 100
        / (
            1 + rs
        )
    )


def ATR(
    df,
    period=14,
):

    previous_close = (
        df["close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["high"]
            - df["low"],

            (
                df["high"]
                - previous_close
            ).abs(),

            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )


def ADX(
    df,
    period=14,
):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (
                up_move
                > down_move
            )
            & (
                up_move
                > 0
            ),
            up_move,
            0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (
                down_move
                > up_move
            )
            & (
                down_move
                > 0
            ),
            down_move,
            0,
        ),
        index=df.index,
    )

    previous_close = (
        close.shift(1)
    )

    tr = pd.concat(
        [
            high - low,

            (
                high
                - previous_close
            ).abs(),

            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_value = (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    plus_di = (
        100
        * (
            plus_dm
            .ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )
        / (
            atr_value
            + 1e-12
        )
    )

    minus_di = (
        100
        * (
            minus_dm
            .ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )
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

    return (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )


# ============================================================
# FEATURES
# ============================================================

def build_features(
    df,
):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]
    volume = x["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    for n in [
        1,
        3,
        6,
        12,
        24,
        48,
        96,
    ]:

        x[
            f"ret_{n}"
        ] = close.pct_change(
            n
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    e5 = EMA(
        close,
        5,
    )

    e20 = EMA(
        close,
        20,
    )

    e50 = EMA(
        close,
        50,
    )

    e100 = EMA(
        close,
        100,
    )

    e200 = EMA(
        close,
        200,
    )

    for period, ema_value in [
        (5, e5),
        (20, e20),
        (50, e50),
        (100, e100),
        (200, e200),
    ]:

        x[
            f"ema_dist_{period}"
        ] = (
            close
            / (
                ema_value
                + 1e-12
            )
            - 1
        )

    x["ema_5_20"] = (
        e5
        / (
            e20
            + 1e-12
        )
        - 1
    )

    x["ema_20_50"] = (
        e20
        / (
            e50
            + 1e-12
        )
        - 1
    )

    x["ema_50_200"] = (
        e50
        / (
            e200
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
        21,
    ]:

        x[
            f"rsi_{period}"
        ] = RSI(
            close,
            period,
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_fast = EMA(
        close,
        12,
    )

    macd_slow = EMA(
        close,
        26,
    )

    macd_line = (
        macd_fast
        - macd_slow
    )

    macd_signal = EMA(
        macd_line,
        9,
    )

    macd_hist = (
        macd_line
        - macd_signal
    )

    x["macd_norm"] = (
        macd_line
        / (
            close
            + 1e-12
        )
    )

    x["macd_signal_norm"] = (
        macd_signal
        / (
            close
            + 1e-12
        )
    )

    x["macd_hist_norm"] = (
        macd_hist
        / (
            close
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb_mid = (
        close
        .rolling(20)
        .mean()
    )

    bb_std = (
        close
        .rolling(20)
        .std()
    )

    bb_upper = (
        bb_mid
        + 2 * bb_std
    )

    bb_lower = (
        bb_mid
        - 2 * bb_std
    )

    x["bb_width"] = (
        bb_upper
        - bb_lower
    ) / (
        bb_mid
        + 1e-12
    )

    x["bb_position"] = (
        close
        - bb_lower
    ) / (
        bb_upper
        - bb_lower
        + 1e-12
    )

    # --------------------------------------------------------
    # ATR / ADX
    # --------------------------------------------------------

    atr_value = ATR(
        x,
        14,
    )

    x["atr_pct"] = (
        atr_value
        / (
            close
            + 1e-12
        )
    )

    x["adx"] = ADX(
        x,
        14,
    )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    lowest = (
        low
        .rolling(14)
        .min()
    )

    highest = (
        high
        .rolling(14)
        .max()
    )

    x["stoch_k"] = (
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

    x["stoch_d"] = (
        x["stoch_k"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle_range = (
        high
        - low
    )

    body = (
        close
        - open_
    )

    x["body_pct"] = (
        body
        / (
            close
            + 1e-12
        )
    )

    x["range_pct"] = (
        candle_range
        / (
            close
            + 1e-12
        )
    )

    x["upper_wick_pct"] = (
        high
        - np.maximum(
            open_,
            close,
        )
    ) / (
        close
        + 1e-12
    )

    x["lower_wick_pct"] = (
        np.minimum(
            open_,
            close,
        )
        - low
    ) / (
        close
        + 1e-12
    )

    x["body_to_range"] = (
        body.abs()
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

    x["volume_ratio"] = (
        volume
        / (
            volume_mean
            + 1e-12
        )
    )

    x["volume_z"] = (
        volume
        - volume_mean
    ) / (
        volume_std
        + 1e-12
    )

    x["volume_change"] = (
        volume.pct_change()
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    one_bar_return = (
        close.pct_change()
    )

    for period in [
        12,
        24,
        48,
        96,
    ]:

        x[
            f"volatility_{period}"
        ] = (
            one_bar_return
            .rolling(period)
            .std()
        )

    # --------------------------------------------------------
    # DISTANCE TO HIGH/LOW
    # --------------------------------------------------------

    high48 = (
        high
        .rolling(48)
        .max()
    )

    low48 = (
        low
        .rolling(48)
        .min()
    )

    high200 = (
        high
        .rolling(200)
        .max()
    )

    low200 = (
        low
        .rolling(200)
        .min()
    )

    x["distance_high_48"] = (
        close
        / (
            high48
            + 1e-12
        )
        - 1
    )

    x["distance_low_48"] = (
        close
        / (
            low48
            + 1e-12
        )
        - 1
    )

    x["distance_high_200"] = (
        close
        / (
            high200
            + 1e-12
        )
        - 1
    )

    x["distance_low_200"] = (
        close
        / (
            low200
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

    x["drawdown"] = (
        close
        / (
            running_high
            + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # PRICE Z-SCORE
    # --------------------------------------------------------

    for period in [
        50,
        200,
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

        x[
            f"price_z_{period}"
        ] = (
            close
            - mean
        ) / (
            std
            + 1e-12
        )

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return x


# ============================================================
# TARGET
# ============================================================

def make_target(
    df,
):

    x = df.copy()

    entry = x["close"]

    high1 = (
        x["high"]
        .shift(-1)
    )

    high2 = (
        x["high"]
        .shift(-2)
    )

    low1 = (
        x["low"]
        .shift(-1)
    )

    low2 = (
        x["low"]
        .shift(-2)
    )

    future_high = pd.concat(
        [
            high1,
            high2,
        ],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [
            low1,
            low2,
        ],
        axis=1,
    ).min(axis=1)

    future_up = (
        future_high
        / entry
        - 1
    )

    future_down = (
        future_low
        / entry
        - 1
    )

    # Exact target hit information
    buy_hit = (
        future_up
        >= BUY_TARGET
    )

    sell_hit = (
        future_down
        <= SELL_TARGET
    )

    # --------------------------------------------------------
    # NO HOLD LABEL
    #
    # If +10% is reached => BUY.
    # If -10% is reached => SELL.
    # If neither is reached, whichever excursion is larger
    # determines the binary direction.
    # --------------------------------------------------------

    target = np.where(
        future_up
        >= future_down.abs(),
        1,
        0,
    ).astype(float)

    # If BUY target is hit and SELL target is not:
    target[
        buy_hit
        & ~sell_hit
    ] = 1

    # If SELL target is hit and BUY target is not:
    target[
        sell_hit
        & ~buy_hit
    ] = 0

    # If both hit, use larger excursion.
    both = (
        buy_hit
        & sell_hit
    )

    target[both] = np.where(
        future_up[both]
        >= future_down[both].abs(),
        1,
        0,
    )

    # Last 2 rows do not have complete future.
    target[
        len(target) - FUTURE_BARS:
    ] = np.nan

    x["target"] = target

    x["buy_target"] = (
        buy_hit.astype(float)
    )

    x["sell_target"] = (
        sell_hit.astype(float)
    )

    x["future_max_return"] = (
        future_up
    )

    x["future_min_return"] = (
        future_down
    )

    x["future_close_return"] = (
        x["close"]
        .shift(-2)
        / x["close"]
        - 1
    )

    return x


# ============================================================
# BALANCE
# ============================================================

def balance_training(
    X,
    y,
):

    temp = X.copy()

    temp["_target"] = (
        y.values
    )

    counts = (
        temp["_target"]
        .value_counts()
    )

    if len(counts) < 2:

        return (
            X,
            y,
        )

    max_count = int(
        counts.max()
    )

    rng = np.random.default_rng(
        42
    )

    parts = []

    for target_value in sorted(
        counts.index
    ):

        part = temp[
            temp["_target"]
            == target_value
        ].copy()

        if len(part) < max_count:

            extra_indices = (
                rng.choice(
                    len(part),
                    size=(
                        max_count
                        - len(part)
                    ),
                    replace=True,
                )
            )

            extra = part.iloc[
                extra_indices
            ]

            part = pd.concat(
                [
                    part,
                    extra,
                ],
                ignore_index=True,
            )

        parts.append(
            part
        )

    balanced = pd.concat(
        parts,
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42,
    ).reset_index(
        drop=True
    )

    return (
        balanced[
            FEATURES
        ],
        balanced[
            "_target"
        ].astype(int),
    )


# ============================================================
# MODEL
# ============================================================

def make_model():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(
                        96,
                        48,
                        24,
                    ),

                    activation="relu",

                    solver="adam",

                    alpha=0.0005,

                    batch_size=128,

                    learning_rate="adaptive",

                    learning_rate_init=0.001,

                    max_iter=250,

                    early_stopping=True,

                    validation_fraction=0.15,

                    n_iter_no_change=20,

                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# TRAIN / PREDICT
# ============================================================

def train_predict(
    df,
    train_limit,
):

    if len(df) < MIN_CANDLES:

        raise DataProviderError(
            f"Yetersiz candle: "
            f"{len(df)} / "
            f"minimum {MIN_CANDLES}"
        )

    features = build_features(
        df
    )

    labeled = make_target(
        features
    )

    usable = (
        labeled
        .dropna(
            subset=FEATURES
            + ["target"]
        )
        .copy()
    )

    # IMPORTANT:
    # We don't filter by buy_target/sell_target.
    # Therefore rows where neither ±10% happens still
    # contribute to the BUY/SELL binary model.

    if len(usable) < MIN_CANDLES:

        raise DataProviderError(
            "Feature/target temizliği "
            f"sonrası {len(usable)} satır kaldı."
        )

    usable["target"] = (
        usable["target"]
        .astype(int)
    )

    usable = (
        usable
        .tail(
            min(
                int(train_limit),
                len(usable),
            )
        )
        .reset_index(drop=True)
    )

    if usable["target"].nunique() < 2:

        raise DataProviderError(
            "Training datasında hem BUY "
            "hem SELL örneği bulunamadı."
        )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(
        usable
    )

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    if train_end < 200:

        raise DataProviderError(
            "Training datası çok küçük."
        )

    if validation_end <= train_end:

        validation_end = (
            train_end + 1
        )

    if validation_end >= n:

        validation_end = n - 1

    train = usable.iloc[
        :train_end
    ]

    validation = usable.iloc[
        train_end:
        validation_end
    ]

    test = usable.iloc[
        validation_end:
    ]

    if len(test) < 20:

        raise DataProviderError(
            "Test datası çok küçük."
        )

    X_train = (
        train[
            FEATURES
        ]
        .astype(float)
    )

    y_train = (
        train["target"]
        .astype(int)
    )

    X_test = (
        test[
            FEATURES
        ]
        .astype(float)
    )

    y_test = (
        test["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    (
        X_balanced,
        y_balanced,
    ) = balance_training(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    model = make_model()

    model.fit(
        X_balanced,
        y_balanced,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    predictions = (
        model.predict(
            X_test
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    # --------------------------------------------------------
    # LATEST
    # --------------------------------------------------------

    latest_rows = (
        labeled
        .dropna(
            subset=FEATURES
        )
    )

    if latest_rows.empty:

        raise DataProviderError(
            "Latest feature satırı bulunamadı."
        )

    latest = (
        latest_rows.iloc[-1]
    )

    X_latest = pd.DataFrame(
        [
            latest[
                FEATURES
            ].astype(float)
        ]
    )

    probabilities = (
        model
        .predict_proba(
            X_latest
        )[0]
    )

    class_probability = {
        int(cls): float(prob)
        for cls, prob in zip(
            model.classes_,
            probabilities,
        )
    }

    # 0 = SELL
    # 1 = BUY

    p_sell = (
        class_probability
        .get(
            0,
            0.0,
        )
    )

    p_buy = (
        class_probability
        .get(
            1,
            0.0,
        )
    )

    # --------------------------------------------------------
    # FORCE BUY / SELL
    # --------------------------------------------------------

    if p_buy >= p_sell:

        signal = "BUY"

        confidence = p_buy

    else:

        signal = "SELL"

        confidence = p_sell

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = float(
        latest["close"]
    )

    # --------------------------------------------------------
    # FUTURE DIAGNOSTICS
    # --------------------------------------------------------

    future_highs = pd.concat(
        [
            df["high"]
            .shift(-1),

            df["high"]
            .shift(-2),
        ],
        axis=1,
    )

    future_lows = pd.concat(
        [
            df["low"]
            .shift(-1),

            df["low"]
            .shift(-2),
        ],
        axis=1,
    )

    max_future_high = (
        future_highs
        .iloc[-1]
        .max()
    )

    min_future_low = (
        future_lows
        .iloc[-1]
        .min()
    )

    projected_upside = (
        max_future_high
        / current_price
        - 1
    )

    projected_downside = (
        min_future_low
        / current_price
        - 1
    )

    # --------------------------------------------------------
    # TECHNICAL SCORE
    # --------------------------------------------------------

    ret24 = safe_float(
        latest.get(
            "ret_24"
        ),
        0,
    )

    ema20_50 = safe_float(
        latest.get(
            "ema_20_50"
        ),
        0,
    )

    ema50_200 = safe_float(
        latest.get(
            "ema_50_200"
        ),
        0,
    )

    volume_ratio = safe_float(
        latest.get(
            "volume_ratio"
        ),
        1,
    )

    rsi14 = safe_float(
        latest.get(
            "rsi_14"
        ),
        50,
    )

    bb_position = safe_float(
        latest.get(
            "bb_position"
        ),
        0.5,
    )

    adx_value = safe_float(
        latest.get(
            "adx"
        ),
        0,
    )

    momentum_score = float(
        np.tanh(
            ret24 * 25
        )
    )

    short_trend_score = float(
        np.tanh(
            ema20_50 * 25
        )
    )

    long_trend_score = float(
        np.tanh(
            ema50_200 * 20
        )
    )

    volume_score = float(
        np.tanh(
            (
                volume_ratio
                - 1
            )
            / 2
        )
    )

    rsi_score = float(
        np.clip(
            (
                rsi14
                - 50
            )
            / 25,
            -1,
            1,
        )
    )

    bb_score = float(
        np.clip(
            (
                bb_position
                - 0.5
            )
            * 2,
            -1,
            1,
        )
    )

    trend_strength = float(
        np.clip(
            adx_value
            / 50,
            0,
            1,
        )
    )

    technical_direction = float(
        np.clip(
            (
                momentum_score
                * 0.25

                + short_trend_score
                * 0.25

                + long_trend_score
                * 0.15

                + volume_score
                * 0.10

                + rsi_score
                * 0.10

                + bb_score
                * 0.10
            )
            * (
                0.8
                + 0.2
                * trend_strength
            ),
            -1,
            1,
        )
    )

    # --------------------------------------------------------
    # FINAL SCORES
    # --------------------------------------------------------

    probability_direction = (
        p_buy
        - p_sell
    )

    buy_score = float(
        np.clip(
            50
            + probability_direction
            * 35

            + max(
                technical_direction,
                0,
            )
            * 20

            + max(
                projected_upside,
                0,
            )
            * 100,

            0,
            100,
        )
    )

    sell_score = float(
        np.clip(
            50
            - probability_direction
            * 35

            + max(
                -technical_direction,
                0,
            )
            * 20

            + max(
                -projected_downside,
                0,
            )
            * 100,

            0,
            100,
        )
    )

    ai_score = (
        buy_score
        if signal == "BUY"
        else sell_score
    )

    # --------------------------------------------------------
    # TARGET HIT RATES
    # --------------------------------------------------------

    buy_target_rate = float(
        labeled[
            "buy_target"
        ]
        .mean()
    )

    sell_target_rate = float(
        labeled[
            "sell_target"
        ]
        .mean()
    )

    future_close_return = (
        df["close"]
        .shift(-2)
        / df["close"]
        - 1
    ).iloc[-1]

    return {
        "signal": signal,

        "confidence": confidence,

        "p_buy": p_buy,

        "p_sell": p_sell,

        "ai_score": ai_score,

        "buy_score": buy_score,

        "sell_score": sell_score,

        "price": current_price,

        "target_up_pct": 10.0,

        "target_down_pct": -10.0,

        "horizon_minutes": 30,

        "projected_upside":
            projected_upside,

        "projected_downside":
            projected_downside,

        "future_close_return":
            future_close_return,

        "buy_target_rate":
            buy_target_rate,

        "sell_target_rate":
            sell_target_rate,

        "rsi14":
            rsi14,

        "bb_position":
            bb_position,

        "atr_pct":
            safe_float(
                latest.get(
                    "atr_pct"
                ),
                0,
            ),

        "adx":
            adx_value,

        "volume_ratio":
            volume_ratio,

        "technical_direction":
            technical_direction,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        "test_rows":
            len(test),

        "candles":
            len(df),

        "last_candle":
            str(
                df[
                    "timestamp"
                ].iloc[-1]
            ),
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)

provider = st.sidebar.selectbox(
    "Data Provider",
    [
        "Binance REST",
        "Kraken REST",
        "Historical CSV",
    ],
)

st.sidebar.markdown(
    "---"
)

all_symbols = st.sidebar.checkbox(
    "🌐 TÜM AKTİF SPOT SEMBOLLER",
    value=True,
)

max_symbols = st.sidebar.number_input(
    "Maksimum sembol",
    min_value=1,
    max_value=2000,
    value=200,
    step=50,
    help=(
        "Tüm semboller seçilirse bu sayı üst "
        "sınır olarak kullanılır."
    ),
)

candle_count = st.sidebar.number_input(
    "Historical Candles",
    min_value=500,
    max_value=MAX_CANDLES,
    value=5000,
    step=500,
)

train_limit = st.sidebar.number_input(
    "AI Training Candles",
    min_value=500,
    max_value=100000,
    value=20000,
    step=1000,
)

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    "**Timeframe:** 15 minutes"
)

st.sidebar.write(
    "**Horizon:** 30 minutes"
)

st.sidebar.write(
    "**BUY:** +10%"
)

st.sidebar.write(
    "**SELL:** -10%"
)

st.sidebar.write(
    "**HOLD:** ❌ Yok"
)


# ============================================================
# CSV UPLOAD
# ============================================================

uploaded_file = None

if provider == "Historical CSV":

    uploaded_file = st.sidebar.file_uploader(
        "📁 Historical CSV",
        type=["csv"],
    )


# ============================================================
# MANUAL SYMBOLS
# ============================================================

manual_symbols_text = st.sidebar.text_area(
    "Manuel semboller "
    "(Tüm semboller kapalıysa)",
    value=(
        "BTCUSDT\n"
        "ETHUSDT\n"
        "BNBUSDT\n"
        "SOLUSDT\n"
        "XRPUSDT"
    ),
    height=130,
)


# ============================================================
# AUTO REFRESH
# ============================================================

auto_refresh = st.sidebar.checkbox(
    "🔄 15 dakikada otomatik tarama",
    value=False,
)


# ============================================================
# SYMBOL LOADING
# ============================================================

def resolve_symbols():

    # --------------------------------------------------------
    # Binance
    # --------------------------------------------------------

    if provider == "Binance REST":

        if all_symbols:

            symbols = get_binance_symbols()

            if not symbols:

                raise DataProviderError(
                    "Binance aktif Spot sembol listesi boş."
                )

            return symbols[
                :int(max_symbols)
            ]

        manual = [
            x.strip().upper()
            for x in manual_symbols_text.splitlines()
            if x.strip()
        ]

        return manual

    # --------------------------------------------------------
    # Kraken
    # --------------------------------------------------------

    if provider == "Kraken REST":

        pairs = get_kraken_symbols()

        if all_symbols:

            return pairs[
                :int(max_symbols)
            ]

        manual = [
            x.strip().upper()
            for x in manual_symbols_text.splitlines()
            if x.strip()
        ]

        lookup = {}

        for pair in pairs:

            lookup[
                pair["display"].upper()
            ] = pair["api"]

            lookup[
                pair["api"].upper()
            ] = pair["api"]

        output = []

        for item in manual:

            if item in lookup:

                output.append(
                    {
                        "api":
                            lookup[item],

                        "display":
                            item,
                    }
                )

        return output

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if provider == "Historical CSV":

        if uploaded_file is None:

            raise DataProviderError(
                "CSV yüklenmedi."
            )

        # CSV single-symbol mode
        return [
            {
                "api": "CSV",
                "display": "CSV",
            }
        ]

    return []


# ============================================================
# LOAD SYMBOL DATA
# ============================================================

def get_symbol_data(
    symbol,
):

    if provider == "Binance REST":

        return get_binance_history(
            symbol,
            int(candle_count),
        )

    if provider == "Kraken REST":

        return get_kraken_history(
            symbol["api"]
        )

    if provider == "Historical CSV":

        return load_csv(
            uploaded_file
        )

    raise DataProviderError(
        "Geçersiz provider."
    )


# ============================================================
# SCAN
# ============================================================

def run_scan():

    st.session_state.errors = []

    st.session_state.results = (
        pd.DataFrame()
    )

    started = time.time()

    try:

        symbols = resolve_symbols()

    except Exception as exc:

        st.error(
            f"Sembol listesi alınamadı: {exc}"
        )

        return

    if not symbols:

        st.error(
            "Tarama için sembol bulunamadı."
        )

        return

    # --------------------------------------------------------
    # CSV special case
    # --------------------------------------------------------

    if provider == "Historical CSV":

        try:

            csv_df = load_csv(
                uploaded_file
            )

            if len(csv_df) < MIN_CANDLES:

                raise DataProviderError(
                    f"CSV'de {len(csv_df)} candle var. "
                    f"Minimum {MIN_CANDLES} gerekli."
                )

        except Exception as exc:

            st.error(
                f"CSV hatası: {exc}"
            )

            return

    else:

        csv_df = None

    results = []

    errors = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(
        symbols
    )

    for index, symbol in enumerate(
        symbols
    ):

        if provider == "Kraken REST":

            display_symbol = (
                symbol["display"]
            )

        elif provider == "Historical CSV":

            display_symbol = "CSV"

        else:

            display_symbol = symbol

        status.write(
            f"⏳ {display_symbol} "
            f"({index + 1}/{total})"
        )

        try:

            if provider == "Historical CSV":

                data = csv_df

            else:

                data = get_symbol_data(
                    symbol
                )

            if len(data) < MIN_CANDLES:

                raise DataProviderError(
                    f"{display_symbol}: "
                    f"{len(data)} candle. "
                    f"Minimum {MIN_CANDLES}."
                )

            result = train_predict(
                data,
                int(train_limit),
            )

            result["symbol"] = (
                display_symbol
            )

            result[
                "processing_seconds"
            ] = (
                time.time()
                - started
            )

            results.append(
                result
            )

        except Binance451Error as exc:

            errors.append(
                {
                    "symbol":
                        display_symbol,

                    "error":
                        str(exc),
                }
            )

            # Binance 451 tüm coinlerde
            # tekrar edeceği için durdur.
            if provider == "Binance REST":

                break

        except Exception as exc:

            errors.append(
                {
                    "symbol":
                        display_symbol,

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                }
            )

        progress.progress(
            (
                index + 1
            )
            / total
        )

    progress.empty()

    status.empty()

    if results:

        result_df = pd.DataFrame(
            results
        )

        # ----------------------------------------------------
        # BUY / SELL ONLY
        # ----------------------------------------------------

        result_df = result_df[
            result_df[
                "signal"
            ].isin(
                [
                    "BUY",
                    "SELL",
                ]
            )
        ]

        result_df = (
            result_df
            .sort_values(
                "ai_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        st.session_state.results = (
            result_df
        )

    st.session_state.errors = (
        errors
    )

    st.session_state.last_scan = (
        now_utc()
    )

    st.session_state.force_scan = False

    st.success(
        f"Tarama tamamlandı. "
        f"{len(results)} sonuç, "
        f"{len(errors)} hata."
    )


# ============================================================
# BUTTON
# ============================================================

if st.sidebar.button(
    "🚀 SCAN NOW",
    type="primary",
    use_container_width=True,
):

    run_scan()


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    st.markdown(
        "🔄 Otomatik tarama açık: "
        "15 dakika."
    )

    # Native fragment if available.
    if hasattr(
        st,
        "fragment",
    ):

        @st.fragment(
            run_every="15m"
        )
        def auto_scanner():

            if st.button(
                "🔄 Auto Scan",
                key="auto_scan_hidden",
            ):

                run_scan()

        # Do not show an additional button.
        # Streamlit reruns the fragment every 15m.
        auto_scanner()


# ============================================================
# STATUS
# ============================================================

st.divider()

c1, c2, c3, c4 = (
    st.columns(4)
)

with c1:

    st.metric(
        "Results",
        len(
            st.session_state.results
        ),
    )

with c2:

    if not st.session_state.results.empty:

        buy_count = int(
            (
                st.session_state.results[
                    "signal"
                ]
                == "BUY"
            ).sum()
        )

    else:

        buy_count = 0

    st.metric(
        "BUY",
        buy_count,
    )

with c3:

    if not st.session_state.results.empty:

        sell_count = int(
            (
                st.session_state.results[
                    "signal"
                ]
                == "SELL"
            ).sum()
        )

    else:

        sell_count = 0

    st.metric(
        "SELL",
        sell_count,
    )

with c4:

    if st.session_state.last_scan:

        last_scan_text = (
            st.session_state.last_scan
            .strftime(
                "%H:%M:%S"
            )
        )

    else:

        last_scan_text = "-"

    st.metric(
        "Last Scan",
        last_scan_text,
    )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.results
)

if results.empty:

    st.info(
        "Henüz sonuç yok. "
        "Sol taraftan SCAN NOW'a basın."
    )

else:

    # --------------------------------------------------------
    # TOP BUY
    # --------------------------------------------------------

    buy = (
        results[
            results[
                "signal"
            ]
            == "BUY"
        ]
        .sort_values(
            [
                "buy_score",
                "confidence",
            ],
            ascending=False,
        )
        .head(10)
        .copy()
    )

    st.subheader(
        "🟢 TOP 10 BUY"
    )

    if buy.empty:

        st.info(
            "BUY sonucu bulunamadı."
        )

    else:

        display = buy[
            [
                "symbol",
                "signal",
                "ai_score",
                "confidence",
                "p_buy",
                "p_sell",
                "projected_upside",
                "rsi14",
                "adx",
                "volume_ratio",
                "price",
                "test_balanced_accuracy",
                "candles",
            ]
        ].copy()

        display[
            "confidence"
        ] *= 100

        display[
            "p_buy"
        ] *= 100

        display[
            "p_sell"
        ] *= 100

        display[
            "projected_upside"
        ] *= 100

        display[
            "test_balanced_accuracy"
        ] *= 100

        display = display.rename(
            columns={
                "symbol":
                    "Symbol",

                "signal":
                    "Signal",

                "ai_score":
                    "AI Score",

                "confidence":
                    "Confidence %",

                "p_buy":
                    "BUY Probability %",

                "p_sell":
                    "SELL Probability %",

                "projected_upside":
                    "Projected +10% Move %",

                "rsi14":
                    "RSI 14",

                "adx":
                    "ADX",

                "volume_ratio":
                    "Volume Ratio",

                "price":
                    "Price",

                "test_balanced_accuracy":
                    "Test Balanced Accuracy %",

                "candles":
                    "Candles",
            }
        )

        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # TOP SELL
    # --------------------------------------------------------

    sell = (
        results[
            results[
                "signal"
            ]
            == "SELL"
        ]
        .sort_values(
            [
                "sell_score",
                "confidence",
            ],
            ascending=False,
        )
        .head(10)
        .copy()
    )

    st.subheader(
        "🔴 TOP 10 SELL"
    )

    if sell.empty:

        st.info(
            "SELL sonucu bulunamadı."
        )

    else:

        display = sell[
            [
                "symbol",
                "signal",
                "ai_score",
                "confidence",
                "p_buy",
                "p_sell",
                "projected_downside",
                "rsi14",
                "adx",
                "volume_ratio",
                "price",
                "test_balanced_accuracy",
                "candles",
            ]
        ].copy()

        display[
            "confidence"
        ] *= 100

        display[
            "p_buy"
        ] *= 100

        display[
            "p_sell"
        ] *= 100

        display[
            "projected_downside"
        ] *= 100

        display[
            "test_balanced_accuracy"
        ] *= 100

        display = display.rename(
            columns={
                "symbol":
                    "Symbol",

                "signal":
                    "Signal",

                "ai_score":
                    "AI Score",

                "confidence":
                    "Confidence %",

                "p_buy":
                    "BUY Probability %",

                "p_sell":
                    "SELL Probability %",

                "projected_downside":
                    "Projected -10% Move %",

                "rsi14":
                    "RSI 14",

                "adx":
                    "ADX",

                "volume_ratio":
                    "Volume Ratio",

                "price":
                    "Price",

                "test_balanced_accuracy":
                    "Test Balanced Accuracy %",

                "candles":
                    "Candles",
            }
        )

        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    with st.expander(
        "📋 TÜM BUY / SELL SONUÇLARI"
    ):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
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
        ].tolist(),
    )

    row = results[
        results[
            "symbol"
        ]
        == selected_symbol
    ].iloc[0]

    a, b, c, d = (
        st.columns(4)
    )

    with a:

        st.metric(
            "Signal",
            row["signal"],
        )

    with b:

        st.metric(
            "Confidence",
            f"{row['confidence'] * 100:.2f}%",
        )

    with c:

        st.metric(
            "AI Score",
            f"{row['ai_score']:.2f}/100",
        )

    with d:

        if row["signal"] == "BUY":

            target_move = (
                row["projected_upside"]
            )

        else:

            target_move = (
                row["projected_downside"]
            )

        st.metric(
            "30m Projected Move",
            f"{target_move * 100:.2f}%",
        )

    detail = pd.DataFrame(
        {
            "Metric": [
                "Signal",

                "Price",

                "BUY Probability",

                "SELL Probability",

                "BUY Score",

                "SELL Score",

                "AI Score",

                "Confidence",

                "Target",

                "Maximum Horizon",

                "Projected +10% Move",

                "Projected -10% Move",

                "Historical +10% Hit Rate",

                "Historical -10% Hit Rate",

                "Future 30m Close Return",

                "RSI 14",

                "Bollinger Position",

                "ATR %",

                "ADX",

                "Volume Ratio",

                "Technical Direction",

                "Test Accuracy",

                "Test Balanced Accuracy",

                "Training Rows",

                "Validation Rows",

                "Test Rows",

                "Candles",

                "Last Candle",

                "Processing Seconds",
            ],

            "Value": [
                row["signal"],

                row["price"],

                f"{row['p_buy'] * 100:.2f}%",

                f"{row['p_sell'] * 100:.2f}%",

                f"{row['buy_score']:.2f}/100",

                f"{row['sell_score']:.2f}/100",

                f"{row['ai_score']:.2f}/100",

                f"{row['confidence'] * 100:.2f}%",

                "+10% BUY / -10% SELL",

                "30 minutes",

                f"{row['projected_upside'] * 100:.2f}%",

                f"{row['projected_downside'] * 100:.2f}%",

                f"{row['buy_target_rate'] * 100:.3f}%",

                f"{row['sell_target_rate'] * 100:.3f}%",

                f"{row['future_close_return'] * 100:.3f}%",

                f"{row['rsi14']:.2f}",

                f"{row['bb_position']:.4f}",

                f"{row['atr_pct'] * 100:.3f}%",

                f"{row['adx']:.2f}",

                f"{row['volume_ratio']:.3f}",

                f"{row['technical_direction']:.4f}",

                f"{row['test_accuracy'] * 100:.2f}%",

                f"{row['test_balanced_accuracy'] * 100:.2f}%",

                row["train_rows"],

                row["validation_rows"],

                row["test_rows"],

                row["candles"],

                row["last_candle"],

                f"{row['processing_seconds']:.2f}",
            ],
        }
    )

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download Results CSV",

        data=(
            results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),

        file_name=(
            "spot_scalping_ai_results.csv"
        ),

        mime="text/csv",

        use_container_width=True,
    )


# ============================================================
# ERRORS
# ============================================================

if st.session_state.errors:

    st.divider()

    st.subheader(
        "⚠️ Symbol Errors"
    )

    error_df = pd.DataFrame(
        st.session_state.errors
    )

    st.dataframe(
        error_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Sistem nasıl çalışıyor?"
):

    st.markdown(
        """
## 🎯 Scalping Target

Sistem 15 dakikalık candle kullanır.

Her candle için sonraki:

**2 candle = 30 dakika**

incelenir.

### BUY

Eğer sonraki 30 dakika içinde:

**High >= Entry × 1.10**

olursa +10% BUY target hit kabul edilir.

### SELL

Eğer sonraki 30 dakika içinde:

**Low <= Entry × 0.90**

olursa -10% SELL target hit kabul edilir.

---

## ❌ HOLD YOK

Modelde HOLD sınıfı bulunmaz.

Her coin:

**BUY**

veya

**SELL**

olarak sınıflandırılır.

+10% veya -10% hedefinin ikisi de gerçekleşmezse,
30 dakikalık yukarı/aşağı excursion hangisi daha büyükse
model eğitim etiketi o yönde oluşturulur.

Böylece eğitim datası yalnızca çok nadir +10% / -10%
hareketlerinden oluşmaz.

---

## 🤖 AI

Model aşağıdaki bilgileri kullanır:

- EMA 5
- EMA 20
- EMA 50
- EMA 100
- EMA 200
- RSI 7
- RSI 14
- RSI 21
- MACD
- Bollinger Bands
- ATR
- ADX
- Stochastic
- Volume
- Volatility
- Candle structure
- Drawdown
- Price Z-score
- High/Low distance

Model:

**70% Training**

**15% Validation**

**15% Test**

şeklinde kronolojik olarak eğitilir.

---

## 🌐 TÜM SEMBOLLER

Binance seçildiğinde sistem:

`exchangeInfo`

üzerinden aktif:

- Spot
- TRADING
- USDT
- USDC

paritelerini otomatik olarak alır.

Örneğin:

BTCUSDT  
ETHUSDT  
BNBUSDT  
SOLUSDT  
XRPUSDT  
ADAUSDT  
DOGEUSDT  
...

ve listede bulunan diğer aktif Spot USDT/USDC
paritelerini tarar.

---

## ⚠️ Önemli

+10% hareketin 30 dakika içinde gerçekleşmesi
çok agresif bir hedeftir.

AI olasılığı kesin fiyat tahmini veya garanti kâr değildir.

Bu uygulama:

**EMİR GÖNDERMEZ.**

Sadece analiz ve BUY/SELL sıralaması üretir.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot Scalping AI Scanner • "
    "15M • +10% BUY / -10% SELL • "
    "30 Minute Horizon • "
    "BUY / SELL Only • "
    "No Order Execution"
)
