import io
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Spot Scalping AI Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Spot Scalping AI Scanner")

st.caption(
    "15M • +10% BUY / -10% SELL • "
    "Maximum 30 Minutes • BUY/SELL Only • "
    "Weekly Low + 3 Rising Candles"
)


# ============================================================
# CONFIG
# ============================================================

BINANCE_URL = "https://api.binance.com"
KRAKEN_URL = "https://api.kraken.com"

TIMEFRAME = "15m"

FUTURE_BARS = 2

HORIZON_MINUTES = 30

BUY_TARGET = 0.10

SELL_TARGET = -0.10

# 7 days × 24 hours × 4 candles/hour
WEEKLY_BARS = 7 * 24 * 4

# Weekly-low proximity
WEEKLY_LOW_DISTANCE = 0.015

MIN_CANDLES = 220

DEFAULT_CANDLES = 1000

DEFAULT_TRAIN_CANDLES = 10000

MAX_CANDLES = 500000

SUPPORTED_QUOTES = {
    "USDT",
    "USDC",
}

REQUEST_TIMEOUT = 20


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "ret24",

    "ema5_20",
    "ema20_50",
    "ema50_200",

    "dist20",
    "dist50",
    "dist200",

    "rsi7",
    "rsi14",
    "rsi21",

    "macd",
    "macd_signal",
    "macd_hist",

    "bb_position",
    "bb_width",

    "atr_pct",
    "adx",

    "stoch_k",
    "stoch_d",

    "body_pct",
    "range_pct",

    "upper_wick",
    "lower_wick",

    "volume_ratio",
    "volume_z",

    "volatility12",
    "volatility24",
    "volatility48",

    "price_z",

    "high_distance",
    "low_distance",
]


# ============================================================
# SESSION
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None


HTTP = requests.Session()

HTTP.headers.update(
    {
        "User-Agent":
            "Mozilla/5.0 SpotScalpingAI/2.0",
        "Accept":
            "application/json",
    }
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def request_json(
    url,
    params=None,
):

    try:

        response = HTTP.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Connection error: {exc}"
        )

    if response.status_code == 451:

        raise RuntimeError(
            "HTTP 451: API erişimi bu "
            "çalışma ortamında bölgesel olarak "
            "kısıtlanıyor."
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    try:

        return response.json()

    except Exception:

        raise RuntimeError(
            "API geçerli JSON döndürmedi."
        )


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

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Eksik OHLCV kolonu: {column}"
            )

    x = df.copy()

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x = x.dropna(
        subset=required
    )

    x = x[
        x["close"] > 0
    ]

    x = x[
        x["high"] >= x["low"]
    ]

    x = (
        x.sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_binance_symbols():

    data = request_json(
        BINANCE_URL
        + "/api/v3/exchangeInfo"
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
                "isSpotTradingAllowed",
                True,
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

        if symbol:

            symbols.append(
                symbol
            )

    return sorted(
        set(symbols)
    )


# ============================================================
# BINANCE HISTORICAL DATA
# ============================================================

def get_binance_history(
    symbol,
    requested_candles,
):

    requested_candles = min(
        int(requested_candles),
        MAX_CANDLES,
    )

    frames = []

    remaining = requested_candles

    end_time = None

    while remaining > 0:

        limit = min(
            remaining,
            1000,
        )

        params = {
            "symbol":
                symbol,

            "interval":
                TIMEFRAME,

            "limit":
                limit,
        }

        if end_time is not None:

            params[
                "endTime"
            ] = end_time

        data = request_json(
            BINANCE_URL
            + "/api/v3/klines",
            params,
        )

        if not data:

            break

        rows = []

        for candle in data:

            rows.append(
                {
                    "timestamp":
                        pd.to_datetime(
                            int(
                                candle[0]
                            ),
                            unit="ms",
                            utc=True,
                        ),

                    "open":
                        float(
                            candle[1]
                        ),

                    "high":
                        float(
                            candle[2]
                        ),

                    "low":
                        float(
                            candle[3]
                        ),

                    "close":
                        float(
                            candle[4]
                        ),

                    "volume":
                        float(
                            candle[5]
                        ),
                }
            )

        part = clean_ohlcv(
            pd.DataFrame(
                rows
            )
        )

        if part.empty:

            break

        frames.append(
            part
        )

        remaining -= len(part)

        oldest = part[
            "timestamp"
        ].min()

        end_time = (
            int(
                oldest.timestamp()
                * 1000
            )
            - 1
        )

        if len(part) < limit:

            break

        time.sleep(
            0.04
        )

    if not frames:

        raise RuntimeError(
            f"{symbol}: candle verisi alınamadı."
        )

    result = clean_ohlcv(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    return (
        result
        .tail(
            requested_candles
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# KRAKEN SYMBOLS
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_kraken_symbols():

    data = request_json(
        KRAKEN_URL
        + "/0/public/AssetPairs"
    )

    result = data.get(
        "result",
        {},
    )

    output = []

    for key, item in result.items():

        if not isinstance(
            item,
            dict,
        ):

            continue

        if item.get(
            "status",
            "online",
        ) != "online":

            continue

        name = str(
            item.get(
                "wsname",
                "",
            )
        )

        api_name = str(
            item.get(
                "altname",
                key,
            )
        )

        upper = name.upper()

        if not (
            upper.endswith(
                "/USD"
            )
            or upper.endswith(
                "/USDT"
            )
            or upper.endswith(
                "/USDC"
            )
        ):

            continue

        output.append(
            {
                "name":
                    name,

                "api":
                    api_name,
            }
        )

    return sorted(
        output,
        key=lambda x:
            x["name"],
    )


# ============================================================
# KRAKEN HISTORY
# ============================================================

def get_kraken_history(
    api_symbol,
):

    data = request_json(
        KRAKEN_URL
        + "/0/public/OHLC",
        {
            "pair":
                api_symbol,

            "interval":
                15,
        },
    )

    result = data.get(
        "result",
        {},
    )

    pair_key = next(
        (
            key
            for key in result
            if key != "last"
        ),
        None,
    )

    if pair_key is None:

        raise RuntimeError(
            f"{api_symbol}: OHLC verisi yok."
        )

    rows = []

    for candle in result[
        pair_key
    ]:

        rows.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(
                            candle[0]
                        ),
                        unit="s",
                        utc=True,
                    ),

                "open":
                    float(
                        candle[1]
                    ),

                "high":
                    float(
                        candle[2]
                    ),

                "low":
                    float(
                        candle[3]
                    ),

                "close":
                    float(
                        candle[4]
                    ),

                "volume":
                    float(
                        candle[6]
                    ),
            }
        )

    df = clean_ohlcv(
        pd.DataFrame(
            rows
        )
    )

    if len(df) < MIN_CANDLES:

        raise RuntimeError(
            f"{api_symbol}: "
            f"{len(df)} candle var."
        )

    return df


# ============================================================
# CSV
# ============================================================

def load_csv(
    uploaded,
):

    raw = uploaded.read()

    df = pd.read_csv(
        io.BytesIO(raw)
    )

    normalized = {
        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        ):
        column

        for column in df.columns
    }

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
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
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
        - (
            100
            / (
                1 + rs
            )
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

    true_range = pd.concat(
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
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
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

    plus_dm = np.where(
        (
            up_move
            > down_move
        )
        & (
            up_move > 0
        ),
        up_move,
        0,
    )

    minus_dm = np.where(
        (
            down_move
            > up_move
        )
        & (
            down_move > 0
        ),
        down_move,
        0,
    )

    previous_close = (
        close.shift(1)
    )

    true_range = pd.concat(
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
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index,
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
        / (
            atr_value
            + 1e-12
        )
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index,
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
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
        )
        .mean()
    )


# ============================================================
# WEEKLY LOW + 3 RISING CANDLES
# ============================================================

def weekly_low_three_rise(
    df,
):

    result = {
        "condition":
            False,

        "weekly_low":
            np.nan,

        "distance_from_low":
            np.nan,

        "rising_candles":
            0,

        "green_last":
            False,

        "near_weekly_low":
            False,

        "dip_index":
            -1,
    }

    if len(df) < (
        WEEKLY_BARS
        + 3
    ):

        return result

    # Son açık/oluşmakta olan mum yerine
    # tamamlanmış mumları kullan.
    x = df.iloc[
        :-1
    ].copy()

    if len(x) < (
        WEEKLY_BARS
        + 3
    ):

        return result

    # --------------------------------------------------------
    # SON 7 GÜN
    # --------------------------------------------------------

    weekly = x.tail(
        WEEKLY_BARS
    )

    weekly_low = float(
        weekly["low"].min()
    )

    result[
        "weekly_low"
    ] = weekly_low

    # Dip nerede?
    dip_position = int(
        weekly["low"]
        .values
        .argmin()
    )

    result[
        "dip_index"
    ] = dip_position

    # --------------------------------------------------------
    # SON 3 TAMAMLANMIŞ MUM
    # --------------------------------------------------------

    last3 = x.tail(3)

    candle1 = last3.iloc[0]
    candle2 = last3.iloc[1]
    candle3 = last3.iloc[2]

    rising1 = (
        candle2["close"]
        > candle1["close"]
    )

    rising2 = (
        candle3["close"]
        > candle2["close"]
    )

    green_last = (
        candle3["close"]
        > candle3["open"]
    )

    rising_count = 0

    if rising1:
        rising_count += 1

    if rising2:
        rising_count += 1

    result[
        "rising_candles"
    ] = (
        3
        if (
            rising1
            and rising2
        )
        else 0
    )

    result[
        "green_last"
    ] = bool(
        green_last
    )

    # --------------------------------------------------------
    # DİP BÖLGESİ
    # --------------------------------------------------------

    recent_low = float(
        last3["low"].min()
    )

    distance = (
        recent_low
        / weekly_low
        - 1
    )

    result[
        "distance_from_low"
    ] = distance

    near_low = (
        distance
        <= WEEKLY_LOW_DISTANCE
    )

    result[
        "near_weekly_low"
    ] = bool(
        near_low
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    result[
        "condition"
    ] = bool(
        rising1
        and rising2
        and green_last
        and near_low
    )

    return result


# ============================================================
# FEATURE ENGINEERING
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
    ]:

        x[
            f"ret{n}"
        ] = close.pct_change(
            n
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema5 = EMA(
        close,
        5,
    )

    ema20 = EMA(
        close,
        20,
    )

    ema50 = EMA(
        close,
        50,
    )

    ema200 = EMA(
        close,
        200,
    )

    x["ema5_20"] = (
        ema5
        / (
            ema20
            + 1e-12
        )
        - 1
    )

    x["ema20_50"] = (
        ema20
        / (
            ema50
            + 1e-12
        )
        - 1
    )

    x["ema50_200"] = (
        ema50
        / (
            ema200
            + 1e-12
        )
        - 1
    )

    x["dist20"] = (
        close
        / (
            ema20
            + 1e-12
        )
        - 1
    )

    x["dist50"] = (
        close
        / (
            ema50
            + 1e-12
        )
        - 1
    )

    x["dist200"] = (
        close
        / (
            ema200
            + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    x["rsi7"] = RSI(
        close,
        7,
    )

    x["rsi14"] = RSI(
        close,
        14,
    )

    x["rsi21"] = RSI(
        close,
        21,
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

    macd = (
        macd_fast
        - macd_slow
    )

    macd_signal = EMA(
        macd,
        9,
    )

    x["macd"] = (
        macd
        / (
            close
            + 1e-12
        )
    )

    x["macd_signal"] = (
        macd_signal
        / (
            close
            + 1e-12
        )
    )

    x["macd_hist"] = (
        macd
        - macd_signal
    ) / (
        close
        + 1e-12
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

    x["bb_position"] = (
        close
        - bb_lower
    ) / (
        bb_upper
        - bb_lower
        + 1e-12
    )

    x["bb_width"] = (
        bb_upper
        - bb_lower
    ) / (
        bb_mid
        + 1e-12
    )

    # --------------------------------------------------------
    # ATR
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

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    x["adx"] = ADX(
        x,
        14,
    )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    low14 = (
        low
        .rolling(14)
        .min()
    )

    high14 = (
        high
        .rolling(14)
        .max()
    )

    x["stoch_k"] = (
        100
        * (
            close
            - low14
        )
        / (
            high14
            - low14
            + 1e-12
        )
    )

    x["stoch_d"] = (
        x["stoch_k"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    candle_range = (
        high
        - low
    )

    x["body_pct"] = (
        close
        - open_
    ) / (
        close
        + 1e-12
    )

    x["range_pct"] = (
        candle_range
        / (
            close
            + 1e-12
        )
    )

    x["upper_wick"] = (
        high
        - pd.concat(
            [
                open_,
                close,
            ],
            axis=1,
        ).max(axis=1)
    ) / (
        close
        + 1e-12
    )

    x["lower_wick"] = (
        pd.concat(
            [
                open_,
                close,
            ],
            axis=1,
        ).min(axis=1)
        - low
    ) / (
        close
        + 1e-12
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

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    returns = (
        close.pct_change()
    )

    x["volatility12"] = (
        returns
        .rolling(12)
        .std()
    )

    x["volatility24"] = (
        returns
        .rolling(24)
        .std()
    )

    x["volatility48"] = (
        returns
        .rolling(48)
        .std()
    )

    # --------------------------------------------------------
    # PRICE Z
    # --------------------------------------------------------

    price_mean = (
        close
        .rolling(50)
        .mean()
    )

    price_std = (
        close
        .rolling(50)
        .std()
    )

    x["price_z"] = (
        close
        - price_mean
    ) / (
        price_std
        + 1e-12
    )

    # --------------------------------------------------------
    # HIGH / LOW DISTANCE
    # --------------------------------------------------------

    rolling_high = (
        high
        .rolling(48)
        .max()
    )

    rolling_low = (
        low
        .rolling(48)
        .min()
    )

    x["high_distance"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
        - 1
    )

    x["low_distance"] = (
        close
        / (
            rolling_low
            + 1e-12
        )
        - 1
    )

    return x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# AI TARGET
# ============================================================

def make_target(
    df,
):

    x = df.copy()

    entry = x["close"]

    future_high = pd.concat(
        [
            x["high"].shift(-1),
            x["high"].shift(-2),
        ],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [
            x["low"].shift(-1),
            x["low"].shift(-2),
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

    buy_hit = (
        future_up
        >= BUY_TARGET
    )

    sell_hit = (
        future_down
        <= SELL_TARGET
    )

    # --------------------------------------------------------
    # NO HOLD
    # --------------------------------------------------------

    target = np.where(
        future_up
        >= future_down.abs(),
        1,
        0,
    ).astype(float)

    target[
        buy_hit
        & ~sell_hit
    ] = 1

    target[
        sell_hit
        & ~buy_hit
    ] = 0

    # Both hit
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

    # Last 2 candles cannot have future data.
    target[
        -FUTURE_BARS:
    ] = np.nan

    x["target"] = target

    x["buy_hit"] = (
        buy_hit.astype(float)
    )

    x["sell_hit"] = (
        sell_hit.astype(float)
    )

    x["future_up"] = (
        future_up
    )

    x["future_down"] = (
        future_down
    )

    return x


# ============================================================
# BALANCE DATA
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

        return X, y

    max_count = int(
        counts.max()
    )

    rng = np.random.default_rng(
        42
    )

    parts = []

    for class_value in counts.index:

        part = temp[
            temp["_target"]
            == class_value
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

def create_model():

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
                        64,
                        32,
                    ),

                    activation="relu",

                    solver="adam",

                    alpha=0.0005,

                    batch_size=128,

                    learning_rate="adaptive",

                    learning_rate_init=0.001,

                    max_iter=200,

                    early_stopping=True,

                    validation_fraction=0.15,

                    n_iter_no_change=15,

                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# TECHNICAL FALLBACK
# ============================================================

def technical_fallback(
    df,
):

    f = build_features(
        df
    )

    last = f.iloc[-1]

    score = 0.0

    # Trend
    score += (
        np.tanh(
            float(
                last["ema20_50"]
            )
            * 30
        )
        * 0.25
    )

    score += (
        np.tanh(
            float(
                last["ema50_200"]
            )
            * 20
        )
        * 0.15
    )

    # Momentum
    score += (
        np.tanh(
            float(
                last["ret24"]
            )
            * 25
        )
        * 0.20
    )

    # RSI
    rsi_value = float(
        last["rsi14"]
    )

    score += (
        np.clip(
            (
                rsi_value
                - 50
            )
            / 25,
            -1,
            1,
        )
        * 0.15
    )

    # MACD
    score += (
        np.tanh(
            float(
                last["macd_hist"]
            )
            * 100
        )
        * 0.10
    )

    # Bollinger
    bb = float(
        last["bb_position"]
    )

    score += (
        np.clip(
            (
                bb
                - 0.5
            )
            * 2,
            -1,
            1,
        )
        * 0.10
    )

    # Volume
    volume_ratio = float(
        last["volume_ratio"]
    )

    score += (
        np.tanh(
            volume_ratio
            - 1
        )
        * 0.05
    )

    score = float(
        np.clip(
            score,
            -1,
            1,
        )
    )

    if score >= 0:

        signal = "BUY"

    else:

        signal = "SELL"

    confidence = (
        0.50
        + abs(score)
        * 0.49
    )

    return {
        "signal":
            signal,

        "confidence":
            confidence,

        "p_buy":
            (
                confidence
                if signal == "BUY"
                else 1 - confidence
            ),

        "p_sell":
            (
                confidence
                if signal == "SELL"
                else 1 - confidence
            ),

        "buy_score":
            (
                confidence * 100
                if signal == "BUY"
                else
                100
                - confidence * 100
            ),

        "sell_score":
            (
                confidence * 100
                if signal == "SELL"
                else
                100
                - confidence * 100
            ),

        "ai_score":
            confidence * 100,

        "mode":
            "TECHNICAL_FALLBACK",

        "test_accuracy":
            np.nan,

        "test_balanced_accuracy":
            np.nan,

        "train_rows":
            0,

        "validation_rows":
            0,

        "test_rows":
            0,

        "buy_hit_rate":
            np.nan,

        "sell_hit_rate":
            np.nan,
    }


# ============================================================
# AI PREDICTION
# ============================================================

def ai_predict(
    df,
    train_limit,
):

    try:

        feature_df = build_features(
            df
        )

        labeled = make_target(
            feature_df
        )

        usable = (
            labeled
            .dropna(
                subset=
                    FEATURES
                    + ["target"]
            )
            .copy()
        )

        if len(usable) < MIN_CANDLES:

            raise RuntimeError(
                "AI temizleme sonrası "
                f"{len(usable)} satır kaldı."
            )

        usable = (
            usable
            .tail(
                min(
                    int(train_limit),
                    len(usable),
                )
            )
            .reset_index(
                drop=True
            )
        )

        if (
            usable[
                "target"
            ].nunique()
            < 2
        ):

            raise RuntimeError(
                "Training datasında "
                "iki yönlü sınıf oluşmadı."
            )

        n = len(
            usable
        )

        train_end = int(
            n * 0.70
        )

        validation_end = int(
            n * 0.85
        )

        train = usable[
            :train_end
        ]

        validation = usable[
            train_end:
            validation_end
        ]

        test = usable[
            validation_end:
        ]

        if len(test) < 10:

            raise RuntimeError(
                "Test datası çok küçük."
            )

        X_train = train[
            FEATURES
        ].astype(float)

        y_train = train[
            "target"
        ].astype(int)

        X_test = test[
            FEATURES
        ].astype(float)

        y_test = test[
            "target"
        ].astype(int)

        X_train, y_train = (
            balance_training(
                X_train,
                y_train,
            )
        )

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

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

        latest = (
            labeled
            .dropna(
                subset=FEATURES
            )
            .iloc[-1]
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

        probability_map = {
            int(cls):
                float(probability)

            for cls, probability
            in zip(
                model.classes_,
                probabilities,
            )
        }

        p_sell = (
            probability_map.get(
                0,
                0.0,
            )
        )

        p_buy = (
            probability_map.get(
                1,
                0.0,
            )
        )

        if p_buy >= p_sell:

            signal = "BUY"

            confidence = p_buy

        else:

            signal = "SELL"

            confidence = p_sell

        directional = (
            p_buy
            - p_sell
        )

        buy_score = float(
            np.clip(
                50
                + directional * 50,
                0,
                100,
            )
        )

        sell_score = float(
            np.clip(
                50
                - directional * 50,
                0,
                100,
            )
        )

        return {
            "signal":
                signal,

            "confidence":
                confidence,

            "p_buy":
                p_buy,

            "p_sell":
                p_sell,

            "buy_score":
                buy_score,

            "sell_score":
                sell_score,

            "ai_score":
                confidence * 100,

            "mode":
                "AI",

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

            "buy_hit_rate":
                float(
                    labeled[
                        "buy_hit"
                    ].mean()
                ),

            "sell_hit_rate":
                float(
                    labeled[
                        "sell_hit"
                    ].mean()
                ),
        }

    except Exception:

        return technical_fallback(
            df
        )


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
    df,
    train_limit,
):

    start = time.time()

    df = clean_ohlcv(
        df
    )

    if len(df) < MIN_CANDLES:

        raise RuntimeError(
            f"{symbol}: "
            f"{len(df)} candle."
        )

    prediction = ai_predict(
        df,
        train_limit,
    )

    feature_df = build_features(
        df
    )

    last = feature_df.iloc[-1]

    # --------------------------------------------------------
    # WEEKLY LOW STRATEGY
    # --------------------------------------------------------

    weekly = (
        weekly_low_three_rise(
            df
        )
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = float(
        df["close"].iloc[-1]
    )

    # --------------------------------------------------------
    # NEXT 30 MINUTE RANGE
    # --------------------------------------------------------

    future_high = pd.concat(
        [
            df["high"].shift(-1),
            df["high"].shift(-2),
        ],
        axis=1,
    ).iloc[-1].max()

    future_low = pd.concat(
        [
            df["low"].shift(-1),
            df["low"].shift(-2),
        ],
        axis=1,
    ).iloc[-1].min()

    projected_upside = (
        future_high
        / price
        - 1
    )

    projected_downside = (
        future_low
        / price
        - 1
    )

    # --------------------------------------------------------
    # WEEKLY LOW BONUS
    # --------------------------------------------------------

    weekly_bonus = 0

    if weekly[
        "condition"
    ]:

        weekly_bonus = 20

        prediction[
            "buy_score"
        ] = min(
            100,
            prediction[
                "buy_score"
            ]
            + weekly_bonus,
        )

        prediction[
            "ai_score"
        ] = min(
            100,
            prediction[
                "ai_score"
            ]
            + weekly_bonus,
        )

        # Bu pattern BUY yönlü olduğu için
        # BUY olasılığını da güçlendiriyoruz.
        prediction[
            "p_buy"
        ] = min(
            0.99,
            prediction[
                "p_buy"
            ]
            + 0.10,
        )

        # Eğer model zaten SELL diyorsa,
        # güçlü weekly-low pattern'i BUY'a çevirebilir.
        if (
            prediction[
                "buy_score"
            ]
            >= prediction[
                "sell_score"
            ]
        ):

            prediction[
                "signal"
            ] = "BUY"

            prediction[
                "confidence"
            ] = prediction[
                "p_buy"
            ]

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    # HOLD kesinlikle yok.
    if (
        prediction[
            "buy_score"
        ]
        >= prediction[
            "sell_score"
        ]
    ):

        final_signal = "BUY"

    else:

        final_signal = "SELL"

    prediction[
        "signal"
    ] = final_signal

    if final_signal == "BUY":

        prediction[
            "confidence"
        ] = max(
            prediction[
                "confidence"
            ],
            prediction[
                "buy_score"
            ]
            / 100,
        )

    else:

        prediction[
            "confidence"
        ] = max(
            prediction[
                "confidence"
            ],
            prediction[
                "sell_score"
            ]
            / 100,
        )

    return {
        "symbol":
            symbol,

        "signal":
            prediction[
                "signal"
            ],

        "mode":
            prediction[
                "mode"
            ],

        "price":
            price,

        "confidence":
            prediction[
                "confidence"
            ],

        "p_buy":
            prediction[
                "p_buy"
            ],

        "p_sell":
            prediction[
                "p_sell"
            ],

        "ai_score":
            prediction[
                "ai_score"
            ],

        "buy_score":
            prediction[
                "buy_score"
            ],

        "sell_score":
            prediction[
                "sell_score"
            ],

        # Weekly low
        "weekly_low":
            weekly[
                "weekly_low"
            ],

        "weekly_low_3_rise":
            weekly[
                "condition"
            ],

        "rising_candles":
            weekly[
                "rising_candles"
            ],

        "green_last":
            weekly[
                "green_last"
            ],

        "near_weekly_low":
            weekly[
                "near_weekly_low"
            ],

        "distance_from_weekly_low":
            weekly[
                "distance_from_low"
            ],

        "weekly_bonus":
            weekly_bonus,

        # 30m
        "projected_upside":
            projected_upside,

        "projected_downside":
            projected_downside,

        # Technical
        "rsi":
            float(
                last["rsi14"]
            ),

        "adx":
            float(
                last["adx"]
            ),

        "bb_position":
            float(
                last[
                    "bb_position"
                ]
            ),

        "atr_pct":
            float(
                last["atr_pct"]
            ),

        "volume_ratio":
            float(
                last[
                    "volume_ratio"
                ]
            ),

        # Test
        "test_accuracy":
            prediction[
                "test_accuracy"
            ],

        "test_balanced_accuracy":
            prediction[
                "test_balanced_accuracy"
            ],

        "train_rows":
            prediction[
                "train_rows"
            ],

        "validation_rows":
            prediction[
                "validation_rows"
            ],

        "test_rows":
            prediction[
                "test_rows"
            ],

        "buy_hit_rate":
            prediction[
                "buy_hit_rate"
            ],

        "sell_hit_rate":
            prediction[
                "sell_hit_rate"
            ],

        "candles":
            len(df),

        "last_candle":
            str(
                df[
                    "timestamp"
                ].iloc[-1]
            ),

        "processing_seconds":
            time.time()
            - start,
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

all_symbols = st.sidebar.checkbox(
    "🌐 TÜM AKTİF COİNLER",
    value=True,
)

max_symbols = st.sidebar.number_input(
    "Coin üst sınırı",
    min_value=1,
    max_value=5000,
    value=5000,
    step=100,
)

candle_count = st.sidebar.number_input(
    "Historical 15M Candles",
    min_value=300,
    max_value=MAX_CANDLES,
    value=DEFAULT_CANDLES,
    step=100,
)

train_limit = st.sidebar.number_input(
    "AI Training Candles",
    min_value=300,
    max_value=100000,
    value=DEFAULT_TRAIN_CANDLES,
    step=500,
)

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    "⏱ Timeframe: **15M**"
)

st.sidebar.write(
    "📅 Weekly lookback: **672 candles**"
)

st.sidebar.write(
    "📈 Rising candles: **3**"
)

st.sidebar.write(
    "🎯 BUY target: **+10% / 30M**"
)

st.sidebar.write(
    "🎯 SELL target: **-10% / 30M**"
)

st.sidebar.write(
    "❌ HOLD: **YOK**"
)

st.sidebar.write(
    "📍 Weekly low distance: **≤ 1.5%**"
)


# ============================================================
# MANUAL SYMBOLS
# ============================================================

manual_symbols = st.sidebar.text_area(
    "Manuel semboller",
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
# CSV
# ============================================================

uploaded_file = None

if provider == "Historical CSV":

    uploaded_file = st.sidebar.file_uploader(
        "📁 15M Historical CSV",
        type=[
            "csv"
        ],
    )


# ============================================================
# SYMBOL RESOLUTION
# ============================================================

def resolve_symbols():

    if provider == "Binance REST":

        symbols = (
            get_binance_symbols()
        )

        if all_symbols:

            return symbols[
                :int(max_symbols)
            ]

        return [
            x.strip().upper()
            for x
            in manual_symbols.splitlines()
            if x.strip()
        ]

    if provider == "Kraken REST":

        pairs = (
            get_kraken_symbols()
        )

        if all_symbols:

            return pairs[
                :int(max_symbols)
            ]

        wanted = {
            x.strip().upper()
            for x
            in manual_symbols.splitlines()
            if x.strip()
        }

        return [
            pair
            for pair in pairs
            if (
                pair[
                    "name"
                ].upper()
                in wanted

                or

                pair[
                    "api"
                ].upper()
                in wanted
            )
        ]

    return [
        {
            "name":
                "CSV",

            "api":
                "CSV",
        }
    ]


# ============================================================
# DATA LOADER
# ============================================================

def load_symbol_data(
    symbol,
):

    if provider == "Binance REST":

        return (
            symbol,

            get_binance_history(
                symbol,
                int(
                    candle_count
                ),
            ),
        )

    if provider == "Kraken REST":

        return (
            symbol[
                "name"
            ],

            get_kraken_history(
                symbol[
                    "api"
                ]
            ),
        )

    if uploaded_file is None:

        raise RuntimeError(
            "CSV yüklenmedi."
        )

    return (
        "CSV",

        load_csv(
            uploaded_file
        ),
    )


# ============================================================
# FULL SCAN
# ============================================================

def run_scan():

    st.session_state.results = (
        pd.DataFrame()
    )

    st.session_state.errors = []

    try:

        symbols = (
            resolve_symbols()
        )

    except Exception as exc:

        st.error(
            f"Coin listesi alınamadı: "
            f"{exc}"
        )

        return

    if not symbols:

        st.error(
            "Hiç coin bulunamadı."
        )

        return

    st.info(
        f"🌐 {len(symbols):,} aktif "
        "Spot sembol taranacak."
    )

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

        if provider == "Binance REST":

            symbol_name = symbol

        elif provider == "Kraken REST":

            symbol_name = symbol[
                "name"
            ]

        else:

            symbol_name = "CSV"

        status.write(
            f"🔎 {symbol_name} "
            f"({index + 1}/{total})"
        )

        try:

            name, df = (
                load_symbol_data(
                    symbol
                )
            )

            result = analyze_symbol(
                name,
                df,
                int(
                    train_limit
                ),
            )

            results.append(
                result
            )

        except Exception as exc:

            errors.append(
                {
                    "symbol":
                        symbol_name,

                    "error":
                        str(exc),
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
        # HOLD YOK
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

        # Weekly pattern first,
        # then AI score.
        result_df[
            "weekly_priority"
        ] = (
            result_df[
                "weekly_low_3_rise"
            ]
            .astype(int)
        )

        result_df = (
            result_df
            .sort_values(
                [
                    "weekly_priority",
                    "ai_score",
                    "confidence",
                ],
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
        datetime.now(
            timezone.utc
        )
    )

    st.success(
        f"Tarama tamamlandı: "
        f"{len(results):,} coin sonuçlandı, "
        f"{len(errors):,} hata."
    )


# ============================================================
# SCAN BUTTON
# ============================================================

if st.sidebar.button(
    "🚀 TÜM COİNLERİ TARA",
    type="primary",
    use_container_width=True,
):

    run_scan()


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.results
)

if results.empty:

    st.info(
        "Henüz tarama yapılmadı."
    )

else:

    # ========================================================
    # SUMMARY
    # ========================================================

    buy_count = int(
        (
            results[
                "signal"
            ]
            == "BUY"
        ).sum()
    )

    sell_count = int(
        (
            results[
                "signal"
            ]
            == "SELL"
        ).sum()
    )

    weekly_count = int(
        results[
            "weekly_low_3_rise"
        ].sum()
    )

    ai_count = int(
        (
            results[
                "mode"
            ]
            == "AI"
        ).sum()
    )

    fallback_count = int(
        (
            results[
                "mode"
            ]
            == "TECHNICAL_FALLBACK"
        ).sum()
    )

    a, b, c, d, e = (
        st.columns(5)
    )

    with a:

        st.metric(
            "Toplam",
            f"{len(results):,}",
        )

    with b:

        st.metric(
            "BUY",
            buy_count,
        )

    with c:

        st.metric(
            "SELL",
            sell_count,
        )

    with d:

        st.metric(
            "Weekly Low + 3 Rise",
            weekly_count,
        )

    with e:

        st.metric(
            "AI",
            ai_count,
        )


    # ========================================================
    # WEEKLY LOW SIGNALS
    # ========================================================

    st.subheader(
        "🔥 WEEKLY LOW + 3 RISING CANDLES"
    )

    weekly_signals = (
        results[
            results[
                "weekly_low_3_rise"
            ]
            == True
        ]
        .sort_values(
            [
                "buy_score",
                "confidence",
            ],
            ascending=False,
        )
        .copy()
    )

    if weekly_signals.empty:

        st.warning(
            "Son 1 haftanın dip bölgesinden "
            "3 ardışık yükselen mum şartını "
            "karşılayan coin bulunamadı."
        )

    else:

        weekly_display = (
            weekly_signals[
                [
                    "symbol",
                    "signal",
                    "mode",
                    "price",
                    "weekly_low",
                    "distance_from_weekly_low",
                    "rising_candles",
                    "green_last",
                    "buy_score",
                    "confidence",
                    "p_buy",
                    "rsi",
                    "adx",
                    "volume_ratio",
                ]
            ]
            .copy()
        )

        weekly_display[
            "distance_from_weekly_low"
        ] *= 100

        weekly_display[
            "confidence"
        ] *= 100

        weekly_display[
            "p_buy"
        ] *= 100

        weekly_display = (
            weekly_display.rename(
                columns={
                    "symbol":
                        "Symbol",

                    "signal":
                        "Signal",

                    "mode":
                        "Model",

                    "price":
                        "Price",

                    "weekly_low":
                        "Weekly Low",

                    "distance_from_weekly_low":
                        "Distance From Low %",

                    "rising_candles":
                        "Rising Candles",

                    "green_last":
                        "Last Green",

                    "buy_score":
                        "BUY Score",

                    "confidence":
                        "Confidence %",

                    "p_buy":
                        "BUY Probability %",

                    "rsi":
                        "RSI",

                    "adx":
                        "ADX",

                    "volume_ratio":
                        "Volume Ratio",
                }
            )
        )

        st.dataframe(
            weekly_display.round(4),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # TOP BUY
    # ========================================================

    st.subheader(
        "🟢 TOP 10 BUY"
    )

    buys = (
        results[
            results[
                "signal"
            ]
            == "BUY"
        ]
        .sort_values(
            [
                "weekly_low_3_rise",
                "buy_score",
                "confidence",
            ],
            ascending=False,
        )
        .head(10)
        .copy()
    )

    if buys.empty:

        st.info(
            "BUY bulunamadı."
        )

    else:

        display = buys[
            [
                "symbol",
                "signal",
                "mode",
                "buy_score",
                "confidence",
                "p_buy",
                "weekly_low_3_rise",
                "rising_candles",
                "distance_from_weekly_low",
                "projected_upside",
                "rsi",
                "adx",
                "price",
            ]
        ].copy()

        display[
            "confidence"
        ] *= 100

        display[
            "p_buy"
        ] *= 100

        display[
            "distance_from_weekly_low"
        ] *= 100

        display[
            "projected_upside"
        ] *= 100

        display = display.rename(
            columns={
                "symbol":
                    "Symbol",

                "signal":
                    "Signal",

                "mode":
                    "Model",

                "buy_score":
                    "BUY Score",

                "confidence":
                    "Confidence %",

                "p_buy":
                    "BUY Probability %",

                "weekly_low_3_rise":
                    "Weekly Low + 3 Rise",

                "rising_candles":
                    "Rising Candles",

                "distance_from_weekly_low":
                    "Distance From Low %",

                "projected_upside":
                    "Projected Upside %",

                "rsi":
                    "RSI",

                "adx":
                    "ADX",

                "price":
                    "Price",
            }
        )

        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # TOP SELL
    # ========================================================

    st.subheader(
        "🔴 TOP 10 SELL"
    )

    sells = (
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

    if sells.empty:

        st.info(
            "SELL bulunamadı."
        )

    else:

        display = sells[
            [
                "symbol",
                "signal",
                "mode",
                "sell_score",
                "confidence",
                "p_sell",
                "weekly_low_3_rise",
                "projected_downside",
                "rsi",
                "adx",
                "price",
            ]
        ].copy()

        display[
            "confidence"
        ] *= 100

        display[
            "p_sell"
        ] *= 100

        display[
            "projected_downside"
        ] *= 100

        display = display.rename(
            columns={
                "symbol":
                    "Symbol",

                "signal":
                    "Signal",

                "mode":
                    "Model",

                "sell_score":
                    "SELL Score",

                "confidence":
                    "Confidence %",

                "p_sell":
                    "SELL Probability %",

                "weekly_low_3_rise":
                    "Weekly Low + 3 Rise",

                "projected_downside":
                    "Projected Downside %",

                "rsi":
                    "RSI",

                "adx":
                    "ADX",

                "price":
                    "Price",
            }
        )

        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # ALL COINS
    # ========================================================

    with st.expander(
        "📋 TÜM COİNLER"
    ):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # DETAIL
    # ========================================================

    st.subheader(
        "🔎 Coin Detail"
    )

    selected_symbol = st.selectbox(
        "Coin seç",
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

    a, b, c, d, e = (
        st.columns(5)
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
            f"{row['ai_score']:.2f}",
        )

    with d:

        st.metric(
            "Weekly Pattern",
            (
                "YES"
                if row[
                    "weekly_low_3_rise"
                ]
                else "NO"
            ),
        )

    with e:

        st.metric(
            "Price",
            f"{row['price']:.8g}",
        )


    detail = pd.DataFrame(
        {
            "Metric": [
                "Symbol",

                "Final Signal",

                "Model",

                "Price",

                "BUY Probability",

                "SELL Probability",

                "BUY Score",

                "SELL Score",

                "AI Score",

                "Confidence",

                "Weekly Low",

                "Weekly Low + 3 Rising",

                "Rising Candles",

                "Last Candle Green",

                "Near Weekly Low",

                "Distance From Weekly Low",

                "Weekly Bonus",

                "30M BUY Target",

                "30M SELL Target",

                "Projected Upside",

                "Projected Downside",

                "RSI 14",

                "ADX",

                "Bollinger Position",

                "ATR %",

                "Volume Ratio",

                "Historical +10% Hit Rate",

                "Historical -10% Hit Rate",

                "Test Accuracy",

                "Test Balanced Accuracy",

                "Training Rows",

                "Validation Rows",

                "Test Rows",

                "Candles",

                "Last Candle",
            ],

            "Value": [
                row["symbol"],

                row["signal"],

                row["mode"],

                row["price"],

                f"{row['p_buy'] * 100:.2f}%",

                f"{row['p_sell'] * 100:.2f}%",

                f"{row['buy_score']:.2f}",

                f"{row['sell_score']:.2f}",

                f"{row['ai_score']:.2f}",

                f"{row['confidence'] * 100:.2f}%",

                row["weekly_low"],

                (
                    "YES"
                    if row[
                        "weekly_low_3_rise"
                    ]
                    else "NO"
                ),

                row[
                    "rising_candles"
                ],

                (
                    "YES"
                    if row[
                        "green_last"
                    ]
                    else "NO"
                ),

                (
                    "YES"
                    if row[
                        "near_weekly_low"
                    ]
                    else "NO"
                ),

                f"{row['distance_from_weekly_low'] * 100:.3f}%",

                f"+{row['weekly_bonus']:.0f}",

                "+10%",

                "-10%",

                f"{row['projected_upside'] * 100:.3f}%",

                f"{row['projected_downside'] * 100:.3f}%",

                f"{row['rsi']:.2f}",

                f"{row['adx']:.2f}",

                f"{row['bb_position']:.4f}",

                f"{row['atr_pct'] * 100:.3f}%",

                f"{row['volume_ratio']:.3f}",

                (
                    f"{row['buy_hit_rate'] * 100:.3f}%"
                    if pd.notna(
                        row[
                            "buy_hit_rate"
                        ]
                    )
                    else "-"
                ),

                (
                    f"{row['sell_hit_rate'] * 100:.3f}%"
                    if pd.notna(
                        row[
                            "sell_hit_rate"
                        ]
                    )
                    else "-"
                ),

                (
                    f"{row['test_accuracy'] * 100:.2f}%"
                    if pd.notna(
                        row[
                            "test_accuracy"
                        ]
                    )
                    else "Fallback"
                ),

                (
                    f"{row['test_balanced_accuracy'] * 100:.2f}%"
                    if pd.notna(
                        row[
                            "test_balanced_accuracy"
                        ]
                    )
                    else "Fallback"
                ),

                row[
                    "train_rows"
                ],

                row[
                    "validation_rows"
                ],

                row[
                    "test_rows"
                ],

                row[
                    "candles"
                ],

                row[
                    "last_candle"
                ],
            ],
        }
    )

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv_output = (
        results
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "⬇️ Sonuçları CSV indir",

        data=csv_output,

        file_name=(
            "spot_scalping_all_coins.csv"
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
        "⚠️ Veri alınamayan coinler"
    )

    st.caption(
        "Bu liste yalnızca borsadan OHLCV "
        "verisi alınamayan veya sembolü geçersiz "
        "olan coinleri gösterir. AI eğitimi "
        "başarısız olduğunda coin silinmez; "
        "TECHNICAL_FALLBACK kullanılır."
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
    "ℹ️ Strateji detayları"
):

    st.markdown(
        """
### 1. Haftalık dip

15M grafik kullanılır.

Son:

**672 × 15 dakika = 7 gün**

içindeki en düşük `LOW` bulunur.

### 2. Dip bölgesi

Son 3 tamamlanmış mumun en düşük seviyesi,
haftalık dip seviyesinin en fazla:

**%1.5**

üzerindeyse coin dip bölgesinde kabul edilir.

### 3. Üç yükselen mum

Son 3 tamamlanmış mum:

```text
Close 1 < Close 2 < Close 3
