import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Binance Spot AI Scanner - 15M BUY / SELL"

BINANCE_URL = "https://api.binance.com"
KRAKEN_URL = "https://api.kraken.com"

TIMEFRAME = "15m"

FUTURE_BARS = 2
HORIZON_MINUTES = 30

BUY_TARGET = 0.10
SELL_TARGET = -0.10

WEEKLY_BARS = 672
WEEKLY_LOW_DISTANCE = 0.015

MIN_CANDLES = 250

DEFAULT_CANDLES = 1000
DEFAULT_TRAIN_CANDLES = 5000

MAX_CANDLES = 500000

REQUEST_TIMEOUT = 20

SUPPORTED_QUOTES = {
    "USDT",
    "USDC"
}


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
    "low_distance"
]


session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 CryptoScanner/1.0"
    }
)


# ============================================================
# BASIC HELPERS
# ============================================================

def request_json(url, params=None, timeout=REQUEST_TIMEOUT):

    response = session.get(
        url,
        params=params,
        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


def clean_ohlcv(df):

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in required:

        if column not in result.columns:
            return pd.DataFrame()

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    if "timestamp" in result.columns:

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce"
        )

        result = result.dropna(
            subset=["timestamp"]
        )

        result = result.set_index(
            "timestamp"
        )

    result = result[
        ~result.index.duplicated(
            keep="last"
        )
    ]

    result = result.sort_index()

    result = result.dropna(
        subset=required
    )

    return result


def empty_result(symbol, error=""):

    return {
        "symbol": symbol,
        "signal": "ERROR",
        "score": 0.0,
        "buy_score": 0.0,
        "sell_score": 0.0,
        "p_buy": np.nan,
        "p_sell": np.nan,
        "price": np.nan,
        "weekly_low": np.nan,
        "distance_from_weekly_low": np.nan,
        "three_rise": False,
        "green_last": False,
        "dip_before_rise": False,
        "ai_accuracy": np.nan,
        "ai_balanced_accuracy": np.nan,
        "rows": 0,
        "error": error
    }


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_binance_symbols():

    data = request_json(
        BINANCE_URL + "/api/v3/exchangeInfo"
    )

    symbols = []

    for item in data.get("symbols", []):

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        quote = item.get(
            "quoteAsset",
            ""
        )

        if quote not in SUPPORTED_QUOTES:
            continue

        symbols.append(
            item["symbol"]
        )

    return sorted(
        set(symbols)
    )


# ============================================================
# BINANCE HISTORICAL DATA
# ============================================================

def get_binance_history(
    symbol,
    candles=1000
):

    candles = min(
        int(candles),
        MAX_CANDLES
    )

    rows = []

    end_time = None

    while len(rows) < candles:

        limit = min(
            1000,
            candles - len(rows)
        )

        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "limit": limit
        }

        if end_time is not None:

            params["endTime"] = end_time

        data = request_json(
            BINANCE_URL + "/api/v3/klines",
            params=params
        )

        if not data:
            break

        rows = data + rows

        oldest = int(
            data[0][0]
        )

        end_time = oldest - 1

        if len(data) < limit:
            break

        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    rows = rows[-candles:]

    records = []

    for item in rows:

        records.append(
            {
                "timestamp": pd.to_datetime(
                    int(item[0]),
                    unit="ms",
                    utc=True
                ),
                "open": safe_float(item[1]),
                "high": safe_float(item[2]),
                "low": safe_float(item[3]),
                "close": safe_float(item[4]),
                "volume": safe_float(item[5])
            }
        )

    return clean_ohlcv(
        pd.DataFrame(records)
    )


# ============================================================
# KRAKEN
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_kraken_symbols():

    try:

        data = request_json(
            KRAKEN_URL + "/0/public/AssetPairs"
        )

        pairs = []

        for key, item in data.get(
            "result",
            {}
        ).items():

            if ".d" in key.lower():
                continue

            base = str(
                item.get(
                    "base",
                    ""
                )
            ).upper()

            quote = str(
                item.get(
                    "quote",
                    ""
                )
            ).upper()

            if quote in {
                "USDT",
                "USDC"
            }:

                pairs.append(key)

        return sorted(
            set(pairs)
        )

    except Exception:

        return []


def get_kraken_history(
    symbol,
    candles=1000
):

    try:

        data = request_json(
            KRAKEN_URL + "/0/public/OHLC",
            params={
                "pair": symbol,
                "interval": 15
            }
        )

        result = data.get(
            "result",
            {}
        )

        keys = [
            key
            for key in result.keys()
            if key != "last"
        ]

        if not keys:
            return pd.DataFrame()

        values = result[
            keys[0]
        ][-candles:]

        records = []

        for item in values:

            records.append(
                {
                    "timestamp": pd.to_datetime(
                        int(item[0]),
                        unit="s",
                        utc=True
                    ),
                    "open": safe_float(item[1]),
                    "high": safe_float(item[2]),
                    "low": safe_float(item[3]),
                    "close": safe_float(item[4]),
                    "volume": safe_float(item[6])
                }
            )

        return clean_ohlcv(
            pd.DataFrame(records)
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# CSV
# ============================================================

def load_csv(uploaded):

    if uploaded is None:
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            uploaded
        )

    except Exception:

        return pd.DataFrame()

    columns = {
        str(c).lower().strip(): c
        for c in df.columns
    }

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

    renamed = {}

    for target, names in aliases.items():

        for name in names:

            if name in columns:

                renamed[
                    columns[name]
                ] = target

                break

    df = df.rename(
        columns=renamed
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if not all(
        x in df.columns
        for x in required
    ):

        return pd.DataFrame()

    if "timestamp" not in df.columns:

        df["timestamp"] = pd.RangeIndex(
            start=0,
            stop=len(df)
        )

    return clean_ohlcv(
        df
    )


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    value = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return value.fillna(50)


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                -
                previous_close
            ).abs(),
            (
                df["low"]
                -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def adx(df, period=14):

    up = df["high"].diff()

    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0.0
        ),
        index=df.index
    )

    atr_value = atr(
        df,
        period
    ).replace(
        0,
        np.nan
    )

    plus_di = (
        100
        *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr_value
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr_value
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        *
        (
            plus_di - minus_di
        ).abs()
        /
        denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean().fillna(0)


def stochastic(
    df,
    period=14,
    smooth=3
):

    low_n = df["low"].rolling(
        period
    ).min()

    high_n = df["high"].rolling(
        period
    ).max()

    denominator = (
        high_n - low_n
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        *
        (
            df["close"] - low_n
        )
        /
        denominator
    )

    d = k.rolling(
        smooth
    ).mean()

    return (
        k.fillna(50),
        d.fillna(50)
    )


# ============================================================
# WEEKLY LOW + THREE RISING CANDLES
# ============================================================

def weekly_low_three_rise(df):

    result = {
        "condition": False,
        "weekly_low": np.nan,
        "distance_from_low": np.nan,
        "rising_candles": 0,
        "green_last": False,
        "near_weekly_low": False,
        "dip_before_rise": False
    }

    if len(df) < WEEKLY_BARS + 5:
        return result

    # Ignore currently forming candle.
    x = df.iloc[:-1].copy()

    if len(x) < WEEKLY_BARS + 3:
        return result

    weekly = x.tail(
        WEEKLY_BARS
    )

    last3 = x.tail(3)

    weekly_low = float(
        weekly["low"].min()
    )

    result["weekly_low"] = weekly_low

    if weekly_low <= 0:
        return result

    dip_idx = weekly["low"].idxmin()

    first_rise_idx = last3.index[0]

    dip_before_rise = (
        dip_idx < first_rise_idx
    )

    result[
        "dip_before_rise"
    ] = bool(
        dip_before_rise
    )

    c1 = last3.iloc[0]
    c2 = last3.iloc[1]
    c3 = last3.iloc[2]

    rise1 = (
        float(c2["close"])
        >
        float(c1["close"])
    )

    rise2 = (
        float(c3["close"])
        >
        float(c2["close"])
    )

    green_last = (
        float(c3["close"])
        >
        float(c3["open"])
    )

    recent_low = float(
        last3["low"].min()
    )

    distance = (
        recent_low
        /
        weekly_low
        -
        1
    )

    near_low = (
        distance
        <=
        WEEKLY_LOW_DISTANCE
    )

    result[
        "rising_candles"
    ] = 3 if (
        rise1 and rise2
    ) else 0

    result[
        "green_last"
    ] = bool(
        green_last
    )

    result[
        "distance_from_low"
    ] = distance

    result[
        "near_weekly_low"
    ] = bool(
        near_low
    )

    result[
        "condition"
    ] = bool(
        rise1
        and rise2
        and green_last
        and near_low
        and dip_before_rise
    )

    return result


# ============================================================
# FEATURES
# ============================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret6"] = close.pct_change(6)
    x["ret12"] = close.pct_change(12)
    x["ret24"] = close.pct_change(24)

    e5 = ema(close, 5)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)

    x["ema5_20"] = (
        e5 / e20 - 1
    )

    x["ema20_50"] = (
        e20 / e50 - 1
    )

    x["ema50_200"] = (
        e50 / e200 - 1
    )

    x["dist20"] = (
        close / e20 - 1
    )

    x["dist50"] = (
        close / e50 - 1
    )

    x["dist200"] = (
        close / e200 - 1
    )

    x["rsi7"] = rsi(
        close,
        7
    )

    x["rsi14"] = rsi(
        close,
        14
    )

    x["rsi21"] = rsi(
        close,
        21
    )

    macd_line = (
        ema(close, 12)
        -
        ema(close, 26)
    )

    macd_signal = ema(
        macd_line,
        9
    )

    x["macd"] = macd_line
    x["macd_signal"] = macd_signal

    x["macd_hist"] = (
        macd_line
        -
        macd_signal
    )

    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    upper = middle + 2 * std
    lower = middle - 2 * std

    band_width = (
        upper - lower
    ).replace(
        0,
        np.nan
    )

    x["bb_position"] = (
        close - lower
    ) / band_width

    x["bb_width"] = (
        band_width
        /
        middle.replace(
            0,
            np.nan
        )
    )

    atr_value = atr(
        x,
        14
    )

    x["atr_pct"] = (
        atr_value
        /
        close.replace(
            0,
            np.nan
        )
    )

    x["adx"] = adx(
        x,
        14
    )

    stoch_k, stoch_d = stochastic(
        x
    )

    x["stoch_k"] = stoch_k
    x["stoch_d"] = stoch_d

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan
    )

    body = (
        close - x["open"]
    ).abs()

    x["body_pct"] = (
        body
        /
        close.replace(
            0,
            np.nan
        )
    )

    x["range_pct"] = (
        candle_range
        /
        close.replace(
            0,
            np.nan
        )
    )

    x["upper_wick"] = (
        high
        -
        np.maximum(
            x["open"],
            close
        )
    ) / candle_range

    x["lower_wick"] = (
        np.minimum(
            x["open"],
            close
        )
        -
        low
    ) / candle_range

    volume_mean = volume.rolling(
        20
    ).mean()

    volume_std = volume.rolling(
        20
    ).std()

    x["volume_ratio"] = (
        volume
        /
        volume_mean.replace(
            0,
            np.nan
        )
    )

    x["volume_z"] = (
        volume
        -
        volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    x["volatility12"] = (
        x["ret1"].rolling(
            12
        ).std()
    )

    x["volatility24"] = (
        x["ret1"].rolling(
            24
        ).std()
    )

    x["volatility48"] = (
        x["ret1"].rolling(
            48
        ).std()
    )

    price_mean = close.rolling(
        48
    ).mean()

    price_std = close.rolling(
        48
    ).std()

    x["price_z"] = (
        close
        -
        price_mean
    ) / price_std.replace(
        0,
        np.nan
    )

    x["high_distance"] = (
        high / close - 1
    )

    x["low_distance"] = (
        low / close - 1
    )

    return x


# ============================================================
# TARGET
# ============================================================

def make_target(df):

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    target = np.full(
        len(df),
        np.nan
    )

    for i in range(
        len(df) - FUTURE_BARS
    ):

        entry = close[i]

        future_high = np.max(
            high[
                i + 1:
                i + 1 + FUTURE_BARS
            ]
        )

        future_low = np.min(
            low[
                i + 1:
                i + 1 + FUTURE_BARS
            ]
        )

        up = (
            future_high
            /
            entry
            -
            1
        )

        down = (
            future_low
            /
            entry
            -
            1
        )

        hit_buy = (
            up >= BUY_TARGET
        )

        hit_sell = (
            down <= SELL_TARGET
        )

        if hit_buy and not hit_sell:

            target[i] = 1

        elif hit_sell and not hit_buy:

            target[i] = 0

        elif hit_buy and hit_sell:

            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

        else:

            # HOLD yok.
            # Hangi yön daha büyük hareket yaptıysa
            # o yön seçiliyor.
            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

    return target


# ============================================================
# BALANCE
# ============================================================

def balance_training(X, y):

    y = np.asarray(y)

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    if len(classes) < 2:
        return X, y

    target_count = int(
        np.max(counts)
    )

    rng = np.random.default_rng(
        42
    )

    parts_x = []
    parts_y = []

    for cls, count in zip(
        classes,
        counts
    ):

        indexes = np.where(
            y == cls
        )[0]

        if count < target_count:

            extra = rng.choice(
                indexes,
                size=(
                    target_count
                    -
                    count
                ),
                replace=True
            )

            use = np.concatenate(
                [
                    indexes,
                    extra
                ]
            )

        else:

            use = indexes

        parts_x.append(
            X[use]
        )

        parts_y.append(
            y[use]
        )

    X2 = np.vstack(
        parts_x
    )

    y2 = np.concatenate(
        parts_y
    )

    order = rng.permutation(
        len(y2)
    )

    return (
        X2[order],
        y2[order]
    )


# ============================================================
# AI
# ============================================================

def create_model():

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
                        64,
                        32
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=128,
                    learning_rate_init=0.001,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=12,
                    random_state=42
                )
            )
        ]
    )


def ai_predict(
    df,
    train_candles
):

    features = build_features(
        df
    )

    target = make_target(
        df
    )

    valid = (
        features[FEATURES]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )

    target_series = pd.Series(
        target,
        index=df.index
    )

    y_valid = target_series.reindex(
        valid.index
    )

    mask = y_valid.notna()

    X_train = valid.loc[
        mask
    ].values

    y_train = y_valid.loc[
        mask
    ].astype(int).values

    if len(X_train) < 400:

        raise ValueError(
            "AI training data is too small."
        )

    if (
        train_candles > 0
        and
        len(X_train) > train_candles
    ):

        X_train = X_train[
            -train_candles:
        ]

        y_train = y_train[
            -train_candles:
        ]

    if len(
        np.unique(y_train)
    ) < 2:

        raise ValueError(
            "Only one AI target class exists."
        )

    split = int(
        len(X_train)
        *
        0.80
    )

    if split < 200:

        raise ValueError(
            "Validation data is too small."
        )

    X_fit = X_train[
        :split
    ]

    y_fit = y_train[
        :split
    ]

    X_val = X_train[
        split:
    ]

    y_val = y_train[
        split:
    ]

    X_fit, y_fit = balance_training(
        X_fit,
        y_fit
    )

    model = create_model()

    model.fit(
        X_fit,
        y_fit
    )

    predictions = model.predict(
        X_val
    )

    accuracy = float(
        np.mean(
            predictions
            ==
            y_val
        )
    )

    recalls = []

    for cls in [
        0,
        1
    ]:

        class_mask = (
            y_val == cls
        )

        if class_mask.sum() > 0:

            recalls.append(
                float(
                    np.mean(
                        predictions[
                            class_mask
                        ]
                        ==
                        cls
                    )
                )
            )

    balanced_accuracy = (
        float(
            np.mean(
                recalls
            )
        )
        if recalls
        else
        np.nan
    )

    latest = (
        features.iloc[
            [-2]
        ][FEATURES]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    if latest.isna().any().any():

        raise ValueError(
            "Latest feature row contains NaN."
        )

    probabilities = (
        model.predict_proba(
            latest.values
        )[0]
    )

    classes_model = list(
        model.classes_
    )

    p_buy = 0.0
    p_sell = 0.0

    for cls, probability in zip(
        classes_model,
        probabilities
    ):

        if int(cls) == 1:

            p_buy = float(
                probability
            )

        else:

            p_sell = float(
                probability
            )

    return {
        "p_buy": p_buy,
        "p_sell": p_sell,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    }


# ============================================================
# TECHNICAL FALLBACK
# ============================================================

def technical_fallback(df):

    features = build_features(
        df
    )

    row = features.iloc[
        -2
    ]

    buy = 0.0
    sell = 0.0

    if row["ema5_20"] > 0:
        buy += 12
    else:
        sell += 12

    if row["ema20_50"] > 0:
        buy += 12
    else:
        sell += 12

    if row["ema50_200"] > 0:
        buy += 10
    else:
        sell += 10

    if row["rsi14"] < 35:
        buy += 12

    elif row["rsi14"] > 65:
        sell += 12

    if row["macd_hist"] > 0:
        buy += 8
    else:
        sell += 8

    if row["bb_position"] < 0.25:
        buy += 12

    elif row["bb_position"] > 0.75:
        sell += 12

    if row["stoch_k"] < 25:
        buy += 8

    elif row["stoch_k"] > 75:
        sell += 8

    if row["close"] > row["open"]:
        buy += 8
    else:
        sell += 8

    if row["adx"] >= 20:

        if row["ema20_50"] > 0:
            buy += 6
        else:
            sell += 6

    return (
        buy,
        sell
    )


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    df,
    train_candles
):

    if df.empty:

        return empty_result(
            symbol,
            "No candle data."
        )

    if len(df) < MIN_CANDLES:

        return empty_result(
            symbol,
            f"Not enough candles: {len(df)}"
        )

    try:

        pattern = weekly_low_three_rise(
            df
        )

        try:

            ai = ai_predict(
                df,
                train_candles
            )

        except Exception:

            ai = None

        technical_buy, technical_sell = (
            technical_fallback(
                df
            )
        )

        if ai is not None:

            buy_score = (
                ai["p_buy"]
                *
                100
            )

            sell_score = (
                ai["p_sell"]
                *
                100
            )

            buy_score += (
                technical_buy
                *
                0.35
            )

            sell_score += (
                technical_sell
                *
                0.35
            )

            p_buy = ai[
                "p_buy"
            ]

            p_sell = ai[
                "p_sell"
            ]

            accuracy = ai[
                "accuracy"
            ]

            balanced = ai[
                "balanced_accuracy"
            ]

        else:

            total = (
                technical_buy
                +
                technical_sell
            )

            if total <= 0:

                p_buy = 0.5
                p_sell = 0.5

            else:

                p_buy = (
                    technical_buy
                    /
                    total
                )

                p_sell = (
                    technical_sell
                    /
                    total
                )

            buy_score = technical_buy
            sell_score = technical_sell

            accuracy = np.nan
            balanced = np.nan

        # ====================================================
        # WEEKLY LOW + 3 RISE BUY BOOST
        # ====================================================

        if pattern["condition"]:

            buy_score += 20

            p_buy = min(
                1.0,
                p_buy + 0.10
            )

        # ====================================================
        # FINAL SIGNAL
        # ====================================================

        if buy_score >= sell_score:

            signal = "BUY"

        else:

            signal = "SELL"

        price = float(
            df["close"].iloc[-2]
        )

        score = max(
            buy_score,
            sell_score
        )

        return {
            "symbol": symbol,
            "signal": signal,
            "score": round(
                score,
                2
            ),
            "buy_score": round(
                buy_score,
                2
            ),
            "sell_score": round(
                sell_score,
                2
            ),
            "p_buy": round(
                float(p_buy),
                4
            ),
            "p_sell": round(
                float(p_sell),
                4
            ),
            "price": price,
            "weekly_low": pattern[
                "weekly_low"
            ],
            "distance_from_weekly_low": pattern[
                "distance_from_low"
            ],
            "three_rise": pattern[
                "condition"
            ],
            "green_last": pattern[
                "green_last"
            ],
            "dip_before_rise": pattern[
                "dip_before_rise"
            ],
            "ai_accuracy": accuracy,
            "ai_balanced_accuracy": balanced,
            "rows": len(df),
            "error": ""
        }

    except Exception as e:

        return empty_result(
            symbol,
            str(e)
        )


# ============================================================
# DATA SOURCE
# ============================================================

def get_symbols(
    source
):

    if source == "Binance":

        return get_binance_symbols()

    if source == "Kraken":

        return get_kraken_symbols()

    return []


def load_symbol_data(
    source,
    symbol,
    candles
):

    if source == "Binance":

        return get_binance_history(
            symbol,
            candles
        )

    if source == "Kraken":

        return get_kraken_history(
            symbol,
            candles
        )

    return pd.DataFrame()


# ============================================================
# FULL SCANNER
# ============================================================

def run_scan(
    source,
    symbols,
    candles,
    train_candles
):

    results = []
    errors = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(
        symbols
    )

    for i, symbol in enumerate(
        symbols,
        start=1
    ):

        status.write(
            f"Scanning {i}/{total}: {symbol}"
        )

        try:

            df = load_symbol_data(
                source,
                symbol,
                candles
            )

            if df.empty:

                raise ValueError(
                    "No data returned."
                )

            result = analyze_symbol(
                symbol,
                df,
                train_candles
            )

            if result["error"]:

                errors.append(
                    result
                )

            else:

                results.append(
                    result
                )

        except Exception as e:

            errors.append(
                empty_result(
                    symbol,
                    str(e)
                )
            )

        progress.progress(
            int(
                i
                /
                max(total, 1)
                *
                100
            )
        )

    status.empty()
    progress.empty()

    return (
        pd.DataFrame(results),
        pd.DataFrame(errors)
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide"
)

st.title(
    "📈 Binance Spot AI Scanner"
)

st.caption(
    "15M | BUY / SELL only | 30-minute horizon | +10% / -10% target"
)

st.warning(
    "Bu uygulama emir göndermez. Yalnızca analiz/sinyal üretir. "
    "+10% hareketin 30 dakika içinde gerçekleşmesi çok agresif bir "
    "hedef olduğundan sinyal garanti değildir."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Scanner Settings"
    )

    source = st.selectbox(
        "Data Source",
        [
            "Binance",
            "Kraken"
        ]
    )

    candles = st.number_input(
        "15M Candles",
        min_value=700,
        max_value=MAX_CANDLES,
        value=DEFAULT_CANDLES,
        step=100
    )

    train_candles = st.number_input(
        "AI Training Candles",
        min_value=500,
        max_value=MAX_CANDLES,
        value=DEFAULT_TRAIN_CANDLES,
        step=500
    )

    auto_refresh = st.checkbox(
        "Refresh page every 15 minutes",
        value=False
    )

    st.divider()

    st.write(
        "AI TARGET"
    )

    st.write(
        "BUY: next 2 candles reach +10%"
    )

    st.write(
        "SELL: next 2 candles reach -10%"
    )

    st.write(
        "HOLD: disabled"
    )

    st.divider()

    st.write(
        "WEEKLY BUY PATTERN"
    )

    st.write(
        "672 completed 15M candles"
    )

    st.write(
        "Weekly minimum before final 3 candles"
    )

    st.write(
        "3 consecutive rising closes"
    )

    st.write(
        "Last candle green"
    )

    st.write(
        "Last 3-candle low within 1.5% of weekly low"
    )

    st.write(
        "Pattern adds +20 BUY score"
    )


# ============================================================
# SYMBOLS
# ============================================================

try:

    symbols = get_symbols(
        source
    )

except Exception as e:

    symbols = []

    st.error(
        f"{source} symbol list could not be loaded: {e}"
    )


if source == "Binance" and symbols:

    st.success(
        f"{len(symbols):,} Binance Spot USDT/USDC symbols found."
    )


if source == "Kraken" and symbols:

    st.success(
        f"{len(symbols):,} Kraken symbols found."
    )


if not symbols:

    st.error(
        "No symbols available. Binance may return HTTP 451/403 "
        "from the current hosting environment."
    )

    st.stop()


# ============================================================
# SCAN SCOPE
# ============================================================

selection_mode = st.radio(
    "Scan Scope",
    [
        "ALL SYMBOLS",
        "SELECT SYMBOLS"
    ],
    horizontal=True
)


if selection_mode == "ALL SYMBOLS":

    selected_symbols = symbols

else:

    selected_symbols = st.multiselect(
        "Symbols",
        symbols,
        default=symbols[:20]
    )


st.write(
    f"Selected symbols: **{len(selected_symbols):,}**"
)


scan_button = st.button(
    "🚀 START FULL SCAN",
    type="primary",
    use_container_width=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    try:

        @st.fragment(run_every="15m")
        def refresh_fragment():

            st.caption(
                "Auto refresh active - page refreshes every 15 minutes."
            )

        refresh_fragment()

    except Exception:

        st.caption(
            "Automatic refresh is unavailable. "
            "Use the browser refresh button."
        )


# ============================================================
# RUN SCAN
# ============================================================

if scan_button:

    if not selected_symbols:

        st.error(
            "Select at least one symbol."
        )

        st.stop()

    start_time = time.time()

    with st.spinner(
        f"Scanning {len(selected_symbols):,} symbols..."
    ):

        results_df, errors_df = run_scan(
            source=source,
            symbols=selected_symbols,
            candles=int(candles),
            train_candles=int(train_candles)
        )

    elapsed = (
        time.time()
        -
        start_time
    )

    st.session_state[
        "results_df"
    ] = results_df

    st.session_state[
        "errors_df"
    ] = errors_df

    st.session_state[
        "scan_time"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    st.session_state[
        "elapsed"
    ] = elapsed


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "results_df" in st.session_state:

    results_df = st.session_state[
        "results_df"
    ]

    errors_df = st.session_state.get(
        "errors_df",
        pd.DataFrame()
    )

    elapsed = st.session_state.get(
        "elapsed",
        0
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Valid Results",
        f"{len(results_df):,}"
    )

    c2.metric(
        "Errors",
        f"{len(errors_df):,}"
    )

    if not results_df.empty:

        buy_count = int(
            (
                results_df["signal"]
                ==
                "BUY"
            ).sum()
        )

        sell_count = int(
            (
                results_df["signal"]
                ==
                "SELL"
            ).sum()
        )

    else:

        buy_count = 0
        sell_count = 0

    c3.metric(
        "BUY",
        f"{buy_count:,}"
    )

    c4.metric(
        "SELL",
        f"{sell_count:,}"
    )

    st.caption(
        f"Scan duration: {elapsed:.1f} sec | "
        f"Last scan: {st.session_state.get('scan_time', '')}"
    )


    # ========================================================
    # WEEKLY PATTERN
    # ========================================================

    if not results_df.empty:

        pattern_df = results_df[
            results_df[
                "three_rise"
            ]
            ==
            True
        ].copy()

        st.subheader(
            "🔥 Weekly Low + 3 Rising Candles"
        )

        if pattern_df.empty:

            st.info(
                "No symbol currently satisfies the complete weekly-low pattern."
            )

        else:

            pattern_df = pattern_df.sort_values(
                [
                    "buy_score",
                    "p_buy"
                ],
                ascending=False
            )

            display = pattern_df[
                [
                    "symbol",
                    "signal",
                    "price",
                    "weekly_low",
                    "distance_from_weekly_low",
                    "buy_score",
                    "sell_score",
                    "p_buy",
                    "ai_accuracy",
                    "rows"
                ]
            ].copy()

            display[
                "distance_from_weekly_low"
            ] = (
                display[
                    "distance_from_weekly_low"
                ]
                *
                100
            )

            display[
                "p_buy"
            ] = (
                display[
                    "p_buy"
                ]
                *
                100
            )

            display = display.rename(
                columns={
                    "symbol": "Symbol",
                    "signal": "Signal",
                    "price": "Price",
                    "weekly_low": "Weekly Low",
                    "distance_from_weekly_low": "Distance %",
                    "buy_score": "BUY Score",
                    "sell_score": "SELL Score",
                    "p_buy": "AI BUY %",
                    "ai_accuracy": "AI Accuracy",
                    "rows": "Candles"
                }
            )

            st.dataframe(
                display.round(3),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # TOP BUY
    # ========================================================

    if not results_df.empty:

        st.subheader(
            "🟢 TOP BUY"
        )

        buy_df = results_df[
            results_df[
                "signal"
            ]
            ==
            "BUY"
        ].copy()

        if buy_df.empty:

            st.info(
                "No BUY results."
            )

        else:

            buy_df = buy_df.sort_values(
                [
                    "buy_score",
                    "p_buy"
                ],
                ascending=False
            ).head(100)

            display = buy_df[
                [
                    "symbol",
                    "signal",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_buy",
                    "three_rise",
                    "distance_from_weekly_low",
                    "ai_accuracy",
                    "ai_balanced_accuracy",
                    "rows"
                ]
            ].copy()

            display[
                "p_buy"
            ] = (
                display[
                    "p_buy"
                ]
                *
                100
            )

            display[
                "distance_from_weekly_low"
            ] = (
                display[
                    "distance_from_weekly_low"
                ]
                *
                100
            )

            st.dataframe(
                display.round(3),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # TOP SELL
    # ========================================================

    if not results_df.empty:

        st.subheader(
            "🔴 TOP SELL"
        )

        sell_df = results_df[
            results_df[
                "signal"
            ]
            ==
            "SELL"
        ].copy()

        if sell_df.empty:

            st.info(
                "No SELL results."
            )

        else:

            sell_df = sell_df.sort_values(
                [
                    "sell_score",
                    "p_sell"
                ],
                ascending=False
            ).head(100)

            display = sell_df[
                [
                    "symbol",
                    "signal",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_sell",
                    "three_rise",
                    "distance_from_weekly_low",
                    "ai_accuracy",
                    "ai_balanced_accuracy",
                    "rows"
                ]
            ].copy()

            display[
                "p_sell"
            ] = (
                display[
                    "p_sell"
                ]
                *
                100
            )

            display[
                "distance_from_weekly_low"
            ] = (
                display[
                    "distance_from_weekly_low"
                ]
                *
                100
            )

            st.dataframe(
                display.round(3),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ALL RESULTS
    # ========================================================

    if not results_df.empty:

        st.subheader(
            "📊 ALL RESULTS"
        )

        all_display = results_df.copy()

        all_display[
            "p_buy"
        ] = (
            all_display[
                "p_buy"
            ]
            *
            100
        )

        all_display[
            "p_sell"
        ] = (
            all_display[
                "p_sell"
            ]
            *
            100
        )

        all_display[
            "distance_from_weekly_low"
        ] = (
            all_display[
                "distance_from_weekly_low"
            ]
            *
            100
        )

        all_display = all_display.sort_values(
            [
                "score",
                "buy_score",
                "sell_score"
            ],
            ascending=False
        )

        st.dataframe(
            all_display.round(3),
            use_container_width=True,
            hide_index=True
        )

        csv = results_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download Results CSV",
            data=csv,
            file_name="spot_ai_scanner_results.csv",
            mime="text/csv"
        )


    # ========================================================
    # ERRORS
    # ========================================================

    if not errors_df.empty:

        with st.expander(
            f"⚠️ Errors ({len(errors_df):,})"
        ):

            st.dataframe(
                errors_df[
                    [
                        "symbol",
                        "error"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


else:

    st.info(
        "Click START FULL SCAN to analyze all selected Spot symbols."
    )

    st.subheader(
        "Strategy"
    )

    st.write(
        "• 15-minute candles"
    )

    st.write(
        "• Next 2 candles = 30-minute prediction horizon"
    )

    st.write(
        "• BUY target = +10%"
    )

    st.write(
        "• SELL target = -10%"
    )

    st.write(
        "• HOLD removed"
    )

    st.write(
        "• Weekly minimum = last 672 completed 15M candles"
    )

    st.write(
        "• Weekly minimum must occur before the latest 3 rising candles"
    )

    st.write(
        "• Three consecutive rising closes required"
    )

    st.write(
        "• Last candle must be green"
    )

    st.write(
        "• Last 3-candle low must be within 1.5% of weekly low"
    )

    st.write(
        "• Matching pattern adds +20 BUY score"
    )
