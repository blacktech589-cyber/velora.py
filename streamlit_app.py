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

st.set_page_config(
    page_title="1000+ Crypto AI Scanner",
    page_icon="📉",
    layout="wide"
)

BINANCE_URL = "https://api.binance.com"
GATE_URL = "https://api.gateio.ws"

MIN_COINS = 1000

CRASH_BARS = 960
WEEKLY_BARS = 672

BUY_TARGET = 0.10
SELL_TARGET = -0.10

PREDICTION_BARS = 2

MIN_DATA = 300

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


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/154.0 Safari/537.36"
        )
    }
)


def get_json(
    url,
    params=None,
    timeout=20
):
    response = session.get(
        url,
        params=params,
        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(ttl=300)
def get_binance_symbols():

    data = get_json(
        BINANCE_URL + "/api/v3/exchangeInfo"
    )

    symbols = []

    for item in data.get("symbols", []):

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        symbols.append(
            {
                "symbol": item["symbol"],
                "base": item.get(
                    "baseAsset",
                    ""
                ),
                "quote": "USDT"
            }
        )

    return pd.DataFrame(symbols)


# ============================================================
# BINANCE VOLUME
# ============================================================

def get_binance_volume():

    data = get_json(
        BINANCE_URL + "/api/v3/ticker/24hr"
    )

    rows = []

    for item in data:

        symbol = item.get(
            "symbol",
            ""
        )

        if not symbol.endswith(
            "USDT"
        ):
            continue

        try:
            volume = float(
                item.get(
                    "quoteVolume",
                    0
                )
            )
        except Exception:
            volume = 0.0

        rows.append(
            {
                "symbol": symbol,
                "volume": volume
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BINANCE CANDLES
# ============================================================

def get_binance_candles(
    symbol,
    limit=960
):

    data = get_json(
        BINANCE_URL + "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "15m",
            "limit": min(
                int(limit),
                1000
            )
        }
    )

    rows = []

    for item in data:

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(item[0]),
                    unit="ms",
                    utc=True
                ),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )


# ============================================================
# GATE SYMBOLS
# ============================================================

@st.cache_data(ttl=300)
def get_gate_symbols():

    data = get_json(
        GATE_URL +
        "/api/v4/spot/currency_pairs"
    )

    rows = []

    for item in data:

        if str(
            item.get(
                "quote",
                ""
            )
        ).upper() != "USDT":
            continue

        status = str(
            item.get(
                "trade_status",
                ""
            )
        ).lower()

        if status and status not in {
            "tradable",
            "buyable",
            "sellable"
        }:
            continue

        symbol = item.get(
            "id"
        )

        if not symbol:
            continue

        rows.append(
            {
                "symbol": symbol,
                "base": item.get(
                    "base",
                    ""
                ),
                "quote": "USDT"
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# GATE VOLUME
# ============================================================

@st.cache_data(ttl=120)
def get_gate_volume():

    data = get_json(
        GATE_URL +
        "/api/v4/spot/tickers"
    )

    rows = []

    for item in data:

        symbol = item.get(
            "currency_pair",
            ""
        )

        if not symbol.endswith(
            "_USDT"
        ):
            continue

        try:
            volume = float(
                item.get(
                    "quote_volume",
                    0
                )
                or 0
            )
        except Exception:
            volume = 0.0

        rows.append(
            {
                "symbol": symbol,
                "volume": volume
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# GATE CANDLES
# ============================================================

def get_gate_candles(
    symbol,
    limit=960
):

    data = get_json(
        GATE_URL +
        "/api/v4/spot/candlesticks",
        params={
            "currency_pair": symbol,
            "interval": "15m",
            "limit": min(
                int(limit),
                1000
            )
        }
    )

    rows = []

    for item in data:

        if len(item) < 6:
            continue

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(item[0]),
                    unit="s",
                    utc=True
                ),
                "volume": float(item[1]),
                "close": float(item[2]),
                "high": float(item[3]),
                "low": float(item[4]),
                "open": float(item[5])
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )


# ============================================================
# SELECT SYMBOLS
# ============================================================

def select_symbols(
    provider,
    count
):

    if provider == "Binance":

        symbols = get_binance_symbols()

        if symbols.empty:
            return symbols

        try:

            volume = get_binance_volume()

            symbols = symbols.merge(
                volume,
                on="symbol",
                how="left"
            )

            symbols["volume"] = (
                symbols["volume"]
                .fillna(0)
            )

            symbols = symbols.sort_values(
                "volume",
                ascending=False
            )

        except Exception:
            pass

        return symbols.head(
            max(
                MIN_COINS,
                count
            )
        )

    if provider == "Gate.io":

        symbols = get_gate_symbols()

        if symbols.empty:
            return symbols

        try:

            volume = get_gate_volume()

            symbols = symbols.merge(
                volume,
                on="symbol",
                how="left"
            )

            symbols["volume"] = (
                symbols["volume"]
                .fillna(0)
            )

            symbols = symbols.sort_values(
                "volume",
                ascending=False
            )

        except Exception:
            pass

        return symbols.head(
            max(
                MIN_COINS,
                count
            )
        )

    return pd.DataFrame()


# ============================================================
# DATA LOADER
# ============================================================

def load_candles(
    provider,
    symbol
):

    if provider == "Binance":

        return get_binance_candles(
            symbol,
            CRASH_BARS
        )

    if provider == "Gate.io":

        return get_gate_candles(
            symbol,
            CRASH_BARS
        )

    return pd.DataFrame()


# ============================================================
# EMA
# ============================================================

def ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(
    close,
    period
):

    delta = close.diff()

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

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return result.fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period=14
):

    previous = df[
        "close"
    ].shift(1)

    tr = pd.concat(
        [
            df["high"] -
            df["low"],

            (
                df["high"] -
                previous
            ).abs(),

            (
                df["low"] -
                previous
            ).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# ADX
# ============================================================

def adx(
    df,
    period=14
):

    up = df[
        "high"
    ].diff()

    down = -df[
        "low"
    ].diff()

    plus_dm = pd.Series(
        np.where(
            (
                up > down
            )
            &
            (
                up > 0
            ),
            up,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (
                down > up
            )
            &
            (
                down > 0
            ),
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
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr_value
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr_value
    )

    denominator = (
        plus_di +
        minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100 *
        (
            plus_di -
            minus_di
        ).abs()
        /
        denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean().fillna(0)


# ============================================================
# FEATURES
# ============================================================

def build_features(df):

    x = df.copy()

    close = x[
        "close"
    ]

    high = x[
        "high"
    ]

    low = x[
        "low"
    ]

    volume = x[
        "volume"
    ]

    x[
        "ret1"
    ] = close.pct_change(1)

    x[
        "ret3"
    ] = close.pct_change(3)

    x[
        "ret6"
    ] = close.pct_change(6)

    x[
        "ret12"
    ] = close.pct_change(12)

    x[
        "ret24"
    ] = close.pct_change(24)

    e5 = ema(
        close,
        5
    )

    e20 = ema(
        close,
        20
    )

    e50 = ema(
        close,
        50
    )

    e200 = ema(
        close,
        200
    )

    x[
        "ema5_20"
    ] = e5 / e20 - 1

    x[
        "ema20_50"
    ] = e20 / e50 - 1

    x[
        "ema50_200"
    ] = e50 / e200 - 1

    x[
        "dist20"
    ] = close / e20 - 1

    x[
        "dist50"
    ] = close / e50 - 1

    x[
        "dist200"
    ] = close / e200 - 1

    x[
        "rsi7"
    ] = rsi(
        close,
        7
    )

    x[
        "rsi14"
    ] = rsi(
        close,
        14
    )

    x[
        "rsi21"
    ] = rsi(
        close,
        21
    )

    macd_line = (
        ema(
            close,
            12
        )
        -
        ema(
            close,
            26
        )
    )

    macd_signal = ema(
        macd_line,
        9
    )

    x[
        "macd"
    ] = macd_line

    x[
        "macd_signal"
    ] = macd_signal

    x[
        "macd_hist"
    ] = (
        macd_line -
        macd_signal
    )

    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    upper = (
        middle +
        2 * std
    )

    lower = (
        middle -
        2 * std
    )

    width = (
        upper -
        lower
    ).replace(
        0,
        np.nan
    )

    x[
        "bb_position"
    ] = (
        close -
        lower
    ) / width

    x[
        "bb_width"
    ] = (
        width /
        middle.replace(
            0,
            np.nan
        )
    )

    atr_value = atr(
        x,
        14
    )

    x[
        "atr_pct"
    ] = (
        atr_value /
        close.replace(
            0,
            np.nan
        )
    )

    x[
        "adx"
    ] = adx(
        x,
        14
    )

    low14 = low.rolling(
        14
    ).min()

    high14 = high.rolling(
        14
    ).max()

    denominator = (
        high14 -
        low14
    ).replace(
        0,
        np.nan
    )

    k = (
        100 *
        (
            close -
            low14
        )
        /
        denominator
    )

    d = k.rolling(
        3
    ).mean()

    x[
        "stoch_k"
    ] = k.fillna(50)

    x[
        "stoch_d"
    ] = d.fillna(50)

    candle_range = (
        high -
        low
    ).replace(
        0,
        np.nan
    )

    body = (
        close -
        x["open"]
    ).abs()

    x[
        "body_pct"
    ] = (
        body /
        close.replace(
            0,
            np.nan
        )
    )

    x[
        "range_pct"
    ] = (
        candle_range /
        close.replace(
            0,
            np.nan
        )
    )

    x[
        "upper_wick"
    ] = (
        high -
        np.maximum(
            x["open"],
            close
        )
    ) / candle_range

    x[
        "lower_wick"
    ] = (
        np.minimum(
            x["open"],
            close
        ) -
        low
    ) / candle_range

    volume_mean = volume.rolling(
        20
    ).mean()

    volume_std = volume.rolling(
        20
    ).std()

    x[
        "volume_ratio"
    ] = (
        volume /
        volume_mean.replace(
            0,
            np.nan
        )
    )

    x[
        "volume_z"
    ] = (
        volume -
        volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    x[
        "volatility12"
    ] = (
        x["ret1"]
        .rolling(12)
        .std()
    )

    x[
        "volatility24"
    ] = (
        x["ret1"]
        .rolling(24)
        .std()
    )

    x[
        "volatility48"
    ] = (
        x["ret1"]
        .rolling(48)
        .std()
    )

    price_mean = close.rolling(
        48
    ).mean()

    price_std = close.rolling(
        48
    ).std()

    x[
        "price_z"
    ] = (
        close -
        price_mean
    ) / price_std.replace(
        0,
        np.nan
    )

    x[
        "high_distance"
    ] = (
        high /
        close -
        1
    )

    x[
        "low_distance"
    ] = (
        low /
        close -
        1
    )

    return x


# ============================================================
# CRASH CALCULATION
# ============================================================

def calculate_crash(
    df,
    threshold
):

    result = {
        "peak": np.nan,
        "price": np.nan,
        "drop": np.nan,
        "is_crash": False
    }

    if len(df) < 20:
        return result

    completed = df.iloc[
        :-1
    ]

    window = completed.tail(
        CRASH_BARS
    )

    if window.empty:
        return result

    peak = float(
        window[
            "high"
        ].max()
    )

    price = float(
        completed[
            "close"
        ].iloc[-1]
    )

    if peak <= 0:
        return result

    drop = (
        1 -
        price / peak
    ) * 100

    result[
        "peak"
    ] = peak

    result[
        "price"
    ] = price

    result[
        "drop"
    ] = drop

    result[
        "is_crash"
    ] = (
        drop >= threshold
    )

    return result


# ============================================================
# WEEKLY PATTERN
# ============================================================

def weekly_pattern(df):

    result = {
        "weekly_low": np.nan,
        "pattern": False,
        "three_rising": False,
        "green": False,
        "near_low": False,
        "dip_before": False
    }

    if len(df) < (
        WEEKLY_BARS + 3
    ):
        return result

    completed = df.iloc[
        :-1
    ]

    weekly = completed.tail(
        WEEKLY_BARS
    )

    last3 = completed.tail(
        3
    )

    weekly_low = float(
        weekly[
            "low"
        ].min()
    )

    recent_low = float(
        last3[
            "low"
        ].min()
    )

    if weekly_low <= 0:
        return result

    distance = (
        recent_low /
        weekly_low -
        1
    )

    rising = (
        last3[
            "close"
        ].iloc[0]
        <
        last3[
            "close"
        ].iloc[1]
        <
        last3[
            "close"
        ].iloc[2]
    )

    green = (
        last3[
            "close"
        ].iloc[2]
        >
        last3[
            "open"
        ].iloc[2]
    )

    weekly_low_index = weekly[
        "low"
    ].idxmin()

    first_index = last3.index[
        0
    ]

    dip_before = (
        weekly_low_index
        <
        first_index
    )

    near_low = (
        distance <= 0.015
    )

    result[
        "weekly_low"
    ] = weekly_low

    result[
        "three_rising"
    ] = bool(
        rising
    )

    result[
        "green"
    ] = bool(
        green
    )

    result[
        "near_low"
    ] = bool(
        near_low
    )

    result[
        "dip_before"
    ] = bool(
        dip_before
    )

    result[
        "pattern"
    ] = bool(
        rising
        and
        green
        and
        near_low
        and
        dip_before
    )

    return result


# ============================================================
# AI TARGET
# ============================================================

def make_target(df):

    target = np.full(
        len(df),
        np.nan
    )

    close = df[
        "close"
    ].values

    high = df[
        "high"
    ].values

    low = df[
        "low"
    ].values

    for i in range(
        len(df) -
        PREDICTION_BARS
    ):

        entry = close[i]

        future_high = np.max(
            high[
                i + 1:
                i + 1 + PREDICTION_BARS
            ]
        )

        future_low = np.min(
            low[
                i + 1:
                i + 1 + PREDICTION_BARS
            ]
        )

        up = (
            future_high /
            entry -
            1
        )

        down = (
            future_low /
            entry -
            1
        )

        buy_hit = (
            up >= BUY_TARGET
        )

        sell_hit = (
            down <= SELL_TARGET
        )

        if buy_hit and not sell_hit:

            target[i] = 1

        elif sell_hit and not buy_hit:

            target[i] = 0

        elif buy_hit and sell_hit:

            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

        else:

            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

    return target


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(df):

    features = build_features(
        df
    )

    row = features.iloc[
        -2
    ]

    buy = 0.0
    sell = 0.0

    if row[
        "ema5_20"
    ] > 0:
        buy += 10
    else:
        sell += 10

    if row[
        "ema20_50"
    ] > 0:
        buy += 10
    else:
        sell += 10

    if row[
        "ema50_200"
    ] > 0:
        buy += 8
    else:
        sell += 8

    if row[
        "rsi14"
    ] < 35:
        buy += 10

    elif row[
        "rsi14"
    ] > 65:
        sell += 10

    if row[
        "macd_hist"
    ] > 0:
        buy += 8
    else:
        sell += 8

    if row[
        "bb_position"
    ] < 0.25:
        buy += 10

    elif row[
        "bb_position"
    ] > 0.75:
        sell += 10

    if row[
        "stoch_k"
    ] < 25:
        buy += 8

    elif row[
        "stoch_k"
    ] > 75:
        sell += 8

    if row[
        "close"
    ] > row[
        "open"
    ]:
        buy += 8
    else:
        sell += 8

    return (
        buy,
        sell
    )


# ============================================================
# AI MODEL
# ============================================================

def create_model():

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
                        48,
                        24
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0007,
                    batch_size=128,
                    learning_rate_init=0.001,
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# AI PREDICTION
# ============================================================

def ai_predict(
    df,
    training_rows
):

    features = build_features(
        df
    )

    target = make_target(
        df
    )

    feature_df = (
        features[
            FEATURES
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    target_series = pd.Series(
        target,
        index=df.index
    )

    valid = (
        feature_df
        .notna()
        .all(
            axis=1
        )
        &
        target_series
        .notna()
    )

    X = feature_df.loc[
        valid
    ].values

    y = target_series.loc[
        valid
    ].astype(
        int
    ).values

    if len(X) < 300:
        raise ValueError(
            "Not enough AI rows"
        )

    if len(X) > training_rows:

        X = X[
            -training_rows:
        ]

        y = y[
            -training_rows:
        ]

    if len(
        np.unique(y)
    ) < 2:

        raise ValueError(
            "Only one AI class"
        )

    split = int(
        len(X) * 0.80
    )

    X_train = X[
        :split
    ]

    y_train = y[
        :split
    ]

    X_test = X[
        split:
    ]

    y_test = y[
        split:
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = float(
        np.mean(
            prediction ==
            y_test
        )
    )

    latest = feature_df.iloc[
        -2:-1
    ]

    probability = model.predict_proba(
        latest.values
    )[0]

    p_buy = 0.0
    p_sell = 0.0

    for cls, prob in zip(
        model.classes_,
        probability
    ):

        if int(cls) == 1:
            p_buy = float(
                prob
            )
        else:
            p_sell = float(
                prob
            )

    return {
        "p_buy": p_buy,
        "p_sell": p_sell,
        "accuracy": accuracy
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    df,
    crash_threshold,
    training_rows
):

    if df.empty:

        raise ValueError(
            "No candle data"
        )

    if len(df) < MIN_DATA:

        raise ValueError(
            "Not enough candles"
        )

    crash = calculate_crash(
        df,
        crash_threshold
    )

    weekly = weekly_pattern(
        df
    )

    tech_buy, tech_sell = (
        technical_score(
            df
        )
    )

    try:

        ai = ai_predict(
            df,
            training_rows
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

        buy_score = (
            p_buy * 100
            +
            tech_buy * 0.35
        )

        sell_score = (
            p_sell * 100
            +
            tech_sell * 0.35
        )

    except Exception:

        total = (
            tech_buy +
            tech_sell
        )

        if total > 0:

            p_buy = (
                tech_buy /
                total
            )

            p_sell = (
                tech_sell /
                total
            )

        else:

            p_buy = 0.5
            p_sell = 0.5

        buy_score = tech_buy
        sell_score = tech_sell

        accuracy = np.nan

    crash_bonus = 0

    if crash[
        "is_crash"
    ]:

        crash_bonus = 20

        buy_score += 20

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    pattern_bonus = 0

    if weekly[
        "pattern"
    ]:

        pattern_bonus = 20

        buy_score += 20

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    if buy_score >= sell_score:

        signal = "BUY"

    else:

        signal = "SELL"

    price = float(
        df.iloc[
            -2
        ][
            "close"
        ]
    )

    return {
        "symbol": symbol,
        "signal": signal,
        "price": price,

        "buy_score": round(
            buy_score,
            2
        ),

        "sell_score": round(
            sell_score,
            2
        ),

        "p_buy": round(
            p_buy * 100,
            2
        ),

        "p_sell": round(
            p_sell * 100,
            2
        ),

        "10d_peak": round(
            crash[
                "peak"
            ],
            10
        ),

        "10d_drop": round(
            crash[
                "drop"
            ],
            2
        ),

        "crash_80": bool(
            crash[
                "is_crash"
            ]
        ),

        "crash_bonus": crash_bonus,

        "weekly_low": round(
            weekly[
                "weekly_low"
            ],
            10
        ),

        "weekly_pattern": bool(
            weekly[
                "pattern"
            ]
        ),

        "three_rising": bool(
            weekly[
                "three_rising"
            ]
        ),

        "green_last": bool(
            weekly[
                "green"
            ]
        ),

        "ai_accuracy": (
            round(
                accuracy * 100,
                2
            )
            if not pd.isna(
                accuracy
            )
            else np.nan
        ),

        "candles": len(df)
    }


# ============================================================
# SCAN
# ============================================================

def run_scan(
    provider,
    symbols,
    crash_threshold,
    training_rows
):

    results = []
    errors = []

    total = len(
        symbols
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    for i, item in enumerate(
        symbols,
        start=1
    ):

        symbol = item[
            "symbol"
        ]

        status.write(
            f"{i:,}/{total:,}  {symbol}"
        )

        try:

            df = load_candles(
                provider,
                symbol
            )

            result = analyze_symbol(
                symbol,
                df,
                crash_threshold,
                training_rows
            )

            results.append(
                result
            )

        except Exception as e:

            errors.append(
                {
                    "symbol": symbol,
                    "error": str(e)
                }
            )

        progress.progress(
            i / total
        )

        time.sleep(
            0.02
        )

    progress.empty()
    status.empty()

    return (
        pd.DataFrame(
            results
        ),
        pd.DataFrame(
            errors
        )
    )


# ============================================================
# UI
# ============================================================

st.title(
    "📉 1000+ Crypto 15M AI Scanner"
)

st.caption(
    "10-day crash detection + AI BUY/SELL"
)

st.warning(
    "Bu uygulama yalnızca analiz yapar. "
    "Otomatik emir göndermez."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Scanner Settings"
    )

    provider_mode = st.selectbox(
        "Data Provider",
        [
            "AUTO",
            "Binance",
            "Gate.io"
        ]
    )

    coin_count = st.number_input(
        "Coin Count",
        min_value=1000,
        max_value=3000,
        value=1000,
        step=100
    )

    crash_threshold = st.slider(
        "10-Day Crash %",
        min_value=50.0,
        max_value=95.0,
        value=80.0,
        step=1.0
    )

    training_rows = st.number_input(
        "AI Training Candles",
        min_value=300,
        max_value=900,
        value=700,
        step=100
    )

    st.divider()

    st.write(
        "15M timeframe"
    )

    st.write(
        "10 days = 960 candles"
    )

    st.write(
        "BUY = +10% / 30 min"
    )

    st.write(
        "SELL = -10% / 30 min"
    )

    st.write(
        "HOLD disabled"
    )

    st.write(
        "Crash bonus = +20"
    )

    st.write(
        "Weekly pattern bonus = +20"
    )


# ============================================================
# PROVIDER
# ============================================================

if provider_mode == "AUTO":

    try:

        test_symbols = (
            get_binance_symbols()
        )

        if len(
            test_symbols
        ) >= MIN_COINS:

            provider = "Binance"

        else:

            provider = "Gate.io"

    except Exception:

        provider = "Gate.io"

else:

    provider = provider_mode


st.info(
    f"Active provider: {provider}"
)


# ============================================================
# SYMBOL LIST
# ============================================================

try:

    symbols_df = select_symbols(
        provider,
        int(coin_count)
    )

except Exception as e:

    symbols_df = pd.DataFrame()

    st.error(
        f"Symbol API error: {e}"
    )


if symbols_df.empty:

    st.error(
        "Coin list could not be loaded."
    )

    st.stop()


available = len(
    symbols_df
)

if available >= MIN_COINS:

    st.success(
        f"{available:,} coin loaded."
    )

else:

    st.warning(
        f"Only {available:,} coins available. "
        f"Target is {MIN_COINS:,}."
    )


# ============================================================
# SCOPE
# ============================================================

scope = st.radio(
    "Scan",
    [
        "ALL",
        "MANUAL"
    ],
    horizontal=True
)

if scope == "ALL":

    selected = symbols_df.copy()

else:

    names = st.multiselect(
        "Coins",
        symbols_df[
            "symbol"
        ].tolist(),
        default=symbols_df[
            "symbol"
        ].tolist()[:20]
    )

    selected = symbols_df[
        symbols_df[
            "symbol"
        ].isin(names)
    ].copy()


st.write(
    f"Selected coins: {len(selected):,}"
)


# ============================================================
# START
# ============================================================

start_scan = st.button(
    "🚀 START SCAN",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTE
# ============================================================

if start_scan:

    start_time = time.time()

    with st.spinner(
        "Scanning coins..."
    ):

        results, errors = run_scan(
            provider,
            selected.to_dict(
                "records"
            ),
            float(
                crash_threshold
            ),
            int(
                training_rows
            )
        )

    elapsed = (
        time.time() -
        start_time
    )

    st.session_state[
        "results"
    ] = results

    st.session_state[
        "errors"
    ] = errors

    st.session_state[
        "elapsed"
    ] = elapsed

    st.session_state[
        "scan_time"
    ] = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state[
        "results"
    ]

    errors = st.session_state[
        "errors"
    ]

    elapsed = st.session_state[
        "elapsed"
    ]

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(
        5
    )

    buy_count = 0
    sell_count = 0
    crash_count = 0

    if not results.empty:

        buy_count = int(
            (
                results[
                    "signal"
                ]
                ==
                "BUY"
            ).sum()
        )

        sell_count = int(
            (
                results[
                    "signal"
                ]
                ==
                "SELL"
            ).sum()
        )

        crash_count = int(
            results[
                "crash_80"
            ].sum()
        )

    col1.metric(
        "Scanned",
        f"{len(results):,}"
    )

    col2.metric(
        "Errors",
        f"{len(errors):,}"
    )

    col3.metric(
        "BUY",
        f"{buy_count:,}"
    )

    col4.metric(
        "SELL",
        f"{sell_count:,}"
    )

    col5.metric(
        "CRASH",
        f"{crash_count:,}"
    )

    st.caption(
        f"Scan duration: {elapsed:.1f} seconds"
    )


    # ========================================================
    # CRASH
    # ========================================================

    st.subheader(
        f"🔥 SON 10 GÜNDE %{crash_threshold:.0f}+ DÜŞENLER"
    )

    if not results.empty:

        crash_df = results[
            results[
                "10d_drop"
            ]
            >=
            crash_threshold
        ].copy()

        crash_df = crash_df.sort_values(
            [
                "10d_drop",
                "buy_score"
            ],
            ascending=False
        )

        if crash_df.empty:

            st.info(
                "Crash filtresine uyan coin yok."
            )

        else:

            st.dataframe(
                crash_df[
                    [
                        "symbol",
                        "signal",
                        "price",
                        "10d_peak",
                        "10d_drop",
                        "p_buy",
                        "buy_score",
                        "sell_score",
                        "weekly_pattern",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # TOP BUY
    # ========================================================

    st.subheader(
        "🟢 TOP BUY"
    )

    if not results.empty:

        buy_df = results[
            results[
                "signal"
            ]
            ==
            "BUY"
        ].copy()

        buy_df = buy_df.sort_values(
            [
                "buy_score",
                "p_buy",
                "10d_drop"
            ],
            ascending=False
        ).head(
            100
        )

        st.dataframe(
            buy_df[
                [
                    "symbol",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_buy",
                    "10d_drop",
                    "crash_80",
                    "weekly_pattern",
                    "three_rising",
                    "green_last",
                    "ai_accuracy"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # TOP SELL
    # ========================================================

    st.subheader(
        "🔴 TOP SELL"
    )

    if not results.empty:

        sell_df = results[
            results[
                "signal"
            ]
            ==
            "SELL"
        ].copy()

        sell_df = sell_df.sort_values(
            [
                "sell_score",
                "p_sell"
            ],
            ascending=False
        ).head(
            100
        )

        st.dataframe(
            sell_df[
                [
                    "symbol",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_sell",
                    "10d_drop",
                    "crash_80",
                    "weekly_pattern",
                    "ai_accuracy"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # WEEKLY PATTERN
    # ========================================================

    st.subheader(
        "🔥 WEEKLY LOW + 3 RISING"
    )

    if not results.empty:

        pattern_df = results[
            results[
                "weekly_pattern"
            ]
            ==
            True
        ].copy()

        pattern_df = pattern_df.sort_values(
            "buy_score",
            ascending=False
        ).head(
            100
        )

        if pattern_df.empty:

            st.info(
                "Weekly pattern bulunamadı."
            )

        else:

            st.dataframe(
                pattern_df[
                    [
                        "symbol",
                        "price",
                        "weekly_low",
                        "buy_score",
                        "p_buy",
                        "10d_drop",
                        "crash_80",
                        "three_rising",
                        "green_last"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ALL
    # ========================================================

    st.subheader(
        "📊 ALL RESULTS"
    )

    if not results.empty:

        all_results = results.sort_values(
            [
                "buy_score",
                "sell_score"
            ],
            ascending=False
        )

        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )

        csv = results.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ DOWNLOAD CSV",
            data=csv,
            file_name="crypto_1000_ai_scan.csv",
            mime="text/csv",
            use_container_width=True
        )


    # ========================================================
    # ERRORS
    # ========================================================

    if not errors.empty:

        with st.expander(
            f"⚠️ ERRORS ({len(errors):,})"
        ):

            st.dataframe(
                errors,
                use_container_width=True,
                hide_index=True
            )

else:

    st.info(
        "START SCAN butonuna basarak taramayı başlat."
    )
