Aşağıdaki kodu `streamlit_app.py` dosyanın tamamının yerine koy:

```python
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
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="1000+ Crypto AI Scanner",
    page_icon="📉",
    layout="wide"
)

APP_TITLE = "1000+ Crypto 15M AI Crash Scanner"

BINANCE_URL = "https://api.binance.com"
GATE_URL = "https://api.gateio.ws"

TIMEFRAME = "15m"

MIN_SYMBOLS = 1000

CRASH_LOOKBACK_BARS = 960
CRASH_THRESHOLD_DEFAULT = 80.0

PREDICTION_BARS = 2

BUY_TARGET = 0.10
SELL_TARGET = -0.10

WEEKLY_BARS = 672

WEEKLY_DISTANCE = 0.015

MIN_TRAIN_ROWS = 400

REQUEST_TIMEOUT = 20

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
# SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/154.0 Safari/537.36"
    }
)


# ============================================================
# HTTP
# ============================================================

def get_json(url, params=None):

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
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

    rows = []

    for item in data.get("symbols", []):

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        rows.append(
            {
                "symbol": item["symbol"],
                "base": item.get("baseAsset", ""),
                "quote": item.get("quoteAsset", "")
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BINANCE 15M DATA
# ============================================================

def get_binance_klines(
    symbol,
    limit=1000
):

    data = get_json(
        BINANCE_URL + "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "15m",
            "limit": min(limit, 1000)
        }
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
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5])
            }
        )

    return pd.DataFrame(rows).set_index(
        "timestamp"
    )


# ============================================================
# GATE SYMBOLS
# ============================================================

@st.cache_data(ttl=300)
def get_gate_symbols():

    data = get_json(
        GATE_URL + "/api/v4/spot/currency_pairs"
    )

    rows = []

    for item in data:

        quote = str(
            item.get("quote", "")
        ).upper()

        status = str(
            item.get("trade_status", "")
        ).lower()

        if quote != "USDT":
            continue

        if status not in {
            "",
            "tradable",
            "buyable",
            "sellable"
        }:
            continue

        symbol = item.get("id")

        if not symbol:
            continue

        rows.append(
            {
                "symbol": symbol,
                "base": item.get("base", ""),
                "quote": quote
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# GATE TICKERS
# ============================================================

@st.cache_data(ttl=120)
def get_gate_tickers():

    data = get_json(
        GATE_URL + "/api/v4/spot/tickers"
    )

    rows = []

    for item in data:

        pair = item.get(
            "currency_pair",
            ""
        )

        if not pair.endswith("_USDT"):
            continue

        last = float(
            item.get("last", 0)
            or 0
        )

        quote_volume = float(
            item.get("quote_volume", 0)
            or 0
        )

        rows.append(
            {
                "symbol": pair,
                "last": last,
                "quote_volume": quote_volume
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SELECT 1000+ SYMBOLS
# ============================================================

def select_symbols(
    provider,
    requested_count
):

    if provider == "Binance":

        symbols = get_binance_symbols()

        if symbols.empty:
            return symbols

        try:

            ticker = get_json(
                BINANCE_URL + "/api/v3/ticker/24hr"
            )

            volume_rows = []

            for item in ticker:

                if not item["symbol"].endswith("USDT"):
                    continue

                try:
                    volume = float(
                        item.get(
                            "quoteVolume",
                            0
                        )
                    )
                except Exception:
                    volume = 0

                volume_rows.append(
                    {
                        "symbol": item["symbol"],
                        "volume": volume
                    }
                )

            volume_df = pd.DataFrame(
                volume_rows
            )

            symbols = symbols.merge(
                volume_df,
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
                MIN_SYMBOLS,
                requested_count
            )
        )

    if provider == "Gate.io":

        symbols = get_gate_symbols()

        if symbols.empty:
            return symbols

        try:

            tickers = get_gate_tickers()

            symbols = symbols.merge(
                tickers[
                    [
                        "symbol",
                        "quote_volume"
                    ]
                ],
                on="symbol",
                how="left"
            )

            symbols["quote_volume"] = (
                symbols["quote_volume"]
                .fillna(0)
            )

            symbols = symbols.sort_values(
                "quote_volume",
                ascending=False
            )

        except Exception:

            pass

        return symbols.head(
            max(
                MIN_SYMBOLS,
                requested_count
            )
        )

    return pd.DataFrame()


# ============================================================
# GATE 15M DATA
# ============================================================

def get_gate_klines(
    symbol,
    limit=960
):

    limit = min(
        int(limit),
        1000
    )

    data = get_json(
        GATE_URL + "/api/v4/spot/candlesticks",
        params={
            "currency_pair": symbol,
            "interval": "15m",
            "limit": limit
        }
    )

    if not data:
        return pd.DataFrame()

    rows = []

    for x in data:

        if len(x) < 6:
            continue

        # Gate format:
        # timestamp
        # volume quote
        # close
        # high
        # low
        # open

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(x[0]),
                    unit="s",
                    utc=True
                ),
                "open": float(x[5]),
                "high": float(x[3]),
                "low": float(x[4]),
                "close": float(x[2]),
                "volume": float(x[1])
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df
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
# LOAD DATA
# ============================================================

def load_data(
    provider,
    symbol,
    candles
):

    if provider == "Binance":

        return get_binance_klines(
            symbol,
            min(candles, 1000)
        )

    if provider == "Gate.io":

        return get_gate_klines(
            symbol,
            min(candles, 1000)
        )

    return pd.DataFrame()


# ============================================================
# RSI
# ============================================================

def calc_rsi(
    close,
    period=14
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

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    ).fillna(50)


# ============================================================
# ATR
# ============================================================

def calc_atr(
    df,
    period=14
):

    previous = df[
        "close"
    ].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous).abs(),
            (df["low"] - previous).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# ADX
# ============================================================

def calc_adx(
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
            (up > down) &
            (up > 0),
            up,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) &
            (down > 0),
            down,
            0
        ),
        index=df.index
    )

    atr_value = calc_atr(
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
        (plus_di - minus_di).abs()
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

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret6"] = close.pct_change(6)
    x["ret12"] = close.pct_change(12)
    x["ret24"] = close.pct_change(24)

    e5 = close.ewm(
        span=5,
        adjust=False
    ).mean()

    e20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    e50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    e200 = close.ewm(
        span=200,
        adjust=False
    ).mean()

    x["ema5_20"] = e5 / e20 - 1
    x["ema20_50"] = e20 / e50 - 1
    x["ema50_200"] = e50 / e200 - 1

    x["dist20"] = close / e20 - 1
    x["dist50"] = close / e50 - 1
    x["dist200"] = close / e200 - 1

    x["rsi7"] = calc_rsi(
        close,
        7
    )

    x["rsi14"] = calc_rsi(
        close,
        14
    )

    x["rsi21"] = calc_rsi(
        close,
        21
    )

    macd = (
        close.ewm(
            span=12,
            adjust=False
        ).mean()
        -
        close.ewm(
            span=26,
            adjust=False
        ).mean()
    )

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    x["macd"] = macd
    x["macd_signal"] = signal
    x["macd_hist"] = macd - signal

    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    upper = middle + 2 * std
    lower = middle - 2 * std

    width = (
        upper - lower
    ).replace(
        0,
        np.nan
    )

    x["bb_position"] = (
        close - lower
    ) / width

    x["bb_width"] = (
        width /
        middle.replace(
            0,
            np.nan
        )
    )

    atr_value = calc_atr(
        x,
        14
    )

    x["atr_pct"] = (
        atr_value /
        close.replace(
            0,
            np.nan
        )
    )

    x["adx"] = calc_adx(
        x,
        14
    )

    lowest = low.rolling(
        14
    ).min()

    highest = high.rolling(
        14
    ).max()

    denominator = (
        highest -
        lowest
    ).replace(
        0,
        np.nan
    )

    k = (
        100 *
        (close - lowest)
        /
        denominator
    )

    d = k.rolling(
        3
    ).mean()

    x["stoch_k"] = k.fillna(50)
    x["stoch_d"] = d.fillna(50)

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
        body /
        close.replace(
            0,
            np.nan
        )
    )

    x["range_pct"] = (
        candle_range /
        close.replace(
            0,
            np.nan
        )
    )

    x["upper_wick"] = (
        high -
        np.maximum(
            x["open"],
            close
        )
    ) / candle_range

    x["lower_wick"] = (
        np.minimum(
            x["open"],
            close
        ) -
        low
    ) / candle_range

    vol_mean = volume.rolling(
        20
    ).mean()

    vol_std = volume.rolling(
        20
    ).std()

    x["volume_ratio"] = (
        volume /
        vol_mean.replace(
            0,
            np.nan
        )
    )

    x["volume_z"] = (
        volume -
        vol_mean
    ) / vol_std.replace(
        0,
        np.nan
    )

    x["volatility12"] = (
        x["ret1"]
        .rolling(12)
        .std()
    )

    x["volatility24"] = (
        x["ret1"]
        .rolling(24)
        .std()
    )

    x["volatility48"] = (
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

    x["price_z"] = (
        close -
        price_mean
    ) / price_std.replace(
        0,
        np.nan
    )

    x["high_distance"] = (
        high /
        close -
        1
    )

    x["low_distance"] = (
        low /
        close -
        1
    )

    return x


# ============================================================
# 10-DAY CRASH
# ============================================================

def calculate_crash(
    df,
    threshold
):

    result = {
        "peak_10d": np.nan,
        "current_price": np.nan,
        "drop_10d": np.nan,
        "crash_80": False,
        "peak_index": None
    }

    if len(df) < 50:
        return result

    # Current candle ignored.
    completed = df.iloc[:-1].copy()

    if len(completed) < 10:
        return result

    window = completed.tail(
        CRASH_LOOKBACK_BARS
    )

    peak = float(
        window["high"].max()
    )

    current = float(
        completed["close"].iloc[-1]
    )

    if peak <= 0:
        return result

    drop = (
        1 -
        current / peak
    ) * 100

    result["peak_10d"] = peak
    result["current_price"] = current
    result["drop_10d"] = drop

    result["crash_80"] = (
        drop >= threshold
    )

    peak_index = window[
        "high"
    ].idxmax()

    result["peak_index"] = peak_index

    return result


# ============================================================
# WEEKLY LOW / THREE RISING
# ============================================================

def weekly_pattern(df):

    result = {
        "weekly_low": np.nan,
        "near_weekly_low": False,
        "three_rising": False,
        "green_last": False,
        "dip_before_rise": False,
        "pattern": False
    }

    if len(df) < 700:
        return result

    completed = df.iloc[:-1]

    weekly = completed.tail(
        WEEKLY_BARS
    )

    last3 = completed.tail(3)

    if len(last3) < 3:
        return result

    weekly_low = float(
        weekly["low"].min()
    )

    recent_low = float(
        last3["low"].min()
    )

    distance = (
        recent_low /
        weekly_low -
        1
    )

    rising = (
        last3["close"].iloc[0]
        <
        last3["close"].iloc[1]
        <
        last3["close"].iloc[2]
    )

    green = (
        last3["close"].iloc[2]
        >
        last3["open"].iloc[2]
    )

    weekly_low_index = weekly[
        "low"
    ].idxmin()

    first_rise_index = last3.index[0]

    dip_before = (
        weekly_low_index
        <
        first_rise_index
    )

    near = (
        distance <= WEEKLY_DISTANCE
    )

    pattern = (
        rising
        and
        green
        and
        near
        and
        dip_before
    )

    result["weekly_low"] = weekly_low
    result["near_weekly_low"] = near
    result["three_rising"] = rising
    result["green_last"] = green
    result["dip_before_rise"] = dip_before
    result["pattern"] = pattern

    return result


# ============================================================
# TARGET
# ============================================================

def make_target(df):

    target = np.full(
        len(df),
        np.nan
    )

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for i in range(
        len(df) - PREDICTION_BARS
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
# BALANCE
# ============================================================

def balance_data(
    X,
    y
):

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    if len(classes) < 2:
        return X, y

    target_count = int(
        counts.max()
    )

    rng = np.random.default_rng(
        42
    )

    xs = []
    ys = []

    for cls, count in zip(
        classes,
        counts
    ):

        idx = np.where(
            y == cls
        )[0]

        if count < target_count:

            extra = rng.choice(
                idx,
                size=(
                    target_count -
                    count
                ),
                replace=True
            )

            idx = np.concatenate(
                [
                    idx,
                    extra
                ]
            )

        xs.append(
            X[idx]
        )

        ys.append(
            y[idx]
        )

    X2 = np.vstack(xs)
    y2 = np.concatenate(ys)

    order = rng.permutation(
        len(y2)
    )

    return (
        X2[order],
        y2[order]
    )


# ============================================================
# MODEL
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
# AI
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
        feature_df.notna().all(axis=1)
        &
        target_series.notna()
    )

    X = feature_df.loc[
        valid
    ].values

    y = target_series.loc[
        valid
    ].astype(int).values

    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(
            "Not enough AI training rows"
        )

    if len(X) > training_rows:

        X = X[-training_rows:]
        y = y[-training_rows:]

    if len(
        np.unique(y)
    ) < 2:

        raise ValueError(
            "Only one target class"
        )

    split = int(
        len(X) * 0.80
    )

    X_fit = X[:split]
    y_fit = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    X_fit, y_fit = balance_data(
        X_fit,
        y_fit
    )

    model = create_model()

    model.fit(
        X_fit,
        y_fit
    )

    predictions = model.predict(
        X_test
    )

    accuracy = float(
        np.mean(
            predictions == y_test
        )
    )

    recalls = []

    for cls in [0, 1]:

        mask = (
            y_test == cls
        )

        if mask.sum() > 0:

            recalls.append(
                float(
                    np.mean(
                        predictions[mask]
                        ==
                        cls
                    )
                )
            )

    balanced_accuracy = (
        float(
            np.mean(recalls)
        )
        if recalls
        else np.nan
    )

    latest = (
        feature_df
        .iloc[-2:-1]
    )

    if latest.isna().any().any():

        raise ValueError(
            "Latest features contain NaN"
        )

    probabilities = (
        model
        .predict_proba(
            latest.values
        )[0]
    )

    p_sell = 0.0
    p_buy = 0.0

    for cls, probability in zip(
        model.classes_,
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
        "balanced_accuracy":
            balanced_accuracy
    }


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(df):

    f = build_features(
        df
    )

    row = f.iloc[-2]

    buy = 0.0
    sell = 0.0

    if row["ema5_20"] > 0:
        buy += 10
    else:
        sell += 10

    if row["ema20_50"] > 0:
        buy += 10
    else:
        sell += 10

    if row["ema50_200"] > 0:
        buy += 8
    else:
        sell += 8

    if row["rsi14"] < 35:
        buy += 10

    elif row["rsi14"] > 65:
        sell += 10

    if row["macd_hist"] > 0:
        buy += 8
    else:
        sell += 8

    if row["bb_position"] < 0.25:
        buy += 10

    elif row["bb_position"] > 0.75:
        sell += 10

    if row["stoch_k"] < 25:
        buy += 8

    elif row["stoch_k"] > 75:
        sell += 8

    if row["close"] > row["open"]:
        buy += 8
    else:
        sell += 8

    return buy, sell


# ============================================================
# ANALYZE
# ============================================================

def analyze(
    symbol,
    df,
    crash_threshold,
    training_rows
):

    if df.empty:

        raise ValueError(
            "No data"
        )

    if len(df) < 300:

        raise ValueError(
            f"Only {len(df)} candles"
        )

    crash = calculate_crash(
        df,
        crash_threshold
    )

    pattern = weekly_pattern(
        df
    )

    tech_buy, tech_sell = (
        technical_score(df)
    )

    try:

        ai = ai_predict(
            df,
            training_rows
        )

        p_buy = ai["p_buy"]
        p_sell = ai["p_sell"]

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

        accuracy = ai[
            "accuracy"
        ]

        balanced = ai[
            "balanced_accuracy"
        ]

    except Exception:

        total = (
            tech_buy +
            tech_sell
        )

        if total <= 0:

            p_buy = 0.5
            p_sell = 0.5

        else:

            p_buy = (
                tech_buy /
                total
            )

            p_sell = (
                tech_sell /
                total
            )

        buy_score = tech_buy
        sell_score = tech_sell

        accuracy = np.nan
        balanced = np.nan

    # ========================================================
    # CRASH BONUS
    # ========================================================

    crash_bonus = 0

    if crash["crash_80"]:

        # Strong crash candidate.
        crash_bonus = 20

        buy_score += crash_bonus

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    # ========================================================
    # WEEKLY PATTERN BONUS
    # ========================================================

    pattern_bonus = 0

    if pattern["pattern"]:

        pattern_bonus = 20

        buy_score += pattern_bonus

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    # ========================================================
    # FINAL
    # ========================================================

    signal = (
        "BUY"
        if buy_score >= sell_score
        else
        "SELL"
    )

    completed = df.iloc[:-1]

    price = float(
        completed[
            "close"
        ].iloc[-1]
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

        "peak_10d": round(
            crash["peak_10d"],
            10
        ),

        "drop_10d": round(
            crash["drop_10d"],
            2
        ),

        "crash_80": crash[
            "crash_80"
        ],

        "crash_bonus": crash_bonus,

        "weekly_low": round(
            pattern["weekly_low"],
            10
        ),

        "weekly_pattern":
            pattern["pattern"],

        "three_rising":
            pattern["three_rising"],

        "green_last":
            pattern["green_last"],

        "ai_accuracy":
            round(
                accuracy * 100,
                2
            )
            if not pd.isna(accuracy)
            else np.nan,

        "balanced_accuracy":
            round(
                balanced * 100,
                2
            )
            if not pd.isna(balanced)
            else np.nan,

        "candles": len(df),

        "error": ""
    }


# ============================================================
# SCAN
# ============================================================

def run_scan(
    provider,
    symbols,
    candles,
    crash_threshold,
    training_rows
):

    results = []
    errors = []

    total = len(symbols)

    progress = st.progress(0)

    status = st.empty()

    for i, row in enumerate(
        symbols,
        start=1
    ):

        symbol = row["symbol"]

        status.write(
            f"📊 {i:,}/{total:,} → {symbol}"
        )

        try:

            df = load_data(
                provider,
                symbol,
                candles
            )

            result = analyze(
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

    progress.empty()
    status.empty()

    return (
        pd.DataFrame(results),
        pd.DataFrame(errors)
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
    "Bu uygulama emir göndermez. "
    "Sadece piyasa verisini analiz eder. "
    "AI tahminleri garanti değildir."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    provider = st.selectbox(
        "Data Provider",
        [
            "AUTO",
            "Binance",
            "Gate.io"
        ]
    )

    coin_count = st.number_input(
        "Minimum Coin Count",
        min_value=1000,
        max_value=3000,
        value=1000,
        step=100
    )

    crash_threshold = st.slider(
        "10-Day Crash Threshold (%)",
        min_value=50.0,
        max_value=95.0,
        value=CRASH_THRESHOLD_DEFAULT,
        step=1.0
    )

    training_rows = st.number_input(
        "AI Training Candles",
        min_value=400,
        max_value=1000,
        value=700,
        step=100
    )

    st.divider()

    st.write(
        "TIMEFRAME"
    )

    st.write(
        "15 minutes"
    )

    st.write(
        "10 days = 960 candles"
    )

    st.divider()

    st.write(
        "TARGET"
    )

    st.write(
        "BUY = +10% / next 2 candles"
    )

    st.write(
        "SELL = -10% / next 2 candles"
    )

    st.write(
        "HOLD = disabled"
    )

    st.divider()

    st.write(
        "CRASH"
    )

    st.write(
        f"Peak → current >= {crash_threshold:.0f}%"
    )

    st.write(
        "Crash BUY bonus = +20"
    )

    st.divider()

    st.write(
        "WEEKLY PATTERN"
    )

    st.write(
        "672 × 15M candles"
    )

    st.write(
        "3 rising candles"
    )

    st.write(
        "Last candle green"
    )

    st.write(
        "Near weekly low"
    )

    st.write(
        "Pattern BUY bonus = +20"
    )


# ============================================================
# AUTO PROVIDER
# ============================================================

active_provider = provider

if provider == "AUTO":

    try:

        binance_test = get_binance_symbols()

        if len(binance_test) >= MIN_SYMBOLS:

            active_provider = "Binance"

        else:

            active_provider = "Gate.io"

    except Exception:

        active_provider = "Gate.io"


st.info(
    f"Active provider: **{active_provider}**"
)


# ============================================================
# SYMBOLS
# ============================================================

try:

    symbols_df = select_symbols(
        active_provider,
        int(coin_count)
    )

except Exception as e:

    symbols_df = pd.DataFrame()

    st.error(
        f"Symbol list error: {e}"
    )


if symbols_df.empty:

    st.error(
        "Coin list could not be loaded."
    )

    st.stop()


# ============================================================
# SYMBOL COUNT CHECK
# ============================================================

available_count = len(
    symbols_df
)

if available_count < MIN_SYMBOLS:

    st.warning(
        f"Bu provider yalnızca "
        f"{available_count:,} uygun USDT Spot çift döndürdü. "
        f"Hedef minimum {MIN_SYMBOLS:,}."
    )

else:

    st.success(
        f"✅ {available_count:,} coin hazır."
    )


# ============================================================
# COIN SELECTION
# ============================================================

scope = st.radio(
    "Scan Scope",
    [
        "ALL SELECTED COINS",
        "MANUAL"
    ],
    horizontal=True
)

if scope == "ALL SELECTED COINS":

    selected = symbols_df.copy()

else:

    selected_names = st.multiselect(
        "Select coins",
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
        ].isin(
            selected_names
        )
    ].copy()


st.write(
    f"### Tarama: {len(selected):,} coin"
)


# ============================================================
# START
# ============================================================

start = st.button(
    "🚀 START 1000+ COIN SCAN",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN
# ============================================================

if start:

    started = time.time()

    with st.spinner(
        "1000+ coin taranıyor..."
    ):

        results, errors = run_scan(
            provider=active_provider,
            symbols=selected.to_dict(
                "records"
            ),
            candles=960,
            crash_threshold=crash_threshold,
            training_rows=int(
                training_rows
            )
        )

    elapsed = (
        time.time()
        -
        started
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

    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Scanned",
        f"{len(results):,}"
    )

    c2.metric(
        "Errors",
        f"{len(errors):,}"
    )

    if not results.empty:

        buy_count = int(
            (
                results["signal"]
                ==
                "BUY"
            ).sum()
        )

        sell_count = int(
            (
                results["signal"]
                ==
                "SELL"
            ).sum()
        )

        crash_count = int(
            results["crash_80"]
            .sum()
        )

    else:

        buy_count = 0
        sell_count = 0
        crash_count = 0

    c3.metric(
        "BUY",
        f"{buy_count:,}"
    )

    c4.metric(
        "SELL",
        f"{sell_count:,}"
    )

    c5.metric(
        "CRASH 80%+",
        f"{crash_count:,}"
    )

    st.caption(
        f"Scan time: {elapsed:.1f} seconds"
    )


    if not results.empty:

        # ====================================================
        # CRASH LIST
        # ====================================================

        st.subheader(
            f"🔥 SON 10 GÜNDE %{crash_threshold:.0f}+ ÇAKILANLAR"
        )

        crash_df = results[
            results["crash_80"]
            ==
            True
        ].copy()

        if crash_df.empty:

            st.info(
                f"Son 10 günde %{crash_threshold:.0f}+ "
                "düşen coin bulunamadı."
            )

        else:

            crash_df = crash_df.sort_values(
                [
                    "drop_10d",
                    "buy_score"
                ],
                ascending=False
            )

            crash_display = crash_df[
                [
                    "symbol",
                    "signal",
                    "price",
                    "peak_10d",
                    "drop_10d",
                    "p_buy",
                    "buy_score",
                    "sell_score",
                    "weekly_pattern",
                    "ai_accuracy"
                ]
            ].copy()

            crash_display = (
                crash_display.rename(
                    columns={
                        "symbol": "COIN",
                        "signal": "SIGNAL",
                        "price": "PRICE",
                        "peak_10d": "10D PEAK",
                        "drop_10d": "DROP %",
                        "p_buy": "AI BUY %",
                        "buy_score": "BUY SCORE",
                        "sell_score": "SELL SCORE",
                        "weekly_pattern": "WEEKLY PATTERN",
                        "ai_accuracy": "AI ACCURACY %"
                    }
                )
            )

            st.dataframe(
                crash_display,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # TOP BUY
        # ====================================================

        st.subheader(
            "🟢 TOP BUY"
        )

        buy_df = results[
            results["signal"]
            ==
            "BUY"
        ].copy()

        buy_df = buy_df.sort_values(
            [
                "buy_score",
                "p_buy",
                "drop_10d"
            ],
            ascending=False
        ).head(100)

        st.dataframe(
            buy_df[
                [
                    "symbol",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_buy",
                    "drop_10d",
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


        # ====================================================
        # TOP SELL
        # ====================================================

        st.subheader(
            "🔴 TOP SELL"
        )

        sell_df = results[
            results["signal"]
            ==
            "SELL"
        ].copy()

        sell_df = sell_df.sort_values(
            [
                "sell_score",
                "p_sell"
            ],
            ascending=False
        ).head(100)

        st.dataframe(
            sell_df[
                [
                    "symbol",
                    "price",
                    "buy_score",
                    "sell_score",
                    "p_sell",
                    "drop_10d",
                    "crash_80",
                    "weekly_pattern",
                    "ai_accuracy"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # WEEKLY PATTERN
        # ====================================================

        st.subheader(
            "🔥 WEEKLY LOW + 3 RISING"
        )

        weekly_df = results[
            results["weekly_pattern"]
            ==
            True
        ].copy()

        weekly_df = weekly_df.sort_values(
            "buy_score",
            ascending=False
        ).head(100)

        if weekly_df.empty:

            st.info(
                "Pattern bulunamadı."
            )

        else:

            st.dataframe(
                weekly_df[
                    [
                        "symbol",
                        "price",
                        "weekly_low",
                        "buy_score",
                        "p_buy",
                        "drop_10d",
                        "crash_80",
                        "three_rising",
                        "green_last"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # ALL RESULTS
        # ====================================================

        st.subheader(
            "📊 ALL RESULTS"
        )

        all_results = results.sort_values(
            "buy_score",
            ascending=False
        )

        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # CSV
        # ====================================================

        csv = results.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ DOWNLOAD ALL RESULTS CSV",
            data=csv,
            file_name=(
                "1000_crypto_ai_scan.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


    # ========================================================
    # ERRORS
    # ========================================================

    if not errors.empty:

        with st.expander(
            f"⚠️ API / DATA ERRORS ({len(errors):,})"
        ):

            st.dataframe(
                errors,
                use_container_width=True,
                hide_index=True
            )


else:

    st.info(
        "START 1000+ COIN SCAN butonuna bas."
    )

    st.markdown(
        """
### Tarama mantığı

**1.000+ Spot USDT coin**

**15 dakika**

**Son 10 gün = 960 mum**

**10 günlük peak → mevcut fiyat**

`Düşüş % = (Peak - Current) / Peak × 100`

Varsayılan:

`%80+ = CRASH`

### BUY

AI + teknik analiz + crash filtresi + haftalık dip patterni.

### SELL

AI + teknik analiz.

### HOLD

Kullanılmıyor.

### Haftalık pattern

Son 672 tamamlanmış 15M mum içerisinde:

- haftalık minimum
- minimumun son 3 mumdan önce olması
- son 3 mumda yükselen kapanış
- son mum yeşil
- son 3 mumun dibi haftalık dibe %1.5 yakın

Bu koşullar oluşursa BUY skoruna ekstra puan eklenir.
"""
    )
```

`requirements.txt` aynı kalabilir:

```text
streamlit==1.50.0
requests==2.32.5
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
```

**Not:** Gate.io'nun resmi API'sinde `/spot/currency_pairs` tüm desteklenen Spot çiftlerini, `/spot/candlesticks` ise 15m dahil çeşitli aralıkları sağlıyor; candlestick endpoint'i sorgu başına maksimum 1.000 nokta döndürüyor.

Bu nedenle bu sürümde **Binance 451 alırsan uygulama tamamen çökmek yerine Gate.io'ya geçebilir**. Ayrıca `CRASH_THRESHOLD` artık arayüzden örneğin **%70 / %75 / %80 / %85 / %90** yapılabilir.
