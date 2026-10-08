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
# PAGE
# ============================================================

st.set_page_config(
    page_title="Binance Spot BUY SELL AI Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

BINANCE_URL = "https://api.binance.com"
GATE_URL = "https://api.gateio.ws"

INTERVAL = "15m"

# 10 gün / 15 dakika
CRASH_BARS = 960

# 1 hafta / 15 dakika
WEEKLY_BARS = 672

# 30 dakika = 2 x 15M
PREDICTION_BARS = 2

# BUY hedefi
BUY_TARGET = 0.10

# SELL hedefi
SELL_TARGET = -0.10

# Minimum AI veri
MIN_DATA = 300

# Varsayılan eğitim
DEFAULT_TRAINING_ROWS = 700


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
    "low_distance"
]


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/154.0 Safari/537.36"
    ),
    "Accept": "application/json"
})


# ============================================================
# GENERIC HTTP
# ============================================================

def get_json(url, params=None, timeout=30):

    response = session.get(
        url,
        params=params,
        timeout=timeout
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

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

        symbol = item.get("symbol")

        if not symbol:
            continue

        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get(
            "isSpotTradingAllowed"
        ) is False:
            continue

        rows.append({
            "symbol": symbol,
            "base": item.get(
                "baseAsset",
                ""
            ),
            "quote": "USDT"
        })

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "Binance USDT Spot pariteleri bulunamadı."
        )

    return df


# ============================================================
# BINANCE 24H DATA
# ============================================================

@st.cache_data(ttl=120)
def get_binance_24h():

    data = get_json(
        BINANCE_URL
        + "/api/v3/ticker/24hr"
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
            quote_volume = float(
                item.get(
                    "quoteVolume",
                    0
                )
                or 0
            )
        except Exception:
            quote_volume = 0.0

        try:
            change = float(
                item.get(
                    "priceChangePercent",
                    0
                )
                or 0
            )
        except Exception:
            change = 0.0

        try:
            trades = int(
                item.get(
                    "count",
                    0
                )
                or 0
            )
        except Exception:
            trades = 0

        rows.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "24h_change": change,
            "trade_count": trades
        })

    return pd.DataFrame(rows)


# ============================================================
# GATE PAIRS
# ============================================================

@st.cache_data(ttl=300)
def get_gate_pairs():

    data = get_json(
        GATE_URL
        + "/api/v4/spot/currency_pairs"
    )

    rows = []

    for item in data:

        quote = str(
            item.get(
                "quote",
                ""
            )
        ).upper()

        if quote != "USDT":
            continue

        status = str(
            item.get(
                "trade_status",
                ""
            )
        ).lower()

        if status:

            if status not in {
                "tradable",
                "buyable",
                "sellable"
            }:
                continue

        pair = item.get("id")

        if not pair:
            continue

        base = item.get(
            "base",
            ""
        )

        rows.append({
            "gate_pair": pair,
            "base": base,
            "quote": "USDT"
        })

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "Gate.io USDT pariteleri alınamadı."
        )

    return df


# ============================================================
# GATE TICKERS
# ============================================================

@st.cache_data(ttl=120)
def get_gate_tickers():

    data = get_json(
        GATE_URL
        + "/api/v4/spot/tickers"
    )

    rows = []

    for item in data:

        pair = item.get(
            "currency_pair",
            ""
        )

        if not pair.endswith(
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

        try:

            change = float(
                item.get(
                    "change_percentage",
                    0
                )
                or 0
            )

        except Exception:

            change = 0.0

        rows.append({
            "gate_pair": pair,
            "gate_volume": volume,
            "gate_change": change
        })

    return pd.DataFrame(rows)


# ============================================================
# BINANCE -> GATE SYMBOL
# ============================================================

def binance_to_gate(symbol):

    if symbol.endswith(
        "USDT"
    ):

        base = symbol[
            :-4
        ]

        return (
            base
            + "_USDT"
        )

    return None


# ============================================================
# BUILD MARKET LIST
# ============================================================

def build_market_list():

    binance = get_binance_symbols()

    try:

        binance_24h = (
            get_binance_24h()
        )

        binance = binance.merge(
            binance_24h,
            on="symbol",
            how="left"
        )

    except Exception:

        binance[
            "quote_volume"
        ] = 0

        binance[
            "24h_change"
        ] = 0

        binance[
            "trade_count"
        ] = 0

    binance[
        "quote_volume"
    ] = pd.to_numeric(
        binance[
            "quote_volume"
        ],
        errors="coerce"
    ).fillna(0)

    binance[
        "24h_change"
    ] = pd.to_numeric(
        binance[
            "24h_change"
        ],
        errors="coerce"
    ).fillna(0)

    binance[
        "trade_count"
    ] = pd.to_numeric(
        binance[
            "trade_count"
        ],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Gate pair
    # --------------------------------------------------------

    binance[
        "gate_pair"
    ] = binance[
        "symbol"
    ].apply(
        binance_to_gate
    )

    try:

        gate_pairs = (
            get_gate_pairs()
        )

        binance = binance.merge(
            gate_pairs[
                [
                    "gate_pair"
                ]
            ],
            on="gate_pair",
            how="left",
            suffixes=(
                "",
                "_gate"
            )
        )

        binance[
            "gate_available"
        ] = binance[
            "gate_pair"
        ].isin(
            gate_pairs[
                "gate_pair"
            ]
        )

    except Exception:

        binance[
            "gate_available"
        ] = False

    # --------------------------------------------------------
    # Gate volume
    # --------------------------------------------------------

    try:

        gate_tickers = (
            get_gate_tickers()
        )

        binance = binance.merge(
            gate_tickers,
            on="gate_pair",
            how="left"
        )

    except Exception:

        binance[
            "gate_volume"
        ] = 0

        binance[
            "gate_change"
        ] = 0

    binance[
        "gate_volume"
    ] = pd.to_numeric(
        binance[
            "gate_volume"
        ],
        errors="coerce"
    ).fillna(0)

    binance[
        "gate_change"
    ] = pd.to_numeric(
        binance[
            "gate_change"
        ],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Activity score
    # --------------------------------------------------------

    volume_log = np.log1p(
        binance[
            "quote_volume"
        ].clip(
            lower=0
        )
    )

    trade_log = np.log1p(
        binance[
            "trade_count"
        ].clip(
            lower=0
        )
    )

    movement = (
        binance[
            "24h_change"
        ]
        .abs()
    )

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum <= minimum:

            return pd.Series(
                1.0,
                index=series.index
            )

        return (
            (series - minimum)
            / (
                maximum
                - minimum
            )
        )

    volume_norm = normalize(
        volume_log
    )

    trade_norm = normalize(
        trade_log
    )

    movement_norm = normalize(
        movement
    )

    binance[
        "activity_score"
    ] = (
        volume_norm * 60
        + trade_norm * 20
        + movement_norm * 20
    )

    binance = (
        binance
        .sort_values(
            [
                "activity_score",
                "quote_volume"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return binance


# ============================================================
# GATE CANDLES
# ============================================================

def get_gate_candles(
    gate_pair,
    limit=960
):

    if not gate_pair:

        return pd.DataFrame()

    data = get_json(
        GATE_URL
        + "/api/v4/spot/candlesticks",
        params={
            "currency_pair": gate_pair,
            "interval": "15m",
            "limit": min(
                int(limit),
                1000
            )
        },
        timeout=30
    )

    rows = []

    for item in data:

        if len(item) < 8:
            continue

        try:

            timestamp = int(
                item[0]
            )

            volume = float(
                item[1]
            )

            close = float(
                item[2]
            )

            high = float(
                item[3]
            )

            low = float(
                item[4]
            )

            open_price = float(
                item[5]
            )

            closed = str(
                item[7]
            ).lower()

        except Exception:

            continue

        rows.append({
            "timestamp": pd.to_datetime(
                timestamp,
                unit="s",
                utc=True
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "closed": closed
        })

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    # Sadece tamamlanmış mumlar
    if "closed" in df.columns:

        closed_values = (
            df["closed"]
            .astype(str)
            .str.lower()
        )

        mask = closed_values.isin(
            [
                "true",
                "1"
            ]
        )

        if mask.any():

            df = df[
                mask
            ].copy()

    df = (
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

    return df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


# ============================================================
# BINANCE DIRECT CANDLES
# ============================================================

def get_binance_candles(
    symbol,
    limit=960
):

    data = get_json(
        BINANCE_URL
        + "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "15m",
            "limit": min(
                int(limit),
                1000
            )
        },
        timeout=30
    )

    rows = []

    for item in data:

        if len(item) < 6:
            continue

        rows.append({
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
        })

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
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100
        - 100
        / (1 + rs)
    )

    return result.fillna(
        50
    )


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period=14
):

    previous = (
        df["close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["high"]
            - df["low"],

            (
                df["high"]
                - previous
            ).abs(),

            (
                df["low"]
                - previous
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

    up = df["high"].diff()

    down = -df["low"].diff()

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

    atr_value = (
        atr(
            df,
            period
        )
        .replace(
            0,
            np.nan
        )
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di
        + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        * (
            plus_di
            - minus_di
        ).abs()
        / denominator
    )

    return (
        dx.ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
        .fillna(0)
    )


# ============================================================
# FEATURES
# ============================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret6"] = close.pct_change(6)
    x["ret12"] = close.pct_change(12)
    x["ret24"] = close.pct_change(24)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_line = (
        ema(close, 12)
        - ema(close, 26)
    )

    macd_signal = ema(
        macd_line,
        9
    )

    x["macd"] = macd_line

    x["macd_signal"] = (
        macd_signal
    )

    x["macd_hist"] = (
        macd_line
        - macd_signal
    )

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    middle = (
        close
        .rolling(20)
        .mean()
    )

    std = (
        close
        .rolling(20)
        .std()
    )

    upper = (
        middle
        + 2 * std
    )

    lower = (
        middle
        - 2 * std
    )

    width = (
        upper
        - lower
    ).replace(
        0,
        np.nan
    )

    x["bb_position"] = (
        (close - lower)
        / width
    )

    x["bb_width"] = (
        width
        / middle.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_value = atr(
        x,
        14
    )

    x["atr_pct"] = (
        atr_value
        / close.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    x["adx"] = adx(
        x,
        14
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

    denominator = (
        high14
        - low14
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        * (close - low14)
        / denominator
    )

    d = (
        k
        .rolling(3)
        .mean()
    )

    x["stoch_k"] = k.fillna(
        50
    )

    x["stoch_d"] = d.fillna(
        50
    )

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan
    )

    body = (
        close
        - x["open"]
    ).abs()

    x["body_pct"] = (
        body
        / close.replace(
            0,
            np.nan
        )
    )

    x["range_pct"] = (
        candle_range
        / close.replace(
            0,
            np.nan
        )
    )

    x["upper_wick"] = (
        high
        - np.maximum(
            x["open"],
            close
        )
    ) / candle_range

    x["lower_wick"] = (
        np.minimum(
            x["open"],
            close
        )
        - low
    ) / candle_range

    # --------------------------------------------------------
    # Volume
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
        / volume_mean.replace(
            0,
            np.nan
        )
    )

    x["volume_z"] = (
        volume
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Price Z
    # --------------------------------------------------------

    price_mean = (
        close
        .rolling(48)
        .mean()
    )

    price_std = (
        close
        .rolling(48)
        .std()
    )

    x["price_z"] = (
        close
        - price_mean
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
# CRASH
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

    completed = df.iloc[:-1]

    window = (
        completed
        .tail(CRASH_BARS)
    )

    if window.empty:
        return result

    peak = float(
        window["high"].max()
    )

    price = float(
        completed[
            "close"
        ].iloc[-1]
    )

    if peak <= 0:
        return result

    drop = (
        1
        - price / peak
    ) * 100

    result["peak"] = peak
    result["price"] = price
    result["drop"] = drop

    result["is_crash"] = (
        drop >= threshold
    )

    return result


# ============================================================
# WEEKLY LOW + 3 RISING
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

    completed = df.iloc[:-1]

    weekly = (
        completed
        .tail(WEEKLY_BARS)
    )

    last3 = (
        completed
        .tail(3)
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
        recent_low
        / weekly_low
        - 1
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

    weekly_low_index = (
        weekly[
            "low"
        ].idxmin()
    )

    first_index = (
        last3.index[0]
    )

    dip_before = (
        weekly_low_index
        < first_index
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
        and green
        and near_low
        and dip_before
    )

    return result


# ============================================================
# TARGET
# ============================================================

def make_target(df):

    target = np.full(
        len(df),
        np.nan
    )

    close = (
        df["close"]
        .values
    )

    high = (
        df["high"]
        .values
    )

    low = (
        df["low"]
        .values
    )

    for i in range(
        len(df)
        - PREDICTION_BARS
    ):

        entry = close[i]

        future_high = np.max(
            high[
                i + 1:
                i + 1
                + PREDICTION_BARS
            ]
        )

        future_low = np.min(
            low[
                i + 1:
                i + 1
                + PREDICTION_BARS
            ]
        )

        up = (
            future_high
            / entry
            - 1
        )

        down = (
            future_low
            / entry
            - 1
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

            # HOLD YOK
            # Daha büyük hareket hangi yöndeyse
            # o sınıfa veriliyor.
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

    row = features.iloc[-2]

    buy = 0.0
    sell = 0.0

    # EMA 5/20
    if row[
        "ema5_20"
    ] > 0:

        buy += 10

    else:

        sell += 10

    # EMA 20/50
    if row[
        "ema20_50"
    ] > 0:

        buy += 10

    else:

        sell += 10

    # EMA 50/200
    if row[
        "ema50_200"
    ] > 0:

        buy += 8

    else:

        sell += 8

    # RSI
    if row[
        "rsi14"
    ] < 35:

        buy += 10

    elif row[
        "rsi14"
    ] > 65:

        sell += 10

    # MACD
    if row[
        "macd_hist"
    ] > 0:

        buy += 8

    else:

        sell += 8

    # Bollinger
    if row[
        "bb_position"
    ] < 0.25:

        buy += 10

    elif row[
        "bb_position"
    ] > 0.75:

        sell += 10

    # Stochastic
    if row[
        "stoch_k"
    ] < 25:

        buy += 8

    elif row[
        "stoch_k"
    ] > 75:

        sell += 8

    # Son mum
    if row[
        "close"
    ] > row[
        "open"
    ]:

        buy += 8

    else:

        sell += 8

    return buy, sell


# ============================================================
# DEEP LEARNING MODEL
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
                        96,
                        48,
                        24
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
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
        .all(axis=1)
        &
        target_series.notna()
    )

    X = feature_df.loc[
        valid
    ].values

    y = (
        target_series.loc[
            valid
        ]
        .astype(int)
        .values
    )

    if len(X) < 300:

        raise ValueError(
            "AI için yeterli veri yok."
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
            "AI tek sınıf gördü."
        )

    split = int(
        len(X) * 0.80
    )

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    if len(
        np.unique(y_train)
    ) < 2:

        raise ValueError(
            "Training set tek sınıf."
        )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    prediction = (
        model.predict(
            X_test
        )
    )

    accuracy = float(
        np.mean(
            prediction
            == y_test
        )
    )

    latest = feature_df.iloc[
        -2:-1
    ]

    probabilities = (
        model.predict_proba(
            latest.values
        )[0]
    )

    p_buy = 0.0
    p_sell = 0.0

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
        "accuracy": accuracy
    }


# ============================================================
# ANALYZE
# ============================================================

def analyze_symbol(
    symbol,
    df,
    crash_threshold,
    training_rows
):

    if df.empty:

        raise ValueError(
            "Mum verisi boş."
        )

    if len(df) < MIN_DATA:

        raise ValueError(
            f"Yetersiz mum: {len(df)}"
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

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

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

        ai_status = "OK"

        buy_score = (
            p_buy * 100
            + tech_buy * 0.35
        )

        sell_score = (
            p_sell * 100
            + tech_sell * 0.35
        )

    except Exception as e:

        # AI başarısız olursa
        # teknik model yine BUY/SELL verir.

        total = (
            tech_buy
            + tech_sell
        )

        if total > 0:

            p_buy = (
                tech_buy
                / total
            )

            p_sell = (
                tech_sell
                / total
            )

        else:

            p_buy = 0.5
            p_sell = 0.5

        buy_score = tech_buy
        sell_score = tech_sell

        accuracy = np.nan

        ai_status = (
            "TECHNICAL FALLBACK: "
            + str(e)
        )

    # --------------------------------------------------------
    # CRASH BONUS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # WEEKLY BONUS
    # --------------------------------------------------------

    weekly_bonus = 0

    if weekly[
        "pattern"
    ]:

        weekly_bonus = 20

        buy_score += 20

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    if buy_score >= sell_score:

        signal = "BUY"

    else:

        signal = "SELL"

    price = float(
        df.iloc[-2][
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

        "crash": bool(
            crash[
                "is_crash"
            ]
        ),

        "crash_bonus": (
            crash_bonus
        ),

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

        "weekly_bonus": (
            weekly_bonus
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

        "ai_status": ai_status,

        "candles": len(df)
    }


# ============================================================
# LOAD DATA FOR ONE PAIR
# ============================================================

def load_pair_data(
    symbol,
    provider,
    gate_pair
):

    # ========================================================
    # BINANCE DIRECT
    # ========================================================

    if provider == "BINANCE":

        return get_binance_candles(
            symbol,
            CRASH_BARS
        )

    # ========================================================
    # GATE FALLBACK
    # ========================================================

    if provider == "GATE_FALLBACK":

        if not gate_pair:

            raise ValueError(
                "Gate.io eşleşmesi yok."
            )

        return get_gate_candles(
            gate_pair,
            CRASH_BARS
        )

    raise ValueError(
        "Bilinmeyen data provider."
    )


# ============================================================
# SCAN
# ============================================================

def run_scan(
    selected,
    provider,
    crash_threshold,
    training_rows,
    delay
):

    results = []

    errors = []

    total = len(
        selected
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    start = time.time()

    for i, item in enumerate(
        selected,
        start=1
    ):

        symbol = item[
            "symbol"
        ]

        status.write(
            f"{i:,}/{total:,} → {symbol}"
        )

        try:

            df = load_pair_data(
                symbol,
                provider,
                item.get(
                    "gate_pair"
                )
            )

            if df.empty:

                raise ValueError(
                    "Mum verisi alınamadı."
                )

            result = analyze_symbol(
                symbol,
                df,
                crash_threshold,
                training_rows
            )

            result[
                "activity_score"
            ] = round(
                float(
                    item.get(
                        "activity_score",
                        0
                    )
                ),
                3
            )

            result[
                "24h_volume"
            ] = float(
                item.get(
                    "quote_volume",
                    0
                )
                or 0
            )

            result[
                "24h_change"
            ] = float(
                item.get(
                    "24h_change",
                    0
                )
                or 0
            )

            result[
                "trade_count"
            ] = int(
                item.get(
                    "trade_count",
                    0
                )
                or 0
            )

            result[
                "gate_pair"
            ] = item.get(
                "gate_pair",
                ""
            )

            result[
                "provider"
            ] = provider

            results.append(
                result
            )

        except Exception as e:

            errors.append({
                "symbol": symbol,
                "gate_pair": item.get(
                    "gate_pair",
                    ""
                ),
                "error": str(e)
            })

        progress.progress(
            i / total
        )

        if delay > 0:

            time.sleep(
                delay
            )

    progress.empty()
    status.empty()

    elapsed = (
        time.time()
        - start
    )

    return (
        pd.DataFrame(
            results
        ),
        pd.DataFrame(
            errors
        ),
        elapsed
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Binance Spot "
    "BUY / SELL Deep Learning Scanner"
)

st.caption(
    "Binance Spot USDT pariteleri • "
    "15M • 30 dakika tahmin • "
    "Deep Learning • BUY / SELL"
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
        "⚙️ Scanner Settings"
    )

    crash_threshold = st.slider(
        "10 Günlük Düşüş %",
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

    delay = st.number_input(
        "API Delay",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.01
    )

    st.divider()

    st.write(
        "📊 Binance Spot"
    )

    st.write(
        "💱 USDT pariteleri"
    )

    st.write(
        "⏱️ 15M"
    )

    st.write(
        "🔮 30 dakika"
    )

    st.write(
        "🟢 BUY +10%"
    )

    st.write(
        "🔴 SELL -10%"
    )

    st.write(
        "🚫 HOLD YOK"
    )

    st.write(
        "🔥 Crash bonus +20"
    )

    st.write(
        "🔥 Weekly bonus +20"
    )


# ============================================================
# MARKET LOAD
# ============================================================

st.subheader(
    "🌐 Binance Spot Pariteleri"
)

# ------------------------------------------------------------
# Önce Binance
# ------------------------------------------------------------

try:

    pairs_df = build_market_list()

    provider = "BINANCE"

    st.success(
        "🟢 Binance API doğrudan erişilebilir."
    )

except Exception as binance_error:

    # --------------------------------------------------------
    # Binance 451
    # --------------------------------------------------------

    st.warning(
        "🟡 Binance API bu sunucudan erişilemiyor. "
        "Gate.io fallback devreye alınıyor."
    )

    with st.expander(
        "Binance API hatası"
    ):

        st.code(
            str(binance_error)
        )

    # --------------------------------------------------------
    # Gate fallback
    # --------------------------------------------------------

    try:

        # Binance paritelerini tekrar almayı
        # denemiyoruz çünkü 451 zaten biliniyor.
        #
        # Gate üzerinden market listesi oluşturuyoruz.
        gate_pairs = get_gate_pairs()

        gate_tickers = get_gate_tickers()

        pairs_df = gate_pairs.copy()

        pairs_df = pairs_df.merge(
            gate_tickers,
            on="gate_pair",
            how="left"
        )

        pairs_df[
            "gate_volume"
        ] = pd.to_numeric(
            pairs_df[
                "gate_volume"
            ],
            errors="coerce"
        ).fillna(0)

        pairs_df[
            "gate_change"
        ] = pd.to_numeric(
            pairs_df[
                "gate_change"
            ],
            errors="coerce"
        ).fillna(0)

        # Binance formatına çevir
        pairs_df[
            "symbol"
        ] = pairs_df[
            "base"
        ].astype(str).str.upper() + "USDT"

        pairs_df[
            "quote"
        ] = "USDT"

        pairs_df[
            "quote_volume"
        ] = pairs_df[
            "gate_volume"
        ]

        pairs_df[
            "24h_change"
        ] = pairs_df[
            "gate_change"
        ]

        pairs_df[
            "trade_count"
        ] = 0

        pairs_df[
            "gate_available"
        ] = True

        # Aktivite
        volume_log = np.log1p(
            pairs_df[
                "quote_volume"
            ].clip(
                lower=0
            )
        )

        movement = (
            pairs_df[
                "24h_change"
            ]
            .abs()
        )

        def normalize_fallback(
            series
        ):

            minimum = series.min()
            maximum = series.max()

            if maximum <= minimum:

                return pd.Series(
                    1.0,
                    index=series.index
                )

            return (
                (
                    series
                    - minimum
                )
                / (
                    maximum
                    - minimum
                )
            )

        pairs_df[
            "activity_score"
        ] = (
            normalize_fallback(
                volume_log
            ) * 80
            +
            normalize_fallback(
                movement
            ) * 20
        )

        pairs_df = (
            pairs_df
            .sort_values(
                "activity_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        provider = (
            "GATE_FALLBACK"
        )

        st.success(
            f"🟢 Gate.io fallback aktif. "
            f"{len(pairs_df):,} USDT paritesi bulundu."
        )

    except Exception as gate_error:

        st.error(
            "Hem Binance hem Gate.io verisi alınamadı."
        )

        st.code(
            f"BINANCE:\n"
            f"{binance_error}\n\n"
            f"GATE.IO:\n"
            f"{gate_error}"
        )

        st.stop()


# ============================================================
# STATISTICS
# ============================================================

total_pairs = len(
    pairs_df
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "USDT Pariteleri",
    f"{total_pairs:,}"
)

if not pairs_df.empty:

    c2.metric(
        "En Aktif",
        pairs_df.iloc[
            0
        ]["symbol"]
    )

    c3.metric(
        "En Yüksek Aktivite",
        f"{float(pairs_df.iloc[0]['activity_score']):.2f}"
    )

else:

    c2.metric(
        "En Aktif",
        "-"
    )

    c3.metric(
        "Aktivite",
        "-"
    )

c4.metric(
    "Provider",
    (
        "Binance"
        if provider == "BINANCE"
        else "Gate Fallback"
    )
)


# ============================================================
# ALL PAIRS
# ============================================================

with st.expander(
    "📋 TÜM PARİTELER"
):

    columns = [
        "symbol",
        "gate_pair",
        "quote_volume",
        "24h_change",
        "trade_count",
        "activity_score"
    ]

    available = [
        column
        for column in columns
        if column in pairs_df.columns
    ]

    st.dataframe(
        pairs_df[
            available
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SCAN MODE
# ============================================================

st.subheader(
    "🔎 Tarama Modu"
)

mode = st.radio(
    "Parite seçimi",
    [
        "ALL PAIRS",
        "TOP ACTIVE",
        "MANUAL"
    ],
    horizontal=True
)


# ============================================================
# ALL
# ============================================================

if mode == "ALL PAIRS":

    selected = pairs_df.copy()

    st.info(
        f"Tüm {len(selected):,} "
        "USDT Spot paritesi taranacak."
    )


# ============================================================
# TOP ACTIVE
# ============================================================

elif mode == "TOP ACTIVE":

    max_count = max(
        1,
        len(pairs_df)
    )

    default_count = min(
        1000,
        max_count
    )

    count = st.number_input(
        "Aktif parite sayısı",
        min_value=1,
        max_value=max_count,
        value=default_count,
        step=100
    )

    selected = (
        pairs_df
        .head(
            int(count)
        )
        .copy()
    )

    st.info(
        f"En aktif {len(selected):,} "
        "parite taranacak."
    )


# ============================================================
# MANUAL
# ============================================================

else:

    symbol_list = (
        pairs_df[
            "symbol"
        ]
        .tolist()
    )

    chosen = st.multiselect(
        "Pariteleri seç",
        symbol_list,
        default=symbol_list[:20]
    )

    selected = (
        pairs_df[
            pairs_df[
                "symbol"
            ].isin(
                chosen
            )
        ]
        .copy()
    )


# ============================================================
# SELECTED
# ============================================================

st.write(
    f"**Seçilen parite:** "
    f"{len(selected):,}"
)


# ============================================================
# START
# ============================================================

start_scan = st.button(
    "🚀 BUY / SELL DEEP LEARNING SCAN",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTE
# ============================================================

if start_scan:

    if selected.empty:

        st.error(
            "Tarama için parite seçilmedi."
        )

        st.stop()

    scan_time = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    with st.spinner(
        "BUY / SELL deep learning "
        "taraması yapılıyor..."
    ):

        results, errors, elapsed = (
            run_scan(
                selected.to_dict(
                    "records"
                ),
                provider,
                float(
                    crash_threshold
                ),
                int(
                    training_rows
                ),
                float(
                    delay
                )
            )
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
    ] = scan_time


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

    scan_time = st.session_state[
        "scan_time"
    ]

    st.divider()

    # ========================================================
    # METRICS
    # ========================================================

    buy_count = 0
    sell_count = 0
    crash_count = 0

    if not results.empty:

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

        crash_count = int(
            results[
                "crash"
            ].sum()
        )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "BAŞARILI",
        f"{len(results):,}"
    )

    c2.metric(
        "HATALI",
        f"{len(errors):,}"
    )

    c3.metric(
        "🟢 BUY",
        f"{buy_count:,}"
    )

    c4.metric(
        "🔴 SELL",
        f"{sell_count:,}"
    )

    c5.metric(
        "🔥 CRASH",
        f"{crash_count:,}"
    )

    st.caption(
        f"Tarama zamanı: {scan_time} | "
        f"Süre: {elapsed:.1f} saniye"
    )


    # ========================================================
    # BUY
    # ========================================================

    st.subheader(
        "🟢 TOP BUY"
    )

    if not results.empty:

        buy_df = (
            results[
                results[
                    "signal"
                ] == "BUY"
            ]
            .sort_values(
                [
                    "buy_score",
                    "p_buy",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if buy_df.empty:

            st.info(
                "BUY sonucu yok."
            )

        else:

            st.dataframe(
                buy_df[
                    [
                        "symbol",
                        "price",
                        "buy_score",
                        "sell_score",
                        "p_buy",
                        "p_sell",
                        "activity_score",
                        "10d_drop",
                        "crash",
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
    # SELL
    # ========================================================

    st.subheader(
        "🔴 TOP SELL"
    )

    if not results.empty:

        sell_df = (
            results[
                results[
                    "signal"
                ] == "SELL"
            ]
            .sort_values(
                [
                    "sell_score",
                    "p_sell",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if sell_df.empty:

            st.info(
                "SELL sonucu yok."
            )

        else:

            st.dataframe(
                sell_df[
                    [
                        "symbol",
                        "price",
                        "sell_score",
                        "buy_score",
                        "p_sell",
                        "p_buy",
                        "activity_score",
                        "10d_drop",
                        "crash",
                        "weekly_pattern",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # CRASH
    # ========================================================

    st.subheader(
        f"🔥 SON 10 GÜNDE "
        f"%{crash_threshold:.0f}+ DÜŞENLER"
    )

    if not results.empty:

        crash_df = (
            results[
                results[
                    "10d_drop"
                ]
                >= crash_threshold
            ]
            .sort_values(
                [
                    "10d_drop",
                    "buy_score"
                ],
                ascending=False
            )
        )

        if crash_df.empty:

            st.info(
                "Crash filtresine uyan "
                "parite yok."
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
                        "activity_score",
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
        "🔥 HAFTALIK DİP + 3 YÜKSELEN MUM"
    )

    if not results.empty:

        weekly_df = (
            results[
                results[
                    "weekly_pattern"
                ] == True
            ]
            .sort_values(
                [
                    "buy_score",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if weekly_df.empty:

            st.info(
                "Weekly pattern bulunamadı."
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
                        "10d_drop",
                        "crash",
                        "three_rising",
                        "green_last",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ALL RESULTS
    # ========================================================

    st.subheader(
        "📊 TÜM BUY / SELL SONUÇLARI"
    )

    if not results.empty:

        all_results = (
            results
            .sort_values(
                [
                    "activity_score",
                    "buy_score"
                ],
                ascending=False
            )
        )

        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )

        csv = (
            all_results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ SONUÇLARI CSV İNDİR",
            data=csv,
            file_name=(
                "binance_buy_sell_ai_scan.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


    # ========================================================
    # ERRORS
    # ========================================================

    st.subheader(
        "⚠️ HATALI PARİTELER"
    )

    if errors.empty:

        st.success(
            "Hatalı parite yok."
        )

    else:

        st.warning(
            f"{len(errors):,} paritede "
            "veri alınamadı."
        )

        st.dataframe(
            errors,
            use_container_width=True,
            hide_index=True
        )

        error_csv = (
            errors
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ HATALI PARİTELERİ İNDİR",
            data=error_csv,
            file_name=(
                "binance_error_pairs.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "🚀 BUY / SELL DEEP LEARNING SCAN "
        "butonuna basarak taramayı başlat."
    )
