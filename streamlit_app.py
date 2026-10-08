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
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Binance All Pairs Deep Learning Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

BINANCE_URL = "https://api.binance.com"

INTERVAL = "15m"

# 10 gün = 960 adet 15 dakikalık mum
CRASH_BARS = 960

# 1 hafta = 672 adet 15 dakikalık mum
WEEKLY_BARS = 672

# 30 dakika = 2 adet 15 dakikalık mum
PREDICTION_BARS = 2

# Hedefler
BUY_TARGET = 0.10
SELL_TARGET = -0.10

# Minimum veri
MIN_DATA = 300

# Varsayılan AI eğitim satırı
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
# API HELPER
# ============================================================

def get_json(url, params=None, timeout=30):

    response = session.get(
        url,
        params=params,
        timeout=timeout
    )

    if response.status_code != 200:

        text = response.text[:500]

        raise RuntimeError(
            f"HTTP {response.status_code}: {text}"
        )

    return response.json()


# ============================================================
# BINANCE - ALL SPOT PAIRS
# ============================================================

@st.cache_data(ttl=300)
def get_all_binance_spot_symbols():

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

        # Spot kontrolü
        if item.get("isSpotTradingAllowed") is False:
            continue

        rows.append({
            "symbol": symbol,
            "base": item.get("baseAsset", ""),
            "quote": item.get("quoteAsset", "USDT")
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Binance Spot USDT paritesi bulunamadı."
        )

    return df


# ============================================================
# BINANCE 24H TICKERS
# ============================================================

@st.cache_data(ttl=120)
def get_binance_24h_tickers():

    data = get_json(
        BINANCE_URL + "/api/v3/ticker/24hr"
    )

    rows = []

    for item in data:

        symbol = item.get("symbol", "")

        if not symbol.endswith("USDT"):
            continue

        try:
            quote_volume = float(
                item.get("quoteVolume", 0) or 0
            )
        except Exception:
            quote_volume = 0.0

        try:
            last_price = float(
                item.get("lastPrice", 0) or 0
            )
        except Exception:
            last_price = 0.0

        try:
            price_change_percent = float(
                item.get("priceChangePercent", 0) or 0
            )
        except Exception:
            price_change_percent = 0.0

        try:
            high_price = float(
                item.get("highPrice", 0) or 0
            )
        except Exception:
            high_price = 0.0

        try:
            low_price = float(
                item.get("lowPrice", 0) or 0
            )
        except Exception:
            low_price = 0.0

        try:
            count = int(
                item.get("count", 0) or 0
            )
        except Exception:
            count = 0

        rows.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "last_price": last_price,
            "price_change_24h": price_change_percent,
            "high_24h": high_price,
            "low_24h": low_price,
            "trade_count": count
        })

    return pd.DataFrame(rows)


# ============================================================
# ACTIVITY SCORE
# ============================================================

def calculate_activity_score(df):

    x = df.copy()

    if x.empty:
        return x

    # Log volume
    x["volume_score"] = np.log1p(
        x["quote_volume"].clip(lower=0)
    )

    # İşlem sayısı
    x["trade_score"] = np.log1p(
        x["trade_count"].clip(lower=0)
    )

    # Günlük hareket
    x["movement_score"] = (
        x["price_change_24h"]
        .abs()
        .fillna(0)
    )

    # 24h high-low hareketi
    x["range_score"] = np.where(
        x["low_24h"] > 0,
        (
            (x["high_24h"] - x["low_24h"])
            / x["low_24h"]
        ) * 100,
        0
    )

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                np.ones(len(series)),
                index=series.index
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    x["volume_norm"] = normalize(
        x["volume_score"]
    )

    x["trade_norm"] = normalize(
        x["trade_score"]
    )

    x["movement_norm"] = normalize(
        x["movement_score"]
    )

    x["range_norm"] = normalize(
        x["range_score"]
    )

    # Aktivite skoru
    x["activity_score"] = (
        x["volume_norm"] * 50
        + x["trade_norm"] * 20
        + x["movement_norm"] * 15
        + x["range_norm"] * 15
    )

    return x.sort_values(
        "activity_score",
        ascending=False
    )


# ============================================================
# SELECT ALL BINANCE PAIRS
# ============================================================

@st.cache_data(ttl=120)
def get_all_ranked_pairs():

    symbols = get_all_binance_spot_symbols()

    tickers = get_binance_24h_tickers()

    if tickers.empty:

        symbols["quote_volume"] = 0
        symbols["last_price"] = 0
        symbols["price_change_24h"] = 0
        symbols["high_24h"] = 0
        symbols["low_24h"] = 0
        symbols["trade_count"] = 0

        return symbols

    result = symbols.merge(
        tickers,
        on="symbol",
        how="left"
    )

    numeric_columns = [
        "quote_volume",
        "last_price",
        "price_change_24h",
        "high_24h",
        "low_24h",
        "trade_count"
    ]

    for column in numeric_columns:

        if column not in result.columns:
            result[column] = 0

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce"
            )
            .fillna(0)
        )

    result = calculate_activity_score(
        result
    )

    return result


# ============================================================
# BINANCE CANDLES
# ============================================================

def get_binance_candles(
    symbol,
    limit=1000
):

    limit = min(
        int(limit),
        1000
    )

    data = get_json(
        BINANCE_URL + "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit
        }
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

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    return df


# ============================================================
# EMA
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(close, period):

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

    rs = avg_gain / loss.replace(
        0,
        np.nan
    )

    result = (
        100
        - (100 / (1 + rs))
    )

    return result.fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs()
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
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
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
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # Returns
    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret6"] = close.pct_change(6)
    x["ret12"] = close.pct_change(12)
    x["ret24"] = close.pct_change(24)

    # EMA
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

    # RSI
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

    # MACD
    macd_line = (
        ema(close, 12)
        - ema(close, 26)
    )

    macd_signal = ema(
        macd_line,
        9
    )

    x["macd"] = macd_line

    x["macd_signal"] = macd_signal

    x["macd_hist"] = (
        macd_line
        - macd_signal
    )

    # Bollinger
    middle = close.rolling(20).mean()

    std = close.rolling(20).std()

    upper = (
        middle
        + 2 * std
    )

    lower = (
        middle
        - 2 * std
    )

    width = (
        upper - lower
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

    # ATR
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

    # ADX
    x["adx"] = adx(
        x,
        14
    )

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()

    denominator = (
        high14 - low14
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        * (close - low14)
        / denominator
    )

    d = k.rolling(3).mean()

    x["stoch_k"] = k.fillna(50)
    x["stoch_d"] = d.fillna(50)

    # Candle
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

    # Volume
    volume_mean = volume.rolling(20).mean()

    volume_std = volume.rolling(20).std()

    x["volume_ratio"] = (
        volume
        / volume_mean.replace(
            0,
            np.nan
        )
    )

    x["volume_z"] = (
        volume - volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    # Volatility
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

    # Price Z
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
        close - price_mean
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
# 10 DAY CRASH
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

    # Son tamamlanmış mum
    completed = df.iloc[:-1]

    window = completed.tail(
        CRASH_BARS
    )

    if window.empty:
        return result

    peak = float(
        window["high"].max()
    )

    price = float(
        completed["close"].iloc[-1]
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
# WEEKLY LOW + 3 RISING CANDLES
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

    if len(df) < WEEKLY_BARS + 3:
        return result

    completed = df.iloc[:-1]

    weekly = completed.tail(
        WEEKLY_BARS
    )

    last3 = completed.tail(3)

    weekly_low = float(
        weekly["low"].min()
    )

    recent_low = float(
        last3["low"].min()
    )

    if weekly_low <= 0:
        return result

    distance = (
        recent_low / weekly_low
        - 1
    )

    rising = (
        last3["close"].iloc[0]
        < last3["close"].iloc[1]
        < last3["close"].iloc[2]
    )

    green = (
        last3["close"].iloc[2]
        > last3["open"].iloc[2]
    )

    weekly_low_index = (
        weekly["low"].idxmin()
    )

    first_index = last3.index[0]

    dip_before = (
        weekly_low_index
        < first_index
    )

    near_low = (
        distance <= 0.015
    )

    result["weekly_low"] = weekly_low

    result["three_rising"] = bool(
        rising
    )

    result["green"] = bool(
        green
    )

    result["near_low"] = bool(
        near_low
    )

    result["dip_before"] = bool(
        dip_before
    )

    result["pattern"] = bool(
        rising
        and green
        and near_low
        and dip_before
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
            future_high / entry
            - 1
        )

        down = (
            future_low / entry
            - 1
        )

        buy_hit = (
            up >= BUY_TARGET
        )

        sell_hit = (
            down <= SELL_TARGET
        )

        # BUY
        if buy_hit and not sell_hit:

            target[i] = 1

        # SELL
        elif sell_hit and not buy_hit:

            target[i] = 0

        # İkisi de
        elif buy_hit and sell_hit:

            if up >= abs(down):
                target[i] = 1
            else:
                target[i] = 0

        # HOLD yok
        else:

            if up >= abs(down):
                target[i] = 1
            else:
                target[i] = 0

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

    # EMA 5 / 20
    if row["ema5_20"] > 0:
        buy += 10
    else:
        sell += 10

    # EMA 20 / 50
    if row["ema20_50"] > 0:
        buy += 10
    else:
        sell += 10

    # EMA 50 / 200
    if row["ema50_200"] > 0:
        buy += 8
    else:
        sell += 8

    # RSI
    if row["rsi14"] < 35:
        buy += 10

    elif row["rsi14"] > 65:
        sell += 10

    # MACD
    if row["macd_hist"] > 0:
        buy += 8
    else:
        sell += 8

    # Bollinger
    if row["bb_position"] < 0.25:
        buy += 10

    elif row["bb_position"] > 0.75:
        sell += 10

    # Stochastic
    if row["stoch_k"] < 25:
        buy += 8

    elif row["stoch_k"] > 75:
        sell += 8

    # Candle
    if row["close"] > row["open"]:
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
        features[FEATURES]
        .replace(
            [np.inf, -np.inf],
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

        X = X[-training_rows:]
        y = y[-training_rows:]

    if len(np.unique(y)) < 2:

        raise ValueError(
            "AI yalnızca tek sınıf gördü."
        )

    split = int(
        len(X) * 0.80
    )

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

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
            prediction == y_test
        )
    )

    latest = feature_df.iloc[
        -2:-1
    ]

    probability = (
        model.predict_proba(
            latest.values
        )[0]
    )

    p_buy = 0.0
    p_sell = 0.0

    for cls, prob in zip(
        model.classes_,
        probability
    ):

        if int(cls) == 1:
            p_buy = float(prob)

        else:
            p_sell = float(prob)

    return {
        "p_buy": p_buy,
        "p_sell": p_sell,
        "accuracy": accuracy
    }


# ============================================================
# ANALYZE ONE SYMBOL
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
        technical_score(df)
    )

    # AI
    try:

        ai = ai_predict(
            df,
            training_rows
        )

        p_buy = ai["p_buy"]
        p_sell = ai["p_sell"]
        accuracy = ai["accuracy"]

        buy_score = (
            p_buy * 100
            + tech_buy * 0.35
        )

        sell_score = (
            p_sell * 100
            + tech_sell * 0.35
        )

        ai_status = "OK"

    except Exception as e:

        total = (
            tech_buy
            + tech_sell
        )

        if total > 0:

            p_buy = (
                tech_buy / total
            )

            p_sell = (
                tech_sell / total
            )

        else:

            p_buy = 0.5
            p_sell = 0.5

        buy_score = tech_buy
        sell_score = tech_sell

        accuracy = np.nan

        ai_status = (
            "FALLBACK: "
            + str(e)
        )

    # Crash bonus
    crash_bonus = 0

    if crash["is_crash"]:

        crash_bonus = 20

        buy_score += 20

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    # Weekly pattern bonus
    pattern_bonus = 0

    if weekly["pattern"]:

        pattern_bonus = 20

        buy_score += 20

        p_buy = min(
            1.0,
            p_buy + 0.10
        )

    # BUY / SELL
    if buy_score >= sell_score:

        signal = "BUY"

    else:

        signal = "SELL"

    price = float(
        df.iloc[-2]["close"]
    )

    return {
        "symbol": symbol,
        "signal": signal,
        "price": price,

        "activity_score": np.nan,

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
            crash["peak"],
            10
        ),

        "10d_drop": round(
            crash["drop"],
            2
        ),

        "crash": bool(
            crash["is_crash"]
        ),

        "crash_bonus": crash_bonus,

        "weekly_low": round(
            weekly["weekly_low"],
            10
        ),

        "weekly_pattern": bool(
            weekly["pattern"]
        ),

        "three_rising": bool(
            weekly["three_rising"]
        ),

        "green_last": bool(
            weekly["green"]
        ),

        "ai_accuracy": (
            round(
                accuracy * 100,
                2
            )
            if not pd.isna(accuracy)
            else np.nan
        ),

        "ai_status": ai_status,

        "candles": len(df)
    }


# ============================================================
# SCAN
# ============================================================

def run_scan(
    pairs,
    crash_threshold,
    training_rows,
    delay_seconds
):

    results = []
    errors = []

    total = len(pairs)

    progress = st.progress(0)

    status = st.empty()

    start = time.time()

    for i, item in enumerate(
        pairs,
        start=1
    ):

        symbol = item["symbol"]

        status.write(
            f"{i:,} / {total:,}  "
            f"{symbol}"
        )

        try:

            df = get_binance_candles(
                symbol,
                CRASH_BARS
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
                4
            )

            result[
                "24h_volume"
            ] = float(
                item.get(
                    "quote_volume",
                    0
                )
            )

            result[
                "24h_change"
            ] = float(
                item.get(
                    "price_change_24h",
                    0
                )
            )

            result[
                "trade_count"
            ] = int(
                item.get(
                    "trade_count",
                    0
                )
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

        if delay_seconds > 0:

            time.sleep(
                delay_seconds
            )

    progress.empty()
    status.empty()

    elapsed = (
        time.time()
        - start
    )

    return (
        pd.DataFrame(results),
        pd.DataFrame(errors),
        elapsed
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Binance ALL Spot Pairs "
    "Deep Learning Scanner"
)

st.caption(
    "Tüm Binance Spot USDT pariteleri • "
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

    delay_seconds = st.number_input(
        "API Delay",
        min_value=0.0,
        max_value=2.0,
        value=0.08,
        step=0.01
    )

    st.divider()

    st.write(
        "📊 Binance Spot"
    )

    st.write(
        "🌐 Tüm USDT pariteleri"
    )

    st.write(
        "⏱️ Timeframe: 15M"
    )

    st.write(
        "🔮 Prediction: 30 dakika"
    )

    st.write(
        "🟢 BUY: +10%"
    )

    st.write(
        "🔴 SELL: -10%"
    )

    st.write(
        "🚫 HOLD yok"
    )

    st.write(
        "🔥 Crash bonus: +20"
    )

    st.write(
        "🔥 Weekly pattern: +20"
    )


# ============================================================
# LOAD ALL PAIRS
# ============================================================

st.subheader(
    "🌐 Binance Tüm Spot Pariteleri"
)

try:

    pairs_df = get_all_ranked_pairs()

except Exception as e:

    st.error(
        f"Binance pariteleri alınamadı: {e}"
    )

    st.info(
        "Eğer HTTP 451 görüyorsanız, "
        "sunucunun bulunduğu bölgeden Binance API erişimi "
        "engelleniyor olabilir."
    )

    st.stop()


# ============================================================
# PAIR STATISTICS
# ============================================================

total_pairs = len(
    pairs_df
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Tüm USDT Spot",
    f"{total_pairs:,}"
)

if not pairs_df.empty:

    active_top = pairs_df.head(
        min(100, len(pairs_df))
    )

    active_volume = (
        active_top[
            "quote_volume"
        ].sum()
    )

else:

    active_volume = 0


col2.metric(
    "Top 100 Hacim",
    f"${active_volume:,.0f}"
)

col3.metric(
    "En Aktif",
    (
        pairs_df.iloc[0]["symbol"]
        if not pairs_df.empty
        else "-"
    )
)

col4.metric(
    "Son Güncelleme",
    datetime.now(
        timezone.utc
    ).strftime(
        "%H:%M:%S UTC"
    )
)


# ============================================================
# ALL PAIRS TABLE
# ============================================================

with st.expander(
    "📋 Binance Tüm Pariteleri Gör"
):

    st.dataframe(
        pairs_df[
            [
                "symbol",
                "base",
                "quote",
                "quote_volume",
                "price_change_24h",
                "trade_count",
                "activity_score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SCAN MODE
# ============================================================

st.subheader(
    "🔎 Tarama"
)

scan_mode = st.radio(
    "Tarama kapsamı",
    [
        "ALL PAIRS",
        "TOP ACTIVE",
        "MANUAL"
    ],
    horizontal=True
)


# ============================================================
# ALL PAIRS
# ============================================================

if scan_mode == "ALL PAIRS":

    selected_pairs = pairs_df.copy()

    st.info(
        f"TÜM Binance USDT Spot pariteleri "
        f"taranacak: {len(selected_pairs):,}"
    )


# ============================================================
# TOP ACTIVE
# ============================================================

elif scan_mode == "TOP ACTIVE":

    active_count = st.number_input(
        "Aktif parite sayısı",
        min_value=10,
        max_value=max(
            10,
            len(pairs_df)
        ),
        value=min(
            1000,
            len(pairs_df)
        ),
        step=100
    )

    selected_pairs = (
        pairs_df
        .head(
            int(active_count)
        )
        .copy()
    )

    st.info(
        f"Aktivite skoruna göre "
        f"en aktif {len(selected_pairs):,} "
        f"parite taranacak."
    )


# ============================================================
# MANUAL
# ============================================================

else:

    selected_symbols = st.multiselect(
        "Pariteleri seç",
        pairs_df[
            "symbol"
        ].tolist(),
        default=pairs_df[
            "symbol"
        ].tolist()[:20]
    )

    selected_pairs = (
        pairs_df[
            pairs_df["symbol"].isin(
                selected_symbols
            )
        ]
        .copy()
    )


# ============================================================
# SELECTED
# ============================================================

st.write(
    f"**Seçilen parite:** "
    f"{len(selected_pairs):,}"
)


# ============================================================
# START SCAN
# ============================================================

start_scan = st.button(
    "🚀 DEEP LEARNING SCAN BAŞLAT",
    type="primary",
    use_container_width=True
)


# ============================================================
# SCAN EXECUTION
# ============================================================

if start_scan:

    if selected_pairs.empty:

        st.error(
            "Tarama için parite yok."
        )

        st.stop()

    scan_time = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    with st.spinner(
        "Binance pariteleri "
        "deep learning ile analiz ediliyor..."
    ):

        results, errors, elapsed = run_scan(
            selected_pairs.to_dict(
                "records"
            ),
            float(
                crash_threshold
            ),
            int(
                training_rows
            ),
            float(
                delay_seconds
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
                results["signal"]
                == "BUY"
            ).sum()
        )

        sell_count = int(
            (
                results["signal"]
                == "SELL"
            ).sum()
        )

        crash_count = int(
            results["crash"]
        .sum()
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Başarılı",
        f"{len(results):,}"
    )

    c2.metric(
        "Hatalı",
        f"{len(errors):,}"
    )

    c3.metric(
        "BUY",
        f"{buy_count:,}"
    )

    c4.metric(
        "SELL",
        f"{sell_count:,}"
    )

    c5.metric(
        "CRASH",
        f"{crash_count:,}"
    )

    st.caption(
        f"Tarama zamanı: {scan_time} | "
        f"Süre: {elapsed:.1f} saniye"
    )


    # ========================================================
    # HYPERACTIVE
    # ========================================================

    st.subheader(
        "⚡ HİPERAKTİF PARİTELER"
    )

    if not results.empty:

        active_results = (
            results
            .sort_values(
                [
                    "activity_score",
                    "24h_volume"
                ],
                ascending=False
            )
            .head(100)
        )

        st.dataframe(
            active_results[
                [
                    "symbol",
                    "signal",
                    "activity_score",
                    "24h_volume",
                    "trade_count",
                    "24h_change",
                    "price",
                    "p_buy",
                    "p_sell",
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

        buy_df = (
            results[
                results["signal"]
                == "BUY"
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
                "BUY paritesi bulunamadı."
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
    # TOP SELL
    # ========================================================

    st.subheader(
        "🔴 TOP SELL"
    )

    if not results.empty:

        sell_df = (
            results[
                results["signal"]
                == "SELL"
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
                "SELL paritesi bulunamadı."
            )

        else:

            st.dataframe(
                sell_df[
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
                results["10d_drop"]
                >= crash_threshold
            ]
            .sort_values(
                [
                    "10d_drop",
                    "activity_score",
                    "buy_score"
                ],
                ascending=False
            )
        )

        if crash_df.empty:

            st.info(
                "Bu crash filtresine uyan "
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
                        "activity_score",
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
    # WEEKLY PATTERN
    # ========================================================

    st.subheader(
        "🔥 HAFTALIK DİP + 3 YÜKSELEN MUM"
    )

    if not results.empty:

        pattern_df = (
            results[
                results[
                    "weekly_pattern"
                ]
                == True
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

        if pattern_df.empty:

            st.info(
                "Bu pattern bulunamadı."
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
                        "activity_score",
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
        "📊 TÜM BAŞARILI SONUÇLAR"
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
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ TÜM SONUÇLARI CSV İNDİR",
            data=csv,
            file_name=(
                "binance_all_pairs_ai_scan.csv"
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

        st.error(
            f"{len(errors):,} paritede hata oluştu."
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
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ HATALI PARİTELERİ CSV İNDİR",
            data=error_csv,
            file_name=(
                "binance_error_pairs.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "🚀 DEEP LEARNING SCAN BAŞLAT "
        "butonuna basarak taramayı başlat."
    )
