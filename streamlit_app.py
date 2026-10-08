import os
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Spot AI Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
).rstrip("/")

INTERVAL = "15m"

KLINE_LIMIT = 1000

DEFAULT_CANDLE_TARGET = 500_000

DEFAULT_SYMBOL_COUNT = 10

FUTURE_BARS = 12

BUY_SELL_THRESHOLD = 0.003

REQUEST_TIMEOUT = 30

MAX_TRAINING_SAMPLES = 3000

RANDOM_STATE = 42


# ============================================================
# SESSION STATE
# ============================================================

if "candle_cache" not in st.session_state:
    st.session_state.candle_cache = {}

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "last_error" not in st.session_state:
    st.session_state.last_error = ""

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0


# ============================================================
# HTTP SESSION
# ============================================================

http = requests.Session()

http.headers.update({
    "User-Agent": "Spot-AI-Scanner/1.0"
})


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):

            return default

        return value

    except Exception:

        return default


def clamp(value, low, high):

    return max(
        low,
        min(
            high,
            value
        )
    )


def utc_now():

    return datetime.now(
        timezone.utc
    )


# ============================================================
# API
# ============================================================

def api_get(
    endpoint,
    params=None
):

    url = (
        BASE_URL
        + endpoint
    )

    response = http.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 451:

        raise RuntimeError(
            "HTTP 451: Binance API bu çalışma "
            "ortamından erişime izin vermiyor. "
            "Bu bir Python veya Streamlit kod hatası değildir."
        )

    if response.status_code == 429:

        raise RuntimeError(
            "HTTP 429: Binance API rate limit. "
            "İstek sayısı azaltılmalı."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXCHANGE INFORMATION
# ============================================================

@st.cache_data(ttl=900)
def get_exchange_info():

    data = api_get(
        "/api/v3/exchangeInfo"
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

        if item.get(
            "isSpotTradingAllowed"
        ) is False:

            continue

        quote = item.get(
            "quoteAsset",
            ""
        )

        if quote not in (
            "USDT",
            "USDC",
            "FDUSD"
        ):

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
# KLINE API
# ============================================================

def get_klines(
    symbol,
    limit=1000,
    start_time=None,
    end_time=None
):

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": min(
            int(limit),
            KLINE_LIMIT
        )
    }

    if start_time is not None:

        params[
            "startTime"
        ] = int(
            start_time
        )

    if end_time is not None:

        params[
            "endTime"
        ] = int(
            end_time
        )

    return api_get(
        "/api/v3/klines",
        params
    )


# ============================================================
# KLINE DATAFRAME
# ============================================================

def convert_klines(data):

    if not data:

        return pd.DataFrame()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce"
    )

    df["close_time"] = pd.to_numeric(
        df["close_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "open_time"
        ]
    )

    df = df.sort_values(
        "open_time"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# DOWNLOAD 500K CANDLES
# ============================================================

def download_history(
    symbol,
    target,
    progress_bar=None
):

    parts = []

    remaining = int(
        target
    )

    end_time = None

    downloaded = 0

    while remaining > 0:

        limit = min(
            KLINE_LIMIT,
            remaining
        )

        data = get_klines(
            symbol=symbol,
            limit=limit,
            end_time=end_time
        )

        if not data:

            break

        batch = convert_klines(
            data
        )

        if batch.empty:

            break

        parts.append(
            batch
        )

        received = len(
            batch
        )

        downloaded += received

        remaining -= received

        oldest = int(
            batch[
                "open_time"
            ].min()
        )

        end_time = (
            oldest - 1
        )

        if progress_bar:

            progress_bar.progress(
                min(
                    downloaded / target,
                    1.0
                )
            )

        if received < limit:

            break

        time.sleep(
            0.05
        )

    if not parts:

        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=[
            "open_time"
        ]
    )

    result = result.sort_values(
        "open_time"
    )

    result = result.tail(
        target
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# UPDATE EXISTING CANDLES
# ============================================================

def update_history(
    symbol,
    existing,
    target
):

    if (
        existing is None
        or existing.empty
    ):

        return download_history(
            symbol,
            target
        )

    last_open_time = int(
        existing[
            "open_time"
        ].max()
    )

    data = get_klines(
        symbol=symbol,
        limit=1000,
        start_time=last_open_time + 1
    )

    if not data:

        return existing

    new_data = convert_klines(
        data
    )

    if new_data.empty:

        return existing

    combined = pd.concat(
        [
            existing,
            new_data
        ],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=[
            "open_time"
        ]
    )

    combined = combined.sort_values(
        "open_time"
    )

    combined = combined.tail(
        target
    )

    return combined.reset_index(
        drop=True
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

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


def atr(
    df,
    period=14
):

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
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
        adjust=False
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

    line = (
        fast - slow
    )

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


def adx(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    previous_high = high.shift(1)

    previous_low = low.shift(1)

    up_move = (
        high
        - previous_high
    )

    down_move = (
        previous_low
        - low
    )

    plus_dm = up_move.where(
        (
            up_move > down_move
        )
        & (
            up_move > 0
        ),
        0
    )

    minus_dm = down_move.where(
        (
            down_move > up_move
        )
        & (
            down_move > 0
        ),
        0
    )

    previous_close = (
        df["close"].shift(1)
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
            ).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    atr_value = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / (
            atr_value
            + 1e-12
        )
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False
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

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURES = [
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "ret24",
    "ret48",

    "ema5_dist",
    "ema20_dist",
    "ema50_dist",
    "ema100_dist",
    "ema200_dist",
    "ema500_dist",
    "ema800_dist",

    "ema20_50",
    "ema50_200",

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

    "volume_ratio",

    "body_pct",
    "range_pct",

    "volatility24",
    "volatility48",

    "distance_high",
    "distance_low",

    "drawdown",

    "price_z"
]


def build_features(
    df
):

    df = df.copy()

    close = df["close"]

    high = df["high"]

    low = df["low"]

    open_price = df["open"]

    volume = df["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["ret1"] = (
        close.pct_change(1)
    )

    df["ret3"] = (
        close.pct_change(3)
    )

    df["ret6"] = (
        close.pct_change(6)
    )

    df["ret12"] = (
        close.pct_change(12)
    )

    df["ret24"] = (
        close.pct_change(24)
    )

    df["ret48"] = (
        close.pct_change(48)
    )

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

    e100 = ema(
        close,
        100
    )

    e200 = ema(
        close,
        200
    )

    e500 = ema(
        close,
        500
    )

    e800 = ema(
        close,
        800
    )

    df["ema5_dist"] = (
        close
        / (
            e5 + 1e-12
        )
    ) - 1

    df["ema20_dist"] = (
        close
        / (
            e20 + 1e-12
        )
    ) - 1

    df["ema50_dist"] = (
        close
        / (
            e50 + 1e-12
        )
    ) - 1

    df["ema100_dist"] = (
        close
        / (
            e100 + 1e-12
        )
    ) - 1

    df["ema200_dist"] = (
        close
        / (
            e200 + 1e-12
        )
    ) - 1

    df["ema500_dist"] = (
        close
        / (
            e500 + 1e-12
        )
    ) - 1

    df["ema800_dist"] = (
        close
        / (
            e800 + 1e-12
        )
    ) - 1

    df["ema20_50"] = (
        e20
        / (
            e50 + 1e-12
        )
    ) - 1

    df["ema50_200"] = (
        e50
        / (
            e200 + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi7"] = rsi(
        close,
        7
    )

    df["rsi14"] = rsi(
        close,
        14
    )

    df["rsi21"] = rsi(
        close,
        21
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_line, macd_signal, macd_hist = macd(
        close
    )

    df["macd"] = (
        macd_line
        / (
            close + 1e-12
        )
    )

    df["macd_signal"] = (
        macd_signal
        / (
            close + 1e-12
        )
    )

    df["macd_hist"] = (
        macd_hist
        / (
            close + 1e-12
        )
    )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb_middle = close.rolling(
        20
    ).mean()

    bb_std = close.rolling(
        20
    ).std()

    bb_upper = (
        bb_middle
        + 2 * bb_std
    )

    bb_lower = (
        bb_middle
        - 2 * bb_std
    )

    df["bb_position"] = (
        close
        - bb_lower
    ) / (
        bb_upper
        - bb_lower
        + 1e-12
    )

    df["bb_width"] = (
        bb_upper
        - bb_lower
    ) / (
        bb_middle.abs()
        + 1e-12
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_value = atr(
        df
    )

    df["atr_pct"] = (
        atr_value
        / (
            close + 1e-12
        )
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    df["adx"] = adx(
        df
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    average_volume = (
        volume
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        volume
        / (
            average_volume
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    df["body_pct"] = (
        close
        - open_price
    ) / (
        close
        + 1e-12
    )

    df["range_pct"] = (
        high
        - low
    ) / (
        close
        + 1e-12
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["volatility24"] = (
        df["ret1"]
        .rolling(24)
        .std()
    )

    df["volatility48"] = (
        df["ret1"]
        .rolling(48)
        .std()
    )

    # --------------------------------------------------------
    # HIGH / LOW
    # --------------------------------------------------------

    rolling_high = (
        high
        .rolling(100)
        .max()
    )

    rolling_low = (
        low
        .rolling(100)
        .min()
    )

    df["distance_high"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    df["distance_low"] = (
        close
        / (
            rolling_low
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    df["drawdown"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # Z SCORE
    # --------------------------------------------------------

    rolling_mean = (
        close
        .rolling(100)
        .mean()
    )

    rolling_std = (
        close
        .rolling(100)
        .std()
    )

    df["price_z"] = (
        close
        - rolling_mean
    ) / (
        rolling_std
        + 1e-12
    )

    return df


# ============================================================
# TARGET CREATION
# ============================================================

def create_targets(
    df
):

    df = df.copy()

    future_return = (
        df["close"]
        .shift(
            -FUTURE_BARS
        )
        / df["close"]
    ) - 1

    target = np.ones(
        len(df),
        dtype=float
    )

    target[
        future_return
        > BUY_SELL_THRESHOLD
    ] = 2

    target[
        future_return
        < -BUY_SELL_THRESHOLD
    ] = 0

    target[
        future_return.isna()
    ] = np.nan

    df["future_return"] = (
        future_return
    )

    df["target"] = (
        target
    )

    return df


# ============================================================
# AI TRAINING
# ============================================================

def train_ai(
    data
):

    if len(data) < 500:

        raise RuntimeError(
            "AI eğitimi için en az "
            "500 temiz satır gerekli."
        )

    train_data = data.tail(
        MAX_TRAINING_SAMPLES
    ).copy()

    x = train_data[
        FEATURES
    ].values

    y = train_data[
        "target"
    ].astype(
        int
    ).values

    returns = train_data[
        "future_return"
    ].values

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(
        x
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(
            128,
            64
        ),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=25,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE
    )

    classifier.fit(
        x_scaled,
        y
    )

    regression = Ridge(
        alpha=1.0
    )

    regression.fit(
        x_scaled,
        returns
    )

    return (
        classifier,
        regression,
        scaler
    )


# ============================================================
# AI PREDICTION
# ============================================================

def predict_ai(
    classifier,
    regression,
    scaler,
    data
):

    latest = data[
        FEATURES
    ].iloc[
        -1:
    ].values

    latest_scaled = scaler.transform(
        latest
    )

    probabilities = (
        classifier.predict_proba(
            latest_scaled
        )[0]
    )

    classes = (
        classifier.classes_
    )

    probability_map = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }

    for cls, probability in zip(
        classes,
        probabilities
    ):

        probability_map[
            int(cls)
        ] = float(
            probability
        )

    sell_probability = (
        probability_map[0]
    )

    hold_probability = (
        probability_map[1]
    )

    buy_probability = (
        probability_map[2]
    )

    probabilities_list = [
        sell_probability,
        hold_probability,
        buy_probability
    ]

    maximum = max(
        probabilities_list
    )

    if (
        buy_probability
        == maximum
    ):

        signal = "BUY"

    elif (
        sell_probability
        == maximum
    ):

        signal = "SELL"

    else:

        signal = "HOLD"

    expected_return = float(
        regression.predict(
            latest_scaled
        )[0]
    )

    return {
        "signal": signal,
        "sell": sell_probability,
        "hold": hold_probability,
        "buy": buy_probability,
        "confidence": maximum,
        "expected": expected_return
    }


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    candles
):

    features = build_features(
        candles
    )

    features = create_targets(
        features
    )

    clean = features[
        FEATURES
        + [
            "target",
            "future_return"
        ]
    ].replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    ).dropna()

    if len(clean) < 500:

        raise RuntimeError(
            f"{symbol}: AI için yeterli "
            "temiz veri bulunamadı."
        )

    classifier, regression, scaler = train_ai(
        clean
    )

    prediction = predict_ai(
        classifier,
        regression,
        scaler,
        clean
    )

    latest = (
        features
        .dropna(
            subset=FEATURES
        )
        .iloc[-1]
    )

    price = safe_float(
        latest["close"]
    )

    rsi_value = safe_float(
        latest["rsi14"],
        50
    )

    adx_value = safe_float(
        latest["adx"]
    )

    volume_ratio = safe_float(
        latest["volume_ratio"],
        1
    )

    volatility = safe_float(
        latest["volatility24"]
    )

    momentum = safe_float(
        latest["ret24"]
    )

    trend = safe_float(
        latest["ema50_200"]
    )

    atr_pct = safe_float(
        latest["atr_pct"]
    )

    trend_score = clamp(
        50
        + trend * 1000,
        0,
        100
    )

    momentum_score = clamp(
        50
        + momentum * 500,
        0,
        100
    )

    liquidity_score = clamp(
        volume_ratio * 40,
        0,
        100
    )

    risk_score = clamp(
        50
        + volatility * 500
        + atr_pct * 300,
        0,
        100
    )

    direction_score = (
        prediction["buy"]
        - prediction["sell"]
    ) * 100

    expected_score = (
        prediction["expected"]
        * 10000
    )

    ai_score = (
        direction_score * 0.35
        + prediction["confidence"]
        * 100
        * 0.20
        + trend_score * 0.15
        + momentum_score * 0.10
        + liquidity_score * 0.05
        + expected_score * 0.15
        - risk_score * 0.05
    )

    return {
        "symbol": symbol,

        "signal":
            prediction["signal"],

        "ai_score":
            float(ai_score),

        "confidence_pct":
            prediction["confidence"] * 100,

        "buy_probability_pct":
            prediction["buy"] * 100,

        "sell_probability_pct":
            prediction["sell"] * 100,

        "hold_probability_pct":
            prediction["hold"] * 100,

        "expected_return_pct":
            prediction["expected"] * 100,

        "price":
            price,

        "rsi":
            rsi_value,

        "adx":
            adx_value,

        "volume_ratio":
            volume_ratio,

        "volatility_pct":
            volatility * 100,

        "atr_pct":
            atr_pct * 100,

        "momentum_pct":
            momentum * 100,

        "trend_pct":
            trend * 100,

        "risk_score":
            risk_score,

        "liquidity_score":
            liquidity_score,

        "candle_count":
            len(candles)
    }


# ============================================================
# SELECT SYMBOLS
# ============================================================

def select_symbols(
    symbols,
    count
):

    usdt = [
        symbol
        for symbol in symbols
        if symbol.endswith(
            "USDT"
        )
    ]

    return usdt[
        :int(count)
    ]


# ============================================================
# COMPLETE SCAN
# ============================================================

def run_scan(
    symbols,
    target,
    progress,
    status
):

    results = []

    errors = []

    total = len(
        symbols
    )

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        status.info(
            f"🔎 {symbol} "
            f"analiz ediliyor "
            f"({index}/{total})"
        )

        progress.progress(
            (index - 1)
            / max(
                total,
                1
            )
        )

        try:

            existing = (
                st.session_state
                .candle_cache
                .get(symbol)
            )

            if (
                existing is None
                or len(existing)
                < target
            ):

                status.info(
                    f"📥 {symbol}: "
                    f"{target:,} mum hazırlanıyor..."
                )

                candles = download_history(
                    symbol,
                    target
                )

            else:

                status.info(
                    f"🔄 {symbol}: "
                    "yeni 15m mumlar ekleniyor..."
                )

                candles = update_history(
                    symbol,
                    existing,
                    target
                )

            if candles.empty:

                raise RuntimeError(
                    "Candle verisi alınamadı."
                )

            st.session_state.candle_cache[
                symbol
            ] = candles

            result = analyze_symbol(
                symbol,
                candles
            )

            results.append(
                result
            )

        except Exception as exc:

            errors.append(
                f"{symbol}: {exc}"
            )

    progress.progress(
        1.0
    )

    if errors:

        st.session_state.last_error = (
            "\n".join(
                errors
            )
        )

    else:

        st.session_state.last_error = ""

    return pd.DataFrame(
        results
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Scanner Settings"
)

st.sidebar.write(
    "PyTorch kullanılmıyor."
)

st.sidebar.write(
    "AI: scikit-learn MLP"
)

st.sidebar.write(
    f"Interval: {INTERVAL}"
)

st.sidebar.write(
    "Orders: DISABLED"
)

candle_target = st.sidebar.number_input(
    "Hedef candle",
    min_value=10_000,
    max_value=500_000,
    value=500_000,
    step=10_000
)

symbol_count = st.sidebar.number_input(
    "Parite sayısı",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

auto_refresh = st.sidebar.checkbox(
    "15 dakikada otomatik yenile",
    value=True
)

st.sidebar.divider()

st.sidebar.write(
    f"API: {BASE_URL}"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Binance Spot AI Deep Learning Scanner"
)

st.markdown(
    "15m Spot market • AI classification • "
    "500K candle target • Top 10 BUY/SELL"
)

st.info(
    "Bu sürüm PyTorch kullanmaz. "
    "Streamlit Cloud üzerinde doğrudan "
    "scikit-learn ile çalışır."
)


# ============================================================
# SCANNER
# ============================================================

def scanner():

    st.subheader(
        "🔎 Market Scanner"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "Son tarama",
            st.session_state.last_scan
            or "-"
        )

    with col2:

        st.metric(
            "Cache parite",
            len(
                st.session_state.candle_cache
            )
        )

    with col3:

        st.metric(
            "Sonuç",
            len(
                st.session_state.results
            )
        )

    with col4:

        st.metric(
            "Model",
            "MLP AI"
        )

    st.divider()

    scan_button = st.button(
        "🚀 ŞİMDİ TARA",
        type="primary",
        use_container_width=True
    )

    first_scan = (
        st.session_state.results.empty
    )

    if scan_button or first_scan:

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            status.info(
                "Binance Spot sembolleri "
                "alınıyor..."
            )

            symbols = get_exchange_info()

            selected = select_symbols(
                symbols,
                symbol_count
            )

            if not selected:

                raise RuntimeError(
                    "USDT Spot paritesi bulunamadı."
                )

            st.info(
                "Taranacak pariteler: "
                + ", ".join(selected)
            )

            results = run_scan(
                selected,
                int(candle_target),
                progress,
                status
            )

            st.session_state.results = (
                results
            )

            st.session_state.last_scan = (
                utc_now().strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )

            st.session_state.scan_count += 1

            status.success(
                "Tarama tamamlandı."
            )

        except Exception as exc:

            st.session_state.last_error = (
                traceback.format_exc()
            )

            st.error(
                f"Tarama hatası: {exc}"
            )

            if "451" in str(exc):

                st.warning(
                    "HTTP 451: Binance API bu "
                    "ortamdan erişime izin vermiyor."
                )

    df = st.session_state.results

    if df.empty:

        st.warning(
            "Henüz sonuç yok."
        )

        if st.session_state.last_error:

            with st.expander(
                "Hata detayları"
            ):

                st.code(
                    st.session_state.last_error
                )

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    buy = df[
        df["signal"] == "BUY"
    ]

    sell = df[
        df["signal"] == "SELL"
    ]

    hold = df[
        df["signal"] == "HOLD"
    ]

    a, b, c, d = st.columns(
        4
    )

    a.metric(
        "🟢 BUY",
        len(buy)
    )

    b.metric(
        "🔴 SELL",
        len(sell)
    )

    c.metric(
        "⚪ HOLD",
        len(hold)
    )

    d.metric(
        "Tarama",
        st.session_state.scan_count
    )

    # ========================================================
    # BUY
    # ========================================================

    st.header(
        "🟢 TOP 10 BUY"
    )

    if not buy.empty:

        buy_display = buy.sort_values(
            [
                "ai_score",
                "confidence_pct"
            ],
            ascending=False
        ).head(
            10
        )

        st.dataframe(
            buy_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "BUY sinyali bulunamadı."
        )

    # ========================================================
    # SELL
    # ========================================================

    st.header(
        "🔴 TOP 10 SELL"
    )

    if not sell.empty:

        sell_display = sell.sort_values(
            [
                "ai_score",
                "confidence_pct"
            ],
            ascending=False
        ).head(
            10
        )

        st.dataframe(
            sell_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "SELL sinyali bulunamadı."
        )

    # ========================================================
    # ALL RESULTS
    # ========================================================

    st.header(
        "📊 Tüm Sonuçlar"
    )

    all_results = df.sort_values(
        "ai_score",
        ascending=False
    )

    st.dataframe(
        all_results,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CSV
    # ========================================================

    csv = all_results.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ CSV indir",
        data=csv,
        file_name="spot_ai_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================
    # DETAIL
    # ========================================================

    st.header(
        "🔍 Coin Detayı"
    )

    selected_symbol = st.selectbox(
        "Parite seç",
        sorted(
            df["symbol"].unique()
        )
    )

    selected = df[
        df["symbol"]
        == selected_symbol
    ]

    if not selected.empty:

        row = selected.iloc[0]

        x1, x2, x3, x4 = st.columns(
            4
        )

        x1.metric(
            "Signal",
            row["signal"]
        )

        x2.metric(
            "AI Score",
            f"{row['ai_score']:.2f}"
        )

        x3.metric(
            "Confidence",
            f"{row['confidence_pct']:.2f}%"
        )

        x4.metric(
            "Expected Return",
            f"{row['expected_return_pct']:.3f}%"
        )

        detail_columns = [
            "symbol",
            "price",
            "signal",
            "ai_score",
            "confidence_pct",
            "buy_probability_pct",
            "sell_probability_pct",
            "hold_probability_pct",
            "expected_return_pct",
            "rsi",
            "adx",
            "volume_ratio",
            "volatility_pct",
            "atr_pct",
            "momentum_pct",
            "trend_pct",
            "risk_score",
            "liquidity_score",
            "candle_count"
        ]

        detail = pd.DataFrame(
            [
                {
                    column: row[column]
                    for column in detail_columns
                    if column in row.index
                }
            ]
        )

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CACHE
    # ========================================================

    st.header(
        "🗄️ Candle Cache"
    )

    cache_rows = []

    for symbol, candles in (
        st.session_state
        .candle_cache
        .items()
    ):

        cache_rows.append(
            {
                "symbol": symbol,
                "candles": len(candles)
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

    # ========================================================
    # ERRORS
    # ========================================================

    if st.session_state.last_error:

        with st.expander(
            "⚠️ Son hatalar"
        ):

            st.code(
                st.session_state.last_error
            )

    # ========================================================
    # WARNING
    # ========================================================

    st.warning(
        "Bu uygulama finansal tavsiye değildir. "
        "AI tahminleri kesin fiyat tahmini değildir. "
        "Otomatik emir gönderilmez."
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    if hasattr(
        st,
        "fragment"
    ):

        @st.fragment(
            run_every="15m"
        )
        def scheduled_scanner():

            scanner()

        scheduled_scanner()

    else:

        scanner()

        st.info(
            "Otomatik 15 dakika yenileme için "
            "Streamlit 1.37 veya daha yeni sürüm gerekir."
        )

else:

    scanner()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Scanner | 15m | 500K target | "
    "MLP AI | No Order Execution"
)
