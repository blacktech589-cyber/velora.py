import os
import time
import math
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PYTORCH SAFE IMPORT
# ============================================================

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = None

torch = None
nn = None
DataLoader = None
TensorDataset = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

except Exception as exc:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Spot AI Deep Learning Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
).rstrip("/")

INTERVAL = "15m"

KLINE_LIMIT = 1000

DEFAULT_CANDLE_TARGET = 500_000

DEFAULT_SYMBOL_LIMIT = 10

SEQ_LEN = 128

FUTURE_HORIZON = 12

CLASS_THRESHOLD = 0.003

REQUEST_TIMEOUT = 30

TRAIN_EPOCHS = 3

BATCH_SIZE = 128

LEARNING_RATE = 1e-4

DEVICE = (
    "cuda"
    if TORCH_AVAILABLE and torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SESSION STATE
# ============================================================

if "candle_cache" not in st.session_state:
    st.session_state.candle_cache = {}

if "last_results" not in st.session_state:
    st.session_state.last_results = pd.DataFrame()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

if "last_error" not in st.session_state:
    st.session_state.last_error = ""


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def safe_float(value, default=0.0):
    try:
        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "Spot-AI-Deep-Learning-Scanner/1.0"
})


# ============================================================
# API
# ============================================================

def api_get(path, params=None):

    url = DEFAULT_BASE_URL + path

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 451:

        raise RuntimeError(
            "HTTP 451: Binance API bu çalışma "
            "ortamından erişimi kısıtlıyor. "
            "Bu bir PyTorch veya Streamlit hatası değildir."
        )

    if response.status_code == 429:

        raise RuntimeError(
            "HTTP 429: Binance API rate limit. "
            "İstek sayısını azaltın."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXCHANGE INFO
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

        if quote not in {
            "USDT",
            "USDC",
            "FDUSD",
            "BTC",
            "ETH",
            "BNB"
        }:
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
# KLINE
# ============================================================

def fetch_klines_page(
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
        ] = int(start_time)

    if end_time is not None:
        params[
            "endTime"
        ] = int(end_time)

    return api_get(
        "/api/v3/klines",
        params=params
    )


def klines_to_dataframe(data):

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

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
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
        subset=["open_time"]
    )

    df = df.sort_values(
        "open_time"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# INITIAL 500K DOWNLOAD
# ============================================================

def fetch_initial_history(
    symbol,
    target_candles,
    progress_callback=None
):

    all_parts = []

    remaining = int(
        target_candles
    )

    end_time = None

    requests_done = 0

    while remaining > 0:

        batch_size = min(
            KLINE_LIMIT,
            remaining
        )

        data = fetch_klines_page(
            symbol=symbol,
            limit=batch_size,
            end_time=end_time
        )

        if not data:
            break

        part = klines_to_dataframe(
            data
        )

        if part.empty:
            break

        all_parts.append(
            part
        )

        received = len(
            part
        )

        remaining -= received

        requests_done += 1

        oldest_time = int(
            part[
                "open_time"
            ].min()
        )

        end_time = (
            oldest_time - 1
        )

        if progress_callback:

            progress_callback(
                min(
                    1.0,
                    (
                        target_candles
                        - max(
                            remaining,
                            0
                        )
                    )
                    / target_candles
                )
            )

        if received < batch_size:
            break

        time.sleep(
            0.05
        )

        max_requests = (
            target_candles
            // KLINE_LIMIT
        ) + 10

        if requests_done > max_requests:
            break

    if not all_parts:
        return pd.DataFrame()

    result = pd.concat(
        all_parts,
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=["open_time"]
    )

    result = result.sort_values(
        "open_time"
    )

    result = result.tail(
        target_candles
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# INCREMENTAL UPDATE
# ============================================================

def fetch_latest_candles(
    symbol,
    existing_df,
    max_candles
):

    if (
        existing_df is None
        or existing_df.empty
    ):

        return fetch_initial_history(
            symbol,
            max_candles
        )

    last_time = int(
        existing_df[
            "open_time"
        ].max()
    )

    start_time = (
        last_time + 1
    )

    data = fetch_klines_page(
        symbol=symbol,
        limit=1000,
        start_time=start_time
    )

    if not data:
        return existing_df

    new_df = klines_to_dataframe(
        data
    )

    if new_df.empty:
        return existing_df

    combined = pd.concat(
        [
            existing_df,
            new_df
        ],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["open_time"]
    )

    combined = combined.sort_values(
        "open_time"
    )

    combined = combined.tail(
        max_candles
    )

    return combined.reset_index(
        drop=True
    )


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def sma(series, period):

    return series.rolling(
        period
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

    rs = avg_gain / (
        avg_loss + 1e-12
    )

    return 100 - (
        100 / (1 + rs)
    )


def true_range(df):

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

    return pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )


def atr(df, period=14):

    return true_range(
        df
    ).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def macd(series):

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


def bollinger(
    series,
    period=20
):

    middle = sma(
        series,
        period
    )

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

    width = (
        upper - lower
    ) / (
        middle.abs()
        + 1e-12
    )

    return (
        middle,
        upper,
        lower,
        width
    )


def stochastic(
    df,
    period=14
):

    lowest = df[
        "low"
    ].rolling(
        period
    ).min()

    highest = df[
        "high"
    ].rolling(
        period
    ).max()

    k = (
        100
        * (
            df["close"]
            - lowest
        )
        / (
            highest
            - lowest
            + 1e-12
        )
    )

    d = k.rolling(
        3
    ).mean()

    return (
        k,
        d
    )


def cci(
    df,
    period=20
):

    typical = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    mean = typical.rolling(
        period
    ).mean()

    deviation = (
        typical
        .rolling(period)
        .apply(
            lambda x:
                np.mean(
                    np.abs(
                        x - x.mean()
                    )
                ),
            raw=True
        )
    )

    return (
        typical - mean
    ) / (
        0.015 * deviation
        + 1e-12
    )


def williams_r(
    df,
    period=14
):

    highest = df[
        "high"
    ].rolling(
        period
    ).max()

    lowest = df[
        "low"
    ].rolling(
        period
    ).min()

    return (
        -100
        * (
            highest
            - df["close"]
        )
        / (
            highest
            - lowest
            + 1e-12
        )
    )


def roc(
    series,
    period=12
):

    return (
        series.pct_change(
            period
        ) * 100
    )


def obv(df):

    direction = np.sign(
        df["close"].diff()
    )

    return (
        direction
        * df["volume"]
    ).fillna(
        0
    ).cumsum()


def mfi(
    df,
    period=14
):

    typical = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    money_flow = (
        typical
        * df["volume"]
    )

    direction = typical.diff()

    positive = money_flow.where(
        direction > 0,
        0
    )

    negative = money_flow.where(
        direction < 0,
        0
    ).abs()

    positive_sum = positive.rolling(
        period
    ).sum()

    negative_sum = negative.rolling(
        period
    ).sum()

    ratio = positive_sum / (
        negative_sum
        + 1e-12
    )

    return (
        100
        - 100 / (
            1 + ratio
        )
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

    tr = true_range(
        df
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
            atr_value + 1e-12
        )
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / (
            atr_value + 1e-12
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
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_48",

    "ema_5_dist",
    "ema_20_dist",
    "ema_50_dist",
    "ema_100_dist",
    "ema_200_dist",
    "ema_500_dist",
    "ema_800_dist",

    "ema_20_50",
    "ema_50_200",

    "rsi_7",
    "rsi_14",
    "rsi_21",

    "macd",
    "macd_signal",
    "macd_hist",

    "bb_position",
    "bb_width",

    "atr_pct",

    "adx",

    "stoch_k",
    "stoch_d",

    "cci",

    "williams_r",

    "roc_12",
    "roc_24",

    "volume_ratio",

    "obv_z",

    "vwap_distance",

    "mfi",

    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",

    "range_pct",

    "volatility_24",
    "volatility_48",

    "distance_high_100",
    "distance_low_100",

    "drawdown_100",

    "price_z_100"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    df = df.copy()

    close = df["close"]

    high = df["high"]

    low = df["low"]

    open_price = df["open"]

    volume = df["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["return_1"] = (
        close.pct_change(1)
    )

    df["return_3"] = (
        close.pct_change(3)
    )

    df["return_6"] = (
        close.pct_change(6)
    )

    df["return_12"] = (
        close.pct_change(12)
    )

    df["return_24"] = (
        close.pct_change(24)
    )

    df["return_48"] = (
        close.pct_change(48)
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema5 = ema(
        close,
        5
    )

    ema20 = ema(
        close,
        20
    )

    ema50 = ema(
        close,
        50
    )

    ema100 = ema(
        close,
        100
    )

    ema200 = ema(
        close,
        200
    )

    ema500 = ema(
        close,
        500
    )

    ema800 = ema(
        close,
        800
    )

    df["ema_5_dist"] = (
        close / (
            ema5 + 1e-12
        )
    ) - 1

    df["ema_20_dist"] = (
        close / (
            ema20 + 1e-12
        )
    ) - 1

    df["ema_50_dist"] = (
        close / (
            ema50 + 1e-12
        )
    ) - 1

    df["ema_100_dist"] = (
        close / (
            ema100 + 1e-12
        )
    ) - 1

    df["ema_200_dist"] = (
        close / (
            ema200 + 1e-12
        )
    ) - 1

    df["ema_500_dist"] = (
        close / (
            ema500 + 1e-12
        )
    ) - 1

    df["ema_800_dist"] = (
        close / (
            ema800 + 1e-12
        )
    ) - 1

    df["ema_20_50"] = (
        ema20 / (
            ema50 + 1e-12
        )
    ) - 1

    df["ema_50_200"] = (
        ema50 / (
            ema200 + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_7"] = rsi(
        close,
        7
    )

    df["rsi_14"] = rsi(
        close,
        14
    )

    df["rsi_21"] = rsi(
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

    (
        bb_middle,
        bb_upper,
        bb_lower,
        bb_width
    ) = bollinger(
        close,
        20
    )

    df["bb_position"] = (
        close
        - bb_lower
    ) / (
        bb_upper
        - bb_lower
        + 1e-12
    )

    df["bb_width"] = bb_width

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_value = atr(
        df,
        14
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
        df,
        14
    )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    stoch_k, stoch_d = stochastic(
        df,
        14
    )

    df["stoch_k"] = stoch_k

    df["stoch_d"] = stoch_d

    # --------------------------------------------------------
    # CCI
    # --------------------------------------------------------

    df["cci"] = cci(
        df,
        20
    )

    # --------------------------------------------------------
    # WILLIAMS R
    # --------------------------------------------------------

    df["williams_r"] = williams_r(
        df,
        14
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    df["roc_12"] = roc(
        close,
        12
    )

    df["roc_24"] = roc(
        close,
        24
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = volume.rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        volume
        / (
            volume_mean
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    obv_value = obv(
        df
    )

    obv_mean = obv_value.rolling(
        100
    ).mean()

    obv_std = obv_value.rolling(
        100
    ).std()

    df["obv_z"] = (
        obv_value
        - obv_mean
    ) / (
        obv_std
        + 1e-12
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        high
        + low
        + close
    ) / 3

    cumulative_volume = (
        volume.cumsum()
    )

    cumulative_pv = (
        typical_price
        * volume
    ).cumsum()

    vwap_value = (
        cumulative_pv
        / (
            cumulative_volume
            + 1e-12
        )
    )

    df["vwap_distance"] = (
        close
        / (
            vwap_value
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # MFI
    # --------------------------------------------------------

    df["mfi"] = mfi(
        df,
        14
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    body = (
        close - open_price
    )

    df["body_pct"] = (
        body
        / (
            close + 1e-12
        )
    )

    df["upper_wick_pct"] = (
        high
        - np.maximum(
            open_price,
            close
        )
    ) / (
        close + 1e-12
    )

    df["lower_wick_pct"] = (
        np.minimum(
            open_price,
            close
        )
        - low
    ) / (
        close + 1e-12
    )

    df["range_pct"] = (
        candle_range
        / (
            close + 1e-12
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["volatility_24"] = (
        df["return_1"]
        .rolling(24)
        .std()
    )

    df["volatility_48"] = (
        df["return_1"]
        .rolling(48)
        .std()
    )

    # --------------------------------------------------------
    # HIGH / LOW
    # --------------------------------------------------------

    rolling_high = high.rolling(
        100
    ).max()

    rolling_low = low.rolling(
        100
    ).min()

    df["distance_high_100"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    df["distance_low_100"] = (
        close
        / (
            rolling_low
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    df["drawdown_100"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # PRICE Z SCORE
    # --------------------------------------------------------

    price_mean = close.rolling(
        100
    ).mean()

    price_std = close.rolling(
        100
    ).std()

    df["price_z_100"] = (
        close
        - price_mean
    ) / (
        price_std
        + 1e-12
    )

    return df


# ============================================================
# TARGET
# ============================================================

def create_targets(
    df,
    horizon=FUTURE_HORIZON,
    threshold=CLASS_THRESHOLD
):

    df = df.copy()

    future_return = (
        df["close"].shift(
            -horizon
        )
        / df["close"]
    ) - 1

    target = np.ones(
        len(df),
        dtype=np.int64
    )

    target[
        future_return > threshold
    ] = 2

    target[
        future_return < -threshold
    ] = 0

    df["future_return"] = (
        future_return
    )

    df["target"] = (
        target.astype(float)
    )

    df.loc[
        future_return.isna(),
        "target"
    ] = np.nan

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_training_data(df):

    columns = (
        FEATURE_COLUMNS
        + [
            "target",
            "future_return"
        ]
    )

    data = df[
        columns
    ].copy()

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    data = data.dropna()

    return data


# ============================================================
# FEATURE SCALER
# ============================================================

class FeatureScaler:

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, x):

        self.mean = np.nanmean(
            x,
            axis=0
        )

        self.std = np.nanstd(
            x,
            axis=0
        )

        self.std[
            self.std < 1e-8
        ] = 1.0

        return self

    def transform(self, x):

        return (
            x - self.mean
        ) / self.std


# ============================================================
# TRANSFORMER
# IMPORTANT:
# ALL MODEL CLASSES ARE ONLY DEFINED IF TORCH EXISTS
# ============================================================

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):

        def __init__(
            self,
            d_model,
            max_len=4096
        ):

            super().__init__()

            position = torch.arange(
                max_len,
                dtype=torch.float32
            ).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(
                    0,
                    d_model,
                    2,
                    dtype=torch.float32
                )
                * (
                    -math.log(10000.0)
                    / d_model
                )
            )

            pe = torch.zeros(
                max_len,
                d_model
            )

            pe[:, 0::2] = torch.sin(
                position * div_term
            )

            pe[:, 1::2] = torch.cos(
                position * div_term
            )

            pe = pe.unsqueeze(0)

            self.register_buffer(
                "pe",
                pe
            )

        def forward(self, x):

            length = x.size(1)

            return (
                x
                + self.pe[
                    :, :length
                ]
            )


    class TransformerScanner(nn.Module):

        def __init__(
            self,
            input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.15
        ):

            super().__init__()

            self.input_projection = nn.Linear(
                input_size,
                d_model
            )

            self.position = PositionalEncoding(
                d_model
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True
                )
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

            self.norm = nn.LayerNorm(
                d_model
            )

            self.classifier = nn.Sequential(
                nn.Linear(
                    d_model,
                    64
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    64,
                    3
                )
            )

            self.regressor = nn.Sequential(
                nn.Linear(
                    d_model,
                    64
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    64,
                    1
                )
            )

        def forward(self, x):

            x = self.input_projection(
                x
            )

            x = self.position(
                x
            )

            x = self.encoder(
                x
            )

            x = self.norm(
                x
            )

            pooled = x.mean(
                dim=1
            )

            classification = (
                self.classifier(
                    pooled
                )
            )

            regression = (
                self.regressor(
                    pooled
                ).squeeze(-1)
            )

            return (
                classification,
                regression
            )

else:

    PositionalEncoding = None
    TransformerScanner = None


# ============================================================
# SEQUENCE PREPARATION
# ============================================================

def prepare_sequences(
    data,
    scaler,
    seq_len=SEQ_LEN,
    max_samples=2500
):

    values = data[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    classes = data[
        "target"
    ].values.astype(
        np.int64
    )

    returns = data[
        "future_return"
    ].values.astype(
        np.float32
    )

    values = scaler.transform(
        values
    ).astype(
        np.float32
    )

    sequences = []

    target_classes = []

    target_returns = []

    start = max(
        seq_len,
        len(values)
        - max_samples
    )

    for i in range(
        start,
        len(values)
    ):

        sequence = values[
            i - seq_len:i
        ]

        if len(sequence) != seq_len:
            continue

        sequences.append(
            sequence
        )

        target_classes.append(
            classes[i]
        )

        target_returns.append(
            returns[i]
        )

    if not sequences:

        return (
            None,
            None,
            None
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            target_classes,
            dtype=np.int64
        ),
        np.asarray(
            target_returns,
            dtype=np.float32
        )
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    data,
    epochs=TRAIN_EPOCHS
):

    if not TORCH_AVAILABLE:

        raise RuntimeError(
            "PyTorch yüklenemedi: "
            + str(
                TORCH_IMPORT_ERROR
            )
        )

    if TransformerScanner is None:

        raise RuntimeError(
            "Transformer modeli "
            "oluşturulamadı."
        )

    if len(data) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            "Model eğitimi için "
            "yeterli veri yok."
        )

    feature_values = data[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    scaler = FeatureScaler()

    scaler.fit(
        feature_values
    )

    (
        x,
        y_class,
        y_return
    ) = prepare_sequences(
        data,
        scaler
    )

    if x is None:

        raise RuntimeError(
            "Sequence oluşturulamadı."
        )

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32
    )

    class_tensor = torch.tensor(
        y_class,
        dtype=torch.long
    )

    return_tensor = torch.tensor(
        y_return,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        x_tensor,
        class_tensor,
        return_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    model = TransformerScanner(
        input_size=len(
            FEATURE_COLUMNS
        )
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    class_loss = nn.CrossEntropyLoss()

    regression_loss = nn.HuberLoss()

    model.train()

    for epoch in range(
        epochs
    ):

        for (
            batch_x,
            batch_class,
            batch_return
        ) in loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_class = batch_class.to(
                DEVICE
            )

            batch_return = batch_return.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits, prediction_return = model(
                batch_x
            )

            loss_class = class_loss(
                logits,
                batch_class
            )

            loss_return = regression_loss(
                prediction_return,
                batch_return
            )

            loss = (
                loss_class
                + 2.0 * loss_return
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

    return (
        model,
        scaler
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_latest(
    model,
    scaler,
    data
):

    if len(data) < SEQ_LEN:

        raise RuntimeError(
            "Tahmin için yeterli "
            "sequence yok."
        )

    values = data[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    values = scaler.transform(
        values
    ).astype(
        np.float32
    )

    latest = values[
        -SEQ_LEN:
    ]

    x = torch.tensor(
        latest,
        dtype=torch.float32
    )

    x = x.unsqueeze(
        0
    ).to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        logits, regression = model(
            x
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        expected_return = (
            regression.item()
        )

    sell_probability = safe_float(
        probabilities[0]
    )

    hold_probability = safe_float(
        probabilities[1]
    )

    buy_probability = safe_float(
        probabilities[2]
    )

    maximum = max(
        sell_probability,
        hold_probability,
        buy_probability
    )

    if buy_probability == maximum:

        signal = "BUY"

    elif sell_probability == maximum:

        signal = "SELL"

    else:

        signal = "HOLD"

    return {
        "signal": signal,
        "sell_probability": sell_probability,
        "hold_probability": hold_probability,
        "buy_probability": buy_probability,
        "confidence": maximum,
        "expected_return": expected_return
    }


# ============================================================
# MARKET METRICS
# ============================================================

def calculate_market_metrics(
    features
):

    valid = features.dropna(
        subset=FEATURE_COLUMNS
    )

    if valid.empty:

        raise RuntimeError(
            "Feature verisi boş."
        )

    row = valid.iloc[-1]

    price = safe_float(
        row["close"]
    )

    rsi_value = safe_float(
        row["rsi_14"],
        50
    )

    adx_value = safe_float(
        row["adx"]
    )

    volatility = safe_float(
        row["volatility_24"]
    )

    momentum = safe_float(
        row["return_24"]
    )

    trend = safe_float(
        row["ema_50_200"]
    )

    volume_ratio = safe_float(
        row["volume_ratio"],
        1
    )

    atr_pct = safe_float(
        row["atr_pct"]
    )

    trend_score = clamp(
        50 + trend * 1000,
        0,
        100
    )

    momentum_score = clamp(
        50 + momentum * 500,
        0,
        100
    )

    if rsi_value < 30:

        rsi_score = 85

    elif rsi_value < 40:

        rsi_score = 70

    elif rsi_value > 70:

        rsi_score = 20

    elif rsi_value > 60:

        rsi_score = 40

    else:

        rsi_score = 55

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

    return {
        "price": price,
        "rsi": rsi_value,
        "adx": adx_value,
        "volatility": volatility,
        "momentum": momentum,
        "trend": trend,
        "volume_ratio": volume_ratio,
        "atr_pct": atr_pct,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "rsi_score": rsi_score,
        "liquidity_score": liquidity_score,
        "risk_score": risk_score
    }


# ============================================================
# AI SCORE
# ============================================================

def calculate_ai_score(
    prediction,
    metrics
):

    buy_probability = (
        prediction[
            "buy_probability"
        ]
    )

    sell_probability = (
        prediction[
            "sell_probability"
        ]
    )

    confidence = (
        prediction[
            "confidence"
        ]
    )

    expected_return = (
        prediction[
            "expected_return"
        ]
    )

    trend_score = (
        metrics[
            "trend_score"
        ]
    )

    momentum_score = (
        metrics[
            "momentum_score"
        ]
    )

    liquidity_score = (
        metrics[
            "liquidity_score"
        ]
    )

    risk_score = (
        metrics[
            "risk_score"
        ]
    )

    direction = (
        buy_probability
        - sell_probability
    ) * 100

    expected_component = (
        expected_return
        * 10000
    )

    score = (
        direction * 0.35
        + confidence * 100 * 0.20
        + trend_score * 0.15
        + momentum_score * 0.10
        + liquidity_score * 0.05
        + expected_component * 0.15
        - risk_score * 0.05
    )

    return float(
        score
    )


# ============================================================
# ANALYZE SYMBOL
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

    clean = clean_training_data(
        features
    )

    if len(clean) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            f"{symbol}: temiz veri "
            "yetersiz."
        )

    model, scaler = train_model(
        clean
    )

    prediction = predict_latest(
        model,
        scaler,
        clean
    )

    metrics = calculate_market_metrics(
        features
    )

    score = calculate_ai_score(
        prediction,
        metrics
    )

    last_time = safe_float(
        candles.iloc[-1][
            "open_time"
        ]
    )

    last_timestamp = (
        datetime.fromtimestamp(
            last_time / 1000,
            tz=timezone.utc
        )
    )

    return {
        "symbol": symbol,

        "signal": prediction[
            "signal"
        ],

        "ai_score": score,

        "confidence_pct":
            prediction[
                "confidence"
            ] * 100,

        "buy_probability_pct":
            prediction[
                "buy_probability"
            ] * 100,

        "sell_probability_pct":
            prediction[
                "sell_probability"
            ] * 100,

        "hold_probability_pct":
            prediction[
                "hold_probability"
            ] * 100,

        "expected_return_pct":
            prediction[
                "expected_return"
            ] * 100,

        "price":
            metrics[
                "price"
            ],

        "rsi":
            metrics[
                "rsi"
            ],

        "adx":
            metrics[
                "adx"
            ],

        "volume_ratio":
            metrics[
                "volume_ratio"
            ],

        "volatility_pct":
            metrics[
                "volatility"
            ] * 100,

        "atr_pct":
            metrics[
                "atr_pct"
            ] * 100,

        "momentum_pct":
            metrics[
                "momentum"
            ] * 100,

        "trend_pct":
            metrics[
                "trend"
            ] * 100,

        "risk_score":
            metrics[
                "risk_score"
            ],

        "liquidity_score":
            metrics[
                "liquidity_score"
            ],

        "candle_count":
            len(candles),

        "last_candle_utc":
            last_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }


# ============================================================
# SYMBOL SELECTION
# ============================================================

def select_symbols(
    symbols,
    limit
):

    usdt = [
        symbol
        for symbol in symbols
        if symbol.endswith(
            "USDT"
        )
    ]

    others = [
        symbol
        for symbol in symbols
        if symbol not in usdt
    ]

    return (
        usdt + others
    )[:int(limit)]


# ============================================================
# SCAN
# ============================================================

def run_scan(
    symbols,
    candle_target,
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
            f"taranıyor "
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

            cached = (
                st.session_state
                .candle_cache
                .get(symbol)
            )

            if (
                cached is None
                or len(cached)
                < candle_target
            ):

                status.info(
                    f"📥 {symbol}: "
                    f"{candle_target:,} "
                    "mum indiriliyor..."
                )

                candles = fetch_initial_history(
                    symbol,
                    candle_target
                )

            else:

                status.info(
                    f"🔄 {symbol}: "
                    "yeni 15m mumlar "
                    "kontrol ediliyor..."
                )

                candles = fetch_latest_candles(
                    symbol,
                    cached,
                    candle_target
                )

            if candles.empty:

                raise RuntimeError(
                    "Mum verisi alınamadı."
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

            errors.append({
                "symbol": symbol,
                "error": str(exc)
            })

    progress.progress(
        1.0
    )

    status.success(
        f"Tarama tamamlandı: "
        f"{len(results)} başarılı / "
        f"{len(errors)} hatalı"
    )

    if errors:

        st.session_state.last_error = (
            str(errors[:20])
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# DISPLAY
# ============================================================

def show_top_table(
    df,
    signal
):

    if df.empty:

        st.info(
            "Sonuç bulunamadı."
        )

        return

    filtered = df[
        df["signal"] == signal
    ].copy()

    if filtered.empty:

        st.warning(
            f"{signal} sinyali yok."
        )

        return

    filtered = filtered.sort_values(
        [
            "ai_score",
            "confidence_pct"
        ],
        ascending=False
    ).head(
        10
    )

    columns = [
        "symbol",
        "signal",
        "ai_score",
        "confidence_pct",
        "expected_return_pct",
        "buy_probability_pct",
        "sell_probability_pct",
        "rsi",
        "adx",
        "volume_ratio",
        "risk_score",
        "price"
    ]

    columns = [
        column
        for column in columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[
            columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Ayarlar"
)

base_url = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_BASE_URL
)

DEFAULT_BASE_URL = (
    base_url.strip().rstrip("/")
)

candle_target = st.sidebar.number_input(
    "Hedef mum sayısı",
    min_value=10_000,
    max_value=500_000,
    value=500_000,
    step=10_000
)

symbol_limit = st.sidebar.number_input(
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
    f"PyTorch: "
    f"{'OK' if TORCH_AVAILABLE else 'HATA'}"
)

st.sidebar.write(
    f"Device: `{DEVICE}`"
)

st.sidebar.write(
    f"Interval: `{INTERVAL}`"
)

st.sidebar.write(
    f"Sequence: `{SEQ_LEN}`"
)

st.sidebar.write(
    f"Future horizon: "
    f"`{FUTURE_HORIZON}`"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Binance Spot AI Deep Learning Scanner"
)

st.markdown(
    """
**15 dakikalık Spot piyasa tarayıcı**

Transformer + teknik göstergeler + gelecekteki
getiri tahmini kullanarak Top 10 BUY / SELL
sonuçları üretir.

> ⚠️ Sistem otomatik emir göndermez.
"""
)


# ============================================================
# PYTORCH ERROR
# ============================================================

if not TORCH_AVAILABLE:

    st.error(
        "❌ PyTorch yüklenemedi."
    )

    st.code(
        TORCH_IMPORT_ERROR
        or
        "Bilinmeyen PyTorch hatası"
    )

    st.markdown(
        """
### `requirements.txt`

```text
streamlit>=1.37,<2.0
requests>=2.32
pandas>=2.2
numpy>=1.26,<2.0
torch==2.4.1
