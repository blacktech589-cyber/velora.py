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
# OPTIONAL TORCH IMPORT
# ============================================================

TORCH_AVAILABLE = True

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


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

DEFAULT_SYMBOL_LIMIT = 20

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

if "manual_scan_counter" not in st.session_state:
    st.session_state.manual_scan_counter = 0


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


def normalize_series(series):
    series = pd.to_numeric(series, errors="coerce")

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


# ============================================================
# HTTP CLIENT
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "Spot-AI-Deep-Learning-Scanner/1.0"
})


def api_get(path, params=None):
    url = DEFAULT_BASE_URL + path

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 451:
        raise RuntimeError(
            "HTTP 451: Binance API bu konumdan/hesaptan "
            "erişimi kısıtlıyor. Bu durum Python, Streamlit "
            "veya PyTorch hatası değildir."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "HTTP 429: Binance rate limit. "
            "İstek sıklığını azaltın."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# BINANCE SPOT EXCHANGE INFO
# ============================================================

@st.cache_data(ttl=900)
def get_exchange_info():
    data = api_get("/api/v3/exchangeInfo")

    symbols = []

    for item in data.get("symbols", []):

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        quote = item.get("quoteAsset", "")

        if quote not in {
            "USDT",
            "USDC",
            "FDUSD",
            "BTC",
            "ETH",
            "BNB"
        }:
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        symbols.append(symbol)

    return sorted(set(symbols))


# ============================================================
# KLINE FETCH
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
        "limit": min(limit, KLINE_LIMIT),
    }

    if start_time is not None:
        params["startTime"] = int(start_time)

    if end_time is not None:
        params["endTime"] = int(end_time)

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
        "ignore",
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
        "taker_buy_quote",
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
        subset=["open_time"]
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# FULL 500K DOWNLOAD
# ============================================================

def fetch_initial_history(
    symbol,
    target_candles=500_000,
    progress_callback=None
):
    """
    Downloads historical 15m candles.

    Binance Spot API allows max 1000 candles per request.

    500,000 candles therefore requires many API requests.
    """

    all_parts = []

    remaining = target_candles

    end_time = None

    request_count = 0

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

        part = klines_to_dataframe(data)

        if part.empty:
            break

        all_parts.append(part)

        request_count += 1

        received = len(part)

        remaining -= received

        oldest_time = int(
            part["open_time"].min()
        )

        end_time = oldest_time - 1

        if progress_callback:
            progress_callback(
                min(
                    1.0,
                    (
                        target_candles
                        - max(remaining, 0)
                    )
                    / target_candles
                )
            )

        if received < batch_size:
            break

        time.sleep(0.05)

        if request_count > (
            target_candles // KLINE_LIMIT
        ) + 10:
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

    result = result.reset_index(drop=True)

    return result


# ============================================================
# INCREMENTAL UPDATE
# ============================================================

def fetch_latest_candles(
    symbol,
    existing_df,
    max_candles=500_000
):
    """
    Updates existing rolling candle cache.

    If existing data exists, only recent candles are requested.
    """

    if existing_df is None or existing_df.empty:
        return fetch_initial_history(
            symbol,
            max_candles
        )

    last_open_time = int(
        existing_df["open_time"].max()
    )

    start_time = last_open_time + 1

    try:
        data = fetch_klines_page(
            symbol=symbol,
            limit=1000,
            start_time=start_time
        )

    except Exception:
        raise

    if data:

        new_df = klines_to_dataframe(data)

        if not new_df.empty:

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

            combined = combined.reset_index(
                drop=True
            )

            return combined

    return existing_df


# ============================================================
# TECHNICAL INDICATORS
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

    previous_close = df["close"].shift(1)

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
    ).max(axis=1)


def atr(df, period=14):

    tr = true_range(df)

    return tr.ewm(
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


def bollinger(series, period=20):

    middle = sma(
        series,
        period
    )

    std = series.rolling(
        period
    ).std()

    upper = middle + (
        2 * std
    )

    lower = middle - (
        2 * std
    )

    width = (
        upper - lower
    ) / (
        middle.abs() + 1e-12
    )

    return (
        middle,
        upper,
        lower,
        width
    )


def stochastic(df, period=14):

    lowest = df["low"].rolling(
        period
    ).min()

    highest = df["high"].rolling(
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


def cci(df, period=20):

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
            lambda x: np.mean(
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


def williams_r(df, period=14):

    highest = df["high"].rolling(
        period
    ).max()

    lowest = df["low"].rolling(
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


def roc(series, period=12):

    return (
        series
        .pct_change(period)
        * 100
    )


def obv(df):

    direction = np.sign(
        df["close"].diff()
    )

    return (
        direction
        * df["volume"]
    ).fillna(0).cumsum()


def mfi(df, period=14):

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

    pos_sum = positive.rolling(
        period
    ).sum()

    neg_sum = negative.rolling(
        period
    ).sum()

    ratio = pos_sum / (
        neg_sum + 1e-12
    )

    return 100 - (
        100 / (
            1 + ratio
        )
    )


def adx(df, period=14):

    high = df["high"]

    low = df["low"]

    previous_high = high.shift(1)

    previous_low = low.shift(1)

    up_move = (
        high - previous_high
    )

    down_move = (
        previous_low - low
    )

    plus_dm = up_move.where(
        (
            up_move > down_move
        ) & (
            up_move > 0
        ),
        0
    )

    minus_dm = down_move.where(
        (
            down_move > up_move
        ) & (
            down_move > 0
        ),
        0
    )

    tr = true_range(df)

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
# FEATURE ENGINEERING
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

    "price_z_100",
]


def build_features(df):

    df = df.copy()

    close = df["close"]

    high = df["high"]

    low = df["low"]

    open_ = df["open"]

    volume = df["volume"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_1"] = close.pct_change(1)

    df["return_3"] = close.pct_change(3)

    df["return_6"] = close.pct_change(6)

    df["return_12"] = close.pct_change(12)

    df["return_24"] = close.pct_change(24)

    df["return_48"] = close.pct_change(48)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_5 = ema(
        close,
        5
    )

    ema_20 = ema(
        close,
        20
    )

    ema_50 = ema(
        close,
        50
    )

    ema_100 = ema(
        close,
        100
    )

    ema_200 = ema(
        close,
        200
    )

    ema_500 = ema(
        close,
        500
    )

    ema_800 = ema(
        close,
        800
    )

    df["ema_5_dist"] = (
        close / ema_5
    ) - 1

    df["ema_20_dist"] = (
        close / ema_20
    ) - 1

    df["ema_50_dist"] = (
        close / ema_50
    ) - 1

    df["ema_100_dist"] = (
        close / ema_100
    ) - 1

    df["ema_200_dist"] = (
        close / ema_200
    ) - 1

    df["ema_500_dist"] = (
        close / ema_500
    ) - 1

    df["ema_800_dist"] = (
        close / ema_800
    ) - 1

    df["ema_20_50"] = (
        ema_20 / ema_50
    ) - 1

    df["ema_50_200"] = (
        ema_50 / ema_200
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
    # Bollinger
    # --------------------------------------------------------

    bb_mid, bb_upper, bb_lower, bb_width = bollinger(
        close,
        20
    )

    df["bb_position"] = (
        close - bb_lower
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
    # Stochastic
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
    # Williams R
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
    # Volume
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
    # Candle structure
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    body = (
        close - open_
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
            open_,
            close
        )
    ) / (
        close + 1e-12
    )

    df["lower_wick_pct"] = (
        np.minimum(
            open_,
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
    # Volatility
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
    # High / Low distance
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
    # Drawdown
    # --------------------------------------------------------

    df["drawdown_100"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    # --------------------------------------------------------
    # Price Z score
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
# TARGET CREATION
# ============================================================

def create_targets(
    df,
    horizon=FUTURE_HORIZON,
    threshold=CLASS_THRESHOLD
):

    future_return = (
        df["close"]
        .shift(-horizon)
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

    df = df.copy()

    df["future_return"] = future_return

    df["target"] = target

    df.loc[
        future_return.isna(),
        "target"
    ] = np.nan

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_training_data(
    feature_df
):

    cols = FEATURE_COLUMNS + [
        "target",
        "future_return"
    ]

    data = feature_df[
        cols
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
# NORMALIZATION
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

        if self.mean is None:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        return (
            x - self.mean
        ) / self.std


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=4096
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
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
            + self.pe[:, :length]
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
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
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
            )
            .squeeze(-1)
        )

        return (
            classification,
            regression
        )


# ============================================================
# MODEL TRAINING
# ============================================================

def prepare_sequences(
    data,
    scaler,
    seq_len=SEQ_LEN,
    max_samples=2500
):

    x = data[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    y_class = data[
        "target"
    ].values.astype(
        np.int64
    )

    y_return = data[
        "future_return"
    ].values.astype(
        np.float32
    )

    x = scaler.transform(
        x
    ).astype(
        np.float32
    )

    sequences = []

    class_targets = []

    return_targets = []

    start = max(
        seq_len,
        len(x) - max_samples
    )

    for i in range(
        start,
        len(x)
    ):

        sequence = x[
            i - seq_len:i
        ]

        if len(sequence) != seq_len:
            continue

        sequences.append(
            sequence
        )

        class_targets.append(
            y_class[i]
        )

        return_targets.append(
            y_return[i]
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
            class_targets,
            dtype=np.int64
        ),
        np.asarray(
            return_targets,
            dtype=np.float32
        )
    )


def train_model(
    data,
    epochs=TRAIN_EPOCHS
):

    if not TORCH_AVAILABLE:

        raise RuntimeError(
            "PyTorch yüklü değil. "
            "requirements.txt içine "
            "torch ekleyin."
        )

    if len(data) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            "Model eğitimi için yeterli "
            "temiz veri yok."
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

    x, y_class, y_return = prepare_sequences(
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

    classification_loss = nn.CrossEntropyLoss()

    regression_loss = nn.HuberLoss()

    model.train()

    for epoch in range(
        epochs
    ):

        epoch_loss = 0.0

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

            logits, regression = model(
                batch_x
            )

            loss_class = (
                classification_loss(
                    logits,
                    batch_class
                )
            )

            loss_return = (
                regression_loss(
                    regression,
                    batch_return
                )
            )

            loss = (
                loss_class
                + 2.0 * loss_return
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

    return (
        model,
        scaler
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_latest(
    model,
    scaler,
    data
):

    if len(data) < SEQ_LEN:

        raise RuntimeError(
            "Tahmin için yeterli veri yok."
        )

    feature_values = data[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    feature_values = scaler.transform(
        feature_values
    ).astype(
        np.float32
    )

    latest = feature_values[
        -SEQ_LEN:
    ]

    x = torch.tensor(
        latest,
        dtype=torch.float32
    ).unsqueeze(0).to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        logits, regression = model(
            x
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

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

    if buy_probability >= max(
        sell_probability,
        hold_probability
    ):

        signal = "BUY"

    elif sell_probability >= max(
        buy_probability,
        hold_probability
    ):

        signal = "SELL"

    else:

        signal = "HOLD"

    confidence = max(
        sell_probability,
        hold_probability,
        buy_probability
    )

    return {
        "signal": signal,
        "sell_probability": sell_probability,
        "hold_probability": hold_probability,
        "buy_probability": buy_probability,
        "confidence": confidence,
        "expected_return": expected_return,
    }


# ============================================================
# MARKET METRICS
# ============================================================

def calculate_market_metrics(
    feature_df
):

    row = feature_df.iloc[-1]

    close = safe_float(
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

    # --------------------------------------------------------
    # Trend score
    # --------------------------------------------------------

    trend_score = clamp(
        50
        + trend * 1000,
        0,
        100
    )

    # --------------------------------------------------------
    # Momentum score
    # --------------------------------------------------------

    momentum_score = clamp(
        50
        + momentum * 500,
        0,
        100
    )

    # --------------------------------------------------------
    # RSI score
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Liquidity score
    # --------------------------------------------------------

    liquidity_score = clamp(
        volume_ratio * 40,
        0,
        100
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_score = clamp(
        50
        + volatility * 500
        + atr_pct * 300,
        0,
        100
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    trend_strength = clamp(
        adx_value,
        0,
        100
    )

    return {
        "price": close,
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
        "risk_score": risk_score,
        "trend_strength": trend_strength,
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

    direction_component = (
        buy_probability
        - sell_probability
    ) * 100

    expected_component = (
        expected_return * 10000
    )

    score = (
        direction_component * 0.35
        + confidence * 100 * 0.20
        + trend_score * 0.15
        + momentum_score * 0.10
        + liquidity_score * 0.05
        + expected_component * 0.15
    )

    score -= (
        risk_score * 0.05
    )

    return float(
        score
    )


# ============================================================
# SINGLE SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    candle_df
):

    if candle_df.empty:

        raise RuntimeError(
            "Candle data boş."
        )

    features = build_features(
        candle_df
    )

    features = create_targets(
        features
    )

    clean_data = clean_training_data(
        features
    )

    if len(clean_data) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            f"{symbol}: yeterli temiz "
            "eğitim verisi yok."
        )

    model, scaler = train_model(
        clean_data
    )

    prediction = predict_latest(
        model,
        scaler,
        clean_data
    )

    metrics = calculate_market_metrics(
        features.dropna(
            subset=FEATURE_COLUMNS
        )
    )

    score = calculate_ai_score(
        prediction,
        metrics
    )

    last_candle = candle_df.iloc[-1]

    timestamp = datetime.fromtimestamp(
        safe_float(
            last_candle[
                "open_time"
            ]
        ) / 1000,
        tz=timezone.utc
    )

    result = {
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
            len(candle_df),

        "last_candle_utc":
            timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    return result


# ============================================================
# SYMBOL SELECTION
# ============================================================

def select_symbols(
    symbols,
    limit
):

    # Stable deterministic selection.
    # USDT pairs are preferred.

    usdt = [
        s for s in symbols
        if s.endswith("USDT")
    ]

    other = [
        s for s in symbols
        if s not in usdt
    ]

    selected = (
        usdt + other
    )[:limit]

    return selected


# ============================================================
# SCAN ENGINE
# ============================================================

def run_scan(
    symbols,
    candle_target,
    progress_placeholder,
    status_placeholder
):

    results = []

    total = len(
        symbols
    )

    errors = []

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        status_placeholder.info(
            f"🔎 {symbol} taranıyor "
            f"({index}/{total})"
        )

        progress_placeholder.progress(
            (
                index - 1
            ) / max(
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
                or len(cached) < candle_target
            ):

                status_placeholder.info(
                    f"📥 {symbol}: "
                    f"{candle_target:,} "
                    "mum hazırlanıyor..."
                )

                candles = fetch_initial_history(
                    symbol,
                    candle_target
                )

            else:

                status_placeholder.info(
                    f"🔄 {symbol}: "
                    "yeni 15m mumları ekleniyor..."
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

            continue

    progress_placeholder.progress(
        1.0
    )

    status_placeholder.success(
        f"Tarama tamamlandı. "
        f"{len(results)} başarılı, "
        f"{len(errors)} hatalı."
    )

    if errors:

        st.session_state.last_error = (
            str(errors[:10])
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_results(df):

    if df.empty:
        return df

    output = df.copy()

    numeric_columns = [
        "ai_score",
        "confidence_pct",
        "buy_probability_pct",
        "sell_probability_pct",
        "hold_probability_pct",
        "expected_return_pct",
        "price",
        "rsi",
        "adx",
        "volume_ratio",
        "volatility_pct",
        "atr_pct",
        "momentum_pct",
        "trend_pct",
        "risk_score",
        "liquidity_score",
        "candle_count",
    ]

    for column in numeric_columns:

        if column in output.columns:

            output[column] = pd.to_numeric(
                output[column],
                errors="coerce"
            )

    return output


def show_top_table(
    df,
    signal,
    count=10
):

    if df.empty:

        st.info(
            "Henüz sonuç yok."
        )

        return

    filtered = df[
        df["signal"] == signal
    ].copy()

    if filtered.empty:

        st.warning(
            f"{signal} sonucu bulunamadı."
        )

        return

    filtered = filtered.sort_values(
        [
            "ai_score",
            "confidence_pct"
        ],
        ascending=False
    ).head(
        count
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
        "price",
    ]

    columns = [
        c for c in columns
        if c in filtered.columns
    ]

    display = filtered[
        columns
    ].copy()

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Scanner Settings"
)

st.sidebar.caption(
    "Binance Spot / 15m / Deep Learning"
)

base_url_input = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_BASE_URL
)

if base_url_input:

    # This only affects this running process.
    DEFAULT_BASE_URL = (
        base_url_input
        .strip()
        .rstrip("/")
    )

candle_target = st.sidebar.number_input(
    "Hedef mum sayısı",
    min_value=10_000,
    max_value=500_000,
    value=500_000,
    step=10_000
)

symbol_limit = st.sidebar.number_input(
    "Tarama yapılacak parite sayısı",
    min_value=1,
    max_value=200,
    value=20,
    step=1
)

refresh_enabled = st.sidebar.checkbox(
    "Otomatik 15 dakika yenile",
    value=True
)

if refresh_enabled:

    run_every = "15m"

else:

    run_every = None

st.sidebar.divider()

st.sidebar.write(
    f"**Device:** `{DEVICE}`"
)

st.sidebar.write(
    f"**PyTorch:** "
    f"`{'OK' if TORCH_AVAILABLE else 'YOK'}`"
)

st.sidebar.write(
    f"**Interval:** `{INTERVAL}`"
)

st.sidebar.write(
    f"**Sequence:** `{SEQ_LEN}`"
)

st.sidebar.write(
    f"**Future horizon:** "
    f"`{FUTURE_HORIZON}` candle"
)

st.sidebar.write(
    f"**Base URL:** `{DEFAULT_BASE_URL}`"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Binance Spot AI Deep Learning Scanner"
)

st.markdown(
    """
### Sistem

Bu uygulama **Spot piyasa verilerini** analiz eder.

Model:

- Transformer Encoder
- Classification: SELL / HOLD / BUY
- Regression: gelecekteki beklenen getiri
- 15 dakikalık mum
- Rolling 500K candle hedefi
- Teknik indikatör feature engineering
- Top 10 BUY
- Top 10 SELL

**Bu sistem emir göndermez.**
"""
)


# ============================================================
# TORCH WARNING
# ============================================================

if not TORCH_AVAILABLE:

    st.error(
        """
PyTorch bulunamadı.

`requirements.txt` içine:

`torch==2.4.1`

ekleyin.

Streamlit Cloud kullanıyorsanız uygulamayı
yeniden deploy edin.
"""
    )

    st.stop()


# ============================================================
# 15 MINUTE FRAGMENT
# ============================================================

def scanner_body():

    st.subheader(
        "🔎 Market Scanner"
    )

    top_col1, top_col2, top_col3 = st.columns(
        3
    )

    with top_col1:

        if st.session_state.last_scan_time:

            st.metric(
                "Son tarama",
                st.session_state.last_scan_time
            )

        else:

            st.metric(
                "Son tarama",
                "-"
            )

    with top_col2:

        st.metric(
            "Cache'deki parite",
            len(
                st.session_state
                .candle_cache
            )
        )

    with top_col3:

        st.metric(
            "Sonuç sayısı",
            len(
                st.session_state.last_results
            )
        )

    st.divider()

    control_col1, control_col2 = st.columns(
        [1, 4]
    )

    with control_col1:

        manual_scan = st.button(
            "🚀 Şimdi Tara",
            use_container_width=True,
            type="primary"
        )

    with control_col2:

        st.caption(
            "Otomatik yenileme açıksa "
            "bu bölüm yaklaşık 15 dakikada "
            "bir tekrar çalışır."
        )

    # --------------------------------------------------------
    # AUTO SCAN DECISION
    # --------------------------------------------------------

    should_scan = (
        manual_scan
        or st.session_state.last_results.empty
    )

    if should_scan:

        progress_placeholder = st.empty()

        status_placeholder = st.empty()

        try:

            status_placeholder.info(
                "Binance Spot market bilgisi alınıyor..."
            )

            symbols = get_exchange_info()

            selected_symbols = select_symbols(
                symbols,
                int(symbol_limit)
            )

            if not selected_symbols:

                raise RuntimeError(
                    "Taranacak sembol bulunamadı."
                )

            st.write(
                "### Taranan pariteler"
            )

            st.write(
                ", ".join(
                    selected_symbols
                )
            )

            results = run_scan(
                selected_symbols,
                int(candle_target),
                progress_placeholder,
                status_placeholder
            )

            results = format_results(
                results
            )

            st.session_state.last_results = (
                results
            )

            st.session_state.last_scan_time = (
                utc_now().strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )

            st.session_state.scan_history.append(
                {
                    "time":
                        st.session_state.last_scan_time,
                    "symbols":
                        len(selected_symbols),
                    "results":
                        len(results),
                }
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
                    """
HTTP 451 tespit edildi.

Binance Spot API bu çalışma
ortamından erişilebilir olmayabilir.

Bu bir PyTorch veya Streamlit
hatası değildir.

İzin verilen alternatif bir market-data
API veya Binance-compatible endpoint
kullanmanız gerekir.
"""
                )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    df = st.session_state.last_results

    if df.empty:

        st.warning(
            "Henüz veri yok. "
            "İlk taramanın tamamlanmasını bekleyin."
        )

        if st.session_state.last_error:

            with st.expander(
                "Son hata detayları"
            ):

                st.code(
                    st.session_state.last_error
                )

        return

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    buy_count = int(
        (
            df["signal"] == "BUY"
        ).sum()
    )

    sell_count = int(
        (
            df["signal"] == "SELL"
        ).sum()
    )

    hold_count = int(
        (
            df["signal"] == "HOLD"
        ).sum()
    )

    best_score = safe_float(
        df["ai_score"].max()
    )

    avg_confidence = safe_float(
        df["confidence_pct"].mean()
    )

    c1, c2, c3, c4, c5 = st.columns(
        5
    )

    c1.metric(
        "BUY",
        buy_count
    )

    c2.metric(
        "SELL",
        sell_count
    )

    c3.metric(
        "HOLD",
        hold_count
    )

    c4.metric(
        "Best AI Score",
        f"{best_score:.2f}"
    )

    c5.metric(
        "Avg Confidence",
        f"{avg_confidence:.1f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    st.header(
        "🟢 Top 10 BUY"
    )

    show_top_table(
        df,
        "BUY",
        10
    )

    st.header(
        "🔴 Top 10 SELL"
    )

    show_top_table(
        df,
        "SELL",
        10
    )

    # --------------------------------------------------------
    # FULL RESULTS
    # --------------------------------------------------------

    st.header(
        "📊 Tüm Sonuçlar"
    )

    full_display = df.sort_values(
        [
            "ai_score",
            "confidence_pct"
        ],
        ascending=False
    )

    st.dataframe(
        full_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    st.header(
        "💾 Export"
    )

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ CSV indir",
        data=csv_data,
        file_name=(
            "spot_ai_scan.csv"
        ),
        mime="text/csv"
    )

    # --------------------------------------------------------
    # SYMBOL DETAIL
    # --------------------------------------------------------

    st.header(
        "🔍 Coin Detayı"
    )

    symbol_options = sorted(
        df["symbol"].tolist()
    )

    if symbol_options:

        selected_symbol = st.selectbox(
            "Parite seç",
            symbol_options
        )

        selected = df[
            df["symbol"]
            == selected_symbol
        ]

        if not selected.empty:

            row = selected.iloc[0]

            detail1, detail2, detail3 = st.columns(
                3
            )

            with detail1:

                st.metric(
                    "Signal",
                    row["signal"]
                )

                st.metric(
                    "AI Score",
                    f"{row['ai_score']:.2f}"
                )

                st.metric(
                    "Confidence",
                    f"{row['confidence_pct']:.2f}%"
                )

            with detail2:

                st.metric(
                    "Expected Return",
                    f"{row['expected_return_pct']:.3f}%"
                )

                st.metric(
                    "BUY Probability",
                    f"{row['buy_probability_pct']:.2f}%"
                )

                st.metric(
                    "SELL Probability",
                    f"{row['sell_probability_pct']:.2f}%"
                )

            with detail3:

                st.metric(
                    "RSI",
                    f"{row['rsi']:.2f}"
                )

                st.metric(
                    "ADX",
                    f"{row['adx']:.2f}"
                )

                st.metric(
                    "Risk Score",
                    f"{row['risk_score']:.2f}"
                )

            st.write(
                "### Teknik ölçümler"
            )

            detail_columns = [
                "symbol",
                "price",
                "rsi",
                "adx",
                "volume_ratio",
                "volatility_pct",
                "atr_pct",
                "momentum_pct",
                "trend_pct",
                "liquidity_score",
                "risk_score",
                "candle_count",
                "last_candle_utc",
            ]

            detail_columns = [
                c
                for c in detail_columns
                if c in row.index
            ]

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            c: row[c]
                            for c in detail_columns
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    st.header(
        "🕒 Scan History"
    )

    if st.session_state.scan_history:

        history_df = pd.DataFrame(
            st.session_state.scan_history
        )

        st.dataframe(
            history_df.tail(20),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # CACHE INFORMATION
    # --------------------------------------------------------

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
                "candles": len(candles),
                "first_candle": (
                    datetime.fromtimestamp(
                        safe_float(
                            candles.iloc[0][
                                "open_time"
                            ]
                        ) / 1000,
                        tz=timezone.utc
                    ).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if not candles.empty
                    else "-"
                ),
                "last_candle": (
                    datetime.fromtimestamp(
                        safe_float(
                            candles.iloc[-1][
                                "open_time"
                            ]
                        ) / 1000,
                        tz=timezone.utc
                    ).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if not candles.empty
                    else "-"
                )
            }
        )

    if cache_rows:

        cache_df = pd.DataFrame(
            cache_rows
        )

        st.dataframe(
            cache_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # SYSTEM INFORMATION
    # --------------------------------------------------------

    st.header(
        "🖥️ System Health"
    )

    health_data = {
        "Python / Streamlit session": "OK",
        "PyTorch": (
            "OK"
            if TORCH_AVAILABLE
            else "MISSING"
        ),
        "Device": DEVICE,
        "Interval": INTERVAL,
        "Target candles": candle_target,
        "Sequence length": SEQ_LEN,
        "Future horizon": FUTURE_HORIZON,
        "Symbols": symbol_limit,
        "Auto refresh": (
            "15 min"
            if refresh_enabled
            else "OFF"
        ),
        "Trading execution": "DISABLED",
        "Base URL": DEFAULT_BASE_URL,
    }

    health_df = pd.DataFrame(
        [
            {
                "Parameter": key,
                "Value": value
            }
            for key, value
            in health_data.items()
        ]
    )

    st.dataframe(
        health_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    st.warning(
        """
Bu uygulama finansal tavsiye değildir.

AI sonucu BUY/SELL olarak sınıflandırılsa bile
bu sonuç gelecekte fiyatın kesin olarak o yönde
hareket edeceği anlamına gelmez.

Model yalnızca geçmiş piyasa verilerinden
olasılıksal bir tahmin üretir.

Uygulama otomatik emir göndermez.
"""
    )


# ============================================================
# RUN FRAGMENT
# ============================================================

if refresh_enabled:

    try:

        scanner_body_fragment = st.fragment(
            run_every="15m"
        )

        with scanner_body_fragment:

            scanner_body()

    except AttributeError:

        st.warning(
            """
Kullandığınız Streamlit sürümü
`st.fragment` desteklemiyor.

Streamlit'i güncelleyin:

pip install -U streamlit
"""
        )

        scanner_body()

else:

    scanner_body()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Deep Learning Scanner | "
    "15m | Transformer | No Order Execution"
)
