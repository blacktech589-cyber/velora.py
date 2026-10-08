import io
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================
# CONFIG
# ============================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)

APP_TITLE = "Spot Scalping AI Scanner"

TIMEFRAME = "15m"

# 15m x 2 candles = maximum 30 minutes
FUTURE_BARS = 2
MAX_HORIZON_MINUTES = 30

BUY_TARGET = 0.10
SELL_TARGET = -0.10

DEFAULT_LOOKBACK = 5000
MAX_LOOKBACK = 500000

REQUEST_TIMEOUT = 20

MIN_TRAIN_ROWS = 500

FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_4",
    "ret_8",
    "ret_16",
    "ret_24",

    "ema_5_20",
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

    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",

    "volume_ratio",
    "volume_z",

    "volatility_8",
    "volatility_24",

    "price_z",
    "distance_ema20",
    "distance_ema50",
    "distance_ema200",
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def sf(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return default


def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b


def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "timestamp": [
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time",
            "open_timestamp",
        ],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "price"],
        "volume": ["volume", "vol", "v"],
    }

    rename = {}

    for standard, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = standard
                break

    df = df.rename(columns=rename)

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV eksik kolonlar: "
            + ", ".join(missing)
        )

    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
                utc=True,
            )

            df["timestamp"] = ts

        except Exception:
            df["timestamp"] = pd.NaT

    else:
        df["timestamp"] = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"),
            periods=len(df),
            freq="15min",
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df[
        df["close"] > 0
    ]

    df = df[
        df["high"] >= df["low"]
    ]

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# DATA PROVIDERS
# ============================================================

class DataProviderError(Exception):
    pass


def fetch_binance_klines(
    symbol,
    limit=1000,
):
    """
    Binance Spot public REST.

    Some regions may receive HTTP 451.
    We do not attempt to bypass regional restrictions.
    """

    symbol = symbol.upper().replace("/", "")

    url = (
        "https://api.binance.com/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": "15m",
        "limit": min(int(limit), 1000),
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 451:
        raise DataProviderError(
            "Binance HTTP 451: API bölgesel olarak "
            "erişilemiyor."
        )

    if response.status_code != 200:
        raise DataProviderError(
            f"Binance HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    raw = response.json()

    if not raw:
        raise DataProviderError(
            f"Binance veri döndürmedi: {symbol}"
        )

    rows = []

    for item in raw:
        rows.append(
            {
                "timestamp": pd.to_datetime(
                    item[0],
                    unit="ms",
                    utc=True,
                ),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            }
        )

    return pd.DataFrame(rows)


def fetch_binance_history(
    symbol,
    candles=5000,
):
    """
    Binance pagination.

    Maximum request = 1000 candles/request.
    """

    symbol = symbol.upper().replace("/", "")

    all_rows = []

    end_time = None

    remaining = int(candles)

    while remaining > 0:

        batch = min(
            remaining,
            1000,
        )

        url = (
            "https://api.binance.com/api/v3/klines"
        )

        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": batch,
        }

        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 451:
            raise DataProviderError(
                "Binance HTTP 451: API bölgesel "
                "erişim kısıtlaması."
            )

        if response.status_code != 200:
            raise DataProviderError(
                f"Binance HTTP "
                f"{response.status_code}"
            )

        raw = response.json()

        if not raw:
            break

        for item in raw:
            all_rows.append(
                {
                    "timestamp": pd.to_datetime(
                        item[0],
                        unit="ms",
                        utc=True,
                    ),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )

        oldest = raw[0][0]

        end_time = oldest - 1

        remaining -= len(raw)

        if len(raw) < batch:
            break

        time.sleep(0.05)

    if not all_rows:
        raise DataProviderError(
            f"Binance geçmiş verisi alınamadı: {symbol}"
        )

    df = pd.DataFrame(
        all_rows
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        "timestamp"
    )

    return df.tail(
        candles
    ).reset_index(drop=True)


def fetch_kraken_ohlc(
    symbol,
    limit=720,
):
    """
    Kraken public OHLC.

    Kraken'in public endpoint'i sınırlı sayıda
    recent candle döndürebilir.
    """

    symbol = symbol.upper().replace("/", "")

    mapping = {
        "BTCUSDT": "XBTUSD",
        "BTCUSD": "XBTUSD",
        "ETHUSDT": "ETHUSD",
        "ETHUSD": "ETHUSD",
        "SOLUSDT": "SOLUSD",
        "XRPUSDT": "XRPUSD",
        "ADAUSDT": "ADAUSD",
        "DOGEUSDT": "DOGEUSD",
    }

    pair = mapping.get(
        symbol,
        symbol,
    )

    url = (
        "https://api.kraken.com/0/public/OHLC"
    )

    params = {
        "pair": pair,
        "interval": 15,
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise DataProviderError(
            f"Kraken HTTP "
            f"{response.status_code}"
        )

    payload = response.json()

    if payload.get("error"):
        raise DataProviderError(
            str(payload["error"])
        )

    result = payload.get(
        "result",
        {},
    )

    data_key = None

    for key in result.keys():
        if key != "last":
            data_key = key
            break

    if data_key is None:
        raise DataProviderError(
            f"Kraken veri yok: {symbol}"
        )

    rows = []

    for item in result[data_key]:

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    int(item[0]),
                    unit="s",
                    utc=True,
                ),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[6]),
            }
        )

    df = pd.DataFrame(rows)

    return df.tail(
        limit
    ).reset_index(drop=True)


def load_csv_data(
    uploaded_file
):
    data = uploaded_file.read()

    df = pd.read_csv(
        io.BytesIO(data)
    )

    return normalize_columns(
        df
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(
    close,
    period=14,
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
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = safe_div(
        avg_gain,
        avg_loss,
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def calculate_atr(
    df,
    period=14,
):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - prev_close
    ).abs()

    tr3 = (
        low - prev_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def calculate_adx(
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
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * safe_div(
            plus_dm.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            ).mean(),
            atr,
        )
    )

    minus_di = (
        100
        * safe_div(
            minus_dm.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            ).mean(),
            atr,
        )
    )

    dx = (
        100
        * safe_div(
            (plus_di - minus_di).abs(),
            plus_di + minus_di,
        )
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx


def calculate_stochastic(
    df,
    period=14,
    smooth=3,
):
    lowest = (
        df["low"]
        .rolling(period)
        .min()
    )

    highest = (
        df["high"]
        .rolling(period)
        .max()
    )

    k = (
        100
        * safe_div(
            df["close"] - lowest,
            highest - lowest,
        )
    )

    d = k.rolling(
        smooth
    ).mean()

    return k, d


def build_features(
    df
):
    x = normalize_columns(
        df
    )

    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]
    volume = x["volume"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    for n in [
        1,
        2,
        3,
        4,
        8,
        16,
        24,
    ]:
        x[f"ret_{n}"] = (
            close.pct_change(n)
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    x["ema5"] = (
        close.ewm(
            span=5,
            adjust=False,
        ).mean()
    )

    x["ema20"] = (
        close.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    x["ema50"] = (
        close.ewm(
            span=50,
            adjust=False,
        ).mean()
    )

    x["ema200"] = (
        close.ewm(
            span=200,
            adjust=False,
        ).mean()
    )

    x["ema_5_20"] = safe_div(
        x["ema5"],
        x["ema20"],
    ) - 1

    x["ema_20_50"] = safe_div(
        x["ema20"],
        x["ema50"],
    ) - 1

    x["ema_50_200"] = safe_div(
        x["ema50"],
        x["ema200"],
    ) - 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    x["rsi_7"] = calculate_rsi(
        close,
        7,
    )

    x["rsi_14"] = calculate_rsi(
        close,
        14,
    )

    x["rsi_21"] = calculate_rsi(
        close,
        21,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    x["macd"] = (
        ema12 - ema26
    )

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # --------------------------------------------------------
    # Bollinger
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

    x["bb_position"] = safe_div(
        close - bb_lower,
        bb_upper - bb_lower,
    )

    x["bb_width"] = safe_div(
        bb_upper - bb_lower,
        bb_mid,
    )

    # --------------------------------------------------------
    # ATR / ADX
    # --------------------------------------------------------

    atr = calculate_atr(
        x,
        14,
    )

    x["atr_pct"] = safe_div(
        atr,
        close,
    )

    x["adx"] = calculate_adx(
        x,
        14,
    )

    # --------------------------------------------------------
    # Stochastic
    # --------------------------------------------------------

    x["stoch_k"], x["stoch_d"] = (
        calculate_stochastic(
            x,
            14,
            3,
        )
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan,
    )

    body = (
        close - open_
    ).abs()

    upper_wick = (
        high
        - pd.concat(
            [open_, close],
            axis=1,
        ).max(axis=1)
    )

    lower_wick = (
        pd.concat(
            [open_, close],
            axis=1,
        ).min(axis=1)
        - low
    )

    x["body_pct"] = safe_div(
        body,
        candle_range,
    )

    x["upper_wick_pct"] = safe_div(
        upper_wick,
        candle_range,
    )

    x["lower_wick_pct"] = safe_div(
        lower_wick,
        candle_range,
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_ma = (
        volume
        .rolling(20)
        .mean()
    )

    volume_std = (
        volume
        .rolling(20)
        .std()
    )

    x["volume_ratio"] = safe_div(
        volume,
        volume_ma,
    )

    x["volume_z"] = safe_div(
        volume - volume_ma,
        volume_std,
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    x["volatility_8"] = (
        close
        .pct_change()
        .rolling(8)
        .std()
    )

    x["volatility_24"] = (
        close
        .pct_change()
        .rolling(24)
        .std()
    )

    # --------------------------------------------------------
    # Price z-score
    # --------------------------------------------------------

    price_ma = (
        close
        .rolling(50)
        .mean()
    )

    price_std = (
        close
        .rolling(50)
        .std()
    )

    x["price_z"] = safe_div(
        close - price_ma,
        price_std,
    )

    # --------------------------------------------------------
    # EMA distances
    # --------------------------------------------------------

    x["distance_ema20"] = safe_div(
        close,
        x["ema20"],
    ) - 1

    x["distance_ema50"] = safe_div(
        close,
        x["ema50"],
    ) - 1

    x["distance_ema200"] = safe_div(
        close,
        x["ema200"],
    ) - 1

    return x


# ============================================================
# TARGET
# ============================================================

def make_target(
    df
):
    """
    STRICT 30-MINUTE TARGET.

    BUY:
        +10% high is reached within next 2 x 15m candles.

    SELL:
        -10% low is reached within next 2 x 15m candles.

    No HOLD.

    Ambiguous cases where both targets are first hit
    in the same candle are excluded from training.
    """

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

    buy1 = (
        high1
        >= entry * 1.10
    )

    buy2 = (
        high2
        >= entry * 1.10
    )

    sell1 = (
        low1
        <= entry * 0.90
    )

    sell2 = (
        low2
        <= entry * 0.90
    )

    buy_hit = (
        buy1 | buy2
    )

    sell_hit = (
        sell1 | sell2
    )

    first_buy = np.where(
        buy1,
        1,
        np.where(
            buy2,
            2,
            999,
        ),
    )

    first_sell = np.where(
        sell1,
        1,
        np.where(
            sell2,
            2,
            999,
        ),
    )

    target = np.full(
        len(x),
        np.nan,
    )

    # Only BUY target hit
    buy_only = (
        buy_hit
        & ~sell_hit
    )

    target[buy_only] = 1

    # Only SELL target hit
    sell_only = (
        sell_hit
        & ~buy_hit
    )

    target[sell_only] = 0

    # Both targets hit
    both = (
        buy_hit
        & sell_hit
    )

    buy_first = (
        both
        & (
            first_buy
            < first_sell
        )
    )

    sell_first = (
        both
        & (
            first_sell
            < first_buy
        )
    )

    target[buy_first] = 1

    target[sell_first] = 0

    # Same candle = ambiguous
    same_candle = (
        both
        & (
            first_buy
            == first_sell
        )
    )

    target[same_candle] = np.nan

    x["target"] = target

    x["buy_target"] = (
        buy_hit.astype(float)
    )

    x["sell_target"] = (
        sell_hit.astype(float)
    )

    x["future_max_return"] = (
        pd.concat(
            [
                high1 / entry - 1,
                high2 / entry - 1,
            ],
            axis=1,
        )
        .max(axis=1)
    )

    x["future_min_return"] = (
        pd.concat(
            [
                low1 / entry - 1,
                low2 / entry - 1,
            ],
            axis=1,
        )
        .min(axis=1)
    )

    x["future_close_return"] = (
        x["close"]
        .shift(-2)
        / x["close"]
        - 1
    )

    return x


# ============================================================
# MODEL
# ============================================================

def balance_training_data(
    X,
    y,
):
    """
    Simple random oversampling.

    No imblearn dependency needed.
    """

    data = X.copy()
    data["_target"] = y.values

    groups = []

    counts = (
        data["_target"]
        .value_counts()
    )

    if len(counts) < 2:
        return X, y

    max_count = counts.max()

    rng = np.random.default_rng(
        42
    )

    for cls in sorted(
        counts.index
    ):

        part = data[
            data["_target"] == cls
        ]

        if len(part) < max_count:

            indices = rng.choice(
                len(part),
                size=max_count,
                replace=True,
            )

            part = pd.concat(
                [
                    part,
                    part.iloc[indices],
                ],
                ignore_index=True,
            )

        groups.append(
            part
        )

    balanced = pd.concat(
        groups,
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
# MODEL TRAINING / PREDICTION
# ============================================================

def train_predict(
    df,
    train_limit=20000,
):
    """
    Binary BUY/SELL AI.

    There is no HOLD class.
    """

    if len(df) < MIN_TRAIN_ROWS:
        raise DataProviderError(
            f"Yetersiz candle: {len(df)}. "
            f"Minimum: {MIN_TRAIN_ROWS}"
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

    if len(usable) < MIN_TRAIN_ROWS:
        raise DataProviderError(
            "Feature/target temizliği "
            "sonrası yeterli veri kalmadı."
        )

    usable["target"] = (
        usable["target"]
        .astype(int)
    )

    # Limit training size while preserving chronology
    usable = usable.tail(
        min(
            int(train_limit),
            len(usable),
        )
    ).reset_index(
        drop=True
    )

    if usable["target"].nunique() < 2:
        raise DataProviderError(
            "Training içinde hem BUY hem SELL "
            "örneği bulunamadı."
        )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(usable)

    train_end = int(
        n * 0.70
    )

    valid_end = int(
        n * 0.85
    )

    if train_end < 200:
        raise DataProviderError(
            "Training datası çok küçük."
        )

    if valid_end <= train_end:
        valid_end = (
            train_end + 1
        )

    if valid_end >= n:
        valid_end = n - 1

    train = usable.iloc[
        :train_end
    ]

    validation = usable.iloc[
        train_end:valid_end
    ]

    test = usable.iloc[
        valid_end:
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
        X_train_balanced,
        y_train_balanced,
    ) = balance_training_data(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    model = make_model()

    model.fit(
        X_train_balanced,
        y_train_balanced,
    )

    test_pred = (
        model.predict(
            X_test
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_pred,
        )
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_pred,
        )
    )

    # --------------------------------------------------------
    # Latest
    # --------------------------------------------------------

    latest_rows = (
        labeled
        .dropna(
            subset=FEATURES
        )
    )

    if latest_rows.empty:
        raise DataProviderError(
            "Latest feature satırı yok."
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

    probability = (
        model.predict_proba(
            X_latest
        )[0]
    )

    classes = (
        model.classes_
    )

    class_prob = {
        int(c): float(p)
        for c, p in zip(
            classes,
            probability,
        )
    }

    p_sell = (
        class_prob.get(
            0,
            0.0,
        )
    )

    p_buy = (
        class_prob.get(
            1,
            0.0,
        )
    )

    # --------------------------------------------------------
    # FORCE BUY OR SELL
    # --------------------------------------------------------

    if p_buy >= p_sell:

        signal = "BUY"

        confidence = p_buy

    else:

        signal = "SELL"

        confidence = p_sell

    # --------------------------------------------------------
    # Current market values
    # --------------------------------------------------------

    current_price = float(
        latest["close"]
    )

    # --------------------------------------------------------
    # Future diagnostics
    # --------------------------------------------------------

    future_highs = pd.concat(
        [
            df["high"].shift(-1),
            df["high"].shift(-2),
        ],
        axis=1,
    )

    future_lows = pd.concat(
        [
            df["low"].shift(-1),
            df["low"].shift(-2),
        ],
        axis=1,
    )

    max_future_high = (
        future_highs.iloc[-1]
        .max()
    )

    min_future_low = (
        future_lows.iloc[-1]
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
    # Technical direction
    # --------------------------------------------------------

    momentum_24 = sf(
        latest.get(
            "ret_24"
        )
    )

    trend_short = sf(
        latest.get(
            "ema_20_50"
        )
    )

    trend_long = sf(
        latest.get(
            "ema_50_200"
        )
    )

    volume_ratio = sf(
        latest.get(
            "volume_ratio"
        ),
        1.0,
    )

    rsi = sf(
        latest.get(
            "rsi_14"
        ),
        50.0,
    )

    bb_position = sf(
        latest.get(
            "bb_position"
        ),
        0.5,
    )

    adx = sf(
        latest.get(
            "adx"
        ),
        0.0,
    )

    momentum_score = float(
        np.tanh(
            momentum_24 * 25
        )
    )

    short_trend_score = float(
        np.tanh(
            trend_short * 25
        )
    )

    long_trend_score = float(
        np.tanh(
            trend_long * 20
        )
    )

    volume_score = float(
        np.tanh(
            (volume_ratio - 1)
            / 2
        )
    )

    rsi_score = float(
        np.clip(
            (rsi - 50)
            / 25,
            -1,
            1,
        )
    )

    bb_score = float(
        np.clip(
            (bb_position - 0.5)
            * 2,
            -1,
            1,
        )
    )

    trend_strength = float(
        np.clip(
            adx / 50,
            0,
            1,
        )
    )

    technical_direction = float(
        np.clip(
            (
                momentum_score * 0.25
                + short_trend_score * 0.25
                + long_trend_score * 0.15
                + volume_score * 0.10
                + rsi_score * 0.10
                + bb_score * 0.10
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
    # BUY / SELL ranking scores
    # --------------------------------------------------------

    probability_direction = (
        p_buy - p_sell
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
    # Historical target rates
    # --------------------------------------------------------

    buy_target_rate = float(
        labeled[
            "buy_target"
        ].mean()
    )

    sell_target_rate = float(
        labeled[
            "sell_target"
        ].mean()
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

        "max_horizon_minutes": 30,

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

        "rsi14": rsi,

        "bb_position":
            bb_position,

        "atr_pct":
            sf(
                latest.get(
                    "atr_pct"
                )
            ),

        "adx": adx,

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
                df["timestamp"]
                .iloc[-1]
            ),
    }


# ============================================================
# SYMBOLS
# ============================================================

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
]


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if "errors" not in st.session_state:
    st.session_state.errors = []


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Spot Scalping AI Scanner"
)

st.caption(
    "15M • 30 Dakika • +%10 BUY / -%10 SELL • "
    "HOLD YOK • Emir göndermez"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)

provider = st.sidebar.selectbox(
    "Data Provider",
    [
        "Kraken REST",
        "Binance REST",
        "Historical CSV",
    ],
)

lookback = st.sidebar.number_input(
    "Historical Candles",
    min_value=500,
    max_value=MAX_LOOKBACK,
    value=DEFAULT_LOOKBACK,
    step=500,
)

train_limit = st.sidebar.number_input(
    "AI Training Candles",
    min_value=500,
    max_value=100000,
    value=20000,
    step=1000,
)

symbols_text = st.sidebar.text_area(
    "Symbols",
    value="\n".join(
        DEFAULT_SYMBOLS
    ),
    height=250,
)

symbols = [
    s.strip().upper()
    for s in symbols_text.splitlines()
    if s.strip()
]

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    "**BUY Target:** +10%"
)

st.sidebar.write(
    "**SELL Target:** -10%"
)

st.sidebar.write(
    "**Horizon:** 2 × 15m = 30 min"
)

st.sidebar.write(
    "**Signal:** BUY / SELL only"
)


# ============================================================
# CSV
# ============================================================

uploaded_file = None

if provider == "Historical CSV":

    uploaded_file = st.file_uploader(
        "📁 15M Historical CSV",
        type=[
            "csv",
        ],
        help=(
            "CSV içinde timestamp, open, high, "
            "low, close, volume kolonları bulunmalıdır."
        ),
    )


# ============================================================
# AUTO REFRESH
# ============================================================

auto_refresh = st.sidebar.checkbox(
    "🔄 Auto Refresh",
    value=False,
)

if auto_refresh:
    try:
        from streamlit_autorefresh import (
            st_autorefresh
        )

        st_autorefresh(
            interval=15 * 60 * 1000,
            key="scanner_refresh",
        )

    except ImportError:

        st.sidebar.warning(
            "Auto refresh için "
            "streamlit-autorefresh kurulmalı. "
            "Manuel tarama yine çalışır."
        )


# ============================================================
# RUN
# ============================================================

run_button = st.button(
    "🚀 SCAN NOW",
    type="primary",
    use_container_width=True,
)


# ============================================================
# DATA CACHE
# ============================================================

@st.cache_data(
    ttl=60 * 10,
    show_spinner=False,
)
def cached_binance_history(
    symbol,
    candles,
):
    return fetch_binance_history(
        symbol,
        candles,
    )


@st.cache_data(
    ttl=60 * 10,
    show_spinner=False,
)
def cached_kraken(
    symbol,
    candles,
):
    return fetch_kraken_ohlc(
        symbol,
        min(
            candles,
            720,
        ),
    )


# ============================================================
# SCAN FUNCTION
# ============================================================

def scan_symbol(
    symbol,
    provider_name,
    candles,
    training_limit,
    csv_df=None,
):
    started = time.time()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if provider_name == "Binance REST":

        df = cached_binance_history(
            symbol,
            candles,
        )

    elif provider_name == "Kraken REST":

        df = cached_kraken(
            symbol,
            candles,
        )

    else:

        if csv_df is None:
            raise DataProviderError(
                "CSV seçilmedi."
            )

        df = csv_df.copy()

    df = normalize_columns(
        df
    )

    # --------------------------------------------------------
    # Need enough candles
    # --------------------------------------------------------

    if len(df) < MIN_TRAIN_ROWS:
        raise DataProviderError(
            f"{symbol}: {len(df)} candle. "
            f"Minimum {MIN_TRAIN_ROWS}."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    result = train_predict(
        df,
        training_limit,
    )

    result["symbol"] = symbol

    result["candles"] = len(df)

    result["processing_seconds"] = (
        time.time()
        - started
    )

    return result


# ============================================================
# EXECUTE SCAN
# ============================================================

if run_button:

    errors = []

    results = []

    csv_df = None

    if provider == "Historical CSV":

        if uploaded_file is None:

            st.error(
                "Önce CSV yüklemelisiniz."
            )

            st.stop()

        try:

            csv_df = load_csv_data(
                uploaded_file
            )

            st.success(
                f"CSV yüklendi: "
                f"{len(csv_df):,} candles"
            )

        except Exception as e:

            st.error(
                f"CSV hatası: {e}"
            )

            st.stop()

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(symbols)

    if total == 0:

        st.error(
            "En az bir symbol girin."
        )

        st.stop()

    for i, symbol in enumerate(
        symbols
    ):

        status.write(
            f"⏳ {symbol} "
            f"({i + 1}/{total})"
        )

        try:

            result = scan_symbol(
                symbol,
                provider,
                lookback,
                train_limit,
                csv_df,
            )

            results.append(
                result
            )

        except Exception as e:

            errors.append(
                {
                    "symbol": symbol,
                    "error": str(e),
                }
            )

        progress.progress(
            (i + 1) / total
        )

    status.empty()

    progress.empty()

    if results:

        result_df = pd.DataFrame(
            results
        )

        result_df = result_df[
            result_df["signal"].isin(
                [
                    "BUY",
                    "SELL",
                ]
            )
        ]

        st.session_state.results = (
            result_df
        )

    else:

        st.session_state.results = (
            pd.DataFrame()
        )

    st.session_state.errors = (
        errors
    )

    st.session_state.last_run = (
        datetime.now(
            timezone.utc
        )
    )

    st.rerun()


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.results
)

if results.empty:

    st.info(
        "Henüz tarama yapılmadı. "
        "SCAN NOW butonuna basın."
    )

else:

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

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

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:
        st.metric(
            "Symbols",
            len(results),
        )

    with c2:
        st.metric(
            "BUY",
            buy_count,
        )

    with c3:
        st.metric(
            "SELL",
            sell_count,
        )

    with c4:

        if st.session_state.last_run:
            st.metric(
                "Last Scan",
                st.session_state.last_run.strftime(
                    "%H:%M:%S"
                ),
            )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buy = (
        results[
            results["signal"]
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

    st.caption(
        "+10% hedefini maksimum "
        "30 dakika içinde yakalama "
        "yönünde en güçlü AI sinyalleri."
    )

    if buy.empty:

        st.info(
            "BUY sonucu yok."
        )

    else:

        buy_display = buy[
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

        buy_display[
            "confidence"
        ] *= 100

        buy_display[
            "p_buy"
        ] *= 100

        buy_display[
            "p_sell"
        ] *= 100

        buy_display[
            "projected_upside"
        ] *= 100

        buy_display[
            "test_balanced_accuracy"
        ] *= 100

        buy_display = (
            buy_display.rename(
                columns={
                    "symbol": "Symbol",
                    "signal": "Signal",
                    "ai_score": "AI Score",
                    "confidence": "Confidence %",
                    "p_buy": "BUY Prob %",
                    "p_sell": "SELL Prob %",
                    "projected_upside": "Projected +%",
                    "rsi14": "RSI",
                    "adx": "ADX",
                    "volume_ratio": "Volume Ratio",
                    "price": "Price",
                    "test_balanced_accuracy": "Test Bal.Acc %",
                    "candles": "Candles",
                }
            )
        )

        st.dataframe(
            buy_display.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    sell = (
        results[
            results["signal"]
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

    st.caption(
        "-10% hedefini maksimum "
        "30 dakika içinde yakalama "
        "yönünde en güçlü AI sinyalleri."
    )

    if sell.empty:

        st.info(
            "SELL sonucu yok."
        )

    else:

        sell_display = sell[
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

        sell_display[
            "confidence"
        ] *= 100

        sell_display[
            "p_buy"
        ] *= 100

        sell_display[
            "p_sell"
        ] *= 100

        sell_display[
            "projected_downside"
        ] *= 100

        sell_display[
            "test_balanced_accuracy"
        ] *= 100

        sell_display = (
            sell_display.rename(
                columns={
                    "symbol": "Symbol",
                    "signal": "Signal",
                    "ai_score": "AI Score",
                    "confidence": "Confidence %",
                    "p_buy": "BUY Prob %",
                    "p_sell": "SELL Prob %",
                    "projected_downside": "Projected -%",
                    "rsi14": "RSI",
                    "adx": "ADX",
                    "volume_ratio": "Volume Ratio",
                    "price": "Price",
                    "test_balanced_accuracy": "Test Bal.Acc %",
                    "candles": "Candles",
                }
            )
        )

        st.dataframe(
            sell_display.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "📋 ALL BUY / SELL"
    )

    st.dataframe(
        results.sort_values(
            "ai_score",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # SYMBOL DETAIL
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
        results["symbol"]
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

                "Historical BUY Target Rate",
                "Historical SELL Target Rate",

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
    # CSV DOWNLOAD
    # --------------------------------------------------------

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
        "⬇️ Download AI Results CSV",
        data=csv_output,
        file_name=(
            "spot_scalping_ai_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# ERRORS
# ============================================================

errors = (
    st.session_state.errors
)

if errors:

    st.markdown(
        "---"
    )

    st.subheader(
        "⚠️ Symbol Errors"
    )

    error_df = pd.DataFrame(
        errors
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
### Model

AI modeli 15 dakikalık candle'lar üzerinde çalışır.

Her candle için:

- sonraki 15 dakika
- sonraki 30 dakika

kontrol edilir.

### BUY

Giriş fiyatından sonraki maksimum 30 dakika içinde:

**High >= Entry × 1.10**

olursa BUY target oluşur.

### SELL

Giriş fiyatından sonraki maksimum 30 dakika içinde:

**Low <= Entry × 0.90**

olursa SELL target oluşur.

### HOLD

**HOLD sınıfı yoktur.**

Model her zaman:

**BUY veya SELL**

seçer.

### AI

Model:

- EMA
- RSI
- MACD
- Bollinger Bands
- ATR
- ADX
- Stochastic
- Volume
- Volatility
- Candle structure
- Price distance

gibi feature'ları kullanır.

Training kronolojik olarak yapılır:

**70% Training → 15% Validation → 15% Test**

Test accuracy ve balanced accuracy ayrıca gösterilir.

### Önemli

+%10 hareketin 30 dakika içinde gerçekleşmesi çok agresif
bir scalping hedefidir. AI sonucu kesin kâr veya kesin fiyat
tahmini değildir.

Bu uygulama **emir göndermez**.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Spot Scalping AI Scanner • "
    "15M • 30 Minute Horizon • "
    "+10% BUY / -10% SELL • "
    "BUY / SELL Only • "
    "No Order Execution"
)
