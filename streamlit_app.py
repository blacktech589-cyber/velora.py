import io
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Spot AI Scanner"

TIMEFRAME = "15m"

KRAKEN_URL = "https://api.kraken.com"
BINANCE_URL = "https://api.binance.com"

REQUEST_TIMEOUT = 30

# Kraken public OHLC yaklaşık 721 candle döndürebilir.
# Bu yüzden artık 1000 istemiyoruz.
MIN_CANDLES = 300

SUPPORTED_QUOTES = [
    "USD",
    "USDT",
    "EUR",
    "USDC",
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)

st.title("📊 Spot AI Scanner")

st.caption(
    "15m | 500K Target | MLP AI | No Order Execution"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "results": pd.DataFrame(),
    "errors": [],
    "cache": {},
    "last_scan": None,
    "last_source": None,
    "status": "READY",
    "force_scan": False,
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent": "Spot-AI-Scanner/1.0",
        "Accept": "application/json",
    })

    return session


HTTP = get_http_session()


# ============================================================
# EXCEPTIONS
# ============================================================

class DataProviderError(Exception):
    pass


class BinanceRestrictedError(Exception):
    pass


# ============================================================
# HELPERS
# ============================================================

def now_utc():

    return datetime.now(timezone.utc)


def to_float(value, default=np.nan):

    try:
        return float(value)

    except Exception:

        return default


def clean_ohlcv(df):

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

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise DataProviderError(
            "Eksik OHLCV kolonları: "
            + ", ".join(missing)
        )

    x = df.copy()

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
        errors="coerce",
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    x = x.dropna(
        subset=required
    )

    x = (
        x
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return x


# ============================================================
# KRAKEN API
# ============================================================

def kraken_get(endpoint, params=None):

    try:

        response = HTTP.get(
            KRAKEN_URL + endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise DataProviderError(
            f"Kraken bağlantı hatası: {exc}"
        )

    if response.status_code >= 400:

        raise DataProviderError(
            f"Kraken HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:

        data = response.json()

    except Exception as exc:

        raise DataProviderError(
            f"Kraken JSON hatası: {exc}"
        )

    errors = data.get("error", [])

    if errors:

        raise DataProviderError(
            "Kraken API: " + str(errors)
        )

    return data.get("result", {})


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_kraken_symbols():

    result = kraken_get(
        "/0/public/AssetPairs"
    )

    symbols = []

    for key, item in result.items():

        if not isinstance(item, dict):
            continue

        status = item.get(
            "status",
            "online",
        )

        if status != "online":
            continue

        altname = str(
            item.get("altname", key)
        )

        wsname = str(
            item.get("wsname", altname)
        )

        quote = str(
            item.get("quote", "")
        ).upper()

        quote_clean = (
            quote
            .replace("X", "")
            .replace("Z", "")
        )

        ws_upper = wsname.upper()

        valid_quote = (
            quote_clean in SUPPORTED_QUOTES
            or ws_upper.endswith("/USD")
            or ws_upper.endswith("/USDT")
            or ws_upper.endswith("/EUR")
            or ws_upper.endswith("/USDC")
        )

        if not valid_quote:
            continue

        if ".D" in altname.upper():
            continue

        symbols.append({
            "api": altname,
            "display": wsname,
        })

    symbols.sort(
        key=lambda x: x["display"]
    )

    return symbols


def get_kraken_ohlc(symbol):

    result = kraken_get(
        "/0/public/OHLC",
        {
            "pair": symbol,
            "interval": 15,
        },
    )

    pair_key = None

    for key in result.keys():

        if key != "last":

            pair_key = key
            break

    if pair_key is None:

        return pd.DataFrame()

    rows = []

    for row in result[pair_key]:

        rows.append({
            "timestamp": pd.to_datetime(
                int(row[0]),
                unit="s",
                utc=True,
            ),
            "open": to_float(row[1]),
            "high": to_float(row[2]),
            "low": to_float(row[3]),
            "close": to_float(row[4]),
            "volume": to_float(row[6]),
        })

    return clean_ohlcv(
        pd.DataFrame(rows)
    )


# ============================================================
# BINANCE API
# ============================================================

def binance_get(endpoint, params=None):

    try:

        response = HTTP.get(
            BINANCE_URL + endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise DataProviderError(
            f"Binance bağlantı hatası: {exc}"
        )

    if response.status_code == 451:

        raise BinanceRestrictedError(
            "Binance HTTP 451: "
            "Bu çalışma ortamından Binance API "
            "erişimi kısıtlanmış."
        )

    if response.status_code >= 400:

        raise DataProviderError(
            f"Binance HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:

        return response.json()

    except Exception as exc:

        raise DataProviderError(
            f"Binance JSON hatası: {exc}"
        )


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_binance_symbols():

    data = binance_get(
        "/api/v3/exchangeInfo"
    )

    symbols = []

    for item in data.get(
        "symbols",
        [],
    ):

        if item.get("status") != "TRADING":
            continue

        if (
            item.get("isSpotTradingAllowed")
            is False
        ):
            continue

        if item.get(
            "quoteAsset"
        ) not in [
            "USDT",
            "USDC",
        ]:
            continue

        symbols.append(
            item["symbol"]
        )

    return sorted(symbols)


def get_binance_history(
    symbol,
    target,
):

    frames = []

    end_time = None

    total = 0

    max_requests = math.ceil(
        target / 1000
    )

    for _ in range(max_requests):

        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": 1000,
        }

        if end_time is not None:

            params["endTime"] = int(
                end_time
            )

        data = binance_get(
            "/api/v3/klines",
            params,
        )

        if not data:
            break

        rows = []

        for row in data:

            rows.append({
                "timestamp": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": to_float(row[1]),
                "high": to_float(row[2]),
                "low": to_float(row[3]),
                "close": to_float(row[4]),
                "volume": to_float(row[5]),
            })

        chunk = clean_ohlcv(
            pd.DataFrame(rows)
        )

        if chunk.empty:
            break

        frames.append(chunk)

        total += len(chunk)

        oldest = chunk[
            "timestamp"
        ].min()

        end_time = (
            int(oldest.timestamp() * 1000)
            - 1
        )

        if (
            len(chunk) < 1000
            or total >= target
        ):
            break

    if not frames:

        return pd.DataFrame()

    result = clean_ohlcv(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    return (
        result
        .tail(target)
        .reset_index(drop=True)
    )


# ============================================================
# CSV
# ============================================================

def load_csv(upload):

    raw = upload.read()

    try:

        df = pd.read_csv(
            io.BytesIO(raw)
        )

    except Exception as exc:

        raise DataProviderError(
            f"CSV okunamadı: {exc}"
        )

    normalized = {}

    for col in df.columns:

        normalized[
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
        ] = col

    aliases = {

        "timestamp": [
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time",
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

    return clean_ohlcv(df)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def EMA(series, period):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def RSI(series, period=14):

    delta = series.diff()

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

    rs = avg_gain / (
        avg_loss + 1e-12
    )

    return (
        100
        - 100 / (1 + rs)
    )


def ATR(df, period=14):

    previous_close = (
        df["close"].shift(1)
    )

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
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def MACD(series):

    fast = EMA(
        series,
        12,
    )

    slow = EMA(
        series,
        26,
    )

    line = fast - slow

    signal = EMA(
        line,
        9,
    )

    histogram = (
        line - signal
    )

    return (
        line,
        signal,
        histogram,
    )


def Bollinger(
    series,
    period=20,
):

    middle = (
        series
        .rolling(period)
        .mean()
    )

    std = (
        series
        .rolling(period)
        .std()
    )

    upper = middle + 2 * std
    lower = middle - 2 * std

    return (
        middle,
        upper,
        lower,
    )


def ADX(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()

    down = -low.diff()

    plus_dm = np.where(
        (up > down) & (up > 0),
        up,
        0,
    )

    minus_dm = np.where(
        (down > up) & (down > 0),
        down,
        0,
    )

    previous_close = (
        close.shift(1)
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
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_value = (
        tr
        .rolling(period)
        .mean()
    )

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index,
        )
        .rolling(period)
        .mean()
        / (atr_value + 1e-12)
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index,
        )
        .rolling(period)
        .mean()
        / (atr_value + 1e-12)
    )

    dx = (
        100
        * (
            plus_di - minus_di
        ).abs()
        / (
            plus_di
            + minus_di
            + 1e-12
        )
    )

    return (
        dx
        .rolling(period)
        .mean()
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

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    for n in [
        1,
        3,
        6,
        12,
        24,
        48,
        96,
    ]:

        x[f"ret_{n}"] = (
            close.pct_change(n)
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for n in [
        5,
        20,
        50,
        100,
        200,
    ]:

        e = EMA(
            close,
            n,
        )

        x[f"ema_dist_{n}"] = (
            close / (
                e + 1e-12
            )
            - 1
        )

    e5 = EMA(
        close,
        5,
    )

    e20 = EMA(
        close,
        20,
    )

    e50 = EMA(
        close,
        50,
    )

    e200 = EMA(
        close,
        200,
    )

    x["ema_5_20"] = (
        e5 / (
            e20 + 1e-12
        )
        - 1
    )

    x["ema_20_50"] = (
        e20 / (
            e50 + 1e-12
        )
        - 1
    )

    x["ema_50_200"] = (
        e50 / (
            e200 + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    for n in [
        7,
        14,
        21,
    ]:

        x[f"rsi_{n}"] = RSI(
            close,
            n,
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_line, macd_signal, macd_hist = MACD(
        close
    )

    x["macd_norm"] = (
        macd_line
        / (close + 1e-12)
    )

    x["macd_signal"] = (
        macd_signal
    )

    x["macd_hist"] = (
        macd_hist
    )

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    middle, upper, lower = Bollinger(
        close,
        20,
    )

    x["bb_width"] = (
        upper - lower
    ) / (
        middle + 1e-12
    )

    x["bb_position"] = (
        close - lower
    ) / (
        upper - lower
        + 1e-12
    )

    # --------------------------------------------------------
    # ATR / ADX
    # --------------------------------------------------------

    x["atr_pct"] = (
        ATR(x)
        / (close + 1e-12)
    )

    x["adx"] = ADX(x)

    # --------------------------------------------------------
    # Stochastic
    # --------------------------------------------------------

    lowest = (
        low
        .rolling(14)
        .min()
    )

    highest = (
        high
        .rolling(14)
        .max()
    )

    x["stoch_k"] = (
        100
        * (
            close - lowest
        )
        / (
            highest
            - lowest
            + 1e-12
        )
    )

    x["stoch_d"] = (
        x["stoch_k"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    body = (
        close - x["open"]
    )

    x["body_pct"] = (
        body
        / (close + 1e-12)
    )

    x["range_pct"] = (
        candle_range
        / (close + 1e-12)
    )

    x["upper_wick"] = (
        high
        - np.maximum(
            x["open"],
            close,
        )
    )

    x["lower_wick"] = (
        np.minimum(
            x["open"],
            close,
        )
        - low
    )

    x["body_to_range"] = (
        body.abs()
        / (
            candle_range
            + 1e-12
        )
    )

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

    x["volume_change"] = (
        volume.pct_change()
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    for n in [
        12,
        24,
        48,
        96,
    ]:

        x[f"volatility_{n}"] = (
            x["ret_1"]
            .rolling(n)
            .std()
        )

    # --------------------------------------------------------
    # High / Low distances
    # --------------------------------------------------------

    for n in [
        48,
        200,
    ]:

        rolling_high = (
            high
            .rolling(n)
            .max()
        )

        rolling_low = (
            low
            .rolling(n)
            .min()
        )

        x[
            f"distance_high_{n}"
        ] = (
            close
            / (
                rolling_high
                + 1e-12
            )
            - 1
        )

        x[
            f"distance_low_{n}"
        ] = (
            close
            / (
                rolling_low
                + 1e-12
            )
            - 1
        )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_high = (
        close.cummax()
    )

    x["drawdown"] = (
        close
        / (
            running_high
            + 1e-12
        )
        - 1
    )

    # --------------------------------------------------------
    # Price Z score
    # --------------------------------------------------------

    for n in [
        50,
        200,
    ]:

        mean = (
            close
            .rolling(n)
            .mean()
        )

        std = (
            close
            .rolling(n)
            .std()
        )

        x[
            f"price_z_{n}"
        ] = (
            close - mean
        ) / (
            std + 1e-12
        )

    return x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# FEATURE LIST
# ============================================================

def feature_columns():

    return [

        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_48",
        "ret_96",

        "ema_dist_5",
        "ema_dist_20",
        "ema_dist_50",
        "ema_dist_100",
        "ema_dist_200",

        "ema_5_20",
        "ema_20_50",
        "ema_50_200",

        "rsi_7",
        "rsi_14",
        "rsi_21",

        "macd_norm",
        "macd_signal",
        "macd_hist",

        "bb_width",
        "bb_position",

        "atr_pct",
        "adx",

        "stoch_k",
        "stoch_d",

        "body_pct",
        "range_pct",
        "upper_wick",
        "lower_wick",
        "body_to_range",

        "volume_ratio",
        "volume_z",
        "volume_change",

        "volatility_12",
        "volatility_24",
        "volatility_48",
        "volatility_96",

        "distance_high_48",
        "distance_low_48",

        "distance_high_200",
        "distance_low_200",

        "drawdown",

        "price_z_50",
        "price_z_200",
    ]


# ============================================================
# TARGET
# ============================================================

def create_target(
    df,
    future_bars,
):

    x = df.copy()

    future_return = (
        x["close"]
        .shift(-future_bars)
        / x["close"]
        - 1
    )

    x["future_return"] = (
        future_return
    )

    volatility = (
        x["volatility_24"]
        .fillna(
            x["ret_1"].std()
        )
    )

    threshold = (
        volatility
        * np.sqrt(future_bars)
    ).clip(
        lower=0.002
    )

    x["target"] = 1

    x.loc[
        future_return > threshold,
        "target",
    ] = 2

    x.loc[
        future_return < -threshold,
        "target",
    ] = 0

    return x


# ============================================================
# AI MODELS
# ============================================================

def create_classifier():

    return Pipeline([
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

                batch_size=128,

                learning_rate_init=0.001,

                max_iter=120,

                early_stopping=True,

                validation_fraction=0.15,

                random_state=42,
            ),
        ),
    ])


def create_regressor():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),

        (
            "ridge",
            Ridge(
                alpha=1.0
            ),
        ),
    ])


# ============================================================
# TRAIN / PREDICT
# ============================================================

def train_and_predict(
    df,
    future_bars,
    train_limit,
):

    features = build_features(
        df
    )

    features = create_target(
        features,
        future_bars,
    )

    cols = feature_columns()

    usable = (
        features
        .dropna(
            subset=cols
            + [
                "target",
                "future_return",
            ]
        )
        .copy()
    )

    if len(usable) < MIN_CANDLES:

        raise DataProviderError(
            f"Model için yeterli temiz "
            f"veri yok: {len(usable)}"
        )

    usable = usable.tail(
        min(
            train_limit,
            len(usable),
        )
    )

    X = usable[
        cols
    ].astype(float)

    y = usable[
        "target"
    ].astype(int)

    future_y = usable[
        "future_return"
    ].astype(float)

    if y.nunique() < 2:

        raise DataProviderError(
            "AI target yalnızca "
            "tek sınıf içeriyor."
        )

    classifier = (
        create_classifier()
    )

    regressor = (
        create_regressor()
    )

    classifier.fit(
        X,
        y,
    )

    regressor.fit(
        X,
        future_y,
    )

    train_prediction = (
        classifier.predict(X)
    )

    training_accuracy = (
        accuracy_score(
            y,
            train_prediction,
        )
    )

    latest = (
        features
        .dropna(
            subset=cols
        )
        .iloc[-1]
    )

    X_latest = pd.DataFrame([
        latest[
            cols
        ].astype(float)
    ])

    probabilities = (
        classifier
        .predict_proba(
            X_latest
        )[0]
    )

    classes = (
        classifier.classes_
    )

    probability_map = {
        int(c): float(p)
        for c, p in zip(
            classes,
            probabilities,
        )
    }

    prediction = int(
        classifier.predict(
            X_latest
        )[0]
    )

    buy_probability = (
        probability_map.get(
            2,
            0.0,
        )
    )

    hold_probability = (
        probability_map.get(
            1,
            0.0,
        )
    )

    sell_probability = (
        probability_map.get(
            0,
            0.0,
        )
    )

    if prediction == 2:

        signal = "BUY"

        confidence = (
            buy_probability
        )

    elif prediction == 0:

        signal = "SELL"

        confidence = (
            sell_probability
        )

    else:

        signal = "HOLD"

        confidence = (
            hold_probability
        )

    expected_return = float(
        regressor.predict(
            X_latest
        )[0]
    )

    momentum = to_float(
        latest.get("ret_24")
    )

    trend = to_float(
        latest.get("ema_20_50")
    )

    volume_ratio = to_float(
        latest.get(
            "volume_ratio"
        )
    )

    volatility = to_float(
        latest.get(
            "volatility_24"
        )
    )

    direction = (
        buy_probability
        - sell_probability
    )

    if np.isnan(momentum):

        momentum_score = 0

    else:

        momentum_score = np.tanh(
            momentum * 20
        )

    if np.isnan(trend):

        trend_score = 0

    else:

        trend_score = np.tanh(
            trend * 20
        )

    if np.isnan(
        volume_ratio
    ):

        volume_score = 0

    else:

        volume_score = np.tanh(
            (volume_ratio - 1)
            / 2
        )

    if np.isnan(
        volatility
    ):

        risk_score = 0

    else:

        risk_score = (
            1
            - np.clip(
                volatility * 30,
                0,
                1,
            )
        )

    score = float(
        np.clip(
            50
            + direction * 25
            + momentum_score * 10
            + trend_score * 7
            + volume_score * 4
            + risk_score * 4,
            0,
            100,
        )
    )

    return {

        "signal": signal,

        "confidence": confidence,

        "p_buy": buy_probability,

        "p_hold": hold_probability,

        "p_sell": sell_probability,

        "expected_return":
            expected_return,

        "price":
            float(
                latest["close"]
            ),

        "score":
            score,

        "rsi14":
            to_float(
                latest.get(
                    "rsi_14"
                )
            ),

        "bb_position":
            to_float(
                latest.get(
                    "bb_position"
                )
            ),

        "atr_pct":
            to_float(
                latest.get(
                    "atr_pct"
                )
            ),

        "adx":
            to_float(
                latest.get(
                    "adx"
                )
            ),

        "volume_ratio":
            volume_ratio,

        "momentum":
            momentum,

        "trend":
            trend,

        "volatility":
            volatility,

        "training_accuracy":
            training_accuracy,
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)


source = st.sidebar.selectbox(

    "Data Source",

    [
        "Kraken REST",
        "CSV / Uploaded Data",
        "Binance REST",
    ],
)


uploaded_file = None

if source == "CSV / Uploaded Data":

    uploaded_file = (
        st.sidebar.file_uploader(
            "OHLCV CSV",
            type=["csv"],
        )
    )

    st.sidebar.caption(
        "Kolonlar: timestamp, "
        "open, high, low, close, volume"
    )


target = st.sidebar.selectbox(

    "Candle Target",

    [
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
    ],

    index=4,
)


symbol_limit = st.sidebar.slider(

    "Maximum Symbols",

    min_value=1,

    max_value=100,

    value=10,
)


future_bars = st.sidebar.slider(

    "Future Bars",

    min_value=3,

    max_value=48,

    value=12,
)


train_limit = st.sidebar.slider(

    "Training Rows",

    min_value=500,

    max_value=30_000,

    value=5_000,

    step=500,
)


minimum_confidence = (
    st.sidebar.slider(

        "Minimum AI Confidence",

        min_value=0.30,

        max_value=0.95,

        value=0.55,

        step=0.01,
    )
)


auto_refresh = st.sidebar.checkbox(

    "15 Dakikada Otomatik Tarama",

    value=True,
)


if st.sidebar.button(
    "🔄 Şimdi Tara",
    use_container_width=True,
):

    st.session_state.force_scan = True


# ============================================================
# DISCOVER SYMBOLS
# ============================================================

def discover_symbols():

    if source == "Kraken REST":

        symbols = (
            get_kraken_symbols()
        )

        if not symbols:

            raise DataProviderError(
                "Kraken Spot sembol bulunamadı."
            )

        return symbols[
            :symbol_limit
        ]

    if source == "Binance REST":

        symbols = (
            get_binance_symbols()
        )

        return symbols[
            :symbol_limit
        ]

    if uploaded_file is None:

        raise DataProviderError(
            "CSV dosyası yüklenmedi."
        )

    return [
        {
            "api": "CSV",
            "display":
                "UPLOADED_DATA",
        }
    ]


# ============================================================
# LOAD DATA
# ============================================================

def load_symbol(pair):

    if source == "Kraken REST":

        api_symbol = pair["api"]

        display_symbol = (
            pair["display"]
        )

        cache_key = (
            "KRAKEN|"
            + api_symbol
        )

        if (
            cache_key
            in st.session_state.cache
        ):

            return (
                display_symbol,
                st.session_state.cache[
                    cache_key
                ].tail(target),
            )

        df = get_kraken_ohlc(
            api_symbol
        )

        if not df.empty:

            st.session_state.cache[
                cache_key
            ] = df.tail(target)

        return (
            display_symbol,
            df,
        )

    if source == "Binance REST":

        symbol = pair

        cache_key = (
            "BINANCE|"
            + symbol
        )

        if (
            cache_key
            in st.session_state.cache
        ):

            return (
                symbol,
                st.session_state.cache[
                    cache_key
                ].tail(target),
            )

        df = get_binance_history(
            symbol,
            target,
        )

        if not df.empty:

            st.session_state.cache[
                cache_key
            ] = df

        return (
            symbol,
            df,
        )

    df = load_csv(
        uploaded_file
    )

    return (
        "UPLOADED_DATA",
        df.tail(target),
    )


# ============================================================
# SCAN
# ============================================================

def run_scan():

    st.session_state.errors = []

    st.session_state.status = (
        "SCANNING"
    )

    results = []

    progress = st.progress(0)

    status_box = st.empty()

    try:

        status_box.info(
            "Spot sembolleri alınıyor..."
        )

        symbols = (
            discover_symbols()
        )

        total = len(symbols)

        for index, pair in enumerate(
            symbols
        ):

            if isinstance(
                pair,
                dict,
            ):

                label = pair[
                    "display"
                ]

            else:

                label = str(pair)

            status_box.info(
                f"Tarama: {label} "
                f"({index + 1}/{total})"
            )

            try:

                symbol, df = (
                    load_symbol(pair)
                )

                # ------------------------------------------------
                # IMPORTANT FIX
                # ------------------------------------------------

                if len(df) < MIN_CANDLES:

                    raise DataProviderError(
                        f"Yetersiz candle: "
                        f"{len(df)}. "
                        f"En az "
                        f"{MIN_CANDLES} "
                        f"candle gerekli."
                    )

                start = time.time()

                prediction = (
                    train_and_predict(
                        df,
                        future_bars,
                        train_limit,
                    )
                )

                results.append({

                    "symbol":
                        symbol,

                    **prediction,

                    "candles":
                        len(df),

                    "last_candle":
                        df[
                            "timestamp"
                        ].iloc[-1],

                    "processing_seconds":
                        time.time()
                        - start,
                })

            except Exception as exc:

                st.session_state.errors.append({

                    "symbol":
                        label,

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                })

            progress.progress(
                int(
                    (
                        index + 1
                    )
                    / total
                    * 100
                )
            )

        result_df = (
            pd.DataFrame(results)
        )

        if not result_df.empty:

            result_df = (
                result_df
                .sort_values(
                    "score",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

        st.session_state.results = (
            result_df
        )

        st.session_state.last_scan = (
            now_utc()
        )

        st.session_state.last_source = (
            source
        )

        if result_df.empty:

            st.session_state.status = (
                "NO_RESULTS"
            )

        else:

            st.session_state.status = (
                "SUCCESS"
            )

    except BinanceRestrictedError as exc:

        st.session_state.errors.append({

            "symbol":
                "BINANCE",

            "error":
                str(exc),
        })

        st.session_state.status = (
            "BINANCE_451"
        )

        st.error(
            str(exc)
        )

        st.info(
            "Binance erişimi bu "
            "ortamda kısıtlı. "
            "Kraken REST veya CSV "
            "kullanabilirsiniz."
        )

    except Exception as exc:

        st.session_state.errors.append({

            "symbol":
                "SYSTEM",

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        })

        st.session_state.status = (
            "ERROR"
        )

        st.error(
            f"Tarama hatası: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    finally:

        progress.empty()

        status_box.empty()

        st.session_state.force_scan = (
            False
        )


# ============================================================
# AUTO SCANNER
# ============================================================

if hasattr(
    st,
    "fragment",
):

    @st.fragment(
        run_every=(
            "15m"
            if auto_refresh
            else None
        )
    )
    def scanner_fragment():

        if (
            st.session_state.force_scan
            or auto_refresh
        ):

            run_scan()


    scanner_fragment()

else:

    if (
        st.session_state.force_scan
        or auto_refresh
    ):

        run_scan()


# ============================================================
# STATUS
# ============================================================

st.divider()

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Status",
        st.session_state.status,
    )


with col2:

    st.metric(
        "Source",
        (
            st.session_state.last_source
            or source
        ),
    )


with col3:

    st.metric(
        "Target",
        f"{target:,}",
    )


with col4:

    last_scan = (
        st.session_state.last_scan
    )

    st.metric(
        "Last Scan",
        (
            last_scan.strftime(
                "%H:%M:%S"
            )
            if last_scan
            else "-"
        ),
    )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.results
)


if results.empty:

    st.warning(
        "Henüz sonuç yok."
    )

    if source == "Kraken REST":

        st.info(
            "Kraken public OHLC "
            "yaklaşık 721 adet "
            "15m candle sağlayabilir. "
            "Bu sürüm 721 candle "
            "ile çalışabilir."
        )

else:

    buy = (
        results[
            (
                results["signal"]
                == "BUY"
            )
            & (
                results[
                    "confidence"
                ]
                >= minimum_confidence
            )
        ]
        .sort_values(
            [
                "score",
                "confidence",
            ],
            ascending=False,
        )
        .head(10)
    )

    sell = (
        results[
            (
                results["signal"]
                == "SELL"
            )
            & (
                results[
                    "confidence"
                ]
                >= minimum_confidence
            )
        ]
        .sort_values(
            [
                "score",
                "confidence",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(10)
    )

    hold = (
        results[
            results["signal"]
            == "HOLD"
        ]
        .sort_values(
            "score",
            ascending=False,
        )
        .head(10)
    )

    def show_table(df):

        if df.empty:

            st.info(
                "Sonuç yok."
            )

            return

        view = df.copy()

        view[
            "confidence"
        ] *= 100

        view[
            "expected_return"
        ] *= 100

        view[
            "p_buy"
        ] *= 100

        view[
            "p_hold"
        ] *= 100

        view[
            "p_sell"
        ] *= 100

        columns = [

            "symbol",

            "signal",

            "score",

            "confidence",

            "expected_return",

            "p_buy",

            "p_hold",

            "p_sell",

            "rsi14",

            "adx",

            "volume_ratio",

            "price",

            "candles",
        ]

        st.dataframe(

            view[
                columns
            ].round(4),

            use_container_width=True,

            hide_index=True,
        )


    st.subheader(
        "🟢 TOP 10 BUY"
    )

    show_table(
        buy
    )


    st.subheader(
        "🔴 TOP 10 SELL"
    )

    show_table(
        sell
    )


    st.subheader(
        "🟡 TOP 10 HOLD"
    )

    show_table(
        hold
    )


    with st.expander(
        "📋 Tüm Sonuçlar"
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
        "🔎 Symbol Detail"
    )

    selected_symbol = (
        st.selectbox(
            "Symbol",
            results[
                "symbol"
            ].tolist(),
        )
    )

    row = (
        results[
            results[
                "symbol"
            ]
            == selected_symbol
        ]
        .iloc[0]
    )


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
            (
                f"{row['confidence'] * 100:.2f}%"
            ),
        )


    with c:

        st.metric(
            "Expected Return",
            (
                f"{row['expected_return'] * 100:.3f}%"
            ),
        )


    with d:

        st.metric(
            "AI Score",
            (
                f"{row['score']:.2f}/100"
            ),
        )


    detail = pd.DataFrame({

        "Metric": [

            "Price",

            "BUY Probability",

            "HOLD Probability",

            "SELL Probability",

            "RSI 14",

            "Bollinger Position",

            "ATR %",

            "ADX",

            "Volume Ratio",

            "Training Accuracy",

            "Candles",

            "Last Candle",

            "Processing Seconds",
        ],

        "Value": [

            row["price"],

            f(
                row["p_buy"] * 100,
                ".2f"
            )
            + "%",

            f(
                row["p_hold"] * 100,
                ".2f"
            )
            + "%",

            f(
                row["p_sell"] * 100,
                ".2f"
            )
            + "%",

            row["rsi14"],

            row["bb_position"],

            row["atr_pct"],

            row["adx"],

            row["volume_ratio"],

            f(
                row[
                    "training_accuracy"
                ]
                * 100,
                ".2f"
            )
            + "%",

            row["candles"],

            row["last_candle"],

            f(
                row[
                    "processing_seconds"
                ],
                ".2f"
            ),
        ],
    })


    st.dataframe(

        detail,

        use_container_width=True,

        hide_index=True,
    )


    st.download_button(

        "⬇️ Download AI Results CSV",

        data=(
            results
            .to_csv(
                index=False
            )
            .encode("utf-8")
        ),

        file_name=(
            "spot_ai_results.csv"
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
        "⚠️ Hata Detayları"
    )

    for item in (
        st.session_state.errors
    ):

        st.error(
            f"{item.get('symbol', 'SYSTEM')}: "
            f"{item.get('error', 'Bilinmeyen hata')}"
        )


# ============================================================
# CACHE
# ============================================================

with st.expander(
    "🗄️ Candle Cache"
):

    if not st.session_state.cache:

        st.info(
            "Candle cache boş."
        )

    else:

        cache_rows = []

        for key, df in (
            st.session_state.cache.items()
        ):

            if df.empty:
                continue

            cache_rows.append({

                "source":
                    key,

                "candles":
                    len(df),

                "first":
                    str(
                        df[
                            "timestamp"
                        ].iloc[0]
                    ),

                "last":
                    str(
                        df[
                            "timestamp"
                        ].iloc[-1]
                    ),
            })

        if cache_rows:

            st.dataframe(

                pd.DataFrame(
                    cache_rows
                ),

                use_container_width=True,

                hide_index=True,
            )

        else:

            st.info(
                "Cache boş."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Scanner | "
    "15m | "
    "500K Target | "
    "MLP AI | "
    "No Order Execution"
)

st.caption(
    "Bu uygulama yalnızca OHLCV "
    "verilerinden istatistiksel "
    "BUY/HOLD/SELL sinyali üretir. "
    "Emir göndermez ve yatırım "
    "tavsiyesi değildir."
)
