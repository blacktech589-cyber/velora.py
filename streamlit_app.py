
import io
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# APP CONFIG
# ============================================================

APP_TITLE = "Spot AI Scanner PRO"

KRAKEN_URL = "https://api.kraken.com"
BINANCE_URL = "https://api.binance.com"

REQUEST_TIMEOUT = 30

TIMEFRAME = "15m"

# Scalping target: price must move at least 10% within the next
# two 15-minute candles (maximum 30 minutes).
SCALP_HORIZON_BARS = 2
SCALP_TARGET_RETURN = 0.10

# Minimum candles needed after feature engineering.
MIN_CANDLES = 300

# The requested target is 500K, but an exchange API may not
# actually provide that amount. CSV/Binance historical data can.
MAX_TARGET = 500_000

SUPPORTED_QUOTES = ["USD", "USDT", "USDC"]

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48", "ret_96",
    "ema_dist_5", "ema_dist_20", "ema_dist_50", "ema_dist_100", "ema_dist_200",
    "ema_5_20", "ema_20_50", "ema_50_200",
    "rsi_7", "rsi_14", "rsi_21",
    "macd_norm", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "atr_pct", "adx",
    "stoch_k", "stoch_d",
    "body_pct", "range_pct", "upper_wick", "lower_wick", "body_to_range",
    "volume_ratio", "volume_z", "volume_change",
    "volatility_12", "volatility_24", "volatility_48", "volatility_96",
    "distance_high_48", "distance_low_48",
    "distance_high_200", "distance_low_200",
    "drawdown", "price_z_50", "price_z_200",
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)

st.title("📊 Spot AI Scanner PRO")
st.caption(
    "15m SCALPING • +10%/-10% Target • Max 30 Minutes • Walk-Forward AI • No Order Execution"
)


# ============================================================
# SESSION
# ============================================================

DEFAULTS = {
    "results": pd.DataFrame(),
    "errors": [],
    "cache": {},
    "last_scan": None,
    "last_source": None,
    "status": "READY",
    "force_scan": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HTTP
# ============================================================

@st.cache_resource
def http_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Spot-AI-Scanner-PRO/1.0",
        "Accept": "application/json",
    })
    return s


HTTP = http_session()


# ============================================================
# EXCEPTIONS / HELPERS
# ============================================================

class DataProviderError(Exception):
    pass


class BinanceRestrictedError(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def sf(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()

    required = ["timestamp", "open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataProviderError(
            "Eksik OHLCV kolonları: " + ", ".join(missing)
        )

    x = df.copy()
    x["timestamp"] = pd.to_datetime(
        x["timestamp"], utc=True, errors="coerce"
    )

    for c in required[1:]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=required)
    x = (
        x.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return x


# ============================================================
# KRAKEN
# ============================================================

def kraken_get(endpoint, params=None):
    try:
        r = HTTP.get(
            KRAKEN_URL + endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise DataProviderError(f"Kraken bağlantı hatası: {e}")

    if r.status_code >= 400:
        raise DataProviderError(
            f"Kraken HTTP {r.status_code}: {r.text[:500]}"
        )

    try:
        payload = r.json()
    except Exception as e:
        raise DataProviderError(f"Kraken JSON hatası: {e}")

    if payload.get("error"):
        raise DataProviderError(
            "Kraken API: " + str(payload["error"])
        )

    return payload.get("result", {})


@st.cache_data(ttl=900, show_spinner=False)
def kraken_symbols():
    result = kraken_get("/0/public/AssetPairs")
    symbols = []

    for key, item in result.items():
        if not isinstance(item, dict):
            continue

        if item.get("status", "online") != "online":
            continue

        alt = str(item.get("altname", key))
        ws = str(item.get("wsname", alt))
        quote = str(item.get("quote", "")).upper()

        q = quote.replace("X", "").replace("Z", "")
        wsu = ws.upper()

        # Prefer USD/USDT/USDC for cleaner comparison.
        if not (
            q in ["USD", "USDT", "USDC"]
            or wsu.endswith("/USD")
            or wsu.endswith("/USDT")
            or wsu.endswith("/USDC")
        ):
            continue

        if ".D" in alt.upper():
            continue

        symbols.append({
            "api": alt,
            "display": ws,
        })

    return sorted(symbols, key=lambda x: x["display"])


def kraken_ohlc(symbol):
    result = kraken_get(
        "/0/public/OHLC",
        {"pair": symbol, "interval": 15},
    )

    pair_key = next(
        (k for k in result.keys() if k != "last"),
        None,
    )

    if pair_key is None:
        return pd.DataFrame()

    rows = []

    for row in result[pair_key]:
        rows.append({
            "timestamp": pd.to_datetime(
                int(row[0]), unit="s", utc=True
            ),
            "open": sf(row[1]),
            "high": sf(row[2]),
            "low": sf(row[3]),
            "close": sf(row[4]),
            "volume": sf(row[6]),
        })

    return clean_ohlcv(pd.DataFrame(rows))


# ============================================================
# BINANCE
# ============================================================

def binance_get(endpoint, params=None):
    try:
        r = HTTP.get(
            BINANCE_URL + endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise DataProviderError(f"Binance bağlantı hatası: {e}")

    if r.status_code == 451:
        raise BinanceRestrictedError(
            "Binance HTTP 451: Bu Streamlit çalışma ortamından "
            "Binance API erişimi kısıtlı."
        )

    if r.status_code >= 400:
        raise DataProviderError(
            f"Binance HTTP {r.status_code}: {r.text[:500]}"
        )

    try:
        return r.json()
    except Exception as e:
        raise DataProviderError(f"Binance JSON hatası: {e}")


@st.cache_data(ttl=900, show_spinner=False)
def binance_symbols():
    data = binance_get("/api/v3/exchangeInfo")
    out = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        if item.get("quoteAsset") not in ["USDT", "USDC"]:
            continue

        out.append(item["symbol"])

    return sorted(out)


def binance_history(symbol, target):
    frames = []
    end_time = None
    total = 0
    max_requests = math.ceil(target / 1000)

    for _ in range(max_requests):
        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": 1000,
        }

        if end_time is not None:
            params["endTime"] = int(end_time)

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
                    int(row[0]), unit="ms", utc=True
                ),
                "open": sf(row[1]),
                "high": sf(row[2]),
                "low": sf(row[3]),
                "close": sf(row[4]),
                "volume": sf(row[5]),
            })

        chunk = clean_ohlcv(pd.DataFrame(rows))

        if chunk.empty:
            break

        frames.append(chunk)
        total += len(chunk)

        oldest = chunk["timestamp"].min()
        end_time = int(oldest.timestamp() * 1000) - 1

        if len(chunk) < 1000 or total >= target:
            break

    if not frames:
        return pd.DataFrame()

    return (
        clean_ohlcv(pd.concat(frames, ignore_index=True))
        .tail(target)
        .reset_index(drop=True)
    )


# ============================================================
# CSV
# ============================================================

def read_csv(upload):
    raw = upload.read()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise DataProviderError(f"CSV okunamadı: {e}")

    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    aliases = {
        "timestamp": [
            "timestamp", "time", "date", "datetime",
            "open_time", "opentime"
        ],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }

    rename = {}

    for target, names in aliases.items():
        for name in names:
            if name in normalized:
                rename[normalized[name]] = target
                break

    return clean_ohlcv(df.rename(columns=rename))


# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(
        span=n,
        adjust=False,
        min_periods=n,
    ).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    al = loss.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    rs = ag / (al + 1e-12)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def bollinger(s, n=20):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid, mid + 2 * std, mid - 2 * std


def adx(df, n=14):
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

    prev = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)

    atr_value = tr.rolling(n).mean()

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index,
        ).rolling(n).mean()
        / (atr_value + 1e-12)
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index,
        ).rolling(n).mean()
        / (atr_value + 1e-12)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di + 1e-12)
    )

    return dx.rolling(n).mean()


# ============================================================
# FEATURES
# ============================================================

def build_features(df):
    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    for n in [1, 3, 6, 12, 24, 48, 96]:
        x[f"ret_{n}"] = close.pct_change(n)

    e5 = ema(close, 5)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e100 = ema(close, 100)
    e200 = ema(close, 200)

    for n, e in [
        (5, e5),
        (20, e20),
        (50, e50),
        (100, e100),
        (200, e200),
    ]:
        x[f"ema_dist_{n}"] = close / (e + 1e-12) - 1

    x["ema_5_20"] = e5 / (e20 + 1e-12) - 1
    x["ema_20_50"] = e20 / (e50 + 1e-12) - 1
    x["ema_50_200"] = e50 / (e200 + 1e-12) - 1

    for n in [7, 14, 21]:
        x[f"rsi_{n}"] = rsi(close, n)

    ml, ms, mh = macd(close)

    x["macd_norm"] = ml / (close + 1e-12)
    x["macd_signal"] = ms
    x["macd_hist"] = mh

    mid, upper, lower = bollinger(close, 20)

    x["bb_width"] = (
        upper - lower
    ) / (mid + 1e-12)

    x["bb_position"] = (
        close - lower
    ) / (
        upper - lower + 1e-12
    )

    x["atr_pct"] = atr(x) / (close + 1e-12)
    x["adx"] = adx(x)

    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    x["stoch_k"] = (
        100
        * (close - lowest)
        / (highest - lowest + 1e-12)
    )

    x["stoch_d"] = x["stoch_k"].rolling(3).mean()

    candle_range = high - low
    body = close - x["open"]

    x["body_pct"] = body / (close + 1e-12)
    x["range_pct"] = candle_range / (close + 1e-12)

    x["upper_wick"] = (
        high - np.maximum(x["open"], close)
    )

    x["lower_wick"] = (
        np.minimum(x["open"], close) - low
    )

    x["body_to_range"] = (
        body.abs() / (candle_range + 1e-12)
    )

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()

    x["volume_ratio"] = (
        volume / (volume_mean + 1e-12)
    )

    x["volume_z"] = (
        volume - volume_mean
    ) / (volume_std + 1e-12)

    x["volume_change"] = volume.pct_change()

    for n in [12, 24, 48, 96]:
        x[f"volatility_{n}"] = (
            x["ret_1"].rolling(n).std()
        )

    for n in [48, 200]:
        rh = high.rolling(n).max()
        rl = low.rolling(n).min()

        x[f"distance_high_{n}"] = (
            close / (rh + 1e-12) - 1
        )

        x[f"distance_low_{n}"] = (
            close / (rl + 1e-12) - 1
        )

    running_high = close.cummax()

    x["drawdown"] = (
        close / (running_high + 1e-12) - 1
    )

    for n in [50, 200]:
        m = close.rolling(n).mean()
        s = close.rolling(n).std()

        x[f"price_z_{n}"] = (
            close - m
        ) / (s + 1e-12)

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================
# TARGET
# ============================================================

def make_target(df, future_bars=SCALP_HORIZON_BARS):
    x = df.copy()

    # IMPORTANT: this is an intraperiod scalp target.
    # We look at the highest high and lowest low in the NEXT
    # two 15-minute candles, not only the close 30 minutes later.
    future_highs = pd.concat(
        [x["high"].shift(-i) for i in range(1, future_bars + 1)],
        axis=1,
    )

    future_lows = pd.concat(
        [x["low"].shift(-i) for i in range(1, future_bars + 1)],
        axis=1,
    )

    x["future_max_high"] = future_highs.max(axis=1, skipna=False)
    x["future_min_low"] = future_lows.min(axis=1, skipna=False)

    x["future_upside"] = (
        x["future_max_high"] / x["close"] - 1
    )

    x["future_downside"] = (
        x["future_min_low"] / x["close"] - 1
    )

    # Legacy-compatible field: for a scalp, the directional return
    # is the maximum excursion in the next 30 minutes.
    x["future_return"] = np.where(
        x["future_upside"] >= abs(x["future_downside"]),
        x["future_upside"],
        x["future_downside"],
    )

    # 0 = SELL, 1 = HOLD, 2 = BUY.
    # BUY means +10% was actually reachable within <=30 minutes.
    # SELL means -10% was actually reachable within <=30 minutes.
    x["target"] = 1

    buy_mask = (
        (x["future_upside"] >= SCALP_TARGET_RETURN)
        & (x["future_upside"] >= x["future_downside"].abs())
    )

    sell_mask = (
        (x["future_downside"] <= -SCALP_TARGET_RETURN)
        & (x["future_downside"].abs() > x["future_upside"])
    )

    x.loc[buy_mask, "target"] = 2
    x.loc[sell_mask, "target"] = 0

    return x


# ============================================================
# MODEL
# ============================================================

def make_mlp():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                solver="adam",
                batch_size=128,
                learning_rate_init=0.001,
                max_iter=160,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=42,
            ),
        ),
    ])


def make_regressor():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(alpha=1.0),
        ),
    ])


def balance_training_data(X, y, max_multiplier=2):
    """
    MLPClassifier does not expose class_weight.
    We balance classes by controlled oversampling.

    This specifically prevents HOLD from dominating the
    classifier simply because it is the most common class.
    """

    frame = X.copy()
    frame["_target_"] = np.asarray(y)

    counts = frame["_target_"].value_counts()

    if len(counts) < 2:
        return X, y

    target_size = min(
        int(counts.max()),
        int(counts.min() * max_multiplier),
    )

    pieces = []

    rng = np.random.RandomState(42)

    for cls, group in frame.groupby("_target_"):

        if len(group) >= target_size:
            sampled = group.sample(
                target_size,
                random_state=42,
            )
        else:
            extra = group.sample(
                target_size - len(group),
                replace=True,
                random_state=rng,
            )

            sampled = pd.concat(
                [group, extra],
                ignore_index=True,
            )

        pieces.append(sampled)

    balanced = pd.concat(
        pieces,
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    y_balanced = balanced.pop(
        "_target_"
    ).astype(int)

    return balanced, y_balanced


# ============================================================
# MODEL TRAINING
# ============================================================

def train_predict(df, future_bars, train_limit):
    features = build_features(df)
    labeled = make_target(
        features,
        future_bars,
    )

    usable = (
        labeled
        .dropna(
            subset=FEATURES + [
                "target",
                "future_return",
                "future_upside",
                "future_downside",
            ]
        )
        .copy()
    )

    if len(usable) < MIN_CANDLES:
        raise DataProviderError(
            f"Temiz model verisi yetersiz: {len(usable)}"
        )

    # Keep chronological order.
    usable = usable.tail(
        min(train_limit, len(usable))
    ).reset_index(drop=True)

    n = len(usable)

    # True chronological split:
    # 70% TRAIN / 15% VALIDATION / 15% TEST
    train_end = max(
        int(n * 0.70),
        200,
    )

    valid_end = max(
        int(n * 0.85),
        train_end + 50,
    )

    if valid_end >= n:
        valid_end = n - 1

    train = usable.iloc[:train_end]
    validation = usable.iloc[
        train_end:valid_end
    ]
    test = usable.iloc[valid_end:]

    if len(test) < 20:
        raise DataProviderError(
            "Test seti çok küçük."
        )

    X_train = train[FEATURES].astype(float)
    y_train = train["target"].astype(int)

    X_test = test[FEATURES].astype(float)
    y_test = test["target"].astype(int)

    # Balance ONLY training data.
    X_train_balanced, y_train_balanced = (
        balance_training_data(
            X_train,
            y_train,
        )
    )

    classifier = make_mlp()
    classifier.fit(
        X_train_balanced,
        y_train_balanced,
    )

    # Honest unseen test.
    test_pred = classifier.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_pred,
        )
    )

    # Train two regression models aligned with the 10%/30-minute target:
    # one estimates maximum upside and the other maximum downside.
    upside_regressor = make_regressor()
    downside_regressor = make_regressor()

    upside_regressor.fit(
        train[FEATURES].astype(float),
        train["future_upside"].astype(float),
    )

    downside_regressor.fit(
        train[FEATURES].astype(float),
        train["future_downside"].astype(float),
    )

    # Latest available feature row.
    latest_rows = (
        labeled
        .dropna(subset=FEATURES)
    )

    if latest_rows.empty:
        raise DataProviderError(
            "Son kullanılabilir feature satırı yok."
        )

    latest = latest_rows.iloc[-1]

    X_latest = pd.DataFrame([
        latest[FEATURES].astype(float)
    ])

    probabilities = (
        classifier
        .predict_proba(X_latest)[0]
    )

    class_map = {
        int(c): float(p)
        for c, p in zip(
            classifier.classes_,
            probabilities,
        )
    }

    p_sell = class_map.get(0, 0.0)
    p_hold = class_map.get(1, 0.0)
    p_buy = class_map.get(2, 0.0)

    expected_upside = float(
        upside_regressor.predict(
            X_latest
        )[0]
    )

    expected_downside = float(
        downside_regressor.predict(
            X_latest
        )[0]
    )

    # Direction-specific expected move.
    expected_return = (
        expected_upside
        if p_buy >= p_sell
        else expected_downside
    )

    # --------------------------------------------------------
    # Directional scoring
    # --------------------------------------------------------

    # Probability difference.
    direction = p_buy - p_sell

    momentum = sf(
        latest.get("ret_24")
    )

    trend = sf(
        latest.get("ema_20_50")
    )

    volume_ratio = sf(
        latest.get("volume_ratio")
    )

    volatility = sf(
        latest.get("volatility_24")
    )

    if np.isnan(momentum):
        momentum_score = 0.0
    else:
        momentum_score = float(
            np.tanh(momentum * 20)
        )

    if np.isnan(trend):
        trend_score = 0.0
    else:
        trend_score = float(
            np.tanh(trend * 20)
        )

    if np.isnan(volume_ratio):
        volume_score = 0.0
    else:
        volume_score = float(
            np.tanh((volume_ratio - 1) / 2)
        )

    if np.isnan(volatility):
        risk_score = 0.0
    else:
        risk_score = float(
            1 - np.clip(
                volatility * 30,
                0,
                1,
            )
        )

    # Expected 30-minute move contribution.
    # A 10% target corresponds to the hard scalp threshold.
    return_score = float(
        np.tanh(expected_return * 10)
    )

    # Directional score:
    # 50 = neutral
    # >50 = bullish
    # <50 = bearish
    directional_score = float(
        np.clip(
            50
            + direction * 30
            + return_score * 15
            + momentum_score * 8
            + trend_score * 5
            + volume_score * 2,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # Strong signal decision
    # --------------------------------------------------------

    # Do NOT allow HOLD probability alone to dominate.
    # BUY/SELL must satisfy both probability and direction.
    buy_edge = p_buy - p_sell
    sell_edge = p_sell - p_buy

    # HARD SCALP RULE:
    # BUY requires the model to estimate that +10% can be reached
    # within the next two 15-minute candles (<=30 minutes).
    # SELL is symmetric at -10%.
    buy_condition = (
        p_buy >= 0.45
        and buy_edge >= 0.10
        and expected_upside >= SCALP_TARGET_RETURN
        and directional_score >= 55
    )

    sell_condition = (
        p_sell >= 0.45
        and sell_edge >= 0.10
        and expected_downside <= -SCALP_TARGET_RETURN
        and directional_score <= 45
    )

    if buy_condition:
        signal = "BUY"
    elif sell_condition:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence is now directional confidence,
    # not simply the HOLD class probability.
    if signal == "BUY":
        confidence = float(
            np.clip(
                0.5 * p_buy
                + 0.3 * max(buy_edge, 0)
                + 0.2 * max(return_score, 0),
                0,
                1,
            )
        )
    elif signal == "SELL":
        confidence = float(
            np.clip(
                0.5 * p_sell
                + 0.3 * max(sell_edge, 0)
                + 0.2 * max(-return_score, 0),
                0,
                1,
            )
        )
    else:
        # HOLD confidence means "lack of directional edge",
        # not "98% chance price will stay flat".
        confidence = float(
            np.clip(
                1
                - abs(direction)
                - min(abs(return_score), 0.5),
                0,
                1,
            )
        )

    # --------------------------------------------------------
    # Ranking scores
    # --------------------------------------------------------

    buy_score = float(
        np.clip(
            50
            + 30 * p_buy
            - 20 * p_sell
            + 15 * max(return_score, 0)
            + 8 * max(momentum_score, 0)
            + 5 * max(trend_score, 0),
            0,
            100,
        )
    )

    sell_score = float(
        np.clip(
            50
            + 30 * p_sell
            - 20 * p_buy
            + 15 * max(-return_score, 0)
            + 8 * max(-momentum_score, 0)
            + 5 * max(-trend_score, 0),
            0,
            100,
        )
    )

    # Main score is directional.
    if buy_score >= sell_score:
        ai_score = buy_score
    else:
        ai_score = 100 - sell_score

    return {
        "signal": signal,
        "confidence": confidence,
        "p_buy": p_buy,
        "p_hold": p_hold,
        "p_sell": p_sell,
        "expected_return": expected_return,
        "expected_upside": expected_upside,
        "expected_downside": expected_downside,
        "scalp_target": SCALP_TARGET_RETURN,
        "scalp_horizon_minutes": SCALP_HORIZON_BARS * 15,
        "price": float(latest["close"]),
        "ai_score": float(ai_score),
        "buy_score": buy_score,
        "sell_score": sell_score,
        "directional_score": directional_score,
        "rsi14": sf(latest.get("rsi_14")),
        "bb_position": sf(latest.get("bb_position")),
        "atr_pct": sf(latest.get("atr_pct")),
        "adx": sf(latest.get("adx")),
        "volume_ratio": volume_ratio,
        "momentum": momentum,
        "trend": trend,
        "volatility": volatility,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

source = st.sidebar.selectbox(
    "Data Source",
    [
        "Kraken REST",
        "CSV / Historical 500K",
        "Binance REST",
    ],
)

uploaded_file = None

if source == "CSV / Historical 500K":
    uploaded_file = st.sidebar.file_uploader(
        "500K OHLCV CSV",
        type=["csv"],
        help=(
            "500,000 adet 15m candle içeren "
            "CSV yükleyebilirsiniz."
        ),
    )

    st.sidebar.caption(
        "Kolonlar: timestamp, open, high, low, close, volume"
    )

target = st.sidebar.selectbox(
    "Historical Candle Target",
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
    1,
    100,
    10,
)

future_bars = SCALP_HORIZON_BARS

st.sidebar.info(
    "Scalping hedefi: +10% / -10%\n"
    "maksimum 30 dakika (2 × 15m mum)."
)

train_limit = st.sidebar.slider(
    "Training Rows",
    500,
    100_000,
    20_000,
    step=500,
)

minimum_signal_confidence = st.sidebar.slider(
    "Minimum Signal Confidence",
    0.30,
    0.90,
    0.55,
    step=0.01,
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
# DISCOVERY
# ============================================================

def discover():
    if source == "Kraken REST":
        symbols = kraken_symbols()

        if not symbols:
            raise DataProviderError(
                "Kraken Spot sembol bulunamadı."
            )

        return symbols[:symbol_limit]

    if source == "Binance REST":
        return binance_symbols()[:symbol_limit]

    if uploaded_file is None:
        raise DataProviderError(
            "500K Historical CSV yüklenmedi."
        )

    return [{
        "api": "CSV",
        "display": "UPLOADED_500K",
    }]


# ============================================================
# DATA LOADING
# ============================================================

def load_symbol(pair):
    if source == "Kraken REST":
        api = pair["api"]
        display = pair["display"]

        key = "KRAKEN|" + api

        if key in st.session_state.cache:
            return (
                display,
                st.session_state.cache[key].tail(target),
            )

        df = kraken_ohlc(api)

        if not df.empty:
            st.session_state.cache[key] = df

        return display, df.tail(target)

    if source == "Binance REST":
        symbol = pair
        key = "BINANCE|" + symbol

        if key in st.session_state.cache:
            return (
                symbol,
                st.session_state.cache[key].tail(target),
            )

        df = binance_history(
            symbol,
            target,
        )

        if not df.empty:
            st.session_state.cache[key] = df

        return symbol, df.tail(target)

    df = read_csv(uploaded_file)

    # CSV mode is designed for the real 500K use case.
    return (
        "UPLOADED_500K",
        df.tail(target),
    )


# ============================================================
# SCAN
# ============================================================

def run_scan():
    st.session_state.errors = []
    st.session_state.status = "SCANNING"

    results = []

    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Semboller hazırlanıyor...")
        symbols = discover()

        total = len(symbols)

        for i, pair in enumerate(symbols):
            label = (
                pair["display"]
                if isinstance(pair, dict)
                else str(pair)
            )

            status.info(
                f"Tarama: {label} "
                f"({i + 1}/{total})"
            )

            try:
                symbol, df = load_symbol(pair)

                if len(df) < MIN_CANDLES:
                    raise DataProviderError(
                        f"Yetersiz candle: {len(df)}. "
                        f"En az {MIN_CANDLES} gerekli."
                    )

                start = time.time()

                prediction = train_predict(
                    df,
                    future_bars,
                    train_limit,
                )

                results.append({
                    "symbol": symbol,
                    **prediction,
                    "candles": len(df),
                    "last_candle": df["timestamp"].iloc[-1],
                    "processing_seconds": time.time() - start,
                })

            except Exception as exc:
                st.session_state.errors.append({
                    "symbol": label,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

            progress.progress(
                int(
                    (i + 1)
                    / total
                    * 100
                )
            )

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            result_df = (
                result_df
                .sort_values(
                    "ai_score",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        st.session_state.results = result_df
        st.session_state.last_scan = utc_now()
        st.session_state.last_source = source
        st.session_state.status = (
            "SUCCESS"
            if not result_df.empty
            else "NO_RESULTS"
        )

    except BinanceRestrictedError as exc:
        st.session_state.errors.append({
            "symbol": "BINANCE",
            "error": str(exc),
        })
        st.session_state.status = "BINANCE_451"
        st.error(str(exc))

    except Exception as exc:
        st.session_state.errors.append({
            "symbol": "SYSTEM",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        })
        st.session_state.status = "ERROR"
        st.error(
            f"Tarama hatası: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        progress.empty()
        status.empty()
        st.session_state.force_scan = False


# ============================================================
# AUTO RUN
# ============================================================

if hasattr(st, "fragment"):

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

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Status",
        st.session_state.status,
    )

with c2:
    st.metric(
        "Source",
        st.session_state.last_source or source,
    )

with c3:
    st.metric(
        "Historical Candle Target",
        f"{target:,}",
    )

with c4:
    last = st.session_state.last_scan

    st.metric(
        "Last Scan",
        last.strftime("%H:%M:%S")
        if last
        else "-",
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

results = st.session_state.results


if results.empty:

    st.warning("Henüz sonuç yok.")
    st.info(
        "Bu sürüm yalnızca 15 dakikalık veride, sonraki 2 mum içinde "
        "+10% yükseliş veya -10% düşüş ihtimalini arar. "
        "10% hedefini karşılamayan sinyaller BUY/SELL olarak listelenmez."
    )

    if source == "Kraken REST":
        st.info(
            "Kraken public OHLC sınırlı recent history "
            "sağlar. Bu nedenle Kraken seçiliyken "
            "500K hedefi otomatik olarak 500K geçmiş "
            "anlamına gelmez. Gerçek 500K için "
            "'CSV / Historical 500K' veya erişilebilen "
            "bir tarihsel veri kaynağı kullanın."
        )

else:

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buy = (
        results[
            (results["signal"] == "BUY")
            & (
                results["confidence"]
                >= minimum_signal_confidence
            )
        ]
        .sort_values(
            [
                "buy_score",
                "confidence",
                "expected_return",
            ],
            ascending=False,
        )
        .head(10)
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    sell = (
        results[
            (results["signal"] == "SELL")
            & (
                results["confidence"]
                >= minimum_signal_confidence
            )
        ]
        .sort_values(
            [
                "sell_score",
                "confidence",
                "expected_return",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(10)
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    def show_signal_table(df):
        if df.empty:
            st.info(
                "Bu filtreyle güçlü sinyal bulunamadı."
            )
            return

        view = df.copy()

        for col in [
            "confidence",
            "p_buy",
            "p_hold",
            "p_sell",
            "expected_return",
            "atr_pct",
        ]:
            if col in view:
                view[col] *= 100

        cols = [
            "symbol",
            "signal",
            "ai_score",
            "buy_score",
            "sell_score",
            "confidence",
            "expected_return",
            "expected_upside",
            "expected_downside",
            "p_buy",
            "p_hold",
            "p_sell",
            "rsi14",
            "adx",
            "volume_ratio",
            "price",
            "candles",
            "test_balanced_accuracy",
        ]

        available = [
            c for c in cols
            if c in view.columns
        ]

        st.dataframe(
            view[available].round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # TOP BUY
    # --------------------------------------------------------

    st.subheader("🟢 TOP 10 BUY")

    show_signal_table(buy)

    # --------------------------------------------------------
    # TOP SELL
    # --------------------------------------------------------

    st.subheader("🔴 TOP 10 SELL")

    show_signal_table(sell)

    # --------------------------------------------------------
    # HOLD - secondary
    # --------------------------------------------------------

    st.subheader("🟡 HOLD / NO CLEAR EDGE")

    st.caption(
        "HOLD ana sıralama değildir. BUY/SELL için yeterli "
        "yönsel edge bulunmayan coinler burada gösterilir."
    )

    hold = (
        results[
            results["signal"] == "HOLD"
        ]
        .sort_values(
            "directional_score",
            ascending=False,
        )
        .head(10)
    )

    show_signal_table(hold)

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    with st.expander("📋 Tüm Sonuçlar"):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # DETAIL
    # --------------------------------------------------------

    st.subheader("🔎 Symbol Detail")

    selected = st.selectbox(
        "Symbol",
        results["symbol"].tolist(),
    )

    row = results[
        results["symbol"] == selected
    ].iloc[0]

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Signal",
            row["signal"],
        )

    with b:
        st.metric(
            "Signal Confidence",
            f"{row['confidence'] * 100:.2f}%",
        )

    with c:
        st.metric(
            "Expected 30m Move",
            f"{row['expected_return'] * 100:.3f}%",
        )

    with d:
        st.metric(
            "AI Score",
            f"{row['ai_score']:.2f}/100",
        )

    detail = pd.DataFrame({
        "Metric": [
            "Price",
            "BUY Probability",
            "HOLD Probability",
            "SELL Probability",
            "Expected Upside (30m)",
            "Expected Downside (30m)",
            "BUY Score",
            "SELL Score",
            "Directional Score",
            "RSI 14",
            "Bollinger Position",
            "ATR %",
            "ADX",
            "Volume Ratio",
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
            row["price"],
            f"{row['p_buy'] * 100:.2f}%",
            f"{row['p_hold'] * 100:.2f}%",
            f"{row['p_sell'] * 100:.2f}%",
            f"{row['expected_upside'] * 100:.3f}%",
            f"{row['expected_downside'] * 100:.3f}%",
            f"{row['buy_score']:.2f}/100",
            f"{row['sell_score']:.2f}/100",
            f"{row['directional_score']:.2f}/100",
            row["rsi14"],
            row["bb_position"],
            f"{row['atr_pct'] * 100:.3f}%",
            row["adx"],
            row["volume_ratio"],
            f"{row['test_accuracy'] * 100:.2f}%",
            f"{row['test_balanced_accuracy'] * 100:.2f}%",
            row["train_rows"],
            row["validation_rows"],
            row["test_rows"],
            row["candles"],
            row["last_candle"],
            f"{row['processing_seconds']:.2f}",
        ],
    })

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download AI Results CSV",
        data=results.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="spot_ai_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# ERRORS
# ============================================================

if st.session_state.errors:

    st.divider()

    st.subheader("⚠️ Hata Detayları")

    for item in st.session_state.errors:

        st.error(
            f"{item.get('symbol', 'SYSTEM')}: "
            f"{item.get('error', 'Bilinmeyen hata')}"
        )


# ============================================================
# CACHE
# ============================================================

with st.expander("🗄️ Candle Cache"):

    if not st.session_state.cache:

        st.info("Cache boş.")

    else:

        cache_rows = []

        for key, df in st.session_state.cache.items():

            if df.empty:
                continue

            cache_rows.append({
                "source": key,
                "candles": len(df),
                "first": str(df["timestamp"].iloc[0]),
                "last": str(df["timestamp"].iloc[-1]),
            })

        if cache_rows:

            st.dataframe(
                pd.DataFrame(cache_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info("Cache boş.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Scanner PRO • 15m Scalping • +10% in ≤30m • "
    "500K Historical Target • No Order Execution"
)

st.caption(
    "Model yalnızca OHLCV verilerinden istatistiksel "
    "BUY/SELL/HOLD sınıflandırması ve yönsel skor üretir. "
    "Bu bir yatırım tavsiyesi değildir."
)
