import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Spot Scalping AI Scanner",
    page_icon="📊",
    layout="wide",
)

BINANCE = "https://api.binance.com"
KRAKEN = "https://api.kraken.com"

TIMEFRAME = "15m"

# 2 x 15 minutes
FUTURE_BARS = 2
HORIZON_MINUTES = 30

BUY_TARGET = 0.10
SELL_TARGET = -0.10

MIN_CANDLES = 200

DEFAULT_CANDLES = 1000
DEFAULT_TRAIN = 10000

MAX_CANDLES = 500000

QUOTES = {
    "USDT",
    "USDC",
}

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
    "low_distance",
]


# ============================================================
# SESSION
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None


session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 SpotScalpingAI/1.0",
        "Accept": "application/json",
    }
)


# ============================================================
# HELPERS
# ============================================================

def request_json(
    url,
    params=None,
):
    try:
        r = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        raise RuntimeError(
            f"Connection error: {e}"
        )

    if r.status_code == 451:
        raise RuntimeError(
            "HTTP 451: API bölgesel olarak erişilemiyor."
        )

    if r.status_code >= 400:
        raise RuntimeError(
            f"HTTP {r.status_code}: {r.text[:250]}"
        )

    try:
        return r.json()
    except Exception:
        raise RuntimeError(
            "API JSON döndürmedi."
        )


def clean_df(df):
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

    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"Eksik kolon: {c}"
            )

    x = df.copy()

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
        errors="coerce",
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    x = x.dropna(
        subset=required
    )

    x = x[
        x["close"] > 0
    ]

    x = x[
        x["high"] >= x["low"]
    ]

    x = (
        x.sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    return x


# ============================================================
# BINANCE SYMBOLS
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def binance_symbols():

    data = request_json(
        BINANCE
        + "/api/v3/exchangeInfo"
    )

    output = []

    for s in data.get(
        "symbols",
        [],
    ):

        if s.get(
            "status"
        ) != "TRADING":
            continue

        if s.get(
            "isSpotTradingAllowed",
            True,
        ) is False:
            continue

        if s.get(
            "quoteAsset"
        ) not in QUOTES:
            continue

        symbol = s.get(
            "symbol"
        )

        if symbol:
            output.append(
                symbol
            )

    return sorted(
        set(output)
    )


# ============================================================
# BINANCE DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def binance_history(
    symbol,
    candles,
):

    candles = int(candles)

    frames = []

    end_time = None

    while candles > 0:

        limit = min(
            candles,
            1000,
        )

        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "limit": limit,
        }

        if end_time is not None:
            params[
                "endTime"
            ] = end_time

        data = request_json(
            BINANCE
            + "/api/v3/klines",
            params,
        )

        if not data:
            break

        rows = []

        for k in data:

            rows.append(
                {
                    "timestamp":
                        pd.to_datetime(
                            k[0],
                            unit="ms",
                            utc=True,
                        ),
                    "open":
                        float(k[1]),
                    "high":
                        float(k[2]),
                    "low":
                        float(k[3]),
                    "close":
                        float(k[4]),
                    "volume":
                        float(k[5]),
                }
            )

        part = clean_df(
            pd.DataFrame(rows)
        )

        if part.empty:
            break

        frames.append(
            part
        )

        candles -= len(part)

        oldest = part[
            "timestamp"
        ].min()

        end_time = (
            int(
                oldest.timestamp()
                * 1000
            )
            - 1
        )

        if len(part) < limit:
            break

        time.sleep(
            0.04
        )

    if not frames:
        raise RuntimeError(
            f"{symbol}: candle verisi yok."
        )

    return clean_df(
        pd.concat(
            frames,
            ignore_index=True,
        )
    ).tail(
        int(candles)
        if False
        else 500000
    ).reset_index(
        drop=True
    )


# ============================================================
# KRAKEN SYMBOLS
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def kraken_symbols():

    data = request_json(
        KRAKEN
        + "/0/public/AssetPairs"
    )

    result = data.get(
        "result",
        {},
    )

    output = []

    for key, item in result.items():

        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get(
            "status",
            "online",
        ) != "online":
            continue

        wsname = str(
            item.get(
                "wsname",
                "",
            )
        )

        altname = str(
            item.get(
                "altname",
                key,
            )
        )

        upper = wsname.upper()

        if not (
            upper.endswith("/USD")
            or upper.endswith("/USDT")
            or upper.endswith("/USDC")
        ):
            continue

        output.append(
            {
                "api": altname,
                "name": wsname,
            }
        )

    return sorted(
        output,
        key=lambda x: x["name"],
    )


# ============================================================
# KRAKEN DATA
# ============================================================

def kraken_history(
    pair,
):

    data = request_json(
        KRAKEN
        + "/0/public/OHLC",
        {
            "pair": pair,
            "interval": 15,
        },
    )

    result = data.get(
        "result",
        {},
    )

    pair_key = next(
        (
            k
            for k in result
            if k != "last"
        ),
        None,
    )

    if pair_key is None:
        raise RuntimeError(
            f"{pair}: Kraken candle yok."
        )

    rows = []

    for k in result[
        pair_key
    ]:

        rows.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(k[0]),
                        unit="s",
                        utc=True,
                    ),
                "open":
                    float(k[1]),
                "high":
                    float(k[2]),
                "low":
                    float(k[3]),
                "close":
                    float(k[4]),
                "volume":
                    float(k[6]),
            }
        )

    df = clean_df(
        pd.DataFrame(rows)
    )

    if len(df) < MIN_CANDLES:
        raise RuntimeError(
            f"{pair}: yalnızca "
            f"{len(df)} candle var."
        )

    return df


# ============================================================
# CSV
# ============================================================

def csv_data(
    uploaded,
):

    raw = uploaded.read()

    df = pd.read_csv(
        io.BytesIO(raw)
    )

    rename = {}

    normalized = {
        str(c)
        .lower()
        .strip()
        .replace(" ", "_"):
        c
        for c in df.columns
    }

    aliases = {
        "timestamp": [
            "timestamp",
            "time",
            "date",
            "datetime",
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

    return clean_df(
        df
    )


# ============================================================
# INDICATORS
# ============================================================

def ema(
    s,
    n,
):
    return s.ewm(
        span=n,
        adjust=False,
    ).mean()


def rsi(
    s,
    n=14,
):

    delta = s.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    ag = gain.ewm(
        alpha=1 / n,
        adjust=False,
    ).mean()

    al = loss.ewm(
        alpha=1 / n,
        adjust=False,
    ).mean()

    rs = ag / (
        al + 1e-12
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(
    df,
    n=14,
):

    pc = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"]
            - df["low"],

            (
                df["high"]
                - pc
            ).abs(),

            (
                df["low"]
                - pc
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False,
    ).mean()


def adx(
    df,
    n=14,
):

    h = df["high"]
    l = df["low"]
    c = df["close"]

    up = h.diff()

    down = -l.diff()

    plus = np.where(
        (up > down)
        & (up > 0),
        up,
        0,
    )

    minus = np.where(
        (down > up)
        & (down > 0),
        down,
        0,
    )

    pc = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - pc).abs(),
            (l - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atrv = tr.ewm(
        alpha=1 / n,
        adjust=False,
    ).mean()

    pdi = (
        100
        * pd.Series(
            plus,
            index=df.index,
        )
        .ewm(
            alpha=1 / n,
            adjust=False,
        )
        .mean()
        / (
            atrv + 1e-12
        )
    )

    mdi = (
        100
        * pd.Series(
            minus,
            index=df.index,
        )
        .ewm(
            alpha=1 / n,
            adjust=False,
        )
        .mean()
        / (
            atrv + 1e-12
        )
    )

    dx = (
        100
        * (pdi - mdi).abs()
        / (
            pdi
            + mdi
            + 1e-12
        )
    )

    return dx.ewm(
        alpha=1 / n,
        adjust=False,
    ).mean()


# ============================================================
# FEATURES
# ============================================================

def features(
    df,
):

    x = df.copy()

    c = x["close"]
    h = x["high"]
    l = x["low"]
    o = x["open"]
    v = x["volume"]

    # Returns
    for n in [
        1,
        3,
        6,
        12,
        24,
    ]:
        x[
            f"ret{n}"
        ] = c.pct_change(n)

    # EMA
    e5 = ema(c, 5)
    e20 = ema(c, 20)
    e50 = ema(c, 50)
    e200 = ema(c, 200)

    x["ema5_20"] = (
        e5 / (e20 + 1e-12)
        - 1
    )

    x["ema20_50"] = (
        e20 / (e50 + 1e-12)
        - 1
    )

    x["ema50_200"] = (
        e50 / (e200 + 1e-12)
        - 1
    )

    x["dist20"] = (
        c / (e20 + 1e-12)
        - 1
    )

    x["dist50"] = (
        c / (e50 + 1e-12)
        - 1
    )

    x["dist200"] = (
        c / (e200 + 1e-12)
        - 1
    )

    # RSI
    x["rsi7"] = rsi(c, 7)
    x["rsi14"] = rsi(c, 14)
    x["rsi21"] = rsi(c, 21)

    # MACD
    m12 = ema(c, 12)
    m26 = ema(c, 26)

    macd = m12 - m26
    signal = ema(macd, 9)

    x["macd"] = (
        macd / (c + 1e-12)
    )

    x["macd_signal"] = (
        signal / (c + 1e-12)
    )

    x["macd_hist"] = (
        (macd - signal)
        / (c + 1e-12)
    )

    # Bollinger
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()

    upper = mid + 2 * std
    lower = mid - 2 * std

    x["bb_position"] = (
        (c - lower)
        / (
            upper
            - lower
            + 1e-12
        )
    )

    x["bb_width"] = (
        (upper - lower)
        / (mid + 1e-12)
    )

    # ATR / ADX
    atrv = atr(
        x,
        14,
    )

    x["atr_pct"] = (
        atrv / (c + 1e-12)
    )

    x["adx"] = adx(
        x,
        14,
    )

    # Stochastic
    lo14 = (
        l.rolling(14)
        .min()
    )

    hi14 = (
        h.rolling(14)
        .max()
    )

    x["stoch_k"] = (
        100
        * (c - lo14)
        / (
            hi14
            - lo14
            + 1e-12
        )
    )

    x["stoch_d"] = (
        x["stoch_k"]
        .rolling(3)
        .mean()
    )

    # Candle
    rng = h - l

    x["body_pct"] = (
        c - o
    ) / (
        c + 1e-12
    )

    x["range_pct"] = (
        rng
        / (
            c + 1e-12
        )
    )

    x["upper_wick"] = (
        h
        - pd.concat(
            [o, c],
            axis=1,
        ).max(axis=1)
    ) / (
        c + 1e-12
    )

    x["lower_wick"] = (
        pd.concat(
            [o, c],
            axis=1,
        ).min(axis=1)
        - l
    ) / (
        c + 1e-12
    )

    # Volume
    vm = (
        v.rolling(20)
        .mean()
    )

    vs = (
        v.rolling(20)
        .std()
    )

    x["volume_ratio"] = (
        v / (
            vm + 1e-12
        )
    )

    x["volume_z"] = (
        v - vm
    ) / (
        vs + 1e-12
    )

    # Volatility
    ret = c.pct_change()

    x["volatility12"] = (
        ret.rolling(12).std()
    )

    x["volatility24"] = (
        ret.rolling(24).std()
    )

    x["volatility48"] = (
        ret.rolling(48).std()
    )

    # Price Z
    pm = (
        c.rolling(50)
        .mean()
    )

    ps = (
        c.rolling(50)
        .std()
    )

    x["price_z"] = (
        c - pm
    ) / (
        ps + 1e-12
    )

    # High / low distance
    hi = (
        h.rolling(48)
        .max()
    )

    lo = (
        l.rolling(48)
        .min()
    )

    x["high_distance"] = (
        c / (
            hi + 1e-12
        )
        - 1
    )

    x["low_distance"] = (
        c / (
            lo + 1e-12
        )
        - 1
    )

    return x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# TARGET
# ============================================================

def make_target(
    df,
):

    x = df.copy()

    entry = x["close"]

    future_high = pd.concat(
        [
            x["high"].shift(-1),
            x["high"].shift(-2),
        ],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [
            x["low"].shift(-1),
            x["low"].shift(-2),
        ],
        axis=1,
    ).min(axis=1)

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

    # --------------------------------------------------------
    # NO HOLD
    #
    # +10 hit -> BUY
    # -10 hit -> SELL
    #
    # Neither hit -> larger directional excursion
    # --------------------------------------------------------

    target = np.where(
        up >= down.abs(),
        1,
        0,
    ).astype(float)

    target[
        buy_hit
        & ~sell_hit
    ] = 1

    target[
        sell_hit
        & ~buy_hit
    ] = 0

    # Last 2 candles cannot have a complete future.
    target[
        -FUTURE_BARS:
    ] = np.nan

    x["target"] = target

    x["buy_hit"] = (
        buy_hit.astype(float)
    )

    x["sell_hit"] = (
        sell_hit.astype(float)
    )

    x["future_up"] = up
    x["future_down"] = down

    return x


# ============================================================
# AI
# ============================================================

def model():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(
                        64,
                        32,
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=128,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=42,
                ),
            ),
        ]
    )


def balance(
    X,
    y,
):

    df = X.copy()

    df["_y"] = y.values

    counts = (
        df["_y"]
        .value_counts()
    )

    if len(counts) < 2:
        return X, y

    target_count = int(
        counts.max()
    )

    rng = np.random.default_rng(
        42
    )

    parts = []

    for cls in counts.index:

        part = df[
            df["_y"] == cls
        ]

        if len(part) < target_count:

            indices = rng.choice(
                len(part),
                size=(
                    target_count
                    - len(part)
                ),
                replace=True,
            )

            part = pd.concat(
                [
                    part,
                    part.iloc[
                        indices
                    ],
                ],
                ignore_index=True,
            )

        parts.append(
            part
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = out.sample(
        frac=1,
        random_state=42,
    ).reset_index(
        drop=True
    )

    return (
        out[FEATURES],
        out["_y"].astype(int),
    )


# ============================================================
# TECHNICAL FALLBACK
# ============================================================

def fallback(
    df,
):

    x = features(
        df
    )

    last = x.iloc[-1]

    score = 0.0

    # Trend
    score += np.tanh(
        float(
            last["ema20_50"]
        )
        * 30
    ) * 0.25

    score += np.tanh(
        float(
            last["ema50_200"]
        )
        * 20
    ) * 0.15

    # Momentum
    score += np.tanh(
        float(
            last["ret24"]
        )
        * 25
    ) * 0.20

    # RSI
    rsi_value = float(
        last["rsi14"]
    )

    score += np.clip(
        (
            rsi_value
            - 50
        )
        / 25,
        -1,
        1,
    ) * 0.15

    # MACD
    score += np.tanh(
        float(
            last["macd_hist"]
        )
        * 100
    ) * 0.10

    # Bollinger
    bb = float(
        last["bb_position"]
    )

    score += np.clip(
        (
            bb
            - 0.5
        )
        * 2,
        -1,
        1,
    ) * 0.10

    # Volume
    volume = float(
        last["volume_ratio"]
    )

    score += np.tanh(
        volume - 1
    ) * 0.05

    score = float(
        np.clip(
            score,
            -1,
            1,
        )
    )

    if score >= 0:

        signal = "BUY"

        confidence = (
            0.50
            + abs(score)
            * 0.49
        )

    else:

        signal = "SELL"

        confidence = (
            0.50
            + abs(score)
            * 0.49
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "p_buy": (
            confidence
            if signal == "BUY"
            else 1 - confidence
        ),
        "p_sell": (
            confidence
            if signal == "SELL"
            else 1 - confidence
        ),
        "ai_score": (
            confidence * 100
        ),
        "buy_score": (
            confidence * 100
            if signal == "BUY"
            else (
                100
                - confidence * 100
            )
        ),
        "sell_score": (
            confidence * 100
            if signal == "SELL"
            else (
                100
                - confidence * 100
            )
        ),
        "mode":
            "TECHNICAL_FALLBACK",
        "test_accuracy":
            np.nan,
        "test_balanced_accuracy":
            np.nan,
        "train_rows":
            0,
        "validation_rows":
            0,
        "test_rows":
            0,
    }


# ============================================================
# AI PREDICTION
# ============================================================

def predict(
    df,
    train_limit,
):

    try:

        f = features(
            df
        )

        labeled = make_target(
            f
        )

        usable = (
            labeled
            .dropna(
                subset=FEATURES
                + ["target"]
            )
            .copy()
        )

        # IMPORTANT:
        # We don't require +10% or -10% hit.
        # All usable rows can train.

        if len(usable) < MIN_CANDLES:

            raise RuntimeError(
                f"AI için {len(usable)} temiz "
                f"satır kaldı."
            )

        usable = (
            usable
            .tail(
                min(
                    int(train_limit),
                    len(usable),
                )
            )
            .reset_index(
                drop=True
            )
        )

        if (
            usable["target"]
            .nunique()
            < 2
        ):

            raise RuntimeError(
                "Tek sınıf oluştu."
            )

        n = len(
            usable
        )

        train_end = int(
            n * 0.70
        )

        test_start = int(
            n * 0.85
        )

        train = usable[
            :train_end
        ]

        validation = usable[
            train_end:test_start
        ]

        test = usable[
            test_start:
        ]

        if len(test) < 10:

            raise RuntimeError(
                "Test datası çok küçük."
            )

        X_train = train[
            FEATURES
        ].astype(float)

        y_train = train[
            "target"
        ].astype(int)

        X_test = test[
            FEATURES
        ].astype(float)

        y_test = test[
            "target"
        ].astype(int)

        X_train, y_train = balance(
            X_train,
            y_train,
        )

        clf = model()

        clf.fit(
            X_train,
            y_train,
        )

        pred = clf.predict(
            X_test
        )

        accuracy = (
            accuracy_score(
                y_test,
                pred,
            )
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                pred,
            )
        )

        latest = (
            labeled
            .dropna(
                subset=FEATURES
            )
            .iloc[-1]
        )

        X_latest = pd.DataFrame(
            [
                latest[
                    FEATURES
                ].astype(float)
            ]
        )

        probs = clf.predict_proba(
            X_latest
        )[0]

        mapping = {
            int(c): float(p)
            for c, p in zip(
                clf.classes_,
                probs,
            )
        }

        p_sell = mapping.get(
            0,
            0.0,
        )

        p_buy = mapping.get(
            1,
            0.0,
        )

        if p_buy >= p_sell:

            signal = "BUY"
            confidence = p_buy

        else:

            signal = "SELL"
            confidence = p_sell

        score_direction = (
            p_buy
            - p_sell
        )

        buy_score = np.clip(
            50
            + score_direction
            * 50,
            0,
            100,
        )

        sell_score = np.clip(
            50
            - score_direction
            * 50,
            0,
            100,
        )

        return {
            "signal": signal,

            "confidence":
                confidence,

            "p_buy":
                p_buy,

            "p_sell":
                p_sell,

            "ai_score":
                confidence * 100,

            "buy_score":
                float(
                    buy_score
                ),

            "sell_score":
                float(
                    sell_score
                ),

            "mode":
                "AI",

            "test_accuracy":
                accuracy,

            "test_balanced_accuracy":
                balanced_accuracy,

            "train_rows":
                len(train),

            "validation_rows":
                len(validation),

            "test_rows":
                len(test),

            "buy_hit_rate":
                float(
                    labeled[
                        "buy_hit"
                    ].mean()
                ),

            "sell_hit_rate":
                float(
                    labeled[
                        "sell_hit"
                    ].mean()
                ),
        }

    except Exception as e:

        result = fallback(
            df
        )

        result["ai_error"] = str(
            e
        )

        return result


# ============================================================
# FULL SYMBOL ANALYSIS
# ============================================================

def analyze(
    symbol_name,
    df,
    train_limit,
):

    started = time.time()

    df = clean_df(
        df
    )

    if len(df) < MIN_CANDLES:

        raise RuntimeError(
            f"{symbol_name}: "
            f"{len(df)} candle."
        )

    prediction = predict(
        df,
        train_limit,
    )

    f = features(
        df
    )

    last = f.iloc[-1]

    price = float(
        df["close"].iloc[-1]
    )

    future_high = pd.concat(
        [
            df["high"].shift(-1),
            df["high"].shift(-2),
        ],
        axis=1,
    ).iloc[-1].max()

    future_low = pd.concat(
        [
            df["low"].shift(-1),
            df["low"].shift(-2),
        ],
        axis=1,
    ).iloc[-1].min()

    projected_up = (
        future_high
        / price
        - 1
    )

    projected_down = (
        future_low
        / price
        - 1
    )

    return {
        "symbol":
            symbol_name,

        "signal":
            prediction["signal"],

        "confidence":
            prediction["confidence"],

        "p_buy":
            prediction["p_buy"],

        "p_sell":
            prediction["p_sell"],

        "ai_score":
            prediction["ai_score"],

        "buy_score":
            prediction["buy_score"],

        "sell_score":
            prediction["sell_score"],

        "mode":
            prediction["mode"],

        "price":
            price,

        "projected_upside":
            projected_up,

        "projected_downside":
            projected_down,

        "rsi":
            float(last["rsi14"]),

        "adx":
            float(last["adx"]),

        "bb_position":
            float(
                last["bb_position"]
            ),

        "atr_pct":
            float(
                last["atr_pct"]
            ),

        "volume_ratio":
            float(
                last["volume_ratio"]
            ),

        "test_accuracy":
            prediction[
                "test_accuracy"
            ],

        "test_balanced_accuracy":
            prediction[
                "test_balanced_accuracy"
            ],

        "train_rows":
            prediction[
                "train_rows"
            ],

        "validation_rows":
            prediction[
                "validation_rows"
            ],

        "test_rows":
            prediction[
                "test_rows"
            ],

        "candles":
            len(df),

        "buy_hit_rate":
            prediction.get(
                "buy_hit_rate",
                np.nan,
            ),

        "sell_hit_rate":
            prediction.get(
                "sell_hit_rate",
                np.nan,
            ),

        "processing_seconds":
            time.time()
            - started,

        "last_candle":
            str(
                df[
                    "timestamp"
                ].iloc[-1]
            ),
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)

provider = st.sidebar.selectbox(
    "Data Provider",
    [
        "Binance REST",
        "Kraken REST",
        "Historical CSV",
    ],
)

all_symbols = st.sidebar.checkbox(
    "🌐 TÜM AKTİF COINLER",
    value=True,
)

max_symbols = st.sidebar.number_input(
    "Maksimum coin",
    min_value=1,
    max_value=5000,
    value=2000,
    step=100,
)

candles = st.sidebar.number_input(
    "Historical Candles",
    min_value=300,
    max_value=MAX_CANDLES,
    value=DEFAULT_CANDLES,
    step=100,
)

train_limit = st.sidebar.number_input(
    "AI Training Candles",
    min_value=300,
    max_value=100000,
    value=DEFAULT_TRAIN,
    step=500,
)

manual_symbols = st.sidebar.text_area(
    "Manuel semboller",
    value=(
        "BTCUSDT\n"
        "ETHUSDT\n"
        "BNBUSDT\n"
        "SOLUSDT\n"
        "XRPUSDT"
    ),
    height=130,
)

uploaded = None

if provider == "Historical CSV":

    uploaded = st.sidebar.file_uploader(
        "Historical 15M CSV",
        type=["csv"],
    )

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    "⏱ Timeframe: **15M**"
)

st.sidebar.write(
    "🎯 BUY: **+10% / 30M**"
)

st.sidebar.write(
    "🎯 SELL: **-10% / 30M**"
)

st.sidebar.write(
    "❌ HOLD: **YOK**"
)


# ============================================================
# SYMBOL RESOLUTION
# ============================================================

def get_symbols():

    if provider == "Binance REST":

        symbols = (
            binance_symbols()
        )

        if all_symbols:

            return symbols[
                :int(max_symbols)
            ]

        return [
            x.strip().upper()
            for x in manual_symbols.splitlines()
            if x.strip()
        ]

    if provider == "Kraken REST":

        pairs = (
            kraken_symbols()
        )

        if all_symbols:

            return pairs[
                :int(max_symbols)
            ]

        wanted = {
            x.strip().upper()
            for x in manual_symbols.splitlines()
            if x.strip()
        }

        return [
            p
            for p in pairs
            if (
                p["name"].upper()
                in wanted
                or p["api"].upper()
                in wanted
            )
        ]

    return [
        {
            "name": "CSV",
            "api": "CSV",
        }
    ]


# ============================================================
# DATA FOR SYMBOL
# ============================================================

def get_data(
    symbol,
):

    if provider == "Binance REST":

        return (
            symbol,
            binance_history(
                symbol,
                int(candles),
            )
        )

    if provider == "Kraken REST":

        return (
            symbol["name"],
            kraken_history(
                symbol["api"]
            )
        )

    return (
        "CSV",
        csv_data(
            uploaded
        )
    )


# ============================================================
# SCAN
# ============================================================

def scan():

    st.session_state.results = (
        pd.DataFrame()
    )

    st.session_state.errors = []

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if provider == "Historical CSV":

        if uploaded is None:

            st.error(
                "CSV yüklemediniz."
            )

            return

    # --------------------------------------------------------
    # Symbols
    # --------------------------------------------------------

    try:

        symbols = get_symbols()

    except Exception as e:

        st.error(
            f"Coin listesi alınamadı: {e}"
        )

        return

    if not symbols:

        st.error(
            "Hiç coin bulunamadı."
        )

        return

    st.info(
        f"Toplam {len(symbols):,} coin "
        "taranacak."
    )

    results = []

    errors = []

    progress = st.progress(
        0
    )

    status = st.empty()

    for i, symbol in enumerate(
        symbols
    ):

        if provider == "Kraken REST":

            name = symbol[
                "name"
            ]

        elif provider == "Binance REST":

            name = symbol

        else:

            name = "CSV"

        status.write(
            f"🔎 {name} "
            f"({i + 1}/{len(symbols)})"
        )

        try:

            symbol_name, data = (
                get_data(
                    symbol
                )
            )

            result = analyze(
                symbol_name,
                data,
                int(train_limit),
            )

            results.append(
                result
            )

        except Exception as e:

            errors.append(
                {
                    "symbol":
                        name,

                    "error":
                        str(e),
                }
            )

        progress.progress(
            (i + 1)
            / len(symbols)
        )

    progress.empty()

    status.empty()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if results:

        result_df = pd.DataFrame(
            results
        )

        # Absolutely no HOLD
        result_df = result_df[
            result_df[
                "signal"
            ].isin(
                [
                    "BUY",
                    "SELL",
                ]
            )
        ]

        result_df = (
            result_df
            .sort_values(
                "ai_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        st.session_state.results = (
            result_df
        )

    st.session_state.errors = (
        errors
    )

    st.session_state.last_scan = (
        datetime.now(
            timezone.utc
        )
    )

    st.success(
        f"Tarama tamamlandı: "
        f"{len(results)} coin sonuçlandı, "
        f"{len(errors)} veri hatası."
    )


# ============================================================
# SCAN BUTTON
# ============================================================

if st.sidebar.button(
    "🚀 TÜM COİNLERİ TARA",
    type="primary",
    use_container_width=True,
):

    scan()


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.results
)

if results.empty:

    st.info(
        "Henüz tarama yapılmadı."
    )

else:

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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

    ai_count = int(
        (
            results[
                "mode"
            ]
            == "AI"
        ).sum()
    )

    fallback_count = int(
        (
            results[
                "mode"
            ]
            == "TECHNICAL_FALLBACK"
        ).sum()
    )

    a, b, c, d, e = (
        st.columns(5)
    )

    with a:
        st.metric(
            "Toplam",
            len(results),
        )

    with b:
        st.metric(
            "BUY",
            buy_count,
        )

    with c:
        st.metric(
            "SELL",
            sell_count,
        )

    with d:
        st.metric(
            "AI",
            ai_count,
        )

    with e:
        st.metric(
            "Fallback",
            fallback_count,
        )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    st.subheader(
        "🟢 TOP 10 BUY"
    )

    buy = (
        results[
            results[
                "signal"
            ]
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

    if not buy.empty:

        table = buy[
            [
                "symbol",
                "signal",
                "mode",
                "ai_score",
                "confidence",
                "p_buy",
                "projected_upside",
                "rsi",
                "adx",
                "volume_ratio",
                "price",
                "candles",
            ]
        ].copy()

        for c in [
            "confidence",
            "p_buy",
            "projected_upside",
        ]:
            table[c] *= 100

        st.dataframe(
            table.round(4),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "BUY sonucu yok."
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    st.subheader(
        "🔴 TOP 10 SELL"
    )

    sell = (
        results[
            results[
                "signal"
            ]
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

    if not sell.empty:

        table = sell[
            [
                "symbol",
                "signal",
                "mode",
                "ai_score",
                "confidence",
                "p_sell",
                "projected_downside",
                "rsi",
                "adx",
                "volume_ratio",
                "price",
                "candles",
            ]
        ].copy()

        for c in [
            "confidence",
            "p_sell",
            "projected_downside",
        ]:
            table[c] *= 100

        st.dataframe(
            table.round(4),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "SELL sonucu yok."
        )

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    with st.expander(
        "📋 TÜM COİNLER"
    ):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # DETAIL
    # --------------------------------------------------------

    st.subheader(
        "🔎 Coin Detail"
    )

    selected = st.selectbox(
        "Coin",
        results[
            "symbol"
        ].tolist(),
    )

    row = results[
        results[
            "symbol"
        ]
        == selected
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
            f"{row['ai_score']:.2f}",
        )

    with d:

        move = (
            row["projected_upside"]
            if row["signal"]
            == "BUY"
            else
            row["projected_downside"]
        )

        st.metric(
            "Projected 30m Move",
            f"{move * 100:.2f}%",
        )

    detail = pd.DataFrame(
        {
            "Metric": [
                "Symbol",
                "Signal",
                "Model Mode",
                "Price",

                "BUY Probability",
                "SELL Probability",

                "BUY Score",
                "SELL Score",
                "AI Score",

                "Confidence",

                "Target",
                "Maximum Horizon",

                "Projected Upside",
                "Projected Downside",

                "RSI 14",
                "ADX",
                "Bollinger Position",
                "ATR %",
                "Volume Ratio",

                "Historical +10% Hit Rate",
                "Historical -10% Hit Rate",

                "Test Accuracy",
                "Test Balanced Accuracy",

                "Training Rows",
                "Validation Rows",
                "Test Rows",

                "Candles",
                "Last Candle",
            ],

            "Value": [
                row["symbol"],
                row["signal"],
                row["mode"],
                row["price"],

                f"{row['p_buy'] * 100:.2f}%",
                f"{row['p_sell'] * 100:.2f}%",

                f"{row['buy_score']:.2f}",
                f"{row['sell_score']:.2f}",
                f"{row['ai_score']:.2f}",

                f"{row['confidence'] * 100:.2f}%",

                "+10% BUY / -10% SELL",
                "30 minutes",

                f"{row['projected_upside'] * 100:.2f}%",
                f"{row['projected_downside'] * 100:.2f}%",

                f"{row['rsi']:.2f}",
                f"{row['adx']:.2f}",
                f"{row['bb_position']:.4f}",
                f"{row['atr_pct'] * 100:.3f}%",
                f"{row['volume_ratio']:.3f}",

                f"{row['buy_hit_rate'] * 100:.3f}%",
                f"{row['sell_hit_rate'] * 100:.3f}%",

                (
                    f"{row['test_accuracy'] * 100:.2f}%"
                    if pd.notna(
                        row["test_accuracy"]
                    )
                    else "Fallback"
                ),

                (
                    f"{row['test_balanced_accuracy'] * 100:.2f}%"
                    if pd.notna(
                        row[
                            "test_balanced_accuracy"
                        ]
                    )
                    else "Fallback"
                ),

                row["train_rows"],
                row["validation_rows"],
                row["test_rows"],

                row["candles"],
                row["last_candle"],
            ],
        }
    )

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Sonuçları CSV indir",

        data=(
            results
            .to_csv(
                index=False
            )
            .encode("utf-8")
        ),

        file_name=(
            "all_coins_scalping_ai.csv"
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
        "⚠️ Veri alınamayan coinler"
    )

    st.caption(
        "Bunlar model hatası nedeniyle "
        "çıkarılan coinler değildir. "
        "Borsadan candle alınamayan coinlerdir."
    )

    error_df = pd.DataFrame(
        st.session_state.errors
    )

    st.dataframe(
        error_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# INFO
# ============================================================

with st.expander(
    "ℹ️ Sistem"
):

    st.markdown(
        """
### Tarama

Sistem aktif Spot coin listesini otomatik alır.

Binance için:

- TRADING
- Spot
- USDT
- USDC

pariteleri alınır.

### AI hedefi

15 dakikalık candle kullanılır.

Sonraki 2 candle:

**15 + 15 = 30 dakika**

incelenir.

BUY:

**+10%**

SELL:

**-10%**

### HOLD

HOLD kesinlikle yoktur.

Her coin:

**BUY veya SELL**

olarak sonuçlanır.

### Fallback

Bir coin için AI eğitimi başarısız olursa coin
silinmez.

Teknik göstergelerle ikinci bir BUY/SELL
hesaplaması yapılır ve:

`TECHNICAL_FALLBACK`

olarak işaretlenir.

Bu nedenle AI'da problem yaşayan coinler de
sonuç tablosunda kalır.

### Emir

Bu uygulama hiçbir emir göndermez.
"""
    )


st.divider()

st.caption(
    "Spot Scalping AI Scanner • "
    "15M • +10% BUY / -10% SELL • "
    "30 Minute Horizon • "
    "BUY / SELL Only • "
    "No Order Execution"
)
