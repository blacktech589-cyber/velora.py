import os
import math
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# SAFE PYTORCH IMPORT
# ============================================================

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = ""

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

except Exception as exc:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)

    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Spot AI Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
).rstrip("/")

INTERVAL = "15m"

MAX_KLINE_LIMIT = 1000

SEQ_LEN = 128

FUTURE_BARS = 12

BUY_SELL_THRESHOLD = 0.003

DEVICE = (
    "cuda"
    if TORCH_AVAILABLE and torch.cuda.is_available()
    else "cpu"
)

REQUEST_TIMEOUT = 30

EPOCHS = 3

BATCH_SIZE = 128

LEARNING_RATE = 0.0001


# ============================================================
# SESSION STATE
# ============================================================

if "cache" not in st.session_state:
    st.session_state.cache = {}

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "error" not in st.session_state:
    st.session_state.error = ""


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:

        return default


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(maximum, value)
    )


def now_utc():

    return datetime.now(
        timezone.utc
    )


# ============================================================
# HTTP
# ============================================================

http = requests.Session()

http.headers.update({
    "User-Agent": "Spot-AI-Scanner/1.0"
})


def api_get(endpoint, params=None):

    url = BASE_URL + endpoint

    response = http.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 451:

        raise RuntimeError(
            "HTTP 451: Binance API bu ortamdan "
            "erişime izin vermiyor."
        )

    if response.status_code == 429:

        raise RuntimeError(
            "HTTP 429: Binance API rate limit."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXCHANGE INFO
# ============================================================

@st.cache_data(ttl=900)
def exchange_info():

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

        if quote not in [
            "USDT",
            "USDC",
            "FDUSD"
        ]:

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
# KLINES
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
            MAX_KLINE_LIMIT
        )
    }

    if start_time is not None:

        params["startTime"] = int(
            start_time
        )

    if end_time is not None:

        params["endTime"] = int(
            end_time
        )

    return api_get(
        "/api/v3/klines",
        params
    )


def klines_dataframe(data):

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

    numeric = [
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

    for column in numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["open_time"] = pd.to_numeric(
        df["open_time"],
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
        "open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# DOWNLOAD HISTORY
# ============================================================

def download_history(
    symbol,
    target,
    progress=None
):

    pieces = []

    remaining = int(
        target
    )

    end_time = None

    completed = 0

    while remaining > 0:

        limit = min(
            MAX_KLINE_LIMIT,
            remaining
        )

        data = get_klines(
            symbol,
            limit,
            end_time=end_time
        )

        if not data:

            break

        part = klines_dataframe(
            data
        )

        if part.empty:

            break

        pieces.append(
            part
        )

        received = len(
            part
        )

        completed += received

        remaining -= received

        oldest = int(
            part[
                "open_time"
            ].min()
        )

        end_time = (
            oldest - 1
        )

        if progress:

            progress.progress(
                min(
                    completed / target,
                    1.0
                )
            )

        if received < limit:

            break

        time.sleep(
            0.05
        )

    if not pieces:

        return pd.DataFrame()

    df = pd.concat(
        pieces,
        ignore_index=True
    )

    df = df.drop_duplicates(
        "open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.tail(
        target
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# UPDATE CACHE
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

    last_time = int(
        existing[
            "open_time"
        ].max()
    )

    data = get_klines(
        symbol,
        1000,
        start_time=last_time + 1
    )

    if not data:

        return existing

    new_data = klines_dataframe(
        data
    )

    if new_data.empty:

        return existing

    df = pd.concat(
        [
            existing,
            new_data
        ],
        ignore_index=True
    )

    df = df.drop_duplicates(
        "open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.tail(
        target
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# INDICATORS
# ============================================================

def EMA(series, period):

    return series.ewm(
        span=period,
        adjust=False
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


def ATR(df, period=14):

    previous = df[
        "close"
    ].shift(1)

    a = (
        df["high"]
        - df["low"]
    )

    b = (
        df["high"]
        - previous
    ).abs()

    c = (
        df["low"]
        - previous
    ).abs()

    tr = pd.concat(
        [
            a,
            b,
            c
        ],
        axis=1
    ).max(
        axis=1
    )

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def MACD(series):

    fast = EMA(
        series,
        12
    )

    slow = EMA(
        series,
        26
    )

    line = (
        fast - slow
    )

    signal = EMA(
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


def ADX(df, period=14):

    high = df["high"]

    low = df["low"]

    previous_high = high.shift(1)

    previous_low = low.shift(1)

    up = (
        high
        - previous_high
    )

    down = (
        previous_low
        - low
    )

    plus_dm = up.where(
        (
            up > down
        )
        & (
            up > 0
        ),
        0
    )

    minus_dm = down.where(
        (
            down > up
        )
        & (
            down > 0
        ),
        0
    )

    previous_close = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
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
# FEATURES
# ============================================================

FEATURES = [
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "ret24",
    "ret48",
    "ema5",
    "ema20",
    "ema50",
    "ema100",
    "ema200",
    "ema500",
    "ema800",
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
    "high_distance",
    "low_distance",
    "drawdown",
    "price_z"
]


def make_features(df):

    df = df.copy()

    close = df["close"]

    high = df["high"]

    low = df["low"]

    open_price = df["open"]

    volume = df["volume"]

    # Returns
    df["ret1"] = close.pct_change(1)

    df["ret3"] = close.pct_change(3)

    df["ret6"] = close.pct_change(6)

    df["ret12"] = close.pct_change(12)

    df["ret24"] = close.pct_change(24)

    df["ret48"] = close.pct_change(48)

    # EMAs
    e5 = EMA(
        close,
        5
    )

    e20 = EMA(
        close,
        20
    )

    e50 = EMA(
        close,
        50
    )

    e100 = EMA(
        close,
        100
    )

    e200 = EMA(
        close,
        200
    )

    e500 = EMA(
        close,
        500
    )

    e800 = EMA(
        close,
        800
    )

    df["ema5"] = (
        close / (
            e5 + 1e-12
        )
    ) - 1

    df["ema20"] = (
        close / (
            e20 + 1e-12
        )
    ) - 1

    df["ema50"] = (
        close / (
            e50 + 1e-12
        )
    ) - 1

    df["ema100"] = (
        close / (
            e100 + 1e-12
        )
    ) - 1

    df["ema200"] = (
        close / (
            e200 + 1e-12
        )
    ) - 1

    df["ema500"] = (
        close / (
            e500 + 1e-12
        )
    ) - 1

    df["ema800"] = (
        close / (
            e800 + 1e-12
        )
    ) - 1

    df["ema20_50"] = (
        e20 / (
            e50 + 1e-12
        )
    ) - 1

    df["ema50_200"] = (
        e50 / (
            e200 + 1e-12
        )
    ) - 1

    # RSI
    df["rsi7"] = RSI(
        close,
        7
    )

    df["rsi14"] = RSI(
        close,
        14
    )

    df["rsi21"] = RSI(
        close,
        21
    )

    # MACD
    m_line, m_signal, m_hist = MACD(
        close
    )

    df["macd"] = (
        m_line
        / (
            close + 1e-12
        )
    )

    df["macd_signal"] = (
        m_signal
        / (
            close + 1e-12
        )
    )

    df["macd_hist"] = (
        m_hist
        / (
            close + 1e-12
        )
    )

    # Bollinger
    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    upper = (
        middle
        + 2 * std
    )

    lower = (
        middle
        - 2 * std
    )

    df["bb_position"] = (
        close
        - lower
    ) / (
        upper
        - lower
        + 1e-12
    )

    df["bb_width"] = (
        upper - lower
    ) / (
        middle.abs()
        + 1e-12
    )

    # ATR
    atr_value = ATR(
        df
    )

    df["atr_pct"] = (
        atr_value
        / (
            close + 1e-12
        )
    )

    # ADX
    df["adx"] = ADX(
        df
    )

    # Volume
    volume_average = (
        volume.rolling(
            20
        ).mean()
    )

    df["volume_ratio"] = (
        volume
        / (
            volume_average
            + 1e-12
        )
    )

    # Candle
    df["body_pct"] = (
        close
        - open_price
    ) / (
        close + 1e-12
    )

    df["range_pct"] = (
        high - low
    ) / (
        close + 1e-12
    )

    # Volatility
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

    # High / Low
    rolling_high = high.rolling(
        100
    ).max()

    rolling_low = low.rolling(
        100
    ).min()

    df["high_distance"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    df["low_distance"] = (
        close
        / (
            rolling_low
            + 1e-12
        )
    ) - 1

    # Drawdown
    df["drawdown"] = (
        close
        / (
            rolling_high
            + 1e-12
        )
    ) - 1

    # Z-score
    mean = close.rolling(
        100
    ).mean()

    std = close.rolling(
        100
    ).std()

    df["price_z"] = (
        close
        - mean
    ) / (
        std + 1e-12
    )

    return df


# ============================================================
# TARGET
# ============================================================

def make_targets(df):

    df = df.copy()

    future = (
        df["close"].shift(
            -FUTURE_BARS
        )
        / df["close"]
    ) - 1

    target = np.ones(
        len(df)
    )

    target[
        future > BUY_SELL_THRESHOLD
    ] = 2

    target[
        future < -BUY_SELL_THRESHOLD
    ] = 0

    target[
        future.isna()
    ] = np.nan

    df["future_return"] = future

    df["target"] = target

    return df


# ============================================================
# SCALER
# ============================================================

class Scaler:

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
# MODEL
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

            div = torch.exp(
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
                position * div
            )

            pe[:, 1::2] = torch.cos(
                position * div
            )

            pe = pe.unsqueeze(0)

            self.register_buffer(
                "pe",
                pe
            )

        def forward(self, x):

            length = x.shape[1]

            return (
                x
                + self.pe[
                    :, :length
                ]
            )


    class AIModel(nn.Module):

        def __init__(
            self,
            input_size
        ):

            super().__init__()

            d_model = 128

            self.input = nn.Linear(
                input_size,
                d_model
            )

            self.position = PositionalEncoding(
                d_model
            )

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=512,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )

            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=4
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
                nn.Dropout(0.15),
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
                nn.Dropout(0.15),
                nn.Linear(
                    64,
                    1
                )
            )

        def forward(self, x):

            x = self.input(
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

            x = x.mean(
                dim=1
            )

            classification = (
                self.classifier(x)
            )

            regression = (
                self.regressor(x)
                .squeeze(-1)
            )

            return (
                classification,
                regression
            )

else:

    PositionalEncoding = None
    AIModel = None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    data,
    scaler
):

    x = data[
        FEATURES
    ].values.astype(
        np.float32
    )

    y = data[
        "target"
    ].values.astype(
        np.int64
    )

    r = data[
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

    classes = []

    returns = []

    start = max(
        SEQ_LEN,
        len(x) - 2500
    )

    for i in range(
        start,
        len(x)
    ):

        sequence = x[
            i - SEQ_LEN:i
        ]

        if len(sequence) != SEQ_LEN:

            continue

        sequences.append(
            sequence
        )

        classes.append(
            y[i]
        )

        returns.append(
            r[i]
        )

    if not sequences:

        return None

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            classes,
            dtype=np.int64
        ),
        np.asarray(
            returns,
            dtype=np.float32
        )
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train(
    data
):

    if not TORCH_AVAILABLE:

        raise RuntimeError(
            "PyTorch bulunamadı: "
            + TORCH_IMPORT_ERROR
        )

    if len(data) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            "Eğitim için yeterli veri yok."
        )

    scaler = Scaler()

    scaler.fit(
        data[
            FEATURES
        ].values.astype(
            np.float32
        )
    )

    prepared = prepare_data(
        data,
        scaler
    )

    if prepared is None:

        raise RuntimeError(
            "Training sequence oluşturulamadı."
        )

    x, y, r = prepared

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    r = torch.tensor(
        r,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        x,
        y,
        r
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AIModel(
        len(FEATURES)
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    class_loss = nn.CrossEntropyLoss()

    regression_loss = nn.HuberLoss()

    model.train()

    for _ in range(
        EPOCHS
    ):

        for batch_x, batch_y, batch_r in loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            batch_r = batch_r.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits, prediction = model(
                batch_x
            )

            loss1 = class_loss(
                logits,
                batch_y
            )

            loss2 = regression_loss(
                prediction,
                batch_r
            )

            loss = (
                loss1
                + 2.0 * loss2
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
# PREDICT
# ============================================================

def predict(
    model,
    scaler,
    data
):

    values = data[
        FEATURES
    ].values.astype(
        np.float32
    )

    values = scaler.transform(
        values
    ).astype(
        np.float32
    )

    sequence = values[
        -SEQ_LEN:
    ]

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        logits, regression = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

        expected = (
            regression.item()
        )

    sell = safe_float(
        probabilities[0]
    )

    hold = safe_float(
        probabilities[1]
    )

    buy = safe_float(
        probabilities[2]
    )

    maximum = max(
        sell,
        hold,
        buy
    )

    if buy == maximum:

        signal = "BUY"

    elif sell == maximum:

        signal = "SELL"

    else:

        signal = "HOLD"

    return {
        "signal": signal,
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "confidence": maximum,
        "expected": expected
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze(
    symbol,
    candles
):

    features = make_features(
        candles
    )

    features = make_targets(
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

    if len(clean) < (
        SEQ_LEN + 100
    ):

        raise RuntimeError(
            f"{symbol}: temiz veri yetersiz."
        )

    model, scaler = train(
        clean
    )

    prediction = predict(
        model,
        scaler,
        clean
    )

    row = features.dropna(
        subset=FEATURES
    ).iloc[-1]

    price = safe_float(
        row["close"]
    )

    rsi_value = safe_float(
        row["rsi14"],
        50
    )

    adx_value = safe_float(
        row["adx"]
    )

    volume_ratio = safe_float(
        row["volume_ratio"],
        1
    )

    volatility = safe_float(
        row["volatility24"]
    )

    trend = safe_float(
        row["ema50_200"]
    )

    momentum = safe_float(
        row["ret24"]
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

    liquidity_score = clamp(
        volume_ratio * 40,
        0,
        100
    )

    risk_score = clamp(
        50
        + volatility * 500
        + safe_float(
            row["atr_pct"]
        ) * 300,
        0,
        100
    )

    score = (
        (
            prediction["buy"]
            - prediction["sell"]
        )
        * 100
        * 0.35
    )

    score += (
        prediction["confidence"]
        * 100
        * 0.20
    )

    score += (
        trend_score
        * 0.15
    )

    score += (
        momentum_score
        * 0.10
    )

    score += (
        liquidity_score
        * 0.05
    )

    score += (
        prediction["expected"]
        * 10000
        * 0.15
    )

    score -= (
        risk_score
        * 0.05
    )

    return {
        "symbol": symbol,
        "signal": prediction["signal"],
        "ai_score": score,
        "confidence_pct": prediction["confidence"] * 100,
        "buy_probability_pct": prediction["buy"] * 100,
        "sell_probability_pct": prediction["sell"] * 100,
        "hold_probability_pct": prediction["hold"] * 100,
        "expected_return_pct": prediction["expected"] * 100,
        "price": price,
        "rsi": rsi_value,
        "adx": adx_value,
        "volume_ratio": volume_ratio,
        "volatility_pct": volatility * 100,
        "momentum_pct": momentum * 100,
        "trend_pct": trend * 100,
        "risk_score": risk_score,
        "liquidity_score": liquidity_score,
        "candle_count": len(candles)
    }


# ============================================================
# SCAN
# ============================================================

def scan(
    symbols,
    target,
    progress,
    status
):

    output = []

    errors = []

    total = len(
        symbols
    )

    for index, symbol in enumerate(
        symbols,
        1
    ):

        status.info(
            f"{symbol} taranıyor "
            f"({index}/{total})"
        )

        progress.progress(
            (index - 1) / total
        )

        try:

            old = (
                st.session_state.cache
                .get(symbol)
            )

            if (
                old is None
                or len(old) < target
            ):

                candles = download_history(
                    symbol,
                    target
                )

            else:

                candles = update_history(
                    symbol,
                    old,
                    target
                )

            if candles.empty:

                raise RuntimeError(
                    "Candle verisi boş."
                )

            st.session_state.cache[
                symbol
            ] = candles

            result = analyze(
                symbol,
                candles
            )

            output.append(
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

        st.session_state.error = (
            "\n".join(
                errors
            )
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Binance Spot AI Deep Learning Scanner"
)

st.write(
    "15m Spot market • Transformer • "
    "500K candle target • Top 10 BUY/SELL"
)


# ============================================================
# TORCH CHECK
# ============================================================

if not TORCH_AVAILABLE:

    st.error(
        "PyTorch yüklenemedi."
    )

    st.code(
        TORCH_IMPORT_ERROR
        or "Bilinmeyen hata"
    )

    st.markdown(
        """
requirements.txt:

streamlit>=1.37,<2.0
requests>=2.32
pandas>=2.2
numpy>=1.26,<2.0
torch==2.4.1

runtime.txt:

python-3.11
"""
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Ayarlar"
)

candle_target = st.sidebar.number_input(
    "Hedef mum",
    min_value=10000,
    max_value=500000,
    value=500000,
    step=10000
)

symbol_count = st.sidebar.number_input(
    "Parite sayısı",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

automatic = st.sidebar.checkbox(
    "15 dakikada yenile",
    value=True
)

st.sidebar.write(
    f"Device: {DEVICE}"
)

st.sidebar.write(
    f"Interval: {INTERVAL}"
)

st.sidebar.write(
    f"Sequence: {SEQ_LEN}"
)

st.sidebar.write(
    f"Future bars: {FUTURE_BARS}"
)


# ============================================================
# SCANNER FUNCTION
# ============================================================

def scanner():

    st.subheader(
        "🔎 Scanner"
    )

    if st.button(
        "🚀 ŞİMDİ TARA",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            status.info(
                "Spot sembolleri alınıyor..."
            )

            all_symbols = exchange_info()

            usdt_symbols = [
                x
                for x in all_symbols
                if x.endswith(
                    "USDT"
                )
            ]

            selected = usdt_symbols[
                :int(symbol_count)
            ]

            if not selected:

                raise RuntimeError(
                    "USDT paritesi bulunamadı."
                )

            st.info(
                "Taranıyor: "
                + ", ".join(selected)
            )

            result = scan(
                selected,
                int(candle_target),
                progress,
                status
            )

            st.session_state.results = (
                result
            )

            st.session_state.last_scan = (
                now_utc().strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )

            status.success(
                "Tarama tamamlandı."
            )

        except Exception as exc:

            st.error(
                f"Tarama hatası: {exc}"
            )

            st.session_state.error = (
                traceback.format_exc()
            )

            if "451" in str(exc):

                st.warning(
                    "Binance API bu ortamdan "
                    "erişime kapalı olabilir."
                )

    df = st.session_state.results

    if df.empty:

        st.info(
            "Henüz sonuç yok. "
            "ŞİMDİ TARA butonuna basın."
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

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "BUY",
        len(buy)
    )

    c2.metric(
        "SELL",
        len(sell)
    )

    c3.metric(
        "HOLD",
        len(hold)
    )

    c4.metric(
        "Son tarama",
        st.session_state.last_scan
        or "-"
    )

    # ========================================================
    # TOP BUY
    # ========================================================

    st.header(
        "🟢 TOP 10 BUY"
    )

    if not buy.empty:

        buy = buy.sort_values(
            "ai_score",
            ascending=False
        ).head(
            10
        )

        st.dataframe(
            buy,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "BUY sonucu yok."
        )

    # ========================================================
    # TOP SELL
    # ========================================================

    st.header(
        "🔴 TOP 10 SELL"
    )

    if not sell.empty:

        sell = sell.sort_values(
            "ai_score",
            ascending=False
        ).head(
            10
        )

        st.dataframe(
            sell,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "SELL sonucu yok."
        )

    # ========================================================
    # ALL
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
        csv,
        "spot_ai_results.csv",
        "text/csv"
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

        a, b, c = st.columns(
            3
        )

        a.metric(
            "Signal",
            row["signal"]
        )

        b.metric(
            "AI Score",
            f"{row['ai_score']:.2f}"
        )

        c.metric(
            "Confidence",
            f"{row['confidence_pct']:.2f}%"
        )

        st.json(
            row.to_dict()
        )

    # ========================================================
    # CACHE
    # ========================================================

    st.header(
        "🗄️ Cache"
    )

    cache_info = []

    for symbol, candles in (
        st.session_state.cache.items()
    ):

        cache_info.append(
            {
                "symbol": symbol,
                "candles": len(candles)
            }
        )

    if cache_info:

        st.dataframe(
            pd.DataFrame(
                cache_info
            ),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # ERROR
    # ========================================================

    if st.session_state.error:

        with st.expander(
            "⚠️ Hata detayları"
        ):

            st.code(
                st.session_state.error
            )


# ============================================================
# RUN
# ============================================================

if automatic and hasattr(
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


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spot AI Deep Learning Scanner | "
    "15m | Transformer | 500K Target | "
    "No Order Execution"
)
