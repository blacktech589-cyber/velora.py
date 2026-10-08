import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# PYTORCH
# ============================================================

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

DEFAULT_BINANCE_URL = "https://api.binance.com"

BINANCE_URL = os.getenv(
    "BINANCE_BASE_URL",
    DEFAULT_BINANCE_URL
).rstrip("/")

INTERVAL = "15m"

TARGET_CANDLES = 500_000

LOOKAHEAD = 12
SEQ_LEN = 96

MIN_TRAIN_ROWS = 500
MAX_TRAIN_SAMPLES = 2500

REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.05


# ============================================================
# PRIORITY COINS
# ============================================================

PRIORITY_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SUIUSDT",
    "SEIUSDT",
]


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ema5",
    "ema12",
    "ema20",
    "ema50",
    "ema100",
    "ema200",
    "ema500",
    "ema800",

    "rsi7",
    "rsi14",
    "rsi21",

    "bb_upper",
    "bb_lower",

    "vol20",
    "vol50",

    "volume_ratio",

    "candle_range",
    "candle_body",
    "upper_wick",
    "lower_wick",

    "momentum12",
    "momentum48",

    "atr14",

    "range_z",

    "close_position",
]


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Binance Spot AI Enterprise",
    page_icon="📊",
    layout="wide",
)

st.title(
    "📊 Binance Spot Deep Learning Scanner"
)

st.caption(
    "15m • Rolling 500,000 candles • "
    "Deep Learning • Top 10 BUY / SELL • "
    "No order execution"
)


# ============================================================
# SESSION STATE
# ============================================================

if "candle_cache" not in st.session_state:
    st.session_state.candle_cache = {}

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "errors" not in st.session_state:
    st.session_state.errors = {}


# ============================================================
# API ERROR HANDLER
# ============================================================

def explain_api_error(response, url):

    if response.status_code == 451:

        return (
            "HTTP 451: Market-data API erişimi bu ağ/bölge "
            "için kısıtlanmış. Bu durum PyTorch veya "
            "Streamlit hatası değildir. Uygulama bu "
            "kısıtlamayı bypass etmez."
        )

    if response.status_code == 429:

        return (
            "HTTP 429: API rate limit. "
            "Daha az coin tarayın veya istekler arasındaki "
            "bekleme süresini artırın."
        )

    return (
        f"HTTP {response.status_code}: "
        f"{url}\n\n"
        f"{response.text[:500]}"
    )


# ============================================================
# HTTP CLIENT
# ============================================================

def api_get(endpoint, params=None):

    url = f"{BINANCE_URL}{endpoint}"

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:

        raise RuntimeError(
            explain_api_error(
                response,
                url,
            )
        )

    return response.json()


# ============================================================
# EXCHANGE INFO
# ============================================================

@st.cache_data(ttl=300)
def get_exchange_info():

    return api_get(
        "/api/v3/exchangeInfo"
    )


# ============================================================
# SPOT SYMBOLS
# ============================================================

def get_spot_symbols():

    info = get_exchange_info()

    symbols = []

    for item in info.get(
        "symbols",
        []
    ):

        if item.get(
            "status"
        ) != "TRADING":
            continue

        if item.get(
            "quoteAsset"
        ) != "USDT":
            continue

        permissions = item.get(
            "permissions",
            []
        )

        if permissions:

            if "SPOT" not in permissions:
                continue

        symbols.append(
            item["symbol"]
        )

    return sorted(
        set(symbols)
    )


# ============================================================
# KLINE CONVERTER
# ============================================================

def convert_klines(rows):

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
        "taker_base",
        "taker_quote",
        "ignore",
    ]

    if not rows:

        return pd.DataFrame(
            columns=columns
        )

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    df = (
        df
        .drop_duplicates(
            "open_time"
        )
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# INITIAL 500K DOWNLOAD
# ============================================================

@st.cache_data(
    show_spinner=False
)
def download_initial_history(
    symbol,
    candle_count
):

    rows = []

    end_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp()
        * 1000
    )

    while len(rows) < candle_count:

        limit = min(
            1000,
            candle_count - len(rows)
        )

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit,
            "endTime": end_ms,
        }

        batch = api_get(
            "/api/v3/klines",
            params
        )

        if not batch:
            break

        rows = batch + rows

        oldest = int(
            batch[0][0]
        )

        end_ms = oldest - 1

        if len(batch) < limit:
            break

        time.sleep(
            REQUEST_SLEEP
        )

    df = convert_klines(
        rows
    )

    if len(df) > candle_count:

        df = (
            df
            .tail(candle_count)
            .reset_index(
                drop=True
            )
        )

    return df


# ============================================================
# NEW 15M CANDLES
# ============================================================

def download_new_history(
    symbol,
    last_open_ms
):

    rows = []

    start_ms = (
        int(last_open_ms)
        + 1
    )

    for _ in range(10):

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": 1000,
            "startTime": start_ms,
        }

        batch = api_get(
            "/api/v3/klines",
            params
        )

        if not batch:
            break

        rows.extend(
            batch
        )

        if len(batch) < 1000:
            break

        start_ms = (
            int(batch[-1][0])
            + 1
        )

        time.sleep(
            REQUEST_SLEEP
        )

    return convert_klines(
        rows
    )


# ============================================================
# ROLLING CACHE
# ============================================================

def update_cache(
    symbol,
    candle_count
):

    cache = (
        st.session_state
        .candle_cache
    )

    old = cache.get(
        symbol
    )

    if old is None or old.empty:

        df = download_initial_history(
            symbol,
            candle_count
        )

    else:

        last_ms = int(
            old[
                "open_time"
            ]
            .iloc[-1]
            .timestamp()
            * 1000
        )

        new = download_new_history(
            symbol,
            last_ms
        )

        if new.empty:

            df = old

        else:

            df = pd.concat(
                [
                    old,
                    new
                ],
                ignore_index=True
            )

            df = (
                df
                .drop_duplicates(
                    "open_time"
                )
                .sort_values(
                    "open_time"
                )
                .tail(
                    candle_count
                )
                .reset_index(
                    drop=True
                )
            )

    cache[
        symbol
    ] = df

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for period in [
        5,
        12,
        20,
        50,
        100,
        200,
        500,
        800,
    ]:

        ema = (
            close
            .ewm(
                span=period,
                adjust=False
            )
            .mean()
        )

        x[
            f"ema{period}"
        ] = (
            ema / close
            - 1
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    for period in [
        7,
        14,
        21,
    ]:

        gain = (
            delta
            .clip(lower=0)
            .rolling(period)
            .mean()
        )

        loss = (
            -delta
            .clip(upper=0)
            .rolling(period)
            .mean()
        )

        rs = (
            gain
            / (loss + 1e-12)
        )

        x[
            f"rsi{period}"
        ] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

    # --------------------------------------------------------
    # BOLLINGER
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

    x["bb_upper"] = (
        middle
        + 2 * std
    ) / close - 1

    x["bb_lower"] = (
        middle
        - 2 * std
    ) / close - 1

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    returns = (
        close
        .pct_change()
    )

    x["vol20"] = (
        returns
        .rolling(20)
        .std()
    )

    x["vol50"] = (
        returns
        .rolling(50)
        .std()
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = (
        volume
        .rolling(50)
        .mean()
    )

    x["volume_ratio"] = (
        volume
        / (
            volume_mean
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    x["candle_range"] = (
        high - low
    ) / close

    x["candle_body"] = (
        (
            x["close"]
            - x["open"]
        ).abs()
        / close
    )

    x["upper_wick"] = (
        high
        - x[
            [
                "open",
                "close"
            ]
        ].max(axis=1)
    ) / close

    x["lower_wick"] = (
        x[
            [
                "open",
                "close"
            ]
        ].min(axis=1)
        - low
    ) / close

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    x["momentum12"] = (
        close
        .pct_change(12)
    )

    x["momentum48"] = (
        close
        .pct_change(48)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr1 = (
        high - low
    )

    tr2 = (
        high
        - close.shift()
    ).abs()

    tr3 = (
        low
        - close.shift()
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(axis=1)

    x["atr14"] = (
        true_range
        .rolling(14)
        .mean()
        / close
    )

    # --------------------------------------------------------
    # RANGE Z-SCORE
    # --------------------------------------------------------

    range_mean = (
        x["candle_range"]
        .rolling(50)
        .mean()
    )

    range_std = (
        x["candle_range"]
        .rolling(50)
        .std()
    )

    x["range_z"] = (
        x["candle_range"]
        - range_mean
    ) / (
        range_std
        + 1e-12
    )

    # --------------------------------------------------------
    # CLOSE POSITION
    # --------------------------------------------------------

    x["close_position"] = (
        close - low
    ) / (
        high - low
        + 1e-12
    )

    # --------------------------------------------------------
    # FUTURE TARGET
    # --------------------------------------------------------

    x["future_return"] = (
        close.shift(
            -LOOKAHEAD
        )
        / close
        - 1
    )

    return (
        x
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


# ============================================================
# TRANSFORMER MODEL
# ============================================================

if TORCH_AVAILABLE:

    class TransformerAI(
        nn.Module
    ):

        def __init__(
            self,
            feature_count
        ):

            super().__init__()

            d_model = 96

            self.projection = (
                nn.Linear(
                    feature_count,
                    d_model
                )
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=384,
                    dropout=0.10,
                    activation="gelu",
                    batch_first=True,
                )
            )

            self.encoder = (
                nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=3
                )
            )

            self.classifier = (
                nn.Sequential(
                    nn.Linear(
                        d_model,
                        64
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        0.10
                    ),
                    nn.Linear(
                        64,
                        2
                    ),
                )
            )

            self.regressor = (
                nn.Sequential(
                    nn.Linear(
                        d_model,
                        64
                    ),
                    nn.GELU(),
                    nn.Linear(
                        64,
                        1
                    ),
                )
            )

        def forward(self, x):

            x = self.projection(
                x
            )

            x = self.encoder(
                x
            )

            x = x[
                :,
                -1,
                :
            ]

            classification = (
                self.classifier(x)
            )

            regression = (
                self.regressor(x)
            )

            return (
                classification,
                regression
            )


# ============================================================
# AI PREDICTION
# ============================================================

def predict_ai(df):

    if not TORCH_AVAILABLE:

        return (
            "UNAVAILABLE",
            0.0,
            0.0
        )

    features = (
        make_features(df)
        .dropna(
            subset=(
                FEATURES
                + [
                    "future_return"
                ]
            )
        )
        .copy()
    )

    if len(features) < (
        SEQ_LEN
        + MIN_TRAIN_ROWS
    ):

        return (
            "INSUFFICIENT_DATA",
            0.0,
            0.0
        )

    values = (
        features[
            FEATURES
        ]
        .astype(
            np.float32
        )
        .values
    )

    training_values = (
        values[:-LOOKAHEAD]
    )

    mean = (
        training_values
        .mean(axis=0)
    )

    std = (
        training_values
        .std(axis=0)
        + 1e-6
    )

    values = (
        values - mean
    ) / std

    returns = (
        features[
            "future_return"
        ]
        .astype(
            np.float32
        )
        .values
    )

    max_samples = min(
        2500,
        len(features)
        - SEQ_LEN
    )

    starts = np.linspace(
        0,
        len(features)
        - SEQ_LEN
        - 1,
        max_samples,
        dtype=int
    )

    X = np.stack(
        [
            values[
                i:i + SEQ_LEN
            ]
            for i in starts
        ]
    )

    target_indices = (
        starts
        + SEQ_LEN
        - 1
    )

    y_return = (
        returns[
            target_indices
        ]
    )

    y_class = (
        y_return > 0
    ).astype(
        np.int64
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TransformerAI(
        len(FEATURES)
    ).to(device)

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=0.0002,
            weight_decay=0.0001,
        )
    )

    loss_class = (
        nn.CrossEntropyLoss()
    )

    loss_reg = (
        nn.SmoothL1Loss()
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    y_tensor = torch.tensor(
        y_class,
        dtype=torch.long,
        device=device
    )

    return_tensor = torch.tensor(
        y_return,
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    model.train()

    batch_size = 64

    for epoch in range(3):

        permutation = (
            torch.randperm(
                len(X_tensor),
                device=device
            )
        )

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            indices = (
                permutation[
                    start:
                    start + batch_size
                ]
            )

            logits, predicted_return = (
                model(
                    X_tensor[
                        indices
                    ]
                )
            )

            classification_loss = (
                loss_class(
                    logits,
                    y_tensor[
                        indices
                    ]
                )
            )

            regression_loss = (
                loss_reg(
                    predicted_return,
                    return_tensor[
                        indices
                    ]
                )
            )

            loss = (
                classification_loss
                + 0.5
                * regression_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

    # --------------------------------------------------------
    # LATEST PREDICTION
    # --------------------------------------------------------

    model.eval()

    latest = torch.tensor(
        values[
            -SEQ_LEN:
        ],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        logits, prediction = (
            model(
                latest
            )
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
        )

        sell_probability = (
            float(
                probabilities[
                    0
                ].item()
            )
        )

        buy_probability = (
            float(
                probabilities[
                    1
                ].item()
            )
        )

        expected_return = (
            float(
                prediction
                .item()
            )
        )

    if (
        buy_probability
        >= sell_probability
    ):

        return (
            "BUY",
            buy_probability * 100,
            expected_return * 100,
        )

    return (
        "SELL",
        sell_probability * 100,
        expected_return * 100,
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    df,
    signal,
    confidence,
    expected_return
):

    features = make_features(
        df
    )

    last = features.iloc[-1]

    momentum = float(
        np.nan_to_num(
            last[
                "momentum48"
            ]
        )
    )

    trend = float(
        np.nan_to_num(
            last[
                "ema200"
            ]
            -
            last[
                "ema800"
            ]
        )
    )

    volatility = float(
        np.nan_to_num(
            last[
                "vol50"
            ]
        )
    )

    quote_volume = float(
        np.nan_to_num(
            df[
                "quote_volume"
            ]
            .tail(96)
            .mean()
        )
    )

    liquidity = min(
        np.log10(
            max(
                quote_volume,
                1
            )
        ) / 10,
        1
    )

    risk = min(
        volatility * 10,
        1
    )

    direction = (
        1
        if signal == "BUY"
        else -1
    )

    score = (
        expected_return
        * 0.45

        + (
            confidence
            / 100
        )
        * 0.25

        + direction
        * momentum
        * 0.12

        + direction
        * trend
        * 0.10

        + liquidity
        * 0.08

        - risk
        * 0.20
    )

    return float(
        score
    )


# ============================================================
# SCAN ONE SYMBOL
# ============================================================

def scan_symbol(
    symbol,
    candle_count
):

    df = update_cache(
        symbol,
        candle_count
    )

    minimum = (
        SEQ_LEN
        + LOOKAHEAD
        + MIN_TRAIN_ROWS
    )

    if len(df) < minimum:

        return None

    average_volume = float(
        df[
            "quote_volume"
        ]
        .tail(96)
        .mean()
    )

    # Low liquidity filter
    if average_volume < 1_000_000:

        return None

    (
        signal,
        confidence,
        expected_return
    ) = predict_ai(
        df
    )

    if signal not in [
        "BUY",
        "SELL"
    ]:

        return None

    score = calculate_score(
        df,
        signal,
        confidence,
        expected_return
    )

    return {
        "symbol": symbol,

        "signal": signal,

        "confidence_%": round(
            confidence,
            2
        ),

        "expected_return_%": round(
            expected_return,
            4
        ),

        "score": round(
            score,
            5
        ),

        "candles": len(df),

        "quote_volume_96": round(
            average_volume,
            2
        ),

        "timestamp": datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }


# ============================================================
# FULL SCAN
# ============================================================

def run_scan(
    symbols,
    candle_count
):

    results = []

    progress = st.progress(
        0
    )

    for index, symbol in enumerate(
        symbols
    ):

        progress.progress(
            int(
                (
                    index + 1
                )
                /
                len(symbols)
                * 100
            ),
            text=(
                f"Scanning "
                f"{symbol} "
                f"({index + 1}/"
                f"{len(symbols)})"
            )
        )

        try:

            result = scan_symbol(
                symbol,
                candle_count
            )

            if result:

                results.append(
                    result
                )

        except Exception as error:

            st.session_state.errors[
                symbol
            ] = str(
                error
            )

    progress.empty()

    if not results:

        return pd.DataFrame()

    return (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "score",
                "confidence_%"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Enterprise Settings"
    )

    candle_count = st.number_input(
        "Rolling candles",
        min_value=10_000,
        max_value=500_000,
        value=500_000,
        step=10_000
    )

    pair_limit = st.number_input(
        "Maximum pairs",
        min_value=1,
        max_value=100,
        value=15
    )

    auto_refresh = st.checkbox(
        "15 dakika otomatik tarama",
        value=True
    )

    st.divider()

    st.write(
        "Timeframe:"
    )

    st.code(
        "15m"
    )

    st.write(
        "Lookahead:"
    )

    st.code(
        f"{LOOKAHEAD} candles"
    )

    st.write(
        "Sequence:"
    )

    st.code(
        f"{SEQ_LEN} candles"
    )

    st.divider()

    if TORCH_AVAILABLE:

        st.success(
            "PyTorch READY"
        )

    else:

        st.error(
            "PyTorch MISSING"
        )

    st.info(
        "Bu uygulama emir göndermez."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader(
    "🩺 System Status"
)

c1, c2, c3, c4 = st.columns(
    4
)

with c1:

    st.metric(
        "PyTorch",
        (
            "READY"
            if TORCH_AVAILABLE
            else "MISSING"
        )
    )

with c2:

    st.metric(
        "Device",
        (
            "CUDA"
            if (
                TORCH_AVAILABLE
                and
                torch.cuda.is_available()
            )
            else "CPU"
        )
    )

with c3:

    st.metric(
        "Cached Pairs",
        len(
            st.session_state
            .candle_cache
        )
    )

with c4:

    st.metric(
        "Timeframe",
        "15m"
    )


# ============================================================
# 451 WARNING
# ============================================================

if BINANCE_URL == DEFAULT_BINANCE_URL:

    st.warning(
        """
HTTP 451 alırsanız bu PyTorch hatası değildir.

Binance API endpoint'i bulunduğunuz ağ/bölge için
erişimi reddediyor olabilir.

Uygulama bu kısıtlamayı bypass etmez.

Erişiminize hukuken açık, Binance-compatible bir
market-data endpoint'i kullanmanız gerekir.
"""
    )


# ============================================================
# SCANNER
# ============================================================

@st.fragment(
    run_every=(
        "15m"
        if auto_refresh
        else None
    )
)
def scanner():

    if st.button(
        "🚀 SCAN NOW",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Loading Spot universe..."
            ):

                universe = (
                    get_spot_symbols()
                )

            ordered = [
                symbol
                for symbol
                in PRIORITY_SYMBOLS
                if symbol
                in universe
            ]

            ordered += [
                symbol
                for symbol
                in universe
                if symbol
                not in ordered
            ]

            symbols = ordered[
                :int(pair_limit)
            ]

            with st.spinner(
                "Deep Learning scan..."
            ):

                results = run_scan(
                    symbols,
                    int(
                        candle_count
                    )
                )

            st.session_state.results = (
                results
            )

            st.session_state.last_scan = (
                datetime.now(
                    timezone.utc
                )
            )

        except Exception as error:

            st.error(
                f"Tarama hatası: {error}"
            )

            return

    results = (
        st.session_state.results
    )

    if results.empty:

        st.info(
            "Henüz tarama yapılmadı."
        )

        return

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buys = (
        results[
            results[
                "signal"
            ]
            == "BUY"
        ]
        .sort_values(
            [
                "score",
                "confidence_%"
            ],
            ascending=False
        )
        .head(10)
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    sells = (
        results[
            results[
                "signal"
            ]
            == "SELL"
        ]
        .sort_values(
            [
                "score",
                "confidence_%"
            ],
            ascending=False
        )
        .head(10)
    )

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    left, right = (
        st.columns(2)
    )

    with left:

        st.subheader(
            "🟢 TOP 10 BUY"
        )

        st.dataframe(
            buys,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.subheader(
            "🔴 TOP 10 SELL"
        )

        st.dataframe(
            sells,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    st.subheader(
        "📊 All AI Signals"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download CSV",
        data=results.to_csv(
            index=False
        ),
        file_name=(
            "binance_ai_signals.csv"
        ),
        mime="text/csv"
    )

    if st.session_state.last_scan:

        st.caption(
            "Last scan: "
            +
            st.session_state.last_scan
            .strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )


scanner()


# ============================================================
# ERRORS
# ============================================================

with st.expander(
    "🛠 API / Symbol Errors"
):

    if not st.session_state.errors:

        st.success(
            "No errors."
        )

    else:

        for symbol, error in (
            st.session_state.errors.items()
        ):

            st.error(
                f"{symbol}: {error}"
            )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

st.markdown(
    """
### Enterprise Architecture

```text
Market Data
     ↓
15m Candle Collector
     ↓
Rolling 500,000 Candle Cache
     ↓
Feature Engineering
     ↓
Transformer Deep Learning
     ↓
Classification + Regression
     ↓
Confidence / Expected Return
     ↓
Liquidity + Risk + Momentum
     ↓
AI Ranking
     ↓
TOP 10 BUY / TOP 10 SELL
