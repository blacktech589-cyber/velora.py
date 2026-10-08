import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# PYTORCH
# =========================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

BINANCE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
).rstrip("/")

INTERVAL = "15m"

TARGET_CANDLES = 500_000

SEQ_LEN = 96
LOOKAHEAD = 12

MAX_TRAIN_SAMPLES = 2500

REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.05


# =========================================================
# PRIORITY SYMBOLS
# =========================================================

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


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# PAGE
# =========================================================

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


# =========================================================
# SESSION STATE
# =========================================================

if "cache" not in st.session_state:
    st.session_state.cache = {}

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = {}

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None


# =========================================================
# API
# =========================================================

def api_get(endpoint, params=None):

    url = BINANCE_URL + endpoint

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Network error: {exc}"
        )

    if response.status_code == 451:
        raise RuntimeError(
            "HTTP 451: Binance API bu isteği "
            "bölgesel/yasal erişim kısıtlaması "
            "nedeniyle reddetti. Bu durum PyTorch "
            "veya Streamlit hatası değildir."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "HTTP 429: Binance API rate limit. "
            "Coin sayısını azaltın veya istek "
            "aralığını artırın."
        )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# =========================================================
# EXCHANGE INFO
# =========================================================

@st.cache_data(ttl=300)
def exchange_info():

    return api_get(
        "/api/v3/exchangeInfo"
    )


# =========================================================
# SPOT COINS
# =========================================================

def get_spot_symbols():

    info = exchange_info()

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


# =========================================================
# KLINE CONVERSION
# =========================================================

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

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
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


# =========================================================
# DOWNLOAD 500K
# =========================================================

@st.cache_data(
    show_spinner=False
)
def download_history(
    symbol,
    candle_count
):

    rows = []

    end_time = int(
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
            "endTime": end_time,
        }

        batch = api_get(
            "/api/v3/klines",
            params
        )

        if not batch:
            break

        rows = batch + rows

        end_time = (
            int(batch[0][0])
            - 1
        )

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
            .reset_index(drop=True)
        )

    return df


# =========================================================
# UPDATE CACHE
# =========================================================

def update_cache(
    symbol,
    candle_count
):

    cache = st.session_state.cache

    if symbol not in cache:

        df = download_history(
            symbol,
            candle_count
        )

        cache[symbol] = df

        return df

    df = cache[symbol]

    return df


# =========================================================
# FEATURES
# =========================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # EMA
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
            ema / close - 1
        )

    # RSI
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
            - 100 / (1 + rs)
        )

    # Bollinger
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
        middle + 2 * std
    ) / close - 1

    x["bb_lower"] = (
        middle - 2 * std
    ) / close - 1

    # Volatility
    returns = close.pct_change()

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

    # Volume
    volume_average = (
        volume
        .rolling(50)
        .mean()
    )

    x["volume_ratio"] = (
        volume
        / (
            volume_average
            + 1e-12
        )
    )

    # Candle
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
            ["open", "close"]
        ].max(axis=1)
    ) / close

    x["lower_wick"] = (
        x[
            ["open", "close"]
        ].min(axis=1)
        - low
    ) / close

    # Momentum
    x["momentum12"] = (
        close.pct_change(12)
    )

    x["momentum48"] = (
        close.pct_change(48)
    )

    # ATR
    tr1 = high - low

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

    # Range Z
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
        range_std + 1e-12
    )

    # Position
    x["close_position"] = (
        close - low
    ) / (
        high - low + 1e-12
    )

    # Target
    x["future_return"] = (
        close.shift(-LOOKAHEAD)
        / close
        - 1
    )

    return (
        x.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


# =========================================================
# TRANSFORMER
# =========================================================

if TORCH_AVAILABLE:

    class AIModel(
        nn.Module
    ):

        def __init__(
            self,
            feature_count
        ):

            super().__init__()

            self.input_layer = (
                nn.Linear(
                    feature_count,
                    96
                )
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=96,
                    nhead=8,
                    dim_feedforward=384,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                )
            )

            self.encoder = (
                nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=3
                )
            )

            self.classifier = nn.Sequential(
                nn.Linear(
                    96,
                    64
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    2
                ),
            )

            self.regression = nn.Sequential(
                nn.Linear(
                    96,
                    64
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    1
                ),
            )

        def forward(self, x):

            x = self.input_layer(x)

            x = self.encoder(x)

            x = x[:, -1, :]

            classification = (
                self.classifier(x)
            )

            regression = (
                self.regression(x)
            )

            return (
                classification,
                regression
            )


# =========================================================
# AI PREDICTION
# =========================================================

def predict(
    df
):

    if not TORCH_AVAILABLE:

        return (
            "ERROR",
            0,
            0
        )

    features = (
        build_features(df)
        .dropna(
            subset=(
                FEATURES
                + ["future_return"]
            )
        )
    )

    if len(features) < (
        SEQ_LEN + 200
    ):

        return (
            "HOLD",
            0,
            0
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

    mean = (
        values[:-LOOKAHEAD]
        .mean(axis=0)
    )

    std = (
        values[:-LOOKAHEAD]
        .std(axis=0)
        + 1e-6
    )

    values = (
        values - mean
    ) / std

    future = (
        features[
            "future_return"
        ]
        .values
        .astype(
            np.float32
        )
    )

    sample_count = min(
        MAX_TRAIN_SAMPLES,
        len(features)
        - SEQ_LEN
    )

    starts = np.linspace(
        0,
        len(features)
        - SEQ_LEN
        - 1,
        sample_count,
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

    indices = (
        starts
        + SEQ_LEN
        - 1
    )

    y_return = (
        future[
            indices
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

    model = AIModel(
        len(FEATURES)
    ).to(device)

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=0.0002
        )
    )

    loss_class = (
        nn.CrossEntropyLoss()
    )

    loss_reg = (
        nn.MSELoss()
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

    for epoch in range(3):

        order = torch.randperm(
            len(X_tensor),
            device=device
        )

        for start in range(
            0,
            len(X_tensor),
            64
        ):

            batch = order[
                start:start + 64
            ]

            logits, regression = (
                model(
                    X_tensor[
                        batch
                    ]
                )
            )

            loss = (
                loss_class(
                    logits,
                    y_tensor[
                        batch
                    ]
                )
                +
                0.5
                *
                loss_reg(
                    regression,
                    return_tensor[
                        batch
                    ]
                )
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1
            )

            optimizer.step()

    model.eval()

    latest = torch.tensor(
        values[-SEQ_LEN:],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        logits, regression = (
            model(latest)
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
        )

        sell_probability = (
            probabilities[0]
            .item()
        )

        buy_probability = (
            probabilities[1]
            .item()
        )

        expected_return = (
            regression
            .item()
        )

    if (
        buy_probability
        >= sell_probability
    ):

        return (
            "BUY",
            buy_probability * 100,
            expected_return * 100
        )

    return (
        "SELL",
        sell_probability * 100,
        expected_return * 100
    )


# =========================================================
# SCORING
# =========================================================

def calculate_score(
    df,
    signal,
    confidence,
    expected_return
):

    features = build_features(
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
        df[
            "quote_volume"
        ]
        .tail(96)
        .mean()
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
        expected_return * 0.45
        + confidence / 100 * 0.25
        + direction * momentum * 0.12
        + direction * trend * 0.10
        + liquidity * 0.08
        - risk * 0.20
    )

    return float(score)


# =========================================================
# SCAN SYMBOL
# =========================================================

def scan_symbol(
    symbol,
    candle_count
):

    df = update_cache(
        symbol,
        candle_count
    )

    if df.empty:
        return None

    if len(df) < (
        SEQ_LEN + 250
    ):
        return None

    avg_volume = float(
        df[
            "quote_volume"
        ]
        .tail(96)
        .mean()
    )

    if avg_volume < 1_000_000:
        return None

    signal, confidence, expected = (
        predict(df)
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
        expected
    )

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(
            confidence,
            2
        ),
        "expected_return": round(
            expected,
            4
        ),
        "AI_score": round(
            score,
            5
        ),
        "candles": len(df),
        "volume_96": round(
            avg_volume,
            2
        ),
    }


# =========================================================
# SCAN ALL
# =========================================================

def run_scan(
    symbols,
    candle_count
):

    results = []

    progress = st.progress(
        0
    )

    for i, symbol in enumerate(
        symbols
    ):

        progress.progress(
            int(
                (i + 1)
                / len(symbols)
                * 100
            ),
            text=(
                f"{symbol} "
                f"{i + 1}/"
                f"{len(symbols)}"
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
            ] = str(error)

    progress.empty()

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            results
        )
        .sort_values(
            "AI_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Enterprise Settings"
    )

    candle_count = st.number_input(
        "Candle count",
        min_value=10_000,
        max_value=500_000,
        value=500_000,
        step=10_000
    )

    pair_limit = st.number_input(
        "Pairs to scan",
        min_value=1,
        max_value=100,
        value=15
    )

    auto_refresh = st.checkbox(
        "15 dakika otomatik yenile",
        value=True
    )

    st.divider()

    st.write(
        "Timeframe:"
    )

    st.code("15m")

    st.write(
        "Deep Learning:"
    )

    if TORCH_AVAILABLE:
        st.success(
            "PyTorch READY"
        )
    else:
        st.error(
            "PyTorch MISSING"
        )

    st.info(
        "Bu sistem emir göndermez."
    )


# =========================================================
# SYSTEM STATUS
# =========================================================

st.subheader(
    "🩺 System Status"
)

a, b, c, d = st.columns(4)

with a:
    st.metric(
        "PyTorch",
        (
            "READY"
            if TORCH_AVAILABLE
            else "MISSING"
        )
    )

with b:

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

with c:

    st.metric(
        "Cached pairs",
        len(
            st.session_state.cache
        )
    )

with d:

    st.metric(
        "Timeframe",
        "15m"
    )


# =========================================================
# 451 WARNING
# =========================================================

st.info(
    """
HTTP 451 görürseniz bu PyTorch problemi değildir.

Binance API erişimi ağ/bölge nedeniyle reddediliyor olabilir.
Bu uygulama bölgesel erişim kısıtlamalarını aşmaya çalışmaz.

Eğer kullanımınıza açık uyumlu bir market-data endpoint'iniz
varsa Streamlit environment variable olarak:

BINANCE_BASE_URL

tanımlayabilirsiniz.
"""
)


# =========================================================
# MAIN SCANNER
# =========================================================

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
                "Loading Spot symbols..."
            ):

                symbols = (
                    get_spot_symbols()
                )

            ordered = [
                x
                for x
                in PRIORITY_SYMBOLS
                if x in symbols
            ]

            ordered += [
                x
                for x
                in symbols
                if x not in ordered
            ]

            selected = ordered[
                :int(pair_limit)
            ]

            with st.spinner(
                "Running Deep Learning..."
            ):

                results = run_scan(
                    selected,
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

        st.warning(
            "Henüz sonuç yok. "
            "SCAN NOW butonuna basın."
        )

        return

    # =====================================================
    # TOP BUY
    # =====================================================

    buys = (
        results[
            results["signal"]
            == "BUY"
        ]
        .sort_values(
            [
                "AI_score",
                "confidence"
            ],
            ascending=False
        )
        .head(10)
    )

    # =====================================================
    # TOP SELL
    # =====================================================

    sells = (
        results[
            results["signal"]
            == "SELL"
        ]
        .sort_values(
            [
                "AI_score",
                "confidence"
            ],
            ascending=False
        )
        .head(10)
    )

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

    # =====================================================
    # ALL RESULTS
    # =====================================================

    st.subheader(
        "📊 ALL AI SIGNALS"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # CSV
    # =====================================================

    st.download_button(
        "⬇️ Download CSV",
        data=results.to_csv(
            index=False
        ),
        file_name=(
            "ai_signals.csv"
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


# =========================================================
# ERRORS
# =========================================================

with st.expander(
    "🛠 API Errors"
):

    if not st.session_state.errors:

        st.success(
            "No symbol errors."
        )

    else:

        for symbol, error in (
            st.session_state.errors.items()
        ):

            st.error(
                f"{symbol}: {error}"
            )
