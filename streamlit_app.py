import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

import torch
import torch.nn as nn


# ============================================================
# AYARLAR
# ============================================================

BINANCE_URL = "https://api.binance.com"

INTERVAL = "15m"

TARGET_CANDLES = 500_000

LOOKAHEAD = 12

SEQ_LEN = 96


# ============================================================
# STREAMLIT AYARLARI
# ============================================================

st.set_page_config(
    page_title="Binance Spot AI Scanner",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>

.block-container {
    max-width: 1450px;
    padding-top: 25px;
}

.buy {
    color: #00d084;
    font-weight: 800;
}

.sell {
    color: #ff4d67;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BINANCE API
# ============================================================

@st.cache_data(ttl=300)
def get_exchange_info():

    response = requests.get(
        BINANCE_URL + "/api/v3/exchangeInfo",
        timeout=30,
        headers={
            "User-Agent": "Binance-Spot-AI"
        }
    )

    response.raise_for_status()

    return response.json()


def get_spot_symbols():

    data = get_exchange_info()

    result = []

    for item in data["symbols"]:

        if item["status"] != "TRADING":
            continue

        if item["quoteAsset"] != "USDT":
            continue

        if not item.get(
            "isSpotTradingAllowed",
            True
        ):
            continue

        result.append(
            item["symbol"]
        )

    return result


# ============================================================
# MUM VERİSİ
# ============================================================

def download_initial_500k(
    symbol,
    candle_count
):

    all_rows = []

    end_time = None

    while len(all_rows) < candle_count:

        limit = min(
            1000,
            candle_count - len(all_rows)
        )

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit
        }

        if end_time is not None:

            params["endTime"] = end_time

        response = requests.get(
            BINANCE_URL + "/api/v3/klines",
            params=params,
            timeout=30,
            headers={
                "User-Agent": "Binance-Spot-AI"
            }
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_rows = batch + all_rows

        end_time = batch[0][0] - 1

        if len(batch) < limit:
            break

        time.sleep(0.04)


    return convert_klines(
        all_rows[-candle_count:]
    )


def download_new_candles(
    symbol,
    last_open_time
):

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": int(last_open_time) + 1,
        "limit": 1000
    }

    response = requests.get(
        BINANCE_URL + "/api/v3/klines",
        params=params,
        timeout=30,
        headers={
            "User-Agent": "Binance-Spot-AI"
        }
    )

    response.raise_for_status()

    rows = response.json()

    if not rows:

        return pd.DataFrame()

    return convert_klines(rows)


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
        "ignore"

    ]

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
        "quote_volume"

    ]


    for column in numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna()

    df = df.drop_duplicates(
        "open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# 500K CACHE
# ============================================================

def update_symbol_cache(
    symbol,
    candle_count
):

    if "candle_cache" not in st.session_state:

        st.session_state.candle_cache = {}


    cache = st.session_state.candle_cache


    # İlk kez
    if symbol not in cache:

        df = download_initial_500k(
            symbol,
            candle_count
        )

        cache[symbol] = df

        return df, True


    # Daha önce var
    old = cache[symbol]


    if old.empty:

        df = download_initial_500k(
            symbol,
            candle_count
        )

        cache[symbol] = df

        return df, True


    last_time = int(
        old["open_time"].iloc[-1]
    )


    new = download_new_candles(
        symbol,
        last_time
    )


    if not new.empty:

        df = pd.concat(
            [
                old,
                new
            ],
            ignore_index=True
        )


        df = (
            df
            .drop_duplicates("open_time")
            .sort_values("open_time")
            .tail(candle_count)
            .reset_index(drop=True)
        )


        cache[symbol] = df

        return df, True


    return old, False


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
    "body",

    "upper_wick",
    "lower_wick",

    "momentum12",
    "momentum48"

]


def create_features(df):

    data = df.copy()

    close = data["close"]

    returns = close.pct_change()


    # EMA

    for period in [
        5,
        12,
        20,
        50,
        100,
        200,
        500,
        800
    ]:

        data[f"ema{period}"] = (

            close
            .ewm(
                span=period,
                adjust=False
            )
            .mean()
            /
            close
            - 1

        )


    # RSI

    for period in [
        7,
        14,
        21
    ]:

        delta = close.diff()

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
            gain /
            (loss + 1e-12)
        )

        data[f"rsi{period}"] = (

            100 -
            (
                100 /
                (1 + rs)
            )

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


    data["bb_upper"] = (

        (middle + 2 * std)
        /
        close
        - 1

    )


    data["bb_lower"] = (

        (middle - 2 * std)
        /
        close
        - 1

    )


    # Volatility

    data["vol20"] = (
        returns
        .rolling(20)
        .std()
    )

    data["vol50"] = (
        returns
        .rolling(50)
        .std()
    )


    # Volume

    data["volume_ratio"] = (

        data["volume"]
        /
        (
            data["volume"]
            .rolling(50)
            .mean()
            + 1e-12
        )

    )


    # Candle

    data["candle_range"] = (

        data["high"]
        -
        data["low"]

    ) / close


    data["body"] = (

        data["close"]
        -
        data["open"]

    ) / data["open"]


    data["upper_wick"] = (

        data["high"]
        -
        data[
            ["open", "close"]
        ].max(axis=1)

    ) / close


    data["lower_wick"] = (

        data[
            ["open", "close"]
        ].min(axis=1)
        -
        data["low"]

    ) / close


    # Momentum

    data["momentum12"] = (
        close.pct_change(12)
    )

    data["momentum48"] = (
        close.pct_change(48)
    )


    # Target

    data["future_return"] = (

        close.shift(
            -LOOKAHEAD
        )
        /
        close
        - 1

    )


    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    data = data.dropna()

    return data.reset_index(
        drop=True
    )


# ============================================================
# TRANSFORMER
# ============================================================

class TransformerModel(
    nn.Module
):

    def __init__(
        self,
        feature_count
    ):

        super().__init__()


        self.input = nn.Linear(
            feature_count,
            96
        )


        layer = (
            nn.TransformerEncoderLayer(
                d_model=96,
                nhead=8,
                batch_first=True,
                dropout=0.10
            )
        )


        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=3
            )
        )


        self.classifier = nn.Linear(
            96,
            2
        )


        self.regression = nn.Linear(
            96,
            1
        )


    def forward(self, x):

        x = self.input(x)

        x = self.encoder(x)

        x = x[:, -1]

        return (
            self.classifier(x),
            self.regression(x)
        )


# ============================================================
# AI PREDICTION
# ============================================================

def predict_ai(data):

    if len(data) < SEQ_LEN + 300:

        return None


    values = (
        data[FEATURES]
        .values
        .astype(np.float32)
    )


    future = (
        data["future_return"]
        .values
        .astype(np.float32)
    )


    usable = (
        len(values)
        -
        LOOKAHEAD
    )


    mean = (
        values[:usable]
        .mean(axis=0)
    )


    std = (
        values[:usable]
        .std(axis=0)
        +
        1e-6
    )


    normalized = (
        values - mean
    ) / std


    sample_count = min(
        2500,
        usable - SEQ_LEN
    )


    indexes = np.linspace(

        SEQ_LEN,

        usable - 1,

        sample_count,

        dtype=int

    )


    X = np.stack([

        normalized[
            i-SEQ_LEN:i
        ]

        for i in indexes

    ])


    returns = future[
        indexes
    ]


    labels = (
        returns > 0
    ).astype(
        np.int64
    )


    if len(
        np.unique(labels)
    ) < 2:

        return None


    device = (
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    model = TransformerModel(
        len(FEATURES)
    ).to(device)


    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=0.0002,
            weight_decay=0.0001
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
        labels,
        dtype=torch.long,
        device=device
    )


    return_tensor = torch.tensor(
        returns,
        dtype=torch.float32,
        device=device
    )


    model.train()


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
            64
        ):

            batch = permutation[
                start:start + 64
            ]


            logits, regression = (
                model(
                    X_tensor[batch]
                )
            )


            loss = (

                loss_class(
                    logits,
                    y_tensor[batch]
                )

                +

                0.35
                *
                loss_reg(
                    regression.squeeze(),
                    return_tensor[batch]
                )

            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


    # Güncel tahmin

    model.eval()


    current = torch.tensor(

        normalized[
            -SEQ_LEN:
        ][None],

        dtype=torch.float32,

        device=device

    )


    with torch.no_grad():

        logits, regression = (
            model(current)
        )


        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
            .cpu()
            .numpy()
        )


        predicted_return = (
            float(
                regression
                .squeeze()
                .cpu()
            )
            * 100
        )


    confidence = (
        max(probabilities)
        * 100
    )


    signal = (

        "BUY"
        if probabilities[1] >= 0.5
        else
        "SELL"

    )


    return (
        signal,
        confidence,
        predicted_return
    )


# ============================================================
# SKOR
# ============================================================

def calculate_scores(
    data,
    prediction
):

    signal, confidence, expected = (
        prediction
    )


    volatility = (
        float(
            data[
                "vol50"
            ].iloc[-1]
        )
        * 100
    )


    momentum = float(

        np.clip(

            (
                data[
                    "momentum48"
                ].iloc[-1]
                + 0.10
            )
            * 500,

            0,
            100

        )

    )


    trend = float(

        np.clip(

            50
            +
            (
                data[
                    "ema200"
                ].iloc[-1]
                -
                data[
                    "ema800"
                ].iloc[-1]
            )
            * 500,

            0,
            100

        )

    )


    risk = float(

        np.clip(

            35
            +
            volatility * 250,

            0,
            100

        )

    )


    quote_volume = float(

        data[
            "quote_volume"
        ]
        .tail(96)
        .mean()

    )


    liquidity = float(

        np.clip(

            np.log10(
                quote_volume + 1
            ) * 10,

            0,
            100

        )

    )


    score = (

        expected * 0.45

        +

        confidence * 0.25

        +

        momentum * 0.12

        +

        trend * 0.10

        +

        liquidity * 0.08

        -

        risk * 0.20

    )


    return {

        "signal": signal,

        "expected": expected,

        "confidence": confidence,

        "risk": risk,

        "momentum": momentum,

        "trend": trend,

        "liquidity": liquidity,

        "score": score

    }


# ============================================================
# TEK PARİTE
# ============================================================

def scan_pair(
    symbol,
    candle_count
):

    df, updated = (
        update_symbol_cache(
            symbol,
            candle_count
        )
    )


    if len(df) < 5000:

        return None


    quote_volume = float(

        df[
            "quote_volume"
        ]
        .tail(96)
        .mean()

    )


    if quote_volume < 1_000_000:

        return None


    features = create_features(
        df
    )


    prediction = predict_ai(
        features
    )


    if prediction is None:

        return None


    scores = calculate_scores(
        features,
        prediction
    )


    return {

        "symbol": symbol,

        **scores,

        "price": float(
            df[
                "close"
            ].iloc[-1]
        ),

        "candles": len(df),

        "new_candles": updated

    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Ayarlar"
    )


    candle_count = st.number_input(

        "Geçmiş mum",

        min_value=5_000,

        max_value=500_000,

        value=500_000,

        step=5_000

    )


    pair_count = st.number_input(

        "Taranacak coin",

        min_value=10,

        max_value=100,

        value=10,

        step=5

    )


    auto_refresh = st.checkbox(

        "15 dakikada otomatik tara",

        value=True

    )


    force_scan = st.button(

        "🚀 Şimdi Tara",

        use_container_width=True

    )


    st.divider()

    st.write(
        "Timeframe: **15 dakika**"
    )

    st.write(
        "Model: **Transformer**"
    )

    st.write(
        "Horizon: **12 mum / 3 saat**"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:

    st.session_state.results = []


if "last_scan" not in st.session_state:

    st.session_state.last_scan = None


if "scan_number" not in st.session_state:

    st.session_state.scan_number = 0


# ============================================================
# TARAMA FONKSİYONU
# ============================================================

def run_scan():

    symbols = (
        get_spot_symbols()
    )


    priority = [

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
        "UNIUSDT"

    ]


    ordered = (

        [
            x
            for x in priority
            if x in symbols
        ]

        +

        [
            x
            for x in symbols
            if x not in priority
        ]

    )


    selected = ordered[
        :int(pair_count)
    ]


    results = []


    progress = st.progress(
        0
    )


    status = st.empty()


    for i, symbol in enumerate(
        selected,
        1
    ):

        status.write(

            f"🔎 {symbol} "
            f"taranıyor "
            f"({i}/{len(selected)})"

        )


        try:

            result = scan_pair(

                symbol,

                int(candle_count)

            )


            if result:

                results.append(
                    result
                )


        except Exception as error:

            st.warning(
                f"{symbol}: "
                f"{str(error)[:100]}"
            )


        progress.progress(
            i / len(selected)
        )


    results.sort(

        key=lambda x:
        x["score"],

        reverse=True

    )


    st.session_state.results = (
        results[:10]
    )


    st.session_state.last_scan = (
        datetime.now(
            timezone.utc
        )
    )


    st.session_state.scan_number += 1


    progress.empty()

    status.empty()


# ============================================================
# 15 DAKİKALIK STREAMLIT FRAGMENT
# ============================================================

run_every = (
    "15m"
    if auto_refresh
    else None
)


@st.fragment(
    run_every=run_every,
    key="ai_scanner"
)
def scanner_fragment():

    # İlk açılış veya manuel tarama
    if (
        not st.session_state.results
        or
        force_scan
    ):

        run_scan()


    # --------------------------------------------------------
    # BAŞLIK
    # --------------------------------------------------------

    st.title(
        "📈 Binance Spot AI Scanner"
    )


    st.caption(

        "500.000 mum · 15 dakika · "
        "Deep Learning · BUY / SELL"

    )


    # --------------------------------------------------------
    # ÜST BİLGİ
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "Piyasa",
            "BINANCE SPOT"
        )


    with col2:

        st.metric(
            "Öneri",
            len(
                st.session_state.results
            )
        )


    with col3:

        st.metric(
            "Tarama",
            st.session_state.scan_number
        )


    with col4:

        st.metric(
            "Aralık",
            "15 dk"
        )


    if st.session_state.last_scan:

        local_time = (
            st.session_state
            .last_scan
            .astimezone()
        )


        st.caption(

            "Son tarama: "
            +
            local_time.strftime(
                "%d.%m.%Y %H:%M:%S"
            )

        )


    # --------------------------------------------------------
    # TABLO
    # --------------------------------------------------------

    results = (
        st.session_state.results
    )


    if not results:

        st.warning(
            "Henüz sinyal oluşmadı."
        )

        return


    table = []


    for rank, item in enumerate(
        results,
        1
    ):

        table.append({

            "#": rank,

            "PARİTE":
            item["symbol"],

            "SİNYAL":
            item["signal"],

            "BEKLENEN %":
            round(
                item["expected"],
                3
            ),

            "CONFIDENCE %":
            round(
                item["confidence"],
                1
            ),

            "RİSK":
            round(
                item["risk"],
                1
            ),

            "MOMENTUM":
            round(
                item["momentum"],
                1
            ),

            "TREND":
            round(
                item["trend"],
                1
            ),

            "SKOR":
            round(
                item["score"],
                2
            ),

            "FİYAT":
            item["price"]

        })


    result_df = pd.DataFrame(
        table
    )


    st.dataframe(

        result_df,

        use_container_width=True,

        hide_index=True

    )


    st.info(

        "Her 15 dakikada yeni mumlar alınır, "
        "500K'lık geçmiş pencere güncellenir ve "
        "Top 10 yeniden hesaplanır."

    )


# ============================================================
# ÇALIŞTIR
# ============================================================

scanner_fragment()
