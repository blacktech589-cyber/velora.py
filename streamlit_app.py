import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import torch
import torch.nn as nn


# =========================================================
# AYARLAR
# =========================================================

BINANCE_URL = "https://api.binance.com"

INTERVAL = "15m"

TARGET_CANDLES = 500_000

LOOKAHEAD = 12

SEQ_LEN = 96


# =========================================================
# STREAMLIT
# =========================================================

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
    font-weight: bold;
}

.sell {
    color: #ff4d67;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# BINANCE EXCHANGE INFO
# =========================================================

@st.cache_data(ttl=300)
def get_exchange_info():

    url = BINANCE_URL + "/api/v3/exchangeInfo"

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Binance-Spot-AI"
        }
    )

    response.raise_for_status()

    return response.json()


def get_spot_symbols():

    data = get_exchange_info()

    symbols = []

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

        symbols.append(
            item["symbol"]
        )

    return symbols


# =========================================================
# 500.000 MUM VERİSİ
# =========================================================

@st.cache_data(ttl=60)
def get_klines(
    symbol,
    candle_count
):

    all_data = []

    end_time = None

    while len(all_data) < candle_count:

        limit = min(
            1000,
            candle_count - len(all_data)
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
                "User-Agent":
                "Binance-Spot-AI"
            }
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_data = batch + all_data

        end_time = batch[0][0] - 1

        if len(batch) < limit:
            break

        time.sleep(0.04)


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
        all_data[-candle_count:],
        columns=columns
    )


    numeric_columns = [

        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"

    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna()

    df = df.drop_duplicates(
        subset=["open_time"]
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.reset_index(
        drop=True
    )


    return df


# =========================================================
# TEKNİK GÖSTERGELER
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

        data[
            f"ema{period}"
        ] = (

            close.ewm(
                span=period,
                adjust=False
            ).mean()
            / close
            - 1

        )


    # RSI

    for period in [

        7,
        14,
        21

    ]:

        delta = close.diff()

        gain = delta.clip(
            lower=0
        ).rolling(
            period
        ).mean()

        loss = (
            -delta.clip(
                upper=0
            )
            .rolling(period)
            .mean()
        )

        rs = gain / (
            loss + 1e-12
        )

        data[
            f"rsi{period}"
        ] = (

            100
            -
            (
                100 /
                (1 + rs)
            )

        )


    # Bollinger Bands

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
        / close
        - 1

    )


    data["bb_lower"] = (

        (middle - 2 * std)
        / close
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


    # Future return

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

    data = data.reset_index(
        drop=True
    )


    return data


# =========================================================
# TRANSFORMER
# =========================================================

class AIModel(
    nn.Module
):

    def __init__(
        self,
        feature_count
    ):

        super().__init__()


        self.projection = nn.Linear(
            feature_count,
            96
        )


        encoder_layer = (

            nn.TransformerEncoderLayer(

                d_model=96,

                nhead=8,

                batch_first=True,

                dropout=0.10

            )

        )


        self.encoder = (

            nn.TransformerEncoder(

                encoder_layer,

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

        x = self.projection(x)

        x = self.encoder(x)

        x = x[:, -1]

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
# DEEP LEARNING
# =========================================================

def deep_learning_prediction(
    data
):

    if len(data) < (
        SEQ_LEN + 300
    ):

        return None


    features = data[
        FEATURES
    ].values.astype(
        np.float32
    )


    future_return = (
        data["future_return"]
        .values
        .astype(np.float32)
    )


    n = len(features) - LOOKAHEAD


    # Normalization

    mean = (
        features[:n]
        .mean(axis=0)
    )

    std = (
        features[:n]
        .std(axis=0)
        + 1e-6
    )


    normalized = (
        features - mean
    ) / std


    # Training samples

    sample_count = min(
        2500,
        n - SEQ_LEN
    )


    indexes = np.linspace(

        SEQ_LEN,

        n - 1,

        sample_count,

        dtype=int

    )


    X = np.stack([

        normalized[
            i-SEQ_LEN:i
        ]

        for i in indexes

    ])


    returns = (
        future_return[
            indexes
        ]
    )


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


    model = AIModel(
        len(FEATURES)
    ).to(device)


    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=0.0002,

            weight_decay=0.0001

        )
    )


    classification_loss = (
        nn.CrossEntropyLoss()
    )


    regression_loss = (
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
                start:start+64
            ]


            logits, predicted_return = (
                model(
                    X_tensor[batch]
                )
            )


            loss = (

                classification_loss(
                    logits,
                    y_tensor[batch]
                )

                +

                0.35
                *
                regression_loss(
                    predicted_return.squeeze(),
                    return_tensor[batch]
                )

            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


    # Current prediction

    model.eval()


    current = torch.tensor(

        normalized[-SEQ_LEN:][None],

        dtype=torch.float32,

        device=device

    )


    with torch.no_grad():

        logits, predicted_return = (
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


        expected_return = (

            float(
                predicted_return
                .squeeze()
                .cpu()
            )

            * 100

        )


    confidence = (
        float(
            max(probabilities)
        )
        * 100
    )


    if probabilities[1] >= 0.5:

        signal = "BUY"

    else:

        signal = "SELL"


    return (

        signal,

        confidence,

        expected_return

    )


# =========================================================
# SCORING
# =========================================================

def calculate_score(
    data,
    prediction
):

    signal, confidence, expected = (
        prediction
    )


    volatility = (

        float(
            data["vol50"]
            .iloc[-1]
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
            volatility
            * 250,
            0,
            100
        )
    )


    liquidity = float(
        np.clip(
            np.log10(
                data[
                    "quote_volume"
                ]
                .tail(96)
                .mean()
                + 1
            )
            * 10,
            0,
            100
        )
    )


    ranking_score = (

        expected
        * 0.45

        +

        confidence
        * 0.25

        +

        momentum
        * 0.12

        +

        trend
        * 0.10

        +

        liquidity
        * 0.08

        -

        risk
        * 0.20

    )


    return {

        "signal": signal,

        "confidence": confidence,

        "expected_return": expected,

        "risk": risk,

        "momentum": momentum,

        "trend": trend,

        "liquidity": liquidity,

        "score": ranking_score

    }


# =========================================================
# PARİTE TARAMA
# =========================================================

def scan_symbol(
    symbol,
    candle_count
):

    df = get_klines(
        symbol,
        candle_count
    )


    if len(df) < 5000:

        return None


    # Likidite filtresi

    average_volume = float(

        df[
            "quote_volume"
        ]
        .tail(96)
        .mean()

    )


    if average_volume < 1_000_000:

        return None


    data = create_features(
        df
    )


    prediction = (
        deep_learning_prediction(
            data
        )
    )


    if prediction is None:

        return None


    score = calculate_score(
        data,
        prediction
    )


    return {

        "symbol": symbol,

        **score,

        "price": float(
            df[
                "close"
            ].iloc[-1]
        ),

        "candles": len(df)

    }


# =========================================================
# BAŞLIK
# =========================================================

st.title(
    "📈 Binance Spot AI Scanner"
)

st.caption(
    "500.000 mum · 15 dakika · "
    "Deep Learning · BUY / SELL"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Tarama Ayarları"
    )


    candle_count = st.number_input(

        "Mum sayısı",

        min_value=5000,

        max_value=500000,

        value=20000,

        step=5000

    )


    pair_count = st.number_input(

        "Taranacak coin",

        min_value=10,

        max_value=100,

        value=10,

        step=5

    )


    auto_refresh = st.checkbox(

        "Her 15 dakikada yenile",

        value=True

    )


    scan_button = st.button(

        "🚀 Şimdi Tara",

        use_container_width=True

    )


    st.divider()


    st.write(
        "Timeframe:",
        "15 dakika"
    )

    st.write(
        "Model:",
        "Transformer"
    )

    st.write(
        "Horizon:",
        "12 mum / 3 saat"
    )


# =========================================================
# 15 DAKİKALIK REFRESH
# =========================================================

if auto_refresh:

    st_autorefresh(

        interval=900_000,

        key="15_minute_refresh"

    )


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:

    st.session_state.results = []


if "last_scan" not in st.session_state:

    st.session_state.last_scan = None


# =========================================================
# SCAN KARARI
# =========================================================

if (
    scan_button
    or
    len(
        st.session_state.results
    ) == 0
):


    try:

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


        progress = st.progress(
            0
        )


        status = st.empty()


        results = []


        for index, symbol in enumerate(
            selected,
            1
        ):


            status.write(

                f"🔎 **{symbol}** "
                f"taranıyor "
                f"({index}/{len(selected)})"

            )


            try:

                result = scan_symbol(

                    symbol,

                    int(
                        candle_count
                    )

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

                index /
                len(selected)

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


        progress.empty()

        status.empty()


    except Exception as error:

        st.error(

            "Binance API bağlantısı "
            "başarısız: "
            + str(error)

        )


# =========================================================
# SONUÇLAR
# =========================================================

results = (
    st.session_state.results
)


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

        len(results)

    )


with col3:

    st.metric(

        "Mum",

        f"{int(candle_count):,}"

    )


with col4:

    st.metric(

        "Yenileme",

        "15 dakika"

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


# =========================================================
# TABLO
# =========================================================

if not results:

    st.warning(
        "Henüz öneri bulunamadı."
    )

else:

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
                item[
                    "expected_return"
                ],
                3
            ),

            "CONFIDENCE %":
            round(
                item[
                    "confidence"
                ],
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


    df_results = pd.DataFrame(
        table
    )


    st.dataframe(

        df_results,

        use_container_width=True,

        hide_index=True

    )


# =========================================================
# UYARI
# =========================================================

st.info(

    "Bu sistem yalnızca BUY/SELL model sinyali üretir. "
    "Binance hesabına emir göndermez. "
    "Beklenen getiri garanti edilmiş kâr değildir. "
    "500.000 mum ve çok sayıda coin taraması ciddi CPU/RAM/API "
    "kaynağı gerektirir."

)
