import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Binance Spot AI Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

BINANCE_URL = "https://api.binance.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3"

INTERVAL = "15m"

CRASH_BARS = 960
WEEKLY_BARS = 672

PREDICTION_BARS = 2

BUY_TARGET = 0.10
SELL_TARGET = -0.10

MIN_DATA = 300

DEFAULT_TRAINING_ROWS = 700


# ============================================================
# FEATURES
# ============================================================

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
    "low_distance"
]


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/154.0 Safari/537.36"
    ),
    "Accept": "application/json"
})


# ============================================================
# GENERIC REQUEST
# ============================================================

def request_json(
    url,
    params=None,
    timeout=30
):

    response = session.get(
        url,
        params=params,
        timeout=timeout
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# ============================================================
# BINANCE DIRECT TEST
# ============================================================

def test_binance():

    try:

        data = request_json(
            BINANCE_URL + "/api/v3/exchangeInfo",
            timeout=10
        )

        if not data.get("symbols"):
            raise RuntimeError(
                "Binance exchangeInfo boş."
            )

        return True, ""

    except Exception as e:

        return False, str(e)


# ============================================================
# BINANCE ALL SPOT SYMBOLS
# ============================================================

@st.cache_data(ttl=300)
def get_binance_symbols_direct():

    data = request_json(
        BINANCE_URL + "/api/v3/exchangeInfo"
    )

    rows = []

    for item in data.get(
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

        if item.get(
            "isSpotTradingAllowed"
        ) is False:

            continue

        symbol = item.get(
            "symbol"
        )

        if not symbol:
            continue

        rows.append({
            "symbol": symbol,
            "base": item.get(
                "baseAsset",
                ""
            ),
            "quote": "USDT"
        })

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "Binance Spot USDT pariteleri boş."
        )

    return df


# ============================================================
# BINANCE 24H DIRECT
# ============================================================

@st.cache_data(ttl=120)
def get_binance_24h_direct():

    data = request_json(
        BINANCE_URL
        + "/api/v3/ticker/24hr"
    )

    rows = []

    for item in data:

        symbol = item.get(
            "symbol",
            ""
        )

        if not symbol.endswith(
            "USDT"
        ):
            continue

        try:

            quote_volume = float(
                item.get(
                    "quoteVolume",
                    0
                )
                or 0
            )

        except Exception:

            quote_volume = 0

        try:

            price_change = float(
                item.get(
                    "priceChangePercent",
                    0
                )
                or 0
            )

        except Exception:

            price_change = 0

        try:

            trade_count = int(
                item.get(
                    "count",
                    0
                )
                or 0
            )

        except Exception:

            trade_count = 0

        rows.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "price_change_24h": price_change,
            "trade_count": trade_count
        })

    return pd.DataFrame(rows)


# ============================================================
# COINGECKO BINANCE MARKETS
# ============================================================

@st.cache_data(ttl=300)
def get_coingecko_binance_markets():

    rows = []

    page = 1

    max_pages = 20

    while page <= max_pages:

        try:

            data = request_json(
                COINGECKO_URL
                + "/exchanges/binance/tickers",
                params={
                    "page": page,
                    "order": "volume_desc"
                },
                timeout=30
            )

        except Exception:

            break

        tickers = data.get(
            "tickers",
            []
        )

        if not tickers:

            break

        for item in tickers:

            target = str(
                item.get(
                    "target",
                    ""
                )
            ).upper()

            base = str(
                item.get(
                    "base",
                    ""
                )
            ).upper()

            market = str(
                item.get(
                    "market",
                    {}).get(
                        "name",
                        ""
                    )
            )

            if target != "USDT":
                continue

            if (
                market
                and "Binance" not in market
            ):
                continue

            if not base:
                continue

            symbol = (
                base
                + "USDT"
            )

            try:

                volume = float(
                    item.get(
                        "converted_volume",
                        {}).get(
                            "usd",
                            0
                        )
                    or 0
                )

            except Exception:

                volume = 0

            try:

                last = float(
                    item.get(
                        "last",
                        0
                    )
                    or 0
                )

            except Exception:

                last = 0

            rows.append({
                "symbol": symbol,
                "base": base,
                "quote": "USDT",
                "quote_volume": volume,
                "last_price": last
            })

        page += 1

        if len(tickers) < 100:
            break

        time.sleep(0.15)

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "CoinGecko üzerinden Binance "
            "USDT pariteleri alınamadı."
        )

    df = (
        df
        .drop_duplicates(
            "symbol"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# COINGECKO COIN ID MAP
# ============================================================

@st.cache_data(ttl=1800)
def get_coingecko_coin_map():

    rows = []

    page = 1

    max_pages = 20

    while page <= max_pages:

        try:

            data = request_json(
                COINGECKO_URL
                + "/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": "false"
                },
                timeout=30
            )

        except Exception:

            break

        if not data:
            break

        for coin in data:

            symbol = str(
                coin.get(
                    "symbol",
                    ""
                )
            ).upper()

            coin_id = coin.get(
                "id"
            )

            if not symbol or not coin_id:
                continue

            rows.append({
                "symbol_short": symbol,
                "coin_id": coin_id,
                "name": coin.get(
                    "name",
                    ""
                )
            })

        if len(data) < 250:
            break

        page += 1

        time.sleep(0.15)

    return pd.DataFrame(rows)


# ============================================================
# COINGECKO CANDLES
# ============================================================

def get_coingecko_candles(
    coin_id,
    days=10
):

    data = request_json(
        COINGECKO_URL
        + f"/coins/{coin_id}/market_chart",
        params={
            "vs_currency": "usd",
            "days": days,
            "interval": "15m"
        },
        timeout=30
    )

    prices = data.get(
        "prices",
        []
    )

    volumes = data.get(
        "total_volumes",
        []
    )

    if not prices:

        return pd.DataFrame()

    price_df = pd.DataFrame(
        prices,
        columns=[
            "timestamp",
            "close"
        ]
    )

    price_df["timestamp"] = (
        pd.to_datetime(
            price_df[
                "timestamp"
            ],
            unit="ms",
            utc=True
        )
    )

    price_df["close"] = (
        pd.to_numeric(
            price_df["close"],
            errors="coerce"
        )
    )

    # CoinGecko market_chart
    # her zaman OHLC sağlamayabilir.
    # Bu yüzden close'dan yaklaşık candle
    # oluşturuyoruz.

    price_df["open"] = (
        price_df["close"].shift(1)
    )

    price_df["open"] = (
        price_df["open"]
        .fillna(
            price_df["close"]
        )
    )

    price_df["high"] = (
        price_df[
            ["open", "close"]
        ].max(axis=1)
    )

    price_df["low"] = (
        price_df[
            ["open", "close"]
        ].min(axis=1)
    )

    if volumes:

        volume_df = pd.DataFrame(
            volumes,
            columns=[
                "volume_timestamp",
                "volume"
            ]
        )

        volume_df[
            "volume_timestamp"
        ] = pd.to_datetime(
            volume_df[
                "volume_timestamp"
            ],
            unit="ms",
            utc=True
        )

        volume_df["volume"] = (
            pd.to_numeric(
                volume_df["volume"],
                errors="coerce"
            )
            .fillna(0)
        )

        price_df = pd.merge_asof(
            price_df.sort_values(
                "timestamp"
            ),
            volume_df.sort_values(
                "volume_timestamp"
            ),
            left_on="timestamp",
            right_on="volume_timestamp",
            direction="nearest"
        )

        price_df = price_df.drop(
            columns=[
                "volume_timestamp"
            ],
            errors="ignore"
        )

    else:

        price_df["volume"] = 0.0

    price_df = (
        price_df
        .dropna(
            subset=[
                "close"
            ]
        )
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )

    return price_df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


# ============================================================
# PROVIDER SELECTION
# ============================================================

def load_market_data():

    direct_ok, direct_error = (
        test_binance()
    )

    if direct_ok:

        try:

            symbols = (
                get_binance_symbols_direct()
            )

            tickers = (
                get_binance_24h_direct()
            )

            if not tickers.empty:

                symbols = symbols.merge(
                    tickers,
                    on="symbol",
                    how="left"
                )

            symbols["data_provider"] = (
                "Binance API"
            )

            return (
                symbols,
                "Binance API",
                ""
            )

        except Exception as e:

            direct_error = str(e)

    # ========================================================
    # FALLBACK
    # ========================================================

    try:

        markets = (
            get_coingecko_binance_markets()
        )

        markets["data_provider"] = (
            "CoinGecko Binance Market Data"
        )

        return (
            markets,
            "CoinGecko Binance",
            direct_error
        )

    except Exception as fallback_error:

        raise RuntimeError(
            "Binance API HTTP 451 veya erişim "
            "problemi ve alternatif veri kaynağı "
            "da başarısız oldu.\n\n"
            f"Binance: {direct_error}\n\n"
            f"Fallback: {fallback_error}"
        )


# ============================================================
# ACTIVITY SCORE
# ============================================================

def activity_score(df):

    x = df.copy()

    if x.empty:
        return x

    if "quote_volume" not in x.columns:

        x["quote_volume"] = 0

    if "trade_count" not in x.columns:

        x["trade_count"] = 0

    if "price_change_24h" not in x.columns:

        x["price_change_24h"] = 0

    x["quote_volume"] = (
        pd.to_numeric(
            x["quote_volume"],
            errors="coerce"
        )
        .fillna(0)
    )

    x["trade_count"] = (
        pd.to_numeric(
            x["trade_count"],
            errors="coerce"
        )
        .fillna(0)
    )

    x["price_change_24h"] = (
        pd.to_numeric(
            x["price_change_24h"],
            errors="coerce"
        )
        .fillna(0)
    )

    volume_log = np.log1p(
        x["quote_volume"]
        .clip(lower=0)
    )

    trades_log = np.log1p(
        x["trade_count"]
        .clip(lower=0)
    )

    movement = (
        x["price_change_24h"]
        .abs()
    )

    def normalize(s):

        mn = s.min()
        mx = s.max()

        if mx <= mn:

            return pd.Series(
                1.0,
                index=s.index
            )

        return (
            (s - mn)
            / (mx - mn)
        )

    x["volume_norm"] = normalize(
        volume_log
    )

    x["trade_norm"] = normalize(
        trades_log
    )

    x["movement_norm"] = normalize(
        movement
    )

    x["activity_score"] = (
        x["volume_norm"] * 60
        + x["trade_norm"] * 20
        + x["movement_norm"] * 20
    )

    return (
        x
        .sort_values(
            "activity_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EMA
# ============================================================

def ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(
    close,
    period
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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100
        - 100 / (1 + rs)
    )

    return result.fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period=14
):

    previous = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous
            ).abs(),
            (
                df["low"]
                - previous
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# ADX
# ============================================================

def adx(
    df,
    period=14
):

    up = df["high"].diff()

    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up > down)
            & (up > 0),
            up,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up)
            & (down > 0),
            down,
            0
        ),
        index=df.index
    )

    atr_value = (
        atr(
            df,
            period
        )
        .replace(
            0,
            np.nan
        )
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di
        + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        * (
            plus_di
            - minus_di
        ).abs()
        / denominator
    )

    return (
        dx.ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
        .fillna(0)
    )


# ============================================================
# FEATURES
# ============================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["ret1"] = (
        close.pct_change(1)
    )

    x["ret3"] = (
        close.pct_change(3)
    )

    x["ret6"] = (
        close.pct_change(6)
    )

    x["ret12"] = (
        close.pct_change(12)
    )

    x["ret24"] = (
        close.pct_change(24)
    )

    e5 = ema(close, 5)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)

    x["ema5_20"] = (
        e5 / e20 - 1
    )

    x["ema20_50"] = (
        e20 / e50 - 1
    )

    x["ema50_200"] = (
        e50 / e200 - 1
    )

    x["dist20"] = (
        close / e20 - 1
    )

    x["dist50"] = (
        close / e50 - 1
    )

    x["dist200"] = (
        close / e200 - 1
    )

    x["rsi7"] = rsi(
        close,
        7
    )

    x["rsi14"] = rsi(
        close,
        14
    )

    x["rsi21"] = rsi(
        close,
        21
    )

    macd = (
        ema(close, 12)
        - ema(close, 26)
    )

    macd_signal = ema(
        macd,
        9
    )

    x["macd"] = macd
    x["macd_signal"] = macd_signal
    x["macd_hist"] = (
        macd - macd_signal
    )

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

    upper = (
        middle + 2 * std
    )

    lower = (
        middle - 2 * std
    )

    width = (
        upper - lower
    ).replace(
        0,
        np.nan
    )

    x["bb_position"] = (
        (close - lower)
        / width
    )

    x["bb_width"] = (
        width
        / middle.replace(
            0,
            np.nan
        )
    )

    atr_value = atr(
        x,
        14
    )

    x["atr_pct"] = (
        atr_value
        / close.replace(
            0,
            np.nan
        )
    )

    x["adx"] = adx(
        x,
        14
    )

    low14 = (
        low
        .rolling(14)
        .min()
    )

    high14 = (
        high
        .rolling(14)
        .max()
    )

    denominator = (
        high14 - low14
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        * (close - low14)
        / denominator
    )

    d = (
        k
        .rolling(3)
        .mean()
    )

    x["stoch_k"] = k.fillna(50)
    x["stoch_d"] = d.fillna(50)

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan
    )

    body = (
        close
        - x["open"]
    ).abs()

    x["body_pct"] = (
        body
        / close.replace(
            0,
            np.nan
        )
    )

    x["range_pct"] = (
        candle_range
        / close.replace(
            0,
            np.nan
        )
    )

    x["upper_wick"] = (
        high
        - np.maximum(
            x["open"],
            close
        )
    ) / candle_range

    x["lower_wick"] = (
        np.minimum(
            x["open"],
            close
        )
        - low
    ) / candle_range

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
        / volume_mean.replace(
            0,
            np.nan
        )
    )

    x["volume_z"] = (
        volume
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    x["volatility12"] = (
        x["ret1"]
        .rolling(12)
        .std()
    )

    x["volatility24"] = (
        x["ret1"]
        .rolling(24)
        .std()
    )

    x["volatility48"] = (
        x["ret1"]
        .rolling(48)
        .std()
    )

    price_mean = (
        close
        .rolling(48)
        .mean()
    )

    price_std = (
        close
        .rolling(48)
        .std()
    )

    x["price_z"] = (
        close
        - price_mean
    ) / price_std.replace(
        0,
        np.nan
    )

    x["high_distance"] = (
        high / close - 1
    )

    x["low_distance"] = (
        low / close - 1
    )

    return x


# ============================================================
# CRASH
# ============================================================

def calculate_crash(
    df,
    threshold
):

    result = {
        "peak": np.nan,
        "price": np.nan,
        "drop": np.nan,
        "is_crash": False
    }

    if len(df) < 20:
        return result

    completed = df.iloc[:-1]

    window = (
        completed
        .tail(CRASH_BARS)
    )

    peak = float(
        window["high"].max()
    )

    price = float(
        completed["close"].iloc[-1]
    )

    if peak <= 0:
        return result

    drop = (
        1
        - price / peak
    ) * 100

    result["peak"] = peak
    result["price"] = price
    result["drop"] = drop

    result["is_crash"] = (
        drop >= threshold
    )

    return result


# ============================================================
# WEEKLY PATTERN
# ============================================================

def weekly_pattern(df):

    result = {
        "weekly_low": np.nan,
        "pattern": False,
        "three_rising": False,
        "green": False,
        "near_low": False,
        "dip_before": False
    }

    if len(df) < (
        WEEKLY_BARS + 3
    ):
        return result

    completed = df.iloc[:-1]

    weekly = (
        completed
        .tail(WEEKLY_BARS)
    )

    last3 = (
        completed
        .tail(3)
    )

    weekly_low = float(
        weekly["low"].min()
    )

    recent_low = float(
        last3["low"].min()
    )

    if weekly_low <= 0:
        return result

    distance = (
        recent_low
        / weekly_low
        - 1
    )

    rising = (
        last3["close"].iloc[0]
        < last3["close"].iloc[1]
        < last3["close"].iloc[2]
    )

    green = (
        last3["close"].iloc[2]
        > last3["open"].iloc[2]
    )

    low_index = (
        weekly["low"].idxmin()
    )

    first_index = (
        last3.index[0]
    )

    dip_before = (
        low_index
        < first_index
    )

    near_low = (
        distance <= 0.015
    )

    result["weekly_low"] = (
        weekly_low
    )

    result["three_rising"] = bool(
        rising
    )

    result["green"] = bool(
        green
    )

    result["near_low"] = bool(
        near_low
    )

    result["dip_before"] = bool(
        dip_before
    )

    result["pattern"] = bool(
        rising
        and green
        and near_low
        and dip_before
    )

    return result


# ============================================================
# TARGET
# ============================================================

def make_target(df):

    target = np.full(
        len(df),
        np.nan
    )

    close = df[
        "close"
    ].values

    high = df[
        "high"
    ].values

    low = df[
        "low"
    ].values

    for i in range(
        len(df)
        - PREDICTION_BARS
    ):

        entry = close[i]

        future_high = np.max(
            high[
                i + 1:
                i + 1
                + PREDICTION_BARS
            ]
        )

        future_low = np.min(
            low[
                i + 1:
                i + 1
                + PREDICTION_BARS
            ]
        )

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

        if (
            buy_hit
            and not sell_hit
        ):

            target[i] = 1

        elif (
            sell_hit
            and not buy_hit
        ):

            target[i] = 0

        elif (
            buy_hit
            and sell_hit
        ):

            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

        else:

            target[i] = (
                1
                if up >= abs(down)
                else 0
            )

    return target


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(df):

    features = build_features(
        df
    )

    row = features.iloc[-2]

    buy = 0
    sell = 0

    if row["ema5_20"] > 0:
        buy += 10
    else:
        sell += 10

    if row["ema20_50"] > 0:
        buy += 10
    else:
        sell += 10

    if row["ema50_200"] > 0:
        buy += 8
    else:
        sell += 8

    if row["rsi14"] < 35:
        buy += 10

    elif row["rsi14"] > 65:
        sell += 10

    if row["macd_hist"] > 0:
        buy += 8
    else:
        sell += 8

    if row["bb_position"] < 0.25:
        buy += 10

    elif row["bb_position"] > 0.75:
        sell += 10

    if row["stoch_k"] < 25:
        buy += 8

    elif row["stoch_k"] > 75:
        sell += 8

    if row["close"] > row["open"]:
        buy += 8
    else:
        sell += 8

    return buy, sell


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(
                        96,
                        48,
                        24
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# AI
# ============================================================

def ai_predict(
    df,
    training_rows
):

    features = build_features(
        df
    )

    target = make_target(
        df
    )

    feature_df = (
        features[
            FEATURES
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    target_series = pd.Series(
        target,
        index=df.index
    )

    valid = (
        feature_df
        .notna()
        .all(axis=1)
        &
        target_series.notna()
    )

    X = feature_df.loc[
        valid
    ].values

    y = (
        target_series.loc[
            valid
        ]
        .astype(int)
        .values
    )

    if len(X) < 300:

        raise ValueError(
            "AI için yeterli veri yok."
        )

    if len(X) > training_rows:

        X = X[
            -training_rows:
        ]

        y = y[
            -training_rows:
        ]

    if len(
        np.unique(y)
    ) < 2:

        raise ValueError(
            "AI tek sınıf gördü."
        )

    split = int(
        len(X) * 0.80
    )

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    prediction = (
        model.predict(
            X_test
        )
    )

    accuracy = float(
        np.mean(
            prediction
            == y_test
        )
    )

    latest = feature_df.iloc[
        -2:-1
    ]

    probabilities = (
        model.predict_proba(
            latest.values
        )[0]
    )

    p_buy = 0
    p_sell = 0

    for cls, probability in zip(
        model.classes_,
        probabilities
    ):

        if int(cls) == 1:

            p_buy = float(
                probability
            )

        else:

            p_sell = float(
                probability
            )

    return {
        "p_buy": p_buy,
        "p_sell": p_sell,
        "accuracy": accuracy
    }


# ============================================================
# ANALYZE
# ============================================================

def analyze_symbol(
    symbol,
    df,
    crash_threshold,
    training_rows
):

    if df.empty:

        raise ValueError(
            "Mum verisi boş."
        )

    if len(df) < MIN_DATA:

        raise ValueError(
            f"Yetersiz mum: {len(df)}"
        )

    crash = calculate_crash(
        df,
        crash_threshold
    )

    weekly = weekly_pattern(
        df
    )

    tech_buy, tech_sell = (
        technical_score(df)
    )

    try:

        ai = ai_predict(
            df,
            training_rows
        )

        p_buy = ai[
            "p_buy"
        ]

        p_sell = ai[
            "p_sell"
        ]

        accuracy = ai[
            "accuracy"
        ]

        ai_status = "OK"

        buy_score = (
            p_buy * 100
            + tech_buy * 0.35
        )

        sell_score = (
            p_sell * 100
            + tech_sell * 0.35
        )

    except Exception as e:

        total = (
            tech_buy
            + tech_sell
        )

        if total > 0:

            p_buy = (
                tech_buy
                / total
            )

            p_sell = (
                tech_sell
                / total
            )

        else:

            p_buy = 0.5
            p_sell = 0.5

        accuracy = np.nan

        buy_score = tech_buy
        sell_score = tech_sell

        ai_status = (
            "FALLBACK: "
            + str(e)
        )

    crash_bonus = 0

    if crash[
        "is_crash"
    ]:

        crash_bonus = 20

        buy_score += 20

        p_buy = min(
            1,
            p_buy + 0.10
        )

    pattern_bonus = 0

    if weekly[
        "pattern"
    ]:

        pattern_bonus = 20

        buy_score += 20

        p_buy = min(
            1,
            p_buy + 0.10
        )

    signal = (
        "BUY"
        if buy_score >= sell_score
        else "SELL"
    )

    price = float(
        df.iloc[-2][
            "close"
        ]
    )

    return {
        "symbol": symbol,

        "signal": signal,

        "price": price,

        "buy_score": round(
            buy_score,
            2
        ),

        "sell_score": round(
            sell_score,
            2
        ),

        "p_buy": round(
            p_buy * 100,
            2
        ),

        "p_sell": round(
            p_sell * 100,
            2
        ),

        "10d_peak": round(
            crash["peak"],
            10
        ),

        "10d_drop": round(
            crash["drop"],
            2
        ),

        "crash": bool(
            crash["is_crash"]
        ),

        "crash_bonus": (
            crash_bonus
        ),

        "weekly_low": round(
            weekly["weekly_low"],
            10
        ),

        "weekly_pattern": bool(
            weekly["pattern"]
        ),

        "three_rising": bool(
            weekly["three_rising"]
        ),

        "green_last": bool(
            weekly["green"]
        ),

        "ai_accuracy": (
            round(
                accuracy * 100,
                2
            )
            if not pd.isna(
                accuracy
            )
            else np.nan
        ),

        "ai_status": ai_status,

        "candles": len(df)
    }


# ============================================================
# LOAD CANDLES
# ============================================================

def load_candles(
    provider,
    symbol,
    coin_map=None
):

    # --------------------------------------------------------
    # DIRECT BINANCE
    # --------------------------------------------------------

    if provider == "Binance API":

        return get_binance_candles_direct(
            symbol
        )

    # --------------------------------------------------------
    # COINGECKO FALLBACK
    # --------------------------------------------------------

    if provider == "CoinGecko Binance":

        if coin_map is None:

            raise ValueError(
                "CoinGecko coin map yok."
            )

        base = symbol.replace(
            "USDT",
            ""
        ).upper()

        matches = coin_map[
            coin_map[
                "symbol_short"
            ] == base
        ]

        if matches.empty:

            raise ValueError(
                f"{symbol} CoinGecko "
                "coin ID bulunamadı."
            )

        # Önce ilk ID
        coin_id = matches.iloc[
            0
        ]["coin_id"]

        return get_coingecko_candles(
            coin_id,
            days=10
        )

    raise ValueError(
        "Bilinmeyen provider."
    )


# ============================================================
# DIRECT BINANCE CANDLES
# ============================================================

def get_binance_candles_direct(
    symbol,
    limit=1000
):

    data = request_json(
        BINANCE_URL
        + "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": min(
                int(limit),
                1000
            )
        }
    )

    rows = []

    for item in data:

        if len(item) < 6:
            continue

        rows.append({
            "timestamp": pd.to_datetime(
                int(item[0]),
                unit="ms",
                utc=True
            ),

            "open": float(
                item[1]
            ),

            "high": float(
                item[2]
            ),

            "low": float(
                item[3]
            ),

            "close": float(
                item[4]
            ),

            "volume": float(
                item[5]
            )
        })

    if not rows:

        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )


# ============================================================
# SCANNER
# ============================================================

def run_scan(
    selected,
    provider,
    coin_map,
    crash_threshold,
    training_rows,
    delay
):

    results = []

    errors = []

    total = len(
        selected
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    start = time.time()

    for i, item in enumerate(
        selected,
        start=1
    ):

        symbol = item[
            "symbol"
        ]

        status.write(
            f"{i:,}/{total:,} "
            f"→ {symbol}"
        )

        try:

            df = load_candles(
                provider,
                symbol,
                coin_map
            )

            if df.empty:

                raise ValueError(
                    "Mum verisi yok."
                )

            result = analyze_symbol(
                symbol,
                df,
                crash_threshold,
                training_rows
            )

            result[
                "activity_score"
            ] = round(
                float(
                    item.get(
                        "activity_score",
                        0
                    )
                ),
                3
            )

            result[
                "24h_volume"
            ] = float(
                item.get(
                    "quote_volume",
                    0
                )
                or 0
            )

            result[
                "24h_change"
            ] = float(
                item.get(
                    "price_change_24h",
                    0
                )
                or 0
            )

            result[
                "trade_count"
            ] = int(
                item.get(
                    "trade_count",
                    0
                )
                or 0
            )

            results.append(
                result
            )

        except Exception as e:

            errors.append({
                "symbol": symbol,
                "error": str(e)
            })

        progress.progress(
            i / total
        )

        if delay > 0:

            time.sleep(
                delay
            )

    progress.empty()
    status.empty()

    elapsed = (
        time.time()
        - start
    )

    return (
        pd.DataFrame(results),
        pd.DataFrame(errors),
        elapsed
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Binance Spot "
    "15M Deep Learning Scanner"
)

st.caption(
    "Tüm Binance Spot USDT pariteleri • "
    "15M • 30 dakika tahmin • "
    "Deep Learning • BUY / SELL"
)

st.warning(
    "Bu uygulama yalnızca analiz yapar. "
    "Otomatik emir göndermez."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    crash_threshold = st.slider(
        "10 Günlük Düşüş %",
        50.0,
        95.0,
        80.0,
        1.0
    )

    training_rows = st.number_input(
        "AI Training Candles",
        300,
        900,
        700,
        100
    )

    delay = st.number_input(
        "API Delay",
        0.0,
        2.0,
        0.10,
        0.01
    )

    st.divider()

    st.write(
        "📊 Spot USDT"
    )

    st.write(
        "⏱️ 15M"
    )

    st.write(
        "🔮 30 dakika"
    )

    st.write(
        "🟢 BUY +10%"
    )

    st.write(
        "🔴 SELL -10%"
    )

    st.write(
        "🚫 HOLD yok"
    )

    st.write(
        "🔥 Crash bonus +20"
    )

    st.write(
        "🔥 Weekly bonus +20"
    )


# ============================================================
# LOAD MARKETS
# ============================================================

st.subheader(
    "🌐 Binance Tüm Spot Pariteleri"
)

try:

    pairs_df, provider, direct_error = (
        load_market_data()
    )

except Exception as e:

    st.error(
        "Piyasa verileri alınamadı."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# ACTIVITY
# ============================================================

pairs_df = activity_score(
    pairs_df
)


# ============================================================
# PROVIDER INFO
# ============================================================

if provider == "Binance API":

    st.success(
        "🟢 Binance API doğrudan erişilebilir."
    )

else:

    st.warning(
        "🟡 Binance API bu sunucudan "
        "erişilemiyor. Binance Spot market "
        "listesi için CoinGecko fallback "
        "kullanılıyor."
    )

    if direct_error:

        with st.expander(
            "Binance API hatası"
        ):

            st.code(
                direct_error
            )


# ============================================================
# STATISTICS
# ============================================================

total_pairs = len(
    pairs_df
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "USDT Pariteleri",
    f"{total_pairs:,}"
)

c2.metric(
    "En Aktif",
    (
        pairs_df.iloc[0]["symbol"]
        if not pairs_df.empty
        else "-"
    )
)

c3.metric(
    "Top Aktivite",
    (
        f"{pairs_df.iloc[0]['activity_score']:.2f}"
        if not pairs_df.empty
        else "-"
    )
)

c4.metric(
    "Provider",
    provider
)


# ============================================================
# PAIRS TABLE
# ============================================================

with st.expander(
    "📋 Tüm Pariteler"
):

    columns = [
        "symbol",
        "base",
        "quote",
        "quote_volume",
        "price_change_24h",
        "trade_count",
        "activity_score"
    ]

    available_columns = [
        c
        for c in columns
        if c in pairs_df.columns
    ]

    st.dataframe(
        pairs_df[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COINGECKO MAP
# ============================================================

coin_map = pd.DataFrame()

if provider == "CoinGecko Binance":

    with st.spinner(
        "Alternatif coin eşleştirmesi hazırlanıyor..."
    ):

        try:

            coin_map = (
                get_coingecko_coin_map()
            )

        except Exception as e:

            st.error(
                f"Coin eşleştirme hatası: {e}"
            )

            st.stop()


# ============================================================
# SCAN MODE
# ============================================================

st.subheader(
    "🔎 Tarama Modu"
)

mode = st.radio(
    "Parite seçimi",
    [
        "ALL PAIRS",
        "TOP ACTIVE",
        "MANUAL"
    ],
    horizontal=True
)


# ============================================================
# ALL
# ============================================================

if mode == "ALL PAIRS":

    selected = pairs_df.copy()

    st.info(
        f"Tüm {len(selected):,} "
        "Binance USDT paritesi taranacak."
    )


# ============================================================
# TOP ACTIVE
# ============================================================

elif mode == "TOP ACTIVE":

    max_count = max(
        1,
        len(pairs_df)
    )

    default_count = min(
        1000,
        max_count
    )

    count = st.number_input(
        "Aktif parite sayısı",
        min_value=1,
        max_value=max_count,
        value=default_count,
        step=100
    )

    selected = (
        pairs_df
        .head(
            int(count)
        )
        .copy()
    )

    st.info(
        f"Aktivite skoruna göre "
        f"en aktif {len(selected):,} "
        "parite taranacak."
    )


# ============================================================
# MANUAL
# ============================================================

else:

    symbols = (
        pairs_df[
            "symbol"
        ]
        .tolist()
    )

    chosen = st.multiselect(
        "Pariteler",
        symbols,
        default=symbols[:20]
    )

    selected = (
        pairs_df[
            pairs_df[
                "symbol"
            ].isin(chosen)
        ]
        .copy()
    )


# ============================================================
# SELECTED
# ============================================================

st.write(
    f"**Seçilen parite:** "
    f"{len(selected):,}"
)


# ============================================================
# START
# ============================================================

start_scan = st.button(
    "🚀 DEEP LEARNING SCAN",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTE
# ============================================================

if start_scan:

    if selected.empty:

        st.error(
            "Parite seçilmedi."
        )

        st.stop()

    start_timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    with st.spinner(
        "Deep Learning taraması devam ediyor..."
    ):

        results, errors, elapsed = (
            run_scan(
                selected.to_dict(
                    "records"
                ),
                provider,
                coin_map,
                float(
                    crash_threshold
                ),
                int(
                    training_rows
                ),
                float(
                    delay
                )
            )
        )

    st.session_state[
        "results"
    ] = results

    st.session_state[
        "errors"
    ] = errors

    st.session_state[
        "elapsed"
    ] = elapsed

    st.session_state[
        "scan_time"
    ] = start_timestamp


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state[
        "results"
    ]

    errors = st.session_state[
        "errors"
    ]

    elapsed = st.session_state[
        "elapsed"
    ]

    scan_time = st.session_state[
        "scan_time"
    ]

    st.divider()

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    buy_count = 0
    sell_count = 0
    crash_count = 0

    if not results.empty:

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

        crash_count = int(
            results[
                "crash"
            ].sum()
        )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Başarılı",
        f"{len(results):,}"
    )

    c2.metric(
        "Hatalı",
        f"{len(errors):,}"
    )

    c3.metric(
        "BUY",
        f"{buy_count:,}"
    )

    c4.metric(
        "SELL",
        f"{sell_count:,}"
    )

    c5.metric(
        "CRASH",
        f"{crash_count:,}"
    )

    st.caption(
        f"Tarama: {scan_time} | "
        f"Süre: {elapsed:.1f} saniye"
    )


    # --------------------------------------------------------
    # TOP BUY
    # --------------------------------------------------------

    st.subheader(
        "🟢 TOP BUY"
    )

    if not results.empty:

        buy_df = (
            results[
                results[
                    "signal"
                ]
                == "BUY"
            ]
            .sort_values(
                [
                    "buy_score",
                    "p_buy",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if not buy_df.empty:

            st.dataframe(
                buy_df[
                    [
                        "symbol",
                        "price",
                        "buy_score",
                        "p_buy",
                        "p_sell",
                        "activity_score",
                        "10d_drop",
                        "crash",
                        "weekly_pattern",
                        "three_rising",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "BUY bulunamadı."
            )


    # --------------------------------------------------------
    # TOP SELL
    # --------------------------------------------------------

    st.subheader(
        "🔴 TOP SELL"
    )

    if not results.empty:

        sell_df = (
            results[
                results[
                    "signal"
                ]
                == "SELL"
            ]
            .sort_values(
                [
                    "sell_score",
                    "p_sell",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if not sell_df.empty:

            st.dataframe(
                sell_df[
                    [
                        "symbol",
                        "price",
                        "sell_score",
                        "p_sell",
                        "p_buy",
                        "activity_score",
                        "10d_drop",
                        "crash",
                        "weekly_pattern",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "SELL bulunamadı."
            )


    # --------------------------------------------------------
    # CRASH
    # --------------------------------------------------------

    st.subheader(
        f"🔥 SON 10 GÜNDE "
        f"%{crash_threshold:.0f}+ DÜŞENLER"
    )

    if not results.empty:

        crash_df = (
            results[
                results[
                    "10d_drop"
                ]
                >= crash_threshold
            ]
            .sort_values(
                [
                    "10d_drop",
                    "activity_score",
                    "buy_score"
                ],
                ascending=False
            )
        )

        if not crash_df.empty:

            st.dataframe(
                crash_df[
                    [
                        "symbol",
                        "signal",
                        "price",
                        "10d_peak",
                        "10d_drop",
                        "p_buy",
                        "buy_score",
                        "activity_score",
                        "weekly_pattern",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Crash filtresine uyan "
                "parite yok."
            )


    # --------------------------------------------------------
    # WEEKLY
    # --------------------------------------------------------

    st.subheader(
        "🔥 HAFTALIK DİP + 3 YÜKSELEN MUM"
    )

    if not results.empty:

        weekly_df = (
            results[
                results[
                    "weekly_pattern"
                ]
                == True
            ]
            .sort_values(
                [
                    "buy_score",
                    "activity_score"
                ],
                ascending=False
            )
            .head(100)
        )

        if not weekly_df.empty:

            st.dataframe(
                weekly_df[
                    [
                        "symbol",
                        "price",
                        "weekly_low",
                        "buy_score",
                        "p_buy",
                        "10d_drop",
                        "crash",
                        "three_rising",
                        "green_last",
                        "ai_accuracy"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Pattern bulunamadı."
            )


    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "📊 TÜM SONUÇLAR"
    )

    if not results.empty:

        all_results = (
            results
            .sort_values(
                [
                    "activity_score",
                    "buy_score"
                ],
                ascending=False
            )
        )

        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )

        csv = (
            all_results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ SONUÇLARI CSV İNDİR",
            data=csv,
            file_name=(
                "binance_spot_ai_scan.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    st.subheader(
        "⚠️ HATALI PARİTELER"
    )

    if errors.empty:

        st.success(
            "Hatalı parite yok."
        )

    else:

        st.error(
            f"{len(errors):,} paritede "
            "hata oluştu."
        )

        st.dataframe(
            errors,
            use_container_width=True,
            hide_index=True
        )

        error_csv = (
            errors
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ HATALI PARİTELERİ İNDİR",
            data=error_csv,
            file_name=(
                "binance_error_pairs.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "🚀 DEEP LEARNING SCAN "
        "butonuna basarak taramayı başlat."
    )
