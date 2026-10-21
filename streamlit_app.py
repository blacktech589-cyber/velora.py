
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Velora AI Crypto Scanner",
    page_icon="📊",
    layout="wide",
)

API_BASE = "https://api.binance.com"
QUOTE_ASSET = "USDT"

STABLECOINS = {
    "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
    "USDE", "USDD", "UST", "USTC", "EUR", "EURC",
    "PAXG", "AEUR", "EURI"
}

LEVERAGED_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR"
)

# =========================================================
# HTTP SESSION
# =========================================================

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "VeloraCryptoScanner/1.0"
})


class BinanceAPIError(Exception):
    """Readable Binance API error."""


def api_get(path, params=None, timeout=15):
    """Request public Binance market data."""
    url = API_BASE + path

    try:
        response = SESSION.get(
            url,
            params=params,
            timeout=timeout,
        )

        if response.status_code == 451:
            raise BinanceAPIError(
                "Binance API bu bölgeden erişimi kısıtlıyor (HTTP 451). "
                "Bu uygulama bölgesel kısıtlamaları aşmaya çalışmaz."
            )

        if response.status_code in (418, 429):
            raise BinanceAPIError(
                "Binance geçici istek sınırına ulaşıldı. "
                "Biraz bekleyip tekrar deneyin."
            )

        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("code", 0) < 0:
            raise BinanceAPIError(
                data.get("msg", "Binance API hatası.")
            )

        return data

    except BinanceAPIError:
        raise
    except requests.RequestException as exc:
        raise BinanceAPIError(
            f"Binance API bağlantı hatası: {exc}"
        ) from exc


# =========================================================
# COIN UNIVERSE
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_coin_universe():
    """Return active USDT spot pairs, ranked by quote volume."""
    exchange_info = api_get("/api/v3/exchangeInfo")
    tickers = api_get("/api/v3/ticker/24hr")

    active_symbols = set()

    for item in exchange_info.get("symbols", []):
        symbol = item.get("symbol", "")
        base = item.get("baseAsset", "")
        quote = item.get("quoteAsset", "")

        if item.get("status") != "TRADING":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        if quote != QUOTE_ASSET:
            continue

        if base in STABLECOINS:
            continue

        if base.endswith(LEVERAGED_SUFFIXES):
            continue

        active_symbols.add(symbol)

    rows = []

    for ticker in tickers:
        symbol = ticker.get("symbol", "")

        if symbol not in active_symbols:
            continue

        try:
            price = float(ticker.get("lastPrice", 0))
            quote_volume = float(
                ticker.get("quoteVolume", 0)
            )
            change = float(
                ticker.get("priceChangePercent", 0)
            )
        except (TypeError, ValueError):
            continue

        if price <= 0:
            continue

        rows.append({
            "symbol": symbol,
            "price": price,
            "change_24h": change,
            "quote_volume": quote_volume,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "quote_volume",
        ascending=False,
    ).reset_index(drop=True)


# =========================================================
# CANDLE DATA
# =========================================================

@st.cache_data(ttl=45, show_spinner=False)
def get_candles(symbol, interval="15m", limit=500):
    """
    Download candles and exclude the current unfinished candle.
    """
    data = api_get(
        "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": min(max(int(limit), 100), 1000),
        },
    )

    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trade_count", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    for col in [
        "open", "high", "low", "close",
        "volume", "quote_volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    # Binance klines are timestamped in UTC.
    # Keep only candles whose close time has passed.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now_utc].copy()

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    return df.reset_index(drop=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Handle uninterrupted moves.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0,
    )

    return rsi


def calculate_indicators(df):
    """Calculate indicators using completed candles."""
    df = df.copy()

    if len(df) < 60:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["ema_9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema_20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema_50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    # Bollinger Bands
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    df["bb_width_pct"] = (
        (df["bb_upper"] - df["bb_lower"])
        / df["bb_mid"].replace(0, np.nan)
        * 100
    )

    # RSI
    df["rsi"] = calculate_rsi(close, 14)

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Stochastic
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()

    denominator = (highest_high - lowest_low).replace(
        0, np.nan
    )

    df["stoch_k"] = (
        (close - lowest_low) / denominator * 100
    )

    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    df["atr_pct"] = (
        df["atr"] / close.replace(0, np.nan) * 100
    )

    # ADX / Directional Indicators
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr_safe = df["atr"].replace(0, np.nan)

    df["plus_di"] = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    df["minus_di"] = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    di_sum = (
        df["plus_di"] + df["minus_di"]
    ).replace(0, np.nan)

    df["dx"] = (
        100
        * (df["plus_di"] - df["minus_di"]).abs()
        / di_sum
    )

    df["adx"] = df["dx"].ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    # Volume
    df["volume_sma"] = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / df["volume_sma"].replace(0, np.nan)
    )

    # Candle body and direction
    df["candle_change_pct"] = (
        (close - df["open"])
        / df["open"].replace(0, np.nan)
        * 100
    )

    # Recent price change
    df["change_3_candles_pct"] = (
        close / close.shift(3) - 1
    ) * 100

    df["change_20_candles_pct"] = (
        close / close.shift(20) - 1
    ) * 100

    # Rolling low for context
    df["low_30d"] = low.rolling(
        min(2880, len(df)),
        min_periods=60,
    ).min()

    df["distance_from_rolling_low_pct"] = (
        (close / df["low_30d"]) - 1
    ) * 100

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def make_signal(df, min_votes=4, min_margin=2):
    """
    Rules-based indicator score.
    This is not a trained or guaranteed predictive model.
    """
    if df is None or len(df) < 60:
        return {
            "signal": "NO SIGNAL",
            "confidence": 0,
            "buy_score": 0,
            "sell_score": 0,
            "reason": "Yeterli mum verisi yok.",
        }

    row = df.iloc[-1]
    previous = df.iloc[-2]

    values_to_check = [
        "close", "ema_20", "ema_50", "rsi",
        "bb_upper", "bb_lower", "macd",
        "macd_signal", "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
        "volume_ratio",
    ]

    if any(pd.isna(row.get(k)) for k in values_to_check):
        return {
            "signal": "NO SIGNAL",
            "confidence": 0,
            "buy_score": 0,
            "sell_score": 0,
            "reason": "İndikatörler henüz hesaplanamadı.",
        }

    buy_score = 0
    sell_score = 0
    reasons = []

    close = float(row["close"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    rsi = float(row["rsi"])
    upper = float(row["bb_upper"])
    lower = float(row["bb_lower"])
    macd = float(row["macd"])
    macd_signal = float(row["macd_signal"])
    stoch_k = float(row["stoch_k"])
    stoch_d = float(row["stoch_d"])
    adx = float(row["adx"])
    plus_di = float(row["plus_di"])
    minus_di = float(row["minus_di"])
    volume_ratio = float(row["volume_ratio"])

    # 1. EMA trend
    if close > ema20 and ema20 > ema50:
        buy_score += 1
        reasons.append("Yükseliş EMA trendi")

    elif close < ema20 and ema20 < ema50:
        sell_score += 1
        reasons.append("Düşüş EMA trendi")

    # 2. Bollinger Bands
    if close <= lower * 1.01:
        buy_score += 1
        reasons.append("Alt Bollinger bandına yakın")

    elif close >= upper * 0.99:
        sell_score += 1
        reasons.append("Üst Bollinger bandına yakın")

    # 3. RSI
    if rsi < 30:
        buy_score += 1
        reasons.append("RSI aşırı satım bölgesinde")

    elif rsi > 70:
        sell_score += 1
        reasons.append("RSI aşırı alım bölgesinde")

    elif 50 < rsi < 68:
        buy_score += 1
        reasons.append("RSI pozitif bölgede")

    elif 32 < rsi < 50:
        sell_score += 1
        reasons.append("RSI negatif bölgede")

    # 4. MACD
    if macd > macd_signal and row["macd_hist"] > 0:
        buy_score += 1
        reasons.append("MACD pozitif")

    elif macd < macd_signal and row["macd_hist"] < 0:
        sell_score += 1
        reasons.append("MACD negatif")

    # 5. Stochastic
    if stoch_k < 20 and stoch_k > stoch_d:
        buy_score += 1
        reasons.append("Stochastic yukarı kesişim")

    elif stoch_k > 80 and stoch_k < stoch_d:
        sell_score += 1
        reasons.append("Stochastic aşağı kesişim")

    # 6. ADX / directional movement
    if adx >= 20:
        if plus_di > minus_di:
            buy_score += 1
            reasons.append("ADX yönü pozitif")
        elif minus_di > plus_di:
            sell_score += 1
            reasons.append("ADX yönü negatif")

    # 7. Volume confirmation
    if volume_ratio >= 1.2:
        if close > float(row["open"]):
            buy_score += 1
            reasons.append("Yüksek hacimli yeşil mum")
        elif close < float(row["open"]):
            sell_score += 1
            reasons.append("Yüksek hacimli kırmızı mum")

    # 8. Recent momentum confirmation
    if (
        not pd.isna(row["change_3_candles_pct"])
        and row["change_3_candles_pct"] > 0
    ):
        buy_score += 1
    elif (
        not pd.isna(row["change_3_candles_pct"])
        and row["change_3_candles_pct"] < 0
    ):
        sell_score += 1

    difference = abs(buy_score - sell_score)
    total_score = max(buy_score, sell_score)

    if (
        buy_score >= min_votes
        and buy_score - sell_score >= min_margin
    ):
        signal = "BUY"
    elif (
        sell_score >= min_votes
        and sell_score - buy_score >= min_margin
    ):
        signal = "SELL"
    else:
        signal = "NO SIGNAL"

    # A score-based indicator agreement percentage.
    # It is not a probability of winning.
    confidence = round(
        100 * total_score / 8
    )

    if signal == "NO SIGNAL":
        reasons.append("Yeterli sinyal uyumu yok")

    return {
        "signal": signal,
        "confidence": min(confidence, 100),
        "buy_score": buy_score,
        "sell_score": sell_score,
        "score_difference": difference,
        "reason": "; ".join(reasons),
    }


# =========================================================
# SCAN ONE COIN
# =========================================================

def scan_coin(
    symbol,
    interval,
    candle_limit,
    min_votes,
    min_margin,
):
    try:
        candles = get_candles(
            symbol,
            interval,
            candle_limit,
        )

        if candles.empty:
            return None

        indicators = calculate_indicators(candles)

        if indicators.empty:
            return None

        result = make_signal(
            indicators,
            min_votes=min_votes,
            min_margin=min_margin,
        )

        last = indicators.iloc[-1]

        def safe_number(value, default=0.0):
            try:
                if pd.isna(value):
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        return {
            "symbol": symbol,
            "price": safe_number(last["close"]),
            "signal": result["signal"],
            "confidence": result["confidence"],
            "buy_score": result["buy_score"],
            "sell_score": result["sell_score"],
            "rsi": safe_number(last["rsi"]),
            "adx": safe_number(last["adx"]),
            "ema20": safe_number(last["ema_20"]),
            "ema50": safe_number(last["ema_50"]),
            "volume_ratio": safe_number(
                last["volume_ratio"]
            ),
            "change_3_candles_pct": safe_number(
                last["change_3_candles_pct"]
            ),
            "change_20_candles_pct": safe_number(
                last["change_20_candles_pct"]
            ),
            "atr_pct": safe_number(last["atr_pct"]),
            "reason": result["reason"],
            "candles": indicators,
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "error": str(exc),
        }


# =========================================================
# DISPLAY HELPERS
# =========================================================

def format_price(value):
    try:
        value = float(value)
        if value >= 1000:
            return f"{value:,.2f}"
        if value >= 1:
            return f"{value:.5f}"
        if value >= 0.001:
            return f"{value:.7f}"
        return f"{value:.10f}"
    except (TypeError, ValueError):
        return "N/A"


def results_to_dataframe(results):
    rows = []

    for item in results:
        if not item or "error" in item:
            continue

        row = {
            key: value
            for key, value in item.items()
            if key != "candles"
        }

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    signal_order = {
        "BUY": 0,
        "SELL": 1,
        "NO SIGNAL": 2,
    }

    df["_signal_order"] = df["signal"].map(
        signal_order
    ).fillna(3)

    df = df.sort_values(
        ["_signal_order", "confidence", "adx"],
        ascending=[True, False, False],
    )

    return df.drop(
        columns=["_signal_order"]
    ).reset_index(drop=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Scanner Settings")

interval = st.sidebar.selectbox(
    "Timeframe",
    options=["5m", "15m", "30m", "1h", "4h"],
    index=1,
)

candle_limit = st.sidebar.selectbox(
    "Completed candles",
    options=[100, 200, 300, 500, 800],
    index=2,
)

coin_count = st.sidebar.slider(
    "Scan top coins by 24h volume",
    min_value=10,
    max_value=150,
    value=30,
    step=10,
)

min_votes = st.sidebar.slider(
    "Minimum indicator votes",
    min_value=3,
    max_value=7,
    value=4,
)

min_margin = st.sidebar.slider(
    "Minimum BUY/SELL score difference",
    min_value=1,
    max_value=4,
    value=2,
)

max_workers = st.sidebar.slider(
    "Concurrent requests",
    min_value=1,
    max_value=8,
    value=4,
)

show_no_signal = st.sidebar.checkbox(
    "Show NO SIGNAL coins",
    value=True,
)

st.sidebar.caption(
    "Higher vote requirements produce fewer signals. "
    "More simultaneous requests may trigger API rate limits."
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 Velora AI Crypto Scanner")

st.write(
    "Binance Spot USDT paritelerini teknik göstergelerle tarar. "
    "Bu uygulama emir göndermez; yalnızca analiz üretir."
)

st.info(
    "BUY / SELL sonuçları teknik gösterge uyumunu ifade eder; "
    "kazanç olasılığı veya kâr garantisi değildir."
)


# =========================================================
# LOAD MARKET
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Timeframe", interval)

with col2:
    st.metric("Mum sayısı", candle_limit)

with col3:
    st.metric("Tarama hedefi", coin_count)

try:
    with st.spinner("Binance Spot pariteleri yükleniyor..."):
        universe = get_coin_universe()

except Exception as exc:
    st.error(
        "Binance piyasa verileri alınamadı.\n\n"
        f"Detay: {exc}"
    )
    st.stop()

if universe.empty:
    st.warning("Aktif USDT Spot paritesi bulunamadı.")
    st.stop()

st.caption(
    f"Bulunan aktif USDT Spot paritesi: {len(universe)}"
)

# Most liquid pairs are scanned first.
scan_symbols = universe.head(coin_count)["symbol"].tolist()

# =========================================================
# SCAN BUTTON
# =========================================================

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None

if st.button(
    "🔍 COIN TARAMASINI BAŞLAT",
    type="primary",
    use_container_width=True,
):
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                scan_coin,
                symbol,
                interval,
                candle_limit,
                min_votes,
                min_margin,
            ): symbol
            for symbol in scan_symbols
        }

        total = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                results.append({
                    "symbol": futures[future],
                    "error": str(exc),
                })

            progress.progress(
                min(completed / total, 1.0)
            )

            status.write(
                f"Taranıyor: {completed}/{total}"
            )

            # Small pause to reduce request pressure.
            time.sleep(0.03)

    st.session_state["scan_results"] = results
    st.session_state["scan_time"] = (
        pd.Timestamp.now(tz="UTC").strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    progress.empty()
    status.empty()


# =========================================================
# RESULTS
# =========================================================

results = st.session_state["scan_results"]

if not results:
    st.warning(
        "Henüz tarama yapılmadı. "
        "Yukarıdaki tarama düğmesine basın."
    )
    st.stop()

errors = [
    item for item in results
    if item and "error" in item
]

valid_results = [
    item for item in results
    if item and "candles" in item
]

if errors:
    st.warning(
        f"{len(errors)} coin taranamadı. "
        "API sınırları veya eksik mum verisi buna neden olabilir."
    )

results_df = results_to_dataframe(valid_results)

if results_df.empty:
    st.error(
        "Geçerli tarama sonucu yok. "
        "Bir süre bekleyip tekrar deneyin."
    )
    st.stop()

if not show_no_signal:
    display_df = results_df[
        results_df["signal"] != "NO SIGNAL"
    ].copy()
else:
    display_df = results_df.copy()

buy_count = int(
    (results_df["signal"] == "BUY").sum()
)

sell_count = int(
    (results_df["signal"] == "SELL").sum()
)

no_signal_count = int(
    (results_df["signal"] == "NO SIGNAL").sum()
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Taranan coin", len(results_df))
m2.metric("BUY", buy_count)
m3.metric("SELL", sell_count)
m4.metric("NO SIGNAL", no_signal_count)

if st.session_state["scan_time"]:
    st.caption(
        "Son tarama: "
        + st.session_state["scan_time"]
    )


# =========================================================
# PRIORITY SIGNALS
# =========================================================

st.subheader("🚨 Öncelikli sinyaller")

priority_df = results_df[
    results_df["signal"].isin(["BUY", "SELL"])
].copy()

if priority_df.empty:
    st.info(
        "Bu taramada yeterli gösterge uyumuna sahip "
        "BUY veya SELL sinyali bulunamadı."
    )
else:
    priority_columns = [
        "symbol",
        "signal",
        "price",
        "confidence",
        "buy_score",
        "sell_score",
        "rsi",
        "adx",
        "volume_ratio",
        "change_3_candles_pct",
        "reason",
    ]

    priority_view = priority_df[
        priority_columns
    ].copy()

    priority_view["price"] = priority_view[
        "price"
    ].map(format_price)

    priority_view["rsi"] = priority_view[
        "rsi"
    ].round(2)

    priority_view["adx"] = priority_view[
        "adx"
    ].round(2)

    priority_view["volume_ratio"] = priority_view[
        "volume_ratio"
    ].round(2)

    priority_view["change_3_candles_pct"] = (
        priority_view["change_3_candles_pct"].round(2)
    )

    st.dataframe(
        priority_view,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# ALL SCANNED COINS
# =========================================================

st.subheader("📋 Tarama sonuçları")

table_columns = [
    "symbol",
    "signal",
    "price",
    "confidence",
    "buy_score",
    "sell_score",
    "rsi",
    "adx",
    "ema20",
    "ema50",
    "volume_ratio",
    "change_3_candles_pct",
    "change_20_candles_pct",
    "atr_pct",
]

if not show_no_signal:
    display_df = display_df[
        display_df["signal"] != "NO SIGNAL"
    ]

table_df = display_df[
    table_columns
].copy()

table_df["price"] = table_df["price"].map(
    format_price
)

for col in [
    "rsi",
    "adx",
    "volume_ratio",
    "change_3_candles_pct",
    "change_20_candles_pct",
    "atr_pct",
]:
    table_df[col] = table_df[col].round(3)

table_df["ema20"] = table_df["ema20"].map(
    format_price
)

table_df["ema50"] = table_df["ema50"].map(
    format_price
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# SELECTED COIN DETAIL
# =========================================================

st.subheader("🔬 Coin detay analizi")

available_symbols = [
    item["symbol"]
    for item in valid_results
]

if available_symbols:
    selected_symbol = st.selectbox(
        "Detayını görmek istediğin coin",
        options=available_symbols,
    )

    selected = next(
        (
            item for item in valid_results
            if item["symbol"] == selected_symbol
        ),
        None,
    )

    if selected:
        detail = selected["candles"]
        last = detail.iloc[-1]

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Son kapanış",
            format_price(last["close"]),
        )

        d2.metric(
            "RSI",
            f"{last['rsi']:.2f}"
            if pd.notna(last["rsi"])
            else "N/A",
        )

        d3.metric(
            "ADX",
            f"{last['adx']:.2f}"
            if pd.notna(last["adx"])
            else "N/A",
        )

        d4.metric(
            "ATR %",
            f"{last['atr_pct']:.2f}%"
            if pd.notna(last["atr_pct"])
            else "N/A",
        )

        signal = selected["signal"]

        if signal == "BUY":
            st.success("BUY — yükseliş yönlü gösterge uyumu")
        elif signal == "SELL":
            st.error("SELL — düşüş yönlü gösterge uyumu")
        else:
            st.warning("NO SIGNAL — yeterli uyum yok")

        st.write(
            f"**BUY puanı:** {selected['buy_score']}  |  "
            f"**SELL puanı:** {selected['sell_score']}  |  "
            f"**Gösterge uyum skoru:** "
            f"{selected['confidence']}%"
        )

        st.write(
            "**Analiz gerekçeleri:** "
            + selected["reason"]
        )

        st.markdown("#### Kapanış fiyatı ve EMA")

        chart_df = detail.set_index("open_time")[
            ["close", "ema_20", "ema_50"]
        ].rename(
            columns={
                "close": "Close",
                "ema_20": "EMA 20",
                "ema_50": "EMA 50",
            }
        )

        st.line_chart(chart_df)

        st.markdown("#### Bollinger Bands")

        bb_df = detail.set_index("open_time")[
            ["close", "bb_upper", "bb_mid", "bb_lower"]
        ].rename(
            columns={
                "close": "Close",
                "bb_upper": "Upper Band",
                "bb_mid": "Middle Band",
                "bb_lower": "Lower Band",
            }
        )

        st.line_chart(bb_df)

        st.markdown("#### RSI")

        rsi_df = detail.set_index("open_time")[
            ["rsi"]
        ].rename(
            columns={"rsi": "RSI"}
        )

        st.line_chart(rsi_df)


# =========================================================
# CSV EXPORT
# =========================================================

st.subheader("📥 Sonuçları indir")

csv_df = results_df.drop(
    columns=["candles"],
    errors="ignore",
)

csv_data = csv_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="CSV dosyasını indir",
    data=csv_data,
    file_name="velora_scan_results.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Educational analysis only. This application does not execute "
    "trades or connect to Binomo. Binance public prices can differ "
    "from Binomo OTC prices. Technical analysis and machine-learning "
    "estimates can be wrong; no signal guarantees profit."
)
