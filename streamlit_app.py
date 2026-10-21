
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Velora AI Crypto Scanner",
    page_icon="📊",
    layout="wide",
)

KRAKEN_API = "https://api.kraken.com/0/public"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "VeloraCryptoScanner/1.0"
})

EXCLUDED_BASES = {
    "USDT", "USDC", "DAI", "USDE", "TUSD",
    "USDP", "FDUSD", "EUR", "GBP", "AUD",
    "CAD", "CHF", "JPY", "PAXG",
}


# =========================================================
# KRAKEN API
# =========================================================

def kraken_get(endpoint, params=None):
    url = f"{KRAKEN_API}/{endpoint}"

    try:
        response = SESSION.get(
            url,
            params=params or {},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Kraken API bağlantı hatası: {exc}"
        ) from exc

    errors = payload.get("error", [])

    if errors:
        raise RuntimeError(
            "Kraken API: " + "; ".join(errors)
        )

    return payload.get("result", {})


def normalize_asset(asset):
    """Normalize common Kraken asset aliases."""
    aliases = {
        "XXBT": "BTC",
        "XBT": "BTC",
        "XETH": "ETH",
        "XXDG": "DOGE",
        "XDG": "DOGE",
        "ZUSD": "USD",
        "ZEUR": "EUR",
        "ZGBP": "GBP",
        "ZJPY": "JPY",
        "ZCAD": "CAD",
        "ZAUD": "AUD",
        "ZCHF": "CHF",
    }

    return aliases.get(asset, asset)


def parse_pair(pair_name, pair_info):
    """Extract base and quote currencies from Kraken pair metadata."""
    wsname = pair_info.get("wsname", "")

    if "/" in wsname:
        base, quote = wsname.split("/", 1)
    else:
        base = normalize_asset(
            pair_info.get("base", "")
        )
        quote = normalize_asset(
            pair_info.get("quote", "")
        )

    base = normalize_asset(base)
    quote = normalize_asset(quote)

    return base, quote


# =========================================================
# MARKET UNIVERSE
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_market_universe():
    pairs_data = kraken_get("AssetPairs")
    ticker_data = kraken_get("Ticker")

    markets = []

    for pair_key, info in pairs_data.items():
        if info.get("status", "online") != "online":
            continue

        if info.get("wsname") is None:
            continue

        base, quote = parse_pair(pair_key, info)

        # Keep USD and USDT quote pairs only.
        if quote not in {"USD", "USDT"}:
            continue

        if base in EXCLUDED_BASES:
            continue

        # Avoid obvious leveraged tokens.
        if base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue

        # Kraken ticker response normally uses pair_key.
        ticker = ticker_data.get(pair_key)

        # Fallback to altname if necessary.
        if ticker is None:
            altname = info.get("altname", "")
            ticker = ticker_data.get(altname)

        if not ticker:
            continue

        try:
            price = float(ticker["c"][0])
            volume_24h = float(ticker["v"][1])
            quote_volume = price * volume_24h
            open_price = float(ticker["o"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue

        if price <= 0:
            continue

        change_24h = (
            (price / open_price - 1) * 100
            if open_price > 0
            else 0.0
        )

        markets.append({
            "pair_key": pair_key,
            "altname": info.get("altname", pair_key),
            "symbol": f"{base}/{quote}",
            "base": base,
            "quote": quote,
            "price": price,
            "volume_24h": volume_24h,
            "quote_volume": quote_volume,
            "change_24h": change_24h,
        })

    df = pd.DataFrame(markets)

    if df.empty:
        return df

    return df.sort_values(
        "quote_volume",
        ascending=False,
    ).reset_index(drop=True)


# =========================================================
# OHLC CANDLES
# =========================================================

@st.cache_data(ttl=45, show_spinner=False)
def get_candles(pair_key, interval=15, candle_limit=300):
    result = kraken_get(
        "OHLC",
        {
            "pair": pair_key,
            "interval": int(interval),
        },
    )

    # Kraken adds "last" as a pagination cursor.
    last_cursor = result.get("last")

    candle_key = next(
        (key for key in result if key != "last"),
        None,
    )

    if candle_key is None:
        return pd.DataFrame()

    raw = result[candle_key]

    if not raw:
        return pd.DataFrame()

    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "count",
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in [
        "open", "high", "low", "close",
        "vwap", "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="s",
        utc=True,
    )

    # Kraken's final OHLC entry is the currently forming candle.
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    df = df.tail(int(candle_limit)).reset_index(drop=True)

    return df


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(close, period=14):
    delta = close.diff()

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
    rsi = 100 - 100 / (1 + rs)

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
    df = df.copy()

    if len(df) < 60:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["ema_9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema_20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["ema_50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    # Bollinger Bands
    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    # RSI
    df["rsi"] = calculate_rsi(close, 14)

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()

    denominator = (high14 - low14).replace(
        0, np.nan
    )

    df["stoch_k"] = (
        (close - low14) / denominator * 100
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

    # ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0,
        ),
        index=df.index,
    )

    smoothed_plus = plus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    smoothed_minus = minus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    atr_safe = df["atr"].replace(0, np.nan)

    df["plus_di"] = 100 * smoothed_plus / atr_safe
    df["minus_di"] = 100 * smoothed_minus / atr_safe

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

    # Volume confirmation
    df["volume_sma"] = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / df["volume_sma"].replace(0, np.nan)
    )

    # Momentum
    df["change_3_candles_pct"] = (
        (close / close.shift(3) - 1) * 100
    )

    df["change_20_candles_pct"] = (
        (close / close.shift(20) - 1) * 100
    )

    return df


# =========================================================
# SIGNAL GENERATION
# =========================================================

def generate_signal(df, min_votes=4, min_margin=2):
    if df is None or len(df) < 60:
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "confidence": 0,
            "reason": "Yeterli mum verisi yok.",
        }

    row = df.iloc[-1]

    required = [
        "close", "ema_20", "ema_50", "rsi",
        "bb_upper", "bb_lower", "macd",
        "macd_signal", "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
        "volume_ratio",
    ]

    if any(pd.isna(row[col]) for col in required):
        return {
            "signal": "NO SIGNAL",
            "buy_score": 0,
            "sell_score": 0,
            "confidence": 0,
            "reason": "İndikatörler hesaplanamadı.",
        }

    buy = 0
    sell = 0
    reasons = []

    close = float(row["close"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    rsi = float(row["rsi"])

    # 1. EMA trend
    if close > ema20 and ema20 > ema50:
        buy += 1
        reasons.append("EMA yükseliş trendi")
    elif close < ema20 and ema20 < ema50:
        sell += 1
        reasons.append("EMA düşüş trendi")

    # 2. Bollinger Bands
    if close <= float(row["bb_lower"]) * 1.01:
        buy += 1
        reasons.append("Alt Bollinger bandına yakın")
    elif close >= float(row["bb_upper"]) * 0.99:
        sell += 1
        reasons.append("Üst Bollinger bandına yakın")

    # 3. RSI
    if rsi < 30:
        buy += 1
        reasons.append("RSI aşırı satım")
    elif rsi > 70:
        sell += 1
        reasons.append("RSI aşırı alım")
    elif 50 < rsi < 68:
        buy += 1
        reasons.append("RSI pozitif")
    elif 32 < rsi < 50:
        sell += 1
        reasons.append("RSI negatif")

    # 4. MACD
    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):
        buy += 1
        reasons.append("MACD pozitif")
    elif (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):
        sell += 1
        reasons.append("MACD negatif")

    # 5. Stochastic
    if (
        row["stoch_k"] < 20
        and row["stoch_k"] > row["stoch_d"]
    ):
        buy += 1
        reasons.append("Stochastic yukarı kesişim")
    elif (
        row["stoch_k"] > 80
        and row["stoch_k"] < row["stoch_d"]
    ):
        sell += 1
        reasons.append("Stochastic aşağı kesişim")

    # 6. ADX directional trend
    if row["adx"] >= 20:
        if row["plus_di"] > row["minus_di"]:
            buy += 1
            reasons.append("ADX pozitif yön")
        elif row["minus_di"] > row["plus_di"]:
            sell += 1
            reasons.append("ADX negatif yön")

    # 7. Volume confirmation
    if row["volume_ratio"] >= 1.2:
        if close > float(row["open"]):
            buy += 1
            reasons.append("Yüksek hacimli yeşil mum")
        elif close < float(row["open"]):
            sell += 1
            reasons.append("Yüksek hacimli kırmızı mum")

    # 8. Three-candle momentum
    momentum = row["change_3_candles_pct"]

    if pd.notna(momentum):
        if momentum > 0:
            buy += 1
        elif momentum < 0:
            sell += 1

    if buy >= min_votes and buy - sell >= min_margin:
        signal = "BUY"
    elif sell >= min_votes and sell - buy >= min_margin:
        signal = "SELL"
    else:
        signal = "NO SIGNAL"
        reasons.append("Yeterli sinyal uyumu yok")

    confidence = round(
        max(buy, sell) / 8 * 100
    )

    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "confidence": min(confidence, 100),
        "reason": "; ".join(reasons),
    }


# =========================================================
# SCAN A MARKET
# =========================================================

def scan_market(
    market,
    interval,
    candle_limit,
    min_votes,
    min_margin,
):
    try:
        candles = get_candles(
            market["pair_key"],
            interval=interval,
            candle_limit=candle_limit,
        )

        if candles.empty or len(candles) < 60:
            return None

        df = calculate_indicators(candles)

        if df.empty:
            return None

        signal = generate_signal(
            df,
            min_votes=min_votes,
            min_margin=min_margin,
        )

        last = df.iloc[-1]

        return {
            "symbol": market["symbol"],
            "pair_key": market["pair_key"],
            "price": float(last["close"]),
            "change_24h": market["change_24h"],
            "quote_volume": market["quote_volume"],
            "signal": signal["signal"],
            "confidence": signal["confidence"],
            "buy_score": signal["buy_score"],
            "sell_score": signal["sell_score"],
            "rsi": float(last["rsi"]),
            "adx": float(last["adx"]),
            "ema20": float(last["ema_20"]),
            "ema50": float(last["ema_50"]),
            "volume_ratio": float(last["volume_ratio"]),
            "change_3_candles_pct": float(
                last["change_3_candles_pct"]
            ),
            "change_20_candles_pct": float(
                last["change_20_candles_pct"]
            ),
            "atr_pct": float(last["atr_pct"]),
            "reason": signal["reason"],
            "candles": df,
        }

    except Exception as exc:
        return {
            "symbol": market.get("symbol", "Unknown"),
            "error": str(exc),
        }


# =========================================================
# FORMATTING
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

    except (ValueError, TypeError):
        return "N/A"


def results_dataframe(results):
    rows = []

    for item in results:
        if not item or "error" in item:
            continue

        rows.append({
            key: value
            for key, value in item.items()
            if key not in ("candles", "pair_key")
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    signal_order = {
        "BUY": 0,
        "SELL": 1,
        "NO SIGNAL": 2,
    }

    df["_order"] = df["signal"].map(
        signal_order
    ).fillna(3)

    df = df.sort_values(
        ["_order", "confidence", "adx"],
        ascending=[True, False, False],
    )

    return df.drop(
        columns=["_order"]
    ).reset_index(drop=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Tarama ayarları")

interval = st.sidebar.selectbox(
    "Timeframe",
    options=[1, 5, 15, 30, 60, 240],
    index=2,
    format_func=lambda x: {
        1: "1 dakika",
        5: "5 dakika",
        15: "15 dakika",
        30: "30 dakika",
        60: "1 saat",
        240: "4 saat",
    }[x],
)

candle_limit = st.sidebar.selectbox(
    "Mum sayısı",
    options=[100, 200, 300, 500, 700],
    index=2,
)

coin_count = st.sidebar.slider(
    "Taranacak coin sayısı",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
)

min_votes = st.sidebar.slider(
    "Minimum gösterge puanı",
    min_value=3,
    max_value=7,
    value=4,
)

min_margin = st.sidebar.slider(
    "Minimum BUY/SELL puan farkı",
    min_value=1,
    max_value=4,
    value=2,
)

max_workers = st.sidebar.slider(
    "Eşzamanlı istek",
    min_value=1,
    max_value=5,
    value=3,
)

show_no_signal = st.sidebar.checkbox(
    "NO SIGNAL coinlerini göster",
    value=True,
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 Velora AI Crypto Scanner")

st.markdown(
    """
    Kraken piyasa verileri üzerinden teknik analiz yapan
    bir kripto para tarayıcısı.

    - **BUY:** Yükseliş yönünde yeterli gösterge uyumu.
    - **SELL:** Düşüş yönünde yeterli gösterge uyumu.
    - **NO SIGNAL:** Yeterli gösterge uyumu yok.
    """
)

st.info(
    "Bu uygulama emir göndermez. Gösterge uyum skoru, "
    "işlemin kazanma olasılığı değildir."
)


# =========================================================
# MARKET LOAD
# =========================================================

try:
    with st.spinner("Kraken piyasa verileri yükleniyor..."):
        universe = get_market_universe()

except Exception as exc:
    st.error(f"Piyasa verileri alınamadı: {exc}")
    st.stop()

if universe.empty:
    st.error(
        "Kraken üzerinde uygun USD/USDT paritesi bulunamadı."
    )
    st.stop()

st.caption(
    f"Bulunan uygun parite sayısı: {len(universe)}"
)

st.caption(
    "Kraken, Binance ile aynı coin listesini veya fiyatları "
    "sunmayabilir."
)


# =========================================================
# SCAN CONTROLS
# =========================================================

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None

if st.button(
    "🔍 TARAMAYI BAŞLAT",
    type="primary",
    use_container_width=True,
):
    selected_markets = universe.head(
        coin_count
    ).to_dict("records")

    results = []

    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                scan_market,
                market,
                interval,
                candle_limit,
                min_votes,
                min_margin,
            ): market["symbol"]
            for market in selected_markets
        }

        total = len(futures)

        for count, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            try:
                result = future.result()

                if result:
                    results.append(result)

            except Exception as exc:
                results.append({
                    "symbol": futures[future],
                    "error": str(exc),
                })

            progress.progress(count / total)

            status.write(
                f"Taranıyor: {count}/{total}"
            )

            time.sleep(0.02)

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
        "Tarama henüz yapılmadı. "
        "Taramayı başlat düğmesine bas."
    )
    st.stop()

errors = [
    result for result in results
    if result and "error" in result
]

valid_results = [
    result for result in results
    if result and "candles" in result
]

if errors:
    st.warning(
        f"{len(errors)} parite taranamadı. "
        "API sınırı veya eksik veri söz konusu olabilir."
    )

df_results = results_dataframe(valid_results)

if df_results.empty:
    st.error(
        "Geçerli sonuç yok. Daha sonra tekrar deneyin."
    )
    st.stop()


# =========================================================
# SUMMARY
# =========================================================

buy_count = int(
    (df_results["signal"] == "BUY").sum()
)

sell_count = int(
    (df_results["signal"] == "SELL").sum()
)

neutral_count = int(
    (df_results["signal"] == "NO SIGNAL").sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Taranan coin", len(df_results))
c2.metric("BUY", buy_count)
c3.metric("SELL", sell_count)
c4.metric("NO SIGNAL", neutral_count)

if st.session_state["scan_time"]:
    st.caption(
        f"Son tarama: {st.session_state['scan_time']}"
    )


# =========================================================
# PRIORITY SIGNALS
# =========================================================

st.subheader("🚨 Öncelikli sinyaller")

priority_df = df_results[
    df_results["signal"].isin(["BUY", "SELL"])
].copy()

if priority_df.empty:
    st.info(
        "Yeterli gösterge uyumuna sahip sinyal bulunamadı. "
        "NO SIGNAL sonucu normaldir."
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

    priority_view["price"] = (
        priority_view["price"].map(format_price)
    )

    for col in [
        "rsi",
        "adx",
        "volume_ratio",
        "change_3_candles_pct",
    ]:
        priority_view[col] = (
            priority_view[col].round(3)
        )

    st.dataframe(
        priority_view,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# ALL COINS
# =========================================================

st.subheader("📋 Bütün tarama sonuçları")

display_df = df_results.copy()

if not show_no_signal:
    display_df = display_df[
        display_df["signal"] != "NO SIGNAL"
    ]

table_columns = [
    "symbol",
    "signal",
    "price",
    "change_24h",
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

table_df = display_df[table_columns].copy()

table_df["price"] = table_df["price"].map(
    format_price
)

table_df["ema20"] = table_df["ema20"].map(
    format_price
)

table_df["ema50"] = table_df["ema50"].map(
    format_price
)

for col in [
    "change_24h",
    "confidence",
    "buy_score",
    "sell_score",
    "rsi",
    "adx",
    "volume_ratio",
    "change_3_candles_pct",
    "change_20_candles_pct",
    "atr_pct",
]:
    table_df[col] = table_df[col].round(3)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# COIN DETAIL
# =========================================================

st.subheader("🔬 Coin detay analizi")

symbols = [
    item["symbol"]
    for item in valid_results
]

if symbols:
    selected_symbol = st.selectbox(
        "Coin seç",
        options=symbols,
    )

    selected = next(
        (
            item for item in valid_results
            if item["symbol"] == selected_symbol
        ),
        None,
    )

    if selected:
        candles = selected["candles"]
        last = candles.iloc[-1]

        a, b, c, d = st.columns(4)

        a.metric(
            "Son kapanış",
            format_price(last["close"]),
        )

        b.metric(
            "RSI",
            f"{last['rsi']:.2f}"
            if pd.notna(last["rsi"])
            else "N/A",
        )

        c.metric(
            "ADX",
            f"{last['adx']:.2f}"
            if pd.notna(last["adx"])
            else "N/A",
        )

        d.metric(
            "ATR %",
            f"{last['atr_pct']:.2f}%"
            if pd.notna(last["atr_pct"])
            else "N/A",
        )

        if selected["signal"] == "BUY":
            st.success("BUY — yükseliş yönlü gösterge uyumu")
        elif selected["signal"] == "SELL":
            st.error("SELL — düşüş yönlü gösterge uyumu")
        else:
            st.warning("NO SIGNAL — yeterli uyum yok")

        st.write(
            f"**BUY puanı:** {selected['buy_score']} | "
            f"**SELL puanı:** {selected['sell_score']} | "
            f"**Gösterge uyum skoru:** "
            f"{selected['confidence']}%"
        )

        st.write(
            "**Gerekçe:** " + selected["reason"]
        )

        chart = candles.set_index("time")[
            ["close", "ema_20", "ema_50"]
        ].rename(
            columns={
                "close": "Close",
                "ema_20": "EMA 20",
                "ema_50": "EMA 50",
            }
        )

        st.markdown("#### Fiyat ve EMA")
        st.line_chart(chart)

        bb_chart = candles.set_index("time")[
            ["close", "bb_upper", "bb_mid", "bb_lower"]
        ].rename(
            columns={
                "close": "Close",
                "bb_upper": "Upper Band",
                "bb_mid": "Middle Band",
                "bb_lower": "Lower Band",
            }
        )

        st.markdown("#### Bollinger Bands")
        st.line_chart(bb_chart)

        rsi_chart = candles.set_index("time")[
            ["rsi"]
        ]

        st.markdown("#### RSI")
        st.line_chart(rsi_chart)

        macd_chart = candles.set_index("time")[
            ["macd", "macd_signal", "macd_hist"]
        ]

        st.markdown("#### MACD")
        st.line_chart(macd_chart)


# =========================================================
# CSV EXPORT
# =========================================================

st.subheader("📥 Sonuçları indir")

csv_df = df_results.copy()

csv_data = csv_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="CSV dosyasını indir",
    data=csv_data,
    file_name="velora_kraken_scan.csv",
    mime="text/csv",
    use_container_width=True,
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Educational analysis only. This application does not execute "
    "trades or connect to Binomo. Kraken prices can differ from "
    "Binance and Binomo OTC prices. Indicator scores are not "
    "probabilities of winning. No signal guarantees profit."
)
