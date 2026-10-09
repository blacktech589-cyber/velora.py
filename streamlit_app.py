
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crypto Spot AI Scanner",
    page_icon="📊",
    layout="wide",
)

BINANCE = "https://api.binance.com"
GATE = "https://api.gateio.ws/api/v4"

INTERVAL = "15m"
BARS_PER_DAY = 96
CRASH_DAYS = 10
CRASH_BARS = CRASH_DAYS * BARS_PER_DAY  # 960 candles

FORECAST_BARS = 2  # 2 x 15-minute candles = 30 minutes
BUY_TARGET = 0.10
SELL_TARGET = -0.10

MIN_CANDLES = 300
REQUEST_TIMEOUT = 15

FEATURES = [
    "return_1",
    "return_2",
    "return_4",
    "return_8",
    "return_16",
    "return_32",
    "ema_gap_9",
    "ema_gap_20",
    "ema_gap_50",
    "rsi_7",
    "rsi_14",
    "atr_pct",
    "bb_position",
    "bb_width",
    "volume_ratio",
    "volume_change",
    "green_ratio",
    "momentum",
    "high_low_range",
    "close_open_return",
]

HEADERS = {
    "User-Agent": "CryptoSpotAIScanner/1.0",
    "Accept": "application/json",
}

# ============================================================
# HTTP HELPERS
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# BINANCE MARKETS
# ============================================================

def binance_markets():
    data = get_json(f"{BINANCE}/api/v3/exchangeInfo")

    markets = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol", "")

        if not symbol:
            continue

        markets.append({
            "symbol": symbol,
            "market_id": symbol,
            "base": item.get("baseAsset", ""),
            "quote": "USDT",
            "provider": "Binance",
        })

    if not markets:
        raise RuntimeError("Binance aktif Spot USDT marketi döndürmedi.")

    return markets


def binance_tickers():
    data = get_json(f"{BINANCE}/api/v3/ticker/24hr")

    result = {}

    for item in data:
        symbol = item.get("symbol")

        if not symbol:
            continue

        try:
            result[symbol] = {
                "price": float(item.get("lastPrice", 0)),
                "quote_volume": float(item.get("quoteVolume", 0)),
                "change_24h": float(item.get("priceChangePercent", 0)),
            }
        except (TypeError, ValueError):
            continue

    return result


# ============================================================
# GATE.IO MARKETS - FALLBACK PROVIDER
# ============================================================

def gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")

    markets = []

    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")
        quote = item.get("quote", "")

        if not market_id or not base or quote != "USDT":
            continue

        # Only include markets that the API reports as tradable.
        trade_status = str(item.get("trade_status", "")).lower()

        if trade_status and trade_status not in (
            "tradable",
            "buyable",
            "sellable",
        ):
            continue

        if item.get("delisted") is True:
            continue

        markets.append({
            "symbol": f"{base}USDT",
            "market_id": market_id,
            "base": base,
            "quote": quote,
            "provider": "Gate.io",
        })

    if not markets:
        raise RuntimeError("Gate.io aktif Spot USDT marketi döndürmedi.")

    return markets


def gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")

    result = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            result[market_id] = {
                "price": float(item.get("last", 0)),
                "quote_volume": float(item.get("quote_volume", 0)),
                "change_24h": float(item.get("change_percentage", 0)),
            }
        except (TypeError, ValueError):
            continue

    return result


# ============================================================
# MARKET ACTIVITY
# ============================================================

def add_activity(markets, tickers):
    rows = []

    for market in markets:
        market_id = market["market_id"]
        ticker = tickers.get(market_id, {})

        row = dict(market)
        row["price"] = ticker.get("price", 0.0)
        row["quote_volume"] = ticker.get("quote_volume", 0.0)
        row["change_24h"] = ticker.get("change_24h", 0.0)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# CANDLE LOADER
# ============================================================

def load_candles(market, limit=1000):
    provider = market["provider"]
    market_id = market["market_id"]

    if provider == "Binance":
        raw = get_json(
            f"{BINANCE}/api/v3/klines",
            params={
                "symbol": market_id,
                "interval": INTERVAL,
                "limit": min(limit, 1000),
            },
        )

        if not raw:
            raise RuntimeError("Binance mum verisi boş.")

        df = pd.DataFrame(
            raw,
            columns=[
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
                "ignore",
            ],
        )

        df = df[[
            "open_time", "open", "high", "low",
            "close", "volume", "quote_volume",
        ]].copy()

        df["timestamp"] = pd.to_numeric(
            df["open_time"], errors="coerce"
        ) / 1000

    elif provider == "Gate.io":
        raw = get_json(
            f"{GATE}/spot/candlesticks",
            params={
                "currency_pair": market_id,
                "interval": INTERVAL,
                "limit": min(limit, 1000),
            },
        )

        if not raw:
            raise RuntimeError("Gate.io mum verisi boş.")

        # Gate candle format:
        # timestamp, quote volume, close, high, low, open, base volume
        rows = []

        for candle in raw:
            if len(candle) < 6:
                continue

            rows.append({
                "timestamp": float(candle[0]),
                "quote_volume": float(candle[1]),
                "close": float(candle[2]),
                "high": float(candle[3]),
                "low": float(candle[4]),
                "open": float(candle[5]),
                "volume": (
                    float(candle[6]) if len(candle) > 6 else 0.0
                ),
            })

        df = pd.DataFrame(rows)

    else:
        raise RuntimeError(f"Bilinmeyen veri sağlayıcı: {provider}")

    required = [
        "timestamp", "open", "high", "low",
        "close", "volume", "quote_volume",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Eksik mum alanı: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ].copy()

    if len(df) < MIN_CANDLES:
        raise RuntimeError(
            f"Yetersiz geçmiş: {len(df)} mum; "
            f"en az {MIN_CANDLES} gerekli."
        )

    return df.reset_index(drop=True)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    return result.replace([np.inf, -np.inf], np.nan)


def build_features(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_price = df["open"]
    volume = df["volume"]

    for period in [9, 20, 50]:
        df[f"ema_{period}"] = ema(close, period)
        df[f"ema_gap_{period}"] = (
            close / df[f"ema_{period}"] - 1
        )

    for period in [1, 2, 4, 8, 16, 32]:
        df[f"return_{period}"] = close.pct_change(period)

    df["rsi_7"] = rsi(close, 7)
    df["rsi_14"] = rsi(close, 14)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(14).mean()
    df["atr_pct"] = atr / close

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    upper = middle + 2 * std
    lower = middle - 2 * std

    band_width = (upper - lower) / middle.replace(0, np.nan)

    df["bb_position"] = (
        (close - lower) / (upper - lower).replace(0, np.nan)
    )

    df["bb_width"] = band_width

    average_volume = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / average_volume.replace(0, np.nan)
    )

    df["volume_change"] = volume.pct_change().replace(
        [np.inf, -np.inf], np.nan
    )

    df["green_ratio"] = (
        (close > open_price).astype(float).rolling(10).mean()
    )

    df["momentum"] = close / close.shift(8) - 1

    df["high_low_range"] = (high - low) / close

    df["close_open_return"] = close / open_price - 1

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# 10-DAY CRASH FILTER
# ============================================================

def crash_last_10_days(df, threshold=80):
    # Drop the latest candle because it may still be forming.
    completed = df.iloc[:-1].tail(CRASH_BARS)

    if len(completed) < CRASH_BARS:
        return {
            "peak": np.nan,
            "current_price": np.nan,
            "drop_pct": np.nan,
            "crash": False,
            "bars": len(completed),
        }

    peak = float(completed["high"].max())
    current = float(completed["close"].iloc[-1])

    if peak <= 0 or current <= 0:
        drop = np.nan
    else:
        drop = (peak - current) / peak * 100

    return {
        "peak": peak,
        "current_price": current,
        "drop_pct": drop,
        "crash": bool(
            pd.notna(drop) and drop >= threshold
        ),
        "bars": len(completed),
    }


# ============================================================
# TECHNICAL BUY / SELL SCORES
# ============================================================

def technical_scores(df):
    row = df.iloc[-1]

    buy = 0.0
    sell = 0.0

    # EMA trend
    if row["ema_9"] > row["ema_20"]:
        buy += 12
    else:
        sell += 12

    if row["ema_20"] > row["ema_50"]:
        buy += 12
    else:
        sell += 12

    # RSI
    rsi_value = row["rsi_14"]

    if pd.notna(rsi_value):
        if 30 <= rsi_value <= 55:
            buy += 10
        elif rsi_value > 70:
            sell += 8
        elif rsi_value < 30:
            buy += 5

    # Bollinger Bands
    bb = row["bb_position"]

    if pd.notna(bb):
        if bb < 0.20:
            buy += 12
        elif bb > 0.80:
            sell += 12

    # Recent momentum
    momentum = row["momentum"]

    if pd.notna(momentum):
        if momentum > 0:
            buy += 10
        elif momentum < 0:
            sell += 10

    # Latest completed candle direction
    if row["close"] > row["open"]:
        buy += 6
    elif row["close"] < row["open"]:
        sell += 6

    # Volume activity
    volume_ratio = row["volume_ratio"]

    if pd.notna(volume_ratio) and volume_ratio > 1.5:
        if row["close"] > row["open"]:
            buy += 5
        else:
            sell += 5

    return min(buy, 100), min(sell, 100)


# ============================================================
# MACHINE LEARNING
# ============================================================

def make_training_data(feature_df):
    """
    Create a binary directional target using the next two
    completed candles. No HOLD class is used.

    Samples where the future move reaches +10% or -10% are
    assigned that direction. If neither threshold is reached,
    the direction of the 30-minute close-to-close return is used.
    """

    data = feature_df.copy()

    future_close = data["close"].shift(-FORECAST_BARS)
    future_return = future_close / data["close"] - 1

    labels = np.where(
        future_return >= BUY_TARGET,
        1,
        np.where(
            future_return <= SELL_TARGET,
            0,
            np.where(future_return >= 0, 1, 0),
        ),
    )

    data["target"] = labels

    # Exclude rows whose future outcome is unavailable.
    data = data.iloc[:-FORECAST_BARS].copy()

    data = data.dropna(subset=FEATURES + ["target"])

    if len(data) < 100:
        return None, None

    X = data[FEATURES]
    y = data["target"].astype(int)

    if y.nunique() < 2:
        return None, None

    return X, y


def ai_scores(feature_df):
    """
    Fit a small neural-network classifier and a Random Forest.
    Combine their probabilities when both are available.
    """

    X, y = make_training_data(feature_df)

    current = feature_df.iloc[[-1]][FEATURES]

    if X is None or y is None:
        # If there is insufficient training data, use a neutral
        # probability and let technical scores guide the signal.
        return 50.0, 50.0, "Technical fallback"

    probabilities = []

    try:
        mlp = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=120,
                early_stopping=True,
                random_state=42,
            )),
        ])

        mlp.fit(X, y)
        probabilities.append(float(mlp.predict_proba(current)[0][1]))

    except Exception:
        pass

    try:
        forest = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        forest.fit(X, y)
        probabilities.append(
            float(forest.predict_proba(current)[0][1])
        )

    except Exception:
        pass

    if not probabilities:
        return 50.0, 50.0, "Technical fallback"

    probability_buy = float(np.mean(probabilities))

    return (
        probability_buy * 100,
        (1 - probability_buy) * 100,
        "MLP + Random Forest" if len(probabilities) == 2 else "ML",
    )


# ============================================================
# ANALYZE ONE MARKET
# ============================================================

def analyze_market(market, ticker, crash_threshold=80):
    candles = load_candles(market, limit=1000)

    # Keep only completed candles for all indicators.
    completed = candles.iloc[:-1].copy()

    if len(completed) < MIN_CANDLES:
        raise RuntimeError("Yeterli tamamlanmış mum yok.")

    feature_df = build_features(completed)

    usable = feature_df.dropna(subset=FEATURES).copy()

    if len(usable) < 100:
        raise RuntimeError("Teknik analiz için yeterli veri yok.")

    tech_buy, tech_sell = technical_scores(usable)

    ai_buy, ai_sell, model_name = ai_scores(usable)

    # Combine technical and ML scores.
    buy_score = 0.45 * tech_buy + 0.55 * ai_buy
    sell_score = 0.45 * tech_sell + 0.55 * ai_sell

    crash = crash_last_10_days(
        candles,
        threshold=crash_threshold,
    )

    # Crash is a separate condition, not proof of a rebound.
    if crash["crash"]:
        buy_score += 5

    total = buy_score + sell_score

    if total > 0:
        buy_percent = buy_score / total * 100
        sell_percent = sell_score / total * 100
    else:
        buy_percent = 50
        sell_percent = 50

    signal = "BUY" if buy_score >= sell_score else "SELL"

    latest = usable.iloc[-1]

    price = float(latest["close"])

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "provider": market["provider"],
        "price": price,
        "change_24h": ticker.get("change_24h", 0.0),
        "volume_24h_usdt": ticker.get("quote_volume", 0.0),
        "signal": signal,
        "buy_score": round(buy_percent, 2),
        "sell_score": round(sell_percent, 2),
        "ai_buy_probability": round(ai_buy, 2),
        "ai_sell_probability": round(ai_sell, 2),
        "model": model_name,
        "week10_peak": crash["peak"],
        "drop_10d_pct": crash["drop_pct"],
        "crash_80": crash["crash"],
        "bars_10d": crash["bars"],
        "rsi_14": round(float(latest["rsi_14"]), 2)
        if pd.notna(latest["rsi_14"]) else np.nan,
        "volume_ratio": round(float(latest["volume_ratio"]), 2)
        if pd.notna(latest["volume_ratio"]) else np.nan,
        "candles_used": len(completed),
        "updated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📊 Crypto Spot AI Scanner")

st.write(
    "Aktif Spot USDT paritelerini tarar, BUY/SELL sinyalleri "
    "üretir ve son 10 günde zirveden en az %80 düşen coinleri "
    "ayrı listeler. Otomatik emir göndermez."
)

st.warning(
    "AI çıktıları kesin fiyat tahmini değildir. Özellikle düşük "
    "likiditeli coinlerde sert fiyat hareketleri, veri hataları "
    "ve yüksek spread riski olabilir."
)

with st.sidebar:
    st.header("Tarama Ayarları")

    crash_threshold = st.slider(
        "10 günlük düşüş eşiği (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5,
    )

    scan_mode = st.selectbox(
        "Tarama modu",
        [
            "En yüksek hacimli pariteler",
            "Tüm aktif USDT pariteleri",
            "İlk N parite",
        ],
    )

    max_markets = st.number_input(
        "Tarama adedi (ilk N modunda)",
        min_value=1,
        max_value=2000,
        value=50,
        step=25,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0,
        value=10000,
        step=10000,
    )

    refresh_note = st.caption(
        "Her tarama, butona basıldığında yeni verileri alır."
    )

scan_button = st.button(
    "🔍 Şimdi Tara",
    type="primary",
    use_container_width=True,
)

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None

if scan_button:
    progress = st.progress(0)
    status_box = st.empty()

    try:
        # Prefer Binance, but switch to Gate.io's own markets if
        # Binance is unavailable from the server.
        try:
            markets = binance_markets()
            tickers = binance_tickers()
            provider_name = "Binance"
        except Exception as binance_error:
            st.warning(
                "Binance API erişilemedi. Gate.io verileri "
                "kullanılıyor. Gate.io pariteleri Binance ile "
                "aynı market listesi olarak kabul edilmez. "
                f"Detay: {str(binance_error)[:200]}"
            )

            markets = gate_markets()
            tickers = gate_tickers()
            provider_name = "Gate.io"

        market_df = add_activity(markets, tickers)

        market_df = market_df[
            market_df["quote_volume"] >= float(min_volume)
        ].copy()

        if scan_mode == "En yüksek hacimli pariteler":
            market_df = market_df.sort_values(
                "quote_volume",
                ascending=False,
            ).head(int(max_markets))

        elif scan_mode == "İlk N parite":
            market_df = market_df.head(int(max_markets))

        else:
            market_df = market_df.sort_values(
                "quote_volume",
                ascending=False,
            )

        selected_markets = market_df.to_dict("records")

        if not selected_markets:
            st.error(
                "Filtrelerden geçen parite bulunamadı. "
                "Minimum hacmi düşürmeyi deneyin."
            )
            st.stop()

        st.info(
            f"Veri kaynağı: {provider_name} | "
            f"Aktif market sayısı: {len(markets)} | "
            f"Taranacak market: {len(selected_markets)}"
        )

        results = []
        errors = []

        for index, market in enumerate(selected_markets):
            symbol = market["symbol"]

            status_box.write(
                f"Tarama: {index + 1}/{len(selected_markets)} — {symbol}"
            )

            ticker = tickers.get(market["market_id"], {})

            try:
                result = analyze_market(
                    market,
                    ticker,
                    crash_threshold=crash_threshold,
                )
                results.append(result)

            except Exception as error:
                errors.append({
                    "symbol": symbol,
                    "market_id": market["market_id"],
                    "provider": market["provider"],
                    "error": str(error)[:250],
                })

            progress.progress(
                (index + 1) / len(selected_markets)
            )

            # Small pause to reduce request bursts.
            time.sleep(0.05)

        results_df = pd.DataFrame(results)
        errors_df = pd.DataFrame(errors)

        st.session_state["scan_results"] = results_df
        st.session_state["scan_errors"] = errors_df
        st.session_state["scan_provider"] = provider_name
        st.session_state["scan_time"] = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        status_box.success(
            f"Tarama tamamlandı: {len(results)} başarılı, "
            f"{len(errors)} hatalı."
        )

    except Exception as error:
        st.error(f"Tarama başlatılamadı: {error}")


# ============================================================
# RESULTS
# ============================================================

results_df = st.session_state.get("scan_results")

if results_df is not None:
    if results_df.empty:
        st.warning(
            "Analiz sonucu bulunamadı. Tarama ayarlarını değiştirip "
            "yeniden deneyin."
        )
    else:
        st.caption(
            f"Veri kaynağı: {st.session_state.get('scan_provider', '-')}"
            f" | Son tarama: {st.session_state.get('scan_time', '-')}"
        )

        buy_df = results_df[
            results_df["signal"] == "BUY"
        ].sort_values("buy_score", ascending=False)

        sell_df = results_df[
            results_df["signal"] == "SELL"
        ].sort_values("sell_score", ascending=False)

        crash_df = results_df[
            (results_df["crash_80"] == True)
            & (results_df["bars_10d"] >= CRASH_BARS)
        ].sort_values("drop_10d_pct", ascending=False)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Analiz edilen coin", len(results_df))
        col2.metric("BUY sinyali", len(buy_df))
        col3.metric("SELL sinyali", len(sell_df))
        col4.metric("10 günde %80+ düşen", len(crash_df))

        display_columns = [
            "symbol",
            "provider",
            "price",
            "change_24h",
            "volume_24h_usdt",
            "signal",
            "buy_score",
            "sell_score",
            "ai_buy_probability",
            "ai_sell_probability",
            "rsi_14",
            "volume_ratio",
        ]

        st.subheader("🟢 BUY Sinyalleri")
        if buy_df.empty:
            st.info("BUY sinyali bulunamadı.")
        else:
            st.dataframe(
                buy_df[display_columns],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "BUY listesini CSV indir",
                data=buy_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="crypto_buy_signals.csv",
                mime="text/csv",
            )

        st.subheader("🔴 SELL Sinyalleri")
        if sell_df.empty:
            st.info("SELL sinyali bulunamadı.")
        else:
            st.dataframe(
                sell_df[display_columns],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "SELL listesini CSV indir",
                data=sell_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="crypto_sell_signals.csv",
                mime="text/csv",
            )

        st.subheader("⚠️ Son 10 Günde Zirveden %80 veya Daha Fazla Düşenler")

        if crash_df.empty:
            st.info(
                "Tarama sonuçlarında yeterli 10 günlük geçmişi olan "
                "ve belirlenen düşüş eşiğini geçen coin bulunamadı."
            )
        else:
            crash_columns = [
                "symbol",
                "market_id",
                "provider",
                "price",
                "week10_peak",
                "drop_10d_pct",
                "signal",
                "buy_score",
                "sell_score",
                "bars_10d",
            ]

            st.dataframe(
                crash_df[crash_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "10 günlük sert düşüş listesini CSV indir",
                data=crash_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="crypto_10_day_crash_80_percent.csv",
                mime="text/csv",
            )

        st.subheader("📥 Tüm Analiz Sonuçları")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tüm sonuçları CSV indir",
            data=results_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="crypto_all_scan_results.csv",
            mime="text/csv",
        )

        errors_df = st.session_state.get("scan_errors")

        if errors_df is not None and not errors_df.empty:
            with st.expander(
                f"Atlanan pariteler / hatalar ({len(errors_df)})"
            ):
                st.dataframe(
                    errors_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Hata listesini CSV indir",
                    data=errors_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="crypto_scan_errors.csv",
                    mime="text/csv",
                )

else:
    st.info(
        "Tarama sonuçlarını görmek için soldaki ayarları belirleyip "
        "'Şimdi Tara' butonuna basın."
    )
