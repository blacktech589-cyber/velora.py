import time
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# ============================================================
# AYARLAR
# ============================================================

st.set_page_config(
    page_title="Hyperactive Deep Learning Spot Scanner",
    page_icon="⚡",
    layout="wide",
)

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

GATE = "https://api.gateio.ws/api/v4"
INTERVAL_SECONDS = 900
BARS_90_DAYS = 90 * 24 * 4
TIMEOUT = 20

session = requests.Session()
session.headers.update({"User-Agent": "HyperactiveSpotScanner/2.0"})


# ============================================================
# API YARDIMCILARI
# ============================================================

def get_json(url, params=None):
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def binance_json(path, params=None):
    errors = []

    for host in BINANCE_HOSTS:
        try:
            response = session.get(
                host + path,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code == 451:
                errors.append(f"{host}: HTTP 451")
                continue

            response.raise_for_status()
            return response.json(), host

        except requests.RequestException as exc:
            errors.append(f"{host}: {str(exc)[:120]}")

    raise RuntimeError(
        "Binance API erişimi başarısız. " + " | ".join(errors)
    )


def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()

    out = df.copy()
    seen = {}

    names = []
    for col in out.columns:
        name = str(col)
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count}")

    out.columns = names
    return out


# ============================================================
# PİYASA LİSTELERİ
# ============================================================

def binance_markets():
    info, host = binance_json("/api/v3/exchangeInfo")
    markets = []

    for item in info.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol")
        if not symbol:
            continue

        markets.append({
            "symbol": symbol,
            "market_id": symbol,
            "base_asset": item.get("baseAsset", ""),
            "provider": "Binance",
        })

    if not markets:
        raise RuntimeError("Aktif Binance Spot USDT piyasası bulunamadı.")

    return markets, host


def binance_tickers():
    data, host = binance_json("/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        try:
            tickers[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h_pct": float(item["priceChangePercent"]),
                "trades_24h": int(item["count"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers, host


def gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")
    markets = []

    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")

        if not market_id or not base:
            continue
        if item.get("quote") != "USDT":
            continue
        if str(item.get("trade_status", "")).lower() != "tradable":
            continue
        if item.get("delisted") is True:
            continue

        markets.append({
            "symbol": base + "USDT",
            "market_id": market_id,
            "base_asset": base,
            "provider": "Gate.io",
        })

    if not markets:
        raise RuntimeError("Aktif Gate.io Spot USDT piyasası bulunamadı.")

    return markets


def gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")
    tickers = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            tickers[market_id] = {
                "price": float(item["last"]),
                "quote_volume": float(item["quote_volume"]),
                "change_24h_pct": float(item["change_percentage"]),
                "trades_24h": np.nan,
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


def discover(exchange):
    if exchange == "Binance only":
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()
        return markets, tickers, "Binance", f"{host1}; ticker={host2}"

    if exchange == "Gate.io only":
        return gate_markets(), gate_tickers(), "Gate.io", GATE

    try:
        markets, host1 = binance_markets()
        tickers, host2 = binance_tickers()
        return markets, tickers, "Binance", f"{host1}; ticker={host2}"
    except Exception as exc:
        markets = gate_markets()
        tickers = gate_tickers()
        return (
            markets,
            tickers,
            "Gate.io",
            f"Gate.io fallback; Binance: {str(exc)[:150]}",
        )


# ============================================================
# MUM VERİLERİ
# ============================================================

def clean_candles(rows):
    cols = [
        "timestamp", "open", "high", "low",
        "close", "volume", "quote_volume",
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=cols[:6])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    df = df[df["timestamp"] < current_start]
    df = df[
        (df["open"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["close"] > 0)
    ]

    return df.reset_index(drop=True)


def binance_candles(symbol):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    start_ms = (
        current_start - (BARS_90_DAYS + 10) * INTERVAL_SECONDS
    ) * 1000
    end_ms = current_start * 1000

    cursor = start_ms
    rows = []

    while cursor < end_ms:
        data, _ = binance_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "15m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not data:
            break

        for candle in data:
            try:
                rows.append({
                    "timestamp": int(candle[0]) // 1000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "quote_volume": float(candle[7]),
                })
            except (ValueError, TypeError, IndexError):
                continue

        next_cursor = int(data[-1][0]) + INTERVAL_SECONDS * 1000

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(data) < 1000:
            break

        time.sleep(0.04)

    df = clean_candles(rows)

    if not df.empty:
        cutoff = current_start - BARS_90_DAYS * INTERVAL_SECONDS
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    return df


def gate_candles(market_id):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = now // INTERVAL_SECONDS * INTERVAL_SECONDS

    cursor = (
        current_start - (BARS_90_DAYS + 10) * INTERVAL_SECONDS
    )
    rows = []

    while cursor < current_start:
        page_end = min(
            cursor + 999 * INTERVAL_SECONDS,
            current_start,
        )

        data = get_json(
            f"{GATE}/spot/candlesticks",
            {
                "currency_pair": market_id,
                "interval": "15m",
                "from": cursor,
                "to": page_end,
                "limit": 1000,
            },
        )

        if not data:
            break

        for candle in data:
            if len(candle) < 6:
                continue

            try:
                rows.append({
                    "timestamp": int(candle[0]),
                    "quote_volume": float(candle[1]),
                    "close": float(candle[2]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "open": float(candle[5]),
                    "volume": float(candle[6]) if len(candle) > 6 else 0.0,
                })
            except (ValueError, TypeError, IndexError):
                continue

        cursor = page_end + INTERVAL_SECONDS
        time.sleep(0.04)

    df = clean_candles(rows)

    if not df.empty:
        cutoff = current_start - BARS_90_DAYS * INTERVAL_SECONDS
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    return df


def load_candles(market):
    if market["provider"] == "Binance":
        return binance_candles(market["market_id"])
    return gate_candles(market["market_id"])


# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================

def rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def build_strategy_frame(df):
    x = df.copy().reset_index(drop=True)

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["ret1"] = close.pct_change()
    x["rsi7"] = rsi(close, 7)
    x["rsi14"] = rsi(close, 14)
    x["rsi21"] = rsi(close, 21)

    for span in [5, 8, 12, 16, 20, 26, 30, 50, 80, 100, 200]:
        x[f"ema{span}"] = close.ewm(
            span=span,
            adjust=False,
        ).mean()

    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()

    x["std20"] = close.rolling(20).std(ddof=0)
    x["bb_upper"] = x["sma20"] + 2 * x["std20"]
    x["bb_lower"] = x["sma20"] - 2 * x["std20"]

    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()

    x["stoch_k"] = (
        100 * (close - low14)
        / (high14 - low14).replace(0, np.nan)
    )
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()

    x["momentum10"] = close.pct_change(10) * 100
    x["momentum20"] = close.pct_change(20) * 100

    x["vol_sma20"] = volume.rolling(20).mean()
    x["volume_ratio"] = volume / x["vol_sma20"].replace(0, np.nan)

    x["atr14"] = atr(x, 14)
    x["atr_pct"] = x["atr14"] / close * 100

    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(14).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14

    x["adx_proxy"] = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return x


# ============================================================
# 800 STRATEJİ KOMBİNASYONU
# ============================================================

def evaluate_800_strategies(df):
    """
    8 strateji ailesi x 100 parametre kombinasyonu = 800.
    Bunlar parametreli teknik stratejilerdir; 800 ayrı yapay zekâ modeli
    oldukları anlamına gelmez.
    """
    x = build_strategy_frame(df)

    empty_result = {
        "strategy_count": 0,
        "strategy_score": np.nan,
        "strategy_signal": "INSUFFICIENT HISTORY",
        "strategy_buy_votes": 0,
        "strategy_sell_votes": 0,
        "strategy_win_rate_pct": np.nan,
        "strategy_avg_next_return_pct": np.nan,
    }

    if len(x) < 80:
        return empty_result

    future = x["close"].shift(-1) / x["close"] - 1

    grid = list(product(
        [5, 8, 12, 16, 20],
        [20, 30, 50, 80],
        [25, 30, 35, 40, 45],
    ))

    configs = []

    for a, b, c in grid:
        configs.append(("ema_rsi", a, b, c))
        configs.append(("bollinger_rsi", a, b, c))
        configs.append(("macd_momentum", a, b, c))
        configs.append(("stoch_volume", a, b, c))
        configs.append(("trend_pullback", a, b, c))
        configs.append(("rsi_momentum", a, b, c))
        configs.append(("adx_trend", a, b, c))
        configs.append(("sma_volume", a, b, c))

    sample = x.tail(min(len(x), 1800)).copy()
    future = future.loc[sample.index]
    trainable = future.notna()

    recent_signals = []
    returns_by_strategy = []
    wins_by_strategy = []

    for family, a, b, c in configs:
        if family == "ema_rsi":
            signal = np.where(
                (sample[f"ema{a}"] > sample[f"ema{b}"])
                & (sample["rsi14"] > c),
                1,
                np.where(
                    (sample[f"ema{a}"] < sample[f"ema{b}"])
                    & (sample["rsi14"] < 100 - c),
                    -1,
                    0,
                ),
            )

        elif family == "bollinger_rsi":
            mid = sample["close"].rolling(a).mean()
            sd = sample["close"].rolling(a).std(ddof=0)

            lower = mid - (b / 20) * sd
            upper = mid + (b / 20) * sd

            signal = np.where(
                (sample["close"] < lower)
                & (sample["rsi14"] < c + 20),
                1,
                np.where(
                    (sample["close"] > upper)
                    & (sample["rsi14"] > 100 - c - 20),
                    -1,
                    0,
                ),
            )

        elif family == "macd_momentum":
            macd = sample[f"ema{a}"] - sample[f"ema{min(b, 26)}"]
            sig = macd.ewm(span=max(3, a), adjust=False).mean()
            mom = sample["close"].pct_change(max(2, a)) * 100

            signal = np.where(
                (macd > sig) & (mom > -c / 10),
                1,
                np.where(
                    (macd < sig) & (mom < c / 10),
                    -1,
                    0,
                ),
            )

        elif family == "stoch_volume":
            k = sample["stoch_k"]
            liquid = sample["volume_ratio"] > 0.8

            signal = np.where(
                (k < c + 10) & liquid,
                1,
                np.where(
                    (k > 100 - c - 10) & liquid,
                    -1,
                    0,
                ),
            )

        elif family == "trend_pullback":
            trend = sample[f"ema{a}"] > sample[f"ema{b}"]

            signal = np.where(
                trend & (sample["rsi7"] < c + 20),
                1,
                np.where(
                    (~trend) & (sample["rsi7"] > 100 - c - 20),
                    -1,
                    0,
                ),
            )

        elif family == "rsi_momentum":
            mom = sample["close"].pct_change(max(2, a)) * 100

            signal = np.where(
                (sample["rsi21"] < c) & (mom > -b / 10),
                1,
                np.where(
                    (sample["rsi21"] > 100 - c) & (mom < b / 10),
                    -1,
                    0,
                ),
            )

        elif family == "adx_trend":
            trend = sample[f"ema{a}"] > sample[f"ema{b}"]
            strong = sample["adx_proxy"] > c

            signal = np.where(
                trend & strong,
                1,
                np.where((~trend) & strong, -1, 0),
            )

        else:
            fast_sma = sample["close"].rolling(a).mean()
            slow_sma = sample["close"].rolling(b).mean()
            liquid = sample["volume_ratio"] > (0.5 + c / 100)

            signal = np.where(
                (fast_sma > slow_sma) & liquid,
                1,
                np.where(
                    (fast_sma < slow_sma) & liquid,
                    -1,
                    0,
                ),
            )

        signal = pd.Series(
            signal,
            index=sample.index,
        ).fillna(0)

        valid = trainable & signal.ne(0)
        strategy_returns = signal[valid] * future[valid]

        if len(strategy_returns):
            returns_by_strategy.append(
                float(strategy_returns.mean() * 100)
            )
            wins_by_strategy.append(
                float((strategy_returns > 0).mean() * 100)
            )

        recent_signals.append(int(signal.iloc[-1]))

    votes = np.asarray(recent_signals, dtype=int)

    buy_votes = int((votes == 1).sum())
    sell_votes = int((votes == -1).sum())
    active_votes = buy_votes + sell_votes

    score = (
        buy_votes / active_votes * 100
        if active_votes
        else 50.0
    )

    if buy_votes > sell_votes:
        signal_text = "BUY BIAS"
    elif sell_votes > buy_votes:
        signal_text = "SELL BIAS"
    else:
        signal_text = "MIXED / NEUTRAL"

    return {
        "strategy_count": len(configs),
        "strategy_score": float(score),
        "strategy_signal": signal_text,
        "strategy_buy_votes": buy_votes,
        "strategy_sell_votes": sell_votes,
        "strategy_win_rate_pct": (
            float(np.mean(wins_by_strategy))
            if wins_by_strategy else np.nan
        ),
        "strategy_avg_next_return_pct": (
            float(np.mean(returns_by_strategy))
            if returns_by_strategy else np.nan
        ),
    }


# ============================================================
# LSTM DERİN ÖĞRENME MODELİ
# ============================================================

if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=5, hidden_size=16):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
            )

            self.head = nn.Sequential(
                nn.Linear(hidden_size, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])


def deep_learning_forecast(df, epochs=3, lookback=32):
    if not TORCH_AVAILABLE:
        return {
            "dl_up_probability_pct": np.nan,
            "dl_signal": "PyTorch missing",
            "dl_train_samples": 0,
        }

    if len(df) < max(lookback + 50, 100):
        return {
            "dl_up_probability_pct": np.nan,
            "dl_signal": "INSUFFICIENT HISTORY",
            "dl_train_samples": 0,
        }

    torch.manual_seed(7)
    torch.set_num_threads(1)

    x = build_strategy_frame(df).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = pd.DataFrame({
        "return": x["ret1"],
        "rsi": x["rsi14"] / 100.0,
        "trend": (x["ema20"] / x["ema50"] - 1).clip(-1, 1),
        "volatility": x["ret1"].rolling(20).std(),
        "volume": np.log1p(x["volume"].clip(lower=0)),
    }).replace([np.inf, -np.inf], np.nan)

    # Son mumun bir sonraki mum hedefi bilinmediğinden eğitim dışında kalır.
    target = (x["close"].shift(-1) > x["close"]).astype(float)

    valid_end = len(x) - 1
    features = features.iloc[:valid_end]
    target = target.iloc[:valid_end]

    split = max(lookback + 20, int(len(features) * 0.8))
    split = min(split, len(features) - 5)

    if split <= lookback:
        return {
            "dl_up_probability_pct": np.nan,
            "dl_signal": "INSUFFICIENT HISTORY",
            "dl_train_samples": 0,
        }

    # Ölçekleme istatistikleri yalnızca eğitim bölgesinden hesaplanır.
    means = features.iloc[:split].mean()
    stds = features.iloc[:split].std().replace(0, 1).fillna(1)

    z = (
        (features - means) / stds
    ).clip(-8, 8).fillna(0).to_numpy(dtype=np.float32)

    labels = target.to_numpy(dtype=np.float32)

    sequences = []
    ys = []

    for end in range(lookback, split):
        sequences.append(z[end - lookback:end])
        ys.append(labels[end])

    if len(sequences) < 20:
        return {
            "dl_up_probability_pct": np.nan,
            "dl_signal": "INSUFFICIENT TRAINING DATA",
            "dl_train_samples": len(sequences),
        }

    X = torch.tensor(np.asarray(sequences), dtype=torch.float32)
    y = torch.tensor(
        np.asarray(ys).reshape(-1, 1),
        dtype=torch.float32,
    )

    model = LSTMClassifier(input_size=z.shape[1], hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCELoss()

    model.train()
    batch_size = min(64, len(X))

    for _ in range(max(1, int(epochs))):
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

    model.eval()

    # En son tamamlanmış mumlardan tahmin.
    last_sequence = torch.tensor(
        z[-lookback:][None, :, :],
        dtype=torch.float32,
    )

    with torch.no_grad():
        probability = float(
            model(last_sequence).item() * 100
        )

    if probability >= 55:
        signal = "UP BIAS"
    elif probability <= 45:
        signal = "DOWN BIAS"
    else:
        signal = "UNCERTAIN"

    return {
        "dl_up_probability_pct": probability,
        "dl_signal": signal,
        "dl_train_samples": len(sequences),
    }


# ============================================================
# DÜŞÜŞ VE DİP ANALİZİ
# ============================================================

def analyze(df, max_distance, min_volatility, crash_threshold=90.0):
    if len(df) < 20:
        return {
            "valid": False,
            "reason": f"Yalnızca {len(df)} tamamlanmış mum var.",
        }

    period = df.tail(BARS_90_DAYS).reset_index(drop=True)

    coverage_days = len(period) * 15 / 60 / 24

    peak_pos = int(period["high"].to_numpy().argmax())
    low_pos = int(period["low"].to_numpy().argmin())

    peak_price = float(period.iloc[peak_pos]["high"])
    low_price = float(period.iloc[low_pos]["low"])
    current_price = float(period.iloc[-1]["close"])

    if min(peak_price, low_price, current_price) <= 0:
        return {"valid": False, "reason": "Geçersiz fiyat verisi."}

    drop_from_peak_pct = (
        (peak_price - current_price) / peak_price * 100
    )

    distance_from_low_pct = (
        (current_price / low_price - 1) * 100
    )

    last3 = period.tail(3)
    closes = last3["close"].to_numpy()
    opens = last3["open"].to_numpy()

    rising3 = bool(closes[0] < closes[1] < closes[2])
    green_last = bool(closes[-1] > opens[-1])

    near_low = bool(
        0 <= distance_from_low_pct <= max_distance
    )

    crashed_enough = bool(
        drop_from_peak_pct >= crash_threshold
    )

    period["ret"] = period["close"].pct_change()
    period["atr"] = atr(period)
    period["rsi"] = rsi(period["close"])

    volume_24h = float(
        period["quote_volume"].tail(96).fillna(0).sum()
    )

    returns_24h = period["ret"].tail(96).dropna()

    volatility_pct = (
        float(returns_24h.std(ddof=1) * np.sqrt(96) * 100)
        if len(returns_24h) >= 48
        else np.nan
    )

    atr_pct = (
        float(period["atr"].iloc[-1] / current_price * 100)
        if pd.notna(period["atr"].iloc[-1])
        else np.nan
    )

    rsi_value = (
        float(period["rsi"].iloc[-1])
        if pd.notna(period["rsi"].iloc[-1])
        else np.nan
    )

    volume_score = min(
        100.0,
        max(0.0, np.log10(max(volume_24h, 1.0)) * 10),
    )

    volatility_score = (
        min(100.0, max(0.0, volatility_pct * 5))
        if pd.notna(volatility_pct)
        else 0.0
    )

    activity_score = (volume_score + volatility_score) / 2

    high_activity = bool(
        volume_24h > 0
        and pd.notna(volatility_pct)
        and volatility_pct >= min_volatility
    )

    crash_bottom_candidate = bool(
        crashed_enough and near_low and high_activity
    )

    confirmed_reversal = bool(
        crash_bottom_candidate and rising3 and green_last
    )

    if confirmed_reversal:
        signal = "90% CRASH / LOW-AREA REVERSAL"
    elif crash_bottom_candidate and rising3:
        signal = "90% CRASH / POSSIBLE BOTTOM"
    elif crashed_enough and near_low:
        signal = "90% CRASH / NEAR 90-DAY LOW"
    elif crashed_enough:
        signal = "90% CRASH / NOT NEAR LOW"
    else:
        signal = "CRASH THRESHOLD NOT MET"

    peak_time = datetime.fromtimestamp(
        int(period.iloc[peak_pos]["timestamp"]),
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    low_time = datetime.fromtimestamp(
        int(period.iloc[low_pos]["timestamp"]),
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "valid": True,
        "current_close_15m": current_price,
        "90d_peak": peak_price,
        "90d_peak_time_utc": peak_time,
        "90d_low": low_price,
        "90d_low_time_utc": low_time,
        "drop_from_90d_peak_pct": drop_from_peak_pct,
        "distance_from_90d_low_pct": distance_from_low_pct,
        "crashed_90pct": crashed_enough,
        "near_90d_low": near_low,
        "three_rising_candles": rising3,
        "latest_candle_green": green_last,
        "volume_24h_candles_usdt": volume_24h,
        "realized_volatility_24h_pct": volatility_pct,
        "atr_pct": atr_pct,
        "rsi_14": rsi_value,
        "volume_score": volume_score,
        "volatility_score": volatility_score,
        "activity_score": activity_score,
        "high_activity": high_activity,
        "crash_bottom_candidate": crash_bottom_candidate,
        "confirmed_reversal": confirmed_reversal,
        "signal": signal,
        "candles_analyzed": len(period),
        "history_days_analyzed": coverage_days,
        "full_90d_history": bool(len(period) >= BARS_90_DAYS),
    }


# ============================================================
# COIN TARAMASI
# ============================================================

def scan_market(
    market,
    ticker,
    max_distance,
    min_volatility,
    crash_threshold,
    dl_enabled=True,
    dl_epochs=3,
):
    df = load_candles(market)

    analysis = analyze(
        df,
        max_distance,
        min_volatility,
        crash_threshold,
    )

    if not analysis.get("valid"):
        raise RuntimeError(
            analysis.get("reason", "Analiz başarısız.")
        )

    strategy = evaluate_800_strategies(df)

    if dl_enabled:
        deep = deep_learning_forecast(
            df,
            epochs=dl_epochs,
        )
    else:
        deep = {
            "dl_up_probability_pct": np.nan,
            "dl_signal": "DISABLED",
            "dl_train_samples": 0,
        }

    candle_price = float(analysis["current_close_15m"])
    ticker_price = float(ticker["price"])

    price_diff = (
        abs(ticker_price - candle_price) / candle_price * 100
    )

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "provider": market["provider"],
        "ticker_price": ticker_price,
        "price_difference_pct": price_diff,
        "price_check_ok": price_diff <= 2,
        **strategy,
        **deep,
        "ticker_quote_volume_24h_usdt": float(ticker["quote_volume"]),
        "change_24h_pct": float(ticker["change_24h_pct"]),
        "trades_24h": ticker.get("trades_24h", np.nan),
        **analysis,
    }


# ============================================================
# STREAMLIT ARAYÜZÜ
# ============================================================

st.title("⚡ Hyperactive Deep Learning Spot Scanner")

st.write(
    "Aktif Spot USDT piyasalarını tarar; zirvesinden düşen coinleri, "
    "dip bölgesine yakınlıklarını, hacimlerini, 800 strateji oyunu ve "
    "LSTM modelinin sonraki mum tahminini gösterir. Emir göndermez."
)

st.warning(
    "Bu araç yatırım tavsiyesi değildir. Sert düşmüş bir coin daha fazla "
    "düşebilir. LSTM tahmini ve strateji puanı kâr garantisi vermez."
)

with st.sidebar:
    exchange = st.selectbox(
        "Borsa",
        [
            "Auto (Binance then Gate.io fallback)",
            "Binance only",
            "Gate.io only",
        ],
    )

    crash_threshold = st.slider(
        "90 günlük zirveden minimum düşüş (%)",
        min_value=80,
        max_value=99,
        value=90,
        step=1,
    )

    max_distance = st.slider(
        "90 günlük dipten maksimum uzaklık (%)",
        min_value=1.0,
        max_value=10.0,
        value=10.0,
        step=0.5,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0,
        value=100000,
        step=50000,
    )

    min_volatility = st.number_input(
        "Minimum 24 saatlik volatilite (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5,
    )

    scan_count = st.number_input(
        "Taranacak aktif piyasa sayısı",
        min_value=1,
        max_value=10000,
        value=50,
        step=25,
    )

    dl_enabled = st.checkbox(
        "LSTM derin öğrenmeyi çalıştır",
        value=True,
    )

    dl_epochs = st.slider(
        "LSTM eğitim turu (epoch)",
        min_value=1,
        max_value=10,
        value=3,
    )

    if dl_enabled and not TORCH_AVAILABLE:
        st.error(
            "PyTorch kurulu değil. requirements.txt dosyasını yükleyip "
            "pip install -r requirements.txt komutunu çalıştır."
        )

    run = st.button(
        "⚡ Taramayı başlat",
        type="primary",
        use_container_width=True,
    )


if "results" not in st.session_state:
    st.session_state.results = None

if "errors" not in st.session_state:
    st.session_state.errors = pd.DataFrame()


if run:
    progress = st.progress(0)
    status = st.empty()

    try:
        markets, tickers, provider, source = discover(exchange)

        candidates = []

        for market in markets:
            ticker = tickers.get(market["market_id"])

            if ticker is None:
                continue

            if float(ticker["quote_volume"]) < min_volume:
                continue

            candidates.append({**market, **ticker})

        candidates.sort(
            key=lambda item: float(item["quote_volume"]),
            reverse=True,
        )

        candidates = candidates[:int(scan_count)]

        if not candidates:
            raise RuntimeError(
                "Hacim filtresini geçen piyasa yok. Minimum hacmi düşür."
            )

        st.info(
            f"Kaynak: {provider} | "
            f"Aktif piyasalar: {len(markets)} | "
            f"Taranacak: {len(candidates)} | API: {source}"
        )

        results = []
        errors = []

        for i, market in enumerate(candidates):
            status.write(
                f"{i + 1}/{len(candidates)} — {market['symbol']}"
            )

            try:
                result = scan_market(
                    market,
                    tickers[market["market_id"]],
                    max_distance,
                    min_volatility,
                    crash_threshold,
                    dl_enabled=dl_enabled,
                    dl_epochs=dl_epochs,
                )

                results.append(result)

            except Exception as exc:
                errors.append({
                    "symbol": market["symbol"],
                    "market_id": market["market_id"],
                    "provider": market["provider"],
                    "error": str(exc)[:250],
                })

            progress.progress((i + 1) / len(candidates))
            time.sleep(0.05)

        st.session_state.results = pd.DataFrame(results)
        st.session_state.errors = pd.DataFrame(errors)
        st.session_state.provider = provider
        st.session_state.scan_time = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        status.success(
            f"Tamamlandı: {len(results)} analiz, {len(errors)} hata."
        )

    except Exception as exc:
        st.error(f"Tarama başarısız: {exc}")


# ============================================================
# SONUÇLAR
# ============================================================

results = st.session_state.get("results")

if results is None:
    st.info("Ayarları seç ve taramayı başlat.")

elif results.empty:
    st.warning(
        "Başarılı analiz yok. Hata listesini kontrol et veya filtreleri gevşet."
    )

else:
    st.caption(
        f"Kaynak: {st.session_state.get('provider', '-')} | "
        f"Tarama zamanı: {st.session_state.get('scan_time', '-')}"
    )

    verified = results[
        results["price_check_ok"] == True
    ].copy()

    all_falling = verified[
        verified["drop_from_90d_peak_pct"] > 0
    ].copy()

    all_falling = all_falling.sort_values(
        "drop_from_90d_peak_pct",
        ascending=False,
    )

    crashed = verified[
        verified["drop_from_90d_peak_pct"] >= crash_threshold
    ].copy()

    crash_bottom = crashed[
        (crashed["distance_from_90d_low_pct"] <= max_distance)
        & (crashed["distance_from_90d_low_pct"] >= 0)
        & (crashed["high_activity"] == True)
    ].copy()

    reversals = crash_bottom[
        (crash_bottom["three_rising_candles"] == True)
        & (crash_bottom["latest_candle_green"] == True)
    ].sort_values(
        "activity_score",
        ascending=False,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Analiz edilen", len(results))
    c2.metric("Fiyatı doğrulanan", len(verified))
    c3.metric("Zirvesinden gerileyen", len(all_falling))
    c4.metric(
        f"Düşüş >= %{crash_threshold}",
        len(crashed),
    )

    columns = [
        "symbol",
        "market_id",
        "provider",
        "current_close_15m",
        "ticker_price",
        "price_difference_pct",
        "90d_peak",
        "90d_peak_time_utc",
        "90d_low",
        "90d_low_time_utc",
        "drop_from_90d_peak_pct",
        "distance_from_90d_low_pct",
        "history_days_analyzed",
        "full_90d_history",
        "volume_24h_candles_usdt",
        "realized_volatility_24h_pct",
        "atr_pct",
        "rsi_14",
        "trades_24h",
        "activity_score",
        "signal",
        "strategy_count",
        "strategy_score",
        "strategy_signal",
        "strategy_buy_votes",
        "strategy_sell_votes",
        "strategy_win_rate_pct",
        "strategy_avg_next_return_pct",
        "dl_up_probability_pct",
        "dl_signal",
        "dl_train_samples",
    ]

    columns = [
        col for col in columns
        if col in results.columns
    ]

    st.subheader(
        "🎯 Adaylar: sert düşüş + dip bölgesi + yükselen mumlar"
    )

    st.dataframe(
        safe_dataframe(reversals[columns]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Adayları CSV indir",
        safe_dataframe(reversals).to_csv(
            index=False
        ).encode("utf-8-sig"),
        "crash_bottom_reversals.csv",
        "text/csv",
    )

    st.subheader("📉 Zirvesinden gerileyen TÜM aktif coinler")

    falling_columns = list(dict.fromkeys(
        columns + [
            "candles_analyzed",
            "history_days_analyzed",
            "full_90d_history",
        ]
    ))

    falling_columns = [
        col for col in falling_columns
        if col in all_falling.columns
    ]

    st.dataframe(
        safe_dataframe(all_falling[falling_columns]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Tüm düşen coinleri CSV indir",
        safe_dataframe(all_falling).to_csv(
            index=False
        ).encode("utf-8-sig"),
        "all_falling_spot_usdt_coins.csv",
        "text/csv",
    )

    st.subheader("📉 Belirlenen eşiği aşan düşüşler")

    st.dataframe(
        safe_dataframe(
            crashed.sort_values(
                "drop_from_90d_peak_pct",
                ascending=False,
            )[columns]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Düşüş listesini CSV indir",
        safe_dataframe(crashed).to_csv(
            index=False
        ).encode("utf-8-sig"),
        "crash_list.csv",
        "text/csv",
    )

    st.subheader("🧠 LSTM + 800 strateji özeti")

    summary_cols = [
        col for col in [
            "symbol",
            "drop_from_90d_peak_pct",
            "strategy_count",
            "strategy_score",
            "strategy_signal",
            "strategy_buy_votes",
            "strategy_sell_votes",
            "strategy_win_rate_pct",
            "strategy_avg_next_return_pct",
            "dl_up_probability_pct",
            "dl_signal",
            "dl_train_samples",
        ]
        if col in results.columns
    ]

    summary = results[summary_cols].copy()

    if "strategy_score" in summary.columns:
        summary = summary.sort_values(
            "strategy_score",
            ascending=False,
            na_position="last",
        )

    st.dataframe(
        safe_dataframe(summary),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "800 strateji teknik indikatör kombinasyonudur; 800 bağımsız "
        "yapay zekâ modeli değildir. Geçmiş getiri ölçümleri işlem "
        "ücretlerini ve kaymayı içermez. LSTM tahmini garanti değildir."
    )

    st.subheader("📋 Tüm tarama sonuçları")

    results_display = safe_dataframe(results)

    st.dataframe(
        results_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Tüm sonuçları CSV indir",
        results_display.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "all_scan_results.csv",
        "text/csv",
    )

    errors = st.session_state.get("errors")

    if errors is not None and not errors.empty:
        with st.expander(f"Piyasa hataları ({len(errors)})"):
            st.dataframe(
                safe_dataframe(errors),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Hata listesini CSV indir",
                errors.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "scan_errors.csv",
                "text/csv",
            )
