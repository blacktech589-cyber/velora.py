
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# =========================================================
# SMART FUTURES SCANNER
# Data: OKX public USDT perpetual swaps
# Hourly market refresh
# Signals: BUY / SELL
# Ranking: highest 24h percentage change first
# No API keys, no order placement, no Plotly
# =========================================================

st.set_page_config(
    page_title="Smart Futures AI Scanner",
    page_icon="📈",
    layout="wide",
)

BASE_URL = "https://www.okx.com"
TIMEOUT = 20
INTERVAL = "15m"
REFRESH_MS = 60 * 60 * 1000

session = requests.Session()
session.headers.update({
    "User-Agent": "SmartFuturesScanner/2.0",
    "Accept": "application/json",
})


# ---------------- AUTOMATIC HOURLY REFRESH ----------------

hourly_tick = st_autorefresh(
    interval=REFRESH_MS,
    limit=None,
    key="hourly_market_refresh",
)

# ---------------- PUBLIC API ----------------

def api_get(path, params=None):
    response = session.get(
        BASE_URL + path,
        params=params,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    if str(result.get("code", "0")) != "0":
        raise RuntimeError(
            result.get("msg", "API request failed")
        )

    return result.get("data", [])


@st.cache_data(ttl=3600, show_spinner=False)
def get_instruments():
    data = api_get(
        "/api/v5/public/instruments",
        {"instType": "SWAP"},
    )

    return [
        item["instId"]
        for item in data
        if item.get("state") == "live"
        and item.get("settleCcy") == "USDT"
        and item.get("ctType") == "linear"
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def get_tickers():
    return api_get(
        "/api/v5/market/tickers",
        {"instType": "SWAP"},
    )


# ---------------- MARKET DATA ----------------

def build_market_table(symbols, tickers, min_volume):
    symbol_set = set(symbols)
    rows = []

    for item in tickers:
        symbol = item.get("instId")

        if symbol not in symbol_set:
            continue

        try:
            last = float(item.get("last", 0))
            open24 = float(item.get("open24h", 0))
            high24 = float(item.get("high24h", 0))
            low24 = float(item.get("low24h", 0))
            volume = float(item.get("vol24h", 0))
        except (TypeError, ValueError):
            continue

        if last <= 0 or open24 <= 0 or volume < min_volume:
            continue

        rows.append({
            "Symbol": symbol,
            "24h change (%)": (last / open24 - 1) * 100,
            "Last price": last,
            "24h high": high24,
            "24h low": low24,
            "24h volume (contracts)": volume,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df.sort_values("24h change (%)", ascending=False)
        .reset_index(drop=True)
        .assign(Rank=lambda x: np.arange(1, len(x) + 1))
    )


def parse_candles(data):
    if not data:
        raise RuntimeError("No candle data returned.")

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "volume_currency", "volume_quote", "confirm",
        ],
    )

    for col in [
        "timestamp", "open", "high", "low",
        "close", "volume", "volume_quote",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["confirm"].astype(str) == "1"].copy()
    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    return df


def get_candles(symbol, target=500):
    frames = []
    seen = set()

    latest_data = api_get(
        "/api/v5/market/candles",
        {
            "instId": symbol,
            "bar": INTERVAL,
            "limit": str(min(target, 300)),
        },
    )

    latest = parse_candles(latest_data)
    frames.append(latest)
    seen.update(latest["timestamp"].astype(int).tolist())

    before = int(latest["timestamp"].min())

    for _ in range(12):
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates("timestamp")

        if len(combined) >= target:
            break

        page = api_get(
            "/api/v5/market/history-candles",
            {
                "instId": symbol,
                "bar": INTERVAL,
                "limit": "100",
                "after": str(before),
            },
        )

        older = parse_candles(page)
        new_rows = older[
            ~older["timestamp"].astype(int).isin(seen)
        ].copy()

        if new_rows.empty:
            break

        seen.update(new_rows["timestamp"].astype(int).tolist())
        frames.append(new_rows)

        new_before = int(new_rows["timestamp"].min())
        if new_before >= before:
            break

        before = new_before
        time.sleep(0.12)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates("timestamp")
    result = result.sort_values("timestamp").tail(target)
    result = result.reset_index(drop=True)

    if len(result) < 150:
        raise RuntimeError(
            f"Not enough completed candles: {len(result)}"
        )

    return result


# ---------------- SMARTER FEATURES ----------------

def make_features(df):
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    # Multi-horizon price momentum.
    for lag in [1, 2, 3, 4, 8, 12, 24, 48, 96]:
        f[f"return_{lag}"] = close.pct_change(lag)

    # Candle geometry.
    f["body"] = (df["close"] - df["open"]) / open_price
    f["range"] = (high - low) / close

    candle_range = (high - low).replace(0, np.nan)
    f["close_position"] = (df["close"] - low) / candle_range

    f["upper_wick"] = (
        high - df[["open", "close"]].max(axis=1)
    ) / close

    f["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - low
    ) / close

    # Momentum consistency and rolling volatility.
    one_return = close.pct_change()

    for window in [4, 8, 16, 32, 64]:
        f[f"mean_return_{window}"] = (
            one_return.rolling(window).mean()
        )
        f[f"volatility_{window}"] = (
            one_return.rolling(window).std()
        )
        f[f"range_mean_{window}"] = (
            f["range"].rolling(window).mean()
        )

    # Volume expansion / contraction.
    for lag in [1, 2, 4, 8]:
        f[f"volume_change_{lag}"] = (
            volume / volume.shift(lag) - 1
        )

    f["relative_volume_24"] = (
        volume / volume.rolling(24).mean() - 1
    )

    f["relative_volume_48"] = (
        volume / volume.rolling(48).mean() - 1
    )

    return f.replace([np.inf, -np.inf], np.nan)


# ---------------- LOGISTIC MODEL ----------------

def sigmoid(z):
    return 1.0 / (
        1.0 + np.exp(-np.clip(z, -35, 35))
    )


def train_model(X, y, epochs=500, learning_rate=0.04):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    for epoch in range(epochs):
        pred = sigmoid(X @ weights + bias)
        error = pred - y

        grad_w = (
            X.T @ error
        ) / n + 0.005 * weights

        grad_b = error.mean()
        lr = learning_rate / (1 + epoch / 200)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def analyze_model(df):
    features = make_features(df)

    # Predict direction four candles ahead (one hour on 15m).
    future_return = df["close"].shift(-4) / df["close"] - 1

    valid = features.notna().all(axis=1) & future_return.notna()
    X = features.loc[valid].to_numpy(dtype=float)
    y = (future_return.loc[valid].to_numpy() > 0).astype(float)

    if len(X) < 180:
        raise RuntimeError("Not enough valid training samples.")

    split = int(len(X) * 0.8)
    split = max(60, min(split, len(X) - 30))

    X_train_raw = X[:split]
    X_test_raw = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # Scale using training data only.
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1.0

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    weights, bias = train_model(X_train, y_train)
    test_prob = sigmoid(X_test @ weights + bias)
    test_pred = (test_prob >= 0.5).astype(float)

    accuracy = float((test_pred == y_test).mean() * 100)

    # Balanced accuracy treats rising/falling classes separately.
    class_scores = []

    for label in [0.0, 1.0]:
        mask = y_test == label
        if mask.any():
            class_scores.append(
                float((test_pred[mask] == y_test[mask]).mean())
            )

    balanced_accuracy = (
        float(np.mean(class_scores) * 100)
        if class_scores else accuracy
    )

    # Retrain using all labeled historical samples.
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1.0

    X_all = (X - mean_all) / std_all
    weights, bias = train_model(X_all, y)

    latest = features.iloc[[-1]].to_numpy(dtype=float)

    if not np.isfinite(latest).all():
        raise RuntimeError("Invalid latest candle features.")

    latest_scaled = (latest - mean_all) / std_all
    p_up = float(
        sigmoid(latest_scaled @ weights + bias)[0]
    )

    return {
        "Signal": "BUY" if p_up >= 0.5 else "SELL",
        "BUY probability (%)": round(p_up * 100, 2),
        "SELL probability (%)": round((1 - p_up) * 100, 2),
        "Test accuracy (%)": round(accuracy, 2),
        "Balanced accuracy (%)": round(balanced_accuracy, 2),
        "Test samples": len(X_test),
    }


# ---------------- UI ----------------

st.title("📈 Smart Futures AI Scanner")
st.caption(
    "OKX USDT perpetual swaps | Hourly refresh | "
    "BUY / SELL predictions | Highest 24h change first"
)

with st.sidebar:
    st.header("Settings")

    min_volume = st.number_input(
        "Minimum 24h volume (contracts)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

    candle_count = st.selectbox(
        "Historical 15-minute candles",
        [200, 300, 500, 800, 1000],
        index=2,
    )

    scan_count = st.selectbox(
        "Model scan",
        ["TOP 20", "TOP 50", "TOP 100", "ALL CONTRACTS"],
        index=0,
    )

    min_probability = st.slider(
        "Minimum BUY/SELL probability (%)",
        50, 95, 60,
    )

    min_accuracy = st.slider(
        "Minimum balanced test accuracy (%)",
        0, 90, 50,
    )

    signal_filter = st.selectbox(
        "Signal filter",
        ["ALL", "BUY", "SELL"],
    )

    manual_refresh = st.button(
        "Refresh now",
        type="primary",
        use_container_width=True,
    )

# Auto refresh count changes each hour.
if "last_refresh_tick" not in st.session_state:
    st.session_state["last_refresh_tick"] = -1

if "market_table" not in st.session_state:
    st.session_state["market_table"] = None

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []

hour_changed = (
    hourly_tick != st.session_state["last_refresh_tick"]
)

should_refresh = (
    manual_refresh
    or st.session_state["market_table"] is None
    or hour_changed
)

if should_refresh:
    try:
        with st.spinner("Refreshing futures market data..."):
            symbols = get_instruments()
            tickers = get_tickers()
            market = build_market_table(
                symbols, tickers, min_volume
            )

            if market.empty:
                st.error("No contracts returned by the public API.")
                st.stop()

            st.session_state["market_table"] = market
            st.session_state["last_refresh_tick"] = hourly_tick
            st.session_state["last_updated_utc"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )

            # Old model predictions are no longer presented as fresh.
            st.session_state["model_results"] = []

    except Exception as exc:
        st.error("Could not refresh market data.")
        st.code(str(exc))
        st.info(
            "This app uses OKX data, not Binance data. "
            "Check public API availability if the request fails."
        )
        st.stop()

market = st.session_state["market_table"].copy()
market = market.sort_values(
    "24h change (%)", ascending=False
).reset_index(drop=True)
market["Rank"] = np.arange(1, len(market) + 1)

top = market.iloc[0]
rising = int((market["24h change (%)"] > 0).sum())
falling = int((market["24h change (%)"] < 0).sum())

a, b, c, d = st.columns(4)
a.metric("Active USDT perpetuals", len(market))
b.metric("Rising", rising)
c.metric("Falling", falling)
d.metric(
    "Top 24h mover",
    f"{top['Symbol']} {top['24h change (%)']:+.2f}%"
)

st.caption(
    "Last market refresh (UTC): "
    + st.session_state.get("last_updated_utc", "Not yet refreshed")
)

st.subheader("🔥 Highest 24-hour percentage change")
st.dataframe(
    market.head(20),
    use_container_width=True,
    hide_index=True,
)

with st.expander(f"All contracts ({len(market)})"):
    st.dataframe(
        market,
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download full market list CSV",
    data=market.to_csv(index=False).encode("utf-8-sig"),
    file_name="futures_market_list.csv",
    mime="text/csv",
)

st.divider()
st.subheader("🤖 Smart BUY / SELL model")

if st.button("START MODEL SCAN"):
    if scan_count == "ALL CONTRACTS":
        selected_market = market
    elif scan_count == "TOP 100":
        selected_market = market.head(100)
    elif scan_count == "TOP 50":
        selected_market = market.head(50)
    else:
        selected_market = market.head(20)

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, row in enumerate(selected_market.to_dict("records")):
        symbol = row["Symbol"]
        status.write(
            f"Analyzing {symbol} ({i + 1}/{len(selected_market)})"
        )

        try:
            candles = get_candles(symbol, candle_count)
            prediction = analyze_model(candles)

            results.append({
                "Symbol": symbol,
                "24h change (%)": row["24h change (%)"],
                "Signal": prediction["Signal"],
                "BUY probability (%)": prediction["BUY probability (%)"],
                "SELL probability (%)": prediction["SELL probability (%)"],
                "Test accuracy (%)": prediction["Test accuracy (%)"],
                "Balanced accuracy (%)": prediction["Balanced accuracy (%)"],
                "Last price": row["Last price"],
                "Test samples": prediction["Test samples"],
                "_candles": candles,
                "_error": "",
            })

        except Exception as exc:
            results.append({
                "Symbol": symbol,
                "24h change (%)": row["24h change (%)"],
                "Signal": "DATA ERROR",
                "BUY probability (%)": np.nan,
                "SELL probability (%)": np.nan,
                "Test accuracy (%)": np.nan,
                "Balanced accuracy (%)": np.nan,
                "Last price": row["Last price"],
                "Test samples": 0,
                "_candles": None,
                "_error": str(exc),
            })

        progress.progress((i + 1) / len(selected_market))
        time.sleep(0.15)

    st.session_state["model_results"] = results
    status.success("Model scan completed.")


results = st.session_state["model_results"]

if results:
    errors = [r for r in results if r["Signal"] == "DATA ERROR"]
    good = [r for r in results if r["Signal"] in ("BUY", "SELL")]

    if errors:
        with st.expander(f"Data errors ({len(errors)})"):
            for row in errors:
                st.write(f"{row['Symbol']}: {row['_error']}")

    if good:
        result_df = pd.DataFrame(good).sort_values(
            "24h change (%)", ascending=False
        )

        filtered = result_df[
            (
                (result_df["BUY probability (%)"] >= min_probability)
                | (result_df["SELL probability (%)"] >= min_probability)
            )
            & (
                result_df["Balanced accuracy (%)"] >= min_accuracy
            )
        ].copy()

        if signal_filter != "ALL":
            filtered = filtered[
                filtered["Signal"] == signal_filter
            ]

        columns = [
            "Symbol",
            "24h change (%)",
            "Signal",
            "BUY probability (%)",
            "SELL probability (%)",
            "Test accuracy (%)",
            "Balanced accuracy (%)",
            "Last price",
            "Test samples",
        ]

        st.subheader("🏆 Model ranking — highest 24h change first")
        st.dataframe(
            filtered[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download model results CSV",
            data=filtered[columns].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="smart_futures_results.csv",
            mime="text/csv",
        )

        selected_symbol = st.selectbox(
            "Select a contract for the price chart",
            [r["Symbol"] for r in good],
        )

        selected = next(
            r for r in good if r["Symbol"] == selected_symbol
        )

        candles = selected["_candles"].set_index("datetime")

        st.line_chart(
            candles[["close"]],
            y="close",
            use_container_width=True,
        )

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Signal", selected["Signal"])
        x2.metric(
            "24h change",
            f"{selected['24h change (%)']:+.2f}%"
        )
        x3.metric(
            "BUY probability",
            f"{selected['BUY probability (%)']:.2f}%"
        )
        x4.metric(
            "SELL probability",
            f"{selected['SELL probability (%)']:.2f}%"
        )

st.divider()
st.caption(
    "Hourly refresh runs while the app is active. "
    "The model predicts direction, not guaranteed profit. "
    "This app uses OKX public data and does not place orders."
)
