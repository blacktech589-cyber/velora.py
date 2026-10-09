
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

st.set_page_config(
    page_title="Smart Futures AI Scanner",
    page_icon="📈",
    layout="wide",
)

BASE_URL = "https://www.okx.com"
TIMEOUT = 20
INTERVAL = "15m"

session = requests.Session()
session.headers.update({
    "User-Agent": "SmartFuturesScanner/2.0",
    "Accept": "application/json",
})


# ---------------- API ----------------

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

    df = df.sort_values(
        "24h change (%)",
        ascending=False,
    ).reset_index(drop=True)

    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


# ---------------- CANDLES ----------------

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

    latest = parse_candles(
        api_get(
            "/api/v5/market/candles",
            {
                "instId": symbol,
                "bar": INTERVAL,
                "limit": str(min(target, 300)),
            },
        )
    )

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
        raise RuntimeError("Not enough completed candles.")

    return result


# ---------------- MODEL ----------------

def make_features(df):
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 4, 8, 12, 24, 48, 96]:
        f[f"return_{lag}"] = close.pct_change(lag)

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

    returns = close.pct_change()

    for window in [4, 8, 16, 32, 64]:
        f[f"mean_return_{window}"] = returns.rolling(window).mean()
        f[f"volatility_{window}"] = returns.rolling(window).std()
        f[f"range_mean_{window}"] = f["range"].rolling(window).mean()

    for lag in [1, 2, 4, 8]:
        f[f"volume_change_{lag}"] = volume / volume.shift(lag) - 1

    f["relative_volume_24"] = volume / volume.rolling(24).mean() - 1
    f["relative_volume_48"] = volume / volume.rolling(48).mean() - 1

    return f.replace([np.inf, -np.inf], np.nan)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_model(X, y, epochs=400, learning_rate=0.04):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    for epoch in range(epochs):
        pred = sigmoid(X @ weights + bias)
        error = pred - y

        grad_w = X.T @ error / n + 0.005 * weights
        grad_b = error.mean()
        lr = learning_rate / (1 + epoch / 200)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def analyze_model(df):
    features = make_features(df)
    future_return = df["close"].shift(-4) / df["close"] - 1

    valid = features.notna().all(axis=1) & future_return.notna()
    X = features.loc[valid].to_numpy(dtype=float)
    y = (future_return.loc[valid].to_numpy() > 0).astype(float)

    if len(X) < 180:
        raise RuntimeError("Not enough training data.")

    split = int(len(X) * 0.8)
    split = max(60, min(split, len(X) - 30))

    X_train_raw = X[:split]
    X_test_raw = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1.0

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    weights, bias = train_model(X_train, y_train)

    test_pred = (
        sigmoid(X_test @ weights + bias) >= 0.5
    ).astype(float)

    accuracy = float((test_pred == y_test).mean() * 100)

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

    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1.0

    X_all = (X - mean_all) / std_all
    weights, bias = train_model(X_all, y)

    latest = features.iloc[[-1]].to_numpy(dtype=float)

    if not np.isfinite(latest).all():
        raise RuntimeError("Invalid latest candle features.")

    p_up = float(
        sigmoid(((latest - mean_all) / std_all) @ weights + bias)[0]
    )

    return {
        "Signal": "BUY" if p_up >= 0.5 else "SELL",
        "BUY probability (%)": round(p_up * 100, 2),
        "SELL probability (%)": round((1 - p_up) * 100, 2),
        "Test accuracy (%)": round(accuracy, 2),
        "Balanced accuracy (%)": round(balanced_accuracy, 2),
        "Test samples": len(X_test),
    }


# ---------------- MARKET REFRESH FRAGMENT ----------------

@st.fragment(run_every="1h")
def market_panel():
    st.subheader("Live Futures Market")

    min_volume = st.number_input(
        "Minimum 24h volume (contracts)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        key="min_volume",
    )

    if st.button("Refresh market now", key="refresh_market"):
        get_instruments.clear()
        get_tickers.clear()

    try:
        symbols = get_instruments()
        tickers = get_tickers()
        market = build_market_table(symbols, tickers, min_volume)

        if market.empty:
            st.warning("No contracts returned by the API.")
            return

        top = market.iloc[0]
        rising = int((market["24h change (%)"] > 0).sum())
        falling = int((market["24h change (%)"] < 0).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("USDT perpetual contracts", len(market))
        c2.metric("Rising", rising)
        c3.metric("Falling", falling)
        c4.metric(
            "Top 24h mover",
            f"{top['Symbol']} {top['24h change (%)']:+.2f}%"
        )

        st.caption(
            "Last market refresh (UTC): "
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )

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
            "Download market CSV",
            data=market.to_csv(index=False).encode("utf-8-sig"),
            file_name="futures_market.csv",
            mime="text/csv",
            key="download_market",
        )

    except Exception as exc:
        st.error("Market data request failed.")
        st.code(str(exc))


# ---------------- MAIN APP ----------------

st.title("📈 Smart Futures AI Scanner")
st.caption(
    "OKX public USDT perpetual data | Hourly market refresh | BUY / SELL"
)

market_panel()

st.divider()
st.subheader("🤖 BUY / SELL Model Scan")

scan_count = st.selectbox(
    "How many contracts should the model analyze?",
    ["TOP 20", "TOP 50", "TOP 100", "ALL CONTRACTS"],
)

candle_count = st.selectbox(
    "Historical 15-minute candles per contract",
    [200, 300, 500, 800, 1000],
    index=2,
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

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []


if st.button("START MODEL SCAN"):
    try:
        with st.spinner("Loading market list..."):
            symbols = get_instruments()
            tickers = get_tickers()
            market = build_market_table(symbols, tickers, 0)

        if market.empty:
            st.error("No market contracts found.")
            st.stop()

        if scan_count == "TOP 20":
            selected_market = market.head(20)
        elif scan_count == "TOP 50":
            selected_market = market.head(50)
        elif scan_count == "TOP 100":
            selected_market = market.head(100)
        else:
            selected_market = market

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

    except Exception as exc:
        st.error("Could not start model scan.")
        st.code(str(exc))


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
            filtered = filtered[filtered["Signal"] == signal_filter]

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

        st.subheader("Model results — highest 24h change first")
        st.dataframe(
            filtered[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download model results CSV",
            data=filtered[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="smart_futures_results.csv",
            mime="text/csv",
        )

        chart_symbol = st.selectbox(
            "Select contract for chart",
            [r["Symbol"] for r in good],
        )

        selected = next(
            r for r in good if r["Symbol"] == chart_symbol
        )

        candles = selected["_candles"].set_index("datetime")
        st.line_chart(
            candles[["close"]],
            y="close",
            use_container_width=True,
        )

        a, b, c, d = st.columns(4)
        a.metric("Signal", selected["Signal"])
        b.metric("24h change", f"{selected['24h change (%)']:+.2f}%")
        c.metric("BUY probability", f"{selected['BUY probability (%)']:.2f}%")
        d.metric("SELL probability", f"{selected['SELL probability (%)']:.2f}%")

st.divider()
st.caption(
    "Market data refreshes every hour while the app is active. "
    "Model predictions do not automatically rerun every hour. "
    "This uses OKX data, not Binance data, and places no orders."
)
