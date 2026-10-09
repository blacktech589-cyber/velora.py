
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ==========================================================
# FUTURES CRYPTO SCANNER
# Data source: OKX public USDT perpetual swap market data
# Signals: BUY / SELL
# Ranking: 24-hour percentage change, highest first
# No API keys, no orders, no Plotly
# ==========================================================

st.set_page_config(
    page_title="Futures Crypto Scanner",
    page_icon="📈",
    layout="wide",
)

BASE_URL = "https://www.okx.com"
TIMEOUT = 20

session = requests.Session()
session.headers.update({
    "User-Agent": "PublicFuturesScanner/1.0",
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


@st.cache_data(ttl=60, show_spinner=False)
def get_instruments():
    data = api_get(
        "/api/v5/public/instruments",
        {"instType": "SWAP"},
    )

    symbols = []

    for item in data:
        if (
            item.get("state") == "live"
            and item.get("settleCcy") == "USDT"
            and item.get("ctType") == "linear"
        ):
            symbols.append(item["instId"])

    return symbols


@st.cache_data(ttl=20, show_spinner=False)
def get_tickers():
    return api_get(
        "/api/v5/market/tickers",
        {"instType": "SWAP"},
    )


def build_market_table(symbols, tickers, min_volume):
    symbol_set = set(symbols)
    rows = []

    for ticker in tickers:
        symbol = ticker.get("instId")

        if symbol not in symbol_set:
            continue

        try:
            last = float(ticker.get("last", 0))
            open24 = float(ticker.get("open24h", 0))
            high24 = float(ticker.get("high24h", 0))
            low24 = float(ticker.get("low24h", 0))
            volume = float(ticker.get("vol24h", 0))
        except (TypeError, ValueError):
            continue

        if last <= 0 or open24 <= 0:
            continue

        change = (last / open24 - 1) * 100

        rows.append({
            "Sembol": symbol,
            "24s değişim (%)": change,
            "Son fiyat": last,
            "24s açılış": open24,
            "24s yüksek": high24,
            "24s düşük": low24,
            "24s hacim (sözleşme)": volume,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df[
        df["24s hacim (sözleşme)"] >= min_volume
    ].copy()

    # Highest 24-hour percentage change first.
    df = df.sort_values(
        "24s değişim (%)",
        ascending=False,
    ).reset_index(drop=True)

    df.insert(0, "Sıra", np.arange(1, len(df) + 1))
    return df


# ---------------- CANDLE DATA ----------------

def parse_candles(data):
    if not data:
        raise RuntimeError("Mum verisi bulunamadı.")

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
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df[df["confirm"].astype(str) == "1"].copy()
    df = df.dropna(
        subset=[
            "timestamp", "open", "high",
            "low", "close", "volume",
        ]
    )
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    return df


def get_candles(symbol, target=1000):
    frames = []
    timestamps = set()

    # Newest candles.
    latest_limit = min(target, 300)

    latest_data = api_get(
        "/api/v5/market/candles",
        {
            "instId": symbol,
            "bar": "15m",
            "limit": str(latest_limit),
        },
    )

    latest = parse_candles(latest_data)
    frames.append(latest)

    timestamps.update(
        latest["timestamp"].astype(int).tolist()
    )

    before = int(latest["timestamp"].min())

    # Older candles in pages.
    for _ in range(15):
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates("timestamp")

        if len(combined) >= target:
            break

        page = api_get(
            "/api/v5/market/history-candles",
            {
                "instId": symbol,
                "bar": "15m",
                "limit": "100",
                "after": str(before),
            },
        )

        older = parse_candles(page)

        if older.empty:
            break

        new_rows = older[
            ~older["timestamp"].astype(int).isin(timestamps)
        ].copy()

        if new_rows.empty:
            break

        timestamps.update(
            new_rows["timestamp"].astype(int).tolist()
        )

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
            f"Yeterli tamamlanmış mum yok: {len(result)}"
        )

    return result


# ---------------- MODEL FEATURES ----------------

def make_features(df):
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    # Price changes. No RSI, EMA, MACD or Bollinger Bands.
    for lag in [1, 2, 3, 4, 8, 12, 24, 48]:
        f[f"return_{lag}"] = close.pct_change(lag)

    f["body"] = (
        df["close"] - df["open"]
    ) / open_price

    f["range"] = (
        high - low
    ) / close

    candle_range = (high - low).replace(0, np.nan)

    f["close_position"] = (
        df["close"] - low
    ) / candle_range

    f["upper_wick"] = (
        high - df[["open", "close"]].max(axis=1)
    ) / close

    f["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - low
    ) / close

    one_return = close.pct_change()

    for window in [4, 8, 16, 32]:
        f[f"mean_return_{window}"] = (
            one_return.rolling(window).mean()
        )
        f[f"volatility_{window}"] = (
            one_return.rolling(window).std()
        )
        f[f"range_mean_{window}"] = (
            f["range"].rolling(window).mean()
        )

    for lag in [1, 2, 4, 8]:
        f[f"volume_change_{lag}"] = (
            volume / volume.shift(lag) - 1
        )

    f["relative_volume"] = (
        volume / volume.rolling(24).mean() - 1
    )

    return f.replace([np.inf, -np.inf], np.nan)


# ---------------- LOGISTIC REGRESSION ----------------

def sigmoid(z):
    return 1.0 / (
        1.0 + np.exp(-np.clip(z, -35, 35))
    )


def train_model(X, y, epochs=350, learning_rate=0.06):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    for epoch in range(epochs):
        prediction = sigmoid(X @ weights + bias)
        error = prediction - y

        grad_w = (
            X.T @ error
        ) / n + 0.001 * weights

        grad_b = error.mean()
        lr = learning_rate / (1 + epoch / 150)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def analyze_model(df):
    features = make_features(df)

    # Direction of close four 15-minute candles ahead.
    future_return = (
        df["close"].shift(-4) / df["close"] - 1
    )

    valid = (
        features.notna().all(axis=1)
        & future_return.notna()
    )

    X = features.loc[valid].to_numpy(dtype=float)
    y = (
        future_return.loc[valid].to_numpy() > 0
    ).astype(float)

    if len(X) < 150:
        raise RuntimeError(
            "Model için yeterli geçmiş veri yok."
        )

    split = int(len(X) * 0.8)
    split = max(50, min(split, len(X) - 30))

    X_train_raw = X[:split]
    X_test_raw = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # Chronological holdout test.
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1.0

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    weights, bias = train_model(X_train, y_train)

    test_predictions = (
        sigmoid(X_test @ weights + bias) >= 0.5
    ).astype(float)

    accuracy = float(
        (test_predictions == y_test).mean() * 100
    )

    # Retrain on all labeled data for the latest prediction.
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1.0

    X_all = (X - mean_all) / std_all
    weights, bias = train_model(X_all, y)

    latest = features.iloc[[-1]].to_numpy(dtype=float)

    if not np.isfinite(latest).all():
        raise RuntimeError(
            "Son mum özellikleri geçersiz."
        )

    latest_scaled = (latest - mean_all) / std_all

    p_up = float(
        sigmoid(latest_scaled @ weights + bias)[0]
    )

    # Corrected English signal names.
    signal = "BUY" if p_up >= 0.5 else "SELL"

    return {
        "Signal": signal,
        "BUY probability (%)": round(p_up * 100, 2),
        "SELL probability (%)": round((1 - p_up) * 100, 2),
        "Test accuracy (%)": round(accuracy, 2),
        "Test samples": len(X_test),
    }


# ---------------- STREAMLIT UI ----------------

st.title("📈 Futures Crypto Scanner")
st.caption(
    "Data source: OKX public USDT perpetual swaps | "
    "Sorted by 24-hour percentage change, highest first"
)

with st.sidebar:
    st.header("Scanner settings")

    min_volume = st.number_input(
        "Minimum 24h volume (contracts)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

    candle_count = st.selectbox(
        "Historical 15-minute candles",
        [200, 300, 500, 800, 1000],
        index=4,
    )

    scan_count = st.selectbox(
        "Model scan",
        ["ALL CONTRACTS", "TOP 100", "TOP 50", "TOP 20"],
    )

    min_probability = st.slider(
        "Minimum BUY/SELL probability (%)",
        50, 95, 60,
    )

    min_accuracy = st.slider(
        "Minimum test accuracy (%)",
        0, 90, 50,
    )

    signal_filter = st.selectbox(
        "Signal filter",
        ["ALL", "BUY", "SELL"],
    )

    refresh = st.button(
        "🔄 Refresh market",
        type="primary",
        use_container_width=True,
    )


if "market_table" not in st.session_state:
    st.session_state["market_table"] = None

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []


if refresh or st.session_state["market_table"] is None:
    try:
        with st.spinner("Loading public futures market data..."):
            symbols = get_instruments()
            tickers = get_tickers()

            market = build_market_table(
                symbols,
                tickers,
                min_volume,
            )

            if market.empty:
                st.error(
                    "No matching contracts. Check API access and filters."
                )
                st.stop()

            st.session_state["market_table"] = market
            st.session_state["model_results"] = []

    except Exception as exc:
        st.error("Could not retrieve market data.")
        st.code(str(exc))
        st.info(
            "This version uses OKX data, not Binance data. "
            "Check whether the public API is accessible from your environment."
        )
        st.stop()


market = st.session_state["market_table"].copy()

# Always sort highest 24h percentage change first.
market = market.sort_values(
    "24s değişim (%)",
    ascending=False,
).reset_index(drop=True)

market["Sıra"] = np.arange(1, len(market) + 1)

top = market.iloc[0]
positive_count = int((market["24s değişim (%)"] > 0).sum())
negative_count = int((market["24s değişim (%)"] < 0).sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("USDT perpetual contracts", len(market))
c2.metric("Rising", positive_count)
c3.metric("Falling", negative_count)
c4.metric(
    "Highest 24h change",
    f"{top['Sembol']} {top['24s değişim (%)']:+.2f}%",
)

st.subheader("🔥 Highest percentage change")

st.dataframe(
    market.head(20),
    use_container_width=True,
    hide_index=True,
)

with st.expander(
    f"ALL CONTRACTS ({len(market)})",
    expanded=True,
):
    st.dataframe(
        market,
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "📥 Download all contracts CSV",
    data=market.to_csv(index=False).encode("utf-8-sig"),
    file_name="usdt_perpetual_market.csv",
    mime="text/csv",
)

st.divider()
st.subheader("🤖 BUY / SELL model scan")

if st.button("🚀 START MODEL SCAN"):
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
        symbol = row["Sembol"]
        status.write(
            f"Analyzing {symbol} ({i + 1}/{len(selected_market)})"
        )

        try:
            candles = get_candles(
                symbol,
                candle_count,
            )

            prediction = analyze_model(candles)

            results.append({
                "Symbol": symbol,
                "24h change (%)": row["24s değişim (%)"],
                "Signal": prediction["Signal"],
                "BUY probability (%)": prediction["BUY probability (%)"],
                "SELL probability (%)": prediction["SELL probability (%)"],
                "Test accuracy (%)": prediction["Test accuracy (%)"],
                "Last price": row["Son fiyat"],
                "Test samples": prediction["Test samples"],
                "_candles": candles,
                "_error": "",
            })

        except Exception as exc:
            results.append({
                "Symbol": symbol,
                "24h change (%)": row["24s değişim (%)"],
                "Signal": "DATA ERROR",
                "BUY probability (%)": np.nan,
                "SELL probability (%)": np.nan,
                "Test accuracy (%)": np.nan,
                "Last price": row["Son fiyat"],
                "Test samples": 0,
                "_candles": None,
                "_error": str(exc),
            })

        progress.progress(
            (i + 1) / len(selected_market)
        )
        time.sleep(0.12)

    st.session_state["model_results"] = results
    status.success("Model scan completed.")


results = st.session_state["model_results"]

if results:
    good = [
        row for row in results
        if row["Signal"] in ("BUY", "SELL")
    ]

    errors = [
        row for row in results
        if row["Signal"] == "DATA ERROR"
    ]

    if errors:
        with st.expander(f"Data errors ({len(errors)})"):
            for row in errors:
                st.write(
                    f"{row['Symbol']}: {row['_error']}"
                )

    if good:
        result_df = pd.DataFrame(good)

        # Highest 24h percentage change remains first.
        result_df = result_df.sort_values(
            "24h change (%)",
            ascending=False,
        )

        filtered = result_df[
            (
                (result_df["BUY probability (%)"] >= min_probability)
                | (result_df["SELL probability (%)"] >= min_probability)
            )
            & (
                result_df["Test accuracy (%)"] >= min_accuracy
            )
        ].copy()

        if signal_filter != "ALL":
            filtered = filtered[
                filtered["Signal"] == signal_filter
            ]

        st.subheader("🏆 Model results — highest percentage first")

        display_columns = [
            "Symbol",
            "24h change (%)",
            "Signal",
            "BUY probability (%)",
            "SELL probability (%)",
            "Test accuracy (%)",
            "Last price",
            "Test samples",
        ]

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download model results CSV",
            data=filtered[display_columns].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="futures_model_results.csv",
            mime="text/csv",
        )

        chart_symbol = st.selectbox(
            "Select contract for chart",
            [row["Symbol"] for row in good],
        )

        selected = next(
            row for row in good
            if row["Symbol"] == chart_symbol
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
            f"{selected['24h change (%)']:+.2f}%",
        )
        x3.metric(
            "BUY probability",
            f"{selected['BUY probability (%)']:.2f}%",
        )
        x4.metric(
            "SELL probability",
            f"{selected['SELL probability (%)']:.2f}%",
        )

st.divider()
st.caption(
    "Public market data only. No orders are placed. "
    "BUY/SELL labels are model predictions, not guarantees of profit. "
    "This version uses OKX data and is not a Binance Futures feed."
)
