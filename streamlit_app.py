
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ============================================================
# CRYPTO FUTURES AI SCANNER
# Public OKX market data only; no API keys or order placement.
# 15-minute candles, logistic regression using NumPy.
# No RSI, EMA, MACD or Bollinger Bands.
# No Plotly dependency.
# ============================================================

st.set_page_config(
    page_title="Crypto Futures AI Scanner",
    page_icon="📊",
    layout="wide",
)

BASE_URL = "https://tr.okx.com"
TIMEFRAME = "15m"
TIMEOUT = 20

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoScanner/1.0",
    "Accept": "application/json",
})


# ---------------- API ----------------

def api_get(path, params=None):
    response = SESSION.get(
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

    instruments = []

    for item in data:
        if item.get("state") != "live":
            continue
        if item.get("settleCcy") != "USDT":
            continue
        if item.get("ctType") != "linear":
            continue

        instruments.append(item["instId"])

    return instruments


@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    return api_get(
        "/api/v5/market/tickers",
        {"instType": "SWAP"},
    )


def get_candles(inst_id, target=1000):
    rows = []
    timestamps = set()
    after = None

    # History endpoint pages contain up to 100 candles.
    for _ in range(15):
        params = {
            "instId": inst_id,
            "bar": TIMEFRAME,
            "limit": "100",
        }

        if after is not None:
            params["after"] = str(after)

        page = api_get(
            "/api/v5/market/history-candles",
            params,
        )

        if not page:
            break

        added = 0

        for row in page:
            ts = int(row[0])
            if ts not in timestamps:
                timestamps.add(ts)
                rows.append(row)
                added += 1

        if len(rows) >= target or added == 0:
            break

        oldest = min(int(row[0]) for row in page)

        if after == oldest:
            break

        after = oldest
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(
            f"No candles returned for {inst_id}"
        )

    rows.sort(key=lambda row: int(row[0]))

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "volume_currency",
            "volume_quote", "confirm",
        ],
    )

    numeric_columns = [
        "timestamp", "open", "high", "low",
        "close", "volume", "volume_quote",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Only use completed candles.
    df = df[df["confirm"].astype(str) == "1"].copy()
    df = df.dropna(
        subset=[
            "timestamp", "open", "high",
            "low", "close", "volume",
        ]
    )
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").tail(target)
    df = df.reset_index(drop=True)

    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    if len(df) < 150:
        raise RuntimeError(
            f"Not enough completed candles: {len(df)}"
        )

    return df


# ---------------- FEATURES ----------------

def build_features(df):
    """
    Uses price changes, candle shape and volume changes.
    Does not use RSI, EMA, MACD or Bollinger Bands.
    """
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    features = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 4, 8, 12, 24, 48]:
        features[f"return_{lag}"] = close.pct_change(lag)

    features["body_return"] = (
        df["close"] - df["open"]
    ) / open_price

    features["high_low_range"] = (
        high - low
    ) / close

    candle_range = (high - low).replace(0, np.nan)

    features["close_position"] = (
        df["close"] - low
    ) / candle_range

    features["upper_wick"] = (
        high - df[["open", "close"]].max(axis=1)
    ) / close

    features["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - low
    ) / close

    one_return = close.pct_change()

    for window in [4, 8, 16, 32]:
        features[f"mean_return_{window}"] = (
            one_return.rolling(window).mean()
        )

        features[f"volatility_{window}"] = (
            one_return.rolling(window).std()
        )

        features[f"range_mean_{window}"] = (
            features["high_low_range"]
            .rolling(window).mean()
        )

    for lag in [1, 2, 4, 8]:
        features[f"volume_change_{lag}"] = (
            volume / volume.shift(lag) - 1
        )

    features["volume_position"] = (
        volume / volume.rolling(24).mean() - 1
    )

    return features.replace(
        [np.inf, -np.inf], np.nan
    )


# ---------------- MODEL ----------------

def sigmoid(z):
    return 1.0 / (
        1.0 + np.exp(-np.clip(z, -35, 35))
    )


def train_logistic(X, y, epochs=400, learning_rate=0.06):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        predictions = sigmoid(X @ weights + bias)
        errors = predictions - y

        grad_w = (
            X.T @ errors
        ) / n_samples + 0.001 * weights

        grad_b = errors.mean()

        lr = learning_rate / (
            1.0 + epoch / 150.0
        )

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def evaluate_and_predict(df):
    features = build_features(df)

    # Predict direction four 15-minute candles ahead.
    future_return = (
        df["close"].shift(-4) / df["close"] - 1
    )

    valid = features.notna().all(axis=1)
    valid &= future_return.notna()

    X = features.loc[valid].to_numpy(dtype=float)
    y = (
        future_return.loc[valid].to_numpy() > 0
    ).astype(float)

    if len(X) < 150:
        raise RuntimeError(
            "Insufficient valid samples for model."
        )

    split = int(len(X) * 0.8)
    split = max(50, min(split, len(X) - 30))

    X_train_raw = X[:split]
    y_train = y[:split]
    X_test_raw = X[split:]
    y_test = y[split:]

    # Fit scaling on training data only for evaluation.
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1.0

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    weights, bias = train_logistic(X_train, y_train)

    test_probabilities = sigmoid(
        X_test @ weights + bias
    )
    test_predictions = (
        test_probabilities >= 0.5
    ).astype(float)

    accuracy = float(
        np.mean(test_predictions == y_test) * 100
    )

    # Retrain on all available labeled samples.
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1.0

    X_all = (X - mean_all) / std_all
    final_weights, final_bias = train_logistic(X_all, y)

    # Latest feature row may not have a future label yet.
    latest_features = features.iloc[[-1]].to_numpy(
        dtype=float
    )

    if not np.isfinite(latest_features).all():
        raise RuntimeError(
            "Latest candle features contain invalid values."
        )

    latest_scaled = (
        latest_features - mean_all
    ) / std_all

    p_up = float(
        sigmoid(
            latest_scaled @ final_weights + final_bias
        )[0]
    )

    return {
        "p_up": p_up,
        "p_down": 1.0 - p_up,
        "accuracy": accuracy,
        "samples": len(X),
        "test_samples": len(X_test),
    }


# ---------------- SCANNER ----------------

def make_universe(instruments, tickers, min_volume):
    ticker_map = {
        row.get("instId"): row
        for row in tickers
    }

    universe = []

    for inst_id in instruments:
        ticker = ticker_map.get(inst_id)
        if not ticker:
            continue

        try:
            price = float(ticker.get("last", 0))
            volume = float(ticker.get("volCcy24h", 0))
        except (TypeError, ValueError):
            continue

        if price <= 0 or volume < min_volume:
            continue

        universe.append({
            "instId": inst_id,
            "price": price,
            "volume": volume,
        })

    universe.sort(
        key=lambda row: row["volume"],
        reverse=True,
    )

    return universe


def scan(universe, candle_count, progress, status):
    results = []

    for index, coin in enumerate(universe):
        inst_id = coin["instId"]

        status.write(
            f"Analiz ediliyor: {inst_id} "
            f"({index + 1}/{len(universe)})"
        )

        try:
            candles = get_candles(
                inst_id, candle_count
            )
            model = evaluate_and_predict(candles)

            p_up = model["p_up"]
            signal = "AL" if p_up >= 0.5 else "SAT"

            results.append({
                "Sözleşme": inst_id,
                "Sinyal": signal,
                "AL olasılığı (%)": round(p_up * 100, 2),
                "SAT olasılığı (%)": round(
                    model["p_down"] * 100, 2
                ),
                "Test doğruluğu (%)": round(
                    model["accuracy"], 2
                ),
                "Son fiyat": coin["price"],
                "24s hacim": coin["volume"],
                "Mum sayısı": len(candles),
                "Test örneği": model["test_samples"],
                "_candles": candles,
                "_error": "",
            })

        except Exception as exc:
            results.append({
                "Sözleşme": inst_id,
                "Sinyal": "VERİ HATASI",
                "AL olasılığı (%)": np.nan,
                "SAT olasılığı (%)": np.nan,
                "Test doğruluğu (%)": np.nan,
                "Son fiyat": coin["price"],
                "24s hacim": coin["volume"],
                "Mum sayısı": 0,
                "Test örneği": 0,
                "_candles": None,
                "_error": str(exc),
            })

        progress.progress(
            (index + 1) / len(universe)
        )
        time.sleep(0.15)

    return results


# ---------------- STREAMLIT UI ----------------

st.title("📊 Crypto Futures AI Scanner")
st.caption(
    "OKX TR public market data | USDT perpetual swaps | "
    "15-minute candles | No API keys | No automated orders"
)

with st.sidebar:
    st.header("Tarama ayarları")

    max_coins = st.slider(
        "Taranacak sözleşme sayısı",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    candle_count = st.select_slider(
        "Geçmiş mum sayısı",
        options=[200, 400, 600, 800, 1000],
        value=1000,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0.0,
        value=100000.0,
        step=100000.0,
    )

    min_probability = st.slider(
        "Minimum olasılık (%)",
        min_value=50,
        max_value=95,
        value=60,
    )

    min_accuracy = st.slider(
        "Minimum test doğruluğu (%)",
        min_value=0,
        max_value=90,
        value=50,
    )

    signal_filter = st.selectbox(
        "Sinyal",
        ["Tümü", "AL", "SAT"],
    )

    run_scan = st.button(
        "🚀 Piyasayı tara",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Tahminler garanti değildir. Uygulama emir göndermez."
    )


if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []


if run_scan:
    st.session_state["scan_results"] = []

    try:
        with st.spinner("Sözleşmeler yükleniyor..."):
            instruments = get_instruments()
            tickers = get_tickers()

            universe = make_universe(
                instruments,
                tickers,
                min_volume,
            )

        if not universe:
            st.error(
                "Uygun sözleşme bulunamadı. "
                "Minimum hacmi azaltmayı dene."
            )
            st.stop()

        universe = universe[:max_coins]

        st.info(
            f"{len(universe)} sözleşme taranacak."
        )

        progress = st.progress(0)
        status = st.empty()

        st.session_state["scan_results"] = scan(
            universe,
            candle_count,
            progress,
            status,
        )

        status.success("Tarama tamamlandı.")

    except Exception as exc:
        st.error(
            "Piyasa verisi alınamadı. API erişimini "
            "ve uygulama günlüklerini kontrol et."
        )
        st.code(str(exc))


results = st.session_state["scan_results"]

if results:
    successful = [
        row for row in results
        if row["Sinyal"] in ("AL", "SAT")
    ]

    failed = [
        row for row in results
        if row["Sinyal"] == "VERİ HATASI"
    ]

    if failed:
        with st.expander(
            f"Veri hataları ({len(failed)})"
        ):
            for row in failed:
                st.write(
                    f"**{row['Sözleşme']}**: "
                    f"{row['_error']}"
                )

    if successful:
        df = pd.DataFrame(successful)

        filtered = df[
            (
                (df["AL olasılığı (%)"] >= min_probability)
                | (df["SAT olasılığı (%)"] >= min_probability)
            )
            & (
                df["Test doğruluğu (%)"] >= min_accuracy
            )
        ].copy()

        if signal_filter != "Tümü":
            filtered = filtered[
                filtered["Sinyal"] == signal_filter
            ]

        filtered = filtered.sort_values(
            by=[
                "AL olasılığı (%)",
                "SAT olasılığı (%)",
            ],
            ascending=False,
        )

        buys = sum(
            row["Sinyal"] == "AL"
            for row in successful
        )
        sells = sum(
            row["Sinyal"] == "SAT"
            for row in successful
        )
        avg_accuracy = float(
            df["Test doğruluğu (%)"].mean()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Analiz edilen", len(successful))
        col2.metric("AL", buys)
        col3.metric("SAT", sells)
        col4.metric(
            "Ortalama test doğruluğu",
            f"{avg_accuracy:.2f}%",
        )

        st.subheader("🏆 Sinyal sıralaması")

        columns = [
            "Sözleşme",
            "Sinyal",
            "AL olasılığı (%)",
            "SAT olasılığı (%)",
            "Test doğruluğu (%)",
            "Son fiyat",
            "24s hacim",
            "Mum sayısı",
            "Test örneği",
        ]

        st.dataframe(
            filtered[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 CSV indir",
            data=filtered[columns].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="crypto_futures_scan.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("📈 Mum grafiği")

        symbols = [
            row["Sözleşme"]
            for row in successful
        ]

        selected_symbol = st.selectbox(
            "Sözleşme seç",
            symbols,
        )

        selected = next(
            row for row in successful
            if row["Sözleşme"] == selected_symbol
        )

        candles = selected["_candles"].copy()
        candles = candles.set_index("datetime")

        st.caption(
            f"{selected_symbol} — son "
            f"{len(candles)} tamamlanmış 15 dakikalık mum"
        )

        # Streamlit built-in chart: no Plotly dependency.
        st.line_chart(
            candles[["close"]],
            y="close",
            use_container_width=True,
        )

        st.subheader("Seçili sözleşme")

        a, b, c, d = st.columns(4)
        a.metric("Sinyal", selected["Sinyal"])
        b.metric(
            "AL olasılığı",
            f"{selected['AL olasılığı (%)']:.2f}%",
        )
        c.metric(
            "SAT olasılığı",
            f"{selected['SAT olasılığı (%)']:.2f}%",
        )
        d.metric(
            "Test doğruluğu",
            f"{selected['Test doğruluğu (%)']:.2f}%",
        )

        st.caption(
            "Model, dört mum sonraki fiyat yönünü tahmin etmeye çalışır. "
            "Test doğruluğu gelecekteki başarıyı veya kârlılığı garanti etmez."
        )

else:
    st.info(
        "Tarama başlatılmadı. Soldan ayarları seçip "
        "'Piyasayı tara' düğmesine bas."
    )


st.divider()
st.caption(
    "Bilgilendirme amaçlıdır. Otomatik emir göndermez. "
    "İşlem ücretleri, fonlama, kayma ve likidasyon model testine dahil değildir."
)
