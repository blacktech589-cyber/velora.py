
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# =========================================================
# BINANCE USD-M FUTURES CRYPTO SCANNER
# All active USDT perpetual contracts
# Sorted by 24-hour percentage change, highest first
# Public market data only - no API keys or orders
# No Plotly, RSI, EMA, MACD or Bollinger Bands
# =========================================================

st.set_page_config(
    page_title="Binance Futures Crypto Scanner",
    page_icon="📈",
    layout="wide"
)

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
]
TIMEOUT = 15
INTERVAL = "15m"

session = requests.Session()
session.headers.update({"User-Agent": "FuturesScanner/1.0"})


def api_get(path, params=None):
    errors = []

    for base_url in BASE_URLS:
        try:
            response = session.get(
                base_url + path,
                params=params,
                timeout=TIMEOUT
            )

            if response.status_code == 451:
                raise RuntimeError(
                    "Binance API HTTP 451: bölgesel erişim kısıtlaması."
                )

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            errors.append(f"{base_url}: {exc}")

            # A regional restriction should not be bypassed.
            if "HTTP 451" in str(exc):
                raise RuntimeError(str(exc)) from exc

    raise RuntimeError("\n".join(errors))


@st.cache_data(ttl=60, show_spinner=False)
def get_exchange_symbols():
    info = api_get("/fapi/v1/exchangeInfo")
    symbols = []

    for item in info.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("marginAsset") == "USDT"
        ):
            symbols.append(item["symbol"])

    return symbols


@st.cache_data(ttl=20, show_spinner=False)
def get_all_tickers():
    return api_get("/fapi/v1/ticker/24hr")


@st.cache_data(ttl=10, show_spinner=False)
def get_klines(symbol, interval="15m", limit=1000):
    raw = api_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
    )

    if not raw:
        raise RuntimeError("Mum verisi alınamadı.")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])

    for col in [
        "open", "high", "low", "close",
        "volume", "quote_volume"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_numeric(
        df["open_time"], errors="coerce"
    )

    df["datetime"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    # Remove the current, unfinished candle.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms].copy()

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df.tail(limit).reset_index(drop=True)


def make_features(df):
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 4, 8, 12, 24, 48]:
        f[f"return_{lag}"] = close.pct_change(lag)

    f["candle_body"] = (df["close"] - df["open"]) / open_price
    f["candle_range"] = (high - low) / close

    candle_range = (high - low).replace(0, np.nan)
    f["close_position"] = (df["close"] - low) / candle_range

    f["upper_wick"] = (
        high - df[["open", "close"]].max(axis=1)
    ) / close

    f["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - low
    ) / close

    one_return = close.pct_change()

    for window in [4, 8, 16, 32]:
        f[f"mean_return_{window}"] = one_return.rolling(window).mean()
        f[f"volatility_{window}"] = one_return.rolling(window).std()
        f[f"range_mean_{window}"] = f["candle_range"].rolling(window).mean()

    for lag in [1, 2, 4, 8]:
        f[f"volume_change_{lag}"] = volume / volume.shift(lag) - 1

    f["relative_volume"] = volume / volume.rolling(24).mean() - 1

    return f.replace([np.inf, -np.inf], np.nan)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_model(X, y, epochs=350, learning_rate=0.06):
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0

    for epoch in range(epochs):
        prediction = sigmoid(X @ w + b)
        error = prediction - y
        grad_w = (X.T @ error) / n + 0.001 * w
        grad_b = error.mean()
        lr = learning_rate / (1 + epoch / 150)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def model_signal(df):
    features = make_features(df)

    # Direction of the close four candles ahead.
    future_return = df["close"].shift(-4) / df["close"] - 1

    valid = features.notna().all(axis=1) & future_return.notna()
    X = features.loc[valid].to_numpy(dtype=float)
    y = (future_return.loc[valid].to_numpy() > 0).astype(float)

    if len(X) < 150:
        raise RuntimeError("Model için yeterli geçmiş veri yok.")

    split = int(len(X) * 0.8)
    split = max(50, min(split, len(X) - 30))

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Chronological holdout test.
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    w, b = train_model(X_train, y_train)
    test_prediction = (sigmoid(X_test @ w + b) >= 0.5).astype(float)
    accuracy = float((test_prediction == y_test).mean() * 100)

    # Retrain using all labeled historical samples.
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1

    X_all = (X - mean_all) / std_all
    w, b = train_model(X_all, y)

    # Latest candle has no future label yet.
    latest = features.iloc[[-1]].to_numpy(dtype=float)

    if not np.isfinite(latest).all():
        raise RuntimeError("Son mum özellikleri geçersiz.")

    p_up = float(sigmoid(((latest - mean_all) / std_all) @ w + b)[0])

    return {
        "signal": "AL" if p_up >= 0.5 else "SAT",
        "p_up": p_up * 100,
        "p_down": (1 - p_up) * 100,
        "accuracy": accuracy,
        "test_samples": len(X_test),
    }


def build_market_table(symbols, tickers):
    symbol_set = set(symbols)
    rows = []

    for item in tickers:
        symbol = item.get("symbol")

        if symbol not in symbol_set:
            continue

        try:
            price = float(item["lastPrice"])
            change = float(item["priceChangePercent"])
            volume = float(item["quoteVolume"])
            high = float(item["highPrice"])
            low = float(item["lowPrice"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append({
            "Sıra": 0,
            "Sembol": symbol,
            "24s değişim (%)": change,
            "Son fiyat": price,
            "24s hacim (USDT)": volume,
            "24s yüksek": high,
            "24s düşük": low,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            "24s değişim (%)",
            ascending=False
        ).reset_index(drop=True)

        df["Sıra"] = np.arange(1, len(df) + 1)

    return df


# ======================= PANEL =======================

st.title("📈 Binance Futures Crypto Scanner")
st.caption(
    "Binance USDⓈ-M Futures | USDT perpetual contracts | "
    "24 saatlik değişime göre sıralama"
)

with st.sidebar:
    st.header("Tarama ayarları")

    scan_count_option = st.selectbox(
        "Model taraması",
        ["Tüm kriptolar", "İlk 100", "İlk 50", "İlk 20"],
        index=0
    )

    candle_count = st.selectbox(
        "15 dakikalık geçmiş mum",
        [200, 500, 1000],
        index=2
    )

    minimum_volume = st.number_input(
        "Minimum 24s hacim (USDT)",
        min_value=0.0,
        value=100000.0,
        step=100000.0
    )

    minimum_probability = st.slider(
        "Model sinyali minimum olasılığı (%)",
        50, 95, 60
    )

    minimum_accuracy = st.slider(
        "Minimum geçmiş test doğruluğu (%)",
        0, 90, 50
    )

    signal_filter = st.selectbox(
        "Model sinyali filtresi",
        ["Tümü", "AL", "SAT"]
    )

    refresh = st.button(
        "🔄 Piyasayı yenile",
        type="primary",
        use_container_width=True
    )

if "market_df" not in st.session_state:
    st.session_state["market_df"] = None

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []

if refresh or st.session_state["market_df"] is None:
    try:
        with st.spinner("Binance Futures sözleşmeleri yükleniyor..."):
            symbols = get_exchange_symbols()
            tickers = get_all_tickers()
            market_df = build_market_table(symbols, tickers)

            if market_df.empty:
                st.error("Binance Futures sözleşmeleri bulunamadı.")
                st.stop()

            st.session_state["market_df"] = market_df
            st.session_state["model_results"] = []

    except Exception as exc:
        st.error("Binance API verisi alınamadı.")
        st.code(str(exc))
        st.warning(
            "HTTP 451 alıyorsan bu bölgesel erişim kısıtlamasıdır. "
            "Bu kod, erişim kısıtlamasını aşamaz."
        )
        st.stop()

market_df = st.session_state["market_df"]

# Filter for display; preserve the highest percentage at the top.
visible_df = market_df[
    market_df["24s hacim (USDT)"] >= minimum_volume
].copy()

visible_df = visible_df.sort_values(
    "24s değişim (%)", ascending=False
).reset_index(drop=True)

visible_df["Sıra"] = np.arange(1, len(visible_df) + 1)

total_count = len(market_df)
positive_count = int((market_df["24s değişim (%)"] > 0).sum())
negative_count = int((market_df["24s değişim (%)"] < 0).sum())

top = market_df.iloc[0]
bottom = market_df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktif USDT perpetual", total_count)
c2.metric("Yükselen", positive_count)
c3.metric("Düşen", negative_count)
c4.metric(
    "En yüksek 24s değişim",
    f"{top['Sembol']} +{top['24s değişim (%)']:.2f}%"
)

st.subheader("🔥 En yüksek yüzde değişimi")
st.dataframe(
    visible_df.head(20),
    use_container_width=True,
    hide_index=True
)

with st.expander(f"Tüm sözleşmeler ({len(visible_df)})", expanded=True):
    st.dataframe(
        visible_df,
        use_container_width=True,
        hide_index=True
    )

st.download_button(
    "📥 Tüm Futures listesini CSV indir",
    data=visible_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="binance_futures_all_usdt_contracts.csv",
    mime="text/csv"
)

st.divider()
st.subheader("🤖 AI model taraması")

if st.button("🚀 Model taramasını başlat"):
    if scan_count_option == "Tüm kriptolar":
        to_scan = visible_df.copy()
    elif scan_count_option == "İlk 100":
        to_scan = visible_df.head(100).copy()
    elif scan_count_option == "İlk 50":
        to_scan = visible_df.head(50).copy()
    else:
        to_scan = visible_df.head(20).copy()

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, row in enumerate(to_scan.to_dict("records")):
        symbol = row["Sembol"]
        status.write(f"Analiz ediliyor: {symbol} ({i + 1}/{len(to_scan)})")

        try:
            candles = get_klines(
                symbol,
                INTERVAL,
                candle_count
            )

            prediction = model_signal(candles)

            results.append({
                "Sembol": symbol,
                "24s değişim (%)": row["24s değişim (%)"],
                "Sinyal": prediction["signal"],
                "AL olasılığı (%)": round(prediction["p_up"], 2),
                "SAT olasılığı (%)": round(prediction["p_down"], 2),
                "Test doğruluğu (%)": round(prediction["accuracy"], 2),
                "Son fiyat": row["Son fiyat"],
                "24s hacim (USDT)": row["24s hacim (USDT)"],
                "Test örneği": prediction["test_samples"],
                "_candles": candles,
                "_error": "",
            })

        except Exception as exc:
            results.append({
                "Sembol": symbol,
                "24s değişim (%)": row["24s değişim (%)"],
                "Sinyal": "VERİ HATASI",
                "AL olasılığı (%)": np.nan,
                "SAT olasılığı (%)": np.nan,
                "Test doğruluğu (%)": np.nan,
                "Son fiyat": row["Son fiyat"],
                "24s hacim (USDT)": row["24s hacim (USDT)"],
                "Test örneği": 0,
                "_candles": None,
                "_error": str(exc),
            })

        progress.progress((i + 1) / len(to_scan))
        time.sleep(0.12)

    st.session_state["model_results"] = results
    status.success("Model taraması tamamlandı.")

model_results = st.session_state["model_results"]

if model_results:
    good = [
        r for r in model_results
        if r["Sinyal"] in ("AL", "SAT")
    ]

    bad = [
        r for r in model_results
        if r["Sinyal"] == "VERİ HATASI"
    ]

    if bad:
        with st.expander(f"Veri hataları ({len(bad)})"):
            for r in bad:
                st.write(f"{r['Sembol']}: {r['_error']}")

    if good:
        result_df = pd.DataFrame(good)

        # Rank by 24h percentage change, highest first.
        result_df = result_df.sort_values(
            "24s değişim (%)",
            ascending=False
        )

        filtered = result_df[
            (
                (result_df["AL olasılığı (%)"] >= minimum_probability)
                | (result_df["SAT olasılığı (%)"] >= minimum_probability)
            )
            & (
                result_df["Test doğruluğu (%)"] >= minimum_accuracy
            )
        ].copy()

        if signal_filter != "Tümü":
            filtered = filtered[
                filtered["Sinyal"] == signal_filter
            ]

        st.subheader("🏆 Model sonuçları — en yüksek yüzde ilk sırada")
        display_columns = [
            "Sembol",
            "24s değişim (%)",
            "Sinyal",
            "AL olasılığı (%)",
            "SAT olasılığı (%)",
            "Test doğruluğu (%)",
            "Son fiyat",
            "24s hacim (USDT)",
            "Test örneği",
        ]

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Model sonuçlarını CSV indir",
            data=filtered[display_columns].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="binance_futures_ai_results.csv",
            mime="text/csv"
        )

        chart_symbol = st.selectbox(
            "15 dakikalık grafiği görüntüle",
            [r["Sembol"] for r in good]
        )

        selected = next(
            r for r in good if r["Sembol"] == chart_symbol
        )

        candles = selected["_candles"].set_index("datetime")

        st.line_chart(
            candles[["close"]],
            y="close",
            use_container_width=True
        )

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Sinyal", selected["Sinyal"])
        x2.metric("24s değişim", f"{selected['24s değişim (%)']:.2f}%")
        x3.metric("AL olasılığı", f"{selected['AL olasılığı (%)']:.2f}%")
        x4.metric("SAT olasılığı", f"{selected['SAT olasılığı (%)']:.2f}%")

st.divider()
st.caption(
    "Veriler herkese açık Binance Futures piyasa uç noktalarından alınır. "
    "Model olasılıkları ve geçmiş test doğruluğu kâr garantisi değildir. "
    "Bu panel otomatik emir göndermez. HTTP 451 gibi bölgesel erişim "
    "kısıtlamaları uygulama koduyla çözülemez."
)
