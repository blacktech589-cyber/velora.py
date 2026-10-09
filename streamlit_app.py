
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ==========================================================
# FUTURES CRYPTO SCANNER
# Data source: OKX public USDT perpetual swap market data
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
session.headers.update({"User-Agent": "PublicFuturesScanner/1.0"})


def api_get(path, params=None):
    response = session.get(
        BASE_URL + path,
        params=params,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    if str(result.get("code", "0")) != "0":
        raise RuntimeError(result.get("msg", "API error"))

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

    # Contract volume is not necessarily USDT notional volume.
    df = df[df["24s hacim (sözleşme)"] >= min_volume].copy()

    df = df.sort_values(
        "24s değişim (%)",
        ascending=False,
    ).reset_index(drop=True)

    df.insert(0, "Sıra", np.arange(1, len(df) + 1))
    return df


@st.cache_data(ttl=15, show_spinner=False)
def get_latest_candles(symbol, limit=300):
    data = api_get(
        "/api/v5/market/candles",
        {
            "instId": symbol,
            "bar": "15m",
            "limit": str(min(limit, 300)),
        },
    )

    return parse_candles(data)


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


def get_candles(symbol, target=1000):
    # Get the newest completed candles first.
    latest = get_latest_candles(symbol, min(target, 300))
    frames = [latest]
    all_timestamps = set(latest["timestamp"].astype(int).tolist())

    # Older historical candles can be retrieved in pages.
    # The history endpoint may not include the most recent two days.
    # The newest page above preserves the latest data.
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
                "bar": "15m",
                "limit": "100",
                "after": str(before),
            },
        )

        older = parse_candles(page)

        if older.empty:
            break

        new_rows = older[
            ~older["timestamp"].astype(int).isin(all_timestamps)
        ].copy()

        if new_rows.empty:
            break

        for ts in new_rows["timestamp"].astype(int):
            all_timestamps.add(ts)

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

    return result


# ------------------- MODEL -------------------

def make_features(df):
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    # Price changes only; no RSI, EMA, MACD or Bollinger Bands.
    for lag in [1, 2, 3, 4, 8, 12, 24, 48]:
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

    one_return = close.pct_change()

    for window in [4, 8, 16, 32]:
        f[f"mean_return_{window}"] = one_return.rolling(window).mean()
        f[f"volatility_{window}"] = one_return.rolling(window).std()
        f[f"range_mean_{window}"] = f["range"].rolling(window).mean()

    for lag in [1, 2, 4, 8]:
        f[f"volume_change_{lag}"] = volume / volume.shift(lag) - 1

    f["relative_volume"] = volume / volume.rolling(24).mean() - 1

    return f.replace([np.inf, -np.inf], np.nan)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_model(X, y, epochs=350, learning_rate=0.06):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    for epoch in range(epochs):
        pred = sigmoid(X @ weights + bias)
        error = pred - y

        grad_w = X.T @ error / n + 0.001 * weights
        grad_b = error.mean()
        lr = learning_rate / (1 + epoch / 150)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def analyze_model(df):
    features = make_features(df)
    future_return = df["close"].shift(-4) / df["close"] - 1

    valid = features.notna().all(axis=1) & future_return.notna()
    X = features.loc[valid].to_numpy(dtype=float)
    y = (future_return.loc[valid].to_numpy() > 0).astype(float)

    if len(X) < 150:
        raise RuntimeError("Model için yeterli geçmiş veri yok.")

    split = int(len(X) * 0.8)
    split = max(50, min(split, len(X) - 30))

    X_train_raw = X[:split]
    X_test_raw = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # Chronological holdout evaluation.
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0)
    std_train[std_train < 1e-9] = 1

    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    weights, bias = train_model(X_train, y_train)

    pred_test = (sigmoid(X_test @ weights + bias) >= 0.5).astype(float)
    accuracy = float((pred_test == y_test).mean() * 100)

    # Retrain on all labeled observations for the latest signal.
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all < 1e-9] = 1

    X_all = (X - mean_all) / std_all
    weights, bias = train_model(X_all, y)

    latest = features.iloc[[-1]].to_numpy(dtype=float)

    if not np.isfinite(latest).all():
        raise RuntimeError("Son mum verisi geçersiz.")

    p_up = float(
        sigmoid(((latest - mean_all) / std_all) @ weights + bias)[0]
    )

    return {
        "Sinyal": "AL" if p_up >= 0.5 else "SAT",
        "AL olasılığı (%)": p_up * 100,
        "SAT olasılığı (%)": (1 - p_up) * 100,
        "Test doğruluğu (%)": accuracy,
        "Test örneği": len(X_test),
    }


# ------------------- STREAMLIT PANEL -------------------

st.title("📈 Futures Crypto Scanner")
st.caption(
    "Veri kaynağı: OKX public USDT perpetual swaps | "
    "Sıralama: 24 saatlik yüzde değişimi, en yüksek ilk sırada"
)

with st.sidebar:
    st.header("Tarama ayarları")

    min_volume = st.number_input(
        "Minimum 24s hacim (sözleşme adedi)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

    candle_count = st.selectbox(
        "Geçmiş 15 dakikalık mum",
        [200, 300, 500, 800, 1000],
        index=4,
    )

    scan_count = st.selectbox(
        "Model taraması",
        ["Tüm sözleşmeler", "İlk 100", "İlk 50", "İlk 20"],
    )

    min_probability = st.slider(
        "Minimum AL/SAT olasılığı (%)",
        50, 95, 60,
    )

    min_accuracy = st.slider(
        "Minimum test doğruluğu (%)",
        0, 90, 50,
    )

    signal_filter = st.selectbox(
        "Model sinyali",
        ["Tümü", "AL", "SAT"],
    )

    refresh = st.button(
        "🔄 Piyasayı yenile",
        type="primary",
        use_container_width=True,
    )

if "market_table" not in st.session_state:
    st.session_state["market_table"] = None

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []

if refresh or st.session_state["market_table"] is None:
    try:
        with st.spinner("Piyasa sözleşmeleri alınıyor..."):
            symbols = get_instruments()
            tickers = get_tickers()

            market = build_market_table(
                symbols, tickers, min_volume
            )

            if market.empty:
                st.error(
                    "Sözleşme bulunamadı. API erişimini ve filtreleri kontrol et."
                )
                st.stop()

            st.session_state["market_table"] = market
            st.session_state["model_results"] = []

    except Exception as exc:
        st.error("Piyasa verisi alınamadı.")
        st.code(str(exc))
        st.info(
            "Bu uygulama OKX verisi kullanır, Binance Futures verisi değildir. "
            "Erişim sorunu varsa sağlayıcının API erişimini kontrol et."
        )
        st.stop()

market = st.session_state["market_table"]

# Highest 24-hour percentage change always first.
market = market.sort_values(
    "24s değişim (%)", ascending=False
).reset_index(drop=True)

market["Sıra"] = np.arange(1, len(market) + 1)

top = market.iloc[0]
positive = int((market["24s değişim (%)"] > 0).sum())
negative = int((market["24s değişim (%)"] < 0).sum())

a, b, c, d = st.columns(4)
a.metric("USDT perpetual sözleşme", len(market))
b.metric("Yükselen sözleşme", positive)
c.metric("Düşen sözleşme", negative)
d.metric(
    "En yüksek değişim",
    f"{top['Sembol']} {top['24s değişim (%)']:+.2f}%"
)

st.subheader("🔥 En yüksek yüzde değişimi")

st.dataframe(
    market.head(20),
    use_container_width=True,
    hide_index=True,
)

with st.expander(f"Tüm sözleşmeler ({len(market)})", expanded=True):
    st.dataframe(
        market,
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "📥 Tüm sözleşmeleri CSV indir",
    data=market.to_csv(index=False).encode("utf-8-sig"),
    file_name="usdt_perpetual_market.csv",
    mime="text/csv",
)

st.divider()
st.subheader("🤖 Model analizi")

if st.button("🚀 Model taramasını başlat"):
    if scan_count == "Tüm sözleşmeler":
        selected_market = market
    elif scan_count == "İlk 100":
        selected_market = market.head(100)
    elif scan_count == "İlk 50":
        selected_market = market.head(50)
    else:
        selected_market = market.head(20)

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, row in enumerate(selected_market.to_dict("records")):
        symbol = row["Sembol"]
        status.write(
            f"Analiz: {symbol} ({i + 1}/{len(selected_market)})"
        )

        try:
            candles = get_candles(symbol, candle_count)
            prediction = analyze_model(candles)

            results.append({
                "Sembol": symbol,
                "24s değişim (%)": row["24s değişim (%)"],
                "Son fiyat": row["Son fiyat"],
                **prediction,
                "_candles": candles,
                "_error": "",
            })

        except Exception as exc:
            results.append({
                "Sembol": symbol,
                "24s değişim (%)": row["24s değişim (%)"],
                "Son fiyat": row["Son fiyat"],
                "Sinyal": "VERİ HATASI",
                "AL olasılığı (%)": np.nan,
                "SAT olasılığı (%)": np.nan,
                "Test doğruluğu (%)": np.nan,
                "Test örneği": 0,
                "_candles": None,
                "_error": str(exc),
            })

        progress.progress((i + 1) / len(selected_market))
        time.sleep(0.12)

    st.session_state["model_results"] = results
    status.success("Model taraması tamamlandı.")

results = st.session_state["model_results"]

if results:
    errors = [r for r in results if r["Sinyal"] == "VERİ HATASI"]
    good = [r for r in results if r["Sinyal"] in ("AL", "SAT")]

    if errors:
        with st.expander(f"Veri hataları ({len(errors)})"):
            for row in errors:
                st.write(f"{row['Sembol']}: {row['_error']}")

    if good:
        result_df = pd.DataFrame(good).sort_values(
            "24s değişim (%)", ascending=False
        )

        filtered = result_df[
            (
                (result_df["AL olasılığı (%)"] >= min_probability)
                | (result_df["SAT olasılığı (%)"] >= min_probability)
            )
            & (result_df["Test doğruluğu (%)"] >= min_accuracy)
        ].copy()

        if signal_filter != "Tümü":
            filtered = filtered[filtered["Sinyal"] == signal_filter]

        columns = [
            "Sembol",
            "24s değişim (%)",
            "Sinyal",
            "AL olasılığı (%)",
            "SAT olasılığı (%)",
            "Test doğruluğu (%)",
            "Son fiyat",
            "Test örneği",
        ]

        st.subheader("🏆 Model sonuçları — en yüksek yüzde ilk sırada")
        st.dataframe(
            filtered[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Model sonuçlarını CSV indir",
            data=filtered[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="futures_model_results.csv",
            mime="text/csv",
        )

        symbol_choice = st.selectbox(
            "Grafik için sözleşme seç",
            [r["Sembol"] for r in good],
        )

        selected = next(
            r for r in good if r["Sembol"] == symbol_choice
        )

        candles = selected["_candles"].set_index("datetime")

        st.line_chart(
            candles[["close"]],
            y="close",
            use_container_width=True,
        )

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Model sinyali", selected["Sinyal"])
        x2.metric(
            "24s değişim",
            f"{selected['24s değişim (%)']:+.2f}%"
        )
        x3.metric(
            "AL olasılığı",
            f"{selected['AL olasılığı (%)']:.2f}%"
        )
        x4.metric(
            "SAT olasılığı",
            f"{selected['SAT olasılığı (%)']:.2f}%"
        )

st.divider()
st.caption(
    "Bu panel OKX verisi kullanır; Binance sözleşmelerinin tamamıyla aynı değildir. "
    "Model olasılıkları kâr garantisi değildir. Otomatik emir gönderilmez."
)
