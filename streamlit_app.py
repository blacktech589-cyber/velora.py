import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# ============================================================
# CRYPTO FUTURES SCANNER - OKX TR PUBLIC MARKET DATA
# Public data only. No API keys. No order placement.
# Model: NumPy logistic regression
# Timeframe: 15 minutes
# Indicators excluded: RSI, EMA, MACD, Bollinger Bands
# ============================================================

st.set_page_config(
    page_title="Crypto Futures AI Scanner",
    page_icon="📊",
    layout="wide"
)

BASE_URL = "https://tr.okx.com"
TIMEFRAME = "15m"
CANDLE_TARGET = 1000
REQUEST_TIMEOUT = 15

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoMarketScanner/1.0",
    "Accept": "application/json"
})


# ---------------------- API HELPERS --------------------------

def api_get(path, params=None):
    url = BASE_URL + path

    response = SESSION.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
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
    """Return active USDT-settled linear perpetual swap contracts."""
    data = api_get(
        "/api/v5/public/instruments",
        {"instType": "SWAP"}
    )

    instruments = []

    for item in data:
        if item.get("state") != "live":
            continue

        if item.get("settleCcy") != "USDT":
            continue

        if item.get("ctType") != "linear":
            continue

        instruments.append({
            "instId": item["instId"],
            "baseCcy": item.get("ctValCcy", ""),
            "settleCcy": item.get("settleCcy", "USDT")
        })

    return instruments


@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    """Fetch all available swap ticker snapshots."""
    return api_get(
        "/api/v5/market/tickers",
        {"instType": "SWAP"}
    )


def get_candles(inst_id, target=1000):
    """
    Fetch historical 15-minute candles.
    Uses the history endpoint in pages, then sorts oldest to newest.
    """
    rows = []
    seen_timestamps = set()
    before = None

    for _ in range(15):
        params = {
            "instId": inst_id,
            "bar": TIMEFRAME,
            "limit": "100"
        }

        if before is not None:
            params["after"] = str(before)

        page = api_get(
            "/api/v5/market/history-candles",
            params
        )

        if not page:
            break

        new_rows = 0

        for row in page:
            ts = int(row[0])

            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                rows.append(row)
                new_rows += 1

        if len(rows) >= target or new_rows == 0:
            break

        oldest_ts = min(int(row[0]) for row in page)

        if before == oldest_ts:
            break

        before = oldest_ts

        # Respect public API rate limits.
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(f"No candle data for {inst_id}")

    rows.sort(key=lambda row: int(row[0]))

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_currency",
            "volume_quote",
            "confirm"
        ]
    )

    for column in [
        "open", "high", "low", "close",
        "volume", "volume_quote"
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    # Exclude incomplete candles.
    df = df[df["confirm"].astype(str) == "1"].copy()

    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").tail(target)
    df = df.reset_index(drop=True)

    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    if len(df) < 100:
        raise RuntimeError(
            f"Not enough completed candles for {inst_id}: {len(df)}"
        )

    return df


# ---------------------- MODEL FEATURES -----------------------

def build_features(df):
    """
    Raw price/volume transformations only.
    No RSI, EMA, MACD, or Bollinger Bands.
    """
    data = df.copy()

    close = data["close"].replace(0, np.nan)
    open_price = data["open"].replace(0, np.nan)
    high = data["high"]
    low = data["low"]
    volume = data["volume"].replace(0, np.nan)

    features = pd.DataFrame(index=data.index)

    # Price changes over several completed candles.
    for lag in [1, 2, 3, 4, 8, 12, 24, 48]:
        features[f"return_{lag}"] = close.pct_change(lag)

    # Candle structure.
    features["body_return"] = (
        data["close"] - data["open"]
    ) / open_price

    features["high_low_range"] = (
        high - low
    ) / close

    features["close_position"] = (
        data["close"] - low
    ) / (high - low).replace(0, np.nan)

    features["upper_wick"] = (
        high - data[["open", "close"]].max(axis=1)
    ) / close

    features["lower_wick"] = (
        data[["open", "close"]].min(axis=1) - low
    ) / close

    # Rolling returns and volatility, not technical indicators.
    one_candle_return = close.pct_change()

    for window in [4, 8, 16, 32]:
        features[f"mean_return_{window}"] = (
            one_candle_return.rolling(window).mean()
        )

        features[f"volatility_{window}"] = (
            one_candle_return.rolling(window).std()
        )

        features[f"range_mean_{window}"] = (
            features["high_low_range"].rolling(window).mean()
        )

    # Relative volume changes.
    for lag in [1, 2, 4, 8]:
        features[f"volume_change_{lag}"] = (
            volume / volume.shift(lag) - 1
        )

    features["volume_position"] = (
        volume / volume.rolling(24).mean() - 1
    )

    # Remove infinite and invalid values.
    features = features.replace(
        [np.inf, -np.inf], np.nan
    )

    return features


# ---------------------- LOGISTIC REGRESSION ------------------

def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(X, y, epochs=450, learning_rate=0.08):
    """Train a small binary classifier using NumPy only."""
    n_samples, n_features = X.shape

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    for epoch in range(epochs):
        probabilities = sigmoid(X @ weights + bias)

        error = probabilities - y

        grad_w = (
            X.T @ error
        ) / n_samples + 0.001 * weights

        grad_b = np.mean(error)

        # Gradually reduce the learning rate.
        lr = learning_rate / (
            1.0 + epoch / 150.0
        )

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def predict_probability(X, weights, bias):
    return sigmoid(X @ weights + bias)


def calculate_model(df):
    """
    Label:
    1 = close after the next 4 candles is higher
    0 = close after the next 4 candles is lower or equal

    Test uses a chronological split, not a random split.
    """
    features = build_features(df)

    future_return = (
        df["close"].shift(-4) / df["close"] - 1
    )

    valid = features.notna().all(axis=1)
    valid &= future_return.notna()

    X_all = features.loc[valid].to_numpy(dtype=float)
    returns = future_return.loc[valid].to_numpy(dtype=float)

    if len(X_all) < 150:
        raise RuntimeError("Not enough usable samples to train model.")

    y_all = (returns > 0).astype(float)

    split = int(len(X_all) * 0.80)
    split = max(50, min(split, len(X_all) - 30))

    X_train_raw = X_all[:split]
    y_train = y_all[:split]

    X_test_raw = X_all[split:]
    y_test = y_all[split:]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-9] = 1.0

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    weights, bias = fit_logistic_regression(
        X_train, y_train
    )

    test_probabilities = predict_probability(
        X_test, weights, bias
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(float)

    accuracy = float(
        np.mean(test_predictions == y_test) * 100
    )

    # Retrain on all usable historical data for the current signal.
    full_mean = X_all.mean(axis=0)
    full_std = X_all.std(axis=0)
    full_std[full_std < 1e-9] = 1.0

    X_full = (X_all - full_mean) / full_std

    final_weights, final_bias = fit_logistic_regression(
        X_full, y_all
    )

    latest_features = features.iloc[[-1]].to_numpy(dtype=float)
    latest_features = (
        latest_features - full_mean
    ) / full_std

    current_probability = float(
        predict_probability(
            latest_features,
            final_weights,
            final_bias
        )[0]
    )

    return {
        "probability_up": current_probability,
        "probability_down": 1.0 - current_probability,
        "accuracy": accuracy,
        "samples": len(X_all),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_all.shape[1]
    }


# ---------------------- UNIVERSE & SCAN ----------------------

def make_universe(instruments, tickers, min_quote_volume):
    ticker_map = {
        item.get("instId"): item
        for item in tickers
    }

    universe = []

    for instrument in instruments:
        inst_id = instrument["instId"]
        ticker = ticker_map.get(inst_id)

        if not ticker:
            continue

        try:
            last_price = float(ticker.get("last", 0))
            quote_volume = float(
                ticker.get("volCcy24h", 0)
            )
        except (TypeError, ValueError):
            continue

        if last_price <= 0:
            continue

        if quote_volume < min_quote_volume:
            continue

        universe.append({
            "instId": inst_id,
            "price": last_price,
            "quote_volume": quote_volume,
            "change_24h": float(
                ticker.get("sodUtc8", 0) or 0
            )
        })

    universe.sort(
        key=lambda item: item["quote_volume"],
        reverse=True
    )

    return universe


def scan_market(universe, candle_count, progress_bar, status_text):
    results = []
    total = len(universe)

    for index, coin in enumerate(universe):
        inst_id = coin["instId"]

        status_text.write(
            f"Analiz ediliyor: **{inst_id}** "
            f"({index + 1}/{total})"
        )

        try:
            candles = get_candles(
                inst_id,
                target=candle_count
            )

            model = calculate_model(candles)

            p_up = model["probability_up"]
            p_down = model["probability_down"]

            # Only BUY or SELL; no HOLD label.
            signal = "AL" if p_up >= 0.5 else "SAT"

            results.append({
                "Sözleşme": inst_id,
                "Sinyal": signal,
                "AL olasılığı (%)": round(p_up * 100, 2),
                "SAT olasılığı (%)": round(p_down * 100, 2),
                "Test doğruluğu (%)": round(model["accuracy"], 2),
                "Son fiyat": coin["price"],
                "24s hacim": coin["quote_volume"],
                "Kullanılan mum": len(candles),
                "Test örneği": model["test_samples"],
                "Güncelleme (UTC)": datetime.now(
                    timezone.utc
                ).strftime("%H:%M:%S"),
                "_candles": candles
            })

        except Exception as exc:
            results.append({
                "Sözleşme": inst_id,
                "Sinyal": "VERİ HATASI",
                "AL olasılığı (%)": np.nan,
                "SAT olasılığı (%)": np.nan,
                "Test doğruluğu (%)": np.nan,
                "Son fiyat": coin["price"],
                "24s hacim": coin["quote_volume"],
                "Kullanılan mum": 0,
                "Test örneği": 0,
                "Güncelleme (UTC)": datetime.now(
                    timezone.utc
                ).strftime("%H:%M:%S"),
                "_error": str(exc)
            })

        progress_bar.progress(
            (index + 1) / max(total, 1)
        )

        # Conservative spacing between coin requests.
        time.sleep(0.15)

    return results


# ---------------------- USER INTERFACE -----------------------

st.title("📊 Crypto Futures AI Scanner")
st.caption(
    "OKX TR public market data • USDT linear perpetual swaps • "
    "15-minute candles • No automated trading"
)

with st.sidebar:
    st.header("Tarama Ayarları")

    max_coins = st.slider(
        "Taranacak sözleşme sayısı",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    candle_count = st.select_slider(
        "Geçmiş mum sayısı",
        options=[200, 400, 600, 800, 1000],
        value=1000
    )

    min_quote_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0.0,
        value=100000.0,
        step=100000.0
    )

    min_probability = st.slider(
        "Minimum sinyal olasılığı (%)",
        min_value=50,
        max_value=95,
        value=60
    )

    min_accuracy = st.slider(
        "Minimum test doğruluğu (%)",
        min_value=0,
        max_value=90,
        value=50
    )

    signal_filter = st.selectbox(
        "Sinyal filtresi",
        ["Tümü", "AL", "SAT"]
    )

    run_scan = st.button(
        "🚀 Piyasayı Tara",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.caption(
        "Olasılıklar model tahminidir; kâr garantisi değildir. "
        "Geçmiş test doğruluğu gelecekteki performansı garanti etmez."
    )


# Persist results across Streamlit reruns.
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

if "scan_errors" not in st.session_state:
    st.session_state["scan_errors"] = []

if run_scan:
    st.session_state["scan_results"] = []
    st.session_state["scan_errors"] = []

    try:
        with st.spinner("Sözleşmeler ve piyasa verileri alınıyor..."):
            instruments = get_instruments()
            tickers = get_tickers()

            universe = make_universe(
                instruments,
                tickers,
                min_quote_volume
            )

        if not universe:
            st.error(
                "Filtrelere uygun sözleşme bulunamadı. "
                "Minimum hacim değerini azaltmayı dene."
            )
            st.stop()

        universe = universe[:max_coins]

        st.info(
            f"{len(instruments)} aktif USDT doğrusal swap sözleşmesi "
            f"bulundu. Hacim filtresinden sonra "
            f"{len(universe)} sözleşme taranacak."
        )

        progress = st.progress(0)
        status = st.empty()

        results = scan_market(
            universe,
            candle_count,
            progress,
            status
        )

        st.session_state["scan_results"] = results

        status.success("Tarama tamamlandı.")

    except Exception as exc:
        st.error(
            "Piyasa verisi alınamadı. OKX TR API erişimini ve "
            "internet bağlantısını kontrol et."
        )
        st.code(str(exc))


# ---------------------- RESULTS ------------------------------

results = st.session_state["scan_results"]

if results:
    valid_results = [
        item for item in results
        if item.get("Sinyal") in ["AL", "SAT"]
    ]

    errors = [
        item for item in results
        if item.get("Sinyal") == "VERİ HATASI"
    ]

    if errors:
        with st.expander(
            f"Veri alınamayan sözleşmeler ({len(errors)})"
        ):
            for item in errors:
                st.write(
                    f"**{item['Sözleşme']}** — "
                    f"{item.get('_error', 'Bilinmeyen hata')}"
                )

    if valid_results:
        df_results = pd.DataFrame(valid_results)

        # Filter by signal and probability.
        df_filtered = df_results[
            (
                (df_results["AL olasılığı (%)"] >= min_probability)
                | (df_results["SAT olasılığı (%)"] >= min_probability)
            )
            & (
                df_results["Test doğruluğu (%)"] >= min_accuracy
            )
        ].copy()

        if signal_filter != "Tümü":
            df_filtered = df_filtered[
                df_filtered["Sinyal"] == signal_filter
            ]

        df_filtered = df_filtered.sort_values(
            by=[
                "AL olasılığı (%)",
                "SAT olasılığı (%)",
                "Test doğruluğu (%)"
            ],
            ascending=False
        )

        total_scanned = len(valid_results)
        buy_count = sum(
            item["Sinyal"] == "AL"
            for item in valid_results
        )
        sell_count = sum(
            item["Sinyal"] == "SAT"
            for item in valid_results
        )

        average_accuracy = float(
            df_results["Test doğruluğu (%)"].mean()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Başarılı analiz", total_scanned)
        c2.metric("AL sinyali", buy_count)
        c3.metric("SAT sinyali", sell_count)
        c4.metric(
            "Ortalama test doğruluğu",
            f"{average_accuracy:.2f}%"
        )

        st.subheader("🏆 Sinyal Sıralaması")

        display_columns = [
            "Sözleşme",
            "Sinyal",
            "AL olasılığı (%)",
            "SAT olasılığı (%)",
            "Test doğruluğu (%)",
            "Son fiyat",
            "24s hacim",
            "Kullanılan mum",
            "Test örneği"
        ]

        st.dataframe(
            df_filtered[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Son fiyat": st.column_config.NumberColumn(
                    format="%.8f"
                ),
                "24s hacim": st.column_config.NumberColumn(
                    format="%,.2f"
                )
            }
        )

        csv = df_filtered[display_columns].to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Sonuçları CSV indir",
            data=csv,
            file_name="crypto_futures_scan.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("📈 Mum Grafiği ve Model Detayı")

        available_symbols = [
            item["Sözleşme"]
            for item in valid_results
        ]

        selected_symbol = st.selectbox(
            "Grafiği görüntülenecek sözleşme",
            available_symbols
        )

        selected = next(
            item for item in valid_results
            if item["Sözleşme"] == selected_symbol
        )

        candles = selected["_candles"]

        chart = go.Figure(
            data=[
                go.Candlestick(
                    x=candles["datetime"],
                    open=candles["open"],
                    high=candles["high"],
                    low=candles["low"],
                    close=candles["close"],
                    name=selected_symbol
                )
            ]
        )

        chart.update_layout(
            title=f"{selected_symbol} — 15 dakikalık mumlar",
            xaxis_title="Zaman (UTC)",
            yaxis_title="Fiyat",
            xaxis_rangeslider_visible=False,
            height=600
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Model sinyali",
            selected["Sinyal"]
        )
        m2.metric(
            "AL olasılığı",
            f"{selected['AL olasılığı (%)']:.2f}%"
        )
        m3.metric(
            "SAT olasılığı",
            f"{selected['SAT olasılığı (%)']:.2f}%"
        )
        m4.metric(
            "Geçmiş test doğruluğu",
            f"{selected['Test doğruluğu (%)']:.2f}%"
        )

        st.caption(
            f"Model örnek sayısı: {selected['Test örneği']} test örneği. "
            f"Grafikte {len(candles)} tamamlanmış mum gösteriliyor."
        )

else:
    st.info(
        "Başlamak için soldaki ayarları seçip "
        "**Piyasayı Tara** düğmesine bas."
    )

st.divider()
st.caption(
    "Bilgilendirme amaçlı yazılım. AL/SAT etiketleri model sınıflandırmasıdır; "
    "yatırım tavsiyesi değildir. Ücretler, fonlama, kayma ve likidasyon "
    "bu tahmin doğruluğuna dahil değildir."
)
