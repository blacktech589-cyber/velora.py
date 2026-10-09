
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime, timezone

# =========================================================
# STREAMLIT AYARLARI
# =========================================================

st.set_page_config(
    page_title="Binance BUY / SELL Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance Spot BUY / SELL Scanner")
st.caption(
    "USDT pariteleri • 15 dakikalık mumlar • "
    "Son 1.000 tamamlanmış mum • Emir göndermez"
)

# =========================================================
# AYARLAR
# =========================================================

INTERVAL = "15m"
CANDLE_LIMIT = 1000
FEATURE_WINDOW = 20
TEST_SIZE = 0.20
EPOCHS = 350
LEARNING_RATE = 0.08
MAX_COINS_DEFAULT = 30
REQUEST_TIMEOUT = 15

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

# =========================================================
# GENEL API İSTEKLERİ
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_exchange_info():
    last_error = None

    for base_url in BASE_URLS:
        try:
            response = requests.get(
                base_url + "/api/v3/exchangeInfo",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            symbols = []

            for item in data.get("symbols", []):
                if (
                    item.get("status") == "TRADING"
                    and item.get("quoteAsset") == "USDT"
                    and item.get("isSpotTradingAllowed", False)
                ):
                    symbols.append(item["symbol"])

            if symbols:
                return symbols

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        "Binance parite listesi alınamadı. "
        f"Son hata: {last_error}"
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol, interval="15m", limit=1000):
    last_error = None

    for base_url in BASE_URLS:
        try:
            response = requests.get(
                base_url + "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()
            raw = response.json()

            if not isinstance(raw, list) or len(raw) < 100:
                continue

            columns = [
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
                "ignore"
            ]

            df = pd.DataFrame(raw, columns=columns)

            numeric_columns = [
                "open", "high", "low", "close",
                "volume", "quote_volume", "trades"
            ]

            for column in numeric_columns:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )

            df["open_time"] = pd.to_datetime(
                df["open_time"], unit="ms", utc=True
            )

            df["close_time"] = pd.to_datetime(
                df["close_time"], unit="ms", utc=True
            )

            df = df.dropna(
                subset=["open", "high", "low", "close", "volume"]
            ).reset_index(drop=True)

            # Son mum henüz tamamlanmadıysa çıkar.
            now = pd.Timestamp.now(tz="UTC")

            if (
                len(df) > 1
                and df.iloc[-1]["close_time"] > now
            ):
                df = df.iloc[:-1].copy()

            return df.reset_index(drop=True)

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"{symbol} mum verisi alınamadı. "
        f"Son hata: {last_error}"
    )


# =========================================================
# ÖZELLİKLER
# İndikatör kullanılmaz: RSI / EMA / MACD / Bollinger yok.
# =========================================================

def create_features(df):
    close = df["close"].to_numpy(dtype=float)
    open_price = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    eps = 1e-12

    # Mum getirileri
    returns = np.zeros(len(close), dtype=float)
    returns[1:] = np.diff(close) / np.maximum(close[:-1], eps)

    # Mum gövdesi ve fitilleri
    candle_body = (close - open_price) / np.maximum(open_price, eps)
    candle_range = (high - low) / np.maximum(close, eps)
    upper_wick = (
        high - np.maximum(open_price, close)
    ) / np.maximum(close, eps)
    lower_wick = (
        np.minimum(open_price, close) - low
    ) / np.maximum(close, eps)

    # Hacim değişimi
    log_volume = np.log1p(np.maximum(volume, 0))
    volume_change = np.zeros(len(volume), dtype=float)
    volume_change[1:] = np.diff(log_volume)

    data = pd.DataFrame({
        "return_1": returns,
        "return_2": pd.Series(returns).rolling(2).sum(),
        "return_3": pd.Series(returns).rolling(3).sum(),
        "return_5": pd.Series(returns).rolling(5).sum(),
        "return_10": pd.Series(returns).rolling(10).sum(),
        "return_20": pd.Series(returns).rolling(20).sum(),
        "volatility_5": pd.Series(returns).rolling(5).std(),
        "volatility_10": pd.Series(returns).rolling(10).std(),
        "volatility_20": pd.Series(returns).rolling(20).std(),
        "candle_body": candle_body,
        "candle_range": candle_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "volume_change": volume_change,
        "volume_mean_ratio": (
            pd.Series(volume)
            / (pd.Series(volume).rolling(20).mean() + eps)
        ),
    })

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


# =========================================================
# BASİT NUMPY LOJİSTİK REGRESYON
# PyTorch gerektirmez.
# Bu model LSTM değildir.
# =========================================================

def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(X, y, epochs=350, learning_rate=0.08):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    # Sınıf ağırlıkları, dengesiz hedeflerde yardımcı olabilir.
    positives = max(float(np.sum(y == 1)), 1.0)
    negatives = max(float(np.sum(y == 0)), 1.0)

    sample_weights = np.where(
        y == 1,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives)
    )

    for epoch in range(epochs):
        probabilities = sigmoid(X @ weights + bias)
        errors = (probabilities - y) * sample_weights

        grad_w = (X.T @ errors) / n_samples
        grad_b = float(np.mean(errors))

        # L2 düzenlileştirme
        grad_w += 0.001 * weights

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def predict_probability(X, weights, bias):
    return sigmoid(X @ weights + bias)


# =========================================================
# MODEL VE GEÇMİŞ TEST
# Gelecek mumun getirisi pozitifse yükseliş sınıfı.
# =========================================================

def train_and_evaluate(df):
    features = create_features(df)

    close = df["close"].to_numpy(dtype=float)

    # Her satırın hedefi: bir sonraki mum yukarı kapanıyor mu?
    future_return = np.full(len(close), np.nan, dtype=float)
    future_return[:-1] = (
        close[1:] - close[:-1]
    ) / np.maximum(close[:-1], 1e-12)

    target = (future_return > 0).astype(float)

    dataset = features.copy()
    dataset["target"] = target
    dataset["future_return"] = future_return
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna().reset_index(drop=True)

    if len(dataset) < 150:
        raise ValueError("Model eğitimi için yeterli veri yok.")

    feature_columns = list(features.columns)

    X_all = dataset[feature_columns].to_numpy(dtype=float)
    y_all = dataset["target"].to_numpy(dtype=float)

    # Zaman sırasını koruyan test ayrımı.
    split = int(len(dataset) * (1 - TEST_SIZE))
    split = max(100, min(split, len(dataset) - 30))

    X_train_raw = X_all[:split]
    y_train = y_all[:split]

    X_test_raw = X_all[split:]
    y_test = y_all[split:]

    # Ölçekleme yalnızca eğitim verisiyle öğrenilir.
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std < 1e-9] = 1.0

    X_train = np.clip((X_train_raw - mean) / std, -10, 10)
    X_test = np.clip((X_test_raw - mean) / std, -10, 10)

    weights, bias = fit_logistic_regression(
        X_train,
        y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    test_probabilities = predict_probability(
        X_test, weights, bias
    )

    test_predictions = (test_probabilities >= 0.50).astype(int)

    accuracy = float(np.mean(test_predictions == y_test) * 100)

    # Test tahminleri, sadece karşılaştırma amacıyla kullanılır.
    baseline = max(
        np.mean(y_test == 0),
        np.mean(y_test == 1)
    ) * 100

    # Son kullanılabilir satırla mevcut sinyal.
    last_x = X_all[-1:]
    last_x = np.clip((last_x - mean) / std, -10, 10)

    up_probability = float(
        predict_probability(last_x, weights, bias)[0]
    )

    # İstenen ikili sinyal: HOLD yok.
    signal = "BUY" if up_probability >= 0.50 else "SELL"

    # Ek bilgiler
    last_row = dataset.iloc[-1]

    return {
        "signal": signal,
        "up_probability": up_probability,
        "down_probability": 1.0 - up_probability,
        "accuracy": accuracy,
        "baseline_accuracy": float(baseline),
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "std": std,
        "feature_columns": feature_columns,
        "dataset": dataset,
        "last_row": last_row,
        "test_predictions": test_predictions,
        "test_targets": y_test,
        "test_probabilities": test_probabilities,
        "test_size": len(y_test),
    }


# =========================================================
# TEK PARİTE ANALİZİ
# =========================================================

def analyze_symbol(symbol):
    df = get_klines(symbol, INTERVAL, CANDLE_LIMIT)
    result = train_and_evaluate(df)

    return {
        "symbol": symbol,
        "price": float(df["close"].iloc[-1]),
        "signal": result["signal"],
        "up_probability": result["up_probability"],
        "down_probability": result["down_probability"],
        "accuracy": result["accuracy"],
        "baseline_accuracy": result["baseline_accuracy"],
        "candles": len(df),
        "df": df,
        "model": result,
    }


# =========================================================
# YAN PANEL
# =========================================================

with st.sidebar:
    st.header("Tarama ayarları")

    max_coins = st.slider(
        "Taranacak parite sayısı",
        min_value=5,
        max_value=100,
        value=MAX_COINS_DEFAULT,
        step=5
    )

    min_quote_volume = st.number_input(
        "Minimum 24 saatlik USDT hacmi",
        min_value=0.0,
        value=1_000_000.0,
        step=500_000.0
    )

    auto_refresh = st.checkbox(
        "Otomatik yenileme",
        value=False
    )

    refresh_seconds = st.selectbox(
        "Yenileme aralığı",
        [60, 120, 300, 600],
        index=2
    )

    run_scan = st.button(
        "🔍 Taramayı başlat",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.caption(
        "Bu panel yalnızca analiz yapar. "
        "BUY/SELL sinyalleri kâr garantisi değildir."
    )


# =========================================================
# TARAYICI
# =========================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None


if run_scan or (
    auto_refresh and not st.session_state.scan_results
):
    all_results = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    try:
        symbols = get_exchange_info()

        # 24 saatlik ticker hacimleri
        ticker_data = None
        last_error = None

        for base_url in BASE_URLS:
            try:
                response = requests.get(
                    base_url + "/api/v3/ticker/24hr",
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                ticker_data = response.json()
                break
            except Exception as exc:
                last_error = str(exc)

        if ticker_data is None:
            raise RuntimeError(
                "24 saatlik hacim verisi alınamadı: "
                + str(last_error)
            )

        ticker_map = {
            item["symbol"]: item
            for item in ticker_data
            if item.get("symbol") in symbols
        }

        liquid_symbols = []

        for symbol in symbols:
            item = ticker_map.get(symbol)

            if not item:
                continue

            try:
                quote_volume = float(item.get("quoteVolume", 0))
            except (ValueError, TypeError):
                continue

            if quote_volume >= min_quote_volume:
                liquid_symbols.append((symbol, quote_volume))

        liquid_symbols.sort(key=lambda item: item[1], reverse=True)
        selected_symbols = [
            item[0] for item in liquid_symbols[:max_coins]
        ]

        if not selected_symbols:
            raise RuntimeError(
                "Seçilen hacim koşullarına uyan parite bulunamadı."
            )

        for index, symbol in enumerate(selected_symbols):
            status.text(
                f"Analiz ediliyor: {symbol} "
                f"({index + 1}/{len(selected_symbols)})"
            )

            try:
                result = analyze_symbol(symbol)

                # Grafiğe gerek olmayan verileri tablo kaydından çıkar.
                all_results.append({
                    "Parite": result["symbol"],
                    "Sinyal": result["signal"],
                    "Fiyat": result["price"],
                    "Yükseliş %": result["up_probability"] * 100,
                    "Düşüş %": result["down_probability"] * 100,
                    "Test doğruluğu %": result["accuracy"],
                    "Temel doğruluk %": result["baseline_accuracy"],
                    "Mum sayısı": result["candles"],
                    "Hacim (USDT)": ticker_map.get(
                        symbol, {}
                    ).get("quoteVolume", 0),
                })

            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

            progress.progress((index + 1) / len(selected_symbols))

            # API'ye aşırı yük bindirmemek için kısa ara.
            time.sleep(0.05)

        st.session_state.scan_results = all_results
        st.session_state.scan_errors = errors

        # Ayrıntılı analiz için seçilen ilk pariteyi sakla.
        if all_results:
            st.session_state.selected_symbol = all_results[0]["Parite"]

        status.success(
            f"Tarama tamamlandı. "
            f"{len(all_results)} parite analiz edildi."
        )

    except Exception as exc:
        st.error(f"Tarama başarısız: {exc}")


# =========================================================
# SONUÇ TABLOSU
# =========================================================

results = st.session_state.scan_results

if results:
    results_df = pd.DataFrame(results)

    # En güçlü sinyal olasılığına göre sırala.
    results_df["Sinyal gücü %"] = np.where(
        results_df["Sinyal"] == "BUY",
        results_df["Yükseliş %"],
        results_df["Düşüş %"]
    )

    results_df = results_df.sort_values(
        "Sinyal gücü %",
        ascending=False
    ).reset_index(drop=True)

    buy_count = int((results_df["Sinyal"] == "BUY").sum())
    sell_count = int((results_df["Sinyal"] == "SELL").sum())

    avg_accuracy = float(
        results_df["Test doğruluğu %"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Analiz edilen parite", len(results_df))
    col2.metric("BUY sinyali", buy_count)
    col3.metric("SELL sinyali", sell_count)
    col4.metric("Ortalama test doğruluğu", f"{avg_accuracy:.2f}%")

    st.subheader("🏆 Parite sıralaması")

    display_df = results_df[[
        "Parite",
        "Sinyal",
        "Fiyat",
        "Yükseliş %",
        "Düşüş %",
        "Sinyal gücü %",
        "Test doğruluğu %",
        "Temel doğruluk %",
        "Hacim (USDT)",
        "Mum sayısı",
    ]].copy()

    for column in [
        "Fiyat",
        "Yükseliş %",
        "Düşüş %",
        "Sinyal gücü %",
        "Test doğruluğu %",
        "Temel doğruluk %",
        "Hacim (USDT)",
    ]:
        display_df[column] = pd.to_numeric(
            display_df[column], errors="coerce"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fiyat": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "Yükseliş %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Düşüş %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Sinyal gücü %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Test doğruluğu %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Temel doğruluk %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Hacim (USDT)": st.column_config.NumberColumn(
                format="%.0f"
            ),
        }
    )

    st.download_button(
        "📥 Sonuçları CSV indir",
        data=display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="binance_buy_sell_scan.csv",
        mime="text/csv"
    )

    # =====================================================
    # PARİTE DETAYI
    # =====================================================

    st.divider()
    st.subheader("📈 Parite detay analizi")

    available_symbols = results_df["Parite"].tolist()

    current_selection = st.session_state.selected_symbol

    if current_selection not in available_symbols:
        current_selection = available_symbols[0]

    selected_symbol = st.selectbox(
        "Grafiğini görüntülemek istediğin parite",
        available_symbols,
        index=available_symbols.index(current_selection)
    )

    if st.button("Seçilen pariteyi analiz et"):
        try:
            with st.spinner(f"{selected_symbol} analiz ediliyor..."):
                detail = analyze_symbol(selected_symbol)
                st.session_state["detail_" + selected_symbol] = detail
        except Exception as exc:
            st.error(str(exc))

    detail_key = "detail_" + selected_symbol

    # İlk tarama sonrasında seçilen pariteyi otomatik yükle.
    if detail_key not in st.session_state:
        try:
            with st.spinner(
                f"{selected_symbol} için fiyat grafiği yükleniyor..."
            ):
                st.session_state[detail_key] = analyze_symbol(
                    selected_symbol
                )
        except Exception as exc:
            st.warning(f"Detay analizi alınamadı: {exc}")

    detail = st.session_state.get(detail_key)

    if detail:
        model = detail["model"]
        df = detail["df"]

        signal = model["signal"]

        if signal == "BUY":
            st.success("Güncel model sinyali: BUY")
        else:
            st.error("Güncel model sinyali: SELL")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Son fiyat",
            f'{detail["price"]:.8f} USDT'
        )

        p2.metric(
            "Yükseliş olasılığı",
            f'{model["up_probability"] * 100:.2f}%'
        )

        p3.metric(
            "Düşüş olasılığı",
            f'{model["down_probability"] * 100:.2f}%'
        )

        p4.metric(
            "Geçmiş test doğruluğu",
            f'{model["accuracy"]:.2f}%'
        )

        st.caption(
            f"Test örneği sayısı: {model['test_size']} | "
            f"Basit çoğunluk tahmini doğruluğu: "
            f"{model['baseline_accuracy']:.2f}%"
        )

        st.subheader(f"{selected_symbol} — 15 dakikalık fiyat grafiği")

        chart_df = df[["close_time", "close"]].tail(200).copy()
        chart_df = chart_df.set_index("close_time")
        chart_df.columns = ["Kapanış fiyatı"]

        st.line_chart(chart_df, use_container_width=True)

        st.subheader("Son tamamlanmış mumlar")

        candle_table = df.tail(20)[[
            "open_time", "open", "high", "low", "close", "volume"
        ]].copy()

        st.dataframe(
            candle_table.sort_values(
                "open_time", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.scan_errors:
        with st.expander(
            f"Atlanan pariteler ({len(st.session_state.scan_errors)})"
        ):
            for error in st.session_state.scan_errors:
                st.write("- " + error)

else:
    st.info(
        "Başlamak için soldaki 'Taramayı başlat' düğmesine bas."
    )

# =========================================================
# OTOMATİK YENİLEME
# =========================================================

if auto_refresh:
    st.caption(
        f"Otomatik yenileme açık. Yaklaşık {refresh_seconds} saniyede "
        "bir sayfayı yenileyebilirsin."
    )
    time.sleep(refresh_seconds)
    st.rerun()

st.divider()

st.caption(
    "Uyarı: BUY/SELL, modelin bir sonraki 15 dakikalık mumun "
    "yönüne ilişkin sınıflandırmasıdır. Gerçekleşecek fiyat hareketini "
    "garanti etmez. Geçmiş test doğruluğu gelecekteki başarı oranı "
    "anlamına gelmez. Bu uygulama emir göndermez."
)
