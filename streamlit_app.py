
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Binance All Coins Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance All Coins BUY / SELL Scanner")
st.caption(
    "Binance Spot USDT | 15 dakika | 1.000 tamamlanmış mum | "
    "Otomatik emir göndermez"
)

# =========================================================
# AYARLAR
# =========================================================

INTERVAL = "15m"
CANDLE_LIMIT = 1000
EPOCHS = 350
LEARNING_RATE = 0.05
TEST_FRACTION = 0.20
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.08

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

session = requests.Session()
session.headers.update({"User-Agent": "AllCoinsScanner/1.0"})


# =========================================================
# BINANCE API
# =========================================================

def api_get(path, params=None):
    errors = []

    for base in BASE_URLS:
        try:
            response = session.get(
                base + path,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429:
                time.sleep(2)
                errors.append(f"{base}: HTTP 429")
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "code" in data:
                if int(data.get("code", 0)) < 0:
                    errors.append(
                        f"{base}: {data.get('msg', 'Binance API hatası')}"
                    )
                    continue

            return data

        except Exception as exc:
            errors.append(f"{base}: {exc}")

    raise RuntimeError(
        f"Binance API erişilemedi. Son hatalar: {' | '.join(errors[-3:])}"
    )


@st.cache_data(ttl=600, show_spinner=False)
def get_all_spot_usdt_symbols():
    info = api_get("/api/v3/exchangeInfo")

    symbols = []

    for item in info.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            symbols.append(item["symbol"])

    if not symbols:
        raise RuntimeError("İşlem yapılabilir Spot USDT paritesi bulunamadı.")

    return sorted(set(symbols))


@st.cache_data(ttl=60, show_spinner=False)
def get_all_24h_tickers():
    return api_get("/api/v3/ticker/24hr")


@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol, interval="15m", limit=1000):
    raw = api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
    )

    if not isinstance(raw, list) or len(raw) < 100:
        raise ValueError("Yeterli mum verisi yok.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )
    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    # Yinelenen mumları kaldır.
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)

    # Fiyat ve mum mantığı doğrulaması.
    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )

    df = df.loc[valid].copy()

    # Henüz kapanmamış son mumu çıkar.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    df = df.sort_values("open_time").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Doğrulamadan sonra yeterli mum kalmadı.")

    # 15 dakikalık aralıkta eksik mum kontrolü.
    gaps = df["open_time"].diff().dropna()
    expected = pd.Timedelta(minutes=15)
    gap_count = int((gaps != expected).sum())

    df.attrs["gap_count"] = gap_count
    return df


# =========================================================
# GÖSTERGELER OLMADAN ÖZELLİK ÜRETİMİ
# RSI, EMA, MACD VE BOLLINGER KULLANILMAZ.
# =========================================================

def create_features(df):
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    eps = 1e-12

    returns = np.zeros(len(close), dtype=float)
    returns[1:] = close[1:] / np.maximum(close[:-1], eps) - 1

    ret = pd.Series(returns)
    vol = pd.Series(volume)
    log_vol = np.log1p(np.maximum(volume, 0))

    features = pd.DataFrame({
        "ret1": ret,
        "ret2": ret.rolling(2).sum(),
        "ret3": ret.rolling(3).sum(),
        "ret5": ret.rolling(5).sum(),
        "ret10": ret.rolling(10).sum(),
        "ret20": ret.rolling(20).sum(),
        "std5": ret.rolling(5).std(),
        "std10": ret.rolling(10).std(),
        "std20": ret.rolling(20).std(),
        "body": (close - open_) / np.maximum(open_, eps),
        "range": (high - low) / np.maximum(close, eps),
        "upper_wick": (
            high - np.maximum(open_, close)
        ) / np.maximum(close, eps),
        "lower_wick": (
            np.minimum(open_, close) - low
        ) / np.maximum(close, eps),
        "volume_change": pd.Series(log_vol).diff(),
        "volume_ratio": vol / (vol.rolling(20).mean() + eps),
    })

    return features.replace([np.inf, -np.inf], np.nan)


# =========================================================
# NUMPY LOJİSTİK REGRESYON
# LSTM DEĞİLDİR; PYTORCH GEREKTİRMEZ.
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(X, y, epochs=EPOCHS, lr=LEARNING_RATE):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    positives = max(np.sum(y == 1), 1)
    negatives = max(np.sum(y == 0), 1)

    sample_weights = np.where(
        y == 1,
        n / (2.0 * positives),
        n / (2.0 * negatives)
    )

    for _ in range(epochs):
        probs = sigmoid(X @ weights + bias)
        errors = (probs - y) * sample_weights

        grad_w = X.T @ errors / n + 0.001 * weights
        grad_b = float(np.mean(errors))

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def calculate_accuracy(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y_true) * 100)


# =========================================================
# GEÇMİŞ TEST VE SON SİNYAL
# =========================================================

def analyze(df):
    features = create_features(df)
    close = df["close"].to_numpy(dtype=float)

    # Her mum için sonraki mumun yönü.
    future_returns = np.full(len(close), np.nan)
    future_returns[:-1] = (
        close[1:] - close[:-1]
    ) / np.maximum(close[:-1], 1e-12)

    targets = (future_returns > 0).astype(float)

    data = features.copy()
    data["target"] = targets
    data["future_return"] = future_returns
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)

    if len(data) < 150:
        raise ValueError("Model için yeterli temiz veri yok.")

    feature_cols = list(features.columns)
    X = data[feature_cols].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=float)

    # Zamana göre ayır: gelecek geçmiş eğitim verisine karışmaz.
    split = int(len(data) * (1 - TEST_FRACTION))
    split = max(100, min(split, len(data) - 25))

    X_train_raw = X[:split]
    y_train = y[:split]
    X_test_raw = X[split:]
    y_test = y[split:]

    # Ölçekleme parametreleri sadece eğitim bölümünden öğrenilir.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-9] = 1.0

    X_train = np.clip((X_train_raw - mean) / std, -10, 10)
    X_test = np.clip((X_test_raw - mean) / std, -10, 10)

    weights, bias = train_logistic(X_train, y_train)

    test_probs = sigmoid(X_test @ weights + bias)
    accuracy = calculate_accuracy(y_test, test_probs)

    buy_mask = test_probs >= 0.5
    sell_mask = ~buy_mask

    buy_accuracy = (
        float(np.mean(y_test[buy_mask] == 1) * 100)
        if buy_mask.any() else np.nan
    )
    sell_accuracy = (
        float(np.mean(y_test[sell_mask] == 0) * 100)
        if sell_mask.any() else np.nan
    )

    # Basit çoğunluk tahminiyle kıyaslama.
    baseline = float(
        max(np.mean(y_test == 0), np.mean(y_test == 1)) * 100
    )

    # Son satırın özelliği için aynı ölçekleme.
    last_raw = X[-1:]
    last_x = np.clip((last_raw - mean) / std, -10, 10)
    up_prob = float(sigmoid(last_x @ weights + bias)[0])

    signal = "BUY" if up_prob >= 0.5 else "SELL"

    return {
        "signal": signal,
        "up_prob": up_prob,
        "down_prob": 1 - up_prob,
        "accuracy": accuracy,
        "buy_accuracy": buy_accuracy,
        "sell_accuracy": sell_accuracy,
        "baseline": baseline,
        "test_count": len(y_test),
        "dataset": data,
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "std": std,
        "feature_cols": feature_cols,
    }


# =========================================================
# KENAR ÇUBUĞU
# =========================================================

with st.sidebar:
    st.header("Tarama ayarları")

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
        help="0 seçilirse hacim eşiği uygulanmaz."
    )

    max_coins = st.number_input(
        "Maksimum coin sayısı (0 = tüm coinler)",
        min_value=0,
        max_value=2000,
        value=0,
        step=50
    )

    st.caption(
        "Tüm coinleri taramak API sınırları nedeniyle uzun sürebilir."
    )

    scan_button = st.button(
        "🔍 Tüm coinleri tara",
        type="primary",
        use_container_width=True
    )

    force_refresh = st.button(
        "♻️ Önbelleği temizle",
        use_container_width=True
    )

if force_refresh:
    get_all_spot_usdt_symbols.clear()
    get_all_24h_tickers.clear()
    get_klines.clear()
    st.rerun()


# =========================================================
# TARAYICI
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "errors" not in st.session_state:
    st.session_state.errors = []

if "details" not in st.session_state:
    st.session_state.details = {}

if scan_button:
    results = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    try:
        with st.spinner("Binance Spot USDT pariteleri alınıyor..."):
            symbols = get_all_spot_usdt_symbols()
            tickers = get_all_24h_tickers()

        ticker_map = {
            t.get("symbol"): t
            for t in tickers
            if t.get("symbol")
        }

        candidates = []

        for symbol in symbols:
            ticker = ticker_map.get(symbol)
            if ticker is None:
                continue

            try:
                volume = float(ticker.get("quoteVolume", 0))
                price = float(ticker.get("lastPrice", 0))
            except (ValueError, TypeError):
                continue

            if price <= 0 or volume < min_volume:
                continue

            candidates.append((symbol, volume, price))

        candidates.sort(key=lambda item: item[1], reverse=True)

        if max_coins > 0:
            candidates = candidates[:int(max_coins)]

        if not candidates:
            raise RuntimeError("Tarama koşullarına uyan coin bulunamadı.")

        st.info(f"{len(candidates)} parite sıraya alındı.")

        for i, (symbol, quote_volume, ticker_price) in enumerate(candidates):
            status.text(
                f"{i + 1}/{len(candidates)} — {symbol} analiz ediliyor"
            )

            try:
                df = get_klines(symbol, INTERVAL, CANDLE_LIMIT)
                model = analyze(df)

                results.append({
                    "Coin": symbol,
                    "Sinyal": model["signal"],
                    "Fiyat": float(df["close"].iloc[-1]),
                    "Yükseliş %": model["up_prob"] * 100,
                    "Düşüş %": model["down_prob"] * 100,
                    "Sinyal gücü %": max(
                        model["up_prob"], model["down_prob"]
                    ) * 100,
                    "Test doğruluğu %": model["accuracy"],
                    "BUY test doğruluğu %": model["buy_accuracy"],
                    "SELL test doğruluğu %": model["sell_accuracy"],
                    "Temel doğruluk %": model["baseline"],
                    "Test örneği": model["test_count"],
                    "Mum sayısı": len(df),
                    "Eksik aralık": int(df.attrs.get("gap_count", 0)),
                    "24s hacim USDT": quote_volume,
                })

                # Grafiği sonradan görüntülemek için detayları sakla.
                st.session_state.details[symbol] = {
                    "df": df,
                    "model": model
                }

            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

            progress.progress((i + 1) / len(candidates))
            time.sleep(REQUEST_DELAY)

        st.session_state.results = results
        st.session_state.errors = errors
        status.success(
            f"Tarama tamamlandı: {len(results)} başarılı, "
            f"{len(errors)} hatalı/atlanan parite."
        )

    except Exception as exc:
        st.error(f"Tarama başlatılamadı: {exc}")


# =========================================================
# SONUÇLAR
# =========================================================

if st.session_state.results:
    result_df = pd.DataFrame(st.session_state.results)

    result_df = result_df.sort_values(
        "Sinyal gücü %",
        ascending=False
    ).reset_index(drop=True)

    buys = int((result_df["Sinyal"] == "BUY").sum())
    sells = int((result_df["Sinyal"] == "SELL").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analiz edilen coin", len(result_df))
    c2.metric("BUY", buys)
    c3.metric("SELL", sells)
    c4.metric(
        "Ortalama test doğruluğu",
        f'{result_df["Test doğruluğu %"].mean():.2f}%'
    )

    st.subheader("🏆 Tüm coinlerin sinyal sıralaması")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fiyat": st.column_config.NumberColumn(format="%.8f"),
            "Yükseliş %": st.column_config.NumberColumn(format="%.2f%%"),
            "Düşüş %": st.column_config.NumberColumn(format="%.2f%%"),
            "Sinyal gücü %": st.column_config.NumberColumn(format="%.2f%%"),
            "Test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "BUY test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "SELL test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "Temel doğruluk %": st.column_config.NumberColumn(format="%.2f%%"),
            "24s hacim USDT": st.column_config.NumberColumn(format="%.0f"),
        }
    )

    st.download_button(
        "📥 Tüm sonuçları CSV indir",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="binance_all_coins_buy_sell.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("📈 Coin grafiği ve ayrıntıları")

    symbols_available = result_df["Coin"].tolist()

    selected_symbol = st.selectbox(
        "Coin seç",
        symbols_available
    )

    if selected_symbol not in st.session_state.details:
        if st.button("Seçilen coin için ayrıntı yükle"):
            try:
                with st.spinner(f"{selected_symbol} yükleniyor..."):
                    df = get_klines(selected_symbol, INTERVAL, CANDLE_LIMIT)
                    model = analyze(df)
                    st.session_state.details[selected_symbol] = {
                        "df": df,
                        "model": model
                    }
            except Exception as exc:
                st.error(str(exc))

    detail = st.session_state.details.get(selected_symbol)

    if detail:
        df = detail["df"]
        model = detail["model"]

        if model["signal"] == "BUY":
            st.success("Model sinyali: BUY")
        else:
            st.error("Model sinyali: SELL")

        a, b, c, d = st.columns(4)
        a.metric("Son kapanış", f'{df["close"].iloc[-1]:.8f}')
        b.metric("Yükseliş olasılığı", f'{model["up_prob"] * 100:.2f}%')
        c.metric("Düşüş olasılığı", f'{model["down_prob"] * 100:.2f}%')
        d.metric("Test doğruluğu", f'{model["accuracy"]:.2f}%')

        e, f, g = st.columns(3)
        e.metric(
            "BUY tahminlerinin isabeti",
            "N/A" if np.isnan(model["buy_accuracy"])
            else f'{model["buy_accuracy"]:.2f}%'
        )
        f.metric(
            "SELL tahminlerinin isabeti",
            "N/A" if np.isnan(model["sell_accuracy"])
            else f'{model["sell_accuracy"]:.2f}%'
        )
        g.metric(
            "Basit tahmin karşılaştırması",
            f'{model["baseline"]:.2f}%'
        )

        chart = df[["close_time", "close"]].tail(200).copy()
        chart = chart.set_index("close_time")
        chart.columns = ["Kapanış fiyatı"]
        st.line_chart(chart, use_container_width=True)

        st.caption(
            f"Son {len(df)} tamamlanmış mum kullanıldı. "
            f"Veri aralığı sorunu sayısı: "
            f"{df.attrs.get('gap_count', 0)}. "
            "Eksik aralıklar tespit edilir ancak otomatik doldurulmaz."
        )

        st.subheader("Son 20 mum")
        st.dataframe(
            df.tail(20)[[
                "open_time", "open", "high", "low", "close", "volume"
            ]].sort_values("open_time", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.errors:
        with st.expander(
            f"Analiz edilemeyen coinler ({len(st.session_state.errors)})"
        ):
            st.write(
                "Bir coin atlandığında bunun sebebi API erişimi, "
                "yetersiz geçmiş veri veya veri doğrulama hatası olabilir."
            )
            for error in st.session_state.errors:
                st.write("- " + error)

else:
    st.info(
        "Tüm uygun coinleri taramak için sol menüden "
        "'Tüm coinleri tara' düğmesine bas."
    )

st.divider()
st.warning(
    "Önemli: BUY/SELL sinyali garanti değildir. Olasılıklar model "
    "çıktısıdır; gerçek piyasa olasılığı veya kesin kâr tahmini değildir. "
    "Test doğruluğu geçmiş örneklerde ölçülür. Bu uygulama emir göndermez."
)
