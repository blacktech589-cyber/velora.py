
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# BINANCE USD-M FUTURES SCANNER
# Public API only. No API keys. No orders.
# =========================================================

st.set_page_config(
    page_title="Binance Futures Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Binance USDⓈ-M Futures Scanner")
st.caption(
    "USDT perpetual | 15 dakikalık mumlar | "
    "BUY / SELL | Sadece analiz, emir gönderilmez"
)

INTERVAL = "15m"
CANDLE_LIMIT = 1000
EPOCHS = 350
LEARNING_RATE = 0.05
TEST_FRACTION = 0.20
TIMEOUT = 20
REQUEST_DELAY = 0.10

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 FuturesScanner/1.0",
    "Accept": "application/json",
})


# =========================================================
# API CONNECTION AND DIAGNOSTICS
# =========================================================

def api_get(path, params=None):
    errors = []

    for base in BASE_URLS:
        url = base + path

        try:
            response = session.get(
                url,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code != 200:
                errors.append(
                    f"{base}: HTTP {response.status_code}; "
                    f"body={response.text[:160]!r}"
                )
                continue

            try:
                data = response.json()
            except ValueError:
                errors.append(
                    f"{base}: JSON olmayan yanıt; "
                    f"Content-Type="
                    f"{response.headers.get('Content-Type', '')!r}; "
                    f"body={response.text[:160]!r}"
                )
                continue

            if isinstance(data, dict) and "code" in data:
                try:
                    code = int(data.get("code", 0))
                except (TypeError, ValueError):
                    code = 0

                if code < 0:
                    errors.append(
                        f"{base}: Binance {code}: "
                        f"{data.get('msg', 'API hatası')}"
                    )
                    continue

            return data

        except requests.RequestException as exc:
            errors.append(
                f"{base}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Binance Futures API erişilemedi:\n"
        + "\n".join(errors[-5:])
    )


# =========================================================
# SYMBOLS AND MARKET DATA
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_futures_symbols():
    info = api_get("/fapi/v1/exchangeInfo")
    symbols = []

    for item in info.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("contractType") == "PERPETUAL"
        ):
            symbols.append(item["symbol"])

    if not symbols:
        raise RuntimeError(
            "İşlem yapılabilir USDT perpetual sözleşme bulunamadı."
        )

    return sorted(set(symbols))


@st.cache_data(ttl=60, show_spinner=False)
def get_tickers():
    data = api_get("/fapi/v1/ticker/24hr")
    if not isinstance(data, list):
        raise RuntimeError("24 saatlik ticker verisi beklenen formatta değil.")
    return data


@st.cache_data(ttl=60, show_spinner=False)
def get_funding_rates():
    data = api_get("/fapi/v1/premiumIndex")
    return data if isinstance(data, list) else [data]


@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol, interval="15m", limit=1000):
    raw = api_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not isinstance(raw, list) or len(raw) < 100:
        raise ValueError("Yeterli Futures mum verisi yok.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades",
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
    df = df.drop_duplicates(
        subset=["open_time"], keep="last"
    )
    df = df.sort_values("open_time").reset_index(drop=True)

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

    # Henüz tamamlanmamış mumu dahil etme.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    df = df.sort_values("open_time").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "Veri doğrulamasından sonra yeterli mum kalmadı."
        )

    gaps = df["open_time"].diff().dropna()
    df.attrs["gap_count"] = int(
        (gaps != pd.Timedelta(minutes=15)).sum()
    )

    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_open_interest(symbol):
    return api_get(
        "/fapi/v1/openInterest",
        {"symbol": symbol},
    )


# =========================================================
# FEATURES
# RSI / EMA / MACD / BOLLINGER KULLANILMAZ.
# =========================================================

def create_features(df):
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    eps = 1e-12

    returns = np.zeros(len(close), dtype=float)
    returns[1:] = (
        close[1:] / np.maximum(close[:-1], eps)
    ) - 1.0

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
# NUMPY LOGISTIC REGRESSION
# LSTM DEĞİLDİR.
# =========================================================

def sigmoid(z):
    return 1.0 / (
        1.0 + np.exp(-np.clip(z, -30, 30))
    )


def fit_model(X, y):
    n, p = X.shape
    weights = np.zeros(p, dtype=float)
    bias = 0.0

    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)

    sample_weights = np.where(
        y == 1,
        n / (2.0 * positives),
        n / (2.0 * negatives),
    )

    for _ in range(EPOCHS):
        probs = sigmoid(X @ weights + bias)
        errors = (probs - y) * sample_weights

        grad_w = X.T @ errors / n + 0.001 * weights
        grad_b = float(np.mean(errors))

        weights -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * grad_b

    return weights, bias


def analyze_model(df):
    features = create_features(df)
    close = df["close"].to_numpy(dtype=float)

    # Her mumun hedefi bir sonraki mumun yönüdür.
    future_returns = np.full(len(close), np.nan)
    future_returns[:-1] = (
        close[1:] - close[:-1]
    ) / np.maximum(close[:-1], 1e-12)

    targets = (future_returns > 0).astype(float)

    data = features.copy()
    data["target"] = targets
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)

    if len(data) < 150:
        raise ValueError("Model eğitimi için yeterli veri yok.")

    feature_cols = list(features.columns)
    X = data[feature_cols].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=float)

    # Zamana göre ayır; veriyi karıştırma.
    split = int(len(data) * (1.0 - TEST_FRACTION))
    split = max(100, min(split, len(data) - 25))

    X_train_raw = X[:split]
    y_train = y[:split]
    X_test_raw = X[split:]
    y_test = y[split:]

    # Ölçekleyiciyi sadece eğitim bölümünde hesapla.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-9] = 1.0

    X_train = np.clip(
        (X_train_raw - mean) / std, -10, 10
    )
    X_test = np.clip(
        (X_test_raw - mean) / std, -10, 10
    )

    weights, bias = fit_model(X_train, y_train)

    test_probs = sigmoid(X_test @ weights + bias)
    test_preds = (test_probs >= 0.5).astype(int)
    accuracy = float(np.mean(test_preds == y_test) * 100)

    buy_mask = test_preds == 1
    sell_mask = test_preds == 0

    buy_accuracy = (
        float(np.mean(y_test[buy_mask] == 1) * 100)
        if buy_mask.any() else np.nan
    )
    sell_accuracy = (
        float(np.mean(y_test[sell_mask] == 0) * 100)
        if sell_mask.any() else np.nan
    )

    baseline = float(
        max(np.mean(y_test == 0), np.mean(y_test == 1)) * 100
    )

    last_x = np.clip(
        (X[-1:] - mean) / std, -10, 10
    )
    up_prob = float(sigmoid(last_x @ weights + bias)[0])

    return {
        "signal": "BUY" if up_prob >= 0.5 else "SELL",
        "up_prob": up_prob,
        "down_prob": 1.0 - up_prob,
        "accuracy": accuracy,
        "buy_accuracy": buy_accuracy,
        "sell_accuracy": sell_accuracy,
        "baseline": baseline,
        "test_count": len(y_test),
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Tarama ayarları")

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
    )

    max_coins = st.number_input(
        "Maksimum coin (0 = tümü)",
        min_value=0,
        max_value=2000,
        value=0,
        step=50,
    )

    st.caption(
        "Tüm sözleşmelerin taranması zaman alabilir. "
        "API limitleri nedeniyle bazıları atlanabilir."
    )

    test_connection = st.button(
        "🔌 Futures API bağlantısını test et",
        use_container_width=True,
    )

    scan = st.button(
        "🔍 Futures coinlerini tara",
        type="primary",
        use_container_width=True,
    )

    clear_cache = st.button(
        "♻️ Verileri yenile",
        use_container_width=True,
    )


# =========================================================
# CONNECTION TEST
# =========================================================

if test_connection:
    try:
        ping = api_get("/fapi/v1/ping")
        st.success("Futures API ping başarılı.")
        st.json(ping)

        info = api_get("/fapi/v1/exchangeInfo")
        st.success(
            f"Exchange bilgileri alındı: "
            f"{len(info.get('symbols', []))} sözleşme."
        )

    except Exception as exc:
        st.error(str(exc))


if clear_cache:
    get_futures_symbols.clear()
    get_tickers.clear()
    get_funding_rates.clear()
    get_klines.clear()
    get_open_interest.clear()
    st.rerun()


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "errors" not in st.session_state:
    st.session_state.errors = []

if "details" not in st.session_state:
    st.session_state.details = {}


# =========================================================
# SCAN ALL ELIGIBLE CONTRACTS
# =========================================================

if scan:
    results = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    try:
        with st.spinner("Futures piyasası bilgileri alınıyor..."):
            symbols = get_futures_symbols()
            tickers = get_tickers()
            funding_data = get_funding_rates()

        ticker_map = {
            item["symbol"]: item
            for item in tickers
            if item.get("symbol")
        }

        funding_map = {
            item["symbol"]: item
            for item in funding_data
            if item.get("symbol")
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
            raise RuntimeError(
                "Tarama koşullarına uyan Futures paritesi bulunamadı."
            )

        st.info(f"{len(candidates)} parite tarama sırasına alındı.")

        for i, (symbol, quote_volume, ticker_price) in enumerate(candidates):
            status.text(
                f"{i + 1}/{len(candidates)} — {symbol} analiz ediliyor"
            )

            try:
                df = get_klines(symbol, INTERVAL, CANDLE_LIMIT)
                model = analyze_model(df)

                funding = funding_map.get(symbol, {})

                try:
                    funding_rate = float(
                        funding.get("lastFundingRate", 0) or 0
                    )
                except (ValueError, TypeError):
                    funding_rate = 0.0

                try:
                    mark_price = float(
                        funding.get("markPrice", ticker_price)
                        or ticker_price
                    )
                except (ValueError, TypeError):
                    mark_price = ticker_price

                results.append({
                    "Coin": symbol,
                    "Sinyal": model["signal"],
                    "Fiyat": float(df["close"].iloc[-1]),
                    "Mark fiyatı": mark_price,
                    "Yükseliş %": model["up_prob"] * 100,
                    "Düşüş %": model["down_prob"] * 100,
                    "Sinyal gücü %": max(
                        model["up_prob"], model["down_prob"]
                    ) * 100,
                    "Test doğruluğu %": model["accuracy"],
                    "BUY test doğruluğu %": model["buy_accuracy"],
                    "SELL test doğruluğu %": model["sell_accuracy"],
                    "Temel doğruluk %": model["baseline"],
                    "Funding rate %": funding_rate * 100,
                    "24s değişim %": float(
                        ticker_map[symbol].get("priceChangePercent", 0)
                    ),
                    "24s hacim USDT": quote_volume,
                    "Test örneği": model["test_count"],
                    "Mum sayısı": len(df),
                    "Eksik aralık": int(df.attrs.get("gap_count", 0)),
                })

                st.session_state.details[symbol] = {
                    "df": df,
                    "model": model,
                }

            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

            progress.progress((i + 1) / len(candidates))
            time.sleep(REQUEST_DELAY)

        st.session_state.results = results
        st.session_state.errors = errors

        status.success(
            f"Tarama tamamlandı: {len(results)} başarılı, "
            f"{len(errors)} atlanan/başarısız."
        )

    except Exception as exc:
        st.error(f"Tarama başarısız:\n{exc}")


# =========================================================
# RESULTS TABLE
# =========================================================

if st.session_state.results:
    result_df = pd.DataFrame(st.session_state.results)
    result_df = result_df.sort_values(
        "Sinyal gücü %",
        ascending=False,
    ).reset_index(drop=True)

    buys = int((result_df["Sinyal"] == "BUY").sum())
    sells = int((result_df["Sinyal"] == "SELL").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analiz edilen coin", len(result_df))
    c2.metric("BUY / Long yönü", buys)
    c3.metric("SELL / Short yönü", sells)
    c4.metric(
        "Ortalama test doğruluğu",
        f'{result_df["Test doğruluğu %"].mean():.2f}%'
    )

    st.subheader("🏆 Futures parite sıralaması")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fiyat": st.column_config.NumberColumn(format="%.8f"),
            "Mark fiyatı": st.column_config.NumberColumn(format="%.8f"),
            "Yükseliş %": st.column_config.NumberColumn(format="%.2f%%"),
            "Düşüş %": st.column_config.NumberColumn(format="%.2f%%"),
            "Sinyal gücü %": st.column_config.NumberColumn(format="%.2f%%"),
            "Test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "BUY test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "SELL test doğruluğu %": st.column_config.NumberColumn(format="%.2f%%"),
            "Temel doğruluk %": st.column_config.NumberColumn(format="%.2f%%"),
            "Funding rate %": st.column_config.NumberColumn(format="%.5f%%"),
            "24s değişim %": st.column_config.NumberColumn(format="%.2f%%"),
            "24s hacim USDT": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    st.download_button(
        "📥 Futures sonuçlarını CSV indir",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="binance_futures_buy_sell.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📈 Coin detayları")

    symbols_available = result_df["Coin"].tolist()

    selected_symbol = st.selectbox(
        "Futures paritesi seç",
        symbols_available,
    )

    if st.button("Seçilen coin detaylarını yenile"):
        try:
            with st.spinner(f"{selected_symbol} güncelleniyor..."):
                df = get_klines(
                    selected_symbol, INTERVAL, CANDLE_LIMIT
                )
                model = analyze_model(df)

                st.session_state.details[selected_symbol] = {
                    "df": df,
                    "model": model,
                }

        except Exception as exc:
            st.error(str(exc))

    detail = st.session_state.details.get(selected_symbol)

    if detail:
        df = detail["df"]
        model = detail["model"]

        if model["signal"] == "BUY":
            st.success("Sinyal: BUY — yükseliş/long yönü")
        else:
            st.error("Sinyal: SELL — düşüş/short yönü")

        a, b, c, d = st.columns(4)
        a.metric("Son kapanış", f'{df["close"].iloc[-1]:.8f}')
        b.metric("Yükseliş olasılığı", f'{model["up_prob"] * 100:.2f}%')
        c.metric("Düşüş olasılığı", f'{model["down_prob"] * 100:.2f}%')
        d.metric("Test doğruluğu", f'{model["accuracy"]:.2f}%')

        e, f, g = st.columns(3)
        e.metric(
            "BUY test isabeti",
            "N/A" if np.isnan(model["buy_accuracy"])
            else f'{model["buy_accuracy"]:.2f}%'
        )
        f.metric(
            "SELL test isabeti",
            "N/A" if np.isnan(model["sell_accuracy"])
            else f'{model["sell_accuracy"]:.2f}%'
        )
        g.metric(
            "Basit temel doğruluk",
            f'{model["baseline"]:.2f}%'
        )

        try:
            oi = get_open_interest(selected_symbol)
            st.metric(
                "Güncel open interest",
                f'{float(oi.get("openInterest", 0)):,.4f}'
            )
        except Exception as exc:
            st.caption(f"Open interest alınamadı: {exc}")

        chart = df[["close_time", "close"]].tail(200).copy()
        chart = chart.set_index("close_time")
        chart.columns = ["Kapanış fiyatı"]

        st.subheader(f"{selected_symbol} — 15 dakikalık grafik")
        st.line_chart(chart, use_container_width=True)

        st.caption(
            f"Tamamlanmış mum: {len(df)} | "
            f"Bulunan eksik zaman aralığı: "
            f"{df.attrs.get('gap_count', 0)}"
        )

        st.subheader("Son 20 mum")
        st.dataframe(
            df.tail(20)[[
                "open_time", "open", "high", "low", "close", "volume"
            ]].sort_values("open_time", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.errors:
        with st.expander(
            f"Atlanan/başarısız coinler "
            f"({len(st.session_state.errors)})"
        ):
            for error in st.session_state.errors:
                st.write("- " + error)

else:
    st.info(
        "Tarama için soldaki 'Futures coinlerini tara' düğmesine bas."
    )

st.divider()

st.warning(
    "BUY/SELL sinyalleri garanti değildir. Gösterilen olasılıklar "
    "model çıktılarıdır; kalibre edilmiş gerçek piyasa olasılıkları "
    "oldukları garanti edilmez. Geçmiş test doğruluğu gelecekteki "
    "başarıyı garanti etmez. Kaldıraç ve likidasyon riski vardır. "
    "Bu uygulama emir göndermez."
)
