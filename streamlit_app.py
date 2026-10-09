
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# AYARLAR
# =========================================================

st.set_page_config(
    page_title="Spot Crypto Dip Scanner",
    page_icon="📉",
    layout="wide",
)

BINANCE = "https://api.binance.com"
GATE = "https://api.gateio.ws/api/v4"

INTERVAL_SECONDS = 15 * 60
BARS_10_DAYS = 10 * 24 * 4
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": "SpotCryptoDipScanner/1.0",
    "Accept": "application/json",
}

session = requests.Session()
session.headers.update(HEADERS)


# =========================================================
# HTTP
# =========================================================

def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# =========================================================
# BINANCE AKTIF SPOT USDT MARKETLERI
# =========================================================

def get_binance_markets():
    info = get_json(f"{BINANCE}/api/v3/exchangeInfo")

    markets = []

    for item in info.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        markets.append({
            "symbol": symbol,
            "market_id": symbol,
            "base": item.get("baseAsset", ""),
            "provider": "Binance",
        })

    if not markets:
        raise RuntimeError("Aktif Binance Spot USDT marketi bulunamadı.")

    return markets


def get_binance_tickers():
    data = get_json(f"{BINANCE}/api/v3/ticker/24hr")

    tickers = {}

    for item in data:
        symbol = item.get("symbol")

        try:
            tickers[symbol] = {
                "price": float(item["lastPrice"]),
                "volume": float(item["quoteVolume"]),
                "change_24h": float(item["priceChangePercent"]),
            }
        except (ValueError, TypeError, KeyError):
            continue

    return tickers


# =========================================================
# GATE.IO AKTIF SPOT USDT MARKETLERI
# =========================================================

def get_gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")

    markets = []

    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")
        quote = item.get("quote", "")

        if not market_id or not base or quote != "USDT":
            continue

        status = str(item.get("trade_status", "")).lower()

        # Bilinmeyen durumları aktif kabul etme.
        if status != "tradable":
            continue

        if item.get("delisted") is True:
            continue

        markets.append({
            "symbol": f"{base}USDT",
            "market_id": market_id,
            "base": base,
            "provider": "Gate.io",
        })

    if not markets:
        raise RuntimeError(
            "Gate.io API aktif Spot USDT marketi döndürmedi."
        )

    return markets


def get_gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")

    tickers = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            tickers[market_id] = {
                "price": float(item["last"]),
                "volume": float(item["quote_volume"]),
                "change_24h": float(item["change_percentage"]),
            }
        except (ValueError, TypeError, KeyError):
            continue

    return tickers


# =========================================================
# MUM VERISI
# =========================================================

def load_candles(market, limit=1000):
    provider = market["provider"]
    market_id = market["market_id"]

    if provider == "Binance":
        raw = get_json(
            f"{BINANCE}/api/v3/klines",
            params={
                "symbol": market_id,
                "interval": "15m",
                "limit": min(limit, 1000),
            },
        )

        rows = []

        for candle in raw:
            rows.append({
                "timestamp": int(candle[0]) // 1000,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            })

    elif provider == "Gate.io":
        raw = get_json(
            f"{GATE}/spot/candlesticks",
            params={
                "currency_pair": market_id,
                "interval": "15m",
                "limit": min(limit, 1000),
            },
        )

        rows = []

        for candle in raw:
            if len(candle) < 6:
                continue

            try:
                rows.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[5]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "close": float(candle[2]),
                    "volume": float(candle[6])
                    if len(candle) > 6 else 0.0,
                })
            except (ValueError, TypeError, IndexError):
                continue

    else:
        raise RuntimeError("Bilinmeyen veri sağlayıcı.")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Mum verisi boş.")

    df = df.dropna().drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    # Yalnızca tamamlanmış 15 dakikalık mumları tut.
    now_seconds = int(datetime.now(timezone.utc).timestamp())
    current_candle_start = (
        now_seconds // INTERVAL_SECONDS
    ) * INTERVAL_SECONDS

    df = df[
        df["timestamp"] < current_candle_start
    ].copy()

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ].reset_index(drop=True)

    return df


# =========================================================
# DIP VE DIPTEN DONUS ANALIZI
# =========================================================

def analyze_bottom(df, max_distance_pct=10.0):
    if len(df) < BARS_10_DAYS:
        return {
            "valid": False,
            "reason": "Son 10 gün için 960 tamamlanmış mum yok.",
        }

    recent = df.tail(BARS_10_DAYS).reset_index(drop=True)

    bottom_position = int(
        recent["low"].to_numpy().argmin()
    )

    bottom_price = float(
        recent.iloc[bottom_position]["low"]
    )

    current_price = float(recent.iloc[-1]["close"])

    distance_pct = (
        (current_price - bottom_price) / bottom_price * 100
        if bottom_price > 0 else np.nan
    )

    last_three = recent.tail(3)

    closes = last_three["close"].to_numpy()
    opens = last_three["open"].to_numpy()

    three_rising = bool(
        len(closes) == 3
        and closes[0] < closes[1] < closes[2]
    )

    latest_green = bool(
        closes[-1] > opens[-1]
    )

    bottom_before_last_three = (
        bottom_position < len(recent) - 3
    )

    near_bottom = bool(
        np.isfinite(distance_pct)
        and 0 <= distance_pct <= max_distance_pct
    )

    reversal = bool(
        three_rising
        and latest_green
        and bottom_before_last_three
        and near_bottom
    )

    # Son mumlar dipten yükselmiyorsa BUY etiketi üretme.
    if reversal:
        signal = "DIPTEN DÖNÜŞ"
    elif three_rising and latest_green:
        signal = "YÜKSELİYOR - DİPTEN UZAK"
    else:
        signal = "BEKLE"

    return {
        "valid": True,
        "bottom_price": bottom_price,
        "current_price": current_price,
        "distance_pct": distance_pct,
        "bottom_position": bottom_position,
        "three_rising": three_rising,
        "latest_green": latest_green,
        "near_bottom": near_bottom,
        "reversal": reversal,
        "signal": signal,
        "bars": len(recent),
        "bottom_time": datetime.fromtimestamp(
            int(recent.iloc[bottom_position]["timestamp"]),
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC"),
    }


# =========================================================
# BIR MARKETI TARA
# =========================================================

def scan_market(market, ticker, max_distance_pct):
    df = load_candles(market, limit=1000)

    result = analyze_bottom(
        df,
        max_distance_pct=max_distance_pct,
    )

    if not result.get("valid"):
        raise RuntimeError(result.get("reason", "Veri yetersiz."))

    # Fiyatı mum verisinden al; farklı marketten eşleştirme yapma.
    candle_price = float(result["current_price"])
    ticker_price = float(ticker.get("price", 0.0))

    # Ticker ve mum fiyatları aynı markete ait olmalıdır.
    # Büyük farkta veriyi sessizce birleştirme.
    price_difference_pct = np.nan

    if candle_price > 0 and ticker_price > 0:
        price_difference_pct = abs(
            ticker_price - candle_price
        ) / candle_price * 100

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "provider": market["provider"],
        "price_15m_close": candle_price,
        "ticker_price": ticker_price,
        "ticker_difference_pct": price_difference_pct,
        "volume_24h_usdt": ticker.get("volume", 0.0),
        "change_24h_pct": ticker.get("change_24h", 0.0),
        "bottom_10d": result["bottom_price"],
        "bottom_time_utc": result["bottom_time"],
        "distance_from_bottom_pct": result["distance_pct"],
        "three_rising_candles": result["three_rising"],
        "latest_candle_green": result["latest_green"],
        "near_bottom": result["near_bottom"],
        "signal": result["signal"],
        "is_reversal": result["reversal"],
        "bars_10d": result["bars"],
    }


# =========================================================
# STREAMLIT ARAYUZU
# =========================================================

st.title("📉 Crypto Spot Dipten Dönüş Tarayıcı")

st.write(
    "Aktif Spot USDT marketlerini kontrol eder. Son 10 günlük "
    "en düşük fiyatı bulur ve dipten sonra yükselen coinleri "
    "listeler. Emir göndermez."
)

st.warning(
    "Düşüş sonrası yükseliş garantili değildir. Bu uygulama "
    "teknik tarama yapar; yatırım tavsiyesi vermez."
)

with st.sidebar:
    st.header("Tarama Ayarları")

    max_distance_pct = st.slider(
        "Dipten izin verilen maksimum uzaklık (%)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0,
        value=10000,
        step=10000,
    )

    scan_count = st.number_input(
        "Taranacak market sayısı",
        min_value=1,
        max_value=2000,
        value=50,
        step=25,
    )

    mode = st.selectbox(
        "Market sıralaması",
        ["En yüksek hacim", "API listesindeki sıra"],
    )

    start = st.button(
        "🔍 Taramayı Başlat",
        type="primary",
        use_container_width=True,
    )


if "results" not in st.session_state:
    st.session_state["results"] = None

if start:
    progress = st.progress(0)
    status = st.empty()

    try:
        try:
            markets = get_binance_markets()
            tickers = get_binance_tickers()
            provider = "Binance"

        except Exception as binance_error:
            st.warning(
                "Binance API erişilemedi; Gate.io API deneniyor. "
                "Bu durumda sonuçlar Gate.io marketlerine aittir. "
                f"Binance hatası: {str(binance_error)[:180]}"
            )

            markets = get_gate_markets()
            tickers = get_gate_tickers()
            provider = "Gate.io"

        # Ticker eşleştirmesi sadece gerçek market ID ile yapılır.
        market_rows = []

        for market in markets:
            ticker = tickers.get(market["market_id"])

            if ticker is None:
                continue

            if ticker.get("volume", 0.0) < min_volume:
                continue

            market_rows.append({
                **market,
                **ticker,
            })

        if mode == "En yüksek hacim":
            market_rows.sort(
                key=lambda x: x.get("volume", 0.0),
                reverse=True,
            )

        market_rows = market_rows[:int(scan_count)]

        if not market_rows:
            raise RuntimeError(
                "Filtrelerden geçen market bulunamadı. "
                "Minimum hacmi düşürmeyi deneyin."
            )

        st.info(
            f"Kaynak: {provider} | "
            f"Borsadaki aktif market: {len(markets)} | "
            f"Taranacak: {len(market_rows)}"
        )

        results = []
        errors = []

        for i, market in enumerate(market_rows):
            status.write(
                f"{i + 1}/{len(market_rows)} — {market['symbol']} "
                f"({market['market_id']})"
            )

            try:
                ticker = tickers[market["market_id"]]

                item = scan_market(
                    market,
                    ticker,
                    max_distance_pct,
                )

                results.append(item)

            except Exception as error:
                errors.append({
                    "symbol": market["symbol"],
                    "market_id": market["market_id"],
                    "provider": market["provider"],
                    "error": str(error)[:250],
                })

            progress.progress((i + 1) / len(market_rows))

            # API'ye aşırı hızlı istek göndermemek için kısa bekleme.
            time.sleep(0.08)

        st.session_state["results"] = pd.DataFrame(results)
        st.session_state["errors"] = pd.DataFrame(errors)
        st.session_state["provider"] = provider
        st.session_state["scan_time"] = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        status.success(
            f"Tamamlandı: {len(results)} başarılı, "
            f"{len(errors)} hatalı market."
        )

    except Exception as error:
        st.error(f"Tarama başarısız: {error}")


# =========================================================
# SONUCLAR
# =========================================================

results = st.session_state.get("results")

if results is not None:
    if results.empty:
        st.warning(
            "Sonuç bulunamadı. Hata listesini inceleyin veya "
            "tarama ayarlarını değiştirin."
        )
    else:
        st.caption(
            f"Kaynak: {st.session_state.get('provider', '-')}"
            f" | Tarama: {st.session_state.get('scan_time', '-')}"
        )

        reversals = results[
            results["is_reversal"] == True
        ].sort_values(
            "distance_from_bottom_pct",
            ascending=True,
        )

        rising = results[
            (results["three_rising_candles"] == True)
            & (results["latest_candle_green"] == True)
        ].sort_values(
            "distance_from_bottom_pct",
            ascending=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Analiz edilen", len(results))
        c2.metric("Dipten dönüş şartlarını karşılayan", len(reversals))
        c3.metric("Son 3 mumu yükselen", len(rising))

        columns = [
            "symbol",
            "market_id",
            "provider",
            "price_15m_close",
            "ticker_price",
            "ticker_difference_pct",
            "bottom_10d",
            "bottom_time_utc",
            "distance_from_bottom_pct",
            "three_rising_candles",
            "latest_candle_green",
            "signal",
            "volume_24h_usdt",
        ]

        st.subheader("🎯 Dipten Dönüş Şartlarını Karşılayanlar")

        if reversals.empty:
            st.info(
                "Bu taramada tüm dipten dönüş şartlarını karşılayan "
                "coin bulunmadı. Bu, filtrelerin başarısız olduğu "
                "anlamına gelmez; şartları sağlayan market çıkmamış olabilir."
            )
        else:
            st.dataframe(
                reversals[columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Dipten dönüş listesini CSV indir",
                data=reversals.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="dipten_donus.csv",
                mime="text/csv",
            )

        st.subheader("📈 Son 3 Mumda Yükselen Coinler")

        st.dataframe(
            rising[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Yükselen coinleri CSV indir",
            data=rising.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="son_uc_mum_yukselenler.csv",
            mime="text/csv",
        )

        st.subheader("📋 Tüm Market Analizleri")

        st.dataframe(
            results[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tüm analizleri CSV indir",
            data=results.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="tum_market_analizleri.csv",
            mime="text/csv",
        )

        errors = st.session_state.get("errors")

        if errors is not None and not errors.empty:
            with st.expander(f"Veri hataları ({len(errors)})"):
                st.dataframe(
                    errors,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Hataları CSV indir",
                    data=errors.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="market_hatalari.csv",
                    mime="text/csv",
                )

else:
    st.info(
        "Tarama ayarlarını seçip 'Taramayı Başlat' düğmesine basın."
    )
