
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
    page_title="Hyperactive Binance Spot Scanner",
    page_icon="⚡",
    layout="wide",
)

API = "https://api.binance.com"
INTERVAL = "15m"
INTERVAL_SECONDS = 900
BARS_30_DAYS = 30 * 24 * 4
REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": "HyperactiveSpotScanner/1.0",
    "Accept": "application/json",
}

session = requests.Session()
session.headers.update(HEADERS)


# =========================================================
# API YARDIMCILARI
# =========================================================

def get_json(path, params=None):
    response = session.get(
        f"{API}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# =========================================================
# AKTİF BINANCE SPOT USDT MARKETLERİ
# =========================================================

def get_markets():
    info = get_json("/api/v3/exchangeInfo")
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
            "base_asset": item.get("baseAsset", ""),
        })

    if not markets:
        raise RuntimeError("Aktif Binance Spot USDT marketi bulunamadı.")

    return markets


def get_tickers():
    data = get_json("/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        try:
            tickers[item["symbol"]] = {
                "price": float(item["lastPrice"]),
                "quote_volume": float(item["quoteVolume"]),
                "change_24h": float(item["priceChangePercent"]),
                "high_24h": float(item["highPrice"]),
                "low_24h": float(item["lowPrice"]),
                "trades_24h": int(item["count"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


# =========================================================
# MUM VERİSİNİ SAYFALAYARAK ÇEK
# =========================================================

def load_30_day_candles(symbol):
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = (now // INTERVAL_SECONDS) * INTERVAL_SECONDS

    # 30 gün + ek mumlar; güncel tamamlanmamış mum sonradan çıkarılır.
    start_time = (
        current_start - (BARS_30_DAYS + 10) * INTERVAL_SECONDS
    ) * 1000

    end_time = current_start * 1000
    cursor = start_time
    rows = []

    while cursor < end_time:
        raw = get_json(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_time,
                "limit": 1000,
            },
        )

        if not raw:
            break

        for candle in raw:
            try:
                rows.append({
                    "timestamp": int(candle[0]) // 1000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "quote_volume": float(candle[7]),
                    "trades": int(candle[8]),
                })
            except (ValueError, TypeError, IndexError):
                continue

        next_cursor = int(raw[-1][0]) + INTERVAL_SECONDS * 1000

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(raw) < 1000:
            break

        time.sleep(0.04)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Sadece tamamlanmış mumlar.
    now = int(datetime.now(timezone.utc).timestamp())
    current_start = (now // INTERVAL_SECONDS) * INTERVAL_SECONDS

    df = df[df["timestamp"] < current_start].copy()

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
    ]

    return df.reset_index(drop=True)


# =========================================================
# TEKNİK GÖSTERGELER
# =========================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calculate_atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


# =========================================================
# AYLIK DİP + HİPERAKTİFLİK
# =========================================================

def analyze_market(df, ticker, max_bottom_distance=10.0):
    if len(df) < BARS_30_DAYS:
        return {
            "valid": False,
            "reason": (
                f"Yeterli aylık geçmiş yok: {len(df)} / "
                f"{BARS_30_DAYS} tamamlanmış mum."
            ),
        }

    month = df.tail(BARS_30_DAYS).reset_index(drop=True)

    # Aylık en düşük fiyat.
    bottom_pos = int(month["low"].to_numpy().argmin())
    bottom_price = float(month.iloc[bottom_pos]["low"])
    current_price = float(month.iloc[-1]["close"])

    if bottom_price <= 0:
        return {"valid": False, "reason": "Geçersiz dip fiyatı."}

    distance_pct = (current_price / bottom_price - 1) * 100

    # Son üç tamamlanmış mum.
    last3 = month.tail(3)
    closes = last3["close"].to_numpy()
    opens = last3["open"].to_numpy()

    three_rising = bool(
        closes[0] < closes[1] < closes[2]
    )

    last_green = bool(closes[-1] > opens[-1])
    bottom_before_last3 = bottom_pos < len(month) - 3

    near_bottom = bool(
        0 <= distance_pct <= max_bottom_distance
    )

    # Hacim ve oynaklık metrikleri.
    month["return"] = month["close"].pct_change()
    month["atr"] = calculate_atr(month)
    month["rsi"] = calculate_rsi(month["close"])

    atr_pct = (
        float(month["atr"].iloc[-1] / current_price * 100)
        if pd.notna(month["atr"].iloc[-1])
        else np.nan
    )

    # Ortalama 24 saatlik quote hacmi:
    # son 96 adet 15 dakikalık mumun quote hacmi toplamı.
    quote_volume = month["quote_volume"].fillna(0)
    volume_24h = float(quote_volume.tail(96).sum())

    # 24 saatlik gerçekleşmiş oynaklık: 15 dakikalık getirilerin
    # standart sapması, 96 dönem üzerinden yıllıklandırılmadan yüzde.
    returns_24h = month["return"].tail(96).dropna()
    volatility_24h_pct = (
        float(returns_24h.std(ddof=1) * np.sqrt(96) * 100)
        if len(returns_24h) >= 48
        else np.nan
    )

    # Hareketliliği ölçen puan. Sıralama puanıdır; olasılık değildir.
    volume_score = min(
        100,
        max(0, np.log10(max(volume_24h, 1)) * 10),
    )

    volatility_score = (
        min(100, max(0, volatility_24h_pct * 5))
        if pd.notna(volatility_24h_pct)
        else 0
    )

    activity_score = (
        0.5 * volume_score + 0.5 * volatility_score
    )

    # Her iki koşulun da aynı anda karşılanması gerekir.
    is_hyperactive = bool(
        volume_24h > 0
        and pd.notna(volatility_24h_pct)
        and volatility_24h_pct > 0
    )

    is_monthly_bottom_reversal = bool(
        near_bottom
        and bottom_before_last3
        and three_rising
        and last_green
        and is_hyperactive
    )

    if is_monthly_bottom_reversal:
        signal = "HİPERAKTİF AYLIK DİP DÖNÜŞÜ"
    elif near_bottom and is_hyperactive:
        signal = "AYLIK DİBE YAKIN / HİPERAKTİF"
    elif three_rising and last_green:
        signal = "YÜKSELİŞ MOMENTUMU"
    else:
        signal = "DİĞER"

    bottom_time = datetime.fromtimestamp(
        int(month.iloc[bottom_pos]["timestamp"]),
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "valid": True,
        "price": current_price,
        "monthly_bottom": bottom_price,
        "bottom_time_utc": bottom_time,
        "distance_from_bottom_pct": distance_pct,
        "three_rising_candles": three_rising,
        "latest_candle_green": last_green,
        "near_monthly_bottom": near_bottom,
        "volume_24h_usdt": volume_24h,
        "volatility_24h_pct": volatility_24h_pct,
        "atr_pct": atr_pct,
        "rsi_14": (
            float(month["rsi"].iloc[-1])
            if pd.notna(month["rsi"].iloc[-1])
            else np.nan
        ),
        "activity_score": activity_score,
        "is_hyperactive": is_hyperactive,
        "is_monthly_bottom_reversal": is_monthly_bottom_reversal,
        "signal": signal,
        "candles_30d": len(month),
    }


# =========================================================
# SCAN
# =========================================================

def scan_one_market(market, ticker, max_bottom_distance):
    df = load_30_day_candles(market["market_id"])

    analysis = analyze_market(
        df,
        ticker,
        max_bottom_distance=max_bottom_distance,
    )

    if not analysis.get("valid"):
        raise RuntimeError(analysis.get("reason", "Analiz yapılamadı."))

    candle_price = float(analysis["price"])
    ticker_price = float(ticker["price"])

    if candle_price <= 0 or ticker_price <= 0:
        raise RuntimeError("Geçersiz fiyat verisi.")

    price_diff_pct = abs(
        ticker_price - candle_price
    ) / candle_price * 100

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "price": candle_price,
        "ticker_price": ticker_price,
        "price_difference_pct": price_diff_pct,
        "price_check_ok": price_diff_pct <= 2,
        "change_24h_pct": ticker["change_24h"],
        "ticker_quote_volume_24h": ticker["quote_volume"],
        "trades_24h": ticker["trades_24h"],
        **analysis,
    }


# =========================================================
# STREAMLIT ARAYÜZÜ
# =========================================================

st.title("⚡ Hiperaktif Binance Spot Coin Tarayıcı")

st.write(
    "Yüksek hacimli ve oynak, son 30 günün dip seviyesine yakın "
    "coinleri bulur. Son üç tamamlanmış 15 dakikalık mumun "
    "kapanışları art arda yükseliyorsa dipten dönüş adayı olarak "
    "işaretler. Emir göndermez."
)

st.warning(
    "Hiperaktiflik puanı bir getiri olasılığı değildir. Yüksek "
    "oynaklık hem hızlı kazanç hem de hızlı kayıp anlamına gelebilir."
)

with st.sidebar:
    st.header("Tarama Ayarları")

    max_bottom_distance = st.slider(
        "Aylık dipten maksimum uzaklık (%)",
        min_value=1.0,
        max_value=10.0,
        value=10.0,
        step=0.5,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0,
        value=100000,
        step=50000,
    )

    min_volatility = st.number_input(
        "Minimum 24 saatlik yıllıklandırılmamış oynaklık (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5,
    )

    scan_count = st.number_input(
        "Taranacak market sayısı",
        min_value=1,
        max_value=2000,
        value=50,
        step=25,
    )

    scan_button = st.button(
        "🔍 Binance Spot Tara",
        type="primary",
        use_container_width=True,
    )


if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None


if scan_button:
    progress = st.progress(0)
    status = st.empty()

    try:
        markets = get_markets()
        tickers = get_tickers()

        candidates = []

        for market in markets:
            ticker = tickers.get(market["market_id"])

            if ticker is None:
                continue

            if ticker["quote_volume"] < min_volume:
                continue

            candidates.append({
                **market,
                **ticker,
            })

        candidates.sort(
            key=lambda item: item["quote_volume"],
            reverse=True,
        )

        candidates = candidates[:int(scan_count)]

        if not candidates:
            raise RuntimeError(
                "Hacim filtresinden geçen aktif market bulunamadı."
            )

        st.info(
            f"Binance aktif Spot USDT marketi: {len(markets)} | "
            f"Analiz edilecek: {len(candidates)}"
        )

        results = []
        errors = []

        for i, market in enumerate(candidates):
            status.write(
                f"{i + 1}/{len(candidates)} — {market['symbol']}"
            )

            ticker = tickers[market["market_id"]]

            try:
                result = scan_one_market(
                    market,
                    ticker,
                    max_bottom_distance,
                )
                results.append(result)

            except Exception as exc:
                errors.append({
                    "symbol": market["symbol"],
                    "market_id": market["market_id"],
                    "error": str(exc)[:250],
                })

            progress.progress((i + 1) / len(candidates))
            time.sleep(0.05)

        st.session_state["scan_results"] = pd.DataFrame(results)
        st.session_state["scan_errors"] = pd.DataFrame(errors)
        st.session_state["scan_time"] = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        status.success(
            f"Tamamlandı: {len(results)} analiz, {len(errors)} hata."
        )

    except Exception as exc:
        st.error(f"Tarama başarısız: {exc}")


# =========================================================
# RESULTS
# =========================================================

results = st.session_state.get("scan_results")

if results is not None:
    if results.empty:
        st.warning(
            "Sonuç yok. Hata listesini kontrol edin veya filtreleri azaltın."
        )
    else:
        st.caption(
            "Son tarama: "
            + st.session_state.get("scan_time", "-")
        )

        verified = results[
            results["price_check_ok"] == True
        ].copy()

        # Apply BOTH activity filters: high volume and volatility.
        hyperactive = verified[
            (verified["ticker_quote_volume_24h"] >= min_volume)
            & (verified["volatility_24h_pct"] >= min_volatility)
            & (verified["is_hyperactive"] == True)
        ].copy()

        monthly_reversals = hyperactive[
            hyperactive["is_monthly_bottom_reversal"] == True
        ].sort_values(
            "activity_score",
            ascending=False,
        )

        rising_near_bottom = hyperactive[
            (hyperactive["near_monthly_bottom"] == True)
            & (hyperactive["three_rising_candles"] == True)
            & (hyperactive["latest_candle_green"] == True)
        ].sort_values(
            "activity_score",
            ascending=False,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analiz edilen", len(results))
        c2.metric("Fiyat kontrolü geçen", len(verified))
        c3.metric("Hiperaktif", len(hyperactive))
        c4.metric("Aylık dipten dönüş", len(monthly_reversals))

        columns = [
            "symbol",
            "market_id",
            "price",
            "ticker_price",
            "price_difference_pct",
            "monthly_bottom",
            "bottom_time_utc",
            "distance_from_bottom_pct",
            "volume_24h_usdt",
            "volatility_24h_pct",
            "atr_pct",
            "rsi_14",
            "trades_24h",
            "activity_score",
            "signal",
        ]

        st.subheader("🎯 Hiperaktif Aylık Dipten Dönüş Coinleri")

        if monthly_reversals.empty:
            st.info(
                "Bu taramada bütün şartları karşılayan coin bulunamadı. "
                "Hacim, oynaklık veya dip filtresi fazla katı olabilir."
            )
        else:
            st.dataframe(
                monthly_reversals[columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Dipten dönüş CSV indir",
                data=monthly_reversals.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="hiperaktif_aylik_dip_donus.csv",
                mime="text/csv",
            )

        st.subheader("📈 Aylık Dibe Yakın ve Yükselenler")

        st.dataframe(
            rising_near_bottom[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("⚡ Hiperaktif Coinlerin Tamamı")

        if hyperactive.empty:
            st.info("Hiperaktif filtrelerinden geçen coin yok.")
        else:
            st.dataframe(
                hyperactive.sort_values(
                    "activity_score",
                    ascending=False,
                )[columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Hiperaktif coinleri CSV indir",
                data=hyperactive.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="hiperaktif_binance_spot.csv",
                mime="text/csv",
            )

        st.subheader("📋 Tüm Analizler")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tüm sonuçları CSV indir",
            data=results.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="binance_spot_tum_analizler.csv",
            mime="text/csv",
        )

        errors = st.session_state.get("scan_errors")

        if errors is not None and not errors.empty:
            with st.expander(f"Market hataları ({len(errors)})"):
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
                    file_name="binance_market_hatalari.csv",
                    mime="text/csv",
                )

else:
    st.info(
        "Tarama ayarlarını seçip 'Binance Spot Tara' düğmesine basın."
    )
