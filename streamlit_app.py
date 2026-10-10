import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.okx.com"
TIMEOUT = 15
MIN_REQUEST_INTERVAL = 0.20
LOOKBACK_WEEKS = 52

st.set_page_config(
    page_title="12 Aylık Dip Coin Tarayıcısı",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Son 12 Ayın Dibindeki ve %50+ Düşmüş Coinler")
st.caption(
    "Aktif OKX Spot USDT pariteleri | En az %50 düşüş | "
    "12 aylık dip kontrolü | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({"User-Agent": "OKX-Spot-Low-Scanner/1.0"})
rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    pass


def api_get(path, params=None, retries=4):
    global last_request_time

    for attempt in range(retries + 1):
        with rate_lock:
            wait = MIN_REQUEST_INTERVAL - (
                time.monotonic() - last_request_time
            )
            if wait > 0:
                time.sleep(wait)
            last_request_time = time.monotonic()

        try:
            response = session.get(
                BASE_URL + path,
                params=params,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 20))
                continue
            raise MarketAPIError(f"Bağlantı hatası: {exc}") from exc

        if response.status_code == 451:
            raise MarketAPIError(
                "HTTP 451: OKX API erişimi bu bağlantı konumundan "
                "kısıtlanmış. Kısıtlamayı aşmaya çalışmadan tarama durduruldu."
            )

        if response.status_code == 429:
            if attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = (
                        float(retry_after)
                        if retry_after
                        else 2 ** (attempt + 1)
                    )
                except ValueError:
                    delay = 2 ** (attempt + 1)

                time.sleep(min(max(delay, 2), 30))
                continue

            raise MarketAPIError(
                "HTTP 429: API istek sınırı aşıldı. "
                "Daha az coin tarayıp sonra tekrar deneyin."
            )

        if not response.ok:
            raise MarketAPIError(
                f"HTTP {response.status_code}: {response.text[:250]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketAPIError("API geçerli JSON döndürmedi.") from exc

        if (
            isinstance(payload, dict)
            and payload.get("code") not in (None, "0")
        ):
            raise MarketAPIError(
                f"OKX API hatası {payload.get('code')}: "
                f"{payload.get('msg', '')}"
            )

        return payload

    raise MarketAPIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols():
    """OKX'te hâlen aktif görünen Spot USDT paritelerini al."""
    payload = api_get(
        "/api/v5/public/instruments",
        {"instType": "SPOT"},
    )

    return [
        item["instId"]
        for item in payload.get("data", [])
        if item.get("state") == "live"
        and item.get("quoteCcy") == "USDT"
        and item.get("instId", "").endswith("-USDT")
    ]


@st.cache_data(ttl=180, show_spinner=False)
def get_tickers():
    payload = api_get(
        "/api/v5/market/tickers",
        {"instType": "SPOT"},
    )

    return {
        item["instId"]: item
        for item in payload.get("data", [])
        if item.get("instId")
    }


def get_candles(symbol, timeframe, limit):
    payload = api_get(
        "/api/v5/market/candles",
        {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit),
        },
    )
    return payload.get("data", [])


def confirmed_candles(candles):
    # OKX mum dizisinde 8. indeks: 1 = tamamlanmış mum.
    return [
        candle
        for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(
    symbol,
    ticker,
    minimum_drop,
    maximum_low_distance,
):
    current_price = float(ticker.get("last", 0) or 0)

    if current_price <= 0:
        return None

    # Son 52 tamamlanmış haftalık mum yaklaşık 12 aylık geçmiş sağlar.
    weekly = confirmed_candles(
        get_candles(symbol, "1W", 60)
    )

    weekly.sort(
        key=lambda candle: int(candle[0]),
        reverse=True,
    )
    weekly = weekly[:LOOKBACK_WEEKS]

    # Yeterli geçmiş verisi olmayan coinleri ele.
    if len(weekly) < LOOKBACK_WEEKS:
        return None

    highest_12m = max(
        float(candle[2]) for candle in weekly
    )
    lowest_12m = min(
        float(candle[3]) for candle in weekly
    )

    if highest_12m <= 0 or lowest_12m <= 0:
        return None

    # Zirveden düşüş yüzdesi.
    drop_pct = (1 - current_price / highest_12m) * 100

    # Koşul 1: en az %50 düşmüş olmalı.
    if drop_pct < minimum_drop:
        return None

    # Dip fiyatına uzaklık:
    # 0% = tam dip seviyesinde
    # Pozitif = dip fiyatının üzerinde
    # Negatif = geçmiş dip seviyesinin altında
    low_distance_pct = (
        current_price / lowest_12m - 1
    ) * 100

    # Koşul 2: dip seviyesinde veya belirlenen tolerans içinde olmalı.
    if low_distance_pct > maximum_low_distance:
        return None

    # Son iki tamamlanmış 15 dakikalık kapanışla yükseliş teyidi.
    candles_15m = confirmed_candles(
        get_candles(symbol, "15m", 6)
    )
    candles_15m.sort(
        key=lambda candle: int(candle[0])
    )

    if len(candles_15m) >= 2:
        previous_close = float(candles_15m[-2][4])
        last_close = float(candles_15m[-1][4])
        rising_15m = last_close > previous_close
    else:
        previous_close = None
        last_close = None
        rising_15m = False

    open_24h = float(ticker.get("open24h", 0) or 0)
    change_24h = (
        (current_price / open_24h - 1) * 100
        if open_24h > 0
        else 0.0
    )

    volume_24h = float(ticker.get("volCcy24h", 0) or 0)

    if low_distance_pct < 0:
        low_status = "12A DİBİNİN ALTINDA"
    elif low_distance_pct <= 0.1:
        low_status = "12A DİBİNDE"
    else:
        low_status = "12A DİBİNE YAKIN"

    return {
        "Coin": symbol,
        "Dip Durumu": low_status,
        "Güncel Fiyat": current_price,
        "12A Zirve": highest_12m,
        "12A En Düşük": lowest_12m,
        "Zirveden Düşüş (%)": round(drop_pct, 2),
        "Dibe Uzaklık (%)": round(low_distance_pct, 3),
        "24S Değişim (%)": round(change_24h, 2),
        "24S Hacim (USDT)": round(volume_24h, 2),
        "Önceki 15D Kapanış": previous_close,
        "Son 15D Kapanış": last_close,
        "15D Yükseliş Teyidi": rising_15m,
    }


# =====================================================
# AYARLAR
# =====================================================

st.sidebar.header("Tarama ayarları")

minimum_drop = st.sidebar.slider(
    "Zirveden minimum düşüş (%)",
    min_value=50.0,
    max_value=95.0,
    value=50.0,
    step=5.0,
)

maximum_low_distance = st.sidebar.slider(
    "12 aylık dipten maksimum uzaklık (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

max_coins = st.sidebar.slider(
    "Taranacak coin sayısı",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0,
    value=50000,
    step=50000,
)

if st.sidebar.button("Önbelleği temizle"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Taramayı başlat", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Ayarları seçip 'Taramayı başlat' düğmesine basın.")
    st.stop()


# =====================================================
# AKTİF PARİTELER
# =====================================================

try:
    with st.spinner("OKX verileri alınıyor..."):
        api_get("/api/v5/public/time")
        symbols = get_active_symbols()
        tickers = get_tickers()

except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"OKX't
