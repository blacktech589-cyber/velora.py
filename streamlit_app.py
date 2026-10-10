import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# OKX Spot public API: veri okur, emir göndermez.
BASE_URL = "https://www.okx.com"
TIMEOUT = 15
MIN_REQUEST_INTERVAL = 0.20

# 52 tamamlanmış haftalık mum yaklaşık 12 aylık dönemdir.
LOOKBACK_WEEKS = 52

st.set_page_config(
    page_title="12 Aylık Dip Coin Tarayıcısı",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Son 12 Ayın Dip Seviyesindeki Coinler")
st.caption(
    "Aktif OKX Spot USDT pariteleri | 12 aylık en düşük fiyat "
    "| 15 dakikalık yükseliş kontrolü | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({
    "User-Agent": "OKX-12M-Low-Scanner/1.0"
})

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

            raise MarketAPIError(
                f"Bağlantı hatası: {exc}"
            ) from exc

        if response.status_code == 451:
            raise MarketAPIError(
                "HTTP 451: API erişimi bu bağlantı konumundan "
                "kısıtlanmış. Kısıtlama aşılmaya çalışılmayacak."
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
                "Taranan coin sayısını azaltıp tekrar deneyin."
            )

        if not response.ok:
            raise MarketAPIError(
                f"HTTP {response.status_code}: {response.text[:250]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketAPIError(
                "API geçerli JSON döndürmedi."
            ) from exc

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
    """OKX tarafından hâlen aktif gösterilen Spot USDT pariteleri."""
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


def get_confirmed_candles(candles):
    # OKX mum alanı [8]: 1 tamamlanmış, 0 tamamlanmamış.
    return [
        candle
        for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(symbol, ticker, max_distance_pct):
    try:
        current_price = float(
            ticker.get("last", 0) or 0
        )

        if current_price <= 0:
            return None

        volume_24h = float(
            ticker.get("volCcy24h", 0) or 0
        )

        open_24h = float(
            ticker.get("open24h", 0) or 0
        )

        change_24h = (
            (current_price / open_24h - 1) * 100
            if open_24h > 0
            else 0.0
        )

        # Son 52 tamamlanmış haftalık mum:
        # Haftalık mumun yüksek/düşük alanları, hafta içinde
        # görülen en yüksek ve en düşük fiyatı içerir.
        candles_weekly = get_confirmed_candles(
            get_candles(symbol, "1W", 60)
        )

        candles_weekly.sort(
            key=lambda candle: int(candle[0]),
            reverse=True,
        )

        candles_weekly = candles_weekly[:LOOKBACK_WEEKS]

        if len(candles_weekly) < LOOKBACK_WEEKS:
            return None

        highest_12m = max(
            float(candle[2])
            for candle in candles_weekly
        )

        lowest_12m = min(
            float(candle[3])
            for candle in candles_weekly
        )

        if lowest_12m <= 0:
            return None

        # Fiyat, 12 aylık dipten yüzde kaç uzakta?
        # Negatif değer, fiyatın geçmiş dip seviyesinin altına
        # inmiş olduğunu gösterir.
        distance_pct = (
            current_price / lowest_12m - 1
        ) * 100

        # Dip seviyesinin altındaki fiyatlar da dahil edilir.
        # Örneğin %1 toleransta, geçmiş dip fiyatının en fazla
        # %1 üzerinde olan coinler listelenir.
        at_low = distance_pct <= max_distance_pct

        if not at_low:
            return None

        # Son iki tamamlanmış 15 dakikalık mumun kapanışını kontrol et.
        candles_15m = get_confirmed_candles(
            get_candles(symbol, "15m", 6)
        )

        candles_15m.sort(
            key=lambda candle: int(candle[0])
        )

        if len(candles_15m) >= 2:
            previous_close = float(candles_15m[-2][4])
            last_close = float(candles_15m[-1][4])
            rising = last_close > previous_close
        else:
            previous_close = None
            last_close = None
            rising = False

        below_previous_low = current_price < lowest_12m

        if below_previous_low:
            status = "YENİ 12A DİBİ"
        elif rising:
            status = "DİP BÖLGESİ + 15D YÜKSELİŞ"
        else:
            status = "DİP BÖLGESİ — TEYİT YOK"

        return {
            "Coin": symbol,
            "Durum": status,
            "Güncel Fiyat": current_price,
            "12A En Yüksek Fiyat": highest_12m,
            "12A En Düşük Fiyat": lowest_12m,
            "Dibe Uzaklık (%)": round(distance_pct, 3),
            "24S Değişim (%)": round(change_24h, 2),
            "24S Hacim (USDT)": round(volume_24h, 2),
            "Önceki 15D Kapanış": previous_close,
            "Son 15D Kapanış": last_close,
            "15D Yükseliş": rising,
            "Geçmiş Dip Altında mı": below_previous_low,
        }

    except (ValueError, TypeError, KeyError, IndexError):
        return None


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

st.sidebar.header("Tarama ayarları")

max_distance_pct = st.sidebar.slider(
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
    st.info("Ayarları belirleyip taramayı başlatın.")
    st.stop()


# --------------------------------------------------
# API ERİŞİMİ
# --------------------------------------------------

try:
    with st.spinner("OKX verileri alınıyor..."):
        api_get("/api/v5/public/time")
        active_symbols = get_active_symbols()
        tickers = get_tickers()

except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"OKX üzerinde aktif görünen USDT Spot paritesi: "
    f"{len(active_symbols)}"
)


# --------------------------------------------------
# TARAMA LİSTESİ
# --------------------------------------------------

universe = []

for symbol in active_symbols:
    ticker = tickers.get(symbol)

    if not ticker:
        continue

    try:
        price = float(ticker.get("last", 0) or 0)
        volume = float(ticker.get("volCcy24h", 0) or 0)

        if price > 0 and volume >= min_volume:
            universe.append((symbol, ticker, volume))

    except (ValueError, TypeError):
        continue

# API limitini azaltmak için önce hacmi yüksek pariteler.
universe.sort(
    key=lambda item: item[2],
    reverse=True,
)

universe = universe[:max_coins]

if not universe:
    st.warning(
        "Filtrelere uygun aktif parite yok. "
        "Minimum hacmi düşürmeyi deneyin."
    )
    st.stop()

st.write(f"**Taranacak parite sayısı:** {len(universe)}")

st.info(
    "Yalnızca OKX üzerinde şu anda state='live' olarak gösterilen "
    "pariteler taranır. Geçmişteki tüm listeleme/delist olaylarının "
    "kayıtları bu endpoint üzerinden doğrulanmaz."
)


# --------------------------------------------------
# TARAMA
# --------------------------------------------------

results = []
errors = 0
fatal_error = None

progress = st.progress(0)
status_text = st.empty()

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(
            analyze_coin,
            symbol,
            ticker,
            max_distance_pct,
        ): symbol
        for symbol, ticker, _volume in universe
    }

    total = len(futures)

    for index, future in enumerate(
        as_completed(futures),
        start=1,
    ):
        try:
            result = future.result()

            if result is not None:
                results.append(result)
            else:
                errors += 1

        except MarketAPIError as exc:
            fatal_error = str(exc)

            for pending in futures:
                pending.cancel()

            break

        progress.progress(index / max(total, 1))
        status_text.text(f"İşlenen: {index}/{total}")

progress.empty()
status_text.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "Dip seviyesine yakın coin bulunamadı. "
        "Coin sayısını artırabilir, hacim eşiğini düşürebilir "
        "veya dipten uzaklık oranını yükseltebilirsiniz."
    )
    st.stop()


# --------------------------------------------------
# SONUÇLAR
# --------------------------------------------------

df = pd.DataFrame(results)

# En düşük mesafeli coinler önce; yeni dip yapanlar da üstte görünür.
df = df.sort_values(
    "Dibe Uzaklık (%)",
    ascending=True,
).reset_index(drop=True)

new_lows = df[
    df["Geçmiş Dip Altında mı"]
].copy()

rising_coins = df[
    df["15D Yükseliş"]
].copy()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Dip bölgesindeki coin", len(df))
c2.metric("Yeni 12A dip fiyatı", len(new_lows))
c3.metric("15D yükselişi olan", len(rising_coins))
c4.metric("Analiz edilemeyen", errors)

st.subheader("12 aylık dip seviyesine en yakın coinler")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Geçmiş 12 aylık dip fiyatının altına inenler")

if new_lows.empty:
    st.info(
        "Taranan coinlerden hiçbiri önceki 12 aylık dip fiyatının "
        "altında görünmüyor."
    )
else:
    st.dataframe(
        new_lows,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Dip bölgesinde olup 15 dakikalık kapanışı yükselenler")

if rising_coins.empty:
    st.info("Yükseliş teyidi veren coin bulunamadı.")
else:
    st.dataframe(
        rising_coins,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# GRAFİK
# --------------------------------------------------

st.subheader("15 dakikalık fiyat grafiği")

selected_coin = st.selectbox(
    "Coin seçin",
    df["Coin"].tolist(),
)

try:
    candles = get_confirmed_candles(
        get_candles(selected_coin, "15m", 100)
    )

    candles.sort(
        key=lambda candle: int(candle[0])
    )

    chart_df = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime(
            [int(candle[0]) for candle in candles],
            unit="ms",
            utc=True,
        ),
        "Kapanış": [
            float(candle[4])
            for candle in candles
        ],
    })

    if not chart_df.empty:
        chart_df = chart_df.set_index("Zaman (UTC)")
        st.line_chart(
            chart_df["Kapanış"],
            use_container_width=True,
        )
    else:
        st.info("Grafik verisi bulunamadı.")

except MarketAPIError as exc:
    st.warning(str(exc))


# --------------------------------------------------
# CSV
# --------------------------------------------------

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_active_coins_12_month_lows.csv",
    mime="text/csv",
)

st.caption(
    "Önemli: 52 tamamlanmış haftalık mum yaklaşık 12 aylık dönemi "
    "temsil eder; takvimdeki son 365 günle birebir aynı olmayabilir. "
    "Tarama yalnızca seçilen hacim eşiğini karşılayan pariteleri inceler. "
    "OKX'te aktif görünen bir coin geçmişte liste dışına çıkarılıp "
    "yeniden listelenmiş olabilir. Fiyatın geçmiş dibine gelmesi, "
    "daha fazla düşmeyeceği veya yükseleceği anlamına gelmez. "
    "Otomatik emir gönderilmez."
)
