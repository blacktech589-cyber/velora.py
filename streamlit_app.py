import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.okx.com"
TIMEOUT = 15
MIN_REQUEST_INTERVAL = 0.25
LOOKBACK_WEEKS = 52

st.set_page_config(
    page_title="12 Aylık Dip Coin Tarayıcısı",
    page_icon="📉",
    layout="wide",
)

st.title("12 Aylık Dipte ve Zirvesinden %50+ Düşmüş Coinler")
st.caption(
    "Aktif OKX Spot USDT pariteleri | 12 aylık zirveden en az %50 düşüş "
    "| 12 aylık dip yakınlığı | Otomatik emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "OKX-Spot-Low-Scanner/1.1"})
rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    """OKX API erişim veya yanıt hatası."""


def api_get(path, params=None, retries=4):
    global last_request_time

    for attempt in range(retries + 1):
        with rate_lock:
            delay = MIN_REQUEST_INTERVAL - (time.monotonic() - last_request_time)
            if delay > 0:
                time.sleep(delay)
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
                "HTTP 451: OKX API erişimi bu bağlantı konumundan kısıtlanmış. "
                "Kısıtlamayı aşmaya çalışmadan tarama durduruldu."
            )

        if response.status_code == 429:
            if attempt < retries:
                raw_retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = (
                        float(raw_retry_after)
                        if raw_retry_after
                        else 2 ** (attempt + 1)
                    )
                except ValueError:
                    wait_seconds = 2 ** (attempt + 1)
                time.sleep(min(max(wait_seconds, 2), 30))
                continue
            raise MarketAPIError(
                "HTTP 429: API istek sınırına ulaşıldı. Coin sayısını azaltıp "
                "bir süre sonra tekrar deneyin."
            )

        if not response.ok:
            raise MarketAPIError(
                f"HTTP {response.status_code}: {response.text[:250]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketAPIError("API geçerli JSON döndürmedi.") from exc

        if isinstance(payload, dict) and payload.get("code") not in (None, "0"):
            raise MarketAPIError(
                f"OKX API hatası {payload.get('code')}: {payload.get('msg', '')}"
            )

        return payload

    raise MarketAPIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols():
    """Yalnızca OKX'in state=live gösterdiği Spot USDT pariteleri."""
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
        {"instId": symbol, "bar": timeframe, "limit": str(limit)},
    )
    return payload.get("data", [])


def confirmed_candles(candles):
    # OKX candle alanı [8]: "1" tamamlanmış mumu belirtir.
    return [
        candle for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(symbol, ticker, minimum_drop_pct, max_low_distance_pct):
    current_price = float(ticker.get("last", 0) or 0)
    if current_price <= 0:
        return None

    # 52 tamamlanmış haftalık mum yaklaşık 12 aylık dönemi kapsar.
    weekly = confirmed_candles(get_candles(symbol, "1W", 60))
    weekly.sort(key=lambda candle: int(candle[0]), reverse=True)
    weekly = weekly[:LOOKBACK_WEEKS]
    if len(weekly) < LOOKBACK_WEEKS:
        return None

    highest_12m = max(float(candle[2]) for candle in weekly)
    lowest_12m = min(float(candle[3]) for candle in weekly)
    if highest_12m <= 0 or lowest_12m <= 0:
        return None

    # Koşul 1: güncel fiyat 12 aylık zirveden en az %50 aşağıda.
    drop_from_high_pct = (1 - current_price / highest_12m) * 100
    if drop_from_high_pct < minimum_drop_pct:
        return None

    # Koşul 2: güncel fiyat 12 aylık dipte veya belirlenen tolerans içinde.
    distance_from_low_pct = (current_price / lowest_12m - 1) * 100
    if distance_from_low_pct > max_low_distance_pct:
        return None

    # Yalnızca bilgi amaçlı son iki tamamlanmış 15 dakikalık kapanış karşılaştırması.
    candles_15m = confirmed_candles(get_candles(symbol, "15m", 6))
    candles_15m.sort(key=lambda candle: int(candle[0]))
    if len(candles_15m) >= 2:
        previous_close = float(candles_15m[-2][4])
        last_close = float(candles_15m[-1][4])
        rising_15m = last_close > previous_close
    else:
        previous_close = None
        last_close = None
        rising_15m = False

    open_24h = float(ticker.get("open24h", 0) or 0)
    change_24h = (current_price / open_24h - 1) * 100 if open_24h > 0 else 0.0
    volume_24h = float(ticker.get("volCcy24h", 0) or 0)

    if distance_from_low_pct < 0:
        low_status = "GEÇMİŞ 12A DİBİNİN ALTINDA"
    elif distance_from_low_pct <= 0.1:
        low_status = "12A DİBİNDE"
    else:
        low_status = "12A DİBİNE YAKIN"

    return {
        "Coin": symbol,
        "Dip Durumu": low_status,
        "Güncel Fiyat": current_price,
        "12A Zirve": highest_12m,
        "12A En Düşük": lowest_12m,
        "Zirveden Düşüş (%)": round(drop_from_high_pct, 2),
        "Dibe Uzaklık (%)": round(distance_from_low_pct, 3),
        "24S Değişim (%)": round(change_24h, 2),
        "24S Hacim (USDT)": round(volume_24h, 2),
        "Önceki 15D Kapanış": previous_close,
        "Son 15D Kapanış": last_close,
        "15D Yükseliş Teyidi": rising_15m,
    }


# -------------------- SIDEBAR AYARLARI --------------------
st.sidebar.header("Tarama ayarları")
minimum_drop_pct = st.sidebar.slider(
    "Zirveden minimum düşüş (%)",
    min_value=50.0,
    max_value=95.0,
    value=50.0,
    step=5.0,
)
max_low_distance_pct = st.sidebar.slider(
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
    st.info("Ayarları belirleyip 'Taramayı başlat' düğmesine basın.")
    st.stop()

# -------------------- API ERİŞİMİ --------------------
try:
    with st.spinner("OKX verileri alınıyor..."):
        api_get("/api/v5/public/time")
        active_symbols = get_active_symbols()
        tickers = get_tickers()
except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"OKX'te aktif görünen Spot USDT paritesi: {len(active_symbols)}"
)

# -------------------- HACİM FİLTRESİ --------------------
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

universe.sort(key=lambda item: item[2], reverse=True)
universe = universe[:max_coins]
if not universe:
    st.warning("Hacim filtresine uygun aktif USDT paritesi bulunamadı.")
    st.stop()

st.write(f"**Taranacak aktif parite sayısı:** {len(universe)}")
st.info(
    "Bir coin listelenmek için hem zirvesinden en az belirlenen yüzde kadar "
    "düşmüş hem de son 12 aylık dip fiyatına yakın olmalıdır."
)

# -------------------- TARAMA --------------------
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
            minimum_drop_pct,
            max_low_distance_pct,
        ): symbol
        for symbol, ticker, _volume in universe
    }
    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
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
        except (ValueError, TypeError, KeyError, IndexError):
            errors += 1

        progress.progress(index / max(total, 1))
        status_text.text(f"İşlenen: {index}/{total}")

progress.empty()
status_text.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "İki koşulu aynı anda sağlayan coin bulunamadı. Coin sayısını artırın, "
        "hacim filtresini düşürün veya dip toleransını yükseltin."
    )
    st.stop()

# -------------------- SONUÇLAR --------------------
df = pd.DataFrame(results)
df = df.sort_values("Dibe Uzaklık (%)", ascending=True).reset_index(drop=True)
new_lows = df[df["Dibe Uzaklık (%)"] < 0].copy()
rising_coins = df[df["15D Yükseliş Teyidi"]].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("İki koşula uyan coin", len(df))
col2.metric("12A dip seviyesinin altında", len(new_lows))
col3.metric("15D yükseliş teyidi", len(rising_coins))
col4.metric("Analiz edilemeyen", errors)

st.subheader("12 aylık dipte ve zirvesinden en az %50 düşmüş coinler")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Geçmiş 12 aylık dip fiyatının altındakiler")
if new_lows.empty:
    st.info("Geçmiş dip fiyatının altında bulunan uygun coin yok.")
else:
    st.dataframe(new_lows, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık kapanışı yükselenler")
if rising_coins.empty:
    st.info("15 dakikalık yükseliş teyidi bulunan coin yok.")
else:
    st.dataframe(rising_coins, use_container_width=True, hide_index=True)

# -------------------- GRAFİK --------------------
st.subheader("15 dakikalık kapanış grafiği")
selected_coin = st.selectbox("Coin seçin", df["Coin"].tolist())

try:
    candles = confirmed_candles(get_candles(selected_coin, "15m", 100))
    candles.sort(key=lambda candle: int(candle[0]))
    chart_df = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime(
            [int(candle[0]) for candle in candles],
            unit="ms",
            utc=True,
        ),
        "Kapanış": [float(candle[4]) for candle in candles],
    })
    if not chart_df.empty:
        chart_df = chart_df.set_index("Zaman (UTC)")
        st.line_chart(chart_df["Kapanış"], use_container_width=True)
    else:
        st.info("Grafik için tamamlanmış mum verisi bulunamadı.")
except MarketAPIError as exc:
    st.warning(str(exc))

# -------------------- CSV --------------------
st.download_button(
    label="CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_12m_lows_down_50_percent.csv",
    mime="text/csv",
)

st.caption(
    "Veri OKX Spot'tan alınır. 52 tamamlanmış haftalık mum yaklaşık 12 aylık "
    "dönemi kapsar. Yalnızca seçilen hacim filtresini ve tarama sınırını "
    "karşılayan pariteler incelenir. OKX'te aktif görünen coin geçmişte "
    "delist edilip yeniden listelenmiş olabilir. Geçmiş dip seviyesine gelmek, "
    "daha fazla düşmeyeceği veya yükseleceği anlamına gelmez. Otomatik emir yok."
)
