import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# OKX public Spot data only. No API keys and no order execution.
BASE_URL = "https://www.okx.com"
TIMEOUT = 15
QUOTE_CURRENCY = "USDT"
MONTHS_LOOKBACK = 12
MIN_SECONDS_BETWEEN_REQUESTS = 0.20

st.set_page_config(
    page_title="OKX Spot 12-Month Low Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("OKX Spot - Son 12 Ayın Dip Tarayıcısı")
st.caption(
    "Aktif Spot USDT pariteleri | Son 12 tamamlanmış aylık mumun en düşüğü | "
    "Tamamlanmış 15 dakikalık yükseliş teyidi | Emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.2"})

_rate_lock = threading.Lock()
_last_request_time = 0.0


class MarketAPIError(Exception):
    pass


def api_get(path, params=None, retries=4):
    global _last_request_time

    for attempt in range(retries + 1):
        with _rate_lock:
            wait = MIN_SECONDS_BETWEEN_REQUESTS - (
                time.monotonic() - _last_request_time
            )
            if wait > 0:
                time.sleep(wait)
            _last_request_time = time.monotonic()

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
                "HTTP 451: Veri sağlayıcısı bu bağlantı konumundan erişimi "
                "kısıtlıyor. Kısıtlama aşılmaya çalışılmayacak."
            )

        if response.status_code == 429:
            if attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(min(max(delay, 2), 30))
                continue
            raise MarketAPIError(
                "HTTP 429: İstek sınırı tekrarlandı. Tarama durduruldu. "
                "Daha az coin seçip bir süre sonra yeniden deneyin."
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
                f"API hatası {payload.get('code')}: {payload.get('msg', '')}"
            )

        return payload

    raise MarketAPIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_spot_symbols():
    payload = api_get(
        "/api/v5/public/instruments",
        {"instType": "SPOT"},
    )
    return [
        row["instId"]
        for row in payload.get("data", [])
        if row.get("state") == "live"
        and row.get("quoteCcy") == QUOTE_CURRENCY
        and row.get("instId", "").endswith("-USDT")
    ]


@st.cache_data(ttl=180, show_spinner=False)
def get_tickers():
    payload = api_get(
        "/api/v5/market/tickers",
        {"instType": "SPOT"},
    )
    return {
        row["instId"]: row
        for row in payload.get("data", [])
        if "instId" in row
    }


def get_candles(inst_id, bar, limit):
    payload = api_get(
        "/api/v5/market/candles",
        {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit),
        },
    )
    return payload.get("data", [])


def confirmed_candles(candles):
    # OKX candle field 8: "1" means confirmed, "0" means incomplete.
    return [
        candle for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(inst_id, ticker, max_distance):
    price = float(ticker.get("last", 0))
    if price <= 0:
        return None

    open24 = float(ticker.get("open24h", 0) or 0)
    change24 = (price / open24 - 1) * 100 if open24 > 0 else 0.0

    # One monthly candle contains that month's lowest traded price.
    # Exclude the current, potentially incomplete month; then use the
    # last 12 confirmed monthly candles to estimate the 12-month low.
    monthly_raw = get_candles(inst_id, "1M", 14)
    monthly = confirmed_candles(monthly_raw)
    monthly = sorted(monthly, key=lambda candle: int(candle[0]), reverse=True)

    if len(monthly) < MONTHS_LOOKBACK:
        return None

    last12_months = monthly[:MONTHS_LOOKBACK]
    low12 = min(float(candle[3]) for candle in last12_months)
    if low12 <= 0:
        return None

    distance = (price / low12 - 1) * 100
    near_low = distance <= max_distance

    previous_close = None
    last_close = None
    rising = False

    # Only query 15m candles when the coin is near its 12-month low.
    if near_low:
        raw15 = get_candles(inst_id, "15m", 6)
        completed15 = confirmed_candles(raw15)
        completed15 = sorted(completed15, key=lambda candle: int(candle[0]))

        if len(completed15) >= 2:
            previous_close = float(completed15[-2][4])
            last_close = float(completed15[-1][4])
            rising = last_close > previous_close

    signal = "BUY" if near_low and rising else "SELL"

    return {
        "Coin": inst_id,
        "Signal": signal,
        "Current Price": price,
        "12M Lowest Price": low12,
        "Distance From 12M Low (%)": round(distance, 3),
        "24H Change (%)": round(change24, 2),
        "Near 12M Low": near_low,
        "15M Previous Close": previous_close,
        "15M Last Completed Close": last_close,
        "15M Rising": rising,
    }


st.sidebar.header("Ayarlar")

max_distance = st.sidebar.slider(
    "12 aylık dip seviyesine maksimum uzaklık (%)",
    min_value=0.1,
    max_value=15.0,
    value=1.0,
    step=0.1,
)

max_coins = st.sidebar.slider(
    "Taranacak coin sayısı",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT karşılığı)",
    min_value=0,
    value=50000,
    step=50000,
)

if st.sidebar.button("Önbelleği temizle"):
    get_active_spot_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Taramayı başlat", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Ayarları belirleyip 'Taramayı başlat' düğmesine basın.")
    st.stop()

try:
    with st.spinner("OKX API erişimi kontrol ediliyor..."):
        api_get("/api/v5/public/time")
        symbols = get_active_spot_symbols()
        tickers = get_tickers()
except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"OKX erişilebilir. Aktif USDT Spot paritesi: {len(symbols)}")

universe = []
for inst_id in symbols:
    ticker = tickers.get(inst_id)
    if not ticker:
        continue
    try:
        price = float(ticker.get("last", 0))
        volume = float(ticker.get("volCcy24h", 0) or 0)
        if price > 0 and volume >= min_volume:
            universe.append((inst_id, ticker, volume))
    except (ValueError, TypeError):
        continue

# Limit scan size to reduce public API rate-limit pressure.
universe.sort(key=lambda row: row[2], reverse=True)
universe = universe[:max_coins]

st.write(f"**Taranacak parite:** {len(universe)}")
st.info(
    "12 aylık dip, son 12 tamamlanmış aylık mumun en düşük seviyesinden "
    "hesaplanır. 15 dakikalık mumlar yalnızca dip eşiğine yakın coinler "
    "için istenir."
)

if not universe:
    st.warning("Filtreye uygun parite bulunamadı. Hacim eşiğini düşürün.")
    st.stop()

results = []
failures = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(analyze_coin, inst_id, ticker, max_distance): inst_id
        for inst_id, ticker, _volume in universe
    }
    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
        try:
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                failures += 1
        except MarketAPIError as exc:
            fatal_error = str(exc)
            for pending in futures:
                pending.cancel()
            break
        except (ValueError, TypeError, KeyError, IndexError):
            failures += 1

        progress.progress(index / max(total, 1))
        status.text(f"İşlenen: {index}/{total}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.info(
        "HTTP 429 devam ederse coin sayısını azaltıp daha sonra tekrar deneyin."
    )
    st.stop()

if not results:
    st.warning("Analiz sonucu alınamadı. Hacim filtresini kontrol edin.")
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Distance From 12M Low (%)",
    ascending=True,
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analiz edilen", len(df))
c2.metric("BUY sinyali", len(buys))
c3.metric("SELL sinyali", len(sells))
c4.metric("Analiz edilemeyen", failures)

st.subheader("Son 12 aylık dip seviyesine en yakın coinler")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("BUY - Dip ve yükseliş teyidi")
if buys.empty:
    st.info("İki koşulu da karşılayan BUY sinyali bulunamadı.")
else:
    st.dataframe(buys, use_container_width=True, hide_index=True)

with st.expander("SELL sinyalleri"):
    st.dataframe(sells, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık kapanış grafiği")
selected = st.selectbox("Coin seçin", df["Coin"].tolist())

try:
    raw15 = get_candles(selected, "15m", 100)
    completed15 = confirmed_candles(raw15)
    completed15 = sorted(completed15, key=lambda candle: int(candle[0]))

    chart = pd.DataFrame({
        "Time": pd.to_datetime(
            [int(candle[0]) for candle in completed15],
            unit="ms",
            utc=True,
        ),
        "Close": [float(candle[4]) for candle in completed15],
    }).set_index("Time")

    st.line_chart(chart["Close"], use_container_width=True)
except MarketAPIError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_spot_12m_dip_scanner.csv",
    mime="text/csv",
)

st.caption(
    "Veri kaynağı OKX Spot'tur; Binance verisi değildir. Emir gönderilmez. "
    "12 aylık dip, son 12 tamamlanmış aylık mumun en düşük fiyatıdır; "
    "bu takvim ayları bazlı bir yaklaşımdır. BUY/SELL yalnızca tarama "
    "etiketleridir ve kâr garantisi vermez."
)
