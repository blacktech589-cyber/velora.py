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
DAILY_LOOKBACK = 90
MIN_SECONDS_BETWEEN_REQUESTS = 0.15

st.set_page_config(
    page_title="OKX Spot 90-Day Low Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("OKX Spot - 3 Aylık Dip Tarayıcı")
st.caption(
    "Aktif Spot USDT pariteleri | Son 90 tamamlanmış günlük mumun dibi | "
    "15 dakikalık yükseliş teyidi | Emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.1"})

_rate_lock = threading.Lock()
_last_request_time = 0.0


class MarketAPIError(Exception):
    pass


def api_get(path, params=None, retries=4):
    """Rate-limited public API call with backoff for HTTP 429."""
    global _last_request_time

    url = BASE_URL + path

    for attempt in range(retries + 1):
        with _rate_lock:
            wait = MIN_SECONDS_BETWEEN_REQUESTS - (
                time.monotonic() - _last_request_time
            )
            if wait > 0:
                time.sleep(wait)
            _last_request_time = time.monotonic()

        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
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
                "HTTP 429: Tekrarlanan beklemelere rağmen istek sınırı aşıldı. "
                "Tarama durduruldu. Daha az coin seçip sonra yeniden deneyin."
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
    """OKX candle field 8: '1' means confirmed; '0' means incomplete."""
    return [candle for candle in candles if len(candle) > 8 and candle[8] == "1"]


def analyze_coin(inst_id, ticker, max_distance):
    price = float(ticker.get("last", 0))
    if price <= 0:
        return None

    open24 = float(ticker.get("open24h", 0) or 0)
    change24 = (price / open24 - 1) * 100 if open24 > 0 else 0.0

    # 91 requested to allow for an incomplete current candle;
    # then use the last 90 confirmed daily candles.
    daily_raw = get_candles(inst_id, "1D", 91)
    daily = confirmed_candles(daily_raw)
    if len(daily) < DAILY_LOOKBACK:
        return None

    # OKX returns newest first. Keep 90 most recent confirmed candles.
    daily_90 = daily[:DAILY_LOOKBACK]
    low90 = min(float(candle[3]) for candle in daily_90)
    if low90 <= 0:
        return None

    distance = (price / low90 - 1) * 100
    near_low = distance <= max_distance

    # Only make the second request for coins near their 90-day low.
    # This greatly reduces calls and helps avoid HTTP 429.
    previous_close = None
    last_close = None
    rising = False

    if near_low:
        raw15 = get_candles(inst_id, "15m", 6)
        candles15 = confirmed_candles(raw15)

        # Sort by timestamp ascending and compare the last two completed bars.
        candles15 = sorted(candles15, key=lambda candle: int(candle[0]))
        if len(candles15) >= 2:
            previous_close = float(candles15[-2][4])
            last_close = float(candles15[-1][4])
            rising = last_close > previous_close

    signal = "BUY" if near_low and rising else "SELL"

    return {
        "Coin": inst_id,
        "Signal": signal,
        "Current Price": price,
        "90D Lowest Price": low90,
        "Distance From 90D Low (%)": round(distance, 3),
        "24H Change (%)": round(change24, 2),
        "Near 90D Low": near_low,
        "15M Previous Close": previous_close,
        "15M Last Completed Close": last_close,
        "15M Rising": rising,
    }


st.sidebar.header("Ayarlar")

max_distance = st.sidebar.slider(
    "90 günlük dip seviyesine maksimum uzaklık (%)",
    min_value=0.5,
    max_value=15.0,
    value=3.0,
    step=0.5,
)

max_coins = st.sidebar.slider(
    "Taranacak coin sayısı",
    min_value=20,
    max_value=300,
    value=100,
    step=20,
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
    st.info("Ayarları seçin ve 'Taramayı başlat' düğmesine basın.")
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
        # volCcy24h is a quote-currency volume field on OKX Spot tickers.
        volume = float(ticker.get("volCcy24h", 0) or 0)
        if price > 0 and volume >= min_volume:
            universe.append((inst_id, ticker, volume))
    except (ValueError, TypeError):
        continue

# Rank by 24h quote volume and cap scan size to reduce API pressure.
universe.sort(key=lambda row: row[2], reverse=True)
universe = universe[:max_coins]

st.write(f"**Taranacak parite:** {len(universe)}")
st.info(
    "Tarama, son 90 tamamlanmış günlük mumun en düşük fiyatını kullanır. "
    "15 dakikalık mum isteği yalnızca 90 günlük dip eşiğine yakın coinler "
    "için gönderilir; bu, API isteklerini azaltır."
)

if not universe:
    st.warning("Filtreye uyan parite bulunamadı. Hacim eşiğini düşürün.")
    st.stop()

results = []
failures = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

# Keep concurrency low; api_get also rate-limits requests globally.
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
        except Exception:
            failures += 1

        progress.progress(index / max(total, 1))
        status.text(f"İşlenen: {index}/{total}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.info(
        "İstek sınırı nedeniyle tarama durduysa taranacak coin sayısını "
        "azaltıp bir süre sonra yeniden deneyin."
    )
    st.stop()

if not results:
    st.warning("Analiz sonucu bulunamadı. Hacim filtresini kontrol edin.")
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Distance From 90D Low (%)",
    ascending=True,
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analiz edilen", len(df))
c2.metric("BUY sinyali", len(buys))
c3.metric("SELL sinyali", len(sells))
c4.metric("Analiz edilemeyen", failures)

st.subheader("90 günlük dip seviyesine en yakın coinler")
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
    raw = get_candles(selected, "15m", 100)
    completed = confirmed_candles(raw)
    completed = sorted(completed, key=lambda candle: int(candle[0]))

    chart = pd.DataFrame({
        "Time": pd.to_datetime(
            [int(candle[0]) for candle in completed],
            unit="ms",
            utc=True,
        ),
        "Close": [float(candle[4]) for candle in completed],
    }).set_index("Time")

    st.line_chart(chart["Close"], use_container_width=True)
except MarketAPIError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_spot_90d_dip_scanner.csv",
    mime="text/csv",
)

st.caption(
    "OKX halka açık Spot verileri kullanılır; Binance verisi değildir. "
    "Emir gönderilmez. BUY/SELL yalnızca tarama etiketleridir ve kâr garantisi "
    "vermez. SELL, otomatik satış emri anlamına gelmez."
)
