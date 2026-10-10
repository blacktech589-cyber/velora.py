import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# Public market data only. No API keys, no order execution.
# Active OKX Spot instruments are filtered by state == "live".
BASE_URL = "https://www.okx.com"
TIMEOUT = 15
MIN_SECONDS_BETWEEN_REQUESTS = 0.20
LOOKBACK_MONTHS = 12
DROP_THRESHOLD_PCT = 90.0

st.set_page_config(
    page_title="OKX Spot - 90% Drawdown Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("OKX Spot — Son 12 Ayda %90+ Düşen Coinler")
st.caption(
    "Yalnızca aktif/delist edilmemiş Spot USDT pariteleri | "
    "12 aylık zirveden en az %90 düşüş | 15 dakikalık mum teyidi | Emir yok"
)

session = requests.Session()
session.headers.update({"User-Agent": "OKX-Spot-Drawdown-Scanner/1.0"})
rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    pass


def api_get(path, params=None, retries=4):
    global last_request_time

    for attempt in range(retries + 1):
        with rate_lock:
            delay = MIN_SECONDS_BETWEEN_REQUESTS - (
                time.monotonic() - last_request_time
            )
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
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else 2 ** (attempt + 1)
                except ValueError:
                    wait_seconds = 2 ** (attempt + 1)
                time.sleep(min(max(wait_seconds, 2), 30))
                continue
            raise MarketAPIError(
                "HTTP 429: API istek sınırı aşıldı. Coin sayısını azaltıp "
                "bir süre sonra tekrar deneyin."
            )

        if not response.ok:
            raise MarketAPIError(f"HTTP {response.status_code}: {response.text[:250]}")

        try:
            data = response.json()
        except ValueError as exc:
            raise MarketAPIError("API geçerli JSON döndürmedi.") from exc

        if isinstance(data, dict) and data.get("code") not in (None, "0"):
            raise MarketAPIError(
                f"OKX API hatası {data.get('code')}: {data.get('msg', '')}"
            )
        return data

    raise MarketAPIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_spot_symbols():
    # OKX public instruments endpoint; only instruments marked live are included.
    payload = api_get("/api/v5/public/instruments", {"instType": "SPOT"})
    return [
        item["instId"]
        for item in payload.get("data", [])
        if item.get("state") == "live"
        and item.get("quoteCcy") == "USDT"
        and item.get("instId", "").endswith("-USDT")
    ]


@st.cache_data(ttl=180, show_spinner=False)
def get_tickers():
    payload = api_get("/api/v5/market/tickers", {"instType": "SPOT"})
    return {item["instId"]: item for item in payload.get("data", []) if item.get("instId")}


def get_candles(inst_id, bar, limit):
    payload = api_get(
        "/api/v5/market/candles",
        {"instId": inst_id, "bar": bar, "limit": str(limit)},
    )
    return payload.get("data", [])


def confirmed(candles):
    # OKX candle field index 8: 1 = completed, 0 = incomplete.
    return [
        candle for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(inst_id, ticker, min_drop_pct, max_drop_pct):
    try:
        current = float(ticker.get("last", 0) or 0)
        if current <= 0:
            return None

        open24 = float(ticker.get("open24h", 0) or 0)
        change24 = (current / open24 - 1) * 100 if open24 > 0 else 0.0

        # Monthly candles, 14 requested to allow excluding the incomplete current month.
        monthly = confirmed(get_candles(inst_id, "1M", 14))
        monthly.sort(key=lambda candle: int(candle[0]), reverse=True)
        monthly = monthly[:LOOKBACK_MONTHS]
        if len(monthly) < LOOKBACK_MONTHS:
            return None

        high12 = max(float(candle[2]) for candle in monthly)
        low12 = min(float(candle[3]) for candle in monthly)
        if high12 <= 0:
            return None

        drop_pct = (1 - current / high12) * 100
        if drop_pct < min_drop_pct or drop_pct > max_drop_pct:
            return None

        # Rise confirmation from the last two completed 15-minute candles.
        candles15 = confirmed(get_candles(inst_id, "15m", 6))
        candles15.sort(key=lambda candle: int(candle[0]))
        if len(candles15) < 2:
            previous_close = None
            last_close = None
            rising = False
        else:
            previous_close = float(candles15[-2][4])
            last_close = float(candles15[-1][4])
            rising = last_close > previous_close

        signal = "BUY" if rising else "SELL"
        return {
            "Coin": inst_id,
            "Signal": signal,
            "Current Price": current,
            "12M Highest Price": high12,
            "12M Lowest Price": low12,
            "Drop From 12M High (%)": round(drop_pct, 2),
            "24H Change (%)": round(change24, 2),
            "Previous 15M Close": previous_close,
            "Last Completed 15M Close": last_close,
            "15M Rising": rising,
        }
    except (ValueError, TypeError, KeyError, IndexError):
        return None


st.sidebar.header("Tarama ayarları")
min_drop = st.sidebar.slider(
    "12 aylık zirveden minimum düşüş (%)",
    min_value=90.0,
    max_value=99.0,
    value=90.0,
    step=1.0,
)
max_drop = st.sidebar.slider(
    "12 aylık zirveden maksimum düşüş (%)",
    min_value=90.0,
    max_value=99.9,
    value=99.9,
    step=0.1,
)
max_coins = st.sidebar.slider(
    "Taranacak en yüksek hacimli coin sayısı",
    min_value=10,
    max_value=150,
    value=50,
    step=10,
)
min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0,
    value=50000,
    step=50000,
)

if min_drop > max_drop:
    st.sidebar.error("Minimum düşüş, maksimum düşüşten büyük olamaz.")

if st.sidebar.button("Önbelleği temizle"):
    get_active_spot_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Taramayı başlat", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Ayarları seçip 'Taramayı başlat' düğmesine basın.")
    st.stop()

if min_drop > max_drop:
    st.error("Düşüş aralığını düzeltin.")
    st.stop()

try:
    with st.spinner("OKX API erişimi ve aktif pariteler kontrol ediliyor..."):
        api_get("/api/v5/public/time")
        symbols = get_active_spot_symbols()
        tickers = get_tickers()
except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"API erişilebilir. Aktif USDT Spot paritesi: {len(symbols)}")

universe = []
for symbol in symbols:
    ticker = tickers.get(symbol)
    if not ticker:
        continue
    try:
        last = float(ticker.get("last", 0) or 0)
        volume = float(ticker.get("volCcy24h", 0) or 0)
        if last > 0 and volume >= min_volume:
            universe.append((symbol, ticker, volume))
    except (ValueError, TypeError):
        continue

universe.sort(key=lambda item: item[2], reverse=True)
universe = universe[:max_coins]

if not universe:
    st.warning("Hacim filtresine uygun aktif USDT paritesi bulunamadı.")
    st.stop()

st.write(f"**Taranacak coin:** {len(universe)}")
st.info(
    "Delist edilmiş pariteler, OKX enstrüman listesinde state='live' olmadığı "
    "için dışarıda bırakılır. Bu tarama OKX Spot USDT paritelerini kullanır; "
    "Binance listesi değildir."
)

results = []
errors = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(analyze_coin, symbol, ticker, min_drop, max_drop): symbol
        for symbol, ticker, _volume in universe
    }
    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
        try:
            item = future.result()
            if item is not None:
                results.append(item)
            else:
                errors += 1
        except MarketAPIError as exc:
            fatal_error = str(exc)
            for pending in futures:
                pending.cancel()
            break

        progress.progress(index / max(total, 1))
        status.text(f"İşlenen: {index}/{total}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "Bu taramada %90+ düşüş koşuluna uyan coin bulunamadı. "
        "Coin sayısını artırabilir veya hacim eşiğini düşürebilirsiniz."
    )
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Drop From 12M High (%)",
    ascending=False,
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

a, b, c, d = st.columns(4)
a.metric("Koşula uyan coin", len(df))
b.metric("BUY etiketi", len(buys))
c.metric("SELL etiketi", len(sells))
d.metric("Veri alınamayan", errors)

st.subheader("12 aylık zirvesinden %90 veya daha fazla düşen aktif coinler")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("BUY — %90+ düşüş ve son 15 dakikalık kapanış yükselişi")
if buys.empty:
    st.info("Koşulları karşılayan BUY etiketi yok.")
else:
    st.dataframe(buys, use_container_width=True, hide_index=True)

with st.expander("SELL etiketleri (15 dakikalık yükseliş teyidi yok)"):
    st.dataframe(sells, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık kapanış grafiği")
selected = st.selectbox("Coin seçin", df["Coin"].tolist())
try:
    candles = confirmed(get_candles(selected, "15m", 100))
    candles.sort(key=lambda candle: int(candle[0]))
    chart = pd.DataFrame({
        "Time (UTC)": pd.to_datetime(
            [int(candle[0]) for candle in candles],
            unit="ms",
            utc=True,
        ),
        "Close": [float(candle[4]) for candle in candles],
    }).set_index("Time (UTC)")
    st.line_chart(chart["Close"], use_container_width=True)
except MarketAPIError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_active_spot_coins_down_90pct_12m.csv",
    mime="text/csv",
)

st.caption(
    "Uyarı: Aylık mumlar son 12 tamamlanmış takvim ayını kapsar; kayan 365 günle "
    "birebir aynı değildir. Aylık mumların en yüksek değeri, o ayın zirvesinin "
    "yaklaşık ölçümüdür. BUY/SELL yalnızca filtre etiketidir, yatırım tavsiyesi "
    "veya kâr garantisi değildir. API'de live görünen enstrümanlar alınır; "
    "OKX'te listelenen aktif coinlerdir, tüm borsaları kapsamaz."
)
