import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.binance.com"
TIMEOUT = 15

st.set_page_config(
    page_title="Binance Spot Dip Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("Binance Spot - 30 Günlük Dip Tarayıcı")
st.caption(
    "Aktif Spot USDT pariteleri | 30 günlük dip sıralaması | "
    "Tamamlanmış 15 dakikalık mumlar | Emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.0"})


class BinanceError(Exception):
    """Raised when Binance public API access fails."""


def api_get(path, params=None):
    try:
        response = session.get(
            BASE_URL + path,
            params=params,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BinanceError(f"Bağlantı hatası: {exc}") from exc

    if response.status_code == 451:
        raise BinanceError(
            "HTTP 451: Binance bu bağlantı konumundan erişimi kısıtlıyor. "
            "Kısıtlama aşılmaya çalışılmayacak."
        )
    if response.status_code == 429:
        raise BinanceError(
            "HTTP 429: Binance istek sınırına ulaşıldı. Tarama durduruldu; "
            "bir süre bekleyip yeniden deneyin."
        )
    if response.status_code == 418:
        raise BinanceError(
            "HTTP 418: Binance geçici IP engeli bildirdi. Tarama durduruldu."
        )
    if not response.ok:
        raise BinanceError(
            f"Binance HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BinanceError("Binance geçerli JSON döndürmedi.") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols():
    data = api_get("/api/v3/exchangeInfo")
    return [
        item["symbol"]
        for item in data.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", True)
        and item.get("symbol", "").endswith("USDT")
    ]


@st.cache_data(ttl=180, show_spinner=False)
def get_tickers():
    data = api_get("/api/v3/ticker/24hr")
    return {item["symbol"]: item for item in data if "symbol" in item}


def get_klines(symbol, interval, limit):
    return api_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )


def analyze_coin(symbol, ticker, max_distance):
    try:
        price = float(ticker["lastPrice"])
        volume = float(ticker["quoteVolume"])
        change24 = float(ticker["priceChangePercent"])
        if price <= 0:
            return None

        # Use the latest 31 daily candles and exclude the possibly
        # incomplete current candle, leaving 30 completed daily candles.
        daily = get_klines(symbol, "1d", 31)
        if len(daily) < 30:
            return None

        completed_daily = daily[:-1][-30:]
        low30 = min(float(candle[3]) for candle in completed_daily)
        if low30 <= 0:
            return None

        distance = (price / low30 - 1.0) * 100.0

        # Exclude the currently forming 15-minute candle.
        candles = get_klines(symbol, "15m", 4)
        if len(candles) < 3:
            return None

        completed = candles[:-1]
        previous_close = float(completed[-2][4])
        last_close = float(completed[-1][4])

        rising = last_close > previous_close
        near_low = distance <= max_distance
        signal = "BUY" if near_low and rising else "SELL"

        return {
            "Coin": symbol,
            "Signal": signal,
            "Current Price": price,
            "30D Lowest Price": low30,
            "Distance From Low (%)": round(distance, 3),
            "24H Change (%)": round(change24, 2),
            "24H Volume USDT": round(volume, 0),
            "15M Previous Close": previous_close,
            "15M Last Completed Close": last_close,
            "Near 30D Low": near_low,
            "15M Rising": rising,
        }
    except BinanceError:
        raise
    except (ValueError, TypeError, KeyError, IndexError):
        return None


st.sidebar.header("Ayarlar")

max_distance = st.sidebar.slider(
    "30 günlük dip seviyesine maksimum uzaklık (%)",
    min_value=0.5,
    max_value=15.0,
    value=3.0,
    step=0.5,
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0,
    value=50000,
    step=50000,
)

workers = st.sidebar.select_slider(
    "Eşzamanlı istek sayısı",
    options=[1, 2, 3, 4],
    value=2,
)

if st.sidebar.button("Önbelleği temizle"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Tüm uygun coinleri tara", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Tarama yapmak için 'Tüm uygun coinleri tara' düğmesine basın.")
    st.stop()

try:
    with st.spinner("Binance API erişimi ve aktif pariteler kontrol ediliyor..."):
        api_get("/api/v3/time")
        symbols = get_active_symbols()
        tickers = get_tickers()
except BinanceError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Binance API erişilebilir. Aktif USDT Spot paritesi: {len(symbols)}")

universe = []
for symbol in symbols:
    ticker = tickers.get(symbol)
    if not ticker:
        continue
    try:
        if (
            float(ticker["lastPrice"]) > 0
            and float(ticker["quoteVolume"]) >= min_volume
        ):
            universe.append((symbol, ticker))
    except (ValueError, TypeError, KeyError):
        continue

st.write(f"**Analiz edilecek parite sayısı:** {len(universe)}")

if not universe:
    st.warning("Hacim filtresine uyan parite yok. Minimum hacmi düşürüp deneyin.")
    st.stop()

results = []
failures = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(analyze_coin, symbol, ticker, max_distance): symbol
        for symbol, ticker in universe
    }
    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
        try:
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                failures += 1
        except BinanceError as exc:
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
    st.stop()

if not results:
    st.warning(
        "Analiz sonucu alınamadı. Binance erişimini ve hacim filtresini kontrol edin."
    )
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Distance From Low (%)", ascending=True
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Analiz edilen", len(df))
col2.metric("BUY sinyali", len(buys))
col3.metric("SELL sinyali", len(sells))
col4.metric("Analiz edilemeyen", failures)

st.subheader("30 günlük dip seviyesine en yakın coinler")
st.caption(
    "Sıralama, güncel fiyatın son 30 tamamlanmış günlük mumun en düşük "
    "seviyesinden yüzde uzaklığına göredir."
)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("BUY - Dip ve yükseliş teyidi")
if buys.empty:
    st.info("Şu anda iki koşulu da karşılayan BUY sinyali bulunamadı.")
else:
    st.dataframe(buys, use_container_width=True, hide_index=True)

with st.expander("SELL sinyalleri"):
    st.dataframe(sells, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık kapanış grafiği")
selected = st.selectbox("Coin seçin", df["Coin"].tolist())

try:
    chart_candles = get_klines(selected, "15m", 100)
    chart_candles = chart_candles[:-1]  # Exclude incomplete candle
    chart = pd.DataFrame(
        {
            "Time": pd.to_datetime(
                [int(candle[0]) for candle in chart_candles],
                unit="ms",
                utc=True,
            ),
            "Close": [float(candle[4]) for candle in chart_candles],
        }
    ).set_index("Time")
    st.line_chart(chart["Close"], use_container_width=True)
except BinanceError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="binance_30d_dip_scanner.csv",
    mime="text/csv",
)

st.caption(
    "Bu uygulama yalnızca halka açık piyasa verilerini okur ve emir göndermez. "
    "BUY/SELL etiketleri basit tarama koşullarıdır; kâr garantisi değildir. "
    "SELL, otomatik satış veya yatırım tavsiyesi anlamına gelmez."
)
