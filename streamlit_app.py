
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.okx.com"
TIMEOUT = 15
QUOTE_CURRENCY = "USDT"

st.set_page_config(
    page_title="Spot 30-Day Low Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("Spot - 30 Günlük Dip Tarayıcı")
st.caption(
    "OKX Spot USDT pariteleri | 30 günlük dip sıralaması | "
    "Tamamlanmış 15 dakikalık mumlar | Emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.0"})


class MarketAPIError(Exception):
    pass


def api_get(path, params=None):
    try:
        response = session.get(
            BASE_URL + path,
            params=params,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise MarketAPIError(f"Bağlantı hatası: {exc}") from exc

    if response.status_code == 451:
        raise MarketAPIError(
            "HTTP 451: Veri sağlayıcısı bu bağlantı konumundan "
            "erişimi kısıtlıyor. Kısıtlama aşılmayacak."
        )

    if response.status_code == 429:
        raise MarketAPIError(
            "HTTP 429: İstek sınırına ulaşıldı. Bir süre sonra tekrar deneyin."
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


def analyze_coin(inst_id, ticker, max_distance):
    price = float(ticker.get("last", 0))
    if price <= 0:
        return None

    open24 = float(ticker.get("open24h", 0) or 0)
    change24 = (price / open24 - 1) * 100 if open24 > 0 else 0.0

    # OKX candles are newest-first.
    # Exclude the newest potentially incomplete daily candle.
    daily = get_candles(inst_id, "1D", 31)
    if len(daily) < 31:
        return None

    completed_daily = daily[1:31]
    low30 = min(float(candle[3]) for candle in completed_daily)

    if low30 <= 0:
        return None

    distance = (price / low30 - 1) * 100

    # Exclude the newest potentially incomplete 15-minute candle.
    candles15 = get_candles(inst_id, "15m", 4)
    if len(candles15) < 3:
        return None

    completed15 = list(reversed(candles15[1:]))
    previous_close = float(completed15[-2][4])
    last_close = float(completed15[-1][4])

    rising = last_close > previous_close
    near_low = distance <= max_distance

    # SELL means the BUY screening rules are not satisfied.
    # It does not execute or recommend an actual sale.
    signal = "BUY" if near_low and rising else "SELL"

    return {
        "Coin": inst_id,
        "Signal": signal,
        "Current Price": price,
        "30D Lowest Price": low30,
        "Distance From Low (%)": round(distance, 3),
        "24H Change (%)": round(change24, 2),
        "15M Previous Close": previous_close,
        "15M Last Completed Close": last_close,
        "Near 30D Low": near_low,
        "15M Rising": rising,
    }


st.sidebar.header("Ayarlar")

max_distance = st.sidebar.slider(
    "30 günlük dip seviyesine maksimum uzaklık (%)",
    min_value=0.5,
    max_value=15.0,
    value=3.0,
    step=0.5,
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT karşılığı)",
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
    get_active_spot_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Tüm uygun coinleri tara", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Taramayı başlatmak için düğmeye basın.")
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
        # volCcy24h is used as an approximate volume filter.
        volume = float(ticker.get("volCcy24h", 0) or 0)

        if price > 0 and volume >= min_volume:
            universe.append((inst_id, ticker))
    except (ValueError, TypeError):
        continue

st.write(f"**Analiz edilecek parite sayısı:** {len(universe)}")

if not universe:
    st.warning("Filtreye uygun parite yok. Minimum hacmi azaltıp deneyin.")
    st.stop()

results = []
failures = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(
            analyze_coin, inst_id, ticker, max_distance
        ): inst_id
        for inst_id, ticker in universe
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
    st.stop()

if not results:
    st.warning(
        "Analiz sonucu alınamadı. API erişimini ve hacim filtresini kontrol edin."
    )
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Distance From Low (%)",
    ascending=True,
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analiz edilen", len(df))
c2.metric("BUY sinyali", len(buys))
c3.metric("SELL sinyali", len(sells))
c4.metric("Analiz edilemeyen", failures)

st.subheader("30 günlük dip seviyesine en yakın coinler")
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
    chart_candles = get_candles(selected, "15m", 100)
    chart_candles = list(reversed(chart_candles[1:]))

    chart = pd.DataFrame({
        "Time": pd.to_datetime(
            [int(candle[0]) for candle in chart_candles],
            unit="ms",
            utc=True,
        ),
        "Close": [float(candle[4]) for candle in chart_candles],
    }).set_index("Time")

    st.line_chart(chart["Close"], use_container_width=True)

except MarketAPIError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_spot_30d_dip_scanner.csv",
    mime="text/csv",
)

st.caption(
    "OKX halka açık Spot verileri kullanılır. Binance verisi kullanılmaz. "
    "Emir gönderilmez. BUY/SELL basit tarama etiketleridir; kâr garantisi "
    "veya yatırım tavsiyesi değildir."
)
