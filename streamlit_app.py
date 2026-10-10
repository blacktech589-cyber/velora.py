```python
import time
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.binance.com"
TIMEOUT = 15

st.set_page_config(
    page_title="Binance All-Spot Dip Scanner",
    page_icon="📉",
    layout="wide"
)

st.title("Binance Spot — 30 Günlük Dip Tarayıcı")
st.caption(
    "Aktif Spot USDT pariteleri | 30 günlük dip sıralaması | "
    "Tamamlanmış 15 dakikalık mumlar | Emir gönderilmez"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.0"})


class BinanceError(Exception):
    pass


def api_get(path, params=None):
    try:
        response = session.get(
            BASE_URL + path,
            params=params,
            timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise BinanceError(f"Bağlantı hatası: {exc}") from exc

    if response.status_code == 451:
        raise BinanceError(
            "HTTP 451: Binance bu bağlantı konumundan erişimi "
            "kısıtlıyor. Kısıtlama aşılmaya çalışılmayacak."
        )

    if response.status_code == 429:
        raise BinanceError(
            "HTTP 429: Binance istek sınırına ulaşıldı. "
            "Tarama durduruldu; bir süre sonra yeniden deneyin."
        )

    if response.status_code == 418:
        raise BinanceError(
            "HTTP 418: Binance geçici IP engeli bildirdi. "
            "Tarama durduruldu."
        )

    if not response.ok:
        raise BinanceError(
            f"Binance HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BinanceError("Geçersiz API yanıtı.") from exc


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


@st.cache_data(ttl=300, show_spinner=False)
def get_tickers():
    data = api_get("/api/v3/ticker/24hr")
    return {x["symbol"]: x for x in data if "symbol" in x}


def get_klines(symbol, interval, limit):
    return api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )


def analyze_coin(symbol, ticker, max_distance):
    try:
        price = float(ticker["lastPrice"])
        volume = float(ticker["quoteVolume"])
        change24 = float(ticker["priceChangePercent"])

        if price <= 0:
            return None

        # 31 günlük mum al; en son mum tamamlanmamış olabileceği için
        # onu çıkar ve son 30 tamamlanmış günlük mumu kullan.
        daily = get_klines(symbol, "1d", 31)

        if len(daily) < 30:
            return None

        completed_daily = daily[:-1][-30:]
        daily_lows = [float(candle[3]) for candle in completed_daily]
        low30 = min(daily_lows)

        if low30 <= 0:
            return None

        distance = (price / low30 - 1) * 100

        # Son iki tamamlanmış 15 dakikalık mumun kapanışları.
        candles = get_klines(symbol, "15m", 4)

        if len(candles) < 3:
            return None

        completed = candles[:-1]
        previous = float(completed[-2][4])
        last = float(completed[-1][4])

        rising = last > previous
        near_low = distance <= max_distance

        # Yalnızca dip bölgesi + yükseliş teyidi BUY olur.
        # Diğer coinler SELL olarak işaretlenir; bu emir değildir.
        signal = "BUY" if near_low and rising else "SELL"

        return {
            "Coin": symbol,
            "Signal": signal,
            "Current Price": price,
            "30D Lowest Price": low30,
            "Distance From Low (%)": round(distance, 3),
            "24H Change (%)": round(change24, 2),
            "24H Volume USDT": round(volume, 0),
            "15M Previous Close": previous,
            "15M Last Completed Close": last,
            "Near 30D Low": near_low,
            "15M Rising": rising
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
    step=0.5
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0,
    value=50000,
    step=50000
)

workers = st.sidebar.select_slider(
    "Eşzamanlı istek sayısı",
    options=[1, 2, 3, 4],
    value=2
)

if st.sidebar.button("Önbelleği temizle ve yeniden tara"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Tüm uygun coinleri tara", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Taramayı başlatmak için sol ayarları belirleyip düğmeye bas.")
    st.stop()

try:
    with st.spinner("Binance API erişimi ve aktif pariteler kontrol ediliyor..."):
        api_get("/api/v3/time")
        symbols = get_active_symbols()
        tickers = get_tickers()
except BinanceError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Binance erişimi başarılı. Aktif USDT Spot paritesi: {len(symbols)}")

# Hacim filtresi uygula ama sabit bir coin sayısıyla sınırlama.
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

st.write(f"**Analiz edilecek parite:** {len(universe)}")

results = []
failures = 0
progress = st.progress(0)
status = st.empty()
fatal_error = None

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(analyze_coin, symbol, ticker, max_distance): symbol
        for symbol, ticker in universe
    }

    total = len(futures)

    for i, future in enumerate(as_completed(futures), start=1):
        try:
            item = future.result()
            if item is not None:
                results.append(item)
            else:
                failures += 1
        except BinanceError as exc:
            fatal_error = str(exc)
            for pending in futures:
                pending.cancel()
            break
        except Exception:
            failures += 1

        progress.progress(i / max(total, 1))
        status.text(f"İşlenen: {i}/{total}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "Analiz sonucu bulunamadı. Hacim filtresini düşürüp tekrar deneyin."
    )
    st.stop()

df = pd.DataFrame(results)

# En yakındaki dipler en üstte.
df = df.sort_values(
    "Distance From Low (%)",
    ascending=True
).reset_index(drop=True)

buys = df[df["Signal"] == "BUY"].copy()
sells = df[df["Signal"] == "SELL"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analiz edilen", len(df))
c2.metric("BUY sinyali", len(buys))
c3.metric("SELL sinyali", len(sells))
c4.metric("Analiz edilemeyen", failures)

st.subheader("En düşük seviyelerine en yakın coinler")
st.caption(
    "Sıralama, mevcut fiyatın son 30 tamamlanmış günlük mumun "
    "en düşük seviyesinden yüzde uzaklığına göredir."
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

st.subheader("BUY — Dip + yükseliş teyidi")

if buys.empty:
    st.info("Belirlenen koşulları karşılayan BUY sinyali yok.")
else:
    st.dataframe(
        buys,
        use_container_width=True,
        hide_index=True
    )

with st.expander("SELL sinyalleri"):
    st.dataframe(
        sells,
        use_container_width=True,
        hide_index=True
    )

st.subheader("Coin grafiği")
selected = st.selectbox("Coin seç", df["Coin"].tolist())

try:
    chart_candles = get_klines(selected, "15m", 100)
    chart_candles = chart_candles[:-1]

    chart = pd.DataFrame({
        "Time": pd.to_datetime(
            [int(x[0]) for x in chart_candles],
            unit="ms",
            utc=True
        ),
        "Close": [float(x[4]) for x in chart_candles]
    }).set_index("Time")

    st.line_chart(chart["Close"])

except BinanceError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV indir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="binance_30d_dip_scanner.csv",
    mime="text/csv"
)

st.caption(
    "Sistem emir göndermez. BUY/SELL etiketleri otomatik alım/satım "
    "değildir. Dip seviyeleri gelecekteki fiyat düşüşlerini engellemez."
)
```
