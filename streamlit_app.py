
import time
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.binance.com"
TIMEOUT = 15

st.set_page_config(
    page_title="Binance Spot - %50+ Düşüş Tarayıcı",
    page_icon="📉",
    layout="wide",
)

st.title("Binance Spot — %50+ Düşen Coin Tarayıcı")
st.caption(
    "Aktif USDT Spot pariteleri | Son 12 tamamlanmış aylık mum "
    "| 15 dakikalık kapanış teyidi | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({"User-Agent": "SpotDrawdownScanner/1.0"})


class APIError(Exception):
    pass


def api_get(path, params=None, retries=3):
    for attempt in range(retries + 1):
        try:
            response = session.get(
                BASE_URL + path,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code == 451:
                raise APIError(
                    "Binance API bu bağlantı konumundan erişilemiyor (451). "
                    "Bölgesel erişim kısıtlamalarını aşmaya çalışmadan durduruldu."
                )

            if response.status_code in (418, 429):
                if attempt < retries:
                    time.sleep(min(2 ** (attempt + 1), 20))
                    continue
                raise APIError(
                    f"Binance istek sınırı hatası: HTTP {response.status_code}"
                )

            if not response.ok:
                raise APIError(
                    f"HTTP {response.status_code}: {response.text[:250]}"
                )

            data = response.json()

            if isinstance(data, dict) and "code" in data:
                if isinstance(data["code"], int) and data["code"] < 0:
                    raise APIError(data.get("msg", "Binance API hatası"))

            return data

        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 15))
                continue
            raise APIError(f"Bağlantı hatası: {exc}") from exc

    raise APIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols():
    data = api_get("/api/v3/exchangeInfo")

    return {
        item["symbol"]
        for item in data["symbols"]
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", False)
        and item.get("baseAsset") != "USDT"
    }


@st.cache_data(ttl=120, show_spinner=False)
def get_tickers():
    data = api_get("/api/v3/ticker/24hr")
    return {
        item["symbol"]: item
        for item in data
        if item.get("symbol")
    }


def get_klines(symbol, interval, limit=15):
    return api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )


def analyze_coin(symbol, ticker, min_drop, max_drop):
    try:
        current = float(ticker["lastPrice"])
        open24 = float(ticker["openPrice"])
        volume = float(ticker["quoteVolume"])

        if current <= 0:
            return None

        change24 = (
            (current / open24 - 1) * 100
            if open24 > 0
            else 0.0
        )

        now_ms = int(time.time() * 1000)

        # Son 12 tamamlanmış aylık mum.
        monthly = get_klines(symbol, "1M", 14)
        monthly = [
            candle for candle in monthly
            if int(candle[6]) < now_ms
        ]

        monthly = monthly[-12:]

        if len(monthly) < 12:
            return None

        high12 = max(float(candle[2]) for candle in monthly)
        low12 = min(float(candle[3]) for candle in monthly)

        if high12 <= 0:
            return None

        drop_pct = (1 - current / high12) * 100

        if drop_pct < min_drop or drop_pct > max_drop:
            return None

        # Son iki tamamlanmış 15 dakikalık mum.
        candles = get_klines(symbol, "15m", 5)
        candles = [
            candle for candle in candles
            if int(candle[6]) < now_ms
        ]

        if len(candles) < 2:
            previous_close = None
            last_close = None
            rising = False
            signal = "YETERSİZ VERİ"
        else:
            previous_close = float(candles[-2][4])
            last_close = float(candles[-1][4])
            rising = last_close > previous_close
            signal = "BUY" if rising else "SELL"

        return {
            "Coin": symbol,
            "Sinyal": signal,
            "Güncel Fiyat": current,
            "12A Zirve": high12,
            "12A Dip": low12,
            "Zirveden Düşüş (%)": round(drop_pct, 2),
            "24S Değişim (%)": round(change24, 2),
            "24S Hacim (USDT)": round(volume, 2),
            "Önceki 15D Kapanış": previous_close,
            "Son 15D Kapanış": last_close,
            "15D Yükseliyor": rising,
        }

    except APIError:
        raise
    except (ValueError, TypeError, KeyError, IndexError):
        return None


st.sidebar.header("Tarama Ayarları")

min_drop = st.sidebar.slider(
    "Minimum düşüş (%)",
    min_value=50,
    max_value=99,
    value=50,
    step=1,
)

max_drop = st.sidebar.slider(
    "Maksimum düşüş (%)",
    min_value=50,
    max_value=99,
    value=99,
    step=1,
)

max_coins = st.sidebar.select_slider(
    "Taranacak coin sayısı (hacme göre)",
    options=[50, 100, 150, 200, 300, 500],
    value=150,
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

if min_drop > max_drop:
    st.sidebar.error("Minimum düşüş maksimum düşüşü aşamaz.")

if st.button("Taramayı Başlat", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Ayarları seç ve Taramayı Başlat düğmesine bas.")
    st.stop()

if min_drop > max_drop:
    st.error("Düşüş aralığını düzelt.")
    st.stop()

try:
    with st.spinner("Binance Spot pariteleri alınıyor..."):
        symbols = get_active_symbols()
        tickers = get_tickers()

except APIError as exc:
    st.error(str(exc))
    st.stop()

universe = []

for symbol in symbols:
    ticker = tickers.get(symbol)

    if ticker is None:
        continue

    try:
        price = float(ticker["lastPrice"])
        volume = float(ticker["quoteVolume"])

        if price > 0 and volume >= min_volume:
            universe.append((symbol, ticker, volume))

    except (ValueError, TypeError, KeyError):
        continue

universe.sort(key=lambda item: item[2], reverse=True)
universe = universe[:max_coins]

st.write(f"**Aktif USDT Spot pariteleri:** {len(symbols)}")
st.write(f"**Taranacak parite:** {len(universe)}")

if not universe:
    st.warning(
        "Hacim filtresine uygun aktif parite bulunamadı. "
        "Minimum hacmi azaltmayı dene."
    )
    st.stop()

results = []
errors = 0
fatal_error = None

progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(
            analyze_coin,
            symbol,
            ticker,
            min_drop,
            max_drop,
        ): symbol
        for symbol, ticker, _ in universe
    }

    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
        try:
            result = future.result()

            if result is not None:
                results.append(result)
            else:
                errors += 1

        except APIError as exc:
            fatal_error = str(exc)
            for pending in futures:
                pending.cancel()
            break

        except Exception:
            errors += 1

        progress.progress(index / max(total, 1))
        status.text(f"İşlenen: {index}/{total}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "Seçilen düşüş aralığına uygun coin bulunamadı. "
        "Taranan coin sayısını artırabilir veya hacim filtresini azaltabilirsin."
    )
    st.stop()

df = pd.DataFrame(results).sort_values(
    "Zirveden Düşüş (%)",
    ascending=False,
).reset_index(drop=True)

buys = df[df["Sinyal"] == "BUY"]
sells = df[df["Sinyal"] == "SELL"]

a, b, c, d = st.columns(4)
a.metric("Bulunan coin", len(df))
b.metric("BUY etiketi", len(buys))
c.metric("SELL etiketi", len(sells))
d.metric("Veri alınamayan", errors)

st.subheader("Zirvesinden seçilen oranda düşmüş aktif coinler")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("BUY — Son 15 dakikalık kapanış yükselmiş")
st.dataframe(buys, use_container_width=True, hide_index=True)

with st.expander("SELL — Son kapanış yükselmemiş"):
    st.dataframe(sells, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık fiyat grafiği")
selected = st.selectbox("Coin seç", df["Coin"].tolist())

try:
    now_ms = int(time.time() * 1000)
    candles = get_klines(selected, "15m", 100)
    candles = [
        candle for candle in candles
        if int(candle[6]) < now_ms
    ]

    chart = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime(
            [int(candle[0]) for candle in candles],
            unit="ms",
            utc=True,
        ),
        "Kapanış": [
            float(candle[4]) for candle in candles
        ],
    }).set_index("Zaman (UTC)")

    st.line_chart(chart["Kapanış"], use_container_width=True)

except APIError as exc:
    st.warning(str(exc))

st.download_button(
    "CSV İndir",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="binance_spot_50pct_drawdown.csv",
    mime="text/csv",
)

st.caption(
    "Düşüş, son 12 tamamlanmış aylık mumun en yüksek fiyatına göre "
    "hesaplanır; kayan 365 günlük zirveyle birebir aynı değildir. "
    "BUY/SELL etiketleri yalnızca son iki tamamlanmış 15 dakikalık "
    "kapanışın karşılaştırmasıdır ve kâr garantisi vermez. "
    "Tarayıcı otomatik emir göndermez."
)
