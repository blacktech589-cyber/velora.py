import time
import threading
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# OKX Spot public API.
# API anahtarı gerekmez, otomatik emir gönderilmez.
BASE_URL = "https://www.okx.com"
TIMEOUT = 15
MIN_REQUEST_INTERVAL = 0.20

# Yaklaşık son 3 ay = 90 tamamlanmış günlük mum.
LOOKBACK_DAYS = 90

st.set_page_config(
    page_title="3 Aylık Düşüş Tarayıcısı",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Son 3 Ayda %60+ Düşen Coin Tarayıcısı")
st.caption(
    "Aktif OKX Spot USDT pariteleri | "
    "90 günlük fiyat zirvesinden düşüş | "
    "15 dakikalık yükseliş teyidi | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({
    "User-Agent": "OKX-Spot-Drop-Scanner/1.0"
})

rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    """OKX API erişim ve yanıt hataları."""
    pass


def api_get(path, params=None, retries=4):
    global last_request_time

    for attempt in range(retries + 1):

        # API isteklerinin çok sık gönderilmesini engelle.
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
                "HTTP 429: API istek sınırına ulaşıldı. "
                "Coin sayısını azaltıp daha sonra tekrar deneyin."
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
    """
    Yalnızca OKX tarafından state='live' olarak işaretlenen
    aktif Spot USDT paritelerini alır.
    """
    payload = api_get(
        "/api/v5/public/instruments",
        {"instType": "SPOT"},
    )

    symbols = []

    for item in payload.get("data", []):
        if (
            item.get("state") == "live"
            and item.get("quoteCcy") == "USDT"
            and item.get("instId", "").endswith("-USDT")
        ):
            symbols.append(item["instId"])

    return symbols


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
    """
    OKX mum verisinde 8. indeks:
    1 = tamamlanmış mum
    0 = tamamlanmamış mum
    """
    return [
        candle
        for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(
    symbol,
    ticker,
    min_drop,
    max_drop,
):
    try:
        current_price = float(
            ticker.get("last", 0) or 0
        )

        if current_price <= 0:
            return None

        # 24 saatlik fiyat değişimi.
        open_24h = float(
            ticker.get("open24h", 0) or 0
        )

        if open_24h > 0:
            change_24h = (
                current_price / open_24h - 1
            ) * 100
        else:
            change_24h = 0.0

        # Son 100 günlük mumu al.
        # Tamamlanmamış günlük mumları ele.
        daily_candles = get_confirmed_candles(
            get_candles(symbol, "1D", 100)
        )

        # En yeni günlük mum önce gelecek şekilde sırala.
        daily_candles.sort(
            key=lambda candle: int(candle[0]),
            reverse=True,
        )

        # Son 90 tamamlanmış günlük mum.
        daily_candles = daily_candles[:LOOKBACK_DAYS]

        # Yeterli geçmiş verisi olmayan coinleri ele.
        if len(daily_candles) < LOOKBACK_DAYS:
            return None

        # OKX mum alanları:
        # [0] zaman, [2] yüksek, [3] düşük, [4] kapanış
        highest_90d = max(
            float(candle[2])
            for candle in daily_candles
        )

        lowest_90d = min(
            float(candle[3])
            for candle in daily_candles
        )

        if highest_90d <= 0:
            return None

        # 90 günlük zirveden yüzde kaç düşmüş?
        drop_pct = (
            1 - current_price / highest_90d
        ) * 100

        # İstenen düşüş aralığına uymuyorsa gösterme.
        if drop_pct < min_drop or drop_pct > max_drop:
            return None

        # 15 dakikalık yükseliş teyidi.
        candles_15m = get_confirmed_candles(
            get_candles(symbol, "15m", 6)
        )

        candles_15m.sort(
            key=lambda candle: int(candle[0])
        )

        if len(candles_15m) >= 2:
            previous_close = float(
                candles_15m[-2][4]
            )

            last_close = float(
                candles_15m[-1][4]
            )

            rising = last_close > previous_close

        else:
            previous_close = None
            last_close = None
            rising = False

        # Bunlar emir değil, tarama etiketleridir.
        signal = "BUY" if rising else "SELL"

        return {
            "Coin": symbol,
            "Sinyal": signal,
            "Güncel Fiyat": current_price,
            "90G Zirve": highest_90d,
            "90G Dip": lowest_90d,
            "Zirveden Düşüş (%)": round(drop_pct, 2),
            "24S Değişim (%)": round(change_24h, 2),
            "Önceki 15D Kapanış": previous_close,
            "Son 15D Kapanış": last_close,
            "15D Yükseliş": rising,
        }

    except (ValueError, TypeError, KeyError, IndexError):
        return None


# ==================================================
# AYARLAR
# ==================================================

st.sidebar.header("Tarama ayarları")

min_drop = st.sidebar.slider(
    "Minimum düşüş (%)",
    min_value=50.0,
    max_value=95.0,
    value=60.0,
    step=5.0,
)

max_drop = st.sidebar.slider(
    "Maksimum düşüş (%)",
    min_value=60.0,
    max_value=99.9,
    value=99.9,
    step=0.1,
)

max_coins = st.sidebar.slider(
    "Taranacak coin sayısı",
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
    st.sidebar.error(
        "Minimum düşüş maksimum düşüşten büyük olamaz."
    )

if st.sidebar.button("Önbelleği temizle"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Taramayı başlat", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info(
        "Ayarları seçip 'Taramayı başlat' düğmesine basın."
    )
    st.stop()

if min_drop > max_drop:
    st.error("Düşüş aralığını düzeltin.")
    st.stop()


# ==================================================
# API ERİŞİMİ
# ==================================================

try:
    with st.spinner(
        "OKX erişimi ve aktif pariteler kontrol ediliyor..."
    ):
        api_get("/api/v5/public/time")
        active_symbols = get_active_symbols()
        tickers = get_tickers()

except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"OKX erişilebilir. Aktif Spot USDT paritesi: "
    f"{len(active_symbols)}"
)


# ==================================================
# TARAMA LİSTESİNİ HAZIRLA
# ==================================================

universe = []

for symbol in active_symbols:
    ticker = tickers.get(symbol)

    if not ticker:
        continue

    try:
        price = float(
            ticker.get("last", 0) or 0
        )

        volume = float(
            ticker.get("volCcy24h", 0) or 0
        )

        if price > 0 and volume >= min_volume:
            universe.append(
                (symbol, ticker, volume)
            )

    except (ValueError, TypeError):
        continue

# Önce işlem hacmi yüksek coinleri tara.
universe.sort(
    key=lambda item: item[2],
    reverse=True,
)

universe = universe[:max_coins]

if not universe:
    st.warning(
        "Hacim kriterlerine uyan aktif USDT paritesi bulunamadı. "
        "Hacim eşiğini düşürmeyi deneyin."
    )
    st.stop()

st.write(
    f"**Taranacak aktif parite sayısı:** {len(universe)}"
)

st.info(
    "Tarama yaklaşık son 90 tamamlanmış günlük mumu kullanır. "
    "Son 3 ayda en az %60 düşüş koşulunu karşılayan coinler gösterilir. "
    "Delist edilmiş pariteler, OKX aktif enstrüman filtresine göre dışarıda bırakılır."
)


# ==================================================
# COINLERİ ANALİZ ET
# ==================================================

results = []
errors = 0
fatal_error = None

progress = st.progress(0)
status = st.empty()

with ThreadPoolExecutor(max_workers=2) as executor:

    futures = {
        executor.submit(
            analyze_coin,
            symbol,
            ticker,
            min_drop,
            max_drop,
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

        progress.progress(
            index / max(total, 1)
        )

        status.text(
            f"İşlenen: {index}/{total}"
        )

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.stop()

if not results:
    st.warning(
        "Belirlenen düşüş aralığına uyan coin bulunamadı. "
        "Daha fazla coin tarayabilir veya hacim eşiğini düşürebilirsiniz."
    )
    st.stop()


# ==================================================
# SONUÇLARI GÖSTER
# ==================================================

df = pd.DataFrame(results)

# En fazla düşen coin üstte olsun.
df = df.sort_values(
    "Zirveden Düşüş (%)",
    ascending=False,
).reset_index(drop=True)

buy_df = df[
    df["Sinyal"] == "BUY"
].copy()

sell_df = df[
    df["Sinyal"] == "SELL"
].copy()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Koşula uyan coin",
    len(df),
)

col2.metric(
    "BUY etiketi",
    len(buy_df),
)

col3.metric(
    "SELL etiketi",
    len(sell_df),
)

col4.metric(
    "Analiz edilemeyen",
    errors,
)

st.subheader(
    "Son 3 ayda en az %60 düşen aktif coinler"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

st.subheader(
    "BUY — Son iki 15 dakikalık kapanış yükselmiş"
)

if buy_df.empty:
    st.info(
        "Düşüş kriterini karşılayıp 15 dakikalık yükseliş "
        "teyidi alan coin bulunamadı."
    )

else:
    st.dataframe(
        buy_df,
        use_container_width=True,
        hide_index=True,
    )

with st.expander(
    "SELL etiketleri — yükseliş teyidi yok"
):
    st.dataframe(
        sell_df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# 15 DAKİKALIK GRAFİK
# ==================================================

st.subheader("15 dakikalık kapanış grafiği")

selected_coin = st.selectbox(
    "Grafiğini görmek istediğiniz coin",
    df["Coin"].tolist(),
)

try:
    candles = get_confirmed_candles(
        get_candles(
            selected_coin,
            "15m",
            100,
        )
    )

    candles.sort(
        key=lambda candle: int(candle[0])
    )

    chart_df = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime(
            [
                int(candle[0])
                for candle in candles
            ],
            unit="ms",
            utc=True,
        ),
        "Kapanış": [
            float(candle[4])
            for candle in candles
        ],
    })

    if not chart_df.empty:
        chart_df = chart_df.set_index(
            "Zaman (UTC)"
        )

        st.line_chart(
            chart_df["Kapanış"],
            use_container_width=True,
        )

    else:
        st.info(
            "Grafik için tamamlanmış mum verisi bulunamadı."
        )

except MarketAPIError as exc:
    st.warning(str(exc))


# ==================================================
# CSV İNDİR
# ==================================================

st.download_button(
    label="CSV indir",
    data=df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="okx_spot_3_month_60_percent_drop.csv",
    mime="text/csv",
)

st.caption(
    "Veri kaynağı OKX Spot'tur; Binance değildir. "
    "Son 90 tamamlanmış günlük mum yaklaşık 3 aylık dönemi temsil eder. "
    "Taranan coinler, seçilen hacim eşiğini de karşılamalıdır. "
    "OKX'te state='live' görünen pariteler kullanılır. "
    "BUY/SELL etiketleri yalnızca tarama sonucudur; "
    "yatırım tavsiyesi veya kâr garantisi değildir. "
    "Otomatik emir gönderilmez."
)
