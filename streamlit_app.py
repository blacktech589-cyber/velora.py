import time
import threading
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://www.okx.com"
TIMEOUT = 20
MIN_REQUEST_INTERVAL = 0.45
LOOKBACK_WEEKS = 52

st.set_page_config(
    page_title="All Active OKX Coins - 12M Lows",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Tüm Aktif Spot USDT Coinleri: 12 Aylık Dip + %50 Düşüş")
st.caption(
    "Tüm aktif OKX Spot USDT pariteleri | "
    "12 aylık zirveden en az %50 düşüş | "
    "12 aylık dip yakınlığı | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({
    "User-Agent": "OKX-All-Active-Spot-Scanner/1.2"
})

rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    pass


def api_get(path, params=None, retries=5):
    global last_request_time

    for attempt in range(retries + 1):
        with rate_lock:
            delay = MIN_REQUEST_INTERVAL - (
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
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            raise MarketAPIError(
                f"Bağlantı hatası: {exc}"
            ) from exc

        if response.status_code == 451:
            raise MarketAPIError(
                "HTTP 451: OKX API erişimi bu bağlantı konumundan "
                "kısıtlanmış. Kısıtlama aşılmaya çalışılmadan durduruldu."
            )

        if response.status_code == 429:
            if attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = (
                        float(retry_after)
                        if retry_after
                        else 2 ** (attempt + 1)
                    )
                except ValueError:
                    wait_seconds = 2 ** (attempt + 1)

                time.sleep(min(max(wait_seconds, 2), 60))
                continue

            raise MarketAPIError(
                "HTTP 429: OKX istek sınırı tekrarlandı. "
                "Bir süre bekleyip yeniden deneyin."
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
    """OKX'in state=live olarak işaretlediği tüm Spot USDT pariteleri."""
    payload = api_get(
        "/api/v5/public/instruments",
        {"instType": "SPOT"},
    )

    return sorted({
        item["instId"]
        for item in payload.get("data", [])
        if item.get("state") == "live"
        and item.get("quoteCcy") == "USDT"
        and item.get("instId", "").endswith("-USDT")
    })


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


def confirmed_candles(candles):
    # OKX mum alanı [8]: "1" tamamlanmış mumu gösterir.
    return [
        candle
        for candle in candles
        if len(candle) > 8 and str(candle[8]) == "1"
    ]


def analyze_coin(
    symbol,
    ticker,
    minimum_drop_pct,
    max_low_distance_pct,
):
    """İki filtreyi de karşılayan coin için sonuç döndürür."""
    try:
        current_price = float(
            (ticker or {}).get("last", 0) or 0
        )
    except (ValueError, TypeError):
        return None

    if current_price <= 0:
        return None

    # Yaklaşık son 12 ay için 52 tamamlanmış haftalık mum.
    weekly = confirmed_candles(
        get_candles(symbol, "1W", 60)
    )
    weekly.sort(
        key=lambda candle: int(candle[0]),
        reverse=True,
    )
    weekly = weekly[:LOOKBACK_WEEKS]

    if len(weekly) < LOOKBACK_WEEKS:
        return None

    try:
        highest_12m = max(
            float(candle[2]) for candle in weekly
        )
        lowest_12m = min(
            float(candle[3]) for candle in weekly
        )
    except (ValueError, TypeError, IndexError):
        return None

    if highest_12m <= 0 or lowest_12m <= 0:
        return None

    # Koşul 1: zirvesinden en az %50 düşmüş olmalı.
    drop_from_high_pct = (
        1 - current_price / highest_12m
    ) * 100

    if drop_from_high_pct < minimum_drop_pct:
        return None

    # Koşul 2: kendi 12 aylık dip seviyesinde veya yakınında olmalı.
    distance_from_low_pct = (
        current_price / lowest_12m - 1
    ) * 100

    if distance_from_low_pct > max_low_distance_pct:
        return None

    # Ek bilgi: son iki tamamlanmış 15 dakikalık mum yükseliyor mu?
    candles_15m = confirmed_candles(
        get_candles(symbol, "15m", 6)
    )
    candles_15m.sort(
        key=lambda candle: int(candle[0])
    )

    if len(candles_15m) >= 2:
        previous_close = float(candles_15m[-2][4])
        last_close = float(candles_15m[-1][4])
        rising_15m = last_close > previous_close
    else:
        previous_close = None
        last_close = None
        rising_15m = False

    try:
        open_24h = float(
            (ticker or {}).get("open24h", 0) or 0
        )
        volume_24h = float(
            (ticker or {}).get("volCcy24h", 0) or 0
        )
    except (ValueError, TypeError):
        open_24h = 0.0
        volume_24h = 0.0

    change_24h = (
        (current_price / open_24h - 1) * 100
        if open_24h > 0
        else 0.0
    )

    if distance_from_low_pct < 0:
        low_status = "12A DİBİNİN ALTINDA"
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
        "Zirveden Düşüş (%)": round(
            drop_from_high_pct, 2
        ),
        "Dibe Uzaklık (%)": round(
            distance_from_low_pct, 3
        ),
        "24S Değişim (%)": round(change_24h, 2),
        "24S Hacim (USDT)": round(volume_24h, 2),
        "Önceki 15D Kapanış": previous_close,
        "Son 15D Kapanış": last_close,
        "15D Yükseliş Teyidi": rising_15m,
    }


# ==================================================
# TARAMA AYARLARI
# ==================================================

st.sidebar.header("Tarama ayarları")

minimum_drop_pct = st.sidebar.slider(
    "12 aylık zirveden minimum düşüş (%)",
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

st.sidebar.info(
    "Coin sayısı sınırı ve hacim filtresi yoktur. "
    "Bütün aktif Spot USDT pariteleri sırayla taranır."
)

if st.sidebar.button("Önbelleği temizle"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Tüm coinleri tara", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info(
        "Başlamak için 'Tüm coinleri tara' düğmesine basın."
    )
    st.stop()


# ==================================================
# AKTİF COIN LİSTESİNİ AL
# ==================================================

try:
    with st.spinner(
        "OKX erişimi ve aktif parite listesi kontrol ediliyor..."
    ):
        api_get("/api/v5/public/time")
        active_symbols = get_active_symbols()
        tickers = get_tickers()

except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"OKX'te aktif görünen Spot USDT paritesi: "
    f"{len(active_symbols)}. Hacim veya coin sayısı sınırı yok."
)

missing_tickers = sum(
    1 for symbol in active_symbols
    if symbol not in tickers
)

if missing_tickers:
    st.warning(
        f"{missing_tickers} parite için ticker verisi bulunamadı. "
        "Güncel fiyat olmadan bu pariteler analiz edilemez."
    )

st.info(
    "Koşullar: son 12 aylık zirveden en az seçilen yüzde kadar düşüş "
    "VE 12 aylık dip fiyatına seçilen tolerans içinde olma. "
    "Hacim nedeniyle coin elenmez."
)


# ==================================================
# TÜM AKTİF PARİTELERİ SIRAYLA TARA
# ==================================================

results = []
errors = 0
no_ticker = 0
fatal_error = None

progress = st.progress(0)
status_text = st.empty()
match_count = st.empty()

total = len(active_symbols)

for index, symbol in enumerate(active_symbols, start=1):
    ticker = tickers.get(symbol)

    if not ticker:
        no_ticker += 1
        progress.progress(index / max(total, 1))
        status_text.text(
            f"Taranıyor: {index}/{total} — {symbol} (ticker yok)"
        )
        continue

    try:
        result = analyze_coin(
            symbol,
            ticker,
            minimum_drop_pct,
            max_low_distance_pct,
        )

        if result is not None:
            results.append(result)

    except MarketAPIError as exc:
        fatal_error = str(exc)
        break

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ):
        errors += 1

    except Exception:
        # Tek coinle ilgili beklenmeyen hata tüm taramayı durdurmasın.
        errors += 1

    progress.progress(index / max(total, 1))
    status_text.text(
        f"Taranıyor: {index}/{total} — {symbol}"
    )
    match_count.caption(
        f"Şu ana kadar koşullara uyan coin: {len(results)}"
    )

progress.empty()
status_text.empty()

if fatal_error:
    st.error(fatal_error)
    st.warning(
        f"Tarama {index}/{total} paritede durdu. "
        f"Bulunan sonuç: {len(results)}. Daha sonra tekrar deneyin."
    )
    st.stop()

if not results:
    st.warning(
        "Taranan aktif pariteler içinde iki koşulu birden sağlayan "
        "coin bulunamadı. Dip toleransını artırmayı deneyebilirsiniz."
    )
    st.caption(
        f"İncelenen: {total} | Ticker verisi olmayan: {no_ticker} | "
        f"Analiz hatası: {errors}"
    )
    st.stop()


# ==================================================
# SONUÇLARI GÖSTER
# ==================================================

df = pd.DataFrame(results)

df = df.sort_values(
    "Dibe Uzaklık (%)",
    ascending=True,
).reset_index(drop=True)

new_lows = df[
    df["Dibe Uzaklık (%)"] < 0
].copy()

rising_coins = df[
    df["15D Yükseliş Teyidi"]
].copy()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Taranan aktif parite", total)
col2.metric("İki koşula uyan coin", len(df))
col3.metric("12A dibinin altında", len(new_lows))
col4.metric("15D yükseliş teyidi", len(rising_coins))

st.caption(
    f"Ticker verisi olmayan: {no_ticker} | "
    f"Analiz hatası: {errors}"
)

st.subheader(
    "12 aylık dipte ve zirvesinden en az %50 düşmüş coinler"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Geçmiş 12 aylık dip seviyesinin altındakiler")

if new_lows.empty:
    st.info(
        "Geçmiş 12 aylık dip seviyesinin altında coin bulunamadı."
    )
else:
    st.dataframe(
        new_lows,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("15 dakikalık kapanışı yükselen coinler")

if rising_coins.empty:
    st.info(
        "15 dakikalık yükseliş teyidi bulunan coin yok."
    )
else:
    st.dataframe(
        rising_coins,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# 15 DAKİKALIK GRAFİK
# ==================================================

st.subheader("15 dakikalık kapanış grafiği")

selected_coin = st.selectbox(
    "Coin seçin",
    df["Coin"].tolist(),
)

try:
    candles = confirmed_candles(
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
            float(candle[4]) for candle in candles
        ],
    })

    if not chart_df.empty:
        chart_df = chart_df.set_index("Zaman (UTC)")
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
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_all_active_12m_lows_down_50_percent.csv",
    mime="text/csv",
)

st.caption(
    "Veri kaynağı OKX Spot'tur. 52 tamamlanmış haftalık mum yaklaşık "
    "12 aylık dönemi temsil eder; kayan 365 günle birebir aynı değildir. "
    "Tüm aktif Spot USDT pariteleri sırayla taranır; API sınırları nedeniyle "
    "işlem zaman alabilir. OKX'te aktif görünen bir coin geçmişte delist "
    "edilip yeniden listelenmiş olabilir. Geçmiş dip fiyatına ulaşmak, "
    "daha fazla düşmeyeceği veya yükseleceği anlamına gelmez. "
    "Otomatik emir gönderilmez."
)
