import time
import threading
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://www.okx.com"
TIMEOUT = 20
MIN_REQUEST_INTERVAL = 0.45
LOOKBACK_DAYS = 365
DAY_MS = 24 * 60 * 60 * 1000

st.set_page_config(
    page_title="OKX 12-Month Low Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("📉 12 Aylık Dipteki ve Zirvesinden %50+ Düşmüş Coinler")
st.caption(
    "OKX Spot USDT | Son 365 günlük günlük mumlar | Aktif pariteler | "
    "Tüm uygun aktif pariteler taranır | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({"User-Agent": "OKX-365D-Low-Scanner/2.0"})
rate_lock = threading.Lock()
last_request_time = 0.0


class MarketAPIError(Exception):
    pass


class RateLimitError(MarketAPIError):
    pass


class RestrictedError(MarketAPIError):
    pass


def api_get(path, params=None, retries=5):
    global last_request_time

    for attempt in range(retries + 1):
        with rate_lock:
            wait = MIN_REQUEST_INTERVAL - (time.monotonic() - last_request_time)
            if wait > 0:
                time.sleep(wait)
            last_request_time = time.monotonic()

        try:
            response = session.get(BASE_URL + path, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            raise MarketAPIError(f"Bağlantı hatası: {exc}") from exc

        if response.status_code == 451:
            raise RestrictedError(
                "HTTP 451: OKX API bu bağlantı konumundan erişimi kısıtlıyor. "
                "Kısıtlama aşılmaya çalışılmadan tarama durduruldu."
            )

        if response.status_code == 429:
            if attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(min(max(delay, 2), 60))
                continue
            raise RateLimitError(
                "HTTP 429: OKX istek sınırı aşıldı. Tarama güvenli şekilde durduruldu. "
                "Bir süre sonra yeniden deneyin."
            )

        if not response.ok:
            raise MarketAPIError(f"HTTP {response.status_code}: {response.text[:250]}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketAPIError("OKX API geçerli JSON döndürmedi.") from exc

        if isinstance(payload, dict) and payload.get("code") not in (None, "0"):
            raise MarketAPIError(
                f"OKX API hatası {payload.get('code')}: {payload.get('msg', '')}"
            )
        return payload

    raise MarketAPIError("API isteği tamamlanamadı.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols():
    payload = api_get("/api/v5/public/instruments", {"instType": "SPOT"})
    return sorted({
        item["instId"]
        for item in payload.get("data", [])
        if item.get("state") == "live"
        and item.get("quoteCcy") == "USDT"
        and item.get("instId", "").endswith("-USDT")
    })


@st.cache_data(ttl=180, show_spinner=False)
def get_tickers():
    payload = api_get("/api/v5/market/tickers", {"instType": "SPOT"})
    return {
        item["instId"]: item
        for item in payload.get("data", [])
        if item.get("instId")
    }


def confirmed_candles(rows):
    # OKX candle schema: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    valid = []
    for row in rows:
        try:
            if len(row) > 8 and str(row[8]) == "1":
                valid.append(row)
        except (TypeError, IndexError):
            continue
    return valid


def get_candles_page(symbol, after=None, limit=300):
    params = {
        "instId": symbol,
        "bar": "1Dutc",
        "limit": str(limit),
    }
    if after is not None:
        params["after"] = str(after)
    payload = api_get("/api/v5/market/candles", params)
    return payload.get("data", [])


def get_last_365d_candles(symbol):
    """Fetch enough daily candles to cover the last 365 days, paging older if needed."""
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - LOOKBACK_DAYS * DAY_MS

    all_rows = get_candles_page(symbol, limit=300)
    if not all_rows:
        return []

    # Usually two requests cover 365 days (API returns max 300 per request).
    for _ in range(3):
        try:
            oldest_ts = min(int(row[0]) for row in all_rows)
        except (ValueError, TypeError, IndexError):
            break

        if oldest_ts <= cutoff_ms:
            break

        older_rows = get_candles_page(symbol, after=oldest_ts, limit=300)
        if not older_rows:
            break

        existing_timestamps = {
            int(row[0]) for row in all_rows
            if row and str(row[0]).isdigit()
        }
        new_rows = []
        for row in older_rows:
            try:
                ts = int(row[0])
                if ts < oldest_ts and ts not in existing_timestamps:
                    new_rows.append(row)
            except (ValueError, TypeError, IndexError):
                continue

        if not new_rows:
            break
        all_rows.extend(new_rows)

    candles = confirmed_candles(all_rows)
    unique = {}
    for row in candles:
        try:
            ts = int(row[0])
            if cutoff_ms <= ts <= now_ms:
                unique[ts] = row
        except (ValueError, TypeError, IndexError):
            continue

    return [unique[ts] for ts in sorted(unique)]


def analyze_coin(symbol, ticker, min_drop_pct, low_tolerance_pct):
    current_price = float((ticker or {}).get("last", 0) or 0)
    if current_price <= 0:
        return None, "no_price"

    candles = get_last_365d_candles(symbol)

    # If there isn't nearly a full year of daily history, don't mislabel a new listing as a 12M low.
    if len(candles) < 360:
        return None, "insufficient_history"

    try:
        high_365d = max(float(row[2]) for row in candles)
        low_365d = min(float(row[3]) for row in candles)
    except (ValueError, TypeError, IndexError):
        return None, "bad_candle_data"

    if high_365d <= 0 or low_365d <= 0:
        return None, "bad_candle_data"

    drop_from_high_pct = (1.0 - current_price / high_365d) * 100.0
    distance_from_low_pct = (current_price / low_365d - 1.0) * 100.0

    # Both conditions are mandatory:
    # 1) at least the specified % below 365d high;
    # 2) current price at or below the chosen distance above the 365d low.
    if drop_from_high_pct < min_drop_pct:
        return None, "not_down_enough"
    if distance_from_low_pct > low_tolerance_pct:
        return None, "not_near_low"

    # Optional 15m confirmation: do not discard an otherwise qualified coin if this extra request fails.
    previous_close = None
    last_close = None
    rising_15m = False
    try:
        payload = api_get(
            "/api/v5/market/candles",
            {"instId": symbol, "bar": "15m", "limit": "6"},
        )
        candles_15m = confirmed_candles(payload.get("data", []))
        candles_15m.sort(key=lambda row: int(row[0]))
        if len(candles_15m) >= 2:
            previous_close = float(candles_15m[-2][4])
            last_close = float(candles_15m[-1][4])
            rising_15m = last_close > previous_close
    except MarketAPIError:
        pass

    open_24h = float((ticker or {}).get("open24h", 0) or 0)
    volume_24h = float((ticker or {}).get("volCcy24h", 0) or 0)
    change_24h = (current_price / open_24h - 1) * 100 if open_24h > 0 else 0.0

    if distance_from_low_pct < 0:
        low_status = "YENİ 365G DİBİNİN ALTINDA"
    elif distance_from_low_pct <= 0.1:
        low_status = "365G DİBİNDE"
    else:
        low_status = "365G DİBİNE YAKIN"

    return {
        "Coin": symbol,
        "Dip Durumu": low_status,
        "Güncel Fiyat": current_price,
        "365G En Yüksek": high_365d,
        "365G En Düşük": low_365d,
        "Zirveden Düşüş (%)": round(drop_from_high_pct, 2),
        "Dibe Uzaklık (%)": round(distance_from_low_pct, 3),
        "24S Değişim (%)": round(change_24h, 2),
        "24S Hacim (USDT)": round(volume_24h, 2),
        "15D Önceki Kapanış": previous_close,
        "15D Son Kapanış": last_close,
        "15D Yükseliş Teyidi": rising_15m,
        "Kullanılan Günlük Mum": len(candles),
    }, "match"


st.sidebar.header("Filtreler")
min_drop_pct = st.sidebar.slider(
    "365 günlük zirveden minimum düşüş (%)",
    min_value=50.0,
    max_value=95.0,
    value=50.0,
    step=5.0,
)
low_tolerance_pct = st.sidebar.slider(
    "365 günlük en düşük fiyattan maksimum uzaklık (%)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="0% = fiyat geçmiş dip seviyesinde veya altında olmalı. Örneğin 1% = en fazla %1 üzerinde olabilir.",
)

st.sidebar.info(
    "Coin sayısı ve hacim filtresi yoktur. Aktif Spot USDT pariteleri sırayla taranır. "
    "Tüm coinler taranırken birkaç dakika sürebilir."
)

if st.sidebar.button("Önbelleği temizle"):
    get_active_symbols.clear()
    get_tickers.clear()
    st.rerun()

if st.button("Tüm aktif coinleri tara", type="primary"):
    st.session_state["run_scan"] = True

if not st.session_state.get("run_scan", False):
    st.info("Tarama başlatmak için düğmeye basın.")
    st.stop()

try:
    with st.spinner("Aktif pariteler ve güncel fiyatlar alınıyor..."):
        api_get("/api/v5/public/time")
        symbols = get_active_symbols()
        tickers = get_tickers()
except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"**OKX'te aktif görünen Spot USDT pariteleri:** {len(symbols)}")
st.caption(
    "Bir coin ancak iki ölçütü de karşılarsa listelenir: son 365 günlük zirveden "
    f"en az %{min_drop_pct:.0f} düşüş ve son 365 günlük dibe en fazla %{low_tolerance_pct:.1f} uzaklık."
)

results = []
error_count = 0
no_ticker_count = 0
short_history_count = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()
match_status = st.empty()

for idx, symbol in enumerate(symbols, start=1):
    ticker = tickers.get(symbol)
    if not ticker:
        no_ticker_count += 1
        progress.progress(idx / max(len(symbols), 1))
        status.text(f"{idx}/{len(symbols)} — {symbol}: ticker yok")
        continue

    try:
        result, reason = analyze_coin(
            symbol,
            ticker,
            min_drop_pct,
            low_tolerance_pct,
        )
        if result is not None:
            results.append(result)
        elif reason == "insufficient_history":
            short_history_count += 1

    except RestrictedError as exc:
        fatal_error = str(exc)
        break
    except RateLimitError as exc:
        fatal_error = str(exc)
        break
    except MarketAPIError:
        error_count += 1
    except (ValueError, TypeError, KeyError, IndexError):
        error_count += 1
    except Exception:
        error_count += 1

    progress.progress(idx / max(len(symbols), 1))
    status.text(f"Taranıyor: {idx}/{len(symbols)} — {symbol}")
    match_status.caption(f"Şu ana kadarki sonuç: {len(results)} coin")

progress.empty()
status.empty()
match_status.empty()

if fatal_error:
    st.error(fatal_error)
    st.warning(
        f"Tarama {idx}/{len(symbols)} paritede durdu. Bulunan sonuçlar aşağıda gösterilebilir."
    )

if not results:
    st.warning(
        "Filtreleri karşılayan coin bulunamadı veya yeterli veri alınamadı. "
        "Dip toleransını artırabilir ya da API erişimini kontrol edebilirsiniz."
    )
    st.caption(
        f"Aktif parite: {len(symbols)} | Ticker olmayan: {no_ticker_count} | "
        f"12 aylık veri yetersiz: {short_history_count} | Hatalar: {error_count}"
    )
    st.stop()

results_df = pd.DataFrame(results).sort_values(
    "Dibe Uzaklık (%)", ascending=True
).reset_index(drop=True)

new_lows = results_df[results_df["Dibe Uzaklık (%)"] < 0].copy()
rising_15m = results_df[results_df["15D Yükseliş Teyidi"]].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktif parite", len(symbols))
c2.metric("İki koşula uyan", len(results_df))
c3.metric("365G dip altında", len(new_lows))
c4.metric("15D yükseliş teyidi", len(rising_15m))

st.caption(
    f"Ticker olmayan: {no_ticker_count} | 365 günlük geçmişi yetersiz: "
    f"{short_history_count} | Veri/API hatası: {error_count}"
)

st.subheader("İki koşulu da karşılayan coinler")
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.subheader("Önceki 365 günlük dip fiyatının altına inenler")
if new_lows.empty:
    st.info("Geçmiş 365 günlük dip seviyesinin altında coin bulunamadı.")
else:
    st.dataframe(new_lows, use_container_width=True, hide_index=True)

st.subheader("Son iki tamamlanmış 15 dakikalık kapanışı yükselenler")
if rising_15m.empty:
    st.info("15 dakikalık yükseliş teyidi olan coin bulunamadı.")
else:
    st.dataframe(rising_15m, use_container_width=True, hide_index=True)

st.subheader("15 dakikalık fiyat grafiği")
selected = st.selectbox("Coin seçin", results_df["Coin"].tolist())
try:
    payload = api_get(
        "/api/v5/market/candles",
        {"instId": selected, "bar": "15m", "limit": "100"},
    )
    candles = confirmed_candles(payload.get("data", []))
    candles.sort(key=lambda row: int(row[0]))
    chart = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime([int(row[0]) for row in candles], unit="ms", utc=True),
        "Kapanış": [float(row[4]) for row in candles],
    })
    if not chart.empty:
        st.line_chart(chart.set_index("Zaman (UTC)")["Kapanış"], use_container_width=True)
except MarketAPIError as exc:
    st.warning(f"Grafik alınamadı: {exc}")

st.download_button(
    "CSV indir",
    data=results_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_all_active_365d_lows_down_50pct.csv",
    mime="text/csv",
)

st.caption(
    "Yalnızca OKX'te state='live' olan Spot USDT pariteleri taranır. "
    "Her coin için son 365 günün tamamlanmış UTC günlük mumları kullanılır; "
    "365 güne yakın geçmişi olmayan yeni listelenmiş coinler dışlanır. "
    "Canlı fiyat günlük mumun son kapanışından farklı olabilir. Aktif görünmek, "
    "bir coinin geçmişte hiç delist edilmediğini garanti etmez. BUY/SELL emri verilmez."
)
