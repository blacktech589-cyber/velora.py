import time
import threading
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://www.okx.com"
TIMEOUT = 20
MIN_REQUEST_INTERVAL = 0.45
LOOKBACK_DAYS = 30
DAY_MS = 24 * 60 * 60 * 1000
MIN_HISTORY_CANDLES = 20

st.set_page_config(
    page_title="OKX 30-Day Low Scanner",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Son 30 Günün Dibine En Yakın Coinler")
st.caption(
    "OKX Spot USDT | Son 30 günlük tamamlanmış günlük mumlar | "
    "Yalnızca aktif pariteler | Otomatik emir yok"
)

session = requests.Session()
session.headers.update({"User-Agent": "OKX-30D-Low-Scanner/3.0"})
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
                "Tarama durduruldu; bölgesel kısıtlamayı aşmaya çalışılmadı."
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
                "HTTP 429: OKX istek sınırı aşıldı. Bir süre sonra yeniden deneyin."
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


@st.cache_data(ttl=900, show_spinner=False)
def get_active_symbols():
    payload = api_get("/api/v5/public/instruments", {"instType": "SPOT"})
    # OKX instrument listesinde state='live' olan aktif Spot USDT çiftlerini al.
    return sorted({
        item["instId"]
        for item in payload.get("data", [])
        if item.get("state") == "live"
        and item.get("quoteCcy") == "USDT"
        and item.get("instId", "").endswith("-USDT")
    })


@st.cache_data(ttl=60, show_spinner=False)
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


def get_last_30d_candles(symbol):
    """Fetch confirmed daily candles covering the most recent 30 days."""
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - LOOKBACK_DAYS * DAY_MS

    payload = api_get(
        "/api/v5/market/candles",
        {"instId": symbol, "bar": "1Dutc", "limit": "100"},
    )
    rows = confirmed_candles(payload.get("data", []))

    unique = {}
    for row in rows:
        try:
            ts = int(row[0])
            if cutoff_ms <= ts <= now_ms:
                unique[ts] = row
        except (ValueError, TypeError, IndexError):
            continue

    return [unique[ts] for ts in sorted(unique)]


def analyze_coin(symbol, ticker, low_tolerance_pct):
    try:
        current_price = float((ticker or {}).get("last", 0) or 0)
        open_24h = float((ticker or {}).get("open24h", 0) or 0)
        volume_24h = float((ticker or {}).get("volCcy24h", 0) or 0)
    except (ValueError, TypeError):
        return None, "bad_ticker"

    if current_price <= 0:
        return None, "no_price"

    candles = get_last_30d_candles(symbol)
    if len(candles) < MIN_HISTORY_CANDLES:
        return None, "insufficient_history"

    try:
        high_30d = max(float(row[2]) for row in candles)
        low_30d = min(float(row[3]) for row in candles)
    except (ValueError, TypeError, IndexError):
        return None, "bad_candle_data"

    if high_30d <= 0 or low_30d <= 0:
        return None, "bad_candle_data"

    drop_from_high_pct = (1.0 - current_price / high_30d) * 100.0
    distance_from_low_pct = (current_price / low_30d - 1.0) * 100.0
    change_24h = (current_price / open_24h - 1.0) * 100.0 if open_24h > 0 else 0.0

    if distance_from_low_pct < 0:
        low_status = "YENİ 30G DİBİNİN ALTINDA"
    elif distance_from_low_pct <= 0.1:
        low_status = "30G DİBİNDE"
    elif distance_from_low_pct <= low_tolerance_pct:
        low_status = "30G DİBİNE YAKIN"
    else:
        low_status = "30G DİBİNDEN UZAK"

    return {
        "Coin": symbol,
        "Dip Durumu": low_status,
        "Güncel Fiyat": current_price,
        "30G En Yüksek": high_30d,
        "30G En Düşük": low_30d,
        "Zirveden Düşüş (%)": round(drop_from_high_pct, 2),
        "Dibe Uzaklık (%)": round(distance_from_low_pct, 3),
        "24S Değişim (%)": round(change_24h, 2),
        "24S Hacim (USDT)": round(volume_24h, 2),
        "Tamamlanmış Günlük Mum": len(candles),
        "Dip Toleransında": distance_from_low_pct <= low_tolerance_pct,
    }, "ok"


st.sidebar.header("Filtreler")
low_tolerance_pct = st.sidebar.slider(
    "30 günlük dibin üstünde izin verilen uzaklık (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.25,
    help="0% = fiyat 30 günlük dipte veya altında olmalı. 2% = dip fiyatının en fazla %2 üzerinde olabilir.",
)

st.sidebar.info(
    "Tüm aktif Spot USDT pariteleri taranır. Yeni listelenen ve yeterli 30 günlük "
    "geçmişi olmayan pariteler dışarıda bırakılır. Tarama birkaç dakika sürebilir."
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
    with st.spinner("Aktif Spot pariteler ve güncel fiyatlar alınıyor..."):
        api_get("/api/v5/public/time")
        symbols = get_active_symbols()
        tickers = get_tickers()
except MarketAPIError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"**Aktif OKX Spot USDT paritesi:** {len(symbols)}")
st.caption(
    "Son 30 gündeki tamamlanmış günlük mumların en düşük fiyatı hesaplanır. "
    "Sonuçlar güncel fiyatın bu dip seviyesine uzaklığına göre sıralanır. "
    "Sadece OKX'te şu anda `live` görünen pariteler taranır."
)

results = []
error_count = 0
no_ticker_count = 0
short_history_count = 0
fatal_error = None
progress = st.progress(0)
status = st.empty()

for idx, symbol in enumerate(symbols, start=1):
    ticker = tickers.get(symbol)
    if not ticker:
        no_ticker_count += 1
        progress.progress(idx / max(len(symbols), 1))
        status.text(f"{idx}/{len(symbols)} — {symbol}: ticker yok")
        continue

    try:
        result, reason = analyze_coin(symbol, ticker, low_tolerance_pct)
        if result is not None:
            results.append(result)
        elif reason == "insufficient_history":
            short_history_count += 1
        else:
            error_count += 1
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
    status.text(f"Taranıyor: {idx}/{len(symbols)} — {symbol} | Bulunan: {len(results)}")

progress.empty()
status.empty()

if fatal_error:
    st.error(fatal_error)
    st.warning(f"Tarama {idx}/{len(symbols)} paritede durdu. Bulunan sonuçlar aşağıda gösterilebilir.")

st.caption(
    f"Aktif parite: {len(symbols)} | Ticker olmayan: {no_ticker_count} | "
    f"30 günlük geçmişi yetersiz: {short_history_count} | Veri/API hatası: {error_count}"
)

if not results:
    st.warning(
        "Sonuç bulunamadı. API erişimini, aktif pariteleri veya yeterli geçmiş veriyi kontrol edin."
    )
    st.stop()

results_df = pd.DataFrame(results).sort_values(
    ["Dibe Uzaklık (%)", "Zirveden Düşüş (%)"],
    ascending=[True, False],
).reset_index(drop=True)

near_low_df = results_df[results_df["Dip Toleransında"]].copy()
new_lows = results_df[results_df["Dibe Uzaklık (%)"] < 0].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktif parite", len(symbols))
c2.metric("Yeterli geçmişi olan", len(results_df))
c3.metric("Seçilen dip toleransında", len(near_low_df))
c4.metric("Yeni 30G dibi yapan", len(new_lows))

st.subheader("30 günlük kendi dibine en yakın coinler")
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.subheader(f"30 günlük dibin en fazla %{low_tolerance_pct:g} üzerinde olanlar")
if near_low_df.empty:
    st.info("Seçilen toleransta coin bulunamadı. Toleransı artırabilir veya tüm sonuçları inceleyebilirsiniz.")
else:
    st.dataframe(near_low_df, use_container_width=True, hide_index=True)

st.subheader("Son 30 günlük dip seviyesinin altındaki coinler")
if new_lows.empty:
    st.info("Güncel fiyatı hesaplanan 30 günlük en düşük seviyenin altında olan coin yok.")
else:
    st.dataframe(new_lows, use_container_width=True, hide_index=True)

st.subheader("30 günlük fiyat grafiği")
selected = st.selectbox("Coin seçin", results_df["Coin"].tolist())
try:
    chart_rows = get_last_30d_candles(selected)
    chart = pd.DataFrame({
        "Zaman (UTC)": pd.to_datetime([int(row[0]) for row in chart_rows], unit="ms", utc=True),
        "Kapanış": [float(row[4]) for row in chart_rows],
    })
    if not chart.empty:
        st.line_chart(chart.set_index("Zaman (UTC)")["Kapanış"], use_container_width=True)
except MarketAPIError as exc:
    st.warning(f"Grafik alınamadı: {exc}")

st.download_button(
    "CSV indir",
    data=results_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="okx_active_spot_usdt_30d_lows.csv",
    mime="text/csv",
)

st.caption(
    "Yalnızca OKX'te tarama anında `state='live'` olan Spot USDT pariteleri dahil edilir. "
    "Bu filtre şu anda aktif olmayan/delist edilmiş pariteleri dışarıda bırakır; "
    "geçmişte hiç delist edilip yeniden listelenmediğini garanti etmez. "
    "Hesaplamada son 30 günlük tamamlanmış UTC günlük mumlar kullanılır. "
    "Bu araç otomatik BUY/SELL emri vermez; dipte olmak fiyatın yükselmesini garanti etmez."
)
