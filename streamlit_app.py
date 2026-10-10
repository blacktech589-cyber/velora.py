```python
import time
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# BINANCE SPOT DIP SCANNER
# Public market data only — NO order execution
# =========================================================

BASE_URL = "https://api.binance.com"
TIMEOUT = 12

st.set_page_config(
    page_title="Binance Spot Dip Scanner",
    page_icon="📉",
    layout="wide"
)

st.title("Binance Spot Dip Scanner")
st.caption(
    "30-day low detection | Completed 15-minute candles | "
    "BUY / SELL signals only | No order execution"
)

session = requests.Session()
session.headers.update({"User-Agent": "Spot-Dip-Scanner/1.0"})


class BinanceAPIError(Exception):
    pass


def api_get(path, params=None):
    """Call Binance public REST API without bypassing restrictions."""
    url = BASE_URL + path

    try:
        response = session.get(
            url,
            params=params,
            timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise BinanceAPIError(
            f"Binance bağlantı hatası: {exc}"
        ) from exc

    if response.status_code == 451:
        raise BinanceAPIError(
            "HTTP 451: Binance bu bağlantı konumundan erişimi "
            "kısıtlıyor. Kısıtlamayı aşma girişimi yapılmadı."
        )

    if response.status_code == 429:
        raise BinanceAPIError(
            "HTTP 429: Binance istek sınırına ulaşıldı. "
            "Bir süre bekleyip yeniden deneyin."
        )

    if response.status_code in (418,):
        raise BinanceAPIError(
            "Binance IP geçici olarak engellenmiş olabilir (HTTP 418)."
        )

    if not response.ok:
        raise BinanceAPIError(
            f"Binance HTTP {response.status_code}: "
            f"{response.text[:250]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BinanceAPIError(
            "Binance geçerli JSON yanıtı döndürmedi."
        ) from exc


@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_symbols():
    """Get active Binance Spot USDT trading pairs."""
    data = api_get("/api/v3/exchangeInfo")

    symbols = []
    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", True)
            and item.get("symbol", "").endswith("USDT")
        ):
            symbols.append(item["symbol"])

    return symbols


@st.cache_data(ttl=300, show_spinner=False)
def get_tickers():
    """Retrieve current 24-hour ticker statistics."""
    data = api_get("/api/v3/ticker/24hr")
    return {
        item["symbol"]: item
        for item in data
        if "symbol" in item
    }


def get_klines(symbol, interval, limit):
    return api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )


def analyze_symbol(symbol, ticker, proximity_pct):
    """
    Dip:
      Current price is within proximity_pct of the lowest daily
      low over the last 30 completed daily candles.

    Confirmation:
      The last two completed 15m candles have rising closes.

    BUY when both conditions hold; otherwise SELL.
    This is a simple screening rule, not a prediction guarantee.
    """
    try:
        price = float(ticker["lastPrice"])
        change_24h = float(ticker["priceChangePercent"])
        quote_volume = float(ticker["quoteVolume"])

        # 31 candles allow exclusion of the current incomplete day.
        daily = get_klines(symbol, "1d", 31)
        if len(daily) < 30:
            return None

        # Exclude the latest potentially incomplete daily candle.
        completed_daily = daily[:-1][-30:]
        lows = [float(candle[3]) for candle in completed_daily]
        low_30d = min(lows)

        if low_30d <= 0:
            return None

        distance_pct = ((price / low_30d) - 1) * 100
        near_low = distance_pct <= proximity_pct

        # Binance returns the latest candle too; exclude it because
        # the current 15m candle may still be forming.
        candles_15m = get_klines(symbol, "15m", 5)
        if len(candles_15m) < 4:
            return None

        completed_15m = candles_15m[:-1]
        last_two = completed_15m[-2:]

        previous_close = float(last_two[0][4])
        last_close = float(last_two[1][4])

        rising_confirmation = last_close > previous_close

        # No HOLD category: every successfully analyzed coin is BUY or SELL.
        signal = (
            "BUY"
            if near_low and rising_confirmation
            else "SELL"
        )

        reasons = []
        if near_low:
            reasons.append("30 günlük dibe yakın")
        else:
            reasons.append("Dip bölgesinin dışında")

        if rising_confirmation:
            reasons.append("15 dk kapanışları yükseliyor")
        else:
            reasons.append("15 dk yükseliş teyidi yok")

        return {
            "Symbol": symbol,
            "Signal": signal,
            "Price": price,
            "30D Low": low_30d,
            "Distance from 30D Low (%)": round(distance_pct, 2),
            "24h Change (%)": round(change_24h, 2),
            "24h Volume (USDT)": quote_volume,
            "15m Previous Close": previous_close,
            "15m Last Completed Close": last_close,
            "Near 30D Low": "Yes" if near_low else "No",
            "15m Rising": "Yes" if rising_confirmation else "No",
            "Reason": " | ".join(reasons)
        }

    except BinanceAPIError:
        raise
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def fmt_price(value):
    if pd.isna(value):
        return "-"
    return f"{value:.10g}"


# ---------------- Sidebar settings ----------------

st.sidebar.header("Scanner Settings")

proximity_pct = st.sidebar.slider(
    "30 günlük dipten maksimum uzaklık (%)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.5
)

max_coins = st.sidebar.slider(
    "Taranacak coin sayısı",
    min_value=10,
    max_value=150,
    value=50,
    step=10
)

min_volume = st.sidebar.number_input(
    "Minimum 24 saatlik hacim (USDT)",
    min_value=0,
    value=100000,
    step=100000
)

auto_refresh = st.sidebar.checkbox(
    "Her 5 dakikada bir yenile",
    value=False
)

if st.sidebar.button("Önbelleği temizle ve yenile"):
    get_exchange_symbols.clear()
    get_tickers.clear()
    st.rerun()


# ---------------- API access test ----------------

try:
    with st.spinner("Binance Spot API erişimi test ediliyor..."):
        # A lightweight public endpoint to test connectivity.
        server_time = api_get("/api/v3/time")
        symbols = get_exchange_symbols()
        tickers = get_tickers()

except BinanceAPIError as exc:
    st.error(str(exc))
    st.info(
        "API erişimi sağlanamadığı için tarama durduruldu. "
        "Bölgesel erişim kısıtlamaları aşılmaya çalışılmaz."
    )
    st.stop()

st.success(
    f"Binance Spot API erişilebilir. "
    f"Spot pariteleri: {len(symbols):,}"
)

# ---------------- Build scan universe ----------------

ticker_rows = []

for symbol in symbols:
    ticker = tickers.get(symbol)
    if not ticker:
        continue

    try:
        volume = float(ticker.get("quoteVolume", 0))
        price = float(ticker.get("lastPrice", 0))

        if volume >= min_volume and price > 0:
            ticker_rows.append({
                "symbol": symbol,
                "ticker": ticker,
                "volume": volume,
                "change": float(ticker.get("priceChangePercent", 0))
            })
    except (TypeError, ValueError):
        continue

# Scan liquid pairs first.
ticker_rows.sort(key=lambda row: row["volume"], reverse=True)
universe = ticker_rows[:max_coins]

st.write(
    f"**Taranacak coin sayısı:** {len(universe)} "
    f"| **Dip eşiği:** %{proximity_pct:.1f} "
    f"| **Zaman dilimi:** 15 dakika"
)

if not universe:
    st.warning(
        "Filtrelere uyan coin bulunamadı. Minimum hacim değerini azaltın."
    )
    st.stop()

# ---------------- Scan ----------------

results = []
errors = 0
progress = st.progress(0)
status = st.empty()

# Small worker pool helps limit concurrent API requests.
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(
            analyze_symbol,
            row["symbol"],
            row["ticker"],
            proximity_pct
        ): row["symbol"]
        for row in universe
    }

    total = len(futures)

    for index, future in enumerate(as_completed(futures), start=1):
        symbol = futures[future]
        try:
            result = future.result()
            if result:
                results.append(result)
            else:
                errors += 1
        except BinanceAPIError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            errors += 1

        progress.progress(index / total)
        status.text(f"Taranıyor: {index}/{total} — {symbol}")

progress.empty()
status.empty()

if not results:
    st.warning(
        "Analiz edilebilir sonuç alınamadı. Daha sonra tekrar deneyin."
    )
    st.stop()

df = pd.DataFrame(results)
df = df.sort_values(
    by=["Signal", "Distance from 30D Low (%)"],
    ascending=[True, True]
).reset_index(drop=True)

buy_df = df[df["Signal"] == "BUY"].copy()
sell_df = df[df["Signal"] == "SELL"].copy()

# ---------------- Summary ----------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Taranan", len(df))
col2.metric("BUY sinyali", len(buy_df))
col3.metric("SELL sinyali", len(sell_df))
col4.metric("Analiz edilemeyen", errors)

st.divider()

# ---------------- BUY opportunities ----------------

st.subheader("BUY — Dip ve yükseliş teyidi")
st.caption(
    "BUY yalnızca fiyat 30 günlük dip eşiğinin içindeyse "
    "ve son iki tamamlanmış 15 dakikalık mumun kapanışları yükseliyorsa verilir."
)

if buy_df.empty:
    st.info("Şu anda tanımlanan iki koşulu birden karşılayan coin yok.")
else:
    st.dataframe(
        buy_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.8g"),
            "30D Low": st.column_config.NumberColumn(format="%.8g"),
            "Distance from 30D Low (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "24h Change (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "24h Volume (USDT)": st.column_config.NumberColumn(
                format="%.0f"
            )
        }
    )

# ---------------- SELL signals ----------------

with st.expander("SELL sinyalleri ve diğer taranan coinler"):
    st.dataframe(
        sell_df,
        use_container_width=True,
        hide_index=True
    )

# ---------------- Selected coin chart ----------------

st.subheader("15 dakikalık mum grafiği")

available_symbols = sorted(df["Symbol"].tolist())
selected_symbol = st.selectbox(
    "Grafiğini görmek istediğin coin",
    available_symbols
)

try:
    chart_candles = get_klines(selected_symbol, "15m", 100)
    # Exclude the currently forming candle.
    chart_candles = chart_candles[:-1]

    chart_df = pd.DataFrame(
        {
            "Open time": pd.to_datetime(
                [int(candle[0]) for candle in chart_candles],
                unit="ms",
                utc=True
            ),
            "Open": [float(candle[1]) for candle in chart_candles],
            "High": [float(candle[2]) for candle in chart_candles],
            "Low": [float(candle[3]) for candle in chart_candles],
            "Close": [float(candle[4]) for candle in chart_candles],
            "Volume": [float(candle[5]) for candle in chart_candles]
        }
    )

    chart_df = chart_df.set_index("Open time")
    st.line_chart(chart_df[["Close"]], use_container_width=True)

    with st.expander("Son tamamlanmış mumların verileri"):
        st.dataframe(
            chart_df.tail(20),
            use_container_width=True
        )

except BinanceAPIError as exc:
    st.warning(str(exc))

# ---------------- Download ----------------

csv_data = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Sonuçları CSV olarak indir",
    data=csv_data,
    file_name="binance_spot_dip_signals.csv",
    mime="text/csv"
)

st.caption(
    "Veriler Binance halka açık piyasa uç noktalarından alınır. "
    "Bu uygulama emir göndermez, yatırım tavsiyesi vermez ve "
    "kâr garantisi sunmaz. BUY/SELL etiketleri basit kurallara dayanır."
)

# Optional refresh. Requires a Streamlit version supporting fragments.
if auto_refresh:
    st.info(
        "Otomatik yenileme için sayfayı 5 dakikada bir yeniden çalıştırın. "
        "Bu sürümde otomatik yenileme zamanlayıcısı etkin değildir."
    )
```
