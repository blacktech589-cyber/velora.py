import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="3000 Candle Market Extremes",
    page_icon="📊",
    layout="wide",
)

LOOKBACK_DEFAULT = 3000

# These are public-market Yahoo Finance symbols where available.
# Binomo OTC instruments may not match these external data sources.
ASSETS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Chainlink": "LINK-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin Cash": "BCH-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/GBP": "EURGBP=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "NZD/CAD": "NZDCAD=X",
    "GBP/CHF": "GBPCHF=X",
    "NZD/CHF": "NZDCHF=X",
    "Microsoft": "MSFT",
    "Boeing": "BA",
    "McDonald's": "MCD",
    "PayPal": "PYPL",
    "Google": "GOOGL",
    "IBM": "IBM",
    "NASDAQ 100": "^NDX",
    "Dow Jones": "^DJI",
    "VOO ETF": "VOO",
    "Russell 2000 ETF": "IWM",
    "Nasdaq 100 ETF": "QQQ",
    "Yum Brands": "YUM",
}

TIMEFRAMES = {
    "1 minute": {"interval": "1m", "range": "7d", "seconds": 60},
    "5 minutes": {"interval": "5m", "range": "60d", "seconds": 300},
    "15 minutes": {"interval": "15m", "range": "60d", "seconds": 900},
    "30 minutes": {"interval": "30m", "range": "60d", "seconds": 1800},
    "1 hour": {"interval": "60m", "range": "730d", "seconds": 3600},
    "1 day": {"interval": "1d", "range": "max", "seconds": 86400},
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol: str, interval: str, period: str, candle_seconds: int) -> pd.DataFrame:
    """Fetch public Yahoo Finance candles and remove the currently forming candle."""
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?range={urllib.parse.quote(period)}&interval={urllib.parse.quote(interval)}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(str(chart["error"].get("description", "Yahoo Finance data error")))

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Bu sembol için veri bulunamadı.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        raise RuntimeError("Veri kaynağı mum verisi döndürmedi.")

    quote = quotes[0]
    df = pd.DataFrame({
        "Time": pd.to_datetime(timestamps, unit="s", utc=True),
        "Open": quote.get("open", []),
        "High": quote.get("high", []),
        "Low": quote.get("low", []),
        "Close": quote.get("close", []),
        "Volume": quote.get("volume", []),
    })

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        raise RuntimeError("Geçerli mum bulunamadı.")

    # Exclude the currently forming candle, if the last timestamp is still open.
    now_ts = int(time.time())
    last_ts = int(df["Time"].iloc[-1].timestamp())
    if now_ts < last_ts + candle_seconds:
        df = df.iloc[:-1].copy()

    if df.empty:
        raise RuntimeError("Tamamlanmış mum bulunamadı.")

    df["Time"] = df["Time"].dt.tz_convert("Europe/Istanbul")
    return df.reset_index(drop=True)


def scan_asset(name: str, symbol: str, tf_name: str, lookback: int) -> dict:
    tf = TIMEFRAMES[tf_name]
    try:
        df = fetch_candles(
            symbol,
            tf["interval"],
            tf["range"],
            tf["seconds"],
        )
        df = df.tail(lookback).copy()

        if len(df) < min(50, lookback):
            raise RuntimeError(f"Yeterli veri yok: yalnızca {len(df)} tamamlanmış mum.")

        low_index = df["Low"].idxmin()
        high_index = df["High"].idxmax()
        period_low = float(df.loc[low_index, "Low"])
        period_high = float(df.loc[high_index, "High"])
        latest_close = float(df["Close"].iloc[-1])

        if period_high <= period_low:
            position = 50.0
        else:
            position = (latest_close - period_low) / (period_high - period_low) * 100
            position = max(0.0, min(100.0, position))

        return {
            "Asset": name,
            "Symbol": symbol,
            "Candles": len(df),
            "LOW": period_low,
            "LOW time": df.loc[low_index, "Time"].strftime("%Y-%m-%d %H:%M"),
            "HIGH": period_high,
            "HIGH time": df.loc[high_index, "Time"].strftime("%Y-%m-%d %H:%M"),
            "Latest": latest_close,
            "Position %": round(position, 2),
            "Status": "OK",
            "_data": df,
        }
    except Exception as exc:
        return {
            "Asset": name,
            "Symbol": symbol,
            "Candles": 0,
            "LOW": None,
            "LOW time": "",
            "HIGH": None,
            "HIGH time": "",
            "Latest": None,
            "Position %": None,
            "Status": f"Data error: {str(exc)[:180]}",
            "_data": None,
        }


def price_text(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.5f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def extreme_card(row, kind: str):
    is_low = kind == "LOW"
    color = "#dc2626" if is_low else "#16a34a"
    label = "EN DİBE YAKIN" if is_low else "EN TEPEYE YAKIN"
    st.markdown(
        f"""
        <div style="border:2px solid {color};border-radius:14px;padding:18px;margin-bottom:8px;">
          <div style="font-size:13px;font-weight:700;color:{color};">{label}</div>
          <div style="font-size:25px;font-weight:800;margin:6px 0;">{row['Asset']}</div>
          <div style="font-size:15px;">Konum: <b>{row['Position %']:.2f}%</b></div>
          <div style="margin-top:8px;">LOW: <b>{price_text(row['LOW'])}</b></div>
          <div>HIGH: <b>{price_text(row['HIGH'])}</b></div>
          <div>Güncel kapanış: <b>{price_text(row['Latest'])}</b></div>
          <div style="font-size:12px;opacity:.8;margin-top:6px;">{row['Candles']} tamamlanmış mum · {row['Symbol']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("📊 3.000 Mum — Piyasa Dip / Tepe Tarayıcı")
st.caption(
    "Her varlık kendi geçmiş aralığına göre değerlendirilir. "
    "Bu ekran alım-satım emri göndermez."
)

with st.sidebar:
    st.header("Tarama ayarları")
    tf_name = st.selectbox("Zaman dilimi", list(TIMEFRAMES.keys()), index=2)
    lookback = st.selectbox(
        "Analiz edilecek tamamlanmış mum",
        [500, 1000, 2000, 3000],
        index=3,
    )
    mode = st.radio("Tarama modu", ["Tüm varlıklar", "Tek varlık"], index=0)
    if mode == "Tek varlık":
        selected_asset = st.selectbox("Varlık", list(ASSETS.keys()))
    else:
        selected_asset = None
    start_scan = st.button("🔍 Taramayı başlat", type="primary", use_container_width=True)
    st.caption("Veriler Yahoo Finance'ten alınır; Binomo OTC fiyatlarıyla birebir aynı olmayabilir.")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_tf" not in st.session_state:
    st.session_state.scan_tf = None
if "scan_lookback" not in st.session_state:
    st.session_state.scan_lookback = None

if start_scan:
    if mode == "Tek varlık":
        assets_to_scan = {selected_asset: ASSETS[selected_asset]}
    else:
        assets_to_scan = ASSETS

    results = []
    progress = st.progress(0)
    status_line = st.empty()

    for i, (name, symbol) in enumerate(assets_to_scan.items(), start=1):
        status_line.write(f"Taranıyor: {name} ({i}/{len(assets_to_scan)})")
        result = scan_asset(name, symbol, tf_name, lookback)
        results.append(result)
        progress.progress(i / len(assets_to_scan))

    st.session_state.scan_results = results
    st.session_state.scan_tf = tf_name
    st.session_state.scan_lookback = lookback
    status_line.success("Tarama tamamlandı.")
    progress.empty()

results = st.session_state.scan_results

# Backward compatibility: older app versions may have stored scan_results as a DataFrame.
if isinstance(results, pd.DataFrame):
    results = results.to_dict(orient="records")
    for item in results:
        item.setdefault("_data", None)
    st.session_state.scan_results = results

if results is not None and len(results) > 0:
    if st.session_state.scan_tf != tf_name or st.session_state.scan_lookback != lookback:
        st.info("Ayarları değiştirdin. Yeni ayarlarla sonuç almak için 'Taramayı başlat' düğmesine bas.")
    df_all = pd.DataFrame([
        {k: v for k, v in item.items() if k != "_data"}
        for item in results
    ])
    valid = df_all[df_all["Status"] == "OK"].copy()
    errors = df_all[df_all["Status"] != "OK"].copy()

    if not valid.empty:
        valid = valid.sort_values("Position %", ascending=True).reset_index(drop=True)
        lowest = valid.iloc[0]
        highest = valid.iloc[-1]

        st.subheader("En uç konumdaki varlıklar")
        left, right = st.columns(2)
        with left:
            extreme_card(lowest, "LOW")
        with right:
            extreme_card(highest, "HIGH")

        st.subheader("Tüm başarılı sonuçlar")
        st.caption("LOW ve HIGH her varlığın seçilen son mum sayısı içindeki en düşük/en yüksek fiyatıdır.")
        display = valid.copy()
        display["LOW"] = display["LOW"].map(price_text)
        display["HIGH"] = display["HIGH"].map(price_text)
        display["Latest"] = display["Latest"].map(price_text)
        display = display[
            ["Asset", "Symbol", "Candles", "LOW", "LOW time", "HIGH", "HIGH time", "Latest", "Position %"]
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("LOW ve HIGH konum sıralaması")
        st.markdown("**En dibe yakın → en tepeye yakın**")
        for _, row in valid.iterrows():
            pos = float(row["Position %"])
            st.markdown(
                f"""
                <div style="margin:8px 0;padding:10px;border:1px solid #8885;border-radius:10px;">
                  <div style="display:flex;justify-content:space-between;gap:10px;">
                    <b>{row['Asset']}</b><span>{pos:.2f}%</span>
                  </div>
                  <div style="height:9px;background:#8883;border-radius:8px;margin-top:7px;">
                    <div style="height:9px;width:{pos}%;background:{'#dc2626' if pos < 30 else '#eab308' if pos < 70 else '#16a34a'};border-radius:8px;"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:12px;opacity:.8;margin-top:4px;">
                    <span>LOW: {price_text(row['LOW'])}</span><span>HIGH: {price_text(row['HIGH'])}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        csv = valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Sonuçları CSV indir",
            data=csv,
            file_name="3000_mum_piyasa_tarama.csv",
            mime="text/csv",
        )
    else:
        st.warning("Başarılı veri alınamadı. Zaman dilimini değiştir veya biraz sonra tekrar dene.")

    if not errors.empty:
        with st.expander(f"Veri alınamayan varlıklar ({len(errors)})"):
            # Older saved results may not contain all columns; add them safely.
            for column in ["Asset", "Symbol", "Status"]:
                if column not in errors.columns:
                    errors[column] = ""
            st.dataframe(
                errors[["Asset", "Symbol", "Status"]],
                use_container_width=True,
                hide_index=True,
            )

    if mode == "Tek varlık":
        matching = [item for item in results if item["Asset"] == selected_asset]
        if matching and matching[0].get("_data") is not None:
            one = matching[0]
            st.subheader(f"{one['Asset']} — son {one['Candles']} tamamlanmış mum")
            chart_df = one["_data"].set_index("Time")[["Close"]]
            st.line_chart(chart_df, use_container_width=True)
else:
    st.info("Başlamak için sol menüden ayarları seç ve 'Taramayı başlat' düğmesine bas.")

st.divider()
st.caption(
    "Uyarı: Bu tarayıcı yalnızca dış piyasa verilerini analiz eder. "
    "Geçmiş aralıkta dibe yakınlık tek başına fiyatın yükseleceği, "
    "tepeye yakınlık da düşeceği anlamına gelmez."
)
