
import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import time
from datetime import datetime, timezone

st.set_page_config(
    page_title="400 Candle High Low Scanner",
    page_icon="📊",
    layout="wide"
)

# Yahoo Finance symbols are external market proxies, not Binomo OTC feeds.
ASSETS = {
    "Crypto IDX": None,
    "Booo": None,
    "Ghosty": None,
    "Dusky": None,

    "Bitcoin (OTC)": "BTC-USD",
    "Ethereum (OTC)": "ETH-USD",
    "Solana (OTC)": "SOL-USD",
    "EUR/USD": "EURUSD=X",
    "AUD/USD (OTC)": None,
    "GBP/NOK": "GBPNOK=X",
    "CAD/JPY": "CADJPY=X",
    "FC Barcelona Token (OTC)": None,
    "AUD/ZAR": "AUDZAR=X",
    "CAD/DKK": "CADDKK=X",
    "GBP/SGD": "GBPSGD=X",
    "Cardano (OTC)": "ADA-USD",
    "EUR/ZAR": "EURZAR=X",
    "GBP/HKD": "GBPHKD=X",
    "CAD/MXN": "CADMXN=X",
    "CHF/JPY": "CHFJPY=X",
    "NZD/USD (OTC)": "NZDUSD=X",
    "EUR/MXN": "EURMXN=X",
    "CAD/SEK": "CADSEK=X",
    "CAD/NOK": "CADNOK=X",
    "AUD/JPY": "AUDJPY=X",
    "Chainlink (OTC)": "LINK-USD",
    "GBP/MXN": "GBPMXN=X",
    "AUD/SGD": "AUDSGD=X",
    "AUD/DKK": "AUDDKK=X",
    "GBP/CZK": "GBPCZK=X",
    "EUR/JPY": "EURJPY=X",
    "NZD/JPY": "NZDJPY=X",
    "GBP/TRY": "GBPTRY=X",
    "AUD/SEK": "AUDSEK=X",
    "GBP/JPY (OTC)": "GBPJPY=X",
    "NZD/CAD": "NZDCAD=X",
    "NZD/DKK": "NZDDKK=X",
    "EUR/GBP (OTC)": "EURGBP=X",
    "NZD/SEK": "NZDSEK=X",
    "SGD/JPY": "SGDJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "USD/NOK": "NOK=X",
    "USD/CAD (OTC)": "CAD=X",
    "Bitcoin Cash (OTC)": "BCH-USD",
    "Kusama (OTC)": "KSM-USD",
    "Aave (OTC)": "AAVE-USD",
    "Pancake Swap (OTC)": "CAKE-USD",
    "Uniswap (OTC)": "UNI-USD",
    "NZD/CHF": "NZDCHF=X",
    "GBP/CAD": "GBPCAD=X",
    "AUD/CHF": "AUDCHF=X",

    "Microsoft": "MSFT",
    "Boeing": "BA",
    "US International Group": "AIG",
    "McDonalds": "MCD",
    "JD": "JD",
    "PayPal": "PYPL",
    "NASDAQ 100": "^NDX",
    "Dow Jones IA": "^DJI",
    "GBP/USD": "GBPUSD=X",
    "VOO": "VOO",
    "iShares Russell 2000 ETF": "IWM",
    "Invesco Nasdaq 100 ETF": "QQQ",
    "ProShares UltraPro QQQ": "TQQQ",
    "Google": "GOOGL",
    "IBM": "IBM",
    "ZAR/JPY": "ZARJPY=X",
    "Litecoin": "LTC-USD",
    "USD/DKK": "DKK=X",
    "GBP/CHF": "GBPCHF=X",
    "Yum Brands": "YUM",
}

INTERVALS = {
    "1 minute": ("1m", "1d"),
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "60d"),
    "30 minutes": ("30m", "60d"),
    "1 hour": ("60m", "730d"),
    "1 day": ("1d", "max"),
}


@st.cache_data(ttl=60, show_spinner=False)
def download_candles(symbol, interval, period, count):
    """Fetch public Yahoo Finance chart data without yfinance."""
    encoded = urllib.parse.quote(symbol, safe="^=")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={period}&interval={interval}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(str(chart["error"]))

    results = chart.get("result")
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp", [])
    quotes = result.get("indicators", {}).get("quote", [{}])[0]

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Open": quotes.get("open", []),
        "High": quotes.get("high", []),
        "Low": quotes.get("low", []),
        "Close": quotes.get("close", []),
        "Volume": quotes.get("volume", [])
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))

    df.index.name = "Time"
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Exclude the current candle if its interval has not finished.
    seconds = {
        "1m": 60, "5m": 300, "15m": 900,
        "30m": 1800, "60m": 3600, "1d": 86400
    }[interval]

    now = pd.Timestamp.now(tz="UTC")
    df = df[(df.index + pd.to_timedelta(seconds, unit="s")) <= now]

    return df.tail(int(count))


def analyze_asset(name, symbol, interval, period, lookback):
    if not symbol:
        return {
            "Asset": name,
            "Highest": None,
            "Highest time": "-",
            "Lowest": None,
            "Lowest time": "-",
            "Latest": None,
            "Distance from high %": None,
            "Distance from low %": None,
            "Candles": 0,
            "Status": "No external symbol configured"
        }, pd.DataFrame()

    try:
        df = download_candles(symbol, interval, period, lookback)

        if df.empty:
            raise ValueError("No candle data returned")

        high_time = df["High"].idxmax()
        low_time = df["Low"].idxmin()

        highest = float(df.loc[high_time, "High"])
        lowest = float(df.loc[low_time, "Low"])
        latest = float(df["Close"].iloc[-1])

        high_distance = (
            (highest - latest) / highest * 100 if highest else None
        )
        low_distance = (
            (latest - lowest) / lowest * 100 if lowest else None
        )

        record = {
            "Asset": name,
            "Highest": highest,
            "Highest time": str(high_time),
            "Lowest": lowest,
            "Lowest time": str(low_time),
            "Latest": latest,
            "Distance from high %": round(high_distance, 3)
                if high_distance is not None else None,
            "Distance from low %": round(low_distance, 3)
                if low_distance is not None else None,
            "Candles": len(df),
            "Status": "OK"
        }

        return record, df

    except Exception as exc:
        return {
            "Asset": name,
            "Highest": None,
            "Highest time": "-",
            "Lowest": None,
            "Lowest time": "-",
            "Latest": None,
            "Distance from high %": None,
            "Distance from low %": None,
            "Candles": 0,
            "Status": "Data unavailable: " + str(exc)[:100]
        }, pd.DataFrame()


def draw_chart(df):
    if df.empty:
        return

    chart = df[["Close"]].copy()
    chart["Highest level"] = df["High"].max()
    chart["Lowest level"] = df["Low"].min()

    st.line_chart(chart, height=400)

    st.caption(
        f"Highest: {df['High'].max():.8g} | "
        f"Lowest: {df['Low'].min():.8g}"
    )

    st.dataframe(
        df.tail(30),
        use_container_width=True
    )


st.title("400 Candle High / Low Scanner")

st.write(
    "Scan each configured asset and display its highest and lowest "
    "prices over the latest completed candles."
)

st.warning(
    "External data is not Binomo's own OTC feed. A symbol marked OTC "
    "uses an external market proxy where configured. Prices can differ."
)

col1, col2 = st.columns(2)

with col1:
    timeframe_label = st.selectbox(
        "Timeframe", list(INTERVALS.keys()), index=2
    )

with col2:
    lookback = st.selectbox(
        "Number of candles", [100, 200, 300, 400], index=3
    )

interval, period = INTERVALS[timeframe_label]

mode = st.radio(
    "Mode",
    ["Single asset", "Scan all assets"],
    horizontal=True
)

if mode == "Single asset":
    selected = st.selectbox("Select asset", list(ASSETS.keys()))

    if st.button("Analyze", type="primary"):
        with st.spinner("Retrieving market data..."):
            record, df = analyze_asset(
                selected, ASSETS[selected], interval, period, int(lookback)
            )

        if record["Status"] != "OK":
            st.error(record["Status"])
        else:
            a, b, c = st.columns(3)
            a.metric("Highest price", f"{record['Highest']:.8g}")
            b.metric("Lowest price", f"{record['Lowest']:.8g}")
            c.metric("Latest price", f"{record['Latest']:.8g}")

            d, e = st.columns(2)
            d.metric(
                "Below high (%)",
                f"{record['Distance from high %']:.3f}%"
            )
            e.metric(
                "Above low (%)",
                f"{record['Distance from low %']:.3f}%"
            )

            st.write("Highest candle time:", record["Highest time"])
            st.write("Lowest candle time:", record["Lowest time"])
            st.write("Completed candles:", record["Candles"])

            draw_chart(df)

else:
    if st.button("Start scanning all assets", type="primary"):
        rows = []
        progress = st.progress(0)
        status = st.empty()

        total = len(ASSETS)

        for i, (name, symbol) in enumerate(ASSETS.items(), start=1):
            record, _ = analyze_asset(
                name, symbol, interval, period, int(lookback)
            )
            rows.append(record)
            progress.progress(i / total)
            status.write(f"{i}/{total}: {name}")
            time.sleep(0.05)

        st.session_state["results"] = pd.DataFrame(rows)

    if "results" in st.session_state:
        table = st.session_state["results"]

        filter_choice = st.selectbox(
            "Filter",
            ["All", "Available only", "Unavailable only"]
        )

        if filter_choice == "Available only":
            shown = table[table["Status"] == "OK"]
        elif filter_choice == "Unavailable only":
            shown = table[table["Status"] != "OK"]
        else:
            shown = table

        st.subheader("Highest and lowest prices per asset")
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True
        )

        valid = table[table["Status"] == "OK"]

        if not valid.empty:
            st.subheader("Compare each asset's extreme prices")
            st.bar_chart(
                valid.set_index("Asset")[["Highest", "Lowest"]]
            )

        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="asset_high_low_400.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "The highest and lowest values are calculated from available "
    "completed candles. They do not predict future price direction."
)
