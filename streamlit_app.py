
import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import time

st.set_page_config(
    page_title="400 Candle LOW HIGH Scanner",
    page_icon="📊",
    layout="wide"
)

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

TIMEFRAMES = {
    "1 minute": ("1m", "1d"),
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "60d"),
    "30 minutes": ("30m", "60d"),
    "1 hour": ("60m", "730d"),
    "1 day": ("1d", "max"),
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol, interval, period, count):
    encoded_symbol = urllib.parse.quote(symbol, safe="^=")

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{encoded_symbol}?range={period}&interval={interval}"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
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

    quote = result.get("indicators", {}).get(
        "quote", [{}]
    )[0]

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True)
    )

    df.index.name = "Time"
    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).sort_index()

    seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "1d": 86400,
    }[interval]

    now = pd.Timestamp.now(tz="UTC")

    # Remove a candle that may still be forming.
    df = df[
        (df.index + pd.to_timedelta(seconds, unit="s")) <= now
    ]

    return df.tail(int(count))


def scan_asset(name, symbol, interval, period, count):
    if symbol is None:
        return {
            "Asset": name,
            "LOW": None,
            "LOW time": "-",
            "HIGH": None,
            "HIGH time": "-",
            "Latest": None,
            "Status": "No external data symbol"
        }

    try:
        df = fetch_candles(symbol, interval, period, count)

        if df.empty:
            raise ValueError("No candles returned")

        low_time = df["Low"].idxmin()
        high_time = df["High"].idxmax()

        low = float(df.loc[low_time, "Low"])
        high = float(df.loc[high_time, "High"])
        latest = float(df["Close"].iloc[-1])

        return {
            "Asset": name,
            "LOW": low,
            "LOW time": str(low_time),
            "HIGH": high,
            "HIGH time": str(high_time),
            "Latest": latest,
            "Status": f"OK — {len(df)} candles"
        }

    except Exception as exc:
        return {
            "Asset": name,
            "LOW": None,
            "LOW time": "-",
            "HIGH": None,
            "HIGH time": "-",
            "Latest": None,
            "Status": "Data error: " + str(exc)[:100]
        }


def show_asset_card(row):
    with st.container(border=True):
        st.markdown(f"### {row['Asset']}")

        if row["Status"].startswith("OK"):
            st.markdown(
                f"<div style='color:#ff5c5c;font-size:23px;"
                f"font-weight:700'>LOW: {row['LOW']:.8g}</div>",
                unsafe_allow_html=True
            )

            st.caption("LOW time: " + row["LOW time"])

            st.markdown(
                f"<div style='color:#32d583;font-size:23px;"
                f"font-weight:700'>HIGH: {row['HIGH']:.8g}</div>",
                unsafe_allow_html=True
            )

            st.caption("HIGH time: " + row["HIGH time"])
            st.write(f"Latest price: {row['Latest']:.8g}")
            st.caption(row["Status"])
        else:
            st.write("LOW: Unavailable")
            st.write("HIGH: Unavailable")
            st.caption(row["Status"])


st.title("400 Candle LOW / HIGH Scanner")

st.write(
    "Her varlığın son tamamlanmış mumları içindeki en düşük "
    "ve en yüksek fiyatları ayrı ayrı gösterilir."
)

st.warning(
    "Yahoo Finance dış piyasa verisi kullanılır. Bu veriler "
    "Binomo OTC fiyatlarıyla aynı olmak zorunda değildir. "
    "Verisi olmayan varlıklarda tahmini fiyat gösterilmez."
)

col1, col2 = st.columns(2)

with col1:
    timeframe_name = st.selectbox(
        "Timeframe",
        list(TIMEFRAMES.keys()),
        index=2
    )

with col2:
    candle_count = st.selectbox(
        "Number of candles",
        [100, 200, 300, 400],
        index=3
    )

interval, period = TIMEFRAMES[timeframe_name]

mode = st.radio(
    "Mode",
    ["One asset", "All assets"],
    horizontal=True
)

if mode == "One asset":
    selected = st.selectbox(
        "Select asset",
        list(ASSETS.keys())
    )

    if st.button("Show LOW and HIGH", type="primary"):
        with st.spinner("Loading candles..."):
            row = scan_asset(
                selected,
                ASSETS[selected],
                interval,
                period,
                int(candle_count)
            )

        show_asset_card(row)

        if row["Status"].startswith("OK"):
            df = fetch_candles(
                ASSETS[selected],
                interval,
                period,
                int(candle_count)
            )

            st.subheader("Price history")
            st.line_chart(df[["Close"]], height=300)

            levels = pd.DataFrame(
                {
                    "LOW": [df["Low"].min()] * len(df),
                    "HIGH": [df["High"].max()] * len(df),
                    "Close": df["Close"]
                },
                index=df.index
            )

            st.subheader("LOW / HIGH levels")
            st.line_chart(levels, height=300)

else:
    if st.button("Scan all assets", type="primary"):
        results = []
        progress = st.progress(0)
        status_text = st.empty()
        total = len(ASSETS)

        for index, (name, symbol) in enumerate(
            ASSETS.items(), start=1
        ):
            row = scan_asset(
                name,
                symbol,
                interval,
                period,
                int(candle_count)
            )

            results.append(row)
            progress.progress(index / total)
            status_text.write(f"{index}/{total}: {name}")

            time.sleep(0.05)

        st.session_state["results"] = pd.DataFrame(results)

    if "results" in st.session_state:
        table = st.session_state["results"]

        filter_mode = st.selectbox(
            "Filter assets",
            ["All", "Available only", "Unavailable only"]
        )

        if filter_mode == "Available only":
            displayed = table[
                table["Status"].str.startswith("OK")
            ]
        elif filter_mode == "Unavailable only":
            displayed = table[
                ~table["Status"].str.startswith("OK")
            ]
        else:
            displayed = table

        st.subheader("Each asset — LOW above HIGH")

        # One card per asset, with LOW and HIGH stacked vertically.
        for _, row in displayed.iterrows():
            show_asset_card(row)

        st.download_button(
            "Download all results CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="low_high_400_candles.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "LOW and HIGH are historical extremes in the retrieved candles. "
    "They do not predict the next market movement."
)
