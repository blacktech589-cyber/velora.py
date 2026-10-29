
import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import time

st.set_page_config(
    page_title="400 Candle Market Extremes",
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
    "AUD/USD (OTC)": "AUDUSD=X",
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
    ticker = urllib.parse.quote(symbol, safe="^=")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={period}&interval={interval}"
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
    quote = result.get("indicators", {}).get("quote", [{}])[0]

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
    df = df[
        (df.index + pd.to_timedelta(seconds, unit="s")) <= now
    ]

    return df.tail(int(count))


def scan_asset(name, symbol, interval, period, lookback):
    if not symbol:
        return {
            "Asset": name,
            "LOW": None,
            "LOW time": "-",
            "HIGH": None,
            "HIGH time": "-",
            "Latest": None,
            "Position %": None,
            "Status": "No external data symbol"
        }

    try:
        df = fetch_candles(symbol, interval, period, lookback)

        if len(df) < 2:
            raise ValueError("Insufficient candle data")

        low_time = df["Low"].idxmin()
        high_time = df["High"].idxmax()

        low = float(df.loc[low_time, "Low"])
        high = float(df.loc[high_time, "High"])
        latest = float(df["Close"].iloc[-1])

        if high > low:
            position = (latest - low) / (high - low) * 100
            position = max(0.0, min(100.0, position))
        else:
            position = 50.0

        return {
            "Asset": name,
            "LOW": low,
            "LOW time": str(low_time),
            "HIGH": high,
            "HIGH time": str(high_time),
            "Latest": latest,
            "Position %": round(position, 2),
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
            "Position %": None,
            "Status": "Data unavailable: " + str(exc)[:100]
        }


def show_extreme_card(row, label, color):
    st.markdown(
        f"<div style='border:1px solid {color};padding:16px;"
        f"border-radius:12px;margin-bottom:8px'>"
        f"<div style='color:{color};font-size:14px'>{label}</div>"
        f"<div style='font-size:25px;font-weight:bold'>"
        f"{row['Asset']}</div>"
        f"<div>Position in 400-candle range: "
        f"{row['Position %']:.2f}%</div>"
        f"<div>LOW: {row['LOW']:.8g}</div>"
        f"<div>HIGH: {row['HIGH']:.8g}</div>"
        f"<div>Latest: {row['Latest']:.8g}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


st.title("400 Candle — Market Extremes")

st.write(
    "Compare each asset's latest price with its own 400-candle range. "
    "Position 0% is the bottom of the range; 100% is the top."
)

st.warning(
    "External Yahoo Finance prices are not Binomo OTC quotes. "
    "Unavailable assets are excluded from the ranking."
)

col1, col2 = st.columns(2)

with col1:
    timeframe_label = st.selectbox(
        "Timeframe", list(TIMEFRAMES.keys()), index=2
    )

with col2:
    lookback = st.selectbox(
        "Lookback candles", [100, 200, 300, 400], index=3
    )

interval, period = TIMEFRAMES[timeframe_label]

mode = st.radio(
    "Scanning mode",
    ["Scan all assets", "Single asset"],
    horizontal=True
)

if mode == "Single asset":
    selected = st.selectbox("Asset", list(ASSETS.keys()))

    if st.button("Analyze asset", type="primary"):
        with st.spinner("Loading candles..."):
            row = scan_asset(
                selected,
                ASSETS[selected],
                interval,
                period,
                int(lookback)
            )

        if row["Position %"] is None:
            st.error(row["Status"])
        else:
            show_extreme_card(
                row, "ASSET RANGE POSITION", "#64748b"
            )

            df = fetch_candles(
                ASSETS[selected], interval, period, int(lookback)
            )
            st.subheader("Price history")
            st.line_chart(df[["Close"]], height=350)

else:
    if st.button("Start scan", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        total = len(ASSETS)

        for i, (name, symbol) in enumerate(ASSETS.items(), start=1):
            result = scan_asset(
                name, symbol, interval, period, int(lookback)
            )
            results.append(result)
            progress.progress(i / total)
            status.write(f"Scanning {i}/{total}: {name}")
            time.sleep(0.05)

        st.session_state["scan_results"] = pd.DataFrame(results)

    if "scan_results" in st.session_state:
        table = st.session_state["scan_results"]

        valid = table[
            table["Position %"].notna()
            & (table["Status"].str.startswith("OK"))
        ].copy()

        unavailable = table[
            ~table.index.isin(valid.index)
        ]

        if valid.empty:
            st.error(
                "No usable market data was returned. "
                "Check whether the data provider is accessible."
            )
        else:
            valid = valid.sort_values("Position %", ascending=True)

            lowest = valid.iloc[0]
            highest = valid.iloc[-1]

            st.subheader("Market extremes")

            left, right = st.columns(2)

            with left:
                show_extreme_card(
                    lowest, "🔴 LOWEST POSITION", "#ef4444"
                )

            with right:
                show_extreme_card(
                    highest, "🟢 HIGHEST POSITION", "#22c55e"
                )

            st.subheader("All assets — bottom to top")

            for _, row in valid.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Asset']}**")
                    st.markdown(f"LOW: `{row['LOW']:.8g}`")
                    st.markdown(f"HIGH: `{row['HIGH']:.8g}`")
                    st.markdown(f"Latest: `{row['Latest']:.8g}`")
                    st.progress(float(row["Position %"]) / 100)
                    st.caption(
                        f"Range position: {row['Position %']:.2f}%"
                    )

            st.subheader("Unavailable assets")
            if not unavailable.empty:
                st.dataframe(
                    unavailable[["Asset", "Status"]],
                    use_container_width=True,
                    hide_index=True
                )

            st.download_button(
                "Download CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="market_extremes_400.csv",
                mime="text/csv"
            )

st.divider()
st.caption(
    "The ranking measures relative position within each asset's own "
    "historical range. It does not guarantee a reversal or profitable trade."
)
