
import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import time

st.set_page_config(
    page_title="400 Candle Extreme Scanner",
    page_icon="📈",
    layout="wide"
)

# Binance public market symbols.
# None means no verified external market feed configured.
ASSETS = {
    "Crypto IDX": None,
    "Booo": None,
    "Ghosty": None,
    "Dusky": None,
    "Bitcoin (OTC)": "BTCUSDT",
    "Ethereum (OTC)": "ETHUSDT",
    "Solana (OTC)": "SOLUSDT",
    "EUR/USD": None,
    "AUD/USD (OTC)": None,
    "GBP/NOK": None,
    "CAD/JPY": None,
    "FC Barcelona Token (OTC)": None,
    "AUD/ZAR": None,
    "CAD/DKK": None,
    "GBP/SGD": None,
    "Cardano (OTC)": "ADAUSDT",
    "EUR/ZAR": None,
    "GBP/HKD": None,
    "CAD/MXN": None,
    "CHF/JPY": None,
    "NZD/USD (OTC)": None,
    "EUR/MXN": None,
    "CAD/SEK": None,
    "CAD/NOK": None,
    "AUD/JPY": None,
    "Chainlink (OTC)": "LINKUSDT",
    "GBP/MXN": None,
    "AUD/SGD": None,
    "AUD/DKK": None,
    "GBP/CZK": None,
    "EUR/JPY": None,
    "NZD/JPY": None,
    "GBP/TRY": None,
    "AUD/SEK": None,
    "GBP/JPY (OTC)": None,
    "NZD/CAD": None,
    "NZD/DKK": None,
    "EUR/GBP (OTC)": None,
    "NZD/SEK": None,
    "SGD/JPY": None,
    "AUD/CAD": None,
    "USD/NOK": None,
    "USD/CAD (OTC)": None,
    "Bitcoin Cash (OTC)": "BCHUSDT",
    "Kusama (OTC)": "KSMUSDT",
    "Aave (OTC)": "AAVEUSDT",
    "Pancake Swap (OTC)": "CAKEUSDT",
    "Uniswap (OTC)": "UNIUSDT",
    "NZD/CHF": None,
    "GBP/CAD": None,
    "AUD/CHF": None,
    "Microsoft": None,
    "Boeing": None,
    "US International Group": None,
    "McDonalds": None,
    "JD": None,
    "PayPal": None,
    "NASDAQ 100": None,
    "Dow Jones IA": None,
    "GBP/USD": None,
    "VOO": None,
    "iShares Russell 2000 ETF": None,
    "Invesco Nasdaq 100 ETF": None,
    "ProShares UltraPro QQQ": None,
    "Google": None,
    "IBM": None,
    "ZAR/JPY": None,
    "Litecoin": "LTCUSDT",
    "USD/DKK": None,
    "GBP/CHF": None,
    "Yum Brands": None,
}

INTERVALS = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
}

API_URL = "https://api.binance.com/api/v3/klines"


@st.cache_data(ttl=30, show_spinner=False)
def download_candles(symbol, interval, limit):
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "limit": min(int(limit) + 2, 1000),
    })

    request = urllib.request.Request(
        API_URL + "?" + params,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        raw = json.loads(response.read().decode("utf-8"))

    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()

    columns = [
        "OpenTime", "Open", "High", "Low", "Close",
        "Volume", "CloseTime", "QuoteVolume", "Trades",
        "TakerBuyBase", "TakerBuyQuote", "Ignore",
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Time"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # The latest candle may still be forming.
    if len(df) > 1:
        df = df.iloc[:-1]

    return df.tail(int(limit))


def analyze_candles(df, lookback, tolerance):
    if df.empty or len(df) < lookback + 1:
        return df, None

    df = df.copy()

    # Rolling levels from the preceding candles, excluding current candle.
    df["PriorLow"] = df["Low"].shift(1).rolling(lookback).min()
    df["PriorHigh"] = df["High"].shift(1).rolling(lookback).max()

    # BUY near a low after a bullish candle.
    df["BUY"] = (
        (df["Low"] <= df["PriorLow"] * (1 + tolerance))
        & (df["Close"] > df["Open"])
    )

    # SELL near a high after a bearish candle.
    df["SELL"] = (
        (df["High"] >= df["PriorHigh"] * (1 - tolerance))
        & (df["Close"] < df["Open"])
    )

    df["Signal"] = "WAIT"
    df.loc[df["BUY"], "Signal"] = "BUY"
    df.loc[df["SELL"], "Signal"] = "SELL"

    last = df.iloc[-1]

    result = {
        "Price": float(last["Close"]),
        "Signal": str(last["Signal"]),
        "Low": float(last["PriorLow"]),
        "High": float(last["PriorHigh"]),
        "Time": str(df.index[-1]),
        "Candles": len(df),
    }

    return df, result


def show_chart(df):
    if df.empty:
        st.info("No chart data available.")
        return

    chart = df[["Open", "High", "Low", "Close"]].copy()
    chart = chart.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
    })

    st.subheader("Price chart")
    st.caption("Streamlit native chart; OHLC values are shown below.")

    st.line_chart(chart[["Close"]], height=350)

    st.subheader("400-candle extreme levels")
    st.line_chart(
        df[["Close", "PriorLow", "PriorHigh"]].tail(200),
        height=300,
    )

    st.subheader("Recent candles and signals")
    recent = df.tail(30).copy()
    recent.index = recent.index.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(recent, use_container_width=True)


st.title("400 Candle Extreme Scanner")

st.write(
    "Analyzes the latest completed candles for potential BUY signals "
    "near lower extremes and SELL signals near upper extremes."
)

st.warning(
    "This app uses Binance public market data, not Binomo's own OTC feed. "
    "Prices and signals can differ from Binomo. No trades are executed."
)

col1, col2, col3 = st.columns(3)

with col1:
    timeframe_label = st.selectbox(
        "Timeframe", list(INTERVALS.keys()), index=2
    )

with col2:
    lookback = st.number_input(
        "Lookback candles", min_value=50, max_value=400,
        value=400, step=50
    )

with col3:
    tolerance_pct = st.slider(
        "Extreme zone (%)", 0.0, 2.0, 0.2, 0.1
    )

timeframe = INTERVALS[timeframe_label]
tolerance = tolerance_pct / 100

mode = st.radio(
    "Scanner mode",
    ["Single asset", "Scan supported assets"],
    horizontal=True,
)

if mode == "Single asset":
    names = list(ASSETS.keys())
    selected = st.selectbox("Select asset", names)
    symbol = ASSETS[selected]

    if symbol is None:
        st.info(
            f"{selected} has no external market symbol configured. "
            "Real Binomo OTC candles are not available through this feed."
        )

    if st.button("Analyze asset", type="primary"):
        if symbol is None:
            st.error("Cannot analyze this asset without real price data.")
        else:
            try:
                with st.spinner("Downloading completed candles..."):
                    df = download_candles(
                        symbol, timeframe, int(lookback) + 1
                    )
                    df, result = analyze_candles(
                        df, int(lookback), tolerance
                    )

                if result is None:
                    st.error(
                        "Not enough completed candles were returned "
                        "to calculate the selected lookback."
                    )
                else:
                    a, b, c = st.columns(3)
                    a.metric("Latest price", f"{result['Price']:.8g}")
                    b.metric("Signal", result["Signal"])
                    c.metric("Candles", result["Candles"])

                    if result["Signal"] == "BUY":
                        st.success("Potential BUY reversal signal")
                    elif result["Signal"] == "SELL":
                        st.error("Potential SELL reversal signal")
                    else:
                        st.info("WAIT: no qualifying reversal on the latest candle")

                    st.caption("Last completed candle: " + result["Time"])
                    show_chart(df)

            except Exception as exc:
                st.error("Could not retrieve market data.")
                st.caption(
                    "Check internet access, symbol availability, "
                    "or API rate limits. Details: " + str(exc)[:300]
                )

else:
    supported = {
        name: symbol
        for name, symbol in ASSETS.items()
        if symbol is not None
    }

    st.caption(
        "Only assets with configured Binance symbols are scanned. "
        "This does not scan Binomo's private OTC prices."
    )

    if st.button("Start full scan", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        total = len(supported)

        for i, (name, symbol) in enumerate(supported.items(), start=1):
            try:
                df = download_candles(
                    symbol, timeframe, int(lookback) + 1
                )
                df, result = analyze_candles(
                    df, int(lookback), tolerance
                )

                if result is None:
                    results.append({
                        "Asset": name,
                        "Symbol": symbol,
                        "Price": None,
                        "Signal": "No data",
                        "Candles": len(df),
                    })
                else:
                    results.append({
                        "Asset": name,
                        "Symbol": symbol,
                        "Price": result["Price"],
                        "Signal": result["Signal"],
                        "Candles": result["Candles"],
                        "Prior low": result["Low"],
                        "Prior high": result["High"],
                        "Last candle": result["Time"],
                    })

            except Exception as exc:
                results.append({
                    "Asset": name,
                    "Symbol": symbol,
                    "Price": None,
                    "Signal": "Error",
                    "Candles": 0,
                    "Error": str(exc)[:150],
                })

            progress.progress(i / total)
            status.write(f"Scanned {i}/{total}: {name}")
            time.sleep(0.15)

        table = pd.DataFrame(results)
        st.session_state["scan_results"] = table

    if "scan_results" in st.session_state:
        table = st.session_state["scan_results"]

        filter_option = st.selectbox(
            "Filter results",
            ["All", "BUY", "SELL", "WAIT", "No data / Error"],
        )

        if filter_option in ["BUY", "SELL", "WAIT"]:
            displayed = table[table["Signal"] == filter_option]
        elif filter_option == "No data / Error":
            displayed = table[
                table["Signal"].isin(["No data", "Error"])
            ]
        else:
            displayed = table

        st.dataframe(
            displayed, use_container_width=True, hide_index=True
        )

        buy_count = int((table["Signal"] == "BUY").sum())
        sell_count = int((table["Signal"] == "SELL").sum())
        wait_count = int((table["Signal"] == "WAIT").sum())

        a, b, c = st.columns(3)
        a.metric("BUY signals", buy_count)
        b.metric("SELL signals", sell_count)
        c.metric("WAIT signals", wait_count)

        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="extreme_scanner.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Analysis only. No order execution. Extreme levels do not guarantee "
    "reversals or profitable trades."
)
