
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Binomo Extreme Scanner",
    page_icon="📈",
    layout="wide"
)

# -------------------- ASSETS --------------------

ASSETS = {
    "Crypto IDX": None,
    "Booo": None,
    "Ghosty": None,
    "Dusky": None,
    "Bitcoin (OTC)": None,
    "Ethereum (OTC)": None,
    "Solana (OTC)": None,
    "EUR/USD": "EURUSD=X",
    "AUD/USD (OTC)": None,
    "GBP/NOK": "GBPNOK=X",
    "CAD/JPY": "CADJPY=X",
    "FC Barcelona Token (OTC)": None,
    "AUD/ZAR": "AUDZAR=X",
    "CAD/DKK": "CADDKK=X",
    "GBP/SGD": "GBPSGD=X",
    "Cardano (OTC)": None,
    "EUR/ZAR": "EURZAR=X",
    "GBP/HKD": "GBPHKD=X",
    "CAD/MXN": "CADMXN=X",
    "CHF/JPY": "CHFJPY=X",
    "NZD/USD (OTC)": None,
    "EUR/MXN": "EURMXN=X",
    "CAD/SEK": "CADSEK=X",
    "CAD/NOK": "CADNOK=X",
    "AUD/JPY": "AUDJPY=X",
    "Chainlink (OTC)": None,
    "GBP/MXN": "GBPMXN=X",
    "AUD/SGD": "AUDSGD=X",
    "AUD/DKK": "AUDDKK=X",
    "GBP/CZK": "GBPCZK=X",
    "EUR/JPY": "EURJPY=X",
    "NZD/JPY": "NZDJPY=X",
    "GBP/TRY": "GBPTRY=X",
    "AUD/SEK": "AUDSEK=X",
    "GBP/JPY (OTC)": None,
    "NZD/CAD": "NZDCAD=X",
    "NZD/DKK": "NZDDKK=X",
    "EUR/GBP (OTC)": None,
    "NZD/SEK": "NZDSEK=X",
    "SGD/JPY": "SGDJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "USD/NOK": "NOK=X",
    "USD/CAD (OTC)": None,
    "Bitcoin Cash (OTC)": None,
    "Kusama (OTC)": None,
    "Aave (OTC)": None,
    "Pancake Swap (OTC)": None,
    "Uniswap (OTC)": None,
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

# -------------------- HEADER --------------------

st.title("📈 Binomo 400-Candle Extreme Scanner")
st.write(
    "Scans available market data for potential BUY-at-the-low "
    "and SELL-at-the-high reversal signals."
)

st.warning(
    "This is an external market-data scanner, not a Binomo-connected "
    "indicator. OTC prices may differ from external market prices. "
    "Signals are not guarantees of future price movement."
)

# -------------------- SETTINGS --------------------

col1, col2, col3 = st.columns(3)

with col1:
    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m", "30m", "1h", "1d"],
        index=1
    )

with col2:
    lookback = st.number_input(
        "Lookback candles",
        min_value=50,
        max_value=400,
        value=400,
        step=50
    )

with col3:
    tolerance = st.slider(
        "Extreme tolerance (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1
    ) / 100

asset_names = list(ASSETS.keys())

mode = st.radio(
    "Scanning mode",
    ["Single asset", "Scan all supported assets"],
    horizontal=True
)

selected_asset = None

if mode == "Single asset":
    selected_asset = st.selectbox("Select asset", asset_names)

# -------------------- DATA FUNCTIONS --------------------

@st.cache_data(ttl=60, show_spinner=False)
def download_data(symbol, interval):
    if not symbol:
        return pd.DataFrame()

    period_map = {
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "1d": "max",
    }

    try:
        df = yf.download(
            symbol,
            period=period_map[interval],
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()

        df = df[needed].copy()

        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        return df

    except Exception:
        return pd.DataFrame()


def analyze_asset(name, symbol, interval, lookback, tolerance):
    if not symbol:
        return {
            "Asset": name,
            "Price": None,
            "Signal": "OTC / no external feed",
            "Candles": 0,
            "Error": "No external symbol configured"
        }, pd.DataFrame()

    df = download_data(symbol, interval)

    if df.empty:
        return {
            "Asset": name,
            "Price": None,
            "Signal": "No data",
            "Candles": 0,
            "Error": "Data unavailable"
        }, df

    df = df.copy()

    # Use the previous N completed bars to establish extremes.
    n = int(lookback)
    df["PreviousLow"] = df["Low"].shift(1).rolling(n).min()
    df["PreviousHigh"] = df["High"].shift(1).rolling(n).max()

    # Potential reversal at the lower extreme:
    # price approaches/breaks the prior low and candle closes bullish.
    df["BUY"] = (
        (df["Low"] <= df["PreviousLow"] * (1 + tolerance))
        & (df["Close"] > df["Open"])
    )

    # Potential reversal at the upper extreme:
    # price approaches/breaks the prior high and candle closes bearish.
    df["SELL"] = (
        (df["High"] >= df["PreviousHigh"] * (1 - tolerance))
        & (df["Close"] < df["Open"])
    )

    df["Signal"] = "WAIT"
    df.loc[df["BUY"], "Signal"] = "BUY"
    df.loc[df["SELL"], "Signal"] = "SELL"

    valid = df.dropna(subset=["PreviousLow", "PreviousHigh"])

    if valid.empty:
        return {
            "Asset": name,
            "Price": float(df["Close"].iloc[-1]),
            "Signal": "Need more candles",
            "Candles": len(df),
            "Error": "Insufficient history"
        }, df

    # Avoid acting on a candle that may still be forming.
    # The most recent bar is displayed, but its signal is provisional
    # until that candle has closed.
    latest = valid.iloc[-1]

    result = {
        "Asset": name,
        "Price": round(float(latest["Close"]), 6),
        "Signal": str(latest["Signal"]),
        "Candles": len(df),
        "400-candle low": round(float(latest["PreviousLow"]), 6),
        "400-candle high": round(float(latest["PreviousHigh"]), 6),
        "Error": ""
    }

    return result, df


# -------------------- SINGLE ASSET VIEW --------------------

def show_chart(name, df):
    if df.empty:
        st.info("No chart data is available for this asset.")
        return

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=name
    ))

    n = int(lookback)
    df = df.copy()
    df["LowLevel"] = df["Low"].shift(1).rolling(n).min()
    df["HighLevel"] = df["High"].shift(1).rolling(n).max()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["LowLevel"],
        mode="lines",
        name="Previous 400-candle low",
        line=dict(color="red", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["HighLevel"],
        mode="lines",
        name="Previous 400-candle high",
        line=dict(color="green", width=1)
    ))

    valid = df.dropna(subset=["LowLevel", "HighLevel"])

    buys = valid[
        (valid["Low"] <= valid["LowLevel"] * (1 + tolerance))
        & (valid["Close"] > valid["Open"])
    ]

    sells = valid[
        (valid["High"] >= valid["HighLevel"] * (1 - tolerance))
        & (valid["Close"] < valid["Open"])
    ]

    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys["Low"],
        mode="markers",
        name="BUY",
        marker=dict(color="lime", size=11, symbol="triangle-up")
    ))

    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells["High"],
        mode="markers",
        name="SELL",
        marker=dict(color="red", size=11, symbol="triangle-down")
    ))

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        title=f"{name} | {timeframe} | {lookback} candles",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


if mode == "Single asset":
    symbol = ASSETS[selected_asset]

    if symbol is None:
        st.error(
            "This asset is an OTC or synthetic listing without a configured "
            "external data feed. Its real Binomo candles cannot be retrieved "
            "by this app."
        )
        st.stop()

    if st.button("Analyze asset", type="primary"):
        with st.spinner(f"Loading {selected_asset}..."):
            result, df = analyze_asset(
                selected_asset, symbol, timeframe, lookback, tolerance
            )

        if result["Price"] is None:
            st.error("Market data could not be retrieved. Try another asset.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest price", result["Price"])
            c2.metric("Signal", result["Signal"])
            c3.metric("Candles received", result["Candles"])

            if result["Signal"] == "BUY":
                st.success("Potential BUY reversal at the lower extreme.")
            elif result["Signal"] == "SELL":
                st.error("Potential SELL reversal at the upper extreme.")
            else:
                st.info("WAIT: no qualifying extreme reversal on the latest bar.")

            show_chart(selected_asset, df)

            st.subheader("Recent candles")
            st.dataframe(df.tail(30), use_container_width=True)

# -------------------- ALL ASSETS SCANNER --------------------

else:
    st.caption(
        "Only assets with configured external symbols can be scanned. "
        "OTC listings without external feeds are reported as unavailable."
    )

    if st.button("Start scan", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        supported = [
            (name, symbol)
            for name, symbol in ASSETS.items()
            if symbol is not None
        ]

        unsupported = [
            {
                "Asset": name,
                "Price": None,
                "Signal": "OTC / no external feed",
                "Candles": 0,
                "400-candle low": None,
                "400-candle high": None,
                "Error": "No external symbol configured"
            }
            for name, symbol in ASSETS.items()
            if symbol is None
        ]

        results.extend(unsupported)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    analyze_asset,
                    name,
                    symbol,
                    timeframe,
                    lookback,
                    tolerance
                ): name
                for name, symbol in supported
            }

            total = len(futures)

            for i, future in enumerate(as_completed(futures), start=1):
                try:
                    result, _ = future.result()
                    results.append(result)
                except Exception:
                    results.append({
                        "Asset": futures[future],
                        "Price": None,
                        "Signal": "Error",
                        "Candles": 0,
                        "400-candle low": None,
                        "400-candle high": None,
                        "Error": "Scan failed"
                    })

                progress.progress(i / total)
                status.write(f"Scanned {i} of {total} supported assets")

        table = pd.DataFrame(results)

        st.session_state["scan_results"] = table

    if "scan_results" in st.session_state:
        table = st.session_state["scan_results"]

        filter_choice = st.selectbox(
            "Filter results",
            ["All", "BUY only", "SELL only", "WAIT only", "Unavailable"]
        )

        if filter_choice == "BUY only":
            shown = table[table["Signal"] == "BUY"]
        elif filter_choice == "SELL only":
            shown = table[table["Signal"] == "SELL"]
        elif filter_choice == "WAIT only":
            shown = table[table["Signal"] == "WAIT"]
        elif filter_choice == "Unavailable":
            shown = table[
                table["Signal"].isin(
                    ["OTC / no external feed", "No data", "Error"]
                )
            ]
        else:
            shown = table

        st.dataframe(shown, use_container_width=True, hide_index=True)

        buys = int((table["Signal"] == "BUY").sum())
        sells = int((table["Signal"] == "SELL").sum())

        a, b, c = st.columns(3)
        a.metric("BUY signals", buys)
        b.metric("SELL signals", sells)
        c.metric("Total listed assets", len(ASSETS))

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan results CSV",
            data=csv,
            file_name="binomo_extreme_scan.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "External quotes can differ from Binomo OTC quotes. This app does not "
    "place trades. A reversal signal is not a guarantee of profitability."
)
