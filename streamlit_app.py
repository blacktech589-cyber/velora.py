
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import urllib.request
import urllib.parse
import json
import time

st.set_page_config(
    page_title="400 Candle Extreme Scanner",
    page_icon="📈",
    layout="wide"
)

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

BINANCE_URL = "https://api.binance.com/api/v3/klines"

INTERVALS = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
}

st.title("400 Candle Extreme Scanner")
st.caption("BUY at lower extremes | SELL at upper extremes")

st.warning(
    "This application uses public Binance market data, not Binomo's "
    "OTC price feed. OTC quotes may differ. No trades are executed."
)

c1, c2, c3 = st.columns(3)

with c1:
    interval_label = st.selectbox(
        "Timeframe", list(INTERVALS.keys()), index=2
    )

with c2:
    lookback = st.number_input(
        "Lookback candles", min_value=50, max_value=400,
        value=400, step=50
    )

with c3:
    tolerance_pct = st.slider(
        "Extreme zone (%)", 0.0, 2.0, 0.2, 0.1
    )

interval = INTERVALS[interval_label]
tolerance = tolerance_pct / 100.0


def get_klines(symbol, interval, limit=400):
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "limit": min(int(limit) + 5, 1000)
    })
    url = BINANCE_URL + "?" + params

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=12) as response:
        raw = json.loads(response.read().decode("utf-8"))

    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()

    columns = [
        "Open time", "Open", "High", "Low", "Close",
        "Volume", "Close time", "Quote volume",
        "Trades", "Taker buy base", "Taker buy quote", "Ignore"
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Open time"] = pd.to_datetime(
        df["Open time"], unit="ms", utc=True
    )

    df = df.set_index("Open time")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Exclude the currently forming candle.
    if len(df) > 1:
        df = df.iloc[:-1]

    return df.tail(int(limit))


def analyze(symbol, interval, lookback, tolerance):
    df = get_klines(symbol, interval, int(lookback) + 1)

    if len(df) < int(lookback) + 1:
        return None, df, "Not enough completed candles"

    df = df.copy()
    n = int(lookback)

    # Previous N candles, excluding the current completed candle.
    df["PriorLow"] = df["Low"].shift(1).rolling(n).min()
    df["PriorHigh"] = df["High"].shift(1).rolling(n).max()

    df["BUY"] = (
        (df["Low"] <= df["PriorLow"] * (1 + tolerance))
        & (df["Close"] > df["Open"])
    )

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
        "Candles": len(df),
        "Time": str(df.index[-1]),
    }

    return result, df, ""


def render_chart(name, df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=name
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["PriorLow"],
        name="Prior 400-candle low",
        line=dict(color="red", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["PriorHigh"],
        name="Prior 400-candle high",
        line=dict(color="green", width=1)
    ))

    buys = df[df["BUY"]]
    sells = df[df["SELL"]]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Low"],
        mode="markers", name="BUY",
        marker=dict(color="lime", size=10, symbol="triangle-up")
    ))

    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["High"],
        mode="markers", name="SELL",
        marker=dict(color="red", size=10, symbol="triangle-down")
    ))

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        title=f"{name} — {interval_label}",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


mode = st.radio(
    "Mode",
    ["Single asset", "Scan all supported crypto"],
    horizontal=True
)

if mode == "Single asset":
    available_assets = [
        name for name, symbol in ASSETS.items()
        if symbol is not None
    ]

    selected = st.selectbox("Select asset", available_assets)

    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Downloading market candles..."):
                result, df, error = analyze(
                    ASSETS[selected], interval, lookback, tolerance
                )

            if error:
                st.error(error)
            elif result:
                a, b, c = st.columns(3)
                a.metric("Price", f"{result['Price']:.8g}")
                b.metric("Signal", result["Signal"])
                c.metric("Candles", result["Candles"])

                if result["Signal"] == "BUY":
                    st.success("Potential BUY reversal signal")
                elif result["Signal"] == "SELL":
                    st.error("Potential SELL reversal signal")
                else:
                    st.info("WAIT: no qualifying extreme reversal")

                st.caption(f"Last completed candle: {result['Time']}")
                render_chart(selected, df)
                st.dataframe(df.tail(30), use_container_width=True)

        except Exception as exc:
            st.error(
                "Market data request failed. Check network access, "
                "symbol availability, or API rate limits."
            )
            st.caption(str(exc))


else:
    crypto_assets = {
        name: symbol
        for name, symbol in ASSETS.items()
        if symbol is not None
    }

    st.caption(
        "Only symbols configured for Binance are scanned. "
        "These are external-market proxies, not confirmed Binomo OTC feeds."
    )

    if st.button("Start scan", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        total = len(crypto_assets)

        for i, (name, symbol) in enumerate(crypto_assets.items(), start=1):
            try:
                result, df, error = analyze(
                    symbol, interval, lookback, tolerance
                )

                if error:
                    results.append({
                        "Asset": name,
                        "Symbol": symbol,
                        "Price": None,
                        "Signal": "No data",
                        "Candles": len(df),
                        "Error": error
                    })
                else:
                    results.append({
                        "Asset": name,
                        "Symbol": symbol,
                        "Price": result["Price"],
                        "Signal": result["Signal"],
                        "Candles": result["Candles"],
                        "Low": result["Low"],
                        "High": result["High"],
                        "Time": result["Time"],
                        "Error": ""
                    })

            except Exception as exc:
                results.append({
                    "Asset": name,
                    "Symbol": symbol,
                    "Price": None,
                    "Signal": "Error",
                    "Candles": 0,
                    "Error": str(exc)[:150]
                })

            progress.progress(i / total)
            status.write(f"Scanned {i}/{total}: {name}")

            # Avoid sending too many requests too quickly.
            time.sleep(0.15)

        table = pd.DataFrame(results)
        st.session_state["results"] = table

    if "results" in st.session_state:
        table = st.session_state["results"]

        filter_choice = st.selectbox(
            "Filter",
            ["All", "BUY", "SELL", "WAIT", "Unavailable"]
        )

        if filter_choice in ["BUY", "SELL", "WAIT"]:
            shown = table[table["Signal"] == filter_choice]
        elif filter_choice == "Unavailable":
            shown = table[
                table["Signal"].isin(["No data", "Error"])
            ]
        else:
            shown = table

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True
        )

        buys = int((table["Signal"] == "BUY").sum())
        sells = int((table["Signal"] == "SELL").sum())
        waits = int((table["Signal"] == "WAIT").sum())

        a, b, c = st.columns(3)
        a.metric("BUY", buys)
        b.metric("SELL", sells)
        c.metric("WAIT", waits)

        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="extreme_scanner.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "No order execution. Signals are analytical only and do not guarantee "
    "a profitable trade. External market prices may differ from Binomo."
)
