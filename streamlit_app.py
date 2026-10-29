
import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import time

st.set_page_config(page_title="400 Candle Lowest Scanner", layout="wide")

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


def get_candles(symbol, interval, limit=400):
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit + 2, 1000)
    })

    request = urllib.request.Request(
        API_URL + "?" + params,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        raw = json.loads(response.read().decode("utf-8"))

    if not isinstance(raw, list) or len(raw) == 0:
        return pd.DataFrame()

    columns = [
        "OpenTime", "Open", "High", "Low", "Close",
        "Volume", "CloseTime", "QuoteVolume", "Trades",
        "BuyBase", "BuyQuote", "Ignore"
    ]

    df = pd.DataFrame(raw, columns=columns)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Time"] = pd.to_datetime(
        df["OpenTime"], unit="ms", utc=True
    )

    # Son mum henüz tamamlanmamış olabilir; onu çıkar.
    df = df.iloc[:-1].copy()
    df = df.dropna(subset=["Low", "High", "Close"])
    df = df.set_index("Time")

    return df.tail(limit)


def find_lowest(name, symbol, interval, lookback):
    if symbol is None:
        return {
            "Asset": name,
            "Lowest price": None,
            "Lowest candle time": "-",
            "Latest price": None,
            "Distance from low (%)": None,
            "Status": "External data unavailable"
        }

    try:
        df = get_candles(symbol, interval, lookback)

        if df.empty:
            raise ValueError("No candles returned")

        lowest_time = df["Low"].idxmin()
        lowest_price = float(df.loc[lowest_time, "Low"])
        latest_price = float(df["Close"].iloc[-1])

        distance = (
            (latest_price - lowest_price) / lowest_price * 100
            if lowest_price else None
        )

        return {
            "Asset": name,
            "Lowest price": lowest_price,
            "Lowest candle time": str(lowest_time),
            "Latest price": latest_price,
            "Distance from low (%)": round(distance, 3)
                if distance is not None else None,
            "Status": f"Scanned {len(df)} completed candles"
        }

    except Exception as e:
        return {
            "Asset": name,
            "Lowest price": None,
            "Lowest candle time": "-",
            "Latest price": None,
            "Distance from low (%)": None,
            "Status": "Data error: " + str(e)[:100]
        }


st.title("400 Candle Lowest Price Scanner")
st.write(
    "Her varlığın seçilen zaman dilimindeki son 400 tamamlanmış "
    "mumunun en düşük fiyatını ayrı ayrı hesaplar."
)

st.warning(
    "OTC fiyatları Binomo'ya özgüdür. Binance verisi Binomo OTC "
    "fiyatıyla aynı olmak zorunda değildir. Verisi olmayan varlıklar "
    "için tahmini fiyat üretilmez."
)

c1, c2 = st.columns(2)

with c1:
    interval_label = st.selectbox(
        "Timeframe", list(INTERVALS.keys()), index=2
    )

with c2:
    lookback = st.selectbox(
        "Number of candles", [100, 200, 300, 400], index=3
    )

interval = INTERVALS[interval_label]

mode = st.radio(
    "Scan mode",
    ["Scan one asset", "Scan all assets one by one"],
    horizontal=True
)

if mode == "Scan one asset":
    name = st.selectbox("Select asset", list(ASSETS.keys()))

    if st.button("Find lowest price", type="primary"):
        with st.spinner(f"Scanning {name}..."):
            result = find_lowest(
                name, ASSETS[name], interval, int(lookback)
            )

        if result["Lowest price"] is None:
            st.error(
                f"{name}: gerçek fiyat verisi alınamadı. "
                f"{result['Status']}"
            )
        else:
            a, b, c = st.columns(3)
            a.metric("Lowest price", f"{result['Lowest price']:.8g}")
            b.metric("Latest price", f"{result['Latest price']:.8g}")
            c.metric(
                "Above lowest (%)",
                f"{result['Distance from low (%)']:.3f}%"
            )

            st.write("Lowest candle time:", result["Lowest candle time"])
            st.write("Data status:", result["Status"])

            st.subheader("Last completed candles")
            df = get_candles(
                ASSETS[name], interval, int(lookback)
            )

            st.line_chart(df[["Low", "High", "Close"]])
            st.dataframe(df.tail(30), use_container_width=True)

else:
    if st.button("Start scanning all assets", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        total = len(ASSETS)

        for i, (name, symbol) in enumerate(
            ASSETS.items(), start=1
        ):
            result = find_lowest(
                name, symbol, interval, int(lookback)
            )
            results.append(result)

            progress.progress(i / total)
            status.write(f"{i}/{total} — {name}")

            time.sleep(0.1)

        table = pd.DataFrame(results)
        st.session_state["lowest_results"] = table

    if "lowest_results" in st.session_state:
        table = st.session_state["lowest_results"]

        st.subheader("Each asset's lowest price")
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        valid = table.dropna(subset=["Lowest price"])

        if not valid.empty:
            st.subheader("Lowest observed prices — comparison")

            st.bar_chart(
                valid.set_index("Asset")[["Lowest price"]]
            )

            st.download_button(
                "Download results CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="lowest_prices.csv",
                mime="text/csv"
            )

st.caption(
    "The lowest price is the minimum Low value among the retrieved "
    "completed candles. It is not a guarantee that the market cannot fall."
)
