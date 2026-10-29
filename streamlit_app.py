
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="400 Candle Extreme Scanner", layout="wide")
st.title("Binomo 400 Candle Extreme Scanner")
st.caption("BUY: 400 mumluk tepe kırılımı | SELL: 400 mumluk dip kırılımı")

assets = {
    "EUR/USD": "EURUSD=X",
    "AUD/USD": "AUDUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CAD": "CAD=X",
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "NZD/USD": "NZDUSD=X",
    "AUD/JPY": "AUDJPY=X",
    "CAD/JPY": "CADJPY=X",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Chainlink": "LINK-USD",
    "Bitcoin Cash": "BCH-USD",
    "NASDAQ 100": "^NDX",
    "Dow Jones IA": "^DJI",
    "Google": "GOOGL",
    "VOO": "VOO",
    "Russell 2000 ETF": "IWM",
    "Nasdaq 100 ETF": "QQQ",
}

asset = st.selectbox("Asset", list(assets.keys()))
interval = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"])
lookback = st.number_input("Lookback candles", 10, 400, 400)

@st.cache_data(ttl=60)
def load_data(symbol, timeframe):
    period = "60d" if timeframe in ["5m", "15m", "30m"] else (
        "730d" if timeframe == "1h" else "max"
    )
    data = yf.download(
        symbol,
        period=period,
        interval=timeframe,
        progress=False,
        auto_adjust=True
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(subset=["High", "Low", "Close"])

data = load_data(assets[asset], interval)

if data.empty:
    st.error("Veri alınamadı. Başka bir varlık veya zaman dilimi deneyin.")
    st.stop()

data = data.copy()
data["High400"] = data["High"].rolling(int(lookback)).max()
data["Low400"] = data["Low"].rolling(int(lookback)).min()

# Sinyalleri önceki 400 mumun seviyesine göre hesapla
data["PreviousHigh"] = data["High"].shift(1).rolling(int(lookback)).max()
data["PreviousLow"] = data["Low"].shift(1).rolling(int(lookback)).min()

data["Signal"] = ""
data.loc[data["High"] > data["PreviousHigh"], "Signal"] = "BUY"
data.loc[data["Low"] < data["PreviousLow"], "Signal"] = "SELL"

valid = data.dropna(subset=["PreviousHigh", "PreviousLow"])
if valid.empty:
    st.warning("400 mumluk analiz için yeterli veri yok.")
    st.stop()

latest = valid.iloc[-1]
buy_count = int((valid["Signal"] == "BUY").sum())
sell_count = int((valid["Signal"] == "SELL").sum())

c1, c2, c3 = st.columns(3)
c1.metric("Current price", f"{latest['Close']:.6g}")
c2.metric("BUY signals", buy_count)
c3.metric("SELL signals", sell_count)

last_signal = latest["Signal"]
if last_signal == "BUY":
    st.success("BUY signal: price broke above the previous lookback high.")
elif last_signal == "SELL":
    st.error("SELL signal: price broke below the previous lookback low.")
else:
    st.info("WAIT: no new extreme breakout on the latest candle.")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name=asset
))
fig.add_trace(go.Scatter(
    x=data.index, y=data["High400"],
    name="400-candle high", line=dict(color="green")
))
fig.add_trace(go.Scatter(
    x=data.index, y=data["Low400"],
    name="400-candle low", line=dict(color="red")
))

signals = data[data["Signal"] != ""]
for signal_name, color, symbol in [
    ("BUY", "green", "triangle-up"),
    ("SELL", "red", "triangle-down"),
]:
    points = signals[signals["Signal"] == signal_name]
    fig.add_trace(go.Scatter(
        x=points.index,
        y=points["Close"],
        mode="markers",
        name=signal_name,
        marker=dict(color=color, size=10, symbol=symbol)
    ))

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    title=f"{asset} — {interval} — {lookback} candles"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent signals")
st.dataframe(
    signals[["Close", "High", "Low", "Signal"]].tail(30),
    use_container_width=True
)
