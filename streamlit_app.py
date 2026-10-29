
 //@version=6
indicator("Binomo 400 Candle Extreme Signals", overlay=true, max_labels_count=500)

lookback = input.int(400, "Lookback Candles", minval=10)
showLevels = input.bool(true, "Show High/Low Levels")

lowest400 = ta.lowest(low, lookback)
highest400 = ta.highest(high, lookback)

// Önceki 400 mumun seviyelerini kullan
previousLow = ta.lowest(low[1], lookback)
previousHigh = ta.highest(high[1], lookback)

// Yeni dip = SELL, yeni tepe = BUY
sellSignal = barstate.isconfirmed and low < previousLow
buySignal = barstate.isconfirmed and high > previousHigh

plot(showLevels ? lowest400 : na, "400 Candle Low", color=color.red, linewidth=2)
plot(showLevels ? highest400 : na, "400 Candle High", color=color.lime, linewidth=2)

plotshape(sellSignal, title="SELL", text="SELL",
     style=shape.labeldown, location=location.abovebar,
     color=color.red, textcolor=color.white, size=size.small)

plotshape(buySignal, title="BUY", text="BUY",
     style=shape.labelup, location=location.belowbar,
     color=color.lime, textcolor=color.black, size=size.small)

alertcondition(sellSignal, title="SELL Alert", message="SELL: New 400-candle low")
alertcondition(buySignal, title="BUY Alert", message="BUY: New 400-candle high")
