from __future__ import annotations
import os, sqlite3, tempfile, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import numpy as np
import pandas as pd
import requests
import streamlit as st

TITLE = "Velora — EMA 15 + Bollinger Bands"
TARGET = 100_000
LIMIT = 1_000
TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 VeloraScanner/3.0", "Accept": "application/json"}
BINANCE = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"]
ASSETS = [
    {"name":"Bitcoin","symbol":"BTCUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Ethereum","symbol":"ETHUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Solana","symbol":"SOLUSDT","category":"Crypto","kind":"crypto"},
    {"name":"XRP","symbol":"XRPUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Dogecoin","symbol":"DOGEUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Cardano","symbol":"ADAUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Avalanche","symbol":"AVAXUSDT","category":"Crypto","kind":"crypto"},
    {"name":"Chainlink","symbol":"LINKUSDT","category":"Crypto","kind":"crypto"},
    {"name":"EUR/USD","symbol":"EURUSD=X","category":"Forex","kind":"yahoo"},
    {"name":"GBP/USD","symbol":"GBPUSD=X","category":"Forex","kind":"yahoo"},
    {"name":"USD/JPY","symbol":"JPY=X","category":"Forex","kind":"yahoo"},
    {"name":"AUD/USD","symbol":"AUDUSD=X","category":"Forex","kind":"yahoo"},
    {"name":"USD/CAD","symbol":"CAD=X","category":"Forex","kind":"yahoo"},
    {"name":"USD/CHF","symbol":"CHF=X","category":"Forex","kind":"yahoo"},
    {"name":"NZD/USD","symbol":"NZDUSD=X","category":"Forex","kind":"yahoo"},
    {"name":"EUR/GBP","symbol":"EURGBP=X","category":"Forex","kind":"yahoo"},
    {"name":"Gold futures","symbol":"GC=F","category":"Other","kind":"yahoo"},
    {"name":"Apple","symbol":"AAPL","category":"Other","kind":"yahoo"},
    {"name":"Microsoft","symbol":"MSFT","category":"Other","kind":"yahoo"},
    {"name":"NVIDIA","symbol":"NVDA","category":"Other","kind":"yahoo"},
]
FALLBACK_SYMBOLS = {"BTCUSDT":"BTCUSDT","ETHUSDT":"ETHUSDT","SOLUSDT":"SOLUSDT","XRPUSDT":"XRPUSDT","DOGEUSDT":"DOGEUSDT","ADAUSDT":"ADAUSDT","AVAXUSDT":"AVAXUSDT","LINKUSDT":"LINKUSDT"}
COINBASE = {"BTCUSDT":"BTC-USD","ETHUSDT":"ETH-USD","SOLUSDT":"SOL-USD","XRPUSDT":"XRP-USD","DOGEUSDT":"DOGE-USD","ADAUSDT":"ADA-USD","AVAXUSDT":"AVAX-USD","LINKUSDT":"LINK-USD"}

def now_utc(): return pd.Timestamp.now(tz="UTC")
def minute_now(): return now_utc().floor("min")
def get(url, params=None): return requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)

def data_dir():
    candidates = [Path(os.environ["VELORA_DATA_DIR"])] if os.environ.get("VELORA_DATA_DIR") else [Path.cwd(), Path("/tmp/velora_data")]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True); test=p/".write_test"; test.write_text("ok"); test.unlink(); return p
        except Exception: pass
    p=Path(tempfile.gettempdir())/"velora_data"; p.mkdir(parents=True,exist_ok=True); return p

DATA_DIR=data_dir(); DB=DATA_DIR/"velora_history.sqlite3"; XLSX=DATA_DIR/"velora_scan_history.xlsx"

def init_db():
    """Create the current schema and migrate databases made by older Velora versions.

    Earlier app versions used `candle_time`; the current app uses `time`.
    CREATE TABLE IF NOT EXISTS does not alter an existing table, so explicitly
    migrate that old schema before reading or writing candles.
    """
    with sqlite3.connect(DB, timeout=30) as c:
        existing = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='candles'"
        ).fetchone() is not None
        required = {"symbol", "time", "open", "high", "low", "close", "volume"}
        columns = (
            {row[1] for row in c.execute("PRAGMA table_info(candles)").fetchall()}
            if existing else set()
        )

        if existing and not required.issubset(columns):
            # Remove the old index before renaming, since SQLite keeps index names
            # attached to the renamed legacy table.
            c.execute("DROP INDEX IF EXISTS idx_symbol_time")
            c.execute("ALTER TABLE candles RENAME TO candles_legacy")
            legacy_columns = {
                row[1] for row in c.execute("PRAGMA table_info(candles_legacy)").fetchall()
            }
            c.execute(
                "CREATE TABLE candles ("
                "symbol TEXT NOT NULL, time TEXT NOT NULL, "
                "open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, "
                "close REAL NOT NULL, volume REAL DEFAULT 0, "
                "PRIMARY KEY(symbol,time))"
            )

            time_column = next(
                (name for name in ("time", "candle_time", "timestamp") if name in legacy_columns),
                None,
            )
            needed_ohlc = {"symbol", "open", "high", "low", "close"}
            if time_column and needed_ohlc.issubset(legacy_columns):
                volume_expr = "volume" if "volume" in legacy_columns else "0"
                c.execute(
                    "INSERT OR REPLACE INTO candles "
                    "(symbol,time,open,high,low,close,volume) "
                    f"SELECT symbol, {time_column}, open, high, low, close, {volume_expr} "
                    "FROM candles_legacy "
                    "WHERE symbol IS NOT NULL AND " + time_column + " IS NOT NULL"
                )
            c.execute("DROP TABLE candles_legacy")

        elif not existing:
            c.execute(
                "CREATE TABLE candles ("
                "symbol TEXT NOT NULL, time TEXT NOT NULL, "
                "open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, "
                "close REAL NOT NULL, volume REAL DEFAULT 0, "
                "PRIMARY KEY(symbol,time))"
            )

        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol,time)"
        )
        c.commit()

def normalize(df):
    cols=["time","open","high","low","close","volume"]
    if df is None or df.empty: return pd.DataFrame(columns=cols)
    d=df.copy()
    for k in cols:
        if k not in d: d[k]=0.0 if k=="volume" else np.nan
    d["time"]=pd.to_datetime(d["time"],utc=True,errors="coerce")
    for k in cols[1:]: d[k]=pd.to_numeric(d[k],errors="coerce")
    d["volume"]=d["volume"].fillna(0)
    d=d.dropna(subset=["time","open","high","low","close"])
    d=d[(d.open>0)&(d.high>0)&(d.low>0)&(d.close>0)&(d.time<minute_now())]
    return d.drop_duplicates("time",keep="last").sort_values("time").reset_index(drop=True)[cols]

def save_cache(symbol, df):
    d=normalize(df)
    if d.empty: return
    rows=[(symbol,pd.Timestamp(r.time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume or 0)) for r in d.itertuples(index=False)]
    init_db()
    with sqlite3.connect(DB,timeout=30) as c: c.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)",rows)

def load_cache(symbol):
    init_db()
    with sqlite3.connect(DB,timeout=30) as c: d=pd.read_sql_query("SELECT time,open,high,low,close,volume FROM candles WHERE symbol=? ORDER BY time",c,params=(symbol,))
    return normalize(d)

def parse_binance(rows):
    d=pd.DataFrame(rows,columns=["stamp","open","high","low","close","volume","close_time","qv","trades","tb","tq","ignore"])
    d["time"]=pd.to_datetime(d.stamp,unit="ms",utc=True,errors="coerce")
    return normalize(d[["time","open","high","low","close","volume"]])

def fetch_binance(symbol,target):
    errs=[]
    for base in BINANCE:
        try:
            pages=[]; end=None
            while True:
                p={"symbol":symbol,"interval":"1m","limit":min(1000,max(1,target-len(pd.concat(pages,ignore_index=True)) if pages else min(1000,target)))}
                if end is not None: p["endTime"]=end
                r=get(base+"/api/v3/klines",p)
                if r.status_code in (403,451): raise RuntimeError(f"HTTP {r.status_code}: endpoint blocked")
                if r.status_code==429: raise RuntimeError("HTTP 429: rate limited")
                r.raise_for_status(); rows=r.json()
                if not isinstance(rows,list) or not rows: raise RuntimeError(f"No candles returned: {str(rows)[:150]}")
                page=parse_binance(rows); pages.append(page); d=normalize(pd.concat(pages,ignore_index=True))
                if len(d)>=target or target<=1000: break
                oldest=d.time.min(); next_end=int(oldest.timestamp()*1000)-1
                if end is not None and next_end>=end: break
                end=next_end; time.sleep(.05)
                if len(pages)>=110: break
            d=normalize(pd.concat(pages,ignore_index=True)).tail(target).reset_index(drop=True)
            return d,base,None
        except Exception as e: errs.append(f"{base}: {type(e).__name__}: {e}")
    raise RuntimeError("Binance failed: "+" | ".join(errs))

def fetch_bybit(symbol,target):
    if not symbol.endswith("USDT"): raise RuntimeError("Unsupported symbol")
    pages=[]; end=None; warning=None
    while True:
        p={"category":"spot","symbol":symbol,"interval":"1","limit":min(1000,max(40,target-len(pd.concat(pages,ignore_index=True)) if pages else min(target,1000)))}
        if end is not None: p["end"]=end
        r=get("https://api.bybit.com/v5/market/kline",p); r.raise_for_status(); j=r.json()
        if j.get("retCode")!=0: raise RuntimeError(f"Bybit {j.get('retCode')}: {j.get('retMsg')}")
        rows=(j.get("result") or {}).get("list") or []
        if not rows: break
        d=pd.DataFrame(rows,columns=["stamp","open","high","low","close","volume","turnover"])
        d["time"]=pd.to_datetime(pd.to_numeric(d.stamp,errors="coerce"),unit="ms",utc=True)
        pages.append(normalize(d[["time","open","high","low","close","volume"]]))
        combined=normalize(pd.concat(pages,ignore_index=True))
        if len(combined)>=target or target<=1000: break
        next_end=int(combined.time.min().timestamp()*1000)-1
        if end is not None and next_end>=end: warning="Bybit history paging stopped."; break
        end=next_end; time.sleep(.05)
        if len(pages)>=110: warning="Bybit sync limited to 110 pages."; break
    if not pages: raise RuntimeError("Bybit returned no candles")
    return normalize(pd.concat(pages,ignore_index=True)).tail(target).reset_index(drop=True),"Bybit public API",warning

def fetch_okx(symbol,target):
    inst=symbol[:-4]+"-USDT"; pages=[]; after=None; warning=None
    while True:
        p={"instId":inst,"bar":"1m","limit":str(min(300,max(40,target-len(pd.concat(pages,ignore_index=True)) if pages else min(target,300))))}
        if after: p["after"]=after
        r=get("https://www.okx.com/api/v5/market/candles",p); r.raise_for_status(); j=r.json()
        if str(j.get("code"))!="0": raise RuntimeError(f"OKX {j.get('code')}: {j.get('msg')}")
        rows=j.get("data") or []
        if not rows: break
        d=pd.DataFrame(rows,columns=["stamp","open","high","low","close","volume","volccy","volquote","confirm"])
        d["time"]=pd.to_datetime(pd.to_numeric(d.stamp,errors="coerce"),unit="ms",utc=True)
        pages.append(normalize(d[["time","open","high","low","close","volume"]]))
        combined=normalize(pd.concat(pages,ignore_index=True))
        if len(combined)>=target or target<=300: break
        next_after=str(int(combined.time.min().timestamp()*1000))
        if after==next_after: warning="OKX history paging stopped."; break
        after=next_after; time.sleep(.05)
        if len(pages)>=340: warning="OKX sync limited to 340 pages."; break
    if not pages: raise RuntimeError("OKX returned no candles")
    return normalize(pd.concat(pages,ignore_index=True)).tail(target).reset_index(drop=True),"OKX public API",warning

def fetch_gate(symbol,target):
    pair=symbol[:-4]+"_USDT"
    r=get("https://api.gateio.ws/api/v4/spot/candlesticks",{"currency_pair":pair,"interval":"1m","limit":min(1000,target)}); r.raise_for_status(); rows=r.json()
    if not isinstance(rows,list) or not rows: raise RuntimeError("Gate.io returned no candles")
    d=pd.DataFrame(rows,columns=["stamp","volume","close","high","low","open","quote"])
    d["time"]=pd.to_datetime(pd.to_numeric(d.stamp,errors="coerce"),unit="s",utc=True)
    return normalize(d[["time","open","high","low","close","volume"]]),"Gate.io public API",None

def fetch_coinbase(symbol):
    product=COINBASE.get(symbol)
    if not product: raise RuntimeError("No Coinbase mapping for symbol")
    end=now_utc().floor("min"); start=end-pd.Timedelta(minutes=299)
    p={"granularity":60,"start":start.isoformat().replace("+00:00","Z"),"end":end.isoformat().replace("+00:00","Z")}
    r=get(f"https://api.exchange.coinbase.com/products/{product}/candles",p); r.raise_for_status(); rows=r.json()
    if not rows: raise RuntimeError("Coinbase returned no candles")
    d=pd.DataFrame(rows,columns=["stamp","low","high","open","close","volume"]); d["time"]=pd.to_datetime(d.stamp,unit="s",utc=True)
    return normalize(d[["time","open","high","low","close","volume"]]),"Coinbase public API",None

def fetch_yahoo(asset):
    symbol=asset["symbol"]
    if asset["kind"]=="crypto": symbol=COINBASE.get(symbol,symbol).replace("-","-")
    errors=[]
    for host in ["query1.finance.yahoo.com","query2.finance.yahoo.com"]:
        try:
            url=f"https://{host}/v8/finance/chart/{requests.utils.quote(symbol,safe='')}"
            r=get(url,{"range":"7d","interval":"1m","includePrePost":"false"}); r.raise_for_status(); j=r.json().get("chart",{})
            if j.get("error"): raise RuntimeError(str(j["error"]))
            items=j.get("result") or []
            if not items: raise RuntimeError("Yahoo returned no result")
            item=items[0]; q=(item.get("indicators",{}).get("quote") or [{}])[0]; stamps=item.get("timestamp") or []
            if not stamps: raise RuntimeError("Yahoo returned no timestamps")
            d=pd.DataFrame({"time":pd.to_datetime(stamps,unit="s",utc=True),"open":q.get("open",[]),"high":q.get("high",[]),"low":q.get("low",[]),"close":q.get("close",[]),"volume":q.get("volume",[])})
            return normalize(d),f"Yahoo Finance ({host})",None
        except Exception as e: errors.append(f"{host}: {type(e).__name__}: {e}")
    raise RuntimeError("Yahoo failed: "+" | ".join(errors))

def get_candles(asset,target):
    symbol=asset["symbol"]; errors=[]; warning=[]; recent=pd.DataFrame(); source=""
    if asset["kind"]=="crypto":
        fetchers=[("Binance",lambda:fetch_binance(symbol,target)),("Bybit",lambda:fetch_bybit(symbol,target)),("OKX",lambda:fetch_okx(symbol,target)),("Gate.io",lambda:fetch_gate(symbol,target)),("Coinbase",lambda:fetch_coinbase(symbol)),("Yahoo",lambda:fetch_yahoo(asset))]
        for name,fn in fetchers:
            try:
                recent,source,note=fn()
                if note: warning.append(note)
                if name!="Binance": warning.append(f"Fallback provider {name} used; quote may differ from Binance USDT spot.")
                break
            except Exception as e: errors.append(f"{name}: {type(e).__name__}: {e}")
    else:
        try: recent,source,note=fetch_yahoo(asset)
        except Exception as e: errors.append(f"Yahoo: {type(e).__name__}: {e}")
    recent=normalize(recent)
    if not recent.empty: save_cache(symbol,recent)
    try: cached=load_cache(symbol)
    except Exception as e: cached=pd.DataFrame(); errors.append(f"SQLite: {e}")
    parts=[x for x in [cached,recent] if x is not None and not x.empty]
    if not parts: raise RuntimeError("No live or cached candles. "+(" | ".join(errors) or "All providers returned empty data."))
    d=normalize(pd.concat(parts,ignore_index=True)).tail(TARGET).reset_index(drop=True)
    if len(d)<40: raise RuntimeError(f"Only {len(d)} completed candles; need at least 40. "+(" | ".join(errors) or ""))
    age=(now_utc()-d.time.iloc[-1]).total_seconds()/60
    if age>5: raise RuntimeError(f"Latest candle is {age:.1f} minutes old; no signal calculated. "+" | ".join(errors))
    return d,source,(" | ".join(warning+errors) if warning or errors else None)

def indicators(df):
    d=df.copy(); d["ema15"]=d.close.ewm(span=15,adjust=False,min_periods=15).mean(); d["bb_mid"]=d.close.rolling(20,min_periods=20).mean(); sd=d.close.rolling(20,min_periods=20).std(ddof=0); d["bb_upper"]=d.bb_mid+2*sd; d["bb_lower"]=d.bb_mid-2*sd; return d

def signal_for(df,lookback,mode):
    # Use the supplied indicator columns if they are already calculated. This
    # ensures the signal shown in the result table and the chart comes from the
    # exact same candle/indicator snapshot.
    d = df.copy() if {"ema15", "bb_mid", "bb_upper", "bb_lower"}.issubset(df.columns) else indicators(df)
    last=d.iloc[-1]; seg=d.tail(lookback)
    empty={"signal":"NO SIGNAL","reason":"Not enough indicator data","change_pct":np.nan,"trend":"Unknown","green_ratio":np.nan,"red_ratio":np.nan,"direction_up_ok":False,"direction_down_ok":False,"price_above_ema":False,"price_below_ema":False,"upper_band_ok":False,"lower_band_ok":False}
    if len(d)<max(40,lookback) or pd.isna(last.ema15) or pd.isna(last.bb_upper) or pd.isna(last.bb_lower): return empty
    first=float(seg.close.iloc[0]); price=float(last.close)
    if first<=0:return {**empty,"reason":"Invalid price"}
    change=(price-first)/first*100; green=float((seg.close>seg.open).mean()); red=float((seg.close<seg.open).mean())
    upper= float(last.high)>=float(last.bb_upper) if mode=="Touch or cross" else price>float(last.bb_upper)
    lower= float(last.low)<=float(last.bb_lower) if mode=="Touch or cross" else price<float(last.bb_lower)
    rising=change>0 and green>=.5; falling=change<0 and red>=.5; above=price>float(last.ema15); below=price<float(last.ema15)
    if rising and above and upper: sig="SELL"; reason=f"Last {lookback} candles rose {change:.3f}%; above EMA15 and upper Bollinger condition met."
    elif falling and below and lower: sig="BUY"; reason=f"Last {lookback} candles fell {change:.3f}%; below EMA15 and lower Bollinger condition met."
    else: sig="NO SIGNAL"; reason=f"Conditions incomplete: change={change:.3f}%, green={green:.0%}, red={red:.0%}, above EMA={above}, below EMA={below}, upper band={upper}, lower band={lower}."
    return {"signal":sig,"reason":reason,"change_pct":change,"trend":"Rising" if rising else "Falling" if falling else "Mixed/sideways","green_ratio":green,"red_ratio":red,"direction_up_ok":rising,"direction_down_ok":falling,"price_above_ema":above,"price_below_ema":below,"upper_band_ok":upper,"lower_band_ok":lower}

def scan(asset,lookback,mode,target):
    try:
        candles,source,warning=get_candles(asset,target); d=indicators(candles); sig=signal_for(d,lookback,mode); last=d.iloc[-1]
        return {**asset,**sig,"price":float(last.close),"ema15":float(last.ema15),"bb_upper":float(last.bb_upper),"bb_mid":float(last.bb_mid),"bb_lower":float(last.bb_lower),"count":len(d),"last_candle_utc":str(last.time),"source":source,"candles":d,"error":None,"warning":warning}
    except Exception as e:
        return {**asset,"signal":"NO SIGNAL","reason":"Data unavailable; no signal calculated.","change_pct":np.nan,"trend":"Unknown","green_ratio":np.nan,"red_ratio":np.nan,"price":np.nan,"ema15":np.nan,"bb_upper":np.nan,"bb_mid":np.nan,"bb_lower":np.nan,"count":0,"last_candle_utc":None,"source":"Unavailable","candles":None,"error":str(e),"warning":None}

def excel_row(r,scan_time):
    keys=["name","symbol","category","kind","signal","reason","trend","change_pct","green_ratio","red_ratio","price","ema15","bb_upper","bb_mid","bb_lower","count","last_candle_utc","source","warning","error"]
    out={"scan_time_utc":scan_time,**{k:r.get(k) for k in keys}}
    for k,v in out.items():
        if isinstance(v,(np.floating,float)) and not np.isfinite(v):out[k]=None
        elif isinstance(v,(np.integer,)):out[k]=int(v)
    return out

def save_excel(results,scan_time):
    rows=pd.DataFrame([excel_row(r,scan_time) for r in results]); errors=rows[rows.error.notna()].copy()
    summary=pd.DataFrame([{"scan_time_utc":scan_time,"assets_scanned":len(results),"data_ok":sum(not r.get("error") for r in results),"data_errors":sum(bool(r.get("error")) for r in results),"BUY":sum(r.get("signal")=="BUY" and not r.get("error") for r in results),"SELL":sum(r.get("signal")=="SELL" and not r.get("error") for r in results),"NO_SIGNAL":sum(r.get("signal")=="NO SIGNAL" and not r.get("error") for r in results)}])
    sheets={"Scan History":summary,"Scan Results":rows,"Data Errors":errors}
    if XLSX.exists():
        try:
            with pd.ExcelFile(XLSX) as book:
                for name,frame in list(sheets.items()):
                    if name in book.sheet_names:sheets[name]=pd.concat([pd.read_excel(book,sheet_name=name),frame],ignore_index=True).tail(1_000_000)
        except Exception:pass
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx",prefix="velora_",dir=DATA_DIR,delete=False) as f:tmp=f.name
        with pd.ExcelWriter(tmp,engine="openpyxl") as writer:
            for name,frame in sheets.items():
                frame.to_excel(writer,sheet_name=name,index=False); ws=writer.sheets[name]; ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
                for cells in ws.columns: ws.column_dimensions[cells[0].column_letter].width=min(55, max(12, max((len(str(c.value)) if c.value is not None else 0) for c in cells[:1000]) + 2))
        os.replace(tmp,XLSX)
    finally:
        if tmp and os.path.exists(tmp):os.remove(tmp)

st.set_page_config(page_title="Velora Scanner",page_icon="📊",layout="wide")
APP_VERSION = "2026-10-21-diagnostics-table-2"
if st.session_state.get("_velora_app_version") != APP_VERSION:
    for key in ("results", "scan_time", "save_msg"):
        st.session_state.pop(key, None)
    st.session_state["_velora_app_version"] = APP_VERSION

st.title("📊 Velora — EMA 15 + Bollinger Bands")
st.caption("1-minute market scanner | BUY / SELL signals only | No order execution")
st.warning("Public market data only. Crypto tries Binance, Bybit, OKX, Gate.io, Coinbase and Yahoo. Forex/stock uses Yahoo Finance. Data availability varies; Yahoo prices are not Binomo OTC quotes. This app never places trades.")
st.sidebar.header("Scanner settings")
category=st.sidebar.selectbox("Asset category",["All","Crypto","Forex","Other"])
lookback=st.sidebar.select_slider("Rise/fall candle count",options=[15,16,17],value=15)
mode=st.sidebar.radio("Bollinger confirmation",["Touch or cross","Close outside band"])
asset_limit=int(st.sidebar.number_input("Maximum assets per scan",min_value=1,max_value=len(ASSETS),value=10))
deep=st.sidebar.checkbox("Sync toward 100,000 candles (slower; crypto providers only)",value=False)
show_no=st.sidebar.checkbox("Show NO SIGNAL results",value=True)
assets=[a for a in ASSETS if category=="All" or a["category"]==category][:asset_limit]
target=TARGET if deep else LIMIT
st.markdown(f"**Strategy:** SELL after rising candles, at least half green, close above EMA15 and upper Bollinger condition. BUY after falling candles, at least half red, close below EMA15 and lower Bollinger condition. Only completed 1-minute candles are used. Otherwise NO SIGNAL.")
st.write(f"Assets selected: **{len(assets)}** · History target: **{target:,} candles per asset**")
st.caption(f"Excel automatically saved to `{XLSX}` after each scan. Download it to preserve a copy; hosted local files may not survive redeployment.")
# Session state may survive a Streamlit rerun/deployment while the app code
# changes. Older versions may have stored a DataFrame in `results`; evaluating
# a DataFrame with `if results` raises pandas ValueError. Reset any incompatible
# legacy value to a plain list before it is used.
if "results" not in st.session_state or not isinstance(st.session_state.get("results"), list):
    st.session_state["results"] = []
if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None

if st.button("🔍 START SCAN",type="primary",use_container_width=True):
    results=[]; bar=st.progress(0.0); status=st.empty()
    with ThreadPoolExecutor(max_workers=min(3,len(assets) or 1)) as pool:
        futures={pool.submit(scan,a,int(lookback),mode,target):a for a in assets}
        for i,f in enumerate(as_completed(futures),1):
            try:results.append(f.result())
            except Exception as e:results.append({**futures[f],"signal":"NO SIGNAL","reason":"Unexpected error","candles":None,"error":str(e)})
            bar.progress(i/max(1,len(futures)));status.write(f"Scanned {i}/{len(futures)} assets")
    order={a["symbol"]:i for i,a in enumerate(assets)};results.sort(key=lambda r:order.get(r.get("symbol"),999))
    stamp=now_utc().strftime("%Y-%m-%d %H:%M:%S UTC");st.session_state.results=results;st.session_state.scan_time=stamp
    try:save_excel(results,stamp);st.session_state.save_msg=f"Excel history saved automatically: {XLSX.name}"
    except Exception as e:st.session_state.save_msg=f"Excel save failed: {e}"
    bar.empty();status.empty()
# Revalidate state on every rerun. Do not use a pandas DataFrame as a Boolean.
results_value = st.session_state.get("results", [])
results = results_value if isinstance(results_value, list) else []
if not isinstance(results_value, list):
    st.session_state["results"] = []
stamp=st.session_state.get("scan_time")
if len(results) == 0:
    st.info("Click START SCAN to retrieve candles and calculate signals.")
    if XLSX.exists():st.download_button("📥 Download Excel history",XLSX.read_bytes(),"velora_scan_history.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.stop()
if hasattr(st.session_state,"save_msg"):
    (st.success if st.session_state.save_msg.startswith("Excel history saved") else st.error)(st.session_state.save_msg)
valid=[r for r in results if r.get("candles") is not None and not r.get("error")];errors=[r for r in results if r.get("error")];buys=[r for r in valid if r["signal"]=="BUY"];sells=[r for r in valid if r["signal"]=="SELL"]
st.caption(f"Last scan: {stamp}")
a,b,c,d=st.columns(4);a.metric("Assets with 100K candles",sum(r["count"]>=TARGET for r in valid));b.metric("BUY",len(buys));c.metric("SELL",len(sells));d.metric("Data errors",len(errors))
st.subheader("🚨 BUY / SELL signals")
signals=buys+sells
if len(signals) > 0:
    signal_table = pd.DataFrame([{
        "Asset": r.get("name"), "Symbol": r.get("symbol"),
        "Signal": r.get("signal"), "Price": r.get("price"),
        "Trend": r.get("trend"), "Change (%)": r.get("change_pct"),
        "EMA 15": r.get("ema15"), "BB upper": r.get("bb_upper"),
        "BB lower": r.get("bb_lower"), "Candles": r.get("count"),
        "Reason": r.get("reason")
    } for r in signals])
    # st.table renders a static table rather than the interactive canvas grid.
    st.table(signal_table)
else:
    st.info("No BUY or SELL setup meets all selected conditions. No signal is forced.")
st.subheader("📋 All scan results")
shown=valid if show_no else [r for r in valid if r["signal"]!="NO SIGNAL"]
if len(shown) > 0:
    def movement_status(r):
        if bool(r.get("direction_up_ok")):
            return "PASS — rising + green-candle rule"
        if bool(r.get("direction_down_ok")):
            return "PASS — falling + red-candle rule"
        return "FAIL — direction/candle-color rule"

    def ema_status(r):
        if bool(r.get("direction_up_ok")):
            return "PASS — above EMA 15" if bool(r.get("price_above_ema")) else "FAIL — needs above EMA 15"
        if bool(r.get("direction_down_ok")):
            return "PASS — below EMA 15" if bool(r.get("price_below_ema")) else "FAIL — needs below EMA 15"
        return f"Above={bool(r.get('price_above_ema'))}; below={bool(r.get('price_below_ema'))}"

    def band_status(r):
        if bool(r.get("direction_up_ok")):
            return "PASS — upper band" if bool(r.get("upper_band_ok")) else "FAIL — upper band not touched"
        if bool(r.get("direction_down_ok")):
            return "PASS — lower band" if bool(r.get("lower_band_ok")) else "FAIL — lower band not touched"
        return f"Upper={bool(r.get('upper_band_ok'))}; lower={bool(r.get('lower_band_ok'))}"

    all_table = pd.DataFrame([{
        "Asset": r.get("name"), "Symbol": r.get("symbol"),
        "Category": r.get("category"), "Signal": r.get("signal"),
        "Price": r.get("price"), "Candles": r.get("count"),
        "Direction check": movement_status(r),
        "EMA check": ema_status(r),
        "Band check": band_status(r),
        "Source": r.get("source"), "Reason": r.get("reason")
    } for r in shown])
    st.table(all_table)
else:
    st.info("No valid scan results. Review Data errors below.")
st.subheader("📈 Asset chart")
chartable=[r for r in valid if isinstance(r.get("candles"),pd.DataFrame) and not r["candles"].empty]
if len(chartable) > 0:
    chart_by_symbol = {r["symbol"]: r for r in chartable}
    chosen_symbol = st.selectbox(
        "Choose an asset",
        options=list(chart_by_symbol.keys()),
        format_func=lambda symbol: chart_by_symbol[symbol]["name"],
        key="velora_chart_symbol",
    )
    chosen = chart_by_symbol[chosen_symbol]
    chart=chosen["candles"].tail(500).set_index("time")
    st.write(f"**{chosen['signal']}** — {chosen['reason']} · {chosen['source']}")
    if chosen.get("warning"):st.warning(chosen["warning"])
    st.line_chart(chart[["close","ema15"]].rename(columns={"close":"Close","ema15":"EMA 15"}))
    st.line_chart(chart[["close","bb_upper","bb_mid","bb_lower"]].rename(columns={"close":"Close","bb_upper":"Upper band","bb_mid":"Middle band","bb_lower":"Lower band"}))
else:st.info("No chart available because no valid current candle data was retrieved.")
st.subheader("⚠️ Data errors — readable details")
if errors:
    st.error(f"{len(errors)} asset(s) could not be scanned. These errors are shown as plain text so they can be copied and diagnosed.")
    error_lines = []
    for result in errors:
        detail = str(result.get("error") or "No error text returned")
        error_lines.append(
            f"ASSET: {result.get('name')} | SYMBOL: {result.get('symbol')} | CATEGORY: {result.get('category')}\n{detail}"
        )
    readable_errors = "\n\n".join(error_lines)
    st.text_area(
        "Copy these provider error messages and share them for diagnosis",
        value=readable_errors,
        height=320,
    )
    for result in errors:
        with st.expander(f"{result.get('name')} ({result.get('symbol')})", expanded=False):
            st.code(str(result.get("error") or "No error text returned"), language="text")
    st.caption("If all providers fail with DNS, timeout, or connection errors, check whether the Streamlit hosting environment can reach external market-data APIs. A code change cannot create live candles when every provider is unreachable.")
else:
    st.success("No data errors in this scan.")
st.subheader("💾 Automatic Excel history")
if XLSX.exists():
    st.download_button("📥 Download Excel history (all scans)",XLSX.read_bytes(),"velora_scan_history.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    st.caption("Workbook sheets: Scan History, Scan Results, Data Errors.")
export=pd.DataFrame([excel_row(r,stamp or "Unknown") for r in results])
st.download_button("📄 Download current scan as CSV",export.to_csv(index=False).encode("utf-8-sig"),"velora_current_scan.csv","text/csv",use_container_width=True)
st.divider();st.caption("Analysis signals only; no trade execution. 100,000 candles are a target, not a guarantee. Save a local copy of the Excel workbook because hosted files may be erased on redeployment.")
