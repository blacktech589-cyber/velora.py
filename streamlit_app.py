def fetch_binance_candles(symbol: str, limit: int = MAX_CANDLES) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper().strip(), "interval": "1m", "limit": min(limit, 1000)}
    # Add requests call and parsing here...
    return pd.DataFrame() # Placeholder

def fetch_market_candles(source: str, asset: str, endpoint: str = "") -> tuple[pd.DataFrame, str]:
    source = source.lower()
    asset_key = asset.upper().strip()
    
    if source == "binomo auto":
        return fetch_binomo_candles(asset, endpoint, limit=MAX_CANDLES), "Binomo Auto 1m"
    if source == "binomo/http":
        return fetch_binomo_candles(asset, endpoint, limit=MAX_CANDLES), "Binomo/HTTP 1m"
    if source == "binance":
        symbol = BINANCE_SYMBOLS.get(asset_key, asset_key.replace("/", ""))
        return fetch_binance_candles(symbol), f"Binance {symbol} 1m"
        
    source_key = source.lower()
    if source_key in {"binomo auto", "binomo/http"}:
        live = fetch_binomo_candles(asset, endpoint, limit=MAX_CANDLES)
        merged = pd.concat([existing, live], ignore_index=True).drop_duplicates(subset=["time"], keep="last")
        return normalize_candles(merged, max_rows=LONG_TRAINING_CANDLES + MAX_CANDLES), "Binomo live"
    if source_key.startswith("yahoo"):
        symbol = YAHOO_SYMBOLS.get(asset.upper().strip(), asset.strip())
        live = fetch_yahoo_candles(symbol)
        # Add return statement here...
        
    return pd.DataFrame(), "Unknown" # Placeholder

def build_excel_from_history(history: list[dict], strategies: Optional[pd.DataFrame] = None, health: Optional[dict] = None) -> bool:
    """Panel gecmisinden kullanici dosyasi istemeden Excel uretir."""
    if not history:
        return False
    signals = pd.DataFrame(history)
    with pd.ExcelWriter(EXPORT_FILE, engine="openpyxl") as writer:
        signals.to_excel(writer, sheet_name="Signals", index=False)
        if strategies is not None and not strategies.empty:
            strategies.to_excel(writer, sheet_name="Strategies", index=False)
        if health is not None:
            pd.DataFrame([health]).to_excel(writer, sheet_name="Health", index=False)
    audit_event("excel_built_from_history", {"file": str(EXPORT_FILE), "rows": len(history)})
    return True

def render_signal_badge(signal: str) -> None:
    if signal == "BUY":
        st.success("BUY")
    # Add other signals...

def render_ui():
    st.markdown(f"<div class='hero-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Binomo verisi oncelikli, 700 ozellikli deep ensemble, EMA15/RSI/ATR, haber, Telegram ve strateji backtest paneli.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Ayarlar")
        asset = st.text_input("Varlik adi", value="EUR/USD")
        source = st.selectbox("Veri kaynagi", ["Binomo Auto", "Binomo/HTTP", "Yahoo 4Y", "Yahoo 1Y", "Yahoo", "Binance 4Y", "Binance 1Y", "Binance", "CSV"], index=0)
        min_conf = st.slider("Minimum guven", 50, 95, DEFAULT_MIN_CONF)
        max_atr = st.number_input("Maks ATR orani", min_value=0.0001, max_value=0.2, value=DEFAULT_MAX_ATR, step=0.0005, format="%.4f")
        endpoint = st.text_input("Binomo mum endpoint", value=BINOMO_ENDPOINT)
        uploaded = st.file_uploader("Opsiyonel CSV mum verisi", type=["csv"])
        st.divider()
        news_api_key = st.text_input("NewsAPI key", value=NEWS_API_KEY, type="password")
        use_news = st.toggle("Haber cek", value=False)
        load_btn = st.button("Veriyi Yukle / Yenile", use_container_width=True)
        train_btn = st.button("Modeli Egit", use_container_width=True)
        scan_btn = st.button("Sinyal Uret", use_container_width=True)
        excel_btn = st.button("Exceli Otomatik Uret", use_container_width=True)

    if uploaded is not None and load_btn and source == "CSV":
        st.session_state.candles = normalize_candles(pd.read_csv(uploaded))
        
    if train_btn:
        msg = st.session_state.model.train(st.session_state.candles)
        st.info(msg)

    if excel_btn:
        strategy_df = pd.DataFrame()
        candles = st.session_state.get("candles", pd.DataFrame())
        if not candles.empty and len(candles) >= 40:
            strategy_state = backtest_strategy_weights(candles)
            strategy_df = pd.DataFrame(
                [
                    {
                        "Strategy": name,
                        "Weight": round(float(weight), 4),
                        "Latest_Vote": round(float(strategy_state.get("latest_votes", {}).get(name, 0)), 2),
                    }
                    for name, weight in sorted(strategy_state.get("weights", {}).items(), key=lambda x: x[1], reverse=True)
                ]
            )
        ok = build_excel_from_history(
            st.session_state.history,
            strategy_df,
            health_report(st.session_state.model, candles, st.session_state.data_source),
        )
        if ok:
            st.success(f"Excel uretildi: {EXPORT_FILE}") 
        else:
            st.warning("Excel icin once en az bir sinyal uret.")

    should_auto_scan = auto and (time.time() - st.session_state.last_scan >= SCAN_SECONDS)
    if scan_btn or should_auto_scan:
        st.session_state.last_scan = time.time()
        
    st.subheader("Sinyal Gecmisi")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if EXPORT_FILE.exists():
            with EXPORT_FILE.open("rb") as f:
                st.download_button(
                    "Excel indir",
                    data=f,
                    file_name=EXPORT_FILE.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    else:
        st.info("Gecmis bos.")
