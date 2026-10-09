
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Monthly Crypto Bottom Scanner",
    page_icon="📉",
    layout="wide",
)

BINANCE = "https://api.binance.com"
GATE = "https://api.gateio.ws/api/v4"

INTERVAL = "15m"
INTERVAL_SECONDS = 900
BARS_30_DAYS = 30 * 24 * 4  # 2,880 completed candles
LOOKBACK_SECONDS = 30 * 24 * 60 * 60
REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": "MonthlyCryptoBottomScanner/1.0",
    "Accept": "application/json",
}

session = requests.Session()
session.headers.update(HEADERS)


# =========================================================
# HTTP
# =========================================================

def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# =========================================================
# BINANCE ACTIVE SPOT USDT MARKETS
# =========================================================

def get_binance_markets():
    data = get_json(f"{BINANCE}/api/v3/exchangeInfo")
    markets = []

    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        markets.append({
            "symbol": symbol,
            "market_id": symbol,
            "base": item.get("baseAsset", ""),
            "provider": "Binance",
        })

    if not markets:
        raise RuntimeError("Binance aktif Spot USDT marketi bulunamadı.")

    return markets


def get_binance_tickers():
    data = get_json(f"{BINANCE}/api/v3/ticker/24hr")
    tickers = {}

    for item in data:
        symbol = item.get("symbol")

        try:
            tickers[symbol] = {
                "price": float(item["lastPrice"]),
                "volume": float(item["quoteVolume"]),
                "change_24h": float(item["priceChangePercent"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


# =========================================================
# GATE.IO ACTIVE SPOT USDT MARKETS
# =========================================================

def get_gate_markets():
    data = get_json(f"{GATE}/spot/currency_pairs")
    markets = []

    for item in data:
        market_id = item.get("id", "")
        base = item.get("base", "")
        quote = item.get("quote", "")

        if not market_id or not base or quote != "USDT":
            continue

        status = str(item.get("trade_status", "")).lower()

        # Exclude unknown or non-tradable market states.
        if status != "tradable":
            continue

        if item.get("delisted") is True:
            continue

        markets.append({
            "symbol": f"{base}USDT",
            "market_id": market_id,
            "base": base,
            "provider": "Gate.io",
        })

    if not markets:
        raise RuntimeError("Gate.io aktif Spot USDT marketi bulunamadı.")

    return markets


def get_gate_tickers():
    data = get_json(f"{GATE}/spot/tickers")
    tickers = {}

    for item in data:
        market_id = item.get("currency_pair", "")

        if not market_id.endswith("_USDT"):
            continue

        try:
            tickers[market_id] = {
                "price": float(item["last"]),
                "volume": float(item["quote_volume"]),
                "change_24h": float(item["change_percentage"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return tickers


# =========================================================
# BINANCE: PAGINATED 30-DAY CANDLES
# =========================================================

def load_binance_candles(market_id):
    now = int(datetime.now(timezone.utc).timestamp())
    current_candle_start = (now // INTERVAL_SECONDS) * INTERVAL_SECONDS

    # Request extra history so that 2,880 completed candles remain
    # after removing the current, potentially incomplete candle.
    start = current_candle_start - (BARS_30_DAYS + 10) * INTERVAL_SECONDS
    end = current_candle_start

    rows = []
    cursor = start * 1000
    end_ms = end * 1000

    while cursor < end_ms:
        raw = get_json(
            f"{BINANCE}/api/v3/klines",
            params={
                "symbol": market_id,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not raw:
            break

        for candle in raw:
            try:
                rows.append({
                    "timestamp": int(candle[0]) // 1000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })
            except (ValueError, TypeError, IndexError):
                continue

        last_open_ms = int(raw[-1][0])
        next_cursor = last_open_ms + INTERVAL_SECONDS * 1000

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(raw) < 1000:
            break

        time.sleep(0.05)

    return clean_candles(rows)


# =========================================================
# GATE.IO: PAGINATED 30-DAY CANDLES
# =========================================================

def load_gate_candles(market_id):
    now = int(datetime.now(timezone.utc).timestamp())
    current_candle_start = (now // INTERVAL_SECONDS) * INTERVAL_SECONDS

    start = current_candle_start - (
        BARS_30_DAYS + 10
    ) * INTERVAL_SECONDS

    end = current_candle_start

    rows = []
    cursor = start

    while cursor < end:
        page_end = min(
            cursor + 999 * INTERVAL_SECONDS,
            end,
        )

        raw = get_json(
            f"{GATE}/spot/candlesticks",
            params={
                "currency_pair": market_id,
                "interval": INTERVAL,
                "from": cursor,
                "to": page_end,
                "limit": 1000,
            },
        )

        if not raw:
            break

        for candle in raw:
            if len(candle) < 6:
                continue

            try:
                rows.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[5]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "close": float(candle[2]),
                    "volume": float(candle[6])
                    if len(candle) > 6 else 0.0,
                })
            except (ValueError, TypeError, IndexError):
                continue

        cursor = page_end + INTERVAL_SECONDS
        time.sleep(0.05)

    return clean_candles(rows)


# =========================================================
# CLEAN AND VALIDATE CANDLES
# =========================================================

def clean_candles(rows):
    df = pd.DataFrame(rows)

    required = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]

    if df.empty:
        return pd.DataFrame(columns=required)

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Remove current incomplete candle.
    now = int(datetime.now(timezone.utc).timestamp())
    current_candle_start = (now // INTERVAL_SECONDS) * INTERVAL_SECONDS

    df = df[df["timestamp"] < current_candle_start].copy()

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ].copy()

    return df.reset_index(drop=True)


def load_candles(market):
    if market["provider"] == "Binance":
        return load_binance_candles(market["market_id"])

    if market["provider"] == "Gate.io":
        return load_gate_candles(market["market_id"])

    raise RuntimeError("Bilinmeyen borsa.")


# =========================================================
# MONTHLY BOTTOM ANALYSIS
# =========================================================

def analyze_monthly_bottom(
    df,
    max_distance_pct=10.0,
    require_three_rising=True,
):
    if len(df) < BARS_30_DAYS:
        return {
            "valid": False,
            "reason": (
                f"Yalnızca {len(df)} tamamlanmış mum var; "
                f"{BARS_30_DAYS} gerekli."
            ),
        }

    # Exact latest 30 days of completed 15-minute candles.
    recent = df.tail(BARS_30_DAYS).reset_index(drop=True)

    bottom_position = int(
        recent["low"].to_numpy().argmin()
    )

    bottom_price = float(
        recent.iloc[bottom_position]["low"]
    )

    current_price = float(
        recent.iloc[-1]["close"]
    )

    distance_pct = (
        (current_price - bottom_price) / bottom_price * 100
        if bottom_price > 0 else np.nan
    )

    last_three = recent.tail(3)

    closes = last_three["close"].to_numpy()
    opens = last_three["open"].to_numpy()

    three_rising = bool(
        len(closes) == 3
        and closes[0] < closes[1] < closes[2]
    )

    latest_green = bool(
        closes[-1] > opens[-1]
    )

    bottom_before_last_three = (
        bottom_position < len(recent) - 3
    )

    near_bottom = bool(
        np.isfinite(distance_pct)
        and 0 <= distance_pct <= max_distance_pct
    )

    reversal = bool(
        near_bottom
        and bottom_before_last_three
        and latest_green
        and (
            three_rising
            if require_three_rising
            else True
        )
    )

    if reversal:
        signal = "AYLIK DİPTEN DÖNÜŞ"
    elif three_rising and latest_green:
        signal = "YÜKSELİYOR"
    else:
        signal = "BEKLE"

    bottom_timestamp = int(
        recent.iloc[bottom_position]["timestamp"]
    )

    bottom_time = datetime.fromtimestamp(
        bottom_timestamp,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "valid": True,
        "bottom_price": bottom_price,
        "current_price": current_price,
        "distance_pct": distance_pct,
        "bottom_time": bottom_time,
        "three_rising": three_rising,
        "latest_green": latest_green,
        "near_bottom": near_bottom,
        "bottom_before_last_three": bottom_before_last_three,
        "reversal": reversal,
        "signal": signal,
        "bars": len(recent),
    }


# =========================================================
# SCAN ONE MARKET
# =========================================================

def scan_market(market, ticker, max_distance_pct):
    df = load_candles(market)

    analysis = analyze_monthly_bottom(
        df,
        max_distance_pct=max_distance_pct,
        require_three_rising=True,
    )

    if not analysis.get("valid"):
        raise RuntimeError(analysis.get("reason", "Veri yetersiz."))

    candle_price = float(analysis["current_price"])
    ticker_price = float(ticker.get("price", 0.0))

    if candle_price <= 0 or ticker_price <= 0:
        raise RuntimeError("Geçersiz fiyat.")

    price_difference_pct = (
        abs(ticker_price - candle_price)
        / candle_price * 100
    )

    # If prices differ substantially, mark the result as unreliable.
    # This can happen when markets move quickly or APIs update at
    # slightly different times.
    price_check = price_difference_pct <= 2.0

    return {
        "symbol": market["symbol"],
        "market_id": market["market_id"],
        "provider": market["provider"],
        "current_close_15m": candle_price,
        "ticker_price": ticker_price,
        "price_difference_pct": round(
            price_difference_pct, 4
        ),
        "price_check_ok": price_check,
        "volume_24h_usdt": ticker.get("volume", 0.0),
        "change_24h_pct": ticker.get("change_24h", 0.0),
        "monthly_bottom": analysis["bottom_price"],
        "monthly_bottom_time_utc": analysis["bottom_time"],
        "distance_from_bottom_pct": round(
            analysis["distance_pct"], 4
        ),
        "three_rising_candles": analysis["three_rising"],
        "latest_candle_green": analysis["latest_green"],
        "near_monthly_bottom": analysis["near_bottom"],
        "signal": analysis["signal"],
        "is_monthly_reversal": analysis["reversal"],
        "candles_30d": analysis["bars"],
    }


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("📉 Aylık Dipten Dönüş Coin Tarayıcı")

st.write(
    "Aktif Spot USDT marketlerini tarar, son 30 günün en düşük "
    "fiyatını bulur ve bu dipten yükselmeye başlayan coinleri "
    "ayrı listeler. Otomatik işlem yapmaz."
)

st.warning(
    "Aylık dipten dönüş sinyali fiyatın yükselmeye devam edeceğini "
    "garanti etmez. Düşük hacimli coinlerde spread ve likidite "
    "riskleri özellikle yüksek olabilir."
)

with st.sidebar:
    st.header("Tarama Ayarları")

    max_distance_pct = st.slider(
        "Aylık dipten maksimum uzaklık (%)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )

    min_volume = st.number_input(
        "Minimum 24 saatlik hacim (USDT)",
        min_value=0,
        value=10000,
        step=10000,
    )

    scan_count = st.number_input(
        "Taranacak market sayısı",
        min_value=1,
        max_value=2000,
        value=50,
        step=25,
    )

    mode = st.selectbox(
        "Market sıralaması",
        ["En yüksek hacim", "API listesindeki sıra"],
    )

    start_scan = st.button(
        "🔍 Taramayı Başlat",
        type="primary",
        use_container_width=True,
    )


if "results" not in st.session_state:
    st.session_state["results"] = None


if start_scan:
    progress = st.progress(0)
    status_box = st.empty()

    try:
        # Prefer Binance. If unreachable, use Gate.io's own markets.
        try:
            markets = get_binance_markets()
            tickers = get_binance_tickers()
            provider = "Binance"

        except Exception as binance_error:
            st.warning(
                "Binance API erişilemedi. Gate.io verileri deneniyor. "
                "Gate.io marketleri Binance marketleriyle aynı kabul "
                "edilmez. Binance hatası: "
                f"{str(binance_error)[:200]}"
            )

            markets = get_gate_markets()
            tickers = get_gate_tickers()
            provider = "Gate.io"

        candidates = []

        for market in markets:
            ticker = tickers.get(market["market_id"])

            if ticker is None:
                continue

            if ticker.get("volume", 0.0) < min_volume:
                continue

            candidates.append({
                **market,
                **ticker,
            })

        if mode == "En yüksek hacim":
            candidates.sort(
                key=lambda item: item.get("volume", 0.0),
                reverse=True,
            )

        candidates = candidates[:int(scan_count)]

        if not candidates:
            raise RuntimeError(
                "Filtrelerden geçen market yok. Minimum hacmi "
                "düşürüp yeniden deneyin."
            )

        st.info(
            f"Kaynak: {provider} | "
            f"Aktif market sayısı: {len(markets)} | "
            f"Taranacak: {len(candidates)}"
        )

        results = []
        errors = []

        for index, market in enumerate(candidates):
            status_box.write(
                f"{index + 1}/{len(candidates)} — "
                f"{market['symbol']} [{market['market_id']}]"
            )

            ticker = tickers[market["market_id"]]

            try:
                result = scan_market(
                    market,
                    ticker,
                    max_distance_pct,
                )
                results.append(result)

            except Exception as error:
                errors.append({
                    "symbol": market["symbol"],
                    "market_id": market["market_id"],
                    "provider": market["provider"],
                    "error": str(error)[:250],
                })

            progress.progress(
                (index + 1) / len(candidates)
            )

            time.sleep(0.08)

        st.session_state["results"] = pd.DataFrame(results)
        st.session_state["errors"] = pd.DataFrame(errors)
        st.session_state["provider"] = provider
        st.session_state["scan_time"] = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        status_box.success(
            f"Tarama tamamlandı: {len(results)} başarılı, "
            f"{len(errors)} hatalı market."
        )

    except Exception as error:
        st.error(f"Tarama başlatılamadı: {error}")


# =========================================================
# DISPLAY RESULTS
# =========================================================

results = st.session_state.get("results")

if results is not None:
    if results.empty:
        st.warning(
            "Analiz sonucu yok. Hata listesini inceleyin veya "
            "tarama ayarlarını değiştirin."
        )

    else:
        st.caption(
            f"Kaynak: {st.session_state.get('provider', '-')}"
            f" | Son tarama: {st.session_state.get('scan_time', '-')}"
        )

        # Only trust results whose market price checks pass.
        verified = results[
            results["price_check_ok"] == True
        ].copy()

        reversals = verified[
            verified["is_monthly_reversal"] == True
        ].sort_values(
            "distance_from_bottom_pct",
            ascending=True,
        )

        rising = verified[
            (verified["three_rising_candles"] == True)
            & (verified["latest_candle_green"] == True)
        ].sort_values(
            "distance_from_bottom_pct",
            ascending=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Analiz edilen", len(results))
        c2.metric("Fiyat kontrolünden geçen", len(verified))
        c3.metric("Aylık dipten dönüş", len(reversals))
        c4.metric("Son 3 mumu yükselen", len(rising))

        columns = [
            "symbol",
            "market_id",
            "provider",
            "current_close_15m",
            "ticker_price",
            "price_difference_pct",
            "monthly_bottom",
            "monthly_bottom_time_utc",
            "distance_from_bottom_pct",
            "three_rising_candles",
            "latest_candle_green",
            "signal",
            "volume_24h_usdt",
            "change_24h_pct",
        ]

        st.subheader("🎯 Aylık Dipten Dönüş Şartlarını Karşılayanlar")

        if reversals.empty:
            st.info(
                "Filtreleri karşılayan coin bulunamadı. Bu durum, "
                "şartların aynı anda gerçekleşmemiş olmasından "
                "kaynaklanabilir."
            )
        else:
            st.dataframe(
                reversals[columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Aylık dipten dönüş CSV indir",
                data=reversals.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="aylik_dipten_donus.csv",
                mime="text/csv",
            )

        st.subheader("📈 Son 3 Mumda Yükselen Coinler")

        st.dataframe(
            rising[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Yükselen coinler CSV indir",
            data=rising.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="son_uc_mum_yukselen_coinler.csv",
            mime="text/csv",
        )

        st.subheader("📋 Tüm Analizler")

        st.dataframe(
            results[columns + [
                "price_check_ok",
                "candles_30d",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tüm sonuçları CSV indir",
            data=results.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="aylik_dip_tum_sonuclar.csv",
            mime="text/csv",
        )

        errors = st.session_state.get("errors")

        if errors is not None and not errors.empty:
            with st.expander(
                f"Veri hataları ({len(errors)})"
            ):
                st.dataframe(
                    errors,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Hataları CSV indir",
                    data=errors.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="aylik_dip_hatalar.csv",
                    mime="text/csv",
                )

else:
    st.info(
        "Sol taraftaki ayarları belirleyip "
        "'Taramayı Başlat' düğmesine basın."
    )
