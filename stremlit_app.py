"""Binomo için CSV tabanlı araştırma ve Excel sinyal uygulaması.

Bu yazılım yatırım tavsiyesi veya otomatik işlem aracı değildir. Binomo'dan
dışa aktarılan mum verisini analiz eder; platform hesabına bağlanmaz.
"""
from __future__ import annotations

import io
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        GRU, LSTM, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
OUTPUT_FILE = APP_DIR / "binomo_sinyalleri.xlsx"
AUTO_CSV_FILE = APP_DIR / "hazir_veri.csv"
LOG_FILE = APP_DIR / "hata_log.txt"
MODEL_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)

FEATURES = [
    # Getiriler ve trend
    "return_1", "return_3", "return_5", "return_10",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "ema_ratio_5", "ema_ratio_9", "ema_ratio_12", "ema_ratio_15",
    "ema_ratio_21", "ema_ratio_26", "ema_ratio_50",
    "ema_15_slope", "ema_15_26_spread",
    # Momentum
    "rsi_7", "rsi_14", "rsi_21", "stoch_k", "stoch_d",
    "williams_r", "roc_10", "cci_20",
    # Trend gücü
    "macd", "macd_signal", "macd_histogram", "adx_14", "plus_di", "minus_di",
    # Oynaklık ve fiyat konumu
    "volatility_10", "volatility_20", "atr_14", "natr_14",
    "bb_position", "bb_width", "bb_percent_b", "range_ratio",
    # Hacim
    "volume_change", "obv_change", "mfi_14", "vwap_ratio",
]

MARKET_SYMBOLS = {
    "Crypto IDX (piyasa vekili: Bitcoin)": "BTC-USD",
    "Bitcoin (OTC vekili)": "BTC-USD",
    "Ethereum (OTC vekili)": "ETH-USD",
    "Solana (OTC vekili)": "SOL-USD",
    "FC Barcelona Token (OTC vekili)": "BAR-USD",
    "Cardano (OTC vekili)": "ADA-USD",
    "Chainlink (OTC vekili)": "LINK-USD",
    "Bitcoin Cash (OTC vekili)": "BCH-USD",
    "Kusama (OTC vekili)": "KSM-USD",
    "Aave (OTC vekili)": "AAVE-USD",
    "PancakeSwap (OTC vekili)": "CAKE-USD",
    "Uniswap (OTC vekili)": "UNI-USD",
    "EUR/USD": "EURUSD=X",
    "EUR/USD (OTC vekili)": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "GBP/USD (OTC vekili)": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/JPY (OTC vekili)": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "AUD/USD (OTC vekili)": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CAD (OTC vekili)": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "NZD/USD (OTC vekili)": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/GBP (OTC vekili)": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/JPY (OTC vekili)": "GBPJPY=X",
    "GBP/CHF (OTC vekili)": "GBPCHF=X",
    "CHF/JPY (OTC vekili)": "CHFJPY=X",
    "EUR/CAD (OTC vekili)": "EURCAD=X",
    "AUD/CAD (OTC vekili)": "AUDCAD=X",
    "GBP/NZD (OTC vekili)": "GBPNZD=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "Bitcoin/USD": "BTC-USD",
    "Ethereum/USD": "ETH-USD",
    "Solana/USD": "SOL-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil (WTI)": "CL=F",
    "Natural Gas": "NG=F",
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "DAX 40": "^GDAXI",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Meta": "META",
    "Google": "GOOGL",
}

INTERVAL_PERIODS = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "2y",
    "1d": "10y",
}


@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(symbol: str, interval: str) -> pd.DataFrame:
    """Ek paket gerektirmeden harici piyasa kaynağından OHLCV indirir."""
    encoded_symbol = urlencode({"symbol": symbol}).split("=", 1)[1]
    query = urlencode({
        "interval": interval,
        "range": INTERVAL_PERIODS[interval],
        "includePrePost": "false",
        "events": "div,splits",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{query}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Piyasa veri servisine erişilemedi: {exc}") from exc

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(str(chart["error"]))
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"{symbol} için çevrim içi veri bulunamadı.")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        raise ValueError(f"{symbol} için mum verisi boş döndü.")
    size = len(timestamps)
    downloaded = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quotes.get("open", [None] * size),
        "high": quotes.get("high", [None] * size),
        "low": quotes.get("low", [None] * size),
        "close": quotes.get("close", [None] * size),
        "volume": quotes.get("volume", [0] * size),
    })
    return normalize_columns(downloaded)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "timestamp": "time", "datetime": "time", "date": "time",
        "tarih": "time", "zaman": "time", "açılış": "open", "acilis": "open",
        "yüksek": "high", "yuksek": "high", "düşük": "low", "dusuk": "low",
        "kapanış": "close", "kapanis": "close", "hacim": "volume",
    }
    out = df.copy()
    out.columns = [aliases.get(str(c).strip().lower(), str(c).strip().lower()) for c in out.columns]
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError("Eksik sütun(lar): " + ", ".join(sorted(missing)))
    if "volume" not in out:
        out["volume"] = 0.0
    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(out) < 150:
        raise ValueError("Eğitim için en az 150 geçerli mum gerekir.")
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50) / 100


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)


def directional_index(frame: pd.DataFrame, period: int = 14):
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    atr = true_range(frame).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx / 100, plus_di / 100, minus_di / 100


def money_flow_index(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    flow = typical * frame["volume"]
    positive = flow.where(typical.diff() > 0, 0.0).rolling(period).sum()
    negative = flow.where(typical.diff() < 0, 0.0).rolling(period).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50) / 100


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    for n in [1, 3, 5, 10]:
        x[f"return_{n}"] = close.pct_change(n)
    for n in [5, 10, 20, 50]:
        x[f"sma_ratio_{n}"] = close / close.rolling(n).mean() - 1
    emas = {}
    for n in [5, 9, 12, 15, 21, 26, 50]:
        emas[n] = close.ewm(span=n, adjust=False).mean()
        x[f"ema_ratio_{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1
    x["ema_15_slope"] = emas[15].pct_change(3)
    x["ema_15_26_spread"] = emas[15] / emas[26] - 1
    returns = close.pct_change()
    x["volatility_10"] = returns.rolling(10).std()
    x["volatility_20"] = returns.rolling(20).std()
    x["rsi_7"] = rsi(close, 7)
    x["rsi_14"] = rsi(close)
    x["rsi_21"] = rsi(close, 21)
    ema12, ema26 = emas[12], emas[26]
    x["macd"] = (ema12 - ema26) / close
    x["macd_signal"] = (ema12 - ema26).ewm(span=9, adjust=False).mean() / close
    x["macd_histogram"] = x["macd"] - x["macd_signal"]
    mid = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    upper, lower = mid + 2 * deviation, mid - 2 * deviation
    band_range = (upper - lower).replace(0, np.nan)
    x["bb_position"] = (close - lower) / band_range
    x["bb_percent_b"] = x["bb_position"]
    x["bb_width"] = band_range / mid.replace(0, np.nan)
    low14 = x["low"].rolling(14).min()
    high14 = x["high"].rolling(14).max()
    x["stoch_k"] = (close - low14) / (high14 - low14).replace(0, np.nan)
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()
    x["williams_r"] = (close - high14) / (high14 - low14).replace(0, np.nan)
    x["roc_10"] = close.pct_change(10)
    typical = (x["high"] + x["low"] + close) / 3
    mean_deviation = typical.rolling(20).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    x["cci_20"] = (typical - typical.rolling(20).mean()) / (0.015 * mean_deviation).replace(0, np.nan)
    atr = true_range(x).ewm(alpha=1 / 14, adjust=False).mean()
    x["atr_14"] = atr / close.replace(0, np.nan)
    x["natr_14"] = x["atr_14"]
    x["adx_14"], x["plus_di"], x["minus_di"] = directional_index(x)
    x["range_ratio"] = (x["high"] - x["low"]) / close.replace(0, np.nan)
    # Forex kaynaklarında gerçek hacim bulunmayabilir. Böyle bir durumda
    # hacim-tabanlı özellikleri sıfır/nötr üretmek eğitim satırlarını korur.
    has_volume = bool((x["volume"].fillna(0) > 0).any())
    effective_volume = x["volume"].fillna(0) if has_volume else pd.Series(1.0, index=x.index)
    x["volume_change"] = effective_volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * effective_volume).cumsum()
    x["obv_change"] = obv.diff(5) / (obv.abs().rolling(20).mean() + 1e-9)
    if has_volume:
        x["mfi_14"] = money_flow_index(x)
    else:
        x["mfi_14"] = 0.5
    cumulative_volume = effective_volume.cumsum()
    vwap = (typical * effective_volume).cumsum() / cumulative_volume
    x["vwap_ratio"] = close / vwap.replace(0, np.nan) - 1
    future_return = close.shift(-horizon) / close - 1
    x["target"] = np.where(future_return.notna(), (future_return > 0).astype(int), np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def sequences(frame: pd.DataFrame, lookback: int):
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    labels = frame["target"].to_numpy()
    xs, ys, rows = [], [], []
    for end in range(lookback - 1, len(frame)):
        if np.isnan(labels[end]):
            continue
        window = values[end - lookback + 1:end + 1]
        med = np.nanmedian(window, axis=0)
        # Tamamen boş kalan opsiyonel bir gösterge nötr (0) kabul edilir.
        med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
        window = np.where(np.isnan(window), med, window)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        xs.append(window)
        ys.append(int(labels[end]))
        rows.append(end)
    return np.asarray(xs), np.asarray(ys), rows


def build_dl_models(shape):
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lstm = Sequential([
        Input(shape=shape), LSTM(48), Dropout(.2), Dense(24, activation="relu"),
        Dense(1, activation="sigmoid")
    ], name="LSTM")
    gru = Sequential([
        Input(shape=shape), GRU(48), Dropout(.2), Dense(24, activation="relu"),
        Dense(1, activation="sigmoid")
    ], name="GRU")
    inp = Input(shape=shape)
    z = Conv1D(48, 3, padding="causal", activation="relu")(inp)
    z = Conv1D(24, 3, padding="causal", activation="relu")(z)
    z = GlobalAveragePooling1D()(z)
    cnn = Model(inp, Dense(1, activation="sigmoid")(z), name="CNN1D")
    for model in (lstm, gru, cnn):
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return [lstm, gru, cnn], early


def train_and_predict(frame, lookback, epochs):
    X, y, row_ids = sequences(frame, lookback)
    if len(X) < 100:
        raise ValueError(
            f"Yalnızca {len(X)} eğitim penceresi oluştu. Daha uzun bir mum "
            "aralığı seçin (15m, 30m veya 1h önerilir)."
        )
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Seçilen dönemde fiyat yalnızca tek yönde hareket etmiş. "
            "Mum aralığını veya tahmin ufkunu değiştirin."
        )
    split = int(len(X) * .8)
    if split < 50 or len(X) - split < 20:
        raise ValueError("Zaman bazlı test bölümü için daha fazla mum gerekiyor.")
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    flat_train = X_train.reshape(len(X_train), -1)
    flat_test = X_test.reshape(len(X_test), -1)
    latest_seq = X[-1:]
    probabilities, names = [], []

    classical = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]
    for name, estimator in classical:
        pipe = Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", estimator)])
        pipe.fit(flat_train, y_train)
        probabilities.append(pipe.predict_proba(flat_test)[:, 1])
        names.append(name)
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    if TF_AVAILABLE:
        dl_models, early = build_dl_models(X_train.shape[1:])
        for model in dl_models:
            model.fit(
                X_train, y_train, validation_split=.2, epochs=epochs, batch_size=32,
                callbacks=[early], verbose=0, shuffle=False,
            )
            probabilities.append(model.predict(X_test, verbose=0).ravel())
            names.append(model.name)
            model.save(MODEL_DIR / f"{model.name}.keras")

    ensemble = np.mean(probabilities, axis=0)
    pred = (ensemble >= .5).astype(int)
    metrics = {
        "Doğruluk": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "Test örneği": len(y_test),
    }

    latest_probs = []
    for name, estimator in classical:
        pipe = joblib.load(MODEL_DIR / f"{name}.joblib")
        latest_probs.append(float(pipe.predict_proba(latest_seq.reshape(1, -1))[0, 1]))
    if TF_AVAILABLE:
        for model in dl_models:
            latest_probs.append(float(model.predict(latest_seq, verbose=0)[0, 0]))
    probability = float(np.mean(latest_probs))
    return probability, metrics, names


def append_excel(record: dict, market_data: pd.DataFrame | None = None) -> bytes:
    new = pd.DataFrame([record])
    if OUTPUT_FILE.exists():
        old = pd.read_excel(OUTPUT_FILE, sheet_name="Sinyaller")
        data = pd.concat([old, new], ignore_index=True).tail(5000)
    else:
        data = new
    # En yüksek güvenli işlemler her zaman ilk sırada görünür.
    data = data.drop(columns=["Öncelik"], errors="ignore")
    data["Güven"] = pd.to_numeric(data["Güven"], errors="coerce")
    data = data.sort_values(
        ["Güven", "Zaman"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    data.insert(0, "Öncelik", np.arange(1, len(data) + 1))
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Sinyaller", index=False)
        ws = writer.sheets["Sinyaller"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        for col in ws.columns:
            letter = col[0].column_letter
            ws.column_dimensions[letter].width = min(32, max(12, max(len(str(c.value or "")) for c in col) + 2))
        if ws.max_row > 1:
            ws.conditional_formatting.add(
                f"F2:F{ws.max_row}",
                ColorScaleRule(start_type="min", start_color="F8696B",
                               mid_type="percentile", mid_value=50, mid_color="FFEB84",
                               end_type="max", end_color="63BE7B"),
            )
            for cell in ws["E"][1:]:
                cell.number_format = "0.00%"
            for cell in ws["F"][1:]:
                cell.number_format = "0.00%"
        pd.DataFrame([{
            "Not": "Araştırma amaçlıdır; yatırım tavsiyesi veya kazanç garantisi değildir.",
            "Veri": "Kullanıcının yüklediği geçmiş OHLC mum verisi",
            "Doğrulama": "Son %20 veri, zaman sırası korunarak test edilir.",
        }]).to_excel(writer, sheet_name="Açıklama", index=False)
        if market_data is not None and not market_data.empty:
            export_data = market_data.tail(5000).copy()
            # Excel saat dilimli datetime değerlerini kabul etmez.
            export_data["time"] = export_data["time"].dt.tz_localize(None)
            export_data.to_excel(writer, sheet_name="Piyasa_Verisi", index=False)
    return OUTPUT_FILE.read_bytes()


def prioritized_signal_table() -> pd.DataFrame:
    """Excel geçmişini ekranda güven yüzdesine göre önceliklendirir."""
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    history = pd.read_excel(OUTPUT_FILE, sheet_name="Sinyaller")
    history["Güven"] = pd.to_numeric(history["Güven"], errors="coerce")
    history = history.sort_values(
        ["Güven", "Zaman"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    history["Öncelik"] = np.arange(1, len(history) + 1)
    history["Güven"] = history["Güven"] * 100
    if "Model olasılığı" in history:
        history["Model olasılığı"] = history["Model olasılığı"] * 100
    if "Test doğruluğu" in history:
        history["Test doğruluğu"] = history["Test doğruluğu"] * 100
    visible = [
        "Öncelik", "Varlık", "Sinyal", "Güven", "Model olasılığı",
        "Tahmin ufku (mum)", "Test doğruluğu", "Zaman",
    ]
    return history[[column for column in visible if column in history.columns]]


def latest_indicators(frame: pd.DataFrame) -> dict:
    row = frame.iloc[-1]
    return {
        "EMA 15": round(float(row["close"] / (1 + row["ema_ratio_15"])), 6),
        "EMA15 eğimi": round(float(row["ema_15_slope"]), 6),
        "RSI 14": round(float(row["rsi_14"] * 100), 2),
        "Bollinger %B": round(float(row["bb_percent_b"]), 4),
        "Bollinger genişliği": round(float(row["bb_width"]), 6),
        "MACD": round(float(row["macd"]), 6),
        "MACD histogram": round(float(row["macd_histogram"]), 6),
        "ATR %": round(float(row["atr_14"] * 100), 4),
        "ADX": round(float(row["adx_14"] * 100), 2),
        "Stochastic %K": round(float(row["stoch_k"] * 100), 2),
        "Williams %R": round(float(row["williams_r"] * 100), 2),
        "CCI 20": round(float(row["cci_20"]), 2),
        "MFI 14": round(float(row["mfi_14"] * 100), 2),
    }


st.set_page_config(page_title="Binomo DL Araştırma", layout="wide")
st.title("Binomo Derin Öğrenme Araştırması")
st.warning("Araştırma amaçlıdır. Otomatik işlem yapmaz; yatırım tavsiyesi veya kazanç garantisi değildir.")
data_source = st.radio(
    "Veri kaynağı",
    ["Çevrim içi veriyi kendisi indir", "CSV kullan"],
    horizontal=True,
)
uploaded = None
interval = "15m"
if data_source == "CSV kullan":
    uploaded = st.file_uploader("Mum verisi CSV", type=["csv"])
else:
    st.info(
        "Veriler harici piyasa kaynağından otomatik indirilir. "
        "Binomo OTC fiyatları halka açık olmadığından OTC seçimlerinde "
        "normal piyasa karşılığı (vekil sembol) kullanılır."
    )
c1, c2, c3 = st.columns(3)
if data_source == "Çevrim içi veriyi kendisi indir":
    asset = c1.selectbox("Varlık", list(MARKET_SYMBOLS), index=0)
    interval = c2.selectbox("Mum aralığı", list(INTERVAL_PERIODS), index=2)
    horizon = c3.number_input("Tahmin ufku (mum)", 1, 20, 1)
else:
    asset = c1.text_input("Varlık", "EUR/USD")
    horizon = c2.number_input("Tahmin ufku (mum)", 1, 20, 1)
    interval = c3.text_input("Mum aralığı", "CSV")
c4, c5 = st.columns(2)
lookback = c4.number_input("Model penceresi (mum)", 20, 120, 40)
epochs = c5.slider("DL epoch", 5, 100, 25)
force_run = st.button("Verileri yenile, analiz et ve Excel'e kaydet", type="primary")

csv_bytes = None
csv_source = None
online_data = None
if data_source == "Çevrim içi veriyi kendisi indir":
    try:
        online_data = download_market_data(MARKET_SYMBOLS[asset], interval)
        csv_bytes = online_data.to_csv(index=False).encode("utf-8")
        csv_source = f"{MARKET_SYMBOLS[asset]} ({interval})"
        st.caption(f"{len(online_data):,} mum otomatik indirildi.")
    except Exception as exc:
        st.error(f"Çevrim içi veri alınamadı: {exc}")
elif uploaded is not None:
    csv_bytes = uploaded.getvalue()
    csv_source = uploaded.name
elif AUTO_CSV_FILE.exists():
    csv_bytes = AUTO_CSV_FILE.read_bytes()
    csv_source = AUTO_CSV_FILE.name
    st.info(f"Hazır veri bulundu: {AUTO_CSV_FILE.name}")

analysis_key = None
if csv_bytes:
    settings = f"{asset}|{horizon}|{lookback}|{epochs}".encode("utf-8")
    analysis_key = hashlib.sha256(csv_bytes + settings).hexdigest()

if "last_saved_analysis" not in st.session_state:
    st.session_state.last_saved_analysis = None

should_analyze = bool(
    csv_bytes
    and (force_run or analysis_key != st.session_state.last_saved_analysis)
)

if should_analyze:
    try:
        if online_data is not None:
            data = online_data
        else:
            raw = pd.read_csv(io.BytesIO(csv_bytes), sep=None, engine="python")
            data = normalize_columns(raw)
        frame = add_features(data, int(horizon))
        with st.spinner("Modeller eğitiliyor..."):
            probability, metrics, model_names = train_and_predict(frame, int(lookback), epochs)
        indicators = latest_indicators(frame)
        signal = "YUKARI" if probability >= .5 else "AŞAĞI"
        confidence = probability if signal == "YUKARI" else 1 - probability
        record = {
            "Zaman": datetime.now().astimezone().isoformat(timespec="seconds"),
            "Varlık": asset,
            "Sinyal": signal,
            "Model olasılığı": round(probability, 4),
            "Güven": round(confidence, 4),
            "Tahmin ufku (mum)": horizon,
            "Model sayısı": len(model_names),
            "Modeller": ", ".join(model_names),
            "Test doğruluğu": round(metrics["Doğruluk"], 4),
            "Test precision": round(metrics["Precision"], 4),
            "Test recall": round(metrics["Recall"], 4),
            "Mum sayısı": len(data),
            **indicators,
        }
        excel_bytes = append_excel(record, data)
        st.session_state.last_saved_analysis = analysis_key

        # Kullanıcının ilk gördüğü bölüm: önceliklendirilmiş işlem listesi.
        st.subheader("Öncelikli işlemler")
        st.caption("En yüksek güven yüzdesi ilk sıradadır.")
        priority_table = prioritized_signal_table()
        st.dataframe(
            priority_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Güven": st.column_config.ProgressColumn(
                    "Güven", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "Model olasılığı": st.column_config.NumberColumn(
                    "Yukarı olasılığı", format="%.1f%%"
                ),
                "Test doğruluğu": st.column_config.NumberColumn(
                    "Test doğruluğu", format="%.1f%%"
                ),
            },
        )
        st.success(
            f"{csv_source} otomatik analiz edildi ve Excel'e kaydedildi. "
            f"Sonuç: {signal} — model güveni %{confidence * 100:.1f}"
        )
        st.json(metrics)
        st.subheader("Son mum teknik göstergeleri")
        st.dataframe(pd.DataFrame([indicators]), use_container_width=True, hide_index=True)
        st.caption("Kullanılan modeller: " + ", ".join(model_names))
        st.download_button(
            "Excel'i indir", excel_bytes, OUTPUT_FILE.name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        logging.exception("Analiz başarısız")
        st.error(f"İşlem tamamlanamadı: {exc}")
elif csv_bytes and st.session_state.last_saved_analysis == analysis_key:
    st.success("Bu veri daha önce analiz edilip Excel'e kaydedildi.")
    existing = prioritized_signal_table()
    if not existing.empty:
        st.subheader("Öncelikli işlemler")
        st.dataframe(existing, use_container_width=True, hide_index=True)
