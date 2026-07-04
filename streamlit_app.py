"""
Binomo 1M Signal Engine

UYARI:
- Bu kod finansal tavsiye degildir.
- Binomo'nun tum verilerini "bilmek" teknik ve hukuki olarak mumkun degildir.
- Algoritma sadece sizin sagladiginiz canli/CSV mum verisini analiz eder.
- 1 dakikalik islemler cok yuksek risklidir; varsayilan mod guven filtresiyle
  emin olmadigi yerde NO-TRADE uretir.

Calistirma:
    pip install streamlit pandas numpy requests scikit-learn openpyxl
    streamlit run binomo_1m_algo.py

Veri secenekleri:
1) CSV yukle: time,open,high,low,close,volume kolonlari tavsiye edilir.
2) HTTP endpoint kullan: JSON olarak mum listesi donen kendi veri servisiniz.
   Ornek mum:
   {"time":"2026-07-04 12:00:00","open":1.1,"high":1.2,"low":1.0,"close":1.15,"volume":10}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    MLPClassifier = None
    StandardScaler = None


APP_TITLE = "Binomo 1M Algo - Deep Learning Streamlit"
SCAN_SECONDS = 60
MIN_CANDLES = 80
MAX_CANDLES = 500
DEFAULT_MIN_CONF = 68
DEFAULT_MAX_ATR = 0.008
EXPORT_FILE = Path("binomo_1m_signals.xlsx")


@dataclass
class SignalResult:
    asset: str
    signal: str
    confidence: int
    reason: str
    last_price: float
    rsi: float
    ema_fast_delta: float
    ema_slow_delta: float
    atr_ratio: float
    trend_score: float
    pullback_score: float
    best_strategy: str
    strategy_score: float
    timestamp: str


def normalize_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Mum verisini standart OHLCV formatina getirir."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    aliases = {
        "timestamp": "time",
        "date": "time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})

    if "close" not in out.columns:
        raise ValueError("Veride 'close' kolonu bulunmali.")

    if "open" not in out.columns:
        out["open"] = out["close"].shift(1).fillna(out["close"])
    if "high" not in out.columns:
        out["high"] = out[["open", "close"]].max(axis=1)
    if "low" not in out.columns:
        out["low"] = out[["open", "close"]].min(axis=1)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "time" not in out.columns:
        out["time"] = pd.date_range(end=pd.Timestamp.now(), periods=len(out), freq="1min")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_values("time").tail(MAX_CANDLES).reset_index(drop=True)
    return out[["time", "open", "high", "low", "close", "volume"]]


def fetch_http_candles(url: str, timeout: int = 8) -> pd.DataFrame:
    """Kendi Binomo veri koprunuzden mum JSON'u okur."""
    if not url.strip():
        return pd.DataFrame()

    r = requests.get(url.strip(), timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict):
        rows = payload.get("candles") or payload.get("data") or payload.get("result") or []
    else:
        rows = payload

    return normalize_candles(pd.DataFrame(rows))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ret1"] = close.pct_change().fillna(0)
    out["ret2"] = close.pct_change(2).fillna(0)
    out["ret3"] = close.pct_change(3).fillna(0)
    out["ema5"] = close.ewm(span=5, adjust=False).mean()
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema15"] = close.ewm(span=15, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["rsi7"] = rsi(close, 7)
    out["rsi14"] = rsi(close, 14)
    out["rsi21"] = rsi(close, 21)
    out["atr14"] = atr(out, 14)
    out["atr_ratio"] = out["atr14"] / out["close"].replace(0, np.nan)
    atr_mean = out["atr_ratio"].rolling(40).mean()
    atr_std = out["atr_ratio"].rolling(40).std()
    out["atr_z"] = (out["atr_ratio"] - atr_mean) / atr_std.replace(0, np.nan)
    out["ema5_delta"] = (close - out["ema5"]) / close.replace(0, np.nan)
    out["ema9_delta"] = (close - out["ema9"]) / close.replace(0, np.nan)
    out["ema15_delta"] = (close - out["ema15"]) / close.replace(0, np.nan)
    out["ema21_delta"] = (close - out["ema21"]) / close.replace(0, np.nan)
    out["ema50_delta"] = (close - out["ema50"]) / close.replace(0, np.nan)
    out["ema5_15_spread"] = (out["ema5"] - out["ema15"]) / close.replace(0, np.nan)
    out["ema15_50_spread"] = (out["ema15"] - out["ema50"]) / close.replace(0, np.nan)
    out["mom3"] = close.pct_change(3)
    out["mom5"] = close.pct_change(5)
    out["mom10"] = close.pct_change(10)
    out["mom15"] = close.pct_change(15)
    out["vol10"] = out["ret1"].rolling(10).std()
    out["vol20"] = out["ret1"].rolling(20).std()
    out["range_pos20"] = range_position(close, 20)
    out["range_pos50"] = range_position(close, 50)
    out["body"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["wick_up"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["wick_down"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range
    out["candle_power"] = (out["close"] - out["open"]) / candle_range
    out["trend_10"] = rolling_slope(close, 10)
    out["trend_20"] = rolling_slope(close, 20)
    out["trend_50"] = rolling_slope(close, 50)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def range_position(close: pd.Series, window: int) -> pd.Series:
    low = close.rolling(window).min()
    high = close.rolling(window).max()
    return ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)


def rolling_slope(close: pd.Series, window: int) -> pd.Series:
    values = []
    x = np.arange(window)
    for i in range(len(close)):
        if i + 1 < window:
            values.append(0.0)
            continue
        seg = close.iloc[i + 1 - window : i + 1].to_numpy(dtype=float)
        mean_price = max(abs(float(np.mean(seg))), 1e-9)
        try:
            values.append(float(np.polyfit(x, seg, 1)[0] / mean_price))
        except Exception:
            values.append(0.0)
    return pd.Series(values, index=close.index)


FEATURES = [
    "ret1",
    "ret2",
    "ret3",
    "rsi7",
    "rsi14",
    "rsi21",
    "atr_ratio",
    "atr_z",
    "ema5_delta",
    "ema9_delta",
    "ema15_delta",
    "ema21_delta",
    "ema50_delta",
    "ema5_15_spread",
    "ema15_50_spread",
    "mom3",
    "mom5",
    "mom10",
    "mom15",
    "vol10",
    "vol20",
    "range_pos20",
    "range_pos50",
    "body",
    "wick_up",
    "wick_down",
    "candle_power",
    "trend_10",
    "trend_20",
    "trend_50",
]


def make_supervised(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sonraki 1 mum yukari mi asagi mi etiketi olusturur."""
    feat = add_indicators(df)
    y = (feat["close"].shift(-1) > feat["close"]).astype(int)
    train = feat.iloc[:-1].copy()
    y = y.iloc[:-1]
    X = train[FEATURES].to_numpy(dtype=float)
    return X, y.to_numpy(dtype=int)


def strategy_votes(df: pd.DataFrame) -> dict[str, pd.Series]:
    feat = add_indicators(df)
    votes = {}

    votes["EMA15_Trend"] = np.sign(feat["ema5_15_spread"] + feat["ema15_50_spread"])
    votes["RSI_Reversal"] = pd.Series(
        np.where(feat["rsi14"] < 35, 1, np.where(feat["rsi14"] > 65, -1, 0)),
        index=feat.index,
    )
    votes["ATR_Momentum"] = np.sign(feat["mom3"] + feat["mom5"]) * (feat["atr_z"].abs() < 2.5).astype(int)
    votes["Pullback_EMA15"] = pd.Series(
        np.where(
            (feat["ema15_50_spread"] > 0) & (feat["ema15_delta"] < 0) & (feat["rsi14"] > 38),
            1,
            np.where(
                (feat["ema15_50_spread"] < 0) & (feat["ema15_delta"] > 0) & (feat["rsi14"] < 62),
                -1,
                0,
            ),
        ),
        index=feat.index,
    )
    votes["Range_Break"] = pd.Series(
        np.where(feat["range_pos20"] > 0.82, 1, np.where(feat["range_pos20"] < 0.18, -1, 0)),
        index=feat.index,
    )

    return {name: pd.Series(vote, index=feat.index).fillna(0).clip(-1, 1) for name, vote in votes.items()}


def backtest_strategy_weights(df: pd.DataFrame, lookback: int = 180) -> dict:
    """Son veride hangi strateji daha iyi calismis, hizli agirlik hesaplar."""
    if len(df) < 40:
        return {"weights": {}, "best": "Yetersiz veri", "score": 0.0, "combined_vote": 0.0}

    feat = add_indicators(df).tail(lookback).reset_index(drop=True)
    votes = strategy_votes(feat)
    next_dir = np.sign(feat["close"].shift(-1) - feat["close"]).fillna(0)

    weights = {}
    best_name = "Yok"
    best_score = 0.0

    for name, vote in votes.items():
        aligned = pd.DataFrame({"vote": vote.iloc[:-1], "next": next_dir.iloc[:-1]}).replace(0, np.nan).dropna()
        if len(aligned) < 12:
            score = 0.0
        else:
            score = float((np.sign(aligned["vote"]) == np.sign(aligned["next"])).mean())
            score = max(0.0, score - 0.50) * 2.0
        weights[name] = score
        if score > best_score:
            best_name = name
            best_score = score

    latest_votes = {name: float(vote.iloc[-1]) for name, vote in votes.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        combined_vote = float(np.mean(list(latest_votes.values()))) if latest_votes else 0.0
    else:
        combined_vote = sum(latest_votes[name] * weights[name] for name in latest_votes) / total_weight

    return {
        "weights": weights,
        "latest_votes": latest_votes,
        "best": best_name,
        "score": best_score,
        "combined_vote": float(np.clip(combined_vote, -1, 1)),
    }


class OneMinuteModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler() if StandardScaler else None
        self.models = []
        self.deep_model = None
        self.trained = False
        self.deep_trained = False

    def train(self, df: pd.DataFrame) -> str:
        if RandomForestClassifier is None:
            self.trained = False
            return "sklearn yok; sadece kural motoru kullaniliyor."

        if len(df) < MIN_CANDLES:
            self.trained = False
            return f"Model icin en az {MIN_CANDLES} mum gerekir; kural motoru kullaniliyor."

        X, y = make_supervised(df)
        if len(np.unique(y)) < 2:
            self.trained = False
            return "Etiketler tek yonlu; model egitilmedi, kural motoru kullaniliyor."

        Xs = self.scaler.fit_transform(X)
        self.models = [
            RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42, class_weight="balanced"),
            GradientBoostingClassifier(random_state=42),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ]
        for model in self.models:
            model.fit(Xs, y)

        if MLPClassifier is not None:
            self.deep_model = MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                solver="adam",
                alpha=0.0008,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            )
            self.deep_model.fit(Xs, y)
            self.deep_trained = True

        self.trained = True
        dl = " + derin ogrenme aktif" if self.deep_trained else ""
        return f"{len(self.models)} klasik model{dl}, son {len(df)} mumla egitildi."

    def predict_proba_up(self, df: pd.DataFrame) -> Optional[float]:
        if not self.trained or not self.models:
            return None
        feat = add_indicators(df).iloc[-1:][FEATURES].to_numpy(dtype=float)
        xs = self.scaler.transform(feat)
        probs = []
        for model in self.models:
            p = model.predict_proba(xs)[0]
            cls = list(model.classes_)
            probs.append(float(p[cls.index(1)] if 1 in cls else 0.5))

        if self.deep_trained and self.deep_model is not None:
            p = self.deep_model.predict_proba(xs)[0]
            cls = list(self.deep_model.classes_)
            deep_prob = float(p[cls.index(1)] if 1 in cls else 0.5)
            probs.extend([deep_prob, deep_prob])

        return float(np.mean(probs))


def rule_scores(df: pd.DataFrame) -> dict:
    feat = add_indicators(df)
    last = feat.iloc[-1]
    prev = feat.iloc[-6:-1]

    ema_fast_delta = (last["ema5"] - last["ema15"]) / max(abs(last["close"]), 1e-9)
    ema_slow_delta = (last["ema15"] - last["ema50"]) / max(abs(last["close"]), 1e-9)
    trend_score = np.tanh((ema_fast_delta + ema_slow_delta) * 250)

    last5_down = int((prev["close"].diff().dropna() < 0).sum())
    last5_up = int((prev["close"].diff().dropna() > 0).sum())
    atr_normal = abs(float(last["atr_z"])) < 2.5
    pullback_buy = 1.0 if trend_score > 0 and last5_down >= 3 and last["rsi14"] < 58 and atr_normal else 0.0
    pullback_sell = 1.0 if trend_score < 0 and last5_up >= 3 and last["rsi14"] > 42 and atr_normal else 0.0

    momentum = np.tanh((last["mom3"] + last["mom5"] + last["mom10"]) * 90)
    rsi_bias = ((50 - last["rsi14"]) / 50 + (50 - last["rsi21"]) / 50) / 2

    return {
        "rsi": float(last["rsi14"]),
        "atr_ratio": float(last["atr_ratio"]),
        "ema_fast_delta": float(ema_fast_delta),
        "ema_slow_delta": float(ema_slow_delta),
        "trend_score": float(trend_score),
        "momentum": float(momentum),
        "rsi_bias": float(rsi_bias),
        "pullback_buy": float(pullback_buy),
        "pullback_sell": float(pullback_sell),
        "pullback_score": float(pullback_buy - pullback_sell),
    }


def generate_signal(
    asset: str,
    candles: pd.DataFrame,
    model: OneMinuteModel,
    min_conf: int,
    max_atr_ratio: float,
) -> SignalResult:
    if len(candles) < 30:
        return SignalResult(asset, "NO-TRADE", 0, "Yetersiz mum verisi", 0, 50, 0, 0, 0, 0, 0, "Yetersiz veri", 0, now_str())

    df = normalize_candles(candles)
    scores = rule_scores(df)
    model_up = model.predict_proba_up(df)
    strategy = backtest_strategy_weights(df)

    if model_up is None:
        model_vote = 0.0
    else:
        model_vote = (model_up - 0.5) * 2

    raw = (
        0.38 * scores["trend_score"]
        + 0.24 * scores["momentum"]
        + 0.18 * scores["rsi_bias"]
        + 0.20 * scores["pullback_score"]
        + 0.35 * model_vote
        + 0.28 * strategy["combined_vote"]
    )
    raw = float(np.clip(raw, -1, 1))

    confidence = int(round(50 + abs(raw) * 49))
    signal = "BUY" if raw > 0 else "SELL"
    reason = f"Trend + momentum + RSI/ATR + EMA15 + deep model + strateji testi ({strategy['best']})"

    if scores["atr_ratio"] > max_atr_ratio:
        signal = "NO-TRADE"
        confidence = min(confidence, 60)
        reason = "Volatilite filtresi: ATR yuksek"
    elif confidence < min_conf:
        signal = "NO-TRADE"
        reason = "Guven skoru dusuk"
    elif scores["rsi"] >= 78 and signal == "BUY":
        signal = "NO-TRADE"
        confidence = min(confidence, 62)
        reason = "RSI asiri alim bolgesinde BUY engellendi"
    elif scores["rsi"] <= 22 and signal == "SELL":
        signal = "NO-TRADE"
        confidence = min(confidence, 62)
        reason = "RSI asiri satim bolgesinde SELL engellendi"

    return SignalResult(
        asset=asset,
        signal=signal,
        confidence=confidence,
        reason=reason,
        last_price=float(df["close"].iloc[-1]),
        rsi=scores["rsi"],
        ema_fast_delta=scores["ema_fast_delta"],
        ema_slow_delta=scores["ema_slow_delta"],
        atr_ratio=scores["atr_ratio"],
        trend_score=scores["trend_score"],
        pullback_score=scores["pullback_score"],
        best_strategy=strategy["best"],
        strategy_score=float(strategy["score"]),
        timestamp=now_str(),
    )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def result_to_dict(r: SignalResult) -> dict:
    return {
        "Time": r.timestamp,
        "Asset": r.asset,
        "Signal": r.signal,
        "Confidence": r.confidence,
        "Reason": r.reason,
        "Last_Price": round(r.last_price, 8),
        "RSI14": round(r.rsi, 2),
        "EMA5_21_Delta": round(r.ema_fast_delta, 8),
        "EMA21_50_Delta": round(r.ema_slow_delta, 8),
        "ATR_Ratio": round(r.atr_ratio, 8),
        "Trend_Score": round(r.trend_score, 4),
        "Pullback_Score": round(r.pullback_score, 4),
        "Best_Strategy": r.best_strategy,
        "Strategy_Score": round(r.strategy_score, 4),
    }


def append_excel(row: dict) -> None:
    new = pd.DataFrame([row])
    if EXPORT_FILE.exists():
        old = pd.read_excel(EXPORT_FILE)
        out = pd.concat([old, new], ignore_index=True).tail(1000)
    else:
        out = new
    out.to_excel(EXPORT_FILE, index=False)


def render_signal_badge(signal: str) -> None:
    if signal == "BUY":
        st.success("BUY")
    elif signal == "SELL":
        st.error("SELL")
    else:
        st.warning("NO-TRADE")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("1 dakikalik mum verisiyle calisir. MLP derin ogrenme + klasik ensemble + risk filtresi kullanir.")

    if "model" not in st.session_state:
        st.session_state.model = OneMinuteModel()
    if "candles" not in st.session_state:
        st.session_state.candles = pd.DataFrame()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_scan" not in st.session_state:
        st.session_state.last_scan = 0.0

    with st.sidebar:
        st.header("Ayarlar")
        asset = st.text_input("Varlik adi", value="EUR/USD OTC")
        min_conf = st.slider("Minimum guven", 50, 95, DEFAULT_MIN_CONF)
        max_atr = st.number_input("Maks ATR orani", min_value=0.0001, max_value=0.2, value=DEFAULT_MAX_ATR, step=0.0005, format="%.4f")
        endpoint = st.text_input("HTTP mum endpoint", value="")
        uploaded = st.file_uploader("CSV mum verisi yukle", type=["csv"])
        auto = st.toggle("60 saniyede otomatik tara", value=False)

        load_btn = st.button("Veriyi Yukle / Yenile", use_container_width=True)
        train_btn = st.button("Modeli Egit", use_container_width=True)
        scan_btn = st.button("Sinyal Uret", use_container_width=True)

    if uploaded is not None and load_btn:
        st.session_state.candles = normalize_candles(pd.read_csv(uploaded))
        st.success(f"CSV yuklendi: {len(st.session_state.candles)} mum")

    if endpoint and load_btn:
        try:
            st.session_state.candles = fetch_http_candles(endpoint)
            st.success(f"Endpoint okundu: {len(st.session_state.candles)} mum")
        except Exception as exc:
            st.error(f"Endpoint okunamadi: {exc}")

    candles = st.session_state.candles

    if train_btn:
        msg = st.session_state.model.train(candles)
        st.info(msg)

    should_auto_scan = auto and (time.time() - st.session_state.last_scan >= SCAN_SECONDS)
    if scan_btn or should_auto_scan:
        st.session_state.last_scan = time.time()
        result = generate_signal(asset, candles, st.session_state.model, min_conf, max_atr)
        row = result_to_dict(result)
        st.session_state.history.insert(0, row)
        try:
            append_excel(row)
        except Exception as exc:
            st.warning(f"Excel yazilamadi: {exc}")

    top1, top2, top3, top4 = st.columns(4)
    last_result = st.session_state.history[0] if st.session_state.history else None

    with top1:
        st.metric("Mum sayisi", len(candles))
    with top2:
        model_state = "Deep aktif" if st.session_state.model.deep_trained else ("Klasik aktif" if st.session_state.model.trained else "Kural motoru")
        st.metric("Model", model_state)
    with top3:
        st.metric("Son fiyat", "-" if candles.empty else f"{candles['close'].iloc[-1]:.8f}")
    with top4:
        st.metric("Son tarama", "-" if not last_result else last_result["Time"])

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Son Sinyal")
        if last_result:
            render_signal_badge(last_result["Signal"])
            st.metric("Guven", f"{last_result['Confidence']}%")
            st.write(last_result["Reason"])
            st.json({k: last_result[k] for k in ["RSI14", "ATR_Ratio", "Trend_Score", "Pullback_Score"]})
        else:
            st.info("Veri yukleyip Sinyal Uret'e basin.")

    with right:
        st.subheader("Mum Grafiği")
        if not candles.empty:
            chart_df = candles.set_index("time")[["close"]].tail(160)
            st.line_chart(chart_df)
            st.dataframe(candles.tail(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Henuz mum verisi yok.")

    st.divider()
    st.subheader("Sinyal Gecmisi")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    else:
        st.info("Gecmis bos.")

    if auto:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
