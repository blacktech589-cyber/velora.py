import os
import io
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy deps (graceful fallback)
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# -----------------------------
# Utility / Indicators (no `ta` dependency)
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def bollinger(close: pd.Series, window=20, n_std=2):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    high = mid + n_std * std
    low = mid - n_std * std
    width = (high - low) / mid.replace(0, np.nan)
    return high, low, mid, width


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Trend
    x["ema_9"] = ema(x["close"], 9)
    x["ema_21"] = ema(x["close"], 21)
    x["sma_20"] = sma(x["close"], 20)

    # Momentum
    x["rsi_14"] = rsi(x["close"], 14)
    m, s, h = macd(x["close"])
    x["macd"] = m
    x["macd_signal"] = s
    x["macd_diff"] = h

    # Volatility
    x["atr_14"] = atr(x, 14)
    bb_h, bb_l, bb_m, bb_w = bollinger(x["close"])
    x["bb_high"] = bb_h
    x["bb_low"] = bb_l
    x["bb_mid"] = bb_m
    x["bb_width"] = bb_w

    # Candle geometry
    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, 1e-9)
    x["body_ratio"] = x["body"] / x["range"]
    x["upper_wick"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x["low"]

    # Returns
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_5"] = x["close"].pct_change(5)

    return x


def make_labels_3class(df: pd.DataFrame, horizon=3, threshold=0.003) -> pd.Series:
    fut = df["close"].shift(-horizon)
    fut_ret = (fut - df["close"]) / df["close"]
    y = np.where(fut_ret > threshold, 2, np.where(fut_ret < -threshold, 0, 1))
    return pd.Series(y, index=df.index, name="label")


FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ema_9", "ema_21", "sma_20",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "atr_14", "bb_high", "bb_low", "bb_mid", "bb_width",
    "body", "range", "body_ratio", "upper_wick", "lower_wick",
    "ret_1", "ret_3", "ret_5"
]


# -----------------------------
# Deep Learning (LSTM + Attention)
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=32):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx + self.seq_len], dtype=torch.long),
        )


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h, _ = self.lstm(x)          # [B,T,H]
        w = torch.softmax(self.attn(h), dim=1)  # [B,T,1]
        ctx = (w * h).sum(dim=1)     # [B,H]
        return self.fc(ctx)          # [B,C]


def train_lstm_attention(X_train, y_train, X_val, y_val, seq_len=32, epochs=8, batch_size=64, lr=1e-3):
    device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    model = LSTMAttention(input_size=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = SeqDataset(X_train, y_train, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model, device


def predict_lstm_attention(model, device, X, seq_len=32):
    model.eval()
    probs = np.zeros((len(X), 3), dtype=np.float32)
    if len(X) <= seq_len:
        probs[:, 1] = 1.0
        return probs

    with torch.no_grad():
        for i in range(seq_len, len(X)):
            seq = torch.tensor(X[i-seq_len:i], dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(seq)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs[i] = p

    probs[:seq_len, 1] = 1.0
    return probs


# -----------------------------
# Risk / Backtest
# -----------------------------
def apply_conf_filter(pred, proba, min_conf=0.55):
    out = pred.copy()
    conf = proba.max(axis=1)
    out[conf < min_conf] = 1  # HOLD
    return out, conf


def run_backtest(df: pd.DataFrame, pred_col="pred", fee=0.0005, atr_mult_sl=1.5, atr_mult_tp=2.5):
    z = df.copy()
    pos_map = {0: -1, 1: 0, 2: 1}
    z["position"] = z[pred_col].map(pos_map).fillna(0)
    z["ret"] = z["close"].pct_change().fillna(0)
    z["atr_pct"] = (z["atr_14"] / z["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    raw = z["position"].shift(1).fillna(0) * z["ret"]
    sl = -(z["atr_pct"] * atr_mult_sl)
    tp = +(z["atr_pct"] * atr_mult_tp)
    capped = np.minimum(np.maximum(raw, sl), tp)

    z["turnover"] = (z["position"] - z["position"].shift(1).fillna(0)).abs()
    z["strategy_ret"] = capped - z["turnover"] * fee
    z["equity"] = (1 + z["strategy_ret"]).cumprod()
    return z


def metrics(equity: pd.Series, returns: pd.Series, bars_per_year=365 * 24):
    r = returns.dropna()
    if len(r) == 0:
        return dict(total_return=0.0, sharpe=0.0, max_drawdown=0.0, profit_factor=0.0)

    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    vol = float(r.std())
    sharpe = float(np.sqrt(bars_per_year) * r.mean() / vol) if vol > 0 else 0.0

    cm = equity.cummax()
    dd = (equity / cm) - 1.0
    mdd = float(dd.min())

    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    pf = gp / gl if gl > 0 else 0.0

    return dict(total_return=total, sharpe=sharpe, max_drawdown=mdd, profit_factor=pf)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Velora DL Trading Lab", layout="wide")
st.title("Velora DL Trading Lab (BUY / SELL / HOLD)")
st.caption("LightGBM + LSTM-Attention + Ensemble + Confidence Filter + ATR Risk")

with st.sidebar:
    st.header("Ayarlar")
    horizon = st.slider("Label Horizon", 1, 20, 3)
    threshold = st.slider("Label Threshold", 0.0005, 0.02, 0.003, step=0.0005, format="%.4f")
    min_conf = st.slider("Min Confidence", 0.30, 0.95, 0.55, step=0.01)
    fee = st.slider("Fee", 0.0, 0.005, 0.0005, step=0.0001, format="%.4f")
    atr_sl = st.slider("ATR SL Mult", 0.5, 5.0, 1.5, step=0.1)
    atr_tp = st.slider("ATR TP Mult", 0.5, 8.0, 2.5, step=0.1)
    seq_len = st.slider("LSTM Seq Len", 8, 128, 32, step=8)
    epochs = st.slider("LSTM Epochs", 1, 30, 8)
    train_btn = st.button("Train & Backtest")

uploaded = st.file_uploader("OHLCV CSV yükle (timestamp, open, high, low, close, volume)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    needed = {"timestamp", "open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        st.error(f"Eksik kolonlar: {needed - set(df.columns)}")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    st.subheader("Ham Veri")
    st.dataframe(df.tail(10), use_container_width=True)

    feat = add_features(df)
    feat["label"] = make_labels_3class(feat, horizon=horizon, threshold=threshold)
    feat = feat.dropna().reset_index(drop=True)

    if len(feat) < 300:
        st.warning("Veri az olabilir (öneri: en az 300+ mum).")

    split = int(len(feat) * 0.8)
    tr, va = feat.iloc[:split], feat.iloc[split:]

    X_train = tr[FEATURE_COLS].values
    y_train = tr["label"].astype(int).values
    X_val = va[FEATURE_COLS].values
    y_val = va["label"].astype(int).values

    if train_btn:
        # LightGBM
        if lgb is None:
            st.error("lightgbm kurulu değil. pip install lightgbm")
            st.stop()

        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        lgbm.fit(X_train, y_train)
        p_lgb = lgbm.predict_proba(va[FEATURE_COLS].values)

        # LSTM
        if TORCH_OK:
            lstm_model, device = train_lstm_attention(
                X_train, y_train, X_val, y_val, seq_len=seq_len, epochs=epochs
            )
            p_lstm = predict_lstm_attention(lstm_model, device, X_val, seq_len=seq_len)
        else:
            st.warning("PyTorch yok; LSTM atlandı, sadece LightGBM kullanılacak.")
            p_lstm = p_lgb.copy()

        # Ensemble
        p_ens = 0.5 * p_lgb + 0.5 * p_lstm
        pred = p_ens.argmax(axis=1)
        pred_f, conf = apply_conf_filter(pred, p_ens, min_conf=min_conf)

        va_bt = va.copy().reset_index(drop=True)
        va_bt["pred"] = pred_f
        va_bt["conf"] = conf

        bt = run_backtest(
            va_bt,
            pred_col="pred",
            fee=fee,
            atr_mult_sl=atr_sl,
            atr_mult_tp=atr_tp
        )
        m = metrics(bt["equity"], bt["strategy_ret"], bars_per_year=365 * 24)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{m['total_return']*100:.2f}%")
        c2.metric("Sharpe", f"{m['sharpe']:.3f}")
        c3.metric("Max Drawdown", f"{m['max_drawdown']*100:.2f}%")
        c4.metric("Profit Factor", f"{m['profit_factor']:.3f}")

        st.subheader("Equity Curve")
        st.line_chart(bt.set_index("timestamp")["equity"])

        label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        last = bt.iloc[-1]
        st.subheader("Son Sinyal")
        st.write({
            "timestamp": str(last["timestamp"]),
            "signal": label_map[int(last["pred"])],
            "confidence": float(last["conf"]),
            "close": float(last["close"]),
            "atr_14": float(last["atr_14"]),
        })

        # Save artifacts locally (Streamlit runtime)
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(lgbm, "artifacts/lgbm_streamlit.pkl")
        joblib.dump(FEATURE_COLS, "artifacts/feature_cols.pkl")
        bt.to_csv("artifacts/backtest_streamlit.csv", index=False)

        st.success("Model/backtest tamamlandı. artifacts/ klasörüne kaydedildi.")
        st.download_button(
            "Backtest CSV indir",
            data=bt.to_csv(index=False).encode("utf-8"),
            file_name="backtest_streamlit.csv",
            mime="text/csv"
        )
else:
    st.info("Başlamak için OHLCV CSV yükle.")
