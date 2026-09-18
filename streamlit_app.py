from __future__ import annotations

# ============================================================
# BINANCE SPOT DEEP AI 200 - SINGLE FILE STREAMLIT APP
# ============================================================
# Kurulum:
#   pip install streamlit pandas numpy requests torch
#
# Çalıştırma:
#   streamlit run streamlit_app.py
#
# Streamlit Cloud kullanıyorsanız repo kökünde requirements.txt:
#   streamlit
#   pandas
#   numpy
#   requests
#   torch
#
# Özellikler:
# - Binance API Key / Secret Key paneli
# - Spot market data
# - 200 causal feature
# - Residual CNN + BiLSTM + BiGRU + Transformer + Attention
# - Model eğitimi
# - BUY / WAIT / SELL olasılıkları
# - Paper trading
# - Live Spot MARKET BUY / SELL
# - Bakiye, pozisyon ve işlem logları
#
# Not:
# - Withdrawal yetkisi vermeyin.
# - Önce Paper Mode kullanın.
# - AI kâr garantisi vermez.
# ============================================================

import os
import json
import time
import math
import hmac
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------- TORCH SAFE IMPORT -----------------------
TORCH_OK = True
TORCH_ERROR = ""

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    TORCH_OK = False
    TORCH_ERROR = str(exc)

# ----------------------- APP CONFIG -----------------------
st.set_page_config(
    page_title="Binance Spot Deep AI 200",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE_BASE = "https://api.binance.com"
FEATURE_COUNT = 200
DEFAULT_MODEL_DIR = "model_artifacts"

RET_H = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
WIN = [5, 8, 13, 21, 34, 55, 89, 144]
RSI_P = [5, 7, 9, 14, 21, 28, 35, 50]
ATR_P = [5, 7, 10, 14, 21, 28, 35, 50]
STO_P = [5, 7, 9, 14, 21, 28, 35, 50]
BB_P = [10, 14, 20, 28, 35, 50, 75, 100]
VOL_P = [5, 8, 13, 21, 34, 55, 89, 144]

# ----------------------- SESSION -----------------------
defaults = {
    "api_key": "",
    "api_secret": "",
    "api_ok": False,
    "account": None,
    "paper_balance": 1000.0,
    "paper_positions": {},
    "trade_log": [],
    "last_scan": [],
    "training_log": [],
    "bot_enabled": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------- STYLE -----------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: #0b111c; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
        border: 1px solid #334155;
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 14px;
    }
    .hero h1 { margin: 0; font-size: 32px; }
    .hero p { margin: 6px 0 0 0; opacity: .75; }
    .good {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #123b2d;
        color: #65e6ad;
        font-weight: 700;
    }
    .bad {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #42202b;
        color: #ff90a2;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# BINANCE CLIENT
# ============================================================

class BinanceSpotClient:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()

    def public_get(self, path: str, params=None):
        response = requests.get(
            BINANCE_BASE + path,
            params=params or {},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def server_time(self) -> int:
        data = self.public_get("/api/v3/time")
        return int(data["serverTime"])

    def signed(self, method: str, path: str, params=None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API Key / Secret Key eksik.")

        payload = dict(params or {})
        payload["timestamp"] = self.server_time()
        payload["recvWindow"] = 5000

        query = urlencode(payload, doseq=True, safe="")

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        url = f"{BINANCE_BASE}{path}?{query}&signature={signature}"

        headers = {
            "X-MBX-APIKEY": self.api_key,
        }

        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=25,
        )

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text

            raise RuntimeError(
                f"Binance HTTP {response.status_code}: {detail}"
            )

        return response.json()

    def account(self):
        return self.signed(
            "GET",
            "/api/v3/account",
        )

    def ticker24(self, symbol: str):
        return self.public_get(
            "/api/v3/ticker/24hr",
            {"symbol": symbol.upper()},
        )

    def exchange_info(self, symbol: str | None = None):
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        return self.public_get(
            "/api/v3/exchangeInfo",
            params,
        )

    def klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 1000,
        start_time=None,
        end_time=None,
    ):
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }

        if start_time is not None:
            params["startTime"] = int(start_time)

        if end_time is not None:
            params["endTime"] = int(end_time)

        return self.public_get(
            "/api/v3/klines",
            params,
        )

    def symbol_filters(self, symbol: str):
        info = self.exchange_info(symbol)

        if not info.get("symbols"):
            raise RuntimeError(
                f"Symbol bulunamadı: {symbol}"
            )

        symbol_data = info["symbols"][0]

        filters = {
            f["filterType"]: f
            for f in symbol_data.get("filters", [])
        }

        return filters, symbol_data

    @staticmethod
    def floor_step(value: float, step: float) -> float:
        if step <= 0:
            return float(value)

        return math.floor(
            value / step
        ) * step

    def normalize_quantity(self, symbol: str, qty: float) -> float:
        filters, _ = self.symbol_filters(symbol)

        lot = filters.get(
            "LOT_SIZE",
            {},
        )

        step = float(
            lot.get(
                "stepSize",
                "0.00000001",
            )
        )

        min_qty = float(
            lot.get(
                "minQty",
                "0",
            )
        )

        normalized = self.floor_step(
            qty,
            step,
        )

        if normalized < min_qty:
            raise RuntimeError(
                f"SELL miktarı minQty altında: "
                f"{normalized} < {min_qty}"
            )

        return normalized

    def market_buy_quote(self, symbol: str, quote_amount: float):
        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{float(quote_amount):.8f}",
            },
        )

    def market_sell_qty(self, symbol: str, qty: float):
        normalized = self.normalize_quantity(
            symbol,
            qty,
        )

        return self.signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol.upper(),
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{normalized:.12f}",
            },
        )

# ============================================================
# DATA HELPERS
# ============================================================

def klines_to_df(rows) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    return df.set_index(
        "open_time"
    )


def fetch_klines_history(
    symbol: str,
    interval: str,
    total: int,
) -> pd.DataFrame:
    client = BinanceSpotClient()

    rows = []
    end_time = None

    while len(rows) < total:
        limit = min(
            1000,
            total - len(rows),
        )

        batch = client.klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            end_time=end_time,
        )

        if not batch:
            break

        rows = batch + rows

        earliest = int(
            batch[0][0]
        )

        end_time = earliest - 1

        time.sleep(0.05)

    rows = rows[-total:]

    if not rows:
        raise RuntimeError(
            "Binance geçmiş veri alınamadı."
        )

    return klines_to_df(
        rows
    )

# ============================================================
# 200 FEATURE ENGINE
# ============================================================

def safe_div(a, b):
    if isinstance(
        b,
        pd.Series,
    ):
        b = b.replace(
            0,
            np.nan,
        )

    return a / b


def ema(
    series: pd.Series,
    period: int,
) -> pd.Series:
    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def rsi(
    series: pd.Series,
    period: int,
) -> pd.Series:
    delta = series.diff()

    up = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    down = (
        -delta.clip(
            upper=0
        )
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = safe_div(
        up,
        down,
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(
    df: pd.DataFrame,
    period: int,
) -> pd.Series:
    previous_close = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def zscore(
    series: pd.Series,
    period: int,
) -> pd.Series:
    mean = (
        series
        .rolling(period)
        .mean()
    )

    std = (
        series
        .rolling(period)
        .std()
    )

    return safe_div(
        series - mean,
        std,
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Eksik kolonlar: "
            f"{sorted(missing)}"
        )

    x = df.copy().sort_index()

    out = pd.DataFrame(
        index=x.index
    )

    o = x["open"]
    h = x["high"]
    l = x["low"]
    c = x["close"]
    v = x["volume"]

    qv = (
        x["quote_volume"]
        if "quote_volume" in x.columns
        else c * v
    )

    trades = (
        x["trades"]
        if "trades" in x.columns
        else pd.Series(
            0.0,
            index=x.index,
        )
    )

    taker_base = (
        x["taker_base"]
        if "taker_base" in x.columns
        else pd.Series(
            0.0,
            index=x.index,
        )
    )

    # 1-20: returns / log returns
    for n in RET_H:
        out[
            f"ret_{n}"
        ] = c.pct_change(n)

    log_close = np.log(
        c.replace(
            0,
            np.nan,
        )
    )

    for n in RET_H:
        out[
            f"logret_{n}"
        ] = log_close.diff(n)

    # 21-36: SMA / EMA deviations
    for w in WIN:
        sma = c.rolling(
            w
        ).mean()

        out[
            f"sma_dev_{w}"
        ] = safe_div(
            c,
            sma,
        ) - 1

    for w in WIN:
        em = ema(
            c,
            w,
        )

        out[
            f"ema_dev_{w}"
        ] = safe_div(
            c,
            em,
        ) - 1

    # 37-52: volatility + zscore
    ret1 = c.pct_change()

    for w in WIN:
        out[
            f"ret_std_{w}"
        ] = (
            ret1
            .rolling(w)
            .std()
        )

    for w in WIN:
        out[
            f"close_z_{w}"
        ] = zscore(
            c,
            w,
        )

    # 53-68: channel
    for w in WIN:
        rolling_min = (
            l.rolling(w).min()
        )

        rolling_max = (
            h.rolling(w).max()
        )

        width = (
            rolling_max
            -
            rolling_min
        ).replace(
            0,
            np.nan,
        )

        out[
            f"range_pos_{w}"
        ] = (
            c - rolling_min
        ) / width

        out[
            f"range_width_{w}"
        ] = (
            width
            /
            c.replace(
                0,
                np.nan,
            )
        )

    # 69-76: RSI
    for p in RSI_P:
        out[
            f"rsi_{p}"
        ] = (
            rsi(
                c,
                p,
            )
            /
            100.0
        )

    # 77-84: ATR
    for p in ATR_P:
        out[
            f"atr_pct_{p}"
        ] = (
            atr(
                x,
                p,
            )
            /
            c.replace(
                0,
                np.nan,
            )
        )

    # 85-100: stochastic
    for p in STO_P:
        low_roll = (
            l.rolling(p).min()
        )

        high_roll = (
            h.rolling(p).max()
        )

        k = (
            c - low_roll
        ) / (
            high_roll
            -
            low_roll
        ).replace(
            0,
            np.nan,
        )

        out[
            f"stoch_k_{p}"
        ] = k

        out[
            f"stoch_d_{p}"
        ] = (
            k
            .rolling(3)
            .mean()
        )

    # 101-116: Bollinger
    for p in BB_P:
        mean = (
            c.rolling(p).mean()
        )

        std = (
            c.rolling(p).std()
        )

        upper = (
            mean + 2 * std
        )

        lower = (
            mean - 2 * std
        )

        out[
            f"bb_pos_{p}"
        ] = (
            c - lower
        ) / (
            upper - lower
        ).replace(
            0,
            np.nan,
        )

        out[
            f"bb_width_{p}"
        ] = (
            upper - lower
        ) / mean.replace(
            0,
            np.nan,
        )

    # 117-140: volume
    for p in VOL_P:
        out[
            f"vol_z_{p}"
        ] = zscore(
            v,
            p,
        )

        out[
            f"vol_ratio_{p}"
        ] = safe_div(
            v,
            v.rolling(p).mean(),
        )

        out[
            f"qvol_ratio_{p}"
        ] = safe_div(
            qv,
            qv.rolling(p).mean(),
        )

    # 141-150: ROC
    for n in RET_H:
        out[
            f"roc_{n}"
        ] = (
            c
            /
            c.shift(
                n
            ).replace(
                0,
                np.nan,
            )
            -
            1
        )

    # 151-160: candle geometry
    candle_range = (
        h - l
    ).replace(
        0,
        np.nan,
    )

    body = (
        c - o
    )

    max_oc = pd.concat(
        [o, c],
        axis=1,
    ).max(
        axis=1
    )

    min_oc = pd.concat(
        [o, c],
        axis=1,
    ).min(
        axis=1
    )

    out[
        "body_pct"
    ] = (
        body
        /
        o.replace(
            0,
            np.nan,
        )
    )

    out[
        "body_to_range"
    ] = (
        body
        /
        candle_range
    )

    out[
        "abs_body_to_range"
    ] = (
        body.abs()
        /
        candle_range
    )

    out[
        "upper_wick_ratio"
    ] = (
        h - max_oc
    ) / candle_range

    out[
        "lower_wick_ratio"
    ] = (
        min_oc - l
    ) / candle_range

    out[
        "close_location"
    ] = (
        c - l
    ) / candle_range

    out[
        "open_location"
    ] = (
        o - l
    ) / candle_range

    out[
        "gap_pct"
    ] = (
        o
        /
        c.shift(1).replace(
            0,
            np.nan,
        )
        -
        1
    )

    out[
        "hl_pct"
    ] = (
        h - l
    ) / c.replace(
        0,
        np.nan,
    )

    out[
        "oc_abs_pct"
    ] = (
        c - o
    ).abs() / o.replace(
        0,
        np.nan,
    )

    # 161-172: microstructure proxies
    signed = np.sign(
        c.diff()
    ).fillna(0)

    obv = (
        signed * v
    ).cumsum()

    for w in [
        5,
        13,
        21,
        34,
    ]:
        out[
            f"obv_z_{w}"
        ] = zscore(
            obv,
            w,
        )

    for w in [
        5,
        13,
        21,
        34,
    ]:
        out[
            f"trade_z_{w}"
        ] = zscore(
            trades,
            w,
        )

    taker_ratio = safe_div(
        taker_base,
        v,
    )

    for w in [
        5,
        13,
        21,
        34,
    ]:
        out[
            f"taker_ratio_ma_{w}"
        ] = (
            taker_ratio
            .rolling(w)
            .mean()
        )

    # 173-184: MACD family
    macd_pairs = [
        (5, 13),
        (8, 21),
        (12, 26),
        (13, 34),
        (21, 55),
        (34, 89),
    ]

    for fast, slow in macd_pairs:
        macd = (
            ema(
                c,
                fast,
            )
            -
            ema(
                c,
                slow,
            )
        )

        signal = ema(
            macd,
            9,
        )

        out[
            f"macd_norm_{fast}_{slow}"
        ] = (
            macd
            /
            c.replace(
                0,
                np.nan,
            )
        )

        out[
            f"macd_hist_{fast}_{slow}"
        ] = (
            macd - signal
        ) / c.replace(
            0,
            np.nan,
        )

    # 185-192: acceleration
    for n in [
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
    ]:
        r = c.pct_change(
            n
        )

        out[
            f"accel_{n}"
        ] = (
            r
            -
            r.shift(n)
        )

    # 193-198: time
    if isinstance(
        out.index,
        pd.DatetimeIndex,
    ):
        minute = (
            out.index.minute
            +
            out.index.hour * 60
        )

        dow = (
            out.index.dayofweek
        )

        out[
            "tod_sin"
        ] = np.sin(
            2 * np.pi
            *
            minute / 1440
        )

        out[
            "tod_cos"
        ] = np.cos(
            2 * np.pi
            *
            minute / 1440
        )

        out[
            "dow_sin"
        ] = np.sin(
            2 * np.pi
            *
            dow / 7
        )

        out[
            "dow_cos"
        ] = np.cos(
            2 * np.pi
            *
            dow / 7
        )

        out[
            "hour_sin"
        ] = np.sin(
            2 * np.pi
            *
            out.index.hour / 24
        )

        out[
            "hour_cos"
        ] = np.cos(
            2 * np.pi
            *
            out.index.hour / 24
        )
    else:
        for name in [
            "tod_sin",
            "tod_cos",
            "dow_sin",
            "dow_cos",
            "hour_sin",
            "hour_cos",
        ]:
            out[name] = 0.0

    # 199-200: cross features
    out[
        "trend_volume_interaction"
    ] = (
        out[
            "ema_dev_21"
        ]
        *
        out[
            "vol_ratio_21"
        ]
    )

    out[
        "momentum_volatility_interaction"
    ] = (
        out[
            "ret_5"
        ]
        *
        out[
            "atr_pct_14"
        ]
    )

    # Tam olarak 200 feature
    base_cols = list(
        out.columns
    )

    i = 0

    while (
        out.shape[1]
        <
        FEATURE_COUNT
    ):
        a = base_cols[
            i % len(base_cols)
        ]

        b = base_cols[
            (
                i * 7 + 11
            )
            %
            len(base_cols)
        ]

        out[
            f"cross_{i:03d}"
        ] = (
            out[a]
            *
            out[b]
        )

        i += 1

    out = out.iloc[
        :,
        :FEATURE_COUNT,
    ]

    out = out.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    out = (
        out
        .ffill()
        .fillna(0.0)
        .clip(
            -1000,
            1000,
        )
        .astype(
            "float32"
        )
    )

    if (
        out.shape[1]
        != FEATURE_COUNT
    ):
        raise RuntimeError(
            f"Feature count mismatch: "
            f"{out.shape[1]}"
        )

    return out

# ============================================================
# DEEP LEARNING MODEL
# ============================================================

if TORCH_OK:

    class ResidualTemporalBlock(
        nn.Module
    ):
        def __init__(
            self,
            channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ):
            super().__init__()

            padding = (
                (
                    kernel_size - 1
                )
                *
                dilation
            ) // 2

            self.net = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(
                    channels
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(
                    channels
                ),
                nn.GELU(),
            )

            self.dropout = (
                nn.Dropout(
                    dropout
                )
            )

        def forward(
            self,
            x,
        ):
            return self.dropout(
                self.net(x)
                +
                x
            )


    class DeepSpotAI(
        nn.Module
    ):
        def __init__(
            self,
            n_features: int = FEATURE_COUNT,
            d_model: int = 160,
            lstm_hidden: int = 112,
            gru_hidden: int = 112,
            heads: int = 8,
            dropout: float = 0.18,
            classes: int = 3,
        ):
            super().__init__()

            self.input_norm = (
                nn.LayerNorm(
                    n_features
                )
            )

            self.project = (
                nn.Sequential(
                    nn.Linear(
                        n_features,
                        256,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        dropout
                    ),
                    nn.Linear(
                        256,
                        d_model,
                    ),
                    nn.GELU(),
                )
            )

            self.temporal = (
                nn.Sequential(
                    ResidualTemporalBlock(
                        d_model,
                        3,
                        1,
                        dropout,
                    ),
                    ResidualTemporalBlock(
                        d_model,
                        5,
                        2,
                        dropout,
                    ),
                    ResidualTemporalBlock(
                        d_model,
                        7,
                        4,
                        dropout,
                    ),
                )
            )

            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )

            self.lstm_proj = (
                nn.Sequential(
                    nn.Linear(
                        lstm_hidden * 2,
                        d_model,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(
                        d_model
                    ),
                )
            )

            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=gru_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )

            self.gru_proj = (
                nn.Sequential(
                    nn.Linear(
                        gru_hidden * 2,
                        d_model,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(
                        d_model
                    ),
                )
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

            self.transformer = (
                nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=2,
                )
            )

            self.attention = (
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            self.attn_norm = (
                nn.LayerNorm(
                    d_model
                )
            )

            self.gate = (
                nn.Sequential(
                    nn.Linear(
                        d_model * 3,
                        d_model,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        d_model,
                        3,
                    ),
                    nn.Softmax(
                        dim=-1
                    ),
                )
            )

            self.classifier = (
                nn.Sequential(
                    nn.Linear(
                        d_model * 3,
                        320,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(
                        320
                    ),
                    nn.Dropout(
                        dropout
                    ),
                    nn.Linear(
                        320,
                        160,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        dropout
                    ),
                    nn.Linear(
                        160,
                        64,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        64,
                        classes,
                    ),
                )
            )

        def forward(
            self,
            x,
        ):
            x = self.input_norm(
                x
            )

            x = self.project(
                x
            )

            temporal = (
                self.temporal(
                    x.transpose(
                        1,
                        2,
                    )
                )
                .transpose(
                    1,
                    2,
                )
            )

            x = x + temporal

            lstm_out, _ = (
                self.lstm(x)
            )

            lstm_out = (
                self.lstm_proj(
                    lstm_out
                )
            )

            x = x + lstm_out

            gru_out, _ = (
                self.gru(x)
            )

            gru_out = (
                self.gru_proj(
                    gru_out
                )
            )

            x = x + gru_out

            x = self.transformer(
                x
            )

            attn_out, _ = (
                self.attention(
                    x,
                    x,
                    x,
                    need_weights=False,
                )
            )

            x = self.attn_norm(
                x + attn_out
            )

            last_pool = (
                x[:, -1, :]
            )

            mean_pool = (
                x.mean(
                    dim=1
                )
            )

            max_pool = (
                x.max(
                    dim=1
                ).values
            )

            gate_input = torch.cat(
                [
                    last_pool,
                    mean_pool,
                    max_pool,
                ],
                dim=-1,
            )

            weights = self.gate(
                gate_input
            )

            weighted_last = (
                last_pool
                *
                weights[:, 0:1]
            )

            weighted_mean = (
                mean_pool
                *
                weights[:, 1:2]
            )

            weighted_max = (
                max_pool
                *
                weights[:, 2:3]
            )

            fused = torch.cat(
                [
                    weighted_last,
                    weighted_mean,
                    weighted_max,
                ],
                dim=-1,
            )

            return self.classifier(
                fused
            )


    class SequenceDataset(
        Dataset
    ):
        def __init__(
            self,
            X,
            y,
            sequence_length: int,
        ):
            self.X = X
            self.y = y
            self.sequence_length = (
                sequence_length
            )

        def __len__(
            self,
        ):
            return max(
                0,
                len(self.y)
                -
                self.sequence_length
                +
                1,
            )

        def __getitem__(
            self,
            index,
        ):
            end = (
                index
                +
                self.sequence_length
                -
                1
            )

            seq = self.X[
                index:
                index
                +
                self.sequence_length
            ]

            target = self.y[
                end
            ]

            return (
                torch.tensor(
                    seq,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    target,
                    dtype=torch.long,
                ),
            )

# ============================================================
# LABELS / TRAINING
# ============================================================

def build_labels(
    close: pd.Series,
    horizon: int,
    threshold: float,
):
    future_return = (
        close.shift(
            -horizon
        )
        /
        close
        -
        1
    )

    labels = np.ones(
        len(close),
        dtype=np.int64,
    )

    labels[
        future_return
        <
        -threshold
    ] = 0

    labels[
        future_return
        >
        threshold
    ] = 2

    return labels


def compute_class_weights(
    y: np.ndarray,
):
    counts = np.bincount(
        y,
        minlength=3,
    ).astype(
        np.float64
    )

    counts[
        counts == 0
    ] = 1

    weights = (
        counts.sum()
        /
        (
            3
            *
            counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def train_model(
    symbol: str,
    interval: str,
    bars: int,
    sequence_length: int,
    horizon: int,
    threshold: float,
    epochs: int,
    batch_size: int,
    out_dir: str,
    learning_rate: float,
):
    if not TORCH_OK:
        raise RuntimeError(
            "PyTorch kurulu değil."
        )

    out = Path(
        out_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    st.session_state.training_log = []

    def log(
        text: str,
    ):
        st.session_state.training_log.append(
            text
        )

    log(
        "Binance geçmiş verisi indiriliyor..."
    )

    df = fetch_klines_history(
        symbol,
        interval,
        bars,
    )

    log(
        f"{len(df)} mum indirildi."
    )

    features = build_features(
        df
    )

    log(
        f"{features.shape[1]} özellik üretildi."
    )

    labels = build_labels(
        df["close"],
        horizon,
        threshold,
    )

    usable = (
        len(df)
        -
        horizon
    )

    features = (
        features.iloc[
            :usable
        ]
    )

    labels = labels[
        :usable
    ]

    train_end = int(
        len(features)
        *
        0.70
    )

    val_end = int(
        len(features)
        *
        0.85
    )

    train_features = (
        features.iloc[
            :train_end
        ]
    )

    mean = (
        train_features
        .mean()
        .to_numpy(
            np.float32
        )
    )

    std = (
        train_features
        .std()
        .replace(
            0,
            1
        )
        .to_numpy(
            np.float32
        )
    )

    X = (
        (
            features.to_numpy(
                np.float32
            )
            -
            mean
        )
        /
        std
    ).clip(
        -8,
        8,
    )

    X_train = X[
        :train_end
    ]

    y_train = labels[
        :train_end
    ]

    X_val = X[
        train_end
        -
        sequence_length:
        val_end
    ]

    y_val = labels[
        train_end
        -
        sequence_length:
        val_end
    ]

    X_test = X[
        val_end
        -
        sequence_length:
    ]

    y_test = labels[
        val_end
        -
        sequence_length:
    ]

    train_ds = SequenceDataset(
        X_train,
        y_train,
        sequence_length,
    )

    val_ds = SequenceDataset(
        X_val,
        y_val,
        sequence_length,
    )

    test_ds = SequenceDataset(
        X_test,
        y_test,
        sequence_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log(
        f"Device: {device}"
    )

    model = DeepSpotAI(
        n_features=FEATURE_COUNT
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
    )

    class_weights = (
        compute_class_weights(
            y_train
        )
        .to(
            device
        )
    )

    loss_fn = (
        nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.03,
        )
    )

    best_val_loss = (
        float("inf")
    )

    patience = 0
    max_patience = 4

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = loss_fn(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_loss_total += (
                loss.item()
            )

            preds = logits.argmax(
                dim=1
            )

            train_correct += (
                preds == yb
            ).sum().item()

            train_total += (
                yb.numel()
            )

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )

                logits = model(
                    xb
                )

                loss = loss_fn(
                    logits,
                    yb,
                )

                val_loss_total += (
                    loss.item()
                )

                preds = logits.argmax(
                    dim=1
                )

                val_correct += (
                    preds == yb
                ).sum().item()

                val_total += (
                    yb.numel()
                )

        train_loss = (
            train_loss_total
            /
            max(
                1,
                len(train_loader),
            )
        )

        val_loss = (
            val_loss_total
            /
            max(
                1,
                len(val_loader),
            )
        )

        train_acc = (
            train_correct
            /
            max(
                1,
                train_total,
            )
        )

        val_acc = (
            val_correct
            /
            max(
                1,
                val_total,
            )
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        log(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.7f}"
        )

        if (
            val_loss
            <
            best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            patience = 0

            torch.save(
                model.state_dict(),
                out / "model.pt",
            )

            np.savez(
                out / "scaler.npz",
                mean=mean,
                std=std,
            )

            metadata = {
                "symbol": symbol,
                "interval": interval,
                "sequence_length": sequence_length,
                "horizon": horizon,
                "threshold": threshold,
                "feature_count": FEATURE_COUNT,
                "features": list(
                    features.columns
                ),
                "classes": [
                    "SELL",
                    "WAIT",
                    "BUY",
                ],
                "architecture": (
                    "ResidualCNN+BiLSTM+BiGRU+"
                    "Transformer+MultiHeadAttention"
                ),
            }

            (
                out / "meta.json"
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

            log(
                "En iyi model kaydedildi."
            )

        else:
            patience += 1

            if (
                patience
                >=
                max_patience
            ):
                log(
                    "Early stopping."
                )
                break

    model.load_state_dict(
        torch.load(
            out / "model.pt",
            map_location=device,
        )
    )

    model.eval()

    test_correct = 0
    test_total = 0

    confusion = np.zeros(
        (
            3,
            3,
        ),
        dtype=int,
    )

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            logits = model(
                xb
            )

            preds = logits.argmax(
                dim=1
            )

            test_correct += (
                preds == yb
            ).sum().item()

            test_total += (
                yb.numel()
            )

            for true, pred in zip(
                yb.cpu().numpy(),
                preds.cpu().numpy(),
            ):
                confusion[
                    int(true),
                    int(pred),
                ] += 1

    test_acc = (
        test_correct
        /
        max(
            1,
            test_total,
        )
    )

    log(
        f"Test accuracy: "
        f"{test_acc:.4f}"
    )

    log(
        "Eğitim tamamlandı."
    )

    return {
        "test_accuracy": test_acc,
        "confusion": confusion.tolist(),
        "model_dir": str(
            out.resolve()
        ),
    }

# ============================================================
# MODEL LOAD / INFERENCE
# ============================================================

_MODEL_CACHE = {}


def clear_model_cache():
    _MODEL_CACHE.clear()


def load_model_artifacts(
    model_dir: str,
):
    if not TORCH_OK:
        raise RuntimeError(
            "PyTorch kurulu değil."
        )

    key = str(
        Path(
            model_dir
        ).resolve()
    )

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[
            key
        ]

    path = Path(
        model_dir
    )

    meta_path = (
        path
        /
        "meta.json"
    )

    model_path = (
        path
        /
        "model.pt"
    )

    scaler_path = (
        path
        /
        "scaler.npz"
    )

    if not (
        meta_path.exists()
        and
        model_path.exists()
        and
        scaler_path.exists()
    ):
        raise RuntimeError(
            "Model dosyaları bulunamadı."
        )

    metadata = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    scaler = np.load(
        scaler_path
    )

    mean = scaler[
        "mean"
    ]

    std = scaler[
        "std"
    ]

    model = DeepSpotAI(
        n_features=int(
            metadata[
                "feature_count"
            ]
        )
    )

    state = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state
    )

    model.eval()

    result = (
        model,
        metadata,
        mean,
        std,
    )

    _MODEL_CACHE[
        key
    ] = result

    return result


def predict_symbol(
    client: BinanceSpotClient,
    symbol: str,
    interval: str,
    model_dir: str,
):
    (
        model,
        metadata,
        mean,
        std,
    ) = load_model_artifacts(
        model_dir
    )

    rows = client.klines(
        symbol,
        interval,
        limit=1000,
    )

    df = klines_to_df(
        rows
    )

    features = build_features(
        df
    )

    features = features.reindex(
        columns=metadata[
            "features"
        ],
        fill_value=0.0,
    )

    values = (
        features
        .to_numpy(
            np.float32
        )
    )

    normalized = (
        (
            values
            -
            mean
        )
        /
        std
    ).clip(
        -8,
        8,
    )

    seq_len = int(
        metadata[
            "sequence_length"
        ]
    )

    if (
        len(normalized)
        <
        seq_len
    ):
        raise RuntimeError(
            "Yetersiz mum."
        )

    sequence = normalized[
        -seq_len:
    ]

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )

    with torch.no_grad():
        logits = model(
            tensor
        )

        probs = torch.softmax(
            logits,
            dim=1,
        ).squeeze(
            0
        ).numpy()

    return {
        "SELL": float(
            probs[0]
            *
            100
        ),
        "WAIT": float(
            probs[1]
            *
            100
        ),
        "BUY": float(
            probs[2]
            *
            100
        ),
    }, df

# ============================================================
# PAPER TRADING
# ============================================================

def paper_buy(
    symbol: str,
    usdt: float,
    price: float,
):
    if (
        st.session_state
        .paper_balance
        <
        usdt
    ):
        raise RuntimeError(
            "Paper bakiye yetersiz."
        )

    qty = (
        usdt
        /
        price
    )

    old = (
        st.session_state
        .paper_positions
        .get(
            symbol,
            {
                "qty": 0.0,
                "avg": 0.0,
            },
        )
    )

    new_qty = (
        old["qty"]
        +
        qty
    )

    new_avg = (
        (
            old["qty"]
            *
            old["avg"]
        )
        +
        (
            qty
            *
            price
        )
    ) / new_qty

    st.session_state[
        "paper_positions"
    ][symbol] = {
        "qty": new_qty,
        "avg": new_avg,
    }

    st.session_state[
        "paper_balance"
    ] -= usdt

    st.session_state[
        "trade_log"
    ].append(
        {
            "time": str(
                datetime.now()
            ),
            "mode": "PAPER",
            "side": "BUY",
            "symbol": symbol,
            "price": price,
            "quote": usdt,
        }
    )


def paper_sell_all(
    symbol: str,
    price: float,
):
    pos = (
        st.session_state
        .paper_positions
        .get(
            symbol
        )
    )

    if not pos:
        raise RuntimeError(
            "Paper pozisyon yok."
        )

    quote = (
        pos["qty"]
        *
        price
    )

    pnl = (
        price
        -
        pos["avg"]
    ) * pos["qty"]

    st.session_state[
        "paper_balance"
    ] += quote

    st.session_state[
        "trade_log"
    ].append(
        {
            "time": str(
                datetime.now()
            ),
            "mode": "PAPER",
            "side": "SELL",
            "symbol": symbol,
            "price": price,
            "quote": quote,
            "pnl": pnl,
        }
    )

    del st.session_state[
        "paper_positions"
    ][symbol]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "🔐 Binance API"
    )

    api_key = st.text_input(
        "API Key",
        value=st.session_state[
            "api_key"
        ],
        type="password",
    )

    api_secret = st.text_input(
        "Secret Key",
        value=st.session_state[
            "api_secret"
        ],
        type="password",
    )

    c1, c2 = st.columns(
        2
    )

    with c1:
        if st.button(
            "Bağlan",
            use_container_width=True,
        ):
            try:
                test_client = (
                    BinanceSpotClient(
                        api_key,
                        api_secret,
                    )
                )

                account = (
                    test_client.account()
                )

                st.session_state[
                    "api_key"
                ] = api_key

                st.session_state[
                    "api_secret"
                ] = api_secret

                st.session_state[
                    "account"
                ] = account

                st.session_state[
                    "api_ok"
                ] = True

                st.success(
                    "Binance API bağlandı."
                )

            except Exception as exc:
                st.session_state[
                    "api_ok"
                ] = False

                st.error(
                    str(exc)
                )

    with c2:
        if st.button(
            "Temizle",
            use_container_width=True,
        ):
            st.session_state[
                "api_key"
            ] = ""

            st.session_state[
                "api_secret"
            ] = ""

            st.session_state[
                "api_ok"
            ] = False

            st.session_state[
                "account"
            ] = None

            st.rerun()

    st.divider()

    st.header(
        "⚙️ Trading"
    )

    mode = st.radio(
        "İşlem modu",
        [
            "Paper",
            "Live",
        ],
        index=0,
    )

    live_confirm = False

    if mode == "Live":
        st.warning(
            "Live mod gerçek Spot emir gönderir."
        )

        live_confirm = (
            st.checkbox(
                "Gerçek emirleri etkinleştir"
            )
        )

    symbols_text = (
        st.text_area(
            "USDT pariteleri",
            "BTCUSDT\nETHUSDT\nBNBUSDT\nSOLUSDT\nXRPUSDT",
            height=120,
        )
    )

    symbols = [
        item.strip().upper()
        for item
        in symbols_text.splitlines()
        if item.strip()
    ]

    interval = st.selectbox(
        "Timeframe",
        [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
        ],
        index=2,
    )

    model_dir = st.text_input(
        "Model klasörü",
        DEFAULT_MODEL_DIR,
    )

    order_usdt = (
        st.number_input(
            "İşlem başına USDT",
            min_value=5.0,
            value=20.0,
            step=5.0,
        )
    )

    buy_threshold = (
        st.slider(
            "BUY güven eşiği %",
            50,
            99,
            75,
        )
    )

    sell_threshold = (
        st.slider(
            "SELL güven eşiği %",
            50,
            99,
            75,
        )
    )

    max_positions = (
        st.slider(
            "Maks. açık Paper pozisyon",
            1,
            20,
            5,
        )
    )

client = BinanceSpotClient(
    st.session_state[
        "api_key"
    ],
    st.session_state[
        "api_secret"
    ],
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Binance Spot Deep AI 200</h1>
        <p>
        200 feature • Residual CNN • BiLSTM • BiGRU •
        Transformer • Multi-Head Attention • Paper / Live
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TORCH_OK:
    st.error(
        "PyTorch (torch) kurulu değil. "
        "Streamlit Cloud kullanıyorsan repo köküne "
        "`requirements.txt` ekleyip içine `torch` yaz."
    )

    st.code(
        "streamlit\n"
        "pandas\n"
        "numpy\n"
        "requests\n"
        "torch\n",
        language="text",
    )

m1, m2, m3, m4 = st.columns(
    4
)

m1.metric(
    "API",
    (
        "Bağlı"
        if st.session_state[
            "api_ok"
        ]
        else "Bağlı değil"
    ),
)

m2.metric(
    "AI Feature",
    FEATURE_COUNT,
)

m3.metric(
    "Paper bakiye",
    f"{st.session_state['paper_balance']:.2f} USDT",
)

m4.metric(
    "Açık Paper pozisyon",
    len(
        st.session_state[
            "paper_positions"
        ]
    ),
)

# ============================================================
# ACCOUNT
# ============================================================

if (
    st.session_state[
        "api_ok"
    ]
    and
    st.session_state[
        "account"
    ]
):
    with st.expander(
        "💳 Binance Hesabı"
    ):
        balances = []

        for balance in (
            st.session_state[
                "account"
            ]
            .get(
                "balances",
                [],
            )
        ):
            free = float(
                balance[
                    "free"
                ]
            )

            locked = float(
                balance[
                    "locked"
                ]
            )

            if (
                free > 0
                or
                locked > 0
            ):
                balances.append(
                    {
                        "Asset":
                        balance[
                            "asset"
                        ],
                        "Free":
                        free,
                        "Locked":
                        locked,
                        "Total":
                        free + locked,
                    }
                )

        if balances:
            st.dataframe(
                pd.DataFrame(
                    balances
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Gösterilecek bakiye yok."
            )

# ============================================================
# TABS
# ============================================================

(
    tab_scan,
    tab_order,
    tab_train,
    tab_model,
    tab_positions,
    tab_logs,
    tab_security,
) = st.tabs(
    [
        "📊 AI Tarayıcı",
        "💸 Emir",
        "🏋 Eğitim",
        "🧠 Model",
        "💼 Pozisyonlar",
        "🧾 Loglar",
        "🔐 Güvenlik",
    ]
)

# ============================================================
# SCANNER
# ============================================================

with tab_scan:
    st.subheader(
        "Deep Learning Coin Tarayıcı"
    )

    if not TORCH_OK:
        st.warning(
            "Tarayıcı için torch kurulmalı."
        )

    elif not Path(
        model_dir,
        "model.pt",
    ).exists():
        st.warning(
            "Model henüz eğitilmedi. "
            "Önce Eğitim sekmesine git."
        )

    if st.button(
        "🧠 AI Taramayı Başlat",
        type="primary",
        disabled=(
            not TORCH_OK
        ),
    ):
        scan_rows = []

        for symbol in (
            symbols[:20]
        ):
            try:
                probs, _ = (
                    predict_symbol(
                        client,
                        symbol,
                        interval,
                        model_dir,
                    )
                )

                ticker = (
                    client.ticker24(
                        symbol
                    )
                )

                buy = probs[
                    "BUY"
                ]

                sell = probs[
                    "SELL"
                ]

                wait = probs[
                    "WAIT"
                ]

                if (
                    buy
                    >=
                    buy_threshold
                    and
                    buy > sell
                ):
                    signal = "BUY"

                elif (
                    sell
                    >=
                    sell_threshold
                    and
                    sell > buy
                ):
                    signal = "SELL"

                else:
                    signal = "WAIT"

                scan_rows.append(
                    {
                        "Symbol":
                        symbol,
                        "Price":
                        float(
                            ticker[
                                "lastPrice"
                            ]
                        ),
                        "24h %":
                        float(
                            ticker[
                                "priceChangePercent"
                            ]
                        ),
                        "BUY %":
                        round(
                            buy,
                            2,
                        ),
                        "WAIT %":
                        round(
                            wait,
                            2,
                        ),
                        "SELL %":
                        round(
                            sell,
                            2,
                        ),
                        "Signal":
                        signal,
                    }
                )

            except Exception as exc:
                st.warning(
                    f"{symbol}: "
                    f"{exc}"
                )

        st.session_state[
            "last_scan"
        ] = scan_rows

    if st.session_state[
        "last_scan"
    ]:
        scan_df = (
            pd.DataFrame(
                st.session_state[
                    "last_scan"
                ]
            )
            .sort_values(
                "BUY %",
                ascending=False,
            )
        )

        st.dataframe(
            scan_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            scan_df
            .set_index(
                "Symbol"
            )[
                [
                    "BUY %",
                    "WAIT %",
                    "SELL %",
                ]
            ]
        )

# ============================================================
# ORDER PANEL
# ============================================================

with tab_order:
    st.subheader(
        "Manuel + AI Destekli Emir"
    )

    if not symbols:
        st.info(
            "Parite gir."
        )
    else:
        selected_symbol = (
            st.selectbox(
                "Parite",
                symbols,
                key="trade_symbol",
            )
        )

        amount = (
            st.number_input(
                "USDT miktarı",
                min_value=5.0,
                value=float(
                    order_usdt
                ),
                step=5.0,
            )
        )

        try:
            ticker = (
                client.ticker24(
                    selected_symbol
                )
            )

            price = float(
                ticker[
                    "lastPrice"
                ]
            )

            st.metric(
                "Anlık fiyat",
                f"{price:.8f}",
            )

        except Exception as exc:
            price = None

            st.error(
                str(exc)
            )

        if (
            TORCH_OK
            and
            Path(
                model_dir,
                "model.pt",
            ).exists()
        ):
            try:
                probs, _ = (
                    predict_symbol(
                        client,
                        selected_symbol,
                        interval,
                        model_dir,
                    )
                )

                a, b, c = st.columns(
                    3
                )

                a.metric(
                    "BUY",
                    f"%{probs['BUY']:.2f}",
                )

                b.metric(
                    "WAIT",
                    f"%{probs['WAIT']:.2f}",
                )

                c.metric(
                    "SELL",
                    f"%{probs['SELL']:.2f}",
                )

            except Exception as exc:
                st.warning(
                    str(exc)
                )

        left, right = st.columns(
            2
        )

        with left:
            if st.button(
                "🟢 BUY",
                use_container_width=True,
            ):
                try:
                    if price is None:
                        raise RuntimeError(
                            "Fiyat alınamadı."
                        )

                    if mode == "Paper":
                        if (
                            len(
                                st.session_state[
                                    "paper_positions"
                                ]
                            )
                            >=
                            max_positions
                            and
                            selected_symbol
                            not in
                            st.session_state[
                                "paper_positions"
                            ]
                        ):
                            raise RuntimeError(
                                "Maksimum açık "
                                "Paper pozisyon sayısına ulaşıldı."
                            )

                        paper_buy(
                            selected_symbol,
                            amount,
                            price,
                        )

                    else:
                        if not st.session_state[
                            "api_ok"
                        ]:
                            raise RuntimeError(
                                "API bağlantısı yok."
                            )

                        if not live_confirm:
                            raise RuntimeError(
                                "Live işlem onayı yok."
                            )

                        result = (
                            client
                            .market_buy_quote(
                                selected_symbol,
                                amount,
                            )
                        )

                        st.session_state[
                            "trade_log"
                        ].append(
                            {
                                "time":
                                str(
                                    datetime.now()
                                ),
                                "mode":
                                "LIVE",
                                "side":
                                "BUY",
                                "symbol":
                                selected_symbol,
                                "result":
                                str(
                                    result
                                ),
                            }
                        )

                    st.success(
                        "BUY işlendi."
                    )

                except Exception as exc:
                    st.error(
                        str(exc)
                    )

        with right:
            sell_qty = (
                st.number_input(
                    "Live SELL coin miktarı",
                    min_value=0.0,
                    value=0.0,
                    format="%.8f",
                )
            )

            if st.button(
                "🔴 SELL",
                use_container_width=True,
            ):
                try:
                    if price is None:
                        raise RuntimeError(
                            "Fiyat alınamadı."
                        )

                    if mode == "Paper":
                        paper_sell_all(
                            selected_symbol,
                            price,
                        )

                    else:
                        if not st.session_state[
                            "api_ok"
                        ]:
                            raise RuntimeError(
                                "API bağlantısı yok."
                            )

                        if not live_confirm:
                            raise RuntimeError(
                                "Live işlem onayı yok."
                            )

                        if (
                            sell_qty
                            <=
                            0
                        ):
                            raise RuntimeError(
                                "SELL miktarı "
                                "0'dan büyük olmalı."
                            )

                        result = (
                            client
                            .market_sell_qty(
                                selected_symbol,
                                sell_qty,
                            )
                        )

                        st.session_state[
                            "trade_log"
                        ].append(
                            {
                                "time":
                                str(
                                    datetime.now()
                                ),
                                "mode":
                                "LIVE",
                                "side":
                                "SELL",
                                "symbol":
                                selected_symbol,
                                "result":
                                str(
                                    result
                                ),
                            }
                        )

                    st.success(
                        "SELL işlendi."
                    )

                except Exception as exc:
                    st.error(
                        str(exc)
                    )

# ============================================================
# TRAINING TAB
# ============================================================

with tab_train:
    st.subheader(
        "Model Eğitimi"
    )

    if not TORCH_OK:
        st.error(
            "PyTorch kurulmadan eğitim yapılamaz."
        )

    train_symbol = (
        st.text_input(
            "Eğitim paritesi",
            "BTCUSDT",
        )
    )

    tc1, tc2, tc3 = (
        st.columns(
            3
        )
    )

    with tc1:
        train_interval = (
            st.selectbox(
                "Eğitim timeframe",
                [
                    "1m",
                    "3m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                ],
                index=2,
                key="train_interval",
            )
        )

        train_bars = (
            st.number_input(
                "Geçmiş mum sayısı",
                min_value=3000,
                max_value=100000,
                value=10000,
                step=1000,
            )
        )

    with tc2:
        seq_len = (
            st.number_input(
                "Sequence uzunluğu",
                min_value=32,
                max_value=256,
                value=96,
                step=16,
            )
        )

        horizon = (
            st.number_input(
                "Tahmin horizon",
                min_value=1,
                max_value=24,
                value=3,
            )
        )

    with tc3:
        threshold = (
            st.number_input(
                "Label threshold",
                min_value=0.0005,
                max_value=0.05,
                value=0.0025,
                step=0.0005,
                format="%.4f",
            )
        )

        epochs = (
            st.number_input(
                "Epoch",
                min_value=1,
                max_value=50,
                value=8,
            )
        )

    batch_size = (
        st.selectbox(
            "Batch size",
            [
                32,
                64,
                128,
                256,
            ],
            index=2,
        )
    )

    lr = (
        st.number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=0.005,
            value=0.0002,
            step=0.00001,
            format="%.5f",
        )
    )

    if st.button(
        "🏋 Modeli Eğit",
        type="primary",
        disabled=(
            not TORCH_OK
        ),
    ):
        try:
            with st.spinner(
                "Model eğitiliyor..."
            ):
                result = train_model(
                    symbol=train_symbol.upper(),
                    interval=train_interval,
                    bars=int(
                        train_bars
                    ),
                    sequence_length=int(
                        seq_len
                    ),
                    horizon=int(
                        horizon
                    ),
                    threshold=float(
                        threshold
                    ),
                    epochs=int(
                        epochs
                    ),
                    batch_size=int(
                        batch_size
                    ),
                    out_dir=model_dir,
                    learning_rate=float(
                        lr
                    ),
                )

                clear_model_cache()

            st.success(
                "Model eğitildi."
            )

            st.json(
                result
            )

        except Exception as exc:
            st.error(
                str(exc)
            )

    if st.session_state[
        "training_log"
    ]:
        st.text_area(
            "Eğitim logu",
            "\n".join(
                st.session_state[
                    "training_log"
                ]
            ),
            height=300,
        )

# ============================================================
# MODEL TAB
# ============================================================

with tab_model:
    st.subheader(
        "Deep Learning Mimarisi"
    )

    st.code(
        """
200 Causal Market Features
        ↓
LayerNorm
        ↓
Dense 200 → 256 → 160
        ↓
Residual Temporal CNN
kernel 3 / 5 / 7
dilation 1 / 2 / 4
        ↓
2-Layer Bidirectional LSTM
        ↓
2-Layer Bidirectional GRU
        ↓
2-Layer Transformer Encoder
        ↓
8-Head Self Attention
        ↓
Gated Last / Mean / Max Pooling
        ↓
Dense 480 → 320 → 160 → 64
        ↓
SELL / WAIT / BUY
        """,
        language="text",
    )

    st.info(
        "Modelin daha büyük olması tek başına "
        "daha doğru tahmin anlamına gelmez. "
        "Asıl kalite eğitim verisi, rejim çeşitliliği, "
        "komisyon/slippage ve walk-forward testten gelir."
    )

# ============================================================
# POSITIONS
# ============================================================

with tab_positions:
    st.subheader(
        "Paper Pozisyonları"
    )

    rows = []

    for (
        symbol,
        pos,
    ) in (
        st.session_state[
            "paper_positions"
        ].items()
    ):
        try:
            current = float(
                client
                .ticker24(
                    symbol
                )[
                    "lastPrice"
                ]
            )
        except Exception:
            current = None

        pnl = (
            (
                current
                -
                pos["avg"]
            )
            *
            pos["qty"]
            if current is not None
            else None
        )

        rows.append(
            {
                "Symbol":
                symbol,
                "Qty":
                pos["qty"],
                "Avg":
                pos["avg"],
                "Current":
                current,
                "PnL":
                pnl,
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Açık Paper pozisyon yok."
        )

# ============================================================
# LOGS
# ============================================================

with tab_logs:
    st.subheader(
        "İşlem Geçmişi"
    )

    if st.session_state[
        "trade_log"
    ]:
        log_df = (
            pd.DataFrame(
                st.session_state[
                    "trade_log"
                ]
            )
        )

        st.dataframe(
            log_df.iloc[
                ::-1
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "CSV indir",
            log_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                "binance_ai_trades.csv"
            ),
            mime="text/csv",
        )
    else:
        st.info(
            "Henüz işlem logu yok."
        )

# ============================================================
# SECURITY
# ============================================================

with tab_security:
    st.subheader(
        "API Güvenliği"
    )

    st.markdown(
        """
- Binance API Secret Key'i kimseyle paylaşma.
- **Withdrawal / çekim yetkisini açma.**
- Sadece gerekli Spot işlem yetkisini kullan.
- Mümkünse API anahtarını sabit IP ile sınırla.
- İlk testleri Paper Mode ile yap.
- Live Mode gerçek para kullanır.
- Derin öğrenme modeli kâr garantisi vermez.
        """
    )

st.caption(
    "Binance Spot Deep AI 200 • "
    "AI tahminleri finansal sonuç garantisi değildir."
)
