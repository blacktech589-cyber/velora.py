from __future__ import annotations

# =============================================================================
# BINANCE TR SPOT DEEP AI SCALPER — SINGLE FILE
# =============================================================================
# This build uses Binance TR official endpoints rather than api.binance.com.
#
# Official REST base:
#     https://www.binance.tr
#
# Official Binance TR market-data endpoints:
#     type 1: https://api.binance.me
#     type 3: https://cloudme-tr.2meta.app
#
# Core capabilities:
# - Binance TR API Key / Secret Key authentication
# - API diagnostics: public time -> symbols -> signed account -> canTrade
# - BTC_USDT style symbols
# - 200 causal market features
# - Multi-scale residual TCN
# - BiLSTM
# - BiGRU
# - Transformer Encoder
# - Multi-head self-attention
# - Learned branch gating
# - Learned temporal pooling
# - MC-dropout uncertainty
# - Chronological train/validation/test split
# - Class-balanced training
# - Early stopping and LR scheduling
# - Multi-timeframe model support
# - Paper trading
# - Live Binance TR Spot MARKET BUY / SELL
# - Percentage-profit scalping exit
# - No automatic stop-loss in this build
# =============================================================================

import json
import math
import time
import hmac
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

TORCH_OK = True
TORCH_ERROR = ""
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    TORCH_OK = False
    TORCH_ERROR = str(exc)

# =============================================================================
# STREAMLIT / GLOBAL CONFIG
# =============================================================================

st.set_page_config(
    page_title="Binance TR Spot Deep AI Scalper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TR_API_BASE = "https://www.binance.tr"
TR_MARKET_MAIN = "https://api.binance.me"
TR_MARKET_ALT = "https://cloudme-tr.2meta.app"

FEATURE_COUNT = 200
DEFAULT_MODEL_ROOT = "models_tr"

RET_H = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
WIN = [5, 8, 13, 21, 34, 55, 89, 144]
RSI_P = [5, 7, 9, 14, 21, 28, 35, 50]
ATR_P = [5, 7, 10, 14, 21, 28, 35, 50]
STO_P = [5, 7, 9, 14, 21, 28, 35, 50]
BB_P = [10, 14, 20, 28, 35, 50, 75, 100]
VOL_P = [5, 8, 13, 21, 34, 55, 89, 144]

STATE_DEFAULTS = {
    "api_key": "",
    "api_secret": "",
    "api_ok": False,
    "account": None,
    "api_diag": {},
    "paper_balance": 1000.0,
    "positions": {},
    "trade_log": [],
    "scan_rows": [],
    "training_log": [],
    "symbols_cache": {},
}
for _key, _value in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #0b111c;
    }
    .hero {
        background: linear-gradient(135deg,#0f172a,#172554);
        border: 1px solid #334155;
        padding: 22px;
        border-radius: 22px;
        margin-bottom: 14px;
    }
    .hero h1 {
        margin: 0;
        font-size: 32px;
    }
    .hero p {
        margin: .35rem 0 0;
        opacity: .76;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# GENERIC HTTP / BINANCE TR RESPONSE HELPERS
# =============================================================================

class BinanceTRAPIError(RuntimeError):
    pass


def parse_json_response(response: requests.Response):
    try:
        return response.json()
    except Exception as exc:
        raise BinanceTRAPIError(
            f"Sunucu JSON döndürmedi. HTTP {response.status_code}: "
            f"{response.text[:600]}"
        ) from exc


def check_http_response(response: requests.Response, label="Binance TR"):
    if response.status_code == 451:
        raise BinanceTRAPIError(
            f"{label} HTTP 451: bu bağlantı sunucu/ağ düzeyinde reddedildi."
        )

    if not response.ok:
        try:
            body = response.json()
        except Exception:
            body = response.text[:1200]

        raise BinanceTRAPIError(
            f"{label} HTTP {response.status_code}: {body}"
        )


def unwrap_binance_tr(payload):
    if not isinstance(payload, dict):
        return payload

    code = payload.get("code", 0)

    if code not in (0, "0", None):
        message = (
            payload.get("msg")
            or payload.get("message")
            or payload
        )

        raise BinanceTRAPIError(
            f"Binance TR API code {code}: {message}"
        )

    if "data" in payload:
        return payload["data"]

    return payload


def request_json(
    method,
    url,
    *,
    params=None,
    data=None,
    headers=None,
    timeout=20,
):
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise BinanceTRAPIError(
            f"Bağlantı hatası: {exc}"
        ) from exc

    check_http_response(response)

    return parse_json_response(response)


# =============================================================================
# SYMBOL / DECIMAL HELPERS
# =============================================================================

def normalize_tr_symbol(symbol):
    value = (
        str(symbol)
        .strip()
        .upper()
        .replace("-", "_")
        .replace("/", "_")
    )

    if "_" in value:
        return value

    for quote in [
        "USDT",
        "TRY",
        "BTC",
        "ETH",
        "BNB",
    ]:
        if (
            value.endswith(quote)
            and
            len(value) > len(quote)
        ):
            return (
                value[:-len(quote)]
                +
                "_"
                +
                quote
            )

    return value


def format_decimal(value, places=12):
    return (
        f"{float(value):.{places}f}"
        .rstrip("0")
        .rstrip(".")
    )


def floor_step(value, step):
    value = float(value)
    step = float(step)

    if step <= 0:
        return value

    return math.floor(value / step) * step


# =============================================================================
# BINANCE TR CLIENT
# =============================================================================

class BinanceTRClient:
    def __init__(
        self,
        api_key="",
        api_secret="",
    ):
        self.api_key = (
            api_key or ""
        ).strip()

        self.api_secret = (
            api_secret or ""
        ).strip()

        self._symbols_cache = None
        self._symbols_cache_at = 0.0

    # -------------------------------------------------------------------------
    # PUBLIC GENERAL ENDPOINTS
    # -------------------------------------------------------------------------

    def server_time(self):
        payload = request_json(
            "GET",
            TR_API_BASE
            +
            "/open/v1/common/time",
        )

        code = payload.get(
            "code",
            0,
        )

        if code not in (
            0,
            "0",
            None,
        ):
            raise BinanceTRAPIError(
                f"Server time API error: {payload}"
            )

        timestamp = payload.get(
            "timestamp"
        )

        if timestamp is None:
            raise BinanceTRAPIError(
                f"Binance TR server timestamp bulunamadı: {payload}"
            )

        return int(timestamp)

    def supported_symbols(
        self,
        force=False,
    ):
        now = time.time()

        if (
            not force
            and
            self._symbols_cache is not None
            and
            now - self._symbols_cache_at < 300
        ):
            return self._symbols_cache

        payload = request_json(
            "GET",
            TR_API_BASE
            +
            "/open/v1/common/symbols",
        )

        data = unwrap_binance_tr(
            payload
        )

        if isinstance(
            data,
            dict,
        ):
            items = data.get(
                "list",
                [],
            )
        elif isinstance(
            data,
            list,
        ):
            items = data
        else:
            items = []

        result = {}

        for item in items:
            symbol = str(
                item.get(
                    "symbol",
                    "",
                )
            ).upper()

            if not symbol:
                continue

            result[symbol] = item

        if not result:
            raise BinanceTRAPIError(
                "Binance TR symbol listesi boş döndü."
            )

        self._symbols_cache = result
        self._symbols_cache_at = now

        return result

    def symbol_info(
        self,
        symbol,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        symbols = self.supported_symbols()

        if symbol not in symbols:
            raise BinanceTRAPIError(
                f"Binance TR'de symbol bulunamadı: {symbol}"
            )

        return symbols[symbol]

    def symbol_type(
        self,
        symbol,
    ):
        info = self.symbol_info(
            symbol
        )

        return int(
            info.get(
                "type",
                1,
            )
        )

    def public_market_symbol(
        self,
        symbol,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        symbol_type = self.symbol_type(
            symbol
        )

        if symbol_type == 1:
            return symbol.replace(
                "_",
                "",
            )

        return symbol

    # -------------------------------------------------------------------------
    # PUBLIC MARKET ENDPOINTS
    # -------------------------------------------------------------------------

    def klines(
        self,
        symbol,
        interval="1m",
        limit=500,
        start_time=None,
        end_time=None,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        symbol_type = self.symbol_type(
            symbol
        )

        public_symbol = (
            self.public_market_symbol(
                symbol
            )
        )

        params = {
            "symbol": public_symbol,
            "interval": interval,
            "limit": int(limit),
        }

        if start_time is not None:
            params["startTime"] = int(
                start_time
            )

        if end_time is not None:
            params["endTime"] = int(
                end_time
            )

        if symbol_type == 1:
            url = (
                TR_MARKET_MAIN
                +
                "/api/v1/klines"
            )
        else:
            url = (
                TR_MARKET_ALT
                +
                "/api/v1/klines"
            )

        payload = request_json(
            "GET",
            url,
            params=params,
        )

        data = unwrap_binance_tr(
            payload
        )

        if not isinstance(
            data,
            list,
        ):
            raise BinanceTRAPIError(
                f"Kline cevabı beklenmeyen formatta: {type(data)}"
            )

        return data

    def depth(
        self,
        symbol,
        limit=20,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        symbol_type = self.symbol_type(
            symbol
        )

        public_symbol = (
            self.public_market_symbol(
                symbol
            )
        )

        params = {
            "symbol": public_symbol,
            "limit": int(limit),
        }

        if symbol_type == 1:
            url = (
                TR_MARKET_MAIN
                +
                "/api/v3/depth"
            )
        else:
            url = (
                TR_MARKET_ALT
                +
                "/api/v1/depth"
            )

        payload = request_json(
            "GET",
            url,
            params=params,
        )

        return unwrap_binance_tr(
            payload
        )

    def best_bid_ask(
        self,
        symbol,
    ):
        book = self.depth(
            symbol,
            limit=5,
        )

        bids = (
            book.get(
                "bids",
                [],
            )
            if isinstance(
                book,
                dict,
            )
            else []
        )

        asks = (
            book.get(
                "asks",
                [],
            )
            if isinstance(
                book,
                dict,
            )
            else []
        )

        if not bids or not asks:
            raise BinanceTRAPIError(
                "Order book bid/ask alınamadı."
            )

        bid = float(
            bids[0][0]
        )

        ask = float(
            asks[0][0]
        )

        return bid, ask

    def last_price(
        self,
        symbol,
    ):
        rows = self.klines(
            symbol,
            interval="1m",
            limit=2,
        )

        if not rows:
            raise BinanceTRAPIError(
                "Son fiyat için kline alınamadı."
            )

        return float(
            rows[-1][4]
        )

    # -------------------------------------------------------------------------
    # SIGNED ENDPOINTS
    # -------------------------------------------------------------------------

    def signed_request(
        self,
        method,
        path,
        params=None,
    ):
        """
        Binance TR SIGNED request.

        Important:
        - X-MBX-APIKEY header is case-sensitive.
        - HMAC-SHA256 is calculated over the EXACT query/body string.
        - GET sends the exact signed string in the URL query.
        - POST sends the exact signed string as application/x-www-form-urlencoded.
        """
        if (
            not self.api_key
            or
            not self.api_secret
        ):
            raise BinanceTRAPIError(
                "API Key / Secret Key girilmedi."
            )

        payload = dict(
            params or {}
        )

        # Use Binance TR server time to avoid local-clock drift.
        server_timestamp = int(
            self.server_time()
        )

        # Keep explicit deterministic order for easier debugging.
        payload["recvWindow"] = int(
            payload.get(
                "recvWindow",
                5000,
            )
        )

        payload["timestamp"] = (
            server_timestamp
        )

        # Build ONE canonical string and use exactly this string for both
        # signature generation and the actual HTTP request.
        unsigned_query = urlencode(
            payload,
            doseq=True,
            safe="",
        )

        signature = hmac.new(
            self.api_secret.encode(
                "utf-8"
            ),
            unsigned_query.encode(
                "utf-8"
            ),
            hashlib.sha256,
        ).hexdigest()

        signed_query = (
            unsigned_query
            +
            "&signature="
            +
            signature
        )

        headers = {
            "X-MBX-APIKEY":
            self.api_key,
            "Accept":
            "application/json",
            "User-Agent":
            "BinanceTR-DeepAI-Scalper/1.0",
        }

        url = (
            TR_API_BASE
            +
            path
        )

        method = method.upper()

        try:
            if method == "GET":
                # Do NOT pass params=dict here. We send the exact query string
                # that was signed so requests cannot re-encode/reorder it.
                response = requests.get(
                    url
                    +
                    "?"
                    +
                    signed_query,
                    headers=headers,
                    timeout=20,
                )
            else:
                post_headers = dict(
                    headers
                )

                post_headers[
                    "Content-Type"
                ] = (
                    "application/"
                    "x-www-form-urlencoded"
                )

                # Send exact signed body string.
                response = requests.request(
                    method,
                    url,
                    data=signed_query,
                    headers=post_headers,
                    timeout=20,
                )

        except requests.RequestException as exc:
            raise BinanceTRAPIError(
                f"Binance TR signed bağlantı hatası: {exc}"
            ) from exc

        # Keep HTTP and Binance JSON error information.
        if response.status_code == 451:
            raise BinanceTRAPIError(
                "Binance TR signed HTTP 451: "
                "sunucu/ağ erişimi reddetti."
            )

        try:
            raw = response.json()
        except Exception:
            raw = None

        if not response.ok:
            detail = (
                raw
                if raw is not None
                else response.text[:1200]
            )

            raise BinanceTRAPIError(
                f"Binance TR signed HTTP "
                f"{response.status_code}: {detail}"
            )

        if raw is None:
            raise BinanceTRAPIError(
                "Binance TR signed endpoint JSON döndürmedi: "
                f"{response.text[:1200]}"
            )

        # Binance TR may return HTTP 200 with code != 0.
        if isinstance(
            raw,
            dict,
        ):
            code_value = raw.get(
                "code",
                0,
            )

            if code_value not in (
                0,
                "0",
                None,
            ):
                message = (
                    raw.get(
                        "msg"
                    )
                    or
                    raw.get(
                        "message"
                    )
                    or
                    str(
                        raw
                    )
                )

                hint = ""

                message_lower = str(
                    message
                ).lower()

                if (
                    "api-key"
                    in message_lower
                    or
                    "api key"
                    in message_lower
                    or
                    "apikey"
                    in message_lower
                ):
                    hint = (
                        " | İpucu: Bu anahtarın Binance TR hesabındaki "
                        "API Management bölümünden oluşturulduğunu kontrol et. "
                        "Binance.com anahtarını Binance TR endpointinde kullanma."
                    )

                elif (
                    "signature"
                    in message_lower
                ):
                    hint = (
                        " | İpucu: Secret Key yanlış olabilir. "
                        "API Key ve Secret Key case-sensitive'dir; "
                        "başında/sonunda boşluk olmamalı."
                    )

                elif (
                    "timestamp"
                    in message_lower
                    or
                    "recvwindow"
                    in message_lower
                ):
                    hint = (
                        " | İpucu: Timestamp/recvWindow reddedildi. "
                        "Kod Binance TR server time kullanıyor; "
                        "yeniden API testi yap."
                    )

                elif (
                    "permission"
                    in message_lower
                    or
                    "trade"
                    in message_lower
                ):
                    hint = (
                        " | İpucu: API anahtarında gerekli hesap/Spot "
                        "yetkilerini kontrol et."
                    )

                raise BinanceTRAPIError(
                    f"Binance TR API code "
                    f"{code_value}: {message}"
                    f"{hint}"
                )

        return unwrap_binance_tr(
            raw
        )

    def account(
        self,
    ):
        return self.signed_request(
            "GET",
            "/open/v1/account/spot",
        )

    def account_asset(
        self,
        asset,
    ):
        return self.signed_request(
            "GET",
            "/open/v1/account/spot/asset",
            {
                "asset":
                str(asset).upper()
            },
        )

    def query_order(
        self,
        order_id=None,
        client_id=None,
    ):
        params = {}

        if order_id is not None:
            params[
                "orderId"
            ] = str(order_id)

        if client_id:
            params[
                "clientId"
            ] = str(client_id)

        if not params:
            raise ValueError(
                "order_id veya client_id gerekli."
            )

        return self.signed_request(
            "GET",
            "/open/v1/orders/detail",
            params,
        )

    def account_trades(
        self,
        symbol,
        limit=100,
    ):
        return self.signed_request(
            "GET",
            "/open/v1/orders/trades",
            {
                "symbol":
                normalize_tr_symbol(
                    symbol
                ),
                "limit":
                int(limit),
            },
        )

    # -------------------------------------------------------------------------
    # FILTERS / ORDER VALIDATION
    # -------------------------------------------------------------------------

    def symbol_filters(
        self,
        symbol,
    ):
        info = self.symbol_info(
            symbol
        )

        result = {}

        for item in (
            info.get(
                "filters",
                [],
            )
            or []
        ):
            filter_type = (
                item.get(
                    "filterType"
                )
            )

            if filter_type:
                result[
                    filter_type
                ] = item

        return result

    def normalize_quantity(
        self,
        symbol,
        quantity,
    ):
        filters = (
            self.symbol_filters(
                symbol
            )
        )

        market_lot = (
            filters.get(
                "MARKET_LOT_SIZE"
            )
        )

        lot = (
            market_lot
            or
            filters.get(
                "LOT_SIZE"
            )
            or
            {}
        )

        step = float(
            lot.get(
                "stepSize",
                0,
            )
            or
            0
        )

        minimum = float(
            lot.get(
                "minQty",
                0,
            )
            or
            0
        )

        maximum = float(
            lot.get(
                "maxQty",
                0,
            )
            or
            0
        )

        normalized = floor_step(
            quantity,
            step,
        )

        if (
            minimum > 0
            and
            normalized < minimum
        ):
            raise BinanceTRAPIError(
                f"quantity minQty altında: "
                f"{normalized} < {minimum}"
            )

        if (
            maximum > 0
            and
            normalized > maximum
        ):
            raise BinanceTRAPIError(
                f"quantity maxQty üstünde: "
                f"{normalized} > {maximum}"
            )

        return normalized

    def can_trade_symbol(
        self,
        symbol,
    ):
        info = self.symbol_info(
            symbol
        )

        enabled = int(
            info.get(
                "spotTradingEnable",
                1,
            )
            or
            0
        )

        return enabled == 1

    # -------------------------------------------------------------------------
    # LIVE MARKET ORDERS
    # -------------------------------------------------------------------------

    def market_buy_quote(
        self,
        symbol,
        quote_amount,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        if not self.can_trade_symbol(
            symbol
        ):
            raise BinanceTRAPIError(
                f"{symbol} Spot trading kapalı."
            )

        quote_amount = float(
            quote_amount
        )

        if quote_amount <= 0:
            raise ValueError(
                "quoteOrderQty 0'dan büyük olmalı."
            )

        return self.signed_request(
            "POST",
            "/open/v1/orders",
            {
                "symbol":
                symbol,
                "side":
                0,
                "type":
                2,
                "quoteOrderQty":
                format_decimal(
                    quote_amount,
                    8,
                ),
                "clientId":
                "ai_"
                +
                uuid.uuid4().hex[:20],
            },
        )

    def market_sell_quantity(
        self,
        symbol,
        quantity,
    ):
        symbol = normalize_tr_symbol(
            symbol
        )

        if not self.can_trade_symbol(
            symbol
        ):
            raise BinanceTRAPIError(
                f"{symbol} Spot trading kapalı."
            )

        normalized = (
            self.normalize_quantity(
                symbol,
                quantity,
            )
        )

        return self.signed_request(
            "POST",
            "/open/v1/orders",
            {
                "symbol":
                symbol,
                "side":
                1,
                "type":
                2,
                "quantity":
                format_decimal(
                    normalized,
                    12,
                ),
                "clientId":
                "ai_"
                +
                uuid.uuid4().hex[:20],
            },
        )

    def wait_order_execution(
        self,
        order_id,
        timeout_seconds=8.0,
    ):
        started = time.time()

        last = None

        while (
            time.time()
            -
            started
            <
            timeout_seconds
        ):
            last = self.query_order(
                order_id=order_id
            )

            if not isinstance(
                last,
                dict,
            ):
                time.sleep(
                    0.35
                )
                continue

            executed_qty = float(
                last.get(
                    "executedQty",
                    0,
                )
                or
                0
            )

            executed_quote = float(
                last.get(
                    "executedQuoteQty",
                    0,
                )
                or
                0
            )

            if (
                executed_qty > 0
                or
                executed_quote > 0
            ):
                return last

            time.sleep(
                0.35
            )

        return last or {}


# =============================================================================
# MARKET DATAFRAME
# =============================================================================

def klines_to_df(
    rows,
):
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

    frame = pd.DataFrame(
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

    for column in numeric:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["open_time"] = (
        pd.to_datetime(
            frame[
                "open_time"
            ],
            unit="ms",
            utc=True,
        )
    )

    return (
        frame
        .set_index(
            "open_time"
        )
        .sort_index()
    )


def download_history(
    client,
    symbol,
    interval,
    total,
):
    rows = []
    end_time = None

    while len(rows) < total:
        remaining = (
            total
            -
            len(rows)
        )

        limit = min(
            1000,
            remaining,
        )

        batch = client.klines(
            symbol,
            interval=interval,
            limit=limit,
            end_time=end_time,
        )

        if not batch:
            break

        rows = (
            batch
            +
            rows
        )

        earliest = int(
            batch[0][0]
        )

        end_time = (
            earliest
            -
            1
        )

        time.sleep(
            0.05
        )

    rows = rows[
        -total:
    ]

    if not rows:
        raise BinanceTRAPIError(
            "Geçmiş mum verisi alınamadı."
        )

    return klines_to_df(
        rows
    )


# =============================================================================
# FEATURE HELPERS
# =============================================================================

def safe_div(
    left,
    right,
):
    if isinstance(
        right,
        pd.Series,
    ):
        right = right.replace(
            0,
            np.nan,
        )

    return left / right


def ema(
    series,
    period,
):
    return series.ewm(
        span=int(period),
        adjust=False,
    ).mean()


def rsi(
    series,
    period,
):
    period = int(
        period
    )

    delta = series.diff()

    gain = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    loss = (
        -delta.clip(
            upper=0
        )
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = safe_div(
        gain,
        loss,
    )

    return (
        100
        -
        100
        /
        (
            1
            +
            rs
        )
    )


def atr(
    frame,
    period,
):
    period = int(
        period
    )

    previous_close = (
        frame[
            "close"
        ]
        .shift(1)
    )

    true_range = (
        pd.concat(
            [
                frame[
                    "high"
                ]
                -
                frame[
                    "low"
                ],
                (
                    frame[
                        "high"
                    ]
                    -
                    previous_close
                ).abs(),
                (
                    frame[
                        "low"
                    ]
                    -
                    previous_close
                ).abs(),
            ],
            axis=1,
        )
        .max(
            axis=1
        )
    )

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def zscore(
    series,
    period,
):
    mean = (
        series
        .rolling(
            period
        )
        .mean()
    )

    std = (
        series
        .rolling(
            period
        )
        .std()
    )

    return safe_div(
        series - mean,
        std,
    )


# =============================================================================
# EXACTLY 200 CAUSAL FEATURES
# =============================================================================

def build_features(
    frame,
):
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
        set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"Eksik kolonlar: {sorted(missing)}"
        )

    x = (
        frame
        .copy()
        .sort_index()
    )

    out = pd.DataFrame(
        index=x.index
    )

    o = x["open"]
    h = x["high"]
    l = x["low"]
    c = x["close"]
    v = x["volume"]

    qv = (
        x[
            "quote_volume"
        ]
        if
        "quote_volume"
        in
        x.columns
        else
        c * v
    )

    trades = (
        x[
            "trades"
        ]
        if
        "trades"
        in
        x.columns
        else
        pd.Series(
            0.0,
            index=x.index,
        )
    )

    taker = (
        x[
            "taker_base"
        ]
        if
        "taker_base"
        in
        x.columns
        else
        pd.Series(
            0.0,
            index=x.index,
        )
    )

    # 001-020 — simple and log returns
    for n in RET_H:
        out[
            f"ret_{n}"
        ] = c.pct_change(
            n
        )

    log_close = np.log(
        c.replace(
            0,
            np.nan,
        )
    )

    for n in RET_H:
        out[
            f"logret_{n}"
        ] = log_close.diff(
            n
        )

    # 021-036 — trend distance
    for w in WIN:
        out[
            f"sma_dev_{w}"
        ] = (
            safe_div(
                c,
                c.rolling(
                    w
                ).mean(),
            )
            -
            1
        )

    for w in WIN:
        out[
            f"ema_dev_{w}"
        ] = (
            safe_div(
                c,
                ema(
                    c,
                    w,
                ),
            )
            -
            1
        )

    # 037-052 — volatility and standardized price
    one_bar_return = (
        c.pct_change()
    )

    for w in WIN:
        out[
            f"ret_std_{w}"
        ] = (
            one_bar_return
            .rolling(
                w
            )
            .std()
        )

    for w in WIN:
        out[
            f"close_z_{w}"
        ] = zscore(
            c,
            w,
        )

    # 053-068 — rolling channel position and width
    for w in WIN:
        rolling_low = (
            l
            .rolling(
                w
            )
            .min()
        )

        rolling_high = (
            h
            .rolling(
                w
            )
            .max()
        )

        width = (
            rolling_high
            -
            rolling_low
        ).replace(
            0,
            np.nan,
        )

        out[
            f"range_pos_{w}"
        ] = (
            c
            -
            rolling_low
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

    # 069-076 — multi-period RSI
    for period in RSI_P:
        out[
            f"rsi_{period}"
        ] = (
            rsi(
                c,
                period,
            )
            /
            100.0
        )

    # 077-084 — ATR normalized by price
    for period in ATR_P:
        out[
            f"atr_pct_{period}"
        ] = (
            atr(
                x,
                period,
            )
            /
            c.replace(
                0,
                np.nan,
            )
        )

    # 085-100 — stochastic
    for period in STO_P:
        low_roll = (
            l
            .rolling(
                period
            )
            .min()
        )

        high_roll = (
            h
            .rolling(
                period
            )
            .max()
        )

        k = (
            c
            -
            low_roll
        ) / (
            high_roll
            -
            low_roll
        ).replace(
            0,
            np.nan,
        )

        out[
            f"stoch_k_{period}"
        ] = k

        out[
            f"stoch_d_{period}"
        ] = (
            k
            .rolling(3)
            .mean()
        )

    # 101-116 — Bollinger position and width
    for period in BB_P:
        middle = (
            c
            .rolling(
                period
            )
            .mean()
        )

        deviation = (
            c
            .rolling(
                period
            )
            .std()
        )

        upper = (
            middle
            +
            2
            *
            deviation
        )

        lower = (
            middle
            -
            2
            *
            deviation
        )

        out[
            f"bb_pos_{period}"
        ] = (
            c
            -
            lower
        ) / (
            upper
            -
            lower
        ).replace(
            0,
            np.nan,
        )

        out[
            f"bb_width_{period}"
        ] = (
            upper
            -
            lower
        ) / middle.replace(
            0,
            np.nan,
        )

    # 117-140 — volume regime
    for period in VOL_P:
        out[
            f"vol_z_{period}"
        ] = zscore(
            v,
            period,
        )

        out[
            f"vol_ratio_{period}"
        ] = safe_div(
            v,
            v
            .rolling(
                period
            )
            .mean(),
        )

        out[
            f"qvol_ratio_{period}"
        ] = safe_div(
            qv,
            qv
            .rolling(
                period
            )
            .mean(),
        )

    # 141-150 — ROC
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

    # 151-160 — candle shape
    candle_range = (
        h
        -
        l
    ).replace(
        0,
        np.nan,
    )

    body = (
        c
        -
        o
    )

    max_oc = (
        pd.concat(
            [
                o,
                c,
            ],
            axis=1,
        )
        .max(
            axis=1
        )
    )

    min_oc = (
        pd.concat(
            [
                o,
                c,
            ],
            axis=1,
        )
        .min(
            axis=1
        )
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
        h
        -
        max_oc
    ) / candle_range

    out[
        "lower_wick_ratio"
    ] = (
        min_oc
        -
        l
    ) / candle_range

    out[
        "close_location"
    ] = (
        c
        -
        l
    ) / candle_range

    out[
        "open_location"
    ] = (
        o
        -
        l
    ) / candle_range

    out[
        "gap_pct"
    ] = (
        o
        /
        c.shift(
            1
        ).replace(
            0,
            np.nan,
        )
        -
        1
    )

    out[
        "hl_pct"
    ] = (
        h
        -
        l
    ) / c.replace(
        0,
        np.nan,
    )

    out[
        "oc_abs_pct"
    ] = (
        body.abs()
        /
        o.replace(
            0,
            np.nan,
        )
    )

    # 161-172 — microstructure proxies
    signed = np.sign(
        c.diff()
    ).fillna(
        0
    )

    obv = (
        signed
        *
        v
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
        taker,
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
            .rolling(
                w
            )
            .mean()
        )

    # 173-184 — multi-scale MACD
    for fast, slow in [
        (5, 13),
        (8, 21),
        (12, 26),
        (13, 34),
        (21, 55),
        (34, 89),
    ]:
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
            macd
            -
            signal
        ) / c.replace(
            0,
            np.nan,
        )

    # 185-192 — acceleration
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
        momentum = (
            c.pct_change(
                n
            )
        )

        out[
            f"accel_{n}"
        ] = (
            momentum
            -
            momentum.shift(
                n
            )
        )

    # 193-198 — cyclical time
    if isinstance(
        out.index,
        pd.DatetimeIndex,
    ):
        minute = (
            out.index.minute
            +
            out.index.hour
            *
            60
        )

        weekday = (
            out.index.dayofweek
        )

        out[
            "tod_sin"
        ] = np.sin(
            2
            *
            np.pi
            *
            minute
            /
            1440
        )

        out[
            "tod_cos"
        ] = np.cos(
            2
            *
            np.pi
            *
            minute
            /
            1440
        )

        out[
            "dow_sin"
        ] = np.sin(
            2
            *
            np.pi
            *
            weekday
            /
            7
        )

        out[
            "dow_cos"
        ] = np.cos(
            2
            *
            np.pi
            *
            weekday
            /
            7
        )

        out[
            "hour_sin"
        ] = np.sin(
            2
            *
            np.pi
            *
            out.index.hour
            /
            24
        )

        out[
            "hour_cos"
        ] = np.cos(
            2
            *
            np.pi
            *
            out.index.hour
            /
            24
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
            out[
                name
            ] = 0.0

    # 199-200 — explicit interactions
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
        "momentum_vol_interaction"
    ] = (
        out[
            "ret_5"
        ]
        *
        out[
            "atr_pct_14"
        ]
    )

    base_columns = list(
        out.columns
    )

    filler_index = 0

    while (
        out.shape[1]
        <
        FEATURE_COUNT
    ):
        left = base_columns[
            filler_index
            %
            len(
                base_columns
            )
        ]

        right = base_columns[
            (
                filler_index
                *
                7
                +
                11
            )
            %
            len(
                base_columns
            )
        ]

        out[
            f"cross_{filler_index:03d}"
        ] = (
            out[
                left
            ]
            *
            out[
                right
            ]
        )

        filler_index += 1

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
        .fillna(
            0.0
        )
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
        !=
        FEATURE_COUNT
    ):
        raise RuntimeError(
            f"Feature count mismatch: {out.shape[1]}"
        )

    return out


# =============================================================================
# DEEP LEARNING MODEL
# =============================================================================

if TORCH_OK:

    class GatedResidualNetwork(
        nn.Module
    ):
        def __init__(
            self,
            size,
            hidden_size,
            dropout,
        ):
            super().__init__()

            self.fc1 = nn.Linear(
                size,
                hidden_size,
            )

            self.fc2 = nn.Linear(
                hidden_size,
                size,
            )

            self.gate = nn.Linear(
                size,
                size,
            )

            self.norm = nn.LayerNorm(
                size
            )

            self.dropout = nn.Dropout(
                dropout
            )

        def forward(
            self,
            x,
        ):
            residual = x

            hidden = self.fc1(
                x
            )

            hidden = torch.nn.functional.gelu(
                hidden
            )

            hidden = self.dropout(
                hidden
            )

            hidden = self.fc2(
                hidden
            )

            gate = torch.sigmoid(
                self.gate(
                    x
                )
            )

            return self.norm(
                residual
                +
                gate
                *
                hidden
            )


    class TemporalResidualBlock(
        nn.Module
    ):
        def __init__(
            self,
            channels,
            kernel_size,
            dilation,
            dropout,
        ):
            super().__init__()

            padding = (
                (
                    kernel_size
                    -
                    1
                )
                *
                dilation
            ) // 2

            self.conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )

            self.conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )

            self.norm1 = nn.BatchNorm1d(
                channels
            )

            self.norm2 = nn.BatchNorm1d(
                channels
            )

            self.dropout = nn.Dropout(
                dropout
            )

        def forward(
            self,
            x,
        ):
            residual = x

            x = self.conv1(
                x
            )

            x = self.norm1(
                x
            )

            x = torch.nn.functional.gelu(
                x
            )

            x = self.dropout(
                x
            )

            x = self.conv2(
                x
            )

            x = self.norm2(
                x
            )

            x = torch.nn.functional.gelu(
                x
            )

            x = self.dropout(
                x
            )

            return (
                x
                +
                residual
            )


    class DeepBinanceTRAI(
        nn.Module
    ):
        """
        Multi-branch temporal architecture.

        Branch A:
            dilated residual TCN

        Branch B:
            bidirectional LSTM -> bidirectional GRU

        Branch C:
            Transformer Encoder

        Fusion:
            learned branch gate

        Refinement:
            multi-head self attention

        Pooling:
            gated last / mean / max pooling

        Classifier:
            SELL / WAIT / BUY
        """

        def __init__(
            self,
            n_features=FEATURE_COUNT,
            d_model=192,
            heads=8,
            dropout=0.20,
            classes=3,
        ):
            super().__init__()

            self.input_norm = nn.LayerNorm(
                n_features
            )

            self.feature_project = nn.Sequential(
                nn.Linear(
                    n_features,
                    320,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    320,
                    d_model,
                ),
                nn.GELU(),
            )

            self.feature_grn_1 = GatedResidualNetwork(
                d_model,
                d_model
                *
                2,
                dropout,
            )

            self.feature_grn_2 = GatedResidualNetwork(
                d_model,
                d_model
                *
                2,
                dropout,
            )

            self.tcn_branch = nn.Sequential(
                TemporalResidualBlock(
                    d_model,
                    3,
                    1,
                    dropout,
                ),
                TemporalResidualBlock(
                    d_model,
                    5,
                    2,
                    dropout,
                ),
                TemporalResidualBlock(
                    d_model,
                    7,
                    4,
                    dropout,
                ),
                TemporalResidualBlock(
                    d_model,
                    9,
                    8,
                    dropout,
                ),
            )

            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )

            self.lstm_project = nn.Sequential(
                nn.Linear(
                    256,
                    d_model,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    d_model
                ),
            )

            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )

            self.gru_project = nn.Sequential(
                nn.Linear(
                    256,
                    d_model,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    d_model
                ),
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=4,
            )

            self.branch_gate = nn.Sequential(
                nn.Linear(
                    d_model
                    *
                    3,
                    256,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    256,
                    3,
                ),
                nn.Softmax(
                    dim=-1
                ),
            )

            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )

            self.attention_norm = nn.LayerNorm(
                d_model
            )

            self.pool_gate = nn.Sequential(
                nn.Linear(
                    d_model
                    *
                    3,
                    160,
                ),
                nn.GELU(),
                nn.Linear(
                    160,
                    3,
                ),
                nn.Softmax(
                    dim=-1
                ),
            )

            self.classifier = nn.Sequential(
                nn.Linear(
                    d_model
                    *
                    3,
                    512,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    512
                ),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    512,
                    256,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    256,
                    128,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    128,
                    64,
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    classes,
                ),
            )

            self.temperature = nn.Parameter(
                torch.ones(
                    1
                )
            )

        def forward(
            self,
            x,
        ):
            x = self.input_norm(
                x
            )

            x = self.feature_project(
                x
            )

            x = self.feature_grn_1(
                x
            )

            x = self.feature_grn_2(
                x
            )

            tcn = (
                self.tcn_branch(
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

            lstm_output, _ = self.lstm(
                x
            )

            recurrent = self.lstm_project(
                lstm_output
            )

            gru_output, _ = self.gru(
                recurrent
            )

            recurrent = (
                recurrent
                +
                self.gru_project(
                    gru_output
                )
            )

            transformer = self.transformer(
                x
            )

            branch_summary = torch.cat(
                [
                    tcn.mean(
                        dim=1
                    ),
                    recurrent.mean(
                        dim=1
                    ),
                    transformer.mean(
                        dim=1
                    ),
                ],
                dim=-1,
            )

            branch_weights = self.branch_gate(
                branch_summary
            )

            fused = (
                tcn
                *
                branch_weights[
                    :,
                    0:1,
                    None,
                ]
                +
                recurrent
                *
                branch_weights[
                    :,
                    1:2,
                    None,
                ]
                +
                transformer
                *
                branch_weights[
                    :,
                    2:3,
                    None,
                ]
            )

            attention_output, _ = self.self_attention(
                fused,
                fused,
                fused,
                need_weights=False,
            )

            fused = self.attention_norm(
                fused
                +
                attention_output
            )

            last_pool = fused[
                :,
                -1,
                :,
            ]

            mean_pool = fused.mean(
                dim=1
            )

            max_pool = fused.max(
                dim=1
            ).values

            pool_input = torch.cat(
                [
                    last_pool,
                    mean_pool,
                    max_pool,
                ],
                dim=-1,
            )

            pool_weights = self.pool_gate(
                pool_input
            )

            pooled = torch.cat(
                [
                    last_pool
                    *
                    pool_weights[
                        :,
                        0:1,
                    ],
                    mean_pool
                    *
                    pool_weights[
                        :,
                        1:2,
                    ],
                    max_pool
                    *
                    pool_weights[
                        :,
                        2:3,
                    ],
                ],
                dim=-1,
            )

            logits = self.classifier(
                pooled
            )

            temperature = torch.clamp(
                self.temperature,
                0.25,
                4.0,
            )

            return (
                logits
                /
                temperature
            )


    class SequenceDataset(
        Dataset
    ):
        def __init__(
            self,
            features,
            labels,
            sequence_length,
        ):
            self.features = features
            self.labels = labels
            self.sequence_length = int(
                sequence_length
            )

        def __len__(
            self,
        ):
            return max(
                0,
                len(
                    self.labels
                )
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

            sequence = (
                self.features[
                    index:
                    index
                    +
                    self.sequence_length
                ]
            )

            label = (
                self.labels[
                    end
                ]
            )

            return (
                torch.tensor(
                    sequence,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
            )


# =============================================================================
# TRAINING / VALIDATION
# =============================================================================

def build_labels(
    close,
    horizon,
    threshold,
):
    future_return = (
        close.shift(
            -int(
                horizon
            )
        )
        /
        close
        -
        1
    )

    labels = np.ones(
        len(
            close
        ),
        dtype=np.int64,
    )

    labels[
        future_return
        <
        -float(
            threshold
        )
    ] = 0

    labels[
        future_return
        >
        float(
            threshold
        )
    ] = 2

    return labels


def calculate_class_weights(
    labels,
):
    counts = np.bincount(
        labels,
        minlength=3,
    ).astype(
        np.float64
    )

    counts[
        counts
        ==
        0
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


def timeframe_model_dir(
    root,
    timeframe,
):
    return (
        Path(
            root
        )
        /
        str(
            timeframe
        )
    )


def train_model(
    client,
    symbol,
    timeframe,
    bars,
    sequence_length,
    horizon,
    threshold,
    epochs,
    batch_size,
    learning_rate,
    model_root,
):
    if not TORCH_OK:
        raise RuntimeError(
            "PyTorch kurulmadı."
        )

    st.session_state[
        "training_log"
    ] = []

    def log(
        text,
    ):
        st.session_state[
            "training_log"
        ].append(
            str(
                text
            )
        )

    symbol = normalize_tr_symbol(
        symbol
    )

    output_dir = timeframe_model_dir(
        model_root,
        timeframe,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        f"{symbol} {timeframe}: "
        f"{bars} mum indiriliyor..."
    )

    frame = download_history(
        client,
        symbol,
        timeframe,
        int(
            bars
        ),
    )

    log(
        f"{len(frame)} mum indirildi."
    )

    features = build_features(
        frame
    )

    labels = build_labels(
        frame[
            "close"
        ],
        horizon,
        threshold,
    )

    usable = (
        len(
            frame
        )
        -
        int(
            horizon
        )
    )

    features = features.iloc[
        :usable
    ]

    labels = labels[
        :usable
    ]

    train_end = int(
        len(
            features
        )
        *
        0.70
    )

    validation_end = int(
        len(
            features
        )
        *
        0.85
    )

    training_features = features.iloc[
        :train_end
    ]

    mean = (
        training_features
        .mean()
        .to_numpy(
            np.float32
        )
    )

    std = (
        training_features
        .std()
        .replace(
            0,
            1,
        )
        .to_numpy(
            np.float32
        )
    )

    matrix = (
        (
            features
            .to_numpy(
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

    train_dataset = SequenceDataset(
        matrix[
            :train_end
        ],
        labels[
            :train_end
        ],
        sequence_length,
    )

    validation_dataset = SequenceDataset(
        matrix[
            train_end
            -
            sequence_length:
            validation_end
        ],
        labels[
            train_end
            -
            sequence_length:
            validation_end
        ],
        sequence_length,
    )

    test_dataset = SequenceDataset(
        matrix[
            validation_end
            -
            sequence_length:
        ],
        labels[
            validation_end
            -
            sequence_length:
        ],
        sequence_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=False,
    )

    device = (
        "cuda"
        if
        torch.cuda.is_available()
        else
        "cpu"
    )

    log(
        f"Device: {device}"
    )

    model = DeepBinanceTRAI(
        n_features=FEATURE_COUNT
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            learning_rate
        ),
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

    weights = calculate_class_weights(
        labels[
            :train_end
        ]
    ).to(
        device
    )

    loss_function = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.03,
    )

    best_validation_loss = (
        float(
            "inf"
        )
    )

    patience = 0
    maximum_patience = 4

    for epoch in range(
        1,
        int(
            epochs
        )
        +
        1,
    ):
        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for (
            feature_batch,
            label_batch,
        ) in train_loader:
            feature_batch = (
                feature_batch
                .to(
                    device
                )
            )

            label_batch = (
                label_batch
                .to(
                    device
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                feature_batch
            )

            loss = loss_function(
                logits,
                label_batch,
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

            predicted = logits.argmax(
                dim=1
            )

            train_correct += (
                predicted
                ==
                label_batch
            ).sum().item()

            train_total += (
                label_batch.numel()
            )

        model.eval()

        validation_loss_total = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for (
                feature_batch,
                label_batch,
            ) in validation_loader:
                feature_batch = (
                    feature_batch
                    .to(
                        device
                    )
                )

                label_batch = (
                    label_batch
                    .to(
                        device
                    )
                )

                logits = model(
                    feature_batch
                )

                loss = loss_function(
                    logits,
                    label_batch,
                )

                validation_loss_total += (
                    loss.item()
                )

                predicted = logits.argmax(
                    dim=1
                )

                validation_correct += (
                    predicted
                    ==
                    label_batch
                ).sum().item()

                validation_total += (
                    label_batch.numel()
                )

        train_loss = (
            train_loss_total
            /
            max(
                1,
                len(
                    train_loader
                ),
            )
        )

        validation_loss = (
            validation_loss_total
            /
            max(
                1,
                len(
                    validation_loader
                ),
            )
        )

        train_accuracy = (
            train_correct
            /
            max(
                1,
                train_total,
            )
        )

        validation_accuracy = (
            validation_correct
            /
            max(
                1,
                validation_total,
            )
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        log(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f} | "
            f"lr={current_lr:.7f}"
        )

        if (
            validation_loss
            <
            best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            patience = 0

            torch.save(
                model.state_dict(),
                output_dir
                /
                "model.pt",
            )

            np.savez(
                output_dir
                /
                "scaler.npz",
                mean=mean,
                std=std,
            )

            metadata = {
                "symbol":
                symbol,
                "timeframe":
                timeframe,
                "sequence_length":
                int(
                    sequence_length
                ),
                "horizon":
                int(
                    horizon
                ),
                "threshold":
                float(
                    threshold
                ),
                "feature_count":
                FEATURE_COUNT,
                "features":
                list(
                    features.columns
                ),
                "classes":
                [
                    "SELL",
                    "WAIT",
                    "BUY",
                ],
                "architecture":
                (
                    "GRN+ResidualTCN+BiLSTM+BiGRU+"
                    "Transformer4+SelfAttention+GatedPooling"
                ),
            }

            (
                output_dir
                /
                "meta.json"
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
                maximum_patience
            ):
                log(
                    "Early stopping."
                )
                break

    model.load_state_dict(
        torch.load(
            output_dir
            /
            "model.pt",
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
        dtype=np.int64,
    )

    with torch.no_grad():
        for (
            feature_batch,
            label_batch,
        ) in test_loader:
            feature_batch = (
                feature_batch
                .to(
                    device
                )
            )

            label_batch = (
                label_batch
                .to(
                    device
                )
            )

            logits = model(
                feature_batch
            )

            predicted = logits.argmax(
                dim=1
            )

            test_correct += (
                predicted
                ==
                label_batch
            ).sum().item()

            test_total += (
                label_batch.numel()
            )

            for (
                true_value,
                predicted_value,
            ) in zip(
                label_batch
                .cpu()
                .numpy(),
                predicted
                .cpu()
                .numpy(),
            ):
                confusion[
                    int(
                        true_value
                    ),
                    int(
                        predicted_value
                    ),
                ] += 1

    test_accuracy = (
        test_correct
        /
        max(
            1,
            test_total,
        )
    )

    log(
        f"Out-of-sample test accuracy: "
        f"{test_accuracy:.4f}"
    )

    return {
        "timeframe":
        timeframe,
        "test_accuracy":
        test_accuracy,
        "confusion_matrix":
        confusion.tolist(),
        "model_dir":
        str(
            output_dir
        ),
    }


# =============================================================================
# MODEL LOADING / MC-DROPOUT INFERENCE
# =============================================================================

MODEL_CACHE = {}


def clear_model_cache():
    MODEL_CACHE.clear()


def load_model(
    model_root,
    timeframe,
):
    if not TORCH_OK:
        raise RuntimeError(
            "PyTorch kurulmadı."
        )

    directory = timeframe_model_dir(
        model_root,
        timeframe,
    )

    cache_key = str(
        directory.resolve()
    )

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[
            cache_key
        ]

    model_file = (
        directory
        /
        "model.pt"
    )

    metadata_file = (
        directory
        /
        "meta.json"
    )

    scaler_file = (
        directory
        /
        "scaler.npz"
    )

    for required_file in [
        model_file,
        metadata_file,
        scaler_file,
    ]:
        if not required_file.exists():
            raise RuntimeError(
                f"Model dosyası eksik: {required_file}"
            )

    metadata = json.loads(
        metadata_file.read_text(
            encoding="utf-8"
        )
    )

    scaler = np.load(
        scaler_file
    )

    mean = scaler[
        "mean"
    ]

    std = scaler[
        "std"
    ]

    model = DeepBinanceTRAI(
        n_features=int(
            metadata[
                "feature_count"
            ]
        )
    )

    state = torch.load(
        model_file,
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

    MODEL_CACHE[
        cache_key
    ] = result

    return result


def enable_mc_dropout(
    model,
):
    model.eval()

    for module in model.modules():
        if isinstance(
            module,
            nn.Dropout,
        ):
            module.train()


def predict_single_timeframe(
    client,
    symbol,
    timeframe,
    model_root,
    mc_samples=8,
):
    (
        model,
        metadata,
        mean,
        std,
    ) = load_model(
        model_root,
        timeframe,
    )

    rows = client.klines(
        symbol,
        interval=timeframe,
        limit=1000,
    )

    frame = klines_to_df(
        rows
    )

    features = build_features(
        frame
    )

    features = features.reindex(
        columns=metadata[
            "features"
        ],
        fill_value=0.0,
    )

    matrix = (
        (
            features
            .to_numpy(
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

    sequence_length = int(
        metadata[
            "sequence_length"
        ]
    )

    if (
        len(
            matrix
        )
        <
        sequence_length
    ):
        raise RuntimeError(
            "Inference için yeterli mum yok."
        )

    sequence = matrix[
        -sequence_length:
    ]

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )

    enable_mc_dropout(
        model
    )

    draws = []

    with torch.no_grad():
        for _ in range(
            max(
                2,
                int(
                    mc_samples
                ),
            )
        ):
            logits = model(
                tensor
            )

            probability = torch.softmax(
                logits,
                dim=1,
            ).squeeze(
                0
            ).numpy()

            draws.append(
                probability
            )

    model.eval()

    stacked = np.stack(
        draws,
        axis=0,
    )

    mean_probability = stacked.mean(
        axis=0
    )

    variance = float(
        stacked.var(
            axis=0
        ).mean()
    )

    safe_probability = np.clip(
        mean_probability.astype(
            np.float64
        ),
        1e-9,
        1.0,
    )

    entropy = float(
        -np.sum(
            safe_probability
            *
            np.log(
                safe_probability
            )
        )
        /
        np.log(
            3.0
        )
    )

    variance_component = min(
        1.0,
        variance
        *
        100.0,
    )

    uncertainty = min(
        100.0,
        100.0
        *
        (
            0.75
            *
            entropy
            +
            0.25
            *
            variance_component
        ),
    )

    confidence = max(
        0.0,
        100.0
        -
        uncertainty,
    )

    return {
        "SELL":
        float(
            mean_probability[
                0
            ]
            *
            100
        ),
        "WAIT":
        float(
            mean_probability[
                1
            ]
            *
            100
        ),
        "BUY":
        float(
            mean_probability[
                2
            ]
            *
            100
        ),
        "CONFIDENCE":
        float(
            confidence
        ),
        "UNCERTAINTY":
        float(
            uncertainty
        ),
        "VARIANCE":
        float(
            variance
        ),
    }, frame


def predict_multi_timeframe(
    client,
    symbol,
    model_root,
    timeframes,
    mc_samples=8,
):
    predictions = []

    timeframe_weights = {
        "1m": 1.00,
        "3m": 1.15,
        "5m": 1.30,
        "15m": 1.45,
        "30m": 1.55,
        "1h": 1.65,
    }

    for timeframe in timeframes:
        directory = timeframe_model_dir(
            model_root,
            timeframe,
        )

        if not (
            directory
            /
            "model.pt"
        ).exists():
            continue

        try:
            prediction, frame = (
                predict_single_timeframe(
                    client,
                    symbol,
                    timeframe,
                    model_root,
                    mc_samples,
                )
            )
        except Exception:
            continue

        weight = (
            timeframe_weights.get(
                timeframe,
                1.0,
            )
            *
            max(
                0.05,
                prediction[
                    "CONFIDENCE"
                ]
                /
                100.0,
            )
        )

        predictions.append(
            (
                timeframe,
                prediction,
                frame,
                weight,
            )
        )

    if not predictions:
        raise RuntimeError(
            "Kullanılabilir eğitilmiş timeframe modeli yok."
        )

    total_weight = sum(
        item[
            3
        ]
        for
        item
        in
        predictions
    )

    total_weight = max(
        total_weight,
        1e-9,
    )

    aggregate = {
        "SELL": 0.0,
        "WAIT": 0.0,
        "BUY": 0.0,
        "CONFIDENCE": 0.0,
        "UNCERTAINTY": 0.0,
    }

    details = []

    for (
        timeframe,
        prediction,
        frame,
        weight,
    ) in predictions:
        normalized_weight = (
            weight
            /
            total_weight
        )

        for key in [
            "SELL",
            "WAIT",
            "BUY",
            "CONFIDENCE",
            "UNCERTAINTY",
        ]:
            aggregate[
                key
            ] += (
                prediction[
                    key
                ]
                *
                normalized_weight
            )

        details.append(
            {
                "timeframe":
                timeframe,
                "weight":
                normalized_weight,
                **prediction,
            }
        )

    directional_gap = abs(
        aggregate[
            "BUY"
        ]
        -
        aggregate[
            "SELL"
        ]
    )

    aggregate[
        "DIRECTION_GAP"
    ] = directional_gap

    return (
        aggregate,
        details,
        predictions[
            0
        ][
            2
        ],
    )


# =============================================================================
# MARKET QUALITY / ROUTING SCORE
# =============================================================================

def spread_bps(
    client,
    symbol,
):
    bid, ask = client.best_bid_ask(
        symbol
    )

    middle = (
        bid
        +
        ask
    ) / 2.0

    if middle <= 0:
        return 9999.0

    return (
        (
            ask
            -
            bid
        )
        /
        middle
        *
        10000
    )


def realized_volatility_score(
    frame,
):
    returns = (
        frame[
            "close"
        ]
        .pct_change()
        .tail(
            50
        )
    )

    volatility = float(
        returns.std()
        *
        100
    )

    if not np.isfinite(
        volatility
    ):
        return 0.0

    if volatility < 0.02:
        return max(
            0.0,
            volatility
            /
            0.02
            *
            25.0,
        )

    if volatility <= 0.8:
        return (
            25.0
            +
            min(
                55.0,
                (
                    volatility
                    -
                    0.02
                )
                /
                0.78
                *
                55.0,
            )
        )

    return max(
        5.0,
        80.0
        -
        (
            volatility
            -
            0.8
        )
        *
        25.0,
    )


def scanner_score(
    prediction,
    spread,
    volatility_score,
):
    buy = prediction[
        "BUY"
    ]

    sell = prediction[
        "SELL"
    ]

    wait = prediction[
        "WAIT"
    ]

    confidence = prediction[
        "CONFIDENCE"
    ]

    uncertainty = prediction[
        "UNCERTAINTY"
    ]

    directional = max(
        buy,
        sell,
    )

    # Binance TR spot bot only buys; SELL is used primarily as an exit/avoid score.
    if sell > buy:
        direction_penalty = 30.0
    else:
        direction_penalty = 0.0

    score = (
        directional
        *
        0.50
        +
        confidence
        *
        0.30
        +
        volatility_score
        *
        0.20
        -
        wait
        *
        0.12
        -
        uncertainty
        *
        0.18
        -
        min(
            30.0,
            spread
            *
            2.2,
        )
        -
        direction_penalty
    )

    return float(
        score
    )


# =============================================================================
# PAPER / LIVE POSITION MANAGEMENT
# =============================================================================

def position_key(
    symbol,
):
    return normalize_tr_symbol(
        symbol
    )


def paper_open(
    symbol,
    quote_amount,
    entry_price,
    target_profit_pct,
):
    symbol = normalize_tr_symbol(
        symbol
    )

    if (
        st.session_state[
            "paper_balance"
        ]
        <
        quote_amount
    ):
        raise RuntimeError(
            "Paper bakiye yetersiz."
        )

    key = position_key(
        symbol
    )

    if key in st.session_state[
        "positions"
    ]:
        raise RuntimeError(
            "Bu symbol için zaten açık takip pozisyonu var."
        )

    quantity = (
        float(
            quote_amount
        )
        /
        float(
            entry_price
        )
    )

    st.session_state[
        "paper_balance"
    ] -= float(
        quote_amount
    )

    st.session_state[
        "positions"
    ][key] = {
        "symbol":
        symbol,
        "mode":
        "PAPER",
        "entry_price":
        float(
            entry_price
        ),
        "quantity":
        float(
            quantity
        ),
        "quote_amount":
        float(
            quote_amount
        ),
        "target_profit_pct":
        float(
            target_profit_pct
        ),
        "opened_at":
        str(
            datetime.now()
        ),
    }


def live_open(
    client,
    symbol,
    quote_amount,
    target_profit_pct,
):
    symbol = normalize_tr_symbol(
        symbol
    )

    key = position_key(
        symbol
    )

    if key in st.session_state[
        "positions"
    ]:
        raise RuntimeError(
            "Bu symbol için zaten açık takip pozisyonu var."
        )

    order_response = client.market_buy_quote(
        symbol,
        quote_amount,
    )

    order_id = None

    if isinstance(
        order_response,
        dict,
    ):
        order_id = (
            order_response.get(
                "orderId"
            )
        )

    if order_id is None:
        raise BinanceTRAPIError(
            f"BUY emri orderId döndürmedi: {order_response}"
        )

    detail = client.wait_order_execution(
        order_id
    )

    executed_quantity = float(
        detail.get(
            "executedQty",
            0,
        )
        or
        0
    )

    executed_quote = float(
        detail.get(
            "executedQuoteQty",
            0,
        )
        or
        0
    )

    executed_price = float(
        detail.get(
            "executedPrice",
            0,
        )
        or
        0
    )

    if (
        executed_price <= 0
        and
        executed_quantity > 0
        and
        executed_quote > 0
    ):
        executed_price = (
            executed_quote
            /
            executed_quantity
        )

    if (
        executed_quantity <= 0
    ):
        raise BinanceTRAPIError(
            f"BUY emri dolum miktarı alınamadı: {detail}"
        )

    if (
        executed_price <= 0
    ):
        executed_price = (
            client.last_price(
                symbol
            )
        )

    st.session_state[
        "positions"
    ][key] = {
        "symbol":
        symbol,
        "mode":
        "LIVE",
        "entry_price":
        float(
            executed_price
        ),
        "quantity":
        float(
            executed_quantity
        ),
        "quote_amount":
        float(
            executed_quote
            or
            quote_amount
        ),
        "target_profit_pct":
        float(
            target_profit_pct
        ),
        "open_order_id":
        str(
            order_id
        ),
        "opened_at":
        str(
            datetime.now()
        ),
    }

    return {
        "new_order":
        order_response,
        "fill_detail":
        detail,
    }


def position_profit_pct(
    position,
    current_price,
):
    entry = float(
        position[
            "entry_price"
        ]
    )

    if entry <= 0:
        return 0.0

    return (
        (
            float(
                current_price
            )
            -
            entry
        )
        /
        entry
        *
        100.0
    )


def close_position(
    client,
    key,
    current_price,
):
    position = (
        st.session_state[
            "positions"
        ][key]
    )

    profit_pct = position_profit_pct(
        position,
        current_price,
    )

    quantity = float(
        position[
            "quantity"
        ]
    )

    if (
        position[
            "mode"
        ]
        ==
        "LIVE"
    ):
        result = client.market_sell_quantity(
            position[
                "symbol"
            ],
            quantity,
        )
    else:
        result = {
            "paper":
            True
        }

    quote_value = (
        quantity
        *
        float(
            current_price
        )
    )

    pnl_usdt = (
        quote_value
        -
        float(
            position[
                "quote_amount"
            ]
        )
    )

    if (
        position[
            "mode"
        ]
        ==
        "PAPER"
    ):
        st.session_state[
            "paper_balance"
        ] += quote_value

    st.session_state[
        "trade_log"
    ].append(
        {
            "time":
            str(
                datetime.now()
            ),
            "action":
            "CLOSE_TP",
            "symbol":
            position[
                "symbol"
            ],
            "mode":
            position[
                "mode"
            ],
            "entry_price":
            position[
                "entry_price"
            ],
            "exit_price":
            float(
                current_price
            ),
            "quantity":
            quantity,
            "profit_pct":
            profit_pct,
            "pnl_usdt_est":
            pnl_usdt,
            "result":
            str(
                result
            ),
        }
    )

    del st.session_state[
        "positions"
    ][key]

    return result


def monitor_take_profit(
    client,
):
    events = []

    for key in list(
        st.session_state[
            "positions"
        ].keys()
    ):
        position = st.session_state[
            "positions"
        ].get(
            key
        )

        if not position:
            continue

        try:
            current_price = client.last_price(
                position[
                    "symbol"
                ]
            )

            profit_pct = position_profit_pct(
                position,
                current_price,
            )

            if (
                profit_pct
                >=
                float(
                    position[
                        "target_profit_pct"
                    ]
                )
            ):
                close_position(
                    client,
                    key,
                    current_price,
                )

                events.append(
                    f"{position['symbol']} "
                    f"%{profit_pct:.3f} kârda kapandı."
                )
        except Exception as exc:
            events.append(
                f"{key} TP kontrol hatası: {exc}"
            )

    return events


# =============================================================================
# API DIAGNOSTICS
# =============================================================================

def run_api_diagnostics(
    api_key,
    api_secret,
    test_symbol,
):
    diagnostics = {}

    client = BinanceTRClient(
        api_key,
        api_secret,
    )

    # Step 1: public server time
    try:
        timestamp = client.server_time()

        diagnostics[
            "1_public_server_time"
        ] = {
            "ok":
            True,
            "timestamp":
            timestamp,
        }
    except Exception as exc:
        diagnostics[
            "1_public_server_time"
        ] = {
            "ok":
            False,
            "error":
            str(
                exc
            ),
        }

        return (
            diagnostics,
            None,
            None,
        )

    # Step 2: public symbols
    try:
        symbols = client.supported_symbols(
            force=True
        )

        diagnostics[
            "2_public_symbols"
        ] = {
            "ok":
            True,
            "count":
            len(
                symbols
            ),
        }
    except Exception as exc:
        diagnostics[
            "2_public_symbols"
        ] = {
            "ok":
            False,
            "error":
            str(
                exc
            ),
        }

        return (
            diagnostics,
            None,
            None,
        )

    # Step 3: market data
    normalized = normalize_tr_symbol(
        test_symbol
    )

    try:
        rows = client.klines(
            normalized,
            interval="1m",
            limit=3,
        )

        diagnostics[
            "3_market_data"
        ] = {
            "ok":
            True,
            "symbol":
            normalized,
            "bars":
            len(
                rows
            ),
        }
    except Exception as exc:
        diagnostics[
            "3_market_data"
        ] = {
            "ok":
            False,
            "error":
            str(
                exc
            ),
        }

    # Step 4: signed account
    try:
        account = client.account()

        can_trade = int(
            account.get(
                "canTrade",
                0,
            )
            or
            0
        )

        diagnostics[
            "4_signed_account"
        ] = {
            "ok":
            True,
            "canTrade":
            can_trade,
            "canDeposit":
            account.get(
                "canDeposit"
            ),
            "canWithdraw":
            account.get(
                "canWithdraw"
            ),
        }

        if can_trade != 1:
            diagnostics[
                "5_trade_permission"
            ] = {
                "ok":
                False,
                "error":
                "Hesap API cevabında canTrade=1 değil.",
            }
        else:
            diagnostics[
                "5_trade_permission"
            ] = {
                "ok":
                True,
            }

        return (
            diagnostics,
            client,
            account,
        )

    except Exception as exc:
        error_text = str(
            exc
        )

        diagnostics[
            "4_signed_account"
        ] = {
            "ok":
            False,
            "error":
            error_text,
            "endpoint":
            (
                TR_API_BASE
                +
                "/open/v1/account/spot"
            ),
            "uses_server_time":
            True,
            "signature":
            "HMAC-SHA256 exact query string",
            "header":
            "X-MBX-APIKEY",
            "hint":
            (
                "API anahtarının Binance TR API Management üzerinden "
                "oluşturulduğunu, Secret Key'in doğru olduğunu ve "
                "API erişim/yetki ayarlarını kontrol et."
            ),
        }

        return (
            diagnostics,
            client,
            None,
        )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header(
        "🔐 Binance TR API"
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

    diagnostic_symbol = st.text_input(
        "API test symbol",
        "BTC_USDT",
    )

    if st.button(
        "🔌 API'yi Tam Test Et",
        use_container_width=True,
    ):
        try:
            (
                diagnostics,
                connected_client,
                account,
            ) = run_api_diagnostics(
                api_key,
                api_secret,
                diagnostic_symbol,
            )

            st.session_state[
                "api_diag"
            ] = diagnostics

            account_ok = (
                diagnostics
                .get(
                    "4_signed_account",
                    {},
                )
                .get(
                    "ok",
                    False,
                )
            )

            trade_ok = (
                diagnostics
                .get(
                    "5_trade_permission",
                    {},
                )
                .get(
                    "ok",
                    False,
                )
            )

            if account_ok:
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
                ] = bool(
                    trade_ok
                )

                if trade_ok:
                    st.success(
                        "API kabul edildi ve canTrade=1."
                    )
                else:
                    st.warning(
                        "İmzalı API kabul edildi fakat canTrade kapalı."
                    )
            else:
                st.session_state[
                    "api_ok"
                ] = False

                st.error(
                    "Signed account aşaması başarısız. "
                    "API teşhis sekmesine bak."
                )

        except Exception as exc:
            st.session_state[
                "api_ok"
            ] = False

            st.error(
                str(
                    exc
                )
            )

    if st.button(
        "API Anahtarlarını Temizle",
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

        st.session_state[
            "api_diag"
        ] = {}

        st.rerun()

    st.divider()

    st.header(
        "⚡ Spot Scalping"
    )

    execution_mode = st.radio(
        "Emir modu",
        [
            "Paper",
            "Live",
        ],
        index=0,
    )

    live_confirm = False

    if (
        execution_mode
        ==
        "Live"
    ):
        if not st.session_state[
            "api_ok"
        ]:
            st.error(
                "Live için önce API tam testinden geçmeli."
            )

        live_confirm = st.checkbox(
            "Gerçek Binance TR Spot emirlerini etkinleştir",
            value=False,
        )

    symbols_text = st.text_area(
        "Taranacak pariteler",
        "BTC_USDT\nETH_USDT\nBNB_USDT\nSOL_USDT\nXRP_USDT",
        height=125,
    )

    scan_symbols = [
        normalize_tr_symbol(
            item
        )
        for item
        in
        symbols_text.splitlines()
        if item.strip()
    ]

    timeframes_text = st.text_input(
        "AI timeframe'leri",
        "1m,3m,5m",
    )

    scan_timeframes = [
        item.strip()
        for item
        in
        timeframes_text.split(
            ","
        )
        if item.strip()
    ]

    quote_amount = st.number_input(
        "İşlem başına USDT",
        min_value=5.0,
        value=20.0,
        step=5.0,
    )

    target_profit_pct = st.number_input(
        "Kârda kapat (%)",
        min_value=0.05,
        max_value=20.0,
        value=0.50,
        step=0.05,
        format="%.2f",
    )

    minimum_buy_probability = st.slider(
        "Minimum BUY %",
        50,
        99,
        70,
    )

    minimum_ai_confidence = st.slider(
        "Minimum AI güven %",
        0,
        99,
        20,
    )

    maximum_spread_bps = st.number_input(
        "Maksimum spread (bps)",
        min_value=0.1,
        max_value=100.0,
        value=20.0,
        step=0.5,
    )

    mc_samples = st.slider(
        "MC-dropout örnek",
        2,
        24,
        8,
    )

    model_root = st.text_input(
        "Model klasörü",
        DEFAULT_MODEL_ROOT,
    )


client = BinanceTRClient(
    st.session_state[
        "api_key"
    ],
    st.session_state[
        "api_secret"
    ],
)


# =============================================================================
# HEADER / KPI
# =============================================================================

st.markdown(
    """
    <div class="hero">
      <h1>Binance TR Spot Deep AI Scalper</h1>
      <p>
      Resmî Binance TR API • 200 özellik • GRN + Residual TCN + BiLSTM +
      BiGRU + Transformer ×4 + Self-Attention + MC-Dropout • yüzde kâr TP
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TORCH_OK:
    st.error(
        "PyTorch kurulamadı: "
        + TORCH_ERROR
    )

    st.code(
        "streamlit\npandas\nnumpy\nrequests\ntorch\n",
        language="text",
    )

k1, k2, k3, k4 = st.columns(
    4
)

k1.metric(
    "API",
    (
        "Kabul edildi"
        if
        st.session_state[
            "api_ok"
        ]
        else
        "Bağlı değil"
    ),
)

k2.metric(
    "Özellik",
    FEATURE_COUNT,
)

k3.metric(
    "Paper bakiye",
    f"{st.session_state['paper_balance']:.2f} USDT",
)

k4.metric(
    "Açık pozisyon",
    len(
        st.session_state[
            "positions"
        ]
    ),
)


# =============================================================================
# TABS
# =============================================================================

(
    tab_scan,
    tab_trade,
    tab_training,
    tab_positions,
    tab_account,
    tab_api,
    tab_model,
) = st.tabs(
    [
        "🤖 AI Tarayıcı",
        "⚡ Scalper",
        "🏋 Eğitim",
        "💼 Pozisyonlar",
        "💳 Hesap",
        "🔎 API Teşhis",
        "🧠 Model",
    ]
)


# =============================================================================
# AI SCANNER
# =============================================================================

with tab_scan:
    st.subheader(
        "Multi-Timeframe Binance TR AI Tarayıcı"
    )

    if not TORCH_OK:
        st.warning(
            "AI için torch gerekli."
        )

    if st.button(
        "🧠 Pariteleri Tara",
        type="primary",
        disabled=not TORCH_OK,
    ):
        rows = []

        for symbol in scan_symbols:
            try:
                (
                    prediction,
                    details,
                    frame,
                ) = predict_multi_timeframe(
                    client,
                    symbol,
                    model_root,
                    scan_timeframes,
                    mc_samples,
                )

                spread = spread_bps(
                    client,
                    symbol,
                )

                volatility_score = (
                    realized_volatility_score(
                        frame
                    )
                )

                score = scanner_score(
                    prediction,
                    spread,
                    volatility_score,
                )

                if (
                    prediction[
                        "BUY"
                    ]
                    >=
                    minimum_buy_probability
                    and
                    prediction[
                        "CONFIDENCE"
                    ]
                    >=
                    minimum_ai_confidence
                    and
                    spread
                    <=
                    maximum_spread_bps
                ):
                    signal = "BUY"
                else:
                    signal = "WAIT"

                rows.append(
                    {
                        "Symbol":
                        symbol,
                        "BUY %":
                        round(
                            prediction[
                                "BUY"
                            ],
                            2,
                        ),
                        "WAIT %":
                        round(
                            prediction[
                                "WAIT"
                            ],
                            2,
                        ),
                        "SELL %":
                        round(
                            prediction[
                                "SELL"
                            ],
                            2,
                        ),
                        "AI Güven %":
                        round(
                            prediction[
                                "CONFIDENCE"
                            ],
                            2,
                        ),
                        "Belirsizlik %":
                        round(
                            prediction[
                                "UNCERTAINTY"
                            ],
                            2,
                        ),
                        "Spread bps":
                        round(
                            spread,
                            3,
                        ),
                        "Vol Score":
                        round(
                            volatility_score,
                            2,
                        ),
                        "Scanner Score":
                        round(
                            score,
                            2,
                        ),
                        "Signal":
                        signal,
                    }
                )

            except Exception as exc:
                rows.append(
                    {
                        "Symbol":
                        symbol,
                        "Error":
                        str(
                            exc
                        ),
                    }
                )

        st.session_state[
            "scan_rows"
        ] = rows

    if st.session_state[
        "scan_rows"
    ]:
        scan_frame = pd.DataFrame(
            st.session_state[
                "scan_rows"
            ]
        )

        if (
            "Scanner Score"
            in
            scan_frame.columns
        ):
            scan_frame = (
                scan_frame
                .sort_values(
                    "Scanner Score",
                    ascending=False,
                )
            )

        st.dataframe(
            scan_frame,
            use_container_width=True,
            hide_index=True,
        )

        valid_buys = [
            row
            for row
            in
            st.session_state[
                "scan_rows"
            ]
            if
            row.get(
                "Signal"
            )
            ==
            "BUY"
            and
            "Scanner Score"
            in
            row
        ]

        if valid_buys:
            best = max(
                valid_buys,
                key=lambda item:
                item[
                    "Scanner Score"
                ],
            )

            st.success(
                f"En yüksek BUY: "
                f"{best['Symbol']} — "
                f"BUY %{best['BUY %']:.2f} — "
                f"skor {best['Scanner Score']:.2f}"
            )

            if st.button(
                "🚀 En İyi BUY'ı Aç",
                type="primary",
            ):
                try:
                    symbol = best[
                        "Symbol"
                    ]

                    current = client.last_price(
                        symbol
                    )

                    if (
                        execution_mode
                        ==
                        "Paper"
                    ):
                        paper_open(
                            symbol,
                            quote_amount,
                            current,
                            target_profit_pct,
                        )

                        result = {
                            "paper":
                            True
                        }

                    else:
                        if not st.session_state[
                            "api_ok"
                        ]:
                            raise RuntimeError(
                                "Live API doğrulanmadı."
                            )

                        if not live_confirm:
                            raise RuntimeError(
                                "Live onay kutusu işaretli değil."
                            )

                        result = live_open(
                            client,
                            symbol,
                            quote_amount,
                            target_profit_pct,
                        )

                    st.session_state[
                        "trade_log"
                    ].append(
                        {
                            "time":
                            str(
                                datetime.now()
                            ),
                            "action":
                            "OPEN_BUY",
                            "symbol":
                            symbol,
                            "mode":
                            execution_mode.upper(),
                            "price":
                            current,
                            "quote_amount":
                            quote_amount,
                            "tp_pct":
                            target_profit_pct,
                            "ai_buy":
                            best[
                                "BUY %"
                            ],
                            "ai_confidence":
                            best[
                                "AI Güven %"
                            ],
                            "scanner_score":
                            best[
                                "Scanner Score"
                            ],
                            "result":
                            str(
                                result
                            ),
                        }
                    )

                    st.success(
                        "Pozisyon açıldı."
                    )

                except Exception as exc:
                    st.error(
                        str(
                            exc
                        )
                    )


# =============================================================================
# SCALPING / TP
# =============================================================================

with tab_trade:
    st.subheader(
        "Yüzde Kâr Scalping"
    )

    st.write(
        f"Yeni pozisyon hedefi: "
        f"**%{target_profit_pct:.2f} kâr**."
    )

    if st.button(
        "🔄 TP Kontrolünü Şimdi Çalıştır",
        use_container_width=True,
    ):
        events = monitor_take_profit(
            client
        )

        if events:
            for event in events:
                st.info(
                    event
                )
        else:
            st.success(
                "Kontrol tamamlandı; kapanan pozisyon yok."
            )

    automatic_tp = st.checkbox(
        "Panel açıkken her 3 saniyede TP kontrolü",
        value=False,
    )

    if (
        automatic_tp
        and
        hasattr(
            st,
            "fragment",
        )
    ):
        @st.fragment(
            run_every="3s"
        )
        def tp_fragment():
            events = monitor_take_profit(
                client
            )

            if events:
                for event in events:
                    st.write(
                        event
                    )

            st.caption(
                "Son kontrol: "
                +
                datetime.now().strftime(
                    "%H:%M:%S"
                )
            )

        tp_fragment()


# =============================================================================
# TRAINING UI
# =============================================================================

with tab_training:
    st.subheader(
        "Derin Öğrenme Model Eğitimi"
    )

    training_symbol = st.text_input(
        "Eğitim paritesi",
        "BTC_USDT",
    )

    training_timeframe = st.selectbox(
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
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        bars = st.number_input(
            "Geçmiş mum",
            min_value=3000,
            max_value=100000,
            value=15000,
            step=1000,
        )

        sequence_length = st.number_input(
            "Sequence",
            min_value=32,
            max_value=256,
            value=96,
            step=16,
        )

    with c2:
        horizon = st.number_input(
            "Tahmin horizon",
            min_value=1,
            max_value=24,
            value=3,
        )

        threshold = st.number_input(
            "Label threshold",
            min_value=0.0005,
            max_value=0.05,
            value=0.0025,
            step=0.0005,
            format="%.4f",
        )

    with c3:
        epochs = st.number_input(
            "Epoch",
            min_value=1,
            max_value=50,
            value=10,
        )

        batch_size = st.selectbox(
            "Batch size",
            [
                32,
                64,
                128,
                256,
            ],
            index=2,
        )

    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.00001,
        max_value=0.005,
        value=0.00020,
        step=0.00001,
        format="%.5f",
    )

    if st.button(
        "🏋 Modeli Eğit",
        type="primary",
        disabled=not TORCH_OK,
    ):
        try:
            with st.spinner(
                "Model eğitiliyor..."
            ):
                result = train_model(
                    client,
                    normalize_tr_symbol(
                        training_symbol
                    ),
                    training_timeframe,
                    int(
                        bars
                    ),
                    int(
                        sequence_length
                    ),
                    int(
                        horizon
                    ),
                    float(
                        threshold
                    ),
                    int(
                        epochs
                    ),
                    int(
                        batch_size
                    ),
                    float(
                        learning_rate
                    ),
                    model_root,
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
                str(
                    exc
                )
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
            height=330,
        )


# =============================================================================
# POSITIONS UI
# =============================================================================

with tab_positions:
    st.subheader(
        "Açık Pozisyonlar"
    )

    position_rows = []

    for (
        key,
        position,
    ) in st.session_state[
        "positions"
    ].items():
        try:
            current_price = client.last_price(
                position[
                    "symbol"
                ]
            )

            profit_pct = position_profit_pct(
                position,
                current_price,
            )

            pnl_usdt = (
                float(
                    position[
                        "quantity"
                    ]
                )
                *
                (
                    current_price
                    -
                    float(
                        position[
                            "entry_price"
                        ]
                    )
                )
            )

        except Exception:
            current_price = None
            profit_pct = None
            pnl_usdt = None

        position_rows.append(
            {
                "Symbol":
                position[
                    "symbol"
                ],
                "Mode":
                position[
                    "mode"
                ],
                "Entry":
                position[
                    "entry_price"
                ],
                "Current":
                current_price,
                "Qty":
                position[
                    "quantity"
                ],
                "TP %":
                position[
                    "target_profit_pct"
                ],
                "PnL %":
                profit_pct,
                "PnL USDT":
                pnl_usdt,
            }
        )

    if position_rows:
        st.dataframe(
            pd.DataFrame(
                position_rows
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Açık pozisyon yok."
        )

    st.subheader(
        "İşlem Geçmişi"
    )

    if st.session_state[
        "trade_log"
    ]:
        trade_frame = pd.DataFrame(
            st.session_state[
                "trade_log"
            ]
        )

        st.dataframe(
            trade_frame.iloc[
                ::-1
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "CSV indir",
            trade_frame.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name="binance_tr_ai_trades.csv",
            mime="text/csv",
        )


# =============================================================================
# ACCOUNT UI
# =============================================================================

with tab_account:
    st.subheader(
        "Binance TR Spot Hesabı"
    )

    if st.session_state[
        "account"
    ]:
        account = st.session_state[
            "account"
        ]

        a1, a2, a3 = st.columns(
            3
        )

        a1.metric(
            "canTrade",
            str(
                account.get(
                    "canTrade"
                )
            ),
        )

        a2.metric(
            "canDeposit",
            str(
                account.get(
                    "canDeposit"
                )
            ),
        )

        a3.metric(
            "canWithdraw",
            str(
                account.get(
                    "canWithdraw"
                )
            ),
        )

        assets = []

        for item in (
            account.get(
                "accountAssets",
                [],
            )
            or []
        ):
            free = float(
                item.get(
                    "free",
                    0,
                )
                or
                0
            )

            locked = float(
                item.get(
                    "locked",
                    0,
                )
                or
                0
            )

            if (
                free > 0
                or
                locked > 0
            ):
                assets.append(
                    {
                        "Asset":
                        item.get(
                            "asset"
                        ),
                        "Free":
                        free,
                        "Locked":
                        locked,
                        "Total":
                        free
                        +
                        locked,
                    }
                )

        if assets:
            st.dataframe(
                pd.DataFrame(
                    assets
                ),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            "Önce API'yi tam test et."
        )


# =============================================================================
# API DIAGNOSTICS UI
# =============================================================================

with tab_api:
    st.subheader(
        "API Kabul / Bağlantı Teşhisi"
    )

    if st.session_state[
        "api_diag"
    ]:
        st.json(
            st.session_state[
                "api_diag"
            ]
        )

    st.markdown(
        """
**Panel şu sırayla test eder:**

1. `GET /open/v1/common/time` — public bağlantı.
2. `GET /open/v1/common/symbols` — Binance TR sembol listesi.
3. Binance TR Kline endpoint'i — piyasa verisi.
4. `GET /open/v1/account/spot` — HMAC-SHA256 imzalı API Key/Secret testi.
5. `canTrade == 1` — gerçek Spot işlem yetkisi.

Bu beşinci adım da başarılıysa panel API durumunu **Kabul edildi** gösterir.
        """
    )

    st.warning(
        "Kod API anahtarının Binance TR tarafından kabul edilmesini garanti edemez. "
        "Yanlış Secret Key, yetki, hesap kısıtı veya Binance tarafındaki bir engel "
        "varsa sunucu isteği reddeder. Panel bu hatayı hangi aşamada aldığını gösterir."
    )


# =============================================================================
# MODEL INFO UI
# =============================================================================

with tab_model:
    st.subheader(
        "Derin Öğrenme Mimarisi"
    )

    st.code(
        """
200 Causal Market Features
        ↓
LayerNorm
        ↓
Dense Feature Projection
        ↓
Gated Residual Feature Network ×2
        ↓
 ┌────────────────────────┬─────────────────────────┬────────────────────────┐
 │ Residual TCN Branch    │ BiLSTM → BiGRU Branch │ Transformer ×4 Branch │
 │ kernels 3/5/7/9       │ temporal memory        │ long-range context    │
 │ dilation 1/2/4/8      │ bidirectional          │ 8-head attention      │
 └────────────────────────┴─────────────────────────┴────────────────────────┘
        ↓
Learned Branch Gating
        ↓
8-Head Self-Attention Refinement
        ↓
Gated Last / Mean / Max Temporal Pooling
        ↓
Dense 576 → 512 → 256 → 128 → 64
        ↓
SELL / WAIT / BUY
        ↓
Learned Temperature
        ↓
Monte-Carlo Dropout
        ↓
Confidence / Uncertainty
        ↓
Multi-Timeframe Weighted Ensemble
        ↓
Spread + Volatility Filter
        ↓
Spot BUY / WAIT
        """,
        language="text",
    )

    st.info(
        "Daha fazla katman tek başına daha iyi sonuç anlamına gelmez. "
        "Modelin gerçekten iyi olup olmadığını out-of-sample sonuçları, "
        "farklı piyasa rejimleri, komisyon ve gerçek paper-trading sonuçları belirler."
    )


st.caption(
    "Binance TR Spot AI Scalper • "
    "Kâr garantisi yoktur. Live mod gerçek para kullanır."
)
# INTERNAL REFERENCE 0001.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0001.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0002.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0003.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0004.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0005.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0006.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0007.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0008.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0009.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0010.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0011.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0012.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0013.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0014.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0015.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0016.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0017.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0018.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0019.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0020.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0021.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0022.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0023.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0024.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0025.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0026.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0027.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0028.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0029.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0030.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0031.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0032.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0033.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0034.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0035.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0036.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0037.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0038.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0039.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0040.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0041.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0042.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0043.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0044.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0045.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0046.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0047.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0048.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0049.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0050.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0051.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0052.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0053.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0054.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0055.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0056.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0057.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0058.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0059.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0060.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0061.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0062.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0063.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0064.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0065.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0066.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0067.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0068.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0069.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0070.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0071.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0072.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0073.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0074.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0075.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0076.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0077.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0078.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0079.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0080.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0081.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0082.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0083.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0084.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0085.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0086.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0087.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0088.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0089.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0090.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0091.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0092.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0093.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0094.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0095.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0096.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0097.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0098.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0099.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0100.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0101.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0102.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0103.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0104.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0105.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0106.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0107.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0108.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0109.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0110.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0111.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0112.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0113.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0114.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0115.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0116.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0117.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0118.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0119.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0120.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0121.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0122.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0123.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0124.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0125.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0126.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0127.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0128.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0129.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0130.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0131.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0132.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0133.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0134.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0135.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0136.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0137.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0138.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0139.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0140.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0141.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0142.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0143.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0144.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0145.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0146.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0147.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0148.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0149.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0150.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0151.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0152.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0153.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0154.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0155.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0156.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0157.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0158.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0159.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0160.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0161.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0162.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0163.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0164.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0165.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0166.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0167.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0168.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0169.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0170.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0171.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0172.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0173.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0174.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0175.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0176.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0177.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0178.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0179.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0180.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0181.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0182.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0183.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0184.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0185.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0186.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0187.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0188.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0189.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0190.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0191.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0192.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0193.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0194.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0195.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0196.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0197.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0198.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0199.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0200.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0201.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0202.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0203.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0204.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0205.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0206.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0207.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0208.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0209.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0210.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0211.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0212.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0213.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0214.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0215.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0216.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0217.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0218.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0219.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0220.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0221.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0222.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0223.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0224.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0225.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0226.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0227.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0228.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0229.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0230.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0231.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0232.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0233.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0234.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0235.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0236.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0237.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0238.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0239.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0240.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0241.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0242.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0243.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0244.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0245.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0246.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0247.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0248.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0249.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0250.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0251.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0252.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0253.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0254.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0255.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0256.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0257.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0258.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0259.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0260.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0261.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0262.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0263.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0264.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0265.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0266.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0267.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0268.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0269.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0270.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0271.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0272.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0273.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0274.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0275.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0276.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0277.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0278.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0279.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0280.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0281.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0282.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0283.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0284.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0285.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0286.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0287.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0288.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0289.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0290.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0291.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0292.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0293.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0294.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0295.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0296.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0297.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0298.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0299.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0300.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0301.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0302.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0303.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0304.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0305.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0306.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0307.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0308.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0309.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0310.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0311.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0312.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0313.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0314.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0315.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0316.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0317.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0318.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0319.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0320.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0321.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0322.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0323.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0324.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0325.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0326.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0327.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0328.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0329.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0330.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0331.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0332.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0333.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0334.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0335.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0336.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0337.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0338.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0339.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0340.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0341.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0342.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0343.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0344.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0345.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0346.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0347.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0348.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0349.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0350.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0351.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0352.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0353.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0354.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0355.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0356.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0357.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.07 — Binance TR symbol-format reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.08 — multi-timeframe ensemble reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.09 — MC-dropout uncertainty reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0358.10 — paper/live execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.01 — API diagnostic reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.02 — feature-engine reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.03 — model architecture reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.04 — training-validation reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.05 — scalping execution reference; reserved for maintenance, audit notes, and future production extensions.
# INTERNAL REFERENCE 0359.06 — risk and exchange-filter reference; reserved for maintenance, audit notes, and future production extensions.
