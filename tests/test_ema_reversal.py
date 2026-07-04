"""
Tests for EmaTrendReversalSignal.

Run with:  python -m pytest tests/test_ema_reversal.py -v
"""

import sys
import os
import time
import importlib
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stub so streamlit_app can be imported without a running Streamlit
# server and without requiring all optional dependencies (xgboost, lightgbm).
# ---------------------------------------------------------------------------
def _make_streamlit_stub():
    """Return a minimal mock of streamlit so imports don't fail."""
    st = types.ModuleType("streamlit")

    class _SS(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    st.session_state = _SS()

    def _noop(*a, **kw):
        return None

    for name in (
        "set_page_config", "title", "markdown", "caption", "header",
        "number_input", "sidebar", "spinner", "success", "error",
        "info", "warning", "metric", "progress", "subheader",
        "dataframe", "bar_chart", "button", "download_button", "rerun",
        "columns", "stop",
    ):
        setattr(st, name, _noop)

    # sidebar needs to work as a context manager
    class _Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def __getattr__(self, name):
            return _noop

    st.sidebar = _Sidebar()

    # number_input must return default value when called
    def _number_input(label, min_value=None, max_value=None, value=0, step=1, **kw):
        return value

    st.number_input = _number_input

    # columns must return a list of context managers
    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def __getattr__(self, name):
            return _noop

    def _columns(spec, **kw):
        n = len(spec) if hasattr(spec, "__len__") else spec
        return [_Col() for _ in range(n)]

    st.columns = _columns
    return st


# Inject stub before importing the app module
sys.modules.setdefault("streamlit", _make_streamlit_stub())

# Suppress optional heavy deps
for _dep in ("textblob", "binomo_scraper", "xgboost", "lightgbm"):
    if _dep not in sys.modules:
        sys.modules[_dep] = types.ModuleType(_dep)

# Add repo root to path and import
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Import the class under test directly from the module
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "streamlit_app",
    os.path.join(_repo, "streamlit_app.py"),
)
_mod = _ilu.module_from_spec(_spec)

# Pre-populate session_state to survive the UI boot code
_mod.st = sys.modules["streamlit"]  # type: ignore

try:
    _spec.loader.exec_module(_mod)
except Exception:
    pass  # UI-level errors are expected when running headlessly

EmaTrendReversalSignal = _mod.EmaTrendReversalSignal  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, 0.3, n))


def _uptrend_with_pullback(lookback=30, pullback=3, reversal=True):
    """Return price array: rising trend → pullback → optional reversal up."""
    prices = list(np.linspace(100, 115, lookback + 10))
    for _ in range(pullback):
        prices.append(prices[-1] - 0.5)
    if reversal:
        prices.append(prices[-1] + 0.8)
    return np.array(prices)


def _downtrend_with_pullback(lookback=30, pullback=3, reversal=True):
    """Return price array: falling trend → pullback → optional reversal down."""
    prices = list(np.linspace(115, 100, lookback + 10))
    for _ in range(pullback):
        prices.append(prices[-1] + 0.5)
    if reversal:
        prices.append(prices[-1] - 0.8)
    return np.array(prices)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEmaTrendDirection:
    """Tests for internal trend detection helpers."""

    def test_trend_up_detected(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10)
        prices = np.linspace(100, 115, 30)
        ema = sig._ema_series(prices)
        assert sig._trend_up(ema)

    def test_trend_down_detected(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10)
        prices = np.linspace(115, 100, 30)
        ema = sig._ema_series(prices)
        assert sig._trend_down(ema)

    def test_trend_flat_not_up(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10)
        prices = np.full(30, 100.0)
        ema = sig._ema_series(prices)
        assert not sig._trend_up(ema)

    def test_trend_flat_not_down(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10)
        prices = np.full(30, 100.0)
        ema = sig._ema_series(prices)
        assert not sig._trend_down(ema)

    def test_insufficient_history_returns_false(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=50)
        prices = np.linspace(100, 115, 20)  # shorter than lookback
        ema = sig._ema_series(prices)
        assert sig._trend_up(ema) is False
        assert sig._trend_down(ema) is False


class TestConsecutiveCandles:
    """Tests for pullback-count helpers."""

    def test_consecutive_down(self):
        prices = np.array([100, 99, 98, 97, 98.5])  # 3 downs then reversal
        assert EmaTrendReversalSignal._consecutive_down_before_last(prices) == 3

    def test_consecutive_up(self):
        prices = np.array([100, 101, 102, 103, 101.5])  # 3 ups then reversal
        assert EmaTrendReversalSignal._consecutive_up_before_last(prices) == 3

    def test_no_consecutive_down(self):
        prices = np.array([100, 101, 100.5])  # last move: up before reversal
        assert EmaTrendReversalSignal._consecutive_down_before_last(prices) == 0

    def test_no_consecutive_up(self):
        prices = np.array([100, 99, 99.5])
        assert EmaTrendReversalSignal._consecutive_up_before_last(prices) == 0


class TestCheckSignalBuy:
    """BUY signal: uptrend + pullback + reversal up + delay expired."""

    def test_buy_emitted_after_delay(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=0.0
        )
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        result = sig.check_signal(prices)
        # With zero delay the signal should fire immediately
        assert result == "BUY"

    def test_buy_pending_before_delay(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0
        )
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        result = sig.check_signal(prices)
        # Delay not elapsed → still pending, no signal yet
        assert result is None
        assert sig.pending_buy_since is not None

    def test_buy_pending_then_emitted_after_delay(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=0.05
        )
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        first = sig.check_signal(prices)
        assert first is None  # too soon
        time.sleep(0.07)
        second = sig.check_signal(prices)
        assert second == "BUY"

    def test_buy_resets_when_conditions_break(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0
        )
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        sig.check_signal(prices)
        assert sig.pending_buy_since is not None
        # New prices that don't satisfy conditions (no pullback, no reversal)
        flat_prices = np.full(len(prices), 100.0)
        sig.check_signal(flat_prices)
        assert sig.pending_buy_since is None


class TestCheckSignalSell:
    """SELL signal: downtrend + pullback + reversal down + delay expired."""

    def test_sell_emitted_after_delay(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=0.0
        )
        prices = _downtrend_with_pullback(lookback=20, pullback=2, reversal=True)
        result = sig.check_signal(prices)
        assert result == "SELL"

    def test_sell_pending_before_delay(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0
        )
        prices = _downtrend_with_pullback(lookback=20, pullback=2, reversal=True)
        result = sig.check_signal(prices)
        assert result is None
        assert sig.pending_sell_since is not None

    def test_sell_requires_min_pullback(self):
        sig = EmaTrendReversalSignal(
            ema_period=5, trend_lookback=10, min_pullback_candles=3, signal_delay_sec=0.0
        )
        # Only 1 pullback candle → below min_pullback_candles
        prices = _downtrend_with_pullback(lookback=20, pullback=1, reversal=True)
        result = sig.check_signal(prices)
        assert result is None


class TestInsufficientData:
    """No signal when there is not enough price history."""

    def test_returns_none_with_short_series(self):
        sig = EmaTrendReversalSignal(ema_period=30, trend_lookback=30, min_pullback_candles=2)
        prices = np.linspace(100, 110, 10)  # way too short
        assert sig.check_signal(prices) is None


class TestUpdateParams:
    """update_params resets pending state when settings change."""

    def test_params_change_resets_pending(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0)
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        sig.check_signal(prices)
        assert sig.pending_buy_since is not None

        sig.update_params(ema_period=10, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0)
        assert sig.pending_buy_since is None

    def test_same_params_preserves_pending(self):
        sig = EmaTrendReversalSignal(ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0)
        prices = _uptrend_with_pullback(lookback=20, pullback=2, reversal=True)
        sig.check_signal(prices)
        ts_before = sig.pending_buy_since

        sig.update_params(ema_period=5, trend_lookback=10, min_pullback_candles=2, signal_delay_sec=60.0)
        assert sig.pending_buy_since == ts_before


class TestGetEmaReversal:
    """_get_ema_reversal creates / reuses per-asset instances."""

    def test_creates_new_instance(self):
        _mod._ema_reversal_instances.pop("TEST_ASSET_NEW", None)
        inst = _mod._get_ema_reversal("TEST_ASSET_NEW", 20, 20, 2, 3.0)
        assert isinstance(inst, EmaTrendReversalSignal)
        assert inst.ema_period == 20

    def test_reuses_existing_instance(self):
        _mod._ema_reversal_instances.pop("TEST_ASSET_REUSE", None)
        inst1 = _mod._get_ema_reversal("TEST_ASSET_REUSE", 20, 20, 2, 3.0)
        inst2 = _mod._get_ema_reversal("TEST_ASSET_REUSE", 20, 20, 2, 3.0)
        assert inst1 is inst2

    def test_updates_params_on_change(self):
        _mod._ema_reversal_instances.pop("TEST_ASSET_CHANGE", None)
        inst = _mod._get_ema_reversal("TEST_ASSET_CHANGE", 20, 20, 2, 3.0)
        inst2 = _mod._get_ema_reversal("TEST_ASSET_CHANGE", 30, 30, 3, 5.0)
        assert inst is inst2
        assert inst2.ema_period == 30
        assert inst2.signal_delay_sec == 5.0
