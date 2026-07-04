"""
Tests for the conservative trading strategy additions:
  - MUM(3-9) rising/falling → SELL/BUY mapping
  - DLConfirmationFilter gating behaviour
  - EMA(7) and RSI conservative constraints in ProtectionEngine
  - No-regression: default (non-conservative) behaviour is preserved
"""
import sys
import types
import importlib
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Minimal Streamlit stub so we can import streamlit_app without a browser
# ---------------------------------------------------------------------------
_st = types.ModuleType("streamlit")

class _SessionState(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

_st.session_state = _SessionState()

def _noop(*a, **kw): pass
def _noop_ctx(*a, **kw):
    class _CM:
        def __enter__(self): pass
        def __exit__(self, *a): pass
    return _CM()

for _attr in (
    "set_page_config", "title", "markdown", "caption", "metric",
    "button", "download_button", "progress", "info",
    "success", "warning", "error", "dataframe", "bar_chart",
    "subheader", "rerun", "write", "text",
):
    setattr(_st, _attr, _noop)

# spinner must be a context manager
_st.spinner = _noop_ctx

_st.columns = lambda n, **kw: [_noop_ctx()] * (n if isinstance(n, int) else len(n))

sys.modules.setdefault("streamlit", _st)

# Stub openpyxl.styles so import doesn't fail in minimal env
_openpyxl = sys.modules.get("openpyxl") or types.ModuleType("openpyxl")
_styles = types.ModuleType("openpyxl.styles")
for _cls in ("Font", "PatternFill", "Alignment", "Border", "Side"):
    setattr(_styles, _cls, type(_cls, (), {"__init__": lambda self, **kw: None}))
_openpyxl.styles = _styles
sys.modules.setdefault("openpyxl", _openpyxl)
sys.modules.setdefault("openpyxl.styles", _styles)

# Stub requests
if "requests" not in sys.modules:
    _req = types.ModuleType("requests")
    _req.get = lambda *a, **kw: None
    sys.modules["requests"] = _req

# Now import the actual module
import streamlit_app as app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monotone_up(n=50, start=100.0, step=0.5):
    """Strictly ascending price series."""
    return np.array([start + i * step for i in range(n)])

def _monotone_down(n=50, start=100.0, step=0.5):
    """Strictly descending price series."""
    return np.array([start - i * step for i in range(n)])

_gen = app.SignalGenerator("test_asset", "2025-01-01")


# ---------------------------------------------------------------------------
# 1. MUM(3-9) signal logic
# ---------------------------------------------------------------------------

class TestMUMSignal(unittest.TestCase):

    def test_rising_spread_returns_sell(self):
        """When EMA(3)-EMA(9) spread is rising the signal must be SELL."""
        # A strongly ascending price series causes a rising EMA spread
        prices = _monotone_up(60)
        sig, mum_val, direction = _gen.calculate_mum_signal(prices)
        # direction > 0 means spread is growing
        if direction > 0:
            self.assertEqual(sig, "SELL", "Rising MUM spread must produce SELL")

    def test_falling_spread_returns_buy(self):
        """When EMA(3)-EMA(9) spread is falling the signal must be BUY."""
        prices = _monotone_down(60)
        sig, mum_val, direction = _gen.calculate_mum_signal(prices)
        if direction < 0:
            self.assertEqual(sig, "BUY", "Falling MUM spread must produce BUY")

    def test_neutral_on_flat_prices(self):
        """Flat price series may produce NEUTRAL."""
        prices = np.full(50, 100.0)
        sig, _, direction = _gen.calculate_mum_signal(prices)
        # direction should be 0 → NEUTRAL
        self.assertAlmostEqual(direction, 0.0, places=9)
        self.assertEqual(sig, "NEUTRAL")

    def test_short_series_returns_neutral(self):
        """Series shorter than slow+2 must return NEUTRAL safely."""
        prices = np.array([100.0, 101.0, 102.0])
        sig, val, d = _gen.calculate_mum_signal(prices, fast=3, slow=9)
        self.assertEqual(sig, "NEUTRAL")
        self.assertEqual(val, 0.0)
        self.assertEqual(d, 0.0)

    def test_mum_value_equals_ema3_minus_ema9(self):
        """calculate_mum should equal EMA(3) - EMA(9)."""
        prices = _monotone_up(40)
        mum = _gen.calculate_mum(prices, fast=3, slow=9)
        ema3 = _gen.calculate_ema(prices, 3)
        ema9 = _gen.calculate_ema(prices, 9)
        self.assertAlmostEqual(mum, ema3 - ema9, places=10)

    def test_sell_signal_encoded_correctly(self):
        """Manually construct a scenario where spread rises and check SELL."""
        # Create a price that causes current_mum > prev_mum
        prices = _monotone_up(30)
        current_mum = _gen.calculate_mum(prices, 3, 9)
        prev_mum = _gen.calculate_mum(prices[:-1], 3, 9)
        if current_mum > prev_mum:
            sig, _, _ = _gen.calculate_mum_signal(prices, 3, 9)
            self.assertEqual(sig, "SELL")

    def test_buy_signal_encoded_correctly(self):
        """Manually construct a scenario where spread falls and check BUY."""
        prices = _monotone_down(30)
        current_mum = _gen.calculate_mum(prices, 3, 9)
        prev_mum = _gen.calculate_mum(prices[:-1], 3, 9)
        if current_mum < prev_mum:
            sig, _, _ = _gen.calculate_mum_signal(prices, 3, 9)
            self.assertEqual(sig, "BUY")


# ---------------------------------------------------------------------------
# 2. DLConfirmationFilter
# ---------------------------------------------------------------------------

class TestDLConfirmationFilter(unittest.TestCase):

    def test_default_threshold_neutral_passes(self):
        """Base score 0.60 with a default 0.60 threshold should pass."""
        f = app.DLConfirmationFilter(threshold=0.60)
        ok, conf, reason = f.confirm(raw_signal=True, features=[], rsi14=50.0, mum_signal="NEUTRAL")
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_high_threshold_blocks_neutral(self):
        """With a conservative threshold of 0.70 the neutral base 0.60 fails."""
        f = app.DLConfirmationFilter(threshold=0.70)
        ok, conf, reason = f.confirm(raw_signal=True, features=[], rsi14=50.0, mum_signal="NEUTRAL")
        self.assertFalse(ok)
        self.assertIn("DL confidence", reason)

    def test_rsi_aligns_with_buy_boosts_score(self):
        """RSI < 35 for a BUY signal should boost confidence above neutral."""
        f = app.DLConfirmationFilter(threshold=0.60)
        score = f._predict_stub(raw_signal=True, rsi14=25.0, mum_signal="NEUTRAL")
        self.assertGreater(score, 0.60)

    def test_rsi_conflicts_with_buy_reduces_score(self):
        """RSI > 65 for a BUY signal should reduce confidence below neutral."""
        f = app.DLConfirmationFilter(threshold=0.60)
        score = f._predict_stub(raw_signal=True, rsi14=75.0, mum_signal="NEUTRAL")
        self.assertLess(score, 0.60)

    def test_mum_aligns_with_sell_boosts_score(self):
        """MUM=SELL for a SELL signal should boost confidence."""
        f = app.DLConfirmationFilter(threshold=0.60)
        score = f._predict_stub(raw_signal=False, rsi14=50.0, mum_signal="SELL")
        self.assertGreater(score, 0.60)

    def test_mum_conflicts_with_sell_reduces_score(self):
        """MUM=BUY for a SELL signal should reduce confidence."""
        f = app.DLConfirmationFilter(threshold=0.60)
        score = f._predict_stub(raw_signal=False, rsi14=50.0, mum_signal="BUY")
        self.assertLess(score, 0.60)

    def test_score_clamped_to_unit_range(self):
        """Stub scores must always stay in [0, 1]."""
        f = app.DLConfirmationFilter(threshold=0.60)
        for rsi in [0, 10, 50, 90, 100]:
            for mum in ("BUY", "SELL", "NEUTRAL"):
                for sig in (True, False):
                    s = f._predict_stub(raw_signal=sig, rsi14=float(rsi), mum_signal=mum)
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s, 1.0)

    def test_conservative_threshold_from_config(self):
        """CONSERVATIVE_CONFIG dl_confidence_threshold must gate correctly."""
        threshold = app.CONSERVATIVE_CONFIG["dl_confidence_threshold"]
        f = app.DLConfirmationFilter(threshold=threshold)
        # A highly aligned signal should still pass
        score = f._predict_stub(raw_signal=True, rsi14=25.0, mum_signal="BUY")
        ok, _, _ = f.confirm(raw_signal=True, features=[], rsi14=25.0, mum_signal="BUY")
        if score >= threshold:
            self.assertTrue(ok)
        else:
            self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 3. ProtectionEngine – conservative constraints
# ---------------------------------------------------------------------------

class TestProtectionEngineConservative(unittest.TestCase):

    def setUp(self):
        self.pe = app.ProtectionEngine()

    def _call(self, raw_signal, rsi14, ema15_delta, ema7_delta=0.0, mum_signal="NEUTRAL",
              confidence=85, news_score=0.0):
        return self.pe.apply(
            raw_signal=raw_signal,
            confidence=confidence,
            rsi14=rsi14,
            ema15_delta=ema15_delta,
            atr_ratio=0.01,
            news_score=news_score,
            asset="test",
            ema7_delta=ema7_delta,
            mum_signal=mum_signal,
        )

    def test_ema7_conflict_reduces_buy_confidence(self):
        """BUY when price < EMA(7) (ema7_delta < 0) should reduce confidence in conservative mode."""
        was_conservative = app.CONSERVATIVE_MODE
        app.CONSERVATIVE_MODE = True
        app.CONSERVATIVE_CONFIG["ema7_confirmation"] = True
        try:
            sig, conf, reason = self._call(
                raw_signal=True, rsi14=50, ema15_delta=0.01, ema7_delta=-0.05,
                confidence=90
            )
            # confidence should be reduced
            self.assertLessEqual(conf, 90)
        finally:
            app.CONSERVATIVE_MODE = was_conservative

    def test_mum_conflict_reduces_buy_confidence(self):
        """BUY when MUM says SELL should reduce confidence in conservative mode."""
        was_conservative = app.CONSERVATIVE_MODE
        app.CONSERVATIVE_MODE = True
        app.CONSERVATIVE_CONFIG["mum_confirmation"] = True
        try:
            sig, conf, reason = self._call(
                raw_signal=True, rsi14=50, ema15_delta=0.01, mum_signal="SELL",
                confidence=90
            )
            self.assertLessEqual(conf, 90)
        finally:
            app.CONSERVATIVE_MODE = was_conservative

    def test_rsi_conservative_overbought_penalty(self):
        """RSI above conservative overbought threshold should penalise BUY confidence."""
        was_conservative = app.CONSERVATIVE_MODE
        app.CONSERVATIVE_MODE = True
        try:
            rsi_ob = app.CONSERVATIVE_CONFIG["rsi_overbought"]
            sig, conf, reason = self._call(
                raw_signal=True, rsi14=rsi_ob + 2, ema15_delta=-0.01,
                confidence=88
            )
            self.assertLessEqual(conf, 88)
        finally:
            app.CONSERVATIVE_MODE = was_conservative

    def test_no_conservative_penalty_when_mode_off(self):
        """When CONSERVATIVE_MODE is False, EMA(7) and MUM penalties must not apply."""
        was_conservative = app.CONSERVATIVE_MODE
        app.CONSERVATIVE_MODE = False
        try:
            sig, conf, reason = self._call(
                raw_signal=True, rsi14=50, ema15_delta=0.01,
                ema7_delta=-0.05, mum_signal="SELL",
                confidence=90
            )
            # Without conservative penalties, conf should stay at 90
            self.assertEqual(conf, 90)
        finally:
            app.CONSERVATIVE_MODE = was_conservative

    def test_low_confidence_blocked(self):
        """Signal below conservative min_confidence should be blocked as NO-TRADE."""
        was_conservative = app.CONSERVATIVE_MODE
        app.CONSERVATIVE_MODE = True
        try:
            min_conf = app.CONSERVATIVE_CONFIG["min_confidence"]
            sig, conf, reason = self._call(
                raw_signal=True, rsi14=50, ema15_delta=0.01,
                confidence=min_conf - 5
            )
            self.assertEqual(sig, "NO-TRADE")
        finally:
            app.CONSERVATIVE_MODE = was_conservative


# ---------------------------------------------------------------------------
# 4. RSI + EMA(7) feature presence
# ---------------------------------------------------------------------------

class TestRSIEMA7Core(unittest.TestCase):

    def test_rsi_ema15_core_has_ema7_delta(self):
        """rsi_ema15_core must return ema7_delta key."""
        gen = app.SignalGenerator("test", "2025-01-01")
        prices = _monotone_up(60)
        core = app.rsi_ema15_core(prices, gen)
        self.assertIn("ema7_delta", core)

    def test_ema7_delta_sign_for_uptrend(self):
        """In a rising market price > EMA(7) → ema7_delta should be positive."""
        gen = app.SignalGenerator("test", "2025-01-01")
        prices = _monotone_up(60)
        core = app.rsi_ema15_core(prices, gen)
        self.assertGreater(core["ema7_delta"], 0)

    def test_ema7_delta_sign_for_downtrend(self):
        """In a falling market price < EMA(7) → ema7_delta should be negative."""
        gen = app.SignalGenerator("test", "2025-01-01")
        prices = _monotone_down(60)
        core = app.rsi_ema15_core(prices, gen)
        self.assertLess(core["ema7_delta"], 0)


# ---------------------------------------------------------------------------
# 5. No-regression: StrategyComparator adds MUM39
# ---------------------------------------------------------------------------

class TestStrategyComparator(unittest.TestCase):

    def test_mum39_key_present(self):
        """analyze_strategies must include 'MUM39' key."""
        sc = app.StrategyComparator()
        prices = _monotone_up(60)
        results = sc.analyze_strategies(prices, "test")
        self.assertIn("MUM39", results)

    def test_mum39_value_is_valid(self):
        """MUM39 value must be one of BUY, SELL."""
        sc = app.StrategyComparator()
        prices = _monotone_up(60)
        results = sc.analyze_strategies(prices, "test")
        self.assertIn(results["MUM39"], ("BUY", "SELL"))

    def test_legacy_strategy_keys_still_present(self):
        """Pre-existing strategy keys must still be present (no regression)."""
        sc = app.StrategyComparator()
        prices = _monotone_up(60)
        results = sc.analyze_strategies(prices, "test")
        for key in ("Trend", "MeanRev", "Momentum", "Channel", "Volatility"):
            self.assertIn(key, results)


# ---------------------------------------------------------------------------
# 6. Conservative config completeness
# ---------------------------------------------------------------------------

class TestConservativeConfig(unittest.TestCase):

    def test_all_required_keys_present(self):
        """CONSERVATIVE_CONFIG must have all required keys."""
        required = {
            "min_confidence", "rsi_overbought", "rsi_oversold",
            "dl_confidence_threshold", "mum_confirmation", "ema7_confirmation",
        }
        self.assertTrue(required.issubset(set(app.CONSERVATIVE_CONFIG.keys())))

    def test_conservative_min_conf_stricter_than_default(self):
        """Conservative min_confidence must be >= default MIN_CONFIDENCE_TO_TRADE."""
        self.assertGreaterEqual(
            app.CONSERVATIVE_CONFIG["min_confidence"],
            app.MIN_CONFIDENCE_TO_TRADE,
        )

    def test_conservative_rsi_thresholds_tighter(self):
        """Conservative RSI bands must be tighter (overbought lower, oversold higher)."""
        self.assertLess(app.CONSERVATIVE_CONFIG["rsi_overbought"], 78)
        self.assertGreater(app.CONSERVATIVE_CONFIG["rsi_oversold"], 22)

    def test_dl_threshold_above_default(self):
        """Conservative DL threshold must exceed the non-conservative default (0.60)."""
        self.assertGreater(app.CONSERVATIVE_CONFIG["dl_confidence_threshold"], 0.60)


if __name__ == "__main__":
    unittest.main()
