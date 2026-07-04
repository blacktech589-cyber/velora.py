# Velora AI — Conservative Trading Strategy

> **Real-time multi-asset signal generator powered by RSI/EMA ensemble, MUM(3-9) momentum spread, and a deep-learning confirmation filter.**

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Conservative Mode

Set `CONSERVATIVE_MODE = True` (default) near the top of `streamlit_app.py` to enable the conservative risk profile. All parameters are grouped in `CONSERVATIVE_CONFIG`:

```python
CONSERVATIVE_CONFIG = {
    "min_confidence":         82,    # ensemble confidence required to trade
    "rsi_overbought":         72,    # RSI overbought level (tighter than default 78)
    "rsi_oversold":           28,    # RSI oversold level  (tighter than default 22)
    "dl_confidence_threshold": 0.65, # DL filter pass threshold
    "mum_confirmation":       True,  # MUM(3-9) directional gating
    "ema7_confirmation":      True,  # EMA(7) trend agreement required
}
```

When `CONSERVATIVE_MODE = False` the system falls back to the original default thresholds (min_confidence 78, RSI 78/22, DL threshold 0.60) so there is **full backward compatibility**.

---

## MUM(3-9) Directional Rule

**MUM(3,9)** is the spread between EMA(3) and EMA(9):

```
MUM = EMA(3) − EMA(9)
```

| MUM Spread Direction | Signal Bias |
|----------------------|-------------|
| Rising (Δ > 0)       | **SELL**    |
| Falling (Δ < 0)      | **BUY**     |
| Flat (Δ = 0)         | NEUTRAL     |

**Rationale:** When the fast EMA is pulling further above the slow EMA the market is in an accelerating uptrend — a contrarian / mean-reversion conservative strategy interprets this as overhead risk and biases SELL. A narrowing spread means momentum is fading, biasing BUY.

MUM signals appear in the `MUM39` column of the signal table and are used in two places:
1. **`DLConfirmationFilter`** — adjusts stub confidence score.
2. **`ProtectionEngine.apply`** — penalises signals that conflict with MUM direction (−8 pts confidence each).

The strategy result also appears as `MUM39` in `StrategyComparator.analyze_strategies`, counted toward `Strategy_Match`.

---

## Deep-Learning Confirmation Filter

`DLConfirmationFilter` sits **between the raw ensemble signal and execution**:

```
raw signal → DLConfirmationFilter.confirm() → ProtectionEngine.apply() → final signal
```

If DL confidence is below `threshold`, a `NO-TRADE` result is returned immediately.

### Conservative behaviour

Conservative mode sets `threshold = CONSERVATIVE_CONFIG["dl_confidence_threshold"]` (default **0.65** vs 0.60 in standard mode). Fewer but higher-quality trades result.

### Stub fallback & modularity

The current implementation ships with a **deterministic, model-free stub** (`_predict_stub`) that scores trades using RSI and MUM alignment heuristics. It is intentionally transparent and safe.

To plug in a real neural network:

```python
class DLConfirmationFilter:
    def __init__(self, threshold=0.65):
        # TODO: load real model, e.g.:
        #   import torch
        #   self.model = torch.load("dl_model.pt")
        self.threshold = threshold

    def _predict_stub(self, raw_signal, rsi14, mum_signal):
        # TODO: replace with real inference, e.g.:
        #   feats = torch.tensor(features).unsqueeze(0)
        #   with torch.no_grad():
        #       return float(self.model(feats).sigmoid())
        ...
```

---

## RSI & EMA(7) Optimisation

### RSI (conservative thresholds)

| Mode         | Overbought | Oversold |
|--------------|-----------|---------|
| Default      | 78         | 22       |
| Conservative | 72         | 28       |

Tighter bands reduce false positives in ranging markets.

### EMA(7) trend confirmation

In conservative mode the `ProtectionEngine` applies an additional −8 confidence penalty when:
- Signal is **BUY** but price is **below** EMA(7) (`ema7_delta < 0`)
- Signal is **SELL** but price is **above** EMA(7) (`ema7_delta > 0`)

This eliminates counter-trend entries in short-term trending conditions.

### EMA(7) in feature engineering

`rsi_ema15_core()` now exposes `ema7_delta` (price relative to EMA(7)) alongside the existing `ema15_delta`, `atr_ratio`, and `trend_stack`. The signal table shows both `EMA7_Delta` and `EMA15_Delta`.

---

## Strategy Parameters Overview

| Parameter                  | Default | Conservative | Rationale                                    |
|----------------------------|---------|-------------|----------------------------------------------|
| `MIN_CONFIDENCE_TO_TRADE`  | 78      | 82          | Fewer, higher-conviction trades              |
| RSI overbought             | 78      | 72          | Earlier exit from overbought conditions      |
| RSI oversold               | 22      | 28          | Earlier re-entry in oversold conditions      |
| `HIGH_VOL_THRESHOLD`       | 0.045   | 0.045       | Unchanged; blocks high-vol regimes           |
| DL confidence threshold    | 0.60    | 0.65        | Requires stronger DL agreement              |
| MUM(3-9) gating penalty    | N/A     | −8 pts      | Eliminates MUM-conflicting signals           |
| EMA(7) confirmation penalty| N/A     | −8 pts      | Requires short-term trend alignment          |

---

## Running Tests

```bash
pip install pytest
python -m pytest test_strategy.py -v
```

Tests cover:

- `TestMUMSignal` — rising/falling MUM spread maps to SELL/BUY
- `TestDLConfirmationFilter` — gating, threshold, RSI/MUM alignment
- `TestProtectionEngineConservative` — EMA(7)/RSI conservative constraints; no-penalty when mode off
- `TestRSIEMA7Core` — `ema7_delta` presence and sign correctness
- `TestStrategyComparator` — `MUM39` key present, legacy keys preserved
- `TestConservativeConfig` — all required keys exist, thresholds tighter than defaults

---

## Architecture

```
SignalGenerator
  ├── generate_realistic_prices()
  ├── calculate_rsi()
  ├── calculate_ema()
  ├── calculate_atr()
  ├── calculate_macd()
  ├── calculate_mum()          ← NEW: EMA(fast) − EMA(slow)
  ├── calculate_mum_signal()   ← NEW: rising→SELL, falling→BUY
  └── calculate_ultra_400_features()

DLConfirmationFilter           ← NEW: modular DL gating
  ├── confirm()                ←   public API
  └── _predict_stub()          ←   deterministic fallback (TODO: swap DL model)

ProtectionEngine
  └── apply()                  ← UPDATED: conservative RSI/EMA(7)/MUM thresholds

RSIRegimeEnsemble              ← unchanged ensemble
StrategyComparator
  └── analyze_strategies()     ← UPDATED: adds MUM39 strategy

rsi_ema15_core()               ← UPDATED: adds ema7_delta
advanced_analyze()             ← UPDATED: wires MUM + DL filter
```
