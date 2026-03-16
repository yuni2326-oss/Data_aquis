# Predictive Maintenance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FFT 기반 예지보전 파이프라인 구현 — 30초 주기로 5초 분량 raw 파형을 수집하여 z-score + Isolation Forest로 장비 이상을 감지하고 UI에 표시한다.

**Architecture:** `AcquisitionWorker.data_ready` → `FeatureCollector` (버퍼 축적 + 30초 타이머) → `AnomalyDetector` (z-score + IsolationForest) → `MainWindow` (상태바 + AnomalyPlot 그래프). 각 컴포넌트는 Qt signal로 연결되며 독립적으로 테스트 가능하다.

**Tech Stack:** Python 3.x, PyQt5, numpy, scikit-learn (IsolationForest), pyqtgraph, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-predictive-maintenance-design.md`

---

## Chunk 1: FFT Feature Extraction (`features.py`)

**Files:**
- Create: `usb5133_daq/analysis/features.py`
- Create: `tests/test_features.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
# tests/test_features.py
import numpy as np
import pytest
from usb5133_daq.analysis.features import extract_features

SAMPLE_RATE = 1_000_000.0
RECORD_LEN = 5_000  # 5ms at 1MHz — small but fast for tests


def _sine(freq_hz: float, n: int = RECORD_LEN, sr: float = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq_hz * t)


class TestExtractFeatures:
    def test_returns_shape_7(self):
        vec = extract_features(_sine(1000.0), SAMPLE_RATE)
        assert vec.shape == (7,)

    def test_returns_float64(self):
        vec = extract_features(_sine(1000.0), SAMPLE_RATE)
        assert vec.dtype == np.float64

    def test_dominant_freq_correct(self):
        freq = 1000.0
        vec = extract_features(_sine(freq), SAMPLE_RATE)
        # dominant_freq (index 0) should be close to 1000 Hz
        assert abs(vec[0] - freq) < 50.0  # within 50 Hz tolerance

    def test_dominant_mag_positive(self):
        vec = extract_features(_sine(1000.0), SAMPLE_RATE)
        assert vec[1] > 0.0

    def test_zero_signal_returns_zeros_no_nan(self):
        zero_wave = np.zeros(RECORD_LEN)
        vec = extract_features(zero_wave, SAMPLE_RATE)
        assert vec.shape == (7,)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))
        np.testing.assert_array_equal(vec, np.zeros(7))

    def test_thd_is_nonnegative(self):
        vec = extract_features(_sine(1000.0), SAMPLE_RATE)
        assert vec[4] >= 0.0

    def test_spectral_centroid_positive(self):
        vec = extract_features(_sine(1000.0), SAMPLE_RATE)
        assert vec[6] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_features.py -v
```

Expected: `ImportError: cannot import name 'extract_features'`

- [ ] **Step 3: Implement `features.py`**

Create `usb5133_daq/analysis/features.py`:

```python
# usb5133_daq/analysis/features.py
from __future__ import annotations

import numpy as np

from usb5133_daq.analysis.fft import compute_fft

_PEAK_NEIGHBOR = 5  # bins to exclude around dominant peak when computing noise floor


def extract_features(waveform: np.ndarray, sample_rate: float) -> np.ndarray:
    """FFT 특징 벡터 추출.

    Args:
        waveform: 1D array shape (N,) — single channel
        sample_rate: sample rate in Hz

    Returns:
        np.ndarray shape (7,), dtype float64:
            [0] dominant_freq (Hz)
            [1] dominant_mag
            [2] second_harmonic_mag
            [3] third_harmonic_mag
            [4] thd  = (2nd + 3rd) / dominant_mag
            [5] noise_floor_rms
            [6] spectral_centroid (Hz)
    """
    freqs, mags = compute_fft(waveform, sample_rate)

    dominant_idx = int(np.argmax(mags))
    dominant_mag = float(mags[dominant_idx])

    # Zero-signal guard — avoids NaN propagation
    if dominant_mag < 1e-9:
        return np.zeros(7, dtype=np.float64)

    dominant_freq = float(freqs[dominant_idx])

    # Harmonic magnitudes — find bin closest to 2x and 3x dominant_freq
    def _mag_at_freq(target_hz: float) -> float:
        if target_hz <= 0 or target_hz >= freqs[-1]:
            return 0.0
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(mags[idx])

    second_mag = _mag_at_freq(2.0 * dominant_freq)
    third_mag = _mag_at_freq(3.0 * dominant_freq)
    thd = (second_mag + third_mag) / dominant_mag

    # Noise floor RMS — exclude _PEAK_NEIGHBOR bins around dominant
    mask = np.ones(len(mags), dtype=bool)
    lo = max(0, dominant_idx - _PEAK_NEIGHBOR)
    hi = min(len(mags), dominant_idx + _PEAK_NEIGHBOR + 1)
    mask[lo:hi] = False
    noise_rms = float(np.sqrt(np.mean(mags[mask] ** 2))) if mask.any() else 0.0

    # Spectral centroid
    total = float(np.sum(mags))
    centroid = float(np.sum(freqs * mags) / total) if total > 0 else 0.0

    return np.array(
        [dominant_freq, dominant_mag, second_mag, third_mag, thd, noise_rms, centroid],
        dtype=np.float64,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_features.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add usb5133_daq/analysis/features.py tests/test_features.py
git commit -m "feat: add FFT feature extraction for predictive maintenance"
```

---

## Chunk 2: Feature Collector (`feature_collector.py`)

**Files:**
- Create: `usb5133_daq/analysis/feature_collector.py`
- Create: `tests/test_feature_collector.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/test_feature_collector.py`:

```python
# tests/test_feature_collector.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import QCoreApplication
import sys

# QApplication needed for QObject/QTimer
app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.analysis.feature_collector import FeatureCollector

SAMPLE_RATE = 100_000.0  # 100 kHz for fast tests
RECORD_LEN = 1000        # 10ms per fetch
WINDOW_SEC = 0.05        # 50ms window — needs 5000 samples


def _make_waveform(n_channels: int = 1, record_len: int = RECORD_LEN) -> np.ndarray:
    """shape (n_channels, record_len) sine waveform"""
    t = np.arange(record_len) / SAMPLE_RATE
    row = np.sin(2 * np.pi * 1000.0 * t)
    return np.tile(row, (n_channels, 1))


class TestFeatureCollector:
    def test_channel_slice(self):
        """on_data should use only waveform[channel]"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,  # timer won't fire during test
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        wave = _make_waveform(n_channels=2)
        wave[1] *= 99  # channel 1 is very different
        collector.on_data(wave)
        # buffer should contain only channel-0 samples (values near ±1)
        buf = list(collector._buf)
        assert max(abs(s) for s in buf) < 2.0

    def test_buffer_skips_when_insufficient_samples(self):
        """_on_timer should not emit features_ready if buffer < n_needed"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        received = []
        collector.features_ready.connect(lambda v: received.append(v))

        # Feed less than n_needed samples (WINDOW_SEC * SAMPLE_RATE = 5000)
        wave = _make_waveform()  # only 1000 samples
        collector.on_data(wave)
        collector._on_timer()  # manually trigger

        assert len(received) == 0

    def test_emits_when_sufficient_samples(self):
        """_on_timer should emit features_ready once buffer >= n_needed"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        received = []
        collector.features_ready.connect(lambda v: received.append(v))

        # Fill buffer: n_needed = 5000 samples, 5 calls × 1000 = 5000
        wave = _make_waveform()
        for _ in range(5):
            collector.on_data(wave)
        collector._on_timer()

        assert len(received) == 1
        assert received[0].shape == (7,)

    def test_timer_starts_on_first_on_data(self):
        """QTimer.start() called exactly once on first on_data"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=30.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        assert not collector._timer.isActive()
        collector.on_data(_make_waveform())
        assert collector._timer.isActive()
        # second call should not restart timer
        interval_before = collector._timer.interval()
        collector.on_data(_make_waveform())
        assert collector._timer.interval() == interval_before
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_feature_collector.py -v
```

Expected: `ImportError: cannot import name 'FeatureCollector'`

- [ ] **Step 3: Implement `feature_collector.py`**

Create `usb5133_daq/analysis/feature_collector.py`:

```python
# usb5133_daq/analysis/feature_collector.py
from __future__ import annotations

import collections

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from usb5133_daq.analysis.features import extract_features


class FeatureCollector(QObject):
    """raw 파형 버퍼 축적 후 주기적으로 FFT 특징 벡터를 추출해 emit.

    Signals:
        features_ready: np.ndarray shape (7,)
    """

    features_ready = pyqtSignal(object)  # np.ndarray shape (7,)

    def __init__(
        self,
        sample_rate: float,
        cycle_sec: float = 30.0,
        collect_window_sec: float = 5.0,
        channel: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._collect_window_sec = collect_window_sec
        self._channel = channel
        self._n_needed = int(collect_window_sec * sample_rate)

        buffer_size = self._n_needed * 4
        self._buf: collections.deque = collections.deque(maxlen=buffer_size)
        self._started = False

        self._timer = QTimer(self)
        self._timer.setInterval(int(cycle_sec * 1000))
        self._timer.timeout.connect(self._on_timer)

    def on_data(self, waveform: np.ndarray) -> None:
        """AcquisitionWorker.data_ready 슬롯.

        Args:
            waveform: shape (num_channels, record_length)
        """
        self._buf.extend(waveform[self._channel])
        if not self._started:
            self._timer.start()
            self._started = True

    def _on_timer(self) -> None:
        if len(self._buf) < self._n_needed:
            return
        samples = np.array(list(self._buf))[-self._n_needed:]
        vec = extract_features(samples, self._sample_rate)
        self.features_ready.emit(vec)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_feature_collector.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add usb5133_daq/analysis/feature_collector.py tests/test_feature_collector.py
git commit -m "feat: add FeatureCollector — buffer accumulation and periodic feature extraction"
```

---

## Chunk 3: Anomaly Detector (`anomaly_detector.py`)

**Files:**
- Create: `usb5133_daq/analysis/anomaly_detector.py`
- Create: `tests/test_anomaly_detector.py`
- Modify: `requirements.txt` (add scikit-learn)

---

- [ ] **Step 1: Add scikit-learn to requirements.txt**

Edit `requirements.txt` — append:
```
scikit-learn>=1.0
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_anomaly_detector.py`:

```python
# tests/test_anomaly_detector.py
import math
import numpy as np
import pytest
import sys
from PyQt5.QtCore import QCoreApplication

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.analysis.anomaly_detector import AnomalyDetector, AnomalyResult

N_FEATURES = 7
BASELINE = 20  # 20+ recommended for stable z-score baseline (spec section 3.3)


def _normal_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1.0, scale=0.05, size=N_FEATURES)


def _fill_baseline(detector: AnomalyDetector, count: int):
    for i in range(count):
        detector.on_features(_normal_vec(seed=i))


class TestAnomalyDetector:
    def test_learning_label_during_baseline(self):
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        _fill_baseline(detector, BASELINE - 1)

        assert len(results) == BASELINE - 1
        for r in results:
            assert r.label == "LEARNING"

    def test_learning_progress_increments(self):
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        _fill_baseline(detector, BASELINE - 1)

        for i, r in enumerate(results):
            assert r.baseline_progress == i + 1
            assert r.baseline_total == BASELINE

    def test_nth_sample_completes_fit_but_emits_learning(self):
        """The baseline_count-th sample completes model fitting but still emits LEARNING."""
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        _fill_baseline(detector, BASELINE)

        assert results[-1].label == "LEARNING"

    def test_normal_vector_after_baseline_is_normal(self):
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        _fill_baseline(detector, BASELINE)
        # Feed a near-identical normal vector
        detector.on_features(_normal_vec(seed=99))

        last = results[-1]
        assert last.label == "NORMAL"
        assert not math.isnan(last.score)
        assert not math.isnan(last.zscore_max)

    def test_anomaly_vector_is_warning_or_alarm(self):
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        _fill_baseline(detector, BASELINE)
        # Feed a wildly different vector
        anomaly_vec = _normal_vec(seed=0) * 1000.0
        detector.on_features(anomaly_vec)

        last = results[-1]
        assert last.label in ("WARNING", "ALARM")

    def test_score_nan_during_learning(self):
        detector = AnomalyDetector(baseline_count=BASELINE)
        results = []
        detector.result_ready.connect(results.append)

        detector.on_features(_normal_vec(seed=0))

        assert math.isnan(results[0].score)
        assert math.isnan(results[0].zscore_max)
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_anomaly_detector.py -v
```

Expected: `ImportError: cannot import name 'AnomalyDetector'`

- [ ] **Step 4: Implement `anomaly_detector.py`**

Create `usb5133_daq/analysis/anomaly_detector.py`:

```python
# usb5133_daq/analysis/anomaly_detector.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.ensemble import IsolationForest

_CONTAMINATION = 0.05


@dataclass
class AnomalyResult:
    label: str           # "LEARNING" | "NORMAL" | "WARNING" | "ALARM"
    score: float         # IsolationForest score_samples() — NaN during LEARNING
    zscore_max: float    # max |z-score| across features — NaN during LEARNING
    baseline_progress: int
    baseline_total: int


class AnomalyDetector(QObject):
    """z-score + IsolationForest 기반 이상 감지.

    Signals:
        result_ready: AnomalyResult
    """

    result_ready = pyqtSignal(object)  # AnomalyResult

    def __init__(self, baseline_count: int = 10, parent=None):
        super().__init__(parent)
        self._baseline_count = baseline_count
        self._baseline: list[np.ndarray] = []
        self._if_model: IsolationForest | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def on_features(self, vec: np.ndarray) -> None:
        """FeatureCollector.features_ready 슬롯."""
        if len(self._baseline) < self._baseline_count:
            self._baseline.append(vec)
            if len(self._baseline) == self._baseline_count:
                X = np.vstack(self._baseline)  # shape (baseline_count, 7)
                self._if_model = IsolationForest(contamination=_CONTAMINATION).fit(X)
                self._mean = X.mean(axis=0)
                self._std = X.std(axis=0) + 1e-9  # prevent divide-by-zero
            self.result_ready.emit(
                AnomalyResult(
                    label="LEARNING",
                    score=math.nan,
                    zscore_max=math.nan,
                    baseline_progress=len(self._baseline),
                    baseline_total=self._baseline_count,
                )
            )
            return

        zscore_max = float(np.max(np.abs((vec - self._mean) / self._std)))
        if_score = float(self._if_model.score_samples([vec])[0])

        if zscore_max < 2.0 and if_score > -0.1:
            label = "NORMAL"
        elif zscore_max < 3.0 and if_score > -0.3:
            label = "WARNING"
        else:
            label = "ALARM"

        self.result_ready.emit(
            AnomalyResult(
                label=label,
                score=if_score,
                zscore_max=zscore_max,
                baseline_progress=self._baseline_count,
                baseline_total=self._baseline_count,
            )
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_anomaly_detector.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 6: Run full test suite to confirm no regressions**

```
pytest -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add usb5133_daq/analysis/anomaly_detector.py tests/test_anomaly_detector.py requirements.txt
git commit -m "feat: add AnomalyDetector with z-score and IsolationForest anomaly detection"
```

---

## Chunk 4: Anomaly Score Plot (`anomaly_plot.py`)

**Files:**
- Create: `usb5133_daq/ui/anomaly_plot.py`

Note: `pyqtgraph>=0.13.0` is already in `requirements.txt`. No new dependency needed.

---

- [ ] **Step 1: Implement `anomaly_plot.py`**

Create `usb5133_daq/ui/anomaly_plot.py`:

```python
# usb5133_daq/ui/anomaly_plot.py
from __future__ import annotations

import math
from collections import deque

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from usb5133_daq.analysis.anomaly_detector import AnomalyResult

_HISTORY = 50
_THRESHOLD_NORMAL = -0.1   # above this = NORMAL
_THRESHOLD_WARNING = -0.3  # above this = WARNING, below = ALARM

_BG_COLORS = {
    "LEARNING": QColor(140, 140, 140, 40),
    "NORMAL":   QColor(0, 180, 0, 40),
    "WARNING":  QColor(255, 200, 0, 40),
    "ALARM":    QColor(220, 50, 50, 40),
}


class AnomalyPlot(QWidget):
    """이상 점수 시계열 그래프.

    update_result(result) 호출로 갱신. score는 IsolationForest score_samples() 값.
    높을수록 정상, 낮을수록 이상.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scores: deque[float] = deque(maxlen=_HISTORY)
        self._labels: deque[str] = deque(maxlen=_HISTORY)

        self._plot_widget = pg.PlotWidget(title="이상 점수 (Anomaly Score)")
        self._plot_widget.setLabel("left", "Score")
        self._plot_widget.setLabel("bottom", "주기 (최근 50회)")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setYRange(-0.6, 0.2)

        # Threshold lines — use Qt.DashLine (from PyQt5.QtCore import Qt)
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=_THRESHOLD_NORMAL, angle=0,
                            pen=pg.mkPen("g", width=1, style=Qt.DashLine),
                            label="NORMAL", labelOpts={"position": 0.05})
        )
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=_THRESHOLD_WARNING, angle=0,
                            pen=pg.mkPen("r", width=1, style=Qt.DashLine),
                            label="ALARM", labelOpts={"position": 0.05})
        )

        self._curve = self._plot_widget.plot(
            pen=pg.mkPen("#00BFFF", width=2),
            symbol="o",
            symbolSize=5,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_result(self, result: AnomalyResult) -> None:
        """AnomalyDetector.result_ready 슬롯에서 호출.

        Note: method is named update_result (not update) to avoid shadowing QWidget.update().
        """
        score = result.score if not math.isnan(result.score) else float("nan")
        self._scores.append(score)
        self._labels.append(result.label)

        # Update background color based on latest label
        color = _BG_COLORS.get(result.label, _BG_COLORS["LEARNING"])
        self._plot_widget.setBackground(color)

        valid = [(i, s) for i, s in enumerate(self._scores) if not math.isnan(s)]
        if valid:
            xs, ys = zip(*valid)
            self._curve.setData(list(xs), list(ys))
        else:
            self._curve.setData([], [])
```

- [ ] **Step 2: Verify widget construction works (smoke test)**

```
python -c "from PyQt5.QtWidgets import QApplication; import sys; app = QApplication(sys.argv); from usb5133_daq.ui.anomaly_plot import AnomalyPlot; w = AnomalyPlot(); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add usb5133_daq/ui/anomaly_plot.py
git commit -m "feat: add AnomalyPlot widget for anomaly score visualization"
```

---

## Chunk 5: MainWindow Integration

**Files:**
- Modify: `usb5133_daq/ui/main_window.py`

---

- [ ] **Step 1: Add imports to `main_window.py`**

At the top of `usb5133_daq/ui/main_window.py`, add after existing imports:

```python
from usb5133_daq.analysis.feature_collector import FeatureCollector
from usb5133_daq.analysis.anomaly_detector import AnomalyDetector, AnomalyResult
from usb5133_daq.ui.anomaly_plot import AnomalyPlot
```

- [ ] **Step 2: Add instance variables in `__init__`**

In `MainWindow.__init__`, after `self._last_sample_rate = 1_000_000.0`, add:

```python
self._collector: FeatureCollector | None = None
self._detector: AnomalyDetector | None = None
```

- [ ] **Step 3: Add UI controls — anomaly settings panel**

In `_build_control_panel()`, after the voltage range combo block (before `layout.addStretch()`), add:

```python
layout.addWidget(QLabel("|"))
layout.addWidget(QLabel("수집주기(초):"))
self._edit_cycle = QLineEdit("30")
self._edit_cycle.setFixedWidth(45)
layout.addWidget(self._edit_cycle)

layout.addWidget(QLabel("수집창(초):"))
self._edit_window = QLineEdit("5")
self._edit_window.setFixedWidth(40)
layout.addWidget(self._edit_window)

layout.addWidget(QLabel("기준선횟수:"))
self._edit_baseline = QLineEdit("10")
self._edit_baseline.setFixedWidth(45)
self._edit_baseline.setPlaceholderText("20 이상 권장")
layout.addWidget(self._edit_baseline)
```

- [ ] **Step 4: Add AnomalyPlot to the splitter in `_build_ui`**

In `_build_ui()`, **remove** the existing `splitter.setSizes([400, 250])` line and **replace** the block ending at that line with:

```python
self._anomaly_plot = AnomalyPlot()
splitter.addWidget(self._anomaly_plot)
splitter.setSizes([400, 200, 150])  # three panels: waveform, FFT, anomaly
```

- [ ] **Step 5: Wire up FeatureCollector and AnomalyDetector in `_on_start`**

In `_on_start()`, after `self._worker.start()`, add:

```python
try:
    cycle_sec = float(self._edit_cycle.text())
    window_sec = float(self._edit_window.text())
    baseline_count = int(self._edit_baseline.text())
except ValueError:
    cycle_sec, window_sec, baseline_count = 30.0, 5.0, 10

self._collector = FeatureCollector(
    sample_rate=self._last_sample_rate,
    cycle_sec=cycle_sec,
    collect_window_sec=window_sec,
    channel=0,
)
self._detector = AnomalyDetector(baseline_count=baseline_count)
self._worker.data_ready.connect(self._collector.on_data)
self._collector.features_ready.connect(self._detector.on_features)
self._detector.result_ready.connect(self._on_anomaly_result)
```

- [ ] **Step 6: Tear down collector/detector in `_on_stop`**

In `_on_stop()`, after `self._worker = None`, add:

```python
self._collector = None
self._detector = None
```

- [ ] **Step 7: Add `_on_anomaly_result` handler**

Add a new method to `MainWindow`:

```python
def _on_anomaly_result(self, result: AnomalyResult) -> None:
    self._anomaly_plot.update_result(result)
    if result.label == "LEARNING":
        tag = f"[학습 중 {result.baseline_progress}/{result.baseline_total}]"
    elif result.label == "NORMAL":
        tag = "[정상]"
    elif result.label == "WARNING":
        tag = "[WARNING]"
    else:
        tag = "[ALARM]"
    self._status_bar.showMessage(f"{tag} 수집 중...")
```

- [ ] **Step 8: Smoke-test the full app with --mock**

```
python main.py --mock
```

Expected:
- App opens without errors
- Three panels visible: waveform, FFT, anomaly score
- New controls (수집주기, 수집창, 기준선횟수) appear in settings panel
- Clicking 연결 → 시작 starts acquisition

After ~30 seconds (or lower the 수집주기 to 5 for quick test), status bar should show `[학습 중 1/10] 수집 중...`

- [ ] **Step 9: Run full test suite**

```
pytest -v
```

Expected: all tests PASS

- [ ] **Step 10: Commit**

```bash
git add usb5133_daq/ui/main_window.py
git commit -m "feat: integrate predictive maintenance pipeline into MainWindow UI"
```

---

## Chunk 6: Final Polish & Push

- [ ] **Step 1: Run full test suite one last time**

```
pytest -v
```

Expected: all tests PASS

- [ ] **Step 2: Push to GitHub**

```bash
git push
```

- [ ] **Step 3: Done**

전체 파이프라인 완성:
- `features.py` — FFT 특징 7개 추출
- `feature_collector.py` — 30초 주기, 5초 창 raw 버퍼 축적
- `anomaly_detector.py` — z-score + IsolationForest 이상 감지
- `anomaly_plot.py` — 이상 점수 시계열 그래프
- `main_window.py` — UI 통합, 파라미터 설정 가능
