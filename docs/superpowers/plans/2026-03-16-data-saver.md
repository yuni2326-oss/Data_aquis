# Data Saver Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically save raw waveform samples and anomaly detection results as CSV file pairs in a `results/` subfolder, one pair per 30-second analysis cycle.

**Architecture:** A new `DataSaver(QObject)` receives two signals — `FeatureCollector.raw_ready(datetime, samples)` and `AnomalyDetector.result_ready(AnomalyResult)` — and writes a `_raw.csv` + `_result.csv` pair per cycle. `FeatureCollector` gains a new `raw_ready` signal emitted just before `features_ready` in `_on_timer`. `MainWindow` constructs and wires `DataSaver` on start, disconnects and clears it on stop.

**Tech Stack:** Python stdlib `csv.writer`, PyQt5 signals/slots, `pathlib.Path`, `datetime`

---

## Chunk 1: DataSaver class + FeatureCollector signal

### Task 1: DataSaver — tests first

**Files:**
- Create: `tests/test_data_saver.py`

**Background for implementer:**

`DataSaver` is a new `QObject` in `usb5133_daq/storage/data_saver.py`. It receives two slots:
- `on_raw(timestamp: datetime, samples: np.ndarray)` — stores `(timestamp, samples)` in `_pending_raw`
- `on_result(result: AnomalyResult)` — retrieves `_pending_raw`, writes two CSV files, clears `_pending_raw`

File naming: `timestamp.strftime("%Y-%m-%d %H-%M-%S")` → stem. Files: `<stem>_raw.csv`, `<stem>_result.csv`.

Raw CSV columns: `sample_index,voltage` (one row per sample).
Result CSV columns: `timestamp,label,score,zscore_max,baseline_progress,baseline_total` where `timestamp` is formatted as `YYYY-MM-DD HH:MM:SS` (colons, ISO 8601 — different from filename stem which uses hyphens for Windows compatibility).

Tests call slots directly — no Qt event loop needed. Use `tmp_path` for isolated directories.

`AnomalyResult` import: `from usb5133_daq.analysis.anomaly_detector import AnomalyResult`

- [ ] **Step 1: Write all 8 test cases**

```python
# tests/test_data_saver.py
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.analysis.anomaly_detector import AnomalyResult
from usb5133_daq.storage.data_saver import DataSaver

N_SAMPLES = 500  # small for fast tests


def _ts() -> datetime:
    return datetime(2026, 3, 16, 14, 30, 0)


def _ts2() -> datetime:
    return datetime(2026, 3, 16, 14, 31, 0)


def _samples() -> np.ndarray:
    return np.linspace(0.0, 1.0, N_SAMPLES)


def _normal_result() -> AnomalyResult:
    return AnomalyResult(
        label="NORMAL",
        score=-0.05,
        zscore_max=1.2,
        baseline_progress=20,
        baseline_total=20,
    )


def _learning_result() -> AnomalyResult:
    return AnomalyResult(
        label="LEARNING",
        score=math.nan,
        zscore_max=math.nan,
        baseline_progress=5,
        baseline_total=20,
    )


class TestDataSaver:
    def test_creates_results_dir(self, tmp_path):
        save_dir = tmp_path / "results"
        assert not save_dir.exists()
        DataSaver(save_dir=save_dir)
        assert save_dir.is_dir()

    def test_raw_csv_written(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        saver.on_result(_normal_result())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        raw_file = tmp_path / f"{stem}_raw.csv"
        assert raw_file.exists()

        with raw_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sample_index", "voltage"]
        assert len(rows) == N_SAMPLES + 1  # header + data rows

    def test_result_csv_written(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        result = _normal_result()
        saver.on_result(result)

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        result_file = tmp_path / f"{stem}_result.csv"
        assert result_file.exists()

        with result_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["timestamp", "label", "score", "zscore_max",
                           "baseline_progress", "baseline_total"]
        assert len(rows) == 2  # header + 1 data row
        assert rows[1][1] == "NORMAL"

    def test_filename_format(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = datetime(2026, 3, 16, 9, 5, 3)
        saver.on_raw(ts, _samples())
        saver.on_result(_normal_result())

        assert (tmp_path / "2026-03-16 09-05-03_raw.csv").exists()
        assert (tmp_path / "2026-03-16 09-05-03_result.csv").exists()

    def test_result_without_raw(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        with patch("usb5133_daq.storage.data_saver._log") as mock_log:
            saver.on_result(_normal_result())
            mock_log.warning.assert_called_once()
        assert list(tmp_path.iterdir()) == []

    def test_multiple_cycles(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts1, ts2 = _ts(), _ts2()

        saver.on_raw(ts1, _samples())
        saver.on_result(_normal_result())

        saver.on_raw(ts2, _samples())
        saver.on_result(_normal_result())

        stem1 = ts1.strftime("%Y-%m-%d %H-%M-%S")
        stem2 = ts2.strftime("%Y-%m-%d %H-%M-%S")
        assert (tmp_path / f"{stem1}_raw.csv").exists()
        assert (tmp_path / f"{stem1}_result.csv").exists()
        assert (tmp_path / f"{stem2}_raw.csv").exists()
        assert (tmp_path / f"{stem2}_result.csv").exists()

    def test_oserror_on_write(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        saver.on_raw(_ts(), _samples())
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            with patch("usb5133_daq.storage.data_saver._log") as mock_log:
                saver.on_result(_normal_result())
                mock_log.warning.assert_called_once()

    def test_result_csv_nan_values(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        saver.on_result(_learning_result())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        with (tmp_path / f"{stem}_result.csv").open() as f:
            rows = list(csv.reader(f))
        assert rows[1][2] == "nan"   # score column
        assert rows[1][3] == "nan"   # zscore_max column
```

- [ ] **Step 2: Run tests to verify they all fail (DataSaver does not exist yet)**

```
cd "e:/My project/Data_aquis"
pytest tests/test_data_saver.py -v
```

Expected: `ModuleNotFoundError: No module named 'usb5133_daq.storage.data_saver'`

---

### Task 2: DataSaver — implementation

**Files:**
- Create: `usb5133_daq/storage/data_saver.py`

Note: `usb5133_daq/storage/__init__.py` already exists — do not recreate it.

- [ ] **Step 3: Implement DataSaver**

```python
# usb5133_daq/storage/data_saver.py
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

from usb5133_daq.analysis.anomaly_detector import AnomalyResult

_log = logging.getLogger(__name__)


class DataSaver(QObject):
    """수집 사이클마다 raw 파형과 이상 감지 결과를 CSV 쌍으로 저장.

    Slots:
        on_raw(datetime, np.ndarray): FeatureCollector.raw_ready 수신
        on_result(AnomalyResult): AnomalyDetector.result_ready 수신
    """

    def __init__(self, save_dir: str | Path = "results", parent=None):
        super().__init__(parent)
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)  # OSError propagates to caller
        self._pending_raw: tuple[datetime, np.ndarray] | None = None

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, samples: np.ndarray) -> None:
        """FeatureCollector.raw_ready 슬롯.

        Architecturally, a pre-existing entry cannot exist under the single-thread
        constraint. This overwrite is a defensive guard only.
        """
        self._pending_raw = (timestamp, samples)

    @pyqtSlot(object)
    def on_result(self, result: AnomalyResult) -> None:
        """AnomalyDetector.result_ready 슬롯."""
        if self._pending_raw is None:
            _log.warning("DataSaver.on_result: no pending raw data — skipping write")
            return

        timestamp, samples = self._pending_raw
        self._pending_raw = None

        stem = timestamp.strftime("%Y-%m-%d %H-%M-%S")
        try:
            self._write_raw(stem, samples)
            self._write_result(stem, timestamp, result)
        except OSError as exc:
            _log.warning("DataSaver: failed to write CSV files: %s", exc)

    def _write_raw(self, stem: str, samples: np.ndarray) -> None:
        path = self._save_dir / f"{stem}_raw.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "voltage"])
            for i, v in enumerate(samples):
                writer.writerow([i, v])

    def _write_result(self, stem: str, timestamp: datetime, result: AnomalyResult) -> None:
        path = self._save_dir / f"{stem}_result.csv"
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")  # ISO 8601 colons for CSV
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "score", "zscore_max",
                              "baseline_progress", "baseline_total"])
            writer.writerow([
                ts_str,
                result.label,
                result.score,
                result.zscore_max,
                result.baseline_progress,
                result.baseline_total,
            ])
```

- [ ] **Step 4: Run tests to verify they all pass**

```
pytest tests/test_data_saver.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add usb5133_daq/storage/data_saver.py tests/test_data_saver.py
git commit -m "feat: add DataSaver — per-cycle CSV writer for raw and result data"
```

---

### Task 3: FeatureCollector — add raw_ready signal

**Files:**
- Modify: `usb5133_daq/analysis/feature_collector.py`
- Modify: `tests/test_feature_collector.py`

**Background:** `FeatureCollector._on_timer()` currently extracts samples and emits `features_ready`. We add a `raw_ready = pyqtSignal(object, object)` signal and emit it (with `datetime.now()` and the samples) before calling `extract_features`. The existing `features_ready` tests must still pass unchanged.

- [ ] **Step 6: Write a failing test for raw_ready**

Add to `tests/test_feature_collector.py` after the existing tests:

```python
from datetime import datetime

def test_raw_ready_emitted_with_timestamp_and_samples():
    """raw_ready is emitted once per timer tick, before features_ready."""
    collector = FeatureCollector(
        sample_rate=SAMPLE_RATE,
        cycle_sec=0.001,  # 1ms — fires almost immediately
        collect_window_sec=WINDOW_SEC,
        channel=0,
    )
    raw_events: list[tuple] = []
    collector.raw_ready.connect(lambda ts, s: raw_events.append((ts, s)))

    # Feed enough data to fill the buffer
    n_needed = int(WINDOW_SEC * SAMPLE_RATE)
    waveform = _make_waveform(record_len=n_needed)
    collector.on_data(waveform)

    # Fire the timer manually
    collector._timer.timeout.emit()

    assert len(raw_events) == 1
    ts, samples = raw_events[0]
    assert isinstance(ts, datetime)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == n_needed
```

- [ ] **Step 7: Run the new test to verify it fails**

```
pytest tests/test_feature_collector.py::test_raw_ready_emitted_with_timestamp_and_samples -v
```

Expected: FAIL — `AttributeError: type object 'FeatureCollector' has no attribute 'raw_ready'`

- [ ] **Step 8: Update FeatureCollector**

Replace the entire file content:

```python
# usb5133_daq/analysis/feature_collector.py
from __future__ import annotations

import collections
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from usb5133_daq.analysis.features import extract_features


class FeatureCollector(QObject):
    """raw 파형 버퍼 축적 후 주기적으로 FFT 특징 벡터를 추출해 emit.

    Signals:
        raw_ready: (datetime, np.ndarray) — 타임스탬프 + shape (n_needed,) 샘플
        features_ready: np.ndarray shape (7,)
    """

    raw_ready = pyqtSignal(object, object)    # (datetime, np.ndarray shape (n_needed,))
    features_ready = pyqtSignal(object)        # np.ndarray shape (7,)

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

    def stop(self) -> None:
        """타이머를 명시적으로 중지. _on_stop에서 None 할당 전에 호출."""
        self._timer.stop()

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
        samples = np.array(self._buf)[-self._n_needed:]
        ts = datetime.now()
        self.raw_ready.emit(ts, samples)
        vec = extract_features(samples, self._sample_rate)
        self.features_ready.emit(vec)
```

- [ ] **Step 9: Run all feature_collector tests**

```
pytest tests/test_feature_collector.py -v
```

Expected: all PASSED (including the new `test_raw_ready_emitted_with_timestamp_and_samples`)

- [ ] **Step 10: Run the full test suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 11: Commit**

```bash
git add usb5133_daq/analysis/feature_collector.py tests/test_feature_collector.py
git commit -m "feat: add raw_ready signal to FeatureCollector"
```

---

## Chunk 2: MainWindow wiring + gitignore

### Task 4: Wire DataSaver in MainWindow

**Files:**
- Modify: `usb5133_daq/ui/main_window.py`

**Background:** `MainWindow` must:
1. Add `self._saver: DataSaver | None = None` to `__init__`.
2. In `_on_start()`, after the last existing wiring line (`self._detector.result_ready.connect(self._on_anomaly_result)` at line 231), construct `DataSaver` in a try/except and connect both signals.
3. In `_on_stop()`, insert a saver-disconnect block **before** `self._collector = None`.
4. Import `DataSaver` at the top.

No new tests are needed for MainWindow (it is the composition root; DataSaver and FeatureCollector are already tested independently). Run the full suite to confirm no regressions.

- [ ] **Step 12: Add DataSaver import**

In `usb5133_daq/ui/main_window.py`, add to the imports block (after the existing `usb5133_daq.storage.csv_writer` import):

```python
from usb5133_daq.storage.data_saver import DataSaver
```

- [ ] **Step 13: Add _saver attribute to __init__**

In `MainWindow.__init__`, after the existing `self._detector: AnomalyDetector | None = None` line, add:

```python
self._saver: DataSaver | None = None
```

- [ ] **Step 14: Wire DataSaver in _on_start**

In `_on_start()`, after the existing line:
```python
self._detector.result_ready.connect(self._on_anomaly_result)
```

Insert immediately after:

```python
        try:
            self._saver = DataSaver(save_dir="results")
        except OSError as exc:
            QMessageBox.critical(self, "저장 오류", f"results/ 폴더를 생성할 수 없습니다:\n{exc}")
            self._on_stop()
            return
        self._collector.raw_ready.connect(self._saver.on_raw)
        self._detector.result_ready.connect(self._saver.on_result)
```

- [ ] **Step 15: Update _on_stop to disconnect DataSaver**

Replace the existing `_on_stop` body with:

```python
    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None
        # Disconnect and clear DataSaver before nulling _collector / _detector
        if self._saver is not None and self._collector is not None and self._detector is not None:
            self._collector.raw_ready.disconnect(self._saver.on_raw)
            self._detector.result_ready.disconnect(self._saver.on_result)
        self._saver = None
        if self._collector is not None:
            self._collector.stop()  # 타이머 명시적 중지 (GC 타이밍 의존 방지)
        self._collector = None
        self._detector = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_bar.showMessage("수집 정지")
```

- [ ] **Step 16: Run the full test suite**

```
pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 17: Commit**

```bash
git add usb5133_daq/ui/main_window.py
git commit -m "feat: wire DataSaver into MainWindow start/stop lifecycle"
```

---

### Task 5: Add results/ to .gitignore

**Files:**
- Modify: `.gitignore`

The existing `.gitignore` already excludes `*.csv`. Add an explicit entry for the `results/` directory.

- [ ] **Step 18: Add results/ to .gitignore**

Append to `.gitignore`:

```
# Runtime output
results/
```

- [ ] **Step 19: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore results/ directory"
```
