# Data Saver Design Spec

## Overview

Add automatic CSV saving of raw waveform data and anomaly detection results to a `results/` subfolder. Saving starts automatically when acquisition starts and stops when acquisition stops. Each 30-second analysis cycle produces a pair of CSV files named with the cycle timestamp.

---

## Requirements

- **Trigger:** Automatic — saving starts when acquisition starts, stops when it stops. No separate UI button.
- **Format:** CSV only.
- **Granularity:** Per 30-second cycle — each cycle produces 2 files (raw + result).
- **Output directory:** `results/` relative to the working directory (created automatically if absent).
- **Filename format:** `YYYY-MM-DD HH-MM-SS_raw.csv` and `YYYY-MM-DD HH-MM-SS_result.csv` where the timestamp is captured at the moment samples are extracted in `_on_timer()`.

---

## Architecture

```
FeatureCollector._on_timer()
  ├─ emit raw_ready(timestamp: datetime, samples: np.ndarray)  ──────────┐
  └─ emit features_ready(vec)                                             │
       └─ AnomalyDetector.on_features()                                  │
            └─ emit result_ready(AnomalyResult)  ──────────────────────  │
                                                                         ▼
                                                                    DataSaver
                                                    results/YYYY-MM-DD HH-MM-SS_raw.csv
                                                    results/YYYY-MM-DD HH-MM-SS_result.csv
```

Both signals fire synchronously within the same `_on_timer` call stack because all objects (`FeatureCollector`, `AnomalyDetector`, `DataSaver`) live on the main thread with direct Qt connections. This guarantees `raw_ready` always arrives before `result_ready` within each cycle, making a simple "hold the last raw" pattern safe. **Constraint:** All three objects must remain on the same thread; moving any of them to a worker thread would require revisiting this design.

---

## Components

### 1. `FeatureCollector` (modified)

**File:** `usb5133_daq/analysis/feature_collector.py`

Add `from __future__ import annotations` (consistent with codebase) and `from datetime import datetime`.

Add one new signal:

```python
raw_ready = pyqtSignal(object, object)  # (datetime, np.ndarray shape (n_needed,))
```

In `_on_timer()`, capture the timestamp and emit `raw_ready` **before** calling `extract_features`:

```python
def _on_timer(self) -> None:
    if len(self._buf) < self._n_needed:
        return
    samples = np.array(self._buf)[-self._n_needed:]
    ts = datetime.now()
    self.raw_ready.emit(ts, samples)
    vec = extract_features(samples, self._sample_rate)
    self.features_ready.emit(vec)
```

---

### 2. `DataSaver` (new)

**File:** `usb5133_daq/storage/data_saver.py`

Include `from __future__ import annotations` at the top (required for `str | Path` union type on Python < 3.10).

```python
class DataSaver(QObject):
    def __init__(self, save_dir: str | Path = "results", parent=None): ...

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, samples: np.ndarray) -> None: ...

    @pyqtSlot(object)
    def on_result(self, result: AnomalyResult) -> None: ...
```

**Internal state:**
- `_save_dir: Path` — ensured to exist at construction time via `Path.mkdir(parents=True, exist_ok=True)`
- `_pending_raw: tuple[datetime, np.ndarray] | None` — holds the most recent raw data until `on_result` arrives

**`__init__` attributes (initialised before signal connections):**
```python
self._save_dir = Path(save_dir)
self._save_dir.mkdir(parents=True, exist_ok=True)  # OSError propagates to caller
self._pending_raw: tuple[datetime, np.ndarray] | None = None
```

**`on_raw` behaviour:**
1. Store `(timestamp, samples)` in `_pending_raw`. Under the thread constraint stated in the Architecture section, a pre-existing unsaved entry cannot exist here (it is architecturally impossible without a threading violation). The overwrite is retained solely as a defensive guard and is treated as dead code in normal operation.

**`on_result` behaviour:**
1. If `_pending_raw` is `None`, log a warning and return (result arrived before any raw data — should not happen in normal operation).
2. Retrieve and clear `_pending_raw`.
3. Derive the stem: `timestamp.strftime("%Y-%m-%d %H-%M-%S")`.
4. Write `<stem>_raw.csv` and `<stem>_result.csv` to `_save_dir`.
5. If a file with the same stem already exists (e.g., rapid stop/restart within the same second), overwrite silently.
6. Any `OSError` during writing is caught and logged via `logging.warning`; acquisition continues uninterrupted.

**Raw CSV columns:** `sample_index,voltage`
- `sample_index`: integer 0-based index
- `voltage`: the floating-point value as produced by the acquisition layer (same unit as the `waveform` array in `AcquisitionWorker.data_ready` — volts for real hardware, arbitrary float for mock)
- One row per sample (5 s × sample_rate rows)

**Result CSV columns:** `timestamp,label,score,zscore_max,baseline_progress,baseline_total`
- `timestamp`: the `datetime` value from `_pending_raw`, formatted as ISO 8601 `YYYY-MM-DD HH:MM:SS` (colons as time separator, compatible with `pandas.read_csv(parse_dates=["timestamp"])`). Note: the filename stem uses hyphens (`HH-MM-SS`) because colons are not valid in Windows filenames — these are two different format strings.
- `label`, `score`, `zscore_max`, `baseline_progress`, `baseline_total`: taken directly from the `AnomalyResult` fields
- During LEARNING phase, `score` and `zscore_max` are `math.nan`. These are written as the Python string `"nan"` by `csv.writer` — consumers must handle this (e.g., `pd.read_csv(..., na_values=["nan"])`).
- One row (header + data row)

**CSV writing library:** Use Python's standard library `csv.writer` for both files. Do not use `numpy.savetxt` (awkward with mixed types in result row) or `pandas` (unnecessary dependency).

---

### 3. `MainWindow` (modified)

**File:** `usb5133_daq/ui/main_window.py`

Add to `__init__`:
```python
self._saver: DataSaver | None = None
```

In `_on_start()`, insert the saver block **after** the existing line `self._detector.result_ready.connect(self._on_anomaly_result)` (which is the last line of the existing wiring block), so that `self._collector` and `self._detector` are guaranteed to exist:
```python
# existing line (already present):
self._detector.result_ready.connect(self._on_anomaly_result)

# new lines — insert immediately after:
try:
    self._saver = DataSaver(save_dir="results")
except OSError as exc:
    QMessageBox.critical(self, "저장 오류", f"results/ 폴더를 생성할 수 없습니다:\n{exc}")
    self._on_stop()
    return
self._collector.raw_ready.connect(self._saver.on_raw)
self._detector.result_ready.connect(self._saver.on_result)
```

In `_on_stop()`, insert the saver block **before** the existing `self._collector = None` line (so that `_collector` and `_detector` are still valid when disconnecting). The complete updated `_on_stop` body:

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
        self._collector.stop()
    self._collector = None
    self._detector = None
    self._btn_start.setEnabled(True)
    self._btn_stop.setEnabled(False)
    self._status_bar.showMessage("수집 정지")
```

---

### 4. `usb5133_daq/storage/__init__.py` (new, empty)

Required to make `storage` a proper package.

---

## File Layout

```
usb5133_daq/
  storage/
    __init__.py          (new, empty)
    data_saver.py        (new)
  analysis/
    feature_collector.py (modified — add raw_ready signal + datetime emit)
  ui/
    main_window.py       (modified — create/connect/clear DataSaver)
tests/
  test_data_saver.py     (new)
results/                 (created at runtime, gitignored)
```

---

## Testing

**File:** `tests/test_data_saver.py`

Test cases (no Qt event loop required — call slots directly):

| Test | Description |
|------|-------------|
| `test_creates_results_dir` | DataSaver creates `save_dir` when it does not exist |
| `test_raw_csv_written` | After `on_raw` + `on_result`, `_raw.csv` exists with correct columns and row count |
| `test_result_csv_written` | After `on_raw` + `on_result`, `_result.csv` exists with correct columns and one data row |
| `test_filename_format` | Filenames match `YYYY-MM-DD HH-MM-SS_raw.csv` / `_result.csv` |
| `test_result_without_raw` | `on_result` called before `on_raw` — no crash, no files written |
| `test_multiple_cycles` | Two sequential `on_raw` + `on_result` pairs with injected distinct `datetime` values produce two file pairs with distinct filenames |
| `test_oserror_on_write` | Simulate write failure (patch `open` to raise `OSError`) — no crash, warning logged |
| `test_result_csv_nan_values` | LEARNING-phase result (`score=nan`, `zscore_max=nan`) is written as string `"nan"` in the CSV |

Tests use `tmp_path` (pytest fixture) for isolated temp directories.

---

## Error Handling

- `OSError` on file write: caught, `logging.warning` emitted, acquisition continues.
- `on_result` before `on_raw`: `logging.warning` emitted, no file written, no crash.
- `save_dir` creation failure (permissions): `OSError` propagated from `DataSaver.__init__` to `MainWindow._on_start`, which catches it, shows `QMessageBox.critical`, and calls `_on_stop` to abort acquisition cleanly.

---

## Out of Scope

- UI controls for save directory selection
- File compression or rotation
- Replay/load of saved files
- Filename collision disambiguation beyond silent overwrite (e.g., `_001` suffix)
