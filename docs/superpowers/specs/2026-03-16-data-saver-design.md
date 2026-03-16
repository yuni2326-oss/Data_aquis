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

`raw_ready` and `result_ready` fire within the same Qt event loop iteration (both originate from `_on_timer`), so `DataSaver` can safely use a simple "hold the last raw" pattern without risking mismatched timestamps.

---

## Components

### 1. `FeatureCollector` (modified)

**File:** `usb5133_daq/analysis/feature_collector.py`

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

Import addition: `from datetime import datetime`

---

### 2. `DataSaver` (new)

**File:** `usb5133_daq/storage/data_saver.py`

```python
class DataSaver(QObject):
    def __init__(self, save_dir: str | Path = "results", parent=None): ...

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, samples: np.ndarray) -> None: ...

    @pyqtSlot(object)
    def on_result(self, result: AnomalyResult) -> None: ...
```

**Internal state:**
- `_save_dir: Path` — ensured to exist at construction time
- `_pending_raw: tuple[datetime, np.ndarray] | None` — holds the most recent raw data until `on_result` arrives

**`on_raw` behaviour:**
1. Store `(timestamp, samples)` in `_pending_raw`, overwriting any previously unsaved entry (this should not happen in normal operation but is safe by design).

**`on_result` behaviour:**
1. If `_pending_raw` is `None`, log a warning and return (result arrived before any raw data — should not happen in normal operation).
2. Retrieve and clear `_pending_raw`.
3. Derive the stem: `timestamp.strftime("%Y-%m-%d %H-%M-%S")`.
4. Write `<stem>_raw.csv` and `<stem>_result.csv` to `_save_dir`.
5. Any `OSError` during writing is caught and logged; it does not crash the application.

**Raw CSV columns:** `sample_index,voltage`
One row per sample (5 s × sample_rate rows).

**Result CSV columns:** `timestamp,label,score,zscore_max,baseline_progress,baseline_total`
One row (header + data row).

---

### 3. `MainWindow` (modified)

**File:** `usb5133_daq/ui/main_window.py`

In `_on_start()`:
```python
self._saver = DataSaver(save_dir="results")
self._collector.raw_ready.connect(self._saver.on_raw)
self._detector.result_ready.connect(self._saver.on_result)
```

In `_on_stop()`:
```python
self._saver = None  # DataSaver has no timer to stop
```

Add `_saver: DataSaver | None = None` to `__init__`.

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
| `test_multiple_cycles` | Two sequential `on_raw` + `on_result` pairs produce two file pairs with distinct timestamps |

Tests use `tmp_path` (pytest fixture) for isolated temp directories.

---

## Error Handling

- `OSError` on file write: caught, warning logged via Python `logging`, acquisition continues.
- `on_result` before `on_raw`: warning logged, no file written, no crash.
- `save_dir` creation failure (permissions): `OSError` propagated at `DataSaver.__init__` time — will surface during `_on_start()` and should be shown to the user in a future enhancement (out of scope here).

---

## Out of Scope

- UI controls for save directory selection
- File compression or rotation
- Replay/load of saved files
