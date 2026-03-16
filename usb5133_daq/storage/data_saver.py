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
        if self._pending_raw is not None:
            _log.warning("DataSaver.on_raw: overwriting unsaved raw data (threading violation?)")
        self._pending_raw = (timestamp, samples)

    @pyqtSlot(object)
    def on_result(self, result: AnomalyResult) -> None:
        """AnomalyDetector.result_ready 슬롯."""
        if self._pending_raw is None:
            _log.warning("DataSaver.on_result: no pending raw data — skipping write")
            return

        timestamp, samples = self._pending_raw
        # Clear before writing so a failed write does not block the next cycle.
        # Intentional trade-off: on OSError the cycle's data is dropped (warning logged).
        self._pending_raw = None

        stem = timestamp.strftime("%Y-%m-%d %H-%M-%S")
        try:
            self._write_raw(stem, samples)
            self._write_result(stem, timestamp, result)
        except OSError as exc:
            _log.warning("DataSaver: failed to write CSV files: %s", exc)

    def _write_raw(self, stem: str, samples: np.ndarray) -> None:
        """samples는 1-D array (FeatureCollector가 단일 채널 슬라이스 후 emit)."""
        path = self._save_dir / f"{stem}_raw.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "voltage"])
            writer.writerows(enumerate(samples))

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
