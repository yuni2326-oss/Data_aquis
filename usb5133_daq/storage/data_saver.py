# usb5133_daq/storage/data_saver.py
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

from usb5133_daq.analysis.fft import compute_fft

_log = logging.getLogger(__name__)


class DataSaver(QObject):
    """수집 사이클마다 raw 파형과 FFT 스펙트럼을 CSV 쌍으로 저장.

    Slots:
        on_raw(datetime, np.ndarray): FeatureCollector.raw_ready 수신
    """

    def __init__(self, sample_rate: float, save_dir: str | Path = "results", parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)  # OSError propagates to caller

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, samples: np.ndarray) -> None:
        """FeatureCollector.raw_ready 슬롯. raw + FFT CSV를 즉시 저장."""
        stem = timestamp.strftime("%Y-%m-%d %H-%M-%S")
        try:
            self._write_raw(stem, samples)
            self._write_fft(stem, samples)
        except OSError as exc:
            _log.warning("DataSaver: failed to write CSV files: %s", exc)

    def _write_raw(self, stem: str, samples: np.ndarray) -> None:
        """samples는 1-D array (FeatureCollector가 단일 채널 슬라이스 후 emit)."""
        path = self._save_dir / f"{stem}_raw.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "voltage"])
            writer.writerows(enumerate(samples))

    def _write_fft(self, stem: str, samples: np.ndarray) -> None:
        """FFT 스펙트럼을 (frequency_hz, magnitude) 쌍으로 저장."""
        freqs, mags = compute_fft(samples, self._sample_rate)
        path = self._save_dir / f"{stem}_fft.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frequency_hz", "magnitude"])
            writer.writerows(zip(freqs, mags))
