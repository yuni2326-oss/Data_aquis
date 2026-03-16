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
        samples = np.array(self._buf)[-self._n_needed:]
        vec = extract_features(samples, self._sample_rate)
        self.features_ready.emit(vec)
