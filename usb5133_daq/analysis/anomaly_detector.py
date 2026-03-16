# usb5133_daq/analysis/anomaly_detector.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
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
        # Baseline IF score statistics for normalised deviation
        self._score_mean: float = 0.0
        self._score_std: float = 1.0

    @pyqtSlot(object)
    def on_features(self, vec: np.ndarray) -> None:
        """FeatureCollector.features_ready slot."""
        if len(self._baseline) < self._baseline_count:
            self._baseline.append(vec)
            if len(self._baseline) == self._baseline_count:
                X = np.vstack(self._baseline)  # shape (baseline_count, n_features)
                self._if_model = IsolationForest(contamination=_CONTAMINATION).fit(X)
                self._mean = X.mean(axis=0)
                self._std = X.std(axis=0) + 1e-9  # prevent divide-by-zero
                # Compute normalised score statistics from baseline (batch call)
                baseline_scores = self._if_model.score_samples(X)
                self._score_mean = float(baseline_scores.mean())
                self._score_std = float(baseline_scores.std()) + 1e-9
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
        # Normalised deviation of the IF score relative to baseline score distribution
        norm_dev = (if_score - self._score_mean) / self._score_std

        if zscore_max < 3.0 and norm_dev > -2.0:
            label = "NORMAL"
        elif zscore_max < 5.0 and norm_dev > -3.0:
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
