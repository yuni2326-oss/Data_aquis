# tests/test_anomaly_detector.py
import math
import numpy as np
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
