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
