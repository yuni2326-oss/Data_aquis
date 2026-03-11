# tests/test_fft.py
import numpy as np
import pytest
from usb5133_daq.analysis.fft import compute_fft


class TestComputeFFT:
    def test_returns_two_arrays(self):
        waveform = np.zeros(1000)
        freqs, mags = compute_fft(waveform, sample_rate=1_000_000.0)
        assert freqs is not None
        assert mags is not None

    def test_output_length_is_half_input(self):
        N = 1000
        waveform = np.zeros(N)
        freqs, mags = compute_fft(waveform, sample_rate=1_000_000.0)
        assert len(freqs) == N // 2
        assert len(mags) == N // 2

    def test_detects_known_frequency(self):
        sample_rate = 100_000.0
        N = 4096
        freq_hz = 1000.0
        t = np.arange(N) / sample_rate
        waveform = np.sin(2 * np.pi * freq_hz * t)
        freqs, mags = compute_fft(waveform, sample_rate=sample_rate)
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - freq_hz) < 50.0

    def test_dc_signal_peak_at_zero(self):
        waveform = np.ones(1000)
        freqs, mags = compute_fft(waveform, sample_rate=1_000_000.0)
        assert freqs[np.argmax(mags)] == pytest.approx(0.0, abs=1.0)

    def test_freqs_are_non_negative(self):
        waveform = np.random.randn(512)
        freqs, mags = compute_fft(waveform, sample_rate=50_000.0)
        assert np.all(freqs >= 0)

    def test_mags_are_non_negative(self):
        waveform = np.random.randn(512)
        freqs, mags = compute_fft(waveform, sample_rate=50_000.0)
        assert np.all(mags >= 0)
