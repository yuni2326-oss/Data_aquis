# usb5133_daq/analysis/features.py
from __future__ import annotations

import numpy as np

from usb5133_daq.analysis.fft import compute_fft

_PEAK_NEIGHBOR = 5  # bins to exclude around dominant peak when computing noise floor


def extract_features(waveform: np.ndarray, sample_rate: float) -> np.ndarray:
    """FFT 특징 벡터 추출.

    Args:
        waveform: 1D array shape (N,) — single channel
        sample_rate: sample rate in Hz

    Returns:
        np.ndarray shape (7,), dtype float64:
            [0] dominant_freq (Hz)
            [1] dominant_mag
            [2] second_harmonic_mag
            [3] third_harmonic_mag
            [4] thd  = (2nd + 3rd) / dominant_mag
            [5] noise_floor_rms
            [6] spectral_centroid (Hz)
    """
    freqs, mags = compute_fft(waveform, sample_rate)

    dominant_idx = int(np.argmax(mags))
    dominant_mag = float(mags[dominant_idx])

    # Zero-signal guard — avoids NaN propagation
    if dominant_mag < 1e-9:
        return np.zeros(7, dtype=np.float64)

    dominant_freq = float(freqs[dominant_idx])

    # Harmonic magnitudes — find bin closest to 2x and 3x dominant_freq
    def _mag_at_freq(target_hz: float) -> float:
        if target_hz <= 0 or target_hz > freqs[-1]:
            return 0.0
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(mags[idx])

    second_mag = _mag_at_freq(2.0 * dominant_freq)
    third_mag = _mag_at_freq(3.0 * dominant_freq)
    thd = (second_mag + third_mag) / dominant_mag

    # Noise floor RMS — exclude _PEAK_NEIGHBOR bins around dominant
    mask = np.ones(len(mags), dtype=bool)
    lo = max(0, dominant_idx - _PEAK_NEIGHBOR)
    hi = min(len(mags), dominant_idx + _PEAK_NEIGHBOR + 1)
    mask[lo:hi] = False
    noise_rms = float(np.sqrt(np.mean(mags[mask] ** 2))) if mask.any() else 0.0

    # Spectral centroid
    total = float(np.sum(mags))
    centroid = float(np.sum(freqs * mags) / total) if total > 0 else 0.0

    return np.array(
        [dominant_freq, dominant_mag, second_mag, third_mag, thd, noise_rms, centroid],
        dtype=np.float64,
    )
