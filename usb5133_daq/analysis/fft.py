# usb5133_daq/analysis/fft.py
from __future__ import annotations

import numpy as np


def compute_fft(waveform: np.ndarray, sample_rate: float):
    """단측 FFT 스펙트럼 계산.

    Args:
        waveform: 1D numpy 배열 (단일 채널 파형)
        sample_rate: 샘플레이트 (Hz)

    Returns:
        (freqs, magnitudes): 주파수 배열(Hz), 크기 배열 (둘 다 길이 N//2)
    """
    N = len(waveform)
    window = np.hanning(N)
    windowed = waveform * window

    fft_vals = np.fft.fft(windowed)
    half = N // 2
    magnitudes = np.abs(fft_vals[:half]) * 2.0 / N
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)[:half]

    return freqs, magnitudes
