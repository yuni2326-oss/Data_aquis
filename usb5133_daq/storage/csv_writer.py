# usb5133_daq/storage/csv_writer.py
from __future__ import annotations

import numpy as np


def save_waveform(waveform: np.ndarray, sample_rate: float, filename: str) -> None:
    """파형 데이터를 CSV로 저장.

    Args:
        waveform: shape (num_channels, record_length)
        sample_rate: 샘플레이트 (Hz) — 시간축 생성에 사용
        filename: 저장할 파일 경로
    """
    num_channels, record_length = waveform.shape
    time_axis = np.arange(record_length) / sample_rate

    data = np.column_stack([time_axis] + [waveform[i] for i in range(num_channels)])
    header = "time," + ",".join(f"ch{i}" for i in range(num_channels))

    # comments="" → 헤더 줄에 '#' 접두사 없이 저장
    np.savetxt(filename, data, delimiter=",", header=header, comments="")
