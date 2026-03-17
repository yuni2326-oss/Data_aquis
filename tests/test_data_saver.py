# tests/test_data_saver.py
from __future__ import annotations

import csv
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.storage.data_saver import DataSaver

SAMPLE_RATE = 10_000.0   # 10 kHz — small enough for fast tests
N_SAMPLES = 1_000        # 0.1 s of data


def _ts() -> datetime:
    return datetime(2026, 3, 16, 14, 30, 0)


def _ts2() -> datetime:
    return datetime(2026, 3, 16, 14, 31, 0)


def _samples() -> np.ndarray:
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    return np.sin(2 * np.pi * 1000.0 * t)


def _saver(tmp_path) -> DataSaver:
    return DataSaver(sample_rate=SAMPLE_RATE, save_dir=tmp_path)


class TestDataSaver:
    def test_creates_results_dir(self, tmp_path):
        save_dir = tmp_path / "results"
        assert not save_dir.exists()
        DataSaver(sample_rate=SAMPLE_RATE, save_dir=save_dir)
        assert save_dir.is_dir()

    def test_raw_csv_written(self, tmp_path):
        saver = _saver(tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        raw_file = tmp_path / f"{stem}_raw.csv"
        assert raw_file.exists()

        with raw_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sample_index", "voltage"]
        assert len(rows) == N_SAMPLES + 1  # header + data rows
        assert int(rows[1][0]) == 0        # first sample_index
        assert float(rows[1][1]) == pytest.approx(0.0, abs=1e-9)  # sin(0) = 0

    def test_fft_csv_written(self, tmp_path):
        saver = _saver(tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        fft_file = tmp_path / f"{stem}_fft.csv"
        assert fft_file.exists()

        with fft_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["frequency_hz", "magnitude"]
        assert len(rows) == N_SAMPLES // 2 + 1  # header + N/2 freq bins
        # first freq bin should be 0 Hz
        assert float(rows[1][0]) == pytest.approx(0.0)
        # all magnitudes should be non-negative
        for row in rows[1:]:
            assert float(row[1]) >= 0.0

    def test_filename_format(self, tmp_path):
        saver = _saver(tmp_path)
        ts = datetime(2026, 3, 16, 9, 5, 3)
        saver.on_raw(ts, _samples())

        assert (tmp_path / "2026-03-16 09-05-03_raw.csv").exists()
        assert (tmp_path / "2026-03-16 09-05-03_fft.csv").exists()

    def test_multiple_cycles(self, tmp_path):
        saver = _saver(tmp_path)
        ts1, ts2 = _ts(), _ts2()

        saver.on_raw(ts1, _samples())
        saver.on_raw(ts2, _samples())

        stem1 = ts1.strftime("%Y-%m-%d %H-%M-%S")
        stem2 = ts2.strftime("%Y-%m-%d %H-%M-%S")
        assert (tmp_path / f"{stem1}_raw.csv").exists()
        assert (tmp_path / f"{stem1}_fft.csv").exists()
        assert (tmp_path / f"{stem2}_raw.csv").exists()
        assert (tmp_path / f"{stem2}_fft.csv").exists()

    def test_oserror_on_write(self, tmp_path):
        saver = _saver(tmp_path)
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            with patch("usb5133_daq.storage.data_saver._log") as mock_log:
                saver.on_raw(_ts(), _samples())
                mock_log.warning.assert_called_once()

    def test_fft_peak_at_signal_frequency(self, tmp_path):
        """1 kHz 사인파의 FFT 피크가 1 kHz 근처에 있는지 검증."""
        saver = _saver(tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        with (tmp_path / f"{stem}_fft.csv").open() as f:
            rows = list(csv.reader(f))[1:]  # skip header

        freqs = [float(r[0]) for r in rows]
        mags = [float(r[1]) for r in rows]
        peak_freq = freqs[mags.index(max(mags))]
        assert abs(peak_freq - 1000.0) < SAMPLE_RATE / N_SAMPLES * 2  # within 2 bins
