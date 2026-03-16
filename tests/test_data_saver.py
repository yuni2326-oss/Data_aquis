# tests/test_data_saver.py
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.analysis.anomaly_detector import AnomalyResult
from usb5133_daq.storage.data_saver import DataSaver

N_SAMPLES = 500  # small for fast tests


def _ts() -> datetime:
    return datetime(2026, 3, 16, 14, 30, 0)


def _ts2() -> datetime:
    return datetime(2026, 3, 16, 14, 31, 0)


def _samples() -> np.ndarray:
    return np.linspace(0.0, 1.0, N_SAMPLES)


def _normal_result() -> AnomalyResult:
    return AnomalyResult(
        label="NORMAL",
        score=-0.05,
        zscore_max=1.2,
        baseline_progress=20,
        baseline_total=20,
    )


def _learning_result() -> AnomalyResult:
    return AnomalyResult(
        label="LEARNING",
        score=math.nan,
        zscore_max=math.nan,
        baseline_progress=5,
        baseline_total=20,
    )


class TestDataSaver:
    def test_creates_results_dir(self, tmp_path):
        save_dir = tmp_path / "results"
        assert not save_dir.exists()
        DataSaver(save_dir=save_dir)
        assert save_dir.is_dir()

    def test_raw_csv_written(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        saver.on_result(_normal_result())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        raw_file = tmp_path / f"{stem}_raw.csv"
        assert raw_file.exists()

        with raw_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sample_index", "voltage"]
        assert len(rows) == N_SAMPLES + 1  # header + data rows
        assert float(rows[1][0]) == 0      # first sample_index
        assert float(rows[1][1]) == pytest.approx(0.0)  # first voltage (linspace starts at 0)

    def test_result_csv_written(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        result = _normal_result()
        saver.on_result(result)

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        result_file = tmp_path / f"{stem}_result.csv"
        assert result_file.exists()

        with result_file.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["timestamp", "label", "score", "zscore_max",
                           "baseline_progress", "baseline_total"]
        assert len(rows) == 2  # header + 1 data row
        assert rows[1][1] == "NORMAL"
        assert rows[1][0] == "2026-03-16 14:30:00"  # ISO 8601 timestamp with colons
        assert float(rows[1][2]) == pytest.approx(-0.05)  # score
        assert int(rows[1][4]) == 20  # baseline_progress
        assert int(rows[1][5]) == 20  # baseline_total

    def test_filename_format(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = datetime(2026, 3, 16, 9, 5, 3)
        saver.on_raw(ts, _samples())
        saver.on_result(_normal_result())

        assert (tmp_path / "2026-03-16 09-05-03_raw.csv").exists()
        assert (tmp_path / "2026-03-16 09-05-03_result.csv").exists()

    def test_result_without_raw(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        with patch("usb5133_daq.storage.data_saver._log") as mock_log:
            saver.on_result(_normal_result())
            mock_log.warning.assert_called_once()
        assert list(tmp_path.iterdir()) == []

    def test_multiple_cycles(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts1, ts2 = _ts(), _ts2()

        saver.on_raw(ts1, _samples())
        saver.on_result(_normal_result())

        saver.on_raw(ts2, _samples())
        saver.on_result(_normal_result())

        stem1 = ts1.strftime("%Y-%m-%d %H-%M-%S")
        stem2 = ts2.strftime("%Y-%m-%d %H-%M-%S")
        assert (tmp_path / f"{stem1}_raw.csv").exists()
        assert (tmp_path / f"{stem1}_result.csv").exists()
        assert (tmp_path / f"{stem2}_raw.csv").exists()
        assert (tmp_path / f"{stem2}_result.csv").exists()

    def test_oserror_on_write(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        saver.on_raw(_ts(), _samples())
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            with patch("usb5133_daq.storage.data_saver._log") as mock_log:
                saver.on_result(_normal_result())
                mock_log.warning.assert_called_once()

    def test_result_csv_nan_values(self, tmp_path):
        saver = DataSaver(save_dir=tmp_path)
        ts = _ts()
        saver.on_raw(ts, _samples())
        saver.on_result(_learning_result())

        stem = ts.strftime("%Y-%m-%d %H-%M-%S")
        with (tmp_path / f"{stem}_result.csv").open() as f:
            rows = list(csv.reader(f))
        assert rows[1][2] == "nan"   # score column
        assert rows[1][3] == "nan"   # zscore_max column
