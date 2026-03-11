# tests/test_csv_writer.py
import csv
from pathlib import Path

import numpy as np
import pytest
from usb5133_daq.storage.csv_writer import save_waveform


class TestSaveWaveform:
    def test_creates_file(self, tmp_path):
        waveform = np.zeros((1, 100))
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=1_000_000.0, filename=out)
        assert Path(out).exists()

    def test_single_channel_columns(self, tmp_path):
        waveform = np.zeros((1, 10))
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=10.0, filename=out)
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["time", "ch0"]

    def test_two_channel_columns(self, tmp_path):
        waveform = np.zeros((2, 10))
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=10.0, filename=out)
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["time", "ch0", "ch1"]

    def test_time_axis_correct(self, tmp_path):
        N = 5
        sr = 10.0
        waveform = np.zeros((1, N))
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=sr, filename=out)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        times = [float(r[0]) for r in rows]
        expected = [i / sr for i in range(N)]
        assert len(times) == N
        for t, e in zip(times, expected):
            assert abs(t - e) < 1e-9

    def test_data_values_preserved(self, tmp_path):
        waveform = np.array([[1.0, 2.0, 3.0]])
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=1.0, filename=out)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        values = [float(r[1]) for r in rows]
        assert values == pytest.approx([1.0, 2.0, 3.0])

    def test_row_count_matches_record_length(self, tmp_path):
        N = 200
        waveform = np.zeros((2, N))
        out = str(tmp_path / "out.csv")
        save_waveform(waveform, sample_rate=1_000.0, filename=out)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        assert len(rows) == N
