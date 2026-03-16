# tests/test_feature_collector.py
import numpy as np
from PyQt5.QtCore import QCoreApplication
import sys

# QApplication needed for QObject/QTimer
app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from usb5133_daq.analysis.feature_collector import FeatureCollector

SAMPLE_RATE = 100_000.0  # 100 kHz for fast tests
RECORD_LEN = 1000        # 10ms per fetch
WINDOW_SEC = 0.05        # 50ms window — needs 5000 samples


def _make_waveform(n_channels: int = 1, record_len: int = RECORD_LEN) -> np.ndarray:
    """shape (n_channels, record_len) sine waveform"""
    t = np.arange(record_len) / SAMPLE_RATE
    row = np.sin(2 * np.pi * 1000.0 * t)
    return np.tile(row, (n_channels, 1))


class TestFeatureCollector:
    def test_channel_slice(self):
        """on_data should use only waveform[channel]"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,  # timer won't fire during test
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        wave = _make_waveform(n_channels=2)
        wave[1] *= 99  # channel 1 is very different
        collector.on_data(wave)
        # buffer should contain only channel-0 samples (values near ±1)
        buf = list(collector._buf)
        assert max(abs(s) for s in buf) < 2.0

    def test_buffer_skips_when_insufficient_samples(self):
        """_on_timer should not emit features_ready if buffer < n_needed"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        received = []
        collector.features_ready.connect(lambda v: received.append(v))

        # Feed less than n_needed samples (WINDOW_SEC * SAMPLE_RATE = 5000)
        wave = _make_waveform()  # only 1000 samples
        collector.on_data(wave)
        collector._on_timer()  # manually trigger

        assert len(received) == 0

    def test_emits_when_sufficient_samples(self):
        """_on_timer should emit features_ready once buffer >= n_needed"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=1000.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        received = []
        collector.features_ready.connect(lambda v: received.append(v))

        # Fill buffer: n_needed = 5000 samples, 5 calls × 1000 = 5000
        wave = _make_waveform()
        for _ in range(5):
            collector.on_data(wave)
        collector._on_timer()

        assert len(received) == 1
        assert received[0].shape == (7,)

    def test_timer_starts_on_first_on_data(self):
        """QTimer.start() called exactly once on first on_data"""
        collector = FeatureCollector(
            sample_rate=SAMPLE_RATE,
            cycle_sec=30.0,
            collect_window_sec=WINDOW_SEC,
            channel=0,
        )
        assert not collector._timer.isActive()
        collector.on_data(_make_waveform())
        assert collector._timer.isActive()
        # second call should not restart timer
        interval_before = collector._timer.interval()
        collector.on_data(_make_waveform())
        assert collector._timer.interval() == interval_before
        assert collector._timer.isActive()  # still active, not stopped and restarted


from datetime import datetime

def test_raw_ready_emitted_with_timestamp_and_samples():
    """raw_ready is emitted once per timer tick, before features_ready."""
    collector = FeatureCollector(
        sample_rate=SAMPLE_RATE,
        cycle_sec=0.001,  # 1ms — fires almost immediately
        collect_window_sec=WINDOW_SEC,
        channel=0,
    )
    raw_events: list[tuple] = []
    collector.raw_ready.connect(lambda ts, s: raw_events.append((ts, s)))

    # Feed enough data to fill the buffer
    n_needed = int(WINDOW_SEC * SAMPLE_RATE)
    waveform = _make_waveform(record_len=n_needed)
    collector.on_data(waveform)

    # Fire the timer manually
    collector._timer.timeout.emit()

    assert len(raw_events) == 1
    ts, samples = raw_events[0]
    assert isinstance(ts, datetime)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == n_needed
