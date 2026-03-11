# tests/test_scope.py
import numpy as np
import pytest
from usb5133_daq.device.scope import MockScope

VALID_VOLTAGE_RANGES = [0.5, 1.0, 2.0, 5.0]

class TestMockScope:
    def test_connect_disconnect(self):
        scope = MockScope()
        scope.connect()
        scope.disconnect()

    def test_context_manager(self):
        with MockScope() as scope:
            pass

    def test_configure_sets_params(self):
        scope = MockScope()
        scope.connect()
        scope.configure(
            sample_rate=1_000_000.0,
            record_length=1000,
            channels=[0],
            voltage_range=1.0,
        )
        assert scope.sample_rate == 1_000_000.0
        assert scope.record_length == 1000
        assert scope.channels == [0]
        assert scope.voltage_range == 1.0

    def test_configure_raises_if_not_connected(self):
        scope = MockScope()
        with pytest.raises(RuntimeError, match="connect"):
            scope.configure(1_000_000.0, 1000, [0], 1.0)

    def test_configure_raises_on_invalid_sample_rate(self):
        scope = MockScope()
        scope.connect()
        with pytest.raises(ValueError, match="sample_rate"):
            scope.configure(200_000_000.0, 1000, [0], 1.0)

    def test_configure_raises_on_invalid_voltage_range(self):
        scope = MockScope()
        scope.connect()
        with pytest.raises(ValueError, match="voltage_range"):
            scope.configure(1_000_000.0, 1000, [0], 3.0)

    def test_fetch_returns_correct_shape_one_channel(self):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        data = scope.fetch()
        assert data.shape == (1, 1000)

    def test_fetch_returns_correct_shape_two_channels(self):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0, 1], 1.0)
        data = scope.fetch()
        assert data.shape == (2, 1000)

    def test_fetch_returns_sine_wave(self):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        data = scope.fetch()
        assert data.min() >= -1.1
        assert data.max() <= 1.1
