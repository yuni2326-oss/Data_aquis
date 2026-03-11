# tests/test_worker.py
import time
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from usb5133_daq.acquisition.worker import AcquisitionWorker
from usb5133_daq.device.scope import MockScope


def _run_worker(worker, qapp, sleep_sec=0.1):
    worker.start()
    time.sleep(sleep_sec)
    worker.stop()
    worker.wait(2000)
    qapp.processEvents()


class TestAcquisitionWorker:
    def test_emits_data_ready(self, qapp):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        worker = AcquisitionWorker(scope)
        received = []
        worker.data_ready.connect(lambda d: received.append(d), Qt.DirectConnection)
        _run_worker(worker, qapp)
        assert len(received) > 0

    def test_data_shape_one_channel(self, qapp):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 500, [0], 1.0)
        worker = AcquisitionWorker(scope)
        received = []
        worker.data_ready.connect(lambda d: received.append(d), Qt.DirectConnection)
        _run_worker(worker, qapp)
        assert received[0].shape == (1, 500)

    def test_data_shape_two_channels(self, qapp):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 500, [0, 1], 1.0)
        worker = AcquisitionWorker(scope)
        received = []
        worker.data_ready.connect(lambda d: received.append(d), Qt.DirectConnection)
        _run_worker(worker, qapp)
        assert received[0].shape == (2, 500)

    def test_stop_terminates_thread(self, qapp):
        scope = MockScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        worker = AcquisitionWorker(scope)
        worker.start()
        time.sleep(0.05)
        worker.stop()
        finished = worker.wait(2000)
        assert finished

    def test_error_signal_on_fetch_failure(self, qapp):
        class FailingScope(MockScope):
            def fetch(self):
                raise RuntimeError("simulated fetch error")
        scope = FailingScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        worker = AcquisitionWorker(scope)
        errors = []
        worker.error_occurred.connect(lambda e: errors.append(e), Qt.DirectConnection)
        _run_worker(worker, qapp)
        assert len(errors) > 0
        assert "simulated fetch error" in errors[0]

    def test_start_acquisition_failure_emits_error(self, qapp):
        class BadStartScope(MockScope):
            def start_acquisition(self):
                raise RuntimeError("start failed")
        scope = BadStartScope()
        scope.connect()
        scope.configure(1_000_000.0, 1000, [0], 1.0)
        worker = AcquisitionWorker(scope)
        errors = []
        worker.error_occurred.connect(lambda e: errors.append(e), Qt.DirectConnection)
        worker.start()
        time.sleep(0.05)
        worker.wait(2000)
        qapp.processEvents()
        assert len(errors) == 1
        assert "start failed" in errors[0]
