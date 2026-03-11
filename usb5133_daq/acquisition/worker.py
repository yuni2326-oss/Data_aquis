# usb5133_daq/acquisition/worker.py
from __future__ import annotations

import threading

from PyQt5.QtCore import QThread, pyqtSignal


class AcquisitionWorker(QThread):
    """백그라운드 수집 루프.

    Signals:
        data_ready: shape (num_channels, record_length)의 np.ndarray 전달
        error_occurred: 오류 메시지 문자열 전달
    """

    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, scope, parent=None):
        super().__init__(parent)
        self._scope = scope
        self._stop_event = threading.Event()

    def run(self):
        self._stop_event.clear()
        try:
            self._scope.start_acquisition()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return  # start 실패 시 UI에 알리고 조용히 종료
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._scope.fetch()
                    self.data_ready.emit(data)
                    # UI 과부하 방지: MockScope는 즉각 반환하므로 제한 필요
                    self.msleep(10)
                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                    break
        finally:
            self._scope.stop_acquisition()

    def stop(self):
        """스레드 안전하게 수집 루프 종료."""
        self._stop_event.set()
