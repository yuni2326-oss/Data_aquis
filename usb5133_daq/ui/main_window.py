# usb5133_daq/ui/main_window.py
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from usb5133_daq.acquisition.worker import AcquisitionWorker
from usb5133_daq.device.scope import NI5133, MockScope
from usb5133_daq.storage.csv_writer import save_waveform
from usb5133_daq.storage.data_saver import DataSaver
from usb5133_daq.ui.fft_plot import FFTPlot
from usb5133_daq.ui.waveform_plot import WaveformPlot
from usb5133_daq.analysis.feature_collector import FeatureCollector
from usb5133_daq.analysis.anomaly_detector import AnomalyDetector, AnomalyResult
from usb5133_daq.ui.anomaly_plot import AnomalyPlot

VOLTAGE_RANGES = [0.5, 1.0, 2.0, 5.0]
MAX_SAMPLE_RATE = 100_000_000.0


class MainWindow(QMainWindow):
    def __init__(self, use_mock: bool = False):
        super().__init__()
        self.setWindowTitle("NI USB-5133 DAQ")
        self.resize(1200, 700)

        self._scope = None
        self._worker = None
        self._last_waveform = None
        self._last_sample_rate = 1_000_000.0
        self._use_mock = use_mock
        self._collector: FeatureCollector | None = None
        self._detector: AnomalyDetector | None = None
        self._saver: DataSaver | None = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_control_panel())

        splitter = QSplitter(Qt.Vertical)
        self._waveform_plot = WaveformPlot()
        self._fft_plot = FFTPlot()
        splitter.addWidget(self._waveform_plot)
        splitter.addWidget(self._fft_plot)
        self._anomaly_plot = AnomalyPlot()
        splitter.addWidget(self._anomaly_plot)
        splitter.setSizes([400, 200, 150])  # three panels: waveform, FFT, anomaly
        root.addWidget(splitter)

        root.addWidget(self._build_action_bar())

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("준비")

    def _build_control_panel(self) -> QGroupBox:
        group = QGroupBox("설정")
        layout = QHBoxLayout(group)

        self._btn_connect = QPushButton("연결")
        self._btn_disconnect = QPushButton("해제")
        self._btn_disconnect.setEnabled(False)
        layout.addWidget(self._btn_connect)
        layout.addWidget(self._btn_disconnect)
        layout.addWidget(QLabel("|"))

        self._chk_ch0 = QCheckBox("CH0")
        self._chk_ch1 = QCheckBox("CH1")
        self._chk_ch0.setChecked(True)
        layout.addWidget(QLabel("채널:"))
        layout.addWidget(self._chk_ch0)
        layout.addWidget(self._chk_ch1)
        layout.addWidget(QLabel("|"))

        layout.addWidget(QLabel("샘플레이트 (Hz):"))
        self._edit_sample_rate = QLineEdit("1000000")
        self._edit_sample_rate.setFixedWidth(120)
        layout.addWidget(self._edit_sample_rate)
        layout.addWidget(QLabel("|"))

        layout.addWidget(QLabel("레코드 길이 (samples):"))
        self._edit_record_length = QLineEdit("1000")
        self._edit_record_length.setFixedWidth(80)
        layout.addWidget(self._edit_record_length)
        layout.addWidget(QLabel("|"))

        layout.addWidget(QLabel("전압 범위:"))
        self._combo_voltage = QComboBox()
        for v in VOLTAGE_RANGES:
            self._combo_voltage.addItem(f"±{v} V", v)
        self._combo_voltage.setCurrentIndex(1)
        layout.addWidget(self._combo_voltage)

        layout.addWidget(QLabel("|"))
        layout.addWidget(QLabel("수집주기(초):"))
        self._edit_cycle = QLineEdit("30")
        self._edit_cycle.setFixedWidth(45)
        layout.addWidget(self._edit_cycle)

        layout.addWidget(QLabel("수집창(초):"))
        self._edit_window = QLineEdit("5")
        self._edit_window.setFixedWidth(40)
        layout.addWidget(self._edit_window)

        layout.addWidget(QLabel("기준선횟수:"))
        self._edit_baseline = QLineEdit("20")
        self._edit_baseline.setFixedWidth(45)
        self._edit_baseline.setPlaceholderText("20 이상 권장")
        layout.addWidget(self._edit_baseline)

        layout.addStretch()
        return group

    def _build_action_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn_start = QPushButton("▶ 시작")
        self._btn_stop = QPushButton("■ 정지")
        self._btn_save = QPushButton("CSV 저장")
        self._btn_stop.setEnabled(False)
        self._btn_save.setEnabled(False)

        layout.addWidget(self._btn_start)
        layout.addWidget(self._btn_stop)
        layout.addStretch()
        layout.addWidget(self._btn_save)
        return widget

    def _connect_signals(self):
        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_save.clicked.connect(self._on_save)

    def _on_connect(self):
        try:
            self._scope = MockScope() if self._use_mock else NI5133()
            self._scope.connect()
            self._btn_connect.setEnabled(False)
            self._btn_disconnect.setEnabled(True)
            self._status_bar.showMessage("디바이스 연결됨")
        except Exception as e:
            QMessageBox.critical(self, "연결 오류", str(e))

    def _on_disconnect(self):
        # 주의: worker.wait(3000) — NI5133 fetch 블로킹 시 최대 3초 지연 가능.
        self._on_stop()
        if self._scope:
            self._scope.disconnect()
            self._scope = None
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._status_bar.showMessage("연결 해제됨")

    def _on_start(self):
        if self._scope is None:
            QMessageBox.warning(self, "경고", "먼저 디바이스를 연결하세요.")
            return

        channels = []
        if self._chk_ch0.isChecked():
            channels.append(0)
        if self._chk_ch1.isChecked():
            channels.append(1)
        if not channels:
            QMessageBox.warning(self, "경고", "최소 하나의 채널을 선택하세요.")
            return

        try:
            sample_rate = float(self._edit_sample_rate.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "샘플레이트에 유효한 숫자를 입력하세요.")
            return

        if sample_rate <= 0 or sample_rate > MAX_SAMPLE_RATE:
            QMessageBox.warning(
                self, "입력 오류",
                f"샘플레이트는 1 ~ {MAX_SAMPLE_RATE:,.0f} Hz 범위여야 합니다."
            )
            return

        try:
            record_length = int(self._edit_record_length.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "레코드 길이에 정수를 입력하세요.")
            return

        voltage_range = self._combo_voltage.currentData()

        try:
            self._scope.configure(sample_rate, record_length, channels, voltage_range)
        except Exception as e:
            QMessageBox.critical(self, "설정 오류", str(e))
            return

        self._last_sample_rate = sample_rate
        self._worker = AcquisitionWorker(self._scope)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        try:
            cycle_sec = float(self._edit_cycle.text())
            window_sec = float(self._edit_window.text())
            baseline_count = int(self._edit_baseline.text())
        except ValueError:
            cycle_sec, window_sec, baseline_count = 30.0, 5.0, 10

        self._collector = FeatureCollector(
            sample_rate=self._last_sample_rate,
            cycle_sec=cycle_sec,
            collect_window_sec=window_sec,
            channel=0,  # CH0 고정: 예지보전은 첫 번째 채널 기준으로 학습
        )
        self._detector = AnomalyDetector(baseline_count=baseline_count)
        # data_ready는 _on_data_ready(UI 갱신)와 collector.on_data(예지보전) 두 슬롯에 연결
        self._worker.data_ready.connect(self._collector.on_data)
        self._collector.features_ready.connect(self._detector.on_features)
        self._detector.result_ready.connect(self._on_anomaly_result)

        try:
            self._saver = DataSaver(save_dir="results")
        except OSError as exc:
            QMessageBox.critical(self, "저장 오류", f"results/ 폴더를 생성할 수 없습니다:\n{exc}")
            self._on_stop()
            return
        self._collector.raw_ready.connect(self._saver.on_raw)
        self._detector.result_ready.connect(self._saver.on_result)

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_bar.showMessage("수집 중...")

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None
        # Disconnect and clear DataSaver before nulling _collector / _detector
        if self._saver is not None and self._collector is not None and self._detector is not None:
            self._collector.raw_ready.disconnect(self._saver.on_raw)
            self._detector.result_ready.disconnect(self._saver.on_result)
        self._saver = None
        if self._collector is not None:
            self._collector.stop()  # 타이머 명시적 중지 (GC 타이밍 의존 방지)
        self._collector = None
        self._detector = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_bar.showMessage("수집 정지")

    def _on_data_ready(self, waveform: np.ndarray):
        self._last_waveform = waveform
        self._waveform_plot.update_waveform(waveform, self._last_sample_rate)
        self._fft_plot.update_fft(waveform, self._last_sample_rate)
        self._btn_save.setEnabled(True)

    def _on_error(self, message: str):
        self._on_stop()
        self._status_bar.showMessage(f"오류: {message}")
        QMessageBox.critical(self, "수집 오류", message)

    def _on_anomaly_result(self, result: AnomalyResult) -> None:
        self._anomaly_plot.update_result(result)
        if result.label == "LEARNING":
            tag = f"[학습 중 {result.baseline_progress}/{result.baseline_total}]"
        elif result.label == "NORMAL":
            tag = "[정상]"
        elif result.label == "WARNING":
            tag = "[WARNING]"
        else:
            tag = "[ALARM]"
        self._status_bar.showMessage(f"{tag} 수집 중...")

    def _on_save(self):
        if self._last_waveform is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "CSV 저장", "", "CSV Files (*.csv)"
        )
        if not filename:
            return
        try:
            save_waveform(self._last_waveform, self._last_sample_rate, filename)
            self._status_bar.showMessage(f"저장 완료: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    def closeEvent(self, event):
        # 주의: NI5133 fetch 블로킹 중일 경우 3초 타임아웃 후 종료. MockScope는 무관.
        self._on_disconnect()
        super().closeEvent(event)
