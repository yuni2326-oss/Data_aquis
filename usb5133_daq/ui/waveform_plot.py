# usb5133_daq/ui/waveform_plot.py
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

CHANNEL_COLORS = ["#00BFFF", "#FF6B6B"]


class WaveformPlot(QWidget):
    """실시간 파형 플롯 위젯. update_waveform(waveform, sample_rate) 호출로 갱신."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot_widget = pg.PlotWidget(title="파형")
        self._plot_widget.setLabel("left", "전압 (V)")
        self._plot_widget.setLabel("bottom", "시간 (s)")
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._curves = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_waveform(self, waveform: np.ndarray, sample_rate: float) -> None:
        num_channels, record_length = waveform.shape
        t = np.arange(record_length) / sample_rate

        if len(self._curves) != num_channels:
            for c in self._curves:
                self._plot_widget.removeItem(c)
            self._curves = [
                self._plot_widget.plot(
                    pen=pg.mkPen(CHANNEL_COLORS[i % len(CHANNEL_COLORS)], width=1),
                    name=f"CH{i}",
                )
                for i in range(num_channels)
            ]

        for i, curve in enumerate(self._curves):
            curve.setData(t, waveform[i])
