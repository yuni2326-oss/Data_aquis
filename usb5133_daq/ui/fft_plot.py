# usb5133_daq/ui/fft_plot.py
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from usb5133_daq.analysis.fft import compute_fft

CHANNEL_COLORS = ["#00BFFF", "#FF6B6B"]


class FFTPlot(QWidget):
    """FFT 스펙트럼 플롯 위젯. update_fft(waveform, sample_rate) 호출로 갱신."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot_widget = pg.PlotWidget(title="FFT 스펙트럼")
        self._plot_widget.setLabel("left", "크기")
        self._plot_widget.setLabel("bottom", "주파수 (Hz)")
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._curves = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_fft(self, waveform: np.ndarray, sample_rate: float) -> None:
        num_channels = waveform.shape[0]

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
            freqs, mags = compute_fft(waveform[i], sample_rate)
            curve.setData(freqs, mags)
