# usb5133_daq/ui/anomaly_plot.py
from __future__ import annotations

import math
from collections import deque

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from usb5133_daq.analysis.anomaly_detector import AnomalyResult

_HISTORY = 50
_THRESHOLD_NORMAL = -0.1   # above this = NORMAL
_THRESHOLD_WARNING = -0.3  # above this = WARNING, below = ALARM

_BG_COLORS = {
    "LEARNING": QColor(140, 140, 140, 40),
    "NORMAL":   QColor(0, 180, 0, 40),
    "WARNING":  QColor(255, 200, 0, 40),
    "ALARM":    QColor(220, 50, 50, 40),
}


class AnomalyPlot(QWidget):
    """이상 점수 시계열 그래프.

    update_result(result) 호출로 갱신. score는 IsolationForest score_samples() 값.
    높을수록 정상, 낮을수록 이상.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scores: deque[float] = deque(maxlen=_HISTORY)
        self._labels: deque[str] = deque(maxlen=_HISTORY)

        self._plot_widget = pg.PlotWidget(title="이상 점수 (Anomaly Score)")
        self._plot_widget.setLabel("left", "Score")
        self._plot_widget.setLabel("bottom", "주기 (최근 50회)")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setYRange(-0.6, 0.2)

        # Threshold lines — use Qt.DashLine (from PyQt5.QtCore import Qt)
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=_THRESHOLD_NORMAL, angle=0,
                            pen=pg.mkPen("g", width=1, style=Qt.DashLine),
                            label="NORMAL", labelOpts={"position": 0.05})
        )
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=_THRESHOLD_WARNING, angle=0,
                            pen=pg.mkPen("r", width=1, style=Qt.DashLine),
                            label="ALARM", labelOpts={"position": 0.05})
        )

        self._curve = self._plot_widget.plot(
            pen=pg.mkPen("#00BFFF", width=2),
            symbol="o",
            symbolSize=5,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_result(self, result: AnomalyResult) -> None:
        """AnomalyDetector.result_ready 슬롯에서 호출.

        Note: method is named update_result (not update) to avoid shadowing QWidget.update().
        """
        score = result.score if not math.isnan(result.score) else float("nan")
        self._scores.append(score)
        self._labels.append(result.label)

        # Update background color based on latest label
        color = _BG_COLORS.get(result.label, _BG_COLORS["LEARNING"])
        self._plot_widget.setBackground(color)

        valid = [(i, s) for i, s in enumerate(self._scores) if not math.isnan(s)]
        if valid:
            xs, ys = zip(*valid)
            self._curve.setData(list(xs), list(ys))
        else:
            self._curve.setData([], [])
