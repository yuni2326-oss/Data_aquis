# usb5133_daq/ui/status_light.py
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget


class _Bulb(QWidget):
    """단일 원형 전구."""

    _SIZE = 24

    def __init__(self, color_on: str, color_off: str, parent=None):
        super().__init__(parent)
        self._on_color = QColor(color_on)
        self._off_color = QColor(color_off)
        self._lit = False
        self.setFixedSize(self._SIZE, self._SIZE)

    def set_lit(self, lit: bool) -> None:
        if self._lit != lit:
            self._lit = lit
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = self._on_color if self._lit else self._off_color
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        margin = 2
        p.drawEllipse(margin, margin, self._SIZE - 2 * margin, self._SIZE - 2 * margin)


class StatusLight(QWidget):
    """신호등 스타일 상태 표시기.

    States: None (대기) | "LEARNING" | "NORMAL" | "WARNING" | "ALARM"
    """

    # (red, yellow, green)
    _LIGHT = {
        None:       (False, False, False),
        "LEARNING": (False, True,  False),
        "NORMAL":   (False, False, True),
        "WARNING":  (False, True,  False),
        "ALARM":    (True,  False, False),
    }
    _DEFAULT_TEXT = {
        None:       "대기",
        "LEARNING": "학습 중",
        "NORMAL":   "정상",
        "WARNING":  "WARNING",
        "ALARM":    "ALARM",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._red    = _Bulb("#FF3333", "#3a0808")
        self._yellow = _Bulb("#FFCC00", "#3a3000")
        self._green  = _Bulb("#33EE55", "#083a15")

        self._label = QLabel("대기")
        self._label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._label.setMinimumWidth(110)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(5)
        layout.addWidget(self._red)
        layout.addWidget(self._yellow)
        layout.addWidget(self._green)
        layout.addWidget(self._label)

    def set_state(self, state: str | None, extra: str = "") -> None:
        r, y, g = self._LIGHT.get(state, (False, False, False))
        self._red.set_lit(r)
        self._yellow.set_lit(y)
        self._green.set_lit(g)
        self._label.setText(extra if extra else self._DEFAULT_TEXT.get(state, "대기"))
