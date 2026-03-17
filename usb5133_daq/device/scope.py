# usb5133_daq/device/scope.py
from __future__ import annotations

import numpy as np

VALID_VOLTAGE_RANGES = {0.5, 1.0, 2.0, 5.0}
MAX_SAMPLE_RATE = 100_000_000.0


class _ScopeBase:
    """NI5133과 MockScope가 공유하는 인터페이스."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def configure(
        self,
        sample_rate: float,
        record_length: int,
        channels: list,
        voltage_range: float,
    ):
        raise NotImplementedError

    def start_acquisition(self) -> None:
        """수집 시작 전 호출. NI5133에서는 initiate() 실행."""

    def stop_acquisition(self) -> None:
        """수집 종료 후 호출. NI5133에서는 abort() 실행."""

    def fetch(self) -> np.ndarray:
        raise NotImplementedError


class MockScope(_ScopeBase):
    """하드웨어 없는 환경에서 테스트용 사인파 생성."""

    def __init__(self):
        self._connected = False
        self.sample_rate = 0.0
        self.record_length = 0
        self.channels = []
        self.voltage_range = 0.0

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def configure(self, sample_rate, record_length, channels, voltage_range):
        if not self._connected:
            raise RuntimeError("connect() must be called before configure()")
        if sample_rate > MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate {sample_rate} exceeds maximum {MAX_SAMPLE_RATE} Hz"
            )
        if voltage_range not in VALID_VOLTAGE_RANGES:
            raise ValueError(
                f"voltage_range {voltage_range} is invalid. "
                f"Valid values: {sorted(VALID_VOLTAGE_RANGES)}"
            )
        self.sample_rate = sample_rate
        self.record_length = record_length
        self.channels = list(channels)
        self.voltage_range = voltage_range

    def start_acquisition(self) -> None:
        """수집 시작 (MockScope에서는 no-op)."""

    def stop_acquisition(self) -> None:
        """수집 종료 (MockScope에서는 no-op)."""

    def fetch(self) -> np.ndarray:
        """shape (num_channels, record_length)의 사인파 반환."""
        t = np.arange(self.record_length) / self.sample_rate
        freq = 1000.0
        sine = np.sin(2 * np.pi * freq * t)
        return np.stack([sine] * len(self.channels))


class NI5133(_ScopeBase):
    """NI USB-5133 디지타이저 래퍼. niscope 드라이버 필요."""

    def __init__(self, resource_name: str = "Dev1"):
        self._resource_name = resource_name
        self._session = None
        self.sample_rate = 0.0
        self.record_length = 0
        self.channels = []
        self.voltage_range = 0.0

    def connect(self):
        import niscope  # noqa: PLC0415

        self._session = niscope.Session(self._resource_name)

    def disconnect(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def configure(self, sample_rate, record_length, channels, voltage_range):
        if self._session is None:
            raise RuntimeError("connect() must be called before configure()")
        if sample_rate > MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate {sample_rate} exceeds maximum {MAX_SAMPLE_RATE} Hz"
            )
        if voltage_range not in VALID_VOLTAGE_RANGES:
            raise ValueError(
                f"voltage_range {voltage_range} is invalid. "
                f"Valid values: {sorted(VALID_VOLTAGE_RANGES)}"
            )
        self.sample_rate = sample_rate
        self.record_length = record_length
        self.channels = list(channels)
        self.voltage_range = voltage_range

        import niscope  # noqa: PLC0415
        ch_str = ",".join(str(c) for c in channels)
        self._session.channels[ch_str].configure_vertical(
            range=voltage_range,
            coupling=niscope.VerticalCoupling.DC,
        )
        self._session.configure_horizontal_timing(
            min_sample_rate=sample_rate,
            min_num_pts=record_length,
            ref_position=50.0,
            num_records=1,
            enforce_realtime=True,
        )

    def start_acquisition(self) -> None:
        """연속 수집 준비. 실제 arm은 fetch() 내부에서 호출마다 수행."""

    def stop_acquisition(self) -> None:
        """연속 수집 중단. AcquisitionWorker.stop() 후 호출."""
        try:
            self._session.abort()
        except Exception:
            pass

    def fetch(self) -> np.ndarray:
        """shape (num_channels, record_length)의 파형 반환.

        num_records=1 제약으로 인해 매 호출마다 abort→initiate로 재-arm한다.
        이 방식이 NI-SCOPE에서 단일 레코드 연속 수집의 표준 패턴이다.
        """
        self._session.abort()    # 이전 상태 초기화 (미시작 시 no-op)
        self._session.initiate()  # 새 레코드 arm
        ch_str = ",".join(str(c) for c in self.channels)
        waveforms = self._session.channels[ch_str].fetch(
            num_samples=self.record_length,
        )
        return np.stack([np.array(w.samples) for w in waveforms])
