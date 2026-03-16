# Predictive Maintenance (예지보전) Feature Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** FFT-based anomaly detection using statistical baseline + Isolation Forest

---

## 1. Overview

펄스 신호를 주입한 장비에서 30초 주기로 5초 분량의 raw 파형을 수집하고, FFT 특징 벡터를 추출하여 이상 유무를 판단하는 예지보전 기능을 추가한다.

**두 단계 감지:**
- **Stage 1:** z-score 기반 통계적 기준선 (baseline 완료 즉시 동작)
- **Stage 2:** Isolation Forest `score_samples()` (baseline 완료 시 학습, 이후 연속 점수 출력)

---

## 2. Architecture

### 파이프라인
```
AcquisitionWorker.data_ready (10ms, shape: (num_channels, record_length))
    → FeatureCollector.on_data(waveform)
        └ waveform[channel] (기본: channel=0) 1D 슬라이스
        └ deque(maxlen=buffer_size) 에 append
        └ 첫 on_data 호출 시 QTimer 시작
        └ 30초 타이머 만료 시: 최근 5초 분량 슬라이스 → features.py → features_ready emit
    → AnomalyDetector.on_features(vec: np.ndarray shape [7])
        └ LEARNING: baseline 리스트에 append
        └ baseline_count 도달 시 (최초 1회):
            np.vstack(baseline) → IsolationForest.fit() + 평균/표준편차 계산
        └ 이후 매 주기: z-score + score_samples() → result_ready emit
    → MainWindow.on_anomaly_result(result: AnomalyResult)
        └ 상태바 텍스트 업데이트
        └ AnomalyPlot.update(result)
```

### 신규/수정 파일
| 파일 | 상태 | 역할 |
|------|------|------|
| `usb5133_daq/analysis/features.py` | NEW | FFT 특징 벡터 추출 |
| `usb5133_daq/analysis/feature_collector.py` | NEW | raw 버퍼 축적 + 주기 타이머 |
| `usb5133_daq/analysis/anomaly_detector.py` | NEW | z-score + IsolationForest 이상 감지 |
| `usb5133_daq/ui/anomaly_plot.py` | NEW | 이상 점수 시계열 그래프 |
| `usb5133_daq/ui/main_window.py` | MODIFY | FeatureCollector 연결, 파라미터 UI 추가 |

---

## 3. Component Specifications

### 3.1 `features.py`

**함수:** `extract_features(waveform: np.ndarray, sample_rate: float) -> np.ndarray`

- `waveform`: 1D array, shape `(N,)` — FeatureCollector이 `waveform_2d[channel]` 으로 슬라이스해서 전달
- 반환: shape `(7,)`, dtype float64
- 내부적으로 기존 `fft.py`의 `compute_fft()` 를 호출

**제로 신호 보호:** `dominant_mag < 1e-9` 이면 `np.zeros(7)` 반환 (NaN 전파 방지)

추출 특징:
| Index | 이름 | 설명 |
|-------|------|------|
| 0 | dominant_freq | 에너지 최대 주파수 (Hz) |
| 1 | dominant_mag | 해당 주파수 크기 |
| 2 | second_harmonic_mag | 2배 주파수 크기 |
| 3 | third_harmonic_mag | 3배 주파수 크기 |
| 4 | thd | (2nd+3rd) / dominant_mag; dominant_mag==0 이면 0.0 |
| 5 | noise_floor_rms | 피크 제외 스펙트럼 RMS |
| 6 | spectral_centroid | 스펙트럼 무게중심 주파수 (Hz) |

---

### 3.2 `feature_collector.py`

**클래스:** `FeatureCollector(QObject)`

**생성자 파라미터 (모두 수정 가능):**
- `sample_rate: float` — 하드웨어 확인된 실제 샘플레이트 (UI 입력값이 아닌 `scope.configure()` 이후의 실제 레이트)
- `cycle_sec: float = 30.0` — 수집 주기 (초)
- `collect_window_sec: float = 5.0` — 수집 창 (초)
- `channel: int = 0` — 특징 추출에 사용할 채널 인덱스

**내부 버퍼:**
```python
buffer_size = int(collect_window_sec * sample_rate) * 4  # 최대 4배 창 크기
_buf = collections.deque(maxlen=buffer_size)
```
- `deque(maxlen=buffer_size)` 사용 → 자동 오래된 샘플 eviction, 고정 메모리

**Signals:**
- `features_ready = pyqtSignal(object)` — `np.ndarray` shape `(7,)`

**동작:**
1. `on_data(waveform: np.ndarray)` 슬롯
   - `_buf.extend(waveform[self.channel])` 로 CH 슬라이스를 샘플 단위로 append
   - 첫 호출 시에만 `QTimer.start(cycle_sec * 1000)` 기동
2. 타이머 만료 슬롯 `_on_timer()`
   - 필요 샘플 수: `n_needed = int(collect_window_sec * sample_rate)`
   - `len(_buf) < n_needed` 이면 skip (수집 부족)
   - `np.array(list(_buf))[-n_needed:]` 슬라이스 → `extract_features()` 호출
   - 결과가 유효하면 `features_ready.emit(vec)`

---

### 3.3 `anomaly_detector.py`

**클래스:** `AnomalyDetector(QObject)`

**생성자 파라미터:**
- `baseline_count: int = 10` — 기준선 수집 횟수

> 주의: `baseline_count < 20`이면 `IsolationForest(contamination=0.05)`의 오염 비율이 정수 표현 불가(0.5샘플). 기본값 10은 동작하지만 contamination은 사실상 0으로 반올림됨. UI 안내 문구 권장: "정확도를 높이려면 20회 이상 권장".

**Signals:**
- `result_ready = pyqtSignal(object)` — `AnomalyResult`

**`AnomalyResult` dataclass:**
```python
@dataclass
class AnomalyResult:
    label: str          # "LEARNING" | "NORMAL" | "WARNING" | "ALARM"
    score: float        # IsolationForest score_samples() 값 (LEARNING 중엔 float('nan'))
    zscore_max: float   # 특징별 최대 z-score (LEARNING 중엔 float('nan'))
    baseline_progress: int  # 현재까지 수집된 baseline 수
    baseline_total: int     # baseline_count 값
```

**동작 (on_features 슬롯, 순차 평가):**
```python
def on_features(self, vec: np.ndarray):
    if len(self._baseline) < self._baseline_count:
        self._baseline.append(vec)
        if len(self._baseline) == self._baseline_count:
            X = np.vstack(self._baseline)          # shape (baseline_count, 7)
            self._if_model = IsolationForest(contamination=0.05).fit(X)
            self._mean = X.mean(axis=0)
            self._std  = X.std(axis=0) + 1e-9      # 0 나누기 방지
        emit LEARNING result
        return

    zscore_max = np.max(np.abs((vec - self._mean) / self._std))
    if_score   = float(self._if_model.score_samples([vec])[0])

    # 순차 평가 (첫 번째 조건 충족 시 적용)
    if zscore_max < 2.0 and if_score > -0.1:
        label = "NORMAL"
    elif zscore_max < 3.0 and if_score > -0.3:
        label = "WARNING"
    else:
        label = "ALARM"

    emit AnomalyResult(label, if_score, zscore_max, ...)
```

**라이프사이클:** 시작 버튼 클릭 시 `FeatureCollector`와 함께 새로 생성 (이전 baseline 및 모델 초기화). 정지 후 재시작 시 baseline 재수집 시작.

---

### 3.4 `anomaly_plot.py`

**클래스:** `AnomalyPlot(QWidget)` (pyqtgraph 사용, 기존 플롯과 동일 패턴)

- 최근 50개 이상 점수(`score`) 시계열 표시
- y축: Isolation Forest `score_samples()` 값 (높을수록 정상)
- 수평 구분선: `-0.1` (NORMAL/WARNING 경계), `-0.3` (WARNING/ALARM 경계)
- LEARNING 구간: 회색 배경
- NORMAL: 초록, WARNING: 노랑, ALARM: 빨강 구분색 배경

---

### 3.5 `main_window.py` 수정

**설정 패널에 새 컨트롤 추가:**
- `수집 주기 (초)`: QLineEdit, 기본값 `30`
- `수집 창 (초)`: QLineEdit, 기본값 `5`
- `기준선 횟수`: QLineEdit, 기본값 `10` (힌트 텍스트: "정확도를 위해 20 이상 권장")

**시작 시 (`_on_start`):**
```python
# scope.configure() 호출 후 실제 sample_rate를 읽거나
# 현재처럼 _last_sample_rate = sample_rate (입력값) 사용
self._collector = FeatureCollector(
    sample_rate=self._last_sample_rate,
    cycle_sec=float(self._edit_cycle.text()),
    collect_window_sec=float(self._edit_window.text()),
    channel=0,
)
self._detector = AnomalyDetector(
    baseline_count=int(self._edit_baseline.text())
)
self._worker.data_ready.connect(self._collector.on_data)
self._collector.features_ready.connect(self._detector.on_features)
self._detector.result_ready.connect(self._on_anomaly_result)
```

**정지 시 (`_on_stop`):** `_collector`와 `_detector`를 `None`으로 해제.

**상태바 포맷 (ASCII-safe):**
- `[학습 중 3/10] 수집 중...`
- `[정상] 수집 중...`
- `[WARNING] 수집 중...`
- `[ALARM] 수집 중...`

---

## 4. Configurable Parameters Summary

| 파라미터 | 기본값 | 위치 | 설명 |
|----------|--------|------|------|
| `cycle_sec` | 30.0 | UI | 수집 주기 (초) |
| `collect_window_sec` | 5.0 | UI | 수집 창 (초) |
| `baseline_count` | 10 | UI | 기준선 수집 횟수 (20 이상 권장) |
| `contamination` | 0.05 | 코드 상수 | IsolationForest 이상 비율 가정 |
| `history_size` | 50 | 코드 상수 | AnomalyPlot 표시 이력 수 |
| `channel` | 0 | 코드 상수 | 특징 추출 채널 인덱스 |

---

## 5. Testing Plan

- `tests/test_features.py`
  - sine 파형 입력 → shape `(7,)` 반환 확인
  - 제로 파형 입력 → `np.zeros(7)` 반환 (NaN 없음)
  - dominant_freq 값 검증 (알려진 주파수 sine 파형)

- `tests/test_feature_collector.py`
  - 버퍼 축적: `n_needed` 미만 데이터 → skip 확인
  - 타이머 첫 호출 시 기동 확인 (mock QTimer)
  - 채널 슬라이스: 2채널 입력 → channel=0 선택 확인

- `tests/test_anomaly_detector.py`
  - `baseline_count`개 입력 전: `LEARNING` label 확인
  - `baseline_count`번째 입력 시: IsolationForest fit 호출 확인
  - 정상 벡터 입력 → `NORMAL`
  - 이상 벡터 입력 → `WARNING` 또는 `ALARM`

---

## 6. Dependencies

- `scikit-learn>=1.0` (IsolationForest) — `requirements.txt`에 추가
- `pyqtgraph` — 기존 사용 여부 확인 후 없으면 추가
