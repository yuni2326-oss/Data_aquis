# NI USB-5133 DAQ — 실시간 이상 감지 시스템

NI USB-5133 오실로스코프로 신호를 수집하고, FFT 특징 추출 → 머신러닝 기반 이상 감지까지 수행하는 PyQt5 데스크탑 애플리케이션입니다.

---

## 실행 방법

```bash
# 실제 하드웨어 연결 시
python main.py

# 하드웨어 없이 Mock 모드로 실행 (개발/테스트)
python main.py --mock
```

---

## 전체 신호 흐름

```
NI USB-5133 (또는 MockScope)
        │  data_ready  (np.ndarray shape: channels × samples)
        ▼
  FeatureCollector
   · 30초마다 최근 5초 분량의 샘플을 잘라냄
   · FFT 특징 벡터 7개 추출
        │  features_ready  (np.ndarray shape: 7)
        ▼
  AnomalyDetector
   · 처음 10회: 베이스라인 학습
   · 이후: z-score + IsolationForest로 판정
        │  result_ready  (AnomalyResult)
        ▼
  UI (StatusLight, AnomalyPlot)
   · LEARNING / NORMAL / WARNING / ALARM 표시
```

---

## 이상 감지 알고리즘 상세

### 1단계 — FFT 특징 추출 (`features.py`)

매 판정 주기마다 5초 분량의 원시 파형에서 다음 **7개 특징**을 추출합니다.

| 인덱스 | 특징명 | 설명 |
|--------|--------|------|
| 0 | `dominant_freq` | FFT 스펙트럼에서 크기가 가장 큰 주파수 (Hz) |
| 1 | `dominant_mag` | 지배 주파수의 크기 |
| 2 | `second_harmonic_mag` | 지배 주파수 × 2 위치의 크기 (2차 고조파) |
| 3 | `third_harmonic_mag` | 지배 주파수 × 3 위치의 크기 (3차 고조파) |
| 4 | `thd` | 고조파 왜곡율 = (2차 + 3차) / 지배 크기 |
| 5 | `noise_floor_rms` | 지배 주파수 주변 ±5 bin을 제외한 나머지 RMS (잡음 수준) |
| 6 | `spectral_centroid` | 스펙트럼 무게중심 주파수 (Hz) |

**FFT 계산 방식 (`fft.py`)**

- 누설(spectral leakage)을 줄이기 위해 **Hanning 윈도우**를 적용한 뒤 FFT 수행
- 단측 스펙트럼(0 ~ Nyquist)만 사용
- 크기 정규화: `|FFT| × 2 / N`

신호가 사실상 0인 경우(`dominant_mag < 1e-9`) 전체 특징을 0으로 반환해 NaN 전파를 방지합니다.

---

### 2단계 — 베이스라인 학습 (`anomaly_detector.py`)

처음 **10회** 특징 벡터가 수집되는 동안은 `LEARNING` 상태를 유지하며 판정하지 않습니다.

10회가 채워지면 다음을 계산해 저장합니다.

| 저장값 | 용도 |
|--------|------|
| `_mean` (shape: 7) | 각 특징의 평균 → z-score 기준점 |
| `_std` (shape: 7) | 각 특징의 표준편차 (0 방지를 위해 `+1e-9`) |
| `_if_model` | 베이스라인 10개로 학습된 IsolationForest 모델 |
| `_score_mean`, `_score_std` | 베이스라인 IF 점수의 평균·표준편차 → 정규화 기준 |

---

### 3단계 — 이중 지표 이상 판정

학습 완료 후 매 주기마다 두 가지 지표를 동시에 계산합니다.

#### 지표 A: z-score (통계적 이탈도)

```
z_i = (x_i - mean_i) / std_i        (i = 0..6, 각 특징별)
zscore_max = max(|z_0|, |z_1|, ..., |z_6|)
```

- 베이스라인 대비 각 특징이 표준편차의 몇 배나 벗어났는지 측정
- 특징 중 **하나라도** 크게 변하면 즉시 반응

#### 지표 B: IsolationForest 정규화 편차 (복합 패턴 이탈도)

```
if_score  = IsolationForest.score_samples([현재 벡터])
              ← 정상일수록 0에 가까운 음수, 이상일수록 더 큰 음수
norm_dev  = (if_score - score_mean) / score_std
              ← 베이스라인 점수 분포 기준으로 표준화
```

- 7개 특징의 **조합 패턴**이 베이스라인과 얼마나 다른지 측정
- z-score가 각 축을 개별로 보는 반면, IsolationForest는 다차원 공간에서 밀도를 봄

> **왜 두 지표를 모두 쓰나?**
> z-score는 단일 특징의 급변에 빠르게 반응하지만 다차원 상관관계를 놓칩니다.
> IsolationForest는 개별 특징이 경계 내에 있어도 조합이 이상한 경우를 잡아냅니다.
> 두 지표를 AND 조건으로 결합하면 **오탐(false positive)을 줄이면서 탐지 범위를 확대**할 수 있습니다.

---

### 4단계 — 판정 기준표

| 상태 | 조건 | 의미 |
|------|------|------|
| `LEARNING` | 베이스라인 수집 중 | 아직 모델 미완성 |
| `NORMAL` | `zscore_max < 3.0` **AND** `norm_dev > -2.0` | 정상 범위 |
| `WARNING` | `zscore_max < 5.0` **AND** `norm_dev > -3.0` | 주의 필요 |
| `ALARM` | 위 조건 모두 불만족 | 즉시 점검 필요 |

---

### 데이터 수집 주기 (`feature_collector.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `cycle_sec` | 30초 | 판정 주기 (타이머 간격) |
| `collect_window_sec` | 5초 | FFT에 사용할 샘플 길이 |
| 버퍼 크기 | `n_needed × 4` | 최신 데이터를 deque로 유지 |

타이머가 발동하면 deque에서 **가장 최근** `n_needed`개 샘플을 꺼내 특징 추출에 사용합니다.

---

## 프로젝트 구조

```
usb5133_daq/
├── acquisition/
│   └── worker.py          # NI 드라이버 호출, data_ready 신호 emit
├── analysis/
│   ├── fft.py             # Hanning 윈도우 FFT
│   ├── features.py        # 7개 FFT 특징 추출
│   ├── feature_collector.py  # 주기적 수집 및 raw_ready/features_ready emit
│   └── anomaly_detector.py   # 베이스라인 학습 + 이중 지표 판정
├── storage/
│   ├── csv_writer.py      # 원시 데이터 CSV 저장
│   └── data_saver.py      # FFT 결과 저장 (start/stop 연동)
├── ui/
│   ├── main_window.py     # 메인 윈도우, 전체 생명주기 관리
│   ├── waveform_plot.py   # 실시간 파형 표시
│   ├── fft_plot.py        # FFT 스펙트럼 표시
│   ├── anomaly_plot.py    # 이상 점수 시계열 표시
│   └── status_light.py    # NORMAL/WARNING/ALARM 상태 표시등
└── device/
    └── scope.py           # NI USB-5133 드라이버 래퍼 (MockScope 포함)
```

---

## 의존성

- Python 3.10+
- PyQt5
- numpy
- scikit-learn
- NI-SCOPE 드라이버 (실제 하드웨어 사용 시)
