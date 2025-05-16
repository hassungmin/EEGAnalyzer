# EEGAnalyzerGUI

> Python 기반 GUI 도구로, EEG(raw) 데이터를 불러와 필터링, ICA, 에포킹, ERP 시각화를 수행할 수 있는 데스크탑 애플리케이션입니다.

## 🧠 프로젝트 개요

**EEGAnalyzerGUI**는 PyQt5 및 MNE-Python을 기반으로 하여, EEG 데이터를 효율적으로 탐색하고 처리할 수 있도록 도와주는 GUI 프로그램입니다. 이 도구는 ERP 분석 실습, 시각적 디버깅, 데이터 전처리 테스트 등에 유용합니다.

## ⚙️ 환경 설정

이 프로그램을 실행하기 위해선 다음의 환경이 필요합니다:

- Python 3.8+
- PyQt5
- MNE
- numpy
- pandas
- matplotlib

설치 예시:
```bash
pip install PyQt5 mne numpy pandas matplotlib
```

## ▶️ 실행 방법

```bash
python eeg_analyzer_gui.py
```

실행 후 GUI 창이 나타나며, 각 단계별로 EEG 분석 기능을 수행할 수 있습니다.

## 📁 구성 파일 설명

| 파일명 | 설명 |
|--------|------|
| `eeg_analyzer_gui.py` | EEG 데이터를 불러오고 전처리(필터, ICA), 에포킹, ERP 분석 및 시각화를 지원하는 메인 GUI 애플리케이션입니다. |
| `generate_test_eeg_data.py` | 임의 EEG 및 이벤트 데이터를 생성하는 스크립트입니다. 실험 전 GUI 기능을 테스트하는 데 유용합니다. |

## 🔬 주요 기능

- **EEG 데이터 불러오기**: CSV 포맷의 EEG 데이터를 불러와 MNE의 `RawArray`로 변환
- **이벤트 데이터 로드**: `sample`, `previous`, `event_id` 형식의 이벤트 타이밍 정보를 로드
- **필터링**: 다양한 EEG 필터(Band-pass, Notch, High-pass 등) 적용 가능
- **ICA**: 독립 성분 분석을 통한 아티팩트 제거
- **에포킹**: 이벤트 기반 시간 창 설정 후 에포크 생성
- **ERP 분석**: 조건별 ERP 파형 시각화 및 평균

## 🧪 샘플 데이터 생성

테스트용 EEG/이벤트 데이터를 아래 명령어로 생성할 수 있습니다:

```bash
python generate_test_eeg_data.py
```

생성된 파일:
- `test_eeg_data_YYYYMMDD_HHMMSS.csv`
- `test_events_YYYYMMDD_HHMMSS.csv`

## 📌 참고 문헌

- Luck, S. J. (2014). *An Introduction to the Event-Related Potential Technique (2nd ed.)*
- Widmann et al. (2015). *Digital filter design for electrophysiological data.*
- Delorme & Makeig (2004). *EEGLAB: an open source toolbox.*
- Tanner et al. (2015). *Common filter artifacts in electrophysiology.*
- Hu et al. (2010). *Zero-phase filtering in EEG.*
