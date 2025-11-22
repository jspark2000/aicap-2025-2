# KNHANES 대사성 질환 예측 프로젝트

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jspark2000/aicap-2025-2/blob/main/preprocessing.ipynb)

> **참고**: Colab에서 직접 노트북 파일을 업로드하는 방법으로도 실행이 가능합니다.

비흡연·비음주 성인을 대상으로 생물심리사회(BPS) 모델을 적용하여 고혈압, 당뇨병, 이상지질혈증 등 대사성 질환 발병 요인을 다차원적으로 분석함

## 프로젝트 개요

이 프로젝트는 한국국민건강영양조사(KNHANES) 2023 데이터를 활용하여:

- **생물학적 요인**: 연령, 성별, BMI, 혈압, 혈당, 혈중 지질 등
- **심리학적 요인**: 스트레스 인지율, GAD-7 점수 등
- **사회/행동적 요인**: 교육수준, 소득, 신체활동, 영양소 섭취량 등

세 가지 영역의 변수들이 대사성 질환 발병에 미치는 영향을 전통적인 회귀 모델과 머신러닝 알고리즘을 통해 분석합니다.

## 프로젝트 구조

```
aicap-2025-2/
├── data/
│   ├── khanes-2023.csv              # 원본 KNHANES 2023 데이터
│   ├── headers.csv                   # 컬럼명-설명 매핑 파일
│   └── khanes_preprocessed.parquet   # 전처리 완료 데이터셋
├── docs/
│   └── methods.pdf                   # 연구 방법론 문서
├── src/
│   ├── __init__.py
│   └── utils.py                      # 데이터 전처리 유틸리티 함수
├── preprocessing.ipynb                # 전처리 및 EDA 노트북
├── environment.yaml                  # Conda 환경 설정 파일
└── README.md                         # 프로젝트 설명서
```

## 시작하기

### 사전 요구사항

- Python 3.11
- Conda (또는 Miniconda)

### 설치 방법

1. **저장소 클론**
   ```bash
   git clone https://github.com/jspark2000/aicap-2025-2.git
   cd aicap-2025-2
   ```

2. **Conda 환경 생성 및 활성화**
   ```bash
   conda env create -f environment.yaml
   conda activate khanes
   ```

3. **데이터 준비**
   - `data/khanes-2023.csv` 파일이 있는지 확인하세요.

### 실행 방법

1. **Jupyter Lab 실행**
   ```bash
   jupyter lab
   ```

2. **전처리 노트북 실행**
   - `preprocessing.ipynb` 파일을 열고 셀을 순차적으로 실행합니다.
   - 전처리 완료 후 `data/khanes_preprocessed.parquet` 파일이 생성됩니다.

## Google Colab에서 실행하기

이 프로젝트는 Google Colab에서도 실행할 수 있습니다. 노트북에 Colab 자동 설정 기능이 포함되어 있어 간편하게 실행할 수 있습니다.

### 방법 1: Colab Badge 클릭

README 상단의 **"Open In Colab"** 배지를 클릭하면 자동으로 Colab 환경이 열립니다.


### 방법 2: 노트북 직접 업로드

1. **Colab 노트북 열기**
   - [Google Colab](https://colab.research.google.com/) 접속
   - `파일 > 노트북 업로드` 또는 `GitHub` 탭에서 `preprocessing.ipynb` 파일 열기

2. **자동 환경 설정**
   - 노트북의 **"Google Colab에서 실행하기"** 섹션의 첫 번째 셀을 실행하면 자동으로:
     - 필요한 패키지 설치
     - 디렉토리 구조 생성 (`data/`, `src/`)
     - `src/utils.py` 파일 다운로드 시도

3. **데이터 파일 업로드**
   - 다음 셀을 실행하여 `khanes-2023.csv` 파일을 업로드합니다.
   - 업로드된 파일은 자동으로 `data/` 폴더로 이동됩니다.

4. **나머지 셀 실행**
   - 이후 셀들을 순차적으로 실행하면 전처리가 완료됩니다.


## 데이터 전처리 과정

전처리 파이프라인은 다음과 같은 단계로 구성됩니다:

1. **데이터 로드**: KNHANES CSV 파일 읽기
2. **상징값 처리**: 설문조사 상징값(8, 88, 888 등)을 NaN으로 변환
3. **필터링**: 비흡연·비음주 성인, 20-64세 대상자만 선별
4. **이상치 처리**: IQR 기반 이상치 제한(capping)
5. **결측치 보간**: 연령 기준 선형 보간법 적용
6. **표준화**: Z-score 표준화 수행
7. **타겟 변수 생성**: 대사성 질환 플래그 생성 (고혈압/당뇨/이상지질혈증 중 하나 이상)

## 주요 변수 설명

### 종속 변수 (Dependent Variable)
- `metabolic_flag`: 대사성 질환 유무 (1=있음, 0=없음)
  - `DI1_dg`: 고혈압 의사진단 여부
  - `DE1_dg`: 당뇨병 의사진단 여부
  - `DI2_dg`: 이상지질혈증 의사진단 여부

### 독립 변수 (Independent Variables)

#### 생물학적 영역 (Biological Domain)
- `sex`: 성별
- `age`: 나이
- `HE_BMI`: 체질량지수
- `HE_sbp`, `HE_dbp`: 수축기/이완기 혈압
- `HE_glu`: 공복혈당
- `HE_HbA1c`: 당화혈색소
- `HE_chol`, `HE_HDL_st2`, `HE_TG`, `HE_LDL_drct`: 혈중 지질 프로필

#### 심리학적 영역 (Psychological Domain)
- `mh_stress`: 스트레스 인지율
- `gad_score`: GAD-7 점수 (BP_GAD_1~7 합계)

#### 사회/행동적 영역 (Social/Behavioral Domain)
- `edu`: 교육수준
- `ho_incm5`: 가구 소득 5분위수
- `marri_1`: 혼인상태
- `pa_aerobic`: 유산소 신체활동 빈도
- `N_EN`, `N_PROT`, `N_FAT`, `N_SFA`, `N_CHO`, `N_TDF`, `N_SUGAR`: 주요 영양소 섭취량
- `N_NA`, `N_K`: 나트륨, 칼륨 섭취량
- `N_VITC`, `N_VITD`: 비타민 C, D 섭취량

## 분석 방법론

본 연구는 생물심리사회(BPS) 모델을 이론적 틀로 하여 다음과 같은 분석을 수행합니다:

1. **기술통계 및 그룹 비교**
   - 대사성 질환 유무에 따른 변수별 차이 검정
   - t-test, Mann-Whitney U test, Chi-square test

2. **단계적 다변량 로지스틱 회귀분석**
   - Model 1: 생물학적 요인만
   - Model 2: Model 1 + 심리학적 요인
   - Model 3: Model 2 + 사회/행동적 요인
   - Model 4: 전체 변수 포함 (완전 조정 모델)

3. **머신러닝 예측 모델**
   - Logistic Regression (기준 모델)
   - AdaBoost, Random Forest, Gradient Boosting
   - XGBoost, Support Vector Machine (SVM)
   - Artificial Neural Network (ANN)

4. **모델 평가 지표**
   - ROC-AUC (주요 지표)
   - Accuracy, Kappa, Sensitivity, Specificity, F1-score

## 유틸리티 함수

`src/utils.py`에는 다음과 같은 전처리 함수들이 포함되어 있습니다:

- `load_khanes_data()`: KNHANES CSV 파일 로드
- `replace_numeric_sentinels()`: 상징값을 NaN으로 변환
- `cap_iqr_outliers()`: IQR 기반 이상치 제한
- `interpolate_numeric_features()`: 선형 보간법 적용
- `standardize_features()`: Z-score 표준화
- `compute_metabolic_flag()`: 대사성 질환 플래그 생성
- `filter_non_smoking_non_drinking()`: 연구 대상자 필터링

## 전처리 결과

전처리 완료 후 생성되는 데이터셋:

- **파일명**: `data/khanes_preprocessed.parquet`
- **행 수**: 1,256명 (비흡연·비음주 성인, 20-64세)
- **열 수**: 38개 (ID, 생물학적/심리학적/사회행동적 변수, 타겟 변수)
- **대사성 질환 유병률**: 28.8% (362명)

## 참고 자료

- [KNHANES 공식 사이트](https://knhanes.kdca.go.kr/)
- 연구 방법론 상세 내용: `docs/methods.pdf`
