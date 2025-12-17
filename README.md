# 네트워크 분석 기반 주식 포트폴리오 구성 프로젝트 (Network-Based Portfolio Strategy)

## 1. 프로젝트 개요 (Introduction)

이 프로젝트는 **주가 데이터로부터 네트워크를 구성**하고, 네트워크 이론(Network Theory)을 적용하여 구성한 포트폴리오가 투자 성과에 어떤 영향을 미치는지 분석합니다.

기존의 단순한 가격 변동 분석을 넘어, **주식들 간의 상호 연관성(Correlation)**을 네트워크 형태로 모델링하여 시장의 구조적 특성을 파악합니다.

**핵심 아이디어:**
> "시장 전체의 공통된 움직임(Market Mode)을 제거한 후 남은 **고유한 연관성(Residual Correlation)**으로 네트워크를 구축하면, 진정한 기업 간의 관계를 볼 수 있지 않을까?"
> "이 네트워크에서 **중심성(Centrality)**이 높은 종목과 낮은 종목을 선별하여 투자하면 어떤 성과가 날까?"

본 프로젝트는 **S&P 500** 종목을 대상으로 분기별 네트워크를 구축하고, 다양한 포트폴리오 전략(중심성 기반, 가중치 기반 등)의 성과를 시뮬레이션(백테스팅)하여 검증합니다.

---

## 2. 분석 방법론 (Methodology)

### 1단계: 데이터 수집 및 전처리
- **대상**: S&P 500 포함 종목
- **소스**: `yfinance` (Yahoo Finance API)
- **데이터**: 수정 주가(Adjusted Close), 거래량, S&P 500 지수(^GSPC)

### 2단계: 네트워크 구축 (Network Construction)
1.  **시장 효과 제거 (Market Residuals)**:
    - CAPM 모델을 활용하여 개별 종목 수익률에서 시장(S&P 500)의 영향을 분리해냅니다.
    - $R_i = \alpha_i + \beta_i R_m + \epsilon_i$ 에서 잔차 $\epsilon_i$를 추출합니다.
2.  **상관관계 계산**:
    - 추출된 잔차들 간의 피어슨 상관계수(Pearson Correlation)를 계산합니다.
3.  **네트워크 필터링 (Thresholding)**:
    - 통계적 유의성 검정 및 임계값(Threshold)을 적용하여, 유의미한 연결(Edge)만을 남깁니다.

### 3단계: 네트워크 분석 및 전략 수립
1.  **커뮤니티 탐지 (Community Detection)**:
    - Louvain 알고리즘을 사용하여 서로 밀접하게 연결된 종목들의 군집(Cluster)을 식별합니다. 이는 유사한 주가 움직임을 보이는 산업/테마 그룹으로 해석될 수 있습니다.
2.  **중심성 분석 (Centrality Analysis)**:
    - 각 커뮤니티 내에서 **고유벡터 중심성(Eigenvector Centrality)**을 계산하여, 그룹 내 영향력이 큰 종목('리더')과 작은 종목('아웃사이더')을 구분합니다.
3.  **포트폴리오 구성**:
    - **Max Centrality**: 각 커뮤니티에서 중심성이 가장 높은 종목들로 구성
    - **Min Centrality**: 각 커뮤니티에서 중심성이 가장 낮은 종목들로 구성
    - **Weighted Sector**: 섹터별 비중을 고려한 가중 포트폴리오 (비교군)
    - **Random**: 무작위로 선택된 포트폴리오 (통계적 검증용 벤치마크)

### 4단계: 백테스팅 (Backtesting)
- **롤링 윈도우(Rolling Window)** 방식을 사용합니다.
- 예: 2020년 1분기 데이터로 네트워크 구축 -> 2020년 2분기에 투자 -> 성과 측정
- **평가 지표**: 누적 수익률, 변동성(Volatility), 샤프 비율(Sharpe Ratio), MDD 등.
- **Top % 분석**: 전략 포트폴리오가 무작위 포트폴리오 분포 상위 몇 %에 위치하는지 분석하여 통계적 유의성을 검증합니다.

---

## 3. 프로젝트 구조 (Project Structure)

```bash
/
├── run.py                      # [메인] 전체 분석 파이프라인 실행 (데이터 로드 -> 분석 -> 결과 저장)
├── config.py                   # 분석 기간, 파라미터, 경로 설정 등 통합 설정 파일
├── analyze_results.py          # 백테스팅 결과 종합 분석 및 요약 리포트 생성
├── plot_metrics.py             # 네트워크 구조 지표(밀도, 모듈성 등) 시각화
├── requirements.txt            # 의존성 패키지 목록
│
├── pipeline/                   # 핵심 분석 로직 패키지
│   ├── data_loader.py          # 데이터 다운로드 및 로드
│   ├── corr_calculator.py      # 잔차 상관관계 계산
│   ├── threshold.py            # 엣지 필터링
│   ├── network_analysis.py     # 커뮤니티 탐지 및 중심성 계산
│   ├── portfolio.py            # 포트폴리오 성과 측정
│   └── weighted_portfolio.py   # 가중치 포트폴리오 로직
│
└── results/                    # (자동 생성) 모든 분석 결과물이 저장되는 곳
    ├── summary/
    │   ├── network_metrics.csv         # 분기별 네트워크 구조 지표
    │   └── ...
    ├── global_figures/
    │   └── network_metrics_timeseries.png # 네트워크 지표 변화 그래프
    └── quarterly/              # 분기별 상세 결과
        ├── 2020Q1/
        │   ├── clusters/       # 커뮤니티 노드/엣지 정보 (CSV)
        │   ├── figures/        # 네트워크 시각화 및 차수 분포 그래프
        │   ├── network_data/   # 상관관계 행렬 및 Gephi용 데이터
        │   └── portfolios/     # 포트폴리오 구성 및 백테스트 결과 (JSON, CSV)
        └── ...
```

---

## 4. 사용 방법 (How to Use)

### 1. 환경 설정
Python 3.9 이상 환경에서 필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. 설정 변경 (선택 사항)
`config.py` 파일에서 분석 기간(`START_DATE`, `END_DATE`)이나 파라미터(`ALPHA`, `CORRELATION_THRESHOLD`)를 수정할 수 있습니다.

### 3. 전체 분석 실행
`run.py`를 실행하면 데이터 수집부터 네트워크 구축, 백테스팅, 결과 요약 및 시각화까지 **모든 과정이 자동으로 수행**됩니다.
```bash
python run.py
```
> **참고**: 실행 시 `results/` 폴더가 생성되며, 진행 상황이 터미널에 출력됩니다.

### 4. 개별 단계 실행 (옵션)
이미 `run.py`를 통해 `results/` 데이터가 생성된 상태라면, 아래 스크립트를 통해 결과 분석만 다시 수행할 수 있습니다.

- **포트폴리오 성과 요약**:
  ```bash
  python analyze_results.py
  ```
- **네트워크 지표 그래프 그리기**:
  ```bash
  python plot_metrics.py
  ```

---

## 5. 결과 확인

분석이 완료되면 `results/` 디렉토리에 다양한 결과물이 생성됩니다.

1.  **네트워크 구조 변화**: `results/global_figures/network_metrics_timeseries.png`를 통해 시장 상황에 따른 네트워크의 응집력(Density, Modularity) 변화를 확인하세요.
2.  **전략 성과**: `run.py` 또는 `analyze_results.py` 실행 후 터미널에 출력되는 **Summary Report**를 확인하세요.
    - **Top % (Return/Sharpe)**: 내 전략이 무작위 투자 대비 상위 몇 %인지 알려줍니다. (낮을수록 우수)
3.  **상세 시각화**: `results/quarterly/YYYYQX/figures/` 폴더에서 각 분기별 주식 네트워크의 모습과 차수 분포(Degree Distribution)를 확인할 수 있습니다.