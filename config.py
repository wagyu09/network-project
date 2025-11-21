"""프로젝트 전체에서 사용되는 주요 설정값을 중앙에서 관리하는 모듈

분석 기간, 파일 경로, 모델 파라미터 등 주요 변수들을 이 파일에서 일괄적으로
수정 및 관리
"""

# --- 분석 기간 설정 ---
# 전체 분석 대상이 되는 기간 정의
START_DATE = '2020-01-01'
END_DATE = '2025-09-30'
START_QUARTER = '2020Q1'
END_QUARTER = '2025Q3'

# --- 파일 경로 설정 ---
# 파이프라인 실행 결과(분기별 테스트)가 저장될 디렉토리
TESTS_OUTPUT_DIR = 'tests'
# 최종 분석 결과 리포트가 저장될 경로 및 파일명
FINAL_REPORT_PATH = 'final_summary_report.csv'

# --- 모델 파라미터 설정 ---
# 상관관계 행렬에서 유의미한 엣지를 필터링할 때 사용되는 유의수준(alpha)
ALPHA = 0.01
# 전략 포트폴리오의 성과 비교를 위해 생성할 무작위 포트폴리오의 개수
NUM_RANDOM_PORTFOLIOS = 1000

# --- 데이터 로딩 설정 ---
# S&P 500 종목 리스트에서 명시적으로 제외할 티커 목록
# 주로 해당 기간에 데이터가 없거나, 분석에 적합하지 않은 종목들
EXCLUDED_TICKERS = [
    'SOLS', 'Q', 'APP', 'HOOD', 'EME', 'IBKR', 'TTD', 'DDOG', 'COIN', 'DASH',
    'TKO', 'WSM', 'EXE', 'LII', 'APO', 'WDAY', 'AMTM', 'PLTR', 'DELL', 'ERIE',
    'KKR', 'CRWD', 'GDDY', 'VST', 'SOLV', 'GEV', 'SMCI', 'DECK', 'BX', 'ABNB',
    'GEHC', 'STLD', 'FSLR',
]