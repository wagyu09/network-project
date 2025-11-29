"""프로젝트 전체에서 사용되는 주요 설정값을 중앙에서 관리하는 모듈

분석 기간, 파일 경로, 모델 파라미터 등 주요 변수들을 이 파일에서 일괄적으로
수정 및 관리
"""
import os

# --- 분석 기간 설정 ---
START_DATE = '2020-01-01'
END_DATE = '2025-09-30'
START_QUARTER = '2020Q1'
END_QUARTER = '2025Q3'

# --- 결과물 경로 설정 ---
BASE_OUTPUT_DIR = 'results'
SUMMARY_DIR = os.path.join(BASE_OUTPUT_DIR, 'summary')
FIGURES_DIR = os.path.join(BASE_OUTPUT_DIR, 'global_figures')
QUARTERLY_DIR = os.path.join(BASE_OUTPUT_DIR, 'quarterly')

# --- 모델 파라미터 설정 ---
ALPHA = 0.01
# [중요] 0.4 기준 유지
CORRELATION_THRESHOLD = 0.4
NUM_RANDOM_PORTFOLIOS = 1000

# --- 데이터 로딩 설정 ---
# 데이터가 불충분하여 분석에서 제외할 종목들
EXCLUDED_TICKERS = [
    'SOLS', 'Q', 'APP', 'HOOD', 'EME', 'IBKR', 'TTD', 'DDOG', 'COIN', 'DASH',
    'TKO', 'WSM', 'EXE', 'LII', 'APO', 'WDAY', 'AMTM', 'PLTR', 'DELL', 'ERIE',
    'KKR', 'CRWD', 'GDDY', 'VST', 'SOLV', 'GEV', 'SMCI', 'DECK', 'BX', 'ABNB',
    'GEHC', 'STLD', 'FSLR',
]