# --- Analysis Period ---
START_DATE = '2020-01-01'
END_DATE = '2025-09-30'
START_QUARTER = '2020Q1'
END_QUARTER = '2025Q3'

# --- File Paths ---
TESTS_OUTPUT_DIR = 'pipeline/tests'
FINAL_REPORT_PATH = 'final_summary_report.csv'

# --- Model Parameters ---
ALPHA = 0.01  # Significance level for thresholding
NUM_RANDOM_PORTFOLIOS = 1000  # Number of random portfolios for backtesting

# --- Data Loading ---
# Tickers to be explicitly excluded from the S&P 500 list
EXCLUDED_TICKERS = [
    'SOLS', 'Q', 'APP', 'HOOD', 'EME', 'IBKR', 'TTD', 'DDOG', 'COIN', 'DASH',
    'TKO', 'WSM', 'EXE', 'LII', 'APO', 'WDAY', 'AMTM', 'PLTR', 'DELL', 'ERIE',
    'KKR', 'CRWD', 'GDDY', 'VST', 'SOLV', 'GEV', 'SMCI', 'DECK', 'BX', 'ABNB',
    'GEHC', 'STLD', 'FSLR',
]
