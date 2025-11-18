from scipy import stats
import numpy as np
import pandas as pd

def threshold(corr_stats, alpha = 0.05, c_min = 0):
    df = corr_stats.copy()
    r = df['Correlation'].to_numpy(float)
    n = df['n'].to_numpy(int)

    # 자유도 계산 (n - 2 - 1), 잔차 상관관계의 경우 모델 파라미터 수(1)를 추가로 빼줌
    dfree = n - 3
    r_safe = np.clip(r, -0.999999, 0.999999)

    # t-statistic 계산, 0으로 나누기 오류 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        tval = r_safe * np.sqrt(np.where(dfree > 0, dfree, np.nan) / (1.0 - r_safe**2))

    # p-value 계산, 유효하지 않은 값 처리
    pval = 2.0 * stats.t.sf(np.abs(tval), df=np.where(dfree > 0, dfree, 1))
    pval[(dfree < 1) | ~np.isfinite(pval)] = 1.0

    df['pval'] = pval

    p_edges = df[(df['pval'] < alpha) & (df['Correlation'].abs() >= c_min)].sort_index()

    return p_edges